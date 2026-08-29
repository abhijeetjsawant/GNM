"""Safe application-side import for NVIDIA body-provider responses.

NVIDIA modules and pickle/Torch artifacts are never imported here.  The
provider process must emit a primitive JSON manifest plus an uncompressed NPZ
whose bytes, members, dtypes, shapes, and semantic claims are checked before a
``SomaMotion`` can be created.

This first implementation intentionally accepts only the separately named,
non-production Apple-Silicon GEM-X preview runtime.  Signed production worker
attestation remains a later gate.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any
import zipfile

import numpy as np

from .acting import TICKS_PER_SECOND
from .soma_motion import (
    GEM_X_CONTACT_NAMES,
    GEM_X_CONTACT_SCHEMA_ID,
    SomaMotion,
    SomaMotionValidationError,
)


GEM_X_RESPONSE_SCHEMA_LEGACY = "autoanim.gem-x-provider-response/1.0"
GEM_X_RESPONSE_SCHEMA = "autoanim.gem-x-provider-response/1.1"
GEM_X_COMMIT = "32992550dba114c62243fb55e361311972dce8f9"
MAX_MANIFEST_BYTES = 1_000_000
MAX_NPZ_BYTES = 160_000_000
ARRAY_NAMES = (
    "source_pts",
    "root_translation_m",
    "local_rotations_xyzw",
    "rest_joint_positions_m",
    "rest_world_rotations_xyzw",
    "joint_positions_m",
    "contacts",
)


class NvidiaBodyProviderError(SomaMotionValidationError):
    """A provider response failed the safe application boundary."""


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise NvidiaBodyProviderError(f"Duplicate JSON member: {key}")
        output[key] = value
    return output


def _exact_members(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NvidiaBodyProviderError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        raise NvidiaBodyProviderError(
            f"{label} members differ: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )
    return value


def _read_regular_file(path: Path, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NvidiaBodyProviderError(f"{label} must be a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise NvidiaBodyProviderError(f"{label} size/type is outside limits")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise NvidiaBodyProviderError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise NvidiaBodyProviderError(f"{label} grew while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_manifest(path: Path) -> dict[str, Any]:
    encoded = _read_regular_file(path, MAX_MANIFEST_BYTES, "Provider manifest")
    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=lambda value: (_ for _ in ()).throw(
                NvidiaBodyProviderError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NvidiaBodyProviderError("Provider manifest is not canonical UTF-8 JSON") from error


def _load_npz(
    encoded: bytes,
    specifications: dict[str, Any],
    frame_count: int,
) -> dict[str, np.ndarray]:
    expected_arrays = {
        "source_pts": ("<i8", [frame_count]),
        "root_translation_m": ("<f4", [frame_count, 3]),
        "local_rotations_xyzw": ("<f4", [frame_count, 77, 4]),
        "rest_joint_positions_m": ("<f4", [77, 3]),
        "rest_world_rotations_xyzw": ("<f4", [77, 4]),
        "joint_positions_m": ("<f4", [frame_count, 77, 3]),
        "contacts": ("|b1", [frame_count, 6]),
    }
    _exact_members(specifications, set(ARRAY_NAMES), "motion_npz.arrays")
    for name, (dtype, shape) in expected_arrays.items():
        specification = _exact_members(
            specifications[name],
            {"dtype", "shape"},
            f"motion_npz.arrays.{name}",
        )
        if specification != {"dtype": dtype, "shape": shape}:
            raise NvidiaBodyProviderError(f"Unsafe array declaration: {name}")
    expected_members = {f"{name}.npy" for name in ARRAY_NAMES}
    try:
        with zipfile.ZipFile(BytesIO(encoded)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise NvidiaBodyProviderError("NPZ contains duplicate members")
            if set(names) != expected_members:
                raise NvidiaBodyProviderError("NPZ member names differ from the contract")
            total_uncompressed = 0
            total_expected_payload = 0
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or len(member.parts) != 1
                    or ".." in member.parts
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise NvidiaBodyProviderError(
                        "NPZ members must be flat, uncompressed NPY files"
                    )
                total_uncompressed += info.file_size
                array_name = member.stem
                with archive.open(info) as source:
                    version = np.lib.format.read_magic(source)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                            source
                        )
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                            source
                        )
                    else:
                        raise NvidiaBodyProviderError("Unsupported NPY format version")
                    header_size = source.tell()
                expected_dtype, expected_shape = expected_arrays[array_name]
                expected_payload = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(
                    expected_dtype
                ).itemsize
                if (
                    fortran_order
                    or dtype.hasobject
                    or dtype.str != expected_dtype
                    or list(shape) != expected_shape
                    or info.file_size != header_size + expected_payload
                ):
                    raise NvidiaBodyProviderError(
                        f"NPY header differs from the contract: {array_name}"
                    )
                total_expected_payload += expected_payload
            if total_uncompressed > MAX_NPZ_BYTES:
                raise NvidiaBodyProviderError("NPZ expanded size is outside limits")
            if total_expected_payload > MAX_NPZ_BYTES:
                raise NvidiaBodyProviderError("NPZ declared payload is outside limits")
        with np.load(BytesIO(encoded), allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in ARRAY_NAMES}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, NvidiaBodyProviderError):
            raise
        raise NvidiaBodyProviderError("Provider motion artifact is not a safe NPZ") from error

    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise NvidiaBodyProviderError("Object arrays are forbidden")
        expected_dtype, expected_shape = expected_arrays[name]
        if array.dtype.str != expected_dtype or list(array.shape) != expected_shape:
            raise NvidiaBodyProviderError(f"Array metadata mismatch: {name}")
    return arrays


def _load_gem_x_response(
    manifest_path: str | Path,
    *,
    expected_runtime_class: str,
    expected_execution_provider: str,
    require_regional_fidelity: bool,
    allow_legacy_schema: bool,
) -> SomaMotion:

    requested_path = Path(manifest_path)
    if requested_path.is_symlink():
        raise NvidiaBodyProviderError("Provider manifest symlinks are forbidden")
    try:
        path = requested_path.resolve(strict=True)
    except OSError as error:
        raise NvidiaBodyProviderError("Provider manifest does not exist") from error
    raw_manifest = _read_manifest(path)
    schema_version = raw_manifest.get("schema_version")
    common_members = {
            "schema_version",
            "provider_id",
            "provider_git_commit_oid",
            "runtime_class",
            "execution_provider",
            "camera_model",
            "operation",
            "motion_kind",
            "production_validated",
            "frame_count",
            "source_time_base",
            "source_coordinate_system",
            "contact_schema",
            "input_sha256",
            "provider_raw_motion_sha256",
            "motion_npz",
    }
    if schema_version == GEM_X_RESPONSE_SCHEMA:
        expected_members = common_members | {"regional_reprojection"}
    elif schema_version == GEM_X_RESPONSE_SCHEMA_LEGACY and allow_legacy_schema:
        expected_members = common_members
    else:
        raise NvidiaBodyProviderError("Unexpected GEM-X response schema")
    manifest = _exact_members(
        raw_manifest,
        expected_members,
        "provider manifest",
    )
    expected_scalars = {
        "schema_version": schema_version,
        "provider_id": "nvidia_gem_x",
        "provider_git_commit_oid": GEM_X_COMMIT,
        "runtime_class": expected_runtime_class,
        "execution_provider": expected_execution_provider,
        "camera_model": "static_camera_assumed",
        "operation": "video_capture",
        "motion_kind": "observed",
        "production_validated": False,
    }
    for name, expected in expected_scalars.items():
        if manifest[name] != expected or type(manifest[name]) is not type(expected):
            raise NvidiaBodyProviderError(f"Unexpected provider claim: {name}")
    if schema_version == GEM_X_RESPONSE_SCHEMA:
        reprojection = manifest["regional_reprojection"]
        if not isinstance(reprojection, dict):
            raise NvidiaBodyProviderError("regional_reprojection must be an object")
        status = reprojection.get("status")
        passed = reprojection.get("passed")
        failures = reprojection.get("failure_reasons")
        if (
            reprojection.get("schema_version")
            != "autoanim.arm-reprojection-audit/1.0"
            or status not in {"passed", "failed", "not_run"}
            or type(passed) is not bool
            or not isinstance(failures, list)
            or not all(isinstance(value, str) and value for value in failures)
            or (status == "passed" and (not passed or failures))
            or (status != "passed" and passed)
        ):
            raise NvidiaBodyProviderError("Invalid regional reprojection claim")
        if status in {"passed", "failed"}:
            required_audit_members = {
                "schema_version",
                "status",
                "passed",
                "confidence_threshold",
                "p95_error_limit_bbox_height",
                "sign_mismatch_limit",
                "sides",
                "failure_reasons",
                "vitpose_sha256",
                "bounding_boxes_sha256",
            }
            _exact_members(
                reprojection,
                required_audit_members,
                "regional_reprojection",
            )
            for digest_name in ("vitpose_sha256", "bounding_boxes_sha256"):
                digest = reprojection[digest_name]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise NvidiaBodyProviderError(
                        f"Invalid regional reprojection digest: {digest_name}"
                    )
        else:
            _exact_members(
                reprojection,
                {"schema_version", "status", "passed", "failure_reasons"},
                "regional_reprojection",
            )
        if require_regional_fidelity and not passed:
            raise NvidiaBodyProviderError(
                "GEM-X regional reprojection fidelity gate did not pass"
            )
    elif require_regional_fidelity:
        raise NvidiaBodyProviderError(
            "Legacy GEM-X responses have no regional reprojection evidence"
        )
    frame_count = manifest["frame_count"]
    if type(frame_count) is not int or not 2 <= frame_count <= 216_001:
        raise NvidiaBodyProviderError("Provider frame_count is outside limits")
    timebase = _exact_members(
        manifest["source_time_base"],
        {"numerator", "denominator"},
        "source_time_base",
    )
    if any(type(timebase[name]) is not int or timebase[name] <= 0 for name in timebase):
        raise NvidiaBodyProviderError("Source timebase must be a positive rational")
    coordinates = _exact_members(
        manifest["source_coordinate_system"],
        {
            "handedness",
            "up_axis",
            "forward_axis",
            "linear_unit_in_meters",
            "source_to_canonical_rotation_xyzw",
        },
        "source_coordinate_system",
    )
    if coordinates != {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "linear_unit_in_meters": 1.0,
        "source_to_canonical_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }:
        raise NvidiaBodyProviderError("Unexpected GEM-X coordinate convention")
    contact = _exact_members(
        manifest["contact_schema"],
        {
            "id",
            "contact_names",
            "provider_joint_indices",
            "velocity_threshold_m_per_s",
            "velocity_timing",
        },
        "contact_schema",
    )
    if (
        contact["id"] != GEM_X_CONTACT_SCHEMA_ID
        or contact["contact_names"] != list(GEM_X_CONTACT_NAMES)
        or contact["provider_joint_indices"] != [69, 70, 74, 75, 14, 42]
        or contact["velocity_threshold_m_per_s"] != 0.15
        or contact["velocity_timing"] != "exact_source_pts"
    ):
        raise NvidiaBodyProviderError("Unexpected GEM-X contact semantics")
    motion = _exact_members(
        manifest["motion_npz"],
        {"file_name", "sha256", "arrays"},
        "motion_npz",
    )
    file_name = motion["file_name"]
    if (
        not isinstance(file_name, str)
        or PurePosixPath(file_name).name != file_name
        or not file_name.endswith(".npz")
    ):
        raise NvidiaBodyProviderError("Unsafe provider motion file name")
    artifact_path = path.parent / file_name
    artifact_bytes = _read_regular_file(
        artifact_path,
        MAX_NPZ_BYTES,
        "Provider motion artifact",
    )
    if sha256(artifact_bytes).hexdigest() != motion["sha256"]:
        raise NvidiaBodyProviderError("Provider motion SHA-256 mismatch")
    arrays = _load_npz(artifact_bytes, motion["arrays"], frame_count)
    source_pts = arrays["source_pts"]
    if source_pts.shape != (frame_count,) or source_pts.dtype.kind not in "iu":
        raise NvidiaBodyProviderError("source_pts does not match frame_count")
    first_pts = int(source_pts[0])
    denominator = int(timebase["denominator"])
    ticks = np.asarray(
        [
            (
                2
                * (int(value) - first_pts)
                * int(timebase["numerator"])
                * TICKS_PER_SECOND
                + denominator
            )
            // (2 * denominator)
            for value in source_pts
        ],
        dtype=np.int64,
    )
    if ticks[-1] <= 0:
        raise NvidiaBodyProviderError("Provider motion duration must be positive")
    tick_steps = np.diff(ticks)
    sample_rate_hz = int(
        round(TICKS_PER_SECOND / float(np.median(tick_steps)))
    )
    return SomaMotion(
        provider_id="nvidia_gem_x",
        provider_git_commit_oid=GEM_X_COMMIT,
        operation="video_capture",
        motion_kind="observed",
        duration_ticks=int(ticks[-1]),
        sample_rate_hz=sample_rate_hz,
        ticks=ticks,
        source_pts=source_pts,
        root_translation_m=arrays["root_translation_m"],
        local_rotations_xyzw=arrays["local_rotations_xyzw"],
        rest_joint_positions_m=arrays["rest_joint_positions_m"],
        rest_world_rotations_xyzw=arrays["rest_world_rotations_xyzw"],
        joint_positions_m=arrays["joint_positions_m"],
        contacts=arrays["contacts"],
        contact_schema_id=GEM_X_CONTACT_SCHEMA_ID,
        contact_names=GEM_X_CONTACT_NAMES,
        source_handedness="right",
        source_up_axis="+Y",
        source_forward_axis="+Z",
        source_linear_unit_in_meters=1.0,
        source_to_canonical_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_time_base_numerator=int(timebase["numerator"]),
        source_time_base_denominator=denominator,
        input_sha256=manifest["input_sha256"],
        provider_raw_motion_sha256=manifest["provider_raw_motion_sha256"],
    )


def load_gem_x_preview_response(
    manifest_path: str | Path,
    *,
    require_regional_fidelity: bool = False,
) -> SomaMotion:
    """Load an unattested local Apple-Silicon preview response."""

    return _load_gem_x_response(
        manifest_path,
        expected_runtime_class="apple_silicon_preview",
        expected_execution_provider="CPUExecutionProvider",
        require_regional_fidelity=require_regional_fidelity,
        allow_legacy_schema=True,
    )


def load_gem_x_cuda_response(
    manifest_path: str | Path,
    *,
    require_regional_fidelity: bool = True,
) -> SomaMotion:
    """Load the official full-image-conditioned CUDA response after 2D audit.

    This proves observation fidelity and structural validity, not signed worker
    attestation or production perceptual qualification.
    """

    return _load_gem_x_response(
        manifest_path,
        expected_runtime_class="cuda_pytorch_regression_full_image_conditioned",
        expected_execution_provider="CUDAExecutionProvider",
        require_regional_fidelity=require_regional_fidelity,
        allow_legacy_schema=False,
    )
