"""Fail-closed import boundary for GestureLSM CUDA-worker responses.

Torch checkpoints and provider packages remain outside the application
process.  The worker emits canonical JSON and an uncompressed numeric NPZ;
this module verifies pins, hashes, member names, headers, shapes, timebase, and
independent forward kinematics before returning trusted motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
import zipfile

import numpy as np

from .speech_motion import (
    GESTURELSM_PROVIDER_ID,
    MAX_NATIVE_FRAMES,
    NativeSpeechMotion,
    SMPLX55_NAMES,
    SMPLX55_SCHEMA_VERSION,
    SMPLX55_SEMANTIC_SHA256,
    SpeechMotionValidationError,
    axis_angle_to_quaternion,
)


GESTURELSM_RESPONSE_SCHEMA = "autoanim.gesturelsm-provider-response/1.0"
GESTURELSM_REQUEST_SCHEMA = "autoanim.gesturelsm-provider-request/1.0"
MAX_MANIFEST_BYTES = 1_000_000
MAX_NPZ_BYTES = 256_000_000
MAX_CANDIDATES = 8
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ARRAY_NAMES = (
    "ticks",
    "root_translation_m",
    "local_axis_angle_rad",
    "rest_joint_positions_m",
    "rest_world_rotations_xyzw",
    "joint_positions_m",
)


class SpeechMotionProviderError(SpeechMotionValidationError):
    """A speech-motion provider response was unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class SpeechMotionRequest:
    """Hash-bound input envelope sent to the isolated CUDA worker."""

    request_id: str
    duration_ticks: int
    audio_file_name: str
    audio_sha256: str
    transcript_file_name: str
    transcript_sha256: str
    alignment_file_name: str
    alignment_sha256: str
    transcript_mode: str
    asr_model_revision: str | None
    candidate_seeds: tuple[int, ...]
    profile: SpeechMotionProfile

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not _REQUEST_ID.fullmatch(
            self.request_id
        ):
            raise SpeechMotionProviderError("request_id is unsafe")
        if type(self.duration_ticks) is not int or self.duration_ticks <= 0:
            raise SpeechMotionProviderError("duration_ticks must be positive")
        for name in ("audio_file_name", "transcript_file_name", "alignment_file_name"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or Path(value).name != value
                or not value
            ):
                raise SpeechMotionProviderError(f"{name} is unsafe")
        for name in ("audio_sha256", "transcript_sha256", "alignment_sha256"):
            _validate_hex(getattr(self, name), 64, name)
        if self.transcript_mode not in {"supplied", "pinned_asr_fallback"}:
            raise SpeechMotionProviderError("transcript_mode is unsupported")
        if self.transcript_mode == "supplied" and self.asr_model_revision is not None:
            raise SpeechMotionProviderError("Supplied transcripts must bypass ASR")
        if self.transcript_mode == "pinned_asr_fallback":
            if self.asr_model_revision is None:
                raise SpeechMotionProviderError("Fallback ASR must be revision-pinned")
            _validate_hex(self.asr_model_revision, 40, "asr_model_revision")
        seeds = tuple(self.candidate_seeds)
        if (
            not 1 <= len(seeds) <= MAX_CANDIDATES
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in seeds)
        ):
            raise SpeechMotionProviderError("candidate_seeds are invalid or duplicated")
        object.__setattr__(self, "candidate_seeds", seeds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GESTURELSM_REQUEST_SCHEMA,
            "provider_id": GESTURELSM_PROVIDER_ID,
            "operation": "speech_motion_generation",
            "request_id": self.request_id,
            "production_validated": False,
            "profile": self.profile.as_dict(),
            "timeline": {
                "ticks_per_second": 48_000,
                "duration_ticks": self.duration_ticks,
                "output_sample_rate_hz": 30,
            },
            "source": {
                "audio": {
                    "file_name": self.audio_file_name,
                    "sha256": self.audio_sha256,
                },
                "transcript": {
                    "file_name": self.transcript_file_name,
                    "sha256": self.transcript_sha256,
                    "mode": self.transcript_mode,
                    "asr_model_revision": self.asr_model_revision,
                },
                "alignment": {
                    "file_name": self.alignment_file_name,
                    "sha256": self.alignment_sha256,
                },
            },
            "candidate_seeds": list(self.candidate_seeds),
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def _validate_hex(value: str, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpeechMotionProviderError(f"{label} must be lowercase hex length {length}")
    return value


@dataclass(frozen=True, slots=True)
class SpeechMotionProfile:
    """Reviewed provider and model pins accepted by one application build."""

    provider_git_commit_oid: str
    model_revision: str
    model_artifact_sha256: str

    def __post_init__(self) -> None:
        _validate_hex(self.provider_git_commit_oid, 40, "provider_git_commit_oid")
        _validate_hex(self.model_revision, 40, "model_revision")
        _validate_hex(self.model_artifact_sha256, 64, "model_artifact_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "provider_git_commit_oid": self.provider_git_commit_oid,
            "model_revision": self.model_revision,
            "model_artifact_sha256": self.model_artifact_sha256,
        }


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SpeechMotionProviderError(f"Duplicate JSON member: {key}")
        output[key] = value
    return output


def _exact_members(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpeechMotionProviderError(f"{label} must be an object")
    observed = set(value)
    if observed != expected:
        raise SpeechMotionProviderError(
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
        raise SpeechMotionProviderError(f"{label} must be a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise SpeechMotionProviderError(f"{label} size/type is outside limits")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise SpeechMotionProviderError(f"{label} changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SpeechMotionProviderError(f"{label} grew while being read")
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
                SpeechMotionProviderError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SpeechMotionProviderError(
            "Provider manifest is not canonical UTF-8 JSON"
        ) from error


def load_speech_motion_request(path: str | Path) -> SpeechMotionRequest:
    """Load and semantically validate one sealed worker request."""

    requested_path = Path(path)
    if requested_path.is_symlink():
        raise SpeechMotionProviderError("Provider request symlinks are forbidden")
    try:
        resolved = requested_path.resolve(strict=True)
    except OSError as error:
        raise SpeechMotionProviderError("Provider request does not exist") from error
    value = _exact_members(
        _read_manifest(resolved),
        {
            "schema_version",
            "provider_id",
            "operation",
            "request_id",
            "production_validated",
            "profile",
            "timeline",
            "source",
            "candidate_seeds",
        },
        "provider request",
    )
    if (
        value["schema_version"] != GESTURELSM_REQUEST_SCHEMA
        or value["provider_id"] != GESTURELSM_PROVIDER_ID
        or value["operation"] != "speech_motion_generation"
        or value["production_validated"] is not False
    ):
        raise SpeechMotionProviderError("Provider request claims are unsupported")
    profile_value = _exact_members(
        value["profile"],
        {"provider_git_commit_oid", "model_revision", "model_artifact_sha256"},
        "provider request profile",
    )
    timeline = _exact_members(
        value["timeline"],
        {"ticks_per_second", "duration_ticks", "output_sample_rate_hz"},
        "provider request timeline",
    )
    if timeline["ticks_per_second"] != 48_000 or timeline["output_sample_rate_hz"] != 30:
        raise SpeechMotionProviderError("Provider request timebase is unsupported")
    source = _exact_members(
        value["source"], {"audio", "transcript", "alignment"}, "provider request source"
    )
    audio = _exact_members(
        source["audio"], {"file_name", "sha256"}, "provider request audio"
    )
    transcript = _exact_members(
        source["transcript"],
        {"file_name", "sha256", "mode", "asr_model_revision"},
        "provider request transcript",
    )
    alignment = _exact_members(
        source["alignment"],
        {"file_name", "sha256"},
        "provider request alignment",
    )
    if not isinstance(value["candidate_seeds"], list):
        raise SpeechMotionProviderError("candidate_seeds must be an array")
    request = SpeechMotionRequest(
        request_id=value["request_id"],
        duration_ticks=timeline["duration_ticks"],
        audio_file_name=audio["file_name"],
        audio_sha256=audio["sha256"],
        transcript_file_name=transcript["file_name"],
        transcript_sha256=transcript["sha256"],
        alignment_file_name=alignment["file_name"],
        alignment_sha256=alignment["sha256"],
        transcript_mode=transcript["mode"],
        asr_model_revision=transcript["asr_model_revision"],
        candidate_seeds=tuple(value["candidate_seeds"]),
        profile=SpeechMotionProfile(**profile_value),
    )
    if request.as_dict() != value:
        raise SpeechMotionProviderError("Provider request is not canonical semantically")
    return request


def _load_npz(
    encoded: bytes,
    specifications: dict[str, Any],
    frame_count: int,
) -> dict[str, np.ndarray]:
    expected_arrays = {
        "ticks": ("<i8", [frame_count]),
        "root_translation_m": ("<f4", [frame_count, 3]),
        "local_axis_angle_rad": ("<f4", [frame_count, 55, 3]),
        "rest_joint_positions_m": ("<f4", [55, 3]),
        "rest_world_rotations_xyzw": ("<f4", [55, 4]),
        "joint_positions_m": ("<f4", [frame_count, 55, 3]),
    }
    _exact_members(specifications, set(ARRAY_NAMES), "motion_npz.arrays")
    for name, (dtype, shape) in expected_arrays.items():
        specification = _exact_members(
            specifications[name], {"dtype", "shape"}, f"motion_npz.arrays.{name}"
        )
        if specification != {"dtype": dtype, "shape": shape}:
            raise SpeechMotionProviderError(f"Unsafe array declaration: {name}")
    expected_members = {f"{name}.npy" for name in ARRAY_NAMES}
    try:
        with zipfile.ZipFile(BytesIO(encoded)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SpeechMotionProviderError("NPZ contains duplicate members")
            if set(names) != expected_members:
                raise SpeechMotionProviderError("NPZ member names differ from the contract")
            total_uncompressed = 0
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or len(member.parts) != 1
                    or ".." in member.parts
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise SpeechMotionProviderError(
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
                        raise SpeechMotionProviderError("Unsupported NPY format version")
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
                    raise SpeechMotionProviderError(
                        f"NPY header differs from the contract: {array_name}"
                    )
            if total_uncompressed > MAX_NPZ_BYTES:
                raise SpeechMotionProviderError("NPZ expanded size is outside limits")
        with np.load(BytesIO(encoded), allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in ARRAY_NAMES}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, SpeechMotionProviderError):
            raise
        raise SpeechMotionProviderError("Provider motion artifact is not a safe NPZ") from error
    return arrays


def load_gesturelsm_response(
    manifest_path: str | Path,
    *,
    profile: SpeechMotionProfile,
    request: SpeechMotionRequest,
) -> NativeSpeechMotion:
    """Load one pinned CUDA-worker response without importing provider code."""

    requested_path = Path(manifest_path)
    if requested_path.is_symlink():
        raise SpeechMotionProviderError("Provider manifest symlinks are forbidden")
    try:
        path = requested_path.resolve(strict=True)
    except OSError as error:
        raise SpeechMotionProviderError("Provider manifest does not exist") from error
    manifest = _exact_members(
        _read_manifest(path),
        {
            "schema_version",
            "provider_id",
            "request_id",
            "request_sha256",
            "provider_git_commit_oid",
            "model_revision",
            "model_artifact_sha256",
            "runtime_class",
            "execution_provider",
            "operation",
            "motion_kind",
            "production_validated",
            "candidate_seed",
            "frame_count",
            "duration_ticks",
            "sample_rate_hz",
            "skeleton",
            "source_coordinate_system",
            "source",
            "motion_npz",
        },
        "provider manifest",
    )
    expected_scalars: dict[str, Any] = {
        "schema_version": GESTURELSM_RESPONSE_SCHEMA,
        "provider_id": GESTURELSM_PROVIDER_ID,
        "provider_git_commit_oid": profile.provider_git_commit_oid,
        "model_revision": profile.model_revision,
        "model_artifact_sha256": profile.model_artifact_sha256,
        "runtime_class": "cuda_worker",
        "execution_provider": "CUDAExecutionProvider",
        "operation": "speech_motion_generation",
        "motion_kind": "generated",
        "production_validated": False,
        "request_id": request.request_id,
        "request_sha256": sha256(request.canonical_json_bytes()).hexdigest(),
    }
    if request.profile != profile:
        raise SpeechMotionProviderError("Request and accepted profile differ")
    for name, expected in expected_scalars.items():
        if manifest[name] != expected or type(manifest[name]) is not type(expected):
            raise SpeechMotionProviderError(f"Unexpected provider claim: {name}")
    frame_count = manifest["frame_count"]
    if type(frame_count) is not int or not 2 <= frame_count <= MAX_NATIVE_FRAMES:
        raise SpeechMotionProviderError("Provider frame_count is outside limits")
    for name in ("duration_ticks", "sample_rate_hz", "candidate_seed"):
        if type(manifest[name]) is not int:
            raise SpeechMotionProviderError(f"{name} must be an integer")
    skeleton = _exact_members(
        manifest["skeleton"],
        {"schema_version", "semantic_sha256", "joint_names"},
        "skeleton",
    )
    if skeleton != {
        "schema_version": SMPLX55_SCHEMA_VERSION,
        "semantic_sha256": SMPLX55_SEMANTIC_SHA256,
        "joint_names": list(SMPLX55_NAMES),
    }:
        raise SpeechMotionProviderError("Provider skeleton contract differs")
    coordinates = _exact_members(
        manifest["source_coordinate_system"],
        {"handedness", "up_axis", "forward_axis", "linear_unit_in_meters"},
        "source_coordinate_system",
    )
    if coordinates != {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "linear_unit_in_meters": 1.0,
    }:
        raise SpeechMotionProviderError("Unexpected source coordinate convention")
    source = _exact_members(
        manifest["source"],
        {"audio_sha256", "transcript_sha256", "alignment_sha256"},
        "source",
    )
    for name, value in source.items():
        _validate_hex(value, 64, f"source.{name}")
    motion = _exact_members(
        manifest["motion_npz"], {"file_name", "sha256", "arrays"}, "motion_npz"
    )
    filename = motion["file_name"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".npz")
    ):
        raise SpeechMotionProviderError("Unsafe motion NPZ file name")
    _validate_hex(motion["sha256"], 64, "motion_npz.sha256")
    artifact_path = path.parent / filename
    encoded = _read_regular_file(artifact_path, MAX_NPZ_BYTES, "Motion NPZ")
    if sha256(encoded).hexdigest() != motion["sha256"]:
        raise SpeechMotionProviderError("Motion NPZ SHA-256 differs")
    arrays = _load_npz(encoded, motion["arrays"], frame_count)
    return NativeSpeechMotion(
        provider_id=manifest["provider_id"],
        provider_git_commit_oid=manifest["provider_git_commit_oid"],
        model_revision=manifest["model_revision"],
        model_artifact_sha256=manifest["model_artifact_sha256"],
        candidate_seed=manifest["candidate_seed"],
        duration_ticks=manifest["duration_ticks"],
        sample_rate_hz=manifest["sample_rate_hz"],
        ticks=arrays["ticks"],
        root_translation_m=arrays["root_translation_m"],
        local_rotations_xyzw=axis_angle_to_quaternion(
            arrays["local_axis_angle_rad"]
        ),
        rest_joint_positions_m=arrays["rest_joint_positions_m"],
        rest_world_rotations_xyzw=arrays["rest_world_rotations_xyzw"],
        joint_positions_m=arrays["joint_positions_m"],
        audio_sha256=source["audio_sha256"],
        transcript_sha256=source["transcript_sha256"],
        alignment_sha256=source["alignment_sha256"],
    )


__all__ = [
    "ARRAY_NAMES",
    "GESTURELSM_REQUEST_SCHEMA",
    "GESTURELSM_RESPONSE_SCHEMA",
    "SpeechMotionRequest",
    "SpeechMotionProfile",
    "SpeechMotionProviderError",
    "load_speech_motion_request",
    "load_gesturelsm_response",
]
