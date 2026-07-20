"""Deterministic, fail-closed ReviewBundle v1 core contract.

ReviewBundle v1 is a portable *review description*, not an approval record and
not a renderer.  It binds a sealed video-performance manifest to the exact
artifact bytes available to a native review client, reconstructs one rational
source-PTS cursor from Capture v1, and exposes immutable candidate layers.
Within-job A/B is deliberately unavailable until there is more than one
renderable revision; v1 instead emits a strict cross-bundle comparison key.

The service boundary is responsible for calling this core only after JobStore
has verified the manifest HMAC and retained input bytes.  This module has
neither the HMAC key nor the retained source path, so those two claims remain
false in its output.  It deliberately grants no production motion authority
and makes no material, correction, approval, or publication claim.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePath
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA_VERSION = "autoanim.review-bundle/1.0"
CLOCK_SCHEMA_VERSION = "autoanim.review-clock/1.0"
COMPARISON_KEY_SCHEMA_VERSION = "autoanim.review-comparison-key/1.0"
LAYER_SCHEMA_VERSION = "autoanim.review-layer/1.0"
REVISION_GRAPH_SCHEMA_VERSION = "autoanim.review-revision-graph/1.0"
CLOSEUP_SCHEMA_VERSION = "autoanim.review-closeup/1.0"
MATERIAL_SCHEMA_VERSION = "autoanim.review-material-channel/1.0"
CORRECTION_SCHEMA_VERSION = "autoanim.review-correction-eligibility/1.0"
BRIDGE_SCHEMA_VERSION = "autoanim.review-bridge/1.0"

MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_ARTIFACTS = 256
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024 * 1024
MAX_FRAMES = 1_800
MAX_CONTROLS_ARCHIVE_BYTES = 64 * 1024 * 1024

LAYER_ORDER = (
    "source",
    "visual_base",
    "audio_repair",
    "acting",
    "authored_correction",
    "physics",
    "final",
)
CLOSEUP_REGIONS = ("mouth", "tongue", "left_eye", "right_eye")
MATERIAL_CHANNELS = (
    ("base_color", "base_color", "srgb"),
    ("normal", "normal", "linear"),
    ("displacement", "displacement", "linear"),
    ("roughness", "roughness", "linear"),
    ("specular", "specular_color", "linear"),
)
BRIDGE_MESSAGE_TYPES = ("cursor", "layer", "selection", "revision")

LIMITATIONS = (
    "manifest_hmac_and_input_bytes_must_be_preverified_by_service_boundary",
    "manifest_hmac_signature_not_verified_by_review_bundle_core",
    "motion_consumption_reported_not_independently_recomputed",
    "closeup_bounds_require_native_semantic_selection",
    "material_hash_references_may_not_have_isolated_bytes",
    "raw_gnm_controls_not_sampled_by_a_measurement_viewport",
    "correction_writer_and_human_approval_not_established",
)

_PERFORMANCE_MEMBERS = frozenset(
    {
        "schema_version.npy",
        "identity.npy",
        "expression.npy",
        "rotations.npy",
        "translation.npy",
        "timestamps_seconds.npy",
        "source_pts.npy",
        "detected.npy",
        "effective_quality.npy",
        "source_lip_geometry_valid.npy",
        "source_lip_gap_interocular.npy",
        "source_lip_contact_confidence.npy",
        "lip_contact_target_gap_interocular.npy",
        "contact_correction_applied.npy",
        "lip_contact_attained.npy",
        "lip_aperture_target_gap_interocular.npy",
        "lip_aperture_correction_applied.npy",
        "lip_aperture_target_attained.npy",
        "provenance_json.npy",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:@+-]{1,200}$")


class ReviewBundleError(ValueError):
    """A ReviewBundle input or document violates the v1 contract."""

    def __init__(self, code: str, message: str, *, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


def _duplicate_free_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewBundleError(
                "DUPLICATE_KEY", f"Duplicate JSON member: {key}", field=key
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReviewBundleError("NONFINITE_NUMBER", f"Non-finite JSON number: {value}")


def _canonical_json(value: Any, *, label: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReviewBundleError(
            "INVALID_DOCUMENT", f"{label} is not finite canonical JSON"
        ) from exc
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ReviewBundleError(
            "DOCUMENT_SIZE", f"{label} exceeds {MAX_DOCUMENT_BYTES} bytes"
        )
    return encoded


def _read_json_document(
    source: str | Path | bytes | bytearray | Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        value: Any = deepcopy(dict(source))
        _canonical_json(value, label=label)
    else:
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            if not isinstance(source, (str, Path)):
                raise ReviewBundleError(
                    "INVALID_TYPE", f"{label} source has an unsupported type"
                )
            try:
                candidate = Path(source)
                raw = candidate.read_bytes()
            except (OSError, TypeError, ValueError) as exc:
                raise ReviewBundleError(
                    "DOCUMENT_UNREADABLE", f"{label} is not readable"
                ) from exc
        if not raw or len(raw) > MAX_DOCUMENT_BYTES:
            raise ReviewBundleError(
                "DOCUMENT_SIZE", f"{label} is empty or exceeds its byte bound"
            )
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_duplicate_free_pairs,
                parse_constant=_reject_constant,
            )
        except ReviewBundleError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewBundleError(
                "INVALID_DOCUMENT", f"{label} is not strict UTF-8 JSON"
            ) from exc
    if not isinstance(value, dict):
        raise ReviewBundleError("INVALID_TYPE", f"{label} must be a JSON object")
    return value


def _object(value: Any, field: str, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewBundleError("INVALID_TYPE", f"{field} must be an object", field=field)
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise ReviewBundleError(
            "INVALID_FIELDS",
            f"{field} fields differ; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}",
            field=field,
        )
    return value


def _sequence(value: Any, field: str, *, maximum: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ReviewBundleError("INVALID_TYPE", f"{field} must be an array", field=field)
    if maximum is not None and len(value) > maximum:
        raise ReviewBundleError("LIMIT_EXCEEDED", f"{field} exceeds its item bound", field=field)
    return value


def _string(value: Any, field: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ReviewBundleError("INVALID_TYPE", f"{field} must be bounded text", field=field)
    return value


def _identifier(value: Any, field: str) -> str:
    text = _string(value, field, maximum=200)
    if not _IDENTIFIER.fullmatch(text):
        raise ReviewBundleError("INVALID_IDENTIFIER", f"{field} is invalid", field=field)
    return text


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReviewBundleError("INVALID_SHA256", f"{field} must be lowercase SHA-256", field=field)
    return value


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReviewBundleError("INVALID_TYPE", f"{field} must be an integer", field=field)
    if not -(2**63) <= value <= 2**63 - 1:
        raise ReviewBundleError(
            "LIMIT_EXCEEDED", f"{field} exceeds signed 64-bit bounds", field=field
        )
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ReviewBundleError("INVALID_TYPE", f"{field} must be boolean", field=field)
    return value


def _optional_string(value: Any, field: str) -> str | None:
    return None if value is None else _identifier(value, field)


def _sha_value(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical_json(value, label=label)).hexdigest()


def review_bundle_payload_sha256(document: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(document))
    payload.pop("bundle_sha256", None)
    return _sha_value(payload, label="ReviewBundle payload")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ReviewBundleError("ARTIFACT_UNREADABLE", f"Artifact {path.name} is unreadable") from exc
    return digest.hexdigest()


def _array_sha256(domain: str, array: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ReviewBundleError(
            "CONTROLS_NONFINITE", f"{domain} contains non-finite values"
        )
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        _canonical_json(
            {"dtype": value.dtype.str, "shape": list(value.shape)},
            label=f"{domain} metadata",
        )
    )
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_pts_sha256(source_pts: Sequence[int]) -> str:
    return _array_sha256(
        "autoanim.review-source-pts/1.0",
        np.asarray(source_pts, dtype=np.int64),
        "<i8",
    )


def _pipeline_array_sha256(array: np.ndarray) -> str:
    """Hash used by the sealed video revision-chain schema."""

    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _load_controls_identity(
    controls_path: Path, *, expected_source_pts: Sequence[int]
) -> tuple[str, str, dict[str, str]]:
    """Read only the bounded, exact Performance v3 archive contract.

    The ledger hash authenticates the whole file; this parser additionally
    rejects ZIP ambiguity/bombs, object arrays, wrong shapes/dtypes, non-finite
    control values, and a controls clock that differs from Capture v1.
    """

    try:
        archive_size = controls_path.stat().st_size
    except OSError as exc:
        raise ReviewBundleError(
            "CONTROLS_ARTIFACT_INVALID", "Performance controls are unreadable"
        ) from exc
    if not 0 < archive_size <= MAX_CONTROLS_ARCHIVE_BYTES:
        raise ReviewBundleError(
            "CONTROLS_BOUNDS", "Performance controls exceed the v1 archive bound"
        )
    try:
        with zipfile.ZipFile(controls_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(set(names))
                or set(names) != _PERFORMANCE_MEMBERS
                or any(
                    member.flag_bits & 0x1
                    or member.is_dir()
                    or member.file_size < 0
                    or member.compress_size < 0
                    for member in members
                )
                or sum(member.file_size for member in members)
                > MAX_CONTROLS_ARCHIVE_BYTES
            ):
                raise ReviewBundleError(
                    "CONTROLS_ARCHIVE_INVALID",
                    "Performance controls have ambiguous or unbounded ZIP members",
                )

        frame_count = len(expected_source_pts)
        expected_arrays: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
            "identity": (np.dtype(np.float32), (253,)),
            "expression": (np.dtype(np.float32), (frame_count, 383)),
            "rotations": (np.dtype(np.float32), (frame_count, 4, 3)),
            "translation": (np.dtype(np.float32), (frame_count, 3)),
            "timestamps_seconds": (np.dtype(np.float64), (frame_count,)),
            "source_pts": (np.dtype(np.int64), (frame_count,)),
            "detected": (np.dtype(np.bool_), (frame_count,)),
            "effective_quality": (np.dtype(np.float32), (frame_count,)),
            "source_lip_geometry_valid": (np.dtype(np.bool_), (frame_count,)),
            "source_lip_gap_interocular": (np.dtype(np.float32), (frame_count,)),
            "source_lip_contact_confidence": (
                np.dtype(np.float32),
                (frame_count,),
            ),
            "lip_contact_target_gap_interocular": (
                np.dtype(np.float32),
                (frame_count,),
            ),
            "contact_correction_applied": (np.dtype(np.bool_), (frame_count,)),
            "lip_contact_attained": (np.dtype(np.bool_), (frame_count,)),
            "lip_aperture_target_gap_interocular": (
                np.dtype(np.float32),
                (frame_count,),
            ),
            "lip_aperture_correction_applied": (
                np.dtype(np.bool_),
                (frame_count,),
            ),
            "lip_aperture_target_attained": (np.dtype(np.bool_), (frame_count,)),
        }
        with np.load(controls_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                member.removesuffix(".npy") for member in _PERFORMANCE_MEMBERS
            }:
                raise ReviewBundleError(
                    "CONTROLS_ARCHIVE_INVALID",
                    "Performance controls member names are not canonical",
                )
            schema = archive["schema_version"]
            provenance = archive["provenance_json"]
            if (
                schema.shape != ()
                or schema.dtype.kind not in {"U", "S"}
                or str(schema.item()) != "autoanim.gnm-performance.v3"
                or provenance.shape != ()
                or provenance.dtype.kind not in {"U", "S"}
            ):
                raise ReviewBundleError(
                    "CONTROLS_SCHEMA_INVALID",
                    "Performance controls metadata is not Performance v3",
                )
            provenance_text = str(provenance.item())
            if not provenance_text or len(provenance_text.encode("utf-8")) > 1024 * 1024:
                raise ReviewBundleError(
                    "CONTROLS_BOUNDS", "Performance provenance is empty or unbounded"
                )
            _read_json_document(
                provenance_text.encode("utf-8"), label="Performance provenance"
            )
            arrays: dict[str, np.ndarray] = {}
            for name, (dtype, shape) in expected_arrays.items():
                value = archive[name]
                if value.dtype != dtype or value.shape != shape:
                    raise ReviewBundleError(
                        "CONTROLS_SHAPE_INVALID",
                        f"Performance controls array {name} has the wrong dtype or shape",
                    )
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise ReviewBundleError(
                        "CONTROLS_NONFINITE",
                        f"Performance controls array {name} contains non-finite values",
                    )
                arrays[name] = value
            if arrays["source_pts"].tolist() != list(expected_source_pts):
                raise ReviewBundleError(
                    "CONTROLS_CLOCK_MISMATCH",
                    "Performance controls and Capture v1 use different source PTS",
                )
            if frame_count > 1 and np.any(np.diff(arrays["timestamps_seconds"]) <= 0):
                raise ReviewBundleError(
                    "CONTROLS_CLOCK_MISMATCH",
                    "Performance controls timestamps are not strictly increasing",
                )
            identity_sha256 = _array_sha256(
                "autoanim.gnm-identity-array/1.0",
                arrays["identity"],
                "<f4",
            )
            revision_bindings = {
                "identity_sha256": _pipeline_array_sha256(
                    arrays["identity"]
                ),
                "source_pts_sha256": _pipeline_array_sha256(
                    arrays["source_pts"]
                ),
                "final_expression_sha256": _pipeline_array_sha256(
                    arrays["expression"]
                ),
            }
    except ReviewBundleError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ReviewBundleError(
            "CONTROLS_ARTIFACT_INVALID",
            "Performance controls could not be parsed as a bounded Performance v3 archive",
        ) from exc
    return identity_sha256, "autoanim.gnm-performance.v3", revision_bindings


def _fraction_pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _verified_rational(value: Any, field: str, *, positive: bool = False) -> Fraction:
    pair = _sequence(value, field)
    if len(pair) != 2:
        raise ReviewBundleError("INVALID_RATIONAL", f"{field} must contain two integers", field=field)
    numerator = _integer(pair[0], f"{field}.0")
    denominator = _integer(pair[1], f"{field}.1")
    if denominator <= 0 or (positive and numerator <= 0):
        raise ReviewBundleError("INVALID_RATIONAL", f"{field} has an invalid sign", field=field)
    fraction = Fraction(numerator, denominator)
    if [fraction.numerator, fraction.denominator] != pair:
        raise ReviewBundleError("INVALID_RATIONAL", f"{field} must be reduced", field=field)
    return fraction


def _manifest_artifact_entries(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not 1 <= len(artifacts) <= MAX_ARTIFACTS:
        raise ReviewBundleError("INVALID_MANIFEST", "Performance artifact ledger is missing or unbounded")
    parsed: dict[str, dict[str, Any]] = {}
    for logical_name, raw in artifacts.items():
        logical = _identifier(logical_name, f"artifacts.{logical_name}.logical_name")
        entry = _object(
            raw,
            f"artifacts.{logical}",
            ("name", "bytes", "sha256", "media_type"),
        )
        name = _string(entry["name"], f"artifacts.{logical}.name", maximum=255)
        size = _integer(entry["bytes"], f"artifacts.{logical}.bytes")
        if PurePath(name).name != name or not 0 < size <= MAX_ARTIFACT_BYTES:
            raise ReviewBundleError("INVALID_ARTIFACT", f"Artifact {logical} name or size is invalid")
        parsed[logical] = {
            "name": name,
            "bytes": size,
            "sha256": _sha(entry["sha256"], f"artifacts.{logical}.sha256"),
            "media_type": _string(entry["media_type"], f"artifacts.{logical}.media_type", maximum=160),
        }
    return parsed


def _verify_artifacts(
    ledger: Mapping[str, dict[str, Any]], artifact_paths: Mapping[str, str | Path]
) -> list[dict[str, Any]]:
    if not isinstance(artifact_paths, Mapping) or set(artifact_paths) != set(ledger):
        raise ReviewBundleError(
            "ARTIFACT_SET_MISMATCH",
            "Artifact paths must exactly match every logical artifact in the performance manifest",
        )
    references: list[dict[str, Any]] = []
    for logical in sorted(ledger):
        entry = ledger[logical]
        path_value = artifact_paths[logical]
        if not isinstance(path_value, (str, Path)):
            raise ReviewBundleError(
                "INVALID_TYPE", f"Artifact path {logical} has an unsupported type"
            )
        path = Path(path_value)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ReviewBundleError("ARTIFACT_UNREADABLE", f"Artifact {logical} is missing") from exc
        if (
            not path.is_file()
            or path.name != entry["name"]
            or size != entry["bytes"]
            or _file_sha256(path) != entry["sha256"]
        ):
            raise ReviewBundleError(
                "ARTIFACT_INTEGRITY", f"Artifact {logical} does not match its sealed ledger"
            )
        references.append(
            {
                "logical_name": logical,
                **entry,
                "bytes_verified": True,
            }
        )
    return references


def _load_capture_clock(
    capture_path: Path, manifest: Mapping[str, Any]
) -> tuple[list[int], Fraction, str]:
    try:
        from .video_capture import load_capture_npz

        capture = load_capture_npz(capture_path)
    except Exception as exc:
        raise ReviewBundleError(
            "CLOCK_ARTIFACT_INVALID", "Capture v1 could not reconstruct the review clock"
        ) from exc
    pts = [int(value) for value in capture.source_pts.tolist()]
    if not pts or len(pts) > MAX_FRAMES:
        raise ReviewBundleError("CLOCK_BOUNDS", "Capture frame count exceeds ReviewBundle v1")
    source_input = manifest.get("input")
    capture_manifest = manifest.get("capture")
    if not isinstance(source_input, dict):
        raise ReviewBundleError("INVALID_MANIFEST", "Performance input ledger is missing")
    if not isinstance(capture_manifest, dict):
        raise ReviewBundleError(
            "INVALID_MANIFEST", "Performance capture ledger is missing"
        )
    manifest_frames = _integer(capture_manifest.get("frames"), "capture.frames")
    if (
        capture.provenance.source_sha256 != source_input.get("sha256")
        or capture.provenance.source_bytes != source_input.get("bytes")
        or manifest_frames != len(pts)
    ):
        raise ReviewBundleError(
            "CLOCK_SOURCE_MISMATCH", "Capture clock is not bound to the performance input and frame count"
        )
    time_base = Fraction(
        capture.provenance.time_base_numerator,
        capture.provenance.time_base_denominator,
    )
    return pts, time_base, capture.schema_version


def _artifact_layer(logical_name: str) -> str | None:
    if logical_name in {
        "capture",
        "capture_jsonl",
        "performance_evidence",
        "pixel_observations",
        "observation_v3",
        "video_capture_run",
        "visual_track",
        "visual_track_summary",
        "capture_session",
        "viewer_media",
        "audio_video_timing",
        "audio_visual_timing_consumption",
        "audio_visual_source",
        "oral_validation",
        "oral_glb_validation",
        "performance_revision_chain",
    }:
        return "source"
    if logical_name.startswith("audio_visual_source_"):
        return "source"
    if logical_name in {"controls", "controls_jsonl", "retarget_calibration"}:
        return "visual_base"
    if logical_name in {
        "audio_visual_repair",
        "audio_visual_repair_arrays",
    }:
        return "audio_repair"
    if logical_name.startswith("acting_") or logical_name.startswith("direction_"):
        return "acting"
    if logical_name in {"mouth_aperture_edit", "mouth_aperture_edit_arrays"}:
        return "authored_correction"
    if logical_name == "physics" or logical_name.startswith("physics_"):
        return "physics"
    if logical_name in {
        "glb",
        "glb_mapping",
    }:
        return "final"
    return None


def _layer_available(layer_id: str, logical_names: Sequence[str]) -> bool:
    names = set(logical_names)
    if layer_id == "audio_repair":
        return {
            "audio_visual_repair",
            "audio_visual_repair_arrays",
        }.issubset(names)
    if layer_id == "authored_correction":
        return {
            "mouth_aperture_edit",
            "mouth_aperture_edit_arrays",
        }.issubset(names)
    if layer_id == "acting":
        return bool(
            names
            & {
                "acting_track",
                "acting_applied_controls",
                "body_track",
                "body_track_arrays",
            }
        )
    if layer_id == "final":
        return "glb" in names
    return bool(names)


def _motion_authority(layer_id: str, available: bool, changes_motion: bool) -> str:
    if layer_id == "source":
        return "reference_only"
    if not available or not changes_motion:
        return "none"
    return {
        "visual_base": "candidate_visual_retarget",
        "audio_repair": "candidate_lower_face_and_tongue_repair",
        "acting": "candidate_acting_override",
        "authored_correction": "candidate_bounded_artist_override",
        "physics": "candidate_simulation",
        "final": "candidate_composite",
    }[layer_id]


def _layer_revision_payload(layer: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in layer.items() if key != "revision_id"}


def _build_layers(
    artifact_names: Sequence[str],
    *,
    changes_motion_reported: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    grouped = {
        layer_id: sorted(
            name for name in artifact_names if _artifact_layer(name) == layer_id
        )
        for layer_id in LAYER_ORDER
    }
    final_available = bool(grouped["final"])
    reported = changes_motion_reported or {}
    layers: list[dict[str, Any]] = []
    previous_revision: str | None = None
    for layer_id in LAYER_ORDER:
        names = grouped[layer_id]
        available = _layer_available(layer_id, names)
        changes_motion = bool(available and reported.get(layer_id, False))
        consumes = [] if previous_revision is None else [previous_revision]
        layer: dict[str, Any] = {
            "schema_version": LAYER_SCHEMA_VERSION,
            "layer_id": layer_id,
            "layer_version": 1,
            "parent_revision_ids": consumes,
            "availability": "available" if available else "unavailable",
            "artifact_logical_names": names,
            "motion_authority": _motion_authority(
                layer_id, available, changes_motion
            ),
            "production_motion_authority": "none",
            "consumption": {
                "consumed_by_final_reported": bool(
                    final_available
                    and changes_motion
                    and layer_id
                    in {
                        "visual_base",
                        "audio_repair",
                        "acting",
                        "authored_correction",
                        "physics",
                    }
                ),
                "consumption_independently_verified": False,
            },
            "changes_motion_reported": changes_motion,
            "production_validated": False,
            "approval_status": "unapproved",
        }
        layer["revision_id"] = "review-revision:" + _sha_value(
            _layer_revision_payload(layer), label=f"Review layer {layer_id}"
        )
        previous_revision = layer["revision_id"]
        layers.append(layer)
    return layers


def _build_revision_graph(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nodes = [
        {
            "revision_id": layer["revision_id"],
            "layer_id": layer["layer_id"],
            "parent_revision_ids": list(layer["parent_revision_ids"]),
            "immutable": True,
            "production_validated": False,
            "approval_status": "unapproved",
        }
        for layer in layers
    ]
    edges = [
        {
            "from_revision_id": layers[index - 1]["revision_id"],
            "to_revision_id": layers[index]["revision_id"],
            "relation": "candidate_layer_composition",
        }
        for index in range(1, len(layers))
    ]
    final_revision_id = layers[-1]["revision_id"]
    return {
        "schema_version": REVISION_GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "ab_pairs": [],
        "ab_scope": "cross_bundle_same_comparison_key_only",
        "renderable_revisions": [
            {
                "revision_id": final_revision_id,
                "artifact_logical_name": "glb",
                "render_role": "final_textured_animation",
                "production_validated": False,
                "approval_status": "unapproved",
            }
        ],
        "immutable": True,
        "undo_redo_mode": "append_only_revision_selection",
        "production_validated": False,
    }


def _build_closeups(artifact_names: Sequence[str]) -> list[dict[str, Any]]:
    review_sources = [
        name for name in ("viewer_media", "glb") if name in artifact_names
    ]
    return [
        {
            "schema_version": CLOSEUP_SCHEMA_VERSION,
            "region_id": region,
            "region_version": 1,
            "selection_space": "semantic_face_region",
            "normalized_bounds": None,
            "selection_status": "native_selection_required",
            "renderable": False,
            "artifact_logical_names": review_sources,
            "production_validated": False,
            "approval_status": "unapproved",
        }
        for region in CLOSEUP_REGIONS
    ]


def _build_material_channels(
    manifest: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    character = model.get("character") if isinstance(model.get("character"), dict) else {}
    hashes = (
        character.get("runtime_material_sha256s")
        if isinstance(character.get("runtime_material_sha256s"), dict)
        else {}
    )
    output: list[dict[str, Any]] = []
    for channel, manifest_key, color_space in MATERIAL_CHANNELS:
        logical_candidates = (
            f"material_{manifest_key}",
            f"character_{manifest_key}",
            manifest_key,
        )
        logical = next((name for name in logical_candidates if name in references), None)
        referenced_hash = hashes.get(manifest_key)
        if referenced_hash is not None:
            referenced_hash = _sha(referenced_hash, f"model.character.runtime_material_sha256s.{manifest_key}")
        if logical is not None and referenced_hash is not None:
            if references[logical]["sha256"] != referenced_hash:
                raise ReviewBundleError(
                    "MATERIAL_BINDING_MISMATCH",
                    f"Material channel {channel} artifact differs from the character reference",
                )
        resolved_hash = references[logical]["sha256"] if logical is not None else referenced_hash
        status = (
            "sealed_artifact"
            if logical is not None
            else "hash_reference_only"
            if resolved_hash is not None
            else "unavailable"
        )
        output.append(
            {
                "schema_version": MATERIAL_SCHEMA_VERSION,
                "channel": channel,
                "manifest_key": manifest_key,
                "color_space": color_space,
                "status": status,
                "artifact_logical_name": logical,
                "sha256": resolved_hash,
                "isolatable": logical is not None,
                "measured": False,
                "production_validated": False,
                "approval_status": "unapproved",
            }
        )
    return output


def _build_correction(layers: Sequence[Mapping[str, Any]], frame_count: int) -> dict[str, Any]:
    by_id = {layer["layer_id"]: layer for layer in layers}
    return {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "candidate_request_eligible": False,
        "candidate_layer_id": "authored_correction",
        "required_parent_revision_id": by_id["final"]["revision_id"],
        "selection_requires_exact_source_pts": True,
        "immutable_revision_required": True,
        "undo_redo_mode": "append_only_revision_selection",
        "protected_anchor_classes": ["contact", "blink", "apex"],
        "writer_implemented": False,
        "production_revision_eligible": False,
        "human_review_recorded": False,
        "approval_status": "unapproved",
        "production_validated": False,
        "reason_codes": [
            "NATIVE_CORRECTION_WRITER_NOT_IMPLEMENTED",
            "CORRECTION_BRIDGE_MESSAGE_NOT_ENABLED",
            "HUMAN_REVIEW_NOT_RECORDED",
            "PRODUCTION_QUALIFICATION_NOT_ESTABLISHED",
        ],
    }


def _bridge_contract() -> dict[str, Any]:
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "allowed_message_types": list(BRIDGE_MESSAGE_TYPES),
        "message_version_required": True,
        "arbitrary_script_messages_allowed": False,
        "production_commands_enabled": False,
    }


def _build_comparison_key(
    *,
    input_sha256: str,
    clock: Mapping[str, Any],
    viewer_media_sha256: str,
    gnm_version: str,
    controls_identity_sha256: str,
    controls_performance_schema_version: str,
) -> dict[str, Any]:
    value = {
        "schema_version": COMPARISON_KEY_SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "clock_sha256": clock["clock_sha256"],
        "source_pts_sha256": _source_pts_sha256(clock["source_pts"]),
        "viewer_media_sha256": viewer_media_sha256,
        "gnm_version": gnm_version,
        "controls_performance_schema_version": controls_performance_schema_version,
        "controls_identity_sha256": controls_identity_sha256,
    }
    value["comparison_key_sha256"] = _sha_value(
        value, label="Review comparison key"
    )
    return value


def _reported_motion_from_revision_chain(
    *,
    artifact_paths: Mapping[str, str | Path],
    references: Mapping[str, Mapping[str, Any]],
    input_sha256: str,
    controls_bindings: Mapping[str, str],
) -> dict[str, bool]:
    """Fail-closed applied-motion claims from the sealed v1 revision chain.

    Retaining a report does not mean its edit was applied.  Any missing,
    malformed, unbound, or contradictory field makes the optional layer false;
    visual retarget and final controls remain the two intrinsic job revisions.
    """

    reported = {layer_id: False for layer_id in LAYER_ORDER}
    reported["visual_base"] = True
    reported["final"] = True
    if "performance_revision_chain" not in references:
        return reported
    try:
        chain = _object(
            _read_json_document(
                artifact_paths["performance_revision_chain"],
                label="Performance revision chain",
            ),
            "performance_revision_chain",
            (
                "chainConsistent",
                "finalPerformanceExpressionSha256",
                "immutableSourceMediaSha256",
                "productionValidated",
                "revisions",
                "schemaVersion",
                "sourcePtsSha256",
                "status",
            ),
        )
        if (
            chain["schemaVersion"] != "autoanim.performance-revision-chain.v1"
            or _boolean(chain["chainConsistent"], "revision_chain.chainConsistent")
            is not True
            or _boolean(
                chain["productionValidated"],
                "revision_chain.productionValidated",
            )
            is not False
            or _sha(
                chain["immutableSourceMediaSha256"],
                "revision_chain.immutableSourceMediaSha256",
            )
            != input_sha256
            or _sha(
                chain["sourcePtsSha256"], "revision_chain.sourcePtsSha256"
            )
            != controls_bindings["source_pts_sha256"]
            or _sha(
                chain["finalPerformanceExpressionSha256"],
                "revision_chain.finalPerformanceExpressionSha256",
            )
            != controls_bindings["final_expression_sha256"]
        ):
            return reported
        _string(chain["status"], "revision_chain.status", maximum=80)
        revisions = _sequence(
            chain["revisions"], "revision_chain.revisions", maximum=16
        )
        if len(revisions) != 3:
            return reported
        visual = _object(
            revisions[0],
            "revision_chain.revisions.0",
            ("inputAuthority", "name", "outputExpressionSha256"),
        )
        audio = _object(
            revisions[1],
            "revision_chain.revisions.1",
            (
                "applied",
                "inputExpressionSha256",
                "name",
                "outputExpressionSha256",
                "reportSha256",
            ),
        )
        correction = _object(
            revisions[2],
            "revision_chain.revisions.2",
            (
                "applied",
                "compositeInputSha256",
                "compositeOutputSha256",
                "inputExpressionSha256",
                "name",
                "outputExpressionSha256",
                "reportSha256",
            ),
        )
        if (
            visual["inputAuthority"] != "immutable_video_snapshot"
            or visual["name"] != "visual_video_retarget"
            or audio["name"] != "learned_audio_visual_repair"
            or correction["name"] != "authored_mouth_aperture"
        ):
            return reported
        visual_output = _sha(
            visual["outputExpressionSha256"],
            "revision_chain.visual.outputExpressionSha256",
        )
        audio_input = _sha(
            audio["inputExpressionSha256"],
            "revision_chain.audio.inputExpressionSha256",
        )
        audio_output = _sha(
            audio["outputExpressionSha256"],
            "revision_chain.audio.outputExpressionSha256",
        )
        correction_input = _sha(
            correction["inputExpressionSha256"],
            "revision_chain.correction.inputExpressionSha256",
        )
        correction_output = _sha(
            correction["outputExpressionSha256"],
            "revision_chain.correction.outputExpressionSha256",
        )
        audio_applied = _boolean(audio["applied"], "revision_chain.audio.applied")
        correction_applied = _boolean(
            correction["applied"], "revision_chain.correction.applied"
        )
        if (
            audio_input != visual_output
            or correction_input != audio_output
            or (not audio_applied and audio_output != audio_input)
            or (not correction_applied and correction_output != correction_input)
            or correction_output
            != controls_bindings["final_expression_sha256"]
        ):
            return reported

        if (
            audio_applied
            and {
                "audio_visual_repair",
                "audio_visual_repair_arrays",
            }.issubset(references)
            and isinstance(audio["reportSha256"], str)
            and audio["reportSha256"]
            == references["audio_visual_repair"]["sha256"]
        ):
            report = _object(
                _read_json_document(
                    artifact_paths["audio_visual_repair"],
                    label="Audio-visual repair report",
                ),
                "audio_visual_repair",
                (
                    "bindings",
                    "caveats",
                    "claims",
                    "clockJoin",
                    "config",
                    "locks",
                    "metrics",
                    "outputRole",
                    "policy",
                    "schemaVersion",
                    "sourceAuthority",
                    "status",
                ),
            )
            claims = report["claims"]
            bindings = report["bindings"]
            if not isinstance(claims, dict) or not isinstance(bindings, dict):
                raise ReviewBundleError(
                    "INVALID_REVISION_EVIDENCE",
                    "Audio repair claims and bindings must be objects",
                )
            repair_schema_policy = {
                "autoanim.audio-visual-repair.v1": (
                    "video_authoritative_conservative_audio_repair_v1"
                ),
                "autoanim.audio-visual-repair.v2": (
                    "video_authoritative_conservative_audio_repair_v2"
                ),
            }
            if (
                repair_schema_policy.get(report["schemaVersion"])
                == report["policy"]
                and report["status"] == "repaired"
                and _boolean(
                    claims.get("changesFinalGNMMotion"),
                    "audio_visual_repair.claims.changesFinalGNMMotion",
                )
                is True
                and _boolean(
                    claims.get("productionValidated"),
                    "audio_visual_repair.claims.productionValidated",
                )
                is False
                and _sha(
                    bindings.get("identitySha256"),
                    "audio_visual_repair.bindings.identitySha256",
                )
                == controls_bindings["identity_sha256"]
                and _sha(
                    bindings.get("inputExpressionSha256"),
                    "audio_visual_repair.bindings.inputExpressionSha256",
                )
                == audio_input
                and _sha(
                    bindings.get("outputExpressionSha256"),
                    "audio_visual_repair.bindings.outputExpressionSha256",
                )
                == audio_output
                and audio_input != audio_output
            ):
                reported["audio_repair"] = True

        if (
            correction_applied
            and {
                "mouth_aperture_edit",
                "mouth_aperture_edit_arrays",
            }.issubset(references)
            and isinstance(correction["reportSha256"], str)
            and correction["reportSha256"]
            == references["mouth_aperture_edit"]["sha256"]
        ):
            report = _object(
                _read_json_document(
                    artifact_paths["mouth_aperture_edit"],
                    label="Mouth aperture edit report",
                ),
                "mouth_aperture_edit",
                (
                    "author",
                    "authored_edit",
                    "bindings",
                    "claims",
                    "config",
                    "frame_reports",
                    "reason",
                    "schema_version",
                    "source_mode",
                    "status",
                    "summary",
                    "timeline",
                ),
            )
            bindings = report["bindings"]
            summary = report["summary"]
            claims = report["claims"]
            if not all(isinstance(value, dict) for value in (bindings, summary, claims)):
                raise ReviewBundleError(
                    "INVALID_REVISION_EVIDENCE",
                    "Mouth edit bindings, summary, and claims must be objects",
                )
            if (
                report["schema_version"]
                == "autoanim.gnm.mouth-aperture-correction.v3"
                and _boolean(
                    report["authored_edit"], "mouth_aperture_edit.authored_edit"
                )
                is True
                and report["status"] != "exact_noop"
                and _integer(
                    summary.get("corrected_frames"),
                    "mouth_aperture_edit.summary.corrected_frames",
                )
                > 0
                and _boolean(
                    claims.get("production_validated"),
                    "mouth_aperture_edit.claims.production_validated",
                )
                is False
                and _sha(
                    bindings.get("identity_sha256"),
                    "mouth_aperture_edit.bindings.identity_sha256",
                )
                == controls_bindings["identity_sha256"]
                and _sha(
                    bindings.get("base_expression_sha256"),
                    "mouth_aperture_edit.bindings.base_expression_sha256",
                )
                == correction_input
                and _sha(
                    bindings.get("revised_expression_sha256"),
                    "mouth_aperture_edit.bindings.revised_expression_sha256",
                )
                == correction_output
                and correction_input != correction_output
            ):
                reported["authored_correction"] = True
    except (ReviewBundleError, OSError, ValueError, KeyError, TypeError):
        return reported
    return reported


def build_review_bundle(
    performance_manifest_source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    artifact_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Reconstruct ReviewBundle v1 from one manifest and its exact artifact files."""

    manifest = _read_json_document(
        performance_manifest_source, label="Performance manifest"
    )
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("kind") != "video_performance"
        or manifest.get("status") != "succeeded"
    ):
        raise ReviewBundleError(
            "UNSUPPORTED_PERFORMANCE",
            "ReviewBundle v1 requires one successful video_performance manifest",
        )
    job_id = _identifier(manifest.get("job_id"), "job_id")
    integrity = _object(
        manifest.get("integrity"),
        "integrity",
        ("schema", "key_id", "signature"),
    )
    if integrity["schema"] != "autoanim.hmac-sha256.v1":
        raise ReviewBundleError("MANIFEST_UNSEALED", "Performance manifest seal schema is unsupported")
    key_id = _string(integrity["key_id"], "integrity.key_id", maximum=64)
    signature = _sha(integrity["signature"], "integrity.signature")
    input_entry = _object(
        manifest.get("input"),
        "input",
        ("name", "sha256", "bytes", "media_type"),
    )
    input_name = _string(input_entry["name"], "input.name", maximum=255)
    input_bytes = _integer(input_entry["bytes"], "input.bytes")
    if (
        PurePath(input_name).name != input_name
        or not 0 < input_bytes <= MAX_ARTIFACT_BYTES
    ):
        raise ReviewBundleError("INVALID_MANIFEST", "Performance input ledger is invalid")
    input_reference = {
        "name": input_name,
        "sha256": _sha(input_entry["sha256"], "input.sha256"),
        "bytes": input_bytes,
        "media_type": _string(input_entry["media_type"], "input.media_type", maximum=160),
        "bytes_verified": False,
    }

    ledger = _manifest_artifact_entries(manifest)
    references_list = _verify_artifacts(ledger, artifact_paths)
    references = {value["logical_name"]: value for value in references_list}
    required_artifacts = {"capture", "controls", "viewer_media", "glb"}
    if not required_artifacts.issubset(references):
        raise ReviewBundleError(
            "REVIEW_ARTIFACT_MISSING",
            "ReviewBundle v1 requires capture, controls, viewer_media, and glb artifacts",
        )
    pts, time_base, capture_schema = _load_capture_clock(
        Path(artifact_paths["capture"]), manifest
    )
    clock_payload = {
        "schema_version": CLOCK_SCHEMA_VERSION,
        "capture_schema_version": capture_schema,
        "cursor_unit": "source_pts",
        "time_base": _fraction_pair(time_base),
        "display_time_mapping": "(source_pts-first_source_pts)*time_base",
        "source_pts": pts,
        "frame_count": len(pts),
        "first_source_pts": pts[0],
        "last_source_pts": pts[-1],
        "first_display_time_exact_rational": [0, 1],
        "source_start_time_exact_rational": _fraction_pair(
            Fraction(pts[0]) * time_base
        ),
        "duration_exact_rational": _fraction_pair(
            Fraction(pts[-1] - pts[0]) * time_base
        ),
    }
    clock_payload["clock_sha256"] = _sha_value(
        clock_payload, label="Review clock"
    )
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ReviewBundleError(
            "INVALID_MANIFEST", "Performance model ledger is missing"
        )
    gnm_version = _identifier(model.get("gnm_version"), "model.gnm_version")
    if gnm_version != "3.0":
        raise ReviewBundleError(
            "UNSUPPORTED_GNM_VERSION", "ReviewBundle v1 requires GNM 3.0"
        )
    controls_identity_sha256, controls_schema, controls_bindings = _load_controls_identity(
        Path(artifact_paths["controls"]), expected_source_pts=pts
    )
    comparison_key = _build_comparison_key(
        input_sha256=input_reference["sha256"],
        clock=clock_payload,
        viewer_media_sha256=references["viewer_media"]["sha256"],
        gnm_version=gnm_version,
        controls_identity_sha256=controls_identity_sha256,
        controls_performance_schema_version=controls_schema,
    )

    artifact_names = [value["logical_name"] for value in references_list]
    reported_motion = _reported_motion_from_revision_chain(
        artifact_paths=artifact_paths,
        references=references,
        input_sha256=input_reference["sha256"],
        controls_bindings=controls_bindings,
    )
    layers = _build_layers(
        artifact_names, changes_motion_reported=reported_motion
    )
    revision_graph = _build_revision_graph(layers)
    closeups = _build_closeups(artifact_names)
    materials = _build_material_channels(manifest, references)
    correction = _build_correction(layers, len(pts))
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": {
            "schema_version": "1.0",
            "job_id": job_id,
            "kind": "video_performance",
            "status": "succeeded",
            "performance_manifest_sha256": _sha_value(
                manifest, label="Performance manifest binding"
            ),
            "manifest_seal": {
                "schema": integrity["schema"],
                "key_id": key_id,
                "signature": signature,
                "signature_verified": False,
            },
            "input": input_reference,
        },
        "clock": clock_payload,
        "comparison_key": comparison_key,
        "artifacts": references_list,
        "layers": layers,
        "revision_graph": revision_graph,
        "closeups": closeups,
        "material_channels": materials,
        "correction_eligibility": correction,
        "bridge": _bridge_contract(),
        "claims": {
            "artifact_ledger_bytes_verified": True,
            "exact_rational_pts_clock_verified": True,
            "manifest_signature_verified": False,
            "motion_consumption_independently_verified": False,
            "materials_approved": False,
            "correction_approved": False,
            "performance_approved": False,
            "production_validated": False,
            "publishable": False,
        },
        "limitations": list(LIMITATIONS),
    }
    document["bundle_sha256"] = review_bundle_payload_sha256(document)
    return load_review_bundle(document)


def _validate_artifact_references(value: Any) -> dict[str, dict[str, Any]]:
    items = _sequence(value, "artifacts", maximum=MAX_ARTIFACTS)
    if not items:
        raise ReviewBundleError("INVALID_ARTIFACT", "ReviewBundle requires artifacts")
    parsed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, raw in enumerate(items):
        field = f"artifacts.{index}"
        item = _object(
            raw,
            field,
            ("logical_name", "name", "bytes", "sha256", "media_type", "bytes_verified"),
        )
        logical = _identifier(item["logical_name"], f"{field}.logical_name")
        name = _string(item["name"], f"{field}.name", maximum=255)
        size = _integer(item["bytes"], f"{field}.bytes")
        if logical in parsed or PurePath(name).name != name or not 0 < size <= MAX_ARTIFACT_BYTES:
            raise ReviewBundleError("INVALID_ARTIFACT", f"{field} is duplicated or invalid")
        if _boolean(item["bytes_verified"], f"{field}.bytes_verified") is not True:
            raise ReviewBundleError("UNVERIFIED_ARTIFACT", f"{field} bytes are not verified")
        parsed[logical] = {
            "logical_name": logical,
            "name": name,
            "bytes": size,
            "sha256": _sha(item["sha256"], f"{field}.sha256"),
            "media_type": _string(item["media_type"], f"{field}.media_type", maximum=160),
            "bytes_verified": True,
        }
        order.append(logical)
    if order != sorted(order):
        raise ReviewBundleError("INVALID_ORDER", "ReviewBundle artifacts must be sorted")
    return parsed


def _validate_clock(value: Any) -> dict[str, Any]:
    clock = _object(
        value,
        "clock",
        (
            "schema_version",
            "capture_schema_version",
            "cursor_unit",
            "time_base",
            "display_time_mapping",
            "source_pts",
            "frame_count",
            "first_source_pts",
            "last_source_pts",
            "first_display_time_exact_rational",
            "source_start_time_exact_rational",
            "duration_exact_rational",
            "clock_sha256",
        ),
    )
    if (
        clock["schema_version"] != CLOCK_SCHEMA_VERSION
        or clock["capture_schema_version"] != "autoanim.capture.v1"
        or clock["cursor_unit"] != "source_pts"
        or clock["display_time_mapping"]
        != "(source_pts-first_source_pts)*time_base"
    ):
        raise ReviewBundleError("INVALID_CLOCK", "Review clock schema or cursor is invalid")
    time_base = _verified_rational(clock["time_base"], "clock.time_base", positive=True)
    raw_pts = _sequence(clock["source_pts"], "clock.source_pts")
    if not 1 <= len(raw_pts) <= MAX_FRAMES:
        raise ReviewBundleError(
            "CLOCK_BOUNDS", "Review clock exceeds the U1 source-PTS bound"
        )
    pts = tuple(
        _integer(item, f"clock.source_pts.{index}")
        for index, item in enumerate(raw_pts)
    )
    frame_count = _integer(clock["frame_count"], "clock.frame_count")
    first_source_pts = _integer(
        clock["first_source_pts"], "clock.first_source_pts"
    )
    last_source_pts = _integer(clock["last_source_pts"], "clock.last_source_pts")
    if (
        not pts
        or frame_count != len(pts)
        or any(right <= left for left, right in zip(pts, pts[1:]))
        or first_source_pts != pts[0]
        or last_source_pts != pts[-1]
    ):
        raise ReviewBundleError("INVALID_CLOCK", "Review clock PTS sequence is inconsistent")
    first = _verified_rational(
        clock["first_display_time_exact_rational"],
        "clock.first_display_time_exact_rational",
    )
    source_start = _verified_rational(
        clock["source_start_time_exact_rational"],
        "clock.source_start_time_exact_rational",
    )
    duration = _verified_rational(
        clock["duration_exact_rational"], "clock.duration_exact_rational"
    )
    if (
        first != 0
        or source_start != Fraction(pts[0]) * time_base
        or duration != Fraction(pts[-1] - pts[0]) * time_base
    ):
        raise ReviewBundleError("INVALID_CLOCK", "Review clock rational times do not match PTS")
    declared_hash = _sha(clock["clock_sha256"], "clock.clock_sha256")
    payload = deepcopy(clock)
    payload.pop("clock_sha256")
    if declared_hash != _sha_value(payload, label="Review clock"):
        raise ReviewBundleError("CLOCK_HASH_MISMATCH", "Review clock hash does not match")
    return clock


def _validate_source_manifest(value: Any) -> dict[str, Any]:
    source = _object(
        value,
        "source_manifest",
        (
            "schema_version",
            "job_id",
            "kind",
            "status",
            "performance_manifest_sha256",
            "manifest_seal",
            "input",
        ),
    )
    if source["schema_version"] != "1.0" or source["kind"] != "video_performance" or source["status"] != "succeeded":
        raise ReviewBundleError("INVALID_SOURCE", "ReviewBundle source manifest is invalid")
    _identifier(source["job_id"], "source_manifest.job_id")
    _sha(source["performance_manifest_sha256"], "source_manifest.performance_manifest_sha256")
    seal = _object(
        source["manifest_seal"],
        "source_manifest.manifest_seal",
        ("schema", "key_id", "signature", "signature_verified"),
    )
    if (
        seal["schema"] != "autoanim.hmac-sha256.v1"
        or not isinstance(seal["key_id"], str)
        or not seal["key_id"]
        or _sha(seal["signature"], "source_manifest.manifest_seal.signature") != seal["signature"]
        or _boolean(seal["signature_verified"], "source_manifest.manifest_seal.signature_verified") is not False
    ):
        raise ReviewBundleError("UNSUPPORTED_CLAIM", "ReviewBundle cannot verify the manifest signature")
    input_ref = _object(
        source["input"],
        "source_manifest.input",
        ("name", "sha256", "bytes", "media_type", "bytes_verified"),
    )
    name = _string(input_ref["name"], "source_manifest.input.name", maximum=255)
    if (
        PurePath(name).name != name
        or not (
            0
            < _integer(input_ref["bytes"], "source_manifest.input.bytes")
            <= MAX_ARTIFACT_BYTES
        )
        or _boolean(input_ref["bytes_verified"], "source_manifest.input.bytes_verified") is not False
    ):
        raise ReviewBundleError("INVALID_SOURCE", "ReviewBundle source input reference is invalid")
    _sha(input_ref["sha256"], "source_manifest.input.sha256")
    _string(input_ref["media_type"], "source_manifest.input.media_type", maximum=160)
    return source


def _validate_comparison_key(
    value: Any,
    *,
    source: Mapping[str, Any],
    clock: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparison = _object(
        value,
        "comparison_key",
        (
            "schema_version",
            "input_sha256",
            "clock_sha256",
            "source_pts_sha256",
            "viewer_media_sha256",
            "gnm_version",
            "controls_performance_schema_version",
            "controls_identity_sha256",
            "comparison_key_sha256",
        ),
    )
    required = {"controls", "viewer_media", "glb", "capture"}
    if not required.issubset(artifacts):
        raise ReviewBundleError(
            "REVIEW_ARTIFACT_MISSING",
            "Review comparison requires capture, controls, viewer_media, and glb",
        )
    if (
        comparison["schema_version"] != COMPARISON_KEY_SCHEMA_VERSION
        or _sha(comparison["input_sha256"], "comparison_key.input_sha256")
        != source["input"]["sha256"]
        or _sha(comparison["clock_sha256"], "comparison_key.clock_sha256")
        != clock["clock_sha256"]
        or _sha(
            comparison["source_pts_sha256"],
            "comparison_key.source_pts_sha256",
        )
        != _source_pts_sha256(clock["source_pts"])
        or _sha(
            comparison["viewer_media_sha256"],
            "comparison_key.viewer_media_sha256",
        )
        != artifacts["viewer_media"]["sha256"]
        or comparison["gnm_version"] != "3.0"
        or comparison["controls_performance_schema_version"]
        != "autoanim.gnm-performance.v3"
    ):
        raise ReviewBundleError(
            "COMPARISON_KEY_MISMATCH",
            "Review comparison key is not bound to source, clock, viewer, and GNM",
        )
    _sha(
        comparison["controls_identity_sha256"],
        "comparison_key.controls_identity_sha256",
    )
    declared = _sha(
        comparison["comparison_key_sha256"],
        "comparison_key.comparison_key_sha256",
    )
    payload = deepcopy(comparison)
    payload.pop("comparison_key_sha256")
    if declared != _sha_value(payload, label="Review comparison key"):
        raise ReviewBundleError(
            "COMPARISON_KEY_HASH_MISMATCH", "Review comparison key hash does not match"
        )
    return comparison


def _validate_layers(value: Any, artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
    layers = _sequence(value, "layers")
    if len(layers) != len(LAYER_ORDER):
        raise ReviewBundleError("INVALID_LAYERS", "ReviewBundle requires exactly seven layers")
    parsed: list[dict[str, Any]] = []
    previous_revision: str | None = None
    final_available = any(
        _artifact_layer(name) == "final" for name in artifacts
    )
    for index, (raw, expected_id) in enumerate(zip(layers, LAYER_ORDER, strict=True)):
        field = f"layers.{index}"
        layer = _object(
            raw,
            field,
            (
                "schema_version",
                "layer_id",
                "layer_version",
                "revision_id",
                "parent_revision_ids",
                "availability",
                "artifact_logical_names",
                "motion_authority",
                "production_motion_authority",
                "consumption",
                "changes_motion_reported",
                "production_validated",
                "approval_status",
            ),
        )
        expected_names = sorted(
            name for name in artifacts if _artifact_layer(name) == expected_id
        )
        names = _sequence(layer["artifact_logical_names"], f"{field}.artifact_logical_names")
        parents = _sequence(layer["parent_revision_ids"], f"{field}.parent_revision_ids")
        expected_parents = [] if previous_revision is None else [previous_revision]
        available = _layer_available(expected_id, expected_names)
        changes_motion = _boolean(
            layer["changes_motion_reported"],
            f"{field}.changes_motion_reported",
        )
        if (
            (expected_id in {"source", "acting", "physics"} and changes_motion)
            or (not available and changes_motion)
            or (
                expected_id in {"visual_base", "final"}
                and available
                and not changes_motion
            )
        ):
            raise ReviewBundleError(
                "UNSUPPORTED_CLAIM",
                f"{field} reports motion without supported sealed evidence",
            )
        consumption = _object(
            layer["consumption"],
            f"{field}.consumption",
            ("consumed_by_final_reported", "consumption_independently_verified"),
        )
        expected_consumed = bool(
            final_available
            and changes_motion
            and expected_id
            in {"visual_base", "audio_repair", "acting", "authored_correction", "physics"}
        )
        if (
            layer["schema_version"] != LAYER_SCHEMA_VERSION
            or layer["layer_id"] != expected_id
            or layer["layer_version"] != 1
            or names != expected_names
            or parents != expected_parents
            or layer["availability"] != ("available" if available else "unavailable")
            or layer["motion_authority"]
            != _motion_authority(expected_id, available, changes_motion)
            or layer["production_motion_authority"] != "none"
            or _boolean(consumption["consumed_by_final_reported"], f"{field}.consumption.consumed_by_final_reported") != expected_consumed
            or _boolean(consumption["consumption_independently_verified"], f"{field}.consumption.consumption_independently_verified") is not False
            or _boolean(layer["production_validated"], f"{field}.production_validated") is not False
            or layer["approval_status"] != "unapproved"
        ):
            raise ReviewBundleError("UNSUPPORTED_CLAIM", f"{field} is inconsistent or escalates authority")
        revision_id = _identifier(layer["revision_id"], f"{field}.revision_id")
        expected_revision = "review-revision:" + _sha_value(
            _layer_revision_payload(layer), label=f"Review layer {expected_id}"
        )
        if revision_id != expected_revision:
            raise ReviewBundleError("REVISION_HASH_MISMATCH", f"{field} revision ID is invalid")
        previous_revision = revision_id
        parsed.append(layer)
    return parsed


def _validate_materials(value: Any, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    channels = _sequence(value, "material_channels")
    if len(channels) != len(MATERIAL_CHANNELS):
        raise ReviewBundleError("INVALID_MATERIAL", "ReviewBundle material channels are incomplete")
    for index, (raw, expected) in enumerate(zip(channels, MATERIAL_CHANNELS, strict=True)):
        field = f"material_channels.{index}"
        item = _object(
            raw,
            field,
            (
                "schema_version",
                "channel",
                "manifest_key",
                "color_space",
                "status",
                "artifact_logical_name",
                "sha256",
                "isolatable",
                "measured",
                "production_validated",
                "approval_status",
            ),
        )
        channel, manifest_key, color_space = expected
        logical = _optional_string(item["artifact_logical_name"], f"{field}.artifact_logical_name")
        digest = None if item["sha256"] is None else _sha(item["sha256"], f"{field}.sha256")
        expected_status = "sealed_artifact" if logical else "hash_reference_only" if digest else "unavailable"
        if logical is not None and (logical not in artifacts or artifacts[logical]["sha256"] != digest):
            raise ReviewBundleError("MATERIAL_BINDING_MISMATCH", f"{field} artifact binding is invalid")
        if (
            item["schema_version"] != MATERIAL_SCHEMA_VERSION
            or (item["channel"], item["manifest_key"], item["color_space"]) != expected
            or item["status"] != expected_status
            or _boolean(item["isolatable"], f"{field}.isolatable") != (logical is not None)
            or _boolean(item["measured"], f"{field}.measured") is not False
            or _boolean(item["production_validated"], f"{field}.production_validated") is not False
            or item["approval_status"] != "unapproved"
        ):
            raise ReviewBundleError("UNSUPPORTED_CLAIM", f"{field} contains an unsupported material claim")


def load_review_bundle(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly load and semantically reconstruct ReviewBundle v1."""

    root = _object(
        _read_json_document(source, label="ReviewBundle"),
        "review_bundle",
        (
            "schema_version",
            "source_manifest",
            "clock",
            "comparison_key",
            "artifacts",
            "layers",
            "revision_graph",
            "closeups",
            "material_channels",
            "correction_eligibility",
            "bridge",
            "claims",
            "limitations",
            "bundle_sha256",
        ),
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ReviewBundleError("UNSUPPORTED_SCHEMA", "ReviewBundle schema is unsupported")
    source_manifest = _validate_source_manifest(root["source_manifest"])
    clock = _validate_clock(root["clock"])
    artifacts = _validate_artifact_references(root["artifacts"])
    _validate_comparison_key(
        root["comparison_key"],
        source=source_manifest,
        clock=clock,
        artifacts=artifacts,
    )
    layers = _validate_layers(root["layers"], artifacts)
    if root["revision_graph"] != _build_revision_graph(layers):
        raise ReviewBundleError("INVALID_REVISION_GRAPH", "ReviewBundle A/B graph is not deterministic")
    if root["closeups"] != _build_closeups(tuple(artifacts)):
        raise ReviewBundleError("INVALID_CLOSEUP", "ReviewBundle closeups are not fail-closed v1 regions")
    _validate_materials(root["material_channels"], artifacts)
    if root["correction_eligibility"] != _build_correction(layers, clock["frame_count"]):
        raise ReviewBundleError("UNSUPPORTED_CLAIM", "Correction eligibility is inconsistent")
    if root["bridge"] != _bridge_contract():
        raise ReviewBundleError("INVALID_BRIDGE", "Review bridge message contract is invalid")
    expected_claims = {
        "artifact_ledger_bytes_verified": True,
        "exact_rational_pts_clock_verified": True,
        "manifest_signature_verified": False,
        "motion_consumption_independently_verified": False,
        "materials_approved": False,
        "correction_approved": False,
        "performance_approved": False,
        "production_validated": False,
        "publishable": False,
    }
    if root["claims"] != expected_claims:
        raise ReviewBundleError("UNSUPPORTED_CLAIM", "ReviewBundle claims must remain fail closed")
    if root["limitations"] != list(LIMITATIONS):
        raise ReviewBundleError("INVALID_LIMITATIONS", "ReviewBundle limitations are incomplete")
    declared_hash = _sha(root["bundle_sha256"], "bundle_sha256")
    if declared_hash != review_bundle_payload_sha256(root):
        raise ReviewBundleError("BUNDLE_HASH_MISMATCH", "ReviewBundle payload hash does not match")
    return json.loads(_canonical_json(root, label="ReviewBundle").decode("utf-8"))


__all__ = [
    "BRIDGE_MESSAGE_TYPES",
    "CLOSEUP_REGIONS",
    "COMPARISON_KEY_SCHEMA_VERSION",
    "LAYER_ORDER",
    "MATERIAL_CHANNELS",
    "MAX_ARTIFACTS",
    "MAX_DOCUMENT_BYTES",
    "MAX_FRAMES",
    "ReviewBundleError",
    "SCHEMA_VERSION",
    "build_review_bundle",
    "load_review_bundle",
    "review_bundle_payload_sha256",
]
