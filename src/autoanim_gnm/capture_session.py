"""Deterministic, immutable capture provenance shared by production workflows.

This first schema is emitted by the video lane.  It binds the retained source,
Capture v1 detector output, Observation v2 compatibility evidence, and the new
pixel-derived Observation v3 sidecars without inferring subject identity,
neutrality, calibration, consent, or production validity.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .serialization import write_json
from .video_capture import (
    CaptureTrack,
    load_capture_npz,
    load_verified_capture_jsonl,
)
from .video_evidence import (
    PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
    REGION_LANDMARKS,
    load_verified_performance_evidence,
)
from .video_observation import (
    OBSERVATION_V3_POLICY,
    OBSERVATION_V3_SCHEMA_VERSION,
    PIXEL_ANALYZER_VERSION,
    PIXEL_OBSERVATION_SCHEMA_VERSION,
    REASON_CODES,
    PixelObservationTrack,
    load_pixel_observations,
    load_verified_observation_v3_summary,
)


CAPTURE_SESSION_SCHEMA_VERSION = "autoanim.capture-session.v1"
CAPTURE_SESSION_POLICY = "immutable_observation_provenance_no_claim_inference"
MAX_CAPTURE_SESSION_BYTES = 4 * 1024 * 1024

_VIDEO_ARTIFACT_SCHEMAS = {
    "capture": "autoanim.capture.v1",
    "capture_jsonl": "autoanim.capture.v1",
    "performance_evidence": PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
    "pixel_observations": PIXEL_OBSERVATION_SCHEMA_VERSION,
    "observation_v3": OBSERVATION_V3_SCHEMA_VERSION,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _decoded_sequence_sha256(observations: PixelObservationTrack) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(observations.decoded_pixel_sha256):
        digest.update(index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _artifact_record(logical_name: str, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Capture-session artifact {logical_name} is unavailable")
    return {
        "logical_name": logical_name,
        "name": path.name,
        "schema_version": _VIDEO_ARTIFACT_SCHEMAS[logical_name],
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _dataclass_values_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            equal_nan = left_value.dtype.kind in {"f", "c"}
            if not np.array_equal(left_value, right_value, equal_nan=equal_nan):
                return False
        elif left_value != right_value:
            return False
    return True


def build_video_capture_session(
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    *,
    artifact_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Build one deterministic, path-free video capture-session document."""

    observations.validate_capture(capture)
    if (
        Path(capture.provenance.source_name).name
        != capture.provenance.source_name
        or Path(capture.provenance.model_name).name
        != capture.provenance.model_name
    ):
        raise ValueError("CaptureSession source and model names must be basenames")
    for command in (
        capture.provenance.ffprobe_command,
        capture.provenance.ffmpeg_command,
    ):
        if any(
            Path(item).is_absolute() or item.startswith("file:")
            for item in command[1:]
        ):
            raise ValueError("CaptureSession command recipes must be path-free")
    if set(artifact_paths) != set(_VIDEO_ARTIFACT_SCHEMAS):
        raise ValueError("Capture-session artifact set does not match the video schema")
    artifacts = [
        _artifact_record(logical_name, Path(artifact_paths[logical_name]))
        for logical_name in _VIDEO_ARTIFACT_SCHEMAS
    ]
    return {
        "schema_version": CAPTURE_SESSION_SCHEMA_VERSION,
        "kind": "capture_session",
        "media_kind": "video",
        "policy": CAPTURE_SESSION_POLICY,
        "source_set": {
            "ordering": "single_job_input",
            "sources": [
                {
                    "source_id": "source-0001",
                    "job_input_ref": {"collection": "input", "index": 0},
                    "original_name": capture.provenance.source_name,
                    "source_sha256": capture.provenance.source_sha256,
                    "source_bytes": capture.provenance.source_bytes,
                    "role": "performance_video",
                    "usage": "performance_observation",
                    "frame_count": capture.frame_count,
                    "frame_size": [capture.width, capture.height],
                    "source_start_pts": int(capture.source_pts[0]),
                    "source_time_base": [
                        capture.provenance.time_base_numerator,
                        capture.provenance.time_base_denominator,
                    ],
                    "source_pts_sha256": _array_sha256(capture.source_pts, "<i8"),
                }
            ],
        },
        "pixel_streams": [
            {
                "stream_id": "decoded-rgb-0001",
                "source_id": "source-0001",
                "format": "rgb24",
                "frame_hash_domain": (
                    "rgb8_hwc_contiguous_after_declared_orientation"
                ),
                "frame_hashes_artifact_ref": "pixel_observations",
                "sequence_sha256": _decoded_sequence_sha256(observations),
                "relationship_to_detector_input": "redecoded_for_evidence",
                "detector_ingress_pixels_retained": False,
            }
        ],
        "detectors": [
            {
                "detector_id": "mediapipe-face-landmarker-video-v1",
                "implementation": "mediapipe.tasks.vision.FaceLandmarker",
                "model_name": capture.provenance.model_name,
                "model_sha256": capture.provenance.model_sha256,
                "runtime": {"mediapipe": capture.provenance.mediapipe_version},
                "running_mode": "VIDEO",
                "configuration": {
                    "num_faces": 1,
                    "confidence_thresholds": None,
                    "confidence_threshold_state": (
                        "not_retained_by_capture_v1"
                    ),
                    "output_face_blendshapes": True,
                    "output_facial_transformation_matrices": True,
                },
                "coordinate_space": "normalized_source_image_xyz",
                "landmark_schema": "mediapipe-face-landmarker-478",
                "identity_continuity_observable": False,
            }
        ],
        "observation_streams": artifacts,
        "calibration": {
            "presence": "absent",
            "type": "monocular_canonical_face_assumption",
            "validation_state": "not_applicable",
            "metric_scale_declared": False,
            "metric_scale_independently_validated": False,
        },
        "subject_binding": {
            "state": "unbound",
            "pseudonymous_subject_id": None,
            "evidence_ref": None,
            "evidence_sha256": None,
        },
        "assessments": {
            "neutrality": {
                "state": "unknown",
                "authority": "none",
                "production_validated": False,
            },
            "identity_continuity": {
                "state": "unknown",
                "authority": "none_num_faces_one_is_not_identity_evidence",
                "production_validated": False,
            },
        },
        "region_contract": {
            "regions": list(REGION_LANDMARKS),
            "reason_codes": list(REASON_CODES),
            "pixel_analyzer_version": PIXEL_ANALYZER_VERSION,
            "observation_v3_policy": OBSERVATION_V3_POLICY,
            "unknown_is_not_neutral": True,
        },
        "claims": {
            "metric_identity_validated": False,
            "neutrality_independently_confirmed": False,
            "identity_continuity_verified": False,
            "occlusion_validated": False,
            "measured_pbr_supported": False,
            "changes_final_gnm_motion": False,
            "production_validated": False,
        },
        "privacy": {
            "contains_biometric_observation_hashes": True,
            "contains_source_pixels": False,
            "retention_policy_bound": False,
            "consent_binding_present": False,
        },
        "caveats": [
            "The local JobStore HMAC is an integrity root for this workstation, not a portable publisher signature.",
            "Decoded-frame hashes and face observations are biometric-sensitive even though source pixels are not embedded.",
            "Capture v1 did not retain detector-ingress pixel hashes; Observation v3 re-decoded the exact retained source.",
            "Capture v1 did not retain the configured MediaPipe confidence thresholds; they remain unknown in this session.",
        ],
    }


def write_video_capture_session(
    path: str | Path,
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    *,
    artifact_paths: Mapping[str, str | Path],
) -> Path:
    return write_json(
        path,
        build_video_capture_session(
            capture,
            observations,
            artifact_paths=artifact_paths,
        ),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def load_verified_video_capture_session(
    path: str | Path,
    *,
    expected_capture: CaptureTrack,
    expected_observations: PixelObservationTrack,
    artifact_paths: Mapping[str, str | Path],
    artifact_contracts_preverified: bool = False,
) -> dict[str, Any]:
    """Reconstruct a video CaptureSession from exact source-side artifacts.

    ``artifact_contracts_preverified`` is reserved for a caller that already
    verified the sealed artifact ledger and every nested contract in this same
    request. It avoids retaining a second decompressed copy of the dense pixel
    track while still reconstructing the CaptureSession document and hashes.
    """

    source = Path(path)
    size = source.stat().st_size
    if size <= 0 or size > MAX_CAPTURE_SESSION_BYTES:
        raise ValueError("CaptureSession size is outside the accepted bounds")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("CaptureSession must be canonical UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("CaptureSession root must be an object")
    if set(artifact_paths) != set(_VIDEO_ARTIFACT_SCHEMAS):
        raise ValueError("Capture-session artifact set does not match the video schema")
    if artifact_contracts_preverified:
        loaded_capture = expected_capture
        loaded_observations = expected_observations
        loaded_observations.validate_capture(loaded_capture)
    else:
        loaded_capture = load_capture_npz(artifact_paths["capture"])
        if not _dataclass_values_equal(loaded_capture, expected_capture):
            raise ValueError(
                "CaptureSession Capture v1 artifact differs from expected capture"
            )
        loaded_observations = load_pixel_observations(
            artifact_paths["pixel_observations"]
        )
        if not _dataclass_values_equal(loaded_observations, expected_observations):
            raise ValueError(
                "CaptureSession pixel observations differ from expected observations"
            )
        loaded_observations.validate_capture(loaded_capture)
        load_verified_capture_jsonl(
            artifact_paths["capture_jsonl"], loaded_capture
        )
        load_verified_performance_evidence(
            artifact_paths["performance_evidence"],
            expected_source_sha256=loaded_capture.provenance.source_sha256,
            expected_frame_count=loaded_capture.frame_count,
            expected_capture=loaded_capture,
        )
        load_verified_observation_v3_summary(
            artifact_paths["observation_v3"],
            pixel_observations_path=artifact_paths["pixel_observations"],
            capture_artifact_path=artifact_paths["capture"],
            expected_capture=loaded_capture,
            expected_observations=loaded_observations,
        )
    expected = build_video_capture_session(
        loaded_capture,
        loaded_observations,
        artifact_paths=artifact_paths,
    )
    if payload != expected:
        raise ValueError("CaptureSession does not reconstruct from sealed artifacts")
    if (
        payload.get("schema_version") != CAPTURE_SESSION_SCHEMA_VERSION
        or payload.get("policy") != CAPTURE_SESSION_POLICY
        or payload.get("claims", {}).get("production_validated") is not False
        or payload.get("claims", {}).get("changes_final_gnm_motion") is not False
        or payload.get("subject_binding", {}).get("state") != "unbound"
        or payload.get("assessments", {})
        .get("identity_continuity", {})
        .get("state")
        != "unknown"
    ):
        raise ValueError("CaptureSession claims are not fail-closed")
    return payload


__all__ = [
    "CAPTURE_SESSION_POLICY",
    "CAPTURE_SESSION_SCHEMA_VERSION",
    "build_video_capture_session",
    "load_verified_video_capture_session",
    "write_video_capture_session",
]
