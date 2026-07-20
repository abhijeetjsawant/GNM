"""Provider-neutral, observation-only visual performance evidence.

VisualTrack v1 is a shadow artifact.  It binds the exact Capture v1 timeline
and Observation v3 pixel diagnostics into one dense, strictly validated
contract, but it is not an animation input.  Evidence that the current
MediaPipe path cannot provide (identity, covariance, reprojection residual,
and occlusion) is represented as unavailable/unknown, never as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from .errors import AutoAnimError
from .serialization import write_json, write_npz
from .video_capture import (
    CAPTURE_SCHEMA_VERSION,
    CaptureTrack,
    LANDMARK_COUNT,
    VideoCaptureRun,
)
from .video_observation import (
    PIXEL_ANALYZER_VERSION,
    PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
    REASON_CODES,
    PixelObservationTrack,
)


VISUAL_TRACK_SCHEMA_VERSION = "autoanim.visual-track/1.0"
VISUAL_TRACK_SUMMARY_SCHEMA_VERSION = "autoanim.visual-track-summary/1.0"
VISUAL_TRACK_POLICY = "shadow_observation_only_no_motion_effect_v1"
MOTION_AUTHORITY = "none"
MAX_VISUAL_TRACK_BYTES = 256 * 1024 * 1024
MAX_VISUAL_TRACK_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
MAX_VISUAL_TRACK_SUMMARY_BYTES = 4 * 1024 * 1024

REGION_NAMES = (
    "lips_contact",
    "jaw",
    "left_eyelid",
    "right_eyelid",
    "gaze",
    "brows",
    "cheeks_nose",
    "silhouette",
    "head",
    "tongue",
)
SUPPORTED_REGION_MAP = {"head": "head"}
UNSUPPORTED_REGION_NAMES = tuple(
    name for name in REGION_NAMES if name not in {*SUPPORTED_REGION_MAP, "tongue"}
)

MEASUREMENT_MISSING = np.uint8(0)
MEASUREMENT_OBSERVED = np.uint8(1)
OCCLUSION_MISSING = np.uint8(0)
OCCLUSION_UNKNOWN = np.uint8(1)
COVARIANCE_UNAVAILABLE = np.uint8(0)
SUBJECT_MISSING = np.uint8(0)
SUBJECT_OBSERVED_UNBOUND = np.uint8(1)
# Compatibility alias for callers of the initial V1.0a contract. Serialized
# value 1 means an observation exists while subject identity remains unbound;
# it never meant that identity selection was verified.
SUBJECT_SELECTED_UNBOUND = SUBJECT_OBSERVED_UNBOUND
REGION_MISSING = np.uint8(0)
REGION_PROVISIONAL_OBSERVED = np.uint8(1)
REGION_UNKNOWN = np.uint8(2)
CONFIDENCE_UNKNOWN = np.uint8(0)

REGION_UNSUPPORTED_REASON_BIT = np.uint64(1 << 32)
TONGUE_UNOBSERVED_REASON_BIT = np.uint64(1 << 33)
_SHA256_HEX = frozenset("0123456789abcdef")


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _parse_canonical_json(value: str) -> dict[str, Any]:
    parsed = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON number: {item}")
        ),
    )
    if not isinstance(parsed, dict) or _canonical_json(parsed) != value:
        raise ValueError("VisualTrack metadata must be canonical JSON")
    return parsed


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _SHA256_HEX
    )


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value).astype(dtype, copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"VisualTrack {label} members do not match the schema")
    return value


def _capture_tracks_equal(left: CaptureTrack, right: CaptureTrack) -> bool:
    for field in fields(CaptureTrack):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            if not np.array_equal(left_value, right_value, equal_nan=True):
                return False
        elif left_value != right_value:
            return False
    return True


def _metadata(
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    capture_run: VideoCaptureRun | None,
) -> dict[str, Any]:
    provenance = capture.provenance
    same_buffer = capture_run is not None
    detector_configuration = (
        capture_run.detector_configuration()
        if capture_run is not None
        else {
            "num_faces": 1,
            "confidence_thresholds": None,
            "confidence_threshold_state": "not_retained_by_capture_v1",
        }
    )
    return {
        "schema_version": VISUAL_TRACK_SCHEMA_VERSION,
        "kind": "visual_track",
        "policy": VISUAL_TRACK_POLICY,
        "motion_authority": MOTION_AUTHORITY,
        "consumed_by_retargeting": False,
        "source": {
            "name": provenance.source_name,
            "sha256": provenance.source_sha256,
            "bytes": provenance.source_bytes,
            "frame_count": capture.frame_count,
            "frame_size": [capture.width, capture.height],
            "source_start_pts": int(capture.source_pts[0]),
            "source_time_base": [
                provenance.time_base_numerator,
                provenance.time_base_denominator,
            ],
            "source_pts_sha256": _array_sha256(capture.source_pts, "<i8"),
            "coordinate_space": "normalized_display_oriented_source_image_xyz",
            "source_pixel_convention": "x_times_width_y_times_height",
            "capture_schema_version": capture.schema_version,
        },
        "provider": {
            "adapter_id": "autoanim.mediapipe-capture-v1-shadow-adapter",
            "implementation": "mediapipe.tasks.vision.FaceLandmarker",
            "model_name": provenance.model_name,
            "model_sha256": provenance.model_sha256,
            "runtime": {"mediapipe": provenance.mediapipe_version},
            "running_mode": "VIDEO",
            "configuration": detector_configuration,
            "landmark_schema": "mediapipe-face-landmarker-478",
            "detector_ingress_pixels_retained": False,
            "detector_ingress_hashes_retained": same_buffer,
            "evidence_pixels_relationship": (
                "per_frame_sha256_equal_to_detector_ingress"
                if same_buffer
                else "redecoded_for_evidence"
            ),
            "license_notice": {"state": "not_bound_by_capture_v1", "sha256": None},
            "calibration_profile": {"state": "absent", "sha256": None},
        },
        "identity": {
            "subject_binding_state": "unbound",
            "subject_id": None,
            "subject_epochs_available": False,
            "identity_continuity_state": "unknown",
            "identity_embedding_state": "not_computed",
            "character_revision_ref": None,
            "selected_character_is_subject_evidence": False,
            "subject_state_value_1_semantics": "observed_unbound_not_identity_selected",
        },
        "point_contract": {
            "point_count": LANDMARK_COUNT,
            "covariance_packing": ["xx", "xy", "xz", "yy", "yz", "zz"],
            "covariance_units": "normalized_image_xyz_squared",
            "covariance_state": "unavailable",
            "reprojection_residual_units": "source_pixels",
            "reprojection_residual_state": "unavailable",
            "occlusion_probability_state": "unknown",
            "unknown_is_not_zero": True,
        },
        "regional_contract": {
            "region_order": list(REGION_NAMES),
            "support_score_source": observations.analyzer_version,
            "support_score_cap": PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
            "support_score_is_calibrated_probability": False,
            "calibrated_confidence_state": "unknown",
            "tongue_state": "unknown_unobserved",
            "reason_codes": list(REASON_CODES)
            + ["REGION_UNSUPPORTED", "TONGUE_UNOBSERVED"],
            "region_unsupported_reason_bit": int(REGION_UNSUPPORTED_REASON_BIT),
            "tongue_unobserved_reason_bit": int(TONGUE_UNOBSERVED_REASON_BIT),
        },
        "epoch_contract": {
            "shot_epoch_authority": "provisional_pixel_cut_candidate",
            "tracking_epoch_authority": "autoanim_observation_continuity_not_provider_private_state",
            "tracking_epoch_starts": [
                "stream_start_if_detected",
                "shot_boundary_if_detected",
                "detection_reacquisition",
            ],
            "subject_epoch_authority": "none",
            "state_across_unknown_subject": "unbound",
        },
        "claims": {
            "changes_final_gnm_motion": False,
            "confidence_calibrated": False,
            "covariance_available": False,
            "occlusion_validated": False,
            "identity_continuity_validated": False,
            "tongue_observed": False,
            "production_validated": False,
        },
    }


def _expected_epochs(
    detected: np.ndarray, cut_candidate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    shot = np.cumsum(cut_candidate.astype(np.int32), dtype=np.int32)
    tracking = np.full(len(detected), -1, dtype=np.int32)
    epoch = -1
    for index, present in enumerate(detected):
        if not bool(present):
            continue
        if index == 0 or not bool(detected[index - 1]) or bool(cut_candidate[index]):
            epoch += 1
        tracking[index] = epoch
    return shot, tracking


@dataclass(frozen=True, slots=True)
class VisualTrack:
    """Immutable dense VisualTrack v1 shadow evidence."""

    metadata_json: str
    source_pts: np.ndarray
    evidence_rgb_sha256: tuple[str, ...]
    detected: np.ndarray
    cut_candidate: np.ndarray
    shot_epoch_index: np.ndarray
    tracking_epoch_index: np.ndarray
    subject_epoch_index: np.ndarray
    subject_state: np.ndarray
    point_xyz_normalized: np.ndarray
    point_xy_source_pixels: np.ndarray
    point_measurement_state: np.ndarray
    point_presence: np.ndarray
    point_visibility: np.ndarray
    point_occlusion_probability: np.ndarray
    point_occlusion_state: np.ndarray
    point_reprojection_residual_px: np.ndarray
    point_covariance_xyz_packed: np.ndarray
    point_covariance_state: np.ndarray
    region_names: tuple[str, ...]
    region_observation_state: np.ndarray
    region_support_score: np.ndarray
    region_calibrated_confidence: np.ndarray
    region_confidence_state: np.ndarray
    region_reason_mask: np.ndarray

    def __post_init__(self) -> None:
        exact_arrays = {
            "source_pts": (self.source_pts, np.int64),
            "detected": (self.detected, np.bool_),
            "cut_candidate": (self.cut_candidate, np.bool_),
            "shot_epoch_index": (self.shot_epoch_index, np.int32),
            "tracking_epoch_index": (self.tracking_epoch_index, np.int32),
            "subject_epoch_index": (self.subject_epoch_index, np.int32),
            "subject_state": (self.subject_state, np.uint8),
            "point_xyz_normalized": (self.point_xyz_normalized, np.float32),
            "point_xy_source_pixels": (self.point_xy_source_pixels, np.float32),
            "point_measurement_state": (self.point_measurement_state, np.uint8),
            "point_presence": (self.point_presence, np.float32),
            "point_visibility": (self.point_visibility, np.float32),
            "point_occlusion_probability": (
                self.point_occlusion_probability,
                np.float32,
            ),
            "point_occlusion_state": (self.point_occlusion_state, np.uint8),
            "point_reprojection_residual_px": (
                self.point_reprojection_residual_px,
                np.float32,
            ),
            "point_covariance_xyz_packed": (
                self.point_covariance_xyz_packed,
                np.float32,
            ),
            "point_covariance_state": (self.point_covariance_state, np.uint8),
            "region_observation_state": (self.region_observation_state, np.uint8),
            "region_support_score": (self.region_support_score, np.float32),
            "region_calibrated_confidence": (
                self.region_calibrated_confidence,
                np.float32,
            ),
            "region_confidence_state": (self.region_confidence_state, np.uint8),
            "region_reason_mask": (self.region_reason_mask, np.uint64),
        }
        for name, (value, dtype) in exact_arrays.items():
            object.__setattr__(self, name, _readonly(value, np.dtype(dtype)))
        object.__setattr__(self, "evidence_rgb_sha256", tuple(self.evidence_rgb_sha256))
        object.__setattr__(self, "region_names", tuple(self.region_names))
        metadata = _parse_canonical_json(self.metadata_json)
        count = len(self.source_pts)
        region_count = len(self.region_names)
        expected_shapes = {
            "detected": (count,),
            "cut_candidate": (count,),
            "shot_epoch_index": (count,),
            "tracking_epoch_index": (count,),
            "subject_epoch_index": (count,),
            "subject_state": (count,),
            "point_xyz_normalized": (count, LANDMARK_COUNT, 3),
            "point_xy_source_pixels": (count, LANDMARK_COUNT, 2),
            "point_measurement_state": (count, LANDMARK_COUNT),
            "point_presence": (count, LANDMARK_COUNT),
            "point_visibility": (count, LANDMARK_COUNT),
            "point_occlusion_probability": (count, LANDMARK_COUNT),
            "point_occlusion_state": (count, LANDMARK_COUNT),
            "point_reprojection_residual_px": (count, LANDMARK_COUNT),
            "point_covariance_xyz_packed": (count, LANDMARK_COUNT, 6),
            "point_covariance_state": (count, LANDMARK_COUNT),
            "region_observation_state": (count, region_count),
            "region_support_score": (count, region_count),
            "region_calibrated_confidence": (count, region_count),
            "region_confidence_state": (count, region_count),
            "region_reason_mask": (count, region_count),
        }
        if count <= 0 or self.region_names != REGION_NAMES:
            raise ValueError("VisualTrack frame or region metadata is invalid")
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"VisualTrack {name} must have shape {shape}")
        if len(self.evidence_rgb_sha256) != count or any(
            not _valid_sha256(value) for value in self.evidence_rgb_sha256
        ):
            raise ValueError("VisualTrack evidence pixel hashes are invalid")
        self._validate_metadata(metadata)
        self._validate_values(metadata)

    @property
    def frame_count(self) -> int:
        return len(self.source_pts)

    @property
    def metadata(self) -> dict[str, Any]:
        return _parse_canonical_json(self.metadata_json)

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        _require_keys(
            metadata,
            {
                "schema_version",
                "kind",
                "policy",
                "motion_authority",
                "consumed_by_retargeting",
                "source",
                "provider",
                "identity",
                "point_contract",
                "regional_contract",
                "epoch_contract",
                "claims",
            },
            "metadata",
        )
        source = _require_keys(
            metadata.get("source"),
            {
                "name",
                "sha256",
                "bytes",
                "frame_count",
                "frame_size",
                "source_start_pts",
                "source_time_base",
                "source_pts_sha256",
                "coordinate_space",
                "source_pixel_convention",
                "capture_schema_version",
            },
            "source",
        )
        provider = _require_keys(
            metadata.get("provider"),
            {
                "adapter_id",
                "implementation",
                "model_name",
                "model_sha256",
                "runtime",
                "running_mode",
                "configuration",
                "landmark_schema",
                "detector_ingress_pixels_retained",
                "detector_ingress_hashes_retained",
                "evidence_pixels_relationship",
                "license_notice",
                "calibration_profile",
            },
            "provider",
        )
        raw_configuration = provider.get("configuration")
        if not isinstance(raw_configuration, dict):
            raise ValueError("VisualTrack provider configuration must be an object")
        configuration = raw_configuration
        runtime = _require_keys(provider.get("runtime"), {"mediapipe"}, "runtime")
        license_notice = _require_keys(
            provider.get("license_notice"), {"state", "sha256"}, "license notice"
        )
        calibration = _require_keys(
            provider.get("calibration_profile"),
            {"state", "sha256"},
            "calibration profile",
        )
        identity = _require_keys(
            metadata.get("identity"),
            {
                "subject_binding_state",
                "subject_id",
                "subject_epochs_available",
                "identity_continuity_state",
                "identity_embedding_state",
                "character_revision_ref",
                "selected_character_is_subject_evidence",
                "subject_state_value_1_semantics",
            },
            "identity",
        )
        point_contract = _require_keys(
            metadata.get("point_contract"),
            {
                "point_count",
                "covariance_packing",
                "covariance_units",
                "covariance_state",
                "reprojection_residual_units",
                "reprojection_residual_state",
                "occlusion_probability_state",
                "unknown_is_not_zero",
            },
            "point contract",
        )
        regional = _require_keys(
            metadata.get("regional_contract"),
            {
                "region_order",
                "support_score_source",
                "support_score_cap",
                "support_score_is_calibrated_probability",
                "calibrated_confidence_state",
                "tongue_state",
                "reason_codes",
                "region_unsupported_reason_bit",
                "tongue_unobserved_reason_bit",
            },
            "regional contract",
        )
        epoch_contract = _require_keys(
            metadata.get("epoch_contract"),
            {
                "shot_epoch_authority",
                "tracking_epoch_authority",
                "tracking_epoch_starts",
                "subject_epoch_authority",
                "state_across_unknown_subject",
            },
            "epoch contract",
        )
        claims = _require_keys(
            metadata.get("claims"),
            {
                "changes_final_gnm_motion",
                "confidence_calibrated",
                "covariance_available",
                "occlusion_validated",
                "identity_continuity_validated",
                "tongue_observed",
                "production_validated",
            },
            "claims",
        )
        if (
            metadata.get("schema_version") != VISUAL_TRACK_SCHEMA_VERSION
            or metadata.get("kind") != "visual_track"
            or metadata.get("policy") != VISUAL_TRACK_POLICY
            or metadata.get("motion_authority") != MOTION_AUTHORITY
            or metadata.get("consumed_by_retargeting") is not False
            or not isinstance(source, dict)
            or not isinstance(source.get("frame_count"), int)
            or isinstance(source.get("frame_count"), bool)
            or source.get("frame_count") != self.frame_count
            or not isinstance(source.get("source_start_pts"), int)
            or isinstance(source.get("source_start_pts"), bool)
            or source.get("source_start_pts") != int(self.source_pts[0])
            or source.get("source_pts_sha256")
            != _array_sha256(self.source_pts, "<i8")
            or source.get("coordinate_space")
            != "normalized_display_oriented_source_image_xyz"
            or source.get("source_pixel_convention")
            != "x_times_width_y_times_height"
            or source.get("capture_schema_version") != CAPTURE_SCHEMA_VERSION
            or not _valid_sha256(source.get("sha256"))
            or not isinstance(source.get("bytes"), int)
            or isinstance(source.get("bytes"), bool)
            or source["bytes"] <= 0
            or not isinstance(source.get("frame_size"), list)
            or len(source["frame_size"]) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in source["frame_size"]
            )
            or not isinstance(source.get("source_time_base"), list)
            or len(source["source_time_base"]) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in source["source_time_base"]
            )
            or not isinstance(provider, dict)
            or provider.get("adapter_id")
            != "autoanim.mediapipe-capture-v1-shadow-adapter"
            or provider.get("implementation")
            != "mediapipe.tasks.vision.FaceLandmarker"
            or provider.get("running_mode") != "VIDEO"
            or provider.get("landmark_schema")
            != "mediapipe-face-landmarker-478"
            or not _valid_sha256(provider.get("model_sha256"))
            or not isinstance(runtime.get("mediapipe"), str)
            or not runtime["mediapipe"]
            or provider.get("detector_ingress_pixels_retained") is not False
            or not isinstance(provider.get("detector_ingress_hashes_retained"), bool)
            or license_notice
            != {"state": "not_bound_by_capture_v1", "sha256": None}
            or calibration != {"state": "absent", "sha256": None}
            or not isinstance(identity, dict)
            or identity.get("subject_binding_state") != "unbound"
            or identity.get("subject_id") is not None
            or identity.get("subject_epochs_available") is not False
            or identity.get("identity_continuity_state") != "unknown"
            or identity.get("identity_embedding_state") != "not_computed"
            or identity.get("character_revision_ref") is not None
            or identity.get("selected_character_is_subject_evidence") is not False
            or identity.get("subject_state_value_1_semantics")
            != "observed_unbound_not_identity_selected"
            or point_contract
            != {
                "point_count": LANDMARK_COUNT,
                "covariance_packing": ["xx", "xy", "xz", "yy", "yz", "zz"],
                "covariance_units": "normalized_image_xyz_squared",
                "covariance_state": "unavailable",
                "reprojection_residual_units": "source_pixels",
                "reprojection_residual_state": "unavailable",
                "occlusion_probability_state": "unknown",
                "unknown_is_not_zero": True,
            }
            or not isinstance(regional, dict)
            or regional.get("region_order") != list(REGION_NAMES)
            or regional.get("support_score_source") != PIXEL_ANALYZER_VERSION
            or regional.get("support_score_cap")
            != PIXEL_DIAGNOSTIC_CONFIDENCE_CAP
            or regional.get("support_score_is_calibrated_probability") is not False
            or regional.get("calibrated_confidence_state") != "unknown"
            or regional.get("tongue_state") != "unknown_unobserved"
            or regional.get("reason_codes")
            != list(REASON_CODES)
            + ["REGION_UNSUPPORTED", "TONGUE_UNOBSERVED"]
            or regional.get("region_unsupported_reason_bit")
            != int(REGION_UNSUPPORTED_REASON_BIT)
            or regional.get("tongue_unobserved_reason_bit")
            != int(TONGUE_UNOBSERVED_REASON_BIT)
            or epoch_contract
            != {
                "shot_epoch_authority": "provisional_pixel_cut_candidate",
                "tracking_epoch_authority": "autoanim_observation_continuity_not_provider_private_state",
                "tracking_epoch_starts": [
                    "stream_start_if_detected",
                    "shot_boundary_if_detected",
                    "detection_reacquisition",
                ],
                "subject_epoch_authority": "none",
                "state_across_unknown_subject": "unbound",
            }
            or not isinstance(claims, dict)
            or claims
            != {
                "changes_final_gnm_motion": False,
                "confidence_calibrated": False,
                "covariance_available": False,
                "occlusion_validated": False,
                "identity_continuity_validated": False,
                "tongue_observed": False,
                "production_validated": False,
            }
        ):
            raise ValueError("VisualTrack metadata is not fail-closed")
        hashes_retained = provider["detector_ingress_hashes_retained"]
        if hashes_retained:
            configuration = _require_keys(
                configuration,
                {
                    "num_faces",
                    "min_face_detection_confidence",
                    "min_face_presence_confidence",
                    "min_tracking_confidence",
                    "running_mode",
                    "output_face_blendshapes",
                    "output_facial_transformation_matrices",
                    "detector_ingress_hash_domain",
                },
                "same-buffer provider configuration",
            )
            thresholds = (
                configuration.get("min_face_detection_confidence"),
                configuration.get("min_face_presence_confidence"),
                configuration.get("min_tracking_confidence"),
            )
            if (
                type(configuration.get("num_faces")) is not int
                or configuration.get("num_faces") != 1
                or configuration.get("running_mode") != "VIDEO"
                or configuration.get("output_face_blendshapes") is not True
                or configuration.get("output_facial_transformation_matrices")
                is not True
                or configuration.get("detector_ingress_hash_domain")
                != "rgb8_hwc_contiguous_exact_mp_image_input"
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                    for value in thresholds
                )
                or provider.get("evidence_pixels_relationship")
                != "per_frame_sha256_equal_to_detector_ingress"
            ):
                raise ValueError(
                    "VisualTrack same-buffer detector configuration is invalid"
                )
        else:
            configuration = _require_keys(
                configuration,
                {
                    "num_faces",
                    "confidence_thresholds",
                    "confidence_threshold_state",
                },
                "legacy provider configuration",
            )
            if (
                type(configuration.get("num_faces")) is not int
                or configuration
                != {
                    "num_faces": 1,
                    "confidence_thresholds": None,
                    "confidence_threshold_state": "not_retained_by_capture_v1",
                }
                or provider.get("evidence_pixels_relationship")
                != "redecoded_for_evidence"
            ):
                raise ValueError("VisualTrack legacy detector configuration is invalid")
        if Path(str(source.get("name", ""))).name != source.get("name") or Path(
            str(provider.get("model_name", ""))
        ).name != provider.get("model_name"):
            raise ValueError("VisualTrack metadata names must be path-free basenames")

    def _validate_values(self, metadata: dict[str, Any]) -> None:
        if self.frame_count > 1 and np.any(np.diff(self.source_pts) <= 0):
            raise ValueError("VisualTrack source PTS must be strictly increasing")
        if bool(self.cut_candidate[0]):
            raise ValueError("VisualTrack first frame cannot be a cut candidate")
        expected_shot, expected_tracking = _expected_epochs(
            self.detected, self.cut_candidate
        )
        if not np.array_equal(self.shot_epoch_index, expected_shot) or not np.array_equal(
            self.tracking_epoch_index, expected_tracking
        ):
            raise ValueError("VisualTrack epochs are inconsistent")
        if np.any(self.subject_epoch_index != -1) or not np.array_equal(
            self.subject_state,
            np.where(
                self.detected, SUBJECT_OBSERVED_UNBOUND, SUBJECT_MISSING
            ).astype(np.uint8),
        ):
            raise ValueError("VisualTrack subject evidence is not unbound")
        observed_points = np.broadcast_to(
            self.detected[:, None], self.point_measurement_state.shape
        )
        if not np.array_equal(
            self.point_measurement_state,
            np.where(
                observed_points, MEASUREMENT_OBSERVED, MEASUREMENT_MISSING
            ).astype(np.uint8),
        ):
            raise ValueError("VisualTrack point measurement states are inconsistent")
        if (
            np.any(~np.isfinite(self.point_xyz_normalized[observed_points]))
            or np.any(np.isfinite(self.point_xyz_normalized[~observed_points]))
            or np.any(~np.isfinite(self.point_xy_source_pixels[observed_points]))
            or np.any(np.isfinite(self.point_xy_source_pixels[~observed_points]))
        ):
            raise ValueError("VisualTrack point availability is inconsistent")
        frame_size = np.asarray(metadata["source"]["frame_size"], dtype=np.float32)
        expected_pixels = self.point_xyz_normalized[:, :, :2] * frame_size
        if not np.array_equal(
            self.point_xy_source_pixels,
            expected_pixels,
            equal_nan=True,
        ):
            raise ValueError("VisualTrack source-pixel points differ from normalized points")
        for name in ("point_presence", "point_visibility"):
            values = getattr(self, name)
            finite = values[np.isfinite(values)]
            if np.any((finite < 0.0) | (finite > 1.0)) or np.any(
                np.isfinite(values[~observed_points])
            ):
                raise ValueError(f"VisualTrack {name} availability is invalid")
        if (
            np.any(np.isfinite(self.point_occlusion_probability))
            or np.any(np.isfinite(self.point_reprojection_residual_px))
            or np.any(np.isfinite(self.point_covariance_xyz_packed))
            or np.any(self.point_covariance_state != COVARIANCE_UNAVAILABLE)
            or not np.array_equal(
                self.point_occlusion_state,
                np.where(
                    observed_points, OCCLUSION_UNKNOWN, OCCLUSION_MISSING
                ).astype(np.uint8),
            )
        ):
            raise ValueError("VisualTrack unsupported point evidence must remain unknown")
        finite_support = self.region_support_score[
            np.isfinite(self.region_support_score)
        ]
        if np.any((finite_support < 0.0) | (finite_support > PIXEL_DIAGNOSTIC_CONFIDENCE_CAP)):
            raise ValueError("VisualTrack provisional regional support is invalid")
        if (
            np.any(np.isfinite(self.region_calibrated_confidence))
            or np.any(self.region_confidence_state != CONFIDENCE_UNKNOWN)
        ):
            raise ValueError("VisualTrack calibrated regional confidence must be unknown")
        supported_indices = [REGION_NAMES.index(name) for name in SUPPORTED_REGION_MAP]
        supported_state = self.region_observation_state[:, supported_indices]
        supported_score = self.region_support_score[:, supported_indices]
        expected_supported_state = np.where(
            ~self.detected[:, None],
            REGION_MISSING,
            np.where(
                np.isfinite(supported_score),
                REGION_PROVISIONAL_OBSERVED,
                REGION_UNKNOWN,
            ),
        ).astype(np.uint8)
        known_reason_mask = np.uint64((1 << len(REASON_CODES)) - 1)
        if (
            not np.array_equal(supported_state, expected_supported_state)
            or np.any(np.isfinite(supported_score[~self.detected]))
            or np.any(
                self.region_reason_mask[:, supported_indices] & ~known_reason_mask
            )
        ):
            raise ValueError("VisualTrack supported regional evidence is inconsistent")
        unsupported_indices = [
            REGION_NAMES.index(name) for name in UNSUPPORTED_REGION_NAMES
        ]
        expected_unsupported_state = np.broadcast_to(
            np.where(self.detected, REGION_UNKNOWN, REGION_MISSING)[:, None],
            (self.frame_count, len(unsupported_indices)),
        ).astype(np.uint8)
        if (
            not np.array_equal(
                self.region_observation_state[:, unsupported_indices],
                expected_unsupported_state,
            )
            or np.any(
                np.isfinite(self.region_support_score[:, unsupported_indices])
            )
            or np.any(
                self.region_reason_mask[self.detected][:, unsupported_indices]
                != REGION_UNSUPPORTED_REASON_BIT
            )
            or np.any(
                self.region_reason_mask[~self.detected][:, unsupported_indices]
                != np.uint64(1)
            )
        ):
            raise ValueError("VisualTrack unsupported regional evidence must remain unknown")
        tongue = len(REGION_NAMES) - 1
        expected_tongue_state = np.where(
            self.detected, REGION_UNKNOWN, REGION_MISSING
        ).astype(np.uint8)
        if (
            not np.array_equal(
                self.region_observation_state[:, tongue], expected_tongue_state
            )
            or np.any(np.isfinite(self.region_support_score[:, tongue]))
            or np.any(
                self.region_reason_mask[self.detected, tongue]
                != TONGUE_UNOBSERVED_REASON_BIT
            )
            or np.any(self.region_reason_mask[~self.detected, tongue] != np.uint64(1))
        ):
            raise ValueError("VisualTrack tongue evidence must remain unknown")

    def validate_inputs(
        self,
        capture: CaptureTrack,
        observations: PixelObservationTrack,
        *,
        expected_capture_run: VideoCaptureRun | None = None,
    ) -> None:
        expected = build_visual_track(
            capture, observations, capture_run=expected_capture_run
        )
        for field in fields(self):
            left = getattr(self, field.name)
            right = getattr(expected, field.name)
            if isinstance(left, np.ndarray):
                if not np.array_equal(left, right, equal_nan=True):
                    raise ValueError(
                        f"VisualTrack {field.name} does not reconstruct from source evidence"
                    )
            elif left != right:
                raise ValueError(
                    f"VisualTrack {field.name} does not reconstruct from source evidence"
                )


def build_visual_track(
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    *,
    capture_run: VideoCaptureRun | None = None,
) -> VisualTrack:
    """Build the V1.0a adapter without granting any channel motion authority."""

    observations.validate_capture(capture)
    if capture_run is not None:
        if not _capture_tracks_equal(capture_run.track, capture):
            raise ValueError("VideoCaptureRun track does not exactly match CaptureTrack")
        if tuple(capture_run.detector_ingress_rgb_sha256) != tuple(
            observations.decoded_pixel_sha256
        ):
            raise ValueError(
                "Detector-ingress hashes do not exactly match ObservationTrack decoded hashes"
            )
    count = capture.frame_count
    cut = np.asarray(observations.cut_candidate, dtype=np.bool_)
    shot, tracking = _expected_epochs(capture.detected, cut)
    subject_epoch = np.full(count, -1, dtype=np.int32)
    subject_state = np.where(
        capture.detected, SUBJECT_OBSERVED_UNBOUND, SUBJECT_MISSING
    ).astype(np.uint8)
    observed_points = np.broadcast_to(
        capture.detected[:, None], (count, LANDMARK_COUNT)
    )
    measurement_state = np.where(
        observed_points, MEASUREMENT_OBSERVED, MEASUREMENT_MISSING
    ).astype(np.uint8)
    point_pixels = np.asarray(capture.landmarks_xyz[:, :, :2], dtype=np.float32) * np.asarray(
        (capture.width, capture.height), dtype=np.float32
    )
    unknown_point = np.full((count, LANDMARK_COUNT), np.nan, dtype=np.float32)
    unknown_covariance = np.full(
        (count, LANDMARK_COUNT, 6), np.nan, dtype=np.float32
    )
    occlusion_state = np.where(
        observed_points, OCCLUSION_UNKNOWN, OCCLUSION_MISSING
    ).astype(np.uint8)
    region_count = len(REGION_NAMES)
    region_state = np.full((count, region_count), REGION_UNKNOWN, dtype=np.uint8)
    region_support = np.full((count, region_count), np.nan, dtype=np.float32)
    region_reasons = np.zeros((count, region_count), dtype=np.uint64)
    for target_name, source_name in SUPPORTED_REGION_MAP.items():
        target_index = REGION_NAMES.index(target_name)
        source_index = observations.region_names.index(source_name)
        available = np.isfinite(observations.confidence[:, source_index])
        region_state[:, target_index] = np.where(
            ~capture.detected,
            REGION_MISSING,
            np.where(available, REGION_PROVISIONAL_OBSERVED, REGION_UNKNOWN),
        ).astype(np.uint8)
        region_support[:, target_index] = observations.confidence[:, source_index]
        region_reasons[:, target_index] = observations.reason_mask[
            :, source_index
        ].astype(np.uint64)
    for region_name in UNSUPPORTED_REGION_NAMES:
        region_index = REGION_NAMES.index(region_name)
        region_state[:, region_index] = np.where(
            capture.detected, REGION_UNKNOWN, REGION_MISSING
        ).astype(np.uint8)
        region_reasons[capture.detected, region_index] = (
            REGION_UNSUPPORTED_REASON_BIT
        )
        region_reasons[~capture.detected, region_index] = np.uint64(1)
    tongue = region_count - 1
    region_state[:, tongue] = np.where(
        capture.detected, REGION_UNKNOWN, REGION_MISSING
    ).astype(np.uint8)
    region_reasons[capture.detected, tongue] = TONGUE_UNOBSERVED_REASON_BIT
    region_reasons[~capture.detected, tongue] = np.uint64(1)
    metadata = _metadata(capture, observations, capture_run)
    return VisualTrack(
        metadata_json=_canonical_json(metadata),
        source_pts=capture.source_pts,
        evidence_rgb_sha256=observations.decoded_pixel_sha256,
        detected=capture.detected,
        cut_candidate=cut,
        shot_epoch_index=shot,
        tracking_epoch_index=tracking,
        subject_epoch_index=subject_epoch,
        subject_state=subject_state,
        point_xyz_normalized=capture.landmarks_xyz,
        point_xy_source_pixels=point_pixels,
        point_measurement_state=measurement_state,
        point_presence=capture.landmark_presence,
        point_visibility=capture.landmark_visibility,
        point_occlusion_probability=unknown_point,
        point_occlusion_state=occlusion_state,
        point_reprojection_residual_px=unknown_point,
        point_covariance_xyz_packed=unknown_covariance,
        point_covariance_state=np.full(
            (count, LANDMARK_COUNT), COVARIANCE_UNAVAILABLE, dtype=np.uint8
        ),
        region_names=REGION_NAMES,
        region_observation_state=region_state,
        region_support_score=region_support,
        region_calibrated_confidence=np.full(
            (count, region_count), np.nan, dtype=np.float32
        ),
        region_confidence_state=np.full(
            (count, region_count), CONFIDENCE_UNKNOWN, dtype=np.uint8
        ),
        region_reason_mask=region_reasons,
    )


def write_visual_track(path: str | Path, track: VisualTrack) -> Path:
    return write_npz(
        path,
        metadata_json=np.asarray(track.metadata_json),
        source_pts=track.source_pts,
        evidence_rgb_sha256=np.asarray(track.evidence_rgb_sha256),
        detected=track.detected,
        cut_candidate=track.cut_candidate,
        shot_epoch_index=track.shot_epoch_index,
        tracking_epoch_index=track.tracking_epoch_index,
        subject_epoch_index=track.subject_epoch_index,
        subject_state=track.subject_state,
        point_xyz_normalized=track.point_xyz_normalized,
        point_xy_source_pixels=track.point_xy_source_pixels,
        point_measurement_state=track.point_measurement_state,
        point_presence=track.point_presence,
        point_visibility=track.point_visibility,
        point_occlusion_probability=track.point_occlusion_probability,
        point_occlusion_state=track.point_occlusion_state,
        point_reprojection_residual_px=track.point_reprojection_residual_px,
        point_covariance_xyz_packed=track.point_covariance_xyz_packed,
        point_covariance_state=track.point_covariance_state,
        region_names=np.asarray(track.region_names),
        region_observation_state=track.region_observation_state,
        region_support_score=track.region_support_score,
        region_calibrated_confidence=track.region_calibrated_confidence,
        region_confidence_state=track.region_confidence_state,
        region_reason_mask=track.region_reason_mask,
    )


_ARRAY_DTYPES = {
    "source_pts": np.dtype(np.int64),
    "detected": np.dtype(np.bool_),
    "cut_candidate": np.dtype(np.bool_),
    "shot_epoch_index": np.dtype(np.int32),
    "tracking_epoch_index": np.dtype(np.int32),
    "subject_epoch_index": np.dtype(np.int32),
    "subject_state": np.dtype(np.uint8),
    "point_xyz_normalized": np.dtype(np.float32),
    "point_xy_source_pixels": np.dtype(np.float32),
    "point_measurement_state": np.dtype(np.uint8),
    "point_presence": np.dtype(np.float32),
    "point_visibility": np.dtype(np.float32),
    "point_occlusion_probability": np.dtype(np.float32),
    "point_occlusion_state": np.dtype(np.uint8),
    "point_reprojection_residual_px": np.dtype(np.float32),
    "point_covariance_xyz_packed": np.dtype(np.float32),
    "point_covariance_state": np.dtype(np.uint8),
    "region_observation_state": np.dtype(np.uint8),
    "region_support_score": np.dtype(np.float32),
    "region_calibrated_confidence": np.dtype(np.float32),
    "region_confidence_state": np.dtype(np.uint8),
    "region_reason_mask": np.dtype(np.uint64),
}
_TEXT_KEYS = {"metadata_json", "evidence_rgb_sha256", "region_names"}
_EXPECTED_KEYS = frozenset(_ARRAY_DTYPES) | _TEXT_KEYS


def load_visual_track(path: str | Path) -> VisualTrack:
    source = Path(path)
    try:
        size = source.stat().st_size
        if size <= 0 or size > MAX_VISUAL_TRACK_BYTES:
            raise ValueError("VisualTrack exceeds its compressed byte limit")
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(set(names))
                or len(names) > 48
                or sum(member.file_size for member in members)
                > MAX_VISUAL_TRACK_UNCOMPRESSED_BYTES
            ):
                raise ValueError(
                    "VisualTrack has duplicate members or exceeds its resource limit"
                )
        with np.load(source, allow_pickle=False) as values:
            if set(values.files) != _EXPECTED_KEYS:
                raise ValueError("VisualTrack arrays do not match the schema")
            if any(values[name].dtype != dtype for name, dtype in _ARRAY_DTYPES.items()):
                raise ValueError("VisualTrack array dtype does not match the schema")
            if any(values[name].dtype.kind != "U" for name in _TEXT_KEYS):
                raise ValueError("VisualTrack text arrays must be Unicode")
            return VisualTrack(
                metadata_json=str(values["metadata_json"].item()),
                evidence_rgb_sha256=tuple(
                    str(value) for value in values["evidence_rgb_sha256"].tolist()
                ),
                region_names=tuple(
                    str(value) for value in values["region_names"].tolist()
                ),
                **{name: values[name] for name in _ARRAY_DTYPES},
            )
    except (
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid VisualTrack: {exc}") from exc


def _epoch_table(
    indices: np.ndarray, source_pts: np.ndarray, *, include_missing: bool
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    index = 0
    while index < len(indices):
        epoch = int(indices[index])
        if epoch < 0 and not include_missing:
            index += 1
            continue
        end = index + 1
        while end < len(indices) and int(indices[end]) == epoch:
            end += 1
        table.append(
            {
                "epochIndex": epoch,
                "startFrame": index,
                "endFrameExclusive": end,
                "startSourcePTS": int(source_pts[index]),
                "endSourcePTSInclusive": int(source_pts[end - 1]),
            }
        )
        index = end
    return table


def _optional_stat(values: np.ndarray, operation: str) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return float(np.median(finite) if operation == "median" else np.max(finite))


def build_visual_track_summary(track: VisualTrack) -> dict[str, Any]:
    metadata = track.metadata
    regions: dict[str, Any] = {}
    for region_index, name in enumerate(track.region_names):
        state = track.region_observation_state[:, region_index]
        regions[name] = {
            "provisionalObservedFrames": int(
                np.count_nonzero(state == REGION_PROVISIONAL_OBSERVED)
            ),
            "unknownFrames": int(np.count_nonzero(state == REGION_UNKNOWN)),
            "missingFrames": int(np.count_nonzero(state == REGION_MISSING)),
            "supportScoreMedian": _optional_stat(
                track.region_support_score[:, region_index], "median"
            ),
            "supportScoreMaximum": _optional_stat(
                track.region_support_score[:, region_index], "maximum"
            ),
            "calibratedConfidenceState": "unknown",
        }
    return {
        "schemaVersion": VISUAL_TRACK_SUMMARY_SCHEMA_VERSION,
        "kind": "visual_track_summary",
        "visualTrackSchemaVersion": VISUAL_TRACK_SCHEMA_VERSION,
        "policy": VISUAL_TRACK_POLICY,
        "motionAuthority": MOTION_AUTHORITY,
        "consumedByRetargeting": False,
        "source": metadata["source"],
        "provider": metadata["provider"],
        "identity": metadata["identity"],
        "epochs": {
            "shotAuthority": metadata["epoch_contract"]["shot_epoch_authority"],
            "shots": _epoch_table(
                track.shot_epoch_index, track.source_pts, include_missing=True
            ),
            "trackingAuthority": metadata["epoch_contract"][
                "tracking_epoch_authority"
            ],
            "tracking": _epoch_table(
                track.tracking_epoch_index, track.source_pts, include_missing=False
            ),
            "subjectAuthority": "none",
            "subjects": [],
        },
        "points": {
            "countPerObservedFrame": LANDMARK_COUNT,
            "observedFrames": int(np.count_nonzero(track.detected)),
            "missingFrames": int(track.frame_count - np.count_nonzero(track.detected)),
            "covarianceState": "unavailable",
            "reprojectionResidualState": "unavailable",
            "occlusionState": "unknown",
        },
        "regions": regions,
        "claims": metadata["claims"],
    }


def write_visual_track_summary(path: str | Path, track: VisualTrack) -> Path:
    return write_json(path, build_visual_track_summary(track))


def load_verified_visual_track_summary(
    path: str | Path,
    *,
    visual_track_path: str | Path,
    expected_capture: CaptureTrack,
    expected_observations: PixelObservationTrack,
    expected_capture_run: VideoCaptureRun | None = None,
) -> dict[str, Any]:
    source = Path(path)
    try:
        size = source.stat().st_size
        if size <= 0 or size > MAX_VISUAL_TRACK_SUMMARY_BYTES:
            raise ValueError("VisualTrack summary size is outside its bounds")
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {item}")
            ),
        )
        if not isinstance(payload, dict):
            raise ValueError("VisualTrack summary root must be an object")
        track = load_visual_track(visual_track_path)
        track.validate_inputs(
            expected_capture,
            expected_observations,
            expected_capture_run=expected_capture_run,
        )
        expected = build_visual_track_summary(track)
        if payload != expected:
            raise ValueError("VisualTrack summary does not reconstruct from dense evidence")
        return payload
    except AutoAnimError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AutoAnimError(
            "MEDIA_INVALID", f"Invalid VisualTrack summary: {exc}"
        ) from exc


__all__ = [
    "CONFIDENCE_UNKNOWN",
    "COVARIANCE_UNAVAILABLE",
    "MAX_VISUAL_TRACK_BYTES",
    "MAX_VISUAL_TRACK_SUMMARY_BYTES",
    "MAX_VISUAL_TRACK_UNCOMPRESSED_BYTES",
    "MOTION_AUTHORITY",
    "REGION_NAMES",
    "REGION_PROVISIONAL_OBSERVED",
    "REGION_UNSUPPORTED_REASON_BIT",
    "REGION_UNKNOWN",
    "SUBJECT_OBSERVED_UNBOUND",
    "SUBJECT_SELECTED_UNBOUND",
    "TONGUE_UNOBSERVED_REASON_BIT",
    "VISUAL_TRACK_POLICY",
    "VISUAL_TRACK_SCHEMA_VERSION",
    "VISUAL_TRACK_SUMMARY_SCHEMA_VERSION",
    "VisualTrack",
    "build_visual_track",
    "build_visual_track_summary",
    "load_verified_visual_track_summary",
    "load_visual_track",
    "write_visual_track",
    "write_visual_track_summary",
]
