"""Motion-inert bidirectional GNM sequence candidates for VisualTrack V2a.

The solver creates a review candidate from the already-retargeted GNM track.
It never returns a replacement performance object and its contract explicitly
forbids consumption by final retargeting.  Cuts, missing tracking epochs, and
known subject epochs are hard temporal boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from .errors import AutoAnimError
from .serialization import write_json, write_npz
from .visual_track import MOTION_AUTHORITY, REGION_NAMES
from .visual_track_calibration import VisualTrackCalibrationEvidence
from .visual_track_provider import (
    VisualTrackProviderResult,
    array_sha256,
    canonical_json,
    file_sha256,
    parse_canonical_json,
    valid_sha256,
)


VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION = "autoanim.video-sequence-candidate/1.0"
VIDEO_SEQUENCE_SUMMARY_SCHEMA_VERSION = "autoanim.video-sequence-summary/1.0"
VIDEO_SEQUENCE_POLICY = "bidirectional_shadow_candidate_never_shipped_v2a"
MAX_SEQUENCE_BYTES = 256 * 1024 * 1024
MAX_SEQUENCE_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
MAX_SEQUENCE_SUMMARY_BYTES = 2 * 1024 * 1024
DEFAULT_EXPRESSION_REGULARIZATION = 0.35
DEFAULT_POSE_REGULARIZATION = 0.20


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Sequence {label} members do not match the schema")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _regularized_solve(
    values: np.ndarray, strength: float, source_pts: np.ndarray
) -> np.ndarray:
    """Solve a first-difference objective weighted by exact PTS intervals."""

    source = np.asarray(values, dtype=np.float64)
    count = len(source)
    pts = np.asarray(source_pts, dtype=np.int64)
    if pts.shape != (count,) or (count > 1 and np.any(np.diff(pts) <= 0)):
        raise ValueError("Sequence solve needs one strictly increasing PTS per value")
    if count < 2 or strength == 0.0:
        return source.copy()
    intervals = np.diff(pts).astype(np.float64)
    reference_interval = float(np.median(intervals))
    edge_strength = strength * np.square(reference_interval / intervals)
    diagonal = np.ones(count, dtype=np.float64)
    diagonal[:-1] += edge_strength
    diagonal[1:] += edge_strength
    lower = -edge_strength
    upper = -edge_strength
    rhs = source.reshape(count, -1).copy()
    for index in range(1, count):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        rhs[index] -= factor * rhs[index - 1]
    solved = np.empty_like(rhs)
    solved[-1] = rhs[-1] / diagonal[-1]
    for index in range(count - 2, -1, -1):
        solved[index] = (
            rhs[index] - upper[index] * solved[index + 1]
        ) / diagonal[index]
    return solved.reshape(source.shape)


def _segments(provider: VisualTrackProviderResult) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    start = 0
    count = provider.frame_count
    while start < count:
        if not provider.detected[start] or provider.tracking_epoch_index[start] < 0:
            result.append((start, start + 1))
            start += 1
            continue
        key = (
            int(provider.shot_epoch_index[start]),
            int(provider.tracking_epoch_index[start]),
            int(provider.subject_epoch_index[start]),
        )
        stop = start + 1
        while stop < count:
            next_key = (
                int(provider.shot_epoch_index[stop]),
                int(provider.tracking_epoch_index[stop]),
                int(provider.subject_epoch_index[stop]),
            )
            if not provider.detected[stop] or next_key != key:
                break
            stop += 1
        result.append((start, stop))
        start = stop
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VideoSequenceCandidate:
    metadata_json: str
    source_pts: np.ndarray
    shot_epoch_index: np.ndarray
    tracking_epoch_index: np.ndarray
    subject_epoch_index: np.ndarray
    hard_anchor_mask: np.ndarray
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "source_pts": (self.source_pts, np.int64),
            "shot_epoch_index": (self.shot_epoch_index, np.int32),
            "tracking_epoch_index": (self.tracking_epoch_index, np.int32),
            "subject_epoch_index": (self.subject_epoch_index, np.int32),
            "hard_anchor_mask": (self.hard_anchor_mask, np.bool_),
            "expression": (self.expression, np.float32),
            "rotations": (self.rotations, np.float32),
            "translation": (self.translation, np.float32),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, _readonly(value, np.dtype(dtype)))
        count = len(self.source_pts)
        if count <= 0:
            raise ValueError("Sequence candidate cannot be empty")
        expected = {
            "shot_epoch_index": (count,),
            "tracking_epoch_index": (count,),
            "subject_epoch_index": (count,),
            "hard_anchor_mask": (count,),
            "expression": (count, 383),
            "rotations": (count, 4, 3),
            "translation": (count, 3),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"Sequence {name} must have shape {shape}")
        if self.frame_count > 1 and np.any(np.diff(self.source_pts) <= 0):
            raise ValueError("Sequence source PTS must be strictly increasing")
        if not all(
            np.isfinite(value).all()
            for value in (self.expression, self.rotations, self.translation)
        ):
            raise ValueError("Sequence candidate controls must be finite")
        self._validate_metadata(parse_canonical_json(self.metadata_json))

    @property
    def frame_count(self) -> int:
        return len(self.source_pts)

    @property
    def metadata(self) -> dict[str, Any]:
        return parse_canonical_json(self.metadata_json)

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        _require_keys(
            metadata,
            {
                "schema_version",
                "kind",
                "status",
                "policy",
                "motion_authority",
                "consumed_by_retargeting",
                "bindings",
                "timeline",
                "solver",
                "regional_assessment",
                "claims",
                "limitations",
            },
            "metadata",
        )
        bindings = _require_keys(
            metadata["bindings"],
            {
                "capture_source_sha256",
                "source_pts_sha256",
                "provider_result_sha256",
                "calibration_evidence_sha256",
                "baseline_expression_sha256",
                "baseline_rotations_sha256",
                "baseline_translation_sha256",
                "shipped_expression_sha256",
                "shipped_rotations_sha256",
                "shipped_translation_sha256",
                "candidate_expression_sha256",
                "candidate_rotations_sha256",
                "candidate_translation_sha256",
            },
            "bindings",
        )
        timeline = _require_keys(
            metadata["timeline"],
            {
                "frame_count",
                "source_start_pts",
                "exact_pts_retained",
                "boundary_count",
                "boundaries",
            },
            "timeline",
        )
        solver = _require_keys(
            metadata["solver"],
            {
                "mode",
                "objective",
                "expression_regularization",
                "pose_regularization",
                "hard_anchor_count",
                "hard_anchor_scope",
                "identity_state",
                "provider_confidence_used",
                "time_weighting",
            },
            "solver",
        )
        regional = _require_keys(
            metadata["regional_assessment"], {"order", "state", "reason"}, "regional assessment"
        )
        claims = _require_keys(
            metadata["claims"],
            {
                "changes_final_gnm_motion",
                "candidate_is_shipped",
                "final_output_arrays_byte_identical",
                "grants_motion_authority",
                "production_validated",
            },
            "claims",
        )
        if (
            metadata["schema_version"] != VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION
            or metadata["kind"] != "video_sequence_candidate"
            or metadata["status"] != "shadow_unqualified"
            or metadata["policy"] != VIDEO_SEQUENCE_POLICY
            or metadata["motion_authority"] != MOTION_AUTHORITY
            or metadata["consumed_by_retargeting"] is not False
            or timeline["frame_count"] != self.frame_count
            or timeline["source_start_pts"] != int(self.source_pts[0])
            or timeline["exact_pts_retained"] is not True
            or not isinstance(timeline["boundaries"], list)
            or timeline["boundary_count"] != len(timeline["boundaries"])
            or solver["mode"] != "offline_bidirectional_shadow"
            or solver["objective"]
            != "data_fidelity_plus_exact_pts_velocity_regularization"
            or solver["identity_state"] != "frozen_external_not_serialized"
            or solver["hard_anchor_scope"] != "expression_only"
            or solver["provider_confidence_used"] is not False
            or solver["time_weighting"]
            != "exact_source_pts_normalized_by_segment_median_interval"
            or solver["hard_anchor_count"] != int(np.count_nonzero(self.hard_anchor_mask))
            or regional["order"] != list(REGION_NAMES)
            or regional["state"] != "unknown_unqualified"
            or not isinstance(regional["reason"], str)
            or not regional["reason"]
            or claims
            != {
                "changes_final_gnm_motion": False,
                "candidate_is_shipped": False,
                "final_output_arrays_byte_identical": True,
                "grants_motion_authority": False,
                "production_validated": False,
            }
        ):
            raise ValueError("Sequence metadata is inconsistent or not fail-closed")
        if not isinstance(metadata["limitations"], list) or any(
            not isinstance(value, str) or not value for value in metadata["limitations"]
        ):
            raise ValueError("Sequence limitations are invalid")
        for key, value in bindings.items():
            if not valid_sha256(value):
                raise ValueError(f"Sequence binding {key} is invalid")
        if bindings["source_pts_sha256"] != array_sha256(self.source_pts):
            raise ValueError("Sequence source PTS hash does not reconstruct")
        if (
            bindings["baseline_expression_sha256"]
            != bindings["shipped_expression_sha256"]
            or bindings["baseline_rotations_sha256"]
            != bindings["shipped_rotations_sha256"]
            or bindings["baseline_translation_sha256"]
            != bindings["shipped_translation_sha256"]
        ):
            raise ValueError("Sequence shadow lane changed shipped motion hashes")
        if (
            bindings["candidate_expression_sha256"]
            != array_sha256(self.expression)
            or bindings["candidate_rotations_sha256"]
            != array_sha256(self.rotations)
            or bindings["candidate_translation_sha256"]
            != array_sha256(self.translation)
        ):
            raise ValueError("Sequence candidate hashes do not reconstruct")
        expected_boundaries: list[dict[str, int]] = []
        start = 0
        while start < self.frame_count:
            if self.tracking_epoch_index[start] < 0:
                stop = start + 1
            else:
                key = (
                    int(self.shot_epoch_index[start]),
                    int(self.tracking_epoch_index[start]),
                    int(self.subject_epoch_index[start]),
                )
                stop = start + 1
                while stop < self.frame_count:
                    next_key = (
                        int(self.shot_epoch_index[stop]),
                        int(self.tracking_epoch_index[stop]),
                        int(self.subject_epoch_index[stop]),
                    )
                    if self.tracking_epoch_index[stop] < 0 or next_key != key:
                        break
                    stop += 1
            expected_boundaries.append(
                {"start_frame": start, "end_frame_exclusive": stop}
            )
            start = stop
        if timeline["boundaries"] != expected_boundaries:
            raise ValueError("Sequence boundaries do not reconstruct from epochs")
        for boundary in timeline["boundaries"]:
            if (
                not isinstance(boundary, dict)
                or set(boundary) != {"start_frame", "end_frame_exclusive"}
                or not isinstance(boundary["start_frame"], int)
                or not isinstance(boundary["end_frame_exclusive"], int)
                or not 0
                <= boundary["start_frame"]
                < boundary["end_frame_exclusive"]
                <= self.frame_count
            ):
                raise ValueError("Sequence boundary is invalid")

    def validate_inputs(
        self,
        provider: VisualTrackProviderResult,
        calibration: VisualTrackCalibrationEvidence,
        *,
        provider_result_sha256: str,
        calibration_evidence_sha256: str,
        expression: np.ndarray,
        rotations: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        bindings = self.metadata["bindings"]
        if (
            not np.array_equal(self.source_pts, provider.source_pts)
            or not np.array_equal(self.shot_epoch_index, provider.shot_epoch_index)
            or not np.array_equal(self.tracking_epoch_index, provider.tracking_epoch_index)
            or not np.array_equal(self.subject_epoch_index, provider.subject_epoch_index)
            or bindings["provider_result_sha256"] != provider_result_sha256
            or bindings["calibration_evidence_sha256"] != calibration_evidence_sha256
            or bindings["baseline_expression_sha256"] != array_sha256(expression)
            or bindings["baseline_rotations_sha256"] != array_sha256(rotations)
            or bindings["baseline_translation_sha256"] != array_sha256(translation)
            or calibration.payload["provider_binding"]["provider_result_sha256"]
            != provider_result_sha256
        ):
            raise ValueError("Sequence candidate does not bind its exact inputs")


def build_shadow_video_sequence_candidate(
    provider: VisualTrackProviderResult,
    calibration: VisualTrackCalibrationEvidence,
    *,
    provider_result_sha256: str,
    calibration_evidence_sha256: str,
    expression: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
    hard_anchor_mask: np.ndarray | None = None,
    expression_regularization: float = DEFAULT_EXPRESSION_REGULARIZATION,
    pose_regularization: float = DEFAULT_POSE_REGULARIZATION,
) -> VideoSequenceCandidate:
    """Build an offline comparison candidate without changing input arrays."""

    if not valid_sha256(provider_result_sha256) or not valid_sha256(
        calibration_evidence_sha256
    ):
        raise ValueError("Sequence artifact bindings must be SHA-256")
    if (
        calibration.payload["provider_binding"]["provider_result_sha256"]
        != provider_result_sha256
        or calibration.payload["provider_binding"]["source_pts_sha256"]
        != array_sha256(provider.source_pts)
    ):
        raise ValueError("Sequence calibration does not bind the provider")
    count = provider.frame_count
    baseline_expression = np.asarray(expression)
    baseline_rotations = np.asarray(rotations)
    baseline_translation = np.asarray(translation)
    if (
        baseline_expression.shape != (count, 383)
        or baseline_rotations.shape != (count, 4, 3)
        or baseline_translation.shape != (count, 3)
        or not all(
            np.isfinite(value).all()
            for value in (baseline_expression, baseline_rotations, baseline_translation)
        )
    ):
        raise ValueError("Sequence baseline GNM controls are invalid")
    if (
        not np.isfinite(expression_regularization)
        or not np.isfinite(pose_regularization)
        or expression_regularization < 0
        or pose_regularization < 0
        or expression_regularization > 10
        or pose_regularization > 10
    ):
        raise ValueError("Sequence regularization is invalid")
    anchors = (
        np.zeros(count, dtype=np.bool_)
        if hard_anchor_mask is None
        else np.asarray(hard_anchor_mask, dtype=np.bool_)
    )
    if anchors.shape != (count,):
        raise ValueError("Sequence hard anchors must have one value per frame")
    before_hashes = (
        array_sha256(baseline_expression),
        array_sha256(baseline_rotations),
        array_sha256(baseline_translation),
    )
    candidate_expression = np.array(baseline_expression, dtype=np.float64, copy=True)
    candidate_rotations = np.array(baseline_rotations, dtype=np.float64, copy=True)
    candidate_translation = np.array(baseline_translation, dtype=np.float64, copy=True)
    segments = _segments(provider)
    for start, stop in segments:
        if stop - start < 2 or not provider.detected[start]:
            continue
        candidate_expression[start:stop] = _regularized_solve(
            baseline_expression[start:stop],
            expression_regularization,
            provider.source_pts[start:stop],
        )
        candidate_rotations[start:stop] = _regularized_solve(
            baseline_rotations[start:stop],
            pose_regularization,
            provider.source_pts[start:stop],
        )
        candidate_translation[start:stop] = _regularized_solve(
            baseline_translation[start:stop],
            pose_regularization,
            provider.source_pts[start:stop],
        )
    candidate_expression[anchors] = baseline_expression[anchors]
    stored_expression = np.asarray(candidate_expression, dtype=np.float32)
    stored_rotations = np.asarray(candidate_rotations, dtype=np.float32)
    stored_translation = np.asarray(candidate_translation, dtype=np.float32)
    after_hashes = (
        array_sha256(baseline_expression),
        array_sha256(baseline_rotations),
        array_sha256(baseline_translation),
    )
    if after_hashes != before_hashes:
        raise RuntimeError("Shadow sequence solve mutated baseline GNM arrays")
    metadata = {
        "schema_version": VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION,
        "kind": "video_sequence_candidate",
        "status": "shadow_unqualified",
        "policy": VIDEO_SEQUENCE_POLICY,
        "motion_authority": MOTION_AUTHORITY,
        "consumed_by_retargeting": False,
        "bindings": {
            "capture_source_sha256": provider.metadata["bindings"]["capture_source_sha256"],
            "source_pts_sha256": array_sha256(provider.source_pts),
            "provider_result_sha256": provider_result_sha256,
            "calibration_evidence_sha256": calibration_evidence_sha256,
            "baseline_expression_sha256": before_hashes[0],
            "baseline_rotations_sha256": before_hashes[1],
            "baseline_translation_sha256": before_hashes[2],
            "shipped_expression_sha256": after_hashes[0],
            "shipped_rotations_sha256": after_hashes[1],
            "shipped_translation_sha256": after_hashes[2],
            "candidate_expression_sha256": array_sha256(stored_expression),
            "candidate_rotations_sha256": array_sha256(stored_rotations),
            "candidate_translation_sha256": array_sha256(stored_translation),
        },
        "timeline": {
            "frame_count": count,
            "source_start_pts": int(provider.source_pts[0]),
            "exact_pts_retained": True,
            "boundary_count": len(segments),
            "boundaries": [
                {"start_frame": start, "end_frame_exclusive": stop}
                for start, stop in segments
            ],
        },
        "solver": {
            "mode": "offline_bidirectional_shadow",
            "objective": "data_fidelity_plus_exact_pts_velocity_regularization",
            "expression_regularization": float(expression_regularization),
            "pose_regularization": float(pose_regularization),
            "hard_anchor_count": int(np.count_nonzero(anchors)),
            "hard_anchor_scope": "expression_only",
            "identity_state": "frozen_external_not_serialized",
            "provider_confidence_used": False,
            "time_weighting": (
                "exact_source_pts_normalized_by_segment_median_interval"
            ),
        },
        "regional_assessment": {
            "order": list(REGION_NAMES),
            "state": "unknown_unqualified",
            "reason": "no_independently_qualified_runtime_regional_calibration",
        },
        "claims": {
            "changes_final_gnm_motion": False,
            "candidate_is_shipped": False,
            "final_output_arrays_byte_identical": True,
            "grants_motion_authority": False,
            "production_validated": False,
        },
        "limitations": [
            "Candidate is regularized from the existing retarget result, not a "
            "calibrated geometry-domain re-solve.",
            "No provider region has runtime motion authority in V2a.",
            "Unknown subject identity and tongue visibility remain unknown.",
            "Pose vectors are regularized component-wise rather than on the rotation manifold.",
        ],
    }
    result = VideoSequenceCandidate(
        metadata_json=canonical_json(metadata),
        source_pts=provider.source_pts,
        shot_epoch_index=provider.shot_epoch_index,
        tracking_epoch_index=provider.tracking_epoch_index,
        subject_epoch_index=provider.subject_epoch_index,
        hard_anchor_mask=anchors,
        expression=stored_expression,
        rotations=stored_rotations,
        translation=stored_translation,
    )
    result.validate_inputs(
        provider,
        calibration,
        provider_result_sha256=provider_result_sha256,
        calibration_evidence_sha256=calibration_evidence_sha256,
        expression=baseline_expression,
        rotations=baseline_rotations,
        translation=baseline_translation,
    )
    return result


_ARRAY_DTYPES = {
    "source_pts": np.dtype(np.int64),
    "shot_epoch_index": np.dtype(np.int32),
    "tracking_epoch_index": np.dtype(np.int32),
    "subject_epoch_index": np.dtype(np.int32),
    "hard_anchor_mask": np.dtype(np.bool_),
    "expression": np.dtype(np.float32),
    "rotations": np.dtype(np.float32),
    "translation": np.dtype(np.float32),
}


def write_video_sequence_candidate(
    path: str | Path, candidate: VideoSequenceCandidate
) -> Path:
    return write_npz(
        path,
        metadata_json=np.asarray(candidate.metadata_json),
        source_pts=candidate.source_pts,
        shot_epoch_index=candidate.shot_epoch_index,
        tracking_epoch_index=candidate.tracking_epoch_index,
        subject_epoch_index=candidate.subject_epoch_index,
        hard_anchor_mask=candidate.hard_anchor_mask,
        expression=candidate.expression,
        rotations=candidate.rotations,
        translation=candidate.translation,
    )


def load_video_sequence_candidate(path: str | Path) -> VideoSequenceCandidate:
    artifact = Path(path)
    try:
        if artifact.stat().st_size <= 0 or artifact.stat().st_size > MAX_SEQUENCE_BYTES:
            raise ValueError("Sequence artifact size is outside its bounds")
        with zipfile.ZipFile(artifact) as archive:
            names = [item.filename for item in archive.infolist()]
            expected = {"metadata_json.npy", *{f"{name}.npy" for name in _ARRAY_DTYPES}}
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("Sequence artifact members do not match the schema")
            if sum(item.file_size for item in archive.infolist()) > MAX_SEQUENCE_UNCOMPRESSED_BYTES:
                raise ValueError("Sequence artifact exceeds its resource limit")
        with np.load(artifact, allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in _ARRAY_DTYPES}
            metadata = np.asarray(payload["metadata_json"])
        if metadata.shape != () or metadata.dtype.kind != "U":
            raise ValueError("Sequence metadata member is invalid")
        for name, dtype in _ARRAY_DTYPES.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f"Sequence {name} dtype does not match the schema")
        return VideoSequenceCandidate(
            metadata_json=str(metadata.item()),
            source_pts=arrays["source_pts"],
            shot_epoch_index=arrays["shot_epoch_index"],
            tracking_epoch_index=arrays["tracking_epoch_index"],
            subject_epoch_index=arrays["subject_epoch_index"],
            hard_anchor_mask=arrays["hard_anchor_mask"],
            expression=arrays["expression"],
            rotations=arrays["rotations"],
            translation=arrays["translation"],
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid video sequence candidate: {exc}") from exc


def build_video_sequence_summary(
    candidate: VideoSequenceCandidate, *, candidate_sha256: str
) -> dict[str, Any]:
    if not valid_sha256(candidate_sha256):
        raise ValueError("Sequence candidate artifact hash must be SHA-256")
    metadata = candidate.metadata
    bindings = metadata["bindings"]
    return {
        "schemaVersion": VIDEO_SEQUENCE_SUMMARY_SCHEMA_VERSION,
        "candidateSchemaVersion": VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION,
        "status": metadata["status"],
        "policy": VIDEO_SEQUENCE_POLICY,
        "motionAuthority": MOTION_AUTHORITY,
        "consumedByRetargeting": False,
        "candidateSha256": candidate_sha256,
        "frameCount": candidate.frame_count,
        "sourcePtsSha256": array_sha256(candidate.source_pts),
        "baselineHashes": {
            "expression": bindings["baseline_expression_sha256"],
            "rotations": bindings["baseline_rotations_sha256"],
            "translation": bindings["baseline_translation_sha256"],
        },
        "shippedHashes": {
            "expression": bindings["shipped_expression_sha256"],
            "rotations": bindings["shipped_rotations_sha256"],
            "translation": bindings["shipped_translation_sha256"],
        },
        "candidateHashes": {
            "expression": array_sha256(candidate.expression),
            "rotations": array_sha256(candidate.rotations),
            "translation": array_sha256(candidate.translation),
        },
        "candidateComparison": {
            "expressionDiffersFromBaseline": (
                array_sha256(candidate.expression)
                != bindings["baseline_expression_sha256"]
            ),
            "rotationsDifferFromBaseline": (
                array_sha256(candidate.rotations)
                != bindings["baseline_rotations_sha256"]
            ),
            "translationDiffersFromBaseline": (
                array_sha256(candidate.translation)
                != bindings["baseline_translation_sha256"]
            ),
            "changedFrameCounts": None,
            "note": "Per-frame deltas require the separately hash-bound baseline artifact.",
        },
        "claims": metadata["claims"],
    }


def write_video_sequence_summary(
    path: str | Path, candidate: VideoSequenceCandidate, *, candidate_sha256: str
) -> Path:
    return write_json(
        path,
        build_video_sequence_summary(candidate, candidate_sha256=candidate_sha256),
    )


def load_verified_video_sequence_summary(
    path: str | Path,
    *,
    candidate_path: str | Path,
) -> dict[str, Any]:
    artifact = Path(path)
    try:
        if artifact.stat().st_size <= 0 or artifact.stat().st_size > MAX_SEQUENCE_SUMMARY_BYTES:
            raise ValueError("Sequence summary size is outside its bounds")
        payload = json.loads(
            artifact.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {item}")
            ),
        )
        candidate = load_video_sequence_candidate(candidate_path)
        expected = build_video_sequence_summary(
            candidate, candidate_sha256=file_sha256(candidate_path)
        )
        if payload != expected:
            raise ValueError("Sequence summary does not reconstruct from candidate")
        return payload
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid video sequence summary: {exc}") from exc


__all__ = [
    "VIDEO_SEQUENCE_CANDIDATE_SCHEMA_VERSION",
    "VIDEO_SEQUENCE_POLICY",
    "VIDEO_SEQUENCE_SUMMARY_SCHEMA_VERSION",
    "VideoSequenceCandidate",
    "build_shadow_video_sequence_candidate",
    "build_video_sequence_summary",
    "load_verified_video_sequence_summary",
    "load_video_sequence_candidate",
    "write_video_sequence_candidate",
    "write_video_sequence_summary",
]
