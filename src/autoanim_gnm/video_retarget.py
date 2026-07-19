"""Confidence-aware MediaPipe performance retargeting into a fixed GNM identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation

from .animation import LipContactCalibration, apply_lip_contact_correction
from .errors import AutoAnimError
from .rig import ControlRig
from .serialization import write_npz
from .video_capture import CaptureTrack, MONOCULAR_SCALE_CAVEAT


PERFORMANCE_SCHEMA_VERSION = "autoanim.gnm-performance.v3"
NEUTRAL_CALIBRATION_CAVEAT = (
    "Neutral calibration is selected and scored from high-quality low-activity frames, but "
    "MediaPipe blendshape bias is person/camera dependent and a heuristic pass is not a "
    "labeled neutral-pose validation."
)
NEUTRAL_BASELINE_SCORE_LIMIT = 1.0
NEUTRAL_WINDOW_MOTION_LIMIT = 0.12
NEUTRAL_SEMANTIC_AMBIGUITY_LIMIT = 0.50
FAST_CONTACT_CONTROLS = frozenset(
    {
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "jawOpen",
        "mouthClose",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthRollLower",
        "mouthRollUpper",
    }
)
GAZE_JOINT_CONTROLS = frozenset(
    {
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
    }
)
QUARANTINED_EXPRESSION_CONTROLS = frozenset({"mouthClose"})
SOURCE_CONTACT_METHOD = "mediapipe_inner_lip_geometry_v1"
SOURCE_INNER_LIP_PAIRS = ((82, 87), (13, 14), (312, 317))
SOURCE_INTEROCULAR_PAIR = (33, 263)
SOURCE_CONTACT_SEAL_GAP_INTEROCULAR = 0.030
SOURCE_CONTACT_RELEASE_GAP_INTEROCULAR = 0.055
SOURCE_CONTACT_CORRECTION_MIN_CONFIDENCE = 0.65
SOURCE_APERTURE_METHOD = "mediapipe_inner_lip_geometry_identity_calibrated_v1"
SOURCE_APERTURE_MIN_GAP_INTEROCULAR = 0.055
SOURCE_APERTURE_MAX_TARGET_INTEROCULAR = 0.250
SOURCE_APERTURE_TARGET_TOLERANCE_INTEROCULAR = 0.004


def _readonly_array(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@runtime_checkable
class SequenceRetargeter(Protocol):
    """The interface implemented by ``ARKitGNMRetargeter`` and calibrated rigs."""

    def retarget_sequence(
        self, weights: np.ndarray, pose_names: Sequence[str], **kwargs: Any
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TemporalFilterConfig:
    """Small causal filter; closures/blinks bypass its low-pass path."""

    slow_time_constant_seconds: float = 0.030
    fast_time_constant_seconds: float = 0.008
    motion_scale_per_second: float = 5.0
    missing_hold_seconds: float = 0.045
    missing_decay_seconds: float = 0.120
    minimum_quality: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.slow_time_constant_seconds,
            self.fast_time_constant_seconds,
            self.motion_scale_per_second,
            self.missing_decay_seconds,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Positive temporal-filter constants are required")
        if (
            not np.isfinite(self.missing_hold_seconds)
            or self.missing_hold_seconds < 0
            or not 0 <= self.minimum_quality <= 1
        ):
            raise ValueError("Invalid temporal-filter hold or quality threshold")
        if self.fast_time_constant_seconds > self.slow_time_constant_seconds:
            raise ValueError("Fast time constant cannot exceed slow time constant")


@dataclass(frozen=True, slots=True)
class FilteredBlendshapes:
    names: tuple[str, ...]
    scores: np.ndarray
    effective_quality: np.ndarray
    contact_passthrough: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "scores", _readonly_array(self.scores, np.float32))
        object.__setattr__(
            self, "effective_quality", _readonly_array(self.effective_quality, np.float32)
        )
        object.__setattr__(
            self, "contact_passthrough", _readonly_array(self.contact_passthrough, np.bool_)
        )
        count = len(self.scores)
        if self.scores.ndim != 2 or self.scores.shape[1] != len(self.names):
            raise ValueError("Filtered blendshape shape/name mismatch")
        if self.effective_quality.shape != (count,) or self.contact_passthrough.shape != (
            count,
            len(self.names),
        ):
            raise ValueError("Filtered blendshape metadata has invalid shape")
        if not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("Filtered blendshape scores must be finite and bounded")


@dataclass(frozen=True, slots=True)
class PerformanceProvenance:
    capture_schema_version: str
    capture_source_sha256: str
    retargeter: str
    filter_config: TemporalFilterConfig
    transform_convention: str
    translation_scale_to_gnm: float
    coordinate_conversion: tuple[tuple[float, float, float], ...]
    eye_range_radians: float
    baseline_frame_indices: tuple[int, ...]
    neutral_blendshape_baseline: tuple[tuple[str, float], ...]
    neutral_baseline_method: str
    neutral_baseline_validated: bool
    neutral_baseline_correction_applied: bool
    neutral_baseline_score: float
    neutral_baseline_score_limit: float
    neutral_baseline_semantic_peak: float
    neutral_baseline_ambiguity_controls: tuple[str, ...]
    quarantined_expression_controls: tuple[str, ...]
    contact_source_method: str
    contact_calibration_hash: str | None
    aperture_source_method: str
    negative_baseline_residual_clipped_fraction: float
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.negative_baseline_residual_clipped_fraction <= 1.0:
            raise ValueError("Negative baseline residual fraction must lie in [0,1]")
        if not self.contact_source_method:
            raise ValueError("Video contact source method is required")
        if not self.aperture_source_method:
            raise ValueError("Video aperture source method is required")
        if len(set(self.quarantined_expression_controls)) != len(
            self.quarantined_expression_controls
        ):
            raise ValueError("Quarantined expression controls must be unique")
        if self.contact_calibration_hash is not None and len(
            self.contact_calibration_hash
        ) != 64:
            raise ValueError("Video contact calibration hash must be SHA-256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture_schema_version": self.capture_schema_version,
            "capture_source_sha256": self.capture_source_sha256,
            "retargeter": self.retargeter,
            "filter_config": asdict(self.filter_config),
            "transform_convention": self.transform_convention,
            "translation_scale_to_gnm": self.translation_scale_to_gnm,
            "coordinate_conversion": [list(row) for row in self.coordinate_conversion],
            "eye_range_radians": self.eye_range_radians,
            "baseline_frame_indices": list(self.baseline_frame_indices),
            "neutral_blendshape_baseline": {
                name: value for name, value in self.neutral_blendshape_baseline
            },
            "neutral_baseline_method": self.neutral_baseline_method,
            "neutral_baseline_validated": self.neutral_baseline_validated,
            "neutral_baseline_correction_applied": self.neutral_baseline_correction_applied,
            "neutral_baseline_score": self.neutral_baseline_score,
            "neutral_baseline_score_limit": self.neutral_baseline_score_limit,
            "neutral_baseline_semantic_peak": self.neutral_baseline_semantic_peak,
            "neutral_baseline_ambiguity_controls": list(
                self.neutral_baseline_ambiguity_controls
            ),
            "quarantined_expression_controls": list(
                self.quarantined_expression_controls
            ),
            "contact_source_method": self.contact_source_method,
            "contact_calibration_hash": self.contact_calibration_hash,
            "aperture_source_method": self.aperture_source_method,
            "negative_baseline_residual_clipped_fraction": (
                self.negative_baseline_residual_clipped_fraction
            ),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True, slots=True)
class NeutralBaselineEstimate:
    frame_indices: tuple[int, ...]
    expression_baseline: np.ndarray
    method: str
    validated: bool
    correction_applied: bool
    score: float
    semantic_peak: float
    ambiguity_controls: tuple[str, ...]
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expression_baseline",
            _readonly_array(self.expression_baseline, np.float32),
        )


@dataclass(frozen=True, slots=True)
class GNMPerformanceTrack:
    """GNM controls with one fixed identity shared by every source frame."""

    identity: np.ndarray
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray
    timestamps_seconds: np.ndarray
    source_pts: np.ndarray
    detected: np.ndarray
    effective_quality: np.ndarray
    source_lip_geometry_valid: np.ndarray
    source_lip_gap_interocular: np.ndarray
    source_lip_contact_confidence: np.ndarray
    lip_contact_target_gap_interocular: np.ndarray
    contact_correction_applied: np.ndarray
    lip_contact_attained: np.ndarray
    lip_aperture_target_gap_interocular: np.ndarray
    lip_aperture_correction_applied: np.ndarray
    lip_aperture_target_attained: np.ndarray
    provenance: PerformanceProvenance
    schema_version: str = PERFORMANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        arrays = {
            "identity": (self.identity, np.float32),
            "expression": (self.expression, np.float32),
            "rotations": (self.rotations, np.float32),
            "translation": (self.translation, np.float32),
            "timestamps_seconds": (self.timestamps_seconds, np.float64),
            "source_pts": (self.source_pts, np.int64),
            "detected": (self.detected, np.bool_),
            "effective_quality": (self.effective_quality, np.float32),
            "source_lip_geometry_valid": (
                self.source_lip_geometry_valid,
                np.bool_,
            ),
            "source_lip_gap_interocular": (
                self.source_lip_gap_interocular,
                np.float32,
            ),
            "source_lip_contact_confidence": (
                self.source_lip_contact_confidence,
                np.float32,
            ),
            "lip_contact_target_gap_interocular": (
                self.lip_contact_target_gap_interocular,
                np.float32,
            ),
            "contact_correction_applied": (
                self.contact_correction_applied,
                np.bool_,
            ),
            "lip_contact_attained": (self.lip_contact_attained, np.bool_),
            "lip_aperture_target_gap_interocular": (
                self.lip_aperture_target_gap_interocular,
                np.float32,
            ),
            "lip_aperture_correction_applied": (
                self.lip_aperture_correction_applied,
                np.bool_,
            ),
            "lip_aperture_target_attained": (
                self.lip_aperture_target_attained,
                np.bool_,
            ),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, _readonly_array(value, dtype))
        count = len(self.timestamps_seconds)
        if self.schema_version != PERFORMANCE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported performance schema: {self.schema_version}")
        if self.identity.shape != (253,):
            raise ValueError("GNM identity must have shape (253,)")
        if self.expression.shape != (count, 383):
            raise ValueError("GNM expression must have shape [frames, 383]")
        if self.rotations.shape != (count, 4, 3) or self.translation.shape != (count, 3):
            raise ValueError("GNM pose controls have invalid shape")
        if self.source_pts.shape != (count,) or self.detected.shape != (count,):
            raise ValueError("GNM frame metadata has invalid shape")
        if self.effective_quality.shape != (count,):
            raise ValueError("GNM quality metadata has invalid shape")
        contact_arrays = (
            self.source_lip_geometry_valid,
            self.source_lip_gap_interocular,
            self.source_lip_contact_confidence,
            self.lip_contact_target_gap_interocular,
            self.contact_correction_applied,
            self.lip_contact_attained,
            self.lip_aperture_target_gap_interocular,
            self.lip_aperture_correction_applied,
            self.lip_aperture_target_attained,
        )
        if any(value.shape != (count,) for value in contact_arrays):
            raise ValueError("GNM contact metadata has invalid shape")
        if np.any(
            (self.source_lip_contact_confidence < 0)
            | (self.source_lip_contact_confidence > 1)
        ):
            raise ValueError("GNM source contact confidence must lie in [0,1]")
        if count == 0 or not all(
            np.isfinite(value).all()
            for value in (
                self.identity,
                self.expression,
                self.rotations,
                self.translation,
                self.timestamps_seconds,
                self.effective_quality,
                self.source_lip_gap_interocular,
                self.source_lip_contact_confidence,
                self.lip_contact_target_gap_interocular,
                self.lip_aperture_target_gap_interocular,
            )
        ):
            raise ValueError("GNM performance controls must be nonempty and finite")
        if count > 1 and (
            np.any(np.diff(self.timestamps_seconds) <= 0) or np.any(np.diff(self.source_pts) <= 0)
        ):
            raise ValueError("GNM performance timestamps must be strictly increasing")

    @property
    def frame_count(self) -> int:
        return len(self.timestamps_seconds)


def effective_capture_quality(track: CaptureTrack) -> np.ndarray:
    """Use detector confidence when exposed, otherwise an explicit geometry proxy."""

    confidence = np.where(
        np.isfinite(track.face_confidence), track.face_confidence, track.tracking_quality
    ).astype(np.float32)
    confidence[~track.detected] = 0.0
    return np.clip(confidence, 0.0, 1.0)


def source_lip_contact_geometry(
    track: CaptureTrack,
    quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure lip seal evidence from tracked image geometry, not ARKit labels.

    MediaPipe's ``mouthClose`` coefficient is not geometrically calibrated for
    GNM and its released Claire row opens this character. The three inner-lip
    distances are therefore measured directly in pixel-corrected image space
    and normalized by the outer-eye distance. A conservative soft band avoids
    treating a rolled lip or press coefficient as a physical seal.

    Missing or degenerate frames carry zero confidence and an explicit false
    validity bit; the stored finite gap is only a serialization-safe sentinel.
    """

    quality_value = np.asarray(quality, dtype=np.float64)
    if quality_value.shape != (track.frame_count,) or not np.isfinite(quality_value).all():
        raise AutoAnimError("INPUT_INVALID", "Video contact quality must be finite [frames]")
    points = np.asarray(track.landmarks_xyz[:, :, :2], dtype=np.float64).copy()
    points[:, :, 0] *= float(track.width)
    points[:, :, 1] *= float(track.height)
    left_eye, right_eye = SOURCE_INTEROCULAR_PAIR
    interocular = np.linalg.norm(points[:, left_eye] - points[:, right_eye], axis=1)
    pair_gaps = np.stack(
        [
            np.linalg.norm(points[:, upper] - points[:, lower], axis=1)
            for upper, lower in SOURCE_INNER_LIP_PAIRS
        ],
        axis=1,
    )
    valid = (
        track.detected
        & np.isfinite(interocular)
        & (interocular > 1e-6)
        & np.isfinite(pair_gaps).all(axis=1)
    )
    gap = np.full(
        track.frame_count,
        SOURCE_CONTACT_RELEASE_GAP_INTEROCULAR,
        dtype=np.float64,
    )
    gap[valid] = np.mean(pair_gaps[valid], axis=1) / interocular[valid]
    normalized = np.clip(
        (SOURCE_CONTACT_RELEASE_GAP_INTEROCULAR - gap)
        / (
            SOURCE_CONTACT_RELEASE_GAP_INTEROCULAR
            - SOURCE_CONTACT_SEAL_GAP_INTEROCULAR
        ),
        0.0,
        1.0,
    )
    # Smoothstep makes the thresholds continuous while retaining exact full
    # confidence inside the calibrated seal band.
    confidence = normalized * normalized * (3.0 - 2.0 * normalized)
    confidence *= np.clip(quality_value, 0.0, 1.0)
    confidence[~valid] = 0.0
    return (
        gap.astype(np.float32),
        confidence.astype(np.float32),
        valid.astype(bool),
    )


def _rig_lip_gap_interocular(rig: ControlRig, expression: np.ndarray) -> float:
    landmarks = rig.compact_landmarks(np.asarray(expression, dtype=np.float32))
    neutral = rig.neutral_landmarks
    interocular = float(np.linalg.norm(neutral[36] - neutral[45]))
    if not np.isfinite(interocular) or interocular <= 0.0:
        raise AutoAnimError("INTERNAL_ERROR", "GNM interocular distance is invalid")
    return float(
        np.mean(
            [
                np.linalg.norm(landmarks[upper] - landmarks[lower]) / interocular
                for upper, lower in ((61, 67), (62, 66), (63, 65))
            ]
        )
    )


def _apply_lip_aperture_match(
    rig: ControlRig,
    expression: np.ndarray,
    calibration: LipContactCalibration,
    *,
    source_gap_interocular: float,
    quality: float,
) -> tuple[np.ndarray, bool, float]:
    """Open the character toward observed inner-lip geometry without changing timing.

    The same identity-calibrated compact-landmark inverse used for contact is
    driven in the opposite direction. The solve preserves the current
    left/right lip shape, modifies only GNM's lower-face block, bounds upper-
    face drift, and caps monocular 2D targets at a plausible production-review
    envelope. This is a measurable aperture correction, not a jaw/collision
    model; physical oral validation remains a separate gate.
    """

    original = np.asarray(expression, dtype=np.float32)
    source_gap = float(source_gap_interocular)
    confidence = float(np.clip(quality, 0.0, 1.0))
    if (
        not np.isfinite(source_gap)
        or source_gap < SOURCE_APERTURE_MIN_GAP_INTEROCULAR
        or confidence < 0.20
    ):
        return original.copy(), False, 0.0
    landmarks = rig.compact_landmarks(original)
    neutral = rig.neutral_landmarks
    interocular = float(np.linalg.norm(neutral[36] - neutral[45]))
    current_gap = _rig_lip_gap_interocular(rig, original)
    observed_target = min(source_gap, SOURCE_APERTURE_MAX_TARGET_INTEROCULAR)
    target_gap = current_gap + confidence * (observed_target - current_gap)
    if target_gap <= current_gap + 1.0e-4:
        return original.copy(), False, float(max(target_gap, 0.0))

    pairs = ((61, 67), (62, 66), (63, 65))
    desired = np.zeros((68, 3), dtype=np.float32)
    current_pair_gaps = np.asarray(
        [
            np.linalg.norm(landmarks[upper] - landmarks[lower]) / interocular
            for upper, lower in pairs
        ],
        dtype=np.float32,
    )
    pair_targets = current_pair_gaps + np.float32(target_gap - current_gap)
    for pair_target, (upper, lower) in zip(pair_targets, pairs, strict=True):
        separation = landmarks[lower] - landmarks[upper]
        length = float(np.linalg.norm(separation))
        wanted = float(max(pair_target, 0.0) * interocular)
        if length <= 1.0e-9 or wanted <= length:
            continue
        expansion = separation * np.float32(wanted / length - 1.0)
        desired[upper] -= np.float32(0.5) * expansion
        desired[lower] += np.float32(0.5) * expansion
    inner_indices = np.asarray((61, 62, 63, 65, 66, 67), dtype=np.int64)
    solved = np.asarray(calibration.inner_response, dtype=np.float32) @ desired[
        inner_indices
    ].reshape(-1)
    if not np.isfinite(solved).all() or np.max(np.abs(solved), initial=0.0) <= 1.0e-9:
        return original.copy(), False, float(target_gap)
    direction = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    direction[200:350] = solved
    zero = np.zeros_like(original)

    best = original.copy()
    best_gap = current_gap
    for alpha in np.linspace(0.05, 1.50, 30, dtype=np.float32):
        candidate, _ = rig.compose(original + alpha * direction, zero)
        candidate_landmarks = rig.compact_landmarks(candidate)
        upper_face_drift = float(
            np.max(
                np.linalg.norm(candidate_landmarks[17:48] - landmarks[17:48], axis=1),
                initial=0.0,
            )
            / interocular
        )
        if upper_face_drift > 0.002:
            continue
        candidate_gap = _rig_lip_gap_interocular(rig, candidate)
        if candidate_gap > best_gap:
            best = candidate
            best_gap = candidate_gap
        if candidate_gap >= target_gap:
            break
    applied = best_gap > current_gap + 1.0e-4
    return best if applied else original.copy(), applied, float(target_gap)


def _adaptive_alpha(
    difference: np.ndarray,
    delta_seconds: float,
    quality: float,
    config: TemporalFilterConfig,
) -> np.ndarray:
    speed = np.abs(difference) / max(delta_seconds, 1e-9)
    blend = np.clip(speed / config.motion_scale_per_second, 0.0, 1.0)
    time_constant = (
        config.slow_time_constant_seconds * (1.0 - blend)
        + config.fast_time_constant_seconds * blend
    )
    alpha = 1.0 - np.exp(-delta_seconds / time_constant)
    confidence_weight = np.clip(
        (quality - config.minimum_quality) / max(1.0 - config.minimum_quality, 1e-9),
        0.0,
        1.0,
    )
    return alpha * confidence_weight


def filter_blendshapes(
    track: CaptureTrack,
    config: TemporalFilterConfig = TemporalFilterConfig(),
) -> FilteredBlendshapes:
    """Filter jitter without delaying fast eye/lip contact controls.

    High-quality observed blink and mouth-contact controls are exact
    passthrough values.  The remaining controls use a short (8--30 ms),
    velocity-adaptive causal filter.  Missing frames are held only briefly and
    then deterministically decay rather than being treated as observations.
    """

    quality = effective_capture_quality(track)
    source = np.asarray(track.blendshape_scores, dtype=np.float64)
    output = np.zeros_like(source)
    passthrough = np.zeros(source.shape, dtype=bool)
    fast_indices = np.asarray(
        [index for index, name in enumerate(track.blendshape_names) if name in FAST_CONTACT_CONTROLS],
        dtype=np.int64,
    )
    previous = np.zeros(source.shape[1], dtype=np.float64)
    initialized = False
    last_detection_time: float | None = None
    for index, timestamp in enumerate(track.timestamps_seconds):
        if index == 0:
            delta = 1.0 / 30.0
        else:
            delta = float(timestamp - track.timestamps_seconds[index - 1])
        if track.detected[index] and quality[index] >= config.minimum_quality:
            raw = source[index]
            if not initialized:
                current = raw.copy()
                initialized = True
            else:
                alpha = _adaptive_alpha(raw - previous, delta, float(quality[index]), config)
                current = previous + alpha * (raw - previous)
            if len(fast_indices):
                # VIDEO mode already applies its tracker; a second broad
                # low-pass here would visibly delay blinks and lip closures.
                current[fast_indices] = raw[fast_indices]
                passthrough[index, fast_indices] = True
            last_detection_time = float(timestamp)
        elif not initialized:
            current = previous.copy()
        else:
            gap = (
                np.inf
                if last_detection_time is None
                else float(timestamp) - last_detection_time
            )
            if gap <= config.missing_hold_seconds:
                current = previous.copy()
            else:
                decay = np.exp(-delta / config.missing_decay_seconds)
                current = previous * decay
        output[index] = np.clip(current, 0.0, 1.0)
        previous = output[index]
    return FilteredBlendshapes(
        names=track.blendshape_names,
        scores=output.astype(np.float32),
        effective_quality=quality,
        contact_passthrough=passthrough,
    )


def _proper_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _neutral_control_limit(name: str) -> float:
    """Conservative absolute gates for rejecting obvious performed motion.

    These are not FACS thresholds. They only keep a large jaw opening, blink,
    pucker, or similarly explicit performance from being silently subtracted as
    tracker bias. Identity-sensitive controls retain a higher tolerance because
    MediaPipe can report large static brow, squint, smile, and press values on a
    visually neutral subject.
    """

    if name == "jawOpen":
        return 0.18
    if name in {"mouthFunnel", "mouthPucker"}:
        return 0.45
    if name.startswith("eyeBlink"):
        return 0.55
    if name.startswith("eyeWide"):
        return 0.60
    if name.startswith("mouthLowerDown") or name.startswith("mouthUpperUp"):
        return 0.50
    return 0.75


def _neutral_window_diagnostics(
    filtered: FilteredBlendshapes,
    indices: np.ndarray,
) -> tuple[float, float, tuple[str, ...]]:
    columns = np.asarray(
        [
            index
            for index, name in enumerate(filtered.names)
            if name != "_neutral" and name not in GAZE_JOINT_CONTROLS
        ],
        dtype=np.int64,
    )
    if not len(columns):
        return 0.0, 0.0, ()
    selected = filtered.scores[np.asarray(indices, dtype=np.int64)][:, columns]
    median = np.median(selected, axis=0)
    names = tuple(filtered.names[index] for index in columns)
    activation_score = max(
        (
            float(value) / _neutral_control_limit(name)
            for name, value in zip(names, median, strict=True)
        ),
        default=0.0,
    )
    if len(selected) > 1:
        motion = float(np.median(np.linalg.norm(np.diff(selected, axis=0), axis=1)))
    else:
        motion = 0.0
    score = max(activation_score, motion / NEUTRAL_WINDOW_MOTION_LIMIT)
    semantic_peak = float(np.max(median, initial=0.0))
    ambiguity = tuple(
        name
        for name, value in zip(names, median, strict=True)
        if float(value) >= NEUTRAL_SEMANTIC_AMBIGUITY_LIMIT
    )
    return score, semantic_peak, ambiguity


def _candidate_neutral_windows(
    track: CaptureTrack,
    eligible: np.ndarray,
    neutral_baseline_seconds: float,
) -> tuple[np.ndarray, ...]:
    if not len(eligible):
        return ()
    if track.frame_count > 1:
        typical_delta = float(np.median(np.diff(track.timestamps_seconds)))
        expected = max(1, int(np.floor(neutral_baseline_seconds / typical_delta + 1e-6)) + 1)
    else:
        expected = 1
    minimum = max(1, min(expected, int(np.ceil(expected * 0.70))))
    windows: list[np.ndarray] = []
    for offset, start_index in enumerate(eligible):
        start_time = float(track.timestamps_seconds[start_index])
        window = eligible[offset:][
            track.timestamps_seconds[eligible[offset:]]
            <= start_time + neutral_baseline_seconds + 1e-9
        ]
        if len(window) >= minimum:
            windows.append(window)
    if not windows:
        windows.append(eligible[: min(len(eligible), expected)])
    return tuple(windows)


def _estimate_neutral_baseline(
    track: CaptureTrack,
    filtered: FilteredBlendshapes,
    *,
    baseline_frame_count: int | None,
    neutral_baseline_seconds: float,
    minimum_quality: float,
) -> NeutralBaselineEstimate:
    eligible = np.flatnonzero(
        track.detected & (filtered.effective_quality >= minimum_quality)
    )
    if not len(eligible):
        raise AutoAnimError("FACE_NOT_FOUND", "No high-quality face frame was available")
    if baseline_frame_count is not None:
        windows = (eligible[:baseline_frame_count],)
        initial_method = "explicit_initial_window"
    else:
        windows = _candidate_neutral_windows(track, eligible, neutral_baseline_seconds)
        initial_method = "initial_window"
    diagnostics = [
        _neutral_window_diagnostics(filtered, window) for window in windows
    ]
    first_passes = diagnostics[0][0] <= NEUTRAL_BASELINE_SCORE_LIMIT
    if first_passes:
        selected_index = 0
        method = initial_method
    else:
        selected_index = min(range(len(windows)), key=lambda index: diagnostics[index][0])
        method = (
            "auto_low_activity_window"
            if diagnostics[selected_index][0] <= NEUTRAL_BASELINE_SCORE_LIMIT
            and selected_index != 0
            else "none_expressive_video"
        )
    selected = windows[selected_index]
    score, semantic_peak, ambiguity = diagnostics[selected_index]
    correction_applied = method != "none_expressive_video"
    baseline = (
        np.median(filtered.scores[selected], axis=0).astype(np.float32)
        if correction_applied
        else np.zeros(len(filtered.names), dtype=np.float32)
    )
    validated = bool(
        correction_applied
        and score <= NEUTRAL_BASELINE_SCORE_LIMIT
        and not ambiguity
    )
    caveats: list[str] = []
    if method == "auto_low_activity_window":
        caveats.append(
            "The initial reference window was expressive or moving, so calibration used a "
            "later low-activity window; it is heuristic and not a labeled neutral pose."
        )
    elif method == "none_expressive_video":
        caveats.append(
            "No neutral-compatible reference window was found; blendshape baseline subtraction "
            "was disabled to avoid erasing a held performance. Static tracker/person bias remains."
        )
    if ambiguity:
        caveats.append(
            "The selected reference has semantic ambiguity in "
            + ", ".join(ambiguity)
            + "; these high coefficients may be tracker/person bias or a held expression, so "
            "the neutral baseline is not production-validated."
        )
    return NeutralBaselineEstimate(
        frame_indices=tuple(int(index) for index in selected),
        expression_baseline=baseline,
        method=method,
        validated=validated,
        correction_applied=correction_applied,
        score=float(score),
        semantic_peak=semantic_peak,
        ambiguity_controls=ambiguity,
        caveats=tuple(caveats),
    )


def _filter_pose_vectors(
    values: np.ndarray,
    timestamps: np.ndarray,
    detected: np.ndarray,
    quality: np.ndarray,
    config: TemporalFilterConfig,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    previous = np.zeros(values.shape[1], dtype=np.float64)
    initialized = False
    last_detection_time: float | None = None
    for index, timestamp in enumerate(timestamps):
        delta = (
            float(timestamp - timestamps[index - 1]) if index else 1.0 / 30.0
        )
        if detected[index] and quality[index] >= config.minimum_quality:
            raw = values[index]
            if not initialized:
                current = raw.copy()
                initialized = True
            else:
                alpha = _adaptive_alpha(raw - previous, delta, float(quality[index]), config)
                current = previous + alpha * (raw - previous)
            last_detection_time = float(timestamp)
        elif not initialized:
            current = previous.copy()
        else:
            gap = np.inf if last_detection_time is None else float(timestamp) - last_detection_time
            if gap <= config.missing_hold_seconds:
                current = previous.copy()
            else:
                # Pose should settle more slowly than a missing expression;
                # this avoids a visible one-frame snap to the origin.
                decay = np.exp(-delta / (2.0 * config.missing_decay_seconds))
                current = previous * decay
        output[index] = current
        previous = current
    return output


def _pose_from_transforms(
    track: CaptureTrack,
    filtered: FilteredBlendshapes,
    config: TemporalFilterConfig,
    *,
    translation_scale_to_gnm: float,
    coordinate_conversion: np.ndarray,
    baseline_indices: np.ndarray,
    eye_range_radians: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    selected = np.asarray(baseline_indices, dtype=np.int64)
    if not len(selected):
        raise AutoAnimError("FACE_NOT_FOUND", "No face was detected in the video")
    matrices = track.facial_transforms[selected].astype(np.float64)
    baseline_rotations = Rotation.from_matrix(
        np.stack([_proper_rotation(matrix[:3, :3]) for matrix in matrices])
    )
    baseline_rotation = baseline_rotations.mean().as_matrix()
    baseline_translation = np.median(matrices[:, :3, 3], axis=0)
    raw_rotation = np.zeros((track.frame_count, 3), dtype=np.float64)
    raw_translation = np.zeros((track.frame_count, 3), dtype=np.float64)
    for index in np.flatnonzero(track.detected):
        matrix = track.facial_transforms[index].astype(np.float64)
        relative = _proper_rotation(matrix[:3, :3]) @ baseline_rotation.T
        converted = coordinate_conversion @ relative @ coordinate_conversion.T
        raw_rotation[index] = Rotation.from_matrix(converted).as_rotvec()
        raw_translation[index] = (
            coordinate_conversion @ (matrix[:3, 3] - baseline_translation)
        ) * translation_scale_to_gnm
    quality = filtered.effective_quality
    head = _filter_pose_vectors(
        raw_rotation,
        track.timestamps_seconds,
        track.detected,
        quality,
        config,
    )
    translation = _filter_pose_vectors(
        raw_translation,
        track.timestamps_seconds,
        track.detected,
        quality,
        config,
    )
    rotations = np.zeros((track.frame_count, 4, 3), dtype=np.float64)
    rotations[:, 1] = head
    columns = {name: index for index, name in enumerate(filtered.names)}

    def score(name: str) -> np.ndarray:
        index = columns.get(name)
        return (
            filtered.scores[:, index].astype(np.float64)
            if index is not None
            else np.zeros(track.frame_count, dtype=np.float64)
        )

    # GNM +X is subject-left, +Y is up, and +Z is forward.  Around +X,
    # positive pitch looks down; around +Y, positive yaw looks subject-left.
    pitch_left = (score("eyeLookDownLeft") - score("eyeLookUpLeft")) * eye_range_radians
    yaw_left = (score("eyeLookOutLeft") - score("eyeLookInLeft")) * eye_range_radians
    pitch_right = (score("eyeLookDownRight") - score("eyeLookUpRight")) * eye_range_radians
    yaw_right = (score("eyeLookInRight") - score("eyeLookOutRight")) * eye_range_radians
    # MediaPipe gaze coefficients contain face/camera-specific neutral bias.
    # GNM's gaze is a dedicated pair of eye joints, so use motion relative to
    # the same detected baseline that defines head pose rather than treating
    # the absolute tracker scores as an identity-independent calibration.
    for values in (pitch_left, yaw_left, pitch_right, yaw_right):
        values -= np.median(values[selected])
    rotations[:, 2, 0] = pitch_left
    rotations[:, 2, 1] = yaw_left
    rotations[:, 3, 0] = pitch_right
    rotations[:, 3, 1] = yaw_right
    return (
        rotations.astype(np.float32),
        translation.astype(np.float32),
        tuple(int(index) for index in selected),
    )


def retarget_capture(
    track: CaptureTrack,
    retargeter: SequenceRetargeter,
    *,
    identity: np.ndarray | None = None,
    filter_config: TemporalFilterConfig = TemporalFilterConfig(),
    translation_scale_to_gnm: float = 0.01,
    coordinate_conversion: np.ndarray | None = None,
    baseline_frame_count: int | None = None,
    neutral_baseline_seconds: float = 0.2,
    eye_range_radians: float = np.deg2rad(25.0),
    retarget_caveats: Sequence[str] | None = None,
    contact_rig: ControlRig | None = None,
    lip_contact_calibration: LipContactCalibration | None = None,
) -> GNMPerformanceTrack:
    """Retarget raw capture while holding a single GNM identity fixed.

    ``translation_scale_to_gnm=0.01`` converts MediaPipe canonical-face
    centimeters to GNM meters.  It is intentionally exposed because monocular
    video does not recover a subject-calibrated metric scale.
    """

    if not isinstance(retargeter, SequenceRetargeter):
        raise TypeError("Retargeter must implement retarget_sequence(weights, pose_names)")
    identity_value = (
        np.zeros(253, dtype=np.float32)
        if identity is None
        else np.asarray(identity, dtype=np.float32)
    )
    if identity_value.shape != (253,) or not np.isfinite(identity_value).all():
        raise AutoAnimError("INPUT_INVALID", "Fixed GNM identity must be a finite (253,) vector")
    if (
        not np.isfinite(translation_scale_to_gnm)
        or translation_scale_to_gnm <= 0
        or (baseline_frame_count is not None and baseline_frame_count <= 0)
        or not np.isfinite(neutral_baseline_seconds)
        or neutral_baseline_seconds <= 0
        or not np.isfinite(eye_range_radians)
        or eye_range_radians <= 0
    ):
        raise AutoAnimError("INPUT_INVALID", "Invalid pose-retarget calibration")
    conversion = (
        np.eye(3, dtype=np.float64)
        if coordinate_conversion is None
        else np.asarray(coordinate_conversion, dtype=np.float64)
    )
    if conversion.shape != (3, 3) or not np.isfinite(conversion).all():
        raise AutoAnimError("INPUT_INVALID", "Coordinate conversion must be a finite 3x3 matrix")
    if not np.allclose(conversion.T @ conversion, np.eye(3), atol=1e-6) or not np.isclose(
        abs(np.linalg.det(conversion)), 1.0, atol=1e-6
    ):
        raise AutoAnimError("INPUT_INVALID", "Coordinate conversion must be orthogonal")
    filtered = filter_blendshapes(track, filter_config)
    (
        source_lip_gap,
        source_lip_contact_confidence,
        source_lip_geometry_valid,
    ) = source_lip_contact_geometry(track, filtered.effective_quality)
    baseline_estimate = _estimate_neutral_baseline(
        track,
        filtered,
        baseline_frame_count=baseline_frame_count,
        neutral_baseline_seconds=neutral_baseline_seconds,
        minimum_quality=filter_config.minimum_quality,
    )
    baseline_indices = np.asarray(baseline_estimate.frame_indices, dtype=np.int64)
    neutral_baseline = baseline_estimate.expression_baseline
    # Only a neutral-compatible reference is allowed to remove static tracker
    # bias. If every candidate is expressive, the estimator returns a zero
    # baseline so the held performance is preserved and the uncertainty is
    # explicit instead of silently baked into the neutral pose.
    baseline_residual = (filtered.scores - neutral_baseline) / np.maximum(
        1.0 - neutral_baseline,
        1e-4,
    )
    residual_columns = np.asarray(
        [
            index
            for index, name in enumerate(filtered.names)
            if name != "_neutral"
            and name not in GAZE_JOINT_CONTROLS
            and name not in QUARANTINED_EXPRESSION_CONTROLS
        ],
        dtype=np.int64,
    )
    negative_residual_fraction = (
        float(np.mean(baseline_residual[:, residual_columns] < 0.0))
        if baseline_estimate.correction_applied and len(residual_columns)
        else 0.0
    )
    expression_scores = np.clip(baseline_residual, 0.0, 1.0).astype(np.float32)
    for index, name in enumerate(filtered.names):
        if name in GAZE_JOINT_CONTROLS or name in QUARANTINED_EXPRESSION_CONTROLS:
            # Do not drive gaze twice through both the dense facial basis and
            # the dedicated GNM eye joints assembled below. ``mouthClose`` is
            # quarantined because its Claire-to-GNM row opens this character;
            # direct landmark geometry drives the contact layer instead.
            expression_scores[:, index] = 0.0
    expression = np.asarray(
        retargeter.retarget_sequence(expression_scores, filtered.names), dtype=np.float32
    )
    if expression.shape != (track.frame_count, 383):
        raise AutoAnimError(
            "INTERNAL_ERROR", "Injected retargeter must return GNM controls with shape [frames,383]"
        )
    if not np.isfinite(expression).all():
        raise AutoAnimError("INTERNAL_ERROR", "Injected retargeter returned non-finite controls")
    if (contact_rig is None) != (lip_contact_calibration is None):
        raise AutoAnimError(
            "INPUT_INVALID",
            "Video lip-contact correction requires both a rig and calibration",
        )
    lip_contact_target_gap = np.zeros(track.frame_count, dtype=np.float32)
    contact_correction_applied = np.zeros(track.frame_count, dtype=bool)
    lip_contact_attained = np.zeros(track.frame_count, dtype=bool)
    lip_aperture_target_gap = np.zeros(track.frame_count, dtype=np.float32)
    lip_aperture_correction_applied = np.zeros(track.frame_count, dtype=bool)
    lip_aperture_target_attained = np.zeros(track.frame_count, dtype=bool)
    if contact_rig is not None and lip_contact_calibration is not None:
        if contact_rig.adapter.expression_dim != expression.shape[1]:
            raise AutoAnimError(
                "INPUT_INVALID",
                "Video contact rig does not match the retargeted GNM expression space",
            )
        for frame in range(track.frame_count):
            if (
                source_lip_geometry_valid[frame]
                and source_lip_contact_confidence[frame] < 0.12
            ):
                (
                    expression[frame],
                    lip_aperture_correction_applied[frame],
                    lip_aperture_target_gap[frame],
                ) = _apply_lip_aperture_match(
                    contact_rig,
                    expression[frame],
                    lip_contact_calibration,
                    source_gap_interocular=float(source_lip_gap[frame]),
                    quality=float(filtered.effective_quality[frame]),
                )
            (
                expression[frame],
                contact_correction_applied[frame],
                lip_contact_target_gap[frame],
            ) = apply_lip_contact_correction(
                contact_rig,
                expression[frame],
                lip_contact_calibration,
                (
                    float(source_lip_contact_confidence[frame])
                    if source_lip_contact_confidence[frame]
                    >= SOURCE_CONTACT_CORRECTION_MIN_CONFIDENCE
                    else 0.0
                ),
            )
            if lip_contact_target_gap[frame] > 0.0:
                lip_contact_attained[frame] = (
                    _rig_lip_gap_interocular(contact_rig, expression[frame])
                    <= float(lip_contact_target_gap[frame]) + 1.0e-3
                )
            if lip_aperture_target_gap[frame] > 0.0:
                lip_aperture_target_attained[frame] = (
                    _rig_lip_gap_interocular(contact_rig, expression[frame])
                    >= float(lip_aperture_target_gap[frame])
                    - SOURCE_APERTURE_TARGET_TOLERANCE_INTEROCULAR
                )
    rotations, translation, baseline_indices = _pose_from_transforms(
        track,
        filtered,
        filter_config,
        translation_scale_to_gnm=translation_scale_to_gnm,
        coordinate_conversion=conversion,
        baseline_indices=baseline_indices,
        eye_range_radians=eye_range_radians,
    )
    retargeter_name = f"{type(retargeter).__module__}.{type(retargeter).__qualname__}"
    expression_caveats = tuple(retarget_caveats) if retarget_caveats is not None else (
        "MediaPipe blendshapes are ARKit-like semantic coefficients, not a calibrated solve for GNM.",
    )
    provenance = PerformanceProvenance(
        capture_schema_version=track.schema_version,
        capture_source_sha256=track.provenance.source_sha256,
        retargeter=retargeter_name,
        filter_config=filter_config,
        transform_convention=(
            "MediaPipe canonical-face transform made relative to the mean baseline rotation and "
            "median baseline translation; axis-angle applied to GNM head joint; baseline-relative "
            "eye-look controls applied only to dedicated GNM eye joints"
        ),
        translation_scale_to_gnm=float(translation_scale_to_gnm),
        coordinate_conversion=tuple(
            tuple(float(value) for value in row) for row in conversion
        ),
        eye_range_radians=float(eye_range_radians),
        baseline_frame_indices=baseline_indices,
        neutral_blendshape_baseline=tuple(
            (name, float(value))
            for name, value in zip(filtered.names, neutral_baseline, strict=True)
        ),
        neutral_baseline_method=baseline_estimate.method,
        neutral_baseline_validated=baseline_estimate.validated,
        neutral_baseline_correction_applied=baseline_estimate.correction_applied,
        neutral_baseline_score=baseline_estimate.score,
        neutral_baseline_score_limit=NEUTRAL_BASELINE_SCORE_LIMIT,
        neutral_baseline_semantic_peak=baseline_estimate.semantic_peak,
        neutral_baseline_ambiguity_controls=baseline_estimate.ambiguity_controls,
        quarantined_expression_controls=tuple(sorted(QUARANTINED_EXPRESSION_CONTROLS)),
        contact_source_method=SOURCE_CONTACT_METHOD,
        contact_calibration_hash=(
            None
            if lip_contact_calibration is None
            else lip_contact_calibration.calibration_hash
        ),
        aperture_source_method=SOURCE_APERTURE_METHOD,
        negative_baseline_residual_clipped_fraction=negative_residual_fraction,
        caveats=(
            MONOCULAR_SCALE_CAVEAT,
            NEUTRAL_CALIBRATION_CAVEAT,
            *baseline_estimate.caveats,
            *(
                (
                    "The one-sided ARKit source basis clips coefficients below the selected "
                    "neutral baseline; bidirectional subject calibration is required to "
                    "preserve those motions.",
                )
                if negative_residual_fraction > 0.0
                else ()
            ),
            *expression_caveats,
        ),
    )
    return GNMPerformanceTrack(
        identity=identity_value,
        expression=expression,
        rotations=rotations,
        translation=translation,
        timestamps_seconds=track.timestamps_seconds,
        source_pts=track.source_pts,
        detected=track.detected,
        effective_quality=filtered.effective_quality,
        source_lip_geometry_valid=source_lip_geometry_valid,
        source_lip_gap_interocular=source_lip_gap,
        source_lip_contact_confidence=source_lip_contact_confidence,
        lip_contact_target_gap_interocular=lip_contact_target_gap,
        contact_correction_applied=contact_correction_applied,
        lip_contact_attained=lip_contact_attained,
        lip_aperture_target_gap_interocular=lip_aperture_target_gap,
        lip_aperture_correction_applied=lip_aperture_correction_applied,
        lip_aperture_target_attained=lip_aperture_target_attained,
        provenance=provenance,
    )


def write_performance_npz(path: str | Path, track: GNMPerformanceTrack) -> Path:
    provenance = json.dumps(
        track.provenance.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return write_npz(
        path,
        schema_version=np.asarray(track.schema_version),
        identity=track.identity,
        expression=track.expression,
        rotations=track.rotations,
        translation=track.translation,
        timestamps_seconds=track.timestamps_seconds,
        source_pts=track.source_pts,
        detected=track.detected,
        effective_quality=track.effective_quality,
        source_lip_geometry_valid=track.source_lip_geometry_valid,
        source_lip_gap_interocular=track.source_lip_gap_interocular,
        source_lip_contact_confidence=track.source_lip_contact_confidence,
        lip_contact_target_gap_interocular=track.lip_contact_target_gap_interocular,
        contact_correction_applied=track.contact_correction_applied,
        lip_contact_attained=track.lip_contact_attained,
        lip_aperture_target_gap_interocular=(
            track.lip_aperture_target_gap_interocular
        ),
        lip_aperture_correction_applied=track.lip_aperture_correction_applied,
        lip_aperture_target_attained=track.lip_aperture_target_attained,
        provenance_json=np.asarray(provenance),
    )


def write_performance_jsonl(path: str | Path, track: GNMPerformanceTrack) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        metadata = {
            "recordType": "metadata",
            "schemaVersion": track.schema_version,
            "frameCount": track.frame_count,
            "identity": track.identity.tolist(),
            "provenance": track.provenance.as_dict(),
        }
        handle.write(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
        for index in range(track.frame_count):
            record = {
                "recordType": "frame",
                "frameIndex": index,
                "sourcePTS": int(track.source_pts[index]),
                "timestampSeconds": float(track.timestamps_seconds[index]),
                "detected": bool(track.detected[index]),
                "effectiveQuality": float(track.effective_quality[index]),
                "sourceLipGeometryValid": bool(
                    track.source_lip_geometry_valid[index]
                ),
                "sourceLipGapInterocular": float(
                    track.source_lip_gap_interocular[index]
                ),
                "sourceLipContactConfidence": float(
                    track.source_lip_contact_confidence[index]
                ),
                "lipContactTargetGapInterocular": float(
                    track.lip_contact_target_gap_interocular[index]
                ),
                "contactCorrectionApplied": bool(
                    track.contact_correction_applied[index]
                ),
                "lipContactAttained": bool(track.lip_contact_attained[index]),
                "lipApertureTargetGapInterocular": float(
                    track.lip_aperture_target_gap_interocular[index]
                ),
                "lipApertureCorrectionApplied": bool(
                    track.lip_aperture_correction_applied[index]
                ),
                "lipApertureTargetAttained": bool(
                    track.lip_aperture_target_attained[index]
                ),
                "expression": track.expression[index].tolist(),
                "rotations": track.rotations[index].tolist(),
                "translation": track.translation[index].tolist(),
            }
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return destination


def serialize_performance(
    directory: str | Path, track: GNMPerformanceTrack
) -> tuple[Path, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    return (
        write_performance_npz(root / "performance.npz", track),
        write_performance_jsonl(root / "performance.jsonl", track),
    )
