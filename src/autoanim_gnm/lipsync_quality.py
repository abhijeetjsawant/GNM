"""Geometry-first lip-sync quality scoring.

The scorer deliberately does not use a recognizer's own cue timeline as
ground truth.  Audio-derived speech activity is useful for motion hygiene,
but timing/content validation requires independently authored annotations and
their expected mouth-pose prototypes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np


MOUTH_LANDMARKS = slice(48, 68)


@dataclass(frozen=True, slots=True)
class TimingAnnotation:
    """An independently annotated phonetic event.

    ``time_seconds`` is the expected apex/contact time, not the start of a
    broad recognizer cue.  ``label`` must identify a prototype supplied to
    :func:`evaluate_lipsync_quality`.
    """

    time_seconds: float
    label: str


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Conservative gates for a 30 fps review track.

    Timing thresholds are expressed in frames to keep the result legible to
    animators.  The evaluator converts annotations from seconds using the
    requested output frame rate.
    """

    mouth_step_max_interocular: float = 0.04
    speech_active_stationary_fraction: float = 0.12
    neutral_return_frames: int = 2
    false_silence_motion_ratio_p95: float = 0.10
    target_contrast_median: float = 0.80
    target_contrast_p10: float = 0.60
    timing_error_median_frames: float = 1.0
    timing_error_p95_frames: float = 2.0
    minimum_independent_events: int = 3


@dataclass(frozen=True, slots=True)
class ProductionGate:
    """The auditable result of applying :class:`QualityThresholds`."""

    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LipsyncQualityReport:
    """Utility metrics, a comparative score, and a strict production gate."""

    score: float
    metrics: Mapping[str, float | int | None]
    production_gate: ProductionGate

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "metrics": dict(self.metrics),
            "production_gate": asdict(self.production_gate),
        }


def _validated_landmarks(value: np.ndarray, *, name: str, frames: bool) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    expected_ndim = 3 if frames else 2
    if output.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions")
    landmark_axis = 1 if frames else 0
    coordinate_axis = 2 if frames else 1
    if output.shape[landmark_axis] < 68 or output.shape[coordinate_axis] not in (2, 3):
        raise ValueError(f"{name} must contain at least 68 2D or 3D landmarks")
    if not np.isfinite(output).all():
        raise ValueError(f"{name} contains nonfinite values")
    return output


def _face_local_single(landmarks: np.ndarray) -> np.ndarray:
    """Remove translation, scale, and in-plane pose using stable face points."""

    left_eye = landmarks[36]
    right_eye = landmarks[45]
    eye_axis = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_axis))
    if interocular <= 1e-8:
        raise ValueError("interocular distance is zero")
    x_axis = eye_axis / interocular
    eye_midpoint = 0.5 * (left_eye + right_eye)
    if landmarks.shape[1] == 2:
        y_axis = np.asarray([-x_axis[1], x_axis[0]])
        if float(np.dot(landmarks[30] - eye_midpoint, y_axis)) < 0:
            y_axis = -y_axis
        axes = np.stack((x_axis, y_axis), axis=1)
    else:
        nose_direction = landmarks[30] - eye_midpoint
        y_axis = nose_direction - np.dot(nose_direction, x_axis) * x_axis
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm <= 1e-8:
            raise ValueError("eye and nose landmarks cannot define a face-local frame")
        y_axis /= y_norm
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
        axes = np.stack((x_axis, y_axis, z_axis), axis=1)
    return ((landmarks[MOUTH_LANDMARKS] - eye_midpoint) @ axes) / interocular


def _face_local_sequence(landmarks: np.ndarray) -> np.ndarray:
    return np.stack([_face_local_single(frame) for frame in landmarks])


def _rms_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left) - np.asarray(right)
    return np.sqrt(np.mean(np.square(difference), axis=(-2, -1)))


def _silence_core(active: np.ndarray, guard_frames: int) -> np.ndarray:
    if guard_frames <= 0 or not np.any(active):
        return ~active
    padded = np.pad(active.astype(np.int8), guard_frames)
    near_speech = np.convolve(
        padded,
        np.ones(2 * guard_frames + 1, dtype=np.int8),
        mode="same",
    )[guard_frames:-guard_frames] > 0
    return ~near_speech


def _annotation_metrics(
    mouth: np.ndarray,
    annotations: Sequence[TimingAnnotation],
    prototypes: Mapping[str, np.ndarray],
    neutral_mouth: np.ndarray,
    fps: float,
    search_frames: int,
) -> tuple[dict[str, float | int | None], list[float], list[float]]:
    contrasts: list[float] = []
    attainments: list[float] = []
    timing_errors: list[float] = []
    timing_confidences: list[float] = []
    prototype_mouth = {label: _face_local_single(value) for label, value in prototypes.items()}

    for annotation in annotations:
        if annotation.label not in prototype_mouth:
            continue
        expected = int(round(annotation.time_seconds * fps))
        if expected < 0 or expected >= len(mouth):
            continue
        target = prototype_mouth[annotation.label]
        target_distance = float(_rms_distance(mouth[expected], target))
        neutral_to_target = float(_rms_distance(neutral_mouth, target))
        attainment = float(np.clip(1.0 - target_distance / max(neutral_to_target, 1e-8), 0.0, 1.0))
        competitor_distances = [
            float(_rms_distance(mouth[expected], candidate))
            for label, candidate in prototype_mouth.items()
            if label != annotation.label
        ]
        if competitor_distances:
            nearest_competitor = min(competitor_distances)
            identity_contrast = nearest_competitor / max(nearest_competitor + target_distance, 1e-8)
        else:
            identity_contrast = attainment
        contrasts.append(float(np.clip(identity_contrast * attainment, 0.0, 1.0)))
        attainments.append(attainment)

        start = max(0, expected - search_frames)
        end = min(len(mouth), expected + search_frames + 1)
        distances = _rms_distance(mouth[start:end], target)
        minimum = float(np.min(distances))
        candidates = np.flatnonzero(np.isclose(distances, minimum, rtol=1e-7, atol=1e-9)) + start
        predicted = int(candidates[np.argmin(np.abs(candidates - expected))])
        error = float(abs(predicted - expected))
        timing_errors.append(error)
        local_attainment = float(np.clip(1.0 - minimum / max(neutral_to_target, 1e-8), 0.0, 1.0))
        timing_confidences.append(local_attainment * math.exp(-error / 2.0))

    metrics: dict[str, float | int | None] = {
        "annotated_event_count": len(annotations),
        "scored_event_count": len(contrasts),
        "target_contrast_median": float(np.median(contrasts)) if contrasts else None,
        "target_contrast_p10": float(np.percentile(contrasts, 10)) if contrasts else None,
        "target_attainment_median": float(np.median(attainments)) if attainments else None,
        "timing_error_median_frames": float(np.median(timing_errors)) if timing_errors else None,
        "timing_error_p95_frames": float(np.percentile(timing_errors, 95)) if timing_errors else None,
        "timing_confidence_mean": float(np.mean(timing_confidences)) if timing_confidences else None,
    }
    return metrics, contrasts, timing_confidences


def evaluate_lipsync_quality(
    landmarks: np.ndarray,
    neutral_landmarks: np.ndarray,
    speech_activity: np.ndarray,
    *,
    fps: float,
    annotations: Sequence[TimingAnnotation] | None = None,
    annotations_are_independent: bool = False,
    target_prototypes: Mapping[str, np.ndarray] | None = None,
    thresholds: QualityThresholds | None = None,
    stationary_step_interocular: float = 5e-4,
    neutral_tolerance_interocular: float = 0.015,
    silence_guard_frames: int = 2,
    timing_search_frames: int = 6,
) -> LipsyncQualityReport:
    """Evaluate a rendered landmark track without circular cue scoring.

    Geometry and audio-VAD metrics are always available.  Target contrast and
    timing error are only computed from ``annotations`` plus explicit geometry
    ``target_prototypes``.  The production gate *also* requires the caller to
    affirm that those annotations were created independently from the system
    being evaluated; supplying its own Rhubarb/ASR cues must leave
    ``annotations_are_independent`` false.
    """

    frames = _validated_landmarks(landmarks, name="landmarks", frames=True)
    neutral = _validated_landmarks(neutral_landmarks, name="neutral_landmarks", frames=False)
    if frames.shape[0] < 2:
        raise ValueError("landmarks must contain at least two frames")
    if frames.shape[2] != neutral.shape[1]:
        raise ValueError("landmarks and neutral_landmarks coordinate dimensions differ")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    active_values = np.asarray(speech_activity, dtype=np.float64)
    if active_values.shape != (len(frames),) or not np.isfinite(active_values).all():
        raise ValueError("speech_activity must be one finite value per frame")
    if stationary_step_interocular < 0 or neutral_tolerance_interocular < 0:
        raise ValueError("geometry tolerances must be nonnegative")
    if silence_guard_frames < 0 or timing_search_frames < 0:
        raise ValueError("frame windows must be nonnegative")

    prototypes: dict[str, np.ndarray] = {}
    for label, value in (target_prototypes or {}).items():
        prototype = _validated_landmarks(value, name=f"target_prototypes[{label!r}]", frames=False)
        if prototype.shape[1] != neutral.shape[1]:
            raise ValueError("target prototype coordinate dimensions differ")
        prototypes[str(label)] = prototype
    annotation_list = tuple(annotations or ())
    if any(not item.label or not np.isfinite(item.time_seconds) for item in annotation_list):
        raise ValueError("annotations require a finite time and nonempty label")

    mouth = _face_local_sequence(frames)
    neutral_mouth = _face_local_single(neutral)
    active = active_values >= 0.5
    step = np.max(np.linalg.norm(np.diff(mouth, axis=0), axis=2), axis=1)
    active_transitions = active[:-1] | active[1:]
    if np.any(active_transitions):
        stationary_fraction: float | None = float(
            np.mean(step[active_transitions] <= stationary_step_interocular)
        )
    else:
        stationary_fraction = None

    deformation = _rms_distance(mouth, neutral_mouth)
    if np.any(active):
        last_active = int(np.flatnonzero(active)[-1])
        return_candidates = np.flatnonzero(
            deformation[last_active + 1 :] <= neutral_tolerance_interocular
        )
        neutral_return: int | None = (
            int(return_candidates[0] + 1) if len(return_candidates) else None
        )
    else:
        neutral_return = 0 if deformation[0] <= neutral_tolerance_interocular else None

    reference_amplitudes = [
        float(_rms_distance(_face_local_single(prototype), neutral_mouth))
        for prototype in prototypes.values()
    ]
    reference_motion = max(reference_amplitudes, default=float(np.max(deformation)))
    silence = _silence_core(active, silence_guard_frames)
    false_silence = (
        float(np.percentile(deformation[silence], 95) / max(reference_motion, 1e-8))
        if np.any(silence)
        else 0.0
    )

    annotation_metrics, contrasts, timing_confidences = _annotation_metrics(
        mouth,
        annotation_list,
        prototypes,
        neutral_mouth,
        fps,
        timing_search_frames,
    )
    metrics: dict[str, float | int | None] = {
        "mouth_step_max_interocular": float(np.max(step)),
        "mouth_step_p95_interocular": float(np.percentile(step, 95)),
        "speech_active_stationary_fraction": stationary_fraction,
        "neutral_return_frames": neutral_return,
        "false_silence_motion_ratio_p95": false_silence,
        **annotation_metrics,
    }

    limits = thresholds or QualityThresholds()
    independent = bool(annotation_list) and bool(annotations_are_independent)
    complete_events = int(metrics["scored_event_count"] or 0) == len(annotation_list)

    def at_most(value: float | int | None, maximum: float) -> bool:
        return value is not None and float(value) <= maximum

    def at_least(value: float | int | None, minimum: float) -> bool:
        return value is not None and float(value) >= minimum

    checks = {
        "independent_annotations": independent,
        "minimum_event_count": len(annotation_list) >= limits.minimum_independent_events,
        "all_events_have_prototypes": bool(annotation_list) and complete_events,
        "speech_present": bool(np.any(active)),
        "mouth_step": at_most(metrics["mouth_step_max_interocular"], limits.mouth_step_max_interocular),
        "speech_active_motion": at_most(
            metrics["speech_active_stationary_fraction"], limits.speech_active_stationary_fraction
        ),
        "neutral_return": at_most(metrics["neutral_return_frames"], limits.neutral_return_frames),
        "false_silence_motion": at_most(
            metrics["false_silence_motion_ratio_p95"], limits.false_silence_motion_ratio_p95
        ),
        "target_contrast_median": at_least(
            metrics["target_contrast_median"], limits.target_contrast_median
        ),
        "target_contrast_p10": at_least(metrics["target_contrast_p10"], limits.target_contrast_p10),
        "timing_error_median": at_most(
            metrics["timing_error_median_frames"], limits.timing_error_median_frames
        ),
        "timing_error_p95": at_most(
            metrics["timing_error_p95_frames"], limits.timing_error_p95_frames
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)

    step_score = min(1.0, limits.mouth_step_max_interocular / max(float(np.max(step)), 1e-8))
    dynamics_score = 0.0 if stationary_fraction is None else 1.0 - stationary_fraction
    neutral_score = 0.0 if neutral_return is None else max(
        0.0, 1.0 - max(0, neutral_return - limits.neutral_return_frames) / 6.0
    )
    silence_score = max(0.0, 1.0 - false_silence / max(limits.false_silence_motion_ratio_p95, 1e-8))
    contrast_score = float(np.mean(contrasts)) if contrasts else 0.0
    timing_score = float(np.mean(timing_confidences)) if timing_confidences else 0.0
    score = 100.0 * (
        0.10 * step_score
        + 0.10 * dynamics_score
        + 0.10 * neutral_score
        + 0.10 * silence_score
        + 0.35 * contrast_score
        + 0.25 * timing_score
    )
    if not independent:
        score = min(score, 49.0)
    return LipsyncQualityReport(
        score=round(float(np.clip(score, 0.0, 100.0)), 3),
        metrics=metrics,
        production_gate=ProductionGate(not failures, checks, failures),
    )
