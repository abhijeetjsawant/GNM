from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from autoanim_gnm.lipsync_quality import TimingAnnotation, evaluate_lipsync_quality


FPS = 30.0
EVENT_FRAMES = (24, 34, 44, 54)
EVENT_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class SyntheticPerformance:
    neutral: np.ndarray
    prototypes: dict[str, np.ndarray]
    annotations: tuple[TimingAnnotation, ...]
    speech_activity: np.ndarray
    correct: np.ndarray


def _neutral_face() -> np.ndarray:
    face = np.zeros((68, 2), dtype=np.float64)
    face[36] = (-0.5, 0.0)
    face[45] = (0.5, 0.0)
    face[30] = (0.0, 0.42)
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    face[48:68, 0] = 0.22 * np.cos(angles)
    face[48:68, 1] = 0.78 + 0.08 * np.sin(angles)
    return face


def _prototypes(neutral: np.ndarray) -> dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    output = {label: neutral.copy() for label in EVENT_LABELS}
    # Four intentionally distinct articulatory targets: closure, opening,
    # rounding, and lateral smile/stretch.
    output["A"][48:68, 1] = 0.78 + 0.012 * np.sin(angles)
    output["B"][48:68, 1] = 0.78 + 0.155 * np.sin(angles)
    output["C"][48:68, 0] = 0.105 * np.cos(angles)
    output["D"][48:68, 0] = 0.305 * np.cos(angles)
    output["D"][48:68, 1] -= 0.025 * np.abs(np.cos(angles))
    return output


def _performance_with_order(
    neutral: np.ndarray,
    prototypes: dict[str, np.ndarray],
    labels: tuple[str, ...],
) -> np.ndarray:
    frame_count = 90
    track = np.repeat(neutral[None], frame_count, axis=0)
    keys = [(14, neutral)]
    keys.extend(zip(EVENT_FRAMES, (prototypes[label] for label in labels), strict=True))
    keys.append((61, neutral))
    for (left_frame, left), (right_frame, right) in zip(keys[:-1], keys[1:], strict=True):
        for frame in range(left_frame, right_frame + 1):
            alpha = (frame - left_frame) / (right_frame - left_frame)
            track[frame] = (1.0 - alpha) * left + alpha * right
    return track


@pytest.fixture(scope="module")
def performance() -> SyntheticPerformance:
    neutral = _neutral_face()
    prototypes = _prototypes(neutral)
    annotations = tuple(
        TimingAnnotation(frame / FPS, label)
        for frame, label in zip(EVENT_FRAMES, EVENT_LABELS, strict=True)
    )
    speech_activity = np.zeros(90, dtype=np.float64)
    speech_activity[14:61] = 1.0
    correct = _performance_with_order(neutral, prototypes, EVENT_LABELS)
    return SyntheticPerformance(neutral, prototypes, annotations, speech_activity, correct)


def _score(performance: SyntheticPerformance, landmarks: np.ndarray):
    return evaluate_lipsync_quality(
        landmarks,
        performance.neutral,
        performance.speech_activity,
        fps=FPS,
        annotations=performance.annotations,
        annotations_are_independent=True,
        target_prototypes=performance.prototypes,
    )


def _shift_with_neutral(track: np.ndarray, neutral: np.ndarray, frames: int) -> np.ndarray:
    shifted = np.repeat(neutral[None], len(track), axis=0)
    if frames > 0:
        shifted[frames:] = track[:-frames]
    elif frames < 0:
        shifted[:frames] = track[-frames:]
    else:
        shifted[:] = track
    return shifted


def _moving_average(track: np.ndarray, width: int) -> np.ndarray:
    radius = width // 2
    padded = np.pad(track, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.concatenate((np.zeros_like(cumulative[:1]), cumulative), axis=0)
    return (cumulative[width:] - cumulative[:-width]) / width


def test_correct_track_passes_and_reports_geometry_timing(performance: SyntheticPerformance) -> None:
    report = _score(performance, performance.correct)
    assert report.production_gate.passed, report.production_gate.failures
    assert report.metrics["mouth_step_max_interocular"] < 0.04
    assert report.metrics["speech_active_stationary_fraction"] < 0.12
    assert report.metrics["neutral_return_frames"] == 1
    assert report.metrics["false_silence_motion_ratio_p95"] == pytest.approx(0.0)
    assert report.metrics["target_contrast_median"] == pytest.approx(1.0)
    assert report.metrics["timing_error_median_frames"] == pytest.approx(0.0)


@pytest.mark.parametrize("shift", (-4, -2, 2, 4))
def test_independent_events_reject_shifted_tracks(
    performance: SyntheticPerformance,
    shift: int,
) -> None:
    correct = _score(performance, performance.correct)
    shifted = _score(performance, _shift_with_neutral(performance.correct, performance.neutral, shift))
    assert shifted.metrics["timing_error_median_frames"] == pytest.approx(abs(shift))
    assert shifted.score < correct.score
    assert not shifted.production_gate.passed
    assert "timing_error_median" in shifted.production_gate.failures


def test_heavy_smoothing_is_not_mistaken_for_high_quality(performance: SyntheticPerformance) -> None:
    correct = _score(performance, performance.correct)
    smoothed = _score(performance, _moving_average(performance.correct, 21))
    assert smoothed.score < correct.score
    assert smoothed.metrics["target_contrast_median"] < correct.metrics["target_contrast_median"]
    assert not smoothed.production_gate.passed
    assert any(name.startswith("target_contrast") for name in smoothed.production_gate.failures)


def test_static_neutral_and_constant_open_are_rejected(performance: SyntheticPerformance) -> None:
    correct = _score(performance, performance.correct)
    static_neutral = np.repeat(performance.neutral[None], len(performance.correct), axis=0)
    constant_open = np.repeat(performance.prototypes["B"][None], len(performance.correct), axis=0)
    for adversarial in (static_neutral, constant_open):
        report = _score(performance, adversarial)
        assert report.score < correct.score
        assert not report.production_gate.passed
        assert "speech_active_motion" in report.production_gate.failures
        assert "target_contrast_median" in report.production_gate.failures
    open_report = _score(performance, constant_open)
    assert "false_silence_motion" in open_report.production_gate.failures
    assert "neutral_return" in open_report.production_gate.failures


def test_cue_permutation_fails_geometric_target_contrast(performance: SyntheticPerformance) -> None:
    correct = _score(performance, performance.correct)
    permuted = _performance_with_order(
        performance.neutral,
        performance.prototypes,
        ("B", "C", "D", "A"),
    )
    report = _score(performance, permuted)
    assert report.score < correct.score
    assert report.metrics["target_contrast_median"] < 0.60
    assert not report.production_gate.passed
    assert "target_contrast_median" in report.production_gate.failures


def test_emotion_only_mouth_motion_during_silence_fails(performance: SyntheticPerformance) -> None:
    correct = _score(performance, performance.correct)
    emotion_in_silence = performance.correct.copy()
    phase = np.linspace(0.0, np.pi, 12)
    emotion_in_silence[:12, 48:68, 0] += 0.065 * np.sin(phase)[:, None]
    report = _score(performance, emotion_in_silence)
    assert report.score < correct.score
    assert report.metrics["false_silence_motion_ratio_p95"] > 0.10
    assert not report.production_gate.passed
    assert "false_silence_motion" in report.production_gate.failures


def test_production_gate_refuses_self_scored_or_missing_annotations(
    performance: SyntheticPerformance,
) -> None:
    missing = evaluate_lipsync_quality(
        performance.correct,
        performance.neutral,
        performance.speech_activity,
        fps=FPS,
    )
    assert not missing.production_gate.passed
    assert "independent_annotations" in missing.production_gate.failures
    assert missing.metrics["timing_error_median_frames"] is None
    assert missing.score <= 49.0

    self_scored = evaluate_lipsync_quality(
        performance.correct,
        performance.neutral,
        performance.speech_activity,
        fps=FPS,
        annotations=performance.annotations,
        annotations_are_independent=False,
        target_prototypes=performance.prototypes,
    )
    assert not self_scored.production_gate.passed
    assert self_scored.production_gate.failures == ("independent_annotations",)
    assert self_scored.score <= 49.0


def test_invalid_geometry_is_rejected(performance: SyntheticPerformance) -> None:
    bad = performance.correct.copy()
    bad[3, 48, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        _score(performance, bad)
