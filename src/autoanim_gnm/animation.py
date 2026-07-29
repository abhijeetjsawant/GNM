"""Timed GNM control composition and mesh-backed video rendering."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .artifacts import sha256 as artifact_sha256
from .articulation_projection import (
    articulation_array_sha256,
    limit_articulation_edge,
    project_articulation_trajectory,
)
from .audio import MouthCue, ProsodyTrack
from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .render import MeshRenderer
from .rig import ControlRig


@dataclass(frozen=True, slots=True)
class AnimationTrack:
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray
    timestamps: np.ndarray
    fps: int
    saturated: bool
    viseme_weights: np.ndarray
    speech_activity: np.ndarray
    energy: np.ndarray
    pitch_semitones: np.ndarray
    accent: np.ndarray
    phrase_id: np.ndarray
    emotion_intensity: np.ndarray
    mouth_speed_limited: np.ndarray
    lip_contact_confidence: np.ndarray
    lip_contact_target_gap: np.ndarray
    contact_correction_applied: np.ndarray
    lip_contact_attained: np.ndarray
    contact_continuity_restored: np.ndarray
    contact_corrected: np.ndarray
    lip_order_repaired: np.ndarray
    articulation_projection_report: dict[str, object] | None = None
    articulation_projection_desired: np.ndarray | None = None
    articulation_projection_output: np.ndarray | None = None
    source_oral_gate: np.ndarray | None = None
    source_neutral_handling: dict[str, object] | None = None
    contact_run_stabilized: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class LipContactCalibration:
    """Character-rig-specific soft-contact solve in GNM expression space.

    ``direction`` is deliberately restricted to GNM's lower-face modes and is
    solved against both the inner-lip target and dense zero-displacement
    constraints outside the lips. ``maximum_alpha`` is the first measured
    minimum of the character's lip gap, so runtime correction cannot continue
    through that minimum into an inverted/reopened pose.
    """

    direction: np.ndarray
    inner_response: np.ndarray
    neutral_pair_gaps_interocular: np.ndarray
    seal_pair_gaps_interocular: np.ndarray
    neutral_gap_interocular: float
    seal_gap_interocular: float
    maximum_alpha: float
    nonmouth_p95_displacement_interocular: float
    nonmouth_max_displacement_interocular: float
    calibration_hash: str


def _smooth_alpha(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 0.5 - 0.5 * math.cos(math.pi * value)


CUE_ORDER = "XABCDEFGH"
_CUE_INDEX = {cue: index for index, cue in enumerate(CUE_ORDER)}
_DOMINANCE = {"X": 1.15, "A": 2.8, "B": 1.0, "C": 1.0, "D": 1.05, "E": 1.1, "F": 1.2, "G": 2.2, "H": 1.8}
LEARNED_SOURCE_NEUTRAL_POLICY_QUIET_MEDIAN = "clip_quiet_median_delta_v1"
LEARNED_SOURCE_NEUTRAL_POLICY_ABSOLUTE_ORAL_GATE = (
    "neutral_relative_absolute_oral_gate_v1"
)
_LEARNED_SOURCE_NEUTRAL_POLICIES = frozenset(
    (
        LEARNED_SOURCE_NEUTRAL_POLICY_QUIET_MEDIAN,
        LEARNED_SOURCE_NEUTRAL_POLICY_ABSOLUTE_ORAL_GATE,
    )
)
_ORAL_GATE_ZERO_ACTIVITY = 0.02
_ORAL_GATE_FULL_ACTIVITY = 0.08
_ORAL_GATE_MINIMUM_SILENCE_SECONDS = 0.20
_ORAL_GATE_TRANSITION_SECONDS = 0.125
_ABSOLUTE_CONTACT_PREPROJECTION_HORIZON_SECONDS = 8.0 / 30.0
_ABSOLUTE_CONTACT_PREPROJECTION_METHOD = "minimal_bidirectional_edge_projection_v1"


@lru_cache(maxsize=1)
def _static_articulation_rig_binding() -> tuple[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return (
        artifact_sha256(project_root / "gnm/shape/data/versions/v3_0/gnm_head.npz"),
        artifact_sha256(project_root / "gnm/shape/data/landmarks/head_sparse_68.txt"),
    )


def _articulation_evidence_bindings(
    rig: ControlRig,
) -> tuple[dict[str, str], dict[str, str]]:
    metric_contract = {
        "mouth_step": "autoanim.face-local-mouth-20-landmarks-IOD/1.0",
        "mouth_gap": "autoanim.inner-lip-3-pair-mean-IOD/1.0",
        "lip_order": "autoanim.inner-lip-order-minimum-IOD/1.0",
    }
    gnm_head_sha256, landmark_regressor_sha256 = _static_articulation_rig_binding()
    rig_binding = {
        "gnm_head_sha256": gnm_head_sha256,
        "landmark_regressor_sha256": landmark_regressor_sha256,
        "identity_array_sha256": articulation_array_sha256(rig.identity),
    }
    return metric_contract, rig_binding


def _articulation_projection_config(
    *, external_face_controls: bool
) -> dict[str, float]:
    """Return the frozen compiler-specific oral projection contract."""

    if external_face_controls:
        maximum_step = 0.039
        maximum_speed = 1.17
        contact_horizon_seconds = 4.0 / 30.0
    else:
        maximum_step = 0.0365
        maximum_speed = 1.095
        contact_horizon_seconds = 8.0 / 30.0
    return {
        "maximum_step": maximum_step,
        "maximum_speed": maximum_speed,
        "contact_tolerance": 0.001,
        "lip_order_floor": -0.0005,
        "contact_horizon_seconds": contact_horizon_seconds,
    }


def _articulation_projection_compiler_metadata(
    *,
    external_face_controls: bool,
    frame_count: int,
    boundary_rest_frames: list[int] | None = None,
) -> dict[str, object]:
    """Describe the exact authored stage captured by the desired snapshot."""

    if external_face_controls:
        if boundary_rest_frames is None:
            snapshot_stage = (
                "post_resample_post_affect_post_contact_post_lip_order_post_boundary_rest"
            )
            boundary_frames = [0, frame_count - 1]
        else:
            snapshot_stage = (
                "post_resample_post_affect_post_contact_post_lip_order_"
                "post_conditional_provider_neutral_boundary"
            )
            boundary_frames = list(boundary_rest_frames)
    else:
        snapshot_stage = "post_resample_post_affect_post_contact_post_terminal_rest"
        boundary_frames = [frame_count - 1]
    return {
        "desired_snapshot_stage": snapshot_stage,
        "boundary_rest_frames": boundary_frames,
        "boundary_rest_range": [200, 382],
        "boundary_rest_included_before_desired_hash": True,
        "lip_order_repair_included_before_desired_hash": external_face_controls,
    }


def _smooth_array(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * value)


def _learned_oral_source_gate(
    speech_activity: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Return a smooth, time-bounded oral silence gate.

    Only contiguous, settled rest runs qualify. Short low-energy dips remain
    fully source-authored so a quiet phone cannot be erased. Cosine shoulders
    live inside each qualified rest run: adjoining speech and contact frames
    therefore keep full authority while protected tongue modes return smoothly
    to the provider's neutral pose.
    """

    activity = np.asarray(speech_activity, dtype=np.float32)
    clock = np.asarray(timestamps, dtype=np.float64)
    if (
        activity.ndim != 1
        or clock.shape != activity.shape
        or not np.isfinite(activity).all()
        or not np.isfinite(clock).all()
        or np.any((activity < 0.0) | (activity > 1.0))
        or (len(clock) > 1 and np.any(np.diff(clock) <= 0.0))
    ):
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Learned source oral gating requires finite speech activity in [0,1]",
        )
    if not len(activity):
        return activity.copy()
    gate = np.ones(len(activity), dtype=np.float64)
    settled_rest = activity <= _ORAL_GATE_ZERO_ACTIVITY
    run_start: int | None = None
    runs: list[tuple[int, int]] = []
    for index in range(len(settled_rest) + 1):
        is_rest = bool(settled_rest[index]) if index < len(settled_rest) else False
        if is_rest and run_start is None:
            run_start = index
        elif not is_rest and run_start is not None:
            run_end = index - 1
            sample_seconds = (
                float(np.median(np.diff(clock))) if len(clock) > 1 else 0.0
            )
            run_seconds = float(clock[run_end] - clock[run_start] + sample_seconds)
            if run_seconds + 1.0e-9 >= _ORAL_GATE_MINIMUM_SILENCE_SECONDS:
                runs.append((run_start, run_end))
            run_start = None

    for start, end in runs:
        gate[start : end + 1] = 0.0
        if start > 0:
            elapsed = clock[start : end + 1] - clock[start]
            release = np.clip(elapsed / _ORAL_GATE_TRANSITION_SECONDS, 0.0, 1.0)
            gate[start : end + 1] = np.maximum(
                gate[start : end + 1], 1.0 - _smooth_array(release)
            )
        if end < len(gate) - 1:
            until_speech = clock[end] - clock[start : end + 1]
            attack = np.clip(
                until_speech / _ORAL_GATE_TRANSITION_SECONDS, 0.0, 1.0
            )
            gate[start : end + 1] = np.maximum(
                gate[start : end + 1], 1.0 - _smooth_array(attack)
            )
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def _activation_matrix(cues: list[MouthCue], timestamps: np.ndarray) -> np.ndarray:
    """Compile bounded, dominance-aware cue activations.

    Cue intervals remain authoritative. Small attack/release shoulders create
    coarticulation while high-value contact shapes win short overlaps.
    """

    if not cues:
        raise AutoAnimError("CUE_INVALID", "At least one normalized mouth cue is required")
    raw = np.zeros((len(timestamps), len(CUE_ORDER)), dtype=np.float64)
    for cue in cues:
        attack = 0.032 if cue.value in "AGH" else (0.045 if cue.value != "X" else 0.055)
        release = 0.045 if cue.value in "AGH" else (0.075 if cue.value != "X" else 0.050)
        activation = np.zeros(len(timestamps), dtype=np.float64)
        inside = (timestamps >= cue.start) & (timestamps <= cue.end)
        activation[inside] = 1.0
        before = (timestamps >= cue.start - attack) & (timestamps < cue.start)
        activation[before] = _smooth_array((timestamps[before] - (cue.start - attack)) / attack)
        after = (timestamps > cue.end) & (timestamps <= cue.end + release)
        activation[after] = 1.0 - _smooth_array((timestamps[after] - cue.end) / release)
        raw[:, _CUE_INDEX[cue.value]] += activation * _DOMINANCE[cue.value]

    # Keep the composer local even for abnormally short cues: no more than
    # the two strongest adjacent influences survive at a frame.
    if raw.shape[1] > 2:
        keep = np.argpartition(raw, -2, axis=1)[:, -2:]
        mask = np.zeros_like(raw, dtype=bool)
        rows = np.arange(len(raw))[:, None]
        mask[rows, keep] = True
        raw = np.where(mask, raw, 0.0)
    totals = raw.sum(axis=1, keepdims=True)
    missing = totals[:, 0] <= 1e-12
    raw[missing, 0] = 1.0
    totals = raw.sum(axis=1, keepdims=True)
    return (raw / totals).astype(np.float32)


def _default_prosody(cues: list[MouthCue], timestamps: np.ndarray) -> ProsodyTrack:
    ends = np.asarray([cue.end for cue in cues], dtype=np.float64)
    indices = np.minimum(np.searchsorted(ends, timestamps, side="right"), len(cues) - 1)
    active = np.asarray([cues[int(index)].value != "X" for index in indices], dtype=np.float32)
    phrase = np.zeros(len(timestamps), dtype=np.int32)
    return ProsodyTrack(
        timestamps=timestamps,
        rms_dbfs=np.where(active > 0, -30.0, -80.0).astype(np.float32),
        energy=(0.55 * active).astype(np.float32),
        speech_activity=active,
        pitch_semitones=np.zeros(len(timestamps), dtype=np.float32),
        accent=(0.45 * active).astype(np.float32),
        phrase_id=phrase,
    )


def _validate_prosody(prosody: ProsodyTrack, timestamps: np.ndarray) -> None:
    arrays = (
        prosody.timestamps,
        prosody.rms_dbfs,
        prosody.energy,
        prosody.speech_activity,
        prosody.pitch_semitones,
        prosody.accent,
        prosody.phrase_id,
    )
    if any(len(array) != len(timestamps) for array in arrays):
        raise AutoAnimError("INTERNAL_ERROR", "Prosody and animation frame counts differ")
    if any(not np.isfinite(array).all() for array in arrays):
        raise AutoAnimError("INTERNAL_ERROR", "Prosody contains nonfinite values")


def _emotion_envelope(
    duration: float,
    fps: int,
    timestamps: np.ndarray,
    prosody: ProsodyTrack,
) -> np.ndarray:
    window = max(3, int(round(0.35 * fps)))
    kernel = np.hanning(window)
    if not np.any(kernel):
        kernel = np.ones(window)
    kernel /= kernel.sum()
    slow_accent = np.convolve(prosody.accent, kernel, mode="same")
    intensity = (0.28 + 0.57 * slow_accent) * (0.10 + 0.90 * prosody.speech_activity)
    edge = min(0.30, duration / 2)
    if edge > 0:
        fade_in = np.asarray([_smooth_alpha(float(t / edge)) if t < edge else 1.0 for t in timestamps])
        fade_out = np.asarray(
            [_smooth_alpha(float((duration - t) / edge)) if duration - t < edge else 1.0 for t in timestamps]
        )
        intensity *= np.minimum(fade_in, fade_out)
    return np.clip(intensity, 0.0, 1.0).astype(np.float32)


def _blink_envelope(timestamps: np.ndarray) -> np.ndarray:
    output = np.zeros(len(timestamps), dtype=np.float32)
    if not len(timestamps):
        return output
    duration = float(timestamps[-1])
    for center in np.arange(3.15, duration, 3.85):
        half = 0.070
        selected = np.abs(timestamps - center) <= half
        phase = 1.0 - np.abs(timestamps[selected] - center) / half
        output[selected] = np.maximum(output[selected], _smooth_array(phase).astype(np.float32))
    return output


def _head_motion(
    timestamps: np.ndarray,
    prosody: ProsodyTrack,
    joint_count: int,
) -> np.ndarray:
    rotations = np.zeros((len(timestamps), joint_count, 3), dtype=np.float32)
    if joint_count < 2 or not len(timestamps):
        return rotations
    speech = np.asarray(prosody.speech_activity, dtype=np.float64)
    accent = np.asarray(prosody.accent, dtype=np.float64)
    # Restrained, band-limited secondary motion. This is intentionally not
    # described as recovered performance: audio does not determine a unique
    # head pose or gaze.  The amplitudes are nevertheless large enough to
    # avoid the sub-degree mannequin motion of the original prototype.
    accent_rate = np.gradient(accent) if len(accent) > 1 else np.zeros_like(accent)
    yaw = (
        np.deg2rad(1.45) * np.sin(2 * np.pi * 0.17 * timestamps + 0.15) * speech
        + np.deg2rad(0.55) * accent_rate
    )
    roll = np.deg2rad(0.55) * np.sin(2 * np.pi * 0.12 * timestamps + 0.8) * speech
    pitch = (
        np.deg2rad(2.35) * accent * np.sin(2 * np.pi * 0.67 * timestamps)
        + np.deg2rad(0.65) * accent_rate
    ) * speech
    duration = float(timestamps[-1] + (timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.0))
    edge = min(0.28, duration / 2)
    if edge > 0:
        settle = np.minimum(
            np.clip(timestamps / edge, 0.0, 1.0),
            np.clip((duration - timestamps) / edge, 0.0, 1.0),
        )
        settle = _smooth_array(settle)
        pitch *= settle
        yaw *= settle
        roll *= settle
    rotations[:, 1, 0] = pitch.astype(np.float32)
    rotations[:, 1, 1] = yaw.astype(np.float32)
    rotations[:, 1, 2] = roll.astype(np.float32)
    rotations[:, 0] = -0.28 * rotations[:, 1]
    if joint_count >= 4:
        # Shared slow gaze with a small vestibulo-ocular counter-rotation.
        # Saccades cannot be recovered from audio, so this track is procedural
        # and deterministic rather than random or falsely source-attributed.
        gaze_yaw = np.deg2rad(0.85) * np.sin(2 * np.pi * 0.095 * timestamps + 1.1)
        gaze_pitch = np.deg2rad(0.40) * np.sin(2 * np.pi * 0.14 * timestamps + 0.35)
        rotations[:, 2, 0] = (gaze_pitch - 0.18 * pitch).astype(np.float32)
        rotations[:, 2, 1] = (gaze_yaw - 0.24 * yaw).astype(np.float32)
        rotations[:, 3, 0] = rotations[:, 2, 0]
        rotations[:, 3, 1] = rotations[:, 2, 1]
    rotations[0] = 0.0
    rotations[-1] = 0.0
    return rotations


def _mouth_gap_interocular(rig: ControlRig, expression: np.ndarray) -> float:
    landmarks = rig.compact_landmarks(expression)
    gap = np.mean(
        [
            np.linalg.norm(landmarks[upper] - landmarks[lower])
            for upper, lower in ((61, 67), (62, 66), (63, 65))
        ]
    )
    neutral = np.asarray(rig.neutral_landmarks, dtype=np.float64)
    interocular = float(np.linalg.norm(neutral[36] - neutral[45]))
    if interocular <= 0.0:
        raise AutoAnimError("INTERNAL_ERROR", "GNM interocular distance is invalid")
    return float(gap / interocular)


def _mouth_lip_order_minimum_interocular(
    rig: ControlRig,
    expression: np.ndarray,
) -> float:
    """Signed inner-lip order in the same geometry convention as oral QA."""

    landmarks = rig.compact_landmarks(expression)
    neutral = np.asarray(rig.neutral_landmarks, dtype=np.float64)
    interocular = float(np.linalg.norm(neutral[36] - neutral[45]))
    face_up = landmarks[27] - landmarks[8]
    face_up_norm = float(np.linalg.norm(face_up))
    if interocular <= 1.0e-8 or face_up_norm <= 1.0e-8:
        raise AutoAnimError("INTERNAL_ERROR", "GNM lip-order frame is invalid")
    face_up = face_up / np.float32(face_up_norm)
    return float(
        min(
            np.dot(landmarks[upper] - landmarks[lower], face_up) / interocular
            for upper, lower in ((61, 67), (62, 66), (63, 65))
        )
    )


def _repair_lip_order_inversion(
    rig: ControlRig,
    expression: np.ndarray,
    *,
    minimum_order: float = -5.0e-4,
) -> tuple[np.ndarray, bool]:
    """Project an inverted lower-lip pose to the nearest safe GNM control.

    Only lower-face modes 200:350 are attenuated. Tongue, upper-face, and
    reserved coefficients remain exact. The solve keeps the largest fraction
    of the incoming performance whose measured inner-lip ordering is valid.
    """

    source = np.asarray(expression, dtype=np.float32)
    if _mouth_lip_order_minimum_interocular(rig, source) >= minimum_order:
        return source.copy(), False

    def candidate(alpha: float) -> np.ndarray:
        output = source.copy()
        output[200:350] = np.float32(alpha) * source[200:350]
        return output

    if _mouth_lip_order_minimum_interocular(rig, candidate(0.0)) < minimum_order:
        raise AutoAnimError(
            "ORAL_LIP_ORDER_UNREPAIRABLE",
            "Upper-face or tongue controls invert the inner lips even with a neutral lower face",
        )
    lower = 0.0
    upper = 1.0
    for _ in range(24):
        middle = 0.5 * (lower + upper)
        if _mouth_lip_order_minimum_interocular(rig, candidate(middle)) >= minimum_order:
            lower = middle
        else:
            upper = middle
    repaired = candidate(lower * 0.999999)
    if _mouth_lip_order_minimum_interocular(rig, repaired) < minimum_order - 1.0e-7:
        raise AutoAnimError(
            "ORAL_LIP_ORDER_UNREPAIRABLE",
            "Lower-face projection did not produce safe inner-lip ordering",
        )
    return repaired.astype(np.float32), True


def _face_local_mouth(rig: ControlRig, expression: np.ndarray) -> np.ndarray:
    """Return the mouth in the same normalized frame as the production gate."""

    landmarks = rig.compact_landmarks(expression)
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    eye_axis = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_axis))
    if interocular <= 1e-8:
        raise AutoAnimError("INTERNAL_ERROR", "GNM interocular distance is invalid")
    x_axis = eye_axis / interocular
    eye_midpoint = np.float32(0.5) * (left_eye + right_eye)
    nose_direction = landmarks[30] - eye_midpoint
    y_axis = nose_direction - np.dot(nose_direction, x_axis) * x_axis
    y_length = float(np.linalg.norm(y_axis))
    if y_length <= 1e-8:
        raise AutoAnimError("INTERNAL_ERROR", "GNM face-local frame is invalid")
    y_axis /= y_length
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
    axes = np.stack((x_axis, y_axis, z_axis), axis=1)
    return ((landmarks[48:68] - eye_midpoint) @ axes) / interocular


def _mouth_step_quality_ratio(
    rig: ControlRig,
    previous: np.ndarray,
    target: np.ndarray,
) -> float:
    before_mouth = _face_local_mouth(rig, previous)
    return float(
        np.max(
            np.linalg.norm(_face_local_mouth(rig, target) - before_mouth, axis=1),
            initial=0.0,
        )
    )


_INNER_LIP_PAIRS = ((61, 67), (62, 66), (63, 65))


def calibrate_lip_contact(rig: ControlRig) -> LipContactCalibration:
    """Solve a spatially local, character-specific GNM lip-contact path.

    A contact direction made by adding generic ``mouthPress`` rows can carry
    large global PCA coupling into the nose and cheeks. This solve instead
    asks the current rig to bring its three inner-lip pairs toward their
    character-neutral midpoints while strongly penalizing motion at all
    non-mouth sparse landmarks and at a dense sample of exterior skin outside
    the upper/lower-lip vertex groups. The coefficient norm is regularized as
    a final tie-breaker. All objectives are normalized by their row count so
    their weights describe intent rather than mesh resolution.

    This is still a blendshape-space approximation, not collision response.
    The calibrated seal is the first minimum reachable by the rig, and runtime
    correction never extrapolates beyond it.
    """

    expression_dim = rig.adapter.expression_dim
    if expression_dim < 350:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "GNM lower-face contact calibration requires expression modes 200:350",
        )
    neutral = np.asarray(rig.neutral_landmarks, dtype=np.float64)
    compact_basis = np.asarray(
        rig.adapter.compact_expression_basis[200:350],
        dtype=np.float64,
    )
    interocular = float(np.linalg.norm(neutral[36] - neutral[45]))
    if interocular <= 0.0 or not np.isfinite(interocular):
        raise AutoAnimError("INTERNAL_ERROR", "GNM interocular distance is invalid")

    inner_indices = np.asarray((61, 62, 63, 65, 66, 67), dtype=np.int64)
    desired = np.zeros((68, 3), dtype=np.float64)
    for upper, lower in _INNER_LIP_PAIRS:
        separation = neutral[lower] - neutral[upper]
        desired[upper] += 0.5 * separation
        desired[lower] -= 0.5 * separation

    inner_system = compact_basis[:, inner_indices].transpose(1, 2, 0).reshape(-1, 150)
    inner_target = desired[inner_indices].reshape(-1)
    nonmouth_system = compact_basis[:, :48].transpose(1, 2, 0).reshape(-1, 150)
    outer_mouth_system = compact_basis[:, 48:61].transpose(1, 2, 0).reshape(-1, 150)

    vertex_basis = np.asarray(
        rig.adapter.model.expression_basis[200:350],
        dtype=np.float64,
    )
    lip_support = np.maximum.reduce(
        (
            rig.adapter.vertex_group("upper_lip_region"),
            rig.adapter.vertex_group("lower_lip_region"),
            rig.adapter.vertex_group("upper_lip"),
            rig.adapter.vertex_group("lower_lip"),
        )
    )
    exterior = rig.adapter.vertex_group("skin_exterior")
    preserve_indices = np.flatnonzero((exterior > 0.20) & (lip_support < 0.05))
    if len(preserve_indices) < 128:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "GNM has too few non-mouth exterior vertices for contact calibration",
        )
    # The full preservation mask is used for the reported audit metrics. A
    # deterministic, topology-ordered sample keeps the solve small enough to
    # run per character without changing its spatial coverage.
    stride = max(1, int(math.ceil(len(preserve_indices) / 3_000)))
    solve_indices = preserve_indices[::stride]
    dense_preserve_system = (
        vertex_basis[:, solve_indices].transpose(1, 2, 0).reshape(-1, 150)
    )

    def normalized(system: np.ndarray, weight: float) -> np.ndarray:
        return np.float64(weight) * system / math.sqrt(max(len(system), 1))

    system = np.vstack(
        (
            normalized(inner_system, 1.0),
            normalized(dense_preserve_system, 20.0),
            normalized(nonmouth_system, 10.0),
            normalized(outer_mouth_system, 0.5),
            np.float64(1.0e-4) * np.eye(150, dtype=np.float64),
        )
    )
    # Precompute the linear response from arbitrary inner-lip target motion to
    # spatially constrained lower-face coefficients. Runtime can then solve a
    # vowel-dependent closure, instead of assuming the neutral contact ray is
    # sufficient for every coarticulated mouth pose.
    inner_response = (
        np.linalg.pinv(system, rcond=1.0e-7)[:, : len(inner_target)]
        / math.sqrt(len(inner_target))
    )
    solved = inner_response @ inner_target
    if not np.isfinite(solved).all() or float(np.max(np.abs(solved), initial=0.0)) <= 1e-8:
        raise AutoAnimError("INTERNAL_ERROR", "GNM lip-contact solve is degenerate")
    maximum_coefficient = float(np.max(np.abs(solved)))
    if maximum_coefficient > 2.25:
        solved *= np.float64(2.25 / maximum_coefficient)

    direction = np.zeros(expression_dim, dtype=np.float32)
    direction[200:350] = solved.astype(np.float32)
    alpha_bound = min(2.0, 2.95 / max(float(np.max(np.abs(solved))), 1e-8))
    samples = np.linspace(0.0, alpha_bound, 81, dtype=np.float64)
    sampled_landmarks = [
        rig.compact_landmarks(np.float32(alpha) * direction) for alpha in samples
    ]
    pair_gaps = np.asarray(
        [
            [
                np.linalg.norm(landmarks[upper] - landmarks[lower]) / interocular
                for upper, lower in _INNER_LIP_PAIRS
            ]
            for landmarks in sampled_landmarks
        ],
        dtype=np.float64,
    )
    gaps = np.mean(pair_gaps, axis=1)
    minimum_index = int(np.argmin(gaps))
    if minimum_index == 0 or minimum_index == len(samples) - 1:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "GNM contact path has no bounded character-specific seal minimum",
        )
    maximum_alpha = float(samples[minimum_index])
    neutral_gap = float(gaps[0])
    seal_gap = float(gaps[minimum_index])
    if not 0.0 <= seal_gap < neutral_gap:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "GNM contact path does not close the character's neutral lip gap",
        )

    full_displacement = np.einsum(
        "i,ijk->jk",
        np.float64(maximum_alpha) * solved,
        vertex_basis,
        optimize=True,
    )
    nonmouth_displacement = (
        np.linalg.norm(full_displacement[preserve_indices], axis=1) / interocular
    )
    digest = sha256()
    digest.update(np.asarray(direction, dtype="<f4").tobytes())
    digest.update(np.asarray(inner_response, dtype="<f4").tobytes())
    digest.update(
        np.asarray(
            (neutral_gap, seal_gap, maximum_alpha),
            dtype="<f8",
        ).tobytes()
    )
    inner_response = np.asarray(inner_response, dtype=np.float32)
    direction.setflags(write=False)
    inner_response.setflags(write=False)
    neutral_pair_gaps = np.asarray(pair_gaps[0], dtype=np.float32)
    seal_pair_gaps = np.asarray(pair_gaps[minimum_index], dtype=np.float32)
    neutral_pair_gaps.setflags(write=False)
    seal_pair_gaps.setflags(write=False)
    return LipContactCalibration(
        direction=direction,
        inner_response=inner_response,
        neutral_pair_gaps_interocular=neutral_pair_gaps,
        seal_pair_gaps_interocular=seal_pair_gaps,
        neutral_gap_interocular=neutral_gap,
        seal_gap_interocular=seal_gap,
        maximum_alpha=maximum_alpha,
        nonmouth_p95_displacement_interocular=float(
            np.percentile(nonmouth_displacement, 95)
        ),
        nonmouth_max_displacement_interocular=float(
            np.max(nonmouth_displacement, initial=0.0)
        ),
        calibration_hash=digest.hexdigest(),
    )


def _apply_lip_contact_correction(
    rig: ControlRig,
    expression: np.ndarray,
    calibration: LipContactCalibration,
    confidence: float,
) -> tuple[np.ndarray, bool, float]:
    """Apply a bounded soft bilabial correction along a calibrated direction.

    The confidence comes from the learned mouth-close/press and jaw tracks,
    optionally reinforced by the coarse closed-mouth cue.  It is deliberately
    a soft fallback: independent phones and character contact surfaces are
    still required for production approval.
    """

    strength = float(np.clip(confidence, 0.0, 1.0))
    if strength < 0.12:
        return np.asarray(expression, dtype=np.float32).copy(), False, 0.0
    direction = np.asarray(calibration.direction, dtype=np.float32)
    response = np.asarray(calibration.inner_response, dtype=np.float32)
    neutral_pair_gaps = np.asarray(
        calibration.neutral_pair_gaps_interocular,
        dtype=np.float32,
    )
    seal_pair_gaps = np.asarray(
        calibration.seal_pair_gaps_interocular,
        dtype=np.float32,
    )
    if (
        direction.shape != (rig.adapter.expression_dim,)
        or response.shape != (150, 18)
        or neutral_pair_gaps.shape != (3,)
        or seal_pair_gaps.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isfinite(response).all()
        or not np.isfinite(neutral_pair_gaps).all()
        or not np.isfinite(seal_pair_gaps).all()
    ):
        raise AutoAnimError("INTERNAL_ERROR", "Lip-contact calibration arrays are invalid")
    if (
        not np.isfinite(calibration.neutral_gap_interocular)
        or not np.isfinite(calibration.seal_gap_interocular)
        or not np.isfinite(calibration.maximum_alpha)
        or calibration.maximum_alpha <= 0.0
        or not 0.0 <= calibration.seal_gap_interocular < calibration.neutral_gap_interocular
    ):
        raise AutoAnimError("INTERNAL_ERROR", "Lip-contact calibration is invalid")
    original = np.asarray(expression, dtype=np.float32)
    original_landmarks = rig.compact_landmarks(original)
    neutral_landmarks = rig.neutral_landmarks
    interocular = float(np.linalg.norm(neutral_landmarks[36] - neutral_landmarks[45]))
    original_pair_gaps = np.asarray(
        [
            np.linalg.norm(original_landmarks[upper] - original_landmarks[lower])
            / interocular
            for upper, lower in _INNER_LIP_PAIRS
        ],
        dtype=np.float32,
    )
    original_gap = float(np.mean(original_pair_gaps))
    character_pair_targets = neutral_pair_gaps + np.float32(strength) * (
        seal_pair_gaps - neutral_pair_gaps
    )
    # A low-confidence onset must not jump directly from an open vowel to the
    # neutral gap. Ease the geometric target in, reaching the full soft-contact
    # target once the learned evidence is strong enough to be trustworthy.
    onset = _smooth_alpha((strength - 0.10) / 0.45)
    target_pair_gaps = original_pair_gaps + np.float32(onset) * (
        character_pair_targets - original_pair_gaps
    )
    # Contact correction is closure-only. If a coarticulated pose already has
    # a smaller pair gap, it must not be opened to match the calibration ray.
    target_pair_gaps = np.minimum(target_pair_gaps, original_pair_gaps)
    target_gap = float(np.mean(target_pair_gaps))
    if onset <= 1e-6 or np.all(original_pair_gaps <= character_pair_targets):
        return original.copy(), False, float(target_gap)

    original_lip_order = _mouth_lip_order_minimum_interocular(rig, original)
    minimum_lip_order = min(original_lip_order, -5.0e-4)
    vertex_basis = np.asarray(
        rig.adapter.model.expression_basis[200:350],
        dtype=np.float32,
    )
    lip_support = np.maximum.reduce(
        (
            rig.adapter.vertex_group("upper_lip_region"),
            rig.adapter.vertex_group("lower_lip_region"),
            rig.adapter.vertex_group("upper_lip"),
            rig.adapter.vertex_group("lower_lip"),
        )
    )
    exterior = rig.adapter.vertex_group("skin_exterior")
    preserve_indices = np.flatnonzero((exterior > 0.20) & (lip_support < 0.05))

    def nonmouth_displacement_interocular(candidate: np.ndarray) -> float:
        delta_mesh = np.einsum(
            "i,ijk->jk",
            np.asarray(candidate[200:350] - original[200:350], dtype=np.float32),
            vertex_basis,
            optimize=True,
        )
        return float(
            np.max(
                np.linalg.norm(delta_mesh[preserve_indices], axis=1),
                initial=0.0,
            )
            / interocular
        )

    def calibrated_ray_fallback() -> tuple[np.ndarray, bool, float]:
        """Close along the audited character ray when the local solve stalls.

        The local inverse solve is more pose-specific, but on a very open
        coarticulated vowel it can become rank-limited. The calibration ray is
        already bounded to the selected identity's first lip-gap minimum and
        has a stricter measured non-mouth displacement than the dynamic cap.
        """

        static_direction = np.asarray(calibration.direction, dtype=np.float32)
        # Clip coefficients individually. ``rig.compose`` rescales an entire
        # expression region when any one coefficient is saturated, which can
        # move unrelated lower-face modes and invalidate the measured spatial
        # bound for this fallback.
        bound = max(0.0, float(calibration.maximum_alpha))
        if bound <= 1.0e-6:
            return original.copy(), False, float(target_gap)

        def candidate(alpha: float) -> np.ndarray:
            return np.clip(
                original + np.float32(alpha) * static_direction,
                -3.0,
                3.0,
            ).astype(
                np.float32,
                copy=False,
            )

        samples = np.linspace(0.0, bound, 33, dtype=np.float32)
        candidates = tuple(candidate(float(alpha)) for alpha in samples)
        gaps = np.asarray(
            [_mouth_gap_interocular(rig, value) for value in candidates]
        )
        lip_orders = np.asarray(
            [_mouth_lip_order_minimum_interocular(rig, value) for value in candidates]
        )
        nonmouth_displacement = np.asarray(
            [nonmouth_displacement_interocular(value) for value in candidates]
        )
        valid = (lip_orders >= minimum_lip_order - 1.0e-7) & (
            nonmouth_displacement <= 2.0e-3 + 1.0e-7
        )
        reached = np.flatnonzero((gaps <= target_gap) & valid)
        if len(reached):
            upper_index = int(reached[0])
            lower = float(samples[max(upper_index - 1, 0)])
            upper = float(samples[upper_index])
            for _ in range(8):
                middle = 0.5 * (lower + upper)
                value = candidate(middle)
                if (
                    _mouth_gap_interocular(rig, value) <= target_gap
                    and _mouth_lip_order_minimum_interocular(rig, value)
                    >= minimum_lip_order - 1.0e-7
                    and nonmouth_displacement_interocular(value)
                    <= 2.0e-3 + 1.0e-7
                ):
                    upper = middle
                else:
                    lower = middle
            corrected_value = candidate(upper)
        else:
            valid_indices = np.flatnonzero(valid)
            if not len(valid_indices):
                return original.copy(), False, float(target_gap)
            best_index = int(valid_indices[int(np.argmin(gaps[valid_indices]))])
            corrected_value = candidates[best_index]
        improved_gap = _mouth_gap_interocular(rig, corrected_value)
        if improved_gap >= original_gap - 1.0e-5:
            return original.copy(), False, float(target_gap)
        return corrected_value, True, float(target_gap)

    desired = np.zeros((68, 3), dtype=np.float32)
    for pair_index, (upper, lower) in enumerate(_INNER_LIP_PAIRS):
        separation = original_landmarks[lower] - original_landmarks[upper]
        length = float(np.linalg.norm(separation))
        wanted = float(target_pair_gaps[pair_index] * interocular)
        if length <= wanted + 1e-9 or length <= 1e-9:
            continue
        closure = separation * np.float32(1.0 - wanted / length)
        desired[upper] += np.float32(0.5) * closure
        desired[lower] -= np.float32(0.5) * closure
    inner_indices = np.asarray((61, 62, 63, 65, 66, 67), dtype=np.int64)
    solved = response @ desired[inner_indices].reshape(-1)
    if not np.isfinite(solved).all() or float(np.max(np.abs(solved), initial=0.0)) <= 1e-9:
        return calibrated_ray_fallback()
    direction = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    direction[200:350] = solved.astype(np.float32)

    # Find the smallest correction that reaches the soft seal band.  The
    # one-dimensional path is cheap and auditable; it cannot masquerade as a
    # full collision solve.  If the band is unreachable, choose the minimum-
    # gap sample rather than extrapolating into an inverted pose.
    def bounded_candidate(alpha: float) -> np.ndarray:
        candidate, _ = rig.compose(
            original + np.float32(alpha) * direction,
            np.zeros_like(original),
        )
        return candidate

    # Keep every corrected coefficient inside the rig contract and cap the
    # dense non-mouth displacement introduced by this particular dynamic solve
    # to 0.2% of interocular distance (about 0.17 mm on an adult-scale face).
    coefficient_alpha = 1.5
    lower_original = original[200:350]
    for value, delta in zip(lower_original, direction[200:350], strict=True):
        if delta > 1e-9:
            coefficient_alpha = min(coefficient_alpha, float((3.0 - value) / delta))
        elif delta < -1e-9:
            coefficient_alpha = min(coefficient_alpha, float((-3.0 - value) / delta))
    dynamic_displacement = np.einsum(
        "i,ijk->jk",
        direction[200:350],
        vertex_basis,
        optimize=True,
    )
    unit_nonmouth_max = float(
        np.max(
            np.linalg.norm(dynamic_displacement[preserve_indices], axis=1),
            initial=0.0,
        )
        / interocular
    )
    spatial_alpha = 2.0e-3 / max(unit_nonmouth_max, 1e-12)
    search_bound = float(max(0.0, min(1.5, coefficient_alpha, spatial_alpha)))
    if search_bound <= 1e-6:
        return calibrated_ray_fallback()
    samples = np.linspace(0.0, search_bound, 25, dtype=np.float32)
    sample_candidates = tuple(bounded_candidate(float(alpha)) for alpha in samples)
    gaps = np.asarray(
        [_mouth_gap_interocular(rig, candidate) for candidate in sample_candidates]
    )
    lip_order = np.asarray(
        [
            _mouth_lip_order_minimum_interocular(rig, candidate)
            for candidate in sample_candidates
        ]
    )
    valid_lip_order = lip_order >= minimum_lip_order - 1.0e-7
    reached = np.flatnonzero((gaps <= target_gap) & valid_lip_order)
    if len(reached):
        upper_index = int(reached[0])
        lower = float(samples[max(upper_index - 1, 0)])
        upper = float(samples[upper_index])
        for _ in range(8):
            middle = 0.5 * (lower + upper)
            middle_candidate = bounded_candidate(middle)
            if (
                _mouth_gap_interocular(rig, middle_candidate) <= target_gap
                and _mouth_lip_order_minimum_interocular(rig, middle_candidate)
                >= minimum_lip_order - 1.0e-7
            ):
                upper = middle
            else:
                lower = middle
        alpha = upper
    else:
        # Apply only the best spatially/coefficent-bounded improvement and let
        # the post-limiter status report that the target remained unresolved.
        # This avoids both a false success and a complete loss of useful
        # closure motion when GNM's affine space cannot reach the requested
        # character seal from the current coarticulated pose.
        valid_indices = np.flatnonzero(valid_lip_order)
        if not len(valid_indices):
            return calibrated_ray_fallback()
        minimum_index = int(valid_indices[int(np.argmin(gaps[valid_indices]))])
        if minimum_index == 0 or gaps[minimum_index] >= original_gap - 1e-5:
            return calibrated_ray_fallback()
        alpha = float(samples[minimum_index])
    corrected = bounded_candidate(alpha)
    if _mouth_lip_order_minimum_interocular(rig, corrected) < minimum_lip_order - 1.0e-7:
        return calibrated_ray_fallback()
    if _mouth_gap_interocular(rig, corrected) > target_gap + 1.0e-7:
        fallback, fallback_applied, _ = calibrated_ray_fallback()
        if (
            fallback_applied
            and _mouth_gap_interocular(rig, fallback)
            < _mouth_gap_interocular(rig, corrected) - 1.0e-7
        ):
            corrected = fallback
    return (
        corrected,
        bool(_mouth_gap_interocular(rig, corrected) < original_gap - 1e-5),
        float(target_gap),
    )


def apply_lip_contact_correction(
    rig: ControlRig,
    expression: np.ndarray,
    calibration: LipContactCalibration,
    confidence: float,
) -> tuple[np.ndarray, bool, float]:
    """Public, audited entry point for the character lip-contact solve.

    Audio and video derive contact evidence differently, but both must use the
    same bounded GNM-space correction and report whether geometry actually
    moved toward the requested seal.
    """

    return _apply_lip_contact_correction(rig, expression, calibration, confidence)


def _stabilize_absolute_contact_runs(
    rig: ControlRig,
    expression: np.ndarray,
    timestamps: np.ndarray,
    contact_target_gap: np.ndarray,
    *,
    shoulder_seconds: float = _ABSOLUTE_CONTACT_PREPROJECTION_HORIZON_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimally pre-project attained v3 contacts onto the export speed bound.

    The learned solver remains authoritative: no contact run is replaced by a
    common pose. Only an edge that exceeds the exact-time visible-mouth limit
    may change, and a changed contact frame is accepted only if its own seal is
    retained. The smallest required adjustment is then propagated outward
    until the original trajectory is reachable again or the bounded horizon is
    exhausted. Protected upper-face, tongue, pupil, and timestamp channels are
    copied byte-exactly from the learned source.
    """

    output = np.asarray(expression, dtype=np.float32).copy()
    original = output.copy()
    clock = np.asarray(timestamps, dtype=np.float64)
    targets = np.asarray(contact_target_gap, dtype=np.float32)
    if (
        output.ndim != 2
        or output.shape[1] != rig.adapter.expression_dim
        or clock.shape != (len(output),)
        or targets.shape != (len(output),)
        or not np.isfinite(output).all()
        or not np.isfinite(clock).all()
        or not np.isfinite(targets).all()
        or (len(clock) > 1 and np.any(np.diff(clock) <= 0.0))
        or not np.isfinite(shoulder_seconds)
        or shoulder_seconds <= 0.0
    ):
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Absolute contact preprojection requires finite aligned controls and timestamps",
        )
    changed = np.zeros(len(output), dtype=bool)
    if len(output) < 2 or not np.any(targets > 0.0):
        return output, changed
    projection_config = _articulation_projection_config(
        external_face_controls=True
    )
    edge_limits = np.minimum(
        np.float64(projection_config["maximum_step"]),
        np.float64(projection_config["maximum_speed"]) * np.diff(clock),
    )

    def limit_edge(
        previous: np.ndarray,
        target: np.ndarray,
        edge_index: int,
    ) -> tuple[np.ndarray, bool]:
        return limit_articulation_edge(
            previous,
            target,
            mouth_step_metric=lambda left, right: _mouth_step_quality_ratio(
                rig, left, right
            ),
            maximum_ratio=float(edge_limits[edge_index]),
        )

    def contact_pose_is_valid(frame_index: int, candidate: np.ndarray) -> bool:
        return bool(
            _mouth_gap_interocular(rig, candidate)
            <= float(targets[frame_index])
            + float(projection_config["contact_tolerance"])
            + 1.0e-7
            and _mouth_lip_order_minimum_interocular(rig, candidate)
            >= float(projection_config["lip_order_floor"]) - 1.0e-7
        )

    def noncontact_pose_is_valid(candidate: np.ndarray) -> bool:
        return bool(
            _mouth_lip_order_minimum_interocular(rig, candidate)
            >= float(projection_config["lip_order_floor"]) - 1.0e-7
        )

    gaps = np.asarray(
        [_mouth_gap_interocular(rig, frame) for frame in output],
        dtype=np.float64,
    )
    attained = (targets > 0.0) & (gaps <= targets + np.float32(0.001))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(len(attained) + 1):
        active = bool(attained[index]) if index < len(attained) else False
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None

    for start, end in runs:
        # First remove only violating edges inside the learned contact run.
        # A forward and reverse pass avoids privileging attack over release.
        for frame_index in range(start + 1, end + 1):
            trial, limited = limit_edge(
                output[frame_index - 1],
                output[frame_index],
                frame_index - 1,
            )
            if limited and contact_pose_is_valid(frame_index, trial):
                output[frame_index] = trial
        for frame_index in range(end - 1, start - 1, -1):
            trial, limited = limit_edge(
                output[frame_index + 1],
                output[frame_index],
                frame_index,
            )
            if limited and contact_pose_is_valid(frame_index, trial):
                output[frame_index] = trial

        # Extend the minimum reachable adjustment into the neighboring phones.
        # Stop as soon as an untouched source pose is already reachable.
        for frame_index in range(start - 1, -1, -1):
            if targets[frame_index] > 0.0:
                break
            distance = float(clock[start] - clock[frame_index])
            if distance > shoulder_seconds + 1.0e-9:
                break
            trial, limited = limit_edge(
                output[frame_index + 1],
                original[frame_index],
                frame_index,
            )
            if not noncontact_pose_is_valid(trial):
                break
            output[frame_index] = trial
            if not limited:
                break
        for frame_index in range(end + 1, len(output)):
            if targets[frame_index] > 0.0:
                break
            distance = float(clock[frame_index] - clock[end])
            if distance > shoulder_seconds + 1.0e-9:
                break
            trial, limited = limit_edge(
                output[frame_index - 1],
                original[frame_index],
                frame_index - 1,
            )
            if not noncontact_pose_is_valid(trial):
                break
            output[frame_index] = trial
            if not limited:
                break
    changed = np.max(np.abs(output - original), axis=1) > 1.0e-7
    return output, changed


def compose_animation(
    cues: list[MouthCue],
    duration: float,
    fps: int,
    rig: ControlRig,
    emotion_name: str,
    prosody: ProsodyTrack | None = None,
    *,
    head_motion: bool = True,
    lip_contact_calibration: LipContactCalibration | None = None,
) -> AnimationTrack:
    if not 12 <= fps <= 60:
        raise AutoAnimError("INPUT_INVALID", "FPS must be in [12, 60]")
    frame_count = int(math.ceil(duration * fps))
    if frame_count <= 0:
        raise AutoAnimError("INPUT_INVALID", "Animation duration is too short")
    timestamps = np.arange(frame_count, dtype=np.float64) / float(fps)
    if not cues:
        raise AutoAnimError("CUE_INVALID", "At least one normalized mouth cue is required")
    prosody = prosody or _default_prosody(cues, timestamps)
    _validate_prosody(prosody, timestamps)
    controls = np.stack([rig.viseme(cue) for cue in CUE_ORDER])
    viseme_weights = _activation_matrix(cues, timestamps)
    lip_contact_confidence = np.clip(
        viseme_weights[:, _CUE_INDEX["A"]], 0.0, 1.0
    ).astype(np.float32)
    expression = np.zeros((frame_count, rig.adapter.expression_dim), dtype=np.float32)
    for frame in range(frame_count):
        cue_scale = np.ones(len(CUE_ORDER), dtype=np.float32)
        cue_scale[2:7] = np.float32(0.87 + 0.28 * prosody.accent[frame])
        speech_gain = np.float32(0.12 + 0.88 * prosody.speech_activity[frame])
        expression[frame] = speech_gain * np.einsum(
            "i,ij->j", viseme_weights[frame] * cue_scale, controls
        )

    # The source timeline can end on a speech cue. Settle only the final
    # output frame to rest, preserving all preceding cue timing.
    if frame_count > 1:
        expression[-1, 200:382] = 0.0
        viseme_weights[-1] = 0.0
        viseme_weights[-1, 0] = 1.0
        lip_contact_confidence[-1] = 0.0
    emotion = rig.emotion(emotion_name)
    emotion_intensity = _emotion_envelope(duration, fps, timestamps, prosody)
    blink = rig.blink()
    blink_intensity = _blink_envelope(timestamps)
    saturated = False
    lip_contact_target_gap = np.zeros(frame_count, dtype=np.float32)
    contact_correction_applied = np.zeros(frame_count, dtype=bool)
    for frame in range(frame_count):
        expression[frame], clipped = rig.compose(
            expression[frame],
            emotion,
            mouth_activity=float(prosody.speech_activity[frame]),
            emotion_strength=float(emotion_intensity[frame]),
        )
        if blink_intensity[frame] > 0:
            expression[frame], blink_clipped = rig.compose(
                expression[frame] + blink * blink_intensity[frame],
                np.zeros_like(emotion),
            )
            clipped |= blink_clipped
        if lip_contact_calibration is not None:
            (
                expression[frame],
                contact_correction_applied[frame],
                lip_contact_target_gap[frame],
            ) = _apply_lip_contact_correction(
                rig,
                expression[frame],
                lip_contact_calibration,
                float(lip_contact_confidence[frame]),
            )
        saturated |= clipped

    # Rest is a hard export contract. The reverse temporal pass below starts
    # relaxing early enough to reach it without a one-frame snap.
    expression[-1, 200:382] = 0.0
    desired_expression = expression.copy()
    metric_contract, rig_binding = _articulation_evidence_bindings(rig)
    projection_config = _articulation_projection_config(
        external_face_controls=False
    )
    projection = project_articulation_trajectory(
        desired_expression,
        timestamps,
        mouth_step_metric=lambda left, right: _mouth_step_quality_ratio(
            rig, left, right
        ),
        lip_order_metric=lambda frame: _mouth_lip_order_minimum_interocular(
            rig, frame
        ),
        contact_target_gap=lip_contact_target_gap,
        mouth_gap_metric=lambda frame: _mouth_gap_interocular(rig, frame),
        **projection_config,
        metric_contract=metric_contract,
        rig_binding=rig_binding,
    )
    expression = projection.expression
    mouth_speed_limited = projection.limited_frames
    contact_continuity_restored = projection.contact_continuity_restored
    lip_contact_attained = projection.contact_attained
    projection_report = {
        **projection.report,
        **_articulation_projection_compiler_metadata(
            external_face_controls=False,
            frame_count=frame_count,
        ),
    }
    contact_corrected = contact_correction_applied & lip_contact_attained
    rotations = (
        _head_motion(timestamps, prosody, rig.adapter.model.num_joints)
        if head_motion
        else np.zeros((frame_count, rig.adapter.model.num_joints, 3), dtype=np.float32)
    )
    translation = np.zeros((frame_count, 3), dtype=np.float32)
    return AnimationTrack(
        expression=expression,
        rotations=rotations,
        translation=translation,
        timestamps=timestamps,
        fps=fps,
        saturated=saturated,
        viseme_weights=viseme_weights,
        speech_activity=prosody.speech_activity.astype(np.float32),
        energy=prosody.energy.astype(np.float32),
        pitch_semitones=prosody.pitch_semitones.astype(np.float32),
        accent=prosody.accent.astype(np.float32),
        phrase_id=prosody.phrase_id.astype(np.int32),
        emotion_intensity=emotion_intensity,
        mouth_speed_limited=mouth_speed_limited,
        lip_contact_confidence=lip_contact_confidence,
        lip_contact_target_gap=lip_contact_target_gap,
        contact_correction_applied=contact_correction_applied,
        lip_contact_attained=lip_contact_attained,
        contact_continuity_restored=contact_continuity_restored,
        contact_corrected=contact_corrected,
        lip_order_repaired=np.zeros(frame_count, dtype=bool),
        articulation_projection_report=projection_report,
        articulation_projection_desired=projection.desired_expression,
        articulation_projection_output=projection.expression,
    )


def compose_learned_animation(
    source_expression: np.ndarray,
    source_timestamps: np.ndarray,
    cues: list[MouthCue],
    duration: float,
    fps: int,
    rig: ControlRig,
    prosody: ProsodyTrack,
    *,
    acting_strength: float = 0.0,
    emotion_delta: np.ndarray | None = None,
    source_eye_rotations_degrees: np.ndarray | None = None,
    emotion_eye_delta_degrees: np.ndarray | None = None,
    source_lip_contact_confidence: np.ndarray | None = None,
    lip_contact_calibration: LipContactCalibration | None = None,
    source_neutral_policy: str = LEARNED_SOURCE_NEUTRAL_POLICY_QUIET_MEDIAN,
    source_neutral_expression: np.ndarray | None = None,
    head_motion: bool = True,
) -> AnimationTrack:
    """Compile continuous learned controls onto the exact export clock.

    Learned providers emit timestamped source clocks (30 fps for Audio2Face
    v2.3 and 60 fps for v3 diffusion) whose final timestamp is not necessarily
    ``duration - 1/fps``. Clip-relative v2.3 controls retain their quiet-frame
    rest subtraction. Neutral-relative v3 post-solver controls remain absolute
    and only their oral regions are smoothly returned to neutral in verified
    silence. No Rhubarb pose is mixed into the learned mouth; its weights remain
    diagnostic timeline metadata.
    """

    if not 12 <= fps <= 60:
        raise AutoAnimError("INPUT_INVALID", "FPS must be in [12, 60]")
    if not cues:
        raise AutoAnimError("CUE_INVALID", "At least one normalized mouth cue is required")
    source = np.asarray(source_expression, dtype=np.float32)
    source_time = np.asarray(source_timestamps, dtype=np.float64)
    if source_neutral_policy not in _LEARNED_SOURCE_NEUTRAL_POLICIES:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            f"Unknown learned source neutral policy: {source_neutral_policy!r}",
        )
    provider_neutral: np.ndarray | None = None
    if source_neutral_policy == LEARNED_SOURCE_NEUTRAL_POLICY_ABSOLUTE_ORAL_GATE:
        if source_neutral_expression is None:
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Neutral-relative learned controls require a provider neutral expression",
            )
        provider_neutral = np.asarray(source_neutral_expression, dtype=np.float32)
        if (
            provider_neutral.shape != (rig.adapter.expression_dim,)
            or not np.isfinite(provider_neutral).all()
        ):
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Provider neutral expression must be a finite GNM expression vector",
            )
    elif source_neutral_expression is not None:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Provider neutral expression is only valid for neutral-relative learned controls",
        )
    if source.ndim != 2 or source.shape[1] != rig.adapter.expression_dim:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            f"Learned controls must have shape [frames,{rig.adapter.expression_dim}]",
        )
    if source_time.shape != (len(source),) or len(source) < 2:
        raise AutoAnimError("INTERNAL_ERROR", "Learned controls need at least two timestamps")
    if (
        not np.isfinite(source).all()
        or not np.isfinite(source_time).all()
        or source_time[0] < 0
        or np.any(np.diff(source_time) <= 0)
    ):
        raise AutoAnimError("INTERNAL_ERROR", "Learned controls or timestamps are invalid")
    affect_source: np.ndarray | None = None
    if emotion_delta is not None:
        affect_source = np.asarray(emotion_delta, dtype=np.float32)
        if affect_source.shape != source.shape or not np.isfinite(affect_source).all():
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Learned emotion delta must match the finite source expression track",
            )
    eye_source: np.ndarray | None = None
    if source_eye_rotations_degrees is not None:
        eye_source = np.asarray(source_eye_rotations_degrees, dtype=np.float32)
        if eye_source.shape != (len(source), 2, 2) or not np.isfinite(eye_source).all():
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Learned eye rotations must be finite [source_frames,2,2] degrees",
            )
    affect_eye_source: np.ndarray | None = None
    if emotion_eye_delta_degrees is not None:
        affect_eye_source = np.asarray(emotion_eye_delta_degrees, dtype=np.float32)
        if (
            affect_eye_source.shape != (len(source), 2, 2)
            or not np.isfinite(affect_eye_source).all()
        ):
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Learned emotion eye delta must be finite [source_frames,2,2] degrees",
            )
    contact_source: np.ndarray | None = None
    contact_calibration: LipContactCalibration | None = None
    if source_lip_contact_confidence is not None:
        contact_source = np.asarray(source_lip_contact_confidence, dtype=np.float32)
        if (
            contact_source.shape != (len(source),)
            or not np.isfinite(contact_source).all()
            or np.any(contact_source < 0.0)
            or np.any(contact_source > 1.0)
        ):
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Learned lip-contact confidence must be finite [source_frames] in [0,1]",
            )
        if lip_contact_calibration is None:
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Learned lip-contact confidence requires a calibrated GNM contact solve",
            )
        contact_direction = np.asarray(lip_contact_calibration.direction, dtype=np.float32)
        if (
            contact_direction.shape != (rig.adapter.expression_dim,)
            or not np.isfinite(contact_direction).all()
        ):
            raise AutoAnimError(
                "INTERNAL_ERROR",
                "Learned lip-contact direction must be a finite GNM expression vector",
            )
        contact_calibration = lip_contact_calibration
    elif lip_contact_calibration is not None:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "A lip-contact calibration cannot be supplied without source confidence",
        )

    frame_count = int(math.ceil(duration * fps))
    if frame_count <= 0:
        raise AutoAnimError("INPUT_INVALID", "Animation duration is too short")
    timestamps = np.arange(frame_count, dtype=np.float64) / float(fps)
    _validate_prosody(prosody, timestamps)
    viseme_weights = _activation_matrix(cues, timestamps)

    source_activity = np.interp(
        source_time,
        timestamps.astype(np.float64),
        prosody.speech_activity.astype(np.float64),
        left=float(prosody.speech_activity[0]),
        right=float(prosody.speech_activity[-1]),
    )
    quiet = source_activity <= _ORAL_GATE_FULL_ACTIVITY
    if source_neutral_policy == LEARNED_SOURCE_NEUTRAL_POLICY_QUIET_MEDIAN:
        if np.count_nonzero(quiet) >= 3:
            rest = np.median(source[quiet], axis=0)
        else:
            rest = source[0]
        prepared_source = source - rest.astype(np.float32)
        source_oral_gate: np.ndarray | None = None
        source_neutral_handling: dict[str, object] = {
            "schema_version": "autoanim.learned-source-neutral-handling/1.0",
            "policy": LEARNED_SOURCE_NEUTRAL_POLICY_QUIET_MEDIAN,
            "source_control_semantics": "clip_relative_delta",
            "rest_subtracted": True,
            "quiet_source_frames": int(np.count_nonzero(quiet)),
            "production_validated": False,
        }
    else:
        assert provider_neutral is not None
        prepared_source = source.copy()
        source_oral_gate = _learned_oral_source_gate(
            prosody.speech_activity,
            timestamps,
        )
        zero_frames = int(np.count_nonzero(source_oral_gate <= 1.0e-7))
        full_frames = int(np.count_nonzero(source_oral_gate >= 1.0 - 1.0e-7))
        source_neutral_handling = {
            "schema_version": "autoanim.learned-source-neutral-handling/1.0",
            "policy": LEARNED_SOURCE_NEUTRAL_POLICY_ABSOLUTE_ORAL_GATE,
            "source_control_semantics": "neutral_relative_absolute",
            "rest_subtracted": False,
            "provider_neutral_method": "pinned_post_solver_zero_input_offsets",
            "provider_neutral_expression_sha256": articulation_array_sha256(
                provider_neutral
            ),
            "oral_gate_expression_range": [200, 382],
            "oral_gate_input": "audio_conditioned_speech_activity",
            "oral_gate_minimum_silence_seconds": (
                _ORAL_GATE_MINIMUM_SILENCE_SECONDS
            ),
            "oral_gate_transition_seconds": _ORAL_GATE_TRANSITION_SECONDS,
            "oral_gate_maximum_observed_step": float(
                np.max(np.abs(np.diff(source_oral_gate)), initial=0.0)
            ),
            "oral_gate_maximum_observed_speed_per_second": float(
                np.max(
                    np.abs(np.diff(source_oral_gate)) / np.diff(timestamps),
                    initial=0.0,
                )
                if len(source_oral_gate) > 1
                else 0.0
            ),
            "oral_gate_zero_activity": _ORAL_GATE_ZERO_ACTIVITY,
            "oral_gate_full_activity": _ORAL_GATE_FULL_ACTIVITY,
            "oral_gate_zero_frames": zero_frames,
            "oral_gate_transition_frames": int(
                len(source_oral_gate) - zero_frames - full_frames
            ),
            "oral_gate_full_frames": full_frames,
            "quiet_source_frames": int(np.count_nonzero(quiet)),
            "production_validated": False,
        }

    expression = np.empty((frame_count, source.shape[1]), dtype=np.float32)
    for channel in range(source.shape[1]):
        expression[:, channel] = np.interp(
            timestamps.astype(np.float64),
            source_time,
            prepared_source[:, channel].astype(np.float64),
        ).astype(np.float32)
    if source_oral_gate is not None:
        assert provider_neutral is not None
        oral_neutral = provider_neutral[200:382]
        expression[:, 200:382] = oral_neutral[None, :] + source_oral_gate[
            :, None
        ] * (expression[:, 200:382] - oral_neutral[None, :])

    affect = np.zeros_like(expression)
    if affect_source is not None:
        for channel in range(source.shape[1]):
            affect[:, channel] = np.interp(
                timestamps.astype(np.float64),
                source_time,
                affect_source[:, channel].astype(np.float64),
            ).astype(np.float32)

    lip_contact_confidence = np.zeros(frame_count, dtype=np.float32)
    if contact_source is not None:
        interpolated_contact = np.interp(
            timestamps.astype(np.float64),
            source_time,
            contact_source.astype(np.float64),
        ).astype(np.float32)
        # Require agreement between continuous learned closure evidence and
        # Rhubarb's coarse P/B/M-like closed-mouth cue. The geometric mean
        # preserves a short contact when both are credible while rejecting
        # the false seals that mouthClose produces on non-bilabial phones.
        cue_contact = viseme_weights[:, _CUE_INDEX["A"]]
        lip_contact_confidence = np.sqrt(
            np.clip(interpolated_contact * cue_contact, 0.0, 1.0)
        ).astype(np.float32)

    saturated = False
    zero = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    blink_control = rig.blink()
    blink_envelope = _blink_envelope(timestamps)
    lip_contact_target_gap = np.zeros(frame_count, dtype=np.float32)
    contact_correction_applied = np.zeros(frame_count, dtype=bool)
    lip_order_repaired = np.zeros(frame_count, dtype=bool)
    emotion_envelope = (
        np.float32(np.clip(acting_strength, 0.0, 1.0))
        * _emotion_envelope(duration, fps, timestamps, prosody)
    )
    for frame in range(frame_count):
        expression[frame], clipped = rig.compose(
            expression[frame],
            affect[frame] if affect_source is not None else zero,
            mouth_activity=float(prosody.speech_activity[frame]),
            emotion_strength=float(emotion_envelope[frame]),
        )
        expression[frame], blink_clipped = rig.compose(
            expression[frame],
            blink_control,
            mouth_activity=float(prosody.speech_activity[frame]),
            emotion_strength=float(0.82 * blink_envelope[frame]),
        )
        if contact_calibration is not None:
            (
                expression[frame],
                contact_correction_applied[frame],
                lip_contact_target_gap[frame],
            ) = _apply_lip_contact_correction(
                rig,
                expression[frame],
                contact_calibration,
                float(lip_contact_confidence[frame]),
            )
        expression[frame], lip_order_repaired[frame] = _repair_lip_order_inversion(
            rig,
            expression[frame],
        )
        saturated |= clipped or blink_clipped

    contact_run_stabilized = np.zeros(frame_count, dtype=bool)
    if provider_neutral is not None and contact_calibration is not None:
        expression, contact_run_stabilized = _stabilize_absolute_contact_runs(
            rig,
            expression,
            timestamps,
            lip_contact_target_gap,
        )
        source_neutral_handling["contact_preprojection_method"] = (
            _ABSOLUTE_CONTACT_PREPROJECTION_METHOD
        )
        source_neutral_handling["contact_preprojection_horizon_seconds"] = (
            _ABSOLUTE_CONTACT_PREPROJECTION_HORIZON_SECONDS
        )
        source_neutral_handling["contact_preprojection_changed_frames"] = int(
            np.count_nonzero(contact_run_stabilized)
        )
        for frame in np.flatnonzero(contact_run_stabilized):
            expression[frame], repaired = _repair_lip_order_inversion(
                rig,
                expression[frame],
            )
            lip_order_repaired[frame] |= repaired

    # Exported clips have a deterministic rest boundary. The bidirectional
    # limiter distributes the release instead of introducing a terminal snap.
    if provider_neutral is None:
        boundary_rest_frames = [0, frame_count - 1]
    else:
        assert source_oral_gate is not None
        boundary_rest_frames = [
            index
            for index in (0, frame_count - 1)
            if source_oral_gate[index] <= 1.0e-7
        ]
    boundary_oral_pose: float | np.ndarray = (
        provider_neutral[200:382] if provider_neutral is not None else 0.0
    )
    for boundary_frame in boundary_rest_frames:
        expression[boundary_frame, 200:382] = boundary_oral_pose
    desired_expression = expression.copy()
    # The learned sequence model owns dedicated tongue and upper-face motion.
    # The emergency visible-mouth guard therefore projects only modes 200:350
    # and uses each exact timestamp delta for its physical-speed bound.
    metric_contract, rig_binding = _articulation_evidence_bindings(rig)
    projection_config = _articulation_projection_config(
        external_face_controls=True
    )
    projection = project_articulation_trajectory(
        desired_expression,
        timestamps,
        mouth_step_metric=lambda left, right: _mouth_step_quality_ratio(
            rig, left, right
        ),
        lip_order_metric=lambda frame: _mouth_lip_order_minimum_interocular(
            rig, frame
        ),
        contact_target_gap=lip_contact_target_gap,
        mouth_gap_metric=lambda frame: _mouth_gap_interocular(rig, frame),
        **projection_config,
        metric_contract=metric_contract,
        rig_binding=rig_binding,
    )
    expression = projection.expression
    mouth_speed_limited = projection.limited_frames
    contact_continuity_restored = projection.contact_continuity_restored
    lip_contact_attained = projection.contact_attained
    projection_report = {
        **projection.report,
        **_articulation_projection_compiler_metadata(
            external_face_controls=True,
            frame_count=frame_count,
            boundary_rest_frames=(
                boundary_rest_frames if provider_neutral is not None else None
            ),
        ),
    }
    contact_corrected = contact_correction_applied & lip_contact_attained

    final_lip_order = np.asarray(
        [_mouth_lip_order_minimum_interocular(rig, frame) for frame in expression],
        dtype=np.float32,
    )
    if np.any(final_lip_order < np.float32(-5.0e-4 - 1.0e-7)):
        raise AutoAnimError(
            "ORAL_LIP_ORDER_UNREPAIRABLE",
            "Continuity processing reintroduced structurally inverted inner lips",
        )

    rotations = (
        _head_motion(timestamps, prosody, rig.adapter.model.num_joints)
        if head_motion
        else np.zeros((frame_count, rig.adapter.model.num_joints, 3), dtype=np.float32)
    )
    if eye_source is not None and rig.adapter.model.num_joints >= 4:
        eye_activity = np.interp(
            source_time,
            timestamps.astype(np.float64),
            prosody.speech_activity.astype(np.float64),
            left=float(prosody.speech_activity[0]),
            right=float(prosody.speech_activity[-1]),
        )
        quiet_eye = eye_activity <= 0.08
        if np.count_nonzero(quiet_eye) >= 3:
            eye_rest = np.median(eye_source[quiet_eye], axis=0)
        else:
            eye_rest = eye_source[0]
        centered_eyes = eye_source - eye_rest.astype(np.float32)
        interpolated_eyes = np.empty((frame_count, 2, 2), dtype=np.float32)
        for eye in range(2):
            for axis in range(2):
                interpolated_eyes[:, eye, axis] = np.interp(
                    timestamps.astype(np.float64),
                    source_time,
                    centered_eyes[:, eye, axis].astype(np.float64),
                ).astype(np.float32)
        # Claire stores right then left; GNM joints are left then right.
        rotations[:, 3, :2] += np.deg2rad(interpolated_eyes[:, 0]).astype(np.float32)
        rotations[:, 2, :2] += np.deg2rad(interpolated_eyes[:, 1]).astype(np.float32)
        rotations[0, 2:4] = 0.0
        rotations[-1, 2:4] = 0.0
    if affect_eye_source is not None and rig.adapter.model.num_joints >= 4:
        interpolated_affect_eyes = np.empty((frame_count, 2, 2), dtype=np.float32)
        for eye in range(2):
            for axis in range(2):
                interpolated_affect_eyes[:, eye, axis] = np.interp(
                    timestamps.astype(np.float64),
                    source_time,
                    affect_eye_source[:, eye, axis].astype(np.float64),
                ).astype(np.float32)
        scaled_affect = interpolated_affect_eyes * emotion_envelope[:, None, None]
        rotations[:, 3, :2] += np.deg2rad(scaled_affect[:, 0]).astype(np.float32)
        rotations[:, 2, :2] += np.deg2rad(scaled_affect[:, 1]).astype(np.float32)
        rotations[0, 2:4] = 0.0
        rotations[-1, 2:4] = 0.0
    return AnimationTrack(
        expression=expression,
        rotations=rotations,
        translation=np.zeros((frame_count, 3), dtype=np.float32),
        timestamps=timestamps,
        fps=fps,
        saturated=saturated,
        viseme_weights=viseme_weights,
        speech_activity=prosody.speech_activity.astype(np.float32),
        energy=prosody.energy.astype(np.float32),
        pitch_semitones=prosody.pitch_semitones.astype(np.float32),
        accent=prosody.accent.astype(np.float32),
        phrase_id=prosody.phrase_id.astype(np.int32),
        emotion_intensity=emotion_envelope.astype(np.float32),
        mouth_speed_limited=mouth_speed_limited,
        lip_contact_confidence=lip_contact_confidence,
        lip_contact_target_gap=lip_contact_target_gap,
        contact_correction_applied=contact_correction_applied,
        lip_contact_attained=lip_contact_attained,
        contact_continuity_restored=contact_continuity_restored,
        contact_corrected=contact_corrected,
        lip_order_repaired=lip_order_repaired,
        articulation_projection_report=projection_report,
        articulation_projection_desired=projection.desired_expression,
        articulation_projection_output=projection.expression,
        source_oral_gate=source_oral_gate,
        source_neutral_handling=source_neutral_handling,
        contact_run_stabilized=contact_run_stabilized,
    )


def render_silent_video(
    track: AnimationTrack,
    adapter: GNMAdapter,
    output_path: str | Path,
    *,
    identity: np.ndarray | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = MeshRenderer(adapter, identity=identity)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "640x640",
        "-framerate", str(track.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-threads", "1", "-metadata", "creation_time=",
        str(output_path),
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise AutoAnimError("DEPENDENCY_MISSING", "ffmpeg is required to render previews") from exc
    assert process.stdin is not None
    try:
        for expression, rotations, translation in zip(
            track.expression, track.rotations, track.translation, strict=True
        ):
            vertices = adapter.mesh(
                identity=identity,
                expression=expression,
                rotations=rotations,
                translation=translation,
            )
            landmarks = adapter.landmarks(
                identity=identity,
                expression=expression,
                rotations=rotations,
                translation=translation,
            )
            process.stdin.write(renderer.render(vertices, landmarks).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        raise
    if return_code:
        raise AutoAnimError("INTERNAL_ERROR", f"ffmpeg video render failed: {stderr.decode(errors='replace')}")
    return output_path


def mux_audio(silent_path: str | Path, wav_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    target_duration = float(probe_av(silent_path)["video_duration"])
    if not np.isfinite(target_duration) or target_duration <= 0.0:
        raise AutoAnimError("INTERNAL_ERROR", "Silent preview has no valid video duration")
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(silent_path), "-i", str(wav_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-af", "apad",
        "-t", f"{target_duration:.9f}",
        "-movflags", "+faststart", "-metadata", "creation_time=", str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=max(60.0, target_duration * 2.0 + 30.0),
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AutoAnimError("INTERNAL_ERROR", "ffmpeg could not mux the preview") from exc
    return output_path


def probe_av(path: str | Path) -> dict[str, float | int | bool]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = float(data["format"]["duration"])
    video_duration = float(video.get("duration", duration)) if video else 0.0
    audio_duration = float(audio.get("duration", duration)) if audio else 0.0
    video_frames = int(video.get("nb_frames", 0)) if video else 0
    video_fps = float(Fraction(video.get("r_frame_rate", "0/1"))) if video else 0.0
    return {
        "duration": duration,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "video_frames": video_frames,
        "video_fps": video_fps,
        "has_audio": audio is not None,
        "has_video": video is not None,
    }
