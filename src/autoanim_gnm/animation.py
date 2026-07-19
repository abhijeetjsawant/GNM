"""Timed GNM control composition and mesh-backed video rendering."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np

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


def _smooth_array(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * value)


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


def _limit_mouth_step_quality_space(
    rig: ControlRig,
    previous: np.ndarray,
    target: np.ndarray,
    *,
    maximum_ratio: float,
) -> tuple[np.ndarray, bool]:
    """Cap motion in the exact face-local metric used by quality scoring.

    ``ControlRig.limit_mouth_step`` measures raw landmark displacement against
    a fixed neutral interocular distance. The production gate removes the
    frame's in-plane pose and normalizes each frame independently, which made
    the old 0.060 compiler cap report as 0.0608 after evaluation. This bounded
    search makes the compiler and its gate share one geometric contract.
    """

    if not np.isfinite(maximum_ratio) or maximum_ratio <= 0.0:
        raise AutoAnimError("INTERNAL_ERROR", "Mouth-step limit must be positive")
    before = np.asarray(previous, dtype=np.float32)
    desired = np.asarray(target, dtype=np.float32)
    if before.shape != desired.shape or before.shape != (rig.adapter.expression_dim,):
        raise AutoAnimError("INTERNAL_ERROR", "Mouth-step controls have invalid shapes")
    def step(candidate: np.ndarray) -> float:
        return _mouth_step_quality_ratio(rig, before, candidate)

    if step(desired) <= maximum_ratio:
        return desired.copy(), False

    def lower_face_candidate(alpha: float) -> np.ndarray:
        output = desired.copy()
        output[200:382] = before[200:382] + np.float32(alpha) * (
            desired[200:382] - before[200:382]
        )
        return output

    candidate_factory = lower_face_candidate
    # Normally upper-face modes are spatially local and the lower-face path
    # begins below the limit. If upper-face PCA leakage alone exceeds it, fall
    # back to interpolating the whole expression so the exported contract is
    # still true instead of silently emitting an unfixable frame.
    if step(lower_face_candidate(0.0)) > maximum_ratio:
        def full_expression_candidate(alpha: float) -> np.ndarray:
            return before + np.float32(alpha) * (desired - before)

        candidate_factory = full_expression_candidate

    lower = 0.0
    upper = 1.0
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        if step(candidate_factory(middle)) <= maximum_ratio:
            lower = middle
        else:
            upper = middle
    return candidate_factory(lower).astype(np.float32), True


def _restore_contact_anchors_quality_space(
    rig: ControlRig,
    desired: np.ndarray,
    projected: np.ndarray,
    *,
    hard_contact_anchors: np.ndarray,
    restore_needed: np.ndarray,
    maximum_ratio: float,
    horizon_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair lost contact locally without breaking the continuity contract.

    The ordinary forward/reverse projection is contact-oblivious: when a seal
    is one step beyond the velocity bound it changes the seal frame itself.
    This repair holds a pre-limit, geometrically attained contact as an anchor
    and moves its approach/release frames instead. The search is deliberately
    bounded to ``horizon_frames`` so an isolated, incompatible target cannot
    smear anticipation across the whole clip.
    """

    target = np.asarray(desired, dtype=np.float32)
    output = np.asarray(projected, dtype=np.float32).copy()
    anchors = np.asarray(hard_contact_anchors, dtype=bool)
    needed = np.asarray(restore_needed, dtype=bool)
    if (
        target.ndim != 2
        or target.shape != output.shape
        or target.shape[1] != rig.adapter.expression_dim
        or anchors.shape != (len(target),)
        or needed.shape != (len(target),)
        or not np.isfinite(target).all()
        or not np.isfinite(output).all()
    ):
        raise AutoAnimError("INTERNAL_ERROR", "Contact-anchor trajectory arrays are invalid")
    if not np.isfinite(maximum_ratio) or maximum_ratio <= 0.0:
        raise AutoAnimError("INTERNAL_ERROR", "Contact-anchor step limit must be positive")
    if horizon_frames < 1:
        raise AutoAnimError("INTERNAL_ERROR", "Contact-anchor horizon must be positive")

    frame_count = len(output)
    tolerance = maximum_ratio + 1.0e-6
    for anchor in np.flatnonzero(needed):
        accepted: np.ndarray | None = None
        for radius in range(1, horizon_frames + 1):
            trial = output.copy()
            trial[anchor] = target[anchor]
            left = max(0, int(anchor) - radius)
            right = min(frame_count - 1, int(anchor) + radius)
            for frame in range(int(anchor) - 1, left - 1, -1):
                if anchors[frame]:
                    trial[frame] = target[frame]
                else:
                    trial[frame], _ = _limit_mouth_step_quality_space(
                        rig,
                        trial[frame + 1],
                        output[frame],
                        maximum_ratio=maximum_ratio,
                    )
            for frame in range(int(anchor) + 1, right + 1):
                if anchors[frame]:
                    trial[frame] = target[frame]
                else:
                    trial[frame], _ = _limit_mouth_step_quality_space(
                        rig,
                        trial[frame - 1],
                        output[frame],
                        maximum_ratio=maximum_ratio,
                    )

            check_left = max(0, left - 1)
            check_right = min(frame_count - 2, right)
            if all(
                _mouth_step_quality_ratio(rig, trial[frame], trial[frame + 1])
                <= tolerance
                for frame in range(check_left, check_right + 1)
            ):
                accepted = trial
                break
        if accepted is not None:
            output = accepted

    restored = needed & np.asarray(
        [
            np.max(np.abs(output[frame] - target[frame]), initial=0.0) <= 1.0e-7
            for frame in range(frame_count)
        ],
        dtype=bool,
    )
    return output, restored


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
        return original.copy(), False, float(target_gap)
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
        return original.copy(), False, float(target_gap)
    samples = np.linspace(0.0, search_bound, 25, dtype=np.float32)
    sample_candidates = tuple(bounded_candidate(float(alpha)) for alpha in samples)
    gaps = np.asarray(
        [_mouth_gap_interocular(rig, candidate) for candidate in sample_candidates]
    )
    original_lip_order = _mouth_lip_order_minimum_interocular(rig, original)
    minimum_lip_order = min(original_lip_order, -5.0e-4)
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
            return original.copy(), False, float(target_gap)
        minimum_index = int(valid_indices[int(np.argmin(gaps[valid_indices]))])
        if minimum_index == 0 or gaps[minimum_index] >= original_gap - 1e-5:
            return original.copy(), False, float(target_gap)
        alpha = float(samples[minimum_index])
    corrected = bounded_candidate(alpha)
    if _mouth_lip_order_minimum_interocular(rig, corrected) < minimum_lip_order - 1.0e-7:
        return original.copy(), False, float(target_gap)
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
    contact_candidates = lip_contact_target_gap > 0.0
    prelimit_contact_attained = np.zeros(frame_count, dtype=bool)
    if np.any(contact_candidates):
        prelimit_gaps = np.asarray(
            [_mouth_gap_interocular(rig, frame) for frame in desired_expression],
            dtype=np.float32,
        )
        prelimit_contact_attained[contact_candidates] = (
            prelimit_gaps[contact_candidates]
            <= lip_contact_target_gap[contact_candidates] + np.float32(1.0e-3)
        )
    # Leave margin for the difference between the face-local projection metric
    # and the raw-landmark export gate while contact anchors are redistributed.
    # The absolute step bound remains active below 30 fps; the speed bound
    # prevents higher delivery rates from silently permitting faster motion.
    speed_limit = min(0.0365, 1.095 / float(fps))
    mouth_speed_limited = np.zeros(frame_count, dtype=bool)
    for frame in range(1, frame_count):
        expression[frame], mouth_speed_limited[frame] = _limit_mouth_step_quality_space(
            rig,
            expression[frame - 1], expression[frame], maximum_ratio=speed_limit
        )
    for frame in range(frame_count - 2, -1, -1):
        expression[frame], reverse_limited = _limit_mouth_step_quality_space(
            rig,
            expression[frame + 1], expression[frame], maximum_ratio=speed_limit
        )
        mouth_speed_limited[frame] |= reverse_limited
    baseline_contact_attained = np.zeros(frame_count, dtype=bool)
    if np.any(contact_candidates):
        baseline_gaps = np.asarray(
            [_mouth_gap_interocular(rig, frame) for frame in expression],
            dtype=np.float32,
        )
        baseline_contact_attained[contact_candidates] = (
            baseline_gaps[contact_candidates]
            <= lip_contact_target_gap[contact_candidates] + np.float32(1.0e-3)
        )
    expression, contact_continuity_restored = _restore_contact_anchors_quality_space(
        rig,
        desired_expression,
        expression,
        hard_contact_anchors=prelimit_contact_attained,
        restore_needed=prelimit_contact_attained & ~baseline_contact_attained,
        maximum_ratio=speed_limit,
        horizon_frames=max(1, int(math.ceil(8.0 * float(fps) / 30.0))),
    )
    mouth_speed_limited = (
        np.max(np.abs(expression - desired_expression), axis=1) > np.float32(1.0e-7)
    )
    lip_contact_attained = np.zeros(frame_count, dtype=bool)
    if np.any(contact_candidates):
        final_gaps = np.asarray(
            [_mouth_gap_interocular(rig, frame) for frame in expression],
            dtype=np.float32,
        )
        lip_contact_attained[contact_candidates] = (
            final_gaps[contact_candidates]
            <= lip_contact_target_gap[contact_candidates] + np.float32(1.0e-3)
        )
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
    head_motion: bool = True,
) -> AnimationTrack:
    """Compile continuous learned controls onto the exact export clock.

    Learned providers emit timestamped source clocks (30 fps for Audio2Face
    v2.3 and 60 fps for v3 diffusion) whose final timestamp is not necessarily
    ``duration - 1/fps``. We interpolate by the supplied times,
    remove the actor-specific rest bias from acoustically quiet frames, and
    apply only a perceptual emergency step limit. No Rhubarb pose is mixed
    into the learned mouth; its weights remain diagnostic timeline metadata.
    """

    if not 12 <= fps <= 60:
        raise AutoAnimError("INPUT_INVALID", "FPS must be in [12, 60]")
    if not cues:
        raise AutoAnimError("CUE_INVALID", "At least one normalized mouth cue is required")
    source = np.asarray(source_expression, dtype=np.float32)
    source_time = np.asarray(source_timestamps, dtype=np.float64)
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

    source_activity = np.interp(
        source_time,
        timestamps.astype(np.float64),
        prosody.speech_activity.astype(np.float64),
        left=float(prosody.speech_activity[0]),
        right=float(prosody.speech_activity[-1]),
    )
    quiet = source_activity <= 0.08
    if np.count_nonzero(quiet) >= 3:
        rest = np.median(source[quiet], axis=0)
    else:
        rest = source[0]
    centered = source - rest.astype(np.float32)

    expression = np.empty((frame_count, source.shape[1]), dtype=np.float32)
    for channel in range(source.shape[1]):
        expression[:, channel] = np.interp(
            timestamps.astype(np.float64),
            source_time,
            centered[:, channel].astype(np.float64),
        ).astype(np.float32)

    affect = np.zeros_like(expression)
    if affect_source is not None:
        for channel in range(source.shape[1]):
            affect[:, channel] = np.interp(
                timestamps.astype(np.float64),
                source_time,
                affect_source[:, channel].astype(np.float64),
            ).astype(np.float32)

    viseme_weights = _activation_matrix(cues, timestamps)
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

    # Exported clips have a deterministic rest boundary. The bidirectional
    # limiter distributes the release instead of introducing a terminal snap.
    expression[0, 200:382] = 0.0
    expression[-1, 200:382] = 0.0
    desired_expression = expression.copy()
    contact_candidates = lip_contact_target_gap > 0.0
    prelimit_contact_attained = np.zeros(frame_count, dtype=bool)
    if np.any(contact_candidates):
        prelimit_gaps = np.asarray(
            [_mouth_gap_interocular(rig, frame) for frame in desired_expression],
            dtype=np.float32,
        )
        prelimit_contact_attained[contact_candidates] = (
            prelimit_gaps[contact_candidates]
            <= lip_contact_target_gap[contact_candidates] + np.float32(1.0e-3)
        )
    # This is an emergency guard, not the primary temporal model. Enforce it
    # in the same face-local geometry used by the production gate. The
    # 1.17-interocular-units/s cap is the former 0.039 limit at 30 fps, expressed
    # in time so a 60 fps delivery cannot silently double permitted speed. The
    # absolute 0.039 safety bound still applies below 30 fps. Together they
    # leave a small margin for viewer morph-target reconstruction error.
    speed_limit = min(0.039, 1.17 / float(fps))
    mouth_speed_limited = np.zeros(frame_count, dtype=bool)
    for frame in range(1, frame_count):
        expression[frame], limited = _limit_mouth_step_quality_space(
            rig,
            expression[frame - 1],
            expression[frame],
            maximum_ratio=speed_limit,
        )
        mouth_speed_limited[frame] |= limited
    for frame in range(frame_count - 2, -1, -1):
        expression[frame], limited = _limit_mouth_step_quality_space(
            rig,
            expression[frame + 1],
            expression[frame],
            maximum_ratio=speed_limit,
        )
        mouth_speed_limited[frame] |= limited

    baseline_contact_attained = np.zeros(frame_count, dtype=bool)
    if np.any(contact_candidates):
        baseline_gaps = np.asarray(
            [_mouth_gap_interocular(rig, frame) for frame in expression],
            dtype=np.float32,
        )
        baseline_contact_attained[contact_candidates] = (
            baseline_gaps[contact_candidates]
            <= lip_contact_target_gap[contact_candidates] + np.float32(1.0e-3)
        )
    restore_needed = prelimit_contact_attained & ~baseline_contact_attained
    expression, contact_continuity_restored = _restore_contact_anchors_quality_space(
        rig,
        desired_expression,
        expression,
        hard_contact_anchors=prelimit_contact_attained,
        restore_needed=restore_needed,
        maximum_ratio=speed_limit,
        horizon_frames=max(1, int(math.ceil(4.0 * float(fps) / 30.0))),
    )
    # A successful repair deliberately returns the contact frame to its desired
    # pose and moves the approach/release instead. Report the frames that still
    # differ from the learned/contact-composed trajectory, not stale mutations
    # made by the first projection pass.
    mouth_speed_limited = (
        np.max(np.abs(expression - desired_expression), axis=1) > np.float32(1.0e-7)
    )

    # The continuity guard may pull a corrected frame away from its contact
    # target. Final status is therefore measured from the exported expression,
    # after both limiter passes, rather than copied from the pre-limiter solve.
    lip_contact_attained = np.zeros(frame_count, dtype=bool)
    if np.any(contact_candidates):
        final_gaps = np.asarray(
            [_mouth_gap_interocular(rig, frame) for frame in expression],
            dtype=np.float32,
        )
        lip_contact_attained[contact_candidates] = (
            final_gaps[contact_candidates]
            <= lip_contact_target_gap[contact_candidates] + np.float32(1.0e-3)
        )
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
