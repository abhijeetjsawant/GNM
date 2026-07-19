"""End-to-end Phase 2 audio service."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .a2f import (
    A2FRunnerError,
    A2FValidationError,
    ClaireSkinSolver,
    ClaireTongueSolver,
    recover_a2f_auxiliary_track,
    run_a2f_runner,
)
from .animated_gltf import AnimationCompressionError, export_animated_gnm_glb
from .animation import (
    calibrate_lip_contact,
    compose_animation,
    compose_learned_animation,
    mux_audio,
    probe_av,
    render_silent_video,
)
from .audio import analyze_emotion, extract_prosody, normalize_audio, normalize_cues, run_rhubarb
from .calibrated_retarget import CalibratedRetargeter
from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .gltf_export import export_gnm_glb
from .lipsync_quality import evaluate_lipsync_quality
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json, write_npz


AUDIO_CAVEAT = "Rhubarb provides coarse viseme timing, not validated phoneme-accurate alignment."
EMOTION_CAVEAT = "Emotion is an unvalidated acoustic/lexical heuristic."
FALLBACK_CAVEAT = "Procedural fallback animation is not a trained facial-performance model."
LEARNED_RETARGET_CAVEAT = (
    "Audio2Face motion uses a geometry-calibrated dense ARKit-to-GNM retarget. "
    "It preserves independent controls, but GNM still has no physical jaw or "
    "lip/tongue collision rig and the result is not artist-approved."
)
A2F_LICENSE_CAVEAT = "Audio2Face model weights are governed by the NVIDIA Open Model License."
SECONDARY_MOTION_CAVEAT = (
    "Blink, gaze, and head motion are deterministic audio-conditioned secondary motion, "
    "not recovered from the speaker."
)
LIP_CONTACT_CAVEAT = (
    "Lip-contact correction is inferred from Audio2Face and coarse cue evidence; "
    "it is not a phone-annotated collision solve or an artist-approved contact pass."
)
LIP_CONTACT_ALIGNMENT_CAVEAT = (
    "Learned lip-closure evidence had no agreeing closed-mouth alignment cue; "
    "the pipeline failed closed and left those contacts unresolved. Provide an exact transcript "
    "or independently timed phone/contact annotations."
)
LEARNED_BACKEND = "audio2face-3d-v2.3.1-claire-mlx+arkit-solve+gnm-dense-calibrated-v3"
LEARNED_CONDITIONER = "detail-preserving-articulation-v4-contact-anchored-quality-space"
_CONTACT_CRITICAL_CONTROLS = frozenset(
    ("mouthClose", "mouthPressLeft", "mouthPressRight", "mouthRollLower", "mouthRollUpper")
)
_BLINK_CONTROLS = frozenset(("eyeBlinkLeft", "eyeBlinkRight"))
_ARTICULATION_CRITICAL_CONTROLS = frozenset(
    (
        "jawOpen",
        "jawForward",
        "jawLeft",
        "jawRight",
        "mouthFunnel",
        "mouthPucker",
        "mouthLeft",
        "mouthRight",
        "mouthShrugLower",
        "mouthShrugUpper",
    )
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_a2f_assets(explicit: str | Path | None) -> Path:
    configured = explicit or os.environ.get("AUTOANIM_A2F_ASSET_DIR")
    root = Path(configured) if configured else _PROJECT_ROOT / ".cache/autoanim_gnm/a2f-claire"
    required = ("model_data.npz", "bs_skin.npz", "bs_skin_config.json", "bs_tongue.npz", "bs_tongue_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise A2FRunnerError(
            f"Claire retarget assets are unavailable in {root} (missing: {', '.join(missing)})"
        )
    return root


def _mouth_aperture(landmarks: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(landmarks[a] - landmarks[b]) for a, b in ((61, 67), (62, 66), (63, 65))]))


def _smooth_control_matrix(
    values: np.ndarray,
    preferred_window: int,
    *,
    detail_gain: float = 0.0,
) -> np.ndarray:
    """Zero-phase conditioning with a bounded amount of source detail.

    ``detail_gain`` restores part of the residual removed by the polynomial
    fit.  A value of zero is the conventional Savitzky-Golay output and one
    is the unfiltered source.  The explicit gain lets fast articulation retain
    its attack without forcing every upper-face channel through the same
    bandwidth.
    """

    controls = np.asarray(values, dtype=np.float32)
    if controls.ndim != 2 or not np.isfinite(controls).all():
        raise A2FValidationError("Learned control matrix must be finite and two-dimensional")
    window = min(preferred_window, len(controls) if len(controls) % 2 else len(controls) - 1)
    if window < 5:
        return controls.copy()
    if not np.isfinite(detail_gain) or not 0.0 <= detail_gain <= 1.0:
        raise A2FValidationError("detail_gain must be finite and in [0,1]")
    baseline = np.asarray(
        savgol_filter(controls, window, 2, axis=0, mode="interp"),
        dtype=np.float32,
    )
    restored = baseline + np.float32(detail_gain) * (controls - baseline)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def _condition_learned_controls(
    skin_weights: np.ndarray,
    skin_pose_names: tuple[str, ...],
    tongue_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce retarget jitter while retaining fast speech-contact controls.

    A fixed nine-frame window erased roughly 267 ms of fast performance at the
    30 fps model clock.  This version uses a five-frame base, restores more
    residual on jaw/lip articulation and blinks, and leaves explicit
    closure/press/roll controls untouched.  Tongue motion uses the same short
    support with half of its residual restored.  Timestamps remain unchanged,
    so the offline zero-phase pass cannot add A/V latency.
    """

    skin = np.asarray(skin_weights, dtype=np.float32)
    names = tuple(skin_pose_names)
    if skin.ndim != 2 or skin.shape[1] != len(names) or len(set(names)) != len(names):
        raise A2FValidationError("Skin controls and pose names are inconsistent")
    conditioned_skin = _smooth_control_matrix(skin, 5, detail_gain=0.20)
    articulation = _smooth_control_matrix(skin, 5, detail_gain=0.55)
    blink = _smooth_control_matrix(skin, 5, detail_gain=0.80)
    for name in _ARTICULATION_CRITICAL_CONTROLS:
        if name in names:
            conditioned_skin[:, names.index(name)] = articulation[:, names.index(name)]
    for name in _BLINK_CONTROLS:
        if name in names:
            conditioned_skin[:, names.index(name)] = blink[:, names.index(name)]
    for name in _CONTACT_CRITICAL_CONTROLS:
        if name in names:
            conditioned_skin[:, names.index(name)] = skin[:, names.index(name)]
    conditioned_tongue = _smooth_control_matrix(
        tongue_weights,
        5,
        detail_gain=0.50,
    )
    return conditioned_skin, conditioned_tongue


def _derive_lip_contact_confidence(
    conditioned_skin: np.ndarray,
    skin_pose_names: tuple[str, ...],
    source_activity: np.ndarray,
) -> np.ndarray:
    """Estimate when the learned performance intends bilabial contact.

    ``mouthClose`` is useful source evidence but cannot be used directly as a
    GNM target direction: Claire's calibrated positive direction opens the
    target lip landmarks. Confidence therefore starts from rest-relative
    mouth-close evidence, uses press/roll only as a tension modifier, and
    applies jaw-plausibility plus active-speech gates. Geometry is corrected
    later along an independently measured GNM direction and only when the
    aligner's closed-mouth cue agrees.
    """

    values = np.asarray(conditioned_skin, dtype=np.float32)
    names = tuple(skin_pose_names)
    activity = np.asarray(source_activity, dtype=np.float32)
    required = (
        "mouthClose",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthRollLower",
        "mouthRollUpper",
        "jawOpen",
    )
    missing = [name for name in required if name not in names]
    if (
        values.ndim != 2
        or values.shape[1] != len(names)
        or activity.shape != (len(values),)
        or not np.isfinite(values).all()
        or not np.isfinite(activity).all()
    ):
        raise A2FValidationError("Lip-contact evidence must align with finite skin controls")
    if missing:
        raise A2FValidationError(
            f"Lip-contact evidence is missing Claire controls: {', '.join(missing)}"
        )

    index = {name: names.index(name) for name in required}
    mouth_close = values[:, index["mouthClose"]]
    quiet = activity <= 0.08
    rest_close = float(
        np.median(mouth_close[quiet])
        if np.count_nonzero(quiet) >= 3
        else np.percentile(mouth_close, 5)
    )
    close_delta = np.clip(mouth_close - rest_close, 0.0, 1.0)
    closure = np.clip((close_delta - 0.12) / 0.30, 0.0, 1.0)
    closure = closure * closure * (3.0 - 2.0 * closure)
    # Press/roll describe seal tension and lip shape; they are not phone
    # evidence and therefore may modulate, but never initiate, a contact.
    tension = np.maximum(
        0.5
        * (
            values[:, index["mouthPressLeft"]]
            + values[:, index["mouthPressRight"]]
        ),
        0.5
        * (
            values[:, index["mouthRollLower"]]
            + values[:, index["mouthRollUpper"]]
        ),
    )
    tension_gain = 0.85 + 0.15 * np.clip(tension / 0.50, 0.0, 1.0)
    jaw_open = values[:, index["jawOpen"]]
    # Jaw openness is a plausibility weight, not a veto: real bilabial seals
    # can coexist with some mandibular opening through lip stretch.
    jaw_gate = 0.25 + 0.75 * np.clip((0.42 - jaw_open) / 0.30, 0.0, 1.0)
    speech_gate = np.clip((activity - 0.02) / 0.18, 0.0, 1.0)
    confidence = closure * tension_gain * jaw_gate * speech_gate
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _quarantine_mouth_close_retarget(
    conditioned_skin: np.ndarray,
    skin_pose_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove the semantically inverted mouth-close row from dense retargeting.

    The calibrated Claire ``mouthClose`` row opens GNM's inner-lip landmarks.
    It remains available to the contact-evidence model, but its target-space
    contribution is zeroed so a later phone-gated correction does not have to
    fight an equal and opposite deformation.
    """

    values = np.asarray(conditioned_skin, dtype=np.float32)
    names = tuple(skin_pose_names)
    if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
        raise A2FValidationError("Mouth-close quarantine requires finite aligned skin controls")
    if "mouthClose" not in names:
        raise A2FValidationError("Claire skin controls are missing mouthClose")
    output = values.copy()
    index = names.index("mouthClose")
    source = output[:, index].copy()
    output[:, index] = 0.0
    return output, {
        "mouth_close_quarantined_peak": float(np.max(source, initial=0.0)),
        "mouth_close_quarantined_frames": float(np.count_nonzero(source > 1e-5)),
    }


def _conditioning_metrics(
    skin: np.ndarray,
    conditioned_skin: np.ndarray,
    skin_names: tuple[str, ...],
    tongue: np.ndarray,
    conditioned_tongue: np.ndarray,
) -> dict[str, float]:
    general_indices = [
        index for index, name in enumerate(skin_names) if name not in _CONTACT_CRITICAL_CONTROLS
    ]
    raw_general = np.column_stack((skin[:, general_indices], tongue))
    filtered_general = np.column_stack(
        (conditioned_skin[:, general_indices], conditioned_tongue)
    )
    if len(raw_general) >= 4:
        raw_jerk = np.linalg.norm(np.diff(raw_general, n=3, axis=0), axis=1)
        filtered_jerk = np.linalg.norm(np.diff(filtered_general, n=3, axis=0), axis=1)
        raw_p95 = float(np.percentile(raw_jerk, 95))
        jerk_ratio = float(np.percentile(filtered_jerk, 95) / max(raw_p95, 1e-8))
    else:
        jerk_ratio = 1.0
    retention: list[float] = []
    for name in _CONTACT_CRITICAL_CONTROLS:
        if name not in skin_names:
            continue
        index = skin_names.index(name)
        raw_peak = float(np.max(skin[:, index], initial=0.0))
        if raw_peak > 1e-5:
            retention.append(
                float(np.max(conditioned_skin[:, index], initial=0.0) / raw_peak)
            )
    articulation_retention: list[float] = []
    for name in _ARTICULATION_CRITICAL_CONTROLS | _CONTACT_CRITICAL_CONTROLS:
        if name not in skin_names:
            continue
        index = skin_names.index(name)
        raw_range = float(np.ptp(skin[:, index]))
        # Tiny solver excursions are noise-scale and make a minimum ratio
        # unstable (for example a 0.01 lateral mouth twitch).  Score only
        # controls with at least two percent of their normalized range.
        if raw_range > 0.02:
            articulation_retention.append(
                float(np.ptp(conditioned_skin[:, index]) / raw_range)
            )

    def rank95(values: np.ndarray) -> int:
        centered = np.asarray(values, dtype=np.float64) - np.mean(values, axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = singular * singular
        total = float(np.sum(energy))
        if total <= 1e-12:
            return 0
        return int(np.searchsorted(np.cumsum(energy) / total, 0.95) + 1)

    raw_rank = rank95(np.column_stack((skin, tongue)))
    conditioned_rank = rank95(np.column_stack((conditioned_skin, conditioned_tongue)))
    return {
        "conditioning_noncontact_jerk_p95_ratio": jerk_ratio,
        "conditioning_contact_peak_retention_min": min(retention, default=1.0),
        "conditioning_articulation_range_retention_min": min(
            articulation_retention,
            default=1.0,
        ),
        "conditioning_rank95_retention": (
            float(conditioned_rank / raw_rank) if raw_rank else 1.0
        ),
    }


def _fuse_jaw_observation(
    conditioned_skin: np.ndarray,
    skin_names: tuple[str, ...],
    jaw_rotation_vectors_degrees: np.ndarray,
    source_activity: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Use Claire's physical jaw track as a soft ``jawOpen`` constraint.

    GNM has no mandible joint, so this does not pretend to be a hinge solve.
    It restores the timing and minimum magnitude of NVIDIA's five-point jaw
    observation in the closest available named source control.  A character-
    calibrated mandible layer remains the production replacement.
    """

    values = np.asarray(conditioned_skin, dtype=np.float32)
    rotations = np.asarray(jaw_rotation_vectors_degrees, dtype=np.float32)
    activity = np.asarray(source_activity, dtype=np.float32)
    if rotations.shape != (len(values), 3) or activity.shape != (len(values),):
        raise A2FValidationError("Jaw observations must align with conditioned controls")
    if not np.isfinite(rotations).all() or not np.isfinite(activity).all():
        raise A2FValidationError("Jaw observations and activity must be finite")
    if "jawOpen" not in skin_names:
        return values.copy(), {
            "jaw_observation_rotation_range_degrees": float(np.ptp(rotations[:, 0])),
            "jaw_observation_fused_frames": 0.0,
        }

    quiet = activity <= 0.08
    rest = float(
        np.median(rotations[quiet, 0])
        if np.count_nonzero(quiet) >= 3
        else np.percentile(rotations[:, 0], 5)
    )
    # Claire jaw X is an exported Maya rotation in degrees. Eighteen degrees
    # is used only to normalize the soft observation; the final GNM response
    # remains bounded by its calibrated source-to-target matrix.
    observed_open = np.clip((rotations[:, 0] - rest) / 18.0, 0.0, 1.0)
    minimum_drive = np.float32(0.55) * observed_open
    output = values.copy()
    index = skin_names.index("jawOpen")
    before = output[:, index].copy()
    output[:, index] = np.maximum(before, minimum_drive).astype(np.float32)
    fused = np.abs(output[:, index] - before) > 1e-5
    return output, {
        "jaw_observation_rotation_range_degrees": float(np.ptp(rotations[:, 0])),
        "jaw_observation_rest_degrees": rest,
        "jaw_observation_rms_control_delta": float(
            np.sqrt(np.mean((output[:, index] - before) ** 2))
        ),
        "jaw_observation_fused_frames": float(np.count_nonzero(fused)),
    }


def _temporal_metrics(track, rig: ControlRig) -> dict[str, float | int]:
    compact = np.stack([rig.compact_landmarks(frame) for frame in track.expression])
    iod = float(np.linalg.norm(compact[0, 36] - compact[0, 45]))
    mouth = compact[:, 48:68]
    if len(mouth) > 1:
        step = np.max(np.linalg.norm(np.diff(mouth, axis=0), axis=2), axis=1) / max(iod, 1e-8)
        lower_velocity = np.linalg.norm(np.diff(track.expression[:, 200:382], axis=0), axis=1)
        stationary = lower_velocity <= 1e-7
    else:
        step = np.zeros(0, dtype=np.float32)
        lower_velocity = np.zeros(0, dtype=np.float32)
        stationary = np.zeros(0, dtype=bool)
    acceleration = np.diff(lower_velocity) if len(lower_velocity) > 1 else np.zeros(0, dtype=np.float32)
    jerk = np.diff(acceleration) if len(acceleration) > 1 else np.zeros(0, dtype=np.float32)
    centered_expression = track.expression - np.mean(track.expression, axis=0, keepdims=True)
    singular = np.linalg.svd(centered_expression.astype(np.float64), compute_uv=False)
    singular_energy = singular * singular
    total_energy = float(np.sum(singular_energy))
    expression_rank95 = (
        int(np.searchsorted(np.cumsum(singular_energy) / total_energy, 0.95) + 1)
        if total_energy > 1e-12
        else 0
    )
    limited_count = int(np.count_nonzero(track.mouth_speed_limited))
    lip_gap = np.mean(
        np.stack(
            [
                np.linalg.norm(compact[:, upper] - compact[:, lower], axis=1)
                for upper, lower in ((61, 67), (62, 66), (63, 65))
            ],
            axis=1,
        ),
        axis=1,
    ) / max(iod, 1e-8)
    contact_candidates = track.lip_contact_target_gap > 0.0
    strong_contact = contact_candidates & (track.lip_contact_confidence >= 0.55)
    contact_targets = track.lip_contact_target_gap
    attained = track.lip_contact_attained[strong_contact]
    attempted_count = int(np.count_nonzero(track.contact_correction_applied))
    corrected_count = int(np.count_nonzero(track.contact_corrected))
    post_limiter_lost = track.contact_correction_applied & ~track.lip_contact_attained
    target_error = np.maximum(lip_gap - contact_targets, 0.0)
    return {
        "mouth_step_max_interocular": float(np.max(step, initial=0.0)),
        "mouth_step_p95_interocular": float(np.percentile(step, 95)) if len(step) else 0.0,
        "lower_face_stationary_fraction": float(np.mean(stationary)) if len(stationary) else 0.0,
        "lower_face_velocity_p95": float(np.percentile(lower_velocity, 95)) if len(lower_velocity) else 0.0,
        "lower_face_acceleration_p95": float(np.percentile(np.abs(acceleration), 95)) if len(acceleration) else 0.0,
        "lower_face_jerk_p95": float(np.percentile(np.abs(jerk), 95)) if len(jerk) else 0.0,
        "mouth_speed_limited_frames": limited_count,
        "mouth_speed_limited_fraction": float(limited_count / max(len(track.expression), 1)),
        "lip_contact_confidence_peak": float(
            np.max(track.lip_contact_confidence, initial=0.0)
        ),
        "lip_contact_candidate_frames": int(np.count_nonzero(contact_candidates)),
        "lip_contact_strong_frames": int(np.count_nonzero(strong_contact)),
        "lip_contact_correction_applied_frames": attempted_count,
        "lip_contact_continuity_restored_frames": int(
            np.count_nonzero(track.contact_continuity_restored)
        ),
        "lip_contact_corrected_frames": corrected_count,
        "lip_contact_corrected_fraction": float(
            corrected_count / max(len(track.expression), 1)
        ),
        "lip_contact_candidate_gap_p95_interocular": (
            float(np.percentile(lip_gap[contact_candidates], 95))
            if np.any(contact_candidates)
            else 0.0
        ),
        "lip_contact_target_attainment_fraction": (
            float(np.mean(attained)) if len(attained) else 1.0
        ),
        "lip_contact_post_limiter_attained_frames": int(
            np.count_nonzero(track.lip_contact_attained)
        ),
        "lip_contact_post_limiter_lost_frames": int(
            np.count_nonzero(post_limiter_lost)
        ),
        "lip_contact_post_limiter_attainment_fraction": (
            float(np.mean(track.lip_contact_attained[contact_candidates]))
            if np.any(contact_candidates)
            else 1.0
        ),
        "lip_contact_post_limiter_gap_error_p95_interocular": (
            float(np.percentile(target_error[contact_candidates], 95))
            if np.any(contact_candidates)
            else 0.0
        ),
        "expression_effective_rank_95": expression_rank95,
        "upper_face_control_range_max": float(
            np.max(np.ptp(track.expression[:, :200], axis=0), initial=0.0)
        ),
        "head_rotation_max_degrees": float(np.rad2deg(np.max(np.linalg.norm(track.rotations[:, :2], axis=2), initial=0.0))),
        "eye_rotation_max_degrees": float(
            np.rad2deg(
                np.max(
                    np.linalg.norm(track.rotations[:, 2:4], axis=2),
                    initial=0.0,
                )
            )
        ),
        "emotion_intensity_range": float(np.ptp(track.emotion_intensity)),
    }


def _quality_speech_activity(
    speech_activity: np.ndarray,
    *,
    hangover_frames: int = 2,
) -> np.ndarray:
    """Expand recognized speech only for silence-quality evaluation.

    The acoustic VAD can emit isolated false-negative frames inside a word.
    A short symmetric hangover keeps those frames out of the true-silence
    measurement without changing animation controls or hiding long silences.
    """

    values = np.asarray(speech_activity, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise AutoAnimError("INTERNAL_ERROR", "Speech activity must be a finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise AutoAnimError("INTERNAL_ERROR", "Speech activity must be in [0, 1]")
    if isinstance(hangover_frames, bool) or not isinstance(hangover_frames, int):
        raise AutoAnimError("INTERNAL_ERROR", "Speech hangover must be an integer")
    if hangover_frames < 0:
        raise AutoAnimError("INTERNAL_ERROR", "Speech hangover must be non-negative")

    expanded = values >= 0.5
    for _ in range(hangover_frames):
        prior = expanded.copy()
        expanded[1:] |= prior[:-1]
        expanded[:-1] |= prior[1:]
    return expanded.astype(np.float32)


def run_audio_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    fps: int = 30,
    emotion: str = "auto",
    dialog: str | None = None,
    rhubarb_bin: str | Path | None = None,
    backend: str = "auto",
    a2f_runner: str | Path | None = None,
    a2f_asset_dir: str | Path | None = None,
    a2f_offline: bool = False,
    emotion_strength: float = 0.65,
    identity: np.ndarray | None = None,
    texture_path: str | Path | None = None,
    texture_triangle_uvs: np.ndarray | None = None,
    character_ref: dict[str, Any] | None = None,
) -> dict:
    if backend not in {"auto", "learned", "fallback"}:
        raise AutoAnimError("INPUT_INVALID", "Backend must be auto, learned, or fallback")
    if not np.isfinite(emotion_strength) or not 0.0 <= emotion_strength <= 1.0:
        raise AutoAnimError("INPUT_INVALID", "Emotion strength must be in [0,1]")
    identity_value = (
        np.zeros(253, dtype=np.float32)
        if identity is None
        else np.asarray(identity, dtype=np.float32).copy()
    )
    if identity_value.shape != (253,) or not np.isfinite(identity_value).all():
        raise AutoAnimError("INPUT_INVALID", "Character identity must be one finite (253,) vector")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "normalized.wav"
    duration = normalize_audio(input_path, normalized_path)
    raw_cues = run_rhubarb(
        normalized_path,
        output_dir / "rhubarb.json",
        rhubarb_bin=rhubarb_bin,
        dialog=dialog,
    )
    cues = normalize_cues(raw_cues, duration)
    analysis = analyze_emotion(normalized_path, cues, manual=emotion, dialog=dialog)
    prosody = extract_prosody(normalized_path, cues, fps)
    adapter = GNMAdapter()
    decoder = ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5")
    rig = ControlRig(adapter, decoder, identity=identity_value)
    # Calibrate against the selected identity once so automatic and explicit
    # fallback retain the same bilabial-contact contract as learned audio.
    lip_contact_calibration = calibrate_lip_contact(rig)
    learned_error: str | None = None
    learned_artifacts: dict[str, str] = {}
    motion_backend = "procedural_fallback"
    backend_name = "procedural-v2+rhubarb-1.14.0"
    emotion_applied = analysis.emotion
    retargeter_name = "semantic_fallback_v1"
    retarget_calibration_hash: str | None = None
    conditioning: dict[str, float] = {}
    emotion_decomposition = "not_applicable"
    if backend in {"auto", "learned"}:
        try:
            asset_root = _resolve_a2f_assets(a2f_asset_dir)
            # Automatic heuristic labels are diagnostic only. Manual or
            # confidence-gated lexical direction is decomposed from a neutral
            # acoustic performance so a held affect does not replace mouth
            # timing or get erased by rest-bias removal.
            a2f_emotion = analysis.emotion if analysis.validated else "neutral"
            applied_strength = emotion_strength if a2f_emotion != "neutral" else 0.0
            raw_path = output_dir / "a2f_raw.jsonl"
            frames = run_a2f_runner(
                normalized_path,
                runner=a2f_runner,
                output_path=raw_path,
                offline=a2f_offline,
                emotion="neutral",
                emotion_strength=0.0,
            )
            source_timestamps = np.asarray(
                [frame.time_seconds for frame in frames], dtype=np.float32
            )
            with np.load(asset_root / "model_data.npz", allow_pickle=False) as model_data:
                if "neutral_jaw" not in model_data:
                    raise A2FValidationError("Claire model_data.npz is missing neutral_jaw")
                neutral_jaw = np.asarray(model_data["neutral_jaw"], dtype=np.float32)
            auxiliary = recover_a2f_auxiliary_track(frames, neutral_jaw)
            source_activity = np.interp(
                source_timestamps.astype(np.float64),
                prosody.timestamps.astype(np.float64),
                prosody.speech_activity.astype(np.float64),
                left=float(prosody.speech_activity[0]),
                right=float(prosody.speech_activity[-1]),
            ).astype(np.float32)
            skin_solver = ClaireSkinSolver.from_directory(asset_root)
            tongue_solver = ClaireTongueSolver.from_directory(asset_root)
            skin_weights = skin_solver.solve_frames(frames)
            tongue_weights = tongue_solver.solve_frames(frames)
            conditioned_skin, conditioned_tongue = _condition_learned_controls(
                skin_weights,
                skin_solver.pose_names,
                tongue_weights,
            )
            conditioned_skin, jaw_fusion = _fuse_jaw_observation(
                conditioned_skin,
                skin_solver.pose_names,
                auxiliary.jaw_rotation_vectors_degrees,
                source_activity,
            )
            source_lip_contact_confidence = _derive_lip_contact_confidence(
                conditioned_skin,
                skin_solver.pose_names,
                source_activity,
            )
            retarget_skin, quarantine_metrics = _quarantine_mouth_close_retarget(
                conditioned_skin,
                skin_solver.pose_names,
            )
            conditioning = _conditioning_metrics(
                skin_weights,
                conditioned_skin,
                skin_solver.pose_names,
                tongue_weights,
                conditioned_tongue,
            )
            conditioning.update(jaw_fusion)
            conditioning.update(quarantine_metrics)
            conditioning.update(
                {
                    "lip_contact_source_evidence_frames": float(
                        np.count_nonzero(source_lip_contact_confidence >= 0.12)
                    ),
                    "lip_contact_source_evidence_peak": float(
                        np.max(source_lip_contact_confidence, initial=0.0)
                    ),
                }
            )
            conditioning["jaw_observation_rms_residual_p95"] = float(
                np.percentile(auxiliary.jaw_rms_residual, 95)
            )
            calibrated_retargeter = CalibratedRetargeter.from_directory(
                asset_root,
                adapter=adapter,
            )
            conditioning.update(
                {
                    "lip_contact_character_neutral_gap_interocular": (
                        lip_contact_calibration.neutral_gap_interocular
                    ),
                    "lip_contact_character_seal_gap_interocular": (
                        lip_contact_calibration.seal_gap_interocular
                    ),
                    "lip_contact_character_maximum_alpha": (
                        lip_contact_calibration.maximum_alpha
                    ),
                    "lip_contact_calibration_nonmouth_p95_displacement_interocular": (
                        lip_contact_calibration.nonmouth_p95_displacement_interocular
                    ),
                    "lip_contact_calibration_nonmouth_max_displacement_interocular": (
                        lip_contact_calibration.nonmouth_max_displacement_interocular
                    ),
                }
            )
            learned_expression = calibrated_retargeter.retarget_sequence(
                retarget_skin,
                skin_solver.pose_names,
                tongue_weights=conditioned_tongue,
                tongue_pose_names=tongue_solver.pose_names,
                strict=True,
            )
            emotion_expression: np.ndarray | None = None
            emotion_skin_weights: np.ndarray | None = None
            conditioned_emotion_skin: np.ndarray | None = None
            emotion_tongue_weights: np.ndarray | None = None
            conditioned_emotion_tongue: np.ndarray | None = None
            emotion_frames = None
            emotion_auxiliary = None
            emotion_eye_delta: np.ndarray | None = None
            if applied_strength > 0.0:
                emotion_path = output_dir / "a2f_emotion_raw.jsonl"
                emotion_frames = run_a2f_runner(
                    normalized_path,
                    runner=a2f_runner,
                    output_path=emotion_path,
                    offline=a2f_offline,
                    emotion=a2f_emotion,
                    emotion_strength=1.0,
                )
                emotion_timestamps = np.asarray(
                    [frame.time_seconds for frame in emotion_frames],
                    dtype=np.float32,
                )
                neutral_timestamps = np.asarray(
                    [frame.time_seconds for frame in frames],
                    dtype=np.float32,
                )
                if (
                    len(emotion_frames) != len(frames)
                    or not np.array_equal(emotion_timestamps, neutral_timestamps)
                ):
                    raise A2FValidationError(
                        "Neutral and emotional Audio2Face passes must share an exact clock"
                    )
                emotion_skin_weights = skin_solver.solve_frames(emotion_frames)
                emotion_tongue_weights = tongue_solver.solve_frames(emotion_frames)
                emotion_auxiliary = recover_a2f_auxiliary_track(
                    emotion_frames,
                    neutral_jaw,
                )
                conditioned_emotion_skin, conditioned_emotion_tongue = (
                    _condition_learned_controls(
                        emotion_skin_weights,
                        skin_solver.pose_names,
                        emotion_tongue_weights,
                    )
                )
                conditioned_emotion_skin, _ = _fuse_jaw_observation(
                    conditioned_emotion_skin,
                    skin_solver.pose_names,
                    emotion_auxiliary.jaw_rotation_vectors_degrees,
                    source_activity,
                )
                retarget_emotion_skin, _ = _quarantine_mouth_close_retarget(
                    conditioned_emotion_skin,
                    skin_solver.pose_names,
                )
                emotional_expression = calibrated_retargeter.retarget_sequence(
                    retarget_emotion_skin,
                    skin_solver.pose_names,
                    tongue_weights=conditioned_emotion_tongue,
                    tongue_pose_names=tongue_solver.pose_names,
                    strict=True,
                )
                emotion_expression = emotional_expression - learned_expression
                emotion_eye_delta = (
                    emotion_auxiliary.eye_rotations_degrees
                    - auxiliary.eye_rotations_degrees
                )
                emotion_decomposition = "neutral_content_plus_explicit_delta_v1"
            else:
                emotion_decomposition = "neutral_content_only"
            calibrated_retargeter.calibration.save(output_dir / "retarget_calibration.npz")
            retargeter_name = "geometry_calibrated_dense_v3_spatial_contact"
            retarget_calibration_hash = calibrated_retargeter.calibration.calibration_hash
            track = compose_learned_animation(
                learned_expression,
                source_timestamps,
                cues,
                duration,
                fps,
                rig,
                prosody,
                acting_strength=applied_strength,
                emotion_delta=emotion_expression,
                source_eye_rotations_degrees=auxiliary.eye_rotations_degrees,
                emotion_eye_delta_degrees=emotion_eye_delta,
                source_lip_contact_confidence=source_lip_contact_confidence,
                lip_contact_calibration=lip_contact_calibration,
            )
            control_arrays: dict[str, np.ndarray] = {
                "timestamps": source_timestamps,
                "skin_weights": skin_weights,
                "conditioned_skin_weights": conditioned_skin,
                "retarget_skin_weights": retarget_skin,
                "skin_pose_names": np.asarray(skin_solver.pose_names),
                "tongue_weights": tongue_weights,
                "conditioned_tongue_weights": conditioned_tongue,
                "tongue_pose_names": np.asarray(tongue_solver.pose_names),
                "jaw_points": auxiliary.jaw_points,
                "jaw_rotation_matrices": auxiliary.jaw_rotation_matrices,
                "jaw_rotation_vectors_degrees": auxiliary.jaw_rotation_vectors_degrees,
                "jaw_translations": auxiliary.jaw_translations,
                "jaw_rms_residual": auxiliary.jaw_rms_residual,
                "eye_rotations_degrees": auxiliary.eye_rotations_degrees,
                "source_lip_contact_confidence": source_lip_contact_confidence,
                "gnm_lip_contact_direction": lip_contact_calibration.direction,
                "gnm_lip_contact_inner_response": lip_contact_calibration.inner_response,
                "gnm_lip_contact_neutral_pair_gaps_interocular": (
                    lip_contact_calibration.neutral_pair_gaps_interocular
                ),
                "gnm_lip_contact_seal_pair_gaps_interocular": (
                    lip_contact_calibration.seal_pair_gaps_interocular
                ),
                "gnm_lip_contact_neutral_gap_interocular": np.asarray(
                    lip_contact_calibration.neutral_gap_interocular,
                    dtype=np.float32,
                ),
                "gnm_lip_contact_seal_gap_interocular": np.asarray(
                    lip_contact_calibration.seal_gap_interocular,
                    dtype=np.float32,
                ),
                "gnm_lip_contact_maximum_alpha": np.asarray(
                    lip_contact_calibration.maximum_alpha,
                    dtype=np.float32,
                ),
                "gnm_lip_contact_calibration_hash": np.asarray(
                    lip_contact_calibration.calibration_hash,
                ),
            }
            if emotion_expression is not None:
                assert emotion_skin_weights is not None
                assert conditioned_emotion_skin is not None
                assert emotion_tongue_weights is not None
                assert conditioned_emotion_tongue is not None
                assert emotion_auxiliary is not None
                control_arrays.update(
                    {
                        "emotion_skin_weights": emotion_skin_weights,
                        "conditioned_emotion_skin_weights": conditioned_emotion_skin,
                        "retarget_emotion_skin_weights": retarget_emotion_skin,
                        "emotion_tongue_weights": emotion_tongue_weights,
                        "conditioned_emotion_tongue_weights": conditioned_emotion_tongue,
                        "gnm_emotion_delta": emotion_expression,
                        "emotion_jaw_rotation_vectors_degrees": (
                            emotion_auxiliary.jaw_rotation_vectors_degrees
                        ),
                        "emotion_jaw_translations": emotion_auxiliary.jaw_translations,
                        "emotion_eye_rotations_degrees": (
                            emotion_auxiliary.eye_rotations_degrees
                        ),
                    }
                )
            write_npz(
                output_dir / "arkit_controls.npz",
                **control_arrays,
            )
            learned_artifacts = {
                "a2f_raw": "a2f_raw.jsonl",
                "arkit_controls": "arkit_controls.npz",
                "retarget_calibration": "retarget_calibration.npz",
            }
            if emotion_frames is not None:
                learned_artifacts["a2f_emotion_raw"] = "a2f_emotion_raw.jsonl"
            motion_backend = "learned_a2f"
            backend_name = LEARNED_BACKEND
            emotion_applied = a2f_emotion
        except (A2FRunnerError, A2FValidationError, OSError, ValueError) as exc:
            learned_error = str(exc)
            if backend == "learned":
                raise AutoAnimError(
                    "DEPENDENCY_MISSING" if isinstance(exc, A2FRunnerError) else "INTERNAL_ERROR",
                    f"Learned Audio2Face backend failed: {exc}",
                ) from exc
            track = compose_animation(
                cues,
                duration,
                fps,
                rig,
                analysis.emotion,
                prosody,
                lip_contact_calibration=lip_contact_calibration,
            )
    else:
        track = compose_animation(
            cues,
            duration,
            fps,
            rig,
            analysis.emotion,
            prosody,
            lip_contact_calibration=lip_contact_calibration,
        )
    apertures = [
        _mouth_aperture(rig.compact_landmarks(frame)) for frame in track.expression
    ]
    write_json(output_dir / "cues.json", {"duration": duration, "cues": [cue.as_dict() for cue in cues]})
    write_npz(
        output_dir / "controls.npz",
        expression=track.expression,
        rotations=track.rotations,
        translation=track.translation,
        timestamps=track.timestamps,
        fps=np.asarray(track.fps, dtype=np.int32),
        viseme_weights=track.viseme_weights,
        speech_activity=track.speech_activity,
        energy=track.energy,
        pitch_semitones=track.pitch_semitones,
        accent=track.accent,
        phrase_id=track.phrase_id,
        emotion_intensity=track.emotion_intensity,
        mouth_speed_limited=track.mouth_speed_limited,
        lip_contact_confidence=track.lip_contact_confidence,
        lip_contact_target_gap=track.lip_contact_target_gap,
        contact_correction_applied=track.contact_correction_applied,
        lip_contact_attained=track.lip_contact_attained,
        contact_continuity_restored=track.contact_continuity_restored,
        contact_corrected=track.contact_corrected,
    )
    write_json(
        output_dir / "timeline.json",
        {
            "version": 9,
            "motion_backend": motion_backend,
            "retargeter": retargeter_name,
            "retarget_calibration_hash": retarget_calibration_hash,
            "temporal_conditioner": LEARNED_CONDITIONER if motion_backend == "learned_a2f" else None,
            "timestamps": track.timestamps.tolist(),
            "cue_order": list("XABCDEFGH"),
            "viseme_weights": track.viseme_weights.tolist(),
            "speech_activity": track.speech_activity.tolist(),
            "energy": track.energy.tolist(),
            "pitch_semitones": track.pitch_semitones.tolist(),
            "accent": track.accent.tolist(),
            "phrase_id": track.phrase_id.tolist(),
            "emotion_intensity": track.emotion_intensity.tolist(),
            "mouth_speed_limited": track.mouth_speed_limited.tolist(),
            "lip_contact_confidence": track.lip_contact_confidence.tolist(),
            "lip_contact_target_gap": track.lip_contact_target_gap.tolist(),
            "contact_correction_applied": track.contact_correction_applied.tolist(),
            "lip_contact_attained": track.lip_contact_attained.tolist(),
            "contact_continuity_restored": track.contact_continuity_restored.tolist(),
            "contact_corrected": track.contact_corrected.tolist(),
            "mouth_aperture": apertures,
        },
    )
    silent_path = render_silent_video(
        track,
        adapter,
        output_dir / "preview-silent.mp4",
        identity=identity_value,
    )
    preview_path = mux_audio(silent_path, normalized_path, output_dir / "preview.mp4")
    av = probe_av(preview_path)
    offset_frames = abs(float(av["video_duration"]) - duration) * fps
    if av["video_frames"] != len(track.expression) or offset_frames > 1.0:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Preview stream does not preserve the complete control track within one audio frame.",
            {
                "control_frames": len(track.expression),
                "video_frames": av["video_frames"],
                "offset_frames": offset_frames,
            },
        )
    viewer_frames = np.stack(
        [
            adapter.mesh(
                identity=identity_value,
                expression=expression,
                rotations=rotations,
                translation=translation,
            )
            for expression, rotations, translation in zip(
                track.expression, track.rotations, track.translation, strict=True
            )
        ]
    )
    viewer_warning: str | None = None
    viewer_artifacts: dict[str, str] = {}
    try:
        viewer_export = export_animated_gnm_glb(
            output_dir / "animation.glb",
            adapter,
            viewer_frames,
            track.timestamps,
            mapping_path=output_dir / "animation-glb-mapping.npz",
            texture_path=texture_path,
            triangle_uvs=texture_triangle_uvs,
        )
        viewer_status = "ready" if viewer_export.rank else "static_only"
        viewer_reconstruction = {
            "expression_pose_rank": viewer_export.rank,
            "validation_scope": "all_frames",
            "mesh_p95_mm": viewer_export.mesh_p95_mm,
            "mesh_max_mm": viewer_export.mesh_max_mm,
            "landmark_p95_mm": viewer_export.landmark_p95_mm,
            "landmark_max_mm": viewer_export.landmark_max_mm,
        }
        viewer_artifacts = {
            "glb": "animation.glb",
            "glb_mapping": "animation-glb-mapping.npz",
            "normalized_audio": "normalized.wav",
        }
    except AnimationCompressionError as exc:
        static_export = export_gnm_glb(
            output_dir / "animation.glb",
            adapter,
            viewer_frames[0],
            mapping_path=output_dir / "animation-glb-mapping.npz",
            texture_path=texture_path,
            triangle_uvs=texture_triangle_uvs,
        )
        viewer_status = "static_only"
        viewer_warning = "VIEWER_RECONSTRUCTION_LIMIT"
        viewer_reconstruction = {
            "expression_pose_rank": int(exc.metrics.get("rank", 0)),
            "validation_scope": "all_frames",
            "mesh_p95_mm": float(exc.metrics.get("mesh_p95_m", 0.0)) * 1000.0,
            "mesh_max_mm": float(exc.metrics.get("mesh_max_m", 0.0)) * 1000.0,
            "landmark_p95_mm": float(exc.metrics.get("landmark_p95_m", 0.0)) * 1000.0,
            "landmark_max_mm": float(exc.metrics.get("landmark_max_m", 0.0)) * 1000.0,
            "static_vertices": static_export.vertex_count,
        }
        viewer_artifacts = {
            "glb": "animation.glb",
            "glb_mapping": "animation-glb-mapping.npz",
            "normalized_audio": "normalized.wav",
        }
    sample_indices = np.unique(
        np.linspace(0, len(track.expression) - 1, min(12, len(track.expression)), dtype=int)
    )
    meshes_finite = bool(np.isfinite(viewer_frames[sample_indices]).all())
    neutral_aperture = _mouth_aperture(rig.compact_landmarks(rig.viseme("X")))
    temporal = _temporal_metrics(track, rig)
    quality_landmarks = np.stack([rig.compact_landmarks(frame) for frame in track.expression])
    quality_activity = _quality_speech_activity(track.speech_activity)
    quality = evaluate_lipsync_quality(
        quality_landmarks,
        rig.compact_landmarks(np.zeros(adapter.expression_dim, dtype=np.float32)),
        quality_activity,
        fps=track.fps,
    )
    warnings: list[str] = []
    if motion_backend == "learned_a2f":
        warnings.extend(
            (
                LEARNED_RETARGET_CAVEAT,
                A2F_LICENSE_CAVEAT,
                SECONDARY_MOTION_CAVEAT,
                LIP_CONTACT_CAVEAT,
            )
        )
        if (
            conditioning.get("lip_contact_source_evidence_frames", 0.0) > 0.0
            and temporal["lip_contact_candidate_frames"] == 0
        ):
            warnings.append(LIP_CONTACT_ALIGNMENT_CAVEAT)
    else:
        warnings.extend((AUDIO_CAVEAT, FALLBACK_CAVEAT))
    if learned_error is not None:
        warnings.append(f"LEARNED_BACKEND_UNAVAILABLE: {learned_error}")
    if not analysis.validated:
        warnings.append(EMOTION_CAVEAT)
    if track.saturated:
        warnings.append("COEFFICIENT_SATURATED")
    if viewer_warning is not None:
        warnings.append(viewer_warning)
    if texture_path is not None:
        warnings.append(
            "CHARACTER_TEXTURE_GLTF_ONLY: the interactive GLB uses the saved texture; "
            "the downloadable MP4 remains an untextured diagnostic preview."
        )
    result = {
        "kind": "audio_animation",
        "status": "succeeded",
        "model": {
            "gnm_version": "3.0",
            "identity_dim": adapter.identity_dim,
            "expression_dim": adapter.expression_dim,
            "character": character_ref,
            "character_texture_applied_to_glb": texture_path is not None,
            "preview_texture_applied": False,
        },
        "audio": {"duration_s": round(duration, 8), "sample_rate": 16000},
        "analysis": {
            "backend": backend_name,
            "motion_backend": motion_backend,
            "retargeter": retargeter_name,
            "retarget_calibration_hash": retarget_calibration_hash,
            "temporal_conditioner": LEARNED_CONDITIONER if motion_backend == "learned_a2f" else None,
            "emotion_decomposition": emotion_decomposition,
            "secondary_motion": (
                "deterministic_audio_conditioned_v2"
                if motion_backend == "learned_a2f"
                else "deterministic_audio_conditioned_v1"
            ),
            "lip_contact": (
                "learned_evidence_plus_character_spatial_contact_v3_anchored"
                if motion_backend == "learned_a2f"
                else "rhubarb_bilabial_plus_character_spatial_contact_v1"
            ),
            "quality_speech_mask": "vad_binary_symmetric_hangover_2_frames",
            "emotion": analysis.emotion,
            "emotion_applied": emotion_applied,
            "emotion_strength": emotion_strength if emotion_applied != "neutral" else 0.0,
            "emotion_confidence": analysis.confidence,
            "emotion_validated": analysis.validated,
            "emotion_source": analysis.source,
            "features": analysis.features,
            "cues": [cue.as_dict() for cue in cues],
        },
        "animation": {
            "fps": track.fps,
            "frames": len(track.expression),
            "expression_shape": list(track.expression.shape),
            "compiler_version": 9 if motion_backend == "learned_a2f" else 2,
            "production_validated": False,
        },
        "viewer": {
            "schema_version": "1.0",
            "status": viewer_status,
            "mode": "animation" if viewer_status == "ready" else "static",
            "model_artifact": "glb",
            "animation_clip": "autoanim" if viewer_status == "ready" else None,
            "clock_artifact": "normalized_audio",
            "timeline_artifact": "timeline",
            "duration_s": round(duration, 8),
            "fps": track.fps,
            "coordinate_system": "+Y_up_+Z_forward_meters",
            "reconstruction": viewer_reconstruction,
        },
        "quality": quality.as_dict(),
        "metrics": {
            "cue_coverage": sum(cue.end - cue.start for cue in cues) / duration,
            "max_abs_coefficient": float(np.max(np.abs(track.expression))),
            "mesh_finite": meshes_finite,
            "quality_speech_hangover_added_frames": int(
                np.count_nonzero((quality_activity >= 0.5) & (track.speech_activity < 0.5))
            ),
            "mouth_aperture_range": float(max(apertures) - min(apertures)),
            "neutral_mouth_aperture": neutral_aperture,
            "preview_duration_s": av["duration"],
            "preview_video_duration_s": av["video_duration"],
            "preview_audio_duration_s": av["audio_duration"],
            "preview_video_frames": av["video_frames"],
            "audio_video_offset_frames": offset_frames,
            **conditioning,
            **temporal,
        },
        "artifacts": {
            "controls": "controls.npz",
            "cues": "cues.json",
            "preview": "preview.mp4",
            "timeline": "timeline.json",
            **viewer_artifacts,
            **learned_artifacts,
        },
        "warnings": warnings,
    }
    write_json(output_dir / "result.json", result)
    return result
