from dataclasses import replace
from pathlib import Path
import json
import os
import wave

import numpy as np
import pytest

from autoanim_gnm.animation import (
    LipContactCalibration,
    _activation_matrix,
    _apply_lip_contact_correction,
    _default_prosody,
    _mouth_gap_interocular,
    calibrate_lip_contact,
    compose_animation,
    compose_learned_animation,
    probe_av,
)
from autoanim_gnm.audio import (
    MouthCue,
    analyze_emotion,
    normalize_cues,
)
from autoanim_gnm.audio_pipeline import (
    AUDIO_CAVEAT,
    EMOTION_CAVEAT,
    FALLBACK_CAVEAT,
    LIP_CONTACT_CAVEAT,
    _condition_learned_controls,
    _derive_lip_contact_confidence,
    _fuse_jaw_observation,
    _quality_speech_activity,
    _quarantine_mouth_close_retarget,
    run_audio_pipeline,
)
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.lipsync_quality import evaluate_lipsync_quality
from autoanim_gnm.rig import ControlRig


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
RAVDESS_ANGRY = FIXTURES / "03-01-05-02-01-01-01.wav"
LIBRISPEECH = FIXTURES / "libri-human-speech-8s.wav"
RHUBARB = CACHE / "rhubarb/rhubarb"
A2F_RUNNER = Path("native/a2f-runner/.build/release/a2f-runner")
A2F_ASSETS = CACHE / "a2f-claire"
A2F_READY = A2F_RUNNER.exists() and all(
    (A2F_ASSETS / name).exists()
    for name in (
        "model_data.npz",
        "bs_skin.npz",
        "bs_skin_config.json",
        "bs_tongue.npz",
        "bs_tongue_config.json",
    )
)


def test_cue_normalization_fills_gaps_and_merges() -> None:
    cues = normalize_cues(
        [
            {"start": 0.2, "end": 0.4, "value": "B"},
            {"start": 0.4, "end": 0.6, "value": "B"},
            {"start": 0.8, "end": 1.0, "value": "C"},
        ],
        1.2,
    )
    assert cues == [
        MouthCue(0.0, 0.2, "X"),
        MouthCue(0.2, 0.6, "B"),
        MouthCue(0.6, 0.8, "X"),
        MouthCue(0.8, 1.0, "C"),
        MouthCue(1.0, 1.2, "X"),
    ]
    assert sum(cue.end - cue.start for cue in cues) == pytest.approx(1.2)


def test_cue_normalization_rejects_conflicts() -> None:
    with pytest.raises(AutoAnimError, match="overlap"):
        normalize_cues(
            [
                {"start": 0.0, "end": 0.8, "value": "B"},
                {"start": 0.5, "end": 1.0, "value": "C"},
            ],
            1.0,
        )
    with pytest.raises(AutoAnimError):
        normalize_cues([{"start": 0, "end": 1, "value": "Z"}], 1.0)


def test_silent_audio_and_missing_rhubarb_are_typed(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.zeros(16_000, dtype="<i2").tobytes())
    with pytest.raises(AutoAnimError) as caught:
        analyze_emotion(silent, [MouthCue(0, 1, "X")])
    assert caught.value.code == "AUDIO_SILENT"
    from autoanim_gnm.audio import resolve_rhubarb

    with pytest.raises(AutoAnimError) as caught:
        resolve_rhubarb(tmp_path / "missing-rhubarb")
    assert caught.value.code == "DEPENDENCY_MISSING"

    incomplete = tmp_path / "rhubarb"
    incomplete.touch()
    with pytest.raises(AutoAnimError, match="companion") as caught:
        resolve_rhubarb(incomplete)
    assert caught.value.code == "DEPENDENCY_MISSING"


def test_composer_shapes_regions_and_continuity(rig: ControlRig) -> None:
    cues = [MouthCue(0, 0.5, "X"), MouthCue(0.5, 1.0, "D"), MouthCue(1.0, 1.5, "X")]
    track = compose_animation(cues, 1.5, 30, rig, "joy")
    assert track.expression.shape == (45, 383)
    assert track.rotations.shape == (45, 4, 3)
    assert track.translation.shape == (45, 3)
    assert not np.any(track.expression[:, 382:])
    assert np.max(np.abs(np.diff(track.expression, axis=0))) < 1.5
    assert np.linalg.norm(track.expression[20]) > np.linalg.norm(track.expression[0])


def test_temporal_compiler_weights_rest_and_motion_contract(rig: ControlRig) -> None:
    cues = [MouthCue(0, 0.4, "X"), MouthCue(0.4, 1.1, "D"), MouthCue(1.1, 1.6, "X")]
    timestamps = np.arange(48, dtype=np.float32) / 30.0
    weights = _activation_matrix(cues, timestamps)
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.max(np.count_nonzero(weights > 1e-7, axis=1)) <= 2
    assert weights[23, 4] >= 0.99  # D owns the center of its interval.

    track = compose_animation(cues, 1.6, 30, rig, "joy")
    np.testing.assert_allclose(track.expression[0, 200:382], 0.0, atol=1e-7)
    np.testing.assert_allclose(track.expression[-1, 200:382], 0.0, atol=1e-7)
    np.testing.assert_allclose(track.rotations[[0, -1]], 0.0, atol=1e-7)
    assert np.max(np.abs(np.rad2deg(track.rotations[:, 1, 0]))) <= 2.5
    assert np.max(np.abs(np.rad2deg(track.rotations[:, 1, 1]))) <= 1.5
    assert np.max(np.abs(np.rad2deg(track.rotations[:, 1, 2]))) <= 0.6
    for left, right in zip(track.expression[:-1], track.expression[1:], strict=True):
        assert rig.mouth_step_ratio(left, right) <= 0.04001


def test_temporal_compiler_is_deterministic_and_rejects_empty_cues(rig: ControlRig) -> None:
    cues = [MouthCue(0, 0.35, "X"), MouthCue(0.35, 0.8, "F"), MouthCue(0.8, 1.2, "X")]
    first = compose_animation(cues, 1.2, 30, rig, "neutral")
    second = compose_animation(cues, 1.2, 30, rig, "neutral")
    np.testing.assert_array_equal(first.expression, second.expression)
    np.testing.assert_array_equal(first.rotations, second.rotations)
    with pytest.raises(AutoAnimError, match="mouth cue"):
        compose_animation([], 1.0, 30, rig, "neutral")


def test_learned_conditioner_reduces_jerk_without_erasing_contact() -> None:
    frames = 31
    skin = np.zeros((frames, 4), dtype=np.float32)
    skin[8:14, 0] = np.asarray((0.1, 0.55, 1.0, 0.88, 0.42, 0.08), dtype=np.float32)
    # Deliberate frame-alternating solver chatter on a non-articulation
    # channel. It must be attenuated without setting the bandwidth for lips.
    skin[:, 1] = np.asarray([0.0, 0.8] * 15 + [0.0], dtype=np.float32)
    skin[9:16, 2] = np.asarray((0.05, 0.32, 0.72, 1.0, 0.70, 0.30, 0.04), dtype=np.float32)
    skin[:, 3] = np.linspace(0.0, 0.6, frames, dtype=np.float32)
    tongue = np.column_stack((skin[:, 1], skin[:, 3])).astype(np.float32)
    conditioned_skin, conditioned_tongue = _condition_learned_controls(
        skin,
        ("mouthClose", "cheekPuff", "jawOpen", "mouthSmileLeft"),
        tongue,
    )
    assert conditioned_skin.shape == skin.shape
    assert conditioned_tongue.shape == tongue.shape
    assert np.isfinite(conditioned_skin).all() and np.isfinite(conditioned_tongue).all()
    assert np.max(conditioned_skin[:, 0]) == pytest.approx(np.max(skin[:, 0]))
    assert abs(int(np.argmax(conditioned_skin[:, 0])) - int(np.argmax(skin[:, 0]))) <= 1
    assert np.max(conditioned_skin[:, 2]) >= 0.92 * np.max(skin[:, 2])
    raw_jerk = np.linalg.norm(np.diff(skin[:, [1, 3]], n=3, axis=0), axis=1)
    conditioned_jerk = np.linalg.norm(
        np.diff(conditioned_skin[:, [1, 3]], n=3, axis=0), axis=1
    )
    jerk_ratio = np.percentile(conditioned_jerk, 95) / np.percentile(raw_jerk, 95)
    assert 0.10 < jerk_ratio < 0.40


def test_raw_jaw_observation_softly_restores_opening_without_exceeding_bounds() -> None:
    values = np.zeros((8, 2), dtype=np.float32)
    values[:, 0] = np.linspace(0.0, 0.25, 8)
    rotations = np.zeros((8, 3), dtype=np.float32)
    rotations[:, 0] = (0.0, 0.0, 0.0, 3.0, 7.0, 12.0, 18.0, 4.0)
    activity = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0), dtype=np.float32)

    fused, metrics = _fuse_jaw_observation(
        values,
        ("jawOpen", "mouthPucker"),
        rotations,
        activity,
    )

    assert fused[6, 0] == pytest.approx(0.55)
    assert np.all(fused >= values)
    assert np.max(fused) <= 1.0
    assert metrics["jaw_observation_rotation_range_degrees"] == pytest.approx(18.0)
    assert metrics["jaw_observation_fused_frames"] >= 1


def test_lip_contact_confidence_requires_close_lips_closed_jaw_and_speech() -> None:
    names = (
        "mouthClose",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthRollLower",
        "mouthRollUpper",
        "jawOpen",
    )
    values = np.zeros((6, len(names)), dtype=np.float32)
    values[1, 0] = 0.55  # learned mouth-close evidence
    values[2, 1:3] = 0.80  # press alone is tension, not phone evidence
    values[3, 0] = 0.55
    values[3, 3:5] = 0.75  # roll may reinforce mouth-close evidence
    values[4, 0] = 0.60
    values[4, 5] = 0.55  # jaw too open for bilabial contact
    values[5, 0] = 0.60  # quiet/rest bias must not create a seal
    activity = np.asarray((0.0, 1.0, 1.0, 1.0, 1.0, 0.0), dtype=np.float32)

    confidence = _derive_lip_contact_confidence(values, names, activity)

    assert confidence.shape == (6,)
    assert np.all((0.0 <= confidence) & (confidence <= 1.0))
    assert confidence[1] > 0.80
    assert confidence[2] == 0.0
    assert confidence[3] > 0.90
    assert 0.0 < confidence[4] < 0.35
    assert confidence[5] == 0.0


def test_mouth_close_is_evidence_only_not_an_opening_retarget_row() -> None:
    values = np.asarray(
        (
            (0.0, 0.1, 0.2),
            (0.6, 0.3, 0.4),
            (0.8, 0.5, 0.6),
        ),
        dtype=np.float32,
    )
    names = ("mouthClose", "jawOpen", "mouthPressLeft")

    retarget_values, metrics = _quarantine_mouth_close_retarget(values, names)

    np.testing.assert_array_equal(
        values[:, 0],
        np.asarray((0.0, 0.6, 0.8), dtype=np.float32),
    )
    np.testing.assert_array_equal(retarget_values[:, 0], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(retarget_values[:, 1:], values[:, 1:])
    assert metrics["mouth_close_quarantined_peak"] == pytest.approx(0.8)
    assert metrics["mouth_close_quarantined_frames"] == 2


def test_spatial_lip_contact_calibration_is_local_and_character_specific(
    rig: ControlRig,
) -> None:
    calibration = calibrate_lip_contact(rig)
    assert isinstance(calibration, LipContactCalibration)
    assert calibration.direction.shape == (383,)
    assert not np.any(calibration.direction[:200])
    assert not np.any(calibration.direction[350:])
    assert 0.0 < calibration.seal_gap_interocular < calibration.neutral_gap_interocular
    assert 0.0 < calibration.maximum_alpha <= 2.0
    assert calibration.nonmouth_p95_displacement_interocular < 5e-4
    assert calibration.nonmouth_max_displacement_interocular < 1e-3
    assert len(calibration.calibration_hash) == 64

    neutral = rig.adapter.compact_template
    sealed = rig.compact_landmarks(
        np.float32(calibration.maximum_alpha) * calibration.direction
    )
    interocular = np.linalg.norm(neutral[36] - neutral[45])
    nonmouth = np.linalg.norm(sealed[:48] - neutral[:48], axis=1) / interocular
    assert np.max(nonmouth) < 5e-4


def test_soft_lip_contact_correction_closes_without_overapplying(rig: ControlRig) -> None:
    opened = np.float32(0.20) * rig.viseme("D")
    calibration = calibrate_lip_contact(rig)
    before_gap = _mouth_gap_interocular(rig, opened)

    unchanged, applied, target = _apply_lip_contact_correction(
        rig,
        opened,
        calibration,
        0.05,
    )
    np.testing.assert_array_equal(unchanged, opened)
    assert not applied
    assert target == 0.0

    corrected, applied, target = _apply_lip_contact_correction(
        rig,
        opened,
        calibration,
        0.80,
    )
    corrected_gap = _mouth_gap_interocular(rig, corrected)
    assert applied
    assert corrected_gap < 0.35 * before_gap
    assert corrected_gap <= target + 2e-4
    assert np.isfinite(corrected).all()
    assert np.max(np.abs(corrected)) <= 3.0
    np.testing.assert_array_equal(corrected[:200], opened[:200])
    np.testing.assert_array_equal(corrected[350:], opened[350:])

    unreachable_calibration = replace(
        calibration,
        direction=np.zeros(383, dtype=np.float32),
        inner_response=np.zeros((150, 18), dtype=np.float32),
        calibration_hash="0" * 64,
    )
    unreachable, applied, target = _apply_lip_contact_correction(
        rig,
        opened,
        unreachable_calibration,
        1.0,
    )
    np.testing.assert_array_equal(unreachable, opened)
    assert not applied
    assert target > 0.0


def test_procedural_composer_retains_character_contact_calibration(
    rig: ControlRig,
) -> None:
    identity = np.zeros(rig.adapter.identity_dim, dtype=np.float32)
    identity[:32] = np.linspace(-0.25, 0.25, 32, dtype=np.float32)
    character_rig = ControlRig(rig.adapter, rig.decoder, identity=identity)
    calibration = calibrate_lip_contact(character_rig)
    cues = [
        MouthCue(0.0, 0.50, "X"),
        MouthCue(0.50, 0.80, "A"),
        MouthCue(0.80, 1.70, "X"),
    ]
    track = compose_animation(
        cues,
        1.70,
        30,
        character_rig,
        "neutral",
        lip_contact_calibration=calibration,
    )

    candidates = track.lip_contact_target_gap > 0.0
    assert np.count_nonzero(candidates) > 0
    assert np.max(track.lip_contact_confidence) > 0.9
    assert np.min(track.lip_contact_target_gap[candidates]) <= (
        calibration.seal_gap_interocular + 1.0e-3
    )
    assert np.count_nonzero(track.contact_correction_applied) > 0
    assert np.count_nonzero(track.contact_corrected) > 0
    candidates = track.lip_contact_target_gap > 0.0
    assert np.mean(track.lip_contact_attained[candidates]) >= 0.75
    assert not np.any(
        track.contact_correction_applied
        & ~track.lip_contact_attained
        & track.contact_continuity_restored
    )


def test_learned_composer_preserves_fast_articulation_and_adds_secondary_motion(
    rig: ControlRig,
) -> None:
    fps = 30
    duration = 4.2
    frame_count = int(np.ceil(duration * fps))
    timestamps = np.arange(frame_count, dtype=np.float32) / fps
    cues = [
        MouthCue(0.0, 0.45, "X"),
        MouthCue(0.45, 3.75, "D"),
        MouthCue(3.75, duration, "X"),
    ]
    prosody = _default_prosody(cues, timestamps)
    source = np.zeros((frame_count, 383), dtype=np.float32)
    open_pose = rig.viseme("D")
    phase = np.linspace(0.0, 12.0 * np.pi, frame_count)
    source[:, 200:350] = (
        (0.50 + 0.50 * np.sin(phase))[:, None] * open_pose[None, 200:350]
    )
    affect = np.repeat(rig.emotion("anger")[None], frame_count, axis=0)
    eyes = np.zeros((frame_count, 2, 2), dtype=np.float32)
    eyes[60, 0, 0] = 4.0  # Claire right eye X
    eyes[60, 1, 1] = -3.0  # Claire left eye Y

    track = compose_learned_animation(
        source,
        timestamps,
        cues,
        duration,
        fps,
        rig,
        prosody,
        acting_strength=0.75,
        emotion_delta=affect,
        source_eye_rotations_degrees=eyes,
    )
    no_eye_track = compose_learned_animation(
        source,
        timestamps,
        cues,
        duration,
        fps,
        rig,
        prosody,
        acting_strength=0.75,
        emotion_delta=affect,
    )

    quality = evaluate_lipsync_quality(
        np.stack([rig.compact_landmarks(frame) for frame in track.expression]),
        rig.compact_landmarks(np.zeros(383, dtype=np.float32)),
        track.speech_activity,
        fps=fps,
    )
    assert quality.metrics["mouth_step_max_interocular"] <= 0.040001
    assert np.count_nonzero(track.mouth_speed_limited) > 0
    assert np.ptp(track.expression[:, :200], axis=0).max() > 0.1
    assert np.max(np.abs(np.rad2deg(track.rotations[:, 1, :2]))) > 1.0
    assert np.max(np.abs(track.rotations[:, 2:4])) > 0.005
    np.testing.assert_allclose(
        np.rad2deg(track.rotations[60, 3, 0] - no_eye_track.rotations[60, 3, 0]),
        4.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.rad2deg(track.rotations[60, 2, 1] - no_eye_track.rotations[60, 2, 1]),
        -3.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(track.rotations[[0, -1]], 0.0, atol=1e-7)


def test_quality_speech_activity_adds_hangover_without_erasing_long_silence() -> None:
    activity = np.zeros(40, dtype=np.float32)
    activity[8:15] = 1.0
    activity[20:28] = 1.0

    quality_activity = _quality_speech_activity(activity, hangover_frames=2)

    np.testing.assert_array_equal(quality_activity[6:17], np.ones(11, dtype=np.float32))
    np.testing.assert_array_equal(quality_activity[18:30], np.ones(12, dtype=np.float32))
    assert not np.any(quality_activity[:6])
    assert not np.any(quality_activity[30:])


def test_learned_composer_applies_contact_before_continuity_guard(
    rig: ControlRig,
) -> None:
    fps = 30
    duration = 2.0
    frame_count = int(duration * fps)
    timestamps = np.arange(frame_count, dtype=np.float32) / fps
    cues = [
        MouthCue(0.0, 0.30, "X"),
        MouthCue(0.30, 0.90, "D"),
        MouthCue(0.90, 1.10, "A"),
        MouthCue(1.10, 1.70, "D"),
        MouthCue(1.70, duration, "X"),
    ]
    prosody = _default_prosody(cues, timestamps)
    source = np.zeros((frame_count, 383), dtype=np.float32)
    source[9:51] = np.float32(0.20) * rig.viseme("D")
    confidence = np.zeros(frame_count, dtype=np.float32)
    confidence[27:34] = np.asarray(
        (0.15, 0.35, 0.65, 1.0, 0.65, 0.35, 0.15),
        dtype=np.float32,
    )
    calibration = calibrate_lip_contact(rig)

    corrected = compose_learned_animation(
        source,
        timestamps,
        cues,
        duration,
        fps,
        rig,
        prosody,
        source_lip_contact_confidence=confidence,
        lip_contact_calibration=calibration,
    )
    baseline = compose_learned_animation(
        source,
        timestamps,
        cues,
        duration,
        fps,
        rig,
        prosody,
    )
    no_phone_cues = [
        MouthCue(0.0, 0.30, "X"),
        MouthCue(0.30, 1.70, "D"),
        MouthCue(1.70, duration, "X"),
    ]
    no_phone_prosody = _default_prosody(no_phone_cues, timestamps)
    no_phone_gate = compose_learned_animation(
        source,
        timestamps,
        no_phone_cues,
        duration,
        fps,
        rig,
        no_phone_prosody,
        source_lip_contact_confidence=confidence,
        lip_contact_calibration=calibration,
    )

    assert corrected.contact_corrected[30]
    assert corrected.contact_correction_applied[30]
    assert corrected.lip_contact_attained[30]
    assert corrected.lip_contact_target_gap[30] > 0.0
    assert np.count_nonzero(corrected.contact_corrected) <= 7
    assert corrected.lip_contact_confidence[30] >= confidence[30]
    assert _mouth_gap_interocular(rig, corrected.expression[30]) < _mouth_gap_interocular(
        rig,
        baseline.expression[30],
    )
    steps = np.asarray(
        [
            rig.mouth_step_ratio(left, right)
            for left, right in zip(
                corrected.expression[:-1],
                corrected.expression[1:],
                strict=True,
            )
        ]
    )
    assert np.max(steps) <= 0.06001
    assert np.isfinite(corrected.expression).all()
    assert not np.any(no_phone_gate.contact_corrected)
    np.testing.assert_array_equal(
        no_phone_gate.lip_contact_confidence,
        np.zeros(frame_count, dtype=np.float32),
    )


def test_contact_aware_projection_redistributes_approach_instead_of_opening_seal(
    rig: ControlRig,
) -> None:
    fps = 30
    duration = 2.0
    frame_count = int(duration * fps)
    timestamps = np.arange(frame_count, dtype=np.float32) / fps
    cues = [
        MouthCue(0.0, 0.20, "X"),
        MouthCue(0.20, 0.90, "D"),
        MouthCue(0.90, 1.08, "A"),
        MouthCue(1.08, 1.80, "D"),
        MouthCue(1.80, duration, "X"),
    ]
    prosody = _default_prosody(cues, timestamps)
    source = np.zeros((frame_count, 383), dtype=np.float32)
    source[6:54] = np.float32(0.20) * rig.viseme("D")
    # The learned target closes naturally immediately after the contact, but
    # the approach frame is too far away for a one-frame 0.039 transition.
    # A contact-oblivious limiter opens the seal at frame 30. A contact-aware
    # projection must instead start the approach on frame 29.
    source[31:36] = 0.0
    confidence = np.zeros(frame_count, dtype=np.float32)
    confidence[30] = 1.0

    track = compose_learned_animation(
        source,
        timestamps,
        cues,
        duration,
        fps,
        rig,
        prosody,
        source_lip_contact_confidence=confidence,
        lip_contact_calibration=calibrate_lip_contact(rig),
    )
    quality = evaluate_lipsync_quality(
        np.stack([rig.compact_landmarks(frame) for frame in track.expression]),
        rig.compact_landmarks(np.zeros(383, dtype=np.float32)),
        track.speech_activity,
        fps=fps,
    )

    assert track.contact_correction_applied[30]
    assert track.lip_contact_attained[30]
    assert track.contact_continuity_restored[30]
    assert track.contact_corrected[30]
    assert not track.mouth_speed_limited[30]
    assert track.mouth_speed_limited[29]
    assert quality.metrics["mouth_step_max_interocular"] <= 0.040001


def test_final_contact_status_does_not_overclaim_unreachable_open_pose(
    rig: ControlRig,
) -> None:
    fps = 30
    duration = 2.0
    frame_count = int(duration * fps)
    timestamps = np.arange(frame_count, dtype=np.float32) / fps
    cues = [
        MouthCue(0.0, 0.30, "X"),
        MouthCue(0.30, 0.90, "D"),
        MouthCue(0.90, 1.10, "A"),
        MouthCue(1.10, 1.70, "D"),
        MouthCue(1.70, duration, "X"),
    ]
    prosody = _default_prosody(cues, timestamps)
    source = np.zeros((frame_count, 383), dtype=np.float32)
    source[9:51] = np.float32(0.40) * rig.viseme("D")
    confidence = np.zeros(frame_count, dtype=np.float32)
    confidence[30] = 1.0

    track = compose_learned_animation(
        source,
        timestamps,
        cues,
        duration,
        fps,
        rig,
        prosody,
        source_lip_contact_confidence=confidence,
        lip_contact_calibration=calibrate_lip_contact(rig),
    )

    assert track.contact_correction_applied[30]
    assert track.lip_contact_target_gap[30] > 0.0
    assert not track.lip_contact_attained[30]
    assert not track.contact_continuity_restored[30]
    assert not track.contact_corrected[30]
    quality = evaluate_lipsync_quality(
        np.stack([rig.compact_landmarks(frame) for frame in track.expression]),
        rig.compact_landmarks(np.zeros(383, dtype=np.float32)),
        track.speech_activity,
        fps=fps,
    )
    assert quality.metrics["mouth_step_max_interocular"] <= 0.040001


@pytest.mark.skipif(not RAVDESS_ANGRY.exists(), reason="official RAVDESS fixture not downloaded")
def test_real_emotional_audio_is_labeled_but_not_overclaimed(tmp_path: Path) -> None:
    from autoanim_gnm.audio import normalize_audio, run_rhubarb

    normalized = tmp_path / "normalized.wav"
    duration = normalize_audio(RAVDESS_ANGRY, normalized)
    raw = run_rhubarb(normalized, tmp_path / "raw.json", rhubarb_bin=RHUBARB)
    cues = normalize_cues(raw, duration)
    automatic = analyze_emotion(normalized, cues)
    assert automatic.emotion == "anger"
    assert automatic.confidence < 0.65
    assert not automatic.validated
    manual = analyze_emotion(normalized, cues, manual="anger")
    assert manual.emotion == "anger"
    assert manual.confidence == 1.0
    assert manual.validated


@pytest.mark.skipif(not RAVDESS_ANGRY.exists() or not RHUBARB.exists(), reason="real emotional E2E fixture unavailable")
def test_real_ravdess_angry_end_to_end(tmp_path: Path) -> None:
    result = run_audio_pipeline(
        RAVDESS_ANGRY, tmp_path, rhubarb_bin=RHUBARB, backend="fallback"
    )
    assert result["analysis"]["emotion"] == "anger"
    assert result["analysis"]["emotion_confidence"] == pytest.approx(0.62)
    assert not result["analysis"]["emotion_validated"]
    assert result["metrics"]["cue_coverage"] == pytest.approx(1.0)
    assert result["metrics"]["mouth_aperture_range"] > 0.005
    assert result["metrics"]["audio_video_offset_frames"] <= 1.0
    assert "COEFFICIENT_SATURATED" not in result["warnings"]
    assert result["viewer"]["glb_covers_full_track"] is True
    assert result["oral_validation"]["all_control_frames_evaluated"] is True
    assert result["oral_validation"]["viewer_structural_reconstruction_validated"] is True
    assert not any(
        "ORAL_GLB_NOT_STRUCTURALLY_VALIDATED" in warning
        for warning in result["warnings"]
    )
    oral_report = json.loads((tmp_path / "oral-validation.json").read_text(encoding="utf-8"))
    assert oral_report["source"]["evaluation_mode"] == "provided_complete_gnm_frames"
    glb_report = json.loads(
        (tmp_path / "oral-glb-validation.json").read_text(encoding="utf-8")
    )
    assert glb_report["structural_reconstruction"]["reference_evaluation_mode"] == (
        "provided_complete_gnm_frames"
    )
    av = probe_av(tmp_path / "preview.mp4")
    assert av["has_audio"] and av["has_video"]
    assert av["video_frames"] == result["animation"]["frames"]


@pytest.mark.skipif(not LIBRISPEECH.exists() or not RHUBARB.exists(), reason="real E2E fixtures unavailable")
def test_real_librispeech_end_to_end(tmp_path: Path) -> None:
    result = run_audio_pipeline(
        LIBRISPEECH, tmp_path, rhubarb_bin=RHUBARB, backend="fallback"
    )
    assert result["status"] == "succeeded"
    assert result["analysis"]["backend"] == "procedural-v2+rhubarb-1.14.0"
    assert result["analysis"]["motion_backend"] == "procedural_fallback"
    assert not result["animation"]["production_validated"]
    assert result["analysis"]["emotion"] == "neutral"
    assert not result["analysis"]["emotion_validated"]
    assert len(result["analysis"]["cues"]) >= 20
    assert result["metrics"]["cue_coverage"] == pytest.approx(1.0)
    assert result["metrics"]["mesh_finite"]
    assert result["metrics"]["mouth_aperture_range"] > 0.005
    assert result["metrics"]["max_abs_coefficient"] <= 3.0
    assert result["metrics"]["audio_video_offset_frames"] <= 1.0
    assert AUDIO_CAVEAT in result["warnings"]
    assert EMOTION_CAVEAT in result["warnings"]
    assert FALLBACK_CAVEAT in result["warnings"]
    assert result["metrics"]["mouth_step_max_interocular"] <= 0.04001
    assert result["metrics"]["lower_face_stationary_fraction"] < 0.08
    assert result["metrics"]["head_rotation_max_degrees"] > 0.1
    assert "COEFFICIENT_SATURATED" not in result["warnings"]
    with np.load(tmp_path / "controls.npz", allow_pickle=False) as controls:
        assert controls["expression"].shape == (240, 383)
        assert not np.any(controls["expression"][:, 382:])
    av = probe_av(tmp_path / "preview.mp4")
    assert av["has_audio"] and av["has_video"]
    assert av["video_frames"] == result["animation"]["frames"]


@pytest.mark.skipif(
    not LIBRISPEECH.exists() or not RHUBARB.exists() or not A2F_READY,
    reason="real learned E2E dependencies unavailable",
)
def test_real_librispeech_learned_end_to_end(tmp_path: Path) -> None:
    result = run_audio_pipeline(
        LIBRISPEECH,
        tmp_path,
        rhubarb_bin=RHUBARB,
        backend="learned",
        a2f_runner=A2F_RUNNER,
        a2f_asset_dir=A2F_ASSETS,
        a2f_offline=True,
        emotion="neutral",
    )
    assert result["analysis"]["motion_backend"] == "learned_a2f"
    assert result["analysis"]["retargeter"] == "geometry_calibrated_dense_v3_spatial_contact"
    assert len(result["analysis"]["retarget_calibration_hash"]) == 64
    assert result["analysis"]["emotion_applied"] == "neutral"
    assert not result["animation"]["production_validated"]
    assert not result["quality"]["production_gate"]["passed"]
    assert "independent_annotations" in result["quality"]["production_gate"]["failures"]
    assert result["metrics"]["lower_face_stationary_fraction"] < 0.01
    assert result["metrics"]["mouth_step_p95_interocular"] < 0.045
    assert 0.25 < result["metrics"]["conditioning_noncontact_jerk_p95_ratio"] < 0.70
    assert result["metrics"]["conditioning_contact_peak_retention_min"] >= 0.99
    assert result["metrics"]["conditioning_articulation_range_retention_min"] >= 0.78
    assert result["metrics"]["conditioning_rank95_retention"] >= 0.80
    assert result["quality"]["metrics"]["mouth_step_max_interocular"] <= 0.040001
    assert result["metrics"]["head_rotation_max_degrees"] > 1.0
    assert result["metrics"]["eye_rotation_max_degrees"] > 0.5
    assert result["metrics"]["mouth_aperture_range"] > 0.006
    assert result["metrics"]["lip_contact_corrected_fraction"] < 0.20
    assert LIP_CONTACT_CAVEAT in result["warnings"]
    assert result["artifacts"]["a2f_raw"] == "a2f_raw.jsonl"
    assert result["artifacts"]["arkit_controls"] == "arkit_controls.npz"
    assert result["artifacts"]["retarget_calibration"] == "retarget_calibration.npz"
    assert (tmp_path / "retarget_calibration.npz").is_file()
    with np.load(tmp_path / "arkit_controls.npz", allow_pickle=False) as values:
        assert values["skin_weights"].shape == (241, 52)
        assert values["tongue_weights"].shape == (241, 16)
        assert values["jaw_rotation_vectors_degrees"].shape == (241, 3)
        assert values["eye_rotations_degrees"].shape == (241, 2, 2)
        assert values["source_lip_contact_confidence"].shape == (241,)
        assert values["gnm_lip_contact_direction"].shape == (383,)
        assert values["gnm_lip_contact_inner_response"].shape == (150, 18)
        assert values["gnm_lip_contact_neutral_pair_gaps_interocular"].shape == (3,)
        assert values["gnm_lip_contact_seal_pair_gaps_interocular"].shape == (3,)
        assert values["gnm_lip_contact_neutral_gap_interocular"].shape == ()
        assert values["gnm_lip_contact_seal_gap_interocular"].shape == ()
        assert values["gnm_lip_contact_maximum_alpha"].shape == ()
        assert values["gnm_lip_contact_calibration_hash"].shape == ()
        assert str(values["gnm_lip_contact_calibration_hash"]) != ""
        close_index = values["skin_pose_names"].tolist().index("mouthClose")
        assert np.max(values["conditioned_skin_weights"][:, close_index]) > 0.0
        np.testing.assert_array_equal(
            values["retarget_skin_weights"][:, close_index],
            np.zeros(241, dtype=np.float32),
        )
        assert np.isfinite(values["skin_weights"]).all()
        assert np.isfinite(values["tongue_weights"]).all()
        assert np.ptp(values["jaw_rotation_vectors_degrees"][:, 0]) > 5.0
    with np.load(tmp_path / "controls.npz", allow_pickle=False) as controls:
        np.testing.assert_allclose(controls["expression"][[0, -1], 200:382], 0.0, atol=1e-6)
        assert controls["lip_contact_confidence"].shape == (240,)
        assert controls["lip_contact_target_gap"].shape == (240,)
        assert controls["contact_correction_applied"].shape == (240,)
        assert controls["lip_contact_attained"].shape == (240,)
        assert controls["contact_continuity_restored"].shape == (240,)
        assert controls["contact_corrected"].shape == (240,)
    timeline = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["motion_backend"] == "learned_a2f"
    assert len(timeline["mouth_aperture"]) == 240
    av = probe_av(tmp_path / "preview.mp4")
    assert av["has_audio"] and av["has_video"]
    assert av["video_frames"] == 240


@pytest.mark.skipif(
    not RAVDESS_ANGRY.exists() or not RHUBARB.exists() or not A2F_READY,
    reason="real learned emotional E2E dependencies unavailable",
)
def test_real_ravdess_learned_emotion_end_to_end(tmp_path: Path) -> None:
    result = run_audio_pipeline(
        RAVDESS_ANGRY,
        tmp_path,
        rhubarb_bin=RHUBARB,
        backend="learned",
        a2f_runner=A2F_RUNNER,
        a2f_asset_dir=A2F_ASSETS,
        a2f_offline=True,
        emotion="anger",
        emotion_strength=0.65,
        dialog="Kids are talking by the door",
    )
    assert result["analysis"]["motion_backend"] == "learned_a2f"
    assert result["analysis"]["retargeter"] == "geometry_calibrated_dense_v3_spatial_contact"
    assert result["analysis"]["emotion_applied"] == "anger"
    assert result["analysis"]["emotion_strength"] == pytest.approx(0.65)
    assert result["analysis"]["emotion_validated"]
    assert result["metrics"]["lower_face_stationary_fraction"] < 0.01
    assert result["metrics"]["mouth_step_p95_interocular"] < 0.045
    assert result["quality"]["metrics"]["mouth_step_max_interocular"] <= 0.040001
    assert 0.25 < result["metrics"]["conditioning_noncontact_jerk_p95_ratio"] < 0.70
    assert result["metrics"]["conditioning_contact_peak_retention_min"] >= 0.99
    assert result["metrics"]["conditioning_articulation_range_retention_min"] >= 0.80
    assert result["metrics"]["conditioning_rank95_retention"] >= 0.80
    assert result["metrics"]["emotion_intensity_range"] > 0.35
    assert result["metrics"]["head_rotation_max_degrees"] > 1.0
    assert result["metrics"]["eye_rotation_max_degrees"] > 0.5
    assert result["metrics"]["upper_face_control_range_max"] > 0.5
    assert result["metrics"]["jaw_observation_rotation_range_degrees"] > 5.0
    assert result["metrics"]["jaw_observation_fused_frames"] > 0
    assert result["metrics"]["lip_contact_candidate_frames"] > 0
    assert result["metrics"]["lip_contact_corrected_frames"] > 0
    assert result["metrics"]["lip_contact_continuity_restored_frames"] >= 0
    assert result["metrics"]["lip_contact_corrected_fraction"] < 0.20
    assert result["metrics"]["lip_contact_target_attainment_fraction"] >= 0.75
    assert result["metrics"]["lip_contact_post_limiter_lost_frames"] == 0
    assert result["metrics"]["lip_contact_post_limiter_attainment_fraction"] >= 0.75
    assert (
        result["metrics"]["lip_contact_calibration_nonmouth_max_displacement_interocular"]
        < 1e-3
    )
    assert result["artifacts"]["a2f_emotion_raw"] == "a2f_emotion_raw.jsonl"
    with np.load(tmp_path / "arkit_controls.npz", allow_pickle=False) as values:
        assert values["gnm_emotion_delta"].shape == (
            values["skin_weights"].shape[0],
            383,
        )
        assert np.max(np.abs(values["gnm_emotion_delta"][:, :200])) > 0.1
    assert not result["quality"]["production_gate"]["passed"]
    assert result["metrics"]["audio_video_offset_frames"] <= 1.0
