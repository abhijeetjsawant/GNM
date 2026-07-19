from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import os
import shutil

import numpy as np
import pytest

import autoanim_gnm.video_pipeline as video_pipeline_module
from autoanim_gnm.audio_visual_repair import (
    AUDIO_VISUAL_REPAIR_SCHEMA_VERSION,
    AudioVisualRepairConfig,
    _limit_introduced_mouth_steps,
    apply_audio_visual_repair,
)
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.serialization import write_npz
from autoanim_gnm.video_retarget import (
    GNMPerformanceTrack,
    PerformanceProvenance,
    TemporalFilterConfig,
)
from autoanim_gnm.video_pipeline import run_video_pipeline


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
A2F_ASSETS = CACHE / "a2f-claire"
A2F_RUNNER = Path("native/a2f-runner/.build/release/a2f-runner")
RHUBARB = CACHE / "rhubarb/rhubarb"
CREMA_D_ANGRY = FIXTURES / "crema-d-1001-dfa-ang.flv"


def _performance(*, count: int = 6) -> GNMPerformanceTrack:
    expression = np.zeros((count, 383), dtype=np.float32)
    expression[:, :200] = np.arange(count, dtype=np.float32)[:, None] / 10.0
    expression[:, 382] = np.arange(count, dtype=np.float32) / 20.0
    quality = np.ones(count, dtype=np.float32)
    quality[2:4] = 0.20
    source_valid = np.ones(count, dtype=bool)
    source_valid[3] = False
    source_gap = np.full(count, 0.08, dtype=np.float32)
    source_contact = np.zeros(count, dtype=np.float32)
    source_contact[4] = 0.95
    provenance = PerformanceProvenance(
        capture_schema_version="capture-test",
        capture_source_sha256="1" * 64,
        retargeter="test",
        filter_config=TemporalFilterConfig(),
        transform_convention="test",
        translation_scale_to_gnm=1.0,
        coordinate_conversion=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        eye_range_radians=0.5,
        baseline_frame_indices=(0,),
        neutral_blendshape_baseline=(),
        neutral_baseline_method="test",
        neutral_baseline_validated=False,
        neutral_baseline_correction_applied=False,
        neutral_baseline_score=0.0,
        neutral_baseline_score_limit=1.0,
        neutral_baseline_semantic_peak=0.0,
        neutral_baseline_ambiguity_controls=(),
        quarantined_expression_controls=(),
        contact_source_method="test_geometry",
        contact_calibration_hash=None,
        aperture_source_method="test_geometry",
        negative_baseline_residual_clipped_fraction=0.0,
        caveats=(),
    )
    return GNMPerformanceTrack(
        identity=np.zeros(253, dtype=np.float32),
        expression=expression,
        rotations=np.arange(count * 12, dtype=np.float32).reshape(count, 4, 3) / 100.0,
        translation=np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 100.0,
        timestamps_seconds=np.arange(count, dtype=np.float64) / 10.0,
        source_pts=np.arange(count, dtype=np.int64) * 100,
        detected=np.ones(count, dtype=bool),
        effective_quality=quality,
        source_lip_geometry_valid=source_valid,
        source_lip_gap_interocular=source_gap,
        source_lip_contact_confidence=source_contact,
        lip_contact_target_gap_interocular=np.zeros(count, dtype=np.float32),
        contact_correction_applied=np.zeros(count, dtype=bool),
        lip_contact_attained=np.zeros(count, dtype=bool),
        lip_aperture_target_gap_interocular=np.zeros(count, dtype=np.float32),
        lip_aperture_correction_applied=np.zeros(count, dtype=bool),
        lip_aperture_target_attained=np.zeros(count, dtype=bool),
        provenance=provenance,
    )


def _controls(path: Path, *, count: int = 6, contact: bool = True) -> Path:
    expression = np.zeros((count, 383), dtype=np.float32)
    expression[:, 200:350] = np.arange(1, count + 1, dtype=np.float32)[:, None] / 10.0
    expression[:, 350:382] = np.arange(1, count + 1, dtype=np.float32)[:, None] / 20.0
    contact_confidence = np.zeros(count, dtype=np.float32)
    if contact:
        contact_confidence[1] = 0.80
    return write_npz(
        path,
        expression=expression,
        timestamps=np.arange(count, dtype=np.float32) / 10.0,
        fps=np.asarray(10, dtype=np.int32),
        speech_activity=np.ones(count, dtype=np.float32),
        lip_contact_confidence=contact_confidence,
        lip_contact_target_gap=np.full(count, 0.01, dtype=np.float32),
        contact_correction_applied=np.ones(count, dtype=bool),
        lip_contact_attained=np.ones(count, dtype=bool),
    )


def _timing(performance: GNMPerformanceTrack) -> dict:
    frames = [
        {
            "sourcePTS": int(source_pts),
            "coveringAudioSampleSpan": [index * 1_600, (index + 1) * 1_600],
            "hasAudioSampleCoverage": True,
            "displayStartExactRational": [index, 10],
        }
        for index, source_pts in enumerate(performance.source_pts)
    ]
    return {
        "schemaVersion": "autoanim.audio-video-timing.v1",
        "policy": "observation_only_no_motion_effect",
        "status": "available_observation",
        "source": {"sha256": performance.provenance.capture_source_sha256},
        "fusionGate": {"exactClockJoinAvailable": True},
        "audioVideoJoin": {
            "video": {"frames": frames},
            "audio": {
                "sampleRate": 16_000,
                "monoPcm": {"sha256": "2" * 64},
                "decodedStartExactRational": [0, 1],
                "decodedEndExactRational": [3, 5],
            },
        },
    }


def _audio_result(*, learned: bool = True) -> dict:
    return {
        "analysis": {
            "motion_backend": "learned_a2f" if learned else "procedural_fallback"
        }
    }


def test_video_authority_and_audio_repair_are_explicitly_partitioned(
    tmp_path: Path,
) -> None:
    performance = _performance()
    result = apply_audio_visual_repair(
        performance,
        audio_controls_path=_controls(tmp_path / "controls.npz"),
        audio_result=_audio_result(),
        timing_evidence=_timing(performance),
        output_dir=tmp_path,
        config=AudioVisualRepairConfig(taper_frames=0),
    )

    revised = result.performance
    # Reliable video mouth frames are unchanged, including the explicit
    # visible-contact frame. Weak frames receive the learned lower face.
    np.testing.assert_array_equal(
        revised.expression[[0, 1, 4, 5], 200:350],
        performance.expression[[0, 1, 4, 5], 200:350],
    )
    assert np.all(result.lower_face_audio_weight[2:4] > 0.0)
    assert np.max(np.abs(revised.expression[2:4, 200:350])) > 0.0
    # Dedicated tongue is audio-driven during speech because the RGB capture
    # has no independent tongue channel.
    assert np.all(result.tongue_audio_weight == 1.0)
    assert np.max(np.abs(revised.expression[:, 350:382])) > 0.0

    np.testing.assert_array_equal(revised.expression[:, :200], performance.expression[:, :200])
    np.testing.assert_array_equal(revised.expression[:, 382:], performance.expression[:, 382:])
    np.testing.assert_array_equal(revised.rotations, performance.rotations)
    np.testing.assert_array_equal(revised.translation, performance.translation)
    np.testing.assert_array_equal(revised.source_pts, performance.source_pts)
    np.testing.assert_array_equal(
        revised.timestamps_seconds, performance.timestamps_seconds
    )

    # Frame 1 has visibly open lips but an audio closure. It is recorded as a
    # conflict and the video lower face stays byte-exact.
    assert result.audio_visual_contact_conflict[1]
    np.testing.assert_array_equal(
        revised.expression[1, 200:350], performance.expression[1, 200:350]
    )
    assert result.report["schemaVersion"] == AUDIO_VISUAL_REPAIR_SCHEMA_VERSION
    assert result.report["metrics"]["audioVisualContactConflictFrames"] >= 1
    assert result.report["locks"] == {
        "upperFaceExact": True,
        "pupilExact": True,
        "headPoseAndTranslationExact": True,
        "sourcePtsAndTimestampsExact": True,
        "trustedVisualContactDisagreementIsDiagnostic": True,
        "visibleContactProtectedByVisualOwnership": True,
        "mouthContinuityGeometryValidated": False,
        "tongueCoefficientContinuityValidated": True,
    }
    assert result.report["claims"]["productionValidated"] is False
    assert (tmp_path / "audio-visual-repair.json").is_file()
    with np.load(tmp_path / "audio-visual-repair.npz", allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact["source_pts"], performance.source_pts)
        np.testing.assert_array_equal(artifact["output_expression"], revised.expression)
        assert "visual_source_lip_gap_interocular" in artifact.files
        assert "audio_contact_target_gap_interocular" in artifact.files
        assert "final_lip_gap_interocular" in artifact.files


def test_exact_sample_join_handles_partial_coverage_without_time_warp(
    tmp_path: Path,
) -> None:
    performance = _performance()
    timing = _timing(performance)
    timing["audioVideoJoin"]["video"]["frames"][3][
        "hasAudioSampleCoverage"
    ] = False
    timing["audioVideoJoin"]["video"]["frames"][3][
        "coveringAudioSampleSpan"
    ] = [0, 0]
    result = apply_audio_visual_repair(
        performance,
        audio_controls_path=_controls(tmp_path / "controls.npz", contact=False),
        audio_result=_audio_result(),
        timing_evidence=timing,
        config=AudioVisualRepairConfig(taper_frames=0),
    )
    assert result.lower_face_audio_weight[3] == 0.0
    assert result.tongue_audio_weight[3] == 0.0
    np.testing.assert_array_equal(
        result.performance.expression[3], performance.expression[3]
    )
    assert result.report["clockJoin"]["mapping"] == (
        "exact_video_display_start_minus_decoded_audio_start_no_time_warp"
    )
    assert result.report["clockJoin"]["coveredVideoFrames"] == 5


def test_exact_join_honors_nonzero_audio_start_and_variable_frame_times(
    tmp_path: Path,
) -> None:
    performance = _performance()
    timing = _timing(performance)
    audio_start = Fraction(1, 8)
    relative_starts = [
        Fraction(0, 1),
        Fraction(1, 24),
        Fraction(1, 10),
        Fraction(11, 75),
        Fraction(1, 4),
        Fraction(7, 20),
    ]
    timing["audioVideoJoin"]["audio"]["decodedStartExactRational"] = [
        audio_start.numerator,
        audio_start.denominator,
    ]
    audio_end = audio_start + Fraction(3, 5)
    timing["audioVideoJoin"]["audio"]["decodedEndExactRational"] = [
        audio_end.numerator,
        audio_end.denominator,
    ]
    for frame, relative in zip(
        timing["audioVideoJoin"]["video"]["frames"],
        relative_starts,
        strict=True,
    ):
        display = audio_start + relative
        frame["displayStartExactRational"] = [display.numerator, display.denominator]
    result = apply_audio_visual_repair(
        performance,
        audio_controls_path=_controls(tmp_path / "controls.npz"),
        audio_result=_audio_result(),
        timing_evidence=timing,
        config=AudioVisualRepairConfig(taper_frames=0),
    )
    np.testing.assert_allclose(
        result.frame_audio_times_seconds,
        np.asarray([float(value) for value in relative_starts]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_repair_validates_auxiliary_shapes_and_visual_contact_protection(
    tmp_path: Path,
) -> None:
    performance = _performance()
    controls = _controls(tmp_path / "controls.npz")
    with np.load(controls, allow_pickle=False) as source:
        malformed = {name: np.asarray(source[name]).copy() for name in source.files}
    malformed["speech_activity"] = malformed["speech_activity"][:, None]
    write_npz(tmp_path / "malformed.npz", **malformed)
    with pytest.raises(AutoAnimError, match="speech_activity") as invalid:
        apply_audio_visual_repair(
            performance,
            audio_controls_path=tmp_path / "malformed.npz",
            audio_result=_audio_result(),
            timing_evidence=_timing(performance),
        )
    assert invalid.value.code == "AUDIO_VISUAL_REPAIR_BLOCKED"

    protected_base = _performance()
    protected_contact = np.asarray(
        protected_base.source_lip_contact_confidence
    ).copy()
    protected_contact[2] = 0.95
    protected = replace(
        protected_base, source_lip_contact_confidence=protected_contact
    )
    result = apply_audio_visual_repair(
        protected,
        audio_controls_path=controls,
        audio_result=_audio_result(),
        timing_evidence=_timing(protected),
        config=AudioVisualRepairConfig(taper_frames=0),
    )
    assert result.lower_face_audio_weight[2] == 0.0
    np.testing.assert_array_equal(
        result.performance.expression[2, 200:350],
        protected.expression[2, 200:350],
    )
    assert result.report["locks"]["visibleContactProtectedByVisualOwnership"]


def test_audio_contact_target_survives_fusion_for_downstream_artist_veto(
    tmp_path: Path,
) -> None:
    performance = _performance()
    controls = _controls(tmp_path / "controls.npz", contact=False)
    with np.load(controls, allow_pickle=False) as source:
        values = {name: np.asarray(source[name]).copy() for name in source.files}
    values["lip_contact_confidence"][2] = 0.9
    values["contact_correction_applied"][2] = True
    write_npz(tmp_path / "contact.npz", **values)
    result = apply_audio_visual_repair(
        performance,
        audio_controls_path=tmp_path / "contact.npz",
        audio_result=_audio_result(),
        timing_evidence=_timing(performance),
        config=AudioVisualRepairConfig(taper_frames=0),
    )
    assert result.lower_face_audio_weight[2] > 0.0
    assert result.performance.lip_contact_target_gap_interocular[2] == pytest.approx(
        0.01
    )
    assert result.performance.contact_correction_applied[2]
    # Without a rig no final geometry claim is possible; the target is still a
    # hard downstream anchor and the attainment flag remains conservative.
    assert not result.performance.lip_contact_attained[2]


def test_tongue_run_is_tapered_and_status_uses_actual_output_changes(
    tmp_path: Path,
) -> None:
    performance = _performance()
    tapered = apply_audio_visual_repair(
        performance,
        audio_controls_path=_controls(tmp_path / "controls.npz"),
        audio_result=_audio_result(),
        timing_evidence=_timing(performance),
    )
    assert 0.0 < tapered.tongue_audio_weight[0] < tapered.tongue_audio_weight[2]
    assert 0.0 < tapered.tongue_audio_weight[-1] < tapered.tongue_audio_weight[2]
    assert tapered.report["metrics"]["finalTongueCoefficientStepMax"] <= max(
        tapered.report["metrics"]["baselineTongueCoefficientStepMax"], 0.80
    ) + 1.1e-5

    with np.load(tmp_path / "controls.npz", allow_pickle=False) as source:
        values = {name: np.asarray(source[name]).copy() for name in source.files}
    values["expression"][:] = 0.0
    write_npz(tmp_path / "no-op.npz", **values)
    no_op = apply_audio_visual_repair(
        performance,
        audio_controls_path=tmp_path / "no-op.npz",
        audio_result=_audio_result(),
        timing_evidence=_timing(performance),
        config=AudioVisualRepairConfig(taper_frames=0),
    )
    assert no_op.report["status"] == "exact_noop"
    assert no_op.report["metrics"]["lowerFaceAudioWeightedFrames"] > 0
    assert no_op.report["metrics"]["lowerFaceRepairedFrames"] == 0
    assert no_op.report["claims"]["changesFinalGNMMotion"] is False


def test_geometry_limiter_reduces_only_the_weak_repair_run() -> None:
    adapter = GNMAdapter()
    rig = ControlRig(
        adapter,
        ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5"),
    )
    base = np.zeros((5, 383), dtype=np.float32)
    audio = base.copy()
    audio[2, 200:350] = np.float32(3.0) * rig.viseme("A")[200:350]
    weights = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    candidate = base.copy()
    candidate[2, 200:350] = audio[2, 200:350]
    output, limited_weights, limited_runs, baseline_steps, final_steps = (
        _limit_introduced_mouth_steps(
            rig=rig,
            base_expression=base,
            audio_expression=audio,
            output_expression=candidate,
            lower_weight=weights,
            lower_eligible=weights > 0.0,
            maximum_step=0.02,
        )
    )
    assert limited_runs == 1
    assert 0.0 <= limited_weights[2] < 1.0
    assert np.max(final_steps, initial=0.0) <= 0.020011
    np.testing.assert_array_equal(baseline_steps, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(output[[0, 1, 3, 4]], base[[0, 1, 3, 4]])


def test_repair_fails_closed_for_fallback_audio_and_timing_tamper(
    tmp_path: Path,
) -> None:
    performance = _performance()
    controls = _controls(tmp_path / "controls.npz")
    with pytest.raises(AutoAnimError, match="learned Audio2Face") as fallback:
        apply_audio_visual_repair(
            performance,
            audio_controls_path=controls,
            audio_result=_audio_result(learned=False),
            timing_evidence=_timing(performance),
        )
    assert fallback.value.code == "AUDIO_VISUAL_REPAIR_BLOCKED"

    timing = _timing(performance)
    timing["audioVideoJoin"]["video"]["frames"][2]["sourcePTS"] += 1
    with pytest.raises(AutoAnimError, match="source PTS") as tampered:
        apply_audio_visual_repair(
            performance,
            audio_controls_path=controls,
            audio_result=_audio_result(),
            timing_evidence=timing,
        )
    assert tampered.value.code == "AUDIO_VISUAL_REPAIR_BLOCKED"


def test_video_source_snapshot_is_removed_when_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "take.mp4"
    source.write_bytes(b"immutable-video")
    output = tmp_path / "output"
    output.mkdir()
    foreign = output / ".video-source-snapshot-foreign"
    foreign.mkdir()
    (foreign / "owned-by-another-run").write_text("keep", encoding="utf-8")

    def fail_capture(_source: Path, _model: Path) -> None:
        raise RuntimeError("forced capture failure")

    monkeypatch.setattr(video_pipeline_module, "capture_video", fail_capture)
    with pytest.raises(RuntimeError, match="forced capture failure"):
        run_video_pipeline(
            source,
            output,
            model_path=tmp_path / "unused.task",
        )
    assert foreign.is_dir()
    assert (foreign / "owned-by-another-run").read_text(encoding="utf-8") == "keep"
    assert [path.name for path in output.glob(".video-source-snapshot-*")] == [
        foreign.name
    ]


@pytest.mark.skipif(
    not CREMA_D_ANGRY.is_file()
    or not MODEL.is_file()
    or not A2F_RUNNER.is_file()
    or not RHUBARB.is_file()
    or not (A2F_ASSETS / "model_data.npz").is_file()
    or not (A2F_ASSETS / "bs_skin.npz").is_file()
    or not (A2F_ASSETS / "bs_tongue.npz").is_file()
    or not shutil.which("ffmpeg")
    or not shutil.which("ffprobe"),
    reason="Real video, learned A2F, Claire, MediaPipe, Rhubarb, or FFmpeg unavailable",
)
def test_real_video_and_audio_drive_one_pts_bound_gnm_performance(
    tmp_path: Path,
) -> None:
    result = run_video_pipeline(
        CREMA_D_ANGRY,
        tmp_path,
        model_path=MODEL,
        a2f_asset_dir=A2F_ASSETS,
        require_audio_visual_repair=True,
        rhubarb_bin=RHUBARB,
        a2f_runner=A2F_RUNNER,
        a2f_offline=True,
    )
    repair = result["retargeting"]["audio_visual_repair"]
    assert repair["schemaVersion"] == AUDIO_VISUAL_REPAIR_SCHEMA_VERSION
    assert repair["status"] == "repaired"
    assert repair["claims"]["audioContentClassified"] is False
    assert repair["claims"]["speechActivityClassified"] is True
    assert repair["claims"]["rhubarbMouthCuesUsed"] is True
    assert repair["claims"]["lexicalTranscriptGenerated"] is False
    assert repair["claims"]["speechMotionInferred"] is True
    assert repair["claims"]["emotionInferredForFinalMotion"] is False
    assert repair["claims"]["productionValidated"] is False
    assert repair["locks"] == {
        "upperFaceExact": True,
        "pupilExact": True,
        "headPoseAndTranslationExact": True,
        "sourcePtsAndTimestampsExact": True,
        "trustedVisualContactDisagreementIsDiagnostic": True,
        "visibleContactProtectedByVisualOwnership": True,
        "mouthContinuityGeometryValidated": True,
        "tongueCoefficientContinuityValidated": True,
    }
    assert repair["metrics"]["dedicatedTongueDrivenFrames"] > 0
    assert result["capture"]["audio_video_timing_consumed_by_retargeting"] is False
    assert result["capture"][
        "audio_video_sample_join_consumed_by_audio_visual_repair"
    ] is True
    assert result["oral_validation"]["tongue_motion_source"] == (
        "mixed_learned_audio_dedicated_plus_gnm_lower_face_basis_coupling"
    )
    assert result["oral_validation"]["tongue_visible_validated"] is False
    assert result["oral_validation"]["tongue_control_active_frames"] > 0
    assert result["oral_validation"]["tongue_teeth_collision_risk_frames"] == 0
    assert not any("ORAL_TONGUE_SOURCE_UNAVAILABLE" in item for item in result["warnings"])
    assert any("ORAL_TONGUE_AUDIO_INFERRED_UNVALIDATED" in item for item in result["warnings"])
    for name in (
        "audio-visual-repair.json",
        "audio-visual-repair.npz",
        "audio-visual-source-controls.npz",
        "audio-visual-source-arkit-controls.npz",
        "audio-visual-source.wav",
        "audio-visual-source-rhubarb.json",
        "audio-visual-source-cues.json",
        "audio-visual-source-timeline.json",
        "audio-visual-timing-consumption.json",
        "performance-revision-chain.json",
        "performance.npz",
        "performance.glb",
    ):
        assert (tmp_path / name).is_file()
    assert not list(tmp_path.glob(".video-source-snapshot-*"))
    source_manifest = json.loads(
        (tmp_path / "audio-visual-source.json").read_text(encoding="utf-8")
    )
    assert source_manifest["bindings"]["immutableSourceMediaSha256"] == (
        repair["bindings"]["captureSourceSha256"]
    )
    assert source_manifest["bindings"]["a2fModelBundle"] == {
        "modelId": "aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX",
        "sha256": source_manifest["bindings"]["a2fModelBundle"]["sha256"],
        "fileCount": 4,
        "resolvedLocally": True,
        "passedAsExplicitModelDirectory": True,
        "unchangedDuringInference": True,
    }
    assert len(source_manifest["bindings"]["a2fModelBundle"]["sha256"]) == 64
    assert source_manifest["causalInputs"] == {
        "speechActivityClassified": True,
        "rhubarbMouthCuesUsed": True,
        "lexicalTranscriptGenerated": False,
        "automaticEmotionApplied": False,
    }
    revision_chain = json.loads(
        (tmp_path / "performance-revision-chain.json").read_text(encoding="utf-8")
    )
    assert revision_chain["chainConsistent"] is True
    mouth_revision = json.loads(
        (tmp_path / "mouth-aperture-edit.json").read_text(encoding="utf-8")
    )
    assert mouth_revision["source_mode"] == (
        "video_primary_with_audio_visual_repair"
    )
    timing_consumption = json.loads(
        (tmp_path / "audio-visual-timing-consumption.json").read_text(
            encoding="utf-8"
        )
    )
    assert timing_consumption["joinMapping"] == (
        "exact_video_display_start_minus_decoded_audio_start_no_time_warp"
    )
    assert timing_consumption["timeWarpApplied"] is False
    assert timing_consumption["repairChangesFinalGNMMotion"] is True
    with np.load(tmp_path / "audio-visual-repair.npz", allow_pickle=False) as repair_arrays:
        np.testing.assert_array_equal(
            repair_arrays["input_expression"][:, :200],
            repair_arrays["output_expression"][:, :200],
        )
        np.testing.assert_array_equal(
            repair_arrays["input_expression"][:, 382:],
            repair_arrays["output_expression"][:, 382:],
        )
        assert np.max(np.abs(repair_arrays["output_expression"][:, 350:382])) > 0.0
        repaired_expression = np.asarray(repair_arrays["output_expression"]).copy()
    with np.load(tmp_path / "performance.npz", allow_pickle=False) as performance:
        np.testing.assert_array_equal(
            performance["expression"], repaired_expression
        )
