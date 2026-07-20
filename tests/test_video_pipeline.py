from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from autoanim_gnm.api import create_app
from autoanim_gnm.animation import calibrate_lip_contact
from autoanim_gnm.calibrated_retarget import CalibratedRetargeter
from autoanim_gnm.capture_session import (
    CAPTURE_SESSION_SCHEMA_VERSION,
    load_verified_video_capture_session,
)
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.video_capture import (
    MEDIAPIPE_BLENDSHAPE_NAMES,
    load_capture_npz,
    probe_video,
)
from autoanim_gnm.video_capture_run import (
    VIDEO_CAPTURE_RUN_SCHEMA_VERSION,
    load_video_capture_run,
)
from autoanim_gnm.video_pipeline import (
    MAX_PROXY_PTS_ERROR_SECONDS,
    _apply_video_mouth_aperture_edit,
    _export_static_performance_glb,
    _final_output_retention_metrics,
    _mouth_aperture_edit_meets_production_gate,
    _proxy_timing_error,
    _proxy_video,
    _rapid_source_mouth_motion,
)
from autoanim_gnm.video_evidence import (
    PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
    build_performance_evidence,
)
from autoanim_gnm.video_observation import (
    OBSERVATION_V3_POLICY,
    OBSERVATION_V3_SCHEMA_VERSION,
    OBSERVATION_V3_VIEW_SCHEMA_VERSION,
    PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
    PIXEL_OBSERVATION_SCHEMA_VERSION,
    analyze_video_pixels,
    load_pixel_observations,
    load_verified_observation_v3_summary,
)
from autoanim_gnm.video_retarget import retarget_capture
from autoanim_gnm.visual_track import (
    MOTION_AUTHORITY,
    VISUAL_TRACK_POLICY,
    VISUAL_TRACK_SCHEMA_VERSION,
    VISUAL_TRACK_SUMMARY_SCHEMA_VERSION,
    load_verified_visual_track_summary,
    load_visual_track,
)


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
A2F_ASSETS = CACHE / "a2f-claire"
CREMA_D_ANGRY = FIXTURES / "crema-d-1001-dfa-ang.flv"
CREMA_D_ANGRY_SHA256 = "10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4"
CREMA_D_ARRAY_BASELINE_SHA256 = {
    "source_pts": "94f984a33588f43a5488e703a81b117e0acb7bef034318c26c97afcdfe18f385",
    "detected": "8e1bc4b6253c93705e7935f7116fba1bda3383997ca588db079d1eab68bd7c0b",
    "landmarks_xyz": "e4b8d8890f8c14201eb3387a763eb3dbfb148b2272e669e70f26ad4359047123",
    "blendshape_scores": "e63bde330c6a8014bbeb50916bc0721d497436048d40ab295d597cd7d29ca63b",
    "facial_transforms": "a97a36c3de6e576b4744fd32ade491409248a3e39215ec47a0f0e053f5615dd3",
    "pixel_confidence": "a0a1d6cbfc8d1e5f59118b3e84cef21adc31573f6f99be54a630f3a92427fdc3",
    "pixel_reason_mask": "5dbe76c62b608df10a97353f0c749eb45ada54d9d171c3d76380867c0e95fa5c",
    "performance_expression": "3e0f035d4e80eae83f2c946ac873167ba3918a3b64f6cb2c334b4af45586e1c6",
    "performance_rotations": "a920f2b101606e21d4e1a469e269461b12c28b7df3e7d685946e714c6be2623b",
}
RETAINED_CREMA_JOB = Path(
    os.environ.get(
        "AUTOANIM_RETAINED_CREMA_JOB",
        "artifacts/jobs/01kxtx72xy7z1hbmv747hgjzdc",
    )
)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg unavailable",
)
@pytest.mark.parametrize(
    "timestamp_filter",
    (
        "setpts=PTS+0.1/TB",
        "setpts=(0.1+0.1*N+0.1*gte(N\\,2)+0.1*gte(N\\,4))/TB",
    ),
    ids=("positive-start-cfr", "positive-start-vfr"),
)
def test_browser_proxy_preserves_positive_start_frame_zero(
    tmp_path: Path,
    timestamp_filter: str,
) -> None:
    source = tmp_path / "positive-start.mkv"
    subprocess.run(
        (
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=0.5",
            "-vf",
            timestamp_filter,
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            str(source),
        ),
        check=True,
        capture_output=True,
    )
    source_probe = probe_video(source)
    assert source_probe.frame_count == 5
    assert source_probe.source_pts[0] > 0

    proxy = _proxy_video(source, tmp_path / "proxy.mp4")
    proxy_probe, error, start = _proxy_timing_error(source, proxy)

    assert proxy_probe.frame_count == source_probe.frame_count
    np.testing.assert_allclose(
        proxy_probe.timestamps_seconds,
        source_probe.timestamps_seconds,
        atol=MAX_PROXY_PTS_ERROR_SECONDS,
        rtol=0.0,
    )
    assert error <= MAX_PROXY_PTS_ERROR_SECONDS
    assert abs(start) <= MAX_PROXY_PTS_ERROR_SECONDS


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg unavailable",
)
def test_browser_proxy_preserves_audio_delay_from_positive_video_origin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "delayed-audio.mkv"
    subprocess.run(
        (
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=10:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.4",
            "-filter_complex",
            "[0:v]setpts=PTS+0.1/TB[v];[1:a]asetpts=PTS+0.2/TB[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(source),
        ),
        check=True,
        capture_output=True,
    )
    proxy = _proxy_video(source, tmp_path / "proxy.mp4")
    proxy_probe, error, start = _proxy_timing_error(source, proxy)
    stream_document = json.loads(
        subprocess.run(
            (
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,start_time",
                "-of",
                "json",
                str(proxy),
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stream_starts = {
        stream["codec_type"]: float(stream["start_time"])
        for stream in stream_document["streams"]
    }

    assert proxy_probe.frame_count == 5
    assert error <= MAX_PROXY_PTS_ERROR_SECONDS
    assert abs(start) <= MAX_PROXY_PTS_ERROR_SECONDS
    assert stream_starts["video"] == pytest.approx(0.0, abs=0.001)
    # AAC exposes one encoder-priming packet before the intended audible
    # content. The stream start therefore precedes the +100 ms source offset by
    # approximately one 1024-sample frame, but must not collapse to zero.
    assert stream_starts["audio"] > 0.05
    assert stream_starts["audio"] == pytest.approx(
        0.1,
        abs=(1024.0 / 48_000.0) + 0.002,
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@pytest.mark.parametrize(
    "timestamps",
    (
        np.arange(10, dtype=np.float64) / 30.0,
        np.arange(20, dtype=np.float64) / 60.0,
        np.asarray((0.0, 0.011, 0.043, 0.091, 0.167, 0.280), dtype=np.float64),
    ),
)
def test_authored_aperture_source_veto_uses_physical_speed_on_exact_pts(
    timestamps: np.ndarray,
) -> None:
    mouth = np.zeros((len(timestamps), 20, 3), dtype=np.float64)
    mouth[:, :, 0] = 2.5 * timestamps[:, None]

    rapid, frame_speed = _rapid_source_mouth_motion(
        mouth,
        timestamps,
        maximum_speed_interocular_per_second=2.4,
        maximum_step_interocular=1.0,
    )

    np.testing.assert_array_equal(rapid, np.ones(len(timestamps), dtype=bool))
    np.testing.assert_allclose(frame_speed, 2.5, rtol=0.0, atol=2.0e-6)

    mouth[:, :, 0] = 2.3 * timestamps[:, None]
    rapid, frame_speed = _rapid_source_mouth_motion(
        mouth,
        timestamps,
        maximum_speed_interocular_per_second=2.4,
        maximum_step_interocular=1.0,
    )
    assert not rapid.any()
    np.testing.assert_allclose(frame_speed, 2.3, rtol=0.0, atol=2.0e-6)


def test_authored_aperture_source_veto_keeps_absolute_gap_safety() -> None:
    timestamps = np.asarray((0.0, 0.1), dtype=np.float64)
    mouth = np.zeros((2, 20, 3), dtype=np.float64)
    mouth[1, :, 0] = 0.05

    rapid, frame_speed = _rapid_source_mouth_motion(mouth, timestamps)

    np.testing.assert_array_equal(rapid, np.ones(2, dtype=bool))
    np.testing.assert_allclose(frame_speed, 0.5, rtol=0.0, atol=1.0e-7)


def test_long_track_static_viewer_preserves_character_uv_layout(tmp_path: Path) -> None:
    adapter = GNMAdapter()
    packed_uvs = np.asarray(adapter.model.triangle_uvs, dtype=np.float32) * 0.5 + 0.1
    texture = tmp_path / "packed.png"
    Image.new("RGB", (16, 16), (110, 80, 65)).save(texture)
    exported = _export_static_performance_glb(
        tmp_path,
        adapter,
        adapter.mesh(),
        texture_path=texture,
        texture_triangle_uvs=packed_uvs,
    )
    with np.load(exported.mapping_path, allow_pickle=False) as mapping:
        np.testing.assert_allclose(
            mapping["uvs_lower_left"][mapping["triangles"]],
            packed_uvs,
        )


@pytest.mark.skipif(
    not (A2F_ASSETS / "bs_skin.npz").is_file()
    or not (A2F_ASSETS / "bs_tongue.npz").is_file(),
    reason="Claire geometry-calibration assets unavailable",
)
def test_mediapipe_controls_use_dense_geometry_calibration() -> None:
    retargeter = CalibratedRetargeter.from_directory(A2F_ASSETS, adapter=GNMAdapter())
    source = np.zeros((5, len(MEDIAPIPE_BLENDSHAPE_NAMES)), dtype=np.float32)
    columns = {name: index for index, name in enumerate(MEDIAPIPE_BLENDSHAPE_NAMES)}
    source[:, columns["jawOpen"]] = (0.0, 0.25, 0.8, 0.3, 0.0)
    source[:, columns["mouthSmileLeft"]] = (0.0, 0.1, 0.5, 0.1, 0.0)
    source[:, columns["browDownRight"]] = (0.0, 0.4, 0.7, 0.3, 0.0)
    expression = retargeter.retarget_sequence(
        source,
        MEDIAPIPE_BLENDSHAPE_NAMES,
        strict=False,
    )
    assert expression.shape == (5, 383)
    assert np.count_nonzero(np.ptp(expression, axis=0) > 1e-7) > 10
    assert np.max(np.abs(expression)) > 0


@pytest.mark.skipif(
    not (RETAINED_CREMA_JOB / "capture.npz").is_file()
    or not (RETAINED_CREMA_JOB / "input.flv").is_file()
    or not (A2F_ASSETS / "bs_skin.npz").is_file(),
    reason="retained checksum-pinned CREMA-D capture unavailable",
)
def test_retained_crema_capture_neutral_audit_and_final_geometry_retention(
    tmp_path: Path,
) -> None:
    source = RETAINED_CREMA_JOB / "input.flv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == CREMA_D_ANGRY_SHA256
    capture = load_capture_npz(RETAINED_CREMA_JOB / "capture.npz")
    assert capture.provenance.source_sha256 == CREMA_D_ANGRY_SHA256
    evidence = build_performance_evidence(capture)
    assert evidence["schemaVersion"] == PERFORMANCE_EVIDENCE_SCHEMA_VERSION
    assert [frame["sourcePTS"] for frame in evidence["frames"]] == (
        capture.source_pts.tolist()
    )
    assert np.all(np.diff([frame["projectTick"] for frame in evidence["frames"]]) > 0)
    assert evidence["summary"]["observedFrames"] == capture.frame_count
    assert evidence["summary"]["missingFrames"] == 0
    for region in ("mouth", "eyes", "upperFace", "head"):
        assert evidence["summary"]["regions"][region]["geometryOnlyFrames"] == (
            capture.frame_count
        )
    assert all(
        frame["neutralityState"] == "unknown" for frame in evidence["frames"]
    )
    adapter = GNMAdapter()
    retargeter = CalibratedRetargeter.from_directory(A2F_ASSETS, adapter=adapter)
    rig = ControlRig(
        adapter,
        ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5"),
    )
    contact_calibration = calibrate_lip_contact(rig)
    performance = retarget_capture(
        capture,
        retargeter,
        contact_rig=rig,
        lip_contact_calibration=contact_calibration,
    )
    observations = analyze_video_pixels(source, capture)
    performance_after_observation = retarget_capture(
        capture,
        retargeter,
        contact_rig=rig,
        lip_contact_calibration=contact_calibration,
    )
    for field in fields(performance):
        before_value = getattr(performance, field.name)
        after_value = getattr(performance_after_observation, field.name)
        if isinstance(before_value, np.ndarray):
            np.testing.assert_array_equal(before_value, after_value)
        else:
            assert before_value == after_value
    observations.validate_capture(capture)

    np.testing.assert_array_equal(performance.source_pts, capture.source_pts)
    np.testing.assert_array_equal(
        performance.timestamps_seconds,
        capture.timestamps_seconds,
    )
    provenance = performance.provenance
    assert provenance.baseline_frame_indices == tuple(range(7))
    assert provenance.neutral_baseline_method == "initial_window"
    assert provenance.neutral_baseline_correction_applied is True
    assert provenance.neutral_baseline_validated is False
    assert provenance.neutral_baseline_score == pytest.approx(0.7356666, abs=1e-5)
    assert provenance.neutral_baseline_ambiguity_controls == (
        "browDownLeft",
        "browDownRight",
    )
    assert provenance.quarantined_expression_controls == ("mouthClose",)
    assert provenance.contact_calibration_hash == contact_calibration.calibration_hash
    assert provenance.negative_baseline_residual_clipped_fraction > 0.40

    # The old semantic-control proxy fired at frame 14 because mouthRollUpper
    # peaked while the lips were visibly apart. Geometry localizes the actual
    # close after that vowel and the final GNM reaches its calibrated target.
    assert performance.source_lip_contact_confidence[14] == pytest.approx(0.0)
    assert performance.source_lip_contact_confidence[17] > 0.95
    assert performance.contact_correction_applied[17]
    assert performance.lip_contact_attained[17]
    assert performance.lip_contact_target_gap_interocular[17] > 0.0

    metrics = _final_output_retention_metrics(capture, performance, adapter)
    assert metrics["final_blink_source_event_count"] == 0
    assert metrics["final_blink_event_retained_fraction"] is None
    assert metrics["final_contact_source_event_count"] == 1
    assert metrics["final_contact_event_retained_fraction"] == pytest.approx(1.0)
    assert metrics["final_contact_motion_correlation"] > 0.20
    assert metrics["final_contact_geometry_attained_fraction"] == pytest.approx(1.0)
    assert metrics["final_expression_source_event_count"] == 3
    assert metrics["final_expression_motion_retained_fraction"] == pytest.approx(1.0)
    assert metrics["final_expression_motion_correlation"] > 0.85
    assert metrics["final_expression_landmark_step_p95_interocular"] > 0.05
    assert metrics["final_lip_aperture_source_output_correlation"] >= 0.90
    assert 0.85 <= metrics["final_lip_aperture_open_p95_ratio"] <= 1.15
    assert 0.85 <= metrics["final_lip_aperture_affine_slope"] <= 1.15
    assert metrics["final_lip_aperture_correction_applied_frames"] > 0
    assert metrics["final_lip_aperture_target_attainment_fraction"] >= 0.95
    assert np.max(np.abs(performance.expression)) <= 3.0

    no_op_dir = tmp_path / "no-op"
    no_op_dir.mkdir()
    no_op, no_op_report = _apply_video_mouth_aperture_edit(
        output_dir=no_op_dir,
        rig=rig,
        performance=performance,
        gain=1.0,
        author=None,
        reason=None,
        source_sha256=capture.provenance.source_sha256,
        model_sha256=capture.provenance.model_sha256,
        retarget_calibration_hash=retargeter.calibration.calibration_hash,
    )
    np.testing.assert_array_equal(no_op.expression, performance.expression)
    assert not no_op_report.correction_applied.any()

    edit_dir = tmp_path / "authored"
    edit_dir.mkdir()
    edited, edit_report = _apply_video_mouth_aperture_edit(
        output_dir=edit_dir,
        rig=rig,
        performance=performance,
        gain=1.08,
        author="Test artist",
        reason="Open-vowel review correction",
        source_sha256=capture.provenance.source_sha256,
        model_sha256=capture.provenance.model_sha256,
        retarget_calibration_hash=retargeter.calibration.calibration_hash,
    )
    assert np.any(edit_report.correction_applied)
    protected = edit_report.protected_contact
    np.testing.assert_array_equal(edited.expression[protected], performance.expression[protected])
    np.testing.assert_array_equal(edited.expression[:, :200], performance.expression[:, :200])
    np.testing.assert_array_equal(edited.expression[:, 350:], performance.expression[:, 350:])
    np.testing.assert_array_equal(edited.source_pts, performance.source_pts)
    np.testing.assert_array_equal(edited.timestamps_seconds, performance.timestamps_seconds)
    edited_metrics = _final_output_retention_metrics(capture, edited, adapter)
    assert edited_metrics["final_lip_aperture_source_output_correlation"] >= 0.95
    assert edited_metrics["final_lip_aperture_open_p95_ratio"] >= (
        metrics["final_lip_aperture_open_p95_ratio"] + 0.004
    )
    assert edited_metrics["final_lip_aperture_open_p95_ratio"] < 0.90
    assert 0.90 <= edited_metrics["final_lip_aperture_affine_slope"] <= 1.10
    payload = json.loads((edit_dir / "mouth-aperture-edit.json").read_text())
    assert payload["timeline"]["source_pts"] == capture.source_pts.tolist()
    assert payload["claims"]["video_pts_byte_identical"] is True
    assert payload["claims"]["contact_is_a_hard_veto"] is True
    assert payload["summary"]["introduced_lip_order_risk_frames"] == 0
    assert payload["summary"]["rapid_source_motion_veto_frames"] >= 2
    assert payload["summary"]["target_attained_fraction"] >= 0.95
    assert not _mouth_aperture_edit_meets_production_gate(
        edited_metrics,
        payload["summary"]["target_attained_fraction"],
    )


@pytest.mark.skipif(
    not CREMA_D_ANGRY.is_file()
    or not MODEL.is_file()
    or not (A2F_ASSETS / "bs_skin.npz").is_file()
    or not shutil.which("ffmpeg")
    or not shutil.which("ffprobe"),
    reason="opt-in CREMA-D/model/Claire/FFmpeg fixtures unavailable",
)
def test_real_crema_d_dense_video_pipeline_e2e(tmp_path: Path) -> None:
    assert CREMA_D_ANGRY.stat().st_size == 265_922
    assert hashlib.sha256(CREMA_D_ANGRY.read_bytes()).hexdigest() == CREMA_D_ANGRY_SHA256
    app = create_app(
        tmp_path / "jobs",
        model_path=MODEL,
        a2f_asset_dir=A2F_ASSETS,
    )
    with CREMA_D_ANGRY.open("rb") as handle:
        response = TestClient(app).post(
            "/api/video",
            files={"file": (CREMA_D_ANGRY.name, handle, "video/x-flv")},
        )
    assert response.status_code == 201, response.text
    result = response.json()
    job_dir = tmp_path / "jobs" / result["job_id"]

    assert result["kind"] == "video_performance"
    assert result["input"]["media_type"] == "video/x-flv"
    assert result["capture"]["frames"] == 67
    assert result["capture"]["detected_frames"] == 67
    assert result["capture"]["identity_fixed_for_all_frames"] is True
    assert result["capture"]["capture_quality_source"].endswith(
        "otherwise_in_frame_fraction"
    )
    assert result["capture"]["performance_evidence_schema_version"] == (
        PERFORMANCE_EVIDENCE_SCHEMA_VERSION
    )
    assert result["capture"]["performance_evidence_policy"] == (
        "observation_only_no_motion_effect"
    )
    assert result["capture"]["observation_v3_schema_version"] == (
        OBSERVATION_V3_SCHEMA_VERSION
    )
    assert result["capture"]["observation_v3_arrays_schema_version"] == (
        PIXEL_OBSERVATION_SCHEMA_VERSION
    )
    assert result["capture"]["observation_v3_policy"] == OBSERVATION_V3_POLICY
    assert result["capture"]["observation_v3_consumed_by_retargeting"] is False
    assert result["capture"]["visual_track_schema_version"] == (
        VISUAL_TRACK_SCHEMA_VERSION
    )
    assert result["capture"]["visual_track_summary_schema_version"] == (
        VISUAL_TRACK_SUMMARY_SCHEMA_VERSION
    )
    assert result["capture"]["visual_track_policy"] == VISUAL_TRACK_POLICY
    assert result["capture"]["visual_track_motion_authority"] == MOTION_AUTHORITY
    assert result["capture"]["visual_track_consumed_by_retargeting"] is False
    assert result["capture"]["visual_track_detector_ingress_hashes_verified"] is True
    assert result["capture"]["video_capture_run_schema_version"] == (
        VIDEO_CAPTURE_RUN_SCHEMA_VERSION
    )
    assert result["capture"]["capture_session_schema_version"] == (
        CAPTURE_SESSION_SCHEMA_VERSION
    )
    assert result["capture"]["production_validated"] is False
    assert any(
        "VISUAL_TRACK_SHADOW_ONLY" in warning for warning in result["warnings"]
    )
    assert result["retargeting"]["backend"] == (
        "geometry_calibrated_dense_contact_aperture_v3"
    )
    assert result["retargeting"]["geometry_calibrated"] is True
    assert result["retargeting"]["audio_visual_repair"]["status"] == "disabled"
    assert len(result["retargeting"]["calibration_hash"]) == 64
    assert result["retargeting"]["matched_source_controls"] == 51
    assert result["retargeting"]["matched_source_fraction"] > 0.98
    assert result["metrics"]["source_fast_contact_filter_passthrough_exact"] is True
    assert result["metrics"]["source_noncontact_filter_variation_retention"] > 0.75
    assert result["metrics"]["effective_capture_quality_median"] > 0.95
    assert result["metrics"]["landmark_in_frame_fraction_median"] > 0.95
    assert result["retargeting"]["neutral_baseline_method"] == "initial_window"
    assert result["retargeting"]["neutral_baseline_correction_applied"] is True
    assert result["retargeting"]["neutral_baseline_validated"] is False
    assert result["metrics"]["neutral_baseline_score"] <= 1.0
    assert result["metrics"]["neutral_baseline_semantic_peak"] > 0.50
    assert result["metrics"]["negative_baseline_residual_clipped_fraction"] > 0.40
    assert any("NEUTRAL_BASELINE_ONE_SIDED_LOSS" in warning for warning in result["warnings"])
    assert result["metrics"]["final_expression_motion_retained_fraction"] >= 0.80
    assert result["metrics"]["final_expression_motion_correlation"] > 0.45
    assert result["metrics"]["final_lip_aperture_source_output_correlation"] >= 0.90
    assert 0.85 <= result["metrics"]["final_lip_aperture_open_p95_ratio"] <= 1.15
    assert result["metrics"]["final_lip_aperture_target_attainment_fraction"] >= 0.95
    assert result["metrics"]["final_blink_source_event_count"] >= 0
    assert result["metrics"]["final_contact_source_event_count"] >= 0
    assert "final_blink_event_retained_fraction" in result["metrics"]
    assert "final_contact_event_retained_fraction" in result["metrics"]
    assert not any("FINAL_CONTACT_TIMING_MISMATCH" in warning for warning in result["warnings"])
    assert result["metrics"]["final_contact_geometry_attained_fraction"] == pytest.approx(1.0)
    assert result["retargeting"]["quarantined_expression_controls"] == ["mouthClose"]
    assert result["retargeting"]["contact_source"] == "mediapipe_inner_lip_geometry_v1"
    assert len(result["retargeting"]["contact_calibration_hash"]) == 64
    assert result["metrics"]["retarget_bound_active_frames"] == 0
    assert not any("safety bound was active" in warning for warning in result["warnings"])
    assert result["retargeting"]["neutral_baseline_frame_indices"] == list(range(7))
    assert result["metrics"]["proxy_pts_max_error_ms"] <= (
        MAX_PROXY_PTS_ERROR_SECONDS * 1_000
    )
    assert abs(result["metrics"]["proxy_video_start_ms"]) <= 0.001
    assert result["viewer"]["status"] == "ready"
    assert result["viewer"]["mode"] == "animation"
    assert result["viewer"]["glb_covers_full_track"] is True
    assert result["viewer"]["clock_artifact"] == "viewer_media"
    assert result["oral_validation"]["all_control_frames_evaluated"] is True
    assert result["oral_validation"]["viewer_structural_reconstruction_validated"] is True
    assert any(
        "ORAL_TONGUE_SOURCE_UNAVAILABLE" in warning
        for warning in result["warnings"]
    )
    assert result["artifacts"]["retarget_calibration"]["media_type"] == (
        "application/octet-stream"
    )
    assert result["artifacts"]["viewer_media"]["media_type"] == "video/mp4"
    assert result["artifacts"]["oral_validation"]["media_type"] == "application/json"
    assert result["artifacts"]["oral_glb_validation"]["media_type"] == "application/json"
    assert result["artifacts"]["performance_evidence"]["media_type"] == (
        "application/json"
    )
    assert result["artifacts"]["pixel_observations"]["media_type"] == (
        "application/octet-stream"
    )
    assert result["artifacts"]["observation_v3"]["media_type"] == (
        "application/json"
    )
    assert result["artifacts"]["visual_track"]["media_type"] == (
        "application/octet-stream"
    )
    assert result["artifacts"]["visual_track_summary"]["media_type"] == (
        "application/json"
    )
    assert result["artifacts"]["video_capture_run"]["media_type"] == (
        "application/json"
    )
    assert result["artifacts"]["capture_session"]["media_type"] == (
        "application/json"
    )

    evidence_report = json.loads(
        (job_dir / "performance-evidence.json").read_text(encoding="utf-8")
    )
    assert evidence_report["schemaVersion"] == PERFORMANCE_EVIDENCE_SCHEMA_VERSION
    assert evidence_report["policy"] == "observation_only_no_motion_effect"
    assert evidence_report["source"]["frameCount"] == 67
    assert evidence_report["summary"]["observedFrames"] == 67
    assert evidence_report["summary"]["missingFrames"] == 0
    assert [frame["sourcePTS"] for frame in evidence_report["frames"]] == (
        probe_video(CREMA_D_ANGRY).source_pts.tolist()
    )
    assert all(
        frame["neutralityState"] == "unknown"
        for frame in evidence_report["frames"]
    )

    capture_track = load_capture_npz(job_dir / "capture.npz")
    assert _array_sha256(capture_track.source_pts) == (
        CREMA_D_ARRAY_BASELINE_SHA256["source_pts"]
    )
    assert _array_sha256(capture_track.detected) == (
        CREMA_D_ARRAY_BASELINE_SHA256["detected"]
    )
    assert _array_sha256(capture_track.landmarks_xyz) == (
        CREMA_D_ARRAY_BASELINE_SHA256["landmarks_xyz"]
    )
    assert _array_sha256(capture_track.blendshape_scores) == (
        CREMA_D_ARRAY_BASELINE_SHA256["blendshape_scores"]
    )
    assert _array_sha256(capture_track.facial_transforms) == (
        CREMA_D_ARRAY_BASELINE_SHA256["facial_transforms"]
    )
    assert "${SOURCE}" in capture_track.provenance.ffprobe_command
    assert "${SOURCE}" in capture_track.provenance.ffmpeg_command
    assert str(job_dir) not in json.dumps(capture_track.provenance.as_dict())
    pixel_observations = load_pixel_observations(
        job_dir / "pixel-observations.npz"
    )
    pixel_observations.validate_capture(capture_track)
    assert _array_sha256(pixel_observations.confidence) == (
        CREMA_D_ARRAY_BASELINE_SHA256["pixel_confidence"]
    )
    assert _array_sha256(pixel_observations.reason_mask) == (
        CREMA_D_ARRAY_BASELINE_SHA256["pixel_reason_mask"]
    )
    np.testing.assert_array_equal(
        pixel_observations.source_pts, capture_track.source_pts
    )
    assert np.nanmax(pixel_observations.confidence) <= (
        PIXEL_DIAGNOSTIC_CONFIDENCE_CAP
    )
    assert np.isfinite(pixel_observations.focus_reference).all()
    assert np.ptp(pixel_observations.confidence[:, 0]) > 0.25
    assert np.mean(
        pixel_observations.confidence == PIXEL_DIAGNOSTIC_CONFIDENCE_CAP
    ) < 0.10
    assert np.min(pixel_observations.roi_pixel_count) > 256
    assert not np.any(pixel_observations.cut_candidate)
    assert not np.any(pixel_observations.photometric_discontinuity_candidate)
    assert np.flatnonzero(pixel_observations.observation_epoch_start).tolist() == [0]
    assert np.sum(np.isfinite(pixel_observations.temporal_innovation), axis=0).tolist() == [
        66,
        66,
        66,
        66,
    ]
    assert np.sum(np.isfinite(pixel_observations.flow_consistency), axis=0).tolist() == [
        66,
        66,
        66,
        66,
    ]
    observation_v3 = load_verified_observation_v3_summary(
        job_dir / "observation-v3.json",
        pixel_observations_path=job_dir / "pixel-observations.npz",
        capture_artifact_path=job_dir / "capture.npz",
        expected_capture=capture_track,
    )
    assert observation_v3["schemaVersion"] == OBSERVATION_V3_SCHEMA_VERSION
    assert observation_v3["consumedByRetargeting"] is False
    assert observation_v3["claims"]["productionValidated"] is False
    assert observation_v3["decodedPixels"]["relationshipToDetectorInput"] == (
        "redecoded_for_evidence"
    )
    assert "frames" not in observation_v3
    assert observation_v3["events"]["identityContinuityState"] == "unknown"
    assert observation_v3["events"][
        "identityOrTrackingJumpCandidateFrames"
    ] is None
    assert all(
        summary["strongFrames"] == 0
        for summary in observation_v3["summary"]["regions"].values()
    )
    visual_track = load_visual_track(job_dir / "visual-track.npz")
    expected_capture_run = load_video_capture_run(
        job_dir / "video-capture-run.json",
        expected_capture=capture_track,
    )
    visual_summary = load_verified_visual_track_summary(
        job_dir / "visual-track.json",
        visual_track_path=job_dir / "visual-track.npz",
        expected_capture=capture_track,
        expected_observations=pixel_observations,
        expected_capture_run=expected_capture_run,
    )
    assert visual_summary["schemaVersion"] == VISUAL_TRACK_SUMMARY_SCHEMA_VERSION
    assert visual_summary["policy"] == VISUAL_TRACK_POLICY
    assert visual_summary["motionAuthority"] == MOTION_AUTHORITY
    assert visual_summary["consumedByRetargeting"] is False
    assert visual_track.evidence_rgb_sha256 == (
        pixel_observations.decoded_pixel_sha256
    )
    capture_session = load_verified_video_capture_session(
        job_dir / "capture-session.json",
        expected_capture=capture_track,
        expected_observations=pixel_observations,
        artifact_paths={
            "capture": job_dir / "capture.npz",
            "capture_jsonl": job_dir / "capture.jsonl",
            "performance_evidence": job_dir / "performance-evidence.json",
            "pixel_observations": job_dir / "pixel-observations.npz",
            "observation_v3": job_dir / "observation-v3.json",
            "video_capture_run": job_dir / "video-capture-run.json",
            "visual_track": job_dir / "visual-track.npz",
            "visual_track_summary": job_dir / "visual-track.json",
        },
        expected_capture_run=expected_capture_run,
    )
    assert capture_session["schema_version"] == CAPTURE_SESSION_SCHEMA_VERSION
    assert capture_session["subject_binding"]["state"] == "unbound"
    assert capture_session["assessments"]["identity_continuity"]["state"] == (
        "unknown"
    )
    assert capture_session["claims"]["changes_final_gnm_motion"] is False
    assert capture_session["claims"]["production_validated"] is False

    readiness = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/production-readiness"
    )
    assert readiness.status_code == 200
    readiness_evidence = readiness.json()["gates"]["performance"]["evidence"]
    assert readiness_evidence["performance_evidence_artifact_verified"] is True
    assert readiness_evidence["observation_v3_artifacts_verified"] is True
    assert readiness_evidence["video_capture_run_artifact_verified"] is True
    assert readiness_evidence["visual_track_artifacts_verified"] is True
    assert readiness_evidence["capture_session_artifact_verified"] is True
    assert readiness_evidence["capture_session_production_claims_verified"] is False

    oral_report = json.loads((job_dir / "oral-validation.json").read_text(encoding="utf-8"))
    assert oral_report["source"]["evaluation_mode"] == "provided_complete_gnm_frames"
    glb_report = json.loads(
        (job_dir / "oral-glb-validation.json").read_text(encoding="utf-8")
    )
    assert glb_report["structural_reconstruction"]["reference_evaluation_mode"] == (
        "provided_complete_gnm_frames"
    )
    assert oral_report["lip_contact"]["order_inversion_risk_frames"] == 0
    assert glb_report["lip_contact"]["order_inversion_risk_frames"] == 0
    assert result["oral_validation"]["control_lip_order_inversion_risk_frames"] == 0
    assert result["oral_validation"]["viewer_lip_order_inversion_risk_frames"] == 0
    assert result["oral_validation"]["lip_order_inversion_risk_frames"] == 0
    assert result["oral_validation"]["tongue_geometry_motion_frames"] > 0
    assert result["oral_validation"]["tongue_motion_source"] == (
        "gnm_lower_face_basis_coupling_no_dedicated_source"
    )
    assert any(
        "ORAL_UNSOURCED_TONGUE_BASIS_COUPLING" in warning
        for warning in result["warnings"]
    )

    with np.load(job_dir / "capture.npz", allow_pickle=False) as capture:
        with np.load(job_dir / "performance.npz", allow_pickle=False) as performance:
            assert _array_sha256(performance["expression"]) == (
                CREMA_D_ARRAY_BASELINE_SHA256["performance_expression"]
            )
            assert _array_sha256(performance["rotations"]) == (
                CREMA_D_ARRAY_BASELINE_SHA256["performance_rotations"]
            )
            np.testing.assert_array_equal(
                performance["timestamps_seconds"], capture["timestamps_seconds"]
            )
            np.testing.assert_array_equal(performance["source_pts"], capture["source_pts"])
            assert np.isfinite(performance["expression"]).all()
            assert np.count_nonzero(np.ptp(performance["expression"], axis=0) > 1e-7) > 25
            assert np.count_nonzero(performance["lip_aperture_correction_applied"]) > 0
            assert np.mean(performance["lip_aperture_target_attained"][
                performance["lip_aperture_target_gap_interocular"] > 0.0
            ]) >= 0.95
            provenance = json.loads(str(performance["provenance_json"].item()))
            assert "CalibratedRetargeter" in provenance["retargeter"]
            assert provenance["baseline_frame_indices"] == list(range(7))
            assert len(provenance["neutral_blendshape_baseline"]) == 52
            assert provenance["neutral_baseline_method"] == "initial_window"
            assert provenance["neutral_baseline_validated"] is False
            assert provenance["neutral_baseline_correction_applied"] is True
            assert any("semantic ambiguity" in value for value in provenance["caveats"])

    source_probe = probe_video(CREMA_D_ANGRY)
    proxy_probe = probe_video(job_dir / "source-proxy.mp4")
    assert proxy_probe.frame_count == source_probe.frame_count
    assert proxy_probe.source_pts[0] == 0
    assert np.max(
        np.abs(proxy_probe.timestamps_seconds - source_probe.timestamps_seconds)
    ) <= MAX_PROXY_PTS_ERROR_SECONDS
    streams = json.loads(
        subprocess.run(
            (
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(job_dir / "source-proxy.mp4"),
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["streams"]
    assert {stream["codec_type"] for stream in streams} == {"audio", "video"}
    assert (job_dir / "performance.glb").stat().st_size > 1_000
    assert (job_dir / "retarget_calibration.npz").is_file()

    viewer = TestClient(app).get(f"/api/jobs/{result['job_id']}/viewer")
    assert viewer.status_code == 200
    assert 'mediaKind="video"' in viewer.text
    assert f"/api/jobs/{result['job_id']}/files/source-proxy.mp4" in viewer.text
    observation_view_url = (
        f"/api/jobs/{result['job_id']}/observation-v3-view"
    )
    assert observation_view_url in viewer.text
    assert '"observation_review": "available"' in viewer.text
    assert '"native_review_bundle": "available"' in viewer.text
    assert "Object.defineProperty(window,'autoanimReview'" in viewer.text
    review_bundle_response = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/review-bundle"
    )
    assert review_bundle_response.status_code == 200
    review_bundle = review_bundle_response.json()
    assert review_bundle["schema_version"] == "autoanim.review-bundle/1.0"
    assert review_bundle["source_manifest"]["job_id"] == result["job_id"]
    assert review_bundle["clock"]["source_pts"] == source_probe.source_pts.tolist()
    assert review_bundle["claims"]["production_validated"] is False
    assert review_bundle["claims"]["publishable"] is False
    assert review_bundle_response.headers[
        "x-autoanim-review-bundle-sha256"
    ] == review_bundle["bundle_sha256"]
    review_layers = {
        layer["layer_id"]: layer for layer in review_bundle["layers"]
    }
    assert review_layers["visual_base"]["changes_motion_reported"] is True
    assert review_layers["audio_repair"]["availability"] == "unavailable"
    assert review_layers["audio_repair"]["changes_motion_reported"] is False
    assert "drawDiagnosticOverlay" in viewer.text
    observation_view_response = TestClient(app).get(observation_view_url)
    assert observation_view_response.status_code == 200
    observation_view = observation_view_response.json()
    assert observation_view["schemaVersion"] == OBSERVATION_V3_VIEW_SCHEMA_VERSION
    assert len(observation_view["frames"]) == 67
    assert [frame["sourcePTS"] for frame in observation_view["frames"]] == (
        source_probe.source_pts.tolist()
    )
    assert observation_view["claims"]["changesFinalGNMMotion"] is False
    assert observation_view["claims"]["confidenceCalibrated"] is False
    assert all(
        frame["regions"]["mouth"]["confidence"]
        <= PIXEL_DIAGNOSTIC_CONFIDENCE_CAP
        for frame in observation_view["frames"]
    )
    review_frame_response = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/review-frames/5.png"
    )
    assert review_frame_response.status_code == 200
    assert review_frame_response.headers["content-type"] == "image/png"
    assert review_frame_response.headers["x-autoanim-frame-index"] == "5"
    assert review_frame_response.headers["x-autoanim-proxy-sha256"] == (
        result["artifacts"]["viewer_media"]["sha256"]
    )
    routed_frame = cv2.imdecode(
        np.frombuffer(review_frame_response.content, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    independently_decoded = subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(job_dir / "source-proxy.mp4"),
            "-vf",
            "select=between(n\\,4\\,6)",
            "-frames:v",
            "3",
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        check=True,
        capture_output=True,
    ).stdout
    independent_frames = np.frombuffer(
        independently_decoded, dtype=np.uint8
    ).reshape(
        3, proxy_probe.height, proxy_probe.width, 3
    )
    cross_decoder_errors = np.abs(
        independent_frames.astype(np.int16) - routed_frame.astype(np.int16)
    )
    selected_error = cross_decoder_errors[1]
    assert int(np.max(selected_error)) <= 3
    assert float(np.quantile(selected_error, 0.99)) <= 3.0
    assert float(np.mean(selected_error)) < float(np.mean(cross_decoder_errors[0]))
    assert float(np.mean(selected_error)) < float(np.mean(cross_decoder_errors[2]))
    out_of_range_review_frame = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/review-frames/67.png"
    )
    assert out_of_range_review_frame.status_code == 400
    assert out_of_range_review_frame.json()["code"] == "INPUT_INVALID"

    proxy_path = job_dir / "source-proxy.mp4"
    proxy_bytes = proxy_path.read_bytes()
    proxy_path.write_bytes(b"tampered-proxy")
    tampered_review_frame = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/review-frames/5.png"
    )
    assert tampered_review_frame.status_code == 404
    assert tampered_review_frame.json()["code"] == "ARTIFACT_NOT_FOUND"
    proxy_path.write_bytes(proxy_bytes)

    capture_run_path = job_dir / "video-capture-run.json"
    capture_run_bytes = capture_run_path.read_bytes()
    capture_run_path.write_text("{}\n", encoding="utf-8")
    capture_run_tampered = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/production-readiness"
    ).json()["gates"]["performance"]["evidence"]
    assert capture_run_tampered["video_capture_run_artifact_verified"] is False
    assert capture_run_tampered["visual_track_artifacts_verified"] is False
    assert capture_run_tampered["capture_session_artifact_verified"] is False
    capture_run_path.write_bytes(capture_run_bytes)

    visual_summary_path = job_dir / "visual-track.json"
    visual_summary_bytes = visual_summary_path.read_bytes()
    visual_summary_path.write_text("{}\n", encoding="utf-8")
    visual_tampered = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/production-readiness"
    ).json()["gates"]["performance"]["evidence"]
    assert visual_tampered["video_capture_run_artifact_verified"] is False
    assert visual_tampered["visual_track_artifacts_verified"] is False
    assert visual_tampered["capture_session_artifact_verified"] is False
    visual_summary_path.write_bytes(visual_summary_bytes)

    (job_dir / "observation-v3.json").write_text("{}\n", encoding="utf-8")
    tampered_readiness = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/production-readiness"
    ).json()
    tampered_evidence = tampered_readiness["gates"]["performance"]["evidence"]
    assert tampered_evidence["observation_v3_artifacts_verified"] is False
    assert tampered_evidence["capture_session_artifact_verified"] is False
    tampered_observation_view = TestClient(app).get(observation_view_url)
    assert tampered_observation_view.status_code == 404
    assert tampered_observation_view.json()["code"] == "ARTIFACT_NOT_FOUND"
    tampered_review_bundle = TestClient(app).get(
        f"/api/jobs/{result['job_id']}/review-bundle"
    )
    assert tampered_review_bundle.status_code == 404
    assert tampered_review_bundle.json()["code"] == "ARTIFACT_NOT_FOUND"
