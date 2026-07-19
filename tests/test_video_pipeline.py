import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from autoanim_gnm.api import create_app
from autoanim_gnm.animation import calibrate_lip_contact
from autoanim_gnm.calibrated_retarget import CalibratedRetargeter
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.video_capture import (
    MEDIAPIPE_BLENDSHAPE_NAMES,
    load_capture_npz,
    probe_video,
)
from autoanim_gnm.video_pipeline import (
    MAX_PROXY_PTS_ERROR_SECONDS,
    _export_static_performance_glb,
    _final_output_retention_metrics,
)
from autoanim_gnm.video_retarget import retarget_capture


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
A2F_ASSETS = CACHE / "a2f-claire"
CREMA_D_ANGRY = FIXTURES / "crema-d-1001-dfa-ang.flv"
CREMA_D_ANGRY_SHA256 = "10dc3fd1f2bc8203657431598bd7dc9312462008f93d08fda786043ae6a8d2f4"
RETAINED_CREMA_JOB = Path(
    os.environ.get(
        "AUTOANIM_RETAINED_CREMA_JOB",
        "artifacts/jobs/01kxtx72xy7z1hbmv747hgjzdc",
    )
)


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
def test_retained_crema_capture_neutral_audit_and_final_geometry_retention() -> None:
    source = RETAINED_CREMA_JOB / "input.flv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == CREMA_D_ANGRY_SHA256
    capture = load_capture_npz(RETAINED_CREMA_JOB / "capture.npz")
    assert capture.provenance.source_sha256 == CREMA_D_ANGRY_SHA256
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
    assert result["capture"]["production_validated"] is False
    assert result["retargeting"]["backend"] == (
        "geometry_calibrated_dense_contact_aperture_v3"
    )
    assert result["retargeting"]["geometry_calibrated"] is True
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
    assert result["viewer"]["clock_artifact"] == "viewer_media"
    assert result["artifacts"]["retarget_calibration"]["media_type"] == (
        "application/octet-stream"
    )
    assert result["artifacts"]["viewer_media"]["media_type"] == "video/mp4"

    with np.load(job_dir / "capture.npz", allow_pickle=False) as capture:
        with np.load(job_dir / "performance.npz", allow_pickle=False) as performance:
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
