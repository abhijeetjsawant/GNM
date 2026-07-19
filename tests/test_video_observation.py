from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace
import warnings
import zipfile

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

import autoanim_gnm.video_observation as video_observation_module
import autoanim_gnm.api as api_module
import autoanim_gnm.capture_session as capture_session_module
from autoanim_gnm.api import create_app
from autoanim_gnm.artifacts import sha256
from autoanim_gnm.capture_session import write_video_capture_session
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.video_capture import (
    CaptureProvenance,
    CaptureTrack,
    MEDIAPIPE_BLENDSHAPE_NAMES,
    VideoProbe,
    serialize_capture,
)
from autoanim_gnm.video_evidence import (
    REGION_LANDMARKS,
    write_performance_evidence,
)
from autoanim_gnm.video_observation import (
    OBSERVATION_V3_POLICY,
    OBSERVATION_V3_VIEW_SCHEMA_VERSION,
    PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
    PixelObservationTrack,
    analyze_rgb_frames,
    build_observation_v3_view,
    build_observation_v3_summary,
    load_pixel_observations,
    load_verified_observation_v3_summary,
    write_observation_v3_summary,
    write_pixel_observations,
)


def _capture(detected: tuple[bool, ...] = (True, True, False, True)) -> CaptureTrack:
    count = len(detected)
    present = np.asarray(detected, dtype=bool)
    landmarks = np.zeros((count, 478, 3), dtype=np.float32)
    indices = np.arange(478, dtype=np.float32)
    landmarks[:, :, 0] = 0.25 + 0.50 * ((indices % 29) / 28.0)
    landmarks[:, :, 1] = 0.18 + 0.64 * (((indices // 29) % 17) / 16.0)
    landmarks[:, :, 2] = 0.0

    def place(region: str, x0: float, x1: float, y0: float, y1: float) -> None:
        region_indices = np.asarray(REGION_LANDMARKS[region], dtype=np.int64)
        phase = np.arange(len(region_indices), dtype=np.float32)
        landmarks[:, region_indices, 0] = x0 + (x1 - x0) * ((phase % 11) / 10.0)
        landmarks[:, region_indices, 1] = y0 + (y1 - y0) * (((phase // 11) % 5) / 4.0)

    place("mouth", 0.40, 0.60, 0.58, 0.68)
    place("eyes", 0.34, 0.66, 0.38, 0.45)
    place("upperFace", 0.33, 0.67, 0.23, 0.34)
    landmarks[~present] = np.nan
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    transforms[~present] = np.nan
    provenance = CaptureProvenance(
        source_name="observation.mov",
        source_sha256="a" * 64,
        source_bytes=1234,
        model_name="face_landmarker.task",
        model_sha256="b" * 64,
        mediapipe_version="test",
        ffprobe_version="test",
        ffmpeg_version="test",
        codec="test",
        time_base_numerator=1,
        time_base_denominator=30,
        source_start_pts=100,
        display_rotation_degrees=0,
        ffprobe_command=("ffprobe", "${SOURCE}"),
        ffmpeg_command=("ffmpeg", "${SOURCE}"),
    )
    return CaptureTrack(
        source_pts=np.arange(100, 100 + count, dtype=np.int64),
        timestamps_seconds=np.arange(count, dtype=np.float64) / 30.0,
        mediapipe_timestamps_ms=np.rint(
            np.arange(count, dtype=np.float64) * 1000.0 / 30.0
        ).astype(np.int64),
        detected=present,
        landmarks_xyz=landmarks,
        landmark_visibility=np.full((count, 478), np.nan, dtype=np.float32),
        landmark_presence=np.full((count, 478), np.nan, dtype=np.float32),
        blendshape_names=MEDIAPIPE_BLENDSHAPE_NAMES,
        blendshape_scores=np.zeros(
            (count, len(MEDIAPIPE_BLENDSHAPE_NAMES)), dtype=np.float32
        ),
        facial_transforms=transforms,
        face_confidence=np.full(count, np.nan, dtype=np.float32),
        tracking_quality=present.astype(np.float32),
        width=160,
        height=120,
        provenance=provenance,
    )


def _texture(seed: int = 0) -> np.ndarray:
    y, x = np.indices((120, 160), dtype=np.int32)
    red = (x * 7 + y * 3 + seed * 11) % 256
    green = (x * 2 + y * 13 + seed * 17) % 256
    blue = ((x // 3 + y // 2 + seed) % 2) * 180 + 35
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _same_frames(count: int) -> list[np.ndarray]:
    frame = _texture()
    return [frame.copy() for _ in range(count)]


def _assert_track_arrays_equal(
    left: PixelObservationTrack,
    right: PixelObservationTrack,
) -> None:
    for field in fields(PixelObservationTrack):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(left_value, right_value)
        else:
            assert left_value == right_value


def _write_bound_artifacts(
    tmp_path: Path,
    capture: CaptureTrack,
    observations: PixelObservationTrack,
) -> tuple[Path, Path, Path, Path]:
    capture_path, capture_jsonl_path = serialize_capture(tmp_path, capture)
    evidence_path = write_performance_evidence(
        tmp_path / "performance-evidence.json", capture
    )
    arrays_path = write_pixel_observations(
        tmp_path / "pixel-observations.npz", observations
    )
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary_path = write_observation_v3_summary(
        tmp_path / "observation-v3.json",
        capture,
        observations,
        capture_artifact_sha256=digest(capture_path),
        capture_artifact_bytes=capture_path.stat().st_size,
        pixel_observations_sha256=digest(arrays_path),
        pixel_observations_bytes=arrays_path.stat().st_size,
    )
    assert capture_jsonl_path.is_file() and evidence_path.is_file()
    write_video_capture_session(
        tmp_path / "capture-session.json",
        capture,
        observations,
        artifact_paths={
            "capture": capture_path,
            "capture_jsonl": capture_jsonl_path,
            "performance_evidence": evidence_path,
            "pixel_observations": arrays_path,
            "observation_v3": summary_path,
        },
    )
    return capture_path, capture_jsonl_path, arrays_path, summary_path


def _review_bindings(capture: CaptureTrack) -> tuple[dict, dict]:
    artifact_names = {
        "capture": "capture.npz",
        "capture_jsonl": "capture.jsonl",
        "performance_evidence": "performance-evidence.json",
        "pixel_observations": "pixel-observations.npz",
        "observation_v3": "observation-v3.json",
        "capture_session": "capture-session.json",
    }
    generation_contract = {
        "schema_version": "autoanim.viewer-display-binding/1.0",
        "artifact": "viewer_media",
        "source_frame_size": [capture.width, capture.height],
        "proxy_frame_size": [capture.width, capture.height],
        "display_rotation_degrees": 0,
        "sample_aspect_ratio": [1, 1],
        "clean_aperture_crop_ltrb": [0, 0, 0, 0],
        "source_to_display_pixel_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "transcode_policy": (
            "ffmpeg_h264_pts_passthrough_no_geometry_filters_v1"
        ),
    }
    return (
        {
            "chainVerified": True,
            "manifestSha256": "c" * 64,
            "sealSchema": "autoanim.hmac-sha256.v1",
            "sealKeyId": "test-key",
            "retainedSource": {
                "sha256": capture.provenance.source_sha256,
                "bytes": capture.provenance.source_bytes,
            },
            "artifacts": {
                logical_name: {
                    "name": name,
                    "sha256": "d" * 64,
                    "bytes": 1,
                }
                for logical_name, name in artifact_names.items()
            },
        },
        {
            "clockVerified": True,
            "artifact": {
                "logicalName": "viewer_media",
                "name": "source-proxy.mp4",
                "sha256": "e" * 64,
                "bytes": 1,
            },
            "frameCount": capture.frame_count,
            "frameSize": [capture.width, capture.height],
            "displayRotationDegrees": 0,
            "frameTimestampsSeconds": capture.timestamps_seconds.tolist(),
            "sampleAspectRatio": [1, 1],
            "cleanApertureCropLTRB": [0, 0, 0, 0],
            "timestampMaxErrorSeconds": 0.0,
            "generationContract": generation_contract,
            "sourceToDisplayPixelTransform": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
    )


def test_missing_is_null_and_epoch_start_does_not_claim_tracker_state() -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    missing = observations.region_record(2, "mouth")
    assert missing["qualityState"] == "missing"
    assert missing["confidence"] is None
    assert missing["focusMetric"] is None
    assert missing["temporalInnovation"] is None
    assert missing["occlusionState"] == "unknown"
    assert observations.observation_epoch_start.tolist() == [True, False, False, True]
    assert "OBSERVATION_EPOCH_START" in observations.region_record(3, "mouth")[
        "reasonCodes"
    ]
    assert all(
        "TRACK_REINITIALIZATION" not in code
        for code in observations.region_record(3, "mouth")["reasonCodes"]
    )


def test_mouth_only_damage_is_regionally_isolated() -> None:
    capture = _capture((True, True))
    clean_frames = _same_frames(capture.frame_count)
    clean = analyze_rgb_frames(capture, clean_frames)
    mouth_box = clean.roi_boxes_xyxy[0, tuple(REGION_LANDMARKS).index("mouth")]
    x0, y0, x1, y1 = mouth_box.tolist()
    damaged_frames = _same_frames(capture.frame_count)
    damaged_frames[0][y0:y1, x0:x1] = 0
    damaged = analyze_rgb_frames(capture, damaged_frames)
    clean_mouth = clean.region_record(0, "mouth")
    damaged_mouth = damaged.region_record(0, "mouth")
    assert damaged_mouth["confidence"] < clean_mouth["confidence"]
    assert "SEVERE_UNDEREXPOSURE" in damaged_mouth["reasonCodes"]
    # The broad head ROI intentionally overlaps the mouth.  The disjoint eye
    # and upper-face channels must remain bit-exact when only the mouth changes.
    for region_name in ("eyes", "upperFace"):
        region_index = tuple(REGION_LANDMARKS).index(region_name)
        for name in (
            "focus_metric",
            "focus_score",
            "luma_mean",
            "shadow_fraction",
            "highlight_fraction",
            "dynamic_range",
            "confidence",
            "reason_mask",
        ):
            np.testing.assert_array_equal(
                getattr(clean, name)[:, region_index],
                getattr(damaged, name)[:, region_index],
            )


def test_blur_support_is_monotonic_for_the_mouth_region() -> None:
    capture = _capture((True, True, True))
    clean = analyze_rgb_frames(capture, _same_frames(3))
    mouth = tuple(REGION_LANDMARKS).index("mouth")
    x0, y0, x1, y1 = clean.roi_boxes_xyxy[0, mouth].tolist()
    frames: list[np.ndarray] = []
    for kernel in (1, 9, 25):
        frame = _texture()
        if kernel > 1:
            frame[y0:y1, x0:x1] = cv2.GaussianBlur(
                frame[y0:y1, x0:x1], (kernel, kernel), 0
            )
        frames.append(frame)
    result = analyze_rgb_frames(capture, frames)
    scores = result.focus_score[:, mouth].tolist()
    assert scores[0] > scores[1] > scores[2]
    assert result.confidence[0, mouth] > result.confidence[1, mouth]
    assert "BLUR_OR_LOW_DETAIL" in result.region_record(1, "mouth")["reasonCodes"]
    assert "BLUR_OR_LOW_DETAIL" in result.region_record(2, "mouth")["reasonCodes"]


def test_flash_is_not_a_cut_but_structural_edit_starts_an_epoch() -> None:
    capture = _capture((True, True, True))
    base = _texture()
    flash = np.full_like(base, 255)
    flash_result = analyze_rgb_frames(capture, [base, flash, base])
    assert flash_result.cut_candidate.tolist() == [False, False, False]
    assert np.isnan(flash_result.cut_thumbnail_zncc[1:]).all()
    assert flash_result.photometric_discontinuity_candidate.tolist() == [
        False,
        True,
        True,
    ]
    assert flash_result.observation_epoch_start.tolist() == [True, True, True]

    hard_capture = _capture((True, True))
    dark = ((base.astype(np.uint16) // 4) + 8).astype(np.uint8)
    inverted = 255 - dark
    hard = analyze_rgb_frames(hard_capture, [dark, inverted])
    assert hard.cut_histogram_distance[1] >= 0.55
    assert hard.cut_thumbnail_mad[1] >= 0.18
    assert hard.cut_thumbnail_zncc[1] <= 0.45
    assert hard.cut_candidate.tolist() == [False, True]
    assert hard.observation_epoch_start.tolist() == [True, True]


def test_offscreen_region_gap_is_unknown_and_resets_temporal_history() -> None:
    capture = _capture((True, True, True))
    landmarks = capture.landmarks_xyz.copy()
    mouth_indices = np.asarray(REGION_LANDMARKS["mouth"], dtype=np.int64)
    landmarks[1, mouth_indices, :2] = 2.0
    capture = replace(capture, landmarks_xyz=landmarks)
    observations = analyze_rgb_frames(capture, _same_frames(3))
    observations.validate_capture(capture)
    assert observations.region_record(1, "mouth")["qualityState"] == "unknown"
    returned = observations.region_record(2, "mouth")
    assert returned["temporalInnovation"] is None
    assert returned["flowConsistency"] is None
    assert "FLOW_UNAVAILABLE" in returned["reasonCodes"]


def test_textureless_flow_is_unavailable_not_zero_or_perfect() -> None:
    capture = _capture((True, True))
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    observations = analyze_rgb_frames(capture, [flat.copy(), flat.copy()])
    record = observations.region_record(1, "mouth")
    assert record["temporalInnovation"] == 0.0
    assert record["flowConsistency"] is None
    assert "FLOW_UNAVAILABLE" in record["reasonCodes"]


def test_analysis_and_npz_roundtrip_are_exactly_deterministic(tmp_path: Path) -> None:
    capture = _capture()
    first = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    second = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    _assert_track_arrays_equal(first, second)
    path = write_pixel_observations(tmp_path / "pixels.npz", first)
    loaded = load_pixel_observations(path)
    _assert_track_arrays_equal(first, loaded)
    assert np.nanmax(first.confidence) <= PIXEL_DIAGNOSTIC_CONFIDENCE_CAP


def test_compact_summary_reconstructs_and_tampering_fails(tmp_path: Path) -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    capture_path, _, arrays_path, summary_path = _write_bound_artifacts(
        tmp_path, capture, observations
    )
    payload = load_verified_observation_v3_summary(
        summary_path,
        pixel_observations_path=arrays_path,
        capture_artifact_path=capture_path,
        expected_capture=capture,
    )
    assert payload["policy"] == OBSERVATION_V3_POLICY
    assert "frames" not in payload
    assert payload["consumedByRetargeting"] is False
    assert payload["claims"]["productionValidated"] is False
    assert payload["summary"]["regions"]["mouth"]["strongFrames"] == 0
    assert summary_path.stat().st_size < 64 * 1024

    payload["summary"]["cutCandidateFrames"] += 1
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reconstruct"):
        load_verified_observation_v3_summary(
            summary_path,
            pixel_observations_path=arrays_path,
            capture_artifact_path=capture_path,
            expected_capture=capture,
        )


def test_observation_v3_view_exposes_exact_diagnostics_without_motion_authority(
    tmp_path: Path,
) -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    capture_path, _, arrays_path, summary_path = _write_bound_artifacts(
        tmp_path, capture, observations
    )
    summary = load_verified_observation_v3_summary(
        summary_path,
        pixel_observations_path=arrays_path,
        capture_artifact_path=capture_path,
        expected_capture=capture,
    )

    evidence_binding, display_binding = _review_bindings(capture)
    view = build_observation_v3_view(
        capture,
        observations,
        summary,
        evidence_binding=evidence_binding,
        display_binding=display_binding,
    )

    assert view["schemaVersion"] == OBSERVATION_V3_VIEW_SCHEMA_VERSION
    assert view["consumedByRetargeting"] is False
    assert view["claims"] == {
        "derivedFromVerifiedSealedEvidence": True,
        "changesFinalGNMMotion": False,
        "confidenceCalibrated": False,
        "occlusionValidated": False,
        "identityContinuityValidated": False,
        "productionValidated": False,
    }
    assert view["source"]["frameSize"] == [160, 120]
    assert [frame["sourcePTS"] for frame in view["frames"]] == [100, 101, 102, 103]
    assert [frame["timestampSeconds"] for frame in view["frames"]] == pytest.approx(
        [0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0]
    )
    missing_mouth = view["frames"][2]["regions"]["mouth"]
    assert missing_mouth["qualityState"] == "missing"
    assert missing_mouth["confidence"] is None
    assert missing_mouth["roiBoxXYXY"] is None
    assert missing_mouth["occlusionState"] == "unknown"
    assert view["frames"][3]["observationEpochStart"] is True
    assert "OBSERVATION_EPOCH_START" in view["frames"][3]["regions"]["eyes"][
        "reasonCodes"
    ]
    assert "controls" not in json.dumps(view).lower()

    forged = dict(summary)
    forged["consumedByRetargeting"] = True
    with pytest.raises(ValueError, match="not a verified"):
        build_observation_v3_view(
            capture,
            observations,
            forged,
            evidence_binding=evidence_binding,
            display_binding=display_binding,
        )


def test_api_reconstructs_observation_v3_view_from_allowlisted_sealed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    store = app.state.service.store
    job_id, job_dir, _, manifest = store.start("video_performance", source, {})
    capture = replace(
        _capture(),
        provenance=replace(
            _capture().provenance,
            source_name=source.name,
            source_sha256=sha256(source),
            source_bytes=source.stat().st_size,
        ),
    )
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    _write_bound_artifacts(job_dir, capture, observations)
    (job_dir / "source-proxy.mp4").write_bytes(b"proxy")
    monkeypatch.setattr(
        capture_session_module,
        "load_pixel_observations",
        lambda path: (_ for _ in ()).throw(
            AssertionError("API must not decompress pixel observations twice")
        ),
    )
    monkeypatch.setattr(
        api_module,
        "probe_video",
        lambda path: VideoProbe(
            path=Path(path),
            width=capture.width,
            height=capture.height,
            codec="h264",
            time_base_numerator=1,
            time_base_denominator=30,
            source_pts=np.arange(capture.frame_count, dtype=np.int64),
            timestamps_seconds=capture.timestamps_seconds,
            mediapipe_timestamps_ms=capture.mediapipe_timestamps_ms,
            display_rotation_degrees=0,
            ffprobe_command=("ffprobe",),
        ),
    )
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "video_performance",
            "status": "succeeded",
            "capture": {
                "frames": capture.frame_count,
                "width": capture.width,
                "height": capture.height,
            },
            "artifacts": {
                "capture": "capture.npz",
                "capture_jsonl": "capture.jsonl",
                "performance_evidence": "performance-evidence.json",
                "pixel_observations": "pixel-observations.npz",
                "observation_v3": "observation-v3.json",
                "capture_session": "capture-session.json",
                "viewer_media": "source-proxy.mp4",
            },
            "viewer": {
                "clock_artifact": "viewer_media",
                "display_geometry": {
                    "schema_version": "autoanim.viewer-display-binding/1.0",
                    "artifact": "viewer_media",
                    "source_frame_size": [capture.width, capture.height],
                    "proxy_frame_size": [capture.width, capture.height],
                    "display_rotation_degrees": 0,
                    "sample_aspect_ratio": [1, 1],
                    "clean_aperture_crop_ltrb": [0, 0, 0, 0],
                    "source_to_display_pixel_transform": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "transcode_policy": (
                        "ffmpeg_h264_pts_passthrough_no_geometry_filters_v1"
                    ),
                },
            },
        },
        {},
    )

    response = TestClient(app).get(f"/api/jobs/{job_id}/observation-v3-view")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    payload = response.json()
    assert payload["schemaVersion"] == OBSERVATION_V3_VIEW_SCHEMA_VERSION
    assert payload["frames"][2]["regions"]["mouth"]["confidence"] is None
    assert payload["claims"]["changesFinalGNMMotion"] is False
    assert payload["claims"]["derivedFromVerifiedSealedEvidence"] is True
    assert payload["evidenceBinding"]["chainVerified"] is True
    assert payload["display"]["clockVerified"] is True
    assert str(tmp_path) not in response.text

    missing = TestClient(app).get(
        "/api/jobs/not-a-video-job/observation-v3-view"
    )
    assert missing.status_code == 404

    result_path = job_dir / "result.json"
    mismatched = store.read(job_id)
    mismatched["input"] = {**mismatched["input"], "sha256": "d" * 64}
    store._write_manifest(result_path, mismatched)
    source_rejected = TestClient(app).get(
        f"/api/jobs/{job_id}/observation-v3-view"
    )
    assert source_rejected.status_code == 409
    assert source_rejected.json()["code"] == "INTEGRITY_FAILED"

    unsealed = json.loads(result_path.read_text(encoding="utf-8"))
    unsealed.pop("integrity")
    result_path.write_text(json.dumps(unsealed), encoding="utf-8")
    rejected = TestClient(app).get(f"/api/jobs/{job_id}/observation-v3-view")
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "INTEGRITY_UNSEALED"


def test_duplicate_json_and_npz_schema_tampering_fail_closed(tmp_path: Path) -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    capture_path, _, arrays_path, summary_path = _write_bound_artifacts(
        tmp_path, capture, observations
    )
    summary_path.write_text('{"schemaVersion": 1, "schemaVersion": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON member"):
        load_verified_observation_v3_summary(
            summary_path,
            pixel_observations_path=arrays_path,
            capture_artifact_path=capture_path,
            expected_capture=capture,
        )

    with np.load(arrays_path, allow_pickle=False) as values:
        tampered = {name: values[name] for name in values.files}
    tampered["source_pts"] = tampered["source_pts"].astype(np.int32)
    np.savez_compressed(arrays_path, **tampered)
    with pytest.raises(AutoAnimError, match="dtype"):
        load_pixel_observations(arrays_path)


def test_observation_v3_view_rejects_oversized_take_before_dense_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    store = app.state.service.store
    job_id, job_dir, _, manifest = store.start("video_performance", source, {})
    artifact_names = {
        "capture": "capture.npz",
        "capture_jsonl": "capture.jsonl",
        "performance_evidence": "performance-evidence.json",
        "pixel_observations": "pixel-observations.npz",
        "observation_v3": "observation-v3.json",
        "capture_session": "capture-session.json",
    }
    for name in artifact_names.values():
        (job_dir / name).write_bytes(b"bounded")
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "video_performance",
            "status": "succeeded",
            "artifacts": artifact_names,
        },
        {},
    )
    monkeypatch.setattr(
        api_module,
        "load_capture_npz",
        lambda path: SimpleNamespace(frame_count=1_801),
    )
    monkeypatch.setattr(
        api_module,
        "load_pixel_observations",
        lambda path: (_ for _ in ()).throw(
            AssertionError("oversized view must fail before dense NPZ decode")
        ),
    )

    response = TestClient(app).get(
        f"/api/jobs/{job_id}/observation-v3-view"
    )
    assert response.status_code == 413
    assert response.json()["code"] == "LIMIT_EXCEEDED"


def test_interactive_review_accepts_exact_1800_frame_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    store = app.state.service.store
    job_id, job_dir, _, manifest = store.start("video_performance", source, {})
    artifact_names = {
        "glb": "performance.glb",
        "capture": "capture.npz",
        "capture_jsonl": "capture.jsonl",
        "performance_evidence": "performance-evidence.json",
        "pixel_observations": "pixel-observations.npz",
        "observation_v3": "observation-v3.json",
        "capture_session": "capture-session.json",
        "viewer_media": "source-proxy.mp4",
    }
    for name in artifact_names.values():
        (job_dir / name).write_bytes(b"bounded")
    display_geometry = {
        "schema_version": "autoanim.viewer-display-binding/1.0",
        "artifact": "viewer_media",
        "source_frame_size": [64, 48],
        "proxy_frame_size": [64, 48],
        "display_rotation_degrees": 0,
        "sample_aspect_ratio": [1, 1],
        "clean_aperture_crop_ltrb": [0, 0, 0, 0],
        "source_to_display_pixel_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "transcode_policy": (
            "ffmpeg_h264_pts_passthrough_no_geometry_filters_v1"
        ),
    }
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "video_performance",
            "status": "succeeded",
            "capture": {"frames": 1_800, "width": 64, "height": 48},
            "artifacts": artifact_names,
            "viewer": {
                "status": "ready",
                "clock_artifact": "viewer_media",
                "display_geometry": display_geometry,
            },
        },
        {},
    )
    timestamps = np.arange(1_800, dtype=np.float64) / 30.0
    fake_capture = SimpleNamespace(
        frame_count=1_800,
        width=64,
        height=48,
        timestamps_seconds=timestamps,
        source_pts=np.arange(1_800, dtype=np.int64),
        provenance=SimpleNamespace(
            source_sha256=sha256(source),
            source_bytes=source.stat().st_size,
        ),
    )

    class FakeObservations:
        def validate_capture(self, capture: object) -> None:
            assert capture is fake_capture

    class FakeDecoder:
        def __init__(self, path: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            self.index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def get(self, prop: int) -> float:
            return float(self.index + 1)

        def release(self) -> None:
            return None

    monkeypatch.setattr(api_module, "load_capture_npz", lambda path: fake_capture)
    monkeypatch.setattr(
        api_module, "load_pixel_observations", lambda path: FakeObservations()
    )
    monkeypatch.setattr(
        api_module, "load_verified_capture_jsonl", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        api_module, "load_verified_performance_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        api_module,
        "load_verified_observation_v3_summary",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        api_module,
        "load_verified_video_capture_session",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        api_module,
        "build_observation_v3_view",
        lambda capture, observations, summary, **kwargs: {
            "schemaVersion": OBSERVATION_V3_VIEW_SCHEMA_VERSION,
            "frameCount": capture.frame_count,
        },
    )
    monkeypatch.setattr(
        api_module,
        "probe_video",
        lambda path: VideoProbe(
            path=Path(path),
            width=64,
            height=48,
            codec="h264",
            time_base_numerator=1,
            time_base_denominator=30,
            source_pts=np.arange(1_800, dtype=np.int64),
            timestamps_seconds=timestamps,
            mediapipe_timestamps_ms=np.rint(timestamps * 1_000).astype(np.int64),
            display_rotation_degrees=0,
            ffprobe_command=("ffprobe",),
        ),
    )
    monkeypatch.setattr(api_module.cv2, "VideoCapture", FakeDecoder)
    monkeypatch.setattr(
        app.state.service,
        "production_readiness",
        lambda requested_job_id: {
            "status": "blocked",
            "passed_required_gate_count": 0,
            "required_gate_count": 1,
            "failures": ["qualification required"],
            "publishable": False,
        },
    )
    client = TestClient(app)

    viewer = client.get(f"/api/jobs/{job_id}/viewer")
    assert viewer.status_code == 200
    assert f"/api/jobs/{job_id}/observation-v3-view" in viewer.text
    assert '"observation_review": "available"' in viewer.text
    observation = client.get(f"/api/jobs/{job_id}/observation-v3-view")
    assert observation.status_code == 200
    assert observation.json()["frameCount"] == 1_800
    last_frame = client.get(f"/api/jobs/{job_id}/review-frames/1799.png")
    assert last_frame.status_code == 200
    assert last_frame.headers["x-autoanim-frame-index"] == "1799"
    rejected = client.get(f"/api/jobs/{job_id}/review-frames/1800.png")
    assert rejected.status_code == 400
    assert rejected.json()["code"] == "INPUT_INVALID"


def test_pixel_observation_npz_rejects_duplicate_members(tmp_path: Path) -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    arrays_path = write_pixel_observations(
        tmp_path / "pixel-observations.npz", observations
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(arrays_path, mode="a") as archive:
            archive.writestr("source_pts.npy", b"ambiguous")
    with pytest.raises(AutoAnimError, match="duplicate members"):
        load_pixel_observations(arrays_path)


def test_pixel_observation_expanded_size_limit_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    arrays_path = write_pixel_observations(
        tmp_path / "pixel-observations.npz", observations
    )
    monkeypatch.setattr(
        video_observation_module,
        "MAX_PIXEL_OBSERVATION_UNCOMPRESSED_BYTES",
        1,
    )
    with pytest.raises(AutoAnimError, match="exceeds its limit"):
        load_pixel_observations(arrays_path)


def test_npz_availability_and_configuration_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    arrays_path = write_pixel_observations(
        tmp_path / "pixel-observations.npz", observations
    )
    with np.load(arrays_path, allow_pickle=False) as values:
        tampered = {name: values[name] for name in values.files}
    confidence = tampered["confidence"].copy()
    confidence[2, 0] = 0.0
    tampered["confidence"] = confidence
    np.savez_compressed(arrays_path, **tampered)
    with pytest.raises(AutoAnimError, match="availability"):
        load_pixel_observations(arrays_path)

    arrays_path = write_pixel_observations(arrays_path, observations)
    with np.load(arrays_path, allow_pickle=False) as values:
        tampered = {name: values[name] for name in values.files}
    configuration = json.loads(str(tampered["configuration_json"].item()))
    configuration["cutThresholds"]["thumbnailMADMinimum"] = 0.0
    tampered["configuration_json"] = np.asarray(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(arrays_path, **tampered)
    with pytest.raises(AutoAnimError, match="configuration"):
        load_pixel_observations(arrays_path)

    arrays_path = write_pixel_observations(arrays_path, observations)
    with np.load(arrays_path, allow_pickle=False) as values:
        tampered = {name: values[name] for name in values.files}
    canonical = str(tampered["configuration_json"].item())
    tampered["configuration_json"] = np.asarray(
        '{"analyzerVersion":"forged",' + canonical[1:]
    )
    np.savez_compressed(arrays_path, **tampered)
    with pytest.raises(AutoAnimError, match="Duplicate JSON member"):
        load_pixel_observations(arrays_path)


def test_capture_detection_roi_and_reason_tampering_are_rejected() -> None:
    capture = _capture((True, True, True, True))
    observations = analyze_rgb_frames(capture, _same_frames(4))
    with pytest.raises(ValueError, match="detection state"):
        observations.validate_capture(_capture((True, True, False, True)))

    boxes = observations.roi_boxes_xyxy.copy()
    boxes[0, 0, 0] += 1
    forged_roi = replace(observations, roi_boxes_xyxy=boxes)
    with pytest.raises(ValueError, match="ROI differs"):
        forged_roi.validate_capture(capture)

    frame = _texture()
    mouth = tuple(REGION_LANDMARKS).index("mouth")
    x0, y0, x1, y1 = observations.roi_boxes_xyxy[0, mouth].tolist()
    frame[y0:y1, x0:x1] = 0
    damaged = analyze_rgb_frames(_capture((True,)), [frame])
    reasons = damaged.reason_mask.copy()
    reasons[0, mouth] = 0
    with pytest.raises(ValueError, match="reasons are inconsistent|reason mask"):
        replace(damaged, reason_mask=reasons)
