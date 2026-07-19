from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import warnings
import zipfile

import cv2
import numpy as np
import pytest

import autoanim_gnm.video_observation as video_observation_module
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.video_capture import (
    CaptureProvenance,
    CaptureTrack,
    MEDIAPIPE_BLENDSHAPE_NAMES,
    serialize_capture,
)
from autoanim_gnm.video_evidence import (
    REGION_LANDMARKS,
    write_performance_evidence,
)
from autoanim_gnm.video_observation import (
    OBSERVATION_V3_POLICY,
    PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
    PixelObservationTrack,
    analyze_rgb_frames,
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
    return capture_path, capture_jsonl_path, arrays_path, summary_path


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
