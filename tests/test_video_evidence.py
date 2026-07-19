from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.video_capture import (
    CaptureProvenance,
    CaptureTrack,
    MEDIAPIPE_BLENDSHAPE_NAMES,
)
from autoanim_gnm.video_evidence import (
    GEOMETRY_ONLY_CONFIDENCE_CAP,
    PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
    MOUTH_LANDMARKS,
    build_performance_evidence,
    load_verified_performance_evidence,
    write_performance_evidence,
)


def _track() -> CaptureTrack:
    count = 3
    detected = np.asarray((True, False, True), dtype=bool)
    landmarks = np.full((count, 478, 3), 0.5, dtype=np.float32)
    landmarks[:, :, 2] = 0.0
    landmarks[~detected] = np.nan
    # The third frame has a returned face, but its mouth support lies outside
    # the conservative image bound.  Other regions remain well supported.
    landmarks[2, np.asarray(MOUTH_LANDMARKS), 0] = 1.2
    visibility = np.full((count, 478), np.nan, dtype=np.float32)
    presence = np.full((count, 478), np.nan, dtype=np.float32)
    visibility[2] = 0.9
    presence[2] = 0.9
    scores = np.zeros((count, len(MEDIAPIPE_BLENDSHAPE_NAMES)), dtype=np.float32)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    in_frame = np.mean(
        (landmarks[2, :, 0] >= -0.05)
        & (landmarks[2, :, 0] <= 1.05)
        & (landmarks[2, :, 1] >= -0.05)
        & (landmarks[2, :, 1] <= 1.05)
    )
    time_base = Fraction(1001, 30_000)
    provenance = CaptureProvenance(
        source_name="fractional.mov",
        source_sha256="a" * 64,
        source_bytes=123,
        model_name="face_landmarker.task",
        model_sha256="b" * 64,
        mediapipe_version="test",
        ffprobe_version="test",
        ffmpeg_version="test",
        codec="test",
        time_base_numerator=time_base.numerator,
        time_base_denominator=time_base.denominator,
        source_start_pts=100,
        display_rotation_degrees=0,
        ffprobe_command=("ffprobe",),
        ffmpeg_command=("ffmpeg",),
    )
    return CaptureTrack(
        source_pts=np.asarray((100, 101, 102), dtype=np.int64),
        timestamps_seconds=np.asarray(
            (0.0, float(time_base), float(2 * time_base)), dtype=np.float64
        ),
        mediapipe_timestamps_ms=np.asarray((0, 33, 67), dtype=np.int64),
        detected=detected,
        landmarks_xyz=landmarks,
        landmark_visibility=visibility,
        landmark_presence=presence,
        blendshape_names=MEDIAPIPE_BLENDSHAPE_NAMES,
        blendshape_scores=scores,
        facial_transforms=transforms,
        face_confidence=np.asarray((np.nan, np.nan, 0.9), dtype=np.float32),
        tracking_quality=np.asarray((1.0, 0.0, in_frame), dtype=np.float32),
        width=1920,
        height=1080,
        provenance=provenance,
    )


def test_observation_v2_preserves_exact_pts_and_fractional_project_time() -> None:
    evidence = build_performance_evidence(_track())
    assert evidence["schemaVersion"] == PERFORMANCE_EVIDENCE_SCHEMA_VERSION
    assert evidence["policy"] == "observation_only_no_motion_effect"
    assert evidence["sourceMode"] == "video_follow"
    assert evidence["consumedByRetargeting"] is False
    assert evidence["confidenceContract"]["unknownIsNotNeutral"] is True
    assert evidence["source"]["sourceTimeBase"] == [1001, 30_000]
    assert [frame["sourcePTS"] for frame in evidence["frames"]] == [100, 101, 102]
    assert [frame["projectTick"] for frame in evidence["frames"]] == [0, 1602, 3203]
    assert evidence["frames"][1]["projectTickExactRational"] == [8008, 5]
    assert evidence["frames"][1]["projectTickWasRounded"] is True
    assert evidence["projectClock"]["roundedFrameCount"] == 2


def test_missing_observation_is_null_and_zero_control_is_not_called_neutral() -> None:
    evidence = build_performance_evidence(_track())
    observed = evidence["frames"][0]
    missing = evidence["frames"][1]
    assert observed["observationState"] == "observed"
    assert observed["neutralityState"] == "unknown"
    assert observed["regions"]["mouth"]["trackerControls"]["jawOpen"] == 0.0
    assert observed["regions"]["mouth"]["neutralityState"] == "unknown"
    assert missing["observationState"] == "missing"
    assert missing["trackerInFrameFraction"] is None
    for region in missing["regions"].values():
        assert region["observationState"] == "missing"
        assert region["semanticState"] == "unknown"
        assert region["neutralityState"] == "unknown"
        assert region["confidence"] is None
        assert region["trackerControls"] is None


def test_region_confidence_is_conservative_and_does_not_mutate_capture() -> None:
    track = _track()
    landmarks_before = track.landmarks_xyz.copy()
    scores_before = track.blendshape_scores.copy()
    evidence = build_performance_evidence(track)
    first = evidence["frames"][0]["regions"]
    third = evidence["frames"][2]["regions"]
    assert first["mouth"]["confidence"] == GEOMETRY_ONLY_CONFIDENCE_CAP
    assert first["mouth"]["confidenceTier"] == "review"
    assert first["mouth"]["confidenceSource"].endswith("capped_at_0.5")
    assert third["mouth"]["observationState"] == "observed"
    assert third["mouth"]["confidence"] == 0.0
    assert third["eyes"]["confidence"] == pytest.approx(0.9)
    assert third["upperFace"]["confidence"] == pytest.approx(0.9)
    assert 0.0 < third["head"]["confidence"] <= 0.9
    np.testing.assert_array_equal(track.landmarks_xyz, landmarks_before)
    np.testing.assert_array_equal(track.blendshape_scores, scores_before)


def test_performance_evidence_json_is_deterministic_and_nan_free(tmp_path: Path) -> None:
    first = write_performance_evidence(tmp_path / "first.json", _track())
    second = write_performance_evidence(tmp_path / "second.json", _track())
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["summary"]["observedFrames"] == 2
    assert payload["summary"]["missingFrames"] == 1
    assert "NaN" not in first.read_text(encoding="utf-8")
    verified = load_verified_performance_evidence(
        first,
        expected_source_sha256="a" * 64,
        expected_frame_count=3,
        expected_capture=_track(),
    )
    assert verified["frames"][1]["observationState"] == "missing"


def test_observation_v2_rejects_an_internally_valid_different_capture_clock(
    tmp_path: Path,
) -> None:
    track = _track()
    path = write_performance_evidence(tmp_path / "evidence.json", track)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"]["sourceStartPTS"] += 10
    for frame in payload["frames"]:
        frame["sourcePTS"] += 10
    path.write_text(json.dumps(payload), encoding="utf-8")

    # The document remains self-consistent, but it is not evidence for this
    # Capture v1 timeline and must fail the portable verifier boundary.
    load_verified_performance_evidence(
        path,
        expected_source_sha256="a" * 64,
        expected_frame_count=3,
    )
    with pytest.raises(ValueError, match="expected capture"):
        load_verified_performance_evidence(
            path,
            expected_source_sha256="a" * 64,
            expected_frame_count=3,
            expected_capture=track,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["source"].update(sha256="c" * 64), "source take"),
        (lambda payload: payload["frames"][1].update(projectTick=1603), "invalid timing"),
        (
            lambda payload: payload["frames"][1]["regions"]["mouth"].update(
                confidence=0.0
            ),
            "invalid mouth state",
        ),
        (lambda payload: payload["summary"].update(missingFrames=0), "summary"),
    ],
)
def test_verified_evidence_rejects_tampered_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    path = write_performance_evidence(tmp_path / "evidence.json", _track())
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_verified_performance_evidence(
            path,
            expected_source_sha256="a" * 64,
            expected_frame_count=3,
        )


def test_evidence_rejects_capture_timing_that_does_not_match_pts() -> None:
    track = _track()
    with pytest.raises(ValueError, match="timestamps do not match"):
        build_performance_evidence(
            CaptureTrack(
                source_pts=track.source_pts,
                timestamps_seconds=np.asarray((0.0, 0.04, 0.08), dtype=np.float64),
                mediapipe_timestamps_ms=track.mediapipe_timestamps_ms,
                detected=track.detected,
                landmarks_xyz=track.landmarks_xyz,
                landmark_visibility=track.landmark_visibility,
                landmark_presence=track.landmark_presence,
                blendshape_names=track.blendshape_names,
                blendshape_scores=track.blendshape_scores,
                facial_transforms=track.facial_transforms,
                face_confidence=track.face_confidence,
                tracking_quality=track.tracking_quality,
                width=track.width,
                height=track.height,
                provenance=track.provenance,
            )
        )
