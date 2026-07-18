from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoanim_gnm.video_capture import (
    CaptureProvenance,
    CaptureTrack,
    MEDIAPIPE_BLENDSHAPE_NAMES,
    VideoProbe,
    capture_video,
    decoded_video_frames,
    load_capture_npz,
    probe_video,
    serialize_capture,
)
from autoanim_gnm.video_retarget import (
    FAST_CONTACT_CONTROLS,
    filter_blendshapes,
    retarget_capture,
    serialize_performance,
)
from autoanim_gnm.a2f import ARKitGNMRetargeter


CACHE = Path(".cache/autoanim_gnm")
MODEL = CACHE / "face_landmarker.task"
ASTRONAUT = CACHE / "fixtures/astronaut.png"


def _provenance() -> CaptureProvenance:
    return CaptureProvenance(
        source_name="synthetic.mp4",
        source_sha256="1" * 64,
        source_bytes=1234,
        model_name="face_landmarker.task",
        model_sha256="2" * 64,
        mediapipe_version="test",
        ffprobe_version="ffprobe test",
        ffmpeg_version="ffmpeg test",
        codec="test",
        time_base_numerator=1,
        time_base_denominator=30,
        source_start_pts=100,
        display_rotation_degrees=0,
        ffprobe_command=("ffprobe", "synthetic.mp4"),
        ffmpeg_command=("ffmpeg", "synthetic.mp4"),
    )


def _capture_track(*, include_missing: bool = False) -> CaptureTrack:
    count = 5
    detected = np.ones(count, dtype=bool)
    if include_missing:
        detected[3] = False
    landmarks = np.zeros((count, 478, 3), dtype=np.float32)
    landmarks[~detected] = np.nan
    visibility = np.full((count, 478), np.nan, dtype=np.float32)
    presence = np.full((count, 478), np.nan, dtype=np.float32)
    scores = np.zeros((count, len(MEDIAPIPE_BLENDSHAPE_NAMES)), dtype=np.float32)
    columns = {name: index for index, name in enumerate(MEDIAPIPE_BLENDSHAPE_NAMES)}
    scores[:, columns["eyeBlinkLeft"]] = (0.0, 0.98, 0.12, 0.0, 0.0)
    scores[:, columns["eyeBlinkRight"]] = (0.0, 0.92, 0.10, 0.0, 0.0)
    scores[:, columns["mouthClose"]] = (0.0, 0.05, 0.96, 0.0, 0.0)
    scores[:, columns["jawOpen"]] = (0.0, 0.80, 0.04, 0.0, 0.0)
    scores[:, columns["mouthSmileLeft"]] = (0.0, 0.7, 0.8, 0.7, 0.0)
    scores[:, columns["eyeLookOutLeft"]] = (0.2, 0.2, 0.6, 0.3, 0.1)
    scores[:, columns["eyeLookInRight"]] = (0.2, 0.2, 0.5, 0.3, 0.1)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    for index in range(count):
        transforms[index, :3, :3] = Rotation.from_euler(
            "y", 4.0 * index, degrees=True
        ).as_matrix()
        transforms[index, :3, 3] = (index, index * 0.5, -index * 0.25)
    quality = np.where(detected, 1.0, 0.0).astype(np.float32)
    return CaptureTrack(
        source_pts=np.arange(100, 105, dtype=np.int64),
        timestamps_seconds=np.asarray((0, 1 / 30, 2 / 30, 3 / 30, 4 / 30)),
        mediapipe_timestamps_ms=np.asarray((0, 33, 67, 100, 133)),
        detected=detected,
        landmarks_xyz=landmarks,
        landmark_visibility=visibility,
        landmark_presence=presence,
        blendshape_names=MEDIAPIPE_BLENDSHAPE_NAMES,
        blendshape_scores=scores,
        facial_transforms=transforms,
        face_confidence=np.full(count, np.nan, dtype=np.float32),
        tracking_quality=quality,
        width=640,
        height=480,
        provenance=_provenance(),
    )


def _capture_with_scores(track: CaptureTrack, scores: np.ndarray) -> CaptureTrack:
    return CaptureTrack(
        source_pts=track.source_pts,
        timestamps_seconds=track.timestamps_seconds,
        mediapipe_timestamps_ms=track.mediapipe_timestamps_ms,
        detected=track.detected,
        landmarks_xyz=track.landmarks_xyz,
        landmark_visibility=track.landmark_visibility,
        landmark_presence=track.landmark_presence,
        blendshape_names=track.blendshape_names,
        blendshape_scores=scores,
        facial_transforms=track.facial_transforms,
        face_confidence=track.face_confidence,
        tracking_quality=track.tracking_quality,
        width=track.width,
        height=track.height,
        provenance=track.provenance,
    )


class _FakeRetargeter:
    def retarget_sequence(self, weights: np.ndarray, pose_names: tuple[str, ...]) -> np.ndarray:
        assert weights.shape[1] == len(pose_names)
        result = np.zeros((len(weights), 383), dtype=np.float32)
        columns = {name: index for index, name in enumerate(pose_names)}
        result[:, 0] = weights[:, columns["eyeBlinkLeft"]]
        result[:, 2] = weights[:, columns["eyeLookOutLeft"]]
        result[:, 200] = weights[:, columns["jawOpen"]]
        result[:, 201] = weights[:, columns["mouthClose"]]
        return result


def test_capture_schema_is_immutable_and_rejects_nonmonotonic_pts() -> None:
    track = _capture_track()
    assert not track.landmarks_xyz.flags.writeable
    assert not track.blendshape_scores.flags.writeable
    with pytest.raises(ValueError):
        track.blendshape_scores[0, 0] = 1
    with pytest.raises(ValueError, match="strictly increasing"):
        VideoProbe(
            path=Path("x.mp4"),
            width=16,
            height=16,
            codec="x",
            time_base_numerator=1,
            time_base_denominator=30,
            source_pts=np.asarray((10, 10)),
            timestamps_seconds=np.asarray((0, 1 / 30)),
            mediapipe_timestamps_ms=np.asarray((0, 33)),
            display_rotation_degrees=0,
            ffprobe_command=("ffprobe",),
        )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg tools unavailable",
)
def test_real_ffmpeg_decode_has_one_rgb_frame_per_exact_pts(tmp_path: Path) -> None:
    video = tmp_path / "timing.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=7:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ),
        check=True,
    )
    probe = probe_video(video)
    assert probe.frame_count == 7
    exact = [
        float(Fraction(int(pts - probe.source_pts[0])) * probe.time_base)
        for pts in probe.source_pts
    ]
    np.testing.assert_array_equal(probe.timestamps_seconds, exact)
    with decoded_video_frames(probe) as frames:
        decoded = list(frames)
    assert [frame.source_pts for frame in decoded] == probe.source_pts.tolist()
    assert [frame.mediapipe_timestamp_ms for frame in decoded] == (
        probe.mediapipe_timestamps_ms.tolist()
    )
    assert all(frame.rgb.shape == (64, 96, 3) for frame in decoded)


def test_filter_preserves_blinks_and_mouth_contacts_without_broad_smoothing() -> None:
    track = _capture_track()
    filtered = filter_blendshapes(track)
    columns = {name: index for index, name in enumerate(filtered.names)}
    for name in FAST_CONTACT_CONTROLS:
        if name in columns:
            np.testing.assert_array_equal(
                filtered.scores[:, columns[name]], track.blendshape_scores[:, columns[name]]
            )
    smile = columns["mouthSmileLeft"]
    assert 0 < filtered.scores[1, smile] < track.blendshape_scores[1, smile]
    assert filtered.scores[1, columns["eyeBlinkLeft"]] == pytest.approx(0.98)
    assert filtered.scores[2, columns["mouthClose"]] == pytest.approx(0.96)


def test_retarget_preserves_fixed_identity_head_pose_translation_and_eye_joints() -> None:
    capture = _capture_track()
    identity = np.linspace(-0.2, 0.2, 253, dtype=np.float32)
    performance = retarget_capture(
        capture, _FakeRetargeter(), identity=identity, baseline_frame_count=1
    )
    assert performance.expression.shape == (5, 383)
    np.testing.assert_array_equal(performance.expression[0], 0.0)
    np.testing.assert_array_equal(performance.expression[:, 201], 0.0)
    np.testing.assert_array_equal(performance.identity, identity)
    assert not performance.identity.flags.writeable
    assert np.linalg.norm(performance.rotations[-1, 1]) > 0
    assert np.linalg.norm(performance.translation[-1]) > 0
    assert performance.rotations[0, 2, 1] == pytest.approx(0.0)
    assert performance.rotations[0, 3, 1] == pytest.approx(0.0)
    assert np.max(np.abs(performance.rotations[:, 2:4])) > 0
    np.testing.assert_array_equal(performance.expression[:, 2], 0.0)
    assert performance.provenance.baseline_frame_indices == (0,)
    assert performance.provenance.quarantined_expression_controls == ("mouthClose",)
    assert np.count_nonzero(performance.source_lip_contact_confidence) == 0
    assert any("monocular" in caveat.lower() for caveat in performance.provenance.caveats)


def test_expressive_lead_in_is_rejected_and_later_neutral_window_is_used() -> None:
    capture = _capture_track()
    scores = np.zeros_like(capture.blendshape_scores)
    columns = {name: index for index, name in enumerate(capture.blendshape_names)}
    scores[:2, columns["jawOpen"]] = 0.90
    scores[:2, columns["mouthSmileLeft"]] = 0.92
    scores[:2, columns["mouthSmileRight"]] = 0.91
    capture = _capture_with_scores(capture, scores)

    performance = retarget_capture(
        capture,
        _FakeRetargeter(),
        neutral_baseline_seconds=1 / 30,
    )

    # The adaptive filter needs one extra frame to settle the deliberately
    # huge smile, so the lowest-activity reference is the final two frames.
    assert performance.provenance.baseline_frame_indices == (3, 4)
    assert performance.provenance.neutral_baseline_method == "auto_low_activity_window"
    assert performance.provenance.neutral_baseline_validated is True
    assert performance.provenance.neutral_baseline_correction_applied is True
    assert performance.expression[0, 200] > 0.85
    np.testing.assert_array_equal(performance.expression[2:, 200], 0.0)


def test_no_neutral_reference_disables_subtraction_instead_of_erasing_expression() -> None:
    capture = _capture_track()
    scores = np.zeros_like(capture.blendshape_scores)
    columns = {name: index for index, name in enumerate(capture.blendshape_names)}
    scores[:, columns["jawOpen"]] = 0.88
    scores[:, columns["mouthSmileLeft"]] = 0.90
    scores[:, columns["mouthSmileRight"]] = 0.90
    capture = _capture_with_scores(capture, scores)

    performance = retarget_capture(
        capture,
        _FakeRetargeter(),
        neutral_baseline_seconds=1 / 30,
    )

    assert performance.provenance.neutral_baseline_method == "none_expressive_video"
    assert performance.provenance.neutral_baseline_validated is False
    assert performance.provenance.neutral_baseline_correction_applied is False
    assert np.all(performance.expression[:, 200] > 0.85)
    assert any(
        "subtraction was disabled" in caveat
        for caveat in performance.provenance.caveats
    )


def test_current_arkit_gnm_retargeter_satisfies_injected_interface(rig) -> None:
    performance = retarget_capture(
        _capture_track(), ARKitGNMRetargeter(rig), baseline_frame_count=1
    )
    assert performance.expression.shape == (5, 383)
    assert np.max(np.abs(performance.expression[:, :350])) > 0
    assert "ARKitGNMRetargeter" in performance.provenance.retargeter


def test_capture_and_performance_serialization_include_provenance(tmp_path: Path) -> None:
    capture = _capture_track(include_missing=True)
    capture_npz, capture_jsonl = serialize_capture(tmp_path / "capture", capture)
    loaded = load_capture_npz(capture_npz)
    np.testing.assert_array_equal(loaded.source_pts, capture.source_pts)
    np.testing.assert_array_equal(loaded.detected, capture.detected)
    assert loaded.provenance.source_sha256 == capture.provenance.source_sha256
    records = [json.loads(line) for line in capture_jsonl.read_text().splitlines()]
    assert records[0]["recordType"] == "metadata"
    assert records[0]["provenance"]["source_sha256"] == "1" * 64
    assert records[4]["landmarksXYZ"] is None
    assert records[1]["faceConfidence"] is None

    performance = retarget_capture(
        capture, _FakeRetargeter(), baseline_frame_count=1
    )
    performance_npz, performance_jsonl = serialize_performance(
        tmp_path / "performance", performance
    )
    with np.load(performance_npz, allow_pickle=False) as values:
        assert values["expression"].shape == (5, 383)
        provenance = json.loads(str(values["provenance_json"].item()))
        assert provenance["capture_source_sha256"] == "1" * 64
        assert len(provenance["neutral_blendshape_baseline"]) == 52
        assert provenance["neutral_baseline_method"] == "explicit_initial_window"
        assert provenance["neutral_baseline_validated"] is True
        assert provenance["neutral_baseline_correction_applied"] is True
        assert provenance["quarantined_expression_controls"] == ["mouthClose"]
        assert provenance["contact_source_method"] == "mediapipe_inner_lip_geometry_v1"
        assert values["source_lip_geometry_valid"].shape == (5,)
        assert values["source_lip_contact_confidence"].shape == (5,)
        assert values["lip_contact_target_gap_interocular"].shape == (5,)
        assert values["contact_correction_applied"].shape == (5,)
        assert values["lip_contact_attained"].shape == (5,)
    metadata = json.loads(performance_jsonl.read_text().splitlines()[0])
    assert metadata["provenance"]["retargeter"].endswith("._FakeRetargeter")
    first_frame = json.loads(performance_jsonl.read_text().splitlines()[1])
    assert "sourceLipContactConfidence" in first_frame
    assert "sourceLipGeometryValid" in first_frame
    assert "lipContactAttained" in first_frame


@pytest.mark.skipif(
    not MODEL.exists()
    or not ASTRONAUT.exists()
    or not shutil.which("ffmpeg")
    or not shutil.which("ffprobe"),
    reason="real face/model/FFmpeg fixtures unavailable",
)
def test_real_face_video_mediapipe_video_mode_integration(tmp_path: Path) -> None:
    video = tmp_path / "real-face.mp4"
    subprocess.run(
        (
            "ffmpeg",
            "-v",
            "error",
            "-loop",
            "1",
            "-i",
            str(ASTRONAUT),
            "-t",
            "0.5",
            "-r",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ),
        check=True,
    )
    track = capture_video(video, MODEL)
    assert track.frame_count == 5
    assert np.count_nonzero(track.detected) == track.frame_count
    assert track.landmarks_xyz.shape == (5, 478, 3)
    assert track.blendshape_scores.shape == (5, 52)
    assert track.facial_transforms.shape == (5, 4, 4)
    assert np.all(track.tracking_quality > 0.95)
    assert track.provenance.source_sha256 != track.provenance.model_sha256
