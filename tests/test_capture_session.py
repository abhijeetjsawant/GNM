from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import autoanim_gnm.capture_session as capture_session_module
from autoanim_gnm.capture_session import (
    CAPTURE_SESSION_SCHEMA_VERSION,
    LEGACY_CAPTURE_SESSION_SCHEMA_VERSION,
    build_video_capture_session,
    load_verified_legacy_video_capture_session_v1,
    load_verified_video_capture_session,
    write_video_capture_session,
)
from autoanim_gnm.serialization import write_json
from autoanim_gnm.video_evidence import write_performance_evidence
from autoanim_gnm.video_observation import (
    analyze_rgb_frames,
    write_observation_v3_summary,
    write_pixel_observations,
)
from autoanim_gnm.video_capture import VideoCaptureRun, serialize_capture
from autoanim_gnm.video_capture_run import write_video_capture_run
from autoanim_gnm.visual_track import (
    build_visual_track,
    write_visual_track,
    write_visual_track_summary,
)

from test_video_observation import _capture, _same_frames


def _artifacts(tmp_path: Path):
    capture = _capture()
    observations = analyze_rgb_frames(capture, _same_frames(capture.frame_count))
    capture_path, jsonl_path = serialize_capture(tmp_path, capture)
    v2_path = write_performance_evidence(tmp_path / "performance-evidence.json", capture)
    arrays_path = write_pixel_observations(
        tmp_path / "pixel-observations.npz", observations
    )
    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    v3_path = write_observation_v3_summary(
        tmp_path / "observation-v3.json",
        capture,
        observations,
        capture_artifact_sha256=digest(capture_path),
        capture_artifact_bytes=capture_path.stat().st_size,
        pixel_observations_sha256=digest(arrays_path),
        pixel_observations_bytes=arrays_path.stat().st_size,
    )
    capture_run = VideoCaptureRun(
        track=capture,
        detector_ingress_rgb_sha256=observations.decoded_pixel_sha256,
        num_faces=1,
        confidence_thresholds=(0.5, 0.5, 0.5),
    )
    capture_run_path = write_video_capture_run(
        tmp_path / "video-capture-run.json", capture_run
    )
    visual_track = build_visual_track(
        capture, observations, capture_run=capture_run
    )
    visual_track_path = write_visual_track(tmp_path / "visual-track.npz", visual_track)
    visual_track_summary_path = write_visual_track_summary(
        tmp_path / "visual-track.json", visual_track
    )
    paths = {
        "capture": capture_path,
        "capture_jsonl": jsonl_path,
        "performance_evidence": v2_path,
        "pixel_observations": arrays_path,
        "observation_v3": v3_path,
        "video_capture_run": capture_run_path,
        "visual_track": visual_track_path,
        "visual_track_summary": visual_track_summary_path,
    }
    return capture, observations, paths


def test_capture_session_is_deterministic_path_free_and_fail_closed(
    tmp_path: Path,
) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    first = write_video_capture_session(
        tmp_path / "session-a.json",
        capture,
        observations,
        artifact_paths=paths,
    )
    second = write_video_capture_session(
        tmp_path / "session-b.json",
        capture,
        observations,
        artifact_paths=paths,
    )
    assert first.read_bytes() == second.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert str(tmp_path) not in text
    assert "job_id" not in text and "created_at" not in text
    payload = json.loads(text)
    assert payload["schema_version"] == CAPTURE_SESSION_SCHEMA_VERSION
    assert payload["subject_binding"]["state"] == "unbound"
    assert payload["assessments"]["identity_continuity"]["state"] == "unknown"
    assert payload["claims"]["production_validated"] is False
    assert payload["claims"]["changes_final_gnm_motion"] is False
    assert payload["pixel_streams"][0]["relationship_to_detector_input"] == (
        "per_frame_sha256_equal_to_detector_ingress"
    )
    detector_configuration = payload["detectors"][0]["configuration"]
    assert detector_configuration["min_face_detection_confidence"] == 0.5
    assert detector_configuration["min_face_presence_confidence"] == 0.5
    assert detector_configuration["min_tracking_confidence"] == 0.5
    verified = load_verified_video_capture_session(
        first,
        expected_capture=capture,
        expected_observations=observations,
        artifact_paths=paths,
    )
    assert verified == payload


def test_capture_session_binds_same_buffer_detector_provenance(tmp_path: Path) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    capture_run = VideoCaptureRun(
        track=capture,
        detector_ingress_rgb_sha256=observations.decoded_pixel_sha256,
        num_faces=1,
        confidence_thresholds=(0.4, 0.5, 0.6),
    )
    visual_track = build_visual_track(
        capture, observations, capture_run=capture_run
    )
    write_video_capture_run(paths["video_capture_run"], capture_run)
    write_visual_track(paths["visual_track"], visual_track)
    write_visual_track_summary(paths["visual_track_summary"], visual_track)
    session = write_video_capture_session(
        tmp_path / "capture-session.json",
        capture,
        observations,
        artifact_paths=paths,
        capture_run=capture_run,
    )
    payload = load_verified_video_capture_session(
        session,
        expected_capture=capture,
        expected_observations=observations,
        artifact_paths=paths,
        expected_capture_run=capture_run,
    )
    stream = payload["pixel_streams"][0]
    assert stream["relationship_to_detector_input"] == (
        "per_frame_sha256_equal_to_detector_ingress"
    )
    assert stream["detector_ingress_hashes_retained"] is True
    assert payload["detectors"][0]["configuration"] == (
        capture_run.detector_configuration()
    )
    reconstructed_from_persisted_run = load_verified_video_capture_session(
        session,
        expected_capture=capture,
        expected_observations=observations,
        artifact_paths=paths,
    )
    assert reconstructed_from_persisted_run == payload


def test_legacy_capture_session_v1_reconstructs_read_only_without_v2_evidence(
    tmp_path: Path,
) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    legacy_paths = {
        name: path
        for name, path in paths.items()
        if name
        in {
            "capture",
            "capture_jsonl",
            "performance_evidence",
            "pixel_observations",
            "observation_v3",
        }
    }
    legacy_document = capture_session_module._build_legacy_video_capture_session_v1(
        capture,
        observations,
        artifact_paths=legacy_paths,
    )
    session = write_json(tmp_path / "capture-session-v1.json", legacy_document)

    verified = load_verified_legacy_video_capture_session_v1(
        session,
        expected_capture=capture,
        expected_observations=observations,
        artifact_paths=legacy_paths,
    )

    assert verified["schema_version"] == LEGACY_CAPTURE_SESSION_SCHEMA_VERSION
    assert verified["claims"]["production_validated"] is False
    assert verified["claims"]["changes_final_gnm_motion"] is False
    assert verified["subject_binding"]["state"] == "unbound"
    assert verified["pixel_streams"][0]["relationship_to_detector_input"] == (
        "redecoded_for_evidence"
    )
    assert "detector_ingress_hashes_retained" not in verified["pixel_streams"][0]
    with pytest.raises(ValueError, match="artifact set"):
        load_verified_video_capture_session(
            session,
            expected_capture=capture,
            expected_observations=observations,
            artifact_paths=legacy_paths,
        )

    forged = json.loads(session.read_text(encoding="utf-8"))
    forged["claims"]["production_validated"] = True
    write_json(session, forged)
    with pytest.raises(ValueError, match="does not reconstruct"):
        load_verified_legacy_video_capture_session_v1(
            session,
            expected_capture=capture,
            expected_observations=observations,
            artifact_paths=legacy_paths,
        )


def test_capture_session_rejects_artifact_and_document_tampering(tmp_path: Path) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    session = write_video_capture_session(
        tmp_path / "capture-session.json",
        capture,
        observations,
        artifact_paths=paths,
    )
    paths["performance_evidence"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported|does not reconstruct"):
        load_verified_video_capture_session(
            session,
            expected_capture=capture,
            expected_observations=observations,
            artifact_paths=paths,
        )

    session.write_text('{"kind": "a", "kind": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON member"):
        load_verified_video_capture_session(
            session,
            expected_capture=capture,
            expected_observations=observations,
            artifact_paths=paths,
        )


def test_capture_session_verifier_reconstructs_observation_v2_from_capture(
    tmp_path: Path,
) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    v2_path = paths["performance_evidence"]
    payload = json.loads(v2_path.read_text(encoding="utf-8"))
    payload["source"]["sourceStartPTS"] += 10
    for frame in payload["frames"]:
        frame["sourcePTS"] += 10
    v2_path.write_text(json.dumps(payload), encoding="utf-8")
    session = write_video_capture_session(
        tmp_path / "capture-session.json",
        capture,
        observations,
        artifact_paths=paths,
    )
    with pytest.raises(ValueError, match="expected capture"):
        load_verified_video_capture_session(
            session,
            expected_capture=capture,
            expected_observations=observations,
            artifact_paths=paths,
        )

def test_capture_session_requires_the_exact_artifact_set(tmp_path: Path) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    paths.pop("observation_v3")
    with pytest.raises(ValueError, match="artifact set"):
        build_video_capture_session(capture, observations, artifact_paths=paths)


def test_capture_session_rejects_absolute_media_paths(tmp_path: Path) -> None:
    capture, observations, paths = _artifacts(tmp_path)
    capture = replace(
        capture,
        provenance=replace(
            capture.provenance,
            ffmpeg_command=("ffmpeg", "/private/tmp/source.mov"),
        ),
    )
    with pytest.raises(ValueError, match="path-free"):
        build_video_capture_session(capture, observations, artifact_paths=paths)
