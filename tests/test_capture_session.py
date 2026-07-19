from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from autoanim_gnm.capture_session import (
    CAPTURE_SESSION_SCHEMA_VERSION,
    build_video_capture_session,
    load_verified_video_capture_session,
    write_video_capture_session,
)
from autoanim_gnm.video_evidence import write_performance_evidence
from autoanim_gnm.video_observation import (
    analyze_rgb_frames,
    write_observation_v3_summary,
    write_pixel_observations,
)
from autoanim_gnm.video_capture import serialize_capture

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
    paths = {
        "capture": capture_path,
        "capture_jsonl": jsonl_path,
        "performance_evidence": v2_path,
        "pixel_observations": arrays_path,
        "observation_v3": v3_path,
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
        "redecoded_for_evidence"
    )
    detector_configuration = payload["detectors"][0]["configuration"]
    assert detector_configuration["confidence_thresholds"] is None
    assert detector_configuration["confidence_threshold_state"] == (
        "not_retained_by_capture_v1"
    )
    verified = load_verified_video_capture_session(
        first,
        expected_capture=capture,
        expected_observations=observations,
        artifact_paths=paths,
    )
    assert verified == payload


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
