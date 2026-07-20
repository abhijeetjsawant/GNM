from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from autoanim_gnm.video_capture import VideoCaptureRun
from autoanim_gnm.video_capture_run import (
    DETECTOR_INGRESS_HASH_DOMAIN,
    VIDEO_CAPTURE_RUN_SCHEMA_VERSION,
    VideoCaptureRunArtifactError,
    build_video_capture_run_document,
    load_video_capture_run,
    video_capture_run_payload_sha256,
    write_video_capture_run,
)

from test_video_capture import _capture_track


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _run() -> VideoCaptureRun:
    capture = _capture_track()
    return VideoCaptureRun(
        track=capture,
        detector_ingress_rgb_sha256=tuple(
            _digest(f"detector-ingress-{index}")
            for index in range(capture.frame_count)
        ),
        num_faces=1,
        confidence_thresholds=(0.41, 0.52, 0.63),
    )


def _canonical_bytes(document: dict) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _rewrite(
    path: Path,
    document: dict,
    *,
    refresh_document_sha: bool = True,
) -> None:
    if refresh_document_sha:
        document["document_sha256"] = video_capture_run_payload_sha256(document)
    path.write_bytes(_canonical_bytes(document))


def test_capture_run_document_is_canonical_deterministic_and_non_authorizing(
    tmp_path: Path,
) -> None:
    run = _run()
    document = build_video_capture_run_document(run)

    assert document["schema_version"] == VIDEO_CAPTURE_RUN_SCHEMA_VERSION
    assert document["source"] == {
        "capture_schema_version": "autoanim.capture.v1",
        "name": run.track.provenance.source_name,
        "sha256": run.track.provenance.source_sha256,
        "bytes": run.track.provenance.source_bytes,
        "frame_count": run.track.frame_count,
        "source_start_pts": int(run.track.source_pts[0]),
        "source_pts_sha256": hashlib.sha256(
            run.track.source_pts.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "time_base": [1, 30],
    }
    assert document["model"] == {
        "name": run.track.provenance.model_name,
        "sha256": run.track.provenance.model_sha256,
    }
    assert document["detector"]["configuration"] == run.detector_configuration()
    assert document["detector_ingress"]["ordered_rgb_sha256"] == list(
        run.detector_ingress_rgb_sha256
    )
    assert document["claims"] == {
        "detector_ingress_pixels_retained": False,
        "detector_ingress_hashes_retained": True,
        "changes_final_gnm_motion": False,
        "production_validated": False,
    }

    first = write_video_capture_run(tmp_path / "first.json", run)
    second = write_video_capture_run(tmp_path / "second.json", run)
    assert first.read_bytes() == second.read_bytes() == _canonical_bytes(document)
    assert b" " not in first.read_bytes()

    loaded = load_video_capture_run(first, expected_capture=run.track)
    assert loaded.track is run.track
    assert loaded.detector_ingress_rgb_sha256 == run.detector_ingress_rgb_sha256
    assert loaded.detector_configuration() == run.detector_configuration()


def test_capture_run_loader_rejects_hash_sequence_tamper(tmp_path: Path) -> None:
    run = _run()
    path = write_video_capture_run(tmp_path / "run.json", run)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["detector_ingress"]["ordered_rgb_sha256"][2] = _digest("forged")
    # Refresh the document digest so this exercises the independent PTS/hash
    # sequence binding instead of merely the outer accidental-tamper checksum.
    _rewrite(path, document)

    with pytest.raises(VideoCaptureRunArtifactError) as caught:
        load_video_capture_run(path, expected_capture=run.track)
    assert caught.value.code == "ORDERED_BINDING_MISMATCH"


def test_capture_run_loader_rejects_configuration_tamper(tmp_path: Path) -> None:
    run = _run()
    path = write_video_capture_run(tmp_path / "run.json", run)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["detector"]["configuration"]["min_tracking_confidence"] = 0.71
    _rewrite(path, document)

    with pytest.raises(VideoCaptureRunArtifactError) as caught:
        load_video_capture_run(path, expected_capture=run.track)
    assert caught.value.code == "CONFIGURATION_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("mutator", "code"),
    (
        (
            lambda document: document["source"].__setitem__(
                "source_pts_sha256", _digest("other-pts")
            ),
            "RECONSTRUCTION_MISMATCH",
        ),
        (
            lambda document: document["source"].__setitem__(
                "sha256", _digest("other-source")
            ),
            "RECONSTRUCTION_MISMATCH",
        ),
        (
            lambda document: document["source"].__setitem__(
                "name", "other-source.mov"
            ),
            "RECONSTRUCTION_MISMATCH",
        ),
        (
            lambda document: document["model"].__setitem__(
                "sha256", _digest("other-model")
            ),
            "RECONSTRUCTION_MISMATCH",
        ),
        (
            lambda document: document["model"].__setitem__(
                "name", "other-model.task"
            ),
            "RECONSTRUCTION_MISMATCH",
        ),
    ),
)
def test_capture_run_loader_rejects_expected_capture_binding_tamper(
    tmp_path: Path, mutator, code: str
) -> None:
    run = _run()
    path = write_video_capture_run(tmp_path / "run.json", run)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutator(document)
    _rewrite(path, document)

    with pytest.raises(VideoCaptureRunArtifactError) as caught:
        load_video_capture_run(path, expected_capture=run.track)
    assert caught.value.code == code


def test_capture_run_loader_rejects_duplicate_nonfinite_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    run = _run()
    path = tmp_path / "run.json"

    path.write_bytes(
        b'{"schema_version":"autoanim.video-capture-run/1.0",'
        b'"schema_version":"forged"}\n'
    )
    with pytest.raises(VideoCaptureRunArtifactError) as duplicate:
        load_video_capture_run(path, expected_capture=run.track)
    assert duplicate.value.code == "DUPLICATE_KEY"

    path.write_bytes(b'{"unexpected":NaN}\n')
    with pytest.raises(VideoCaptureRunArtifactError) as nonfinite:
        load_video_capture_run(path, expected_capture=run.track)
    assert nonfinite.value.code == "NONFINITE_NUMBER"

    document = build_video_capture_run_document(run)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(VideoCaptureRunArtifactError) as noncanonical:
        load_video_capture_run(path, expected_capture=run.track)
    assert noncanonical.value.code == "NONCANONICAL_JSON"


@pytest.mark.parametrize(
    "mutator",
    (
        lambda document: document["source"].__setitem__("bytes", True),
        lambda document: document["source"].__setitem__("frame_count", True),
        lambda document: document["source"]["time_base"].__setitem__(0, True),
        lambda document: document["detector"]["configuration"].__setitem__(
            "num_faces", True
        ),
        lambda document: document["detector"]["configuration"].__setitem__(
            "min_tracking_confidence", True
        ),
    ),
)
def test_capture_run_loader_rejects_bool_as_number(
    tmp_path: Path, mutator
) -> None:
    run = _run()
    path = write_video_capture_run(tmp_path / "run.json", run)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutator(document)
    # Refresh every redundant digest so strict primitive typing is the failure.
    document["detector"]["configuration_sha256"] = hashlib.sha256(
        json.dumps(
            document["detector"]["configuration"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _rewrite(path, document)

    with pytest.raises(VideoCaptureRunArtifactError) as caught:
        load_video_capture_run(path, expected_capture=run.track)
    assert caught.value.code == "INVALID_TYPE"


def test_capture_run_loader_rejects_claim_and_unknown_field_tamper(
    tmp_path: Path,
) -> None:
    run = _run()
    path = write_video_capture_run(tmp_path / "run.json", run)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["claims"]["production_validated"] = True
    _rewrite(path, document)
    with pytest.raises(VideoCaptureRunArtifactError) as claim:
        load_video_capture_run(path, expected_capture=run.track)
    assert claim.value.code == "UNSUPPORTED_CLAIM"

    document = build_video_capture_run_document(run)
    document["detector_ingress"]["unexpected"] = None
    _rewrite(path, document)
    with pytest.raises(VideoCaptureRunArtifactError) as fields:
        load_video_capture_run(path, expected_capture=run.track)
    assert fields.value.code == "INVALID_FIELDS"


def test_capture_run_loader_rejects_wrong_hash_domain_and_outer_digest(
    tmp_path: Path,
) -> None:
    run = _run()
    path = write_video_capture_run(tmp_path / "run.json", run)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["detector_ingress"]["hash_domain"] = "ambiguous"
    _rewrite(path, document)
    with pytest.raises(VideoCaptureRunArtifactError) as domain:
        load_video_capture_run(path, expected_capture=run.track)
    assert domain.value.code == "INVALID_HASH_DOMAIN"

    document = build_video_capture_run_document(run)
    document["document_sha256"] = _digest("wrong-document")
    _rewrite(path, document, refresh_document_sha=False)
    with pytest.raises(VideoCaptureRunArtifactError) as digest:
        load_video_capture_run(path, expected_capture=run.track)
    assert digest.value.code == "DOCUMENT_HASH_MISMATCH"
    assert run.hash_domain == DETECTOR_INGRESS_HASH_DOMAIN
