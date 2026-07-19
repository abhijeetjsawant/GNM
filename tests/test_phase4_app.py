import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.animation import probe_av
from autoanim_gnm.artifacts import JobStore, new_ulid


CACHE = Path(os.environ.get("AUTOANIM_CACHE_DIR", ".cache/autoanim_gnm"))
FIXTURES = Path(os.environ.get("AUTOANIM_TEST_FIXTURES", CACHE / "fixtures"))
MODEL = CACHE / "face_landmarker.task"
PORTRAIT = FIXTURES / "official-portrait.jpg"
RHUBARB = CACHE / "rhubarb/rhubarb"
LIBRISPEECH = FIXTURES / "libri-human-speech-8s.wav"


def _normalized_result(result: dict) -> bytes:
    normalized = json.loads(json.dumps(result))
    for key in ("job_id", "created_at", "updated_at"):
        normalized.pop(key, None)
    # Integrity envelopes are transport-local because API and CLI roots use
    # different owner-only signing keys; semantic/artifact parity remains exact.
    normalized.pop("integrity", None)
    normalized.get("input", {}).pop("sha256", None)
    for artifact in normalized.get("artifacts", {}).values():
        artifact.pop("sha256", None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()


def _reference_frame_hashes(path: Path) -> list[str]:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count > 2
        targets = {0, count // 2, count - 1}
        hashes: list[str] = []
        for index in sorted(targets):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        return hashes
    finally:
        capture.release()


def test_ulid_and_interrupted_recovery(tmp_path: Path) -> None:
    job_id = new_ulid()
    assert len(job_id) == 26 and job_id == job_id.lower()
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    store = JobStore(tmp_path / "jobs")
    started_id, _, _, _ = store.start("test", source, {})
    recovered = JobStore(tmp_path / "jobs").read(started_id)
    assert recovered["status"] == "failed"
    assert recovered["error"]["code"] == "PROCESS_INTERRUPTED"


def test_recent_jobs_are_minimized_ordered_and_bounded(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    store = app.state.service.store
    source = tmp_path / "private-input.wav"
    source.write_bytes(b"input")
    first_id, first_dir, _, first = store.start("audio_animation", source, {})
    (first_dir / "animation.glb").write_bytes(b"glTF")
    store.finish(first, first_dir, {"kind": "audio_animation", "warnings": [], "artifacts": {"glb": "animation.glb"}}, {})
    second_id, second_dir, _, second = store.start("image_fit", source, {})
    store.finish(second, second_dir, {"kind": "image_fit", "warnings": ["LOW_CONFIDENCE"], "artifacts": {}}, {})

    client = TestClient(app)
    jobs = client.get("/api/jobs?limit=1").json()["jobs"]
    assert [job["job_id"] for job in jobs] == [second_id]
    assert jobs[0]["warning_count"] == 1
    assert jobs[0]["viewable"] is False
    assert "sha256" not in jobs[0]["input"]
    all_jobs = client.get("/api/jobs?limit=50").json()["jobs"]
    assert [job["job_id"] for job in all_jobs] == [second_id, first_id]
    assert all_jobs[1]["viewable"] is True


def test_home_and_health(tmp_path: Path) -> None:
    client = TestClient(
        create_app(tmp_path / "jobs", model_path=MODEL, rhubarb_bin=RHUBARB)
    )
    home = client.get("/")
    assert home.status_code == 200
    assert "Audio → animation" in home.text
    assert "Learned Audio2Face motion is preferred" in home.text
    assert "Recent local runs" in home.text
    assert "mouth_aperture" in home.text
    assert "contact attained" in home.text
    assert "baseline loss" in home.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert health.json()["checks"]["a2f_provenance"]["ready"] is True


def test_api_and_cli_require_authorship_for_nondefault_mouth_edit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"not needed: authorship fails before media decode")
    client = TestClient(
        create_app(tmp_path / "api-jobs", model_path=MODEL, rhubarb_bin=RHUBARB)
    )
    with source.open("rb") as handle:
        response = client.post(
            "/api/audio",
            files={"file": (source.name, handle, "audio/wav")},
            data={"mouth_aperture_gain": "1.08"},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "INPUT_INVALID"
    assert "author" in response.json()["message"].casefold()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoanim_gnm.cli",
            "audio",
            str(source),
            "--out",
            str(tmp_path / "cli-jobs"),
            "--mouth-aperture-gain",
            "1.08",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    error = json.loads(completed.stderr)
    assert error["code"] == "INPUT_INVALID"
    assert "author" in error["message"].casefold()


def test_api_and_cli_real_image_parity_and_allowlist(tmp_path: Path) -> None:
    if not MODEL.exists() or not PORTRAIT.exists():
        pytest.skip("real image parity fixtures unavailable")
    api_root = tmp_path / "api"
    client = TestClient(create_app(api_root, model_path=MODEL, rhubarb_bin=RHUBARB))
    with PORTRAIT.open("rb") as handle:
        response = client.post(
            "/api/image",
            files={"file": (PORTRAIT.name, handle, "image/jpeg")},
            data={"modes": "20", "allow_low_confidence": "false"},
        )
    assert response.status_code == 201, response.text
    api_result = response.json()
    assert api_result["status"] == "succeeded"
    assert api_result["input"]["name"] == PORTRAIT.name
    assert api_result["input"]["media_type"] == "image/jpeg"
    overlay = api_result["artifacts"]["overlay"]
    download = client.get(
        f"/api/jobs/{api_result['job_id']}/files/{overlay['name']}"
    )
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("image/png")
    assert client.get(
        f"/api/jobs/{api_result['job_id']}/files/result.json"
    ).status_code == 404

    cli_root = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoanim_gnm.cli",
            "--model-path",
            str(MODEL),
            "--rhubarb-bin",
            str(RHUBARB),
            "image",
            str(PORTRAIT),
            "--out",
            str(cli_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_result = json.loads(completed.stdout)
    assert cli_result["fit"] == api_result["fit"]
    with np.load(api_root / api_result["job_id"] / "fit.npz", allow_pickle=False) as api_fit:
        with np.load(cli_root / cli_result["job_id"] / "fit.npz", allow_pickle=False) as cli_fit:
            for key in api_fit.files:
                np.testing.assert_array_equal(api_fit[key], cli_fit[key])


def test_api_and_cli_real_audio_parity(tmp_path: Path) -> None:
    if not LIBRISPEECH.exists() or not RHUBARB.exists():
        pytest.skip("real audio parity fixtures unavailable")
    api_root = tmp_path / "api"
    client = TestClient(create_app(api_root, model_path=MODEL, rhubarb_bin=RHUBARB))
    with LIBRISPEECH.open("rb") as handle:
        response = client.post(
            "/api/audio",
            files={"file": (LIBRISPEECH.name, handle, "audio/wav")},
            data={"fps": "30", "emotion": "neutral"},
        )
    assert response.status_code == 201, response.text
    api_result = response.json()

    cli_root = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoanim_gnm.cli",
            "--model-path",
            str(MODEL),
            "--rhubarb-bin",
            str(RHUBARB),
            "audio",
            str(LIBRISPEECH),
            "--out",
            str(cli_root),
            "--emotion",
            "neutral",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_result = json.loads(completed.stdout)
    assert _normalized_result(api_result) == _normalized_result(cli_result)

    api_job = api_root / api_result["job_id"]
    cli_job = cli_root / cli_result["job_id"]
    with np.load(api_job / "controls.npz", allow_pickle=False) as api_controls:
        with np.load(cli_job / "controls.npz", allow_pickle=False) as cli_controls:
            assert api_controls.files == cli_controls.files
            for key in api_controls.files:
                np.testing.assert_array_equal(api_controls[key], cli_controls[key])

    api_preview = api_job / api_result["artifacts"]["preview"]["name"]
    cli_preview = cli_job / cli_result["artifacts"]["preview"]["name"]
    assert probe_av(api_preview) == probe_av(cli_preview)
    assert _reference_frame_hashes(api_preview) == _reference_frame_hashes(cli_preview)


def test_api_blank_image_returns_typed_error(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "jobs", model_path=MODEL, rhubarb_bin=RHUBARB))
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.zeros((256, 256, 3), dtype=np.uint8))
    with blank.open("rb") as handle:
        response = client.post(
            "/api/image",
            files={"file": ("blank.png", handle, "image/png")},
        )
    assert response.status_code == 422
    assert response.json()["code"] == "FACE_NOT_FOUND"
