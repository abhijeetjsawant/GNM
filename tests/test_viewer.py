from pathlib import Path

from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.viewer import VIEWER_VENDOR_FILES


def _viewer_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "viewer"
    for name in VIEWER_VENDOR_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("export {};", encoding="utf-8")
    return root


def test_viewer_resolves_only_allowlisted_glb(tmp_path: Path):
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        viewer_vendor_root=_viewer_bundle(tmp_path),
    )
    store = app.state.service.store
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    job_id, job_dir, _, manifest = store.start("image_fit", source, {})
    (job_dir / "fitted.glb").write_bytes(b"glTF\x02\0\0\0")
    store.finish(
        manifest,
        job_dir,
        {"kind": "image_fit", "status": "succeeded", "artifacts": {"glb": "fitted.glb"}},
        {},
    )
    client = TestClient(app)

    page = client.get(f"/api/jobs/{job_id}/viewer")
    assert page.status_code == 200
    assert "GLTFLoader" in page.text
    assert f"/api/jobs/{job_id}/files/fitted.glb" in page.text
    assert "Surface + topology" in page.text
    assert "cdn.jsdelivr.net" not in page.text
    assert '"three":"/api/viewer/vendor/0.183.2/three.module.js"' in page.text
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert "connect-src 'self' blob:" in page.headers["content-security-policy"]
    assert page.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert 'aria-label="Interactive 3D GNM head"' in page.text
    assert "controls.enablePan=false" in page.text
    assert "controls.maxDistance=radius*8" in page.text
    assert "window.addEventListener('pagehide'" in page.text
    assert "WebGL is unavailable" in page.text
    assert "lines.join('\\n')" in page.text
    asset = client.get(f"/api/jobs/{job_id}/files/fitted.glb")
    assert asset.status_code == 200
    assert asset.headers["content-type"].startswith("model/gltf-binary")
    module = client.get("/api/viewer/vendor/0.183.2/three.module.js")
    assert module.status_code == 200
    assert module.headers["content-type"].startswith("text/javascript")
    assert module.headers["x-content-type-options"] == "nosniff"
    assert module.headers["cross-origin-resource-policy"] == "same-origin"
    core_module = client.get("/api/viewer/vendor/0.183.2/three.core.js")
    assert core_module.status_code == 200
    assert core_module.headers["content-type"].startswith("text/javascript")
    assert client.get("/api/viewer/vendor/0.183.2/../../result.json").status_code == 404
    assert client.get("/api/viewer/vendor/0.182.0/three.module.js").status_code == 404


def test_viewer_rejects_jobs_without_glb(tmp_path: Path):
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        viewer_vendor_root=_viewer_bundle(tmp_path),
    )
    store = app.state.service.store
    source = tmp_path / "input.bin"
    source.write_bytes(b"input")
    job_id, job_dir, _, manifest = store.start("image_fit", source, {})
    store.finish(manifest, job_dir, {"kind": "image_fit", "artifacts": {}}, {})

    response = TestClient(app).get(f"/api/jobs/{job_id}/viewer")
    assert response.status_code == 404
    assert response.json()["code"] == "ARTIFACT_NOT_FOUND"


def test_animation_viewer_uses_allowlisted_media_clock(tmp_path: Path):
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        viewer_vendor_root=_viewer_bundle(tmp_path),
    )
    store = app.state.service.store
    source = tmp_path / "input.wav"
    source.write_bytes(b"input")
    job_id, job_dir, _, manifest = store.start("audio_animation", source, {})
    (job_dir / "animation.glb").write_bytes(b"glTF\x02\0\0\0")
    (job_dir / "normalized.wav").write_bytes(b"RIFF")
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "audio_animation",
            "viewer": {"clock_artifact": "normalized_audio"},
            "artifacts": {
                "glb": "animation.glb",
                "normalized_audio": "normalized.wav",
            },
        },
        {},
    )

    page = TestClient(app).get(f"/api/jobs/{job_id}/viewer")
    assert page.status_code == 200
    assert f"/api/jobs/{job_id}/files/normalized.wav" in page.text
    assert "animationAction.time=Math.min(Math.max(media.currentTime,0),animationDuration)" in page.text
    assert "animationAction.paused=true" in page.text
    assert "mixer.setTime(media.currentTime)" not in page.text
    assert 'mediaKind="audio"' in page.text
    assert "animationAction.setLoop(THREE.LoopOnce,1)" in page.text


def test_video_viewer_uses_video_element_and_source_clock(tmp_path: Path):
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        viewer_vendor_root=_viewer_bundle(tmp_path),
    )
    store = app.state.service.store
    source = tmp_path / "input.flv"
    source.write_bytes(b"input")
    job_id, job_dir, _, manifest = store.start("video_performance", source, {})
    (job_dir / "performance.glb").write_bytes(b"glTF\x02\0\0\0")
    (job_dir / "source-proxy.mp4").write_bytes(b"video")
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "video_performance",
            "viewer": {"clock_artifact": "viewer_media"},
            "artifacts": {
                "glb": "performance.glb",
                "viewer_media": "source-proxy.mp4",
            },
        },
        {},
    )

    page = TestClient(app).get(f"/api/jobs/{job_id}/viewer")
    assert page.status_code == 200
    assert f"/api/jobs/{job_id}/files/source-proxy.mp4" in page.text
    assert 'mediaKind="video"' in page.text
    assert "document.createElement(mediaKind)" in page.text
    assert "media.playsInline=true" in page.text
    assert "animationAction.time=Math.min(Math.max(media.currentTime,0),animationDuration)" in page.text
    assert "animationAction.paused=true" in page.text
    assert "mixer.setTime(media.currentTime)" not in page.text


def test_missing_viewer_bundle_fails_closed(tmp_path: Path):
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        viewer_vendor_root=tmp_path / "missing-viewer",
    )
    client = TestClient(app)
    health = client.get("/api/health").json()
    assert health["checks"]["viewer_bundle"]["ready"] is False
    assert health["status"] == "degraded"
    response = client.get("/api/viewer/vendor/0.183.2/three.module.js")
    assert response.status_code == 424
    assert response.json()["code"] == "DEPENDENCY_MISSING"
