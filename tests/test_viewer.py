import json
import re
import shutil
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from autoanim_gnm.api import _review_display_geometry_compatible, create_app
from autoanim_gnm.viewer import VIEWER_VENDOR_FILES, viewer_html


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
    result_path = job_dir / "result.json"
    unsealed = json.loads(result_path.read_text(encoding="utf-8"))
    unsealed.pop("integrity")
    result_path.write_text(json.dumps(unsealed), encoding="utf-8")
    blocked = client.get(f"/api/jobs/{job_id}/viewer")
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "INTEGRITY_UNSEALED"


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
    assert "animationAction.time=Math.min(Math.max(clockTime,0),animationDuration)" in page.text
    assert "mediaKind==='video'&&presentedMediaTime!==null?presentedMediaTime:media.currentTime" in page.text
    assert "presentationClockState==='verified'?'presented-frame synchronized':'media-clock fallback'" in page.text
    assert "else if(media.currentTime<=.01)status.textContent='Ready · media controls drive exact 3D time'" in page.text
    assert "else status.textContent=`Paused ${time} s · ${clockLabel}`" in page.text
    assert "animationAction.paused=true" in page.text
    assert "mixer.setTime(media.currentTime)" not in page.text
    assert 'mediaKind="audio"' in page.text
    assert "animationAction.setLoop(THREE.LoopOnce,1)" in page.text
    assert '"production_status": "blocked"' in page.text
    assert '"release_blockers":' in page.text


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
    (job_dir / "performance-evidence.json").write_text("{}", encoding="utf-8")
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "video_performance",
            "viewer": {"clock_artifact": "viewer_media"},
            "artifacts": {
                "glb": "performance.glb",
                "viewer_media": "source-proxy.mp4",
                "performance_evidence": "performance-evidence.json",
            },
        },
        {},
    )

    page = TestClient(app).get(f"/api/jobs/{job_id}/viewer")
    assert page.status_code == 200
    assert f"/api/jobs/{job_id}/files/source-proxy.mp4" in page.text
    assert (
        f'"url": "/api/jobs/{job_id}/files/performance-evidence.json"'
        in page.text
    )
    assert '"schemaVersion": "autoanim.viewer-review-layers/1.0"' in page.text
    assert 'mediaKind="video"' in page.text
    assert (
        '"observation_review": "unavailable: complete sealed Observation-v3 '
        'evidence is missing"' in page.text
    )
    assert "document.createElement(mediaKind)" in page.text
    assert "media.playsInline=true" in page.text
    assert "animationAction.time=Math.min(Math.max(clockTime,0),animationDuration)" in page.text
    assert "mediaKind==='video'&&presentedMediaTime!==null?presentedMediaTime:media.currentTime" in page.text
    assert "animationAction.paused=true" in page.text
    assert "mixer.setTime(media.currentTime)" not in page.text
    assert '"production_status": "blocked"' in page.text


def test_viewer_optional_evidence_lane_steps_exact_source_frames() -> None:
    page = viewer_html(
        asset_url="/api/jobs/01abc/files/performance.glb",
        title="Evidence review",
        media_url="/api/jobs/01abc/files/source-proxy.mp4",
        media_type="video/mp4",
        performance_evidence_url=(
            "/api/jobs/01abc/files/performance-evidence.json"
        ),
        observation_v3_url="/api/jobs/01abc/observation-v3-view",
    )
    assert (
        '"url": "/api/jobs/01abc/files/performance-evidence.json"'
        in page
    )
    assert 'id="evidence-panel"' in page
    assert '"url": "/api/jobs/01abc/observation-v3-view"' in page
    assert '"kind": "regional_tracker_evidence"' in page
    assert '"kind": "regional_pixel_roi_evidence"' in page
    assert '"kind": "display_proxy_frame_review_image"' in page
    assert "/review-frames/{frameIndex}.png" in page
    assert "reviewLayers.find(layer=>layer.kind==='regional_tracker_evidence')" in page
    assert 'id="evidence-diagnostic-state"' in page
    assert "autoanim.observation-v3-view/1.0" in page
    assert "Observation-v3 · provisional diagnostic only · never motion authority" in page
    assert "drawDiagnosticOverlay" in page
    assert "region.roiBoxXYXY" in page
    assert "staticMetrics=['clippedFraction','focusMetric','focusScore','lumaMean','shadowFraction','highlightFraction','dynamicRange']" in page
    assert "region.roiPixelCount===(box[2]-box[0])*(box[3]-box[1])" in page
    assert "temporalExpected=roiAvailable&&Array.isArray(previousBox)&&!frame.observationEpochStart" in page
    assert "payload.claims?.changesFinalGNMMotion!==false" in page
    assert "payload.source?.sha256!==performanceEvidenceSource?.sha256" in page
    assert "frame.sourcePTS!==evidence.sourcePTS" in page
    assert 'id="previous-source-frame"' in page
    assert 'id="next-source-frame"' in page
    assert "frame.sourcePTS" in page
    assert "frame.projectTick" in page
    assert "frame.timestampSeconds.toFixed(3)" in page
    assert "frameTimestampsSeconds?.[pendingEvidenceFrameIndex]" in page
    assert "else media.currentTime=targetTime" in page
    assert "pendingEvidenceFrameIndex=target" in page
    assert "waiting for browser presentation" in page
    assert "showEvidenceFrame(target)" not in page
    assert "media.pause()" in page
    assert "requestVideoFrameCallback(presentedVideoFrameLoop)" in page
    assert "displayExactReviewFrame" in page
    assert "const controller=new AbortController()" in page
    assert "if(reviewFrameAbortController===controller)reviewFrameAbortController=null" in page
    assert "if(forcePauseAfterPresentation)return" in page
    assert "reviewFrameAbortController?.abort();reviewFrameAbortController=null;pendingEvidenceFrameIndex=-1" in page
    assert "staticReviewInternalSeekTime=presentedMediaTime" in page
    assert "media.addEventListener('seeking'" in page
    assert "reviewFrameUrlTemplate&&diagnosticsReady&&media.paused&&evidenceReady" in page
    assert "server-decoded proxy frame verified" in page
    assert "const result=applyPresentedMediaTime(mediaTime,true)" in page
    assert "forcePausedSeekPresentation" in page
    assert "media.muted=true" in page
    assert "forcePauseAfterPresentation" in page
    assert "cancelVideoFrameCallback(videoFrameCallbackId)" in page
    assert "presented frame verified" in page
    assert "retrying while diagnostics follow the presented frame" in page
    assert "evidenceIndexForPresentedMediaTime" in page
    assert "pendingPresentationAttempt>3" in page
    assert "Timed out waiting for the browser" in page
    assert "const requiredEvidenceRegions=['mouth','eyes','upperFace','head']" in page
    assert "MISSING · no face observation at this source frame" in page
    assert "UNKNOWN · observed tracker values are not a labeled neutral" in page
    assert "region.confidence===null?'—'" in page
    assert "parsed.origin!==window.location.origin" in page
    assert "credentials:'same-origin'" in page
    assert "cache:'no-store'" in page
    assert "payload.consumedByRetargeting!==false" in page
    assert "animationAction.time=Math.min(Math.max(clockTime,0),animationDuration)" in page
    assert "mixer.setTime(media.currentTime)" not in page
    assert "cdn.jsdelivr.net" not in page


def test_viewer_rejects_evidence_layers_from_different_jobs() -> None:
    with pytest.raises(ValueError, match="must belong to the same job"):
        viewer_html(
            asset_url="/api/jobs/01abc/files/performance.glb",
            title="Evidence review",
            media_url="/api/jobs/01abc/files/source-proxy.mp4",
            media_type="video/mp4",
            performance_evidence_url=(
                "/api/jobs/01abc/files/performance-evidence.json"
            ),
            observation_v3_url="/api/jobs/02xyz/observation-v3-view",
        )


def test_review_display_geometry_preflight_is_fail_closed() -> None:
    capture = {"width": 480, "height": 360}
    valid = {
        "clock_artifact": "viewer_media",
        "display_geometry": {
            "schema_version": "autoanim.viewer-display-binding/1.0",
            "artifact": "viewer_media",
            "source_frame_size": [480, 360],
            "proxy_frame_size": [480, 360],
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
    }
    assert _review_display_geometry_compatible(valid, capture)
    for field, value in (
        ("sample_aspect_ratio", [4, 3]),
        ("clean_aperture_crop_ltrb", [1, 0, 0, 0]),
        ("display_rotation_degrees", 90),
        ("proxy_frame_size", [360, 480]),
    ):
        invalid = json.loads(json.dumps(valid))
        invalid["display_geometry"][field] = value
        assert not _review_display_geometry_compatible(invalid, capture)


@pytest.mark.parametrize(
    "evidence_url",
    (
        "https://example.com/performance-evidence.json",
        "/api/jobs/../files/performance-evidence.json",
        "/api/jobs/01abc/files/other.json",
        "/api/jobs/01abc/files/performance-evidence.json?download=1",
    ),
)
def test_viewer_rejects_non_allowlisted_evidence_url(evidence_url: str) -> None:
    with pytest.raises(ValueError, match="allowlisted job artifact URL"):
        viewer_html(
            asset_url="/api/jobs/01abc/files/performance.glb",
            title="Evidence review",
            media_url="/api/jobs/01abc/files/source-proxy.mp4",
            media_type="video/mp4",
            performance_evidence_url=evidence_url,
        )


@pytest.mark.parametrize(
    "diagnostic_url",
    (
        "https://example.com/api/jobs/01abc/observation-v3-view",
        "/api/jobs/../observation-v3-view",
        "/api/jobs/01abc/files/observation-v3.json",
        "/api/jobs/01abc/observation-v3-view?download=1",
    ),
)
def test_viewer_rejects_non_allowlisted_observation_v3_url(
    diagnostic_url: str,
) -> None:
    with pytest.raises(ValueError, match="allowlisted job review URL"):
        viewer_html(
            asset_url="/api/jobs/01abc/files/performance.glb",
            title="Evidence review",
            media_url="/api/jobs/01abc/files/source-proxy.mp4",
            media_type="video/mp4",
            performance_evidence_url=(
                "/api/jobs/01abc/files/performance-evidence.json"
            ),
            observation_v3_url=diagnostic_url,
        )


def test_audio_and_static_viewer_leave_optional_evidence_lane_inactive() -> None:
    audio = viewer_html(
        asset_url="/api/jobs/01abc/files/animation.glb",
        title="Audio",
        media_url="/api/jobs/01abc/files/normalized.wav",
        media_type="audio/wav",
    )
    static = viewer_html(
        asset_url="/api/jobs/01abc/files/fitted.glb",
        title="Static",
    )
    for page in (audio, static):
        assert '"layers": []' in page
        assert "?.url??null" in page
        assert 'id="evidence-panel"' in page
        assert "if(!performanceEvidenceUrl||!media||mediaKind!=='video')return" in page
    assert 'mediaKind="audio"' in audio
    assert "mediaUrl=null" in static


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_viewer_generated_module_has_valid_javascript_syntax(tmp_path: Path) -> None:
    page = viewer_html(
        asset_url="/api/jobs/01abc/files/performance.glb",
        title="Evidence review",
        media_url="/api/jobs/01abc/files/source-proxy.mp4",
        media_type="video/mp4",
        performance_evidence_url=(
            "/api/jobs/01abc/files/performance-evidence.json"
        ),
    )
    match = re.search(r'<script type="module">(.*?)</script>', page, re.DOTALL)
    assert match is not None
    module = tmp_path / "viewer.mjs"
    module.write_text(match.group(1), encoding="utf-8")
    result = subprocess.run(
        (shutil.which("node") or "node", "--check", str(module)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
