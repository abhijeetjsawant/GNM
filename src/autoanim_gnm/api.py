"""FastAPI transport and a dependency-free local web UI."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .errors import AutoAnimError
from .service import ApplicationService
from .viewer import VIEWER_THREE_VERSION, VIEWER_VENDOR_FILES, viewer_html


STATUS_BY_CODE = {
    "INPUT_INVALID": 400,
    "MEDIA_INVALID": 400,
    "AUDIO_SILENT": 400,
    "CUE_INVALID": 400,
    "JOB_NOT_FOUND": 404,
    "ARTIFACT_NOT_FOUND": 404,
    "BUSY": 409,
    "LIMIT_EXCEEDED": 413,
    "FACE_NOT_FOUND": 422,
    "MULTIPLE_FACES": 422,
    "FIT_REJECTED": 422,
    "DEPENDENCY_MISSING": 424,
    "INTERNAL_ERROR": 500,
}


def _error_response(error: AutoAnimError) -> JSONResponse:
    return JSONResponse(status_code=STATUS_BY_CODE.get(error.code, 500), content=error.as_dict())


def _retain_upload(upload: UploadFile, *, max_bytes: int = 100 * 1024 * 1024) -> Path:
    suffix = Path(upload.filename or "input.bin").suffix[:16]
    size = 0
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        while True:
            block = upload.file.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > max_bytes:
                path.unlink(missing_ok=True)
                limit_mib = max_bytes / (1024 * 1024)
                raise AutoAnimError(
                    "LIMIT_EXCEEDED", f"Upload exceeds {limit_mib:g} MiB"
                )
            handle.write(block)
    if size == 0:
        path.unlink(missing_ok=True)
        raise AutoAnimError("INPUT_INVALID", "Uploaded file is empty")
    return path


def create_app(
    artifact_root: str | Path,
    *,
    model_path: str | Path | None = None,
    rhubarb_bin: str | Path | None = None,
    a2f_runner: str | Path | None = None,
    a2f_asset_dir: str | Path | None = None,
    a2f_offline: bool = False,
    viewer_vendor_root: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="AutoAnim GNM", version="0.1.0")
    service = ApplicationService(
        artifact_root,
        model_path=model_path,
        rhubarb_bin=rhubarb_bin,
        a2f_runner=a2f_runner,
        a2f_asset_dir=a2f_asset_dir,
        a2f_offline=a2f_offline,
        viewer_vendor_root=viewer_vendor_root,
    )
    operation_lock = threading.Lock()
    app.state.service = service

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return UI_HTML

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health")
    def health() -> dict:
        return service.health()

    @app.get("/api/viewer/vendor/{version}/{name:path}", include_in_schema=False)
    def viewer_vendor(version: str, name: str):
        if version != VIEWER_THREE_VERSION or name not in VIEWER_VENDOR_FILES:
            return _error_response(
                AutoAnimError("ARTIFACT_NOT_FOUND", "Viewer module is not allowlisted")
            )
        path = service.viewer_vendor_root / name
        if not path.is_file():
            return _error_response(
                AutoAnimError(
                    "DEPENDENCY_MISSING",
                    "The local Three.js viewer bundle is missing; run scripts/bootstrap_viewer.sh",
                )
            )
        return FileResponse(
            path,
            media_type="text/plain" if name == "LICENSE" else "text/javascript",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "Cross-Origin-Resource-Policy": "same-origin",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/api/audio", status_code=201)
    def audio(
        file: UploadFile = File(...),
        dialog: str | None = Form(None),
        emotion: str = Form("auto"),
        emotion_strength: float = Form(0.65),
        backend: str = Form("auto"),
        fps: int = Form(30),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(AutoAnimError("BUSY", "Another job is currently running", retryable=True))
        temporary: Path | None = None
        try:
            temporary = _retain_upload(file)
            return service.audio(
                temporary,
                fps=fps,
                emotion=emotion,
                emotion_strength=emotion_strength,
                backend=backend,
                dialog=dialog,
                input_name=file.filename,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            operation_lock.release()

    @app.post("/api/image", status_code=201)
    def image(
        file: UploadFile = File(...),
        modes: int = Form(20),
        allow_low_confidence: bool = Form(False),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(AutoAnimError("BUSY", "Another job is currently running", retryable=True))
        temporary: Path | None = None
        try:
            temporary = _retain_upload(file)
            return service.image(
                temporary,
                modes=modes,
                allow_low_confidence=allow_low_confidence,
                input_name=file.filename,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            operation_lock.release()

    @app.post("/api/multiview", status_code=201)
    def multiview(
        files: list[UploadFile] = File(...),
        roles: str = Form(""),
        texture_size: int = Form(256),
        focal_scale: float = Form(1.25),
        mirror_fill: bool = Form(False),
        calibration: UploadFile | None = File(None),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(
                AutoAnimError("BUSY", "Another job is currently running", retryable=True)
            )
        temporary: list[Path] = []
        retained_calibration: Path | None = None
        try:
            if not 2 <= len(files) <= 12:
                raise AutoAnimError("INPUT_INVALID", "Upload 2-12 ordered face photos")
            total = 0
            for upload in files:
                retained = _retain_upload(upload)
                temporary.append(retained)
                total += retained.stat().st_size
                if total > 250 * 1024 * 1024:
                    raise AutoAnimError(
                        "LIMIT_EXCEEDED", "Combined multi-view upload exceeds 250 MiB"
                    )
            parsed_roles = tuple(value.strip() for value in roles.split(",") if value.strip())
            if calibration is not None:
                retained_calibration = _retain_upload(
                    calibration, max_bytes=1_000_000
                )
            return service.multiview(
                temporary,
                roles=parsed_roles or None,
                texture_size=texture_size,
                focal_scale=focal_scale,
                mirror_fill=mirror_fill,
                input_names=tuple(upload.filename or f"view-{index + 1}.bin" for index, upload in enumerate(files)),
                camera_bundle_path=retained_calibration,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
            if retained_calibration is not None:
                retained_calibration.unlink(missing_ok=True)
            operation_lock.release()

    @app.post("/api/video", status_code=201)
    def video(file: UploadFile = File(...)):
        if not operation_lock.acquire(blocking=False):
            return _error_response(
                AutoAnimError("BUSY", "Another job is currently running", retryable=True)
            )
        temporary: Path | None = None
        try:
            temporary = _retain_upload(file)
            return service.video(temporary, input_name=file.filename)
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            operation_lock.release()

    @app.get("/api/jobs")
    def jobs(limit: int = 20):
        return {"jobs": service.store.list_recent(limit=max(1, min(limit, 50)))}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return service.store.read(job_id)
        except FileNotFoundError:
            return _error_response(AutoAnimError("JOB_NOT_FOUND", "Job was not found"))

    @app.get("/api/jobs/{job_id}/files/{name}")
    def artifact(job_id: str, name: str):
        try:
            path = service.store.artifact(job_id, name)
            manifest = service.store.read(job_id)
            media_type = next(
                (
                    entry.get("media_type")
                    for entry in manifest.get("artifacts", {}).values()
                    if entry.get("name") == name
                ),
                None,
            )
            return FileResponse(path, media_type=media_type)
        except FileNotFoundError:
            return _error_response(AutoAnimError("ARTIFACT_NOT_FOUND", "Artifact was not found or allowlisted"))

    @app.get("/api/jobs/{job_id}/viewer", response_class=HTMLResponse)
    def viewer(job_id: str):
        try:
            manifest = service.store.read(job_id)
            artifacts = manifest.get("artifacts", {})
            glb = artifacts.get("textured_glb") or artifacts.get("glb")
            if not isinstance(glb, dict) or not isinstance(glb.get("name"), str):
                raise FileNotFoundError(job_id)
            name = glb["name"]
            # Resolve through the same manifest allowlist before producing a URL.
            service.store.artifact(job_id, name)
            media_url = None
            media_type = None
            viewer_contract = manifest.get("viewer", {})
            clock_key = viewer_contract.get("clock_artifact")
            if isinstance(clock_key, str):
                clock = artifacts.get(clock_key)
                if isinstance(clock, dict) and isinstance(clock.get("name"), str):
                    clock_name = clock["name"]
                    service.store.artifact(job_id, clock_name)
                    media_url = f"/api/jobs/{job_id}/files/{clock_name}"
                    media_type = (
                        clock.get("media_type")
                        if isinstance(clock.get("media_type"), str)
                        else None
                    )
            return HTMLResponse(
                viewer_html(
                    asset_url=f"/api/jobs/{job_id}/files/{name}",
                    title=(
                        "AutoAnim fitted face"
                        if manifest.get("kind") in {"image_fit", "multiview_reconstruction"}
                        else "AutoAnim face animation"
                    ),
                    media_url=media_url,
                    media_type=media_type,
                ),
                headers={
                    "Content-Security-Policy": (
                        "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                        "style-src 'unsafe-inline'; img-src 'self' data: blob:; "
                        "media-src 'self'; connect-src 'self' blob:; worker-src 'self' blob:; "
                        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                        "form-action 'none'"
                    ),
                    "Referrer-Policy": "no-referrer",
                    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except FileNotFoundError:
            return _error_response(
                AutoAnimError("ARTIFACT_NOT_FOUND", "This job has no viewable 3D asset")
            )

    return app


UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AutoAnim GNM</title>
  <style>
    :root{color-scheme:dark;--bg:#0b0d0f;--panel:#15191d;--line:#2b3239;--text:#f0f2f4;--muted:#9ba6b0;--accent:#d8ff63;--danger:#ff7d7d}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#202a30 0,transparent 32rem),var(--bg);color:var(--text);font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
    main{max-width:1180px;margin:auto;padding:54px 24px 80px}header{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:34px}
    h1{font:700 clamp(38px,7vw,82px)/.9 system-ui,sans-serif;letter-spacing:-.07em;margin:0}h1 span{color:var(--accent)}
    header p{max-width:420px;color:var(--muted);margin:0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
    .card{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 18px 50px #0005}
    h2{font:650 24px system-ui,sans-serif;margin:0 0 6px}.card>p{color:var(--muted);min-height:46px}label{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin:15px 0 6px}
    input,select,textarea,button{width:100%;border:1px solid var(--line);border-radius:10px;background:#0d1114;color:var(--text);padding:12px;font:inherit}textarea{min-height:76px;resize:vertical}
    button{background:var(--accent);color:#10130a;border:0;font-weight:800;margin-top:18px;cursor:pointer}button:disabled{opacity:.45;cursor:wait}
    .status{margin-top:18px;min-height:24px;color:var(--muted)}.error{color:var(--danger)}.result{display:none;margin-top:18px;border-top:1px solid var(--line);padding-top:18px}
    video,.result img{display:block;width:100%;max-height:520px;object-fit:contain;background:#050607;border-radius:12px}.links{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.links a{color:var(--accent);border:1px solid var(--line);border-radius:8px;padding:7px 10px;text-decoration:none}
    pre{overflow:auto;max-height:260px;background:#090b0d;padding:12px;border-radius:10px;font-size:11px}.note{margin-top:22px;color:var(--muted);border-left:3px solid var(--accent);padding-left:12px}.quality{margin:0 0 12px;padding:12px;border:1px solid var(--line);border-radius:10px;background:#0d1114}.quality strong{color:var(--accent)}.quality small{display:block;color:var(--muted);margin-top:5px}.timeline-wrap{margin-top:12px;padding:10px;background:#090b0d;border:1px solid var(--line);border-radius:10px}.timeline-wrap canvas{display:block;width:100%;height:112px}.timeline-readout{color:var(--muted);font-size:11px;margin-top:5px}.recent{margin-top:28px}.recent-head{display:flex;align-items:center;justify-content:space-between;gap:16px}.recent-head button{width:auto;margin:0;padding:8px 12px}.recent-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}.recent-job{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:14px;border:1px solid var(--line);border-radius:12px;background:#0d1114}.recent-job small{display:block;color:var(--muted)}.recent-job a{color:var(--accent);white-space:nowrap}.empty{color:var(--muted)}
    @media(max-width:760px){header{display:block}header p{margin-top:20px}.grid,.recent-list{grid-template-columns:1fr}}
  </style>
</head>
<body><main>
  <header><h1>Face motion,<br><span>made inspectable.</span></h1><p>Local GNM Head 3.0 workflows. Inputs stay on this machine; every result exposes controls, confidence, caveats, and downloadable artifacts. Runtime readiness is not production validation.</p></header>
  <section class="grid">
    <form class="card" id="audio-form"><h2>Audio → animation</h2><p>Learned Audio2Face motion is preferred, solved through named ARKit controls, then retargeted into GNM with a transparent procedural fallback.</p>
      <label for="audio-file">Audio</label><input id="audio-file" name="file" type="file" accept="audio/*" required>
      <label for="backend">Motion backend</label><select id="backend" name="backend"><option value="auto">Auto · learned preferred</option><option value="learned">Learned · require Audio2Face</option><option value="fallback">Procedural fallback</option></select>
      <label for="emotion">Emotion</label><select id="emotion" name="emotion"><option>auto</option><option>neutral</option><option>joy</option><option>sad</option><option>anger</option><option>fear</option><option>disgust</option><option>surprise</option><option>contempt</option></select>
      <label for="emotion-strength">Acting strength</label><input id="emotion-strength" name="emotion_strength" type="range" min="0" max="1" step="0.05" value="0.65">
      <label for="dialog">Optional dialog</label><textarea id="dialog" name="dialog" placeholder="Helps Rhubarb and lexical emotion hints"></textarea><input name="fps" type="hidden" value="30">
      <button>Build animation</button><div class="status"></div><div class="result"></div>
    </form>
    <form class="card" id="image-form"><h2>Image → neutral GNM</h2><p>A confidence-gated visible-geometry estimate. This is not a metric 3D clone.</p>
      <label for="image-file">Single face photo</label><input id="image-file" name="file" type="file" accept="image/png,image/jpeg,image/webp" required>
      <label for="modes">Observable identity modes</label><select id="modes" name="modes"><option value="20">20 · recommended</option><option value="10">10 · conservative</option></select>
      <label><input style="width:auto" type="checkbox" name="allow_low_confidence" value="true"> Allow low-confidence download</label>
      <button>Fit GNM face</button><div class="status"></div><div class="result"></div>
    </form>
    <form class="card" id="multiview-form"><h2>Multi-view → textured GNM</h2><p>One shared identity from ordered front, ¾, and profile captures, with directly observed texture clearly separated from filled regions.</p>
      <label for="multiview-files">Ordered face photos</label><input id="multiview-files" name="files" type="file" accept="image/png,image/jpeg,image/webp" multiple required>
      <label for="multiview-roles">Roles, in file order</label><input id="multiview-roles" name="roles" placeholder="front,left_3q,right_3q,left_profile,right_profile">
      <label for="multiview-calibration">Calibrated camera bundle (optional)</label><input id="multiview-calibration" name="calibration" type="file" accept="application/json,.json"><small>Production audit mode requires at least 3 fit cameras and 1 held-out camera. Filenames and upload order must match exactly.</small>
      <label for="texture-size">Texture atlas</label><select id="texture-size" name="texture_size"><option value="256">256 · test / fast</option><option value="512">512 · review</option><option value="1024">1024 · high detail</option><option value="128">128 · diagnostic</option></select>
      <input name="focal_scale" type="hidden" value="1.25">
      <button>Build textured face</button><div class="status"></div><div class="result"></div>
    </form>
    <form class="card" id="video-form"><h2>Video → performance</h2><p>Frame-accurate MediaPipe VIDEO tracking drives expression, head pose, translation, and gaze while keeping identity fixed. Begin with at least 0.2 seconds looking forward with a neutral face for tracker-bias calibration.</p>
      <label for="video-file">Face performance video</label><input id="video-file" name="file" type="file" accept="video/*" required>
      <button>Capture performance</button><div class="status"></div><div class="result"></div>
    </form>
  </section>
  <section class="recent" aria-labelledby="recent-title"><div class="recent-head"><h2 id="recent-title">Recent local runs</h2><button id="refresh-jobs" type="button">Refresh</button></div><div id="recent-list" class="recent-list" aria-live="polite"><p class="empty">Loading recent jobs…</p></div></section>
  <p class="note" id="health">Checking local model and native-tool readiness…</p>
</main>
<script>
const health=document.querySelector('#health');fetch('/api/health').then(r=>r.json()).then(x=>health.textContent=`Health: ${x.status}. GNM ${x.checks.gnm.ready?'ready':'missing'}, Audio2Face ${x.checks.a2f_runner?.ready&&x.checks.a2f_assets?.ready&&x.checks.a2f_provenance?.ready?'ready':'missing'}, MediaPipe ${x.checks.mediapipe_model.ready?'ready':'missing'}, Rhubarb ${x.checks.rhubarb.ready?'ready':'missing'}, offline viewer ${x.checks.viewer_bundle?.ready?'ready':'missing'}.`);
function artifactUrl(job,name){return `/api/jobs/${job}/files/${encodeURIComponent(name)}`}
const recentList=document.querySelector('#recent-list');async function refreshJobs(){try{const response=await fetch('/api/jobs?limit=8'),data=await response.json();recentList.innerHTML='';if(!data.jobs.length){recentList.innerHTML='<p class="empty">No jobs yet.</p>';return}for(const job of data.jobs){const row=document.createElement('article');row.className='recent-job';const copy=document.createElement('div');const title=document.createElement('strong');title.textContent=job.kind.replaceAll('_',' ');const detail=document.createElement('small');detail.textContent=`${job.input.name} · ${job.status} · ${job.warning_count} warning${job.warning_count===1?'':'s'}`;copy.append(title,detail);row.append(copy);if(job.viewable){const link=document.createElement('a');link.href=`/api/jobs/${job.job_id}/viewer`;link.target='_blank';link.textContent='Open 3D';row.append(link)}recentList.append(row)}}catch(error){recentList.innerHTML='<p class="empty">Recent jobs unavailable.</p>'}}
document.querySelector('#refresh-jobs').addEventListener('click',refreshJobs);refreshJobs();
for(const [formId,endpoint] of [['audio-form','/api/audio'],['image-form','/api/image'],['multiview-form','/api/multiview'],['video-form','/api/video']]){
 const form=document.getElementById(formId),status=form.querySelector('.status'),result=form.querySelector('.result'),button=form.querySelector('button');
 form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;status.className='status';status.textContent='Processing locally…';result.style.display='none';
  try{const response=await fetch(endpoint,{method:'POST',body:new FormData(form)}),data=await response.json();if(!response.ok)throw new Error(`${data.code}: ${data.message}`);status.textContent=`Succeeded · ${data.job_id}`;result.innerHTML='';
   if(data.kind==='audio_animation'){const q=document.createElement('div');q.className='quality';const learned=data.analysis.motion_backend==='learned_a2f';const title=document.createElement('strong');title.textContent=learned?'Learned motion · geometry calibrated':'Procedural fallback · not production';q.append(title);const detail=document.createElement('small');detail.textContent=`Stationary speech transitions ${(100*data.metrics.lower_face_stationary_fraction).toFixed(1)}% · mouth-step p95 ${data.metrics.mouth_step_p95_interocular.toFixed(3)} IOD · limited frames ${data.metrics.mouth_speed_limited_frames}. ${data.warnings.join(' ')}`;q.append(detail);result.append(q)}
   if(data.kind==='image_fit'){const q=document.createElement('div');q.className='quality';const title=document.createElement('strong');title.textContent=`Visible-geometry fit · ${data.fit.confidence} confidence`;q.append(title);const detail=document.createElement('small');detail.textContent=`Landmark NME ${data.fit.nme.toFixed(4)} · stability ${data.fit.stability_rms.toFixed(4)} · ${(100*data.fit.coefficient_bound_fraction).toFixed(1)}% coefficients at bounds. This neutral fit does not reconstruct hidden geometry or metric depth. ${data.warnings.join(' ')}`;q.append(detail);result.append(q)}
	   if(data.kind==='video_performance'){const q=document.createElement('div');q.className='quality';const title=document.createElement('strong');title.textContent=data.retargeting.geometry_calibrated?'Video performance · geometry calibrated':'Video performance · semantic fallback';q.append(title);const contact=data.metrics.final_contact_geometry_attained_fraction;const contactText=contact===null?'no scored closure':`${(100*contact).toFixed(1)}% contact attained`;const detail=document.createElement('small');detail.textContent=`Face presence ${(100*data.metrics.face_presence_fraction).toFixed(1)}% · ${contactText} · expression timing ${data.metrics.final_expression_motion_correlation===null?'n/a':data.metrics.final_expression_motion_correlation.toFixed(3)} · baseline loss ${(100*data.metrics.negative_baseline_residual_clipped_fraction).toFixed(1)}% · proxy timing error ${data.metrics.proxy_pts_max_error_ms.toFixed(2)} ms. ${data.warnings.join(' ')}`;q.append(detail);result.append(q)}
   if(data.kind==='multiview_reconstruction'){const q=document.createElement('div');q.className='quality';const title=document.createElement('strong');title.textContent='Shared identity · provenance-aware texture';q.append(title);const detail=document.createElement('small');const holdout=data.capture.held_out?.evaluated?` · held-out NME ${data.capture.held_out.aggregate_nme.toFixed(4)} (${data.capture.held_out.passed?'pass':'fail'})`:'';detail.textContent=`Fit NME ${data.fit.nme.toFixed(4)}${holdout} · direct texture ${(100*data.texture.observed_fraction).toFixed(1)}% · ${data.capture.accepted_view_indices.length}/${data.capture.view_count} views accepted. ${data.warnings.join(' ')}`;q.append(detail);result.append(q)}
   let video=null;const media=data.artifacts.preview||data.artifacts.viewer_media||data.artifacts.overlay||data.artifacts.mesh_preview;if(media){const url=artifactUrl(data.job_id,media.name);if(media.media_type==='video/mp4'){video=document.createElement('video');video.controls=true;video.playsInline=true;video.src=url;result.append(video)}else{const image=document.createElement('img');image.src=url;image.alt='Result preview';result.append(image)}}
   if(video&&data.artifacts.timeline){const wrap=document.createElement('div');wrap.className='timeline-wrap';const canvas=document.createElement('canvas');canvas.width=800;canvas.height=112;const readout=document.createElement('div');readout.className='timeline-readout';wrap.append(canvas,readout);result.append(wrap);fetch(artifactUrl(data.job_id,data.artifacts.timeline.name)).then(r=>r.json()).then(t=>{const ctx=canvas.getContext('2d'),n=t.timestamps.length,ap=t.mouth_aperture||[],en=t.energy||[],amin=Math.min(...ap),ar=Math.max(...ap)-amin||1;function curve(values,color,norm){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{const x=i/(n-1)*canvas.width,y=canvas.height-8-norm(v)*(canvas.height-16);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#090b0d';ctx.fillRect(0,0,canvas.width,canvas.height);curve(en,'#6fb9ff',v=>v);curve(ap,'#d8ff63',v=>(v-amin)/ar);const duration=t.timestamps[n-1]||video.duration||1,x=Math.min(1,video.currentTime/duration)*canvas.width;ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();const cue=(data.analysis.cues||[]).find(c=>video.currentTime>=c.start&&video.currentTime<c.end);readout.textContent=`${video.currentTime.toFixed(2)} s · cue ${cue?cue.value:'X'} · green aperture · blue energy`}video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);draw()}).catch(()=>{readout.textContent='Timeline unavailable'})}
   const links=document.createElement('div');links.className='links';if(data.artifacts.glb||data.artifacts.textured_glb){const view=document.createElement('a');view.href=`/api/jobs/${data.job_id}/viewer`;view.textContent='Open interactive 3D';view.target='_blank';links.append(view)}for(const [key,item] of Object.entries(data.artifacts)){const a=document.createElement('a');a.href=artifactUrl(data.job_id,item.name);a.textContent=key;a.download=item.name;links.append(a)}result.append(links);const pre=document.createElement('pre');pre.textContent=JSON.stringify(data,null,2);result.append(pre);result.style.display='block';refreshJobs();
  }catch(error){status.className='status error';status.textContent=error.message}finally{button.disabled=false}}
 );
}
</script></body></html>"""
