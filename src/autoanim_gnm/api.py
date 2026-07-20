"""FastAPI transport and a dependency-free local web UI."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
import re
import secrets
import tempfile
import threading

import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .errors import AutoAnimError
from .artifacts import sha256
from .capture_session import (
    load_verified_legacy_video_capture_session_v1,
    load_verified_video_capture_session,
)
from .service import ApplicationService
from .video_capture import (
    load_capture_npz,
    load_verified_capture_jsonl,
    probe_video,
)
from .video_evidence import load_verified_performance_evidence
from .video_observation import (
    MAX_OBSERVATION_V3_VIEW_FRAMES,
    build_observation_v3_view,
    load_pixel_observations,
    load_verified_observation_v3_summary,
)
from .viewer import VIEWER_THREE_VERSION, VIEWER_VENDOR_FILES, viewer_html


STATUS_BY_CODE = {
    "SESSION_UNAUTHORIZED": 401,
    "HOST_INVALID": 400,
    "INPUT_INVALID": 400,
    "MEDIA_INVALID": 400,
    "AUDIO_SILENT": 400,
    "CUE_INVALID": 400,
    "JOB_NOT_FOUND": 404,
    "CHARACTER_NOT_FOUND": 404,
    "ARTIFACT_NOT_FOUND": 404,
    "CONSENT_REQUIRED": 422,
    "CONSENT_REVOKED": 403,
    "CONSENT_EXPIRED": 403,
    "CONSENT_SCOPE_DENIED": 403,
    "RIGHTS_EXPIRED": 403,
    "MATERIAL_INVALID": 422,
    "MATERIAL_BINDING_REQUIRED": 422,
    "MATERIAL_BINDING_MISMATCH": 409,
    "MATERIAL_LAYOUT_UNSUPPORTED": 422,
    "MATERIAL_RUNTIME_UNSUPPORTED": 422,
    "REVISION_CONFLICT": 409,
    "INTEGRITY_FAILED": 409,
    "INTEGRITY_UNSEALED": 409,
    "BUSY": 409,
    "LIMIT_EXCEEDED": 413,
    "FACE_NOT_FOUND": 422,
    "MULTIPLE_FACES": 422,
    "FIT_REJECTED": 422,
    "PHONE_EVIDENCE_INVALID": 422,
    "IDENTITY_QUALIFICATION_INVALID": 422,
    "INPUT_CHANGED": 409,
    "DEPENDENCY_MISSING": 424,
    "LLM_UNAVAILABLE": 424,
    "LLM_TIMEOUT": 504,
    "LLM_OUTPUT_TOO_LARGE": 413,
    "LLM_EXIT_NONZERO": 502,
    "LLM_STREAM_PROTOCOL": 502,
    "LLM_TOOL_USE_FORBIDDEN": 422,
    "LLM_JSON_PARSE": 502,
    "LLM_SCHEMA_INVALID": 422,
    "LLM_SEMANTIC_INVALID": 422,
    "LLM_REFUSAL": 422,
    "LLM_NEEDS_INPUT": 422,
    "INTERNAL_ERROR": 500,
}

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})
_HEX_TOKEN = re.compile(r"[0-9a-fA-F]+\Z")
_IDENTITY_PIXEL_TRANSFORM = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
_OBSERVATION_V3_LEGACY_CHAIN_NAMES = (
    "capture",
    "capture_jsonl",
    "performance_evidence",
    "pixel_observations",
    "observation_v3",
    "capture_session",
)
_OBSERVATION_V3_V2_EXTENSION_NAMES = (
    "video_capture_run",
    "visual_track",
    "visual_track_summary",
)
_OBSERVATION_V3_V2_CHAIN_NAMES = (
    *_OBSERVATION_V3_LEGACY_CHAIN_NAMES[:-1],
    *_OBSERVATION_V3_V2_EXTENSION_NAMES,
    "capture_session",
)


def _observation_v3_chain_names(artifacts: dict[str, object]) -> tuple[str, ...]:
    """Select v2 if any v2 evidence is declared; never downgrade partial v2."""

    if any(name in artifacts for name in _OBSERVATION_V3_V2_EXTENSION_NAMES):
        return _OBSERVATION_V3_V2_CHAIN_NAMES
    return _OBSERVATION_V3_LEGACY_CHAIN_NAMES


def _review_display_geometry_compatible(
    viewer_contract: object,
    capture_summary: object,
) -> bool:
    """Return whether a job can safely expose display-proxy ROI review."""

    if not isinstance(viewer_contract, dict) or not isinstance(
        capture_summary, dict
    ):
        return False
    width = capture_summary.get("width")
    height = capture_summary.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        return False
    return (
        viewer_contract.get("clock_artifact") == "viewer_media"
        and viewer_contract.get("display_geometry")
        == {
            "schema_version": "autoanim.viewer-display-binding/1.0",
            "artifact": "viewer_media",
            "source_frame_size": [width, height],
            "proxy_frame_size": [width, height],
            "display_rotation_degrees": 0,
            "sample_aspect_ratio": [1, 1],
            "clean_aperture_crop_ltrb": [0, 0, 0, 0],
            "source_to_display_pixel_transform": _IDENTITY_PIXEL_TRANSFORM,
            "transcode_policy": (
                "ffmpeg_h264_pts_passthrough_no_geometry_filters_v1"
            ),
        }
    )


def _error_response(error: AutoAnimError) -> JSONResponse:
    return JSONResponse(status_code=STATUS_BY_CODE.get(error.code, 500), content=error.as_dict())


def _session_token_digest(session_token: str) -> bytes:
    """Validate a transport token and return a fixed-length comparison value."""
    if not isinstance(session_token, str):
        raise AutoAnimError("INPUT_INVALID", "Session token must be text")
    try:
        encoded = session_token.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AutoAnimError("INPUT_INVALID", "Session token must be valid UTF-8") from exc
    if not encoded or len(encoded) > 4096:
        raise AutoAnimError(
            "INPUT_INVALID", "Session token must encode between 32 and 4096 bytes"
        )
    # A hex-looking token encodes four bits per character, so it must contain
    # at least 64 characters. Other opaque tokens must contain at least 32 bytes.
    if _HEX_TOKEN.fullmatch(session_token):
        if len(session_token) % 2 or len(session_token) < 64:
            raise AutoAnimError(
                "INPUT_INVALID", "Hex session token must encode at least 256 bits"
            )
    elif len(encoded) < 32:
        raise AutoAnimError(
            "INPUT_INVALID", "Session token must contain at least 32 random bytes"
        )
    return hashlib.sha256(encoded).digest()


def _parse_host_header(value: str) -> tuple[str, int | None] | None:
    """Parse the narrow Host grammar accepted by the native loopback server."""
    if not value or value != value.strip() or any(char.isspace() for char in value):
        return None
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            return None
        hostname = value[1:closing].lower()
        remainder = value[closing + 1 :]
        if remainder:
            port_text = remainder[1:]
            if (
                not remainder.startswith(":")
                or not port_text.isdigit()
                or len(port_text) > 5
            ):
                return None
            port = int(port_text)
        else:
            port = None
    else:
        if "[" in value or "]" in value or value.count(":") > 1:
            return None
        if ":" in value:
            hostname, separator, port_text = value.rpartition(":")
            if (
                not separator
                or not hostname
                or not port_text.isdigit()
                or len(port_text) > 5
            ):
                return None
            port = int(port_text)
        else:
            hostname = value
            port = None
        hostname = hostname.lower()
    if hostname not in _LOOPBACK_HOSTS or (port is not None and not 1 <= port <= 65535):
        return None
    return hostname, port


def _host_matches_bound_port(request: Request) -> bool:
    raw_hosts = [
        value.decode("latin-1")
        for name, value in request.scope.get("headers", ())
        if name.lower() == b"host"
    ]
    if len(raw_hosts) != 1:
        return False
    parsed = _parse_host_header(raw_hosts[0])
    if parsed is None:
        return False
    _, presented_port = parsed
    server = request.scope.get("server")
    if not server or len(server) < 2 or not isinstance(server[1], int):
        return False
    bound_port = server[1]
    if presented_port is not None:
        return presented_port == bound_port
    default_port = 443 if request.url.scheme == "https" else 80
    return bound_port == default_port


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
    character_root: str | Path | None = None,
    session_token: str | None = None,
) -> FastAPI:
    session_digest = (
        _session_token_digest(session_token) if session_token is not None else None
    )
    app = FastAPI(title="AutoAnim GNM", version="0.1.0")
    service = ApplicationService(
        artifact_root,
        model_path=model_path,
        rhubarb_bin=rhubarb_bin,
        a2f_runner=a2f_runner,
        a2f_asset_dir=a2f_asset_dir,
        a2f_offline=a2f_offline,
        viewer_vendor_root=viewer_vendor_root,
        character_root=character_root,
    )
    operation_lock = threading.Lock()
    review_decode_lock = threading.Lock()
    app.state.service = service

    if session_digest is not None:

        @app.middleware("http")
        async def require_native_session(request: Request, call_next):
            if not _host_matches_bound_port(request):
                return _error_response(
                    AutoAnimError(
                        "HOST_INVALID",
                        "Host must match the bound loopback server and port",
                    )
                )

            header_values = [
                value.decode("latin-1")
                for name, value in request.scope.get("headers", ())
                if name.lower() == b"x-autoanim-token"
            ]
            candidates: list[str] = []
            if len(header_values) == 1:
                candidates.append(header_values[0])
            cookie_token = request.cookies.get("autoanim_session")
            if cookie_token is not None:
                candidates.append(cookie_token)
            authenticated = any(
                secrets.compare_digest(
                    hashlib.sha256(candidate.encode("utf-8")).digest(), session_digest
                )
                for candidate in candidates
                if len(candidate.encode("utf-8")) <= 4096
            )
            if not authenticated:
                response = _error_response(
                    AutoAnimError(
                        "SESSION_UNAUTHORIZED",
                        "A valid native session credential is required",
                    )
                )
                response.headers["Cache-Control"] = "no-store"
                return response
            return await call_next(request)

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
        phone_textgrid: UploadFile | None = File(None),
        dialog: str | None = Form(None),
        emotion: str = Form("auto"),
        emotion_strength: float = Form(0.65),
        mouth_aperture_gain: float = Form(1.0),
        mouth_aperture_author: str = Form(""),
        mouth_aperture_reason: str = Form(""),
        backend: str = Form("auto"),
        fps: int = Form(30),
        a2f_v3_local_seed: int = Form(0),
        character_id: str = Form(""),
        character_revision_id: str = Form(""),
        usage_scope: str = Form("production"),
        phone_annotations_reviewed: bool = Form(False),
        phone_reviewer: str = Form(""),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(AutoAnimError("BUSY", "Another job is currently running", retryable=True))
        temporary: Path | None = None
        temporary_textgrid: Path | None = None
        try:
            temporary = _retain_upload(file)
            if phone_textgrid is not None:
                temporary_textgrid = _retain_upload(
                    phone_textgrid, max_bytes=8 * 1024 * 1024
                )
            return service.audio(
                temporary,
                fps=fps,
                emotion=emotion,
                emotion_strength=emotion_strength,
                mouth_aperture_gain=mouth_aperture_gain,
                mouth_aperture_author=mouth_aperture_author or None,
                mouth_aperture_reason=mouth_aperture_reason or None,
                backend=backend,
                a2f_v3_local_seed=a2f_v3_local_seed,
                dialog=dialog,
                input_name=file.filename,
                character_id=character_id or None,
                character_revision_id=character_revision_id or None,
                usage_scope=usage_scope,
                phone_annotation_path=temporary_textgrid,
                phone_annotation_name=(
                    phone_textgrid.filename if phone_textgrid is not None else None
                ),
                phone_annotations_independently_reviewed=phone_annotations_reviewed,
                phone_annotation_reviewer=phone_reviewer or None,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if temporary_textgrid is not None:
                temporary_textgrid.unlink(missing_ok=True)
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
    def video(
        file: UploadFile = File(...),
        character_id: str = Form(""),
        character_revision_id: str = Form(""),
        usage_scope: str = Form("production"),
        audio_visual_repair: bool = Form(False),
        mouth_aperture_gain: float = Form(1.0),
        mouth_aperture_author: str = Form(""),
        mouth_aperture_reason: str = Form(""),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(
                AutoAnimError("BUSY", "Another job is currently running", retryable=True)
            )
        temporary: Path | None = None
        try:
            temporary = _retain_upload(file)
            return service.video(
                temporary,
                input_name=file.filename,
                character_id=character_id or None,
                character_revision_id=character_revision_id or None,
                usage_scope=usage_scope,
                audio_visual_repair=audio_visual_repair,
                mouth_aperture_gain=mouth_aperture_gain,
                mouth_aperture_author=mouth_aperture_author or None,
                mouth_aperture_reason=mouth_aperture_reason or None,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            operation_lock.release()

    @app.get("/api/jobs")
    def jobs(limit: int = 20):
        return {"jobs": service.store.list_recent(limit=max(1, min(limit, 50)))}

    @app.get("/api/jobs/{job_id}/production-readiness")
    def production_readiness(
        job_id: str,
        direction_job_id: str | None = None,
        identity_qualification_job_id: str | None = None,
        require_acting: bool = False,
        require_body: bool = False,
        require_pbr: bool = True,
    ):
        try:
            return service.production_readiness(
                job_id,
                direction_job_id=direction_job_id,
                identity_qualification_job_id=identity_qualification_job_id,
                require_acting=require_acting,
                require_body=require_body,
                require_pbr=require_pbr,
            )
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get("/api/jobs/{job_id}/identity-qualification")
    def identity_qualification(job_id: str):
        try:
            return service.identity_qualification(job_id)
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.post("/api/direction", status_code=201)
    def direction(
        source_job_id: str = Form(...),
        provider: str = Form("codex"),
        instructions: str = Form(""),
        transcript: str = Form(""),
        character_id: str = Form(""),
        character_revision_id: str = Form(""),
        usage_scope: str = Form("production"),
        model: str = Form(""),
        timeout_seconds: int = Form(180),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(
                AutoAnimError("BUSY", "Another job is currently running", retryable=True)
            )
        try:
            return service.direct(
                source_job_id,
                provider=provider,
                instructions=instructions,
                transcript=transcript,
                character_id=character_id or None,
                character_revision_id=character_revision_id or None,
                usage_scope=usage_scope,
                model=model or None,
                timeout_seconds=timeout_seconds,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            operation_lock.release()

    @app.get("/api/characters")
    def characters(limit: int = 100):
        return {"characters": service.characters.list(limit=max(1, min(limit, 200)))}

    @app.post("/api/characters/from-job", status_code=201)
    def create_character_from_job(
        job_id: str = Form(...),
        name: str = Form(...),
        consent_attested: bool = Form(False),
        consent_subject: str = Form(""),
        consent_attester: str = Form(""),
        consent_scope: str = Form("production"),
        consent_evidence_ref: str = Form(""),
        consent_evidence: UploadFile = File(...),
        consent_expires_at: str | None = Form(None),
        consent_note: str | None = Form(None),
    ):
        if not operation_lock.acquire(blocking=False):
            return _error_response(
                AutoAnimError("BUSY", "Another job is currently running", retryable=True)
            )
        evidence_path: Path | None = None
        try:
            evidence_path = _retain_upload(consent_evidence, max_bytes=10 * 1024 * 1024)
            return service.promote_character(
                job_id,
                name=name,
                consent_attested=consent_attested,
                consent_subject=consent_subject,
                consent_attester=consent_attester,
                consent_scope=consent_scope,
                consent_evidence_ref=consent_evidence_ref,
                consent_evidence_sha256=sha256(evidence_path),
                consent_expires_at=consent_expires_at,
                consent_note=consent_note,
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            if evidence_path is not None:
                evidence_path.unlink(missing_ok=True)
            operation_lock.release()

    @app.get("/api/characters/{character_id}")
    def character(character_id: str):
        try:
            return service.characters.read(character_id)
        except FileNotFoundError:
            return _error_response(AutoAnimError("CHARACTER_NOT_FOUND", "Character was not found"))
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.post("/api/characters/{character_id}/revoke")
    def revoke_character(
        character_id: str,
        reason: str = Form(...),
        revoked_by: str = Form(...),
    ):
        try:
            return service.characters.revoke(
                character_id, reason=reason, revoked_by=revoked_by
            )
        except FileNotFoundError:
            return _error_response(
                AutoAnimError("CHARACTER_NOT_FOUND", "Character was not found")
            )
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get(
        "/api/characters/{character_id}/revisions/{revision_id}"
    )
    def character_revision(
        character_id: str,
        revision_id: str,
        usage_scope: str = "personal",
    ):
        try:
            revision = service.characters.resolve(
                character_id, revision_id, usage_scope=usage_scope
            )
            return {
                "character_id": revision.character_id,
                "revision_id": revision.revision_id,
                "name": revision.name,
                "revision_manifest_sha256": revision.manifest_sha256,
                "identity_sha256": revision.identity_sha256,
                "texture_uvs_sha256": revision.texture_uvs_sha256,
                "texture_uvs_array_sha256": revision.texture_uvs_array_sha256,
                "material_descriptor_sha256": revision.material_manifest_sha256,
                "material_map_sha256s": dict(revision.material_sha256s),
                "runtime_material_sha256s": dict(
                    revision.runtime_material_sha256s
                ),
                "appearance": revision.manifest.get("appearance"),
                "production_validated": bool(
                    revision.manifest.get("production_validated", False)
                ),
            }
        except FileNotFoundError:
            return _error_response(
                AutoAnimError("CHARACTER_NOT_FOUND", "Character revision was not found")
            )
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get(
        "/api/characters/{character_id}/revisions/{revision_id}/files/{logical_name}"
    )
    def character_asset(
        character_id: str,
        revision_id: str,
        logical_name: str,
        usage_scope: str = "personal",
    ):
        try:
            revision = service.characters.resolve(
                character_id, revision_id, usage_scope=usage_scope
            )
            path = service.characters.asset(
                character_id,
                revision_id,
                logical_name,
                usage_scope=usage_scope,
                _resolved=revision,
            )
            return FileResponse(path)
        except FileNotFoundError:
            return _error_response(
                AutoAnimError("ARTIFACT_NOT_FOUND", "Character artifact was not found")
            )
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get("/api/characters/{character_id}/viewer", response_class=HTMLResponse)
    def character_viewer(
        character_id: str,
        revision_id: str | None = None,
        usage_scope: str = "personal",
    ):
        try:
            revision = service.characters.resolve(
                character_id, revision_id, usage_scope=usage_scope
            )
            url = (
                f"/api/characters/{character_id}/revisions/{revision.revision_id}"
                f"/files/preview?usage_scope={usage_scope}"
            )
            return HTMLResponse(
                viewer_html(
                    asset_url=url,
                    title=f"AutoAnim character · {revision.name}",
                    metadata={
                        "revision": revision.revision_id,
                        "package": (
                            revision.manifest.get("appearance", {}).get(
                                "material_package_id"
                            )
                            if isinstance(
                                revision.manifest.get("appearance"), dict
                            )
                            else None
                        ),
                        "resolution_claim": (
                            revision.manifest.get("appearance", {}).get(
                                "resolution_label"
                            )
                            if isinstance(
                                revision.manifest.get("appearance"), dict
                            )
                            else None
                        ),
                        "runtime_maps": sorted(revision.runtime_material_sha256s),
                        "retained_maps": sorted(revision.material_sha256s),
                        "pore_frequency_validated": False,
                        "unseen_light_validated": False,
                        "production_validated": False,
                    },
                ),
                headers={
                    "Content-Security-Policy": (
                        "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                        "style-src 'unsafe-inline'; img-src 'self' data: blob:; "
                        "connect-src 'self' blob:; worker-src 'self' blob:; "
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
                AutoAnimError("CHARACTER_NOT_FOUND", "Character was not found")
            )
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return service.store.read(job_id)
        except FileNotFoundError:
            return _error_response(AutoAnimError("JOB_NOT_FOUND", "Job was not found"))
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get("/api/jobs/{job_id}/files/{name}")
    def artifact(job_id: str, name: str):
        try:
            path = service.store.artifact(job_id, name)
            manifest = service.store.require_sealed(job_id)
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
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get("/api/jobs/{job_id}/observation-v3-view")
    def observation_v3_view(job_id: str):
        """Reconstruct a bounded viewer document from sealed video evidence."""

        try:
            manifest = service.store.require_sealed(job_id)
            if (
                manifest.get("kind") != "video_performance"
                or manifest.get("status") != "succeeded"
            ):
                raise FileNotFoundError(job_id)
            artifacts = manifest.get("artifacts", {})
            if not isinstance(artifacts, dict):
                raise ValueError("Sealed artifact ledger is invalid")

            def artifact_path(logical_name: str) -> Path:
                entry = artifacts.get(logical_name)
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                    raise FileNotFoundError(logical_name)
                return service.store.artifact(job_id, entry["name"])

            chain_names = _observation_v3_chain_names(artifacts)
            is_v2_chain = chain_names == _OBSERVATION_V3_V2_CHAIN_NAMES
            paths = {name: artifact_path(name) for name in chain_names}
            capture_path = paths["capture"]
            pixels_path = paths["pixel_observations"]
            summary_path = paths["observation_v3"]
            capture = load_capture_npz(capture_path)
            if capture.frame_count > MAX_OBSERVATION_V3_VIEW_FRAMES:
                raise AutoAnimError(
                    "LIMIT_EXCEEDED",
                    "Interactive Observation-v3 review is limited to "
                    f"{MAX_OBSERVATION_V3_VIEW_FRAMES} frames",
                )
            observations = load_pixel_observations(pixels_path)
            input_ledger = manifest.get("input")
            retained_inputs = [
                path
                for path in service.store.job_dir(job_id).glob("input.*")
                if path.is_file()
            ]
            if (
                not isinstance(input_ledger, dict)
                or len(retained_inputs) != 1
                or retained_inputs[0].stat().st_size != input_ledger.get("bytes")
                or sha256(retained_inputs[0]) != input_ledger.get("sha256")
                or capture.provenance.source_sha256 != input_ledger.get("sha256")
                or capture.provenance.source_bytes != input_ledger.get("bytes")
            ):
                raise ValueError("Capture source does not match the sealed retained input")
            observations.validate_capture(capture)
            load_verified_capture_jsonl(paths["capture_jsonl"], capture)
            load_verified_performance_evidence(
                paths["performance_evidence"],
                expected_source_sha256=capture.provenance.source_sha256,
                expected_frame_count=capture.frame_count,
                expected_capture=capture,
            )
            summary = load_verified_observation_v3_summary(
                summary_path,
                pixel_observations_path=pixels_path,
                capture_artifact_path=capture_path,
                expected_capture=capture,
                expected_observations=observations,
            )
            capture_session_artifact_paths = {
                name: paths[name] for name in chain_names if name != "capture_session"
            }
            if is_v2_chain:
                load_verified_video_capture_session(
                    paths["capture_session"],
                    expected_capture=capture,
                    expected_observations=observations,
                    artifact_paths=capture_session_artifact_paths,
                    artifact_contracts_preverified=True,
                )
            else:
                load_verified_legacy_video_capture_session_v1(
                    paths["capture_session"],
                    expected_capture=capture,
                    expected_observations=observations,
                    artifact_paths=capture_session_artifact_paths,
                    artifact_contracts_preverified=True,
                )
            viewer_contract = manifest.get("viewer")
            capture_summary = manifest.get("capture")
            if not _review_display_geometry_compatible(
                viewer_contract, capture_summary
            ):
                raise ValueError("Viewer display geometry is not review-safe")
            clock_key = (
                viewer_contract.get("clock_artifact")
                if isinstance(viewer_contract, dict)
                else None
            )
            if not isinstance(clock_key, str):
                raise ValueError("Viewer clock artifact is not declared")
            display_path = artifact_path(clock_key)
            display_probe = probe_video(display_path)
            timestamp_error = max(
                (
                    abs(float(left) - float(right))
                    for left, right in zip(
                        capture.timestamps_seconds,
                        display_probe.timestamps_seconds,
                        strict=True,
                    )
                ),
                default=0.0,
            )
            display_start = float(
                Fraction(int(display_probe.source_pts[0]))
                * display_probe.time_base
            )
            display_geometry = viewer_contract.get("display_geometry")
            expected_display_geometry = {
                "schema_version": "autoanim.viewer-display-binding/1.0",
                "artifact": clock_key,
                "source_frame_size": [capture.width, capture.height],
                "proxy_frame_size": [
                    display_probe.width,
                    display_probe.height,
                ],
                "display_rotation_degrees": (
                    display_probe.display_rotation_degrees
                ),
                "sample_aspect_ratio": [
                    display_probe.sample_aspect_ratio_numerator,
                    display_probe.sample_aspect_ratio_denominator,
                ],
                "clean_aperture_crop_ltrb": list(
                    display_probe.clean_aperture_crop
                ),
                "source_to_display_pixel_transform": _IDENTITY_PIXEL_TRANSFORM,
                "transcode_policy": (
                    "ffmpeg_h264_pts_passthrough_no_geometry_filters_v1"
                ),
            }
            if (
                display_geometry != expected_display_geometry
                or display_probe.frame_count != capture.frame_count
                or [display_probe.width, display_probe.height]
                != [capture.width, capture.height]
                or display_probe.display_rotation_degrees != 0
                or display_probe.sample_aspect_ratio_numerator
                != display_probe.sample_aspect_ratio_denominator
                or display_probe.clean_aperture_crop != (0, 0, 0, 0)
                or timestamp_error > 0.002
                or abs(display_start) > 0.002
            ):
                raise ValueError("Viewer proxy pixels or presentation clock differ from capture")
            integrity = manifest.get("integrity", {})
            evidence_binding = {
                "chainVerified": True,
                "manifestSha256": sha256(
                    service.store.job_dir(job_id) / "result.json"
                ),
                "sealSchema": integrity.get("schema"),
                "sealKeyId": integrity.get("key_id"),
                "retainedSource": {
                    "sha256": input_ledger["sha256"],
                    "bytes": input_ledger["bytes"],
                },
                "artifacts": {
                    name: {
                        "name": artifacts[name]["name"],
                        "sha256": artifacts[name]["sha256"],
                        "bytes": artifacts[name]["bytes"],
                    }
                    for name in chain_names
                },
            }
            display_entry = artifacts[clock_key]
            display_binding = {
                "clockVerified": True,
                "artifact": {
                    "logicalName": clock_key,
                    "name": display_entry["name"],
                    "sha256": display_entry["sha256"],
                    "bytes": display_entry["bytes"],
                },
                "frameCount": display_probe.frame_count,
                "frameSize": [display_probe.width, display_probe.height],
                "displayRotationDegrees": display_probe.display_rotation_degrees,
                "timestampMaxErrorSeconds": timestamp_error,
                "frameTimestampsSeconds": [
                    float(value) for value in display_probe.timestamps_seconds
                ],
                "mediaStartSeconds": display_start,
                "sampleAspectRatio": [
                    display_probe.sample_aspect_ratio_numerator,
                    display_probe.sample_aspect_ratio_denominator,
                ],
                "cleanApertureCropLTRB": list(
                    display_probe.clean_aperture_crop
                ),
                "coordinateSpace": "display_oriented_rgb_pixels",
                "sourceToDisplayPixelTransform": _IDENTITY_PIXEL_TRANSFORM,
                "generationContract": expected_display_geometry,
            }
            return JSONResponse(
                build_observation_v3_view(
                    capture,
                    observations,
                    summary,
                    evidence_binding=evidence_binding,
                    display_binding=display_binding,
                ),
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except FileNotFoundError:
            return _error_response(
                AutoAnimError(
                    "ARTIFACT_NOT_FOUND",
                    "This job has no complete Observation-v3 review evidence",
                )
            )
        except ValueError as exc:
            return _error_response(
                AutoAnimError(
                    "INTEGRITY_FAILED",
                    f"Observation-v3 review evidence did not verify: {exc}",
                )
            )
        except AutoAnimError as exc:
            return _error_response(exc)

    @app.get("/api/jobs/{job_id}/review-frames/{frame_index}.png")
    def review_frame(job_id: str, frame_index: int):
        """Decode one manifest-bound proxy frame for deterministic paused review.

        Observation-v3 verification is deliberately a separate route and client
        prerequisite. This endpoint binds only the returned pixels to the sealed
        ``viewer_media`` artifact so unrelated diagnostic corruption cannot make
        otherwise valid source pixels unavailable.
        """

        if not review_decode_lock.acquire(blocking=False):
            return _error_response(
                AutoAnimError(
                    "BUSY",
                    "Another exact review frame is being decoded",
                    retryable=True,
                )
            )
        try:
            manifest = service.store.require_sealed(job_id)
            if (
                manifest.get("kind") != "video_performance"
                or manifest.get("status") != "succeeded"
            ):
                raise FileNotFoundError(job_id)
            capture_summary = manifest.get("capture")
            frame_count = (
                capture_summary.get("frames")
                if isinstance(capture_summary, dict)
                else None
            )
            if (
                not isinstance(frame_count, int)
                or frame_count <= 0
                or frame_count > MAX_OBSERVATION_V3_VIEW_FRAMES
                or frame_index < 0
                or frame_index >= frame_count
            ):
                raise AutoAnimError(
                    "INPUT_INVALID",
                    "Review frame index is outside the interactive take",
                )
            viewer_contract = manifest.get("viewer")
            if not _review_display_geometry_compatible(
                viewer_contract, capture_summary
            ):
                raise ValueError("Viewer display geometry is not review-safe")
            display_geometry = viewer_contract["display_geometry"]
            artifacts = manifest.get("artifacts", {})
            display_entry = (
                artifacts.get("viewer_media")
                if isinstance(artifacts, dict)
                else None
            )
            if not isinstance(display_entry, dict) or not isinstance(
                display_entry.get("name"), str
            ):
                raise FileNotFoundError("viewer_media")
            display_path = service.store.artifact(
                job_id, display_entry["name"]
            )
            decoder = cv2.VideoCapture(str(display_path))
            try:
                if not decoder.isOpened() or not decoder.set(
                    cv2.CAP_PROP_POS_FRAMES, frame_index
                ):
                    raise ValueError("Viewer proxy frame seek failed")
                ok, frame = decoder.read()
                decoded_index = int(round(decoder.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            finally:
                decoder.release()
            expected_size = display_geometry.get("proxy_frame_size")
            if (
                not ok
                or decoded_index != frame_index
                or frame is None
                or frame.ndim != 3
                or frame.shape[2] != 3
                or not isinstance(expected_size, list)
                or expected_size != [frame.shape[1], frame.shape[0]]
            ):
                raise ValueError("Viewer proxy returned the wrong review frame")
            encoded_ok, encoded = cv2.imencode(
                ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )
            if not encoded_ok or encoded.nbytes <= 0 or encoded.nbytes > 64 * 1024 * 1024:
                raise ValueError("Review frame PNG is outside the accepted bounds")
            return Response(
                content=encoded.tobytes(),
                media_type="image/png",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "X-AutoAnim-Frame-Index": str(frame_index),
                    "X-AutoAnim-Proxy-SHA256": display_entry["sha256"],
                },
            )
        except FileNotFoundError:
            return _error_response(
                AutoAnimError(
                    "ARTIFACT_NOT_FOUND",
                    "Exact review frame source is unavailable",
                )
            )
        except ValueError as exc:
            return _error_response(
                AutoAnimError(
                    "INTEGRITY_FAILED",
                    f"Exact review frame did not verify: {exc}",
                )
            )
        except AutoAnimError as exc:
            return _error_response(exc)
        finally:
            review_decode_lock.release()

    @app.get("/api/jobs/{job_id}/viewer", response_class=HTMLResponse)
    def viewer(job_id: str):
        try:
            manifest = service.store.require_sealed(job_id)
            artifacts = manifest.get("artifacts", {})
            glb = artifacts.get("textured_glb") or artifacts.get("glb")
            if not isinstance(glb, dict) or not isinstance(glb.get("name"), str):
                raise FileNotFoundError(job_id)
            name = glb["name"]
            # Resolve through the same manifest allowlist before producing a URL.
            service.store.artifact(job_id, name)
            media_url = None
            media_type = None
            performance_evidence_url = None
            observation_v3_url = None
            observation_review_status = None
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
            if manifest.get("kind") == "video_performance":
                observation_review_status = (
                    "unavailable: complete sealed Observation-v3 evidence is missing"
                )
                evidence = artifacts.get("performance_evidence")
                if isinstance(evidence, dict) and isinstance(evidence.get("name"), str):
                    evidence_name = evidence["name"]
                    service.store.artifact(job_id, evidence_name)
                    performance_evidence_url = (
                        f"/api/jobs/{job_id}/files/{evidence_name}"
                    )
                observation_entries = tuple(
                    artifacts.get(name)
                    for name in (
                        *_observation_v3_chain_names(artifacts),
                        "viewer_media",
                    )
                )
                capture_summary = manifest.get("capture", {})
                entries_complete = all(
                    isinstance(entry, dict) and isinstance(entry.get("name"), str)
                    for entry in observation_entries
                )
                frame_count = (
                    capture_summary.get("frames")
                    if isinstance(capture_summary, dict)
                    else None
                )
                geometry_compatible = _review_display_geometry_compatible(
                    viewer_contract, capture_summary
                )
                if entries_complete and isinstance(frame_count, int) and (
                    frame_count > MAX_OBSERVATION_V3_VIEW_FRAMES
                ):
                    observation_review_status = (
                        "unavailable: take exceeds the "
                        f"{MAX_OBSERVATION_V3_VIEW_FRAMES}-frame interactive limit"
                    )
                elif entries_complete and not geometry_compatible:
                    observation_review_status = (
                        "unavailable: display proxy is not identity-mapped "
                        "square-pixel video"
                    )
                elif (
                    entries_complete
                    and isinstance(frame_count, int)
                    and 0 < frame_count <= MAX_OBSERVATION_V3_VIEW_FRAMES
                    and geometry_compatible
                ):
                    for entry in observation_entries:
                        service.store.artifact(job_id, entry["name"])
                    observation_v3_url = (
                        f"/api/jobs/{job_id}/observation-v3-view"
                    )
                    observation_review_status = "available"
            model_document = manifest.get("model")
            character_document = (
                model_document.get("character")
                if isinstance(model_document, dict)
                else None
            )
            runtime_hashes = (
                character_document.get("runtime_material_sha256s")
                if isinstance(character_document, dict)
                else None
            )
            viewer_metadata = (
                {
                    "character_revision": character_document.get("revision_id"),
                    "runtime_maps": (
                        sorted(runtime_hashes)
                        if isinstance(runtime_hashes, dict)
                        else []
                    ),
                }
                if isinstance(character_document, dict)
                else {}
            )
            if observation_review_status is not None:
                viewer_metadata["observation_review"] = observation_review_status
            if manifest.get("kind") in {"audio_animation", "video_performance"}:
                readiness = service.production_readiness(job_id)
                viewer_metadata.update(
                    {
                        "production_status": readiness["status"],
                        "release_gates": (
                            f"{readiness['passed_required_gate_count']}/"
                            f"{readiness['required_gate_count']}"
                        ),
                        "release_blockers": readiness["failures"],
                        "production_validated": readiness["publishable"],
                    }
                )
                repair = (
                    manifest.get("retargeting", {}).get("audio_visual_repair", {})
                    if manifest.get("kind") == "video_performance"
                    and isinstance(manifest.get("retargeting"), dict)
                    else {}
                )
                if isinstance(repair, dict) and repair.get("status") not in {
                    None,
                    "disabled",
                }:
                    metrics = repair.get("metrics", {})
                    viewer_metadata.update(
                        {
                            "audio_visual_repair": repair.get("status"),
                            "repair_quality": "candidate_unqualified",
                            "lower_face_repaired_frames": (
                                metrics.get("lowerFaceRepairedFrames")
                                if isinstance(metrics, dict)
                                else None
                            ),
                            "audio_tongue_frames": (
                                metrics.get("dedicatedTongueDrivenFrames")
                                if isinstance(metrics, dict)
                                else None
                            ),
                            "audio_visual_contact_conflicts": (
                                metrics.get("audioVisualContactConflictFrames")
                                if isinstance(metrics, dict)
                                else None
                            ),
                        }
                    )
            elif viewer_metadata:
                viewer_metadata["production_validated"] = False
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
                    performance_evidence_url=performance_evidence_url,
                    observation_v3_url=observation_v3_url,
                    metadata=viewer_metadata or None,
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
        except AutoAnimError as exc:
            return _error_response(exc)

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
    pre{overflow:auto;max-height:260px;background:#090b0d;padding:12px;border-radius:10px;font-size:11px}.note{margin-top:22px;color:var(--muted);border-left:3px solid var(--accent);padding-left:12px}.quality{margin:0 0 12px;padding:12px;border:1px solid var(--line);border-radius:10px;background:#0d1114}.quality strong{color:var(--accent)}.quality.blocked{border-color:#6a3d3d}.quality.blocked strong{color:var(--danger)}.quality small{display:block;color:var(--muted);margin-top:5px}.timeline-wrap{margin-top:12px;padding:10px;background:#090b0d;border:1px solid var(--line);border-radius:10px}.timeline-wrap canvas{display:block;width:100%;height:112px}.timeline-readout{color:var(--muted);font-size:11px;margin-top:5px}.recent{margin-top:28px}.recent-head{display:flex;align-items:center;justify-content:space-between;gap:16px}.recent-head button,.inline-action{width:auto;margin:0;padding:8px 12px}.recent-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}.recent-job{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:14px;border:1px solid var(--line);border-radius:12px;background:#0d1114}.recent-job small{display:block;color:var(--muted)}.recent-job a{color:var(--accent);white-space:nowrap}.job-actions{display:flex;gap:8px;align-items:center}.empty{color:var(--muted)}.library{margin-bottom:28px}.library-layout{display:grid;grid-template-columns:minmax(280px,.8fr) minmax(0,1.2fr);gap:18px}.character-list{display:grid;gap:10px}.character-row{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:12px;border:1px solid var(--line);border-radius:10px;background:#0d1114}.character-row small{display:block;color:var(--muted)}.character-row a{color:var(--accent)}
    .workspace{margin-bottom:28px;border-color:#65772d;background:linear-gradient(125deg,#172014,#11171b)}.workspace-grid{display:grid;grid-template-columns:minmax(220px,1fr) minmax(170px,.55fr) minmax(220px,1fr);gap:18px;align-items:end}.workspace label{margin-top:0}.workspace-state{padding:11px 12px;border:1px solid var(--line);border-radius:10px;background:#0d1114;color:var(--muted);min-height:48px}.workspace-state strong{display:block;color:var(--text);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.workspace-state span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    @media(max-width:760px){header{display:block}header p{margin-top:20px}.grid,.recent-list,.library-layout,.workspace-grid{grid-template-columns:1fr}}
  </style>
</head>
<body><main>
  <header><h1>Face motion,<br><span>made inspectable.</span></h1><p>Local GNM Head 3.0 workflows. Inputs stay on this machine; every result exposes controls, confidence, caveats, and downloadable artifacts. Runtime readiness is not production validation.</p></header>
  <section class="card workspace" aria-labelledby="workspace-title"><h2 id="workspace-title">Production context</h2><p>Choose the character and rights context once. Reconstruction, performance capture, acting direction, and review then share one active workspace.</p><div class="workspace-grid">
    <div><label for="workspace-character">Active character</label><select id="workspace-character"><option value="">Default neutral GNM</option></select></div>
    <div><label for="workspace-scope">Intended use</label><select id="workspace-scope"><option value="production">Production</option><option value="commercial">Commercial</option><option value="personal">Personal</option><option value="research">Research</option></select></div>
    <div class="workspace-state"><strong>Active performance</strong><span id="workspace-performance">None yet — run audio or video</span></div>
  </div></section>
  <section class="library" aria-labelledby="library-title"><div class="recent-head"><h2 id="library-title">Character library</h2><button id="refresh-characters" type="button">Refresh</button></div><div class="library-layout">
    <form class="card" id="character-form"><h2>Save reconstruction</h2><p>Promote a successful image or multiview job into an immutable character revision. Source photos remain in the job ledger.</p>
      <label for="character-job">Source job ID</label><input id="character-job" name="job_id" required placeholder="01…">
      <label for="character-name">Character name</label><input id="character-name" name="name" required maxlength="120">
      <label for="consent-subject">Performer / subject</label><input id="consent-subject" name="consent_subject" required maxlength="160">
      <label for="consent-attester">Rights attested by</label><input id="consent-attester" name="consent_attester" required maxlength="160">
      <label for="consent-scope">Authorized use</label><select id="consent-scope" name="consent_scope"><option value="production">Production</option><option value="commercial">Commercial</option><option value="personal">Personal</option><option value="research">Research</option></select>
      <label for="consent-evidence-ref">Release / evidence reference</label><input id="consent-evidence-ref" name="consent_evidence_ref" required maxlength="300" placeholder="Contract or release ID">
      <label for="consent-evidence">Release evidence file (hashed, not retained)</label><input id="consent-evidence" name="consent_evidence" type="file" required>
      <label for="consent-expiry">Expiry (optional, ISO-8601 with timezone)</label><input id="consent-expiry" name="consent_expires_at" placeholder="2030-12-31T23:59:59Z">
      <label><input style="width:auto" type="checkbox" name="consent_attested" value="true" required> I attest performer/rights-holder consent for this reusable biometric character.</label>
      <label for="consent-note">Consent note (optional)</label><textarea id="consent-note" name="consent_note" maxlength="500" placeholder="Release, project, or rights reference"></textarea>
      <button>Save character revision</button><div class="status"></div>
    </form>
    <div class="card"><h2>Reusable characters</h2><p>Pick one below in Audio or Video. Identity, the exact sealed UV layout, and any imported PBR runtime maps are applied to every interactive GLB; the audio MP4 remains an untextured diagnostic preview. Large 2K/4K/8K material packages are imported locally with <code>character material-template</code> then <code>character import-material</code>. The immutable revision retains source-precision maps while the viewer renders sealed base-color, normal, roughness, and specular derivatives. Browser upload is intentionally disabled until streamed archive limits are implemented.</p><div id="character-list" class="character-list" aria-live="polite"><p class="empty">Loading characters…</p></div></div>
  </div></section>
  <section class="grid">
    <form class="card" id="audio-form"><h2>Audio → animation</h2><p>Run the genuine temporal Audio2Face v3 weights locally at 60 Hz, use the established v2.3 path, or retain the transparent procedural fallback. Local v3 is an unqualified candidate pending SDK parity and artist review.</p>
      <label for="audio-file">Audio</label><input id="audio-file" name="file" type="file" accept="audio/*" required>
      <label for="audio-character">Character</label><select id="audio-character" class="character-select" name="character_id"><option value="">Default neutral GNM</option></select>
      <label for="audio-scope">Intended use</label><select id="audio-scope" class="usage-scope" name="usage_scope"><option value="production">Production</option><option value="commercial">Commercial</option><option value="personal">Personal</option><option value="research">Research</option></select>
      <label for="backend">Motion backend</label><select id="backend" name="backend"><option value="a2f-v3-local">Local v3 · temporal 60 Hz candidate</option><option value="auto">Auto · v2.3 learned preferred</option><option value="learned">v2.3 learned · require Audio2Face</option><option value="fallback">Procedural fallback</option></select>
      <label for="emotion">Emotion</label><select id="emotion" name="emotion"><option>auto</option><option>neutral</option><option>joy</option><option>sad</option><option>anger</option><option>fear</option><option>disgust</option><option>surprise</option><option>contempt</option></select>
      <label for="emotion-strength">Acting strength</label><input id="emotion-strength" name="emotion_strength" type="range" min="0" max="1" step="0.05" value="0.65">
      <label for="audio-mouth-aperture">Mouth opening correction · <output id="audio-mouth-aperture-value">1.00×</output></label><input id="audio-mouth-aperture" name="mouth_aperture_gain" type="range" min="1" max="1.25" step="0.01" value="1"><small>1.00 is byte-exact off. Higher values request a bounded, contact-preserving geometry edit; they do not scale all mouth controls.</small>
      <label for="audio-mouth-author">Edit author (required above 1.00)</label><input id="audio-mouth-author" name="mouth_aperture_author" maxlength="160" placeholder="Artist or operator">
      <label for="audio-mouth-reason">Edit reason (required above 1.00)</label><textarea id="audio-mouth-reason" name="mouth_aperture_reason" maxlength="500" placeholder="Example: source performance reads too closed on this character"></textarea>
      <label for="audio-phone-textgrid">Phone timing evidence (optional)</label><input id="audio-phone-textgrid" name="phone_textgrid" type="file" accept=".TextGrid,text/plain"><small>Praat/MFA long TextGrid. Imported evidence is scored and retained but does not alter motion in this phase.</small>
      <label><input style="width:auto" type="checkbox" name="phone_annotations_reviewed" value="true"> Phone and apex annotations were independently reviewed</label>
      <label for="audio-phone-reviewer">Phone annotation reviewer</label><input id="audio-phone-reviewer" name="phone_reviewer" maxlength="160" placeholder="Required only for reviewed evidence">
      <label for="dialog">Optional dialog</label><textarea id="dialog" name="dialog" placeholder="Helps Rhubarb and lexical emotion hints"></textarea><input id="audio-fps" name="fps" type="hidden" value="60"><input name="a2f_v3_local_seed" type="hidden" value="0">
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
      <label for="multiview-calibration">Calibrated camera bundle (optional)</label><input id="multiview-calibration" name="calibration" type="file" accept="application/json,.json"><small>I0 identity-capture evidence requires two independent sessions, each with at least 5 fit and 2 held-out cameras spanning 120°. A bundle here establishes declaration and coverage evidence only; it cannot validate identity, scan accuracy, PBR, texture, or production readiness. Filenames and upload order must match exactly.</small>
      <label for="texture-size">Texture atlas</label><select id="texture-size" name="texture_size"><option value="256">256 · test / fast</option><option value="512">512 · review</option><option value="1024">1024 · high detail</option><option value="128">128 · diagnostic</option></select>
      <input name="focal_scale" type="hidden" value="1.25">
      <button>Build textured face</button><div class="status"></div><div class="result"></div>
    </form>
    <form class="card" id="video-form"><h2>Video → performance</h2><p>Frame-accurate MediaPipe VIDEO tracking drives expression, head pose, translation, and gaze while keeping identity fixed. Begin with at least 0.2 seconds looking forward with a neutral face for tracker-bias calibration.</p>
      <label for="video-file">Face performance video</label><input id="video-file" name="file" type="file" accept="video/*" required>
      <label for="video-character">Target character</label><select id="video-character" class="character-select" name="character_id"><option value="">Default neutral GNM</option></select>
      <label for="video-scope">Intended use</label><select id="video-scope" class="usage-scope" name="usage_scope"><option value="production">Production</option><option value="commercial">Commercial</option><option value="personal">Personal</option><option value="research">Research</option></select>
      <label><input id="audio-visual-repair" style="width:auto" type="checkbox" name="audio_visual_repair" value="true"> Conservative learned audio repair · candidate</label><small>Off by default. Video keeps head, gaze, upper-face acting, reliable mouth shapes, and visible contacts. Learned audio uses a global tracker-quality heuristic to repair weak/missing lip evidence and supplies an unvalidated dedicated tongue track.</small>
      <label for="video-mouth-aperture">Mouth opening correction · <output id="video-mouth-aperture-value">1.00×</output></label><input id="video-mouth-aperture" name="mouth_aperture_gain" type="range" min="1" max="1.25" step="0.01" value="1"><small>Video remains authoritative. Confirmed closures are protected, and 1.00 preserves the retarget byte-for-byte.</small>
      <label for="video-mouth-author">Edit author (required above 1.00)</label><input id="video-mouth-author" name="mouth_aperture_author" maxlength="160" placeholder="Artist or operator">
      <label for="video-mouth-reason">Edit reason (required above 1.00)</label><textarea id="video-mouth-reason" name="mouth_aperture_reason" maxlength="500" placeholder="Example: increase open vowels without weakening visible bilabials"></textarea>
      <button>Capture performance</button><div class="status"></div><div class="result"></div>
    </form>
    <form class="card" id="direction-form"><h2>Performance → acting beats</h2><p>Claude or Codex reads a bounded transcript plus measured audio/video motion windows and proposes editable intent. It cannot write visemes, rig coefficients, files, or commands.</p>
      <label for="direction-job">Audio/video job ID</label><input id="direction-job" name="source_job_id" required placeholder="01…">
      <label for="direction-provider">Terminal provider</label><select id="direction-provider" name="provider"><option value="codex">Codex CLI</option><option value="claude">Claude Code CLI</option></select>
      <label for="direction-character">Character</label><select id="direction-character" class="character-select" name="character_id"><option value="">No saved character capability profile</option></select>
      <label for="direction-scope">Intended use</label><select id="direction-scope" class="usage-scope" name="usage_scope"><option value="production">Production</option><option value="commercial">Commercial</option><option value="personal">Personal</option><option value="research">Research</option></select>
      <label for="direction-instructions">Acting instructions</label><textarea id="direction-instructions" name="instructions" maxlength="4000" placeholder="Restrained, reassuring, with one small open-palm beat"></textarea>
      <label for="direction-transcript">Transcript</label><textarea id="direction-transcript" name="transcript" maxlength="80000" placeholder="Quoted dialog; treated as untrusted content"></textarea>
      <input name="timeout_seconds" type="hidden" value="180">
      <button>Propose acting beats</button><div class="status"></div><div class="result"></div>
    </form>
  </section>
  <section class="recent" aria-labelledby="recent-title"><div class="recent-head"><h2 id="recent-title">Recent local runs</h2><button id="refresh-jobs" type="button">Refresh</button></div><div id="recent-list" class="recent-list" aria-live="polite"><p class="empty">Loading recent jobs…</p></div></section>
  <p class="note" id="health">Checking local model and native-tool readiness…</p>
</main>
<script>
const health=document.querySelector('#health'),audioVisualRepair=document.querySelector('#audio-visual-repair'),localV3Option=document.querySelector('#backend option[value="a2f-v3-local"]');fetch('/api/health').then(r=>r.json()).then(x=>{const a2fReady=x.checks.a2f_runner?.ready&&x.checks.a2f_assets?.ready&&x.checks.a2f_provenance?.ready&&x.checks.rhubarb?.ready,localV3Ready=x.checks.a2f_v3_local?.ready;audioVisualRepair.disabled=!a2fReady;audioVisualRepair.title=a2fReady?'Unvalidated learned repair is available':'Learned Audio2Face and Rhubarb dependencies are unavailable';localV3Option.disabled=!localV3Ready;if(!localV3Ready&&audioBackend.value==='a2f-v3-local'){audioBackend.value='auto';syncAudioBackend()}health.textContent=`Health: ${x.status}. GNM ${x.checks.gnm.ready?'ready':'missing'}, local v3 ${localV3Ready?'ready':'missing'}, Audio2Face v2.3 ${a2fReady?'ready':'missing'}, MediaPipe ${x.checks.mediapipe_model.ready?'ready':'missing'}, Rhubarb ${x.checks.rhubarb.ready?'ready':'missing'}, offline viewer ${x.checks.viewer_bundle?.ready?'ready':'missing'}.`});
const audioBackend=document.querySelector('#backend'),audioFps=document.querySelector('#audio-fps');function syncAudioBackend(){audioFps.value=audioBackend.value==='a2f-v3-local'?'60':'30'}audioBackend.addEventListener('change',syncAudioBackend);syncAudioBackend();
function artifactUrl(job,name){return `/api/jobs/${job}/files/${encodeURIComponent(name)}`}
const characterList=document.querySelector('#character-list');
const workspaceCharacter=document.querySelector('#workspace-character');
const workspaceScope=document.querySelector('#workspace-scope');
const workspacePerformance=document.querySelector('#workspace-performance');
for(const prefix of ['audio','video']){const slider=document.querySelector(`#${prefix}-mouth-aperture`),value=document.querySelector(`#${prefix}-mouth-aperture-value`);const update=()=>value.textContent=`${Number(slider.value).toFixed(2)}×`;slider.addEventListener('input',update);update()}
function syncWorkspaceCharacter(value){
 workspaceCharacter.value=value;
 for(const select of document.querySelectorAll('.character-select'))if([...select.options].some(option=>option.value===value))select.value=value;
}
function syncWorkspaceScope(value){workspaceScope.value=value;for(const select of document.querySelectorAll('.usage-scope'))select.value=value}
function setActivePerformance(jobId,kind){document.querySelector('#direction-job').value=jobId;workspacePerformance.textContent=`${kind.replaceAll('_',' ')} · ${jobId}`}
workspaceCharacter.addEventListener('change',()=>syncWorkspaceCharacter(workspaceCharacter.value));
workspaceScope.addEventListener('change',()=>syncWorkspaceScope(workspaceScope.value));
for(const select of document.querySelectorAll('.character-select'))select.addEventListener('change',()=>syncWorkspaceCharacter(select.value));
for(const select of document.querySelectorAll('.usage-scope'))select.addEventListener('change',()=>syncWorkspaceScope(select.value));
async function refreshCharacters(){try{
 const response=await fetch('/api/characters?limit=100'),data=await response.json();
 const selected=workspaceCharacter.value;characterList.innerHTML='';
 for(const select of [workspaceCharacter,...document.querySelectorAll('.character-select')]){
  select.innerHTML='<option value="">Default neutral GNM</option>';
  for(const item of data.characters){if(item.consent_status!=='active'||!['active','not_applicable'].includes(item.material_rights_status))continue;const option=document.createElement('option');option.value=item.character_id;option.textContent=`${item.name} · ${item.appearance_status}`;select.append(option)}
 }
 syncWorkspaceCharacter([...workspaceCharacter.options].some(option=>option.value===selected)?selected:'');
 if(!data.characters.length){characterList.innerHTML='<p class="empty">No saved characters yet. Run Image or Multi-view, then promote its job ID.</p>';return}
 for(const item of data.characters){
  const row=document.createElement('article');row.className='character-row';const copy=document.createElement('div');const title=document.createElement('strong');title.textContent=item.name;const detail=document.createElement('small');detail.textContent=`${item.appearance_status} · ${item.consent_scope} consent ${item.consent_status} · material rights ${item.material_rights_status} · body ${item.body_status} · production ${item.production_validated?'approved':'not validated'}`;copy.append(title,detail);
  const actions=document.createElement('div');actions.className='job-actions';
  if(item.consent_status==='active'&&['active','not_applicable'].includes(item.material_rights_status)){const use=document.createElement('button');use.type='button';use.className='inline-action';use.textContent='Use';use.addEventListener('click',()=>syncWorkspaceCharacter(item.character_id));actions.append(use)}
  const link=document.createElement('a');link.href=`/api/characters/${item.character_id}/viewer?usage_scope=${encodeURIComponent(item.consent_scope)}`;link.target='_blank';link.textContent='Open 3D';actions.append(link);row.append(copy,actions);characterList.append(row)
 }
}catch(error){characterList.innerHTML='<p class="empty">Character library unavailable.</p>'}}
document.querySelector('#refresh-characters').addEventListener('click',refreshCharacters);refreshCharacters();
{const form=document.querySelector('#character-form'),status=form.querySelector('.status'),button=form.querySelector('button');form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;status.className='status';status.textContent='Saving immutable revision…';try{const response=await fetch('/api/characters/from-job',{method:'POST',body:new FormData(form)}),data=await response.json();if(!response.ok)throw new Error(`${data.code}: ${data.message}`);status.textContent=`Saved ${data.name} · ${data.character_id}`;await refreshCharacters();syncWorkspaceCharacter(data.character_id)}catch(error){status.className='status error';status.textContent=error.message}finally{button.disabled=false}})}
const recentList=document.querySelector('#recent-list');async function refreshJobs(){try{const response=await fetch('/api/jobs?limit=8'),data=await response.json();recentList.innerHTML='';if(!data.jobs.length){recentList.innerHTML='<p class="empty">No jobs yet.</p>';return}for(const job of data.jobs){
 const row=document.createElement('article');row.className='recent-job';const copy=document.createElement('div');const title=document.createElement('strong');title.textContent=job.kind.replaceAll('_',' ');const detail=document.createElement('small');detail.textContent=`${job.input.name} · ${job.status} · ${job.warning_count} warning${job.warning_count===1?'':'s'}`;copy.append(title,detail);row.append(copy);
 const actions=document.createElement('div');actions.className='job-actions';
 if(job.status==='succeeded'&&(job.kind==='audio_animation'||job.kind==='video_performance')){const use=document.createElement('button');use.type='button';use.className='inline-action';use.textContent='Direct';use.addEventListener('click',()=>setActivePerformance(job.job_id,job.kind));actions.append(use)}
 if(job.status==='succeeded'&&(job.kind==='image_fit'||job.kind==='multiview_reconstruction')){const use=document.createElement('button');use.type='button';use.className='inline-action';use.textContent='Promote';use.addEventListener('click',()=>{document.querySelector('#character-job').value=job.job_id;document.querySelector('#character-form').scrollIntoView({behavior:'smooth'})});actions.append(use)}
 if(job.viewable){const link=document.createElement('a');link.href=`/api/jobs/${job.job_id}/viewer`;link.target='_blank';link.textContent='Open 3D';actions.append(link)}
 if(actions.childElementCount)row.append(actions);recentList.append(row)
}}catch(error){recentList.innerHTML='<p class="empty">Recent jobs unavailable.</p>'}}
document.querySelector('#refresh-jobs').addEventListener('click',refreshJobs);refreshJobs();
async function appendProductionReadiness(data,result){
 if(data.kind!=='audio_animation'&&data.kind!=='video_performance')return;
 const response=await fetch(`/api/jobs/${data.job_id}/production-readiness`),report=await response.json();
 if(!response.ok)return;
 const q=document.createElement('div');q.className=`quality ${report.publishable?'':'blocked'}`;
 const title=document.createElement('strong');title.textContent=report.publishable?'Production release evidence complete':`Production blocked · ${report.failures.length} required gate${report.failures.length===1?'':'s'}`;q.append(title);
 const detail=document.createElement('small');const failed=report.failures.map(name=>name.replaceAll('_',' ')).join(', '),performanceEvidence=report.gates?.performance?.evidence||{},performanceReasons=[...(performanceEvidence.phone_articulation_failures||[]),performanceEvidence.phone_evidence_artifact_failure_reason].filter(Boolean).map(name=>name.replaceAll('_',' '));detail.textContent=`${report.passed_required_gate_count}/${report.required_gate_count} required gates pass${failed?` · missing: ${failed}`:''}.${performanceReasons.length?` Performance evidence: ${performanceReasons.join(', ')}.`:''} ${report.claim}`;q.append(detail);result.append(q)
}
for(const [formId,endpoint] of [['audio-form','/api/audio'],['image-form','/api/image'],['multiview-form','/api/multiview'],['video-form','/api/video'],['direction-form','/api/direction']]){
 const form=document.getElementById(formId),status=form.querySelector('.status'),result=form.querySelector('.result'),button=form.querySelector('button');
 form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;status.className='status';status.textContent='Processing locally…';result.style.display='none';
  try{const response=await fetch(endpoint,{method:'POST',body:new FormData(form)}),data=await response.json();if(!response.ok)throw new Error(`${data.code}: ${data.message}`);status.textContent=`Succeeded · ${data.job_id}`;result.innerHTML='';
	   if(data.kind==='audio_animation'){const q=document.createElement('div');q.className='quality';const backend=data.analysis.motion_backend||'',learned=backend==='learned_a2f',sequence=backend==='unverified_external_sequence_controls_candidate',localV3=backend==='local_a2f_v3_candidate_unqualified';const title=document.createElement('strong');title.textContent=localV3?'Genuine local v3 sequence · unqualified candidate':sequence?'Unverified external controls · claimed v3 profile':learned?'Learned face + tongue controls · geometry calibrated':'Procedural fallback · not production';q.append(title);const articulation=data.phone_articulation||null,families=articulation?.families||{},f1=value=>value?.classification?.f1==null?'n/a':value.classification.f1.toFixed(3),proxyFailureValues=articulation?.phone_span_proxy_gate?.failures||[],proxyFailures=proxyFailureValues.slice(0,3).map(name=>name.replaceAll('_',' ')),proxyFailureRemainder=proxyFailureValues.length-proxyFailures.length,articulationText=articulation?` Phone-span proxy diagnostics: P/B/M F1 ${f1(families.bilabial)}, F/V proximity F1 ${f1(families.labiodental)}, tongue/teeth proximity F1 ${f1(families.tongue_upper_teeth)}, rounding-width F1 ${f1(families.rounded)} · proxy gate ${articulation.phone_span_proxy_gate?.passed?'pass':'blocked'}${proxyFailures.length?` (${proxyFailures.join(', ')}${proxyFailureRemainder?`, +${proxyFailureRemainder} more`:''})`:''} · production blocked by schema.`:'';const detail=document.createElement('small');detail.textContent=`Stationary speech transitions ${(100*data.metrics.lower_face_stationary_fraction).toFixed(1)}% · mouth speed p95 ${data.metrics.mouth_speed_p95_interocular_per_second.toFixed(3)} IOD/s · absolute step p95 ${data.metrics.mouth_step_p95_interocular.toFixed(3)} IOD · limited frames ${data.metrics.mouth_speed_limited_frames}.${articulationText} Tongue collision and perceptual speech quality still require review. ${data.warnings.join(' ')}`;q.append(detail);result.append(q);setActivePerformance(data.job_id,data.kind)}
   if(data.kind==='image_fit'){const q=document.createElement('div');q.className='quality';const title=document.createElement('strong');title.textContent=`Visible-geometry fit · ${data.fit.confidence} confidence`;q.append(title);const detail=document.createElement('small');detail.textContent=`Landmark NME ${data.fit.nme.toFixed(4)} · stability ${data.fit.stability_rms.toFixed(4)} · ${(100*data.fit.coefficient_bound_fraction).toFixed(1)}% coefficients at bounds. This neutral fit does not reconstruct hidden geometry or metric depth. ${data.warnings.join(' ')}`;q.append(detail);result.append(q);document.querySelector('#character-job').value=data.job_id}
		   if(data.kind==='video_performance'){const q=document.createElement('div');q.className='quality';const repair=data.retargeting.audio_visual_repair||{status:'disabled'},repaired=repair.status!=='disabled';const title=document.createElement('strong');title.textContent=`Video performance · ${data.retargeting.geometry_calibrated?'geometry calibrated':'semantic fallback'}${repaired?' · audiovisual repair candidate':''}`;q.append(title);const contact=data.metrics.final_contact_geometry_attained_fraction;const contactText=contact===null?'no scored closure':`${(100*contact).toFixed(1)}% contact attained`;const aperture=data.metrics.final_lip_aperture_open_p95_ratio;const apertureText=aperture===null?'aperture n/a':`aperture amplitude ${(100*aperture).toFixed(1)}%`;const authority=repaired?`Visual head/gaze/upper face/reliable lips locked · audio changed ${repair.metrics.lowerFaceRepairedFrames} globally low-quality/missing-observation lower-face frames, drove ${repair.metrics.dedicatedTongueDrivenFrames} tongue frames, and diagnosed ${repair.metrics.audioVisualContactConflictFrames} trusted-frame contact disagreements. Candidate is not production validated.`:'Visual-only motion; audio is playback only and tongue is not inferred.';const detail=document.createElement('small');detail.textContent=`Face presence ${(100*data.metrics.face_presence_fraction).toFixed(1)}% · ${contactText} · ${apertureText} · expression timing ${data.metrics.final_expression_motion_correlation===null?'n/a':data.metrics.final_expression_motion_correlation.toFixed(3)} · baseline loss ${(100*data.metrics.negative_baseline_residual_clipped_fraction).toFixed(1)}% · proxy timing error ${data.metrics.proxy_pts_max_error_ms.toFixed(2)} ms. ${authority} ${data.warnings.join(' ')}`;q.append(detail);result.append(q);setActivePerformance(data.job_id,data.kind)}
   if(data.kind==='multiview_reconstruction'){const q=document.createElement('div');q.className='quality';const title=document.createElement('strong');title.textContent='Shared identity · provenance-aware texture';q.append(title);const detail=document.createElement('small');const holdout=data.capture.held_out?.evaluated?` · held-out NME ${data.capture.held_out.aggregate_nme.toFixed(4)} (${data.capture.held_out.passed?'pass':'fail'})`:'';detail.textContent=`Fit NME ${data.fit.nme.toFixed(4)}${holdout} · direct texture ${(100*data.texture.observed_fraction).toFixed(1)}% · ${data.capture.accepted_view_indices.length}/${data.capture.view_count} views accepted. ${data.warnings.join(' ')}`;q.append(detail);result.append(q);document.querySelector('#character-job').value=data.job_id}
   if(data.kind==='acting_direction'){const q=document.createElement('div');q.className='quality';const title=document.createElement('strong');title.textContent=`Acting proposal + unapproved body preview · ${data.direction.beat_count} beat${data.direction.beat_count===1?'':'s'}`;q.append(title);const detail=document.createElement('small');detail.textContent=`${data.direction.summary} · evidence ${data.source.motion_evidence}. Lipsync overrides are disabled; approve/edit and recompile before publish. ${data.warnings.join(' ')}`;q.append(detail);result.append(q)}
   await appendProductionReadiness(data,result);
   let video=null;const media=data.artifacts.preview||data.artifacts.viewer_media||data.artifacts.overlay||data.artifacts.mesh_preview;if(media){const url=artifactUrl(data.job_id,media.name);if(media.media_type==='video/mp4'){video=document.createElement('video');video.controls=true;video.playsInline=true;video.src=url;result.append(video)}else{const image=document.createElement('img');image.src=url;image.alt='Result preview';result.append(image)}}
   if(video&&data.artifacts.timeline){const wrap=document.createElement('div');wrap.className='timeline-wrap';const canvas=document.createElement('canvas');canvas.width=800;canvas.height=112;const readout=document.createElement('div');readout.className='timeline-readout';wrap.append(canvas,readout);result.append(wrap);fetch(artifactUrl(data.job_id,data.artifacts.timeline.name)).then(r=>r.json()).then(t=>{const ctx=canvas.getContext('2d'),n=t.timestamps.length,ap=t.mouth_aperture||[],en=t.energy||[],amin=Math.min(...ap),ar=Math.max(...ap)-amin||1;function curve(values,color,norm){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>{const x=i/(n-1)*canvas.width,y=canvas.height-8-norm(v)*(canvas.height-16);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#090b0d';ctx.fillRect(0,0,canvas.width,canvas.height);curve(en,'#6fb9ff',v=>v);curve(ap,'#d8ff63',v=>(v-amin)/ar);const duration=t.timestamps[n-1]||video.duration||1,x=Math.min(1,video.currentTime/duration)*canvas.width;ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();const cue=(data.analysis.cues||[]).find(c=>video.currentTime>=c.start&&video.currentTime<c.end);readout.textContent=`${video.currentTime.toFixed(2)} s · cue ${cue?cue.value:'X'} · green aperture · blue energy`}video.addEventListener('timeupdate',draw);video.addEventListener('seeked',draw);draw()}).catch(()=>{readout.textContent='Timeline unavailable'})}
   const links=document.createElement('div');links.className='links';if(data.artifacts.glb||data.artifacts.textured_glb){const view=document.createElement('a');view.href=`/api/jobs/${data.job_id}/viewer`;view.textContent='Open interactive 3D';view.target='_blank';links.append(view)}for(const [key,item] of Object.entries(data.artifacts)){const a=document.createElement('a');a.href=artifactUrl(data.job_id,item.name);a.textContent=key;a.download=item.name;links.append(a)}result.append(links);const pre=document.createElement('pre');pre.textContent=JSON.stringify(data,null,2);result.append(pre);result.style.display='block';refreshJobs();
  }catch(error){status.className='status error';status.textContent=error.message}finally{button.disabled=false}}
 );
}
</script></body></html>"""
