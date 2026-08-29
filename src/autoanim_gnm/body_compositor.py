"""Ownership-safe, hash-bound composition of GNM face and SOMA body tracks.

This module creates the canonical source bundle for a unified preview.  It
does not pretend that independently exported meshes are a production
character: publish readiness remains blocked until the exact character/body
pair has a calibrated head socket and a validated neck seam.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from .body import BodyTrack, attachment_contract, validate_body_track
from .body_provider import sha256_file
from .serialization import write_json, write_npz
from .soma_motion import SomaMotion


UNIFIED_PERFORMANCE_SCHEMA_VERSION = "autoanim.unified-performance/1.0"
UNIFIED_ARRAYS_SCHEMA_VERSION = "autoanim.unified-performance-arrays/1.0"
MAX_FACE_PERFORMANCE_BYTES = 256 * 1024 * 1024
_FACE_ARRAYS = {
    "identity",
    "expression",
    "rotations",
    "translation",
    "timestamps_seconds",
    "source_pts",
}


class BodyCompositionError(ValueError):
    """Face/body inputs cannot be safely composed under the ownership contract."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    dtype = array.dtype.str.encode("ascii")
    shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    digest.update(len(dtype).to_bytes(4, "little"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(4, "little"))
    digest.update(shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _artifact(path: str | Path, *, role: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise BodyCompositionError(f"{role} artifact is missing or is a symbolic link")
    return {
        "role": role,
        "name": candidate.name,
        "bytes": candidate.stat().st_size,
        "sha256": sha256_file(candidate),
    }


def _load_face_arrays(path: str | Path) -> dict[str, np.ndarray]:
    candidate = Path(path)
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.stat().st_size <= 0
        or candidate.stat().st_size > MAX_FACE_PERFORMANCE_BYTES
    ):
        raise BodyCompositionError("Face performance NPZ is unavailable or too large")
    try:
        with zipfile.ZipFile(candidate) as archive:
            infos = archive.infolist()
            if (
                len(infos) != len({info.filename for info in infos})
                or any(info.flag_bits & 0x1 for info in infos)
                or sum(info.file_size for info in infos) > MAX_FACE_PERFORMANCE_BYTES
            ):
                raise BodyCompositionError("Face performance NPZ layout is unsafe")
        with np.load(candidate, allow_pickle=False) as archive:
            if not _FACE_ARRAYS.issubset(archive.files):
                raise BodyCompositionError("Face performance is missing canonical arrays")
            values = {name: np.array(archive[name], copy=True) for name in _FACE_ARRAYS}
    except BodyCompositionError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise BodyCompositionError("Face performance NPZ cannot be verified") from exc
    identity = values["identity"]
    expression = values["expression"]
    rotations = values["rotations"]
    translation = values["translation"]
    timestamps = values["timestamps_seconds"]
    source_pts = values["source_pts"]
    frame_count = len(timestamps)
    if (
        identity.shape != (253,)
        or expression.shape != (frame_count, 383)
        or rotations.shape != (frame_count, 4, 3)
        or translation.shape != (frame_count, 3)
        or source_pts.shape != (frame_count,)
        or source_pts.dtype.kind not in "iu"
        or timestamps.dtype.kind != "f"
        or any(value.dtype.kind not in "fiub" for value in values.values())
        or frame_count < 1
        or any(not np.isfinite(value).all() for value in values.values())
        or (frame_count > 1 and np.any(np.diff(timestamps) <= 0.0))
        or (frame_count > 1 and np.any(np.diff(source_pts) <= 0))
    ):
        raise BodyCompositionError("Face performance arrays have invalid shape or values")
    return values


def _validate_derivation(
    value: Mapping[str, Any],
    *,
    canonical_source_sha256: str,
    face_source_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "operation",
        "parent_source_sha256",
        "derived_source_sha256",
        "crop_ltrb_pixels",
        "output_size_pixels",
        "timing_policy",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise BodyCompositionError("Face-source derivation fields are missing or unknown")
    if (
        value["schema_version"] != "autoanim.source-derivation/1.0"
        or value["operation"] != "spatial_crop_and_scale"
        or value["parent_source_sha256"] != canonical_source_sha256
        or value["derived_source_sha256"] != face_source_sha256
        or value["timing_policy"] != "exact_source_pts_preserved"
    ):
        raise BodyCompositionError("Face-source derivation does not bind the sealed source")
    crop = value["crop_ltrb_pixels"]
    size = value["output_size_pixels"]
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(type(item) is not int or item < 0 for item in crop)
        or crop[2] <= crop[0]
        or crop[3] <= crop[1]
        or not isinstance(size, list)
        or len(size) != 2
        or any(type(item) is not int or item <= 0 for item in size)
    ):
        raise BodyCompositionError("Face-source crop or output size is invalid")
    return dict(value)


def compose_unified_performance(
    output_manifest_path: str | Path,
    *,
    canonical_source_path: str | Path,
    face_source_path: str | Path,
    face_source_derivation: Mapping[str, Any],
    face_performance_path: str | Path,
    face_glb_path: str | Path,
    soma_track: SomaMotion,
    body_track: BodyTrack,
    body_glb_path: str | Path,
    body_manifest_path: str | Path,
    body_asset_path: str | Path,
    arrays_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compose one exact timeline and ownership report from verified inputs."""

    validate_body_track(body_track)
    canonical_source = _artifact(canonical_source_path, role="canonical_source_video")
    if canonical_source["sha256"] != soma_track.input_sha256:
        raise BodyCompositionError("SOMA motion does not belong to the canonical source")
    face_source = _artifact(face_source_path, role="derived_face_source_video")
    derivation = _validate_derivation(
        face_source_derivation,
        canonical_source_sha256=canonical_source["sha256"],
        face_source_sha256=face_source["sha256"],
    )
    face = _load_face_arrays(face_performance_path)
    if not np.array_equal(face["source_pts"], soma_track.source_pts):
        raise BodyCompositionError("Face and body source PTS do not match exactly")
    if not np.array_equal(body_track.ticks, soma_track.ticks):
        raise BodyCompositionError("Projected body ticks do not match SOMA source ticks")
    expected_seconds = soma_track.ticks.astype(np.float64) / body_track.ticks_per_second
    if not np.array_equal(face["timestamps_seconds"], expected_seconds):
        raise BodyCompositionError("Face and body timestamps do not share the exact timeline")
    if len(face["timestamps_seconds"]) != len(body_track.ticks):
        raise BodyCompositionError("Face and body frame counts do not match")

    # GNM joints 0/1 are its base neck/head chain; body owns those channels in
    # this first composition version. GNM eye joints 2/3 and all facial/oral
    # expression coefficients remain byte-identical.
    owned_face_rotations = np.array(face["rotations"], dtype=np.float32, copy=True)
    suppressed_neck_head = int(np.count_nonzero(owned_face_rotations[:, :2]))
    owned_face_rotations[:, :2] = 0.0
    suppressed_translation = int(np.count_nonzero(face["translation"]))
    owned_face_translation = np.zeros_like(face["translation"], dtype=np.float32)
    body_eye_delta = np.asarray(body_track.gnm_eye_rotations_xyzw, dtype=np.float64)
    identity_eye = np.zeros_like(body_eye_delta)
    identity_eye[..., 3] = 1.0
    conflicts: list[dict[str, Any]] = []
    if not np.allclose(body_eye_delta, identity_eye, atol=1.0e-7):
        conflicts.append(
            {
                "channel": "eyes",
                "owners": ["GNM", "BodyTrack.gnm_eye_rotations_xyzw"],
                "resolution": "blocked",
            }
        )
    if conflicts:
        raise BodyCompositionError("Composition contains an unresolved ownership conflict")

    expression_sha256 = _array_sha256(face["expression"])
    eye_sha256 = _array_sha256(face["rotations"][:, 2:])
    output_arrays = (
        Path(arrays_path)
        if arrays_path is not None
        else Path(output_manifest_path).with_suffix(".npz")
    )
    write_npz(
        output_arrays,
        schema_version=np.asarray(UNIFIED_ARRAYS_SCHEMA_VERSION),
        ticks=body_track.ticks,
        source_pts=soma_track.source_pts,
        timestamps_seconds=face["timestamps_seconds"],
        gnm_identity=face["identity"],
        gnm_expression=face["expression"],
        gnm_owned_rotations=owned_face_rotations,
        gnm_owned_translation=owned_face_translation,
        body_root_translation_m=body_track.root_translation_m,
        body_local_rotations_xyzw=body_track.local_rotations_xyzw,
        body_foot_contacts=body_track.foot_contacts,
    )

    contract = attachment_contract()
    contract_sha256 = sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifacts = {
        item["role"]: item
        for item in (
            canonical_source,
            face_source,
            _artifact(face_performance_path, role="face_performance"),
            _artifact(face_glb_path, role="face_review_glb"),
            _artifact(body_glb_path, role="body_review_glb"),
            _artifact(body_manifest_path, role="body_manifest"),
            _artifact(body_asset_path, role="body_asset"),
            _artifact(output_arrays, role="unified_arrays"),
        )
    }
    body_manifest = json.loads(Path(body_manifest_path).read_text(encoding="utf-8"))
    body_joint_count = len(body_manifest.get("skeleton", {}).get("joint_names", ()))
    attachment_calibrated = bool(
        body_manifest.get("gnm_head_socket", {}).get("attachment_calibrated", False)
    )
    blockers = []
    if not attachment_calibrated:
        blockers.append("GNM_HEAD_SOCKET_UNCALIBRATED")
    blockers.extend(
        (
            "PROVIDER_HEAD_NOT_REMOVED",
            "NECK_SEAM_NOT_VALIDATED",
            (
                "SOMA_25_PROJECTION_PREVIEW_ONLY"
                if body_joint_count == 25
                else "SOMA_DETAILED_HANDS_PROVIDER_PREVIEW_ONLY"
            ),
            "GNM_FACE_TRACK_NOT_PRODUCTION_CALIBRATED",
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": UNIFIED_PERFORMANCE_SCHEMA_VERSION,
        "status": "diagnostic_unified_preview",
        "production_validated": False,
        "publish_ready": False,
        "source": {
            "canonical": canonical_source,
            "face_derivation": derivation,
            "face_derived": face_source,
        },
        "timeline": {
            "ticks_per_second": body_track.ticks_per_second,
            "duration_ticks": body_track.duration_ticks,
            "frame_count": len(body_track.ticks),
            "source_time_base": {
                "numerator": soma_track.source_time_base_numerator,
                "denominator": soma_track.source_time_base_denominator,
            },
            "source_pts_sha256": _array_sha256(soma_track.source_pts),
            "ticks_sha256": _array_sha256(body_track.ticks),
            "face_body_exact_pts_match": True,
            "face_body_exact_ticks_match": True,
        },
        "ownership": {
            "contract_schema_version": contract["schema_version"],
            "contract_sha256": contract_sha256,
            "body": ["root_translation", "Root_to_Head_base_rotation"],
            "gnm": [
                "identity",
                "facial_expression",
                "lipsync",
                "eyes",
                "teeth",
                "tongue",
                "oral_contact",
            ],
            "gnm_neck_head_samples_suppressed": suppressed_neck_head,
            "gnm_translation_samples_suppressed": suppressed_translation,
            "gnm_expression_input_sha256": expression_sha256,
            "gnm_expression_output_sha256": _array_sha256(face["expression"]),
            "gnm_eye_input_sha256": eye_sha256,
            "gnm_eye_output_sha256": _array_sha256(owned_face_rotations[:, 2:]),
            "authoritative_face_arrays_byte_identical": True,
            "base_head_applied_once": True,
            "conflicts": [],
        },
        "attachment": {
            "calibrated": attachment_calibrated,
            "head_included_in_body_review_glb": True,
            "gnm_head_included_in_body_review_glb": False,
            "single_connected_character_mesh": False,
        },
        "artifacts": artifacts,
        "publish_blockers": blockers,
    }
    payload_sha256 = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest["payload_sha256"] = payload_sha256
    write_json(output_manifest_path, manifest)
    return manifest


def write_unified_preview_html(
    path: str | Path,
    *,
    source_video_url: str,
    body_glb_url: str,
    face_glb_url: str,
    manifest_url: str,
    vendor_base_url: str,
    body_camera_position: tuple[float, float, float] = (0.0, 0.15, 4.2),
    body_camera_target: tuple[float, float, float] = (0.0, 0.15, 0.0),
) -> Path:
    """Write a synchronized three-pane diagnostic review page.

    The page deliberately presents the body and face as separate panes because
    an uncalibrated head socket must never be disguised as a joined character.
    All three panes use the source video's presented-time clock.
    """

    values = {
        "source": source_video_url,
        "body": body_glb_url,
        "face": face_glb_url,
        "manifest": manifest_url,
        "three": f"{vendor_base_url.rstrip('/')}/three.module.js",
        "addons": f"{vendor_base_url.rstrip('/')}/addons/",
        "body_camera_position": list(body_camera_position),
        "body_camera_target": list(body_camera_target),
    }
    encoded = {
        key: json.dumps(value, ensure_ascii=True).replace("</", "<\\/")
        for key, value in values.items()
    }
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>AutoAnim unified performance diagnostic</title>
<style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
*{{box-sizing:border-box}}body{{margin:0;background:#0d0f12;color:#f1f3f5}}
header{{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;padding:20px 24px;border-bottom:1px solid #2a2e34}}
h1{{font-size:18px;margin:0 0 6px}}p{{margin:0;color:#aeb5bf;font-size:13px;line-height:1.45}}
.badge{{padding:7px 10px;border:1px solid #8b5e21;background:#2b2114;color:#ffd69a;border-radius:6px;font-size:12px;white-space:nowrap}}
main{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:#2a2e34;height:calc(100vh - 147px)}}
.pane{{position:relative;min-width:0;background:#111419;overflow:hidden}}
.pane h2{{position:absolute;z-index:2;top:12px;left:14px;margin:0;padding:5px 8px;border-radius:5px;background:#090b0dcc;font-size:12px;font-weight:600}}
video,canvas{{width:100%;height:100%;display:block;object-fit:contain}}
footer{{height:72px;padding:12px 24px;border-top:1px solid #2a2e34;display:grid;grid-template-columns:auto 1fr auto;gap:16px;align-items:center}}
button{{border:1px solid #444b55;border-radius:6px;background:#23272e;color:#fff;padding:8px 14px;cursor:pointer}}
input{{width:100%}}#time{{font-variant-numeric:tabular-nums;color:#bbc2cc;font-size:12px}}
@media(max-width:900px){{main{{grid-template-columns:1fr;height:auto}}.pane{{height:48vh}}footer{{position:sticky;bottom:0;background:#0d0f12}}}}
</style>
<script type="importmap">{{"imports":{{"three":{encoded["three"]},"three/addons/":{encoded["addons"]}}}}}</script>
</head>
<body>
<header><div><h1>Unified performance · exact-time diagnostic</h1>
<p>One sealed source clock drives the observed SOMA body and ownership-safe GNM face.<br>
The panes remain separate until head-socket calibration, provider-head removal, and neck-seam validation pass.</p></div>
<div class="badge">NOT PUBLISHABLE · UNCALIBRATED ATTACHMENT</div></header>
<main>
<section class="pane"><h2>Sealed source</h2><video id="source" src={encoded["source"]} playsinline preload="auto"></video></section>
<section class="pane"><h2>MPFB skin · SOMA body owner</h2><canvas id="body"></canvas></section>
<section class="pane"><h2>GNM · expression / eyes / oral owner</h2><canvas id="face"></canvas></section>
</main>
<footer><button id="play" type="button">Play</button><input id="scrub" type="range" min="0" max="1" step="0.0001" value="0"><span id="time">0.000 / 0.000 s</span></footer>
<script type="module">
import * as THREE from 'three';
import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';
const source=document.querySelector('#source'),play=document.querySelector('#play'),scrub=document.querySelector('#scrub'),time=document.querySelector('#time');
const animations=[];
function stage(canvas,kind){{
 const renderer=new THREE.WebGLRenderer({{canvas,antialias:true,alpha:false}});renderer.outputColorSpace=THREE.SRGBColorSpace;
 const scene=new THREE.Scene();scene.background=new THREE.Color(0x111419);
 const camera=new THREE.PerspectiveCamera(34,1,0.01,20);
 if(kind==='body'){{camera.position.fromArray({encoded["body_camera_position"]});camera.lookAt(...{encoded["body_camera_target"]})}}else{{camera.position.set(0,0.24,0.82);camera.lookAt(0,0.235,0.03)}}
 scene.add(new THREE.HemisphereLight(0xffffff,0x263040,2.1));
 const key=new THREE.DirectionalLight(0xffffff,2.7);key.position.set(1.5,2.5,2);scene.add(key);
 const fill=new THREE.DirectionalLight(0x8fb6ff,1.2);fill.position.set(-2,1,1);scene.add(fill);
 return {{renderer,scene,camera,canvas}};
}}
const body=stage(document.querySelector('#body'),'body'),face=stage(document.querySelector('#face'),'face');
function load(stage,url){{
 new GLTFLoader().load(url,gltf=>{{stage.scene.add(gltf.scene);if(gltf.animations.length){{const clip=gltf.animations[0],mixer=new THREE.AnimationMixer(gltf.scene),action=mixer.clipAction(clip),duration=clip.duration;action.setLoop(THREE.LoopOnce,1);action.clampWhenFinished=true;action.play();action.paused=true;action.time=0;mixer.update(0);animations.push({{mixer,action,duration}})}}}},undefined,error=>{{console.error(error);}});
}}
load(body,{encoded["body"]});load(face,{encoded["face"]});
function resize(stage){{const w=stage.canvas.clientWidth,h=stage.canvas.clientHeight;if(stage.canvas.width!==w||stage.canvas.height!==h){{stage.renderer.setSize(w,h,false);stage.camera.aspect=w/h;stage.camera.updateProjectionMatrix()}}}}
function frame(){{resize(body);resize(face);for(const {{mixer,action,duration}} of animations){{action.time=Math.min(Math.max(source.currentTime,0),duration);mixer.update(0)}}body.renderer.render(body.scene,body.camera);face.renderer.render(face.scene,face.camera);if(source.duration){{scrub.value=source.currentTime/source.duration;time.textContent=`${{source.currentTime.toFixed(3)}} / ${{source.duration.toFixed(3)}} s`}}play.textContent=source.paused?'Play':'Pause';requestAnimationFrame(frame)}}requestAnimationFrame(frame);
play.addEventListener('click',()=>source.paused?source.play():source.pause());
scrub.addEventListener('input',()=>{{if(source.duration)source.currentTime=Number(scrub.value)*source.duration}});
fetch({encoded["manifest"]}).then(r=>r.json()).then(m=>{{if(m.production_validated||m.attachment.single_connected_character_mesh)throw new Error('Diagnostic manifest claimed unsafe readiness')}}).catch(console.error);
</script>
</body></html>
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return output


__all__ = [
    "BodyCompositionError",
    "UNIFIED_ARRAYS_SCHEMA_VERSION",
    "UNIFIED_PERFORMANCE_SCHEMA_VERSION",
    "compose_unified_performance",
    "write_unified_preview_html",
]
