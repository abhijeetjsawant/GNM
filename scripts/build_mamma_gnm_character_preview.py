#!/usr/bin/env python3
"""Build a traceable connected GNM + MAMMA character attachment review.

This is deliberately not an acting-quality claim.  It proves that a verified
MAMMA 55-joint body sequence can drive the connected-character exporter while
GNM retains facial ownership.  The shipped MAMMA example has no usable dialogue
facial close-up, so this builder writes an explicitly neutral GNM face track.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from autoanim_gnm.body_export import export_animated_body_glb
from autoanim_gnm.body_provider import sha256_file
from autoanim_gnm.body_compositor import compose_unified_performance
from autoanim_gnm.gltf_export import export_gnm_glb
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.mamma_compositor import (
    mamma_capture_timeline,
    write_neutral_gnm_face_performance,
)
from autoanim_gnm.mamma_motion import load_mamma_body_track, rebase_mamma_body_track
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.unified_gltf import export_unified_character_glb
from autoanim_gnm.viewer import VIEWER_THREE_VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODY_RUN = ROOT / ".cache" / "autoanim_gnm" / "body-provider" / "run" / "detailed-hands"
DEFAULT_BINDING = ROOT / "artifacts" / "n5-1-production-assembly"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_review(path: Path, *, duration_s: float) -> None:
    """Write a source-clocked review page for the single connected GLB."""

    path.write_text(
        f"""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>GNM + MAMMA attachment review</title><style>body{{margin:0;background:#101216;color:#eef1f5;font:14px system-ui}}header,footer{{padding:16px 22px}}header{{border-bottom:1px solid #2d3440}}h1{{font-size:18px;margin:0 0 6px}}p{{margin:0;color:#b3bdca}}main{{height:calc(100vh - 138px);display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2d3440}}section{{position:relative;background:#151920}}video,canvas{{width:100%;height:100%;display:block;object-fit:contain}}.tag{{position:absolute;top:12px;left:12px;background:#080a0dcc;padding:5px 8px;border-radius:4px}}footer{{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center}}button,input{{accent-color:#85b4ff}}button{{padding:7px 12px;background:#28303c;color:white;border:1px solid #536277;border-radius:5px}}</style><script type=\"importmap\">{{\"imports\":{{\"three\":\"/api/viewer/vendor/{VIEWER_THREE_VERSION}/three.module.js\",\"three/addons/\":\"/api/viewer/vendor/{VIEWER_THREE_VERSION}/addons/\"}}}}</script></head>
<body><header><h1>GNM face + MAMMA multiview body</h1><p>Attachment proof only: GNM head follows MAMMA neck/head motion. Face is intentionally neutral because this lifting clip has no qualified dialogue-face evidence.</p></header><main><section><span class=\"tag\">A001 source · frames 60–89</span><video id=\"source\" src=\"source.mp4\" playsinline preload=\"auto\"></video></section><section><span class=\"tag\">Connected character · preview blockers retained</span><canvas id=\"view\"></canvas></section></main><footer><button id=\"play\">Play</button><input id=\"scrub\" type=\"range\" min=\"0\" max=\"1\" step=\"0.0001\" value=\"0\"><span id=\"time\">0.000 / {duration_s:.3f} s</span></footer>
<script type=\"module\">import * as THREE from 'three';import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';const video=document.querySelector('#source'),canvas=document.querySelector('#view'),play=document.querySelector('#play'),scrub=document.querySelector('#scrub'),time=document.querySelector('#time');const r=new THREE.WebGLRenderer({{canvas,antialias:true}}),s=new THREE.Scene(),c=new THREE.PerspectiveCamera(28,1,.01,20);r.outputColorSpace=THREE.SRGBColorSpace;s.background=new THREE.Color(0x151920);c.position.set(0,1.32,3.15);c.lookAt(0,1.28,0);s.add(new THREE.HemisphereLight(0xffffff,0x263040,2.2));const key=new THREE.DirectionalLight(0xffffff,2.8);key.position.set(1.5,2.5,2);s.add(key);let mixer,action,duration={duration_s:.9f};new GLTFLoader().load('connected-character.glb',g=>{{s.add(g.scene);mixer=new THREE.AnimationMixer(g.scene);action=mixer.clipAction(g.animations[0]);duration=g.animations[0].duration;action.setLoop(THREE.LoopOnce,1);action.clampWhenFinished=true;action.play();action.paused=true;action.time=0;mixer.update(0)}});function draw(){{const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w||canvas.height!==h){{r.setSize(w,h,false);c.aspect=w/h;c.updateProjectionMatrix()}}if(mixer&&action){{action.time=Math.min(video.currentTime,duration);mixer.update(0)}}r.render(s,c);if(video.duration){{scrub.value=video.currentTime/video.duration;time.textContent=`${{video.currentTime.toFixed(3)}} / ${{video.duration.toFixed(3)}} s`}}play.textContent=video.paused?'Play':'Pause';requestAnimationFrame(draw)}}draw();play.onclick=()=>video.paused?video.play():video.pause();scrub.oninput=()=>{{if(video.duration)video.currentTime=Number(scrub.value)*video.duration}};video.onended=()=>{{video.currentTime=0;if(action){{action.time=0;mixer.update(0)}}}};</script></body></html>""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamma-params", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Trimmed A-camera source matching the MAMMA window")
    parser.add_argument("--source-start-frame", type=int, required=True)
    parser.add_argument("--source-size", type=int, nargs=2, default=(3840, 2160))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-run", type=Path, default=DEFAULT_BODY_RUN)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    arguments = parser.parse_args()

    parameter_path = arguments.mamma_params.resolve(strict=True)
    source = arguments.source.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    body_manifest = (arguments.body_run / "neutral-body.json").resolve(strict=True)
    body_asset = (arguments.body_run / "neutral-body.npz").resolve(strict=True)
    binding_manifest = (arguments.binding / "binding.json").resolve(strict=True)
    binding_arrays = (arguments.binding / "binding.npz").resolve(strict=True)

    imported_body_track = load_mamma_body_track(parameter_path)
    capture_world_root_origin_m = imported_body_track.root_translation_m[0].astype(
        float
    ).tolist()
    body_track = rebase_mamma_body_track(imported_body_track)
    source_sha256 = _sha256_file(source)
    timeline = mamma_capture_timeline(
        body_track,
        input_sha256=source_sha256,
        source_start_frame=arguments.source_start_frame,
    )
    write_json(output / "body-track.json", body_track.as_dict())
    write_npz(
        output / "body-track.npz",
        ticks=body_track.ticks,
        source_pts=timeline.source_pts,
        root_translation_m=body_track.root_translation_m,
        local_rotations_xyzw=body_track.local_rotations_xyzw,
        foot_contacts=body_track.foot_contacts,
    )
    body_glb = export_animated_body_glb(
        output / "body.glb",
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
        track=body_track,
        mapping_path=output / "body-mapping.npz",
    )
    face_performance = output / "neutral-gnm-face-performance.npz"
    write_neutral_gnm_face_performance(str(face_performance), timeline=timeline)
    gnm = GNMAdapter()
    neutral_face_glb = export_gnm_glb(
        output / "neutral-gnm-face.glb",
        gnm,
        gnm.mesh(),
        mapping_path=output / "neutral-gnm-face-mapping.npz",
    )
    capture = SimpleNamespace(
        input_sha256=source_sha256,
        source_pts=timeline.source_pts,
        ticks=timeline.ticks,
        source_time_base_numerator=timeline.source_time_base_numerator,
        source_time_base_denominator=timeline.source_time_base_denominator,
    )
    source_width, source_height = arguments.source_size
    composition = compose_unified_performance(
        output / "composition.json",
        arrays_path=output / "composition.npz",
        canonical_source_path=source,
        face_source_path=source,
        face_source_derivation={
            "schema_version": "autoanim.source-derivation/1.0",
            "operation": "spatial_crop_and_scale",
            "parent_source_sha256": source_sha256,
            "derived_source_sha256": source_sha256,
            "crop_ltrb_pixels": [0, 0, source_width, source_height],
            "output_size_pixels": [source_width, source_height],
            "timing_policy": "exact_source_pts_preserved",
        },
        face_performance_path=face_performance,
        face_glb_path=neutral_face_glb.path,
        soma_track=capture,
        body_track=body_track,
        body_glb_path=body_glb.path,
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
    )
    connected = export_unified_character_glb(
        output / "connected-character.glb",
        report_path=output / "connected-character-report.json",
        mapping_path=output / "connected-character-mapping.npz",
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
        binding_manifest_path=binding_manifest,
        binding_arrays_path=binding_arrays,
        composition_manifest_path=output / "composition.json",
        composition_arrays_path=output / "composition.npz",
        face_performance_path=face_performance,
        track=body_track,
        adapter=gnm,
    )
    shutil.copy2(source, output / "source.mp4")
    _write_review(
        output / "review.html",
        duration_s=body_track.duration_ticks / body_track.ticks_per_second,
    )
    report = {
        "schema_version": "autoanim.mamma-gnm-character-preview/1.0",
        "status": "attachment_and_body_motion_preview",
        "mamma_parameter_sha256": sha256_file(parameter_path),
        "canonical_source_sha256": source_sha256,
        "source_camera": "A001",
        "source_frame_range_inclusive": [
            arguments.source_start_frame,
            arguments.source_start_frame + len(body_track.ticks) - 1,
        ],
        "capture_world_root_origin_m": capture_world_root_origin_m,
        "root_space": "character_local_rebased_from_first_mamma_sample",
        "frame_count": len(body_track.ticks),
        "timeline_exact": bool(
            composition["timeline"]["face_body_exact_pts_match"]
            and composition["timeline"]["face_body_exact_ticks_match"]
        ),
        "motion_ownership": {
            "mamma": ["root", "torso", "limbs", "wrists", "fingers", "macro_neck_head"],
            "gnm": ["identity", "face_mesh", "eyes", "lips", "tongue", "teeth"],
            "facial_mode": "neutral_no_inference",
        },
        "connected_glb_sha256": sha256_file(connected.path),
        "publish_ready": False,
        "publish_blockers": list(connected.publish_blockers)
        + ["MAMMA_SOURCE_HAS_NO_QUALIFIED_DIALOGUE_FACE_EVIDENCE"],
    }
    write_json(output / "mamma-gnm-report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
