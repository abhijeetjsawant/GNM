#!/usr/bin/env python3
"""Build an exact-clock video-driven GNM face + SOMA body review shot."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np

from autoanim_gnm.body_compositor import compose_unified_performance
from autoanim_gnm.body import DETAILED_HUMANOID
from autoanim_gnm.body_export import export_animated_body_glb
from autoanim_gnm.body_provider import sha256_file
from autoanim_gnm.body_projection import (
    constrain_arms_to_video_keypoints,
    constrain_hands_to_video_keypoints,
)
from autoanim_gnm.nvidia_body_provider import (
    load_gem_x_cuda_response,
    load_gem_x_preview_response,
)
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.soma_motion import project_soma_to_detailed_body_track
from autoanim_gnm.unified_gltf import export_unified_character_glb
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODY_RUN = (
    ROOT / ".cache" / "autoanim_gnm" / "body-provider" / "run" / "detailed-hands"
)
DEFAULT_BINDING = ROOT / "artifacts" / "n5-1-production-assembly"


def _condition_gnm_head_rotations(
    rotations: np.ndarray,
    detected: np.ndarray,
) -> np.ndarray:
    """Fill profile-dropout gaps and suppress single-frame head snaps.

    GNM's tracker already carries values across missed detections, but a long
    profile dropout can decay toward neutral and then snap to the next valid
    observation. Interpolating only between observed anchors keeps the actual
    turn, and a short symmetric polynomial filter removes tracker jitter
    without introducing timing lag.
    """

    conditioned = np.asarray(rotations, dtype=np.float64).copy()
    reliable = np.asarray(detected, dtype=bool)
    if conditioned.ndim != 3 or conditioned.shape[1:] != (4, 3):
        raise ValueError("GNM rotations must have shape [frames, 4, 3]")
    if reliable.shape != (conditioned.shape[0],):
        raise ValueError("GNM detection mask must match the rotation clock")
    anchors = np.flatnonzero(reliable)
    if len(anchors) < 2:
        raise ValueError("At least two detected face frames are required")

    clock = np.arange(len(conditioned), dtype=np.float64)
    window = min(7, len(conditioned) if len(conditioned) % 2 else len(conditioned) - 1)
    for channel in (0, 1):
        for axis in range(3):
            filled = np.interp(
                clock,
                anchors.astype(np.float64),
                conditioned[anchors, channel, axis],
            )
            if window >= 5:
                filled = savgol_filter(filled, window, 2, mode="interp")
            conditioned[:, channel, axis] = filled
    return conditioned


def _write_viewer(path: Path, *, duration_s: float) -> None:
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video-driven acting review</title><style>
:root{{color-scheme:dark;font-family:Inter,system-ui,sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:#0d1014;color:#f4f5f7}}
header{{padding:18px 22px;border-bottom:1px solid #292e36}}h1{{font-size:18px;margin:0 0 5px}}p{{font-size:12px;color:#aeb6c2;margin:0}}
main{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#292e36;height:calc(100vh - 145px)}}section{{position:relative;background:#11151b;overflow:hidden}}
video,canvas{{width:100%;height:100%;object-fit:contain;display:block}}.label{{position:absolute;z-index:2;left:12px;top:12px;background:#090b0dcc;padding:6px 8px;border-radius:5px;font-size:12px}}
footer{{height:72px;display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;padding:12px 22px}}button{{padding:8px 15px;background:#242a32;border:1px solid #46505d;border-radius:6px;color:white}}input{{width:100%}}span{{font-size:12px;font-variant-numeric:tabular-nums;color:#b7bec8}}
</style><script type="importmap">{{"imports":{{"three":"/vendor/three.module.js","three/addons/":"/vendor/addons/"}}}}</script></head>
<body><header><h1>Video → connected 3D acting</h1><p>GNM owns face, eyes, lips and oral geometry; GEM-X/SOMA owns torso, arms and hands. One exact source clock drives both.</p></header>
<main><section><div class="label">Source dialogue</div><video id="source" src="source.mp4" playsinline preload="auto"></video></section><section><div class="label">Connected character · research preview</div><canvas id="view"></canvas></section></main>
<footer><button id="play">Play</button><input id="scrub" type="range" min="0" max="1" step="0.0001" value="0"><span id="time">0.000 / {duration_s:.3f} s</span></footer>
<script type="module">import * as THREE from 'three';import {{GLTFLoader}} from 'three/addons/loaders/GLTFLoader.js';
const source=document.querySelector('#source'),canvas=document.querySelector('#view'),play=document.querySelector('#play'),scrub=document.querySelector('#scrub'),time=document.querySelector('#time');
const renderer=new THREE.WebGLRenderer({{canvas,antialias:true}});renderer.outputColorSpace=THREE.SRGBColorSpace;const scene=new THREE.Scene();scene.background=new THREE.Color(0x11151b);
const camera=new THREE.PerspectiveCamera(28,1,.01,20);camera.position.set(0,1.32,3.15);camera.lookAt(0,1.27,0);scene.add(new THREE.HemisphereLight(0xffffff,0x263040,2.2));const key=new THREE.DirectionalLight(0xffffff,2.8);key.position.set(1.5,2.5,2);scene.add(key);const fill=new THREE.DirectionalLight(0x8fb6ff,1.1);fill.position.set(-2,1,1);scene.add(fill);
let mixer=null,action=null,duration={duration_s:.9f};new GLTFLoader().load('connected-character.glb',g=>{{scene.add(g.scene);mixer=new THREE.AnimationMixer(g.scene);action=mixer.clipAction(g.animations[0]);duration=g.animations[0].duration;action.setLoop(THREE.LoopOnce,1);action.clampWhenFinished=true;action.play();action.paused=true;action.time=0;mixer.update(0)}});
function resize(){{const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w||canvas.height!==h){{renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}}}}function frame(){{resize();if(mixer&&action){{action.time=Math.min(source.currentTime,duration);mixer.update(0)}}renderer.render(scene,camera);if(source.duration){{scrub.value=source.currentTime/source.duration;time.textContent=`${{source.currentTime.toFixed(3)}} / ${{source.duration.toFixed(3)}} s`}}play.textContent=source.paused?'Play':'Pause';requestAnimationFrame(frame)}}frame();
play.addEventListener('click',()=>source.paused?source.play():source.pause());scrub.addEventListener('input',()=>{{if(source.duration)source.currentTime=Number(scrub.value)*source.duration}});source.addEventListener('ended',()=>{{source.currentTime=0;if(action){{action.time=0;mixer.update(0)}}}});
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--face-source", type=Path)
    parser.add_argument(
        "--face-crop", type=int, nargs=4, default=(0, 0, 640, 832)
    )
    parser.add_argument(
        "--face-output-size", type=int, nargs=2, default=(640, 832)
    )
    parser.add_argument("--soma-manifest", type=Path, required=True)
    parser.add_argument(
        "--cuda-soma",
        action="store_true",
        help=(
            "Load a full-image-conditioned CUDA GEM-X response. A failed raw "
            "regional audit is accepted only when trusted visual keypoints are "
            "also supplied and the downstream correction gates pass."
        ),
    )
    parser.add_argument(
        "--visual-keypoints",
        type=Path,
        help="Trusted finite [frames,77,3] NumPy evidence for video arm directions",
    )
    parser.add_argument("--face-job", type=Path, required=True)
    parser.add_argument("--face-performance", type=Path)
    parser.add_argument("--transfer-gnm-head", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body-run", type=Path, default=DEFAULT_BODY_RUN)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    face_source = (
        arguments.face_source.resolve() if arguments.face_source else source
    )
    face_job = arguments.face_job.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    body_manifest = (arguments.body_run / "neutral-body.json").resolve()
    body_asset = (arguments.body_run / "neutral-body.npz").resolve()

    if arguments.cuda_soma:
        soma = load_gem_x_cuda_response(
            arguments.soma_manifest.resolve(),
            require_regional_fidelity=arguments.visual_keypoints is None,
        )
    else:
        soma = load_gem_x_preview_response(arguments.soma_manifest.resolve())
    track = project_soma_to_detailed_body_track(soma)
    visual_arm_diagnostics = None
    visual_hand_diagnostics = None
    if arguments.visual_keypoints is not None:
        keypoints_path = arguments.visual_keypoints.resolve()
        observed_keypoints = np.load(keypoints_path, allow_pickle=False)
        with np.load(body_asset, allow_pickle=False) as body_archive:
            provider_local_rest = np.asarray(
                body_archive["local_rest_matrices"], dtype=np.float64
            )
            provider_parents = np.asarray(body_archive["parents"], dtype=np.int64)
        track, visual_arm_diagnostics = constrain_arms_to_video_keypoints(
            track,
            observed_keypoints,
            provider_local_rest_matrices=provider_local_rest,
            provider_parents=provider_parents,
        )
        if not visual_arm_diagnostics.passed:
            raise ValueError(
                "Video arm correction failed its relative-direction gate: "
                + ", ".join(visual_arm_diagnostics.failure_reasons)
            )
        track, visual_hand_diagnostics = constrain_hands_to_video_keypoints(
            track,
            observed_keypoints,
            provider_local_rest_matrices=provider_local_rest,
            provider_parents=provider_parents,
        )
        if not visual_hand_diagnostics.passed:
            raise ValueError(
                "Video hand correction failed its rendered-direction gate: "
                + ", ".join(visual_hand_diagnostics.failure_reasons)
            )
    face_performance = (
        arguments.face_performance.resolve()
        if arguments.face_performance
        else face_job / "performance.npz"
    )
    if arguments.transfer_gnm_head:
        with np.load(face_job / "performance.npz", allow_pickle=False) as archive:
            observed_rotations = np.asarray(archive["rotations"], dtype=np.float64)
            detected = np.asarray(archive["detected"], dtype=bool)
        if observed_rotations.shape != (len(track.ticks), 4, 3):
            raise ValueError("GNM head observations do not match the body clock")
        observed_rotations = _condition_gnm_head_rotations(
            observed_rotations,
            detected,
        )
        body_rotations = np.array(track.local_rotations_xyzw, copy=True)
        body_rotations[:, DETAILED_HUMANOID.index("Neck")] = Rotation.from_rotvec(
            observed_rotations[:, 0]
        ).as_quat().astype(np.float32)
        body_rotations[:, DETAILED_HUMANOID.index("Head")] = Rotation.from_rotvec(
            observed_rotations[:, 1]
        ).as_quat().astype(np.float32)
        provenance = sha256()
        provenance.update(b"autoanim.gnm-head-to-body/1.0\0")
        provenance.update(b"detected-anchor-interpolation+savgol-7-2\0")
        provenance.update(track.source_plan_sha256.encode("ascii"))
        provenance.update(sha256_file(face_job / "performance.npz").encode("ascii"))
        track = replace(
            track,
            local_rotations_xyzw=body_rotations,
            source_plan_sha256=provenance.hexdigest(),
        )
    write_json(output / "body-track.json", track.as_dict())
    write_npz(
        output / "body-track.npz",
        ticks=track.ticks,
        root_translation_m=track.root_translation_m,
        local_rotations_xyzw=track.local_rotations_xyzw,
        foot_contacts=track.foot_contacts,
        gaze_direction_body=track.gaze_direction_body,
        gaze_strength=track.gaze_strength,
        gnm_eye_rotations_xyzw=track.gnm_eye_rotations_xyzw,
    )
    body = export_animated_body_glb(
        output / "body.glb",
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
        track=track,
        mapping_path=output / "body-mapping.npz",
    )

    face_glb = face_job / "performance.glb"
    source_sha = sha256_file(source)
    face_source_sha = sha256_file(face_source)
    composition = compose_unified_performance(
        output / "composition.json",
        arrays_path=output / "composition.npz",
        canonical_source_path=source,
        face_source_path=face_source,
        face_source_derivation={
            "schema_version": "autoanim.source-derivation/1.0",
            "operation": "spatial_crop_and_scale",
            "parent_source_sha256": source_sha,
            "derived_source_sha256": face_source_sha,
            "crop_ltrb_pixels": list(arguments.face_crop),
            "output_size_pixels": list(arguments.face_output_size),
            "timing_policy": "exact_source_pts_preserved",
        },
        face_performance_path=face_performance,
        face_glb_path=face_glb,
        soma_track=soma,
        body_track=track,
        body_glb_path=body.path,
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
    )
    connected = export_unified_character_glb(
        output / "connected-character.glb",
        report_path=output / "connected-character-report.json",
        mapping_path=output / "connected-character-mapping.npz",
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
        binding_manifest_path=(arguments.binding / "binding.json").resolve(),
        binding_arrays_path=(arguments.binding / "binding.npz").resolve(),
        composition_manifest_path=output / "composition.json",
        composition_arrays_path=output / "composition.npz",
        face_performance_path=face_performance,
        track=track,
    )
    shutil.copy2(source, output / "source.mp4")
    duration_s = track.duration_ticks / track.ticks_per_second
    _write_viewer(output / "review.html", duration_s=duration_s)
    report = {
        "schema_version": "autoanim.video-acting-shot/1.0",
        "status": "research_preview",
        "source_sha256": source_sha,
        "frame_count": len(track.ticks),
        "duration_s": duration_s,
        "source_pts_exact_match": composition["timeline"]["face_body_exact_pts_match"],
        "face_performance_sha256": sha256_file(face_performance),
        "soma_motion_sha256": soma.content_sha256(),
        "connected_glb_sha256": sha256_file(connected.path),
        "head_motion_owner": (
            "body_channels_transferred_from_gnm_video_observation"
            if arguments.transfer_gnm_head
            else "soma_body_track"
        ),
        "head_motion_conditioning": (
            "detected_anchor_interpolation_plus_savgol_7_frame_order_2"
            if arguments.transfer_gnm_head
            else "none"
        ),
        "visual_arm_constraint": (
            visual_arm_diagnostics.as_dict()
            if visual_arm_diagnostics is not None
            else None
        ),
        "visual_hand_constraint": (
            visual_hand_diagnostics.as_dict()
            if visual_hand_diagnostics is not None
            else None
        ),
        "capture_scope": {
            "source_framing": "waist_up",
            "observed_regions": [
                "face",
                "head",
                "torso",
                "raised_arm",
                "raised_hand",
            ],
            "partially_observed_regions": ["other_arm", "other_hand"],
            "unobserved_regions": ["hips", "legs", "feet"],
            "review_policy": "unobserved_lower_body_excluded_from_final_render",
        },
        "publish_ready": False,
        "publish_blockers": list(connected.publish_blockers),
    }
    write_json(output / "shot-report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
