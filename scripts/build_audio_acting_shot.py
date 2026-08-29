#!/usr/bin/env python3
"""Build a real audio-driven face + deterministic acting body preview.

This is deliberately a preview artifact.  The audio face track is the retained
Audio2Face-derived GNM result, while the body is an editable acting-plan
compile.  The exporter rejects mismatched ticks, ownership, identity, and
composition hashes before it writes the connected GLB.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np

from autoanim_gnm.body import (
    DETAILED_HUMANOID,
    BodyTrack,
    _euler_xyz_quaternion,
    _quaternion_multiply,
    attachment_contract,
    compile_body_track,
)
from autoanim_gnm.acting import TICKS_PER_SECOND, validate_acting_plan
from autoanim_gnm.body_provider import sha256_file
from autoanim_gnm.body_projection import (
    constrain_restrained_root_travel,
    project_generated_foot_contacts,
)
from autoanim_gnm.speech_motion_candidates import (
    SpeechMotionCandidate,
    build_speech_motion_candidates,
    candidate_report,
    resample_generated_body_track,
)
from autoanim_gnm.speech_motion_provider import (
    load_gesturelsm_response,
    load_speech_motion_request,
)
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.unified_gltf import _composition_array_sha256, export_unified_character_glb


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "audio-acting-shot"
AUDIO_JOB = ROOT / "artifacts" / "production-next-audio" / "01kxw6h0d3x9zbzcpd47fsbk59"
BODY_RUN = ROOT / ".cache" / "autoanim_gnm" / "body-provider" / "run" / "detailed-hands"
BINDING = ROOT / "artifacts" / "n5-1-production-assembly"
PRESENTER_PROFILE = "relaxed-standing-a-pose/v1"


def _acting_plan(duration_ticks: int) -> dict:
    quarter = duration_ticks // 4
    beats = []
    definitions = (
        (
            "listen",
            "grounded",
            ["small", "head_nod"],
            "restrained",
            0.38,
            "unspecified",
            0.0,
        ),
        ("persuade", "open", ["shrug", "small"], "warm", 0.68, "unspecified", 0.0),
        ("challenge", "forward", ["point", "small"], "resolve", 0.82, "unspecified", 0.0),
        (
            "reassure",
            "grounded",
            ["hand_to_chest", "small"],
            "warm",
            0.52,
            "unspecified",
            0.0,
        ),
    )
    for index, (
        intent,
        stance,
        gesture_tags,
        expression,
        arousal,
        gaze,
        gaze_strength,
    ) in enumerate(definitions):
        start = index * quarter
        end = (
            duration_ticks
            if index == len(definitions) - 1
            else (index + 1) * quarter
        )
        beats.append(
            {
                "id": f"beat_{index:04d}",
                "start_tick": start,
                "end_tick": end,
                "intent": intent,
                "valence": 0.35 if expression == "warm" else 0.08,
                "arousal": arousal,
                "body": {"stance": stance, "gesture_tags": gesture_tags, "energy": arousal},
                "face": {"expression_tags": [expression], "intensity": arousal},
                "gaze": {"target": gaze, "strength": gaze_strength},
                "constraints": {
                    "preserve_lipsync": True,
                    "preserve_foot_contacts": True,
                },
            }
        )
    return {
        "schema_version": "autoanim.acting-plan/1.0",
        "status": "ok",
        "summary": "Audio-driven dialogue with restrained explanatory gestures and a resolving finish.",
        "beats": beats,
        "diagnostics": [
            "Body motion is deterministic and editable; it is not claimed as recovered from the speaker.",
            "GNM lipsync, tongue, eyes, and facial expression remain audio-owned.",
        ],
    }


def _apply_presenter_profile(track: BodyTrack) -> BodyTrack:
    """Overlay the explicit relaxed standing base for this review shot.

    ``compile_body_track`` correctly treats the canonical skeleton's identity
    rotations as a neutral authoring pose.  That is a horizontal-arm T-pose,
    which is useful for interchange but not a credible person at rest.  This
    shot-level profile lowers the arms into a symmetric A-pose before its
    editable acting gestures are applied.  It intentionally leaves root,
    hips, legs, feet, and fingers untouched: it is not presented as captured
    body motion or a substitute for the forthcoming body-motion provider.
    """

    if track.joint_names != DETAILED_HUMANOID.names:
        raise ValueError(
            "The audio acting presenter profile requires DETAILED_HUMANOID"
        )

    base_angles = np.zeros((len(DETAILED_HUMANOID.joints), 3), dtype=np.float64)
    # Rotating the local +/-X upper-arm bones around Z puts both wrists below
    # their shoulders.  The small forearm continuation avoids a rigid straight
    # horizontal silhouette without changing any contact-constrained joint.
    base_angles[DETAILED_HUMANOID.index("LeftUpperArm"), 2] = np.deg2rad(35.0)
    base_angles[DETAILED_HUMANOID.index("RightUpperArm"), 2] = np.deg2rad(-35.0)
    base_angles[DETAILED_HUMANOID.index("LeftLowerArm"), 2] = np.deg2rad(8.0)
    base_angles[DETAILED_HUMANOID.index("RightLowerArm"), 2] = np.deg2rad(-8.0)
    base = _euler_xyz_quaternion(base_angles)
    base_frames = np.broadcast_to(base, track.local_rotations_xyzw.shape).copy()
    rotations = _quaternion_multiply(base_frames, track.local_rotations_xyzw)
    rotations /= np.linalg.norm(rotations, axis=2, keepdims=True)
    return replace(track, local_rotations_xyzw=rotations.astype(np.float32))


def _write_face_performance(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(AUDIO_JOB / "controls.npz", allow_pickle=False) as archive:
        controls = {name: np.array(archive[name], copy=True) for name in archive.files}
    frame_count = len(controls["timestamps"])
    ticks = np.rint(controls["timestamps"].astype(np.float64) * TICKS_PER_SECOND).astype(
        np.int64
    )
    timestamps = ticks.astype(np.float64) / TICKS_PER_SECOND
    identity = np.zeros(253, dtype=np.float32)
    source_pts = np.arange(frame_count, dtype=np.int64)
    write_npz(
        path,
        identity=identity,
        expression=controls["expression"].astype(np.float32),
        rotations=controls["rotations"].astype(np.float32),
        translation=controls["translation"].astype(np.float32),
        timestamps_seconds=timestamps,
        source_pts=source_pts,
    )
    return ticks, timestamps, source_pts


def _write_composition(
    manifest_path: Path,
    arrays_path: Path,
    *,
    ticks: np.ndarray,
    timestamps: np.ndarray,
    source_pts: np.ndarray,
    identity: np.ndarray,
    expression: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
    body_track,
    face_performance_path: Path,
    body_asset_path: Path,
    body_manifest_path: Path,
    body_backend: str,
    body_profile: str,
) -> None:
    owned_rotations = rotations.copy()
    owned_rotations[:, :2] = 0.0
    owned_translation = np.zeros_like(translation)
    write_npz(
        arrays_path,
        schema_version=np.asarray("autoanim.unified-performance-arrays/1.0"),
        ticks=ticks,
        source_pts=source_pts,
        timestamps_seconds=timestamps,
        gnm_identity=identity,
        gnm_expression=expression,
        gnm_owned_rotations=owned_rotations,
        gnm_owned_translation=owned_translation,
        body_root_translation_m=body_track.root_translation_m,
        body_local_rotations_xyzw=body_track.local_rotations_xyzw,
        body_foot_contacts=body_track.foot_contacts,
    )
    contract = attachment_contract()
    contract_sha256 = sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "autoanim.unified-performance/1.0",
        "status": "diagnostic_audio_acting_preview",
        "production_validated": False,
        "publish_ready": False,
        "source": {
            "audio": {
                "name": "normalized.wav",
                "role": "canonical_source_audio",
                "sha256": sha256_file(AUDIO_JOB / "normalized.wav"),
            },
            "face_backend": "retained Audio2Face-derived GNM controls",
            "body_backend": body_backend,
            "body_profile": body_profile,
        },
        "timeline": {
            "ticks_per_second": TICKS_PER_SECOND,
            "duration_ticks": int(ticks[-1]),
            "frame_count": int(len(ticks)),
            "source_pts_sha256": _composition_array_sha256(source_pts),
            "ticks_sha256": _composition_array_sha256(ticks),
            "face_body_exact_pts_match": True,
            "face_body_exact_ticks_match": True,
        },
        "ownership": {
            "contract_schema_version": contract["schema_version"],
            "contract_sha256": contract_sha256,
            "body": ["root_translation", "Root_to_Head_base_rotation", "acting_gestures"],
            "gnm": ["identity", "facial_expression", "lipsync", "eyes", "teeth", "tongue", "oral_contact"],
            "authoritative_face_arrays_byte_identical": True,
            "base_head_applied_once": True,
            "conflicts": [],
        },
        "attachment": {"calibrated": False, "single_connected_character_mesh": True},
        "artifacts": {
            "unified_arrays": {"sha256": sha256_file(arrays_path)},
            "body_asset": {"sha256": sha256_file(body_asset_path)},
            "body_manifest": {"sha256": sha256_file(body_manifest_path)},
            "face_performance": {"sha256": sha256_file(face_performance_path)},
        },
        "publish_blockers": [
            "CHARACTER_BODY_IDENTITY_NOT_CALIBRATED",
            (
                "GESTURELSM_CANDIDATE_NOT_ANIMATOR_APPROVED"
                if body_backend.startswith("GestureLSM")
                else "DETERMINISTIC_ACTING_BODY_PREVIEW_ONLY"
            ),
            "GNM_FACE_TRACK_NOT_PRODUCTION_CALIBRATED",
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["payload_sha256"] = sha256(payload).hexdigest()
    write_json(manifest_path, manifest)


def _write_viewer(path: Path, *, body_description: str) -> None:
    document = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AutoAnim audio acting shot</title>
<style>:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#0d0f12;color:#f4f6f8}header{display:flex;justify-content:space-between;gap:24px;padding:18px 22px;border-bottom:1px solid #2b3037}h1{font-size:18px;margin:0 0 5px}p{color:#aeb6c0;font-size:12px;margin:0;line-height:1.45}.badge{border:1px solid #795221;background:#2c2114;color:#ffd69a;padding:7px 10px;border-radius:6px;font-size:11px;height:max-content;white-space:nowrap}main{height:calc(100vh - 176px);min-height:440px;position:relative;background:#111419;overflow:hidden}canvas{display:block;width:100%;height:100%}.overlay{position:absolute;left:14px;top:14px;display:flex;gap:8px;flex-wrap:wrap}.overlay button,footer button{border:1px solid #47505b;background:#222830;color:#fff;border-radius:6px;padding:7px 11px;cursor:pointer}.overlay button.active{border-color:#86b7ff;color:#d9e9ff}footer{display:grid;grid-template-columns:auto auto 1fr auto;gap:12px;align-items:center;padding:12px 22px;border-top:1px solid #2b3037}audio{width:250px}input{width:100%}#time{font-size:12px;color:#b9c2cc;font-variant-numeric:tabular-nums}@media(max-width:800px){audio{width:180px}main{height:65vh}}
</style><script type="importmap">{"imports":{"three":"/.cache/autoanim_gnm/viewer/three-0.183.2/three.module.js","three/addons/":"/.cache/autoanim_gnm/viewer/three-0.183.2/addons/"}}</script></head>
<body><header><div><h1>Acting shot · audio → face + body</h1><p>Real Audio2Face-derived GNM motion drives lipsync, tongue, eyes, and expression.<br>__BODY_DESCRIPTION__</p></div><div class="badge">PREVIEW · NOT PUBLISHABLE</div></header><main><canvas id="viewport"></canvas><div class="overlay"><button data-camera="body" class="active">Full body</button><button data-camera="face">Face close-up</button><button data-camera="hands">Hands</button></div></main><footer><button id="play">Play</button><audio id="audio" controls preload="auto" src="normalized.wav"></audio><input id="scrub" type="range" min="0" max="1" step="0.0001" value="0"><span id="time">0.000 / 0.000 s</span></footer>
<script type="module">import * as THREE from 'three';import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const canvas=document.querySelector('#viewport'),audio=document.querySelector('#audio'),play=document.querySelector('#play'),scrub=document.querySelector('#scrub'),time=document.querySelector('#time');const renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.outputColorSpace=THREE.SRGBColorSpace;const scene=new THREE.Scene();scene.background=new THREE.Color(0x111419);const camera=new THREE.PerspectiveCamera(31,1,.01,20);const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.target.set(0,.6,0);scene.add(new THREE.HemisphereLight(0xffffff,0x263040,2.0));const key=new THREE.DirectionalLight(0xffffff,2.7);key.position.set(1.5,2.5,2);scene.add(key);const fill=new THREE.DirectionalLight(0x8fb6ff,1.2);fill.position.set(-2,1,1);scene.add(fill);
const presets={body:{position:[2.65,.65,4.1],target:[0,.7,0]},face:{position:[0,1.62,1.2],target:[0,1.55,0]},hands:{position:[1.15,1.0,1.7],target:[0,1.0,0]}};let preset='body';function applyPreset(name){preset=name;camera.position.fromArray(presets[name].position);controls.target.fromArray(presets[name].target);controls.update();document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('active',b.dataset.camera===name))}document.querySelectorAll('[data-camera]').forEach(b=>b.addEventListener('click',()=>applyPreset(b.dataset.camera)));applyPreset('body');let mixer=null,loaded=false;new GLTFLoader().load('connected-character.glb',gltf=>{scene.add(gltf.scene);mixer=new THREE.AnimationMixer(gltf.scene);const action=mixer.clipAction(gltf.animations[0]);action.play();mixer.setTime(0);loaded=true},undefined,e=>console.error(e));
function resize(){const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w||canvas.height!==h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}}function frame(){resize();if(mixer&&loaded)mixer.setTime(Math.min(audio.currentTime,7.0));controls.update();renderer.render(scene,camera);if(audio.duration){scrub.value=audio.currentTime/audio.duration;time.textContent=`${audio.currentTime.toFixed(3)} / ${audio.duration.toFixed(3)} s`}play.textContent=audio.paused?'Play':'Pause';requestAnimationFrame(frame)}requestAnimationFrame(frame);play.addEventListener('click',()=>audio.paused?audio.play():audio.pause());scrub.addEventListener('input',()=>{if(audio.duration)audio.currentTime=Number(scrub.value)*audio.duration});audio.addEventListener('ended',()=>{if(mixer)mixer.setTime(0)});</script></body></html>'''
    path.write_text(
        document.replace("__BODY_DESCRIPTION__", body_description), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesturelsm-request", type=Path)
    parser.add_argument("--gesturelsm-response", type=Path)
    arguments = parser.parse_args(argv)
    if (arguments.gesturelsm_request is None) != (
        arguments.gesturelsm_response is None
    ):
        parser.error("GestureLSM request and response must be supplied together")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(AUDIO_JOB / "controls.npz", allow_pickle=False) as archive:
        controls = {name: np.array(archive[name], copy=True) for name in archive.files}
    ticks = np.rint(controls["timestamps"].astype(np.float64) * TICKS_PER_SECOND).astype(np.int64)
    duration_ticks = int(ticks[-1])
    plan = _acting_plan(duration_ticks)
    validate_acting_plan(plan, duration_ticks=duration_ticks)
    write_json(OUTPUT / "acting-plan.json", plan)
    body_backend = "deterministic approved-plan compiler"
    body_profile = PRESENTER_PROFILE
    candidate: SpeechMotionCandidate | None = None
    body_resampled_to_face_clock = False
    generated_body_frame_count: int | None = None
    if arguments.gesturelsm_request is not None:
        request = load_speech_motion_request(arguments.gesturelsm_request)
        native = load_gesturelsm_response(
            arguments.gesturelsm_response,
            profile=request.profile,
            request=request,
        )
        candidate = build_speech_motion_candidates((native,))[0]
        generated_body_frame_count = len(candidate.raw_track.ticks)
        if np.array_equal(candidate.raw_track.ticks, ticks):
            body_track = candidate.projected_track
        else:
            body_resampled_to_face_clock = True
            resampled = resample_generated_body_track(candidate.raw_track, ticks)
            restrained, resampled_root_stabilization = constrain_restrained_root_travel(
                resampled
            )
            body_track, resampled_projection = project_generated_foot_contacts(
                restrained
            )
        body_backend = "GestureLSM Shortcut-ReFlow generated speech motion"
        body_profile = f"gesturelsm-seed-{native.candidate_seed}"
        report = candidate_report((candidate,))
        report["composition_timing"] = {
            "source_frame_count": generated_body_frame_count,
            "target_frame_count": len(ticks),
            "quaternion_resampled_to_face_clock": body_resampled_to_face_clock,
            "contacts_reprojected_after_resampling": body_resampled_to_face_clock,
            "resampled_root_stabilization": (
                resampled_root_stabilization.as_dict()
                if body_resampled_to_face_clock
                else None
            ),
            "resampled_projection": (
                resampled_projection.as_dict() if body_resampled_to_face_clock else None
            ),
        }
        write_json(OUTPUT / "candidate-report.json", report)
    else:
        (OUTPUT / "candidate-report.json").unlink(missing_ok=True)
        body_track = _apply_presenter_profile(
            compile_body_track(
                plan,
                duration_ticks=duration_ticks,
                sample_rate_hz=int(controls["fps"]),
                skeleton=DETAILED_HUMANOID,
            )
        )
    write_json(OUTPUT / "body-track.json", body_track.as_dict())
    write_npz(
        OUTPUT / "body-track.npz",
        ticks=body_track.ticks,
        root_translation_m=body_track.root_translation_m,
        local_rotations_xyzw=body_track.local_rotations_xyzw,
        foot_contacts=body_track.foot_contacts,
        gaze_direction_body=body_track.gaze_direction_body,
        gaze_strength=body_track.gaze_strength,
        gnm_eye_rotations_xyzw=body_track.gnm_eye_rotations_xyzw,
    )
    face_performance = OUTPUT / "face-performance.npz"
    face_ticks, timestamps, source_pts = _write_face_performance(face_performance)
    if not np.array_equal(face_ticks, body_track.ticks):
        raise RuntimeError("Audio face and acting body ticks diverged")
    identity = np.zeros(253, dtype=np.float32)
    body_asset = BODY_RUN / "neutral-body.npz"
    body_manifest = BODY_RUN / "neutral-body.json"
    arrays = OUTPUT / "composition.npz"
    composition = OUTPUT / "composition.json"
    _write_composition(
        composition,
        arrays,
        ticks=face_ticks,
        timestamps=timestamps,
        source_pts=source_pts,
        identity=identity,
        expression=controls["expression"].astype(np.float32),
        rotations=controls["rotations"].astype(np.float32),
        translation=controls["translation"].astype(np.float32),
        body_track=body_track,
        face_performance_path=face_performance,
        body_asset_path=body_asset,
        body_manifest_path=body_manifest,
        body_backend=body_backend,
        body_profile=body_profile,
    )
    output = export_unified_character_glb(
        OUTPUT / "connected-character.glb",
        report_path=OUTPUT / "connected-character-report.json",
        mapping_path=OUTPUT / "connected-character-mapping.npz",
        body_manifest_path=body_manifest,
        body_asset_path=body_asset,
        binding_manifest_path=BINDING / "binding.json",
        binding_arrays_path=BINDING / "binding.npz",
        composition_manifest_path=composition,
        composition_arrays_path=arrays,
        face_performance_path=face_performance,
        track=body_track,
    )
    shutil.copy2(AUDIO_JOB / "normalized.wav", OUTPUT / "normalized.wav")
    _write_viewer(
        OUTPUT / "review.html",
        body_description=(
            "A pinned GestureLSM candidate drives the detailed body and hands on the same timeline."
            if candidate is not None
            else "An editable acting plan drives the detailed body and head gesture on the same timeline."
        ),
    )
    composition_blockers = [
        "CHARACTER_BODY_IDENTITY_NOT_CALIBRATED",
        (
            "GESTURELSM_CANDIDATE_NOT_ANIMATOR_APPROVED"
            if candidate is not None
            else "DETERMINISTIC_ACTING_BODY_PREVIEW_ONLY"
        ),
        "GNM_FACE_TRACK_NOT_PRODUCTION_CALIBRATED",
    ]
    publish_blockers = list(
        dict.fromkeys([*output.publish_blockers, *composition_blockers])
    )
    summary = {
        "duration_s": duration_ticks / TICKS_PER_SECOND,
        "frame_count": len(face_ticks),
        "fps": int(controls["fps"]),
        "body_joint_count": len(DETAILED_HUMANOID.joints),
        "face_expression_frames": int(len(controls["expression"])),
        "audio_sha256": sha256_file(OUTPUT / "normalized.wav"),
        "glb": str(output.path),
        "publish_ready": output.publish_ready and not publish_blockers,
        "publish_blockers": publish_blockers,
        "body_backend": body_backend,
        "candidate_seed": None if candidate is None else candidate.seed,
        "generated_body_frame_count": generated_body_frame_count,
        "composition_frame_count": len(body_track.ticks),
        "body_resampled_to_face_clock": body_resampled_to_face_clock,
    }
    write_json(OUTPUT / "shot-report.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
