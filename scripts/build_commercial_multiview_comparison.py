#!/usr/bin/env python3
"""Run the clean-room multiview body pipeline on a calibrated clip."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

from autoanim_gnm.body_export import export_animated_body_glb
from autoanim_gnm.commercial_multiview import (
    JOINT_NAMES,
    load_camera_rig,
    load_observation_jsonl,
    reconstruct_multiview,
)
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.viewer import VIEWER_THREE_VERSION


ROOT = Path(__file__).resolve().parents[1]
SWIFT_SOURCE = ROOT / "workers" / "commercial_multiview" / "apple_vision_pose.swift"
SOMA77_WORKER = ROOT / "workers" / "commercial_multiview" / "soma77_pose.py"
SOMA77_MODEL = ROOT / ".cache" / "autoanim_gnm" / "gem-x" / "inputs" / "onnx" / "vitpose.onnx"
WORKER_ROOT = ROOT / ".cache" / "autoanim_gnm" / "commercial-multiview"
WORKER = WORKER_ROOT / "apple_vision_pose"
MODULE_CACHE = ROOT / ".cache" / "clang-module-cache"
DEFAULT_BODY_RUN = ROOT / ".cache" / "autoanim_gnm" / "body-provider" / "run" / "detailed-hands-fbd9784b"  # regenerated 2026-09-02 under the corrected joint map; the name carries its request hash


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=capture,
    )
    if completed.returncode:
        detail = completed.stderr[-6000:] if capture else ""
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed


def _compile_worker() -> None:
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    MODULE_CACHE.mkdir(parents=True, exist_ok=True)
    if WORKER.is_file() and WORKER.stat().st_mtime_ns >= SWIFT_SOURCE.stat().st_mtime_ns:
        return
    command = [
        "swiftc",
        "-O",
        "-framework",
        "Vision",
        "-framework",
        "ImageIO",
        "-framework",
        "CoreGraphics",
        str(SWIFT_SOURCE),
        "-o",
        str(WORKER),
    ]
    environment = dict(**__import__("os").environ)
    environment["CLANG_MODULE_CACHE_PATH"] = str(MODULE_CACHE)
    environment["SWIFT_MODULECACHE_PATH"] = str(MODULE_CACHE)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-6000:])


FRAME_JPEG_QUALITY = 2


def _extract_frames(
    video: Path,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    width: int,
    source_sha256: str,
) -> tuple[list[Path], bool]:
    destination.mkdir(parents=True, exist_ok=True)
    expected = [destination / f"{frame:06d}.jpg" for frame in range(start_frame, end_frame)]
    # Frames are cached by path alone, so a cache written under different source
    # footage or encoder settings would silently survive a change to them.
    # Record what produced these frames and re-extract when any of it differs.
    stamp = destination / "extraction.json"
    settings = {"width": width, "jpeg_quality": FRAME_JPEG_QUALITY,
                "frame_window": [start_frame, end_frame],
                "source_sha256": source_sha256}
    stale = True
    if stamp.is_file():
        try:
            stale = json.loads(stamp.read_text(encoding="utf-8")) != settings
        except json.JSONDecodeError:
            stale = True
    extracted = stale or not all(path.is_file() for path in expected)
    if extracted:
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                f"select=between(n\\,{start_frame}\\,{end_frame - 1}),scale={width}:-2",
                # Without an explicit quality the mjpeg encoder degrades as the
                # frame grows, so detector quality silently tracked resolution.
                # Measured: pinning this lifts valid joints 82.8% -> 88.2% and
                # cuts temporal rejections 33 -> 14 at width 1280.
                "-q:v",
                str(FRAME_JPEG_QUALITY),
                "-vsync",
                "0",
                "-start_number",
                str(start_frame),
                str(destination / "%06d.jpg"),
            ]
        )
    if not all(path.is_file() for path in expected):
        raise RuntimeError(f"Frame extraction is incomplete for {video}")
    stamp.write_text(json.dumps(settings, sort_keys=True), encoding="utf-8")
    return expected, extracted


def _detect(frames: list[Path], output: Path) -> None:
    if output.is_file() and len(output.read_text(encoding="utf-8").splitlines()) == len(frames):
        return
    completed = _run([str(WORKER), *(str(path) for path in frames)], capture=True)
    lines = completed.stdout.splitlines()
    if len(lines) != len(frames):
        raise RuntimeError(f"Apple Vision returned {len(lines)} frames for {len(frames)} inputs")
    output.write_text(completed.stdout, encoding="utf-8")


def _carries_head_landmarks(lines: list[str]) -> bool:
    """True when some person in a cached detection file carries the 77-point array.

    Scans the whole file rather than the first line: a frame can legitimately hold no
    people, and testing only line one would call an entire run stale because the take
    opens on an empty plate.
    """

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False
        for person in record.get("people", ()):
            if person.get("landmarks_soma77"):
                return True
    return False


def _detect_soma77(frames: list[Path], boxes: Path, output: Path) -> None:
    """Second detection pass: SOMA-77 whole-body keypoints.

    Top-down, so it needs a person box per subject. Boxes come from the Apple
    Vision pass rather than from GEM-X's bundled YOLOX, whose checkpoint is
    Human-Art-trained and authorised for non-commercial use only. Reusing boxes
    we already have keeps that asset out of the pipeline entirely.
    """

    # Cache on line count AND on the schema. A cached file written before the worker
    # emitted `landmarks_soma77` has the right number of lines and the wrong contents,
    # so the line count alone reuses it, the head solve finds no head, and the build
    # delivers a head welded to the torso -- exit 0, report green. That happened. A
    # stale cache must be re-detected, never silently accepted.
    if output.is_file():
        cached = output.read_text(encoding="utf-8").splitlines()
        if len(cached) == len(frames) and _carries_head_landmarks(cached):
            return
    completed = _run(
        [
            sys.executable,
            str(SOMA77_WORKER),
            str(SOMA77_MODEL.resolve(strict=True)),
            str(boxes),
            *(str(path) for path in frames),
        ],
        capture=True,
    )
    lines = completed.stdout.splitlines()
    if len(lines) != len(frames):
        raise RuntimeError(f"SOMA-77 returned {len(lines)} frames for {len(frames)} inputs")
    output.write_text(completed.stdout, encoding="utf-8")


# SOMA-77 indices for the five landmarks that are rigid to the skull. `Head` is the
# skeletal joint inside the skull -- NOT a nose, whatever the 19-joint contract calls it
# (workers/commercial_multiview/soma77_pose.py:80). `HeadEnd` gives the skull its long
# axis and the eyes give the lateral one, which together fix an absolute orientation;
# without them the fit is correct only up to a constant and must not be shipped.
HEAD_LANDMARK_NAMES = ("Head", "HeadEnd", "Jaw", "LeftEye", "RightEye")
_SOMA77_HEAD_INDICES = (6, 7, 8, 9, 10)
# The BALL of each foot -- `LeftToeBase`, `RightToeBase`. `ToeEnd` (71, 76) is deliberately
# excluded: it fails the same length-stability instrument that killed `HeadEnd`, at
# 12.8-63.3 % against body controls at 2.5-4.1 %, while the ball survives at 5.0-9.6 %.
# docs/FEET_MEASURED.md section 1.
_SOMA77_TOE_INDICES = (70, 75)


def _toe_landmarks(records: list[dict[str, Any]]) -> list[list[Any]]:
    """Per-frame, per-person ball-of-foot landmarks, or empty when absent.

    `[frame][person] -> (2, 3)`, left then right. Same contract as `_head_landmarks`: a
    detector without `landmarks_soma77` yields empty rows and the feet keep the previous
    behaviour, which the run report records rather than passing off as a solve.
    """

    import numpy as np

    out: list[list[Any]] = []
    for record in records:
        people: list[Any] = []
        for person in record.get("people", ()):
            marks = person.get("landmarks_soma77")
            if not marks or len(marks) <= max(_SOMA77_TOE_INDICES):
                people.append(None)
                continue
            people.append(
                np.asarray([marks[index] for index in _SOMA77_TOE_INDICES], dtype=float)
            )
        out.append(people)
    return out


def _head_landmarks(records: list[dict[str, Any]]) -> list[list[Any]]:
    """Per-frame, per-person head landmarks, or empty when the detector has none.

    Returns `[frame][person] -> (5, 3)`. A detector without `landmarks_soma77` yields
    empty rows, the head solve is not attempted, and the delivered head stays welded to
    the torso -- which the diagnostics record explicitly, because that fallback is a
    constant and not a neutral default.
    """
    import numpy as np

    out: list[list[Any]] = []
    for record in records:
        people: list[Any] = []
        for person in record.get("people", ()):
            marks = person.get("landmarks_soma77")
            if not marks or len(marks) <= max(_SOMA77_HEAD_INDICES):
                people.append(None)
                continue
            people.append(
                np.asarray([marks[index] for index in _SOMA77_HEAD_INDICES], dtype=float)
            )
        out.append(people)
    return out


def _camera_rig_from_mamma_fixture(path: Path) -> dict[str, Any]:
    """Convert only fixture calibration fields; no MAMMA runtime is imported."""

    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    cameras: dict[str, Any] = {}
    for name in ("A001", "B001", "C001", "D001"):
        camera = source["cameras"][name]
        cameras[name] = {
            "resolution": list(camera["resolution"]),
            "intrinsics": [float(item) for item in camera["intrinsics"]],
            "camera_center_world_m": [float(item) for item in camera["translation"]],
            "camera_to_world_quaternion_wxyz": [
                float(item) for item in camera["rotation_quaternion"]
            ],
        }
    return {"schema_version": "autoanim.calibrated-camera-rig/1.0", "cameras": cameras}


def _review_html(*, only_3d: bool, duration_s: float, subject_count: int) -> str:
    source_panel = "" if only_3d else """<section class=source><span class=tag>Same A001 benchmark window</span><video id=source src=source.mp4 playsinline preload=auto></video></section>"""
    columns = "1fr" if only_3d else "1fr 1fr"
    title = "Clean-room multiview · 3D only" if only_3d else "Clean-room multiview comparison"
    glbs = json.dumps([f"subject-{index:02d}.glb" for index in range(subject_count)])
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>{title}</title><style>
html,body{{margin:0;height:100%;background:#0b0e12;color:#edf2f7;font:14px system-ui}}body{{display:grid;grid-template-rows:auto 1fr auto}}header{{padding:15px 20px;border-bottom:1px solid #28303c}}h1{{font-size:18px;margin:0 0 5px}}p{{margin:0;color:#9faec0}}main{{min-height:0;display:grid;grid-template-columns:{columns};gap:1px;background:#28303c}}section{{position:relative;min-width:0;background:#11161d}}video,canvas{{display:block;width:100%;height:100%;object-fit:contain}}.tag{{position:absolute;z-index:2;top:12px;left:12px;padding:5px 8px;background:#05070acc;border-radius:4px}}footer{{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;padding:12px 20px}}button{{padding:7px 13px;background:#202936;color:white;border:1px solid #48576b;border-radius:5px}}input{{accent-color:#79aef8}}</style><script type=importmap>{{"imports":{{"three":"/api/viewer/vendor/{VIEWER_THREE_VERSION}/three.module.js","three/addons/":"/api/viewer/vendor/{VIEWER_THREE_VERSION}/addons/"}}}}</script></head><body>
<header><h1>{title}</h1><p>Apple Vision observations → our calibrated association, robust triangulation, temporal solve, IK, contacts and AutoAnim-55 retarget. No MAMMA code, weights, or outputs are consumed.</p></header><main>{source_panel}<section><span class=tag>{subject_count} reconstructed performers</span><canvas id=view></canvas></section></main><footer><button id=play>Play</button><input id=scrub type=range min=0 max=1 step=.0001 value=0><span id=time>0.000 / {duration_s:.3f} s</span></footer>
<script type=module>import * as THREE from 'three';import{{GLTFLoader}}from'three/addons/loaders/GLTFLoader.js';const canvas=document.querySelector('#view'),video=document.querySelector('#source'),play=document.querySelector('#play'),scrub=document.querySelector('#scrub'),time=document.querySelector('#time');const renderer=new THREE.WebGLRenderer({{canvas,antialias:true}}),scene=new THREE.Scene(),characters=new THREE.Group(),camera=new THREE.PerspectiveCamera(30,1,.01,100);renderer.outputColorSpace=THREE.SRGBColorSpace;scene.background=new THREE.Color(0x11161d);scene.add(characters);scene.add(new THREE.HemisphereLight(0xffffff,0x263044,2.4));const key=new THREE.DirectionalLight(0xffffff,3);key.position.set(2,4,3);scene.add(key);const grid=new THREE.GridHelper(12,24,0x3b485a,0x232c37);scene.add(grid);let actions=[],mixers=[],loaded=0,duration={duration_s:.9f},manualTime=0,playing=false,last=performance.now();const files={glbs};for(const file of files)new GLTFLoader().load(file,g=>{{characters.add(g.scene);const mixer=new THREE.AnimationMixer(g.scene),action=mixer.clipAction(g.animations[0]);action.setLoop(THREE.LoopRepeat,Infinity);action.play();action.paused=true;mixers.push(mixer);actions.push(action);duration=Math.max(duration,g.animations[0].duration);loaded++;if(loaded===files.length){{const box=new THREE.Box3().setFromObject(characters),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),span=Math.max(size.x,size.y,size.z,1.2);camera.position.set(center.x,center.y+span*.12,center.z+span*1.8);camera.lookAt(center.x,center.y+size.y*.1,center.z)}}}});function seek(t){{manualTime=Math.max(0,Math.min(duration,t));for(const [i,a]of actions.entries()){{a.time=manualTime;mixers[i].update(0)}}if(video&&Math.abs(video.currentTime-manualTime)>.04)video.currentTime=manualTime;scrub.value=duration?manualTime/duration:0;time.textContent=`${{manualTime.toFixed(3)}} / ${{duration.toFixed(3)}} s`}}function loop(now){{const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w||canvas.height!==h){{renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}}if(playing){{let next=manualTime+(now-last)/1000;if(next>=duration)next%=duration;seek(next)}}last=now;renderer.render(scene,camera);play.textContent=playing?'Pause':'Play';requestAnimationFrame(loop)}}requestAnimationFrame(loop);play.onclick=()=>{{playing=!playing;if(video)playing?video.play():video.pause();last=performance.now()}};scrub.oninput=()=>seek(Number(scrub.value)*duration);if(video)video.onended=()=>seek(0);</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--calibration-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=60)
    parser.add_argument("--end-frame", type=int, default=210)
    parser.add_argument("--detector-width", type=int, default=1280)
    parser.add_argument("--subject-count", type=int, default=2)
    parser.add_argument(
        "--detector",
        choices=("apple_vision", "soma77"),
        default="apple_vision",
        help="soma77 runs Apple Vision first for person boxes, then SOMA-77 for keypoints",
    )
    parser.add_argument("--body-run", type=Path, default=DEFAULT_BODY_RUN)
    arguments = parser.parse_args()
    if arguments.end_frame - arguments.start_frame < 2:
        raise ValueError("Comparison requires at least two frames")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _compile_worker()
    camera_names = ("A001", "B001", "C001", "D001")
    observations: list[list[dict[str, Any]]] = []
    # Head landmarks beyond the 19-joint contract, when the detector emits them. They turn
    # the delivered head from a constant welded to the torso into a solved orientation --
    # see docs/HEAD_ORIENTATION_MEASURED.md. Only SOMA-77 carries them today, so any other
    # detector keeps the previous behaviour and the run report says so.
    head_landmarks: list[list[list[Any]]] = []
    toe_landmarks: list[list[list[Any]]] = []
    input_hashes: dict[str, str] = {}
    for camera_name in camera_names:
        video = (arguments.videos / f"{camera_name}.mp4").resolve(strict=True)
        input_hashes[camera_name] = sha256(video.read_bytes()).hexdigest()
        frames, extracted = _extract_frames(
            video,
            output / "work" / "frames" / camera_name,
            start_frame=arguments.start_frame,
            end_frame=arguments.end_frame,
            width=arguments.detector_width,
            source_sha256=input_hashes[camera_name],
        )
        observation_path = output / "work" / f"{camera_name}-observations.jsonl"
        if extracted:
            # _detect caches on line count alone, which is unchanged when the
            # frames behind it are re-encoded, re-sized, or taken from a
            # different window. Re-extracted frames must invalidate detections
            # or the report would certify settings the detector never saw.
            observation_path.unlink(missing_ok=True)
        _detect(frames, observation_path)
        if arguments.detector == "soma77":
            soma_path = output / "work" / f"{camera_name}-soma77-observations.jsonl"
            if extracted:
                soma_path.unlink(missing_ok=True)
            _detect_soma77(frames, observation_path, soma_path)
            observation_path = soma_path
        records = load_observation_jsonl(observation_path)
        observations.append(records)
        head_landmarks.append(_head_landmarks(records))
        toe_landmarks.append(_toe_landmarks(records))
    rig_value = _camera_rig_from_mamma_fixture(arguments.calibration_yaml.resolve(strict=True))
    rig_path = output / "camera-rig.json"
    write_json(rig_path, rig_value)
    cameras = load_camera_rig(rig_path)
    # Every camera must contribute at least one real head observation. `any(frame ...)`
    # alone would pass on a detector that emits people with no head landmarks at all.
    solvable = all(
        any(person is not None for frame in camera for person in frame)
        for camera in head_landmarks
    )
    # A SOMA-77 run reaching here without head landmarks would build a head welded to
    # the torso and exit 0 -- a delivered constant surviving a green build, which is
    # how the welded head went unnoticed in the first place. The worker emits
    # `landmarks_soma77` unconditionally, so there is no legitimate soma77 run without
    # them: absence means stale inputs, and the build must stop rather than deliver.
    if arguments.detector == "soma77" and not solvable:
        raise RuntimeError(
            "SOMA-77 was requested but at least one camera carries no head landmarks, "
            "so the head would ship welded to the torso. Delete "
            f"{output / 'work'}/*-soma77-observations.jsonl and re-detect."
        )
    tracks, diagnostics, world_positions, raw_world_positions = reconstruct_multiview(
        cameras,
        observations,
        head_landmarks_by_camera=head_landmarks if solvable else None,
        head_landmark_names=HEAD_LANDMARK_NAMES if solvable else (),
        toe_landmarks_by_camera=toe_landmarks if solvable else None,
        subject_count=arguments.subject_count,
        sample_rate_hz=30,
    )
    body_manifest = (arguments.body_run / "neutral-body.json").resolve(strict=True)
    body_asset = (arguments.body_run / "neutral-body.npz").resolve(strict=True)
    for subject, track in enumerate(tracks):
        prefix = output / f"subject-{subject:02d}"
        write_json(prefix.with_suffix(".body-track.json"), track.as_dict())
        write_npz(
            prefix.with_suffix(".body-track.npz"),
            ticks=track.ticks,
            root_translation_m=track.root_translation_m,
            local_rotations_xyzw=track.local_rotations_xyzw,
            foot_contacts=track.foot_contacts,
            # D3: the per-performer rest this track was solved on. An npz-only
            # instrument that ran forward kinematics on `DETAILED_HUMANOID` would
            # rebuild a different body from the same rotations; with this on disk it
            # cannot do so silently.
            rest_translations_m=track.rest_translations_m,
            triangulated_world_positions_z_up_m=world_positions[subject],
            # Pre-interpolation triangulation, NaNs intact. The smoothed array
            # above cannot measure raw reconstruction noise.
            raw_triangulated_world_positions_z_up_m=raw_world_positions[subject],
        )
        export_animated_body_glb(
            prefix.with_suffix(".glb"),
            body_manifest_path=body_manifest,
            body_asset_path=body_asset,
            track=track,
            mapping_path=prefix.with_suffix(".mapping.npz"),
        )
    duration_s = (len(observations[0]) - 1) / 30.0
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{arguments.start_frame / 30:.9f}",
            "-i",
            str((arguments.videos / "A001.mp4").resolve(strict=True)),
            "-t",
            f"{duration_s:.9f}",
            "-vf",
            "scale=1280:-2",
            "-an",
            str(output / "source.mp4"),
        ]
    )
    (output / "review.html").write_text(
        _review_html(only_3d=False, duration_s=duration_s, subject_count=len(tracks)),
        encoding="utf-8",
    )
    (output / "only-3d-review.html").write_text(
        _review_html(only_3d=True, duration_s=duration_s, subject_count=len(tracks)),
        encoding="utf-8",
    )
    report = diagnostics.as_dict()
    report.update(
        {
            "test_fixture_license_scope": "MAMMA example footage/calibration: research comparison only",
            "runtime_dependencies": {
                "mamma": False,
                "mamma_outputs": False,
                "mamma_weights": False,
                "smplx_model": False,
                # Named from the observations, not hardcoded. This field read
                # "Apple Vision" on every SOMA-77 run -- a report naming the
                # wrong detector is exactly the provenance defect the SHA chain
                # exists to prevent. SOMA-77 still uses Apple Vision upstream
                # for the person boxes, so the chain is named in full.
                "detector": (
                    "Apple Vision VNDetectHumanBodyPoseRequest (person boxes) "
                    "-> NVIDIA GEM-X SOMA-77 (keypoints)"
                    if arguments.detector == "soma77"
                    else "Apple Vision VNDetectHumanBodyPoseRequest"
                ),
                "body_asset": "existing AutoAnim detailed-hands asset",
            },
            "frame_window": [arguments.start_frame, arguments.end_frame],
            # taken from the observations rather than the CLI argument, so the
            # field describes the data actually reconstructed from
            "detector_width": observations[0][0]["width"],
            # There are three detectors now, so the artifact must say which one
            # produced it. Read from the observations rather than the argument.
            "detector": observations[0][0].get("detector", "apple_vision"),
            "frame_jpeg_quality": FRAME_JPEG_QUALITY,
            "input_sha256": input_hashes,
            "joint_names": list(JOINT_NAMES),
            "production_claim": False,
            "limitations": [
                (
                    # SOMA-77 emits 77 joints including fingers and toes, but the
                    # adapter maps 17 of them -- see docs/HEAD_FEET_HANDS_PLAN.md.
                    # The delivered track has no articulated fingers either way;
                    # the reason differs and the report must say which.
                    "SOMA-77 emits finger and toe landmarks, but the adapter maps "
                    "neither, so the delivered track has no articulated fingers."
                    if arguments.detector == "soma77"
                    else "Apple Vision body observations contain no articulated finger landmarks."
                ),
                "This fixture is research-only and cannot qualify commercial capture data.",
                "Sparse IK preserves canonical body proportions instead of estimating a dense shape model.",
            ],
        }
    )
    write_json(output / "run-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
