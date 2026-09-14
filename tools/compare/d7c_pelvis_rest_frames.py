#!/usr/bin/env python3
"""D7c's report frames: where the pelvis points, over the footage, before and after.

INSTRUMENT ONLY, and it decides nothing. It draws the one thing this step is about.

WHAT IS DRAWN, AND WHY IT IS THE RIGHT PICTURE. The defect is that the delivered pelvis was
pitched about the hip line by a CONSTANT belonging to SOMA-77's rest, not this rig's. The
pelvis's own up axis is the rig's `Hips` -> `Spine` direction, and the thing it is supposed to
point at is the captured `Spine1` landmark seen from the hip midpoint. So each cell draws:

  * YELLOW -- the captured landmarks: the two hip joints and `Spine1`. IDENTICAL in both
    columns, because this is a converter-only change on byte-identical landmarks. If the
    yellow ever moved between columns the change did more than refit the pelvis.
  * GREEN -- the delivered rig, forward-kinematicked from the GLB'S OWN BYTES through
    `d3_skeleton_gate.glb_joint_positions`, the reader every other D7c instrument uses.
  * CYAN -- the pelvis's own up axis, from the hip midpoint through the rig's `Spine`,
    EXTENDED past it. Before, it misses the yellow `Spine1` by about 9 degrees; after, by
    1.4 and 2.1 degrees. That angle is the whole step.
  * MAGENTA -- the gap between the delivered `Spine` joint and where the pelvis axis would
    put it if it aimed at the captured point. It is the 30 mm this step moves.

NO MAGNIFICATION IS USED and none is needed: 9 degrees of pelvis rotation and 30 mm at `Spine`
are visible at native scale, unlike D9b's one-pixel re-aim. The frames are whole and the reader
can see the performer.

PERFORMER 1 IS DRAWN because it is the one the lever guard acts on: the nine-frame demoted run
38-46 (frame ids 98-106) is included with unaffected frames either side, so the reader can see
what the guard does and what it does not.

Nothing is re-detected, nothing is re-triangulated, nothing is written under
`artifacts/commercial-multiview-soma77/`.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_frames.py

Writes JPEG frames to `artifacts/compare/d7c-pelvis-rest/report/frames/` plus `frames.json`
(the frame-player script the report page consumes -- JPEG frames in a JSON list, never an
animated image: the viewer blocks `<video>` from data: and blob: URLs) and a full-rate mp4
beside them for SendUserFile.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(f"PYTHONPATH trap: {autoanim_gnm.__file__}")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402
import captured_limb_stability as stability  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7c-pelvis-rest/report/frames"
BEFORE = ROOT / "artifacts/commercial-multiview-soma77"
# The Spine1 array is not in a delivered `.npz`, and the SHIPPED build has no
# `converter-inputs/` dump. The hygiene rebuild IS the shipped build (8 of 8 byte-identical)
# and its watcher dumped the inputs, so it supplies the same landmarks, byte for byte.
SPINE_SOURCE = {"D9b (before)": ROOT / "artifacts/compare/d7c-pelvis-rest/delivery-hygiene",
                "D7c (after)": ROOT / "artifacts/compare/d7c-pelvis-rest/delivery"}
AFTER = ROOT / "artifacts/compare/d7c-pelvis-rest/delivery"
FRAMES = BEFORE / "work/frames"
SUBJECT = 1
FIRST_FRAME_ID = 60
RANGES = ((90, 110),)                     # array 30-50: the demoted run 38-46 with neighbours
PANEL_CAMERAS = ("A001", "D001")
CELL = (640, 360)
NATIVE = (1280, 720)
DEMOTED = (24, 25, 28, 29, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45, 46,
           65, 68, 70, 78, 79, 80, 81, 140, 141, 144, 145, 147, 148, 149)

RIG_BONES = (
    ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "UpperChest"), ("UpperChest", "Neck"),
    ("Neck", "Head"),
    ("UpperChest", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftLowerArm"), ("LeftLowerArm", "LeftHand"),
    ("UpperChest", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightLowerArm"), ("RightLowerArm", "RightHand"),
    ("Hips", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToes"),
    ("Hips", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"), ("RightFoot", "RightToes"),
)
YELLOW, GREEN, CYAN, MAGENTA = (255, 214, 10), (52, 211, 153), (56, 189, 248), (236, 72, 153)
INK, DIM, SHADOW = (250, 250, 250), (150, 150, 158), (10, 10, 12)


def rig_to_capture(points: np.ndarray) -> np.ndarray:
    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def load(directory: Path, spine_from: Path) -> dict:
    with np.load(directory / f"subject-{SUBJECT:02d}.body-track.npz") as archive:
        captured = np.asarray(archive["triangulated_world_positions_z_up_m"], np.float64)
    names, positions, _ = d3.glb_joint_positions(directory / f"subject-{SUBJECT:02d}.glb")
    with np.load(spine_from / f"converter-inputs/call-{SUBJECT:02d}.npz") as archive:
        spine = np.asarray(archive["spine_world_z_up_m"], np.float64)
    return {"captured": captured, "spine": spine, "names": names,
            "rig": rig_to_capture(np.asarray(positions, np.float64))}


def label(draw, xy, text, colour, font) -> None:
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((xy[0] + dx, xy[1] + dy), text, fill=SHADOW, font=font)
    draw.text(xy, text, fill=colour, font=font)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--page", action="store_true",
                        help="render the PAGE-SIZED set: half-scale sheets at q38 and every "
                             "second frame. 'The Solve So Far' is at 9.27 of its ~9.5 MB "
                             "cap, so a v9 tab has about 0.2 MB of headroom and the "
                             "full-quality set below is 1.9 MB. This one is what a tab can "
                             "carry; the full set and the mp4 stay for SendUserFile.")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cameras = dict(zip(stability.CAMERAS, stability.cameras_scaled()))
    builds = {label: load(path, SPINE_SOURCE[label])
              for label, path in (("D9b (before)", BEFORE), ("D7c (after)", AFTER))}
    if not np.array_equal(builds["D9b (before)"]["spine"], builds["D7c (after)"]["spine"]):
        raise SystemExit("the two builds do not share their Spine1 array")
    index_of = {n: i for i, n in enumerate(builds["D9b (before)"]["names"])}
    joint = dict(cm.JOINT_INDEX)
    if not np.array_equal(builds["D9b (before)"]["captured"], builds["D7c (after)"]["captured"]):
        raise SystemExit("the two builds do not share their captured landmarks")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
        cell_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
    except OSError:
        font = cell_font = small = ImageFont.load_default()

    ids = [v for lo, hi in RANGES for v in range(lo, hi + 1)]
    if args.page:
        ids = ids[::2]
    written, angles = [], []
    for frame_id in ids:
        i = frame_id - FIRST_FRAME_ID
        sheet = Image.new("RGB", (CELL[0] * 2, CELL[1] * 2 + 34), (12, 12, 14))
        header = ImageDraw.Draw(sheet)
        demoted = i in DEMOTED
        per_column = []
        for column, (name, build) in enumerate(builds.items()):
            hip_mid = 0.5 * (build["captured"][i, joint["left_hip"]]
                             + build["captured"][i, joint["right_hip"]])
            spine_point = build["spine"][i]
            pelvis_up = build["rig"][i, index_of["Spine"]] - hip_mid
            length = float(np.linalg.norm(pelvis_up))
            unit = pelvis_up / max(length, 1e-9)
            wanted = spine_point - hip_mid
            angle = float(np.degrees(np.arccos(np.clip(
                np.dot(unit, wanted / max(np.linalg.norm(wanted), 1e-9)), -1.0, 1.0))))
            per_column.append(angle)
            for row, camera_name in enumerate(PANEL_CAMERAS):
                camera = cameras[camera_name]
                source = FRAMES / camera_name / f"{frame_id:06d}.jpg"
                full = (Image.open(source).convert("RGB") if source.exists()
                        else Image.new("RGB", NATIVE, (24, 24, 28)))
                if full.size != NATIVE:
                    full = full.resize(NATIVE, Image.LANCZOS)
                cell = full.copy()
                draw = ImageDraw.Draw(cell)

                def project(point):
                    (u, v), depth = camera.project(point)
                    return (u, v) if depth > 0.0 else None

                for first, second in RIG_BONES:
                    if first not in index_of or second not in index_of:
                        continue
                    a, b = project(build["rig"][i, index_of[first]]), project(
                        build["rig"][i, index_of[second]])
                    if a and b:
                        draw.line((*a, *b), fill=GREEN, width=3)
                near, far = project(hip_mid), project(hip_mid + unit * length * 2.1)
                if near and far:
                    draw.line((*near, *far), fill=CYAN, width=4)
                for landmark in ("left_hip", "right_hip"):
                    point = project(build["captured"][i, joint[landmark]])
                    if point:
                        draw.ellipse((point[0] - 7, point[1] - 7, point[0] + 7,
                                      point[1] + 7), fill=YELLOW)
                point = project(spine_point)
                if point:
                    draw.ellipse((point[0] - 9, point[1] - 9, point[0] + 9, point[1] + 9),
                                 outline=YELLOW, width=4)
                a = project(build["rig"][i, index_of["Spine"]])
                b = project(hip_mid + wanted / max(np.linalg.norm(wanted), 1e-9) * length)
                if a and b:
                    draw.line((*a, *b), fill=MAGENTA, width=5)
                label(draw, (18, 12), f"{camera_name}   frame {frame_id}", INK, cell_font)
                label(draw, (18, 44), "yellow: the captured hips, and Spine1 (ringed)",
                      YELLOW, small)
                label(draw, (18, 68), "cyan: where the delivered pelvis points", CYAN, small)
                label(draw, (18, 92), "magenta: how far that leaves `Spine` from its target",
                      MAGENTA, small)
                label(draw, (18, 116), "green: the delivered rig, from the GLB's own bytes",
                      GREEN, small)
                sheet.paste(cell.resize(CELL, Image.LANCZOS),
                            (column * CELL[0], 34 + row * CELL[1]))  # noqa: E501
            label(header, (14 + column * CELL[0], 8),
                  f"{name}    pelvis axis vs the captured Spine1: {angle:5.2f} deg"
                  + ("    [the guard demoted this frame]" if demoted and column else ""),
                  MAGENTA if demoted and column else (INK if column else DIM), font)
        angles.append([round(v, 3) for v in per_column])
        name = out / f"{frame_id:06d}.jpg"
        if args.page:
            sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)
        sheet.save(name, quality=38 if args.page else 42, optimize=True)
        written.append(name.name)
        print(f"wrote {name.name}  before {per_column[0]:5.2f} deg  after {per_column[1]:5.2f} deg"
              + ("  DEMOTED" if demoted else ""))

    (out / "frames.json").write_text(json.dumps({
        "what": ("D7c report frames. Performer 1, cameras A001 and D001, the shipped D9b "
                 "build beside the D7c build. Yellow: the captured hips and `Spine1`. Cyan: "
                 "where the delivered pelvis points. Magenta: how far that leaves the rig's "
                 "`Spine` from the captured direction. Green: the delivered rig, from the "
                 "GLB's own bytes."),
        "subject": SUBJECT, "cameras": list(PANEL_CAMERAS),
        "ranges": [list(pair) for pair in RANGES], "frame_ids": ids, "files": written,
        "fps": 10,
        "pelvis_axis_vs_captured_spine1_deg": angles,
        "demoted_frame_ids": [f for f in ids if (f - FIRST_FRAME_ID) in DEMOTED],
        "why_no_magnification": ("9 degrees of pelvis rotation and 30 mm at `Spine` are "
                                 "visible at native scale, unlike D9b's one-pixel re-aim"),
        "the_landmarks_are_identical_between_columns": True,
    }, indent=1), encoding="utf-8")
    mp4 = out.parent / "d7c-pelvis-rest.mp4"
    try:
        if args.page:
            raise OSError("the page-sized set does not carry an mp4")
        subprocess.run(["ffmpeg", "-y", "-framerate", "10", "-pattern_type", "glob",
                        "-i", str(out / "*.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", str(mp4)], check=True, capture_output=True)
        print(f"wrote {mp4}")
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"mp4 not written ({error}); the frame player does not need it")
    print(f"\n{len(written)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
