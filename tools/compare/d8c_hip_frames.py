#!/usr/bin/env python3
"""D8c's report frames: the captured hips and the delivered rig over the footage, before and after.

INSTRUMENT ONLY, and it decides nothing. It draws what the two builds put on the person so a
reader can see the two failures rather than take them on a number.

Per output frame, a 2x2 panel:

    camera A001, D8b (before) | camera A001, D8c (after)
    camera D001, D8b (before) | camera D001, D8c (after)

and in each cell, over that camera's own extracted frame:

  * YELLOW -- the CAPTURED landmark skeleton, the delivered
    `triangulated_world_positions_z_up_m`, projected into that camera. This is what the
    length rule acts on.
  * GREEN -- the DELIVERED RIG, forward-kinematicked from the GLB's own bytes (the same
    `d3_skeleton_gate.glb_joint_positions` every other D8c instrument reads) and converted
    back through the converter's own change of basis. This is what ships.
  * a MAGENTA hip line with its length in millimetres, because the whole step is that one
    number, and the performer's own take median beside it.

FRAME IDS ARE ABSOLUTE, the observations' own `frame_index`, and the extracted frames are
named by them. The two ranges are the card's: 80-125 covers the one-hip run (84-86) and the
inward collapse (110-119) with a run-up either side; 155-172 covers the outward stretch
(158-168).

Nothing is re-detected, nothing is re-triangulated, nothing is written under
`artifacts/commercial-multiview-soma77/`.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_frames.py

Writes JPEG frames to `artifacts/compare/d8c-hip/frames/` plus `frames.json` (the
frame-player script the report page consumes: JPEG frames in a JSON list, never an animated
image -- the viewer blocks `<video>` from data: and blob: URLs) and a full-rate mp4 beside
them for the user.
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
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402
import captured_limb_stability as stability  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d8c-hip/frames"
D8B = ROOT / "artifacts/commercial-multiview-soma77"
D8C = ROOT / "artifacts/compare/d8c-hip/delivery"
FRAMES = D8B / "work/frames"
SUBJECT = 1
FIRST_FRAME_ID = 60
RANGES = ((80, 125), (155, 172))
PANEL_CAMERAS = ("A001", "D001")
CELL = (640, 360)
NATIVE = (1280, 720)

CAPTURED_BONES = (
    ("neck", "root"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("root", "left_hip"), ("root", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("neck", "nose"),
)
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
YELLOW = (255, 214, 10)
GREEN = (52, 211, 153)
MAGENTA = (236, 72, 153)
INK = (250, 250, 250)
SHADOW = (10, 10, 12)


def rig_to_capture(points: np.ndarray) -> np.ndarray:
    """The converter's change of basis, inverted.

    `positions_to_body_track` does `rig = (x, z, -y)` on capture Z-up points, so the inverse
    is `capture = (X, -Z, Y)`. Written here rather than eyeballed because getting it wrong
    draws a plausible-looking skeleton in the wrong place.
    """

    out = np.empty_like(points)
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def load(directory: Path) -> dict:
    with np.load(directory / f"subject-{SUBJECT:02d}.body-track.npz") as archive:
        smoothed = np.asarray(archive["triangulated_world_positions_z_up_m"],
                              dtype=np.float64)
    names, positions, _rest = d3.glb_joint_positions(
        directory / f"subject-{SUBJECT:02d}.glb")
    return {"captured": smoothed, "rig_names": names,
            "rig": rig_to_capture(np.asarray(positions, dtype=np.float64))}


def draw_skeleton(draw, camera, points, bones, index_of, colour, width) -> None:
    for first, second in bones:
        if first not in index_of or second not in index_of:
            continue
        a, b = points[index_of[first]], points[index_of[second]]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        (ua, va), depth_a = camera.project(a)
        (ub, vb), depth_b = camera.project(b)
        if depth_a <= 0.0 or depth_b <= 0.0:
            continue
        draw.line((ua, va, ub, vb), fill=colour, width=width)
    for point in points:
        if not np.isfinite(point).all():
            continue
        (u, v), depth = camera.project(point)
        if depth <= 0.0:
            continue
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=colour)


def label(draw, xy, text, colour, font) -> None:
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=SHADOW, font=font)
    draw.text((x, y), text, fill=colour, font=font)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cameras = {name: camera for name, camera
               in zip(stability.CAMERAS, stability.cameras_scaled())}
    builds = {"D8b (before)": load(D8B), "D8c (after)": load(D8C)}
    rig_index = {name: index
                 for index, name in enumerate(builds["D8b (before)"]["rig_names"])}
    joint_index = dict(cm.JOINT_INDEX)

    medians = {}
    for name, build in builds.items():
        series = np.linalg.norm(
            build["captured"][:, joint_index["left_hip"]]
            - build["captured"][:, joint_index["right_hip"]], axis=1) * 1000.0
        medians[name] = float(np.median(series[np.isfinite(series)]))

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
        # The cell is rendered at 1280x720 and pasted at half that, so its own type has to
        # be drawn at twice the size it should read at.
        cell_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
        cell_small = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    except OSError:
        font = cell_font = cell_small = ImageFont.load_default()

    ids = [value for lo, hi in RANGES for value in range(lo, hi + 1)]
    written: list[str] = []
    for frame_id in ids:
        index = frame_id - FIRST_FRAME_ID
        sheet = Image.new("RGB", (CELL[0] * 2, CELL[1] * 2 + 34), (12, 12, 14))
        header = ImageDraw.Draw(sheet)
        for column, (name, build) in enumerate(builds.items()):
            width_mm = float(np.linalg.norm(
                build["captured"][index, joint_index["left_hip"]]
                - build["captured"][index, joint_index["right_hip"]])) * 1000.0
            fraction = width_mm / medians[name] - 1.0
            label(header, (14 + column * CELL[0], 8),
                  f"{name}    hip line {width_mm:6.1f} mm "
                  f"({fraction:+.1%} of his own {medians[name]:.0f} mm)",
                  MAGENTA if abs(fraction) > 0.15 else INK, font)
            for row, camera_name in enumerate(PANEL_CAMERAS):
                source = FRAMES / camera_name / f"{frame_id:06d}.jpg"
                cell = (Image.open(source).convert("RGB") if source.exists()
                        else Image.new("RGB", NATIVE, (24, 24, 28)))
                if cell.size != NATIVE:
                    cell = cell.resize(NATIVE, Image.LANCZOS)
                draw = ImageDraw.Draw(cell)
                camera = cameras[camera_name]
                draw_skeleton(draw, camera, build["captured"][index], CAPTURED_BONES,
                              joint_index, YELLOW, 3)
                draw_skeleton(draw, camera, build["rig"][index], RIG_BONES,
                              rig_index, GREEN, 2)
                left = build["captured"][index, joint_index["left_hip"]]
                right = build["captured"][index, joint_index["right_hip"]]
                if np.isfinite(left).all() and np.isfinite(right).all():
                    (ua, va), da = camera.project(left)
                    (ub, vb), db = camera.project(right)
                    if da > 0.0 and db > 0.0:
                        draw.line((ua, va, ub, vb), fill=MAGENTA, width=8)
                        for u, v in ((ua, va), (ub, vb)):
                            draw.ellipse((u - 7, v - 7, u + 7, v + 7), fill=MAGENTA)
                label(draw, (18, 16), f"{camera_name}   frame {frame_id}", INK, cell_font)
                label(draw, (18, 54), "yellow: captured landmarks", YELLOW, cell_small)
                label(draw, (18, 84), "green: the delivered rig", GREEN, cell_small)
                label(draw, (18, 114), "magenta: the hip line", MAGENTA, cell_small)
                sheet.paste(cell.resize(CELL, Image.LANCZOS),
                            (column * CELL[0], 34 + row * CELL[1]))
        name = out / f"{frame_id:06d}.jpg"
        sheet.save(name, quality=88, optimize=True)
        written.append(name.name)
        print(f"wrote {name.name}")

    (out / "frames.json").write_text(json.dumps({
        "what": ("D8c report frames. Performer 1, cameras A001 and D001, the D8b build "
                 "beside the D8c build, with the captured landmark skeleton in yellow, the "
                 "delivered rig in green and the hip line in magenta."),
        "subject": SUBJECT,
        "cameras": list(PANEL_CAMERAS),
        "ranges": [list(pair) for pair in RANGES],
        "frame_ids": ids,
        "files": written,
        "fps": 30,
        "note": ("JPEG frames for a frame PLAYER, not an animated image: the artifact "
                 "viewer blocks <video> from data: and blob: URLs (CLAUDE.md)."),
    }, indent=1), encoding="utf-8")

    listing = out / "sequence.txt"
    listing.write_text("".join(f"file '{name}'\nduration 0.0333333\n"
                               for name in written)
                       + f"file '{written[-1]}'\n", encoding="utf-8")
    movie = out.parent / "d8c-hip-line-before-and-after.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-vsync", "cfr", "-r", "30", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "18", str(movie)],
        capture_output=True, text=True)
    print(result.stderr[-1500:] if result.returncode else f"wrote {movie}")
    print(f"\n{len(written)} frames in {out}")
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
