#!/usr/bin/env python3
"""D9b's report frames: the ray each bone points along, over the footage, before and after.

INSTRUMENT ONLY, and it decides nothing. It draws the one thing this step is about.

THE PROBLEM WITH DRAWING THIS STEP. The re-aim moves a joint by 3.6-6.4 mm median. At the
subject's distance in this rig that is roughly ONE PIXEL of 1280x720, so a whole-frame
overlay of the two builds would be two identical pictures and would misinform the reader by
looking like nothing happened. Two things are done about it, and neither is a trick:

  * every cell is a 3x MAGNIFIED CROP centred on the performer's own arm, so a millimetre
    is a few pixels rather than a fraction of one. The crop box is computed from the
    CAPTURED landmarks, which are byte-identical between the two builds, so the two columns
    are cropped identically and a shift cannot come from the framing.
  * the quantity drawn is not the joint but the RAY -- the line the bone actually points
    along, from the origin the delivered file gives it, extended past its target. The
    defect is that the landmark sits OFF that line; the fix is that it sits ON it. A ray
    extended over half a metre turns a one-pixel joint error into a visibly separated line,
    and the MAGENTA tick is the perpendicular from the landmark to the ray -- the exact
    quantity B1 bands, drawn at the same scale as the picture.

Per output frame, a 2x2 panel:

    camera A001, D8c (before) | camera A001, D9b (after)
    camera D001, D8c (before) | camera D001, D9b (after)

  * YELLOW -- the captured landmarks the bones are aimed at (identical in both columns).
  * GREEN -- the delivered rig, forward-kinematicked from the GLB's own bytes through
    `d3_skeleton_gate.glb_joint_positions`, the same reader every other D9b instrument uses.
  * CYAN -- the ray from each aimed bone's own origin through its child, extended.
  * MAGENTA -- the perpendicular from the landmark to that ray. It is the error.

FRAME IDS ARE ABSOLUTE, the observations' own `frame_index`. The two ranges are chosen from
the gate's own hoisted-frame list: 129-140 is the 12-frame right-foot contact run at array
indices 69-80, and 96-104 the 10-frame run at 35-44, each with unhoisted frames either side
so the reader can see the two columns coincide exactly where the root did not move.

Nothing is re-detected, nothing is re-triangulated, nothing is written under
`artifacts/commercial-multiview-soma77/`.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_frames.py

Writes JPEG frames to `artifacts/compare/d9b-hoist/report/frames/` plus `frames.json` (the
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
import d9b_hoist_gate as gate  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d9b-hoist/report/frames"
BEFORE = ROOT / "artifacts/commercial-multiview-soma77"
AFTER = ROOT / "artifacts/compare/d9b-hoist/delivery"
FRAMES = BEFORE / "work/frames"
SUBJECT = 0
FIRST_FRAME_ID = 60
RANGES = ((127, 143), (94, 106))
PANEL_CAMERAS = ("A001", "D001")
CELL = (640, 360)
NATIVE = (1280, 720)
ZOOM = 3
CROP = (NATIVE[0] // ZOOM, NATIVE[1] // ZOOM)

# The three bones drawn: the clavicle, the upper arm and the forearm of the LEFT side,
# which is the side the crop follows. All seven root-dependent bones are measured by the
# gate; three fit in one crop without the picture becoming a diagram.
DRAWN = (
    ("LeftShoulder", "LeftUpperArm", "left_shoulder"),
    ("LeftUpperArm", "LeftLowerArm", "left_elbow"),
    ("LeftLowerArm", "LeftHand", "left_wrist"),
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
CYAN = (56, 189, 248)
MAGENTA = (236, 72, 153)
INK = (250, 250, 250)
DIM = (150, 150, 158)
SHADOW = (10, 10, 12)


def rig_to_capture(points: np.ndarray) -> np.ndarray:
    """`positions_to_body_track` does `rig = (x, z, -y)`, so the inverse is `(X, -Z, Y)`."""
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
    builds = {"D8c (before)": load(BEFORE), "D9b (after)": load(AFTER)}
    rig_index = {name: index
                 for index, name in enumerate(builds["D8c (before)"]["rig_names"])}
    joint_index = dict(cm.JOINT_INDEX)

    # The hoist, from the gate's own reader, so the number printed on the picture and the
    # number in `gate.json` are the same number.
    data = gate.read_build(BEFORE, SUBJECT)
    hoist_mm = 1e3 * np.linalg.norm(data["hoist"], axis=1)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
        cell_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
        cell_small = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    except OSError:
        font = cell_font = cell_small = ImageFont.load_default()

    ids = [value for lo, hi in RANGES for value in range(lo, hi + 1)]
    written: list[str] = []
    for frame_id in ids:
        index = frame_id - FIRST_FRAME_ID
        sheet = Image.new("RGB", (CELL[0] * 2, CELL[1] * 2 + 34), (12, 12, 14))
        header = ImageDraw.Draw(sheet)
        hoisted = hoist_mm[index] > gate.HOIST_REPORT_CUT_MM
        for column, (name, build) in enumerate(builds.items()):
            worst = 0.0
            for row, camera_name in enumerate(PANEL_CAMERAS):
                camera = cameras[camera_name]
                # THE CROP BOX COMES FROM THE CAPTURED LANDMARKS, which are byte-identical
                # between the arms, so the two columns are framed identically.
                anchor = 0.5 * (build["captured"][index, joint_index["left_shoulder"]]
                                + build["captured"][index, joint_index["left_wrist"]])
                (cu, cv), depth = camera.project(anchor)
                if depth <= 0.0:
                    cu, cv = NATIVE[0] / 2, NATIVE[1] / 2
                x0 = int(np.clip(cu - CROP[0] / 2, 0, NATIVE[0] - CROP[0]))
                y0 = int(np.clip(cv - CROP[1] / 2, 0, NATIVE[1] - CROP[1]))
                source = FRAMES / camera_name / f"{frame_id:06d}.jpg"
                full = (Image.open(source).convert("RGB") if source.exists()
                        else Image.new("RGB", NATIVE, (24, 24, 28)))
                if full.size != NATIVE:
                    full = full.resize(NATIVE, Image.LANCZOS)
                cell = full.crop((x0, y0, x0 + CROP[0], y0 + CROP[1])).resize(
                    (CROP[0] * ZOOM, CROP[1] * ZOOM), Image.LANCZOS)
                draw = ImageDraw.Draw(cell)

                def to_cell(point):
                    (u, v), d = camera.project(point)
                    if d <= 0.0:
                        return None
                    return ((u - x0) * ZOOM, (v - y0) * ZOOM)

                for first, second in RIG_BONES:
                    if first not in rig_index or second not in rig_index:
                        continue
                    a = to_cell(build["rig"][index, rig_index[first]])
                    b = to_cell(build["rig"][index, rig_index[second]])
                    if a and b:
                        draw.line((*a, *b), fill=GREEN, width=3)
                for parent, child, landmark in DRAWN:
                    origin = build["rig"][index, rig_index[parent]]
                    tip = build["rig"][index, rig_index[child]]
                    axis = tip - origin
                    length = float(np.linalg.norm(axis))
                    if length < 1e-9:
                        continue
                    axis = axis / length
                    target = build["captured"][index, joint_index[landmark]]
                    delta = target - origin
                    foot = origin + float(np.dot(delta, axis)) * axis
                    worst = max(worst, 1e3 * float(np.linalg.norm(target - foot)))
                    far = to_cell(origin + axis * (length * 1.9))
                    near = to_cell(origin - axis * (length * 0.25))
                    if far and near:
                        draw.line((*near, *far), fill=CYAN, width=2)
                    a, b = to_cell(target), to_cell(foot)
                    if a and b:
                        draw.line((*a, *b), fill=MAGENTA, width=5)
                    if a:
                        draw.ellipse((a[0] - 6, a[1] - 6, a[0] + 6, a[1] + 6), fill=YELLOW)
                label(draw, (18, 14), f"{camera_name}   frame {frame_id}   {ZOOM}x",
                      INK, cell_font)
                label(draw, (18, 48), "yellow: the captured landmark", YELLOW, cell_small)
                label(draw, (18, 74), "cyan: the ray the bone points along", CYAN,
                      cell_small)
                label(draw, (18, 100), "magenta: how far the landmark is off it", MAGENTA,
                      cell_small)
                label(draw, (18, 126), "green: the delivered rig", GREEN, cell_small)
                sheet.paste(cell.resize(CELL, Image.LANCZOS),
                            (column * CELL[0], 34 + row * CELL[1]))
            label(header, (14 + column * CELL[0], 8),
                  f"{name}    root moved {hoist_mm[index]:5.2f} mm    "
                  f"worst miss drawn {worst:5.2f} mm",
                  MAGENTA if (hoisted and column == 0) else (INK if hoisted else DIM), font)
        name = out / f"{frame_id:06d}.jpg"
        sheet.save(name, quality=88, optimize=True)
        written.append(name.name)
        print(f"wrote {name.name}  hoist {hoist_mm[index]:.2f} mm")

    (out / "frames.json").write_text(json.dumps({
        "what": ("D9b report frames. Performer 0, cameras A001 and D001, the shipped D8c "
                 "build beside the D9b build, 3x magnified on the left arm, with the "
                 "captured landmark in yellow, the ray each bone points along in cyan and "
                 "the perpendicular from the landmark to that ray in magenta."),
        "subject": SUBJECT,
        "cameras": list(PANEL_CAMERAS),
        "ranges": [list(pair) for pair in RANGES],
        "frame_ids": ids,
        "files": written,
        "fps": 30,
        "hoist_mm": [round(float(hoist_mm[i - FIRST_FRAME_ID]), 3) for i in ids],
        "why_magnified": ("the re-aim moves a joint 3.6-6.4 mm median, about one pixel of "
                          "1280x720 at this distance. The crop box is computed from the "
                          "captured landmarks, which are byte-identical between the two "
                          "builds, so the columns are framed identically."),
        "note": ("JPEG frames for a frame PLAYER, not an animated image: the artifact "
                 "viewer blocks <video> from data: and blob: URLs (CLAUDE.md)."),
    }, indent=1), encoding="utf-8")

    listing = out / "sequence.txt"
    listing.write_text("".join(f"file '{name}'\nduration 0.0333333\n"
                               for name in written)
                       + f"file '{written[-1]}'\n", encoding="utf-8")
    movie = out.parent / "d9b-hoist-reaim-before-and-after.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-vsync", "cfr", "-r", "10", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "18", str(movie)],
        capture_output=True, text=True)
    print(result.stderr[-1500:] if result.returncode else f"wrote {movie}")
    print(f"\n{len(written)} frames in {out}")
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
