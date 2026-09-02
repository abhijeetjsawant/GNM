#!/usr/bin/env python3
"""D2: a PICTURE of the clavicle jitter, because a still cannot show it.

The gate measures the regression -- clavicle-chain frames above a human's peak joint rate
go 32 -> 48 and 11 -> 49 -- but a number that says "the arm snaps" is not the same as
seeing it snap, and the hand close-ups in `hand_closeup.py` are single frames and so are
structurally blind to it.

WHAT THIS DOES. Finds the worst single-frame clavicle step in the D2 rebuild, renders the
delivered build and the rebuild over a window around it through
`tools/swap-harness/camera_overlay.py` -- the real overlay instrument, wrapped, never
re-implemented, and the one that SETS THE SCENE FPS BEFORE THE IMPORT (glTF keyframe times
are seconds and Blender converts them with the scene fps; a 30 fps motion in the 24 fps
factory scene runs 25 % fast and freezes after the last converted frame, which reads as a
placement error and is a timebase error -- CLAUDE.md). Then crops to the arm, labels each
frame with the step angle it is about to take, and writes a contact sheet, a tight zoom on
the spike, and a side-by-side mp4.

Run:  PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2_jitter_sheet.py
Writes: artifacts/compare/d2-clavicle/jitter/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from autoanim_gnm.body import DETAILED_HUMANOID  # noqa: E402

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
OVERLAY = ROOT / "tools/swap-harness/camera_overlay.py"
DELIVERED = ROOT / "artifacts/commercial-multiview-soma77"
REBUILD = ROOT / "artifacts/compare/d2-clavicle/delivery"
OUT = ROOT / "artifacts/compare/d2-clavicle/jitter"
CAMERA = "A001"
SUBJECT = 1                 # the worse of the two: a 160.6 deg single-frame step
JOINT = "LeftShoulder"
WINDOW = range(40, 71)
CEILING_DEG = 800.0 / 30.0  # a human joint's peak rate at this take's 30 fps


def step_angles(quaternions):
    a, b = quaternions[:-1], quaternions[1:]
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), -1.0, 1.0)))


def render(build: Path, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for frame in WINDOW:
        target = folder / f"f{frame:03d}.jpg"
        if target.is_file():
            continue
        subprocess.run(
            [BLENDER, "--background", "--python", str(OVERLAY), "--", str(target),
             CAMERA, str(frame), "noflip", str(build)],
            check=True, capture_output=True, cwd=ROOT)


def tile(path: Path, crop, scale, caption, hot, header=34, size=22):
    image = Image.open(path).crop(crop)
    image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    pad = Image.new("RGB", (image.width, image.height + header), (232, 234, 236))
    pad.paste(image, (0, header))
    draw = ImageDraw.Draw(pad)
    draw.text((8, 6), caption, fill=(190, 30, 30) if hot else (20, 20, 20),
              font=ImageFont.load_default(size))
    if hot:
        draw.rectangle([0, 0, pad.width - 1, pad.height - 1], outline=(190, 30, 30), width=4)
    return pad


def sheet(rows, columns, title, subtitle, path):
    width, height = rows[0][0].width, rows[0][0].height
    top = 64 if subtitle else 40
    out = Image.new("RGB", (columns * width, len(rows) * height + top), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill=(20, 20, 20), font=ImageFont.load_default(22))
    if subtitle:
        draw.text((8, 34), subtitle, fill=(70, 70, 70), font=ImageFont.load_default(18))
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            out.paste(image, (c * width, top + r * height))
    out.save(path)
    print(f"wrote {path}  {out.size}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = DETAILED_HUMANOID.index(JOINT)
    steps = {}
    for label, build in (("before", DELIVERED), ("after", REBUILD)):
        render(build, OUT / label)
        track = np.load(build / f"subject-{SUBJECT:02d}.body-track.npz")
        steps[label] = step_angles(track["local_rotations_xyzw"][:, index])
    worst = int(np.argmax(steps["after"]))
    print(f"worst {JOINT} step in the rebuild: {steps['after'][worst]:.2f} deg "
          f"at track frame {worst} -> {worst + 1} "
          f"(delivered build at the same frame: {steps['before'][worst]:.2f} deg)")

    frames = [f for f in WINDOW if worst - 4 <= f <= worst + 6]
    rows = [[tile(OUT / w / f"f{f:03d}.jpg", (340, 165, 600, 395), 2,
                  f"f{f}  step {steps[w][f]:5.1f} deg", steps[w][f] > CEILING_DEG,
                  header=26, size=16) for f in frames]
            for w in ("before", "after")]
    sheet(rows, len(frames),
          f"Subject {SUBJECT}, camera {CAMERA}, {JOINT}. The "
          f"{steps['after'][worst]:.1f} deg single-frame step is the red tile "
          f"({worst} -> {worst + 1}).",
          "Top: delivered build. Bottom: D2 rebuild. Same frames, same camera, scene fps "
          "set to 30 before import.",
          OUT / f"jitter-contact-sheet-subject{SUBJECT:02d}-{JOINT}.png")

    near = [worst - 1, worst, worst + 1, worst + 2]
    rows = [[tile(OUT / w / f"f{f:03d}.jpg", (430, 175, 580, 300), 5,
                  f"{w.upper()}  f{f}   next step {steps[w][f]:.1f} deg",
                  steps[w][f] > CEILING_DEG) for f in near]
            for w in ("before", "after")]
    sheet(rows, len(near),
          f"Subject {SUBJECT}, {JOINT} spike. Top: delivered. Bottom: D2 rebuild. "
          f"Frames {near[0]}-{near[-1]}, camera {CAMERA}.", "",
          OUT / f"jitter-zoom-subject{SUBJECT:02d}-f{near[0]:03d}-{near[-1]:03d}.png")

    pairs = OUT / "pairs"
    pairs.mkdir(exist_ok=True)
    for frame in WINDOW:
        left, right = (tile(OUT / w / f"f{frame:03d}.jpg", (400, 170, 600, 340), 4,
                            f"{w.upper()}   frame {frame}", False, header=30, size=20)
                       for w in ("before", "after"))
        canvas = Image.new("RGB", (left.width * 2, left.height), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        canvas.save(pairs / f"p{frame:03d}.png")
    video = OUT / f"jitter-{CAMERA}-subject{SUBJECT:02d}-before-vs-after.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6",
         "-start_number", str(WINDOW.start), "-i", str(pairs / "p%03d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(video)], check=True)
    for stale in pairs.glob("*.png"):
        stale.unlink()
    pairs.rmdir()
    print(f"wrote {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
