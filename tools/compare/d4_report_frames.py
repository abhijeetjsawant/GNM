"""D4's report frames: the delivered body drawn over the footage, the rig beside MHR.

Uses the committed silhouette instrument's own rasteriser and camera rig, so what is drawn here
is exactly what B1 scored -- a report picture that could disagree with the band would be worse
than no picture. Each panel is one camera, one frame: the rig delivery's outline on the left, the
MHR delivery's on the right, both over the same source frame, with the SAM2 mask outline drawn
under them so the reader can see what is being covered.

    .venv/bin/python tools/compare/d4_report_frames.py --out artifacts/compare/d4-body/report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
import silhouette as si  # noqa: E402

D4 = ROOT / "artifacts/compare/d4-body"
FRAMES_DIR = D4 / "delivery/work/frames"
START_FRAME = 60
ARMS = {"the rig (before)": ROOT / "artifacts/compare/i6/delivered-mesh.npz",
        "MHR (after)": D4 / "work-delivery/delivered-mesh.npz"}
# BGR, and the lane's validated roles: BLUE is ours (the MHR delivery), AQUA is the alternative
# (the rig delivery this replaces). Orange is reserved for MAMMA everywhere in this lane and
# MAMMA is not drawn here at all, so it must not appear.
COLOURS = {"the rig (before)": (225, 225, 90), "MHR (after)": (255, 150, 40)}
MASK_COLOUR = (120, 120, 120)
WIDTH, HEIGHT = 1280, 720
PANEL_WIDTH = 640
JPEG_QUALITY = 42
EVERY = 6


def outline(image: np.ndarray, binary: np.ndarray, colour, thickness: int = 2) -> None:
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colour, thickness)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", default="A001,D001")
    parser.add_argument("--mp4-camera", default="A001")
    arguments = parser.parse_args()
    out = arguments.out.resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cameras = tuple(arguments.cameras.split(","))

    rig = {c.name: c for c in si.load_camera_rig(si.RIG_PATH)}
    scaled = {name: camera.scaled(WIDTH, HEIGHT) for name, camera in rig.items()}
    si.WORK = ROOT / "artifacts/compare/i6"
    masks = si.MaskStore(4, si.CAMERAS)
    mask_shape = (si.NATIVE[0] // 4, si.NATIVE[1] // 4)
    meshes = {name: dict(np.load(path)) for name, path in ARMS.items()}
    tracklet = si.json.load(open(ROOT / "artifacts/compare/i6/silhouette-1920x1080-A001.json")
                            )["identity"]["our_subject_to_mask_tracklet"]

    written, mp4_dir = [], out / "mp4-frames"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        for frame in range(si.FRAMES):
            source = cv2.imread(str(FRAMES_DIR / camera / f"{START_FRAME + frame:06d}.jpg"))
            if source is None:
                raise SystemExit(f"missing source frame {camera} {frame}")
            panels = []
            for name, mesh in meshes.items():
                panel = source.copy()
                for subject in (0, 1):
                    mask = masks.get(camera, int(tracklet[camera][f"subject_{subject:02d}"]))[frame]
                    outline(panel, cv2.resize(mask.astype(np.uint8), (WIDTH, HEIGHT),
                                              interpolation=cv2.INTER_NEAREST),
                            MASK_COLOUR, 1)
                    verts = mesh[f"verts_{subject:02d}"]
                    if frame < verts.shape[0]:
                        render = si.rasterise(verts[frame], mesh[f"faces_{subject:02d}"],
                                              scaled[camera], (WIDTH, HEIGHT))
                        outline(panel, render, COLOURS[name], 2)
                cv2.rectangle(panel, (0, 0), (WIDTH, 44), (20, 20, 20), -1)
                cv2.putText(panel, f"{camera}  {name}", (14, 31), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, COLOURS[name], 2, cv2.LINE_AA)
                panels.append(cv2.resize(panel, (PANEL_WIDTH, PANEL_WIDTH * HEIGHT // WIDTH)))
            pair = np.hstack(panels)
            if camera == arguments.mp4_camera:
                cv2.imwrite(str(mp4_dir / f"{frame:06d}.png"), pair)
            if frame % EVERY == 0:
                path = out / "frames" / f"{camera}-{frame:06d}.jpg"
                cv2.imwrite(str(path), pair, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                written.append(path)
        print(f"{camera}: done", flush=True)

    mp4 = out / f"d4-body-{arguments.mp4_camera}.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", "30",
                    "-i", str(mp4_dir / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(mp4)], check=True)
    total = sum(p.stat().st_size for p in written)
    print(f"WROTE {len(written)} jpegs ({total / 1024:.0f} kB) and {mp4} "
          f"({mp4.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
