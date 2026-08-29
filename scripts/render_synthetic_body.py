#!/usr/bin/env python3
"""Render the skinned SOMA body through the real calibrated rig.

Arm (i) of `docs/BATTLE2_SYNTHETIC_TRUTH_FIXTURE.md`: put a human whose joints we
know exactly in front of our own cameras, run SOMA-77 on the pictures, and compare
its 2D against the projected truth. That is the only instrument this project has
that can see the *cross-view-coherent* part of the detector's error -- a per-joint
bias, or a whole limb displaced -- which epipolar disagreement and reprojection
residual are both structurally blind to.

Deliberately **not** Blender. The design's §3 camera replication is a real risk: a
flipped principal-point sign is invisible at frame centre and worth 28 px at the
corners, and the conversion would have to be verified before anything downstream
could be trusted. Projecting with `CalibratedCamera.project` -- the same function
the pipeline uses -- removes that risk entirely. The cost is a flat-shaded image
rather than a photoreal one, which is a domain-gap question this script measures
rather than assumes: it reports detection rate and confidence alongside error, and
if the detector does not behave the error numbers mean nothing.

Painter's algorithm with backface culling, Lambertian shading, one key light and an
ambient term. 36k triangles per view.

    python scripts/render_synthetic_body.py --frames 8 --out artifacts/arm-i
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers" / "commercial_multiview"))

from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.soma_body_mesh import SomaBodyMesh
from soma77_pose import SOMA77_TO_AUTOANIM

CAMERAS = ("A001", "B001", "C001", "D001")
WIDTH, HEIGHT = 1280, 720
# +90 degrees about X, determinant +1: SOMA is Y-up, the rig world is Z-up.
SOMA_TO_WORLD = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
SKIN = np.asarray([134.0, 160.0, 196.0])          # BGR, as cv2 writes it: a warm mid skin tone
BACKGROUND_TOP, BACKGROUND_BOTTOM = 70.0, 32.0
KEY_LIGHT = np.asarray([0.35, -0.55, 0.75])
AMBIENT = 0.42


def render(camera, vertices, triangles, image=None):
    import cv2

    uv = np.empty((len(vertices), 2))
    depth = np.empty(len(vertices))
    for index, point in enumerate(vertices):
        pixel, z = camera.project(point)
        uv[index] = pixel
        depth[index] = z
    if image is None:
        image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        ramp = np.linspace(BACKGROUND_TOP, BACKGROUND_BOTTOM, HEIGHT)
        image[:] = ramp[:, None, None].astype(np.uint8)
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths < 1e-12, 1.0, lengths)
    centroids = (a + b + c) / 3.0
    towards = camera.camera_center_world_m - centroids
    towards /= np.linalg.norm(towards, axis=1, keepdims=True)
    facing = np.einsum("ij,ij->i", normals, towards) > 0.0
    visible = facing & (depth[triangles] > 0.0).all(axis=1)
    order = np.argsort(-np.linalg.norm(centroids - camera.camera_center_world_m, axis=1))
    light = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0) * (1.0 - AMBIENT) + AMBIENT
    for index in order:
        if not visible[index]:
            continue
        polygon = uv[triangles[index]].astype(np.int32)
        cv2.fillConvexPoly(image, polygon, tuple(float(v) for v in SKIN * shade[index]))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/arm-i"))
    parser.add_argument("--rig", type=Path, default=Path("artifacts/soma77-full/camera-rig.json"))
    parser.add_argument("--clip", default="autoanim_dialogue/amy-cuddy-dialogue-body")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--place", type=float, nargs=2, default=(-0.71, 4.76))
    args = parser.parse_args()
    import cv2

    cameras = tuple(c.scaled(WIDTH, HEIGHT) for c in cm.load_camera_rig(args.rig))
    body = SomaBodyMesh()
    motion = np.load(
        Path(".cache/autoanim_gnm/gem-x/outputs") / args.clip / "soma_motion.npz",
        allow_pickle=True,
    )
    frames = min(args.frames, len(motion["root_translation_m"]))
    vertices, joints = body.pose(
        motion["root_translation_m"][:frames], motion["local_rotations_xyzw"][:frames]
    )
    vertices = vertices @ SOMA_TO_WORLD.T
    joints = joints @ SOMA_TO_WORLD.T
    # Stand the character where a performer stood, feet on the floor.
    shift = np.zeros(3)
    shift[:2] = np.asarray(args.place) - joints[0, 0, :2]
    vertices += shift
    joints += shift
    drop = vertices[..., 2].min()
    vertices[..., 2] -= drop
    joints[..., 2] -= drop

    args.out.mkdir(parents=True, exist_ok=True)
    truth = {}
    for camera_index, name in enumerate(CAMERAS):
        (args.out / name).mkdir(exist_ok=True)
        rows = []
        for frame in range(frames):
            image = render(cameras[camera_index], vertices[frame], body.triangles)
            cv2.imwrite(str(args.out / name / f"{frame:06d}.jpg"), image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            projected = {}
            for joint_name, soma in SOMA77_TO_AUTOANIM.items():
                pixel, z = cameras[camera_index].project(joints[frame, soma])
                if z > 0:
                    projected[joint_name] = [float(pixel[0]), float(pixel[1])]
            rows.append(projected)
        truth[name] = rows
        print(f"{name}: {frames} frames rendered")
    (args.out / "truth-2d.json").write_text(json.dumps({
        "schema": "autoanim.arm-i-truth/1",
        "clip": args.clip, "frames": frames, "width": WIDTH, "height": HEIGHT,
        "projected_joints_px": truth,
    }, indent=1), encoding="utf-8")
    np.savez_compressed(args.out / "truth-3d.npz", joints_world_m=joints)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
