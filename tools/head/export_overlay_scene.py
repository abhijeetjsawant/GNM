#!/usr/bin/env python3
"""Project BOTH skeletons into camera A001, for an overlay on the footage.

Two arms, because they answer different questions and are routinely confused:

  capture  -- the 19 triangulated joints. What the cameras actually saw.
  rig      -- the delivered 55-joint AutoAnim rig, through forward kinematics on the
              shipped `subject-*.body-track.npz`. What a user receives.

They are NOT the same thing and the gap between them is the retarget, which preserves
canonical body proportions rather than estimating a per-performer shape. Showing only the
rig makes the capture look worse than it is; showing only the capture makes the delivered
product look better than it is.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_joint_names  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, JOINT_NAMES, load_camera_rig,
)
from sized_skeleton import sized_skeleton  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare"))
from smplx_body import SMPLXBody, BODY_JOINTS  # noqa: E402

TRACKS = Path("artifacts/commercial-multiview-soma77")
OUT = Path("artifacts/head-lane/overlay-scene.json")
W, H = 1280, 720

CAPTURE_BONES = [
    ("neck", "root"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("root", "left_hip"), ("root", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("neck", "nose"),
]

REGION = {"Neck": "neck", "Head": "head", "LeftEye": "head", "RightEye": "head",
          "LeftFoot": "feet", "RightFoot": "feet", "LeftToes": "feet", "RightToes": "feet"}


def region_for(name: str) -> str:
    if name in REGION:
        return REGION[name]
    if any(k in name for k in ("Thumb", "Index", "Middle", "Ring", "Little")):
        return "fingers"
    return "body"


def project(camera, pts_z_up: np.ndarray) -> np.ndarray:
    """[N,3] world Z-up metres -> [N,2] pixels, NaN behind the camera."""
    P = camera.projection_matrix
    homo = np.concatenate([pts_z_up, np.ones((len(pts_z_up), 1))], axis=1)
    proj = homo @ P.T
    depth = proj[:, 2]
    out = np.full((len(pts_z_up), 2), np.nan)
    ok = np.isfinite(depth) & (depth > 1e-6)
    out[ok] = proj[ok, :2] / depth[ok, None]
    return out


def main() -> None:
    rig = {c.name: c for c in load_camera_rig(TRACKS / "camera-rig.json")}
    camera = rig["A001"].scaled(W, H)
    names = json.loads((TRACKS / "subject-00.body-track.json").read_text())["joint_names"]
    skeleton = skeleton_for_joint_names(names)
    rig_bones = [[j.parent, i] for i, j in enumerate(skeleton.joints) if j.parent >= 0]

    subjects = []
    for s in (0, 1):
        track = np.load(TRACKS / f"subject-{s:02d}.body-track.npz")
        cap = track["triangulated_world_positions_z_up_m"]           # Z-up
        world = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=skeleton
        )
        # the SAME delivered rotations on a skeleton scaled to this performer's own bones.
        # This is the arm that shows the MOCAP; the canonical one shows mocap plus retarget.
        fitted, limbs = sized_skeleton(skeleton, cap)
        world_fit = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=fitted
        )
        fitz = np.stack([world_fit[..., 0], -world_fit[..., 2], world_fit[..., 1]], axis=-1)
        # rig Y-up -> capture Z-up is the inverse of (x, z, -y)
        rigz = np.stack([world[..., 0], -world[..., 2], world[..., 1]], axis=-1)
        frames = len(cap)
        cap_px = np.stack([project(camera, cap[f]) for f in range(frames)])
        rig_px = np.stack([project(camera, rigz[f]) for f in range(frames)])
        fit_px = np.stack([project(camera, fitz[f]) for f in range(frames)])
        # RUNG 2: our capture driving SMPL-X. Already in the capture's Z-up world, so it
        # projects through the same lens with no change of basis.
        smx = np.load(Path("artifacts/compare") / f"smplx-posed-subject-{s:02d}.npz")["joints"]
        smx_px = np.stack([project(camera, np.nan_to_num(smx[f], nan=1e6)) for f in range(min(frames, len(smx)))])
        if len(smx_px) < frames:
            smx_px = np.concatenate([smx_px, np.full((frames - len(smx_px), BODY_JOINTS, 2), np.nan)])
        smx_px[~np.isfinite(smx[:frames]).all(axis=2)] = np.nan
        subjects.append({
            "capture": np.round(np.nan_to_num(cap_px, nan=-9999.0), 1).tolist(),
            "rig": np.round(np.nan_to_num(rig_px, nan=-9999.0), 1).tolist(),
            "fitted": np.round(np.nan_to_num(fit_px, nan=-9999.0), 1).tolist(),
            "smplx": np.round(np.nan_to_num(smx_px, nan=-9999.0), 1).tolist(),
            "limbs": limbs,
        })
        good = np.isfinite(cap_px).all(axis=2).mean()
        print(f"subject {s}: {frames} frames, capture points on screen {good:.1%}, "
              f"median capture x {np.nanmedian(cap_px[..., 0]):.0f}px y {np.nanmedian(cap_px[..., 1]):.0f}px")

    scene = {
        "width": W, "height": H, "fps": 30,
        "capture_names": list(JOINT_NAMES),
        "capture_bones": [[JOINT_INDEX[a], JOINT_INDEX[b]] for a, b in CAPTURE_BONES],
        "rig_names": names,
        "rig_bones": rig_bones,
        "rig_region": [region_for(n) for n in names],
        "smplx_bones": [[int(p), j] for j, p in enumerate(SMPLXBody().parents[:BODY_JOINTS]) if p >= 0],
        "subjects": subjects,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scene, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
