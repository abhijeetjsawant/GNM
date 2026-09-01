#!/usr/bin/env python3
"""Minimal SMPL-X shape+pose forward model, enough to FIT it to our own capture.

**Instrument only, never the shipping path.** SMPL-X is MPI research-licensed; the
`commercial_multiview` run report declares `smplx_model: false` and that must stay false for
delivered artifacts. This exists to answer one question -- how much of the ~100 mm the
retarget loses is recovered by a proper parametric body -- and to let the substitution
proceed one part at a time, our capture into their body.

Deliberately NOT the `smplx` package: this is the shape and kinematic terms only, which is
all a joint fit needs. Pose blendshapes and the mesh are skipped -- they move vertices, not
joints, and nothing here scores a vertex.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MODEL = Path(".cache/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz")
BODY_JOINTS = 22  # pelvis..wrists; hands and face are separate chains we do not fit


class SMPLXBody:
    """Rest joints from shape, and posed joints from a body pose. Joints only."""

    def __init__(self, path: Path = MODEL, shape_terms: int = 10) -> None:
        data = np.load(path, allow_pickle=True)
        self.v_template = np.asarray(data["v_template"], dtype=np.float64)
        self.shapedirs = np.asarray(data["shapedirs"], dtype=np.float64)[:, :, :shape_terms]
        self.J_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        self.parents = np.asarray(data["kintree_table"], dtype=np.int64)[0].copy()
        self.parents[0] = -1
        self.shape_terms = shape_terms

    def rest_joints(self, betas: np.ndarray) -> np.ndarray:
        """[55,3] rest joint positions for a body shape."""
        verts = self.v_template + self.shapedirs @ np.asarray(betas, dtype=np.float64)
        return self.J_regressor @ verts

    def posed_joints(
        self, betas: np.ndarray, body_pose_rotvec: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """[frames,22,3] world joints. `body_pose_rotvec` is [frames,22,3] axis-angle."""
        rest = self.rest_joints(betas)[:BODY_JOINTS]
        frames = len(body_pose_rotvec)
        out = np.zeros((frames, BODY_JOINTS, 3))
        rot = Rotation.from_rotvec(
            np.asarray(body_pose_rotvec, dtype=np.float64).reshape(-1, 3)
        ).as_matrix().reshape(frames, BODY_JOINTS, 3, 3)
        for f in range(frames):
            world_rot = np.zeros((BODY_JOINTS, 3, 3))
            pos = np.zeros((BODY_JOINTS, 3))
            for j in range(BODY_JOINTS):
                p = self.parents[j]
                if p < 0:
                    world_rot[j] = rot[f, j]
                    pos[j] = rest[j]
                else:
                    world_rot[j] = world_rot[p] @ rot[f, j]
                    pos[j] = pos[p] + world_rot[p] @ (rest[j] - rest[p])
            out[f] = pos + np.asarray(translation[f], dtype=np.float64)
        return out


if __name__ == "__main__":
    body = SMPLXBody()
    zero = body.rest_joints(np.zeros(body.shape_terms))
    print(f"model loaded: {body.v_template.shape[0]} vertices, {len(body.parents)} joints")
    print(f"neutral shoulder span: "
          f"{1000*np.linalg.norm(zero[16]-zero[17]):.1f} mm   "
          f"hip span {1000*np.linalg.norm(zero[1]-zero[2]):.1f} mm   "
          f"height {1000*(zero[:22,1].max()-zero[:22,1].min()):.1f} mm")
    for i, b in enumerate([-2.0, 0.0, 2.0]):
        v = np.zeros(body.shape_terms); v[0] = b
        j = body.rest_joints(v)
        print(f"  beta0={b:+.1f}: shoulder span {1000*np.linalg.norm(j[16]-j[17]):.1f} mm, "
              f"upper arm {1000*np.linalg.norm(j[16]-j[18]):.1f} mm")
