#!/usr/bin/env python3
"""Which MAMMA body is which of our subjects? Derived, never assumed.

**MAMMA's subject indices do not match ours on this fixture.** `body_id-00` is our
subject **1** and `body_id-01` is our subject **0**. Nothing in either output says so,
both are two-element lists indexed 0 and 1, and the mislabelling is invisible in every
per-subject statistic taken separately -- it corrupts only the *pairings* between them,
which is exactly what a parity table is made of.

Caught 2026-08-31 when a corroboration between MAMMA's head yaw and Apple Vision's ear
yaw returned |r| = 0.21 / 0.03 against shuffled controls of 0.18 / 0.22 -- i.e. nothing.
Under the derived correspondence the same data gives +0.40 / +0.34.

The correspondence is resolved by 3D position, which is unambiguous here: MAMMA's world
frame **is** the camera-rig world frame (matched pelvis-to-root median 41-55 mm), and
the mismatched pairings sit 1.36-1.38 m apart -- a 25x margin. The resolver asserts that
margin rather than trusting it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.commercial_multiview import JOINT_INDEX  # noqa: E402

MA3D = Path(
    "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
)
PELVIS = 0
MINIMUM_MARGIN = 5.0


def mamma_index_for(our_positions_world_z_up_m: np.ndarray) -> dict[int, int]:
    """Map our subject index -> MAMMA body_id, from pelvis agreement.

    `our_positions_world_z_up_m` is [subject, frame, joint, 3] in the 19-joint contract.
    """
    distances = np.zeros((2, 2))
    for ours in range(2):
        root = our_positions_world_z_up_m[ours][:, JOINT_INDEX["root"]]
        valid = np.isfinite(root).all(axis=1)
        for theirs in range(2):
            joints = np.load(
                MA3D / f"verts_joints_body_id-{theirs:02d}.npz", allow_pickle=True
            )["pred_joints"].astype(np.float64)
            distances[ours, theirs] = float(
                np.median(np.linalg.norm(joints[valid, PELVIS] - root[valid], axis=1)) * 1000.0
            )
    straight = distances[0, 0] + distances[1, 1]
    crossed = distances[0, 1] + distances[1, 0]
    mapping = {0: 0, 1: 1} if straight < crossed else {0: 1, 1: 0}
    margin = max(straight, crossed) / max(min(straight, crossed), 1e-9)
    if margin < MINIMUM_MARGIN:
        raise SystemExit(
            f"subject correspondence is ambiguous (margin {margin:.2f}x, "
            f"distances mm:\n{distances}) -- refusing to guess"
        )
    return mapping


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    blob = np.load("artifacts/head-lane/three-detector-cache.npz")
    mapping = mamma_index_for(blob["apple_vision"])
    for ours, theirs in sorted(mapping.items()):
        print(f"our subject {ours}  <->  MAMMA body_id-{theirs:02d}")
