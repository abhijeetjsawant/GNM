#!/usr/bin/env python3
"""Our delivered head, measured with the SAME instrument as MAMMA's.

`mamma_head_bar.py` composes a parent chain to world and takes geodesic
frame-to-frame rotations. `docs/HEAD_FEET_HANDS_PLAN.md` §1c quotes 4.48 deg
median / 43.79 p95 for `c_head`, but that is a **local, parent-relative**
rotation read off a momentum fit -- a different quantity from a different
pipeline. Quoting a ratio between the two would be the lane's own
same-denominator rule broken inside its headline number.

So this composes the delivered `DETAILED_HUMANOID` chain
Root -> Hips -> Spine -> Chest -> UpperChest -> Neck -> Head from
`local_rotations_xyzw` in the retained body track, using the *same* functions
`mamma_head_bar.py` uses, and reports the same statistics.

Blind to: accuracy, exactly as the MAMMA bar is. Both sides are tracking
statistics of an estimate. What this makes legitimate is the *comparison*, not
either number as truth.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoanim_gnm.body import DETAILED_HUMANOID  # noqa: E402
from mamma_head_bar import geodesic_deg, stats  # noqa: E402

TRACKS = Path("artifacts/commercial-multiview-soma77")
NAME = {joint.name: index for index, joint in enumerate(DETAILED_HUMANOID.joints)}


def quaternion_matrix(xyzw: np.ndarray) -> np.ndarray:
    q = xyzw / np.linalg.norm(xyzw, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def chain_to(joint: str) -> list[int]:
    chain: list[int] = []
    index = NAME[joint]
    while index >= 0:
        chain.append(index)
        index = DETAILED_HUMANOID.joints[index].parent
    return list(reversed(chain))


def world_of(rotations: np.ndarray, joint: str) -> np.ndarray:
    chain = chain_to(joint)
    out = quaternion_matrix(rotations[:, chain[0]])
    for index in chain[1:]:
        out = out @ quaternion_matrix(rotations[:, index])
    return out


def main() -> None:
    report: dict = {}
    for subject in range(2):
        rotations = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
            "local_rotations_xyzw"
        ].astype(np.float64)
        head = world_of(rotations, "Head")
        neck = world_of(rotations, "Neck")
        thorax = world_of(rotations, "UpperChest")

        relative = np.einsum("nji,njk->nik", thorax, head)
        mean = relative.mean(axis=0)
        u, _, vt = np.linalg.svd(mean)
        mean = u @ vt

        report[f"{subject:02d}"] = {
            "chain": [DETAILED_HUMANOID.joints[i].name for i in chain_to("Head")],
            "frame_to_frame_deg": {
                "head": stats(geodesic_deg(head[1:], head[:-1])),
                "neck": stats(geodesic_deg(neck[1:], neck[:-1])),
                "thorax_upperchest": stats(geodesic_deg(thorax[1:], thorax[:-1])),
            },
            "head_relative_to_thorax_deg": {
                "offset_from_identity": stats(
                    geodesic_deg(relative, np.broadcast_to(np.eye(3), relative.shape))
                ),
                "spread_about_take_mean": stats(
                    geodesic_deg(relative, np.broadcast_to(mean, relative.shape))
                ),
                "frame_to_frame_travel": stats(geodesic_deg(relative[1:], relative[:-1])),
            },
        }
    Path("artifacts/head-lane/our-head-bar.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
