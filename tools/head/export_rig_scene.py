#!/usr/bin/env python3
"""Export the DELIVERED rig as scene data for the four-region viewer.

Forward kinematics on the shipped `subject-*.body-track.npz`, so the page shows the file a
user receives rather than a run held in memory -- the distinction §6f and §6j were written
about. All 55 joints, both performers, with the parent-child bone list taken from the
skeleton rather than hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_joint_names  # noqa: E402

TRACKS = Path("artifacts/commercial-multiview-soma77")
OUT = Path("artifacts/head-lane/rig-scene.json")

REGION = {}
for name in ("Neck",):
    REGION[name] = "neck"
for name in ("Head", "LeftEye", "RightEye"):
    REGION[name] = "head"
for name in ("LeftFoot", "RightFoot", "LeftToes", "RightToes"):
    REGION[name] = "feet"


def region_for(name: str) -> str:
    if name in REGION:
        return REGION[name]
    if any(k in name for k in ("Thumb", "Index", "Middle", "Ring", "Little")):
        return "fingers"
    return "body"


def main() -> None:
    names = json.loads((TRACKS / "subject-00.body-track.json").read_text())["joint_names"]
    skeleton = skeleton_for_joint_names(names)
    bones = [
        [joint.parent, index]
        for index, joint in enumerate(skeleton.joints)
        if joint.parent >= 0
    ]
    subjects = []
    for s in (0, 1):
        track = np.load(TRACKS / f"subject-{s:02d}.body-track.npz")
        world = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=skeleton
        )
        q = track["local_rotations_xyzw"]
        local_deg = np.degrees(2.0 * np.arccos(np.clip(np.abs(q[..., 3]), 0.0, 1.0)))
        subjects.append({
            "joints": np.round(world, 4).tolist(),
            "local_deg": np.round(local_deg, 3).tolist(),
        })
        print(f"subject {s}: {world.shape[0]} frames x {world.shape[1]} joints")
    scene = {
        "joint_names": names,
        "bones": bones,
        "region": [region_for(n) for n in names],
        "subjects": subjects,
        "fps": 30,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scene, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    counts: dict[str, int] = {}
    for n in names:
        counts[region_for(n)] = counts.get(region_for(n), 0) + 1
    print("joints per region:", counts)


if __name__ == "__main__":
    main()
