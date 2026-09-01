#!/usr/bin/env python3
"""Scale the rig to each performer's OWN measured bones, so the mocap can be judged.

The delivered rig is a fixed character: 540 mm across the shoulders where these performers
measure 346 and 363. Judging the capture through it confounds two separate things -- how
well the motion was captured, and how well it was retargeted onto a body of a different
size. **You cannot see a mocap error through a retarget error.**

So this builds a per-performer skeleton, the way a parametric body (SMPL-X and kin) is fitted
to a subject before anyone looks at the pose. Bone rest offsets are scaled to the lengths
`estimate_limb_lengths_m` already measures per subject per take -- the pipeline computes them
to constrain triangulation and then the retarget rebuilds on canonical offsets and discards
them.

This is an EVALUATION skeleton, not a delivery change. Nothing downstream is touched: the
shipped character is still the fixed MPFB asset, whose mesh is skinned to canonical bind-pose
bone lengths and would deform if those lengths moved. Retargeting onto a differently-sized
character is a separate problem and is deliberately not addressed here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.body import HumanoidSkeleton, JointSpec  # noqa: E402
from autoanim_gnm.commercial_multiview import estimate_limb_lengths_m  # noqa: E402

# rig bone -> the measured limb whose length it should take.
# Chains that share one measurement are listed together and split proportionally, because
# the capture has no landmark between them: there is no measured spine segmentation, only
# root->neck, so the four spine bones keep their canonical RATIOS and take the measured TOTAL.
DIRECT = {
    "LeftLowerArm": ("left_shoulder", "left_elbow"),
    "LeftHand": ("left_elbow", "left_wrist"),
    "RightLowerArm": ("right_shoulder", "right_elbow"),
    "RightHand": ("right_elbow", "right_wrist"),
    "LeftLowerLeg": ("left_hip", "left_knee"),
    "LeftFoot": ("left_knee", "left_ankle"),
    "RightLowerLeg": ("right_hip", "right_knee"),
    "RightFoot": ("right_knee", "right_ankle"),
}
SPINE_CHAIN = ("Spine", "Chest", "UpperChest", "Neck")   # sums to root->neck
HALF_SHOULDER = ("LeftShoulder", "LeftUpperArm", "RightShoulder", "RightUpperArm")
HALF_HIP = ("LeftUpperLeg", "RightUpperLeg")


def sized_skeleton(
    skeleton: HumanoidSkeleton, positions_world_z_up_m: np.ndarray
) -> tuple[HumanoidSkeleton, dict]:
    """A copy of `skeleton` with rest offsets scaled to this subject's measured limbs."""

    lengths = estimate_limb_lengths_m(positions_world_z_up_m)
    joints = list(skeleton.joints)
    index = {j.name: i for i, j in enumerate(joints)}
    report: dict[str, float] = {}

    def rescale(name: str, target_m: float) -> None:
        i = index[name]
        offset = np.asarray(joints[i].rest_translation_m, dtype=np.float64)
        norm = float(np.linalg.norm(offset))
        if norm <= 1e-9 or not np.isfinite(target_m) or target_m <= 1e-9:
            return
        joints[i] = dataclasses.replace(
            joints[i], rest_translation_m=tuple(offset * (target_m / norm))
        )
        report[name] = round(1000.0 * target_m, 1)

    for name, limb in DIRECT.items():
        if limb in lengths:
            rescale(name, lengths[limb])

    # spine: canonical ratios, measured total
    if ("root", "neck") in lengths:
        canon = [float(np.linalg.norm(joints[index[n]].rest_translation_m)) for n in SPINE_CHAIN]
        total = sum(canon)
        if total > 1e-9:
            for name, c in zip(SPINE_CHAIN, canon, strict=True):
                rescale(name, lengths[("root", "neck")] * c / total)

    # shoulder span: the arm ROOT must sit at half the measured shoulder width. The rig
    # reaches it through two bones, so both scale by the same factor.
    if ("left_shoulder", "right_shoulder") in lengths:
        half = 0.5 * lengths[("left_shoulder", "right_shoulder")]
        canon_half = abs(
            float(joints[index["LeftShoulder"]].rest_translation_m[0])
            + float(joints[index["LeftUpperArm"]].rest_translation_m[0])
        )
        if canon_half > 1e-9:
            k = half / canon_half
            for name in HALF_SHOULDER:
                off = np.asarray(joints[index[name]].rest_translation_m, dtype=np.float64)
                joints[index[name]] = dataclasses.replace(
                    joints[index[name]], rest_translation_m=tuple(off * k)
                )
            report["shoulder_span_mm"] = round(2000.0 * half, 1)
            report["shoulder_span_canonical_mm"] = round(2000.0 * canon_half, 1)

    # hip span, same construction
    if ("left_hip", "right_hip") in lengths:
        half = 0.5 * lengths[("left_hip", "right_hip")]
        canon_half = abs(float(joints[index["LeftUpperLeg"]].rest_translation_m[0]))
        if canon_half > 1e-9:
            k = half / canon_half
            for name in HALF_HIP:
                off = np.asarray(joints[index[name]].rest_translation_m, dtype=np.float64)
                joints[index[name]] = dataclasses.replace(
                    joints[index[name]],
                    rest_translation_m=(float(off[0] * k), float(off[1]), float(off[2])),
                )
            report["hip_span_mm"] = round(2000.0 * half, 1)
    return HumanoidSkeleton(joints=tuple(joints), schema_version=skeleton.schema_version), report


if __name__ == "__main__":
    import json
    from autoanim_gnm.body import skeleton_for_joint_names

    tracks = Path("artifacts/commercial-multiview-soma77")
    names = json.loads((tracks / "subject-00.body-track.json").read_text())["joint_names"]
    base = skeleton_for_joint_names(names)
    for s in (0, 1):
        pos = np.load(tracks / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        _, report = sized_skeleton(base, pos)
        print(f"subject {s}: {json.dumps(report, indent=None)}")
