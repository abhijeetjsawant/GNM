"""One rest skeleton per performer, built from OUR OWN capture and nothing else.

Ladder step **D3**.  The delivered rig used to be a fixed character: 540 mm across the
shoulders where these performers measure 346 and 363 mm.  Judging a capture through it
confounds two separate things -- how well the motion was captured, and how well it was
retargeted onto a body of a different size -- and *you cannot see a mocap error through a
retarget error*.

**Where this came from.** It is `tools/head/sized_skeleton.py`, moved into `src` so that
the instrument arm and the delivery path cannot diverge; that file now re-exports from
here.  The algorithm is unchanged except for one repair, folded in below and marked, and
that repair is a DERIVATION from the rig's own rest geometry, not a constant.

**Provenance.** Every number this function produces comes from
:func:`autoanim_gnm.commercial_multiview.estimate_limb_lengths_m` run on our own
triangulated capture, and from the rest geometry of the skeleton it is handed.  No
reference fitter, no MAMMA output, and no fitted constant enters.  The only literals here
are ``0.5`` (a midpoint) and a degeneracy tolerance.

**What it does NOT do, and why.** It does not redesign the shoulder construction.  The
arm root must sit at half the measured shoulder width, and the rig reaches it through two
bones (``Shoulder`` then ``UpperArm``); scaling both uniformly also scales their VERTICAL
component, which lowers the arm root by roughly 36 mm on these performers, where an
X-only scaling would not.  Which of the two is right is a *selection* question about
where a shoulder landmark sits relative to the glenohumeral joint, it needs its own
evidence, and it is **D5's**, not D3's.  D3 propagates one skeleton; it does not choose
the skeleton.

It also does not touch head, hands, fingers or toes: the capture measures no bone there
(`docs/HEAD_FEET_HANDS_PLAN.md`), so those keep the canonical offsets and are honest
about it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .body import HumanoidSkeleton


# Rig bone -> the measured limb whose length it should take.  Chains that share one
# measurement are listed together and split proportionally, because the capture has no
# landmark between them: there is no measured spine segmentation, only root->neck, so the
# four spine bones keep their canonical RATIOS and take the measured TOTAL.
DIRECT_LIMBS: dict[str, tuple[str, str]] = {
    "LeftLowerArm": ("left_shoulder", "left_elbow"),
    "LeftHand": ("left_elbow", "left_wrist"),
    "RightLowerArm": ("right_shoulder", "right_elbow"),
    "RightHand": ("right_elbow", "right_wrist"),
    "LeftLowerLeg": ("left_hip", "left_knee"),
    "LeftFoot": ("left_knee", "left_ankle"),
    "RightLowerLeg": ("right_hip", "right_knee"),
    "RightFoot": ("right_knee", "right_ankle"),
}
SPINE_CHAIN = ("Spine", "Chest", "UpperChest", "Neck")  # sums to Hips->Neck
HALF_SHOULDER = ("LeftShoulder", "LeftUpperArm", "RightShoulder", "RightUpperArm")
HALF_HIP = ("LeftUpperLeg", "RightUpperLeg")
_DEGENERATE_M = 1.0e-9


def _leg_root_drop_m(rest: np.ndarray, index: dict[str, int]) -> float:
    """How far the rig's leg roots hang BELOW its ``Hips``, in the rig's own rest.

    Read from the skeleton, never written down.  It is the same quantity D2b's
    ``_leg_root_offset`` uses to place the root, and it is here for the same reason: the
    hip landmarks are femoral joint centres, so their midpoint is where the LEG ROOTS go,
    and ``Hips`` sits this far above it.
    """

    return -0.5 * float(rest[index["LeftUpperLeg"], 1] + rest[index["RightUpperLeg"], 1])


def performer_skeleton(
    skeleton: HumanoidSkeleton,
    positions_world_z_up_m: np.ndarray,
) -> tuple[HumanoidSkeleton, dict[str, float]]:
    """``skeleton``'s joints on THIS performer's measured bone lengths.

    Returns the skeleton and a report of the targets it hit, in **metres** -- no rounding
    and no unit conversion happen in the shipping path, so a reader of the report and a
    reader of the skeleton cannot disagree.
    """

    # Imported here, not at module scope: `commercial_multiview` imports the body track
    # machinery which imports this module's neighbours, and a top-level import closes the
    # cycle.  It is also the only dependency this module has beyond numpy.
    from .commercial_multiview import JOINT_INDEX, estimate_limb_lengths_m

    positions = np.asarray(positions_world_z_up_m, dtype=np.float64)
    lengths = estimate_limb_lengths_m(positions)
    names = skeleton.names
    index = {name: position for position, name in enumerate(names)}
    rest = np.array(skeleton.rest_translations_m, dtype=np.float64, copy=True)
    canonical = skeleton.rest_translations_m
    report: dict[str, float] = {}

    def rescale(name: str, target_m: float) -> None:
        offset = canonical[index[name]]
        norm = float(np.linalg.norm(offset))
        if (
            norm <= _DEGENERATE_M
            or not np.isfinite(target_m)
            or target_m <= _DEGENERATE_M
        ):
            return
        rest[index[name]] = offset * (target_m / norm)
        report[name] = float(target_m)

    for name, limb in DIRECT_LIMBS.items():
        if limb in lengths:
            rescale(name, lengths[limb])

    # ---------------------------------------------------------------- the spine chain
    # THE REPAIR, folded in from `tools/swap-harness/retarget_cost.py::scaled_skeleton`.
    #
    # The chain Spine->Chest->UpperChest->Neck sums to the rig's **Hips -> Neck**
    # distance.  What the capture measures is the distance from the **hip-landmark
    # midpoint** to the neck -- and since D2b the root places the rig's LEG ROOTS on that
    # midpoint, so `Hips` sits `drop` above it.  Scaling the chain to the measured span
    # directly therefore overshoots by exactly `drop`, and it did: the delivered neck sat
    # at 80.00 mm on both performers, which is the canonical `drop` to the millimetre,
    # and it made torso scaling look harmful when it is not.
    #
    # `drop` is read from `rest`.  No constant arrives.
    pelvis = 0.5 * (
        positions[:, JOINT_INDEX["left_hip"]] + positions[:, JOINT_INDEX["right_hip"]]
    )
    neck = positions[:, JOINT_INDEX["neck"]]
    usable = np.isfinite(pelvis).all(axis=1) & np.isfinite(neck).all(axis=1)
    canonical_chain = [
        float(np.linalg.norm(canonical[index[name]])) for name in SPINE_CHAIN
    ]
    total = sum(canonical_chain)
    if usable.any() and total > _DEGENERATE_M:
        span = float(np.median(np.linalg.norm(neck[usable] - pelvis[usable], axis=1)))
        drop = _leg_root_drop_m(canonical, index)
        target = span - drop
        report["hip_midpoint_to_neck_m"] = span
        report["leg_root_drop_m"] = drop
        if target > _DEGENERATE_M:
            for name, canonical_length in zip(SPINE_CHAIN, canonical_chain, strict=True):
                rescale(name, target * canonical_length / total)

    # ---------------------------------------------------------------- the spans
    # The arm ROOT must sit at half the measured shoulder width.  The rig reaches it
    # through two bones, so both scale by the same factor.  See the module docstring:
    # whether this should be X-only is D5's question, deliberately left open here.
    if ("left_shoulder", "right_shoulder") in lengths:
        half = 0.5 * lengths[("left_shoulder", "right_shoulder")]
        canonical_half = abs(
            float(canonical[index["LeftShoulder"], 0])
            + float(canonical[index["LeftUpperArm"], 0])
        )
        if canonical_half > _DEGENERATE_M:
            factor = half / canonical_half
            for name in HALF_SHOULDER:
                rest[index[name]] = canonical[index[name]] * factor
            report["shoulder_half_span_m"] = half
            report["shoulder_half_span_canonical_m"] = canonical_half

    # The hip span, X only: the vertical part of the upper-leg offset is the `drop` the
    # spine chain above is corrected against, and scaling it would move the two together.
    if ("left_hip", "right_hip") in lengths:
        half = 0.5 * lengths[("left_hip", "right_hip")]
        canonical_half = abs(float(canonical[index["LeftUpperLeg"], 0]))
        if canonical_half > _DEGENERATE_M:
            factor = half / canonical_half
            for name in HALF_HIP:
                rest[index[name], 0] = canonical[index[name], 0] * factor
            report["hip_half_span_m"] = half

    return skeleton.with_rest_translations(rest), report


def performer_skeleton_report(
    skeleton: HumanoidSkeleton,
    positions_world_z_up_m: np.ndarray,
) -> dict[str, Any]:
    """The sizing report alone, for an instrument that wants the numbers not the rig."""

    _, report = performer_skeleton(skeleton, positions_world_z_up_m)
    return dict(report)


__all__ = [
    "DIRECT_LIMBS",
    "HALF_HIP",
    "HALF_SHOULDER",
    "SPINE_CHAIN",
    "performer_skeleton",
    "performer_skeleton_report",
]
