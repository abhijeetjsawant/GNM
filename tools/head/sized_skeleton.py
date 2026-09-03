#!/usr/bin/env python3
"""Scale the rig to each performer's OWN measured bones -- now a thin re-export.

**D3, 2026-09-03: the algorithm moved into the shipping path.** It lives in
`src/autoanim_gnm/performer_skeleton.py` and the delivery build calls it directly, so the
instrument arm and the delivered rig are the SAME sizing by construction and cannot drift
apart. This file stays because a dozen instruments import `sized_skeleton` by name; it
adds nothing but the legacy report shape (millimetres, rounded), which several reports
and `tools/compare/ladder.py` read by key.

**One behavioural change came with the move, and it is a repair, not a redesign.** The
spine chain used to take `estimate_limb_lengths_m`'s `("root", "neck")` length outright.
That length is measured from the detector's `root` landmark, while the chain it scales
sums the rig's **Hips -> Neck** distance, and since D2b the root places the rig's LEG
ROOTS on the hip-landmark midpoint -- so `Hips` sits one upper-leg drop above it. The
target is now the hip-midpoint-to-neck span MINUS that drop, read from the skeleton's own
rest. The same repair had already been made in
`tools/swap-harness/retarget_cost.py::scaled_skeleton`, where the uncorrected version
showed up as a delivered neck sitting at exactly 80.00 mm on both performers.

Anything re-run through this file after 2026-09-03 therefore reports a different sized
arm than it did before, by design. The frozen D1/D2 gates are not re-run.

This is no longer *only* an evaluation skeleton: `reconstruct_multiview` builds one per
performer and serialises it on the track.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.body import HumanoidSkeleton  # noqa: E402
from autoanim_gnm.performer_skeleton import (  # noqa: E402
    DIRECT_LIMBS,
    HALF_HIP,
    HALF_SHOULDER,
    SPINE_CHAIN,
    performer_skeleton,
)

DIRECT = DIRECT_LIMBS


def sized_skeleton(
    skeleton: HumanoidSkeleton, positions_world_z_up_m: np.ndarray
) -> tuple[HumanoidSkeleton, dict]:
    """A copy of `skeleton` with rest offsets scaled to this subject's measured limbs."""

    fitted, metres = performer_skeleton(skeleton, positions_world_z_up_m)
    report: dict[str, float] = {}
    for name in tuple(DIRECT_LIMBS) + SPINE_CHAIN:
        if name in metres:
            report[name] = round(1000.0 * metres[name], 1)
    if "shoulder_half_span_m" in metres:
        report["shoulder_span_mm"] = round(2000.0 * metres["shoulder_half_span_m"], 1)
        report["shoulder_span_canonical_mm"] = round(
            2000.0 * metres["shoulder_half_span_canonical_m"], 1
        )
    if "hip_half_span_m" in metres:
        report["hip_span_mm"] = round(2000.0 * metres["hip_half_span_m"], 1)
    return fitted, report


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
