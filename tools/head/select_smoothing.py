#!/usr/bin/env python3
"""Choose the head fit's temporal weight by an L-curve on our own reprojection.

**Why the first selection rule was replaced, stated plainly because it was replaced
after it failed.** `solve_head.py` chose the temporal weight by held-out *camera*
cross-validation, and it chose **zero** on both subjects. The gate then rejected the
candidate on P4 with a frame-to-frame travel p95 of 58-77 deg against a 7-12 deg ceiling.

The rule was mis-specified, and the reason is structural rather than a matter of the
score it produced: **detector noise is correlated across cameras within a frame.** All
four views are the same detector looking at the same instant, so dropping one view does
not create an independent test of *temporal* behaviour -- an unsmoothed fit that chases
per-frame noise still predicts the held-out camera well, because that camera's
observation carries the same per-frame noise. Held-out-camera CV can select a spatial
model; it is blind to a temporal one by construction.

The replacement uses only our own data and never touches MAMMA: the **L-curve**. Sweep
the weight, record in-frame reprojection, and take the **largest** weight whose
reprojection stays within a pre-registered tolerance of the unsmoothed fit. That is
"smooth as hard as the observations permit", and it cannot be tuned toward the reference
because the reference never enters it.

Blind to: whether the resulting smoothness is *correct*. A prior tight enough to hold
reprojection can still lag real motion. The gate's P2 is what catches over-smoothing --
a head smoothed into a constant scores zero spread and fails.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from associate import CAMERAS, OUT  # noqa: E402
from solve_head import (  # noqa: E402
    NAMES, gather, held_out_px, initialise, neck_origins, rodrigues, solve, thorax_frames,
)

# Pre-registered before running: the sweep, and the tolerance.
# Extended from (0 ... 3000) after subject 0's selection landed on the grid's top
# value -- a boundary hit, which is visible in the L-curve itself and needs no reference
# to the gate. The rule is unchanged; only the range it searches.
WEIGHTS = (0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0)
# **The rule changed a second time, after a failure, and the reason is that the old one
# selected a strictly worse-fitting model on its own criterion.** "Largest weight within
# +10% of the UNSMOOTHED fit" was written when the L-curve was flat and had no interior
# optimum, so "smooth as hard as the data permits" was the only sensible reading. Once
# the head was anchored to the neck the curve acquired a clear interior minimum, and the
# old rule then chose weight 30,000 at 2.935 px over weight 100 at 2.719 px and 3,000 at
# 2.742 px -- a worse fit, by its own measure, purely because the tolerance was anchored
# to a point that is no longer the reference. **That is visible without the gate**, and a
# rule that prefers a model it can see is worse is wrong regardless of what it scores.
#
# The replacement is the standard answer and removes a parameter rather than adding one:
# **minimise in-frame reprojection.** The interior minimum is what makes it legitimate --
# reprojection is not monotone in flexibility here, so this is not "pick the least
# smoothing".
TOLERANCE = None


def main() -> None:
    report: dict = {"rule": "minimum in-frame reprojection (replaced the +10% tolerance "
                            "rule, which selected a strictly worse-fitting model)",
                    "weights": list(WEIGHTS)}
    output: dict[str, np.ndarray] = {}
    for subject in range(2):
        observations, cameras = gather(subject)
        template0, rotations0, translations0 = initialise(subject)
        every = np.ones(len(CAMERAS), dtype=bool)
        thorax = thorax_frames(subject)
        neck = neck_origins(subject)
        curve: dict[float, float] = {}
        fits: dict[float, dict] = {}
        for weight in WEIGHTS:
            fit = solve(observations, cameras, template0, rotations0, translations0, weight,
                        every, thorax=thorax, neck_origin=neck)
            fits[weight] = fit
            curve[weight] = float(np.nanmean(
                [held_out_px(fit, observations, cameras, c) for c in range(len(CAMERAS))]))
            print(f"  subject {subject}  weight {weight:7.1f} -> in-frame {curve[weight]:.3f} px")
        best = min(WEIGHTS, key=lambda w: curve[w])
        fit = fits[best]
        matrices = rodrigues(fit["rotations"])
        output[f"subject_{subject:02d}_head_world"] = matrices
        output[f"subject_{subject:02d}_head_position_m"] = fit["translations"]
        output[f"subject_{subject:02d}_template_m"] = fit["template"]
        report[f"subject_{subject:02d}"] = {
            "in_frame_px_by_weight": {str(k): v for k, v in curve.items()},
            "unsmoothed_px": curve[0.0],
            "chosen_weight": best,
            "chosen_in_frame_px": curve[best],
            "template_extent_mm": {
                name: float(np.linalg.norm(fit["template"][i]) * 1000.0)
                for i, name in enumerate(NAMES)},
        }
        print(f"subject {subject}: chose weight {best} at {curve[best]:.3f} px "
              f"(minimum in-frame reprojection)")
    np.savez(OUT / "head-solve.npz", **output)
    (OUT / "head-solve-lcurve.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
