#!/usr/bin/env python3
"""Is the temporal-weight selection rule sound, or is it picking on a blind criterion?

The pipeline picks the temporal weight by **argmin in-sample reprojection**. That rule has
never been checked against the property it implies. With a temporal prior added to a
data term, increasing the prior weight can only make the DATA FIT worse: reprojection
should rise monotonically in the weight, and argmin should therefore always land at 0.

It does not -- the shipped solve selects 100 on both subjects. Exactly one of these is
true and they need different fixes:

  * the curve IS monotonic and something else selects, or
  * the curve is NOT monotonic, which means the low-weight solves are not converging to
    their own optimum -- and "convergence" was excluded from the head's failure analysis
    on the pre-audit code, so that exclusion would not survive.

This prints the curve. It deliberately reports ONLY reprojection and the solve's own
diagnostics -- no gate arm, no reference. Choosing a weight by looking at the gate would
be gate-tuning, which this lane has refused nine times, and the point here is the rule,
not the winner.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, _thorax_frames, load_camera_rig, load_observation_jsonl,
)
from autoanim_gnm.head_orientation import DEFAULT_WEIGHTS, solve_head_orientation  # noqa: E402
from associate import CAMERAS, OUT, RIG, WORK  # noqa: E402

HEAD_INDICES = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}


def main() -> None:
    rig = {c.name: c for c in load_camera_rig(RIG)}
    obs_raw = [load_observation_jsonl(WORK / f"{n}-observations.jsonl") for n in CAMERAS]
    cameras = [rig[n].scaled(obs_raw[0][0]["width"], obs_raw[0][0]["height"]) for n in CAMERAS]
    head = [[[np.asarray([p["landmarks_soma77"][i] for i in HEAD_INDICES.values()], float)
              for p in rec["people"]] for rec in cam] for cam in obs_raw]
    assignment = np.load(OUT / "association.npz")["assignment"]
    frames = len(obs_raw[0])
    report = {}
    for subject in (0, 1):
        smoothed = np.load(
            f"artifacts/commercial-multiview-soma77/subject-{subject:02d}.body-track.npz"
        )["triangulated_world_positions_z_up_m"]
        obs = np.full((frames, len(CAMERAS), len(HEAD_INDICES), 3), np.nan)
        for f in range(frames):
            for c in range(len(CAMERAS)):
                person = int(assignment[f, subject, c])
                if person >= 0:
                    obs[f, c] = head[c][f][person]
        rows = []
        for w in DEFAULT_WEIGHTS:
            solved = solve_head_orientation(
                cameras, obs, tuple(HEAD_INDICES),
                thorax_world=_thorax_frames(smoothed),
                neck_origin_world_m=smoothed[:, JOINT_INDEX["neck"]],
                weights=(float(w),),
            )
            rows.append({"weight": float(w), "reprojection_px": float(solved.reprojection_px)})
            print(f"  subject {subject}  weight {w:>9.1f}  reprojection {solved.reprojection_px:7.4f} px")
        px = [r["reprojection_px"] for r in rows]
        rises = all(b >= a - 1e-9 for a, b in zip(px, px[1:]))
        arg = int(np.argmin(px))
        report[f"subject_{subject:02d}"] = {
            "curve": rows, "monotonic_nondecreasing": bool(rises),
            "argmin_weight": rows[arg]["weight"], "argmin_px": px[arg],
            "px_at_zero_weight": px[0],
        }
        print(f"  -> subject {subject}: monotonic={rises}  argmin at weight "
              f"{rows[arg]['weight']} ({px[arg]:.4f} px), weight 0 gives {px[0]:.4f} px\n")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "weight-curve.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
