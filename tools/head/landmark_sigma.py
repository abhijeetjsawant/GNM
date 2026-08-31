#!/usr/bin/env python3
"""Per-landmark reliability for the five head points, measured from the INPUT alone.

`head_orientation.py` weights each observation by the DETECTOR's confidence and by nothing
else, so all five head landmarks enter the fit equally trusted. §2 measured that they are
not: `HeadEnd` varies 66.5-115.0 % in length against body controls at 2.5-4.2 %, and the
eye baseline is 59 mm with a standard deviation of 133-201 mm. The solve is being told that
a landmark we have shown to be noise is as good as one we have not.

This derives a per-landmark sigma the same way the HeadEnd and toe gates work -- from
**geometric self-consistency**, never from the reference or the gate:

    sigma_i = median over j != i of  SD_t | x_i(t) - x_j(t) |

A point rigid to the skull holds its distance to every other skull point; one the detector
is inventing does not. The statistic touches only our own triangulated 3D, so it is
computable with MAMMA deleted.

Blind to: accuracy and to COMMON-MODE error. Five landmarks that drift together look
perfectly consistent here, so this ranks reliability WITHIN the set and cannot detect that
the whole set is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triangulate_soma import triangulate  # noqa: E402

NAMES = ("Head", "HeadEnd", "Jaw", "LeftEye", "RightEye")
IDX = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}


def main() -> None:
    positions, _, used = triangulate([IDX[n] for n in NAMES])
    report = {}
    for subject in range(positions.shape[0]):
        pos = positions[subject]
        common = np.isfinite(pos).all(axis=(1, 2)) & used[:, subject]
        sig = {}
        for i, a in enumerate(NAMES):
            sds = []
            for j, b in enumerate(NAMES):
                if i == j:
                    continue
                d = np.linalg.norm(pos[common, i] - pos[common, j], axis=1) * 1000.0
                sds.append(float(d.std()))
            sig[a] = {"sigma_mm": float(np.median(sds)),
                      "per_partner_sd_mm": [round(v, 1) for v in sds]}
        base = min(v["sigma_mm"] for v in sig.values())
        for a in NAMES:
            sig[a]["relative_to_best"] = round(sig[a]["sigma_mm"] / base, 2)
            sig[a]["weight_1_over_sigma"] = round(base / sig[a]["sigma_mm"], 3)
        report[f"subject_{subject:02d}"] = {"frames": int(common.sum()), "landmarks": sig}
        print(f"\n=== subject {subject}  ({int(common.sum())} frames) ===")
        print(f"{'landmark':10s} {'sigma mm':>9s} {'x best':>7s} {'weight':>7s}")
        for a in NAMES:
            v = sig[a]
            print(f"{a:10s} {v['sigma_mm']:9.1f} {v['relative_to_best']:7.2f} "
                  f"{v['weight_1_over_sigma']:7.3f}")
    Path("artifacts/head-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/head-lane/landmark-sigma.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
