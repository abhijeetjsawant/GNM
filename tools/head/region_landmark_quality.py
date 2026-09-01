#!/usr/bin/env python3
"""Input quality for the three unbuilt regions: NECK, FEET, FINGERS. One denominator.

The head lane's lesson, applied before building anything: measure the input, against body
controls this lane already trusts, on one common frame set. Segment length is constant by
construction for a rigid segment, so its standard deviation is pure measurement error.

`HeadEnd` failed this at 66.5-115 % against controls at 2.5-4.2 %; the toe TIP failed at
14-145 %; the toe BALL survived at 5.0-9.6 %. Same instrument here, so the four regions sit
on one axis and can be compared.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triangulate_soma import triangulate  # noqa: E402

IDX = {
    "Chest": 3, "Neck1": 4, "Neck2": 5, "Head": 6,
    "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14,
    "LeftThumb1": 15, "LeftThumb2": 16, "LeftIndex1": 19, "LeftIndex2": 20, "LeftIndex3": 21,
    "LeftMiddle1": 24, "LeftMiddle2": 25, "LeftPinky1": 34,
    "RightArm": 40, "RightForeArm": 41, "RightHand": 42,
    "RightIndex1": 47, "RightIndex2": 48, "RightMiddle1": 52,
    "LeftLeg": 67, "LeftShin": 68, "LeftFoot": 69, "LeftToeBase": 70, "LeftToeEnd": 71,
    "RightLeg": 72, "RightShin": 73, "RightFoot": 74, "RightToeBase": 75, "RightToeEnd": 76,
}
ORDER = list(IDX)
SLOT = {n: i for i, n in enumerate(ORDER)}

SEGMENTS = [
    ("NECK   Chest->Neck1", "Chest", "Neck1", "~100-160"),
    ("NECK   Neck1->Neck2", "Neck1", "Neck2", "~40-80"),
    ("NECK   Neck2->Head", "Neck2", "Head", "~80-110"),
    ("FEET   ball L  Foot->ToeBase", "LeftFoot", "LeftToeBase", "~120-180"),
    ("FEET   ball R  Foot->ToeBase", "RightFoot", "RightToeBase", "~120-180"),
    ("FEET   tip L   ToeBase->ToeEnd", "LeftToeBase", "LeftToeEnd", "~40-90"),
    ("FING   palm L  Hand->Index1", "LeftHand", "LeftIndex1", "~85-100"),
    ("FING   palm R  Hand->Index1", "RightHand", "RightIndex1", "~85-100"),
    ("FING   prox L  Index1->Index2", "LeftIndex1", "LeftIndex2", "~40-50"),
    ("FING   prox R  Index1->Index2", "RightIndex1", "RightIndex2", "~40-50"),
    ("FING   inter L Index2->Index3", "LeftIndex2", "LeftIndex3", "~22-30"),
    ("FING   span L  Index1->Pinky1", "LeftIndex1", "LeftPinky1", "~65-85"),
    ("FING   thumb L Hand->Thumb1", "LeftHand", "LeftThumb1", "~35-50"),
    ("CTRL   shin L", "LeftShin", "LeftFoot", "~390-410"),
    ("CTRL   thigh L", "LeftLeg", "LeftShin", "~400-450"),
    ("CTRL   forearm L", "LeftForeArm", "LeftHand", "~240-270"),
    ("CTRL   upper arm L", "LeftArm", "LeftForeArm", "~270-290"),
]


def main() -> None:
    positions, support, used = triangulate([IDX[n] for n in ORDER])
    report = {}
    for subject in range(positions.shape[0]):
        pos = positions[subject]
        have = np.isfinite(pos).all(axis=2)
        needed = sorted({SLOT[a] for _, a, b, _ in SEGMENTS} | {SLOT[b] for _, a, b, _ in SEGMENTS})
        common = have[:, needed].all(axis=1) & used[:, subject]
        rows = {}
        for label, a, b, expected in SEGMENTS:
            d = np.linalg.norm(pos[common, SLOT[a]] - pos[common, SLOT[b]], axis=1) * 1000.0
            rows[label] = {"mean_mm": float(d.mean()), "sd_mm": float(d.std()),
                           "sd_pct": float(100 * d.std() / d.mean()), "anatomical_mm": expected}
        report[f"subject_{subject:02d}"] = {"frames": int(common.sum()), "segments": rows}
        print(f"\n=== subject {subject}  ({int(common.sum())} common frames) ===")
        print(f"{'segment':32s} {'mean':>8s} {'sd':>8s} {'sd%':>8s}   anatomical")
        for label, r in rows.items():
            print(f"{label:32s} {r['mean_mm']:8.1f} {r['sd_mm']:8.1f} {r['sd_pct']:7.1f}%   {r['anatomical_mm']}")
    Path("artifacts/head-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/head-lane/region-landmark-quality.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
