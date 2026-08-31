#!/usr/bin/env python3
"""Are SOMA-77's toe landmarks a usable foot axis, or are they HeadEnd again?

`docs/HEAD_FEET_HANDS_PLAN.md` §2 says a toe point "turns the foot from one landmark into
two, which is the difference between unobservable and observable orientation" -- `l_foot`
has no rotation channel at all today. It also warns, twice, that **a better bet is not a
result**: fingers looked like free capability and failed bone-length stability at 4.4x the
body control, and `HeadEnd` -- called "the single most promising head lead on this page"
-- failed at 66.5 % / 115.0 % length variation against controls at 2.5-4.2 %.

So the toes get the same instrument before anything is built on them, deliberately copying
`tools/head/headend_gate.py`. The verdict is **segment-length stability against body
controls the lane already trusts**, plus one invariant the head had no equivalent of: the
**ankle angle**, the angle between the foot axis and the shin, which a real foot holds near
a right angle and a hallucinated toe has no reason to.

Same denominator throughout: every quantity is triangulated by `triangulate_point` at
production settings, on the pipeline's own association, restricted to frames where **every**
compared quantity resolved. A candidate scored on 40 easy frames must not be compared with
a control scored on 150.

Blind to:
  * ACCURACY. A stable segment can be stably wrong -- these are skeletal conventions, not
    measured anatomy, and nothing here says where the real ball of the foot is.
  * Whether the detector MEASURES the toe or predicts it from the crop. Cross-view
    geometric consistency is the only discriminator available without ground truth, which
    is why every invariant is scored across four independently-viewed cameras.
  * Ground contact. A foot can be stable in length and still float; that is the
    foot-contact question and it is not this one.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head"))
from triangulate_soma import triangulate  # noqa: E402

# SOMA-77 indices, read from src/autoanim_gnm/data/somaskel77-v1.json.
IDX = {
    "LeftLeg": 67, "LeftShin": 68, "LeftFoot": 69, "LeftToeBase": 70, "LeftToeEnd": 71,
    "RightLeg": 72, "RightShin": 73, "RightFoot": 74, "RightToeBase": 75, "RightToeEnd": 76,
    "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14,
    "RightArm": 40, "RightForeArm": 41, "RightHand": 42,
    "Chest": 3, "Head": 6,
}
ORDER = list(IDX)
SLOT = {name: i for i, name in enumerate(ORDER)}

# (label, a, b, anatomical range in mm) -- candidates first, then the controls this lane
# already trusts. The controls are the same ones HeadEnd was measured against, so the
# two verdicts sit on one axis.
SEGMENTS = [
    ("CANDIDATE ball L   Foot->ToeBase", "LeftFoot", "LeftToeBase", "~120-180"),
    ("CANDIDATE ball R   Foot->ToeBase", "RightFoot", "RightToeBase", "~120-180"),
    ("CANDIDATE toe L    ToeBase->ToeEnd", "LeftToeBase", "LeftToeEnd", "~40-90"),
    ("CANDIDATE toe R    ToeBase->ToeEnd", "RightToeBase", "RightToeEnd", "~40-90"),
    ("CONTROL   shin L   Shin->Foot", "LeftShin", "LeftFoot", "~390-410"),
    ("CONTROL   shin R   Shin->Foot", "RightShin", "RightFoot", "~390-410"),
    ("CONTROL   thigh L  Leg->Shin", "LeftLeg", "LeftShin", "~400-450"),
    ("CONTROL   thigh R  Leg->Shin", "RightLeg", "RightShin", "~400-450"),
    ("CONTROL   forearm L", "LeftForeArm", "LeftHand", "~240-270"),
    ("CONTROL   upper arm L", "LeftArm", "LeftForeArm", "~270-290"),
]

# Directions whose frame-to-frame rotation is the orientation signal the foot would gain.
AXES = [
    ("CANDIDATE foot axis L  Foot->ToeBase", "LeftFoot", "LeftToeBase"),
    ("CANDIDATE foot axis R  Foot->ToeBase", "RightFoot", "RightToeBase"),
    ("CONTROL   shin axis L", "LeftShin", "LeftFoot"),
    ("CONTROL   shin axis R", "RightShin", "RightFoot"),
    ("CONTROL   shoulder axis", "RightArm", "LeftArm"),
]

# The invariant the head had no equivalent of. A standing ankle holds the foot near a right
# angle to the shin; a toe predicted from a crop has no reason to respect it.
ANKLES = [("L", "LeftShin", "LeftFoot", "LeftToeBase"), ("R", "RightShin", "RightFoot", "RightToeBase")]


def summarise(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean_mm": float(values.mean()),
        "median_mm": float(np.median(values)),
        "sd_mm": float(values.std()),
        "sd_pct": float(100.0 * values.std() / values.mean()),
    }


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main() -> None:
    positions, support, used = triangulate([IDX[name] for name in ORDER])
    report: dict = {}

    for subject in range(positions.shape[0]):
        pos = positions[subject]
        have = np.isfinite(pos).all(axis=2)
        needed = sorted({SLOT[a] for _, a, b, _ in SEGMENTS} | {SLOT[b] for _, a, b, _ in SEGMENTS})
        common = have[:, needed].all(axis=1) & used[:, subject]
        if common.sum() < 10:
            report[f"subject_{subject:02d}"] = {"frames_used": int(common.sum()),
                                                "note": "too few common frames to score"}
            continue

        lengths = {
            label: summarise(
                np.linalg.norm(pos[common, SLOT[a]] - pos[common, SLOT[b]], axis=1) * 1000.0
            ) | {"anatomical_mm": expected}
            for label, a, b, expected in SEGMENTS
        }

        rotations = {}
        for label, a, b in AXES:
            v = pos[:, SLOT[b]] - pos[:, SLOT[a]]
            ok = common & np.isfinite(v).all(axis=1)
            u = np.full_like(v, np.nan)
            u[ok] = unit(v[ok])
            idx = np.flatnonzero(ok)
            pairs = idx[1:][np.diff(idx) == 1]
            ang = np.degrees(np.arccos(np.clip(
                np.einsum("ni,ni->n", u[pairs], u[pairs - 1]), -1.0, 1.0)))
            rotations[label] = {"n": int(ang.size), "median_deg": float(np.median(ang)),
                                "p95_deg": float(np.percentile(ang, 95)),
                                "max_deg": float(ang.max())}

        ankles = {}
        for side, shin, foot, toe in ANKLES:
            down = unit(pos[common, SLOT[foot]] - pos[common, SLOT[shin]])
            fwd = unit(pos[common, SLOT[toe]] - pos[common, SLOT[foot]])
            ang = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", down, fwd), -1.0, 1.0)))
            ankles[side] = {"median_deg": float(np.median(ang)),
                            "sd_deg": float(ang.std()),
                            "p05_deg": float(np.percentile(ang, 5)),
                            "p95_deg": float(np.percentile(ang, 95)),
                            "frames_beyond_50_130_deg": int(((ang < 50) | (ang > 130)).sum()),
                            "frames": int(common.sum())}

        report[f"subject_{subject:02d}"] = {
            "frames_used": int(common.sum()),
            "frames_accepted_by_pipeline": int(used[:, subject].sum()),
            "camera_support_mean": {n: float(support[subject, common, SLOT[n]].mean()) for n in ORDER},
            "segment_length_stability": lengths,
            "axis_frame_to_frame_rotation": rotations,
            "ankle_angle_shin_to_foot_axis": ankles,
        }

    Path("artifacts/feet-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/feet-lane/toe-gate.json").write_text(json.dumps(report, indent=2))

    for subject, block in report.items():
        if "note" in block:
            print(f"\n=== {subject}: {block['note']} ==="); continue
        print(f"\n=== {subject}  ({block['frames_used']} common frames of "
              f"{block['frames_accepted_by_pipeline']}) ===")
        print(f"{'segment':38s} {'mean':>8s} {'sd':>8s} {'sd%':>8s}   anatomical")
        for label, s in block["segment_length_stability"].items():
            print(f"{label:38s} {s['mean_mm']:8.1f} {s['sd_mm']:8.1f} {s['sd_pct']:7.1f}%   {s['anatomical_mm']}")
        print(f"{'axis frame-to-frame rotation':38s} {'median':>8s} {'p95':>8s} {'max':>8s}")
        for label, r in block["axis_frame_to_frame_rotation"].items():
            print(f"{label:38s} {r['median_deg']:8.2f} {r['p95_deg']:8.2f} {r['max_deg']:8.2f}")
        for side, a in block["ankle_angle_shin_to_foot_axis"].items():
            print(f"ankle {side}: median {a['median_deg']:.1f}deg sd {a['sd_deg']:.1f} "
                  f"p05-p95 {a['p05_deg']:.1f}-{a['p95_deg']:.1f}, "
                  f"{a['frames_beyond_50_130_deg']}/{a['frames']} frames outside 50-130deg")


if __name__ == "__main__":
    main()
