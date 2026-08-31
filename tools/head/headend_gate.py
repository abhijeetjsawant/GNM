#!/usr/bin/env python3
"""Is `HeadEnd` a usable skull axis, or is it the fingers again?

`docs/HEAD_FEET_HANDS_PLAN.md` §2 calls HeadEnd "the single most promising head
lead on this page" and marks it UNMEASURED. This measures it, deliberately
copying the harness shape of `docs/FINGER_TRIANGULATION_GATE.md`, whose lesson
was that **reprojection flatters a bad result and the physical invariant is the
verdict**. Fingers passed reprojection at 1.2x the body control and failed
bone-length stability at 4.4x. So the verdict here is segment-length stability,
with reprojection reported but never decisive.

Same denominator throughout: every quantity below is triangulated by
`triangulate_point` at production settings, on the pipeline's own association,
restricted to **frames where every quantity being compared is available**. A
candidate that resolves on 40 easy frames must not be compared against a control
that resolves on 150.

Blind to:
  * ACCURACY. A stable segment can be stably in the wrong place -- SOMA-77's
    HeadEnd is a skeletal convention, not a measured skull apex, and this says
    nothing about where the real skull top is.
  * Whether the detector is *measuring* HeadEnd or predicting it from the crop.
    Cross-view geometric consistency is the discriminator available here, which
    is why the invariant is scored across four independently-viewed cameras.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triangulate_soma import triangulate  # noqa: E402

# SOMA-77 indices, read from src/autoanim_gnm/data/somaskel77-v1.json.
IDX = {
    "Chest": 3, "Neck1": 4, "Neck2": 5, "Head": 6, "HeadEnd": 7, "Jaw": 8,
    "LeftEye": 9, "RightEye": 10,
    "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14,
    "RightArm": 40, "RightForeArm": 41, "RightHand": 42,
    "LeftLeg": 67, "LeftShin": 68, "LeftFoot": 69,
}
ORDER = list(IDX)
SLOT = {name: i for i, name in enumerate(ORDER)}

# (name, a, b, anatomical range in mm or None) -- the head candidates, then the
# body controls this lane already trusts, then the incumbent head instrument.
SEGMENTS = [
    ("HeadEnd  Head->HeadEnd", "Head", "HeadEnd", "skull half-height, ~110-140"),
    ("Jaw      Head->Jaw", "Head", "Jaw", "~60-90"),
    ("Neck     Neck2->Head", "Neck2", "Head", "~80-110"),
    ("Neck1    Chest->Neck1", "Chest", "Neck1", "~100-160"),
    ("CONTROL  forearm L", "LeftForeArm", "LeftHand", "~240-270"),
    ("CONTROL  forearm R", "RightForeArm", "RightHand", "~240-270"),
    ("CONTROL  upper arm L", "LeftArm", "LeftForeArm", "~270-290"),
    ("CONTROL  shin L", "LeftShin", "LeftFoot", "~390-410"),
    ("INCUMBENT eye baseline", "LeftEye", "RightEye", "~63"),
]

# Direction pairs whose frame-to-frame rotation is the orientation signal.
AXES = [
    ("skull long axis  Head->HeadEnd", "Head", "HeadEnd"),
    ("skull fore axis  Head->Jaw", "Head", "Jaw"),
    ("INCUMBENT eye axis  R->L", "RightEye", "LeftEye"),
    ("CONTROL shoulder axis", "RightArm", "LeftArm"),
    ("CONTROL forearm L", "LeftForeArm", "LeftHand"),
]


def summarise(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean_mm": float(values.mean()),
        "median_mm": float(np.median(values)),
        "sd_mm": float(values.std()),
        "sd_pct": float(100.0 * values.std() / values.mean()),
    }


def main() -> None:
    positions, support, used = triangulate([IDX[name] for name in ORDER])
    report: dict = {}

    for subject in range(positions.shape[0]):
        pos = positions[subject]
        have = np.isfinite(pos).all(axis=2)  # [frame, slot]

        # Same denominator: one common frame set on which EVERY landmark below
        # resolved. Otherwise an easy landmark is scored on easy frames.
        needed = sorted({SLOT[a] for _, a, b, _ in SEGMENTS} | {SLOT[b] for _, a, b, _ in SEGMENTS})
        common = have[:, needed].all(axis=1) & used[:, subject]

        lengths = {}
        for label, a, b, expected in SEGMENTS:
            d = np.linalg.norm(pos[common, SLOT[a]] - pos[common, SLOT[b]], axis=1) * 1000.0
            lengths[label] = summarise(d) | {"anatomical_mm": expected}

        rotations = {}
        for label, a, b in AXES:
            v = pos[:, SLOT[b]] - pos[:, SLOT[a]]
            ok = common & np.isfinite(v).all(axis=1)
            unit = np.full_like(v, np.nan)
            unit[ok] = v[ok] / np.linalg.norm(v[ok], axis=1, keepdims=True)
            # Consecutive COMMON frames only, so a gap is not read as a jump.
            idx = np.flatnonzero(ok)
            pairs = idx[1:][np.diff(idx) == 1]
            dots = np.einsum("ni,ni->n", unit[pairs], unit[pairs - 1])
            ang = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
            rotations[label] = {
                "n": int(ang.size),
                "median_deg": float(np.median(ang)),
                "p95_deg": float(np.percentile(ang, 95)),
                "max_deg": float(ang.max()),
            }

        # The §1b failure mode, re-scored for the skull axis: an axis that
        # opposes the body cannot be right. For the eye axis the reference is
        # the shoulders; for the skull long axis it is world up (+Z), since a
        # head cannot point downward through the neck while the person stands.
        eye = pos[:, SLOT["LeftEye"]] - pos[:, SLOT["RightEye"]]
        shoulder = pos[:, SLOT["LeftArm"]] - pos[:, SLOT["RightArm"]]
        skull = pos[:, SLOT["HeadEnd"]] - pos[:, SLOT["Head"]]
        neck_up = pos[:, SLOT["Head"]] - pos[:, SLOT["Neck2"]]

        def unit(v):
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        eye_dot = np.einsum("ni,ni->n", unit(eye[common]), unit(shoulder[common]))
        skull_dot = np.einsum("ni,ni->n", unit(skull[common]), unit(neck_up[common]))
        skull_up = unit(skull[common])[:, 2]

        report[f"subject_{subject:02d}"] = {
            "frames_used": int(common.sum()),
            "frames_accepted_by_pipeline": int(used[:, subject].sum()),
            "camera_support_mean": {
                name: float(support[subject, common, SLOT[name]].mean()) for name in ORDER
            },
            "segment_length_stability": lengths,
            "axis_frame_to_frame_rotation": rotations,
            "sign_sanity": {
                "eye_axis_opposes_shoulders_frames": int((eye_dot < 0).sum()),
                "skull_axis_opposes_neck_frames": int((skull_dot < 0).sum()),
                "skull_axis_points_below_horizon_frames": int((skull_up < 0).sum()),
                "frames": int(common.sum()),
            },
        }

    Path("artifacts/head-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/head-lane/headend-gate.json").write_text(json.dumps(report, indent=2))

    for subject, block in report.items():
        print(f"\n=== {subject}  ({block['frames_used']} common frames of "
              f"{block['frames_accepted_by_pipeline']}) ===")
        print(f"{'segment':32s} {'mean':>8s} {'sd':>8s} {'sd%':>8s}   anatomical")
        for label, s in block["segment_length_stability"].items():
            print(f"{label:32s} {s['mean_mm']:8.1f} {s['sd_mm']:8.1f} {s['sd_pct']:7.1f}%   {s['anatomical_mm']}")
        print(f"{'axis frame-to-frame rotation':32s} {'median':>8s} {'p95':>8s} {'max':>8s}")
        for label, r in block["axis_frame_to_frame_rotation"].items():
            print(f"{label:32s} {r['median_deg']:8.2f} {r['p95_deg']:8.2f} {r['max_deg']:8.2f}")
        print("sign sanity:", json.dumps(block["sign_sanity"]))


if __name__ == "__main__":
    main()
