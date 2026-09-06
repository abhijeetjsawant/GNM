#!/usr/bin/env python3
"""D8c's B4: the frames-off count on a FIXED denominator, and on the moving one beside it.

THE DEFECT THIS EXISTS FOR is in CLAUDE.md: "`captured_limb_stability.py` recomputes the
performer's median per build -- a moving denominator; compare builds with a fixed
reference." A rule that withholds the frames furthest from the median MOVES the median, so
"frames off by more than 15 %" measured against each build's own median can fall for two
quite different reasons: because the bad frames were repaired, or because the reference
slid toward them. Only the first is the step working.

So every segment is measured three ways and all three are reported:

  off_before_own_median      the BEFORE build against its own median   (the D8b figure)
  off_after_own_median       the AFTER build against its own median    (moving denominator)
  off_after_before_median    the AFTER build against the BEFORE build's median  <-- B4

B4's band is read on the third. The card's must-fail is on the first: the before build must
read 23 on performer 1's hip line, or the band is being read against the wrong baseline.

THE CANDIDATE OPTIMISES THIS DIRECTLY -- it is a count of the very frames the rule fires on
-- which is why the card puts it in the merge rule ONLY PAIRED with the photographs. Two
degenerates it does catch, and they are the reason it is in the merge rule at all: a
recovery that leaves the hips on the inward rays passes every other band and fails this,
and a smoothing smear that pulls the honest frames toward a new median fails it on the
FIXED medians while passing on the moving ones.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_fixed_denominator.py

Writes `artifacts/compare/d8c-hip/b4-fixed-denominator.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402

OUT = ROOT / "artifacts/compare/d8c-hip/b4-fixed-denominator.json"
BEFORE = ROOT / "artifacts/commercial-multiview-soma77"
AFTER = ROOT / "artifacts/compare/d8c-hip/delivery"
FIRST_FRAME_ID = 60
CEILING = 0.15

SEGMENTS = (
    ("shoulder_line", "left_shoulder", "right_shoulder"),
    ("upper_arm_L", "left_shoulder", "left_elbow"),
    ("forearm_L", "left_elbow", "left_wrist"),
    ("upper_arm_R", "right_shoulder", "right_elbow"),
    ("forearm_R", "right_elbow", "right_wrist"),
    ("hip_line", "left_hip", "right_hip"),
    ("thigh_L", "left_hip", "left_knee"),
    ("shin_L", "left_knee", "left_ankle"),
    ("thigh_R", "right_hip", "right_knee"),
    ("shin_R", "right_knee", "right_ankle"),
)
ARRAYS = {"smoothed": "triangulated_world_positions_z_up_m",
          "raw": "raw_triangulated_world_positions_z_up_m"}


def lengths(positions: np.ndarray, a: str, b: str) -> np.ndarray:
    return 1000.0 * np.linalg.norm(positions[:, cm.JOINT_INDEX[a]]
                                   - positions[:, cm.JOINT_INDEX[b]], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--before", type=Path, default=BEFORE)
    parser.add_argument("--after", type=Path, default=AFTER)
    parser.add_argument("--before-label", default="D8b")
    parser.add_argument("--after-label", default="D8c")
    args = parser.parse_args()
    before_dir = args.before if args.before.is_absolute() else ROOT / args.before
    after_dir = args.after if args.after.is_absolute() else ROOT / args.after

    report: dict = {
        "title": "D8c B4 -- captured segment lengths off by more than 15 %, on a FIXED "
                 "denominator and on the moving one",
        "before": {"label": args.before_label, "path": str(before_dir.relative_to(ROOT))},
        "after": {"label": args.after_label, "path": str(after_dir.relative_to(ROOT))},
        "ceiling_fraction": CEILING,
        "how_to_read_it": (
            "`off_after_before_median` is B4: the AFTER build scored against the BEFORE "
            "build's own median, so the reference cannot slide under the count. "
            "`off_after_own_median` is the moving-denominator reading and is beside it, "
            "never instead of it"),
        "blind_to": [
            "TRUTH -- a limb welded to its own median scores zero here",
            "DIRECTION -- a length invariant cannot score direction",
            "and the candidate OPTIMISES this directly, which is why the card pairs it with "
            "the photographs in the merge rule rather than letting it stand alone",
        ],
        "arrays": {},
    }

    for array, key in ARRAYS.items():
        block: dict = {}
        for subject in (0, 1):
            with np.load(before_dir / f"subject-{subject:02d}.body-track.npz") as a, \
                 np.load(after_dir / f"subject-{subject:02d}.body-track.npz") as b:
                before = np.asarray(a[key], dtype=np.float64)
                after = np.asarray(b[key], dtype=np.float64)
            for name, first, second in SEGMENTS:
                lb = lengths(before, first, second)
                la = lengths(after, first, second)
                fb, fa = np.isfinite(lb), np.isfinite(la)
                mb = float(np.median(lb[fb])) if fb.any() else float("nan")
                ma = float(np.median(la[fa])) if fa.any() else float("nan")
                off_b = fb & (np.abs(lb - mb) / mb > CEILING)
                off_a_own = fa & (np.abs(la - ma) / ma > CEILING)
                off_a_fixed = fa & (np.abs(la - mb) / mb > CEILING)
                block[f"subject_{subject:02d}/{name}"] = {
                    f"median_{args.before_label}_mm": round(mb, 2),
                    f"median_{args.after_label}_mm": round(ma, 2),
                    "median_shift_mm": round(ma - mb, 3),
                    "off_before_own_median": int(off_b.sum()),
                    "off_after_own_median": int(off_a_own.sum()),
                    "off_after_before_median": int(off_a_fixed.sum()),
                    "off_before_ids": [i + FIRST_FRAME_ID
                                       for i in np.flatnonzero(off_b).tolist()],
                    "off_after_fixed_ids": [i + FIRST_FRAME_ID
                                            for i in np.flatnonzero(off_a_fixed).tolist()],
                    "frames_measured_before": int(fb.sum()),
                    "frames_measured_after": int(fa.sum()),
                }
        report["arrays"][array] = block

    smoothed = report["arrays"]["smoothed"]
    report["B4"] = {
        "band": {"subject_01/hip_line": "23 -> <= 4 on the FIXED denominator",
                 "subject_00/hip_line": "0 -> 0"},
        "must_fail": f"the {args.before_label} build must read 23 on subject_01/hip_line",
        "measured": {
            "subject_01/hip_line": {
                "before_own_median": smoothed["subject_01/hip_line"][
                    "off_before_own_median"],
                "after_fixed_denominator": smoothed["subject_01/hip_line"][
                    "off_after_before_median"],
                "after_moving_denominator": smoothed["subject_01/hip_line"][
                    "off_after_own_median"]},
            "subject_00/hip_line": {
                "before_own_median": smoothed["subject_00/hip_line"][
                    "off_before_own_median"],
                "after_fixed_denominator": smoothed["subject_00/hip_line"][
                    "off_after_before_median"],
                "after_moving_denominator": smoothed["subject_00/hip_line"][
                    "off_after_own_median"]},
        },
    }
    report["B4"]["must_fail_holds"] = bool(
        smoothed["subject_01/hip_line"]["off_before_own_median"] == 23)
    report["B4"]["passes"] = bool(
        report["B4"]["must_fail_holds"]
        and smoothed["subject_01/hip_line"]["off_after_before_median"] <= 4
        and smoothed["subject_00/hip_line"]["off_after_before_median"] == 0)

    for array in ARRAYS:
        print(f"\n{array.upper()}   (ceiling {CEILING})")
        print(f"  {'subject/segment':<26}{'med'+args.before_label:>10}"
              f"{'med'+args.after_label:>10}{'off before':>12}{'off after':>11}"
              f"{'off@fixed':>11}")
        for name, row in report["arrays"][array].items():
            print(f"  {name:<26}{row['median_'+args.before_label+'_mm']:>10.2f}"
                  f"{row['median_'+args.after_label+'_mm']:>10.2f}"
                  f"{row['off_before_own_median']:>12d}{row['off_after_own_median']:>11d}"
                  f"{row['off_after_before_median']:>11d}")
    print("\nB4:", json.dumps(report["B4"], indent=1))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if report["B4"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
