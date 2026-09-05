#!/usr/bin/env python3
"""D7, step 1: are SOMA-77's lower-spine landmarks rigid to the pelvis on the REAL take?

INSTRUMENT ONLY. Nothing ships from here, and the pelvis frame is not trusted until this
has been read. It is the instrument that killed `HeadEnd` (66.5 % / 115.0 % length
variation against body controls at 2.5-4.2 %) and passed `ToeBase` (5.0-9.6 %):
**segment-length stability over the take against the body controls this lane already
trusts**, on one common frame set.

READING RULE, fixed in the pre-registration before these numbers existed
(`docs/reviews/pelvis-frame-2026-09-04.md` section 0.5, band B3):

    sd % punishes a short lever arithmetically -- 3 mm on a 43 mm segment reads 7 % where
    10 mm on a 400 mm shin reads 2.5 %. THE VERDICT IS TAKEN ON sd_mm against the
    controls' sd_mm; sd % is quoted beside it and is NOT the verdict. A candidate whose
    sd_mm exceeds the worst body control's sd_mm is not trusted.

WRAP THE PIPELINE, NEVER RE-IMPLEMENT IT. The cross-view association comes from
`tools/head/associate.recover()`, which runs the real `reconstruct_multiview` with a
recording associator and refuses to return unless the replay reproduces the retained
`raw_triangulated_world_positions_z_up_m` exactly. A hand replication of that loop drifted
9-19 mm. The triangulation is `triangulate_point` at production settings through
`tools/head/triangulate_soma.triangulate`, so no landmark here enjoys a gate the mapped
body joints do not.

BLIND TO:
  * ACCURACY. A stable segment can be stably wrong. Nothing here says where a real
    L5/S1 is, only that the detector puts its `Spine1` in a stable place relative to the
    pelvis.
  * CROSS-VIEW SELF-AGREEMENT IS NOT RIGIDITY. Apple Vision's ears were the most
    epipolar-consistent landmarks measured in this lane and fit a rigid skull worst.
    This is a between-frame instrument on purpose.
  * Whether the detector MEASURES `Spine1` or predicts it from the crop.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_pelvis_rigidity.py

Writes `artifacts/compare/d7-pelvis-frame/rigidity.json` and the association it recovered
under the same directory. It never writes to `artifacts/head-lane` or to
`artifacts/commercial-multiview-soma77`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src."
    )

import associate  # noqa: E402
import triangulate_soma as ts  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"

# SOMA-77 indices, read from src/autoanim_gnm/data/somaskel77-v1.json.
IDX = {
    "Hips": 0, "Spine1": 1, "Spine2": 2, "Chest": 3, "Neck2": 5,
    "LeftLeg": 67, "LeftShin": 68, "LeftFoot": 69,
    "RightLeg": 72, "RightShin": 73, "RightFoot": 74,
    "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14,
}
ORDER = list(IDX)
SLOT = {name: i for i, name in enumerate(ORDER)}

# (label, a, b) with a virtual point allowed as `a`. The controls are the same ones
# HeadEnd and ToeBase were measured against, so all three verdicts sit on one axis.
SEGMENTS = [
    ("CANDIDATE  root->Spine1", "Hips", "Spine1"),
    ("CANDIDATE  midhips->Spine1", "@hipmid", "Spine1"),
    ("CANDIDATE  root->Spine2", "Hips", "Spine2"),
    ("REPORTED   Spine1->Spine2", "Spine1", "Spine2"),
    ("CONTROL    shin L", "LeftShin", "LeftFoot"),
    ("CONTROL    shin R", "RightShin", "RightFoot"),
    ("CONTROL    thigh L", "LeftLeg", "LeftShin"),
    ("CONTROL    thigh R", "RightLeg", "RightShin"),
    ("CONTROL    forearm L", "LeftForeArm", "LeftHand"),
    ("CONTROL    upper arm L", "LeftArm", "LeftForeArm"),
    ("CONTROL    hip span", "LeftLeg", "RightLeg"),
]


def point(pos: np.ndarray, name: str) -> np.ndarray:
    if name == "@hipmid":
        return 0.5 * (pos[:, SLOT["LeftLeg"]] + pos[:, SLOT["RightLeg"]])
    return pos[:, SLOT[name]]


def summarise(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean_mm": round(float(values.mean()), 2),
        "median_mm": round(float(np.median(values)), 2),
        "sd_mm": round(float(values.std()), 3),
        "sd_pct": round(float(100.0 * values.std() / values.mean()), 2),
        "iqr_mm": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 3),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assignment, used, association_report = associate.recover()
    np.savez(OUT_DIR / "association.npz", assignment=assignment, used=used)
    # `triangulate_soma.triangulate` reads OUT/"association.npz"; point it at OURS so
    # nothing under artifacts/head-lane is touched, and the real function still runs.
    ts.OUT = OUT_DIR
    positions, support, used2 = ts.triangulate([IDX[name] for name in ORDER])

    report: dict = {
        "instrument": "segment-length stability over the take against body controls -- the "
                      "HeadEnd/ToeBase instrument",
        "reference": "NONE. This is a reference-free INVARIANT: a segment between two points "
                     "rigid to one bone has constant length whatever the pose. It scores "
                     "STABILITY, never accuracy.",
        "reading_rule": (
            "the verdict is on sd_mm against the controls' sd_mm. sd_pct is quoted beside it "
            "and is NOT the verdict: sd % punishes a short lever arithmetically (3 mm on a "
            "43 mm segment reads 7 %, 10 mm on a 400 mm shin reads 2.5 %). Fixed in the "
            "pre-registration before these numbers existed."),
        "association_provenance": association_report,
        "subjects": {},
    }
    for subject in range(positions.shape[0]):
        pos = positions[subject]
        have = np.isfinite(pos).all(axis=2)
        needed = sorted({SLOT[n] for _, a, b in SEGMENTS for n in (a, b) if n != "@hipmid"}
                        | {SLOT["LeftLeg"], SLOT["RightLeg"]})
        common = have[:, needed].all(axis=1) & used2[:, subject]
        if common.sum() < 10:
            report["subjects"][f"subject_{subject:02d}"] = {
                "frames_used": int(common.sum()), "note": "too few common frames to score"}
            continue
        lengths = {}
        for label, a, b in SEGMENTS:
            values = np.linalg.norm(
                point(pos[common], a) - point(pos[common], b), axis=1) * 1000.0
            lengths[label] = summarise(values)
        controls = [v["sd_mm"] for k, v in lengths.items() if k.startswith("CONTROL")]
        body_controls = [v["sd_mm"] for k, v in lengths.items()
                         if k.startswith("CONTROL") and "hip span" not in k]
        verdicts = {}
        for label, values in lengths.items():
            if not label.startswith("CANDIDATE"):
                continue
            verdicts[label] = {
                "sd_mm": values["sd_mm"],
                "sd_pct": values["sd_pct"],
                "worst_body_control_sd_mm": round(float(max(body_controls)), 3),
                "best_body_control_sd_mm": round(float(min(body_controls)), 3),
                "trusted_on_sd_mm": bool(values["sd_mm"] <= max(body_controls)),
                "ratio_to_worst_body_control": round(
                    float(values["sd_mm"] / max(body_controls)), 3),
            }
        report["subjects"][f"subject_{subject:02d}"] = {
            "frames_used": int(common.sum()),
            "frames_accepted_by_pipeline": int(used2[:, subject].sum()),
            "camera_support_mean": {n: round(float(support[subject, common, SLOT[n]].mean()), 3)
                                    for n in ORDER},
            "segment_length_stability": lengths,
            "verdict_on_sd_mm": verdicts,
            "all_control_sd_mm": [round(float(v), 3) for v in controls],
        }
    (OUT_DIR / "rigidity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    for name, block in report["subjects"].items():
        if "note" in block:
            print(f"\n=== {name}: {block['note']} ===")
            continue
        print(f"\n=== {name}  ({block['frames_used']} common frames of "
              f"{block['frames_accepted_by_pipeline']} accepted) ===")
        print(f"{'segment':30s} {'mean':>9s} {'sd_mm':>9s} {'sd%':>7s} {'support':>8s}")
        for label, s in block["segment_length_stability"].items():
            print(f"{label:30s} {s['mean_mm']:9.1f} {s['sd_mm']:9.2f} {s['sd_pct']:6.1f}%")
        for label, v in block["verdict_on_sd_mm"].items():
            flag = "TRUSTED" if v["trusted_on_sd_mm"] else "NOT TRUSTED"
            print(f"  {label:28s} sd {v['sd_mm']:6.2f} mm vs worst body control "
                  f"{v['worst_body_control_sd_mm']:6.2f} mm  ({v['ratio_to_worst_body_control']:.2f}x) -> {flag}")
    print(f"\nwrote {OUT_DIR / 'rigidity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
