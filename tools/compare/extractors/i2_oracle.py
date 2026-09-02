#!/usr/bin/env python3
"""Ladder extractor for I2, the perfect-2D oracle.

Registry-ready but NOT registered: `tools/compare/ladder.py` has one owner, and two
agents editing it collide. Drop `x_oracle_2d` into the rung-4/5/7 row (or its own
"pipeline floor" row) there; this file only has to keep the shape.

The figure this contributes is a FLOOR, not a score. It belongs beside the pose and
delivered rungs -- it shares their reference exactly, MAMMA's `pred_joints` on the same
15 joints and the same 150 frames -- and it answers the question those rungs cannot:
of the 36-41 mm our capture sits at, how much is the reconstruction itself? On this
fixture, 1.1 mm. The rest is upstream of 2D.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOWER = "lower is better"
HIGHER = "higher is better"


def _load(rel: str) -> dict | None:
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def fig(label: str, value: float | int | None, unit: str, reference: str, better: str = LOWER,
        key: str | None = None, note: str = "") -> dict:
    return {"key": key or label, "label": label, "value": value, "unit": unit,
            "reference": reference, "better": better, "note": note}


def x_oracle_2d(_: dict) -> tuple[list, list]:
    r = _load("artifacts/compare/oracle-2d.json")
    if not r:
        return [], []
    ref = ("MAMMA's pred_joints, 15 body joints, 150 frames -- the SAME reference as the "
           "pose and delivered rungs; the 2D is MAMMA's own skeleton reprojected, so the "
           "detector is absent by construction")
    exact = r["arms"]["exact"]
    noise = r["arms"]["noise"]["across_seeds"]
    figs = []
    for subject, entry in sorted(exact["subjects"].items()):
        tag = subject.replace("our_subject_", "subject ")
        figs.append(fig(f"pipeline floor on perfect 2D, absolute, {tag}",
                        entry["absolute"]["median_mm"], "mm median", ref,
                        key=f"floor_abs_{subject}",
                        note="block-bootstrap 95% CI "
                             f"{entry['absolute'].get('median_ci95_mm')} mm"))
        figs.append(fig(f"pipeline floor on perfect 2D, root-relative, {tag}",
                        entry["relative"]["median_mm"], "mm median", ref,
                        key=f"floor_rel_{subject}",
                        note="block-bootstrap 95% CI "
                             f"{entry['relative'].get('median_ci95_mm')} mm"))
    before = exact.get("before_the_temporal_stage")
    if before:
        figs.append(fig("of that floor, what raw triangulation costs (same run, pre-fill, "
                        "pre-Savitzky-Golay), absolute",
                        before["overall"]["absolute"]["median_mm"], "mm median", ref,
                        key="floor_before_temporal",
                        note="the difference against the figures above is what the sequence "
                             "solve, the fill and the smoothing window cost; coverage here is "
                             f"{before['overall']['absolute']['coverage']}, its own, because "
                             "the NaNs are intact"))
    figs.append(fig("pipeline floor, absolute, p95 over both subjects",
                    exact["overall"]["absolute"]["p95_mm"], "mm p95", ref, key="floor_abs_p95"))
    figs.append(fig(f"with MAMMA-grade 2D noise injected, absolute, mean of "
                    f"{r['arms']['noise']['seeds']} seeds",
                    noise["absolute"]["median_mm"]["mean"], "mm median", ref, key="noise_abs",
                    note=f"sd {noise['absolute']['median_mm']['sd']} mm across seeds, "
                         f"range {noise['absolute']['median_mm']['min']}-"
                         f"{noise['absolute']['median_mm']['max']}; MAMMA's residual is "
                         "regularised toward its own landmarks and injected i.i.d., so this "
                         "is a lower bound on what a real detector of that class costs"))
    figs.append(fig("coverage on the 15 scored joints",
                    exact["overall"]["absolute"]["coverage"], "fraction",
                    "same denominator on every arm", HIGHER, key="floor_coverage"))

    controls = r["controls"]
    ctrls = []
    for name, label, note in (
        ("shuffled_subject_pairing", "control: the exact arm scored against the OTHER performer",
         "MAMMA's body_id-00 is our subject 1; pairing by index would have landed here"),
        ("frozen_skeleton", "control: frame 0's skeleton projected on every frame (a constant)",
         "it reconstructs at 0.0 px reprojection and agrees with nothing -- the demonstration "
         "that no reprojection figure is an accuracy figure"),
        ("time_shift_+1", "control: camera A001 advanced +1 frame", ""),
        ("time_shift_-1", "control: camera A001 delayed -1 frame",
         "the weakest of the four controls, and it costs about what MAMMA-grade 2D noise "
         "costs; the two ratios below are like against like"),
    ):
        control = controls.get(name)
        if not control:
            continue
        ctrls.append(fig(label, control["overall"]["absolute"]["median_mm"], "mm median", ref,
                         HIGHER, key=name,
                         note=(note + (" | " if note else "") +
                               f"{control['ratio_to_oracle_floor']['absolute']}x the floor "
                               f"absolute, {control['ratio_to_oracle_floor']['relative']}x "
                               "root-relative")))
    ctrls.append(fig("projection convention check: MAMMA's 512 fitted landmarks reprojected "
                     "through our rig vs its own ma_2d, median",
                     r["projection"]["dense_residual_px_native"]["values"][2],
                     "px at 3840 (a sanity check, NOT a score)",
                     "MAMMA's ma_2d", LOWER, key="projection_check",
                     note="single digits means the convention matches; an axis swap or a "
                          "mirrored principal point reads in the hundreds"))
    return figs, ctrls


def stamp() -> dict:
    path = ROOT / "artifacts/compare/oracle-2d.json"
    if not path.exists():
        return {"path": "artifacts/compare/oracle-2d.json", "exists": False}
    return {"path": "artifacts/compare/oracle-2d.json", "exists": True,
            "mtime": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")}


if __name__ == "__main__":
    figures, controls = x_oracle_2d({})
    for row in figures:
        print(f"  FIG  {row['label']}: {row['value']} {row['unit']}")
    for row in controls:
        print(f"  CTRL {row['label']}: {row['value']} {row['unit']}")
