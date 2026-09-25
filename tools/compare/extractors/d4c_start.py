#!/usr/bin/env python3
"""Ladder extractor for D4c -- the calibration's start: the fitted body's REST lengths against synthetic truth.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the registrations collide).
This file supplies the `x_*`-shaped function and the proposed `VISUALS` entry. To register it, add to `ladder.py`:

    from extractors.d4c_start import x_body_model_start

and route its figures to rung 7 (the converter), beside D4's `bodymodel_o1_*` and D4b's `bodymodel_o1b_*`.

It reads exactly one report, `artifacts/compare/d4c-start/gate.json` (`tools/compare/d4c_start_gate.py`).

ONE REFERENCE: SYNTHETIC MHR TRUTH -- twelve bodies (six new seeds x two donors) whose identity is a draw on the
limit-aware drawn set (eight channels, the trunk scored), MAMMA-FREE. Every figure is a RATIO to its own paired
tolerance (the sum of the segment's two endpoint floors on the same fixture): 1.0 is the band edge.

WHAT BELONGS ON THE PAGE BESIDE THE BARS, FIRST: **D4c: FAIL on L -- the trunk misses its tolerance on ONE of
twelve fixtures (20261106 donor 0: 2.13 mm against 1.94 mm, 1.10x); every other conjunct holds; D4 stays open.**
The landmark start moved the trunk from 3/18 within tolerance (zero start) to 15/18 on the development set and 11/12
on the untouched acceptance set, B1 held (+0.156 / +0.118 over the rig), and the fit_one change stays on the branch.

Self-check:  python3 tools/compare/extractors/d4c_start.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d4c-start/gate.json"
REF_SYNTH = ("SYNTHETIC MHR TRUTH: twelve bodies (six new seeds x two donors), identity drawn on the limit-aware "
             "drawn set, the donor pose clamped to the model's own limits. Each figure is |fitted - truth| rest length "
             "over its paired tolerance; 1.0 is the band edge. MAMMA-FREE")


def _worst(segments: dict, arm: str, names: list[str]) -> float | None:
    ratios = [row[arm][n]["error_mm"] / row[arm][n]["tolerance_mm"] for row in segments.values() for n in names
              if row.get(arm, {}).get(n, {}).get("tolerance_mm")]
    return round(max(ratios), 4) if ratios else None


def _closest(segments: dict, arm: str, names: list[str]) -> float | None:
    per = []
    for row in segments.values():
        ratios = [row[arm][n]["error_mm"] / row[arm][n]["tolerance_mm"] for n in names
                  if row.get(arm, {}).get(n, {}).get("tolerance_mm")]
        if ratios:
            per.append(max(ratios))
    return round(min(per), 4) if per else None


def x_body_model_start(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    segments = report.get("reported", {}).get("segments_all_arms", {})
    scored = report.get("segments_scored", [])
    limbs = [s for s in scored if s not in ("trunk", "neck_head")]
    verdict = f"Gate: {report.get('verdict')} ({report.get('reason')})"
    figs = [
        fig("D4c L: the fitted trunk (landmark start), worst of twelve", _worst(segments, "candidate", ["trunk"]),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_start_trunk", note=verdict),
        fig("D4c L: the fitted limbs (landmark start), worst of twelve", _worst(segments, "candidate", limbs),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_start_limbs", note=verdict),
    ]
    ctrls = [
        fig("D4c before arm: the zero start (the D4 fitter), trunk, worst of twelve",
            _worst(segments, "legacy", ["trunk"]), "x tolerance", REF_SYNTH, LOWER, key="bodymodel_start_alt_legacy",
            note="REPORTED, never banded: the fitter D4 shipped, on the same fixtures"),
        fig("D4c must-fail (iv): the start held with no calibration, closest of twelve",
            _closest(segments, "init_only", scored), "x tolerance", REF_SYNTH, LOWER, key="bodymodel_start_ctrl_init",
            note="must exceed 1.0 on every fixture (else the oracle cannot score the calibration)"),
        fig("D4c must-fail (i): the MEAN body, closest of twelve", _closest(segments, "mean_body", scored),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_start_ctrl_mean_body", note="must exceed 1.0 everywhere"),
        fig("D4c must-fail (ii): the spine displaced 0.149 units, trunk (scored), closest of twelve",
            _closest(segments, "spine_displaced", ["trunk"]), "x tolerance", REF_SYNTH, LOWER,
            key="bodymodel_start_ctrl_spine", note="the trunk IS scored under the limit-aware rule; must exceed 1.0"),
    ]
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D4c: does starting the body fit from the measured bone lengths get the proportions right?",
             plain="Almost, not quite: the trunk is now right on eleven of twelve test bodies (the old fitter got "
                   "two), but the one miss fails the registered test, so this does not close D4. Each bar is the "
                   "fitted bone length's error divided by what the pose solve alone already costs -- lower is "
                   "better, 1.0 is the edge.",
             better="lower",
             bars=[dict(label="Our fit, the trunk (worst of 12)", role="ours", key="bodymodel_start_trunk"),
                   dict(label="Our fit, arms and legs (worst)", role="ours", key="bodymodel_start_limbs"),
                   dict(label="The old fitter (zero start), the trunk", role="alt", key="bodymodel_start_alt_legacy"),
                   dict(label="Starting guess only, no fit (deliberately wrong)", role="control",
                        key="bodymodel_start_ctrl_init"),
                   dict(label="No body fit at all (deliberately wrong)", role="control",
                        key="bodymodel_start_ctrl_mean_body"),
                   dict(label="Spine shortened on purpose (deliberately wrong)", role="control",
                        key="bodymodel_start_ctrl_spine")]),
    ],
}


if __name__ == "__main__":
    figures, controls = x_body_model_start({})
    for item in figures + controls:
        print(f"{item['key']:34s} {item['value']!s:>10} {item['unit']}  {item['note'][:90]}")
    keys = {item["key"] for item in figures + controls}
    missing = [bar["key"] for chart in VISUALS["converter"] for bar in chart["bars"] if bar["key"] not in keys]
    print("VISUALS keys resolved:", not missing, missing)
