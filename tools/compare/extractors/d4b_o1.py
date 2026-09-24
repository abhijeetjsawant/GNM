#!/usr/bin/env python3
"""Ladder extractor for D4b -- O1 re-registered prospectively: the fitted body's REST lengths against synthetic truth.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the registrations
collide). This file supplies the `x_*`-shaped function and the proposed `VISUALS` entry. To register it, add
to `ladder.py`:

    from extractors.d4b_o1 import x_body_model_o1

and route its figures to rung 7 (the converter), beside D4's `bodymodel_o1_*`.

It reads exactly one report, `artifacts/compare/d4b-o1/gate.json` (`tools/compare/d4b_o1_gate.py`).

ONE REFERENCE: SYNTHETIC MHR TRUTH -- twelve bodies (six seeds x two donors) whose identity is a draw on the
frozen drawn set, MAMMA-FREE. Every figure is a RATIO to its own paired tolerance (the sum of the segment's two
endpoint floors on the same fixture), so bars from different segments share one unit: 1.0 is the band edge.

WHAT BELONGS ON THE PAGE BESIDE THE BARS: the card's conjunction reads PASS on the scored segments, and **D4
stays open**. The drawn-set rule, frozen before any fit, left `scale_spine_length` and `scale_shoulder_width`
out (their displacement lies in the pose Jacobian's column space at first order), so the trunk is REPORTED,
not scored -- and the card says a trunk that cannot be scored is not a demonstrated trunk failure and such a
run cannot close D4. Reported, the trunk is beyond its tolerance on 9 of 12 fixtures.

Self-check:  python3 tools/compare/extractors/d4b_o1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d4b-o1/gate.json"
REGEN = (
    "/tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --drive --out artifacts/compare/d4b-o1/fresh && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_glb_closure.py --pair ... "
    "--out artifacts/compare/d4b-o1/closure.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate.py --cells artifacts/compare/d4b-o1/fresh "
    "--closure artifacts/compare/d4b-o1/closure.json --out artifacts/compare/d4b-o1/gate.json"
)
REF_SYNTH = ("SYNTHETIC MHR TRUTH: twelve bodies (six new seeds x two donors), identity drawn on the frozen "
             "drawn set, the donor pose clamped to the model's own limits. Each figure is |fitted - truth| rest "
             "length over its paired tolerance; 1.0 is the band edge. MAMMA-FREE")


def _worst_ratio(segments: dict, arm: str, names: list[str]) -> float | None:
    ratios = [row[arm][name]["error_mm"] / row[arm][name]["tolerance_mm"]
              for row in segments.values() for name in names
              if row.get(arm, {}).get(name, {}).get("tolerance_mm")]
    return round(max(ratios), 4) if ratios else None


def _best_ratio_of_worst(segments: dict, arm: str, names: list[str]) -> float | None:
    """The fixture on which the arm comes CLOSEST to passing: min over fixtures of the worst ratio."""
    per_fixture = []
    for row in segments.values():
        ratios = [row[arm][n]["error_mm"] / row[arm][n]["tolerance_mm"] for n in names
                  if row.get(arm, {}).get(n, {}).get("tolerance_mm")]
        if ratios:
            per_fixture.append(max(ratios))
    return round(min(per_fixture), 4) if per_fixture else None


def x_body_model_o1(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    segments = report.get("reported", {}).get("segments_all_arms", {})
    scored = report.get("segments_scored", [])
    figs = [
        fig("D4b L: the fitted rest lengths, worst scored segment over twelve fixtures",
            _worst_ratio(segments, "oracle", scored), "x tolerance", REF_SYNTH, LOWER,
            key="bodymodel_o1b_ours",
            note=f"verdict {report.get('verdict')} on the scored set; D4: {report.get('d4_disposition')}"),
        fig("D4b trunk (REPORTED, not scored): the fitted trunk, worst of twelve",
            _worst_ratio(segments, "oracle", ["trunk"]), "x tolerance", REF_SYNTH, LOWER,
            key="bodymodel_o1b_trunk",
            note="the spine fell below the frozen drawn-set rule; beyond tolerance on "
                 f"{report.get('clauses', {}).get('L_trunk', {}).get('measured', {}).get('fixtures_beyond_tolerance')}"
                 " of 12"),
    ]
    ctrls = [
        fig("D4b probe: CONVERGED (max_iter 300), worst scored segment", _worst_ratio(segments, "converged", scored),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_o1b_alt_converged",
            note="REPORTED, never selecting: adopting it here would select a constant on the oracle that scores it"),
        fig("D4b must-fail (i): the MEAN body, closest of twelve", _best_ratio_of_worst(segments, "mean_body", scored),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_o1b_ctrl_mean_body",
            note="must exceed 1.0 on every fixture"),
        fig("D4b must-fail (ii): the spine displaced 0.149 units, trunk, closest of twelve",
            _best_ratio_of_worst(segments, "spine_displaced", ["trunk"]), "x tolerance", REF_SYNTH, LOWER,
            key="bodymodel_o1b_ctrl_spine",
            note="must exceed 1.0 on every fixture, read whether or not the trunk is scored"),
    ]
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D4b: does the fitted body have the right proportions?",
             plain="Twelve synthetic bodies with a known answer; each bar is the fitted bone length's error "
                   "divided by what the pose solve alone already costs -- below 1.0 passes, lower is better. "
                   "The blue bar is our fit on the bones the check can score: it passes. The second blue bar "
                   "is the trunk, which the check could not score (the pose can mimic a longer spine), and it "
                   "is well past 1.0 -- so this run cannot close D4. The hatched bars are deliberately wrong "
                   "bodies and must sit above 1.0.",
             better="lower",
             bars=[dict(label="Our fit, scored bones (worst)", role="ours", key="bodymodel_o1b_ours"),
                   dict(label="Our fit, the trunk (not scored)", role="ours", key="bodymodel_o1b_trunk"),
                   dict(label="Solver run 10x longer (probe)", role="alt", key="bodymodel_o1b_alt_converged"),
                   dict(label="No body fit at all (deliberately wrong)", role="control",
                        key="bodymodel_o1b_ctrl_mean_body"),
                   dict(label="Spine shortened on purpose (deliberately wrong)", role="control",
                        key="bodymodel_o1b_ctrl_spine")]),
    ],
}


if __name__ == "__main__":
    figures, controls = x_body_model_o1({})
    for item in figures + controls:
        print(f"{item['key']:32s} {item['value']!s:>10} {item['unit']}  {item['note'][:90]}")
    keys = {item["key"] for item in figures + controls}
    missing = [bar["key"] for chart in VISUALS["converter"] for bar in chart["bars"] if bar["key"] not in keys]
    print("VISUALS keys resolved:", not missing, missing)
