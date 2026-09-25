#!/usr/bin/env python3
"""Ladder extractor for D4d -- a second calibration pass on the landmark start: the fitted body's REST lengths against
synthetic truth.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the registrations collide).
This file supplies the `x_*`-shaped function and the proposed `VISUALS` entry. To register it, add to `ladder.py`:

    from extractors.d4d_twopass import x_body_model_twopass

and route its figures to rung 7 (the converter), beside D4c's `bodymodel_start_*`.

It reads exactly one report, `artifacts/compare/d4d-twopass/gate.json` (`tools/compare/d4d_twopass_gate.py`).

ONE REFERENCE: SYNTHETIC MHR TRUTH -- twelve untouched bodies (seeds 20261201-06 x two donors) drawn on D4c's
limit-aware drawn set (eight channels, the trunk scored), MAMMA-FREE. Every figure is a RATIO to its own paired
tolerance (the sum of the segment's two endpoint floors on the same fixture): 1.0 is the band edge. The Phase-1
figure (20261106 donor 0, D4c's one failure) is on a BURNED fixture and says so.

WHAT BELONGS ON THE PAGE BESIDE THE BARS, FIRST: **D4d: PASS -- D4's O1 is superseded by the combined fitter (D4c's
landmark start + a second calibration pass). But the twelve acceptance bodies did not include the class D4c failed on
(a strongly shortened spine on donor 0): D4c's one-pass fitter also passes all twelve, so the acceptance set cannot tell
the two apart; the evidence that the second pass repairs that class is Phase 1's, on burned fixtures.**

Self-check:  python3 tools/compare/extractors/d4d_twopass.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d4d-twopass/gate.json"
REF_SYNTH = ("SYNTHETIC MHR TRUTH: twelve untouched bodies (six new seeds x two donors), identity drawn on the "
             "limit-aware drawn set, the donor pose clamped to the model's own limits. Each figure is |fitted - truth| "
             "rest length over its paired tolerance; 1.0 is the band edge. MAMMA-FREE")
SCORED = ("trunk", "neck_head", "shoulder_width", "l_upper_arm", "r_upper_arm", "l_forearm", "r_forearm", "hip_width",
          "l_thigh", "r_thigh", "l_shin", "r_shin")


def _worst(segments: dict, arm: str, names) -> float | None:
    ratios = [row[arm][n]["error_mm"] / row[arm][n]["tolerance_mm"] for row in segments.values() for n in names
              if row.get(arm, {}).get(n, {}).get("tolerance_mm")]
    return round(max(ratios), 4) if ratios else None


def _closest(segments: dict, arm: str, names) -> float | None:
    per = []
    for row in segments.values():
        ratios = [row[arm][n]["error_mm"] / row[arm][n]["tolerance_mm"] for n in names
                  if row.get(arm, {}).get(n, {}).get("tolerance_mm")]
        if ratios:
            per.append(max(ratios))
    return round(min(per), 4) if per else None


def x_body_model_twopass(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    reported = report.get("reported", {})
    segments = reported.get("segments_all_arms", {})
    limbs = [s for s in SCORED if s not in ("trunk", "neck_head")]
    verdict = f"Gate: {report.get('verdict')} ({report.get('reason')})"
    fork = reported.get("phase1_fork", {})
    figs = [
        fig("D4d L: the fitted trunk (landmark start + two passes), worst of twelve",
            _worst(segments, "candidate", ["trunk"]), "x tolerance", REF_SYNTH, LOWER, key="bodymodel_twopass_trunk",
            note=verdict),
        fig("D4d L: the fitted limbs and widths (two passes), worst of twelve", _worst(segments, "candidate", limbs),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_twopass_limbs", note=verdict),
        fig("D4d Phase 1 (BURNED): two passes on D4c's failing body 20261106/d0, worst segment",
            (lambda v: None if v is None else round(v, 4))((fork.get("two_pass_worst_segment_ratio") or {}).get(
                "d4c/20261106_d0")), "x tolerance", REF_SYNTH, LOWER,
            key="bodymodel_twopass_phase1_failing_body",
            note="burned development evidence, never acceptance; D4c's one pass read 1.10x there"),
    ]
    ctrls = [
        fig("D4d reported arm: D4c's one pass, trunk, worst of twelve", _worst(segments, "one_pass", ["trunk"]),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_twopass_alt_one_pass",
            note="REPORTED, never banded: it also passes all twelve here (the failing class was not drawn)"),
        fig("D4d reported arm: the zero start (the D4 fitter), trunk, worst of twelve",
            _worst(segments, "legacy", ["trunk"]), "x tolerance", REF_SYNTH, LOWER, key="bodymodel_twopass_alt_legacy",
            note="REPORTED, never banded"),
        fig("D4d must-fail (iv): the start held with no calibration, closest of twelve",
            _closest(segments, "init_only", SCORED), "x tolerance", REF_SYNTH, LOWER, key="bodymodel_twopass_ctrl_init",
            note="must exceed 1.0 on every fixture"),
        fig("D4d must-fail (i): the MEAN body, closest of twelve", _closest(segments, "mean_body", SCORED),
            "x tolerance", REF_SYNTH, LOWER, key="bodymodel_twopass_ctrl_mean_body", note="must exceed 1.0 everywhere"),
        fig("D4d must-fail (ii): the spine displaced 0.149 units, trunk, closest of twelve",
            _closest(segments, "spine_displaced", ["trunk"]), "x tolerance", REF_SYNTH, LOWER,
            key="bodymodel_twopass_ctrl_spine", note="must exceed 1.0 everywhere"),
    ]
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D4d: does a second pass of the body fit get the proportions right?",
             plain="Yes on all twelve new test bodies, so the registered test passes. But none of the twelve had the "
                   "very short spine that tripped the one-pass fitter, and the one-pass fitter also passes all twelve "
                   "-- the proof that the second pass fixes that case is on bodies we had already studied. Each bar is "
                   "the fitted bone length's error divided by what the pose solve alone already costs -- lower is "
                   "better, 1.0 is the edge.",
             better="lower",
             bars=[dict(label="Our fit, the trunk (worst of 12)", role="ours", key="bodymodel_twopass_trunk"),
                   dict(label="Our fit, arms, legs and widths (worst)", role="ours", key="bodymodel_twopass_limbs"),
                   dict(label="Our fit on the one body the old fit missed (studied before)", role="ours",
                        key="bodymodel_twopass_phase1_failing_body"),
                   dict(label="One pass only (D4c's fitter), the trunk", role="alt", key="bodymodel_twopass_alt_one_pass"),
                   dict(label="The original fitter (zero start), the trunk", role="alt",
                        key="bodymodel_twopass_alt_legacy"),
                   dict(label="Starting guess only, no fit (deliberately wrong)", role="control",
                        key="bodymodel_twopass_ctrl_init"),
                   dict(label="No body fit at all (deliberately wrong)", role="control",
                        key="bodymodel_twopass_ctrl_mean_body"),
                   dict(label="Spine shortened on purpose (deliberately wrong)", role="control",
                        key="bodymodel_twopass_ctrl_spine")]),
    ],
}


if __name__ == "__main__":
    figures, controls = x_body_model_twopass({})
    for item in figures + controls:
        print(f"{item['key']:40s} {item['value']!s:>10} {item['unit']}  {item['note'][:90]}")
    keys = {item["key"] for item in figures + controls}
    missing = [bar["key"] for chart in VISUALS["converter"] for bar in chart["bars"] if bar["key"] not in keys]
    print("VISUALS keys resolved:", not missing, missing)
