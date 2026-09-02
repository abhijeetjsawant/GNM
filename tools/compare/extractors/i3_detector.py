#!/usr/bin/env python3
"""Ladder extractor for I3 -- the 2D detector rung's THREE reports.

A STUB, deliberately: `tools/compare/ladder.py` owns the registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN §2). This file only supplies the
`x_*`-shaped function for whoever wires the `2D landmarks` rung up, and is imported by
the instruments' tests.

**THREE REFERENCES, THREE AXES, AND THEY DO NOT MEET.**

  1. `detector-self-agreement.json` -- REFERENCE-FREE. Pixels at 1280, one-sided
     epipolar. Nothing outside our own detector and our own rig enters it.
  2. `detector-common-mode.json` -- MAMMA `pred_joints` projected through our rig, scored
     in MILLIMETRES after re-triangulation, with the offsets ORACLE-FITTED against that
     same reference. Every gain on it is a ceiling.
  3. `detector-vs-mamma.json` -- MAMMA `pred_joints` projected through our rig, scored in
     PIXELS at 1280, offsets nowhere in it. Beside it, MAMMA's own residual against its
     own fit, which is DEFLATED and enters as an exposure count, never as a score.

2 and 3 name the same MAMMA array and are still not one axis: one is millimetres after a
geometric reconstruction with an oracle-fitted correction removed, the other is raw
pixels. The `reference` strings say which is which and they must stay distinct.

**THE 2.2x HEADLINE IS RETIRED** (report 3's `RETIRED_HEADLINE`) and no figure here
restores it. It divided our detector's disagreement with RAW TRIANGULATION by MAMMA's
disagreement with a BODY-REGULARISED FIT THAT CONSUMED ITS OWN LANDMARKS: two
denominators, one axis.

**TWO CONTROLS PASS, ON PURPOSE.** `CONTROL_our_own_3d_reprojected` passes both bands of
report 1 -- that is what cross-view self-agreement is blind to, not a broken control. And
`CONTROL_including_the_semantic_mismatch_nose` does NOT discriminate on this fixture: the
I3 gate card expected it to, and it does not. Both notes travel with their figures.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT_SELF_AGREEMENT = ROOT / "artifacts/compare/detector-self-agreement.json"
REPORT_COMMON_MODE = ROOT / "artifacts/compare/detector-common-mode.json"
REPORT_VS_MAMMA = ROOT / "artifacts/compare/detector-vs-mamma.json"

sys.path.insert(0, str(ROOT / "tools" / "compare"))
try:  # pragma: no cover - exercised by whichever import path is available
    from ladder import COUNT, HIGHER, LOWER, _load, fig  # type: ignore
except Exception:  # pragma: no cover
    LOWER = "lower is better"
    HIGHER = "higher is better"
    COUNT = "exposure count, not a score"

    def fig(label, value, unit, reference, better=LOWER, key=None, note=""):
        return {"key": key or label, "label": label, "value": value, "unit": unit,
                "reference": reference, "better": better, "note": note}

    def _load(rel):
        path = ROOT / rel
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None


REF_SELF = ("REFERENCE-FREE -- our detector against itself across four calibrated views, "
            "one-sided epipolar px at 1280. No MAMMA, no fitted body, no joint mapping. "
            "Never on an axis with any millimetre figure or with either MAMMA-referenced "
            "figure below.")
REF_COMMON = ("MAMMA pred_joints projected through our rig -- scored in MILLIMETRES after "
              "raw per-frame re-triangulation, with per-camera offsets ORACLE-FITTED "
              "against that same reference. Every gain is a CEILING, not an achievable "
              "figure, and the four offsets are reported and never shipped.")
REF_VS_MAMMA = ("MAMMA pred_joints projected through our rig -- scored in PIXELS at 1280, "
                "no offsets anywhere, not deflated. Same MAMMA array as the millimetre "
                "figures above and NOT the same axis: those are millimetres through a "
                "reconstruction with an oracle correction removed.")
REF_DEFLATED = ("MAMMA's own ma_2d against our rig's projection of its OWN fitted surface. "
                "DEFLATED -- the fit consumed these landmarks. Shape and exposure only; "
                "not a score, and never a denominator.")


def _read(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _self_agreement(figs: list, ctrls: list) -> None:
    report = _read(REPORT_SELF_AGREEMENT)
    if not report:
        return
    arms, headline = report["arms"], report["headline"]
    figs.append(fig(
        "our detector's cross-view self-agreement, one-sided epipolar median",
        headline["one_sided_epipolar_median_px1280"], "px at 1280 (x3 for native 3840)",
        REF_SELF, LOWER, key="self_agreement_p50",
        note="the symmetric distance HALVED -- _epipolar_distance_px returns the sum of "
             "the two one-sided distances (CLAUDE.md). The measured symmetric/one-sided "
             f"ratio here is "
             f"{report['units']['measured_symmetric_over_one_sided_ratio']['median']} "
             "against the 1.962 on record."))
    figs.append(fig(
        "same, p95", arms["ours"]["one_sided_quantiles_px1280"]["95"],
        "px at 1280", REF_SELF, LOWER, key="self_agreement_p95"))
    figs.append(fig(
        "median per-frame 2D displacement of the detections (the liveness band)",
        arms["ours"]["liveness_px_per_frame"], "px/frame at 1280", REF_SELF, COUNT,
        key="liveness",
        note="the second band of the gate. A frozen skeleton projected into calibrated "
             "cameras is epipolar-perfect and scores exactly 0 here, which is why "
             "self-agreement alone cannot be an acceptance gate."))
    ctrls.append(fig(
        "control: shuffled cross-view pairing -- same detections, wrong partner (must be "
        f">= {report['gate']['G1_discrimination'].split('>=')[-1].strip()}x worse)",
        arms["CONTROL_shuffled_cross_view_pairing"]["one_sided_median_px1280"],
        "px at 1280", REF_SELF, HIGHER, key="self_shuffled",
        note=f"shuffled / ours = {headline['shuffled_over_ours']}x"))
    ctrls.append(fig(
        "control: frozen frame-0 skeleton projected into all four cameras (must fail)",
        arms["CONTROL_frozen_skeleton_projected"]["liveness_px_per_frame"],
        "px/frame at 1280", REF_SELF, HIGHER, key="self_frozen",
        note="it passes the epipolar band at exactly 0.000 px and fails the liveness band "
             "at exactly 0 px/frame. This is the constant the gate must reject."))
    ctrls.append(fig(
        "control that PASSES BY DESIGN: our own 3D reprojected -- the blindness",
        arms["CONTROL_our_own_3d_reprojected"]["one_sided_median_px1280"],
        "px at 1280", REF_SELF, COUNT, key="self_own_3d",
        note="0.000 px and alive, so it clears both bands while carrying no detector "
             "information beyond a fit. Cross-view consistency is a within-frame geometric "
             "property; it says nothing about whether a landmark is on the right part of "
             "the body. Report 1 is a diagnostic, not an acceptance gate."))
    ctrls.append(fig(
        "exposure: (subject, frame, camera) slots the associator left unassigned",
        _slots(report), "of 1200", "our pipeline's own association stage, not the detector",
        COUNT, key="unassigned_slots"))


def _slots(report: dict):
    text = report["population"]["what_the_common_denominator_costs"][
        "camera_slots_the_associator_left_unassigned"]
    try:
        return int(text.split()[0])
    except (IndexError, ValueError):
        return None


def _common_mode(figs: list, ctrls: list) -> None:
    report = _read(REPORT_COMMON_MODE)
    if not report:
        return
    arms = report["3d_error_vs_mamma_projections_mm"]
    verdict = report["THE_VERDICT"]
    figs.append(fig(
        "3D error of our re-triangulated joints, no offset applied",
        arms["as_measured"]["median_mm"], "mm median", REF_COMMON, LOWER, key="cm_base",
        note=report["why_the_baseline_is_not_rung_7s_number"]))
    figs.append(fig(
        "the same after four per-camera offsets fitted on the OTHER half of the frames",
        arms["SPLIT_HALF_held_out"]["median_mm"], "mm median", REF_COMMON, LOWER,
        key="cm_heldout",
        note=f"G_heldout = {verdict['values']['G_heldout']}, "
             f"P(beats baseline) = {verdict['values']['P_heldout']} on the moving-block "
             "bootstrap. The offsets are REPORTED AND NEVER SHIPPED -- they are fitted on "
             "a MAMMA-derived arm."))
    figs.append(fig(
        "CEILING: the same with a free 2D translation per subject per frame per camera",
        arms["ORACLE_full_common_mode_removed"]["median_mm"], "mm median", REF_COMMON,
        LOWER, key="cm_ceiling",
        note=f"G_ceiling = {verdict['values']['G_ceiling']}, so "
             f"{verdict['values']['share_surviving_the_ceiling']:.0%} of the error survives "
             "the best possible per-view TRANSLATION fix. That share is the decision "
             "rule's input for the pseudo-label destination."))
    figs.append(fig(
        "share of the 3D error surviving the best possible per-view translation fix "
        f"(I3 DECISION RULE VERDICT: {verdict['verdict']})",
        verdict["values"]["share_surviving_the_ceiling"], "of the baseline median",
        REF_COMMON, LOWER, key="cm_survives_ceiling",
        note="the rule (i3_decision.py) was written before the first figure and fires "
             "destination (a), the pseudo-label campaign, at >= 0.50. "
             f"(a) pseudo-label campaign: {verdict['a_pseudo_label_campaign']['triggered']}; "
             f"(b) per-camera offset fix: {verdict['b_per_camera_offset_fix']['triggered']} "
             f"-- clauses {verdict['b_per_camera_offset_fix']['clauses']}. "
             + report["are_the_offsets_the_same_size_on_both_halves"]))
    ctrls.append(fig(
        "control: the SAME offsets fitted and scored on the SAME frames (must be better "
        "than the held-out arm -- the difference is the inflation)",
        arms["CONTROL_same_frames_fit_and_score"]["median_mm"], "mm median", REF_COMMON,
        LOWER, key="cm_same_frames",
        note=f"gain {arms['CONTROL_same_frames_fit_and_score']['gain_vs_as_measured']} "
             f"against the held-out arm's {arms['SPLIT_HALF_held_out']['gain_vs_as_measured']}"))
    ctrls.append(fig(
        "control: interleaved (even/odd) halves instead of contiguous ones",
        arms["CONTROL_interleaved_half_held_out"]["median_mm"], "mm median", REF_COMMON,
        LOWER, key="cm_interleaved",
        note="with this much autocorrelation an even/odd split is a same-frames fit wearing "
             "a held-out label; its gain sits with the same-frames arm's, which is why the "
             "halves are contiguous."))
    ctrls.append(fig(
        "control: each camera given the NEXT camera's offset (must NOT reproduce the gain)",
        arms["CONTROL_shuffled_camera_assignment"]["median_mm"], "mm median", REF_COMMON,
        LOWER, key="cm_shuffled_cameras",
        note="IT DOES reproduce it -- gain "
             f"{arms['CONTROL_shuffled_camera_assignment']['gain_vs_as_measured']} against "
             f"{arms['SPLIT_HALF_held_out']['gain_vs_as_measured']}. That is the finding, "
             "not a broken control: the four offsets all point up-and-left by a similar "
             "amount, so the effect is one detector-wide shift, not four per-camera ones. "
             "Clause 4 of the rule fails and destination (b) as written is not supported."))
    posthoc = report["what_clause_4_of_the_rule_revealed"]
    ctrls.append(fig(
        "POST HOC: one global 2D offset for all four cameras, held out",
        arms["POSTHOC_global_single_offset_held_out"]["median_mm"], "mm median", REF_COMMON,
        LOWER, key="cm_global",
        note=f"gain {posthoc['global_single_offset_px1280']['held_out_gain']} -- ONE vector "
             f"beats the four at {posthoc['global_single_offset_px1280']['four_number_held_out_gain']}. "
             + posthoc["and_how_that_rule_came_out"]["verdict"]))
    ctrls.append(fig(
        "POST HOC control: the global offset turned 90 degrees (must hurt)",
        arms["POSTHOC_CONTROL_global_offset_rotated_90deg"]["median_mm"], "mm median",
        REF_COMMON, HIGHER, key="cm_global_rotated",
        note=posthoc["and_how_that_rule_came_out"]["and_it_is_left_failing_on_purpose"]))
    ctrls.append(fig(
        "POST HOC control: the global offset applied backwards (must hurt)",
        arms["POSTHOC_CONTROL_global_offset_negated"]["median_mm"], "mm median",
        REF_COMMON, HIGHER, key="cm_global_negated",
        note=f"gain {arms['POSTHOC_CONTROL_global_offset_negated']['gain_vs_as_measured']} "
             "-- it hurts, as it must."))


def _vs_mamma(figs: list, ctrls: list) -> None:
    report = _read(REPORT_VS_MAMMA)
    if not report:
        return
    ours = report["arms"]["ours_vs_mamma_projections"]
    theirs = report["arms"]["MAMMA_own_residual_DEFLATED"]
    controls = report["controls"]
    retired = report["RETIRED_HEADLINE"]
    figs.append(fig(
        "our detector's 2D residual against MAMMA's projected pred_joints, median",
        ours["median_px1280"], "px at 1280 (x3 for native 3840)", REF_VS_MAMMA, LOWER,
        key="vs_mamma_p50",
        note=f"{ours['n']} slots on {report['population']['scored_against_mamma']} joints; "
             f"{report['population']['joints_excluded_count']} of the 19 contract joints "
             "excluded for joint semantics and counted in the report. Distribution shape: "
             f"{ours['shape']['better']}."))
    figs.append(fig(
        "same, p95", ours["quantiles_px1280"]["95"], "px at 1280", REF_VS_MAMMA, LOWER,
        key="vs_mamma_p95"))
    ctrls.append(fig(
        "MAMMA's own 2D residual against its own fit -- DEFLATED, exposure only, NOT a "
        "score and never a denominator",
        theirs["median_px3840"], "px at native 3840", REF_DEFLATED, COUNT,
        key="mamma_deflated_p50",
        note="is_a_score: false. The fit consumed these landmarks, so this is a lower "
             "bound on MammaNet's error and can only be read for its shape. It is the pool "
             "tools/compare/oracle_2d.py samples for its noise arm, and it reproduces that "
             "file's quantiles exactly "
             f"(identical: {theirs['cross_check'].get('identical')}). "
             "THE 2.2x HEADLINE IS RETIRED: it divided " + retired[
                 "what_it_actually_compared"]["numerator"] + " by " + retired[
                 "what_it_actually_compared"]["denominator"] + ". Two denominators, one "
             "axis. Do not divide this figure by the one above it."))
    ctrls.append(fig(
        "control: our subject scored against the OTHER performer's MAMMA body (must be far "
        "worse)",
        controls["CONTROL_shuffled_subject_pairing"]["median_px1280"], "px at 1280",
        REF_VS_MAMMA, HIGHER, key="vs_mamma_shuffled_subject",
        note="pairing is derived from pelvis agreement (tools/head/subject_map.py), never "
             "from the index -- body_id-00 is our subject 1 on this fixture."))
    ctrls.append(fig(
        "control: our frame-0 detections held constant on every frame (must be far worse)",
        controls["CONTROL_frozen_frame0_detection"]["median_px1280"], "px at 1280",
        REF_VS_MAMMA, HIGHER, key="vs_mamma_frozen"))
    ctrls.append(fig(
        "control: left and right crossed (must be far worse)",
        controls["CONTROL_left_right_swapped"]["median_px1280"], "px at 1280",
        REF_VS_MAMMA, HIGHER, key="vs_mamma_lr_swap",
        note="the joint-semantic control that actually discriminates on this fixture."))
    ctrls.append(fig(
        "control that DOES NOT DISCRIMINATE: the semantically mismatched `nose` put back in",
        controls["CONTROL_including_the_semantic_mismatch_nose"]["median_px1280"],
        "px at 1280", REF_VS_MAMMA, HIGHER, key="vs_mamma_with_nose",
        note=report["what_the_nose_exclusion_turned_out_to_cost"]["finding"] + " "
             + report["what_the_nose_exclusion_turned_out_to_cost"]["why_it_stays_excluded"]))


def x_detector_reports(_: dict) -> tuple[list, list]:
    """Three figures lists' worth of detector evidence, in the `x_*` shape.

    Returns (figures, controls). Every figure carries one of three distinct `reference`
    strings and they are never mixed onto one axis.
    """
    figs: list = []
    ctrls: list = []
    _self_agreement(figs, ctrls)
    _common_mode(figs, ctrls)
    _vs_mamma(figs, ctrls)
    return figs, ctrls


if __name__ == "__main__":
    figures, controls = x_detector_reports({})
    for row in figures + controls:
        value = row["value"]
        print(f"{row['label'][:92]:94s} "
              f"{value if value is None else round(value, 3)} {row['unit']}")
    print(f"\n{len(figures)} figures, {len(controls)} controls, "
          f"{len({row['reference'] for row in figures})} distinct references on the figures")
