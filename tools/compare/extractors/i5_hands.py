#!/usr/bin/env python3
"""Ladder extractor for I5 -- the hand fit's held-out camera, as tracked figures.

A STUB, deliberately: `tools/compare/ladder.py` owns the registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN §2). This file only supplies the
`x_*`-shaped function for whoever wires rung 8 up, and is imported by the
instrument's tests.

`fig` is imported from `ladder.py` when it is importable and mirrored locally
otherwise, so this module never depends on `tools/compare/` being on the path.

**Two references, and they must never share an axis.**

* `HELD_OUT` -- reprojection into a camera the fit never saw, in pixels at 1280 and
  in millimetres at the subject. Reference-free: no MAMMA in it anywhere.
* `MAMMA` -- MAMMA's `pred_joints` hand joints. An **agreement** figure only, on
  its own string. MAMMA is a research-licensed instrument that never ships and
  never selects anything.

The articulation figures (amplitude, jitter, roughness) carry a third character:
**jitter is a quantity the solver optimises directly** -- `pose_smooth_weight`
penalises exactly this acceleration -- so it is a knob setting, not evidence. It is
registered because rung 8 asks for the thrash figure, and every control below names
the figure that actually rejects it, which is never jitter.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "artifacts/hands-lane/hand-fit-heldout.json"

sys.path.insert(0, str(ROOT / "tools" / "compare"))
try:  # pragma: no cover - exercised by whichever import path is available
    from ladder import COUNT, HIGHER, LOWER, fig  # type: ignore
except Exception:  # pragma: no cover
    LOWER = "lower is better"
    HIGHER = "higher is better"
    COUNT = "exposure count, not a score"

    def fig(label, value, unit, reference, better=LOWER, key=None, note=""):
        return {"key": key or label, "label": label, "value": value, "unit": unit,
                "reference": reference, "better": better, "note": note}


HELD_OUT = (
    "held-out camera: `fit_hand_sequence` on three cameras (cross-view weights "
    "computed on those three only, so no held-out pixel reaches the fit even through "
    "a sigma), reprojected into the fourth, scored on the retained cross-view gate's "
    "surviving observations in that view. Reference-free: no other system's output "
    "enters it anywhere. Blind to "
    "depth along the held-out camera's rays."
)
MAMMA_REF = (
    "MAMMA `pred_joints` hand joints (SMPL-X block derived from the file, tip "
    "permutation recovered by optimal assignment), on the MHR chain's own slots, "
    "subject pairing via tools/head/subject_map.py. AGREEMENT, never accuracy: "
    "gt_joints is a byte-copy of pred_joints and MAMMA's own two-person hand error on "
    "this footage is ~48 mm. NOT the same reference as the held-out figures; the two "
    "must not share an axis."
)
SOLVER_OPTIMISES = (
    "wrist-local fingertip position, a parameterisation-invariant frame built from "
    "positions only. THE SOLVER OPTIMISES THE JITTER READING (`pose_smooth_weight` "
    "penalises wrist-relative joint acceleration), so it is a knob setting and not "
    "evidence -- it rejects no control on this rung."
)


def x_hands(_: dict) -> tuple[list, list]:
    if not REPORT.exists():
        return [], []
    try:
        report = json.loads(REPORT.read_text())
    except (OSError, ValueError):
        return [], []

    figures: list[dict] = []
    controls: list[dict] = []
    hands = report.get("hands", {})

    figures.append(fig(
        "held-out camera reprojection, mean over every hand and fold",
        report["summary"].get("held_out_mean_over_hands_mm"), "mm at the subject",
        HELD_OUT, LOWER, key="heldout_mean_mm",
        note=f"{report['summary'].get('hands_measured')} hands x 4 leave-one-camera-out "
             "folds at the SHIPPED solver defaults (pose_smooth_weight 0.25). The fold "
             "spread is wide and is reported per fold rather than averaged away."))

    for name, hand in hands.items():
        summary = hand.get("held_out_summary")
        if summary:
            figures.append(fig(
                f"held-out camera reprojection, {name}, mean of four folds",
                summary["mean_over_folds_mm"], "mm at the subject", HELD_OUT, LOWER,
                key=f"heldout_{name}",
                note=f"best fold {summary['best_fold_mm']} mm, worst "
                     f"{summary['worst_fold_mm']} mm on {summary['folds_measured']} folds"))
            figures.append(fig(
                f"held-out camera reprojection, {name}, WORST fold",
                summary["worst_fold_mm"], "mm at the subject", HELD_OUT, LOWER,
                key=f"heldout_worst_{name}",
                note="the worst fold, not the mean: on a four-camera rig one fold can be "
                     "three times another and the mean is then not well determined"))
        for camera, fold in hand.get("folds", {}).items():
            figures.append(fig(
                f"held-out {camera}, {name}", fold["arms"]["candidate"]["median_px"],
                "px median at 1280 wide", HELD_OUT, LOWER, key=f"fold_{name}_{camera}",
                note=f"p95 {fold['arms']['candidate']['p95_px']} px; "
                     f"{fold['arms']['candidate']['median_mm_at_the_subject']} mm at the "
                     f"subject; {fold['test_observations_gated']} test observations of "
                     f"{fold['test_observations_ungated']} present; lag-1 "
                     f"autocorrelation of the residual series "
                     f"{fold.get('lag1_autocorrelation_of_the_candidate_residual_series')}"))

        art = hand.get("articulation") or {}
        ours = art.get("ours")
        if ours:
            figures.append(fig(
                f"wrist-local fingertip amplitude, {name}", ours["amplitude_mm"], "mm",
                SOLVER_OPTIMISES, COUNT, key=f"amplitude_{name}",
                note=f"{ours['amplitude_percent_of_hand_length']} % of the "
                     f"{ours['hand_length_wrist_to_middle1_mm']} mm wrist->middle1 length; "
                     f"MAMMA's is {art['AGREEMENT_mamma']['amplitude_mm']} mm "
                     f"({art['AGREEMENT_mamma']['amplitude_percent_of_hand_length']} %) on "
                     "its own reference string. Amplitude answers *is it moving* and is "
                     "the clause a frozen hand fails."))
            figures.append(fig(
                f"THE THRASH FIGURE -- wrist-local fingertip jitter, {name}",
                ours["jitter_mm"], "mm median second difference", SOLVER_OPTIMISES,
                COUNT, key=f"jitter_{name}",
                note=f"roughness {ours['roughness']}; MAMMA's jitter "
                     f"{art['AGREEMENT_mamma']['jitter_mm']} mm at roughness "
                     f"{art['AGREEMENT_mamma']['roughness']}. This replaces the void "
                     "18-24 deg vs 5.6 deg angle-sd comparison (SMPL-X axis-angle and MHR "
                     "Euler spread are incomparable). A KNOB SETTING: the solver "
                     "regularises this quantity, so it rejects nothing."))
            figures.append(fig(
                f"fingertip jitter with wrist TRANSLATION removed only, {name}",
                ours["fingertip_jitter_wrist_translation_removed_only_mm"], "mm",
                SOLVER_OPTIMISES, COUNT, key=f"jitter_translation_only_{name}",
                note="the wrist-local frame turns with the wrist and is blind to a "
                     "thrashing wrist; this reading is not. The wrist anchor's own "
                     "translational jitter is "
                     f"{ours['wrist_anchor_translational_jitter_mm']} mm -- an input to "
                     "the fit, not a variable, so it is a floor under every world-space "
                     "hand figure and is never subtracted from one."))
        for arm, row in (art.get("CONTROLS") or {}).items():
            controls.append(fig(
                f"control: {arm.replace('_', ' ')}, {name} -- fingertip AMPLITUDE "
                "(must be near zero for a frozen arm)",
                row["amplitude_mm"], "mm", SOLVER_OPTIMISES, COUNT,
                key=f"amplitude_{arm}_{name}",
                note=f"its jitter is {row['jitter_mm']} mm against the candidate's "
                     f"{ours['jitter_mm'] if ours else '?'} mm -- a control that is "
                     "SMOOTHER than the candidate is the demonstration that a jitter "
                     f"band proves nothing. Rejected by: {row['rejected_by']}"))

        agreement = art.get("agreement_fingertips")
        if agreement:
            figures.append(fig(
                f"AGREEMENT with MAMMA's fingertips, {name}", agreement["median_mm"],
                "mm median", MAMMA_REF, COUNT, key=f"mamma_agreement_{name}",
                note=f"p95 {agreement['p95_mm']} mm, anchored at the knuckle centroid and "
                     "normalised by knuckle-to-tip span because MHR's hand root is not "
                     "SMPL-X's wrist. Agreement with another estimator, never accuracy."))

    # ------------------------------------------------------------------ controls
    seen: set[str] = set()
    for name, hand in hands.items():
        for camera, fold in hand.get("folds", {}).items():
            for arm, row in fold["arms"].items():
                if arm == "candidate":
                    continue
                label = arm.replace("_", " ")
                if arm.startswith("ORACLE"):
                    controls.append(fig(
                        f"ORACLE: MAMMA's own hand joints into held-out {camera}, {name} "
                        "(the floor the protocol imposes)",
                        row["median_mm_at_the_subject"], "mm at the subject", MAMMA_REF,
                        LOWER, key=f"oracle_{name}_{camera}",
                        note="NOT a held-out score -- MAMMA saw all four cameras. It "
                             "bundles SOMA-77's own 2D error with the MHR/SMPL-X joint "
                             "convention gap, which is reported per slot in millimetres "
                             "in the report."))
                    continue
                controls.append(fig(
                    f"control: {label}, {name} held-out {camera} (must fail)",
                    row["median_mm_at_the_subject"], "mm at the subject", HELD_OUT,
                    LOWER, key=f"{arm}_{name}_{camera}",
                    note=f"P(candidate beats it) = "
                         f"{row['P_candidate_beats_this_arm_block_bootstrap']} on 15-frame "
                         f"moving blocks, identical draws; "
                         f"{'rejected' if row['rejected'] else 'NOT REJECTED'}. Rejected by: "
                         f"{row['rejected_by']}"))
                seen.add(arm)

    summary = report.get("summary", {})
    controls.append(fig(
        "degenerate arms rejected, over every hand x fold x control",
        summary.get("controls_rejected"), f"of {summary.get('controls_scored')}",
        "each control rejected when P(candidate beats it) >= 0.95 on 15-frame moving "
        "blocks with identical draws -- and every one of them is rejected by the "
        "held-out camera or by fingertip amplitude, never by the jitter the solver "
        "regularises", HIGHER, key="controls_rejected"))
    return figures, controls


if __name__ == "__main__":
    figs, ctrls = x_hands({})
    for row in figs + ctrls:
        value = row["value"]
        print(f"{row['label'][:96]:98s} {value if value is None else round(value, 3)}")
