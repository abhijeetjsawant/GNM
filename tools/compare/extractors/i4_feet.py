#!/usr/bin/env python3
"""Ladder extractor for I4 -- MAMMA's feet as the bar, and ours against it.

A STUB, deliberately: `tools/compare/ladder.py` owns the registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN §2). This file only supplies the
`x_*`-shaped function for whoever wires the `feet` rung up, and is imported by the
instrument's tests.

`fig` is imported from `ladder.py` when it is importable and mirrored locally otherwise,
so this module never depends on `tools/compare/` being on the path.

Reference on every figure: **MAMMA `pred_joints`, foot direction in the shin frame.**
That is NOT the reference the retired `delivered-foot.json` figures carry (our own
triangulated `ToeBase` direction), so **the two must never share an axis.**
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "artifacts/feet-lane/mamma-feet-bar.json"

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


REF = ("MAMMA pred_joints: ankle 7/8 -> foot 10/11 as a direction in the SHIN FRAME "
       "(origin ankle, d = knee->ankle, m = d x pelvic axis). Parity, never truth -- and "
       "NOT the same reference as the retired delivered-foot figures, which are ours on "
       "both sides. The two must not share an axis.")


def x_feet_bar(_: dict) -> tuple[list, list]:
    if not REPORT.exists():
        return [], []
    try:
        r = json.loads(REPORT.read_text())
    except (OSError, ValueError):
        return [], []

    figs, ctrls = [], []
    rejected = considered = 0
    for s in ("subject_00", "subject_01"):
        subject = r["subjects"].get(s, {})
        for side in ("L", "R"):
            e = subject.get("feet", {}).get(side)
            if not e or "arms" not in e:
                continue
            t = f"{s.replace('_', ' ')} foot {side}"
            arms, verdicts = e["arms"], e["verdicts"]
            bar = e["THE_BAR_mamma_foot_range_of_motion_in_its_own_shin_frame"]

            figs.append(fig(
                f"THE BAR -- MAMMA's own foot range of motion in the shin frame, {t}",
                bar["spread_about_take_mean_direction_deg"]["median_deg"],
                "deg median about the take mean", REF, COUNT, key=f"bar_{s}_{side}",
                note=f"dorsi/plantarflexion range {bar['dorsi_plantarflexion_phi']['range_deg']:.1f} deg, "
                     f"ab/adduction range {bar['ab_adduction_psi_medial_positive']['range_deg']:.1f} deg, "
                     f"on {e['valid_frames']} valid frames -- this is the bar, not a score"))
            for arm, short in (("ours_delivered", "delivered foot"),
                               ("ours_triangulated_toebase", "triangulated ToeBase")):
                figs.append(fig(
                    f"our {short} vs MAMMA, constant offset removed, {t}",
                    arms[arm]["spread_after_offset_removed_deg"]["median_deg"],
                    "deg median", REF, LOWER, key=f"{arm}_{s}_{side}",
                    note=f"constant offset {arms[arm]['mean_direction_offset_deg']:.1f} deg, "
                         f"reported separately and never summed; "
                         f"{verdicts[arm]['verdict']}"))

            where = e["where_the_constant_offset_lives"]
            figs.append(fig(
                f"constant offset, AB/ADDUCTION component (toes inward, positive = medial), {t}",
                where["ours_triangulated_toebase"]["ab_adduction_offset_deg"],
                "deg, take mean", REF, COUNT, key=f"offset_abduction_{s}_{side}",
                note=f"the flexion component of the same offset is "
                     f"{where['ours_triangulated_toebase']['dorsi_plantarflexion_offset_deg']:+.1f} deg, "
                     f"and the ORACLE -- MAMMA's own foot through our shin frame -- carries "
                     f"{where['ORACLE_mamma_foot_in_our_triangulated_shin_frame']['ab_adduction_offset_deg']:+.1f} deg "
                     "of the ab/adduction, so the shin frame explains only part of it. A "
                     "flexion offset is a benign ball-joint placement convention; an "
                     "ab/adduction offset is toes pointing inward on screen. Reported "
                     "separately from the spread and never summed with it."))

            ctrls.append(fig(
                f"ORACLE: MAMMA's own foot through OUR shin frame, {t} (the floor our "
                "frame definitions impose -- must be small)",
                arms["ORACLE_mamma_foot_in_our_triangulated_shin_frame"][
                    "spread_after_offset_removed_deg"]["median_deg"],
                "deg median", REF, LOWER, key=f"oracle_{s}_{side}"))
            for control, short in (
                ("CONTROL_welded_to_shin_zero_articulation", "a foot welded to the shin"),
                ("CONTROL_time_shuffled_mamma", "time-shuffled MAMMA (right distribution, no tracking)"),
                ("CONTROL_mirrored_anterior_axis", "a mirrored forward axis"),
            ):
                p = verdicts["ours_triangulated_toebase"]["G3_controls"][control]
                ctrls.append(fig(
                    f"control: {short}, {t} (must fail)",
                    arms[control]["spread_after_offset_removed_deg"]["median_deg"],
                    "deg median", REF, LOWER, key=f"{control}_{s}_{side}",
                    note=("rejected by G1: its take-mean foot direction is "
                          f"{arms[control]['mean_direction_offset_deg']:.0f} deg from the "
                          "reference's, which no ankle can produce"
                          if p["control_fails_G1_orientation_zero"] else
                          f"P(our triangulated arm beats it) = "
                          f"{p['P_candidate_beats_min_over_blocks']:.3f} on the worst "
                          f"block size; {'rejected' if p['control_rejected'] else 'NOT REJECTED'}")))
            for arm in ("ours_delivered", "ours_triangulated_toebase"):
                for row in verdicts[arm]["G3_controls"].values():
                    considered += 1
                    rejected += bool(row["control_rejected"])

    ctrls.append(fig("degenerate arms rejected, over every candidate x control pair",
                     rejected, f"of {considered}",
                     "each control rejected by G1 or by P(candidate beats it) >= 0.95 on "
                     "the worst of three moving-block sizes", HIGHER, key="controls_rejected"))
    return figs, ctrls


if __name__ == "__main__":
    figures, controls = x_feet_bar({})
    for row in figures + controls:
        value = row["value"]
        print(f"{row['label'][:96]:98s} {value if value is None else round(value, 2)}")
