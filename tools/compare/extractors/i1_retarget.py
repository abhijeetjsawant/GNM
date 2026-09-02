#!/usr/bin/env python3
"""Ladder extractor for step I1 -- the retarget split (rung 7, the converter).

Stub only. `tools/compare/ladder.py` owns the registry and wires this in; nothing
here edits it. Reads exactly one report, `artifacts/compare/retarget-cost.json`,
written by `python3 tools/swap-harness/retarget_cost.py` (SYSTEM python3).

TWO REFERENCES, NEVER ONE AXIS. The own-capture arms score the converter against
OUR triangulated joints; arm B scores it against MAMMA's `pred_joints` fed through
the scoreboard's PAIRS. Different bodies, different poses, a different joint
convention -- the `reference` strings differ verbatim so the page cannot stack them.

Self-check:  python3 tools/compare/extractors/i1_retarget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/retarget-cost.json"
REGEN = "python3 tools/swap-harness/retarget_cost.py"

REF_OURS = ("our own triangulated capture, root-relative (hip midpoint), 13 landmarks -- "
            "the converter scored against its OWN input")
REF_MAMMA = ("MAMMA pred_joints through the scoreboard's PAIRS, root-relative -- the converter "
             "scored against ITS OWN MAMMA input; NOT the same axis as the own-capture figures")


def _g(arms: dict, key: str, group: str) -> float | None:
    try:
        return arms[key]["per_group"][group]["median_mm"]
    except (KeyError, TypeError):
        return None


def x_retarget(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []
    for s in ("subject_00", "subject_01"):
        d = r.get("subjects", {}).get(s)
        if not d:
            continue
        t = s.replace("_", " ")
        a, c = d["arms"], d["controls"]
        b = d["arm_B_mamma_joints_in"]
        ba, bc = b["arms"], b["controls"]

        # --- what the delivered configuration costs, and how much of it is the body
        figs.append(fig(f"retarget cost on the arms, canonical rig (as delivered), {t}",
                        _g(a, "performer_canonical", "arms"), "mm median", REF_OURS,
                        key=f"canonical_arms_{s}", note=f"regenerate: {REGEN}"))
        figs.append(fig(f"retarget cost on the legs, canonical rig (as delivered), {t}",
                        _g(a, "performer_canonical", "legs"), "mm median", REF_OURS,
                        key=f"canonical_legs_{s}"))
        figs.append(fig(f"same arms, rig sized to the performer (re-solved), {t}",
                        _g(a, "performer_sized_resolved", "arms"), "mm median", REF_OURS,
                        key=f"sized_arms_{s}",
                        note="sizing is bit-identical re-solved or replayed: the converter "
                             "turns rest DIRECTIONS and sizing does not turn them"))
        figs.append(fig(f"same legs, rig sized to the performer (re-solved), {t}",
                        _g(a, "performer_sized_resolved", "legs"), "mm median", REF_OURS,
                        key=f"sized_legs_{s}"))
        figs.append(fig(f"converter's own cost on the arms after sizing (round-trip oracle), {t}",
                        _g(a, "ORACLE_roundtrip_sized", "arms"), "mm median",
                        "a body with the sized rig's proportions BY CONSTRUCTION",
                        key=f"converter_floor_arms_{s}",
                        note="what sizing cannot reach. NOT a proportion figure: on our own capture the gap above this floor is the input's bone-length wander plus the landmark-to-joint-origin convention, the clavicle origin among them (D2)"))

        # --- arm B: the same converter priced on MAMMA's joints. Separate reference.
        figs.append(fig(f"converter on MAMMA's own joints, arms, canonical rig, {t}",
                        _g(ba, "mamma_joints_canonical", "arms"), "mm median", REF_MAMMA,
                        key=f"mammaB_canonical_arms_{s}"))
        figs.append(fig(f"converter on MAMMA's own joints, arms, sized rig, {t}",
                        _g(ba, "mamma_joints_sized_resolved", "arms"), "mm median", REF_MAMMA,
                        key=f"mammaB_sized_arms_{s}"))
        figs.append(fig(f"converter on MAMMA's own joints, legs, sized rig, {t}",
                        _g(ba, "mamma_joints_sized_resolved", "legs"), "mm median", REF_MAMMA,
                        key=f"mammaB_sized_legs_{s}"))

        # --- oracles (must pass) and controls (must fail)
        ctrls.append(fig(f"ORACLE canonical round-trip, legs, {t} (must be 0.00)",
                         _g(a, "ORACLE_roundtrip_canonical", "legs"), "mm median",
                         "a canonical-proportioned body BY CONSTRUCTION", LOWER,
                         key=f"oracle_legs_{s}"))
        ctrls.append(fig(f"ORACLE canonical round-trip, arms, {t} (the KNOWN 36-47 mm converter cost)",
                         _g(a, "ORACLE_roundtrip_canonical", "arms"), "mm median",
                         "a canonical-proportioned body BY CONSTRUCTION", LOWER,
                         key=f"oracle_arms_{s}",
                         note="if this DEPARTS from 36-47 mm the instrument is broken, not the pipeline"))
        ctrls.append(fig(f"ORACLE round-trip inside arm B, arms, {t}",
                         _g(ba, "ORACLE_roundtrip_from_mamma_solve", "arms"), "mm median",
                         "a canonical body built from arm B's own output", LOWER,
                         key=f"oracle_mammaB_arms_{s}"))
        ctrls.append(fig(f"CONTROL wrong joint permutation, arms, {t} (must fail)",
                         _g(c, "CONTROL_wrong_joint_permutation", "arms"), "mm median",
                         REF_OURS, LOWER, key=f"ctrl_perm_{s}"))
        ctrls.append(fig(f"CONTROL left/right swap, legs, {t} (must fail)",
                         _g(c, "CONTROL_left_right_swap", "legs"), "mm median",
                         REF_OURS, LOWER, key=f"ctrl_lr_{s}"))
        ctrls.append(fig(f"CONTROL facing yawed 180 deg, arms, {t} (must fail)",
                         _g(c, "CONTROL_facing_yaw_180", "arms"), "mm median",
                         REF_OURS, LOWER, key=f"ctrl_yaw_{s}",
                         note="under a per-frame rotational alignment it scores exactly the "
                              "canonical arm's figure -- which is why this instrument removes "
                              "translation only"))
        cp = c.get("CONTROL_input_copied_through", {})
        ctrls.append(fig(f"CONTROL input positions copied through, arms, {t}",
                         _g(c, "CONTROL_input_copied_through", "arms"), "mm median",
                         REF_OURS, LOWER, key=f"ctrl_copy_{s}",
                         note="0.00 mm: the POSITIONAL score alone cannot reject a bag of points "
                              "with no rotations. The integrity figure below is what rejects it."))
        ctrls.append(fig(f"CONTROL copied-through bone-length wander, {t} (a real rig scores 0.00)",
                         cp.get("integrity", {}).get("bone_length_std_mm_max"), "mm std across frames",
                         "the rig's own rest lengths", HIGHER, key=f"ctrl_copy_integrity_{s}",
                         note="the degenerate must NOT be zero here; every converter arm is"))
        ctrls.append(fig(f"CONTROL wrong joint permutation inside arm B, arms, {t} (must fail)",
                         _g(bc, "CONTROL_wrong_joint_permutation", "arms"), "mm median",
                         REF_MAMMA, LOWER, key=f"ctrl_perm_mammaB_{s}"))
    return figs, ctrls


if __name__ == "__main__":
    f, c = x_retarget({})
    for row in f:
        print(f"FIG   {row['value']!s:>10}  {row['unit']:<24} {row['label']}")
    for row in c:
        print(f"CTRL  {row['value']!s:>10}  {row['unit']:<24} {row['label']}")
    print(f"\n{len(f)} figures, {len(c)} controls/oracles")
