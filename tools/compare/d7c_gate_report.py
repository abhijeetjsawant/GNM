#!/usr/bin/env python3
"""D7c's gate: every clause, its predicted value, its measured value and a verdict.

IT COMPUTES NOTHING AND IT ASSERTS NOTHING. It loads the reports each instrument wrote and
DERIVES every verdict from the numbers in them, so a clause can only pass because a
measurement says so.

WHY THIS FILE WAS REWRITTEN. Astra's merge review round 2 set the wrong-origin control's
residual to zero in memory and the gate still said MERGE; it changed the (a)/(b) split to
one-better/five-worse with `S_verdict: SPLIT` and the gate still said MERGE. Both clauses were
receiving a LITERAL `"PASS"`. The flip experiment that shipped beside them flipped
already-assigned verdicts, which proves the final conjunction's WIRING and not that a
measurement can fail its condition. Two changes follow:

  * `build(reports)` is a pure function of the loaded artifacts. No verdict is a literal; each
    one is an expression over numbers that came out of a report.
  * `INPUT_MUTATIONS` demonstrates failure AT THE INPUT LEVEL. For every conjunct, an input
    artifact is mutated IN MEMORY -- the wrong-origin residual set to zero, the (a)/(b) split
    reversed, a delivered SHA changed, a contact anchor moved past its tolerance -- and the
    whole gate is rebuilt from the mutated inputs. Every one must read NOT MERGE. That is the
    experiment Astra ran by hand, run by the instrument on itself, and its table is written
    into the log and into `gate.json`.

TWO CLAUSES READ **FAIL** AND STAY THAT WAY: S at the card's own fixture (sigma 1.0) and the
calibration under its own frozen monotonicity precondition. `selector.json` and
`selector-calibrated.json` are immutable, and the two reviewer amendments that followed are
recorded as POST HOC.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_gate_report.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/compare/d7c-pelvis-rest"

REPORTS = {
    "hygiene": "delivery-hygiene-build.json",
    "tripwire": "tripwire-mode-c-build.json",
    "delivery": "delivery-build.json",
    "oracle": "instrument-d7c.json",
    "take": "instrument-take.json",
    "sigma1": "selector.json",
    "calibration": "selector-calibrated.json",
    "admissibility": "selector-calibrated-amended.json",
    "reread": "selector-reread-sigma0.335546875.json",
    "projection": "projection-preservation.json",
    "p_oracle": "projection-preservation-oracle.json",
    "control2": "projection-preservation-control-clear-contacts.json",
    "p1_controls": "p1-controls.json",
    "silhouette": "silhouette-partwise.json",
    "b1_attribution": "b1-attribution.json",
    "b2": "b2-delivered-vs-capture.json",
    "b3": "b3-hoist-and-contacts.json",
    "b6": "b6-delivered-bytes.json",
}

# The bands, restated here so the derivation is visible beside the number it tests. None is
# new and none is moved; each is the card's.
O1_TILT_DEG, O1_ORIGIN_MM, O1_RESIDUAL_M = 0.01, 0.01, 1.0e-6
O2_LEG_MM, O2_HOIST_MM = 0.1, 0.05
CONTACT_TOLERANCE_M = 1.0e-5
CALIBRATION_TARGET_MM = 8.7636
CALIBRATION_BRACKET = (0.10, 1.00)
TIE_DEG, TIE_MM = 0.1, 0.1
ORACLE_SEEDS = ("20260903", "20260904", "20260905", "20260906", "20260907", "20260908")
PERFORMERS = ("subject_00", "subject_01")
DELIVERED_FILES = tuple(
    f"subject-{s:02d}{suffix}" for s in (0, 1)
    for suffix in (".glb", ".body-track.json", ".body-track.npz", ".mapping.npz"))
FOLLOWER_RATIO, FOLLOWER_FLOOR_DEG = 2.0, 2.0
CALIBRATION_TAU_MM = 0.05


def load_all() -> dict:
    out = {}
    for key, name in REPORTS.items():
        path = BASE / name
        out[key] = json.loads(path.read_text()) if path.exists() else {}
    return out


def verdict(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def build(r: dict) -> dict:
    """Every clause, derived. A pure function of the loaded reports."""
    clauses: list[dict] = []

    def add(name, predicted, measured, value, note=""):
        clauses.append({"clause": name, "predicted": predicted, "measured": measured,
                        "verdict": value, **({"note": note} if note else {})})

    # ------------------------------------------------------------------------- hygiene
    # POPULATION COVERAGE IS PART OF THE CLAUSE. `all()` over whatever happens to be in the
    # map is vacuous on an empty map and satisfied by a single surviving row: Astra's round 3
    # deleted seven of the eight comparisons, left one matching hash, and the gate still said
    # MERGE. The eight files are NAMED and each must be present AND equal.
    def eight_files(node, label):
        present = [name for name in DELIVERED_FILES if name in node]
        matching = [name for name in present if node[name]["rebuild"] == node[name]["shipped"]]
        return (len(present) == len(DELIVERED_FILES) and len(matching) == len(present),
                f"{len(matching)} of {len(present)} present ({len(DELIVERED_FILES)} required) "
                f"{label}")

    files = r["hygiene"].get("hygiene", {}).get("delivered_files_vs_shipped", {})
    ok, detail = eight_files(files, "SHAs equal")
    add("hygiene: today's code rebuilds the shipped delivery byte-identically",
        f"all {len(DELIVERED_FILES)} named files present and equal", detail, verdict(ok))

    # ------------------------------------------------------------------------ tripwire
    trip = r["tripwire"].get("hygiene", {}).get("delivered_files_vs_shipped", {})
    held = r["tripwire"].get("pelvis_mode_held")
    trip_ok, trip_detail = eight_files(trip, "SHAs equal")
    add("REFACTOR TRIPWIRE (i): mode C held, the refactored function reproduces D9b",
        f"all {len(DELIVERED_FILES)} named files equal, mode held at C_kabsch_pelvis",
        f"{trip_detail}, mode held {held!r}",
        verdict(trip_ok and held == "C_kabsch_pelvis"))

    seeds = list((r["oracle"].get("oracle", {}).get("seeds") or {}).keys())
    # POPULATION COVERAGE. Every oracle clause is "on every seed", so a missing seed is a
    # missing measurement and not a passing one; `max()` over five of six says nothing about
    # the sixth. The same reasoning gives the take's clauses their two-performer requirement.
    seeds_complete = set(seeds) == set(ORACLE_SEEDS)

    def arm(seed, name, *path):
        node = r["oracle"]["oracle"]["seeds"][seed]["arms"][name]
        for key in path:
            node = node[key]
        return node

    if seeds:
        c_tilt = [arm(s, "C_soma_template", "pelvis_vs_truth_deg", "angle", "median")
                  for s in seeds]
        add("REFACTOR TRIPWIRE (ii): the SAME six-body C execution read against exact rig truth",
            f"6.865 deg on every seed, i.e. still failing O1's {O1_TILT_DEG} deg band",
            f"{min(c_tilt):.4f}-{max(c_tilt):.4f} deg",
            verdict(min(c_tilt) > O1_TILT_DEG),
            "ONE execution, two references, two verdicts. If this execution ever came within "
            "the O1 band the tripwire would be comparing against something that is no longer "
            "the defect.")

        # ----------------------------------------------------------------------- O1
        tmax = max(arm(s, "src_default", "pelvis_vs_truth_deg", "angle", "max") for s in seeds)
        smax = max(arm(s, "src_default", "spine_origin_miss_mm", "hoist_subtracted", "max")
                   for s in seeds)
        hmax = max(arm(s, "src_default", "hips_origin_miss_mm", "hoist_subtracted", "max")
                   for s in seeds)
        tor = [arm(s, "src_default", "ABSOLUTE_groups_mm", "unhoisted_frames", "torso")
               for s in seeds]
        rmax = max(arm(s, "src_default", "three_point_residual_m", "max") for s in seeds)
        add("O1 pelvis vs truth, every seed, every frame",
            f"<= {O1_TILT_DEG} deg (from 6.865) on all {len(ORACLE_SEEDS)} seeds",
            f"max {tmax} deg over {len(seeds)} seeds",
            verdict(seeds_complete and tmax <= O1_TILT_DEG))
        add("O1 `Spine` origin miss, hoist-subtracted", f"<= {O1_ORIGIN_MM} mm (from 21-28)",
            f"max {smax} mm over {len(seeds)} seeds",
            verdict(seeds_complete and smax <= O1_ORIGIN_MM))
        add("O1 `Hips` origin miss, hoist-subtracted", f"<= {O1_ORIGIN_MM} mm (from 10)",
            f"max {hmax} mm over {len(seeds)} seeds",
            verdict(seeds_complete and hmax <= O1_ORIGIN_MM))
        add("O1 torso on the unhoisted frames, ABSOLUTE row", "0.00 (from 8.98-12.09)",
            f"{max(tor)} over {len(seeds)} seeds",
            verdict(seeds_complete and max(tor) <= O1_ORIGIN_MM))
        add("O1 unnormalised three-point positional residual", f"<= {O1_RESIDUAL_M} m",
            f"max {rmax:.3e} m over {len(seeds)} seeds",
            verdict(seeds_complete and rmax <= O1_RESIDUAL_M),
            "the clause that discriminates the wrong-origin control")

        # ------------------------------------------------------- the must-fails, DERIVED
        wt = max(arm(s, "wrong_origin", "pelvis_vs_truth_deg", "angle", "max") for s in seeds)
        wr = min(arm(s, "wrong_origin", "three_point_residual_m", "max") for s in seeds)
        add("must-fail: the WRONG-ORIGIN template (a KNOWN BLINDNESS realised)",
            "invisible to the tilt band AND caught by the residual band",
            f"tilt max {wt} deg (inside the {O1_TILT_DEG} band), residual min "
            f"{1e3 * wr:.1f} mm (outside the {O1_RESIDUAL_M} m band)",
            verdict(wt <= O1_TILT_DEG and wr > O1_RESIDUAL_M),
            "DERIVED, and it is the clause Astra's counter-example broke: with the residual "
            "set to zero this control is caught by NOTHING, and the residual band it is the "
            "sole evidence for means nothing either. The verdict now reads that.")
        fz = [arm(s, "frozen_upright", "pelvis_vs_truth_deg", "angle", "median")
              for s in seeds]
        add("must-fail: a pelvis frozen upright (D7's control)", "fails O1's tilt band",
            f"{min(fz):.4f}-{max(fz):.4f} deg", verdict(min(fz) > O1_TILT_DEG))

        # ----------------------------------------------------------------------- O2
        o2 = [r["oracle"]["oracle"]["seeds"][s]["O2_vs_baseline"] for s in seeds]
        leg = max(row["leg_foot_toe_max_mm"] for row in o2)
        hoist = max(row["hoist_change_mm"]["max"] for row in o2)
        contacts = all(row["contacts_identical"] for row in o2)
        add("O2 legs, feet and toes vs the shipped build's FK", f"<= {O2_LEG_MM} mm per seed",
            f"max {leg} mm over {len(seeds)} seeds",
            verdict(seeds_complete and leg <= O2_LEG_MM),
            "bit-identity is NOT claimed: a pelvis frame is whole-take")
        add("O2 contacts identical on the oracle bodies",
            f"identical on all {len(ORACLE_SEEDS)} seeds",
            f"{contacts} over {len(seeds)} seeds", verdict(seeds_complete and contacts))
        add("O2 hoist change", f"<= {O2_HOIST_MM} mm", f"max {hoist} mm",
            verdict(seeds_complete and hoist <= O2_HOIST_MM))
        o3o = [arm(s, "src_default", "ALIGNED_rc_score_groups_mm", "arms") for s in seeds]
        o3c = [arm(s, "C_soma_template", "ALIGNED_rc_score_groups_mm", "arms") for s in seeds]
        add("O3 the D3 gate's own leg-root-ALIGNED gauge, arms (REPORTED)",
            "1.32-2.72 -> 0.07-0.60", f"{min(o3c)}-{max(o3c)} -> {min(o3o)}-{max(o3o)}",
            "REPORT", "that gauge subtracts the leg-root midpoint per frame and is blind to a "
                      "root move; no band reads it")

    # -------------------------------------------------------------- the two recorded STOPs
    fol1 = r["sigma1"].get("frozen_pitch_follower_bent_tercile", {})
    if fol1:
        below = [s for s, row in fol1.items() if row["ratio"] < FOLLOWER_RATIO]
        add("S at the CARD'S OWN FIXTURE (sigma 1.0): the frozen-pitch follower >= 2x on EVERY body",
            f">= {FOLLOWER_RATIO}x on 6 of 6",
            f"{min(v['ratio'] for v in fol1.values()):.3f}-"
            f"{max(v['ratio'] for v in fol1.values()):.3f}x; {len(below)} of "
            f"{len(fol1)} below {FOLLOWER_RATIO}x",
            verdict(not below),
            "the step STOPPED here and the stop stays recorded in `selector.json`, which is "
            "immutable. Attributed to the fixture's noise amplitude.")
    bis = r["calibration"].get("calibration", {}).get("bisection", {})
    if bis:
        add("the amended card's FIXTURE CALIBRATION, under its own frozen monotonicity precondition",
            "monotone across the evaluations",
            f"largest decrease {bis.get('largest_violation_mm')} mm; status "
            f"{bis.get('status')}",
            verdict(bool(bis.get("monotone_across_the_evaluations"))),
            "the step STOPPED again; `selector-calibrated.json` keeps "
            "`monotone_across_the_evaluations: false`. Diagnosed to the frame.")
    adm = r["admissibility"].get("admissibility", {})
    if adm:
        # RECOMPUTED FROM THE EVALUATIONS, not read off the saved checks. Astra's round 3 set
        # the accepted sample's residual to 1 mm with `passes` false and the gate still said
        # MERGE, because it only asked whether an accepted sigma EXISTED. The three
        # conditions of the amended rule are re-derived here from the (sigma, statistic)
        # pairs themselves, and the accepted sample must actually sit inside the tolerance.
        evaluations = r["calibration"].get(
            "calibration", {}).get("bisection", {}).get("evaluations", [])
        target = adm.get("target_mm", CALIBRATION_TARGET_MM)
        bracket = adm.get("bracket", list(CALIBRATION_BRACKET))
        ordered = sorted(evaluations, key=lambda row: row["sigma_scale"])
        values = [row["median_of_six_guard_kept_sd_mm"] for row in ordered]
        sigmas = [row["sigma_scale"] for row in ordered]
        worst_decrease = max(
            (values[i] - values[j] for i in range(len(values))
             for j in range(i + 1, len(values)) if values[j] < values[i]), default=0.0)
        signs = [(-1 if v < target else 1) for v in values if v != target]
        sign_changes = sum(1 for x, y in zip(signs, signs[1:]) if x != y)
        accepted = adm.get("replay_of_the_frozen_stopping_rule", {}).get(
            "accepted_sigma_scale_exact")
        # the frozen artifact predates `sigma_scale_exact`, so the evaluation is identified
        # by the six-place display rounding the bisection recorded -- which is exactly how
        # the replay identifies it too.
        accepted_value = next(
            (row["median_of_six_guard_kept_sd_mm"] for row in evaluations
             if accepted is not None
             and (row.get("sigma_scale_exact") == accepted
                  or row["sigma_scale"] == round(accepted, 6))), None)
        inside = (accepted_value is not None
                  and abs(accepted_value - target) <= CALIBRATION_TAU_MM)
        in_bracket = (accepted is not None
                      and bracket[0] <= accepted <= bracket[1])
        add("the SAME frozen evaluations under Astra round 7's amended admissibility rule",
            f"A <= {CALIBRATION_TAU_MM} mm, B <= 1 sign change, C an accepted sigma whose "
            f"OWN statistic is within {CALIBRATION_TAU_MM} mm of the target and inside the "
            "bracket",
            f"A {worst_decrease:.4f} mm over {len(values)} evaluations, B {sign_changes} "
            f"sign change(s), C sigma {accepted} -> {accepted_value} mm against target "
            f"{target} (inside tolerance {inside}, inside bracket {in_bracket})",
            verdict(bool(evaluations) and worst_decrease <= CALIBRATION_TAU_MM
                    and sign_changes <= 1 and inside and in_bracket),
            "an OBSERVED TOLERANCE MATCH, never monotonicity or uniqueness. POST HOC.")

    # ------------------------------------------------------------------------ S, reread
    re_ = r["reread"]
    if re_:
        agg = re_["aggregated_median_of_six"]
        # THE SIX CELLS ARE RECOMPUTED FROM THE NUMBERS AND THE TIE RULE. Reading the saved
        # `b_vs_a` classification strings is what let Astra's round 3 set (b)'s whole-take
        # orientation error to 4 deg against (a)'s 5.28 -- a genuine SPLIT -- while the
        # strings still said "worse" six times and the gate said MERGE.
        cells, rebuilt = [], {}
        for population in ("whole_take", "bent_tercile"):
            for metric, tie in (("i_orientation_deg", TIE_DEG), ("ii_step_deg", TIE_DEG),
                                ("iii_root_step_mm", TIE_MM)):
                b_value = agg["b_hipline_guarded"][population][metric]
                a_value = agg["a_kabsch_guarded"][population][metric]
                cell = ("tied" if abs(b_value - a_value) <= tie
                        else "better" if b_value < a_value else "worse")
                cells.append(cell)
                rebuilt[f"b_vs_a__{population}__{metric}"] = cell
        b_wins = all(c in ("better", "tied") for c in cells) and "better" in cells
        a_wins = all(c in ("worse", "tied") for c in cells) and "worse" in cells
        decided = b_wins or a_wins
        implied = ("D_rig_rest_hipline" if b_wins
                   else "E_rig_rest_kabsch" if a_wins else None)
        shipped = re_.get("winner", {}).get("mode")
        add("S REREAD: (a) vs (b), all three metrics, both populations",
            "one of them better-or-tied everywhere and strictly better somewhere, else SPLIT",
            f"recomputed cells {cells}; implies {implied}; the file ships {shipped}; "
            f"S_verdict {re_.get('S_verdict')}",
            verdict(decided and implied is not None and implied == shipped
                    and re_.get("S_verdict") == "PROCEED"),
            "DERIVED from the six aggregated medians and the tie rule, and cross-checked "
            "against the mode the file says it ships: a decision the numbers do not support "
            "is a SPLIT, and the card says a SPLIT STOPS the step.")
        if rebuilt != re_.get("b_vs_a"):
            add("S REREAD: the saved (a)/(b) classifications agree with the numbers",
                "the recomputed cells equal the stored ones",
                f"recomputed {rebuilt} against stored {re_.get('b_vs_a')}", "FAIL",
                "a stored classification that disagrees with its own numbers is a corrupted "
                "report, and the gate must not prefer either one silently")
        winner_arm = re_.get("winner", {}).get("arm", "a_kabsch_guarded")
        beats = []
        for population in ("whole_take", "bent_tercile"):
            beats.append(agg[winner_arm][population]["i_orientation_deg"]
                         < agg["C_on_SOMA"][population]["i_orientation_deg"])
            for metric, tie in (("ii_step_deg", 0.1), ("iii_root_step_mm", 0.1)):
                beats.append(agg[winner_arm][population][metric]
                             <= agg["C_on_SOMA"][population][metric] + tie)
        add("S REREAD: the winner strictly better than C-on-SOMA on (i), both populations",
            "strictly better on (i), better-or-tied on (ii) and (iii)",
            f"whole {agg[winner_arm]['whole_take']['i_orientation_deg']} vs "
            f"{agg['C_on_SOMA']['whole_take']['i_orientation_deg']}; bent "
            f"{agg[winner_arm]['bent_tercile']['i_orientation_deg']} vs "
            f"{agg['C_on_SOMA']['bent_tercile']['i_orientation_deg']}",
            verdict(all(beats)))
        fol = re_["frozen_pitch_follower_bent_tercile"]
        # THE RATIO IS RECOMPUTED from the two numbers on every body, and the six bodies must
        # be PRESENT. Astra's round 3 set one row's `winner_i_deg` to 100 while leaving its
        # stored ratio at 3.05 and the gate still said MERGE.
        recomputed = {seed: (row["follower_i_deg"] / row["winner_i_deg"]
                             if row["winner_i_deg"] > 0 else float("inf"))
                      for seed, row in fol.items()}
        ok = (set(fol) == set(ORACLE_SEEDS)
              and all(recomputed[seed] >= FOLLOWER_RATIO
                      and row["follower_i_deg"] >= FOLLOWER_FLOOR_DEG
                      for seed, row in fol.items()))
        add("S REREAD: the frozen-pitch follower >= 2x the winner AND >= 2 deg, on EVERY body",
            f">= {FOLLOWER_RATIO}x and >= {FOLLOWER_FLOOR_DEG} deg on 6 of 6",
            f"{len(fol)} of {len(ORACLE_SEEDS)} bodies; recomputed ratios "
            f"{min(recomputed.values()):.3f}-{max(recomputed.values()):.3f}x, "
            f"{min(v['follower_i_deg'] for v in fol.values()):.2f}-"
            f"{max(v['follower_i_deg'] for v in fol.values()):.2f} deg",
            verdict(ok), "the clause that stopped the step at sigma 1.0")
        g1 = re_["G1_missing_only"]["bodies"]
        holds = all((not row["effective_masks_identical"])
                    or row["interpolated_arrays_bit_identical"] for row in g1.values())
        add("G1 (missing-only): identical masks and retained samples => bit-identical ARRAYS",
            "holds on every body", f"{sum(1 for row in g1.values() if not row['effective_masks_identical'])} "
            f"of {len(g1)} bodies have a different effective mask; identity holds wherever "
            "the masks agree", verdict(holds),
            "an EQUIVALENCE and an error measurement; no superiority claim")
        g2 = re_["G2_finite_only"]
        per_i = all(row["guarded"]["i_on_corrupted_frames_deg"]
                    < row["unguarded"]["i_on_corrupted_frames_deg"]
                    for row in g2["bodies"].values())
        per_ii = all(row["guarded"]["ii_on_transition_pairs_deg"]
                     < row["unguarded"]["ii_on_transition_pairs_deg"]
                     for row in g2["bodies"].values())
        med = g2["median_of_six"]
        add("G2 (finite-only): the guard beats the unguarded winner on BOTH (i) and (ii), every body",
            "both metrics, every body and the median of six",
            f"(i) {med['guarded_i_deg']} vs {med['unguarded_i_deg']} deg; (ii) "
            f"{med['guarded_ii_deg']} vs {med['unguarded_ii_deg']} deg; per-body wins "
            f"{per_i} / {per_ii}",
            verdict(per_i and per_ii
                    and med["guarded_i_deg"] < med["unguarded_i_deg"]
                    and med["guarded_ii_deg"] < med["unguarded_ii_deg"]),
            "where the guard EARNS its place: on the clean fixture it rejects almost nothing "
            "and cannot lose, so its win in S's main arms proves nothing about it")
        wv = re_.get("world_vertical_vs_truth_tilt", {})
        if wv:
            add("the world-vertical control against the truth PELVIS's own tilt", "REPORT",
                f"{wv.get('world_vertical_i_bent_deg')} deg against the truth pelvis's "
                f"{wv.get('truth_PELVIS_bent_tilt_median_deg')} deg -- limitation applies: "
                f"{wv.get('stated_limitation_if_within_2_deg_of_the_tilt')}",
                "REPORT", "S's stops are unchanged; the follower carries the argument")

    # -------------------------------------------------------------------- the delivery
    hyg = r["delivery"].get("hygiene", {})
    raw = hyg.get("raw_triangulation_byte_identical_same_denominator", {})
    smoothed = hyg.get("smoothed_triangulation_byte_identical", {})
    # BOTH PERFORMERS AND BOTH ARRAYS MUST BE PRESENT. `all({})` is True, and Astra's round 3
    # emptied both maps and the gate still said MERGE. A missing comparison is not a passing
    # one.
    covered = (set(raw) == set(PERFORMERS) and set(smoothed) == set(PERFORMERS))
    same = covered and all(raw.values()) and all(smoothed.values())
    add("the delivery: BOTH landmark arrays byte-identical (the same denominator)",
        f"raw AND smoothed identical on {len(PERFORMERS)} performers",
        f"raw {raw or '{} MISSING'}; smoothed {smoothed or '{} MISSING'}",
        verdict(bool(same)),
        "a converter-only change cannot move either array, and an ABSENT comparison is not a "
        "passing one")
    pf = r["delivery"].get("diagnostics", {}).get("pelvis_frame", [])
    if pf:
        add("the delivered run-report records the mode and the guard's demoted frames",
            "E_rig_rest_kabsch; 0 and 29 demoted",
            f"{pf[0]['mode']}; demoted {pf[0]['lever_guard']['demoted_count']} and "
            f"{pf[1]['lever_guard']['demoted_count']}",
            verdict(all(row["mode"] == "E_rig_rest_kabsch" for row in pf)
                    and [row["lever_guard"]["demoted_count"] for row in pf] == [0, 29]))

    # --------------------------------------------------------------------------- P
    proj = r["projection"]
    if proj:
        subjects = proj["subjects"]
        p1_ok = (set(subjects) == set(PERFORMERS)
                 and all(row["P1_channel_preservation"]["authentication"]["authenticated"]
                         and not row["P1_channel_preservation"]["failing_channels"]
                         for row in subjects.values()))
        add("P1 channel preservation -- the delivery, both performers",
            f"every protected channel bit-identical on {len(PERFORMERS)} performers, on an "
            "AUTHENTICATED track",
            f"{len(subjects)} of {len(PERFORMERS)} performers; failing "
            + str({s: row["P1_channel_preservation"]["failing_channels"]
                   for s, row in subjects.items()}), verdict(p1_ok))
        worst = max(row["P2_anchor_lock"]["worst_travel_m"] for row in subjects.values())
        p2_covered = set(subjects) == set(PERFORMERS)
        add("P2 anchor lock -- the delivery, every accepted run, on the GLB's own arrays",
            f"<= {CONTACT_TOLERANCE_M} m at every run's first KEYED sample",
            f"worst {worst:.3e} m over {len(subjects)} of {len(PERFORMERS)} performers; runs "
            + str({s: len(row["P2_anchor_lock"]["runs"]) for s, row in subjects.items()}),
            verdict(p2_covered and worst <= CONTACT_TOLERANCE_M))
        add("P3 planted-foot travel on the frozen UNION of both builds' runs", "REPORT",
            f"{sum(len(row['P3_travel_report']['intervals']) for row in subjects.values())} "
            "intervals", "REPORT")
        oracle_p2 = proj.get("P2_on_the_oracle_bodies", {})
        if oracle_p2:
            worst_o = oracle_p2["worst_travel_m_over_all_seeds"]
            oracle_p2_covered = set(oracle_p2["seeds"]) == set(ORACLE_SEEDS)
            add("P2 anchor lock on EVERY ORACLE BODY, from each exported GLB's own arrays",
                f"<= {CONTACT_TOLERANCE_M} m on all {len(ORACLE_SEEDS)} seeds",
                f"worst {worst_o:.3e} m over {len(oracle_p2['seeds'])} seeds; runs "
                + str({k: v["runs"] for k, v in oracle_p2["seeds"].items()}),
                verdict(oracle_p2_covered and worst_o <= CONTACT_TOLERANCE_M))
    po = r["p_oracle"].get("seeds", {})
    if po:
        ok = (set(po) == set(ORACLE_SEEDS)
              and all(row["root_bit_identical"] and row["contacts_bit_identical"]
                      and not row["failing_channels"] for row in po.values()))
        add("P1 on EVERY ORACLE BODY (the card says the take AND every oracle body)",
            f"every protected channel bit-identical on all {len(ORACLE_SEEDS)} seeds",
            f"{sum(1 for row in po.values() if not row['failing_channels'])} of {len(po)} "
            f"clean, {len(po)} of {len(ORACLE_SEEDS)} seeds present",
            verdict(ok))
    ctrl = r["p1_controls"].get("controls", {})
    if ctrl:
        c1 = all(row["control_1_foot_locals_overwritten"]["failing_channels"]
                 for row in ctrl.values())
        c2 = all(row["control_2_contact_mask_cleared"]["failing_channels"]
                 for row in ctrl.values())
        clean = all(not row["the_unmutated_delivery"]["failing_channels"]
                    for row in ctrl.values())
        add("P1's CONTROL 1 -- the projection's foot locals overwritten", "must FAIL P1",
            "the SHIPPING PATH REFUSES TO BUILD IT (`validate_body_track`: left foot contact "
            "moved 0.00884243 m). Applied to the delivered bytes instead, P1 detects it: "
            + str({s: row["control_1_foot_locals_overwritten"]["failing_channels"]
                   for s, row in ctrl.items()}), verdict(c1),
            "refused by the shipping path is STRONGER than caught by a gate")
        add("P1's CONTROL 2 -- the nonempty contact mask cleared (offline)",
            "must FAIL P1 on the mask",
            str({s: row["control_2_contact_mask_cleared"]["failing_channels"]
                 for s, row in ctrl.items()}), verdict(c2))
        add("the UNMUTATED delivery through the same comparison", "PASS",
            str({s: row["the_unmutated_delivery"]["P1"] for s, row in ctrl.items()}),
            verdict(clean))
    c2b = r["control2"].get("subjects", {})
    if c2b:
        ok = all(row["P1_channel_preservation"]["failing_channels"] for row in c2b.values())
        add("P1's CONTROL 2, BUILT and run through the P instrument", "must FAIL P1",
            str({s: row["P1_channel_preservation"]["failing_channels"]
                 for s, row in c2b.items()}), verdict(ok),
            "an INSTRUMENT DEFECT was found by this very control: the first build read P1 "
            "PASS because the watcher saved the snapshot AFTER the mutation")

    # -------------------------------------------------------------------------- B1
    sil = r["silhouette"].get("preregistered_clause_verdicts", {})
    if sil:
        cells = [(s, name, cell) for s, row in sil.items() if s.startswith("subject_")
                 for name, cell in row.items() if name.startswith("clause_")]
        upper = [cell["ci95"][1] >= 0.0 for _, _, cell in cells]
        b1_covered = ({s for s in sil if s.startswith("subject_")} == set(PERFORMERS)
                      and len(cells) == 8)
        add("B1 the photographs: worsening NOT ESTABLISHED (ci95 upper bound >= 0), 8 cells",
            "ci95[1] >= 0 on every cell",
            f"{sum(upper)} of {len(upper)} cells with the upper bound at or above zero, "
            f"over {len({s for s in sil if s.startswith('subject_')})} performers",
            verdict(b1_covered and all(upper)),
            "it does NOT establish non-worsening; a wide interval passes it for want of power")
        oracle_cell = sil.get("clause_mamma_mesh_oracle", {})
        worst = oracle_cell.get(
            "this_instruments_split_oracle_vs_the_committed_unsplit_one_worst_abs_difference")
        add("B1 the MAMMA mesh oracle bit-identical", "< 1e-9", str(worst),
            verdict(worst is not None and worst < 1e-9))

    # -------------------------------------------------------------------------- B2
    b2_same = r["b2"].get("same_denominator")
    if b2_same is not None:
        if isinstance(b2_same, dict):
            b2_same = b2_same.get("verdict", b2_same)
        add("B2 `delivered_vs_capture.py --reference smoothed`: the same-denominator clause",
            "PASS (landmarks byte-identical)", str(b2_same),
            verdict(str(b2_same).upper() in ("TRUE", "PASS")),
            "CHANGED would mean the change did more than refit the pelvis")

    # --------------------------------------------------------------- the REPORT blocks
    take = r["take"].get("take", {}).get("subjects", {})
    if take:
        add("B4 the pelvis and the root's motion (REPORTED, never banded)", "REPORT",
            "; ".join(
                f"{s}: pitch {row['vs_baseline']['pelvis_change_deg']['pitch_about_hip_line_signed_median']} deg, "
                f"root {row['vs_baseline']['root_move_mm_hoist_subtracted']['median']} mm, "
                f"step p95 {row['pelvis_step_deg_per_frame']['p95']} deg, "
                f"{row['frames_over_800_deg_per_s']} frame(s) over 800 deg/s"
                for s, row in take.items()), "REPORT",
            "the 800 deg/s line is a physical REFERENCE, not a band")
        add("B4 the leg-root midpoint stays on the captured hip midpoint", "0.0 mm",
            "; ".join(f"{s}: max {row['leg_roots_on_captured_hip_midpoint_mm']['max']} mm"
                      for s, row in take.items()), "REPORT",
            "NOT (b)-specific: the card listed this as conditional on (b) in error")
        add("B2/B4 the hip residual under (a) -- a REPORT, and NO band may be made from it",
            "REPORT", "; ".join(
                f"{s}: full p95 "
                f"{row['vs_baseline']['hip_residual_REPORT_never_a_band']['baseline']['full_positional_mm']['p95']}"
                f" -> {row['vs_baseline']['hip_residual_REPORT_never_a_band']['candidate']['full_positional_mm']['p95']} mm"
                for s, row in take.items()), "REPORT")
    att = r["b1_attribution"].get("subjects", {})
    if att:
        rows = []
        for s, row in att.items():
            cell = row["torso"]["whole_take"]
            rows.append(
                f"{s}: both {cell['candidate_minus_D9b__both_effects']['median_difference']:+.5f} "
                f"= articulation "
                f"{cell['ablation_minus_D9b__the_ARTICULATION_alone']['median_difference']:+.5f} "
                f"+ root "
                f"{cell['candidate_minus_ablation__the_ROOT_TRANSLATION_alone']['median_difference']:+.5f} "
                f"(root CI "
                f"{cell['candidate_minus_ablation__the_ROOT_TRANSLATION_alone']['ci95']})")
        add("B1 attribution of the three rising torso cells (DIAGNOSTIC)", "REPORT",
            "; ".join(rows), "REPORT",
            "a POINT-ESTIMATE decomposition. Performer 1's root share has a CI through zero, "
            "so a definite positive root effect is NOT established.")
    b6 = r["b6"].get("builds", {}).get("D7c", {})
    if b6:
        row = b6["subject_00"]
        add("B6 the delivered bytes (REPORT)", "REPORT",
            f"LINEAR samplers, {row['sampler_input_times']['frames']} frames, "
            f"{row['duration_s']:.4f} s, 1 translation + 55 rotation channels; quaternion "
            f"norms 1 +- 4e-8; ZERO negative adjacent dots; track->GLB positional closure max "
            f"{row['track_to_glb_closure']['positional_mm']['max']} mm", "REPORT")
        closure = row["track_to_glb_closure"].get("rotational_deg_frame_corrected")
        if closure:
            add("B6 track->GLB ROTATIONAL closure, after undoing the exporter's bind change",
                "REPORT", f"median {closure['median']} deg, max {closure['max']} deg",
                "REPORT", "the raw comparison is 32 deg and is a change of FRAME, not an error")
        playback = row.get("between_key_playback_mm", {})
        if playback:
            add("B6 between-key playback (rotation AND translation interpolated)", "REPORT",
                json.dumps(playback.get("maxima_mm", {})), "REPORT",
                "B6's report, never P2's clause, which reads KEYED samples only")
        mesh = row.get("mesh_deformation_pelvis_hip_thigh", {})
        if mesh:
            add("B6 the mesh-deformation reading on the pelvis / hip / thigh region", "REPORT",
                json.dumps({k: mesh.get(k) for k in
                            ("triangles", "inverted_triangles_frame_relative",
                             "area_ratio", "edge_length_ratio")})[:300], "REPORT",
                "no deformation acceptance band is invented")
        add("B6 the `Root` / eye / finger local invariants vs D9b (a TRACK-ARRAY claim)",
            "bit-identical", str(row.get("invariants_vs_the_other_build_TRACK_ARRAYS")),
            "REPORT")
        head = r["b6"].get("B5b_head_world_between_builds", {})
        if head:
            add("B5b the delivered `Head` WORLD rotation, from the GLB's own bytes", "REPORT",
                "; ".join(f"{s}: per-frame difference "
                          f"{row_['per_frame_difference_deg']['median']} deg median, "
                          f"{row_['per_frame_difference_deg']['max']} max"
                          for s, row_ in head.items()), "REPORT",
                "NOT identical, and the earlier claim of identity is withdrawn. What IS shown "
                "is that a ~9 deg pelvis change reaches the head at the 1e-5 deg level.")
    b3 = r["b3"].get("arms", {})
    if b3:
        add("B3 the hoist and the contacts (REPORTED)", "REPORT",
            "; ".join(f"{label} {s}: hoist p95 {row['hoist_mm']['p95']} mm, contacts "
                      f"{row['contacts']['count']}"
                      for label, blk in b3.items() for s, row in blk["subjects"].items()),
            "REPORT")

    # ------------------------------------------------------------ the card's merge rule
    def verdict_of(prefix):
        hits = [c for c in clauses if c["clause"].startswith(prefix)]
        return None if not hits else (
            "PASS" if all(c["verdict"] in ("PASS", "REPORT") for c in hits) else "FAIL")

    def all_of(prefixes):
        values = [verdict_of(p) for p in prefixes]
        return None if any(v is None for v in values) else (
            "PASS" if all(v == "PASS" for v in values) else "FAIL")

    conjuncts = {name: all_of(prefixes) for name, prefixes in CONJUNCTS}
    missing = [k for k, v in conjuncts.items() if v is None]
    return {
        "clauses": clauses, "conjuncts": conjuncts, "not_yet_measured": missing,
        "verdict": ("MERGE" if not missing and all(v == "PASS" for v in conjuncts.values())
                    else "INCOMPLETE" if missing else "NO MERGE"),
    }


S_STOPS = (
    "the SAME frozen evaluations under Astra round 7's amended admissibility rule",
    "S REREAD: (a) vs (b)",
    "S REREAD: the winner strictly better than C-on-SOMA",
    "S REREAD: the frozen-pitch follower",
    "G1 (missing-only)",
    "G2 (finite-only)",
)
MUST_FAILS = (
    "REFACTOR TRIPWIRE (ii)",
    "must-fail: the WRONG-ORIGIN template",
    "must-fail: a pelvis frozen upright",
    "P1's CONTROL 1",
    "P1's CONTROL 2 -- the nonempty contact mask cleared",
    "P1's CONTROL 2, BUILT",
)
CONJUNCTS = (
    ("hygiene", ("hygiene:",)),
    ("the refactor tripwire (both readings)",
     ("REFACTOR TRIPWIRE (i)", "REFACTOR TRIPWIRE (ii)")),
    ("O1", ("O1 ",)),
    ("O2", ("O2 ",)),
    ("every must-fail still fails", MUST_FAILS),
    ("P1 on the take",
     ("P1 channel preservation", "the UNMUTATED delivery through the same comparison")),
    ("P2 on the take", ("P2 anchor lock -- the delivery",)),
    ("P1 on every oracle body", ("P1 on EVERY ORACLE BODY",)),
    ("P2 on every oracle body", ("P2 anchor lock on EVERY ORACLE BODY",)),
    ("S (every stop of the reread, G1 and G2 included)", S_STOPS),
    ("B1 on both performers, oracle included",
     ("B1 the photographs", "B1 the MAMMA mesh oracle")),
    ("the same denominator (B2 and both landmark arrays)",
     ("B2 `delivered_vs_capture.py", "the delivery: BOTH landmark arrays byte-identical")),
)
OUTSIDE = {
    "the delivered run-report records the mode and the guard's demoted frames":
        "a REPORT clause; the card does not band the diagnostics block. Astra's round 2 "
        "accepted this exclusion explicitly.",
}


# ------------------------------------------------------- failure AT THE INPUT LEVEL
def _set(node, path, value):
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def mutate_wrong_origin_residual(r):
    for seed in r["oracle"]["oracle"]["seeds"].values():
        seed["arms"]["wrong_origin"]["three_point_residual_m"] = {"median": 0.0, "max": 0.0}


def mutate_ab_split(r):
    keys = list(r["reread"]["b_vs_a"])
    r["reread"]["b_vs_a"] = {k: ("better" if i == 0 else "worse")
                             for i, k in enumerate(keys)}
    r["reread"]["S_verdict"] = "SPLIT"


INPUT_MUTATIONS = (
    ("hygiene", "one delivered SHA in the hygiene rebuild is changed",
     lambda r: _set(r["hygiene"]["hygiene"]["delivered_files_vs_shipped"],
                    ("subject-00.glb", "rebuild"), "0" * 64)),
    ("the refactor tripwire (both readings)",
     "one delivered SHA in the mode-C tripwire rebuild is changed",
     lambda r: _set(r["tripwire"]["hygiene"]["delivered_files_vs_shipped"],
                    ("subject-01.glb", "rebuild"), "0" * 64)),
    ("O1", "one seed's three-point residual is raised above 1e-6 m",
     lambda r: _set(list(r["oracle"]["oracle"]["seeds"].values())[0]["arms"]["src_default"],
                    ("three_point_residual_m", "max"), 1.0e-3)),
    ("O2", "one seed's leg/foot/toe move is raised above 0.1 mm",
     lambda r: _set(list(r["oracle"]["oracle"]["seeds"].values())[0],
                    ("O2_vs_baseline", "leg_foot_toe_max_mm"), 1.0)),
    ("every must-fail still fails",
     "ASTRA'S OWN COUNTER-EXAMPLE: the wrong-origin control's residual set to zero",
     mutate_wrong_origin_residual),
    ("every must-fail still fails",
     "the frozen-upright control's tilt brought inside O1's band",
     lambda r: [seed["arms"]["frozen_upright"]["pelvis_vs_truth_deg"]["angle"].update(
         {"median": 0.001}) for seed in r["oracle"]["oracle"]["seeds"].values()]),
    ("P1 on the take", "one protected channel is marked as differing on the delivery",
     lambda r: list(r["projection"]["subjects"].values())[0][
         "P1_channel_preservation"]["failing_channels"].append("local::LeftFoot")),
    ("P2 on the take", "one accepted run's anchor travel is raised past 1e-5 m",
     lambda r: _set(list(r["projection"]["subjects"].values())[0],
                    ("P2_anchor_lock", "worst_travel_m"), 1.0e-3)),
    ("P1 on every oracle body", "one oracle body's root is marked as not bit-identical",
     lambda r: _set(list(r["p_oracle"]["seeds"].values())[0],
                    ("root_bit_identical",), False)),
    ("P2 on every oracle body", "the oracle anchor lock's worst travel is raised past 1e-5 m",
     lambda r: _set(r["projection"]["P2_on_the_oracle_bodies"],
                    ("worst_travel_m_over_all_seeds",), 1.0e-3)),
    ("S (every stop of the reread, G1 and G2 included)",
     "ASTRA'S OWN COUNTER-EXAMPLE: (b) vs (a) set to one-better/five-worse, S_verdict SPLIT",
     mutate_ab_split),
    ("S (every stop of the reread, G1 and G2 included)",
     "G2's guarded arm made worse than the unguarded on one body",
     lambda r: _set(list(r["reread"]["G2_finite_only"]["bodies"].values())[0],
                    ("guarded", "i_on_corrupted_frames_deg"), 999.0)),
    ("S (every stop of the reread, G1 and G2 included)",
     "ASTRA ROUND 3 (iv): one body's winner_i_deg set to 100 while its STORED ratio is left "
     "at 3.05 -- the ratio must be recomputed from the two numbers",
     lambda r: _set(list(r["reread"]["frozen_pitch_follower_bent_tercile"].values())[0],
                    ("winner_i_deg",), 100.0)),
    ("S (every stop of the reread, G1 and G2 included)",
     "one body's follower error dropped below the 2 deg floor",
     lambda r: _set(list(r["reread"]["frozen_pitch_follower_bent_tercile"].values())[0],
                    ("follower_i_deg",), 0.5)),
    ("S (every stop of the reread, G1 and G2 included)",
     "ASTRA ROUND 3 (i): (b)'s whole-take orientation error set to 4 deg against (a)'s "
     "5.28078 -- a genuine SPLIT that the stored classification strings still call 'worse'",
     lambda r: _set(r["reread"]["aggregated_median_of_six"]["b_hipline_guarded"],
                    ("whole_take", "i_orientation_deg"), 4.0)),
    ("S (every stop of the reread, G1 and G2 included)",
     "ASTRA ROUND 3 (ii): the accepted calibration sample's own statistic moved 1 mm off the "
     "target -- an accepted sigma that is no longer inside the tolerance",
     lambda r: [_set(row, ("median_of_six_guard_kept_sd_mm",), 9.7636)
                for row in r["calibration"]["calibration"]["bisection"]["evaluations"]
                if row["sigma_scale"] == 0.335547]),
    ("S (every stop of the reread, G1 and G2 included)",
     "COVERAGE: one of the six bodies removed from the follower table",
     lambda r: r["reread"]["frozen_pitch_follower_bent_tercile"].pop(
         list(r["reread"]["frozen_pitch_follower_bent_tercile"])[0])),
    ("S (every stop of the reread, G1 and G2 included)",
     "COVERAGE: the calibration's evaluations emptied",
     lambda r: _set(r["calibration"]["calibration"]["bisection"], ("evaluations",), [])),
    ("S (every stop of the reread, G1 and G2 included)",
     "G1's arrays made to differ where the effective masks agree",
     lambda r: list(r["reread"]["G1_missing_only"]["bodies"].values())[0].update(
         {"effective_masks_identical": True, "interpolated_arrays_bit_identical": False})),
    ("B1 on both performers, oracle included",
     "one silhouette cell's CI upper bound driven below zero",
     lambda r: _set(
         [cell for s, row in r["silhouette"]["preregistered_clause_verdicts"].items()
          if s.startswith("subject_") for name, cell in row.items()
          if name.startswith("clause_")][0], ("ci95",), [-0.02, -0.01])),
    ("the same denominator (B2 and both landmark arrays)",
     "the delivery's smoothed landmark array marked as moved",
     lambda r: _set(r["delivery"]["hygiene"]["smoothed_triangulation_byte_identical"],
                    ("subject_00",), False)),
    ("the same denominator (B2 and both landmark arrays)",
     "ASTRA ROUND 3 (v): COVERAGE -- both landmark comparison maps emptied, so `all({})` is "
     "vacuously true",
     lambda r: [_set(r["delivery"]["hygiene"], (key,), {}) for key in
                ("raw_triangulation_byte_identical_same_denominator",
                 "smoothed_triangulation_byte_identical")]),
    ("the same denominator (B2 and both landmark arrays)",
     "COVERAGE: one performer dropped from the smoothed comparison",
     lambda r: r["delivery"]["hygiene"]["smoothed_triangulation_byte_identical"].pop(
         "subject_01")),
    ("hygiene",
     "ASTRA ROUND 3 (iii): COVERAGE -- seven of the eight hygiene comparisons deleted, "
     "leaving one matching hash",
     lambda r: _set(r["hygiene"]["hygiene"], ("delivered_files_vs_shipped",),
                    {"subject-00.glb": next(iter(
                        r["hygiene"]["hygiene"]["delivered_files_vs_shipped"].values()))})),
    ("the refactor tripwire (both readings)",
     "COVERAGE: one delivered file dropped from the tripwire comparison",
     lambda r: r["tripwire"]["hygiene"]["delivered_files_vs_shipped"].pop(
         "subject-01.mapping.npz")),
    ("O1", "COVERAGE: one of the six oracle seeds removed",
     lambda r: r["oracle"]["oracle"]["seeds"].pop(
         list(r["oracle"]["oracle"]["seeds"])[0])),
    ("P1 on every oracle body", "COVERAGE: one oracle body removed from the P1 report",
     lambda r: r["p_oracle"]["seeds"].pop(list(r["p_oracle"]["seeds"])[0])),
    ("P2 on every oracle body", "COVERAGE: one oracle body removed from the P2 report",
     lambda r: r["projection"]["P2_on_the_oracle_bodies"]["seeds"].pop(
         list(r["projection"]["P2_on_the_oracle_bodies"]["seeds"])[0])),
    ("P1 on the take", "COVERAGE: one performer removed from the take's P report",
     lambda r: r["projection"]["subjects"].pop("subject_01")),
    ("B1 on both performers, oracle included",
     "COVERAGE: one performer removed from the silhouette verdicts",
     lambda r: r["silhouette"]["preregistered_clause_verdicts"].pop("subject_01")),
    ("hygiene", "COVERAGE: the hygiene report absent altogether",
     lambda r: _set(r, ("hygiene",), {})),
)


def main() -> int:
    reports = load_all()
    built = build(reports)

    # every conjunct, broken at the INPUT and rebuilt from the mutated artifacts
    demonstration = {
        "what": ("for every conjunct, an INPUT ARTIFACT is mutated in memory and the whole "
                 "gate is rebuilt from the mutated reports. Every one must read NOT MERGE. "
                 "This is the experiment Astra's merge review ran by hand -- and it is not "
                 "the same as flipping an already-assigned verdict, which only proves the "
                 "conjunction's wiring."),
        "trials": [],
    }
    for conjunct, description, mutation in INPUT_MUTATIONS:
        mutated = copy.deepcopy(reports)
        mutation(mutated)
        result = build(mutated)
        demonstration["trials"].append({
            "conjunct": conjunct, "mutation": description,
            "conjunct_after": result["conjuncts"].get(conjunct),
            "gate_verdict_after": result["verdict"],
            "detected": result["verdict"] != "MERGE"})
    demonstration["every_mutation_detected"] = all(
        t["detected"] for t in demonstration["trials"])
    demonstration["undetected"] = [t for t in demonstration["trials"] if not t["detected"]]
    covered = {t["conjunct"] for t in demonstration["trials"]}
    demonstration["conjuncts_without_an_input_level_demonstration"] = [
        name for name, _ in CONJUNCTS if name not in covered]

    report = {
        "title": ("D7c -- the pelvis on the rig's own rest. Every clause, predicted / "
                  "measured / verdict, DERIVED from the reports."),
        "shipping_mode": "E_rig_rest_kabsch",
        "no_verdict_is_a_literal": (
            "every clause's verdict is an expression over numbers loaded from an instrument's "
            "report. Astra's round 2 found two literals -- the wrong-origin control and the "
            "(a)/(b) split -- and both of its counter-examples are now trials below."),
        "the_two_recorded_stops": (
            "S at the card's own fixture (sigma 1.0) and the calibration under its own frozen "
            "monotonicity precondition both read FAIL and stay that way. `selector.json` and "
            "`selector-calibrated.json` are immutable; the two amendments are POST HOC."),
        "clauses": built["clauses"],
        "merge_rule": {
            "source": ("the D7c card: hygiene AND the tripwire AND O1 AND O2 AND P1 and P2 on "
                       "the take and every seed AND S AND B1 on both performers AND B2's "
                       "same-denominator PASS; O3, B3, B4, B5, B6 report"),
            "conjuncts": built["conjuncts"],
            "not_yet_measured": built["not_yet_measured"],
            "verdict": built["verdict"],
        },
        "deliberately_outside_the_predicate": OUTSIDE,
        "input_level_failure_demonstration": demonstration,
    }
    (BASE / "gate.json").write_text(json.dumps(report, indent=1))

    for clause in built["clauses"]:
        print(f"{clause['verdict']:7s} {clause['clause'][:74]:74s} "
              f"{str(clause['measured'])[:44]}")
    print()
    print("MERGE RULE:", json.dumps(built["conjuncts"], indent=1))
    print("verdict:", built["verdict"], "| missing:", built["not_yet_measured"])
    print()
    print("INPUT-LEVEL FAILURE DEMONSTRATION")
    for trial in demonstration["trials"]:
        print(f"  {'detected' if trial['detected'] else 'NOT DETECTED':12s} "
              f"{trial['conjunct'][:42]:42s} -> {trial['gate_verdict_after']:10s} "
              f"{trial['mutation'][:70]}")
    print(f"  every mutation detected: {demonstration['every_mutation_detected']}")
    if demonstration["conjuncts_without_an_input_level_demonstration"]:
        print("  NO DEMONSTRATION FOR:",
              demonstration["conjuncts_without_an_input_level_demonstration"])
    print(f"\nwrote {BASE / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
