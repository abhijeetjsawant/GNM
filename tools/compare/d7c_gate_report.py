#!/usr/bin/env python3
"""D7c's gate: every clause, its predicted value, its measured value and a verdict.

IT COMPUTES NOTHING AND IT ASSERTS NOTHING. It loads the reports each instrument wrote and
DERIVES every verdict from the numbers in them.

WHY THIS FILE HAS THREE RULES RATHER THAN A LIST OF PATCHES. Astra's merge review broke it in
four successive rounds -- literal verdicts, then saved classifications, then partial
populations, then stored aggregates -- and each round was answered hole by hole. The fifth
round found six more. The holes were never the problem; the absence of a rule was. So:

  1. EVERY VALUE IS DERIVED FROM NAMED CONSTITUENTS, OR CROSS-CHECKED AGAINST THEM. An
     aggregate the gate reads without recomputing is an aggregate an attacker can write.
     Where a report also stores a summary, the stored and the derived value must AGREE, and a
     disagreement is a FAIL -- a report that contradicts itself is corrupt whichever half
     would have passed.
  2. A MISSING FIELD OR SET MEMBER IS A FAIL, NEVER A NO-OP. `all()` over an empty map is
     True; `max()` over a subset says nothing about the whole. Every read goes through
     `Reader`, which raises on an absent path, and the clause that needed it FAILS with the
     path named.
  3. EVERY SET IS CHECKED BY IDENTITY, NOT BY COUNT. The eight delivered files, the six oracle
     seeds, the two performers, the eight B1 cells, the six G1/G2/follower bodies, and the
     contact RUNS by their `(side, start, end)` identity taken from the frozen mask. A renamed
     cell, a duplicated run and a dropped seed all survive a count.

AND IT IS PROVED RATHER THAN ASSERTED. `tools/compare/d7c_gate_fuzz.py` walks every leaf of
every report this gate reads, mutates each one in turn, and requires NO MERGE from every leaf
any clause depends on -- reporting, with a justification, the leaves that are genuinely inert.
The mutation count in `gate.json` is the number of leaves visited, not a hand-picked table.

TWO CLAUSES READ **FAIL** AND STAY THAT WAY: S at the card's own fixture (sigma 1.0) and the
calibration under its own frozen monotonicity precondition. `selector.json` and
`selector-calibrated.json` are immutable, and the two reviewer amendments that followed are
recorded as POST HOC.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_gate_report.py
"""

from __future__ import annotations

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

# The bands, restated beside the numbers they test. None is new and none is moved.
O1_TILT_DEG, O1_ORIGIN_MM, O1_RESIDUAL_M = 0.01, 0.01, 1.0e-6
O2_LEG_MM, O2_HOIST_MM = 0.1, 0.05
CONTACT_TOLERANCE_M = 1.0e-5
FOLLOWER_RATIO, FOLLOWER_FLOOR_DEG = 2.0, 2.0
CALIBRATION_TARGET_MM, CALIBRATION_TAU_MM = 8.7636, 0.05
CALIBRATION_BRACKET = (0.10, 1.00)
TIE_DEG, TIE_MM = 0.1, 0.1
AGREEMENT = 1.0e-4                      # stored-vs-derived agreement, in each row's own unit

ORACLE_SEEDS = ("20260903", "20260904", "20260905", "20260906", "20260907", "20260908")
PERFORMERS = ("subject_00", "subject_01")
DELIVERED_FILES = tuple(
    f"subject-{s:02d}{suffix}" for s in (0, 1)
    for suffix in (".glb", ".body-track.json", ".body-track.npz", ".mapping.npz"))
B1_CELLS = tuple(f"clause_{part}_{cut}_worsening_not_established_vs_D9b"
                 for part in ("arm", "torso")
                 for cut in ("whole_take", "bent_tercile"))
PROTECTED = ("Root", "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg",
             "RightLowerLeg", "LeftFoot", "RightFoot", "LeftToes", "RightToes")
SIDE_JOINTS = {0: ("LeftFoot", "LeftToes"), 1: ("RightFoot", "RightToes")}
POPULATIONS = ("whole_take", "bent_tercile")
METRICS = (("i_orientation_deg", TIE_DEG), ("ii_step_deg", TIE_DEG),
           ("iii_root_step_mm", TIE_MM))


class Missing(Exception):
    """An absent path, an absent set member, or a value of the wrong kind."""


class Reader:
    """Every read goes through here, so an absent path FAILS its clause instead of vanishing."""

    def __init__(self, reports: dict) -> None:
        self.reports = reports
        self.touched: set[tuple] = set()

    def at(self, *path):
        node = self.reports
        for step in path:
            if isinstance(node, list):
                if not isinstance(step, int) or not -len(node) <= step < len(node):
                    raise Missing("/".join(map(str, path)))
                node = node[step]
            elif isinstance(node, dict):
                if step not in node:
                    raise Missing("/".join(map(str, path)))
                node = node[step]
            else:
                raise Missing("/".join(map(str, path)))
        self.touched.add(tuple(map(str, path)))
        return node

    def num(self, *path) -> float:
        value = self.at(*path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Missing("/".join(map(str, path)) + " is not a number")
        return float(value)

    def flag(self, *path) -> bool:
        value = self.at(*path)
        if not isinstance(value, bool):
            raise Missing("/".join(map(str, path)) + " is not a boolean")
        return value

    def text(self, *path) -> str:
        value = self.at(*path)
        if not isinstance(value, str):
            raise Missing("/".join(map(str, path)) + " is not a string")
        return value

    def named(self, *path, expect):
        """A map whose keys must BE the named set -- not merely contain or count like it."""
        node = self.at(*path)
        if not isinstance(node, dict) or set(node) != set(expect):
            raise Missing(f"{'/'.join(map(str, path))} keys {sorted(node) if isinstance(node, dict) else node!r} != {sorted(expect)}")
        return node

    def listing(self, *path, minimum=1):
        node = self.at(*path)
        if not isinstance(node, list) or len(node) < minimum:
            raise Missing(f"{'/'.join(map(str, path))} is not a list of >= {minimum}")
        return node


def median(values):
    ordered = sorted(values)
    if not ordered:
        raise Missing("median of an empty population")
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle]))


def agrees(stored, derived, tolerance=AGREEMENT) -> bool:
    return abs(float(stored) - float(derived)) <= tolerance


def build(reports: dict) -> dict:
    """Every clause, derived. A pure function of the loaded reports."""
    r = Reader(reports)
    clauses: list[dict] = []

    def clause(name, predicted, note=""):
        """Run one clause; an absent path FAILS it with the path named."""
        def wrap(fn):
            try:
                detail, ok = fn()
                clauses.append({"clause": name, "predicted": predicted, "measured": detail,
                                "verdict": ("REPORT" if ok == "REPORT"
                                            else "PASS" if ok else "FAIL"),
                                **({"note": note} if note else {})})
            except Missing as absent:
                clauses.append({"clause": name, "predicted": predicted,
                                "measured": f"MISSING or malformed: {absent}",
                                "verdict": "FAIL",
                                "note": note or "a missing measurement is a FAIL, never a "
                                                "no-op"})
        return wrap

    # ------------------------------------------------------------- hygiene and the tripwire
    def eight_files(report_key, label):
        node = r.named(report_key, "hygiene", "delivered_files_vs_shipped",
                       expect=DELIVERED_FILES)
        equal = {name: r.text(report_key, "hygiene", "delivered_files_vs_shipped", name,
                              "rebuild")
                 == r.text(report_key, "hygiene", "delivered_files_vs_shipped", name,
                           "shipped")
                 for name in DELIVERED_FILES}
        return (f"{sum(equal.values())} of {len(DELIVERED_FILES)} named files equal {label}",
                all(equal.values()))

    @clause("hygiene: today's code rebuilds the shipped delivery byte-identically",
            f"all {len(DELIVERED_FILES)} named files present and SHA-equal")
    def _():
        return eight_files("hygiene", "(SHA)")

    @clause("REFACTOR TRIPWIRE (i): mode C held, the refactored function reproduces D9b",
            f"all {len(DELIVERED_FILES)} named files equal, mode held at C_kabsch_pelvis")
    def _():
        detail, ok = eight_files("tripwire", "(SHA)")
        held = r.text("tripwire", "pelvis_mode_held")
        return f"{detail}, mode held {held!r}", ok and held == "C_kabsch_pelvis"

    # ------------------------------------------------------------------------- the oracle
    def arm(seed, name, *path):
        return r.num("oracle", "oracle", "seeds", seed, "arms", name, *path)

    def over_seeds(name, *path):
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        return {seed: arm(seed, name, *path) for seed in ORACLE_SEEDS}

    @clause("REFACTOR TRIPWIRE (ii): the SAME six-body C execution read against exact rig truth",
            f"6.865 deg on every seed, still outside O1's {O1_TILT_DEG} deg band",
            "ONE execution, two references, two verdicts")
    def _():
        values = over_seeds("C_soma_template", "pelvis_vs_truth_deg", "angle", "median")
        return (f"{min(values.values()):.4f}-{max(values.values()):.4f} deg over "
                f"{len(values)} seeds", min(values.values()) > O1_TILT_DEG)

    @clause("O1 pelvis vs truth, every seed, every frame",
            f"<= {O1_TILT_DEG} deg (from 6.865) on all {len(ORACLE_SEEDS)} seeds")
    def _():
        values = over_seeds("src_default", "pelvis_vs_truth_deg", "angle", "max")
        return (f"max {max(values.values())} deg over {len(values)} seeds",
                max(values.values()) <= O1_TILT_DEG)

    @clause("O1 `Spine` origin miss, hoist-subtracted",
            f"<= {O1_ORIGIN_MM} mm (from 21-28) on all {len(ORACLE_SEEDS)} seeds")
    def _():
        values = over_seeds("src_default", "spine_origin_miss_mm", "hoist_subtracted", "max")
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O1_ORIGIN_MM)

    @clause("O1 `Hips` origin miss, hoist-subtracted",
            f"<= {O1_ORIGIN_MM} mm (from 10) on all {len(ORACLE_SEEDS)} seeds")
    def _():
        values = over_seeds("src_default", "hips_origin_miss_mm", "hoist_subtracted", "max")
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O1_ORIGIN_MM)

    @clause("O1 torso on the unhoisted frames, ABSOLUTE row", "0.00 (from 8.98-12.09)")
    def _():
        values = over_seeds("src_default", "ABSOLUTE_groups_mm", "unhoisted_frames", "torso")
        return (f"max {max(values.values())} over {len(values)} seeds",
                max(values.values()) <= O1_ORIGIN_MM)

    @clause("O1 unnormalised three-point positional residual", f"<= {O1_RESIDUAL_M} m",
            "the clause that discriminates the wrong-origin control")
    def _():
        values = over_seeds("src_default", "three_point_residual_m", "max")
        return (f"max {max(values.values()):.3e} m over {len(values)} seeds",
                max(values.values()) <= O1_RESIDUAL_M)

    @clause("must-fail: the WRONG-ORIGIN template (a KNOWN BLINDNESS realised)",
            "invisible to the tilt band AND caught by the residual band",
            "with the residual zeroed this control is caught by NOTHING, and the residual "
            "band it is the sole evidence for means nothing either")
    def _():
        tilt = over_seeds("wrong_origin", "pelvis_vs_truth_deg", "angle", "max")
        residual = over_seeds("wrong_origin", "three_point_residual_m", "max")
        return (f"tilt max {max(tilt.values())} deg (inside {O1_TILT_DEG}), residual min "
                f"{1e3 * min(residual.values()):.1f} mm (outside {O1_RESIDUAL_M} m)",
                max(tilt.values()) <= O1_TILT_DEG
                and min(residual.values()) > O1_RESIDUAL_M)

    @clause("must-fail: a pelvis frozen upright (D7's control)", "fails O1's tilt band")
    def _():
        values = over_seeds("frozen_upright", "pelvis_vs_truth_deg", "angle", "median")
        return (f"{min(values.values()):.4f}-{max(values.values()):.4f} deg",
                min(values.values()) > O1_TILT_DEG)

    @clause("O2 legs, feet and toes vs the shipped build's FK",
            f"<= {O2_LEG_MM} mm on all {len(ORACLE_SEEDS)} seeds",
            "bit-identity is NOT claimed: a pelvis frame is whole-take")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        values = {s: r.num("oracle", "oracle", "seeds", s, "O2_vs_baseline",
                           "leg_foot_toe_max_mm") for s in ORACLE_SEEDS}
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O2_LEG_MM)

    @clause("O2 contacts identical on the oracle bodies",
            f"identical on all {len(ORACLE_SEEDS)} seeds")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        values = {s: r.flag("oracle", "oracle", "seeds", s, "O2_vs_baseline",
                            "contacts_identical") for s in ORACLE_SEEDS}
        return f"{all(values.values())} over {len(values)} seeds", all(values.values())

    @clause("O2 hoist change", f"<= {O2_HOIST_MM} mm on all {len(ORACLE_SEEDS)} seeds")
    def _():
        r.named("oracle", "oracle", "seeds", expect=ORACLE_SEEDS)
        values = {s: r.num("oracle", "oracle", "seeds", s, "O2_vs_baseline",
                           "hoist_change_mm", "max") for s in ORACLE_SEEDS}
        return (f"max {max(values.values())} mm over {len(values)} seeds",
                max(values.values()) <= O2_HOIST_MM)

    @clause("O3 the D3 gate's own leg-root-ALIGNED gauge, arms (REPORTED)",
            "1.32-2.72 -> 0.07-0.60",
            "that gauge is blind to a root move; no band reads it")
    def _():
        ours = over_seeds("src_default", "ALIGNED_rc_score_groups_mm", "arms")
        before = over_seeds("C_soma_template", "ALIGNED_rc_score_groups_mm", "arms")
        return (f"{min(before.values())}-{max(before.values())} -> "
                f"{min(ours.values())}-{max(ours.values())}", "REPORT")

    # ------------------------------------------------------ the two recorded STOPs, derived
    @clause("S at the CARD'S OWN FIXTURE (sigma 1.0): the frozen-pitch follower >= 2x on EVERY body",
            f">= {FOLLOWER_RATIO}x on all {len(ORACLE_SEEDS)} bodies",
            "the step STOPPED here; `selector.json` is immutable")
    def _():
        rows = r.named("sigma1", "frozen_pitch_follower_bent_tercile", expect=ORACLE_SEEDS)
        ratios = {s: r.num("sigma1", "frozen_pitch_follower_bent_tercile", s, "follower_i_deg")
                  / r.num("sigma1", "frozen_pitch_follower_bent_tercile", s, "winner_i_deg")
                  for s in rows}
        below = [s for s, v in ratios.items() if v < FOLLOWER_RATIO]
        return (f"{min(ratios.values()):.3f}-{max(ratios.values()):.3f}x; {len(below)} of "
                f"{len(ratios)} below {FOLLOWER_RATIO}x", not below)

    @clause("the amended card's FIXTURE CALIBRATION, under its own frozen monotonicity precondition",
            "monotone across the evaluations",
            "the step STOPPED again; `selector-calibrated.json` is immutable")
    def _():
        evaluations = r.listing("calibration", "calibration", "bisection", "evaluations",
                                minimum=2)
        ordered = sorted(range(len(evaluations)),
                         key=lambda i: r.num("calibration", "calibration", "bisection",
                                             "evaluations", i, "sigma_scale"))
        values = [r.num("calibration", "calibration", "bisection", "evaluations", i,
                        "median_of_six_guard_kept_sd_mm") for i in ordered]
        worst = max((values[i] - values[j] for i in range(len(values))
                     for j in range(i + 1, len(values)) if values[j] < values[i]), default=0.0)
        return (f"largest earlier-to-later decrease {worst:.4f} mm over {len(values)} "
                "evaluations", worst <= 0.0)

    @clause("the SAME frozen evaluations under Astra round 7's amended admissibility rule",
            f"A <= {CALIBRATION_TAU_MM} mm, B <= 1 sign change, C an accepted sigma whose OWN "
            f"statistic is within {CALIBRATION_TAU_MM} mm of the target and inside the bracket",
            "an OBSERVED TOLERANCE MATCH, never monotonicity or uniqueness. POST HOC.")
    def _():
        evaluations = r.listing("calibration", "calibration", "bisection", "evaluations",
                                minimum=2)
        target = r.num("admissibility", "admissibility", "target_mm")
        bracket = r.listing("admissibility", "admissibility", "bracket", minimum=2)
        pairs = sorted(
            (r.num("calibration", "calibration", "bisection", "evaluations", i,
                   "sigma_scale"),
             r.num("calibration", "calibration", "bisection", "evaluations", i,
                   "median_of_six_guard_kept_sd_mm"))
            for i in range(len(evaluations)))
        values = [value for _sigma, value in pairs]
        worst = max((values[i] - values[j] for i in range(len(values))
                     for j in range(i + 1, len(values)) if values[j] < values[i]), default=0.0)
        signs = [(-1 if v < target else 1) for v in values if v != target]
        changes = sum(1 for x, y in zip(signs, signs[1:]) if x != y)
        accepted = r.num("admissibility", "admissibility",
                         "replay_of_the_frozen_stopping_rule", "accepted_sigma_scale_exact")
        matched = [value for sigma, value in pairs if abs(sigma - accepted) <= 5e-7]
        if not matched:
            raise Missing("the accepted sigma is not among the frozen evaluations")
        inside = abs(matched[0] - target) <= CALIBRATION_TAU_MM
        return (f"A {worst:.4f} mm, B {changes} sign change(s), C sigma {accepted} -> "
                f"{matched[0]} mm against target {target}",
                worst <= CALIBRATION_TAU_MM and changes <= 1 and inside
                and float(bracket[0]) <= accepted <= float(bracket[1]))

    # ------------------------------------------------------------------- S, the reread
    def per_body(arm_name, population, metric):
        """THE AGGREGATE, DERIVED. `aggregated_median_of_six` is never read as evidence."""
        r.named("reread", "bodies", expect=ORACLE_SEEDS)
        return {seed: r.num("reread", "bodies", seed, "arms", arm_name, population, metric)
                for seed in ORACLE_SEEDS}

    def aggregate(arm_name, population, metric):
        derived = median(list(per_body(arm_name, population, metric).values()))
        stored = r.num("reread", "aggregated_median_of_six", arm_name, population, metric)
        if not agrees(stored, derived, 1e-3):
            raise Missing(f"aggregated_median_of_six/{arm_name}/{population}/{metric} "
                          f"stores {stored} against {derived} derived from the six bodies")
        return derived

    @clause("S REREAD: (a) vs (b), all three metrics, both populations",
            "one better-or-tied everywhere and strictly better somewhere, else SPLIT",
            "the six cells are recomputed from the per-body medians, and the implied winner "
            "is cross-checked against the mode the file says it ships")
    def _():
        cells, rebuilt = [], {}
        for population in POPULATIONS:
            for metric, tie in METRICS:
                b_value = aggregate("b_hipline_guarded", population, metric)
                a_value = aggregate("a_kabsch_guarded", population, metric)
                cell = ("tied" if abs(b_value - a_value) <= tie
                        else "better" if b_value < a_value else "worse")
                cells.append(cell)
                rebuilt[f"b_vs_a__{population}__{metric}"] = cell
        stored = r.named("reread", "b_vs_a", expect=rebuilt)
        if stored != rebuilt:
            raise Missing(f"b_vs_a stores {stored} against {rebuilt} derived")
        b_wins = all(c in ("better", "tied") for c in cells) and "better" in cells
        a_wins = all(c in ("worse", "tied") for c in cells) and "worse" in cells
        implied = ("D_rig_rest_hipline" if b_wins
                   else "E_rig_rest_kabsch" if a_wins else None)
        shipped = r.text("reread", "winner", "mode")
        status = r.text("reread", "S_verdict")
        return (f"recomputed cells {cells}; implies {implied}; ships {shipped}; "
                f"S_verdict {status}",
                implied is not None and implied == shipped and status == "PROCEED")

    @clause("S REREAD: the winner strictly better than C-on-SOMA on (i), both populations",
            "strictly better on (i), better-or-tied on (ii) and (iii)")
    def _():
        winner = r.text("reread", "winner", "arm")
        beats, detail = [], []
        for population in POPULATIONS:
            ours = aggregate(winner, population, "i_orientation_deg")
            theirs = aggregate("C_on_SOMA", population, "i_orientation_deg")
            beats.append(ours < theirs)
            detail.append(f"{population} {ours:.5f} vs {theirs:.5f}")
            for metric, tie in METRICS[1:]:
                beats.append(aggregate(winner, population, metric)
                             <= aggregate("C_on_SOMA", population, metric) + tie)
        return "; ".join(detail), all(beats)

    @clause("S REREAD: the frozen-pitch follower >= 2x the winner AND >= 2 deg, on EVERY body",
            f">= {FOLLOWER_RATIO}x and >= {FOLLOWER_FLOOR_DEG} deg on all "
            f"{len(ORACLE_SEEDS)} bodies",
            "the clause that stopped the step at sigma 1.0")
    def _():
        rows = r.named("reread", "frozen_pitch_follower_bent_tercile", expect=ORACLE_SEEDS)
        ok, ratios = True, {}
        for seed in rows:
            follower = r.num("reread", "frozen_pitch_follower_bent_tercile", seed,
                             "follower_i_deg")
            winner = r.num("reread", "frozen_pitch_follower_bent_tercile", seed,
                           "winner_i_deg")
            if winner <= 0.0:
                raise Missing(f"follower/{seed}/winner_i_deg is not positive")
            ratios[seed] = follower / winner
            stored = r.num("reread", "frozen_pitch_follower_bent_tercile", seed, "ratio")
            if not agrees(stored, ratios[seed], 1e-2):
                raise Missing(f"follower/{seed}/ratio stores {stored} against "
                              f"{ratios[seed]:.5f} derived")
            # the follower's own error is also a per-body number the reread carries
            if not agrees(follower, r.num("reread", "bodies", seed, "arms",
                                          "frozen_pitch_follower", "bent_tercile",
                                          "i_orientation_deg"), 1e-3):
                raise Missing(f"follower/{seed}/follower_i_deg disagrees with its body row")
            ok &= (ratios[seed] >= FOLLOWER_RATIO and follower >= FOLLOWER_FLOOR_DEG)
        return (f"{len(rows)} bodies; recomputed ratios {min(ratios.values()):.3f}-"
                f"{max(ratios.values()):.3f}x", ok)

    @clause("G1 (missing-only): identical masks and retained samples => bit-identical ARRAYS",
            f"holds on all {len(ORACLE_SEEDS)} named bodies",
            "an EQUIVALENCE and an error measurement; no superiority claim")
    def _():
        rows = r.named("reread", "G1_missing_only", "bodies", expect=ORACLE_SEEDS)
        holds, differing = True, 0
        for seed in rows:
            masks = r.flag("reread", "G1_missing_only", "bodies", seed,
                           "effective_masks_identical")
            arrays = r.flag("reread", "G1_missing_only", "bodies", seed,
                            "interpolated_arrays_bit_identical")
            differing += (not masks)
            holds &= ((not masks) or arrays)
        return (f"{len(rows)} bodies; {differing} have a different effective mask; identity "
                "holds wherever the masks agree", holds)

    @clause("G2 (finite-only): the guard beats the unguarded winner on BOTH (i) and (ii), every body",
            f"both metrics, all {len(ORACLE_SEEDS)} bodies and the median over them",
            "where the guard EARNS its place")
    def _():
        rows = r.named("reread", "G2_finite_only", "bodies", expect=ORACLE_SEEDS)
        wins, samples = True, {}
        for key, metric in (("i", "i_on_corrupted_frames_deg"),
                            ("ii", "ii_on_transition_pairs_deg")):
            guarded = [r.num("reread", "G2_finite_only", "bodies", s, "guarded", metric)
                       for s in rows]
            unguarded = [r.num("reread", "G2_finite_only", "bodies", s, "unguarded", metric)
                         for s in rows]
            wins &= all(g < u for g, u in zip(guarded, unguarded))
            samples[key] = (median(guarded), median(unguarded))
            for label, derived in (("guarded", samples[key][0]),
                                   ("unguarded", samples[key][1])):
                stored = r.num("reread", "G2_finite_only", "median_of_six",
                               f"{label}_{key}_deg")
                if not agrees(stored, derived, 1e-3):
                    raise Missing(f"G2 median_of_six/{label}_{key}_deg stores {stored} "
                                  f"against {derived:.5f} derived from {len(rows)} bodies")
            wins &= samples[key][0] < samples[key][1]
        return (f"{len(rows)} bodies; medians (i) {samples['i'][0]:.5f} vs "
                f"{samples['i'][1]:.5f} deg, (ii) {samples['ii'][0]:.5f} vs "
                f"{samples['ii'][1]:.5f} deg", wins)

    @clause("the world-vertical control against the truth PELVIS's own tilt", "REPORT",
            "S's stops are unchanged; the follower carries the argument")
    def _():
        control = r.num("reread", "world_vertical_vs_truth_tilt", "world_vertical_i_bent_deg")
        tilt = r.num("reread", "world_vertical_vs_truth_tilt",
                     "truth_PELVIS_bent_tilt_median_deg")
        return (f"{control} deg against the truth pelvis's {tilt} deg; limitation applies: "
                f"{abs(control - tilt) < 2.0}", "REPORT")

    # ------------------------------------------------------------------- the delivery
    @clause("the delivery: BOTH landmark arrays byte-identical (the same denominator)",
            f"raw AND smoothed identical on {len(PERFORMERS)} named performers",
            "an ABSENT comparison is not a passing one")
    def _():
        ok = True
        for key in ("raw_triangulation_byte_identical_same_denominator",
                    "smoothed_triangulation_byte_identical"):
            r.named("delivery", "hygiene", key, expect=PERFORMERS)
            ok &= all(r.flag("delivery", "hygiene", key, s) for s in PERFORMERS)
        return f"both arrays on {len(PERFORMERS)} performers: {ok}", ok

    @clause("the delivered run-report records the mode and the guard's demoted frames",
            "E_rig_rest_kabsch; 0 and 29 demoted")
    def _():
        rows = r.listing("delivery", "diagnostics", "pelvis_frame", minimum=2)
        modes = [r.text("delivery", "diagnostics", "pelvis_frame", i, "mode")
                 for i in range(len(rows))]
        demoted = [len(r.at("delivery", "diagnostics", "pelvis_frame", i, "lever_guard",
                            "demoted_frames")) for i in range(len(rows))]
        counts = [r.num("delivery", "diagnostics", "pelvis_frame", i, "lever_guard",
                        "demoted_count") for i in range(len(rows))]
        return (f"{modes}; demoted {demoted}",
                all(m == "E_rig_rest_kabsch" for m in modes) and demoted == [0, 29]
                and [int(c) for c in counts] == demoted)

    # --------------------------------------------------------------------------- P
    def run_identities(*path):
        """The runs, by (side, start, end), from the FROZEN MASK -- and the measurement rows
        must BE that set. A duplicated run keeps the count and the maximum; identity is what
        catches it."""
        expected = {(int(side), int(start), int(end))
                    for side, start, end in r.listing(*path, "mask_run_identities")}
        rows = r.listing(*path, "runs" if path[-1] != "P2_on_the_oracle_bodies"
                         else "run_measurements", minimum=0) \
            if False else None
        return expected

    def measured_runs(*path, field):
        rows = r.listing(*path, field, minimum=1)
        seen, worst, holds = [], 0.0, True
        for index in range(len(rows)):
            side = int(r.num(*path, field, index, "side_index"))
            span = r.listing(*path, field, index, "run", minimum=2)
            identity = (side, int(span[0]), int(span[1]))
            if identity in seen:
                raise Missing(f"{'/'.join(map(str, path))}/{field} repeats run {identity}")
            seen.append(identity)
            for joint in SIDE_JOINTS[side]:          # BOTH named fields, per side
                worst = max(worst, r.num(*path, field, index, f"{joint}_max_m"))
            holds &= r.flag(*path, field, index, "holds")
        return set(seen), worst, holds

    @clause("P1 channel preservation -- the delivery, both performers",
            f"every protected channel bit-identical on {len(PERFORMERS)} performers, on an "
            "AUTHENTICATED track")
    def _():
        r.named("projection", "subjects", expect=PERFORMERS)
        ok, failing = True, {}
        for performer in PERFORMERS:
            ok &= r.flag("projection", "subjects", performer, "P1_channel_preservation",
                         "authentication", "authenticated")
            channels = r.at("projection", "subjects", performer, "P1_channel_preservation",
                            "channels")
            for name in ("root_translation_m", "foot_contacts", *(f"local::{j}"
                                                                 for j in PROTECTED)):
                if name not in channels:
                    raise Missing(f"P1/{performer}/channels/{name}")
                ok &= r.flag("projection", "subjects", performer, "P1_channel_preservation",
                             "channels", name, "bit_identical")
            failing[performer] = r.at("projection", "subjects", performer,
                                      "P1_channel_preservation", "failing_channels")
            ok &= not failing[performer]
        return f"{len(PERFORMERS)} performers; failing {failing}", ok

    @clause("P2 anchor lock -- the delivery, every accepted run, on the GLB's own arrays",
            f"<= {CONTACT_TOLERANCE_M} m at every run's first KEYED sample, runs matching "
            "the frozen mask by identity")
    def _():
        r.named("projection", "subjects", expect=PERFORMERS)
        ok, worst_all, detail = True, 0.0, {}
        for performer in PERFORMERS:
            path = ("projection", "subjects", performer, "P2_anchor_lock")
            expected = {(int(a), int(b), int(c)) for a, b, c
                        in r.listing(*path, "mask_run_identities", minimum=1)}
            seen, worst, holds = measured_runs(*path, field="runs")
            if seen != expected:
                raise Missing(f"P2/{performer} runs {sorted(seen)} != mask {sorted(expected)}")
            stored = r.num(*path, "worst_travel_m")
            if not agrees(stored, worst, 1e-12):
                raise Missing(f"P2/{performer}/worst_travel_m stores {stored} against "
                              f"{worst} derived from {len(seen)} runs")
            worst_all = max(worst_all, worst)
            ok &= holds and worst <= CONTACT_TOLERANCE_M
            detail[performer] = len(seen)
        return f"worst {worst_all:.3e} m; runs {detail}", ok

    @clause("P2 anchor lock on EVERY ORACLE BODY, from each exported GLB's own arrays",
            f"<= {CONTACT_TOLERANCE_M} m on all {len(ORACLE_SEEDS)} seeds, runs matching the "
            "frozen mask by identity")
    def _():
        r.named("projection", "P2_on_the_oracle_bodies", "seeds", expect=ORACLE_SEEDS)
        ok, worst_all, detail = True, 0.0, {}
        for seed in ORACLE_SEEDS:
            path = ("projection", "P2_on_the_oracle_bodies", "seeds", seed)
            expected = {(int(a), int(b), int(c)) for a, b, c
                        in r.listing(*path, "mask_run_identities", minimum=1)}
            seen, worst, holds = measured_runs(*path, field="run_measurements")
            if seen != expected:
                raise Missing(f"oracle P2/{seed} runs {sorted(seen)} != mask "
                              f"{sorted(expected)}")
            stored = r.num(*path, "worst_travel_m")
            if not agrees(stored, worst, 1e-12):
                raise Missing(f"oracle P2/{seed}/worst_travel_m stores {stored} against "
                              f"{worst} derived")
            if int(r.num(*path, "runs")) != len(seen):
                raise Missing(f"oracle P2/{seed}/runs count disagrees with its rows")
            worst_all = max(worst_all, worst)
            ok &= holds and worst <= CONTACT_TOLERANCE_M
            detail[seed] = len(seen)
        global_stored = r.num("projection", "P2_on_the_oracle_bodies",
                              "worst_travel_m_over_all_seeds")
        if not agrees(global_stored, worst_all, 1e-12):
            raise Missing(f"oracle P2 global summary stores {global_stored} against "
                          f"{worst_all} derived from the seeds")
        return f"worst {worst_all:.3e} m re-derived from the runs; runs {detail}", ok

    @clause("P3 planted-foot travel on the frozen UNION of both builds' runs", "REPORT")
    def _():
        r.named("projection", "subjects", expect=PERFORMERS)
        total = sum(len(r.listing("projection", "subjects", s, "P3_travel_report",
                                  "intervals", minimum=1)) for s in PERFORMERS)
        return f"{total} intervals", "REPORT"

    @clause("P1 on EVERY ORACLE BODY (the card says the take AND every oracle body)",
            f"every protected channel bit-identical on all {len(ORACLE_SEEDS)} seeds")
    def _():
        r.named("p_oracle", "seeds", expect=ORACLE_SEEDS)
        ok, clean = True, 0
        for seed in ORACLE_SEEDS:
            ok &= r.flag("p_oracle", "seeds", seed, "root_bit_identical")
            ok &= r.flag("p_oracle", "seeds", seed, "contacts_bit_identical")
            for joint in PROTECTED:
                ok &= r.flag("p_oracle", "seeds", seed, f"local::{joint}")
            failing = r.at("p_oracle", "seeds", seed, "failing_channels")
            ok &= not failing
            clean += not failing
        return f"{clean} of {len(ORACLE_SEEDS)} clean", ok

    @clause("P1's CONTROL 1 -- the projection's foot locals overwritten", "must FAIL P1",
            "refused by the shipping path is STRONGER than caught by a gate")
    def _():
        r.named("p1_controls", "controls", expect=PERFORMERS)
        failing = {s: r.at("p1_controls", "controls", s,
                           "control_1_foot_locals_overwritten", "failing_channels")
                   for s in PERFORMERS}
        return (f"the shipping path REFUSES to build it; applied to the delivered bytes P1 "
                f"detects it: {failing}", all(failing.values()))

    @clause("P1's CONTROL 2 -- the nonempty contact mask cleared (offline)",
            "must FAIL P1 on the mask")
    def _():
        r.named("p1_controls", "controls", expect=PERFORMERS)
        failing = {s: r.at("p1_controls", "controls", s, "control_2_contact_mask_cleared",
                           "failing_channels") for s in PERFORMERS}
        return str(failing), all(failing.values())

    @clause("the UNMUTATED delivery through the same comparison", "PASS")
    def _():
        r.named("p1_controls", "controls", expect=PERFORMERS)
        failing = {s: r.at("p1_controls", "controls", s, "the_unmutated_delivery",
                           "failing_channels") for s in PERFORMERS}
        return str(failing), not any(failing.values())

    @clause("P1's CONTROL 2, BUILT and run through the P instrument", "must FAIL P1",
            "an INSTRUMENT DEFECT was found by this very control")
    def _():
        r.named("control2", "subjects", expect=PERFORMERS)
        failing = {s: r.at("control2", "subjects", s, "P1_channel_preservation",
                           "failing_channels") for s in PERFORMERS}
        return str(failing), all(failing.values())

    # -------------------------------------------------------------------------- B1, B2
    @clause("B1 the photographs: worsening NOT ESTABLISHED (ci95 upper bound >= 0), 8 cells",
            f"ci95[1] >= 0 on all {len(B1_CELLS) * len(PERFORMERS)} NAMED cells",
            "it does NOT establish non-worsening; a wide interval passes it for want of power")
    def _():
        ok, seen = True, 0
        for performer in PERFORMERS:
            for name in B1_CELLS:
                interval = r.listing("silhouette", "preregistered_clause_verdicts",
                                     performer, name, "ci95", minimum=2)
                ok &= float(interval[1]) >= 0.0
                seen += 1
        return f"{seen} of {len(B1_CELLS) * len(PERFORMERS)} named cells checked", ok

    @clause("B1 the MAMMA mesh oracle bit-identical", "< 1e-9")
    def _():
        worst = r.num("silhouette", "preregistered_clause_verdicts",
                      "clause_mamma_mesh_oracle",
                      "this_instruments_split_oracle_vs_the_committed_unsplit_one_worst_abs_difference")
        return str(worst), worst < 1e-9

    @clause("B2 `delivered_vs_capture.py --reference smoothed`: the same-denominator clause",
            "PASS (landmarks byte-identical)",
            "CHANGED would mean the change did more than refit the pelvis")
    def _():
        value = r.at("b2", "same_denominator")
        if isinstance(value, dict):
            value = value.get("verdict", value)
        return str(value), str(value).upper() in ("TRUE", "PASS")

    # --------------------------------------------------------------- the REPORT blocks
    @clause("B4 the pelvis and the root's motion (REPORTED, never banded)", "REPORT",
            "the 800 deg/s line is a physical REFERENCE, not a band")
    def _():
        r.named("take", "take", "subjects", expect=PERFORMERS)
        rows = [f"{s}: pitch "
                f"{r.num('take', 'take', 'subjects', s, 'vs_baseline', 'pelvis_change_deg', 'pitch_about_hip_line_signed_median')} deg, "
                f"root {r.num('take', 'take', 'subjects', s, 'vs_baseline', 'root_move_mm_hoist_subtracted', 'median')} mm, "
                f"step p95 {r.num('take', 'take', 'subjects', s, 'pelvis_step_deg_per_frame', 'p95')} deg, "
                f"{int(r.num('take', 'take', 'subjects', s, 'frames_over_800_deg_per_s'))} over 800 deg/s"
                for s in PERFORMERS]
        return "; ".join(rows), "REPORT"

    @clause("B4 the leg-root midpoint stays on the captured hip midpoint", "0.0 mm",
            "NOT (b)-specific: the card listed this as conditional on (b) in error")
    def _():
        r.named("take", "take", "subjects", expect=PERFORMERS)
        return ("; ".join(
            f"{s}: max "
            f"{r.num('take', 'take', 'subjects', s, 'leg_roots_on_captured_hip_midpoint_mm', 'max')} mm"
            for s in PERFORMERS), "REPORT")

    @clause("B2/B4 the hip residual under (a) -- a REPORT, and NO band may be made from it",
            "REPORT")
    def _():
        r.named("take", "take", "subjects", expect=PERFORMERS)
        return ("; ".join(
            f"{s}: full p95 "
            f"{r.num('take', 'take', 'subjects', s, 'vs_baseline', 'hip_residual_REPORT_never_a_band', 'baseline', 'full_positional_mm', 'p95')}"
            f" -> {r.num('take', 'take', 'subjects', s, 'vs_baseline', 'hip_residual_REPORT_never_a_band', 'candidate', 'full_positional_mm', 'p95')} mm"
            for s in PERFORMERS), "REPORT")

    @clause("B1 attribution of the three rising torso cells (DIAGNOSTIC)", "REPORT",
            "a POINT-ESTIMATE decomposition; only performer 0's ARTICULATION shares have "
            "intervals clear of zero")
    def _():
        r.named("b1_attribution", "subjects", expect=PERFORMERS)
        rows = []
        for performer in PERFORMERS:
            base = ("b1_attribution", "subjects", performer, "torso", "whole_take")
            rows.append(
                f"{performer}: both "
                f"{r.num(*base, 'candidate_minus_D9b__both_effects', 'median_difference'):+.5f}"
                f" = articulation "
                f"{r.num(*base, 'ablation_minus_D9b__the_ARTICULATION_alone', 'median_difference'):+.5f}"
                f" + root "
                f"{r.num(*base, 'candidate_minus_ablation__the_ROOT_TRANSLATION_alone', 'median_difference'):+.5f}"
                f" (root CI "
                f"{r.listing(*base, 'candidate_minus_ablation__the_ROOT_TRANSLATION_alone', 'ci95', minimum=2)})")
        return "; ".join(rows), "REPORT"

    @clause("B3 the hoist and the contacts (REPORTED)", "REPORT")
    def _():
        rows = []
        for label in ("D9b_shipped", "D7c_candidate"):
            r.named("b3", "arms", label, "subjects", expect=PERFORMERS)
            for performer in PERFORMERS:
                rows.append(
                    f"{label} {performer}: hoist p95 "
                    f"{r.num('b3', 'arms', label, 'subjects', performer, 'hoist_mm', 'p95')} mm, "
                    f"contacts {r.at('b3', 'arms', label, 'subjects', performer, 'contacts', 'count')}")
        return "; ".join(rows), "REPORT"

    @clause("B6 the delivered bytes (REPORT)", "REPORT")
    def _():
        base = ("b6", "builds", "D7c", "subject_00")
        return (f"LINEAR samplers, {int(r.num(*base, 'sampler_input_times', 'frames'))} "
                f"frames, {r.num(*base, 'duration_s'):.4f} s; track->GLB positional closure "
                f"max {r.num(*base, 'track_to_glb_closure', 'positional_mm', 'max')} mm; "
                f"rotational closure median "
                f"{r.num(*base, 'track_to_glb_closure', 'rotational_deg_frame_corrected', 'median')} deg",
                "REPORT")

    @clause("B6 the carried-tetrahedron PROXY (NOT an inversion count)", "REPORT",
            "a PROXY with two demonstrated failure modes -- it mis-classifies a proper rigid "
            "motion under varying weights and it is vertex-order dependent -- so NO inversion "
            "claim is made. The sound measurement is the skinning Jacobian with spatially "
            "varying weights (Kavan, direct methods eq. 17) and is D6's.")
    def _():
        base = ("b6", "builds", "D7c", "subject_00", "mesh_deformation_pelvis_hip_thigh")
        low = int(r.num(*base, "carried_tetrahedron_PROXY", "per_frame_min"))
        high = int(r.num(*base, "carried_tetrahedron_PROXY", "per_frame_max"))
        return (f"fires on {low}-{high} of {int(r.num(*base, 'triangles'))} triangles per "
                f"frame", "REPORT")

    @clause("B6 the `Root` / eye / finger local invariants vs D9b (a TRACK-ARRAY claim)",
            "bit-identical")
    def _():
        node = r.at("b6", "builds", "D7c", "subject_00",
                    "invariants_vs_the_other_build_TRACK_ARRAYS")
        return str(node), "REPORT"

    @clause("B5b the delivered `Head` WORLD rotation, from the GLB's own bytes", "REPORT",
            "NOT identical; the earlier claim of identity is withdrawn")
    def _():
        r.named("b6", "B5b_head_world_between_builds", expect=PERFORMERS)
        return ("; ".join(
            f"{s}: {r.num('b6', 'B5b_head_world_between_builds', s, 'per_frame_difference_deg', 'median')} deg "
            f"median, {r.num('b6', 'B5b_head_world_between_builds', s, 'per_frame_difference_deg', 'max')} max"
            for s in PERFORMERS), "REPORT")

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
        "touched": sorted("/".join(path) for path in r.touched),
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


def load_all() -> dict:
    out = {}
    for key, name in REPORTS.items():
        path = BASE / name
        out[key] = json.loads(path.read_text()) if path.exists() else {}
    return out


def main() -> int:
    reports = load_all()
    built = build(reports)
    fuzz_path = BASE / "gate-fuzz.json"
    fuzz = json.loads(fuzz_path.read_text()) if fuzz_path.exists() else {
        "status": "not run -- `tools/compare/d7c_gate_fuzz.py` has not been executed"}
    report = {
        "title": ("D7c -- the pelvis on the rig's own rest. Every clause, predicted / "
                  "measured / verdict, DERIVED from the reports."),
        "shipping_mode": "E_rig_rest_kabsch",
        "three_rules": [
            "every value is DERIVED from named constituents or CROSS-CHECKED against them; a "
            "stored summary that disagrees with its constituents is a FAIL",
            "a MISSING field or set member is a FAIL, never a no-op",
            "every set is checked by IDENTITY -- files, seeds, performers, cells, and contact "
            "runs by their (side, start, end) identity from the frozen mask",
        ],
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
        "leaf_level_fuzz": fuzz,
    }
    (BASE / "gate.json").write_text(json.dumps(report, indent=1))
    for entry in built["clauses"]:
        print(f"{entry['verdict']:7s} {entry['clause'][:74]:74s} "
              f"{str(entry['measured'])[:44]}")
    print()
    print("MERGE RULE:", json.dumps(built["conjuncts"], indent=1))
    print("verdict:", built["verdict"], "| missing:", built["not_yet_measured"])
    print(f"fuzz: {fuzz.get('summary', fuzz.get('status'))}")
    print(f"\nwrote {BASE / 'gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
