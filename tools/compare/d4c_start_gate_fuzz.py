#!/usr/bin/env python3
"""D4c: PROVE the ONE verdict by mutating its INPUTS, never its verdicts (the `d7c_gate_fuzz.py` pattern).

Two parts.

1. THE CONJUNCTS, TURNED. For every conjunct of the card's conjunction -- precondition 0, stage 0b, validity, L,
   closure, must-fails i-iv, B1 (band, binding, frozen-pose control, MAMMA unchanged), B2, hygiene (rig rebuild,
   tripwire, source diff) -- and for the population, construction and freeze-order rules, a named mutation of the
   gate's INPUTS must move the verdict from its baseline to the verdict the card gives that failure (STOP for
   precondition 0, stage 0b and init-only passing; INVALID for validity; FAIL for the rest).

2. A BOUNDED LEAF WALK over one acceptance fixture's six cells, its closure row, the B1 paired rows and the B2
   checks: every leaf is set to a mismatched value and DELETED in turn, and classified ENFORCED (the verdict moves)
   or REPORTED (it moves only a reported value, or nothing). A REPORTED leaf must match a named justification or
   the fuzz exits non-zero. Long numeric lists are walked at sampled members (first, middle, last).

WHAT A PASS DOES NOT PROVE: that the gate reads the RIGHT things -- only that what it reads, it depends on.
IF A LEAF ESCAPES, FIX THE GATE, NOT THE FUZZER.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate_fuzz.py --out artifacts/compare/d4c-start/fuzz.json
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
import d4c_start_gate as gate  # noqa: E402

FIXTURE = (20261101, 0)
MISMATCH = "0" * 64

CONJUNCT_KEYS = {"validity": "validity", "L": "L", "closure": "closure", "must_fail_i": "must_fail_i_mean_body_misses_L",
                 "must_fail_ii": "must_fail_ii_spine_displaced_fails_L_at_the_trunk",
                 "must_fail_iii": "must_fail_iii_exact_identity_reads_zero_and_passes", "B1": "B1", "B2": "B2",
                 "hygiene": "hygiene", "population": None, "freeze_order": None, "construction": None}

JUSTIFIED = {
    "cell(*).distance_mm*": "REPORTED as J and the pooled statistic; shape and finiteness are the population "
                            "(sampled members move a reported value only)",
    "cell(*).locator_offset_mm_max": "DIAGNOSTIC: the soft pin's reach; not a card clause",
    "cell(*).calibration_iterations.per_call*": "REPORTED: the cap-exhaustion counts (card: reported)",
    "cell(*).start.record*": "REPORTED: the fitter's own record of its start; the start itself is recomputed",
    "cell(*).fixture.drawn_identity*": "LABEL: the draw is REGENERATED from (seed, donor) and compared to the truth",
    "cell(*).fixture.truth_motion_sha256": "LABEL: the truth is compared across arms by its identity and rest arrays",
    "cell(*).glb": "ENFORCED via its hash on the candidate arm only",
    "cell(*).track": "ENFORCED via its hash on the candidate arm only",
    "cell(*-candidate).fitted_identity*": "REPORTED: the recovered per-channel error (card: reported); L reads the rest",
    "cell(*-legacy).*": "REPORTED: the legacy zero-start arm is the before arm, never banded",
    "cell(*-init_only).distance_mm*": "REPORTED",
    "cell(*).arm_note.*": "LABEL: spine_displaced's construction is read from its identities",
    "closure.p95_abs_m": "DERIVED by d4_glb_closure; not a clause",
    "closure.band_m": "DERIVED: the gate carries D3's band itself",
    "closure.within_band": "DERIVED verdict; re-derived from max_abs_m",
    "closure.glb": "a path; the binding is by content (glb_sha256)",
    "closure.track": "a path; the binding is by content (track_sha256)",
    "b1.*.lag1_autocorrelation": "REPORTED by the bootstrap",
    "b1.*.population_as_named": "DERIVED; the gate re-derives it from n and cells_required",
    "b1.*.cells_required": "DERIVED; the gate re-derives it from the mask exclusions",
    "b1.*D4_fitted_MHR_lod2_subject_*.median_difference": "REPORTED: D4c - D4 is not a band",
    "b1.*D4_fitted_MHR_lod2_subject_*.ci95_of_the_median_difference*": "REPORTED: D4c - D4 is not a band",
    "b1.*baseline_D7c_rig_subject_*.median_difference": "REPORTED beside the band (the band reads the lower CI)",
    "b1.*baseline_D7c_rig_subject_*.ci95_of_the_median_difference.1": "the upper CI bound is reported",
    "b2.*.landmarks_*": "REPORTED: what the route consumes",
    "b2.*.declared_landmark_to_joint*": "REPORTED",
    "b2.*.occluded_marker_fraction": "REPORTED",
    "b2.*.max_abs_marker_difference_cm": "DERIVED; the check 4a is read",
    "b2.*.verdict": "DERIVED verdict; the gate reads the numbered checks, never this leaf",
    "b2.*.consumed_array_key": "LABEL",
    "b2.*.omitted*": "REPORTED",
}


def run(inputs: dict) -> str:
    return gate.build(inputs)["verdict"]


FAILING = (20261106, 0)   # the one acceptance fixture whose trunk misses L on the measured run


def repaired(base: dict) -> dict:
    """The PASS baseline: the measured inputs with the ONE failing candidate trunk brought onto the truth.

    The measured verdict is FAIL (L, one fixture), and a FAIL baseline makes every FAIL-direction mutation vacuous.
    So each conjunct is turned from THIS baseline, which must read PASS -- itself the proof that L's FAIL is read
    from the measurement and that a PASS is reachable -- and the conjunct each mutation targets is also checked to
    flip on the measured inputs."""
    inputs = copy.deepcopy(base)
    c = cell(inputs, "candidate", *FAILING)
    c["fitted_rest_mapped_cm"] = copy.deepcopy(c["truth_rest_mapped_cm"])
    return inputs


def cell(inputs: dict, arm: str, seed: int = FIXTURE[0], donor: int = FIXTURE[1]) -> dict:
    return inputs["acceptance"]["cells"][f"cell-{seed}-d{donor}-{arm}.json"]


def targeted(base: dict) -> list[dict]:
    rows = []

    passing = repaired(base)
    measured_conjuncts = gate.build(base)["conjuncts"]

    def attempt(name: str, conjunct: str, want: str, mutate):
        inputs = copy.deepcopy(passing)
        mutate(inputs)
        got = run(inputs)
        # the same mutation on the MEASURED inputs: the targeted conjunct (or the population rule) must fall
        measured = copy.deepcopy(base)
        mutate(measured)
        report = gate.build(measured)
        key = CONJUNCT_KEYS.get(conjunct)
        # a conjunct already failing on the measured inputs (L) is turned by the REPAIR row instead; here it must
        # stay failed
        flipped = (not report["population_ok"]) if key is None else (
            report["conjuncts"].get(key) is False and measured_conjuncts.get(key) in (True, False))
        if conjunct in ("precondition_0", "stage_0b", "must_fail_iv"):
            flipped = report["verdict"] == want
        turns = got == want and flipped
        rows.append({"mutation": name, "conjunct": conjunct, "want": want, "got_from_the_PASS_baseline": got,
                     "measured_verdict": report["verdict"], "conjunct_flipped_on_the_measured_inputs": flipped,
                     "turns": turns})
        print(f"  {name:72s} want {want:8s} got {got:8s} measured-flip {str(flipped):5s} "
              f"{'ok' if turns else 'MISSED'}", flush=True)

    got = run(passing)
    rows.append({"mutation": "REPAIR: the failing fixture's candidate trunk brought onto the truth", "conjunct": "L",
                 "want": "PASS", "got_from_the_PASS_baseline": got, "turns": got == "PASS"})
    print(f"  {'REPAIR: the one failing trunk onto the truth (the PASS baseline)':72s} want PASS     got {got}", flush=True)

    def drawn_without_spine(i):
        d = json.loads(i["drawn_set_bytes"])
        d["per_donor"]["0"]["bounded_residual_median_mm"]["scale_spine_length"] = 1.0
        d["drawn_set"].remove("scale_spine_length")
        i["drawn_set_bytes"] = json.dumps(d).encode()

    def solver_failure(i):
        d = json.loads(i["drawn_set_bytes"])
        d["precondition_0"]["solver_failures"] = [{"donor": 0, "frame": 0, "channel": "scale_spine_length",
                                                   "end": "low", "status": 0}]
        i["drawn_set_bytes"] = json.dumps(d).encode()

    def dev_spine_passes(i):
        c = i["development"]["cells"]["cell-20260922-d0-spine_displaced.json"]
        c["fitted_rest_mapped_cm"] = c["truth_rest_mapped_cm"]

    def drop(i, arm):
        del i["acceptance"]["cells"][f"cell-{FIXTURE[0]}-d{FIXTURE[1]}-{arm}.json"]

    def truncate(i):
        cell(i, "candidate")["distance_mm"] = cell(i, "candidate")["distance_mm"][:15]

    def nonfinite(i):
        cell(i, "candidate")["fitted_rest_mapped_cm"][0][0] = float("nan")

    def invalid(i):
        c = cell(i, "exact_identity")
        c["distance_mm"] = (np.asarray(c["distance_mm"]) * 5.0 + 1.0).tolist()

    def move_trunk(i, arm, mm):
        c = cell(i, arm)
        c["fitted_rest_mapped_cm"] = copy.deepcopy(c["truth_rest_mapped_cm"]) if mm is None else c["fitted_rest_mapped_cm"]
        if mm is not None:
            c["fitted_rest_mapped_cm"][1][1] += mm / 10.0

    def truth_all_arms(i, arm):
        c = cell(i, arm)
        c["fitted_rest_mapped_cm"] = copy.deepcopy(c["truth_rest_mapped_cm"])

    def init_only_passes_everywhere(i):
        for s, d in gate.fx.POPULATIONS["acceptance"]:
            c = cell(i, "init_only", s, d)
            c["fitted_rest_mapped_cm"] = copy.deepcopy(c["truth_rest_mapped_cm"])

    def exact_off(i):
        c = cell(i, "exact_identity")
        c["fitted_rest_mapped_cm"][0][0] += 0.01

    def closure_far(i):
        i["closure"]["pairs"][f"{FIXTURE[0]}_d{FIXTURE[1]}"]["max_abs_m"] = 0.01

    def closure_hash(i):
        i["closure"]["pairs"][f"{FIXTURE[0]}_d{FIXTURE[1]}"]["glb_sha256"] = MISMATCH

    def b1_ci(i):
        i["b1-paired.json"]["paired"][f"{gate.B1_CANDIDATE}_minus_{gate.B1_BASELINE}_subject_01"][
            "ci95_of_the_median_difference"][0] = -0.01

    def b1_ci_zero(i):
        i["b1-paired.json"]["paired"][f"{gate.B1_CANDIDATE}_minus_{gate.B1_BASELINE}_subject_00"][
            "ci95_of_the_median_difference"][0] = 0.0

    def b1_short(i):
        i["b1-paired.json"]["population"]["frames_consumed_per_arm"][gate.B1_CANDIDATE]["subject_00"] = 15

    def b1_frozen(i):
        i["silhouette-delivery.json"]["arms"]["control_frozen_pose_tracked"]["A001"]["subject_00"]["iou"]["median"] = 1.0

    def b1_mamma(i):
        i["silhouette-delivery.json"]["arms"]["ORACLE_mamma_mesh"]["B001"]["subject_01"]["iou"]["median"] += 1e-9

    def b1_binding(i):
        i["b1_files"]["D4c_mesh"] = MISMATCH

    def b1_glb(i):
        i["silhouette-delivery.json"]["input_sha256"]["subject-00.glb"] = MISMATCH

    def b2_check(i):
        i["b2-same-denominator.json"]["subjects"]["subject_01"][
            "1a_handed_array_is_byte_identical_to_the_rig_converter_input_on_this_build"] = False

    def b2_path(i):
        i["b2-same-denominator.json"]["delivery"] = str(ROOT / "artifacts/compare/d4-body/delivery")

    def hygiene_rig(i):
        i["hygiene"]["rebuild"]["subject-00.glb"] = MISMATCH

    def tripwire(i):
        key = next(iter(i["tripwire_files"]))
        i["tripwire_files"][key][1] = MISMATCH

    def tripwire_record(i):
        i["tripwire_record"]["all_byte_identical_or_equal_after_normalising"] = False

    def source_diff(i):
        i["fitter_source"] = i["fitter_source"].replace("tracking.max_iter = max_iter", "tracking.max_iter = 300")

    def order(i):
        i["commits"]["development_json_time"] = i["commits"]["acceptance_manifest_time"] + 1

    def manifest(i):
        name = next(iter(i["manifest"]["cells_sha256"]))
        i["manifest"]["cells_sha256"][name] = MISMATCH

    def draw(i):
        for arm in gate.fx.ACCEPTANCE_ARMS:
            c = cell(i, arm)
            k = c["identity_channel_names"].index("scale_uparms")
            c["truth_identity"][k] += 0.01
            if arm == "exact_identity":
                c["fitted_identity"][k] += 0.01
            if arm == "spine_displaced":
                c["fitted_identity"][k] += 0.01

    def init_start(i):
        c = cell(i, "init_only")
        k = c["identity_channel_names"].index("scale_spine_length")
        c["fitted_identity"][k] += 0.01
        c["start"]["start_identity"][k] += 0.01

    def statistic(i):
        i["fitter_source"] = i["fitter_source"].replace(
            f'TRUNK_STATISTIC: str | None = "{i["_chosen"]}"', 'TRUNK_STATISTIC: str | None = "p95"'
            if i["_chosen"] != "p95" else 'TRUNK_STATISTIC: str | None = "median"')

    attempt("precondition 0: the spine below the rule on one donor", "precondition_0", "STOP", drawn_without_spine)
    attempt("precondition 0: a solver failure recorded", "precondition_0", "STOP", solver_failure)
    attempt("stage 0b: a development spine control reads the truth's trunk", "stage_0b", "STOP", dev_spine_passes)
    attempt("population: a candidate cell deleted", "population", "FAIL", lambda i: drop(i, "candidate"))
    attempt("population: a candidate cell truncated to 15 frames", "population", "FAIL", truncate)
    attempt("population: a non-finite rest", "population", "FAIL", nonfinite)
    attempt("validity: the exact floor inflated past 1 mm", "validity", "INVALID", invalid)
    attempt("L: the candidate's c_neck moved 5 mm (trunk)", "L", "FAIL", lambda i: move_trunk(i, "candidate", 5.0))
    attempt("closure: 0.01 m", "closure", "FAIL", closure_far)
    attempt("closure: the measured GLB hash is another file's", "closure", "FAIL", closure_hash)
    attempt("must-fail i: the mean body reads the truth's rest", "must_fail_i", "FAIL",
            lambda i: truth_all_arms(i, "mean_body"))
    attempt("must-fail ii: the displaced spine reads the truth's rest", "must_fail_ii", "FAIL",
            lambda i: truth_all_arms(i, "spine_displaced"))
    attempt("must-fail iii: the exact arm's rest moved 0.1 mm", "must_fail_iii", "FAIL", exact_off)
    attempt("must-fail iv: init-only reads the truth's rest on every fixture", "must_fail_iv", "STOP",
            init_only_passes_everywhere)
    attempt("B1: performer 1's lower CI below zero", "B1", "FAIL", b1_ci)
    attempt("B1: performer 0's lower CI exactly zero", "B1", "FAIL", b1_ci_zero)
    attempt("B1: the candidate arm consumed 15 frames", "B1", "FAIL", b1_short)
    attempt("B1: the frozen-pose control lifted to 1.0", "B1", "FAIL", b1_frozen)
    attempt("B1: MAMMA's arm no longer bit-identical", "B1", "FAIL", b1_mamma)
    attempt("B1: the scored mesh is not the rebuilt delivery's", "B1", "FAIL", b1_binding)
    attempt("B1: the silhouette read another GLB", "B1", "FAIL", b1_glb)
    attempt("B2: check 1a false on performer 1", "B2", "FAIL", b2_check)
    attempt("B2: the report is D4's delivery, not the rebuilt one", "B2", "FAIL", b2_path)
    attempt("hygiene: one rig file's sha256 changed", "hygiene", "FAIL", hygiene_rig)
    attempt("hygiene: one tripwire file differs", "hygiene", "FAIL", tripwire)
    attempt("hygiene: the tripwire JSON comparison false", "hygiene", "FAIL", tripwire_record)
    attempt("hygiene: fit_one changes beyond the two starts (tracking max_iter)", "hygiene", "FAIL", source_diff)
    attempt("freeze order: the development JSON committed after the manifest", "freeze_order", "FAIL", order)
    attempt("freeze order: the manifest names another cell", "freeze_order", "FAIL", manifest)
    attempt("construction: the uparms draw is not the regenerated one", "construction", "FAIL", draw)
    attempt("construction: init-only's held start is not the frozen rule's", "construction", "FAIL", init_start)
    attempt("construction: the fitter's TRUNK_STATISTIC is not the development choice", "construction", "FAIL",
            statistic)
    return rows


# ------------------------------------------------------------------------------------ the leaf walk

def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list) and node and all(isinstance(x, (int, float)) for x in node) and len(node) > 3:
        for i in sorted({0, len(node) // 2, len(node) - 1}):
            yield path + (i,), node[i]
    elif isinstance(node, list) and node and isinstance(node[0], list) and len(node) > 3:
        for i in sorted({0, len(node) // 2, len(node) - 1}):
            yield from leaves(node[i], path + (i,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (i,))
    else:
        yield path, node


def set_at(root, path, value, delete=False):
    node = root
    for key in path[:-1]:
        node = node[key]
    if delete:
        if isinstance(node, list):
            del node[path[-1]]
        else:
            del node[path[-1]]
    else:
        node[path[-1]] = value


def mismatch(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1e3 if value == value else 0.0
    if isinstance(value, str):
        return value + "-mutated"
    if value is None:
        return "mutated"
    return None


def walk(base: dict, baseline: str) -> list[dict]:
    targets = []
    for arm in gate.fx.ACCEPTANCE_ARMS:
        name = f"cell-{FIXTURE[0]}-d{FIXTURE[1]}-{arm}.json"
        targets.append((f"cell({FIXTURE[0]}-d{FIXTURE[1]}-{arm})", ("acceptance", "cells", name)))
    targets.append(("closure", ("closure", "pairs", f"{FIXTURE[0]}_d{FIXTURE[1]}")))
    targets.append(("b1", ("b1-paired.json", "paired")))
    targets.append(("b2", ("b2-same-denominator.json", "subjects")))
    out = []
    for label, root_path in targets:
        root = base
        for key in root_path:
            root = root[key]
        for path, value in list(leaves(root)):
            moved = False
            for delete in (False, True):
                inputs = copy.deepcopy(base)
                node = inputs
                for key in root_path:
                    node = node[key]
                try:
                    set_at(node, path, None if delete else mismatch(value), delete=delete)
                except (KeyError, IndexError, TypeError):
                    continue
                try:
                    got = run(inputs)
                except Exception:           # a crash on a mutated input is a hole, not a pass
                    got = "CRASH"
                if got != baseline:
                    moved = True
                    break
            dotted = label + "." + ".".join(str(p) for p in path)
            reason = next((why for pattern, why in JUSTIFIED.items() if fnmatch.fnmatch(dotted, pattern)
                           or fnmatch.fnmatch(dotted.replace(f"{FIXTURE[0]}-d{FIXTURE[1]}-", "*-"), pattern)), None)
            out.append({"leaf": dotted, "class": "ENFORCED" if moved else "REPORTED",
                        "justification": None if moved else reason})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-walk", action="store_true")
    arguments = parser.parse_args()
    base = gate.load_inputs()
    report = gate.build(base)
    baseline = report["verdict"]
    base["_chosen"] = report["chosen_trunk_statistic"]
    print("measured baseline:", baseline, "--", report["reason"], flush=True)
    rows = targeted(base)
    walked = [] if arguments.no_walk else walk(repaired(base), "PASS")
    unjustified = [w["leaf"] for w in walked if w["class"] == "REPORTED" and not w["justification"]]
    result = {"baseline": baseline, "walk_baseline": "PASS (the repaired inputs)", "targeted": rows, "all_targeted_turn": all(r["turns"] for r in rows),
              "leaf_walk": walked, "enforced": sum(w["class"] == "ENFORCED" for w in walked),
              "reported_justified": sum(w["class"] == "REPORTED" and bool(w["justification"]) for w in walked),
              "unjustified": unjustified,
              "bounded": "one acceptance fixture's six cells walked leaf by leaf (long lists at first/middle/last), "
                         "its closure row, the B1 paired rows and the B2 checks; the other eleven fixtures run "
                         "through the same code path"}
    args_out = arguments.out
    args_out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"targeted: {sum(r['turns'] for r in rows)}/{len(rows)} turn; walk: {result['enforced']} enforced, "
          f"{result['reported_justified']} reported (justified), {len(unjustified)} unjustified")
    for leaf in unjustified[:40]:
        print("  UNJUSTIFIED:", leaf)
    return 0 if result["all_targeted_turn"] and not unjustified else 1


if __name__ == "__main__":
    raise SystemExit(main())
