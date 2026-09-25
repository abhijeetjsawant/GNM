#!/usr/bin/env python3
"""D4d: PROVE the ONE verdict by mutating its INPUTS, never its verdicts (the `d7c_gate_fuzz.py` pattern), with
CRASH kept as its OWN class (D4c's fuzz counted a crash on a mutated input as ENFORCED; Astra's merge round).

Two parts.

1. THE CONJUNCTS, TURNED. For every conjunct of the card's conjunction -- precondition 0, the Phase-1 population,
   stage 0b, the WARM fork, the TWO-PASS rule, the committed decision, the Phase-2 population / construction /
   freeze order, validity, L, closure, must-fails i-iv, B1 (band, binding, frozen-pose control, MAMMA unchanged),
   B2 (every numbered check, turned by mutating the ARRAYS it is re-derived from), hygiene (rig rebuild, tripwire
   bytes, source diff) -- a named mutation of the gate's INPUTS must move the verdict from the measured PASS to the
   verdict the card gives that failure. A mutation that CRASHES the gate is a hole, never a turn.
   Two NEGATIVE controls must NOT move it: a tripwire difference confined to a named normalisation, and a REPORTED
   value.

2. A BOUNDED LEAF WALK over one Phase-2 fixture's seven cells, its closure row, the B1 paired rows, and one
   Phase-1 fork fixture's three new cells: every leaf is set to a mismatched value and DELETED in turn, and
   classified ENFORCED (the verdict moves to a non-PASS verdict), CRASH (the gate raised), or REPORTED (nothing
   moved). A REPORTED leaf must match a named justification; any CRASH or unjustified leaf exits non-zero. Long
   numeric lists are walked at sampled members (first, middle, last). Phase-1 leaves are walked TWICE: as committed
   (a changed file re-hashes, and the committed decision records every Phase-1 file by hash) and with the decision
   RE-BOUND to the mutated cells and their hashes untouched (which isolates what the frozen rule and the construction
   checks enforce). Phase-2 leaves are walked rule-isolated: the manifest's hashes are NOT moved with the cell, so
   only the clauses and construction checks can react.

WHAT A PASS DOES NOT PROVE: that the gate reads the RIGHT things -- only that what it reads, it depends on.
IF A LEAF ESCAPES, FIX THE GATE, NOT THE FUZZER.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate_fuzz.py --out artifacts/compare/d4d-twopass/fuzz.json
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
import d4d_twopass_gate as gate  # noqa: E402

P2 = (20261201, 1)                     # the Phase-2 fixture walked (the worst candidate trunk on the measured run)
P1 = ("d4c", 20261106, 0)              # the Phase-1 fork fixture walked (D4c's failing fixture)
MISMATCH = "0" * 64

JUSTIFIED = {
    "p2cell(*).distance_mm*": "REPORTED as J and the pooled statistic (the exact arm's is the floor: see ENFORCED); "
                              "shape and finiteness are the population",
    "p2cell(*).locator_offset_mm_max": "DIAGNOSTIC: the soft pin's reach; not a card clause",
    "p2cell(*).calibration_iterations.per_call*": "REPORTED: cap exhaustion per stage and per pass (card: reported)",
    "p2cell(*).calibration_calls.*.locator_offset_mm_max_after": "REPORTED: per-stage diagnostic",
    "p2cell(*).calibration_calls.*.rest_mapped_cm_after*": "REPORTED: the per-stage readings",
    "p2cell(*).start.landmark_start_record*": "REPORTED: the fitter's own record of its start; the start is recomputed",
    "p2cell(*).fixture.drawn_identity*": "LABEL: the draw is REGENERATED from (seed, donor) and compared to the truth",
    "p2cell(*).fixture.truth_motion_sha256": "LABEL: the arms share one fixture dict (compared whole) and one truth",
    "p2cell(*).glb": "ENFORCED via its hash on the candidate arm only",
    "p2cell(*).track": "ENFORCED via its hash on the candidate arm only",
    "p2cell(*-candidate).fitted_identity*": "REPORTED: the recovered per-channel error; L reads the rest",
    "p2cell(*-one_pass).*": "REPORTED: the one-pass (D4c) arm is never banded; its identity is checked against the "
                           "candidate's first pass only",
    "p2cell(*-legacy).*": "REPORTED: the legacy zero-start arm is the before arm, never banded",
    "p2cell(*-init_only).distance_mm*": "REPORTED",
    "p2cell(*).arm_note.*": "LABEL: spine_displaced's construction is read from its identities",
    "p2cell(*).fitted_mapped*": "not a field",
    "closure.p95_abs_m": "DERIVED by d4_glb_closure; not a clause",
    "closure.band_m": "DERIVED: the gate carries D3's band itself",
    "closure.within_band": "DERIVED verdict; re-derived from max_abs_m",
    "closure.glb": "a path; the binding is by content (glb_sha256)",
    "closure.track": "a path; the binding is by content (track_sha256)",
    "b1.*.lag1_autocorrelation": "REPORTED by the bootstrap",
    "b1.*.population_as_named": "DERIVED; the gate re-derives it from n and cells_required",
    "b1.*.cells_required": "DERIVED; the gate re-derives it from the mask exclusions",
    "b1.*D4c_fitted_MHR_lod2_subject_*.median_difference": "REPORTED: combined - D4c is not a band",
    "b1.*D4c_fitted_MHR_lod2_subject_*.ci95_of_the_median_difference*": "REPORTED: combined - D4c is not a band",
    "b1.*baseline_D7c_rig_subject_*.median_difference": "REPORTED beside the band (the band reads the lower CI)",
    "b1.*baseline_D7c_rig_subject_*.ci95_of_the_median_difference.1": "the upper CI bound is reported",
    # Phase 1 with the decision RE-BOUND: what only the report (never the rule or a construction check) reads
    "rebound:p1cell(*).distance_mm*": "REPORTED: the Phase-1 pooled statistic; the fork reads rests",
    "rebound:p1cell(*).locator_offset_mm_max": "DIAGNOSTIC",
    "rebound:p1cell(*).calibration_iterations.per_call*": "REPORTED: cap exhaustion",
    "rebound:p1cell(*).calibration_calls.*.locator_offset_mm_max_after": "REPORTED: per-stage diagnostic",
    "rebound:p1cell(*).calibration_calls.*.rest_mapped_cm_after*": "REPORTED: the per-stage readings",
    "rebound:p1cell(*).start.landmark_start_record*": "REPORTED: the start is recomputed from the landmarks",
    "rebound:p1cell(*).fixture.*": "LABEL: the truth is compared across arms by its identity and rest arrays and the "
                                   "draw regenerated; truth_motion_sha256 is compared across arms (ENFORCED as committed)",
    "rebound:p1cell(*).fitted_identity*": "REPORTED: the recovered error; the fork reads rests (the fitted identity is "
                                          "bound to the last stage-B return)",
    "rebound:p1cell(*-warm).truth_rest*": "REPORTED arm: WARM's truth rest must equal the exact arm's (ENFORCED); a "
                                          "sampled member moved by 1e3 is caught -- see the class",
    "rebound:p1cell(*-sw_star).*": "REPORTED: SW* is a report, never a candidate; its construction is checked",
    "rebound:p1cell(*).arm_note.*": "LABEL",
    "rebound:p1cell(*).arrays": "the two_pass arrays file is read; the others are bound by hash only",
}


# ------------------------------------------------------------------------------------------ plumbing

class Memo:
    """`build` re-derives Phase 1 (180 cells) and the hygiene tripwire (216 files, the debug logs among them) on every
    call. Both are pure functions of their inputs, so a result is reused while the input object is the untouched
    base; any mutated copy is recomputed."""

    def __init__(self, p1: dict, h: dict):
        self.p1_id, self.h_id = id(p1), id(h)
        self._phase1, self._hygiene = gate.phase1, gate.hygiene
        self.phase1_base = self._phase1(p1)
        self.hygiene_base = self._hygiene(h)
        gate.phase1 = self.phase1
        gate.hygiene = self.hygiene

    def phase1(self, inputs):
        return self.phase1_base if id(inputs) == self.p1_id else self._phase1(inputs)   # build never mutates it

    def hygiene(self, h=None):
        return self.hygiene_base if id(h) == self.h_id else self._hygiene(h)


LAST: dict = {}


def run(p1: dict, p2: dict, h: dict) -> str:
    try:
        report = gate.build(p1, p2, h)
    except Exception as error:  # a crash is its OWN class: a hole, never a turn  # noqa: BLE001
        LAST.clear()
        return f"CRASH: {type(error).__name__}: {error}"[:160]
    LAST.clear()
    LAST.update(conjuncts=report["conjuncts"], reason=report["reason"])
    return report["verdict"]


# The conjunct each targeted mutation must turn (so a mutation cannot "turn" through an unrelated failure).
TARGET = {"precondition_0": "precondition_0", "phase1_complete": "phase1_complete", "stage_0b": "stage_0b",
          "WARM": "the_rule_selects_TWO_PASS", "TWO_PASS": "the_rule_selects_TWO_PASS",
          "decision": "phase1_decision_committed_and_rederived", "validity": "validity", "L": "L",
          "closure": "closure", "must_fail_i": "must_fail_i_mean_body_misses_L",
          "must_fail_ii": "must_fail_ii_spine_displaced_fails_L_at_the_trunk",
          "must_fail_iii": "must_fail_iii_exact_identity_reads_zero_and_passes",
          "must_fail_iv": "must_fail_iv_init_only_misses_L", "B1": "B1", "B2": "B2", "hygiene": "hygiene"}
REASON = {"population": "phase 2", "freeze_order": "phase 2", "construction": "phase 2"}


def rebound(p1: dict) -> dict:
    """The decision JSON re-bound to (possibly mutated) Phase-1 cells: isolates the rule from the binding."""
    out = dict(p1)
    data = json.dumps(gate._phase1_raw(out), indent=1, default=float).encode()
    out["decision_bytes"] = out["decision_committed_bytes"] = data
    return out


def p1cell(p1: dict, arm: str, pop: str = P1[0], seed: int = P1[1], donor: int = P1[2]) -> dict:
    if arm in ("warm", "two_pass", "sw_star"):
        return p1["new"][f"cell-{seed}-d{donor}-{arm}.json"]
    return p1["retained"][f"{pop}/{seed}/d{donor}/{arm}"]["record"]


def p2cell(p2: dict, arm: str, seed: int = P2[0], donor: int = P2[1]) -> dict:
    return p2["cells"][f"cell-{seed}-d{donor}-{arm}.json"]


def rebind_retained(q1: dict, key: str) -> None:
    """A CONSISTENT alternative world for a mutated retained D4c cell: D4c's own record (development JSON or
    acceptance manifest), the stage-1 provenance that binds that record, and every new cell's provenance field that
    carries it, all rewritten to the mutated content -- so the binding holds and only the RULE can react."""
    got = q1["retained"][key]
    record = got["record"]
    got["file_sha256"] = gate.sha256_bytes(json.dumps(record).encode())
    got["sorted_dump_sha256"] = gate.sha256_bytes(json.dumps(record, sort_keys=True).encode())
    if got["stage"] == "acceptance":
        manifest = json.loads(q1["d4c_manifest_bytes"])
        manifest["cells_sha256"][got["name"]] = got["file_sha256"]
        q1["d4c_manifest_bytes"] = json.dumps(manifest).encode()
        prefix, field = "D4c acceptance manifest ", None
        data = q1["d4c_manifest_bytes"]
    else:
        development = json.loads(q1["d4c_development_bytes"])
        development["cells_sha256"][got["name"]] = got["sorted_dump_sha256"]
        q1["d4c_development_bytes"] = json.dumps(development).encode()
        prefix, field = "D4c development JSON ", "d4c_development_json_sha256"
        data = q1["d4c_development_bytes"]
    q1["stage1"] = copy.deepcopy(q1["stage1"])
    name = next(k for k in q1["stage1"]["sha256"] if k.startswith(prefix))
    q1["stage1"]["sha256"][name] = gate.sha256_bytes(data)
    if field:
        for cell in q1["new"].values():
            cell["provenance"][field] = gate.sha256_bytes(data)


# ------------------------------------------------------------------------------------ the targeted part

def targeted(p1: dict, p2: dict, h: dict) -> list[dict]:
    rows = []

    def attempt(name: str, conjunct: str, want: str, mutate, target: str = "p2"):
        q1, q2, qh = p1, p2, h
        if target == "p1":
            q1 = copy.deepcopy(p1)
        elif target == "p2":
            q2 = copy.deepcopy(p2)
        elif target == "h":
            qh = dict(h, files=dict(h["files"]), rig=dict(h["rig"]))
        mutate(q1, q2, qh)
        got = run(q1, q2, qh)
        crash = got.startswith("CRASH")
        conjuncts, reason = dict(LAST.get("conjuncts") or {}), LAST.get("reason", "")
        if want == "PASS":
            targeted_ok = True
        elif conjunct in TARGET:
            targeted_ok = conjuncts.get(TARGET[conjunct]) is False
            if conjunct in ("WARM", "TWO_PASS"):
                targeted_ok = targeted_ok and ("WARM" in reason if conjunct == "WARM" else "TWO-PASS does not" in reason)
        else:
            targeted_ok = REASON.get(conjunct, "") in reason
        turns = (got == want) and not crash and targeted_ok
        rows.append({"mutation": name, "conjunct": conjunct, "want": want, "got": got, "reason": reason,
                     "the_targeted_conjunct_fell": targeted_ok,
                     "class": "CRASH" if crash else ("TURNS" if turns else "MISSED")})
        print(f"  {name:78s} want {want:8s} got {got[:40]:40s} {rows[-1]['class']}", flush=True)

    # precondition 0 and Phase 1
    def drawn_byte(q1, q2, qh):
        q1["drawn_bytes"] = q1["drawn_bytes"].replace(b"scale_spine_length", b"scale_spine_lengtX", 1)

    def phase1_cell_missing(q1, q2, qh):
        del q1["new"][f"cell-{P1[1]}-d{P1[2]}-two_pass.json"]

    def phase1_retained_unbound(q1, q2, qh):
        rec = q1["retained"][f"{P1[0]}/{P1[1]}/d{P1[2]}/exact_identity"]
        rec["file_sha256"] = MISMATCH

    def stage0b(q1, q2, qh):
        c = p1cell(q1, "spine_displaced", "d4", 20260922, 0)
        c["fitted_rest_mapped_cm"] = copy.deepcopy(c["truth_rest_mapped_cm"])
        rebind_retained(q1, "d4/20260922/d0/spine_displaced")

    def warm_drifts(q1, q2, qh):
        c = p1cell(q1, "warm")
        c["fitted_rest_mapped_cm"][1][1] += 0.5      # c_neck +5 mm in rest: the trunk beyond tolerance

    def two_pass_misses(q1, q2, qh):
        c = p1cell(q1, "two_pass")
        c["fitted_rest_mapped_cm"][1][1] += 0.5

    def two_pass_pass1_not_d4c(q1, q2, qh):
        c = p1cell(q1, "two_pass")
        k = c["identity_channel_names"].index("scale_uparms")
        for i in (1, 2, 3):
            if i == 1:
                c["calibration_calls"][1]["identity_returned"][k] += 0.01
            else:
                c["calibration_calls"][i]["identity_handed"][k] += 0.01

    def sw_star_start(q1, q2, qh):
        c = p1cell(q1, "sw_star")
        k = c["identity_channel_names"].index("scale_neck_length")
        c["start"]["start_identity"][k] += 0.01
        for i in (0, 1):
            c["calibration_calls"][i]["identity_handed"][k] += 0.01

    def decision_byte(q1, q2, qh):
        q1["decision_bytes"] = q1["decision_bytes"].replace(b'"TWO-PASS"', b'"TWO-PASS" ', 1)

    def decision_uncommitted(q1, q2, qh):
        q1["decision_committed_bytes"] = b""

    attempt("precondition 0: one byte of the drawn set (the spine renamed)", "precondition_0", "STOP", drawn_byte, "p1")
    attempt("phase 1: the fork fixture's two-pass cell deleted", "phase1_complete", "FAIL", phase1_cell_missing, "p1")
    attempt("phase 1: a retained D4c cell no longer bound to D4c's record", "phase1_complete", "FAIL",
            phase1_retained_unbound, "p1")
    attempt("stage 0b: a burned spine control reads the truth's trunk (binding kept)", "stage_0b", "STOP", stage0b, "p1")
    attempt("the fork: WARM's trunk +5 mm on 20261106/d0", "WARM", "STOP", warm_drifts, "p1")
    attempt("the fork: TWO-PASS's trunk +5 mm on 20261106/d0", "TWO_PASS", "STOP", two_pass_misses, "p1")
    attempt("construction: two-pass's first pass is not D4c's one-pass identity", "phase1_complete", "FAIL",
            two_pass_pass1_not_d4c, "p1")
    attempt("construction: SW*'s start moved off the landmark start elsewhere", "phase1_complete", "FAIL",
            sw_star_start, "p1")
    attempt("decision: one byte of the committed decision JSON on disk", "decision", "FAIL", decision_byte, "p1")
    attempt("decision: not committed", "decision", "FAIL", decision_uncommitted, "p1")

    # Phase 2
    def p2_missing(q1, q2, qh):
        del q2["cells"][f"cell-{P2[0]}-d{P2[1]}-candidate.json"]

    def p2_stamp(q1, q2, qh):
        p2cell(q2, "legacy")["phase1_decision_sha256"] = MISMATCH

    def p2_truncate(q1, q2, qh):
        c = p2cell(q2, "candidate")
        c["distance_mm"] = c["distance_mm"][:15]

    def p2_nonfinite(q1, q2, qh):
        p2cell(q2, "candidate")["fitted_rest_mapped_cm"][0][0] = float("nan")

    def p2_order(q1, q2, qh):
        q2["commits"]["decision_time"] = q2["commits"]["manifest_time"] + 1

    def p2_manifest(q1, q2, qh):
        name = next(iter(q2["manifest"]["cells_sha256"]))
        q2["manifest"]["cells_sha256"][name] = MISMATCH

    def p2_manifest_decision(q1, q2, qh):
        q2["manifest"]["phase1_decision_sha256"] = MISMATCH

    def p2_draw(q1, q2, qh):
        for arm in gate.fx4.PHASE2_ARMS:
            c = p2cell(q2, arm)
            k = c["identity_channel_names"].index("scale_uparms")
            c["truth_identity"][k] += 0.01
            if arm in ("exact_identity", "spine_displaced"):
                c["fitted_identity"][k] += 0.01

    def p2_candidate_one_pass(q1, q2, qh):
        c = p2cell(q2, "candidate")
        c["settings"]["passes"] = 1

    def p2_candidate_handoff(q1, q2, qh):
        c = p2cell(q2, "candidate")
        k = c["identity_channel_names"].index("scale_uparms")
        c["calibration_calls"][2]["identity_handed"][k] += 0.01

    def p2_init_start(q1, q2, qh):
        c = p2cell(q2, "init_only")
        k = c["identity_channel_names"].index("scale_spine_length")
        c["fitted_identity"][k] += 0.01
        c["start"]["held_identity"][k] += 0.01

    def validity(q1, q2, qh):
        c = p2cell(q2, "exact_identity")
        c["distance_mm"] = (np.asarray(c["distance_mm"]) * 5.0 + 1.0).tolist()

    def move(arm, mm):
        def mutate(q1, q2, qh):
            p2cell(q2, arm)["fitted_rest_mapped_cm"][1][1] += mm / 10.0
        return mutate

    def truth_rest(arm, everywhere=False):
        def mutate(q1, q2, qh):
            for s, d in (gate.fx4.POPULATIONS["phase2"] if everywhere else [P2]):
                c = p2cell(q2, arm, s, d)
                c["fitted_rest_mapped_cm"] = copy.deepcopy(c["truth_rest_mapped_cm"])
        return mutate

    def exact_off(q1, q2, qh):
        p2cell(q2, "exact_identity")["fitted_rest_mapped_cm"][0][0] += 0.01

    def closure_far(q1, q2, qh):
        q2["closure"]["pairs"][f"{P2[0]}_d{P2[1]}"]["max_abs_m"] = 0.01

    def closure_hash(q1, q2, qh):
        q2["closure"]["pairs"][f"{P2[0]}_d{P2[1]}"]["glb_sha256"] = MISMATCH

    def b1_ci(q1, q2, qh):
        q2["b1-paired.json"]["paired"][f"{gate.B1_CANDIDATE}_minus_{gate.B1_BASELINE}_subject_01"][
            "ci95_of_the_median_difference"][0] = -0.01

    def b1_short(q1, q2, qh):
        q2["b1-paired.json"]["population"]["frames_consumed_per_arm"][gate.B1_CANDIDATE]["subject_00"] = 15

    def b1_frozen(q1, q2, qh):
        q2["silhouette-delivery.json"]["arms"]["control_frozen_pose_tracked"]["A001"]["subject_00"]["iou"]["median"] = 1.0

    def b1_mamma(q1, q2, qh):
        q2["silhouette-delivery.json"]["arms"]["ORACLE_mamma_mesh"]["B001"]["subject_01"]["iou"]["median"] += 1e-9

    def b1_mesh(q1, q2, qh):
        q2["b1_files"]["D4d_mesh"] = MISMATCH

    def b1_glb(q1, q2, qh):
        q2["silhouette-delivery.json"]["input_sha256"]["subject-00.glb"] = MISMATCH

    def b1_passes(q1, q2, qh):
        q2["delivery"]["subjects"]["subject_01"]["passes_json"]["passes"] = 1

    def b1_start(q1, q2, qh):
        ident = q2["delivery"]["subjects"]["subject_00"]["start_json"]["identity"]
        ident["scale_spine_length"] += 0.01

    def b2_marker(q1, q2, qh):
        m = q2["delivery"]["subjects"]["subject_01"]["markers"]
        m["marker_positions_mhr_cm"] = m["marker_positions_mhr_cm"] + 1e-6

    def b2_rig_dtype(q1, q2, qh):
        r = q2["delivery"]["rig_converter_input"]
        r["subject_00_triangulated_world_positions_z_up_m"] = r["subject_00_triangulated_world_positions_z_up_m"].astype(np.float32)

    def b2_rig_track(q1, q2, qh):
        t = q2["delivery"]["subjects"]["subject_00"]["rig_track"]
        a = np.array(t["triangulated_world_positions_z_up_m"])
        a[0, 0, 0] += 1e-9
        t["triangulated_world_positions_z_up_m"] = a

    def b2_declared(q1, q2, qh):
        d = q2["delivery"]["subjects"]["subject_00"]["track_json"]["landmark_to_joint"]
        d["left_ear"] = "l_ear"

    def b2_ticks(q1, q2, qh):
        c = q2["delivery"]["subjects"]["subject_01"]["consumed"]
        c["ticks"] = np.asarray(c["ticks"])[::-1].copy()

    def b2_key(q1, q2, qh):
        q2["delivery"]["subjects"]["subject_00"]["markers"]["source_array_key"] = np.array(
            "raw_triangulated_world_positions_z_up_m")

    attempt("population: the fixture's candidate cell deleted", "population", "FAIL", p2_missing)
    attempt("provenance: a Phase-2 cell not stamped with the committed decision", "population", "FAIL", p2_stamp)
    attempt("population: a candidate cell truncated to 15 frames", "population", "FAIL", p2_truncate)
    attempt("population: a non-finite rest", "population", "FAIL", p2_nonfinite)
    attempt("freeze order: the decision committed after the manifest", "freeze_order", "FAIL", p2_order)
    attempt("freeze order: the manifest names another cell", "freeze_order", "FAIL", p2_manifest)
    attempt("freeze order: the manifest under another decision", "freeze_order", "FAIL", p2_manifest_decision)
    attempt("construction: the uparms draw is not the regenerated one", "construction", "FAIL", p2_draw)
    attempt("construction: the candidate recorded as one pass", "construction", "FAIL", p2_candidate_one_pass)
    attempt("construction: pass 2 not handed pass 1's identity", "construction", "FAIL", p2_candidate_handoff)
    attempt("construction: init-only's held start is not the frozen rule's", "construction", "FAIL", p2_init_start)
    attempt("validity: the exact floor inflated past 1 mm", "validity", "INVALID", validity)
    attempt("L: the candidate's c_neck moved 5 mm (trunk)", "L", "FAIL", move("candidate", 5.0))
    attempt("L: the candidate's r_uparm moved 5 mm (a limb)", "L", "FAIL",
            lambda q1, q2, qh: p2cell(q2, "candidate")["fitted_rest_mapped_cm"][4].__setitem__(
                0, p2cell(q2, "candidate")["fitted_rest_mapped_cm"][4][0] + 0.5))
    attempt("closure: 0.01 m", "closure", "FAIL", closure_far)
    attempt("closure: the measured GLB hash is another file's", "closure", "FAIL", closure_hash)
    attempt("must-fail i: the mean body reads the truth's rest", "must_fail_i", "FAIL", truth_rest("mean_body"))
    attempt("must-fail ii: the displaced spine reads the truth's rest", "must_fail_ii", "FAIL", truth_rest("spine_displaced"))
    attempt("must-fail iii: the exact arm's rest moved 0.1 mm", "must_fail_iii", "FAIL", exact_off)
    attempt("must-fail iv: init-only reads the truth's rest on ONE fixture", "must_fail_iv", "STOP", truth_rest("init_only"))
    attempt("B1: performer 1's lower CI below zero", "B1", "FAIL", b1_ci)
    attempt("B1: the candidate arm consumed 15 frames", "B1", "FAIL", b1_short)
    attempt("B1: the frozen-pose control lifted to 1.0", "B1", "FAIL", b1_frozen)
    attempt("B1: MAMMA's arm no longer bit-identical", "B1", "FAIL", b1_mamma)
    attempt("B1: the scored mesh is not the rebuilt delivery's", "B1", "FAIL", b1_mesh)
    attempt("B1: the silhouette read another GLB", "B1", "FAIL", b1_glb)
    attempt("B1: the delivery ran one pass", "B1", "FAIL", b1_passes)
    attempt("B1: the delivery's start is not the frozen rule's", "B1", "FAIL", b1_start)
    attempt("B2 4a: one marker value moved 1e-6 cm", "B2", "FAIL", b2_marker)
    attempt("B2 1a: the rig converter input cast to float32", "B2", "FAIL", b2_rig_dtype)
    attempt("B2 1b/2b: the rig build's delivered array moved 1e-9 m", "B2", "FAIL", b2_rig_track)
    attempt("B2 4/names: the declared map gains a landmark", "B2", "FAIL", b2_declared)
    attempt("B2 3: frame order reversed", "B2", "FAIL", b2_ticks)
    attempt("B2 1c: the raw array handed", "B2", "FAIL", b2_key)

    # hygiene, and the two negative controls
    def trip_byte(q1, q2, qh):
        key = next(k for k in qh["files"] if k.endswith(".glb"))
        a, b = qh["files"][key]
        qh["files"][key] = (a, b[:-1] + bytes([b[-1] ^ 1]))

    def trip_cell_value(q1, q2, qh):
        key = next(k for k in qh["files"] if k.split("/")[-1].startswith("cell-"))
        a, b = qh["files"][key]
        rec = json.loads(b)
        rec["fitted_identity"][0] += 1e-9
        qh["files"][key] = (a, json.dumps(rec).encode())

    def trip_missing(q1, q2, qh):
        key = next(k for k in qh["files"] if k.startswith("acceptance/"))
        del qh["files"][key]

    def trip_normalised_only(q1, q2, qh):
        key = next(k for k in qh["files"] if k.split("/")[-1].startswith("cell-"))
        a, b = qh["files"][key]
        rec = json.loads(b)
        rec["provenance"]["fitter_sha256"] = MISMATCH
        qh["files"][key] = (a, json.dumps(rec).encode())

    def trip_log_value(q1, q2, qh):
        key = next(k for k in qh["files"] if k.endswith(".calibration-debug.log"))
        a, b = qh["files"][key]
        qh["files"][key] = (a, b.replace(b"Iteration: 1,", b"Iteration: 2,", 1))

    def trip_log_timestamp_only(q1, q2, qh):
        key = next(k for k in qh["files"] if k.endswith(".calibration-debug.log"))
        a, b = qh["files"][key]
        qh["files"][key] = (a, gate.TIMESTAMP.sub(b"[2000-01-01 00:00:00.000]", b))

    def rig(q1, q2, qh):
        qh["rig"]["subject-00.glb"] = [MISMATCH, qh["rig"]["subject-00.glb"][1]]

    def source(q1, q2, qh):
        qh["fitter_source"] = qh["fitter_source"].replace("tracking.max_iter = max_iter", "tracking.max_iter = 300")

    def source_copy(q1, q2, qh):
        qh["fitter_source"] = qh["fitter_source"].replace(
            "mt.calibrate_markers(character, first.copy(), markers, stage_a)",
            "mt.calibrate_markers(character, first, markers, stage_a)")

    attempt("hygiene: one GLB byte of the tripwire", "hygiene", "FAIL", trip_byte, "h")
    attempt("hygiene: a tripwire cell's fitted identity moved 1e-9", "hygiene", "FAIL", trip_cell_value, "h")
    attempt("hygiene: a tripwire file missing", "hygiene", "FAIL", trip_missing, "h")
    attempt("hygiene: a tripwire debug log's iteration index changed", "hygiene", "FAIL", trip_log_value, "h")
    attempt("hygiene: one rig file's sha256 changed", "hygiene", "FAIL", rig, "h")
    attempt("hygiene: fit_one changes beyond the second pass (tracking max_iter)", "hygiene", "FAIL", source, "h")
    attempt("hygiene: pass 2 hands stage A the live identity (no copy)", "hygiene", "FAIL", source_copy, "h")
    attempt("NEGATIVE: a tripwire cell differs ONLY in provenance.fitter_sha256", "hygiene", "PASS",
            trip_normalised_only, "h")
    attempt("NEGATIVE: a tripwire log differs ONLY in its timestamps", "hygiene", "PASS", trip_log_timestamp_only, "h")
    attempt("NEGATIVE: a reported value (combined - D4c median) moved", "reported", "PASS",
            lambda q1, q2, qh: q2["b1-paired.json"]["paired"][f"{gate.B1_CANDIDATE}_minus_{gate.B1_D4C}_subject_00"]
            .__setitem__("median_difference", 0.5))
    return rows


# ------------------------------------------------------------------------------------ the leaf walk

def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list) and node and all(isinstance(x, (int, float)) for x in node) and len(node) > 3:
        for i in sorted({0, len(node) // 2, len(node) - 1}):
            yield path + (i,), node[i]
    elif isinstance(node, list) and node and isinstance(node[0], (list, dict)) and len(node) > 3:
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


def walk(p1: dict, p2: dict, h: dict) -> list[dict]:
    targets = [(f"p2cell({P2[0]}-d{P2[1]}-{a})", "p2", ("cells", f"cell-{P2[0]}-d{P2[1]}-{a}.json"))
               for a in gate.fx4.PHASE2_ARMS]
    targets.append(("closure", "p2", ("closure", "pairs", f"{P2[0]}_d{P2[1]}")))
    targets.append(("b1", "p2", ("b1-paired.json", "paired")))
    targets += [(f"p1cell({P1[1]}-d{P1[2]}-{a})", "p1", ("new", f"cell-{P1[1]}-d{P1[2]}-{a}.json"))
                for a in gate.fx4.PHASE1_NEW_ARMS]
    out = []
    for label, which, root_path in targets:
        base = p2 if which == "p2" else p1
        root = base
        for key in root_path:
            root = root[key]
        modes = ("as_committed", "rebound") if which == "p1" else ("as_committed",)
        for path, value in list(leaves(root)):
            for mode in modes:
                verdicts = []
                for delete in (False, True):
                    mutated = copy.deepcopy(root)
                    try:
                        set_at(mutated, path, None if delete else mismatch(value), delete=delete)
                    except (KeyError, IndexError, TypeError):
                        continue
                    q = dict(base)
                    q[root_path[0]] = dict(base[root_path[0]])
                    node = q[root_path[0]]
                    for key in root_path[1:-1]:
                        node[key] = dict(node[key])
                        node = node[key]
                    node[root_path[-1]] = mutated
                    if which == "p1" and mode == "as_committed":
                        # a faithful model of a changed FILE: the gate hashes the bytes it loads, so the mutated
                        # cell's file hash moves with it (the decision JSON records every Phase-1 file by hash)
                        q["new_files"] = dict(base["new_files"])
                        q["new_files"][root_path[-1]] = gate.sha256_bytes(json.dumps(mutated).encode())
                    if which == "p2":
                        got = run(p1, q, h)
                    else:
                        try:
                            q = rebound(q) if mode == "rebound" else q
                        except Exception as error:  # the gate's own Phase-1 derivation raised  # noqa: BLE001
                            verdicts.append(f"CRASH: {type(error).__name__}: {error}"[:160])
                            continue
                        got = run(q, p2, h)
                    verdicts.append(got)
                crash = any(v.startswith("CRASH") for v in verdicts)
                moved = any(v != "PASS" and not v.startswith("CRASH") for v in verdicts)
                dotted = ("rebound:" if mode == "rebound" else "") + label + "." + ".".join(str(p) for p in path)
                generic = dotted.replace(f"{P2[0]}-d{P2[1]}-", "*-").replace(f"{P1[1]}-d{P1[2]}-", "*-")
                reason = next((why for pattern, why in JUSTIFIED.items()
                               if fnmatch.fnmatch(dotted, pattern) or fnmatch.fnmatch(generic, pattern)), None)
                cls = "CRASH" if crash else ("ENFORCED" if moved else "REPORTED")
                out.append({"leaf": dotted, "class": cls, "verdicts": sorted(set(verdicts)),
                            "justification": reason if cls == "REPORTED" else None})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-walk", action="store_true")
    arguments = parser.parse_args()
    p1, p2, h = gate.load_phase1(), gate.load_phase2(), gate.load_hygiene()
    gate._phase1_raw = gate.phase1
    Memo(p1, h)
    baseline = run(p1, p2, h)
    print("measured baseline:", baseline, flush=True)
    if baseline != "PASS":
        raise SystemExit("the fuzz turns conjuncts from the measured PASS; the baseline is not PASS")
    rebound_baseline = run(rebound(p1), p2, h)
    if rebound_baseline != "PASS":
        raise SystemExit(f"the re-bound decision does not reproduce the PASS ({rebound_baseline})")
    rows = targeted(p1, p2, h)
    walked = [] if arguments.no_walk else walk(p1, p2, h)
    unjustified = [w["leaf"] for w in walked if w["class"] == "REPORTED" and not w["justification"]]
    crashes = [w["leaf"] for w in walked if w["class"] == "CRASH"]
    missed = [r["mutation"] for r in rows if r["class"] != "TURNS"]
    result = {
        "baseline": baseline, "targeted": rows,
        "targeted_turns": sum(r["class"] == "TURNS" for r in rows), "targeted_total": len(rows),
        "targeted_crash": sum(r["class"] == "CRASH" for r in rows), "targeted_missed": missed,
        "leaf_walk": walked, "enforced": sum(w["class"] == "ENFORCED" for w in walked),
        "crash": crashes, "reported_justified": sum(w["class"] == "REPORTED" and bool(w["justification"]) for w in walked),
        "unjustified": unjustified,
        "classes": "TURNS / MISSED / CRASH (targeted); ENFORCED / REPORTED / CRASH (walk). A CRASH is never counted as "
                   "enforcement.",
        "bounded": f"Phase-2 fixture {P2[0]}/d{P2[1]}'s seven cells, its closure row and the B1 paired rows, and Phase-1 "
                   f"fork fixture {P1[1]}/d{P1[2]}'s three new cells (as committed and with the decision re-bound), "
                   "walked leaf by leaf (long lists at first/middle/last); the rest run through the same code path",
    }
    args_out = arguments.out
    args_out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"targeted: {result['targeted_turns']}/{len(rows)} turn, {result['targeted_crash']} crash; walk: "
          f"{result['enforced']} enforced, {len(crashes)} crash, {result['reported_justified']} reported (justified), "
          f"{len(unjustified)} unjustified")
    for leaf in missed[:20]:
        print("  MISSED:", leaf)
    for leaf in crashes[:40]:
        print("  CRASH:", leaf)
    for leaf in unjustified[:60]:
        print("  UNJUSTIFIED:", leaf)
    return 0 if not missed and not crashes and not unjustified else 1


if __name__ == "__main__":
    raise SystemExit(main())
