#!/usr/bin/env python3
"""D4b O1: PROVE the gate by mutating its INPUTS, never its verdicts (the `d7c_gate_fuzz.py` pattern).

Two parts.

1. THE CONJUNCTS, TURNED. For each conjunct of the card's conjunction (validity, L, closure, must-fails
   i-iii) and for the population rule ("a missing, short or non-finite cell is FAIL"), a named mutation of
   the INPUT arrays -- never of a verdict -- must move the gate from its baseline to the verdict the card
   gives that failure: INVALID for validity, FAIL for the rest, and STOP for (ii).

2. THE LEAF WALK. Every leaf of every input the gate reads -- the 72 cells, the closure report, the frozen
   drawn set (mutated as BYTES, so its sha256 moves with it), the stage-1 provenance record and the hashes
   of the delivered GLBs and tracks -- is mutated in turn:

     numbers   set to 1e6, to -1e6, to 0, and DELETED
     strings   set to a mismatched value, and DELETED
     booleans  flipped, and DELETED
     lists     emptied, first member dropped, first member duplicated
     maps      emptied, one member dropped, one member duplicated under a new key

   and each leaf is classified by what it moves: ENFORCED (some mutation turns the verdict away from the
   baseline), REPORTED (it moves a clause's measured or reported values, never the verdict) or UNREAD. For an
   ENFORCED numeric leaf the failing direction is pushed six more orders of magnitude (the leaf's own value
   +/- 1e12) and must still fail: a leaf that fails at 1e6 and passes at 1e12 is matched, not banded.

   BOUNDED, and said so: a numeric list longer than 17 members is walked at sampled members (the first, the
   thirds, the middle, the last, and for the 68-channel identity lists every one of D4's ten named channels)
   plus the list-level mutations, and the per-member walk covers every cell of two fixtures (one per donor)
   and the top-level keys of every other cell (the same code path reads every cell). A REPORTED or UNREAD leaf must match a named
   justification below, or the fuzz exits non-zero.

WHAT A PASS DOES NOT PROVE: that the gate reads the RIGHT things -- only that what it reads, it depends on.
IF A LEAF ESCAPES, FIX THE GATE, NOT THE FUZZER.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate_fuzz.py --out artifacts/compare/d4b-o1/fuzz.json
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
import d4b_o1_gate as gate  # noqa: E402

BIG, FAR = 1.0e6, 1.0e12
MISMATCH = "0" * 64
FULL_FIXTURES = ("20261001-d0", "20261006-d1")
FIXTURE_IDENTITY_SAMPLES = ("scale_spine_length", "scale_shoulder_width")

# REPORTED / UNREAD leaves, each with its reason. A leaf outside these patterns fails the fuzz.
JUSTIFIED = {
    # reported by the card, never banded
    "cell(*-converged).calibration_iterations*": "REPORTED: the CONVERGED probe's iteration record (card: reported)",
    "cell(*).distance_mm*": "REPORTED as J and the pooled statistic; its shape and finiteness are the population",
    "cell(*-oracle).fitted_identity*": "REPORTED: the per-channel parameter error and undrawn drift (card: reported)",
    "cell(*-warm).fitted_identity*": "REPORTED: the WARM probe",
    "cell(*-converged).fitted_identity*": "REPORTED: the CONVERGED probe",
    "cell(*-warm).fitted_rest_mapped_cm*": "REPORTED: the WARM probe's segments",
    "cell(*-converged).fitted_rest_mapped_cm*": "REPORTED: the CONVERGED probe's segments",
    # labels and diagnostics
    "cell(*).locator_offset_mm_max": "DIAGNOSTIC: the soft pin's reach (D4 measured 0.0009 mm); not a card clause",
    "cell(*).fixture.generator": "ENFORCED by identity; a mutated string that still reads the same is impossible",
    "closure.pairs.*.p95_abs_m": "DERIVED by d4_glb_closure; the gate re-derives the band from max_abs_m",
    "closure.pairs.*.band_m": "DERIVED: the gate carries D3's band itself",
    "closure.pairs.*.within_band": "DERIVED verdict; the gate re-derives it from max_abs_m (never a literal)",
    "closure.all_within_band": "DERIVED verdict; re-derived",
    "closure.worst_max_abs_m": "DERIVED aggregate; re-derived per pair",
    "closure.band_m": "DERIVED: the gate carries D3's band itself",
    "provenance.environment.*": "LABEL: the environment record; the pymomentum version is read by name",
    "provenance.environment_reproduces_D4*": "PROVENANCE NOTE: the stage-1 reproduction, not a gate input",
    "provenance.fitter_sha256_*": "LABEL: the card's own sha is a gate constant",
    "provenance.step": "LABEL", "provenance.stage": "LABEL", "provenance.date": "LABEL",
    "provenance.base_commit": "LABEL",
    "provenance.sha256.joint layout*": "PROVENANCE: the joint layout file is hashed at stage 1; the fixture reads "
                                       "its names only and the mapped joints are checked by identity",
    "files.*": "the GLB/track hashes of the files on disk; ENFORCED through closure identity",
    "closure.pairs.*.joints_missing_from_the_glb": "READ (must be []); an empty list admits no walk mutation but "
                                                   "emptying -- the targeted case 'closure: a joint missing' turns it",
    "cell(*-converged).calibration_iterations.note": "LABEL",
    "cell(*).arm_note": "READ (must be {} on every arm but spine_displaced); an empty map admits no walk mutation "
                        "but emptying -- the targeted case 'arm_note on the oracle' turns it",
}


def walk(node, path=()):
    """Yield (path, value) for every leaf and every container, with bounded sampling of long numeric lists."""
    yield path, node
    if isinstance(node, dict):
        for key in list(node):
            yield from walk(node[key], path + (key,))
    elif isinstance(node, list):
        indices = range(len(node))
        if len(node) > 17:
            picks = {0, len(node) // 3, len(node) // 2, 2 * len(node) // 3, len(node) - 1}
            if len(node) == gate.IDENTITY_CHANNELS and path and path[-1] in ("truth_identity", "fitted_identity"):
                picks |= set(range(10))   # the names are checked by identity; sample the head densely
            indices = sorted(picks)
        for index in indices:
            yield from walk(node[index], path + (index,))


def kind_of(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "map"
    return "other"


def mutations_for(value):
    kind = kind_of(value)
    if kind == "number":
        return [("=1e6", BIG), ("=-1e6", -BIG), ("=0", 0), ("delete", None)]
    if kind == "string":
        return [("mismatch", MISMATCH if value != MISMATCH else "1" * 64), ("delete", None)]
    if kind == "bool":
        return [("flip", not value), ("delete", None)]
    if kind == "list":
        out = [("empty", [])]
        if value:
            out += [("drop-first", value[1:]), ("dup-first", [value[0]] + value)]
        return out
    if kind == "map":
        out = [("empty", {})]
        if value:
            key = next(iter(value))
            dropped = dict(value)
            dropped.pop(key)
            dup = dict(value)
            dup[f"{key}__dup"] = value[key]
            out += [("drop-one", dropped), ("dup-one", dup)]
        return out
    return []


def set_at(root, path, value, delete=False):
    node = root
    for key in path[:-1]:
        node = node[key]
    if delete:
        if isinstance(node, dict):
            node.pop(path[-1])
        else:
            node.pop(path[-1])
    else:
        node[path[-1]] = value


def get_at(root, path):
    node = root
    for key in path:
        node = node[key]
    return node


def mutated_inputs(inputs: dict, target: str, path: tuple, value, delete: bool) -> dict:
    """A shallow copy of `inputs` with ONE target deep-copied and mutated at `path`."""
    new = dict(inputs)
    if target.startswith("cell:"):
        name = target[5:]
        cells = dict(inputs["cells"])
        record = copy.deepcopy(cells[name])
        if path:
            set_at(record, path, value, delete)
            cells[name] = record
        elif delete:
            cells.pop(name)
        else:
            cells[name] = value
        new["cells"] = cells
    elif target == "drawn_set":
        document = json.loads(inputs["drawn_set_bytes"])
        if path:
            set_at(document, path, value, delete)
        else:
            document = {} if delete else value
        new["drawn_set_bytes"] = json.dumps(document, indent=1).encode()
    elif target in ("closure", "provenance", "files"):
        document = copy.deepcopy(inputs[target])
        if path:
            set_at(document, path, value, delete)
        else:
            document = None if delete else value
        new[target] = document
    return new


def signature(report: dict) -> tuple:
    """Everything a mutation may move that the card bands: the verdict, STOP, the population and validity
    preconditions, and each conjunct. Under the registered reading B the real baseline is FAIL with STOP, so the
    verdict alone could not show a conjunct turning; each conjunct is compared on its own."""
    return (report["verdict"], report["STOP"], report["population_ok"], report["validity"],
            tuple(sorted(report["conjuncts"].items())))


def generic(target: str, path: tuple) -> str:
    if target.startswith("cell:"):
        name = target[5:].removeprefix("cell-").removesuffix(".json")
        head = f"cell({name})"
    else:
        head = target
    parts = []
    for key in path:
        parts.append("*" if isinstance(key, int) else str(key))
    return head + ("." + ".".join(parts) if parts else "")


def justified(pattern_path: str) -> str | None:
    for pattern, reason in JUSTIFIED.items():
        if fnmatch.fnmatch(pattern_path, pattern):
            return reason
    return None


_STATE: dict = {}


def _init(cells: str, closure: str) -> None:
    inputs = gate.load_inputs(Path(cells), Path(closure))
    report = gate.build(inputs)
    _STATE.update(inputs=inputs, baseline=signature(report),
                  clauses=json.dumps([report["clauses"], report["reported"]], sort_keys=True))


def walk_target(job: tuple) -> tuple[int, dict]:
    """Every mutation of every leaf of ONE input target (run in a worker holding the loaded inputs)."""
    target, full = job
    inputs, baseline, base_clauses = _STATE["inputs"], _STATE["baseline"], _STATE["clauses"]
    root = (inputs["cells"][target[5:]] if target.startswith("cell:") else
            json.loads(inputs["drawn_set_bytes"]) if target == "drawn_set" else inputs[target])
    rows, visited = {}, 0
    for path, value in list(walk(root)):
        if not path:
            continue
        if not full and len(path) > 1:
            continue          # top-level keys only, outside the two fully walked fixtures
        visited += 1
        key = generic(target, path)
        row = rows.setdefault(key, {"enforced": False, "reported": False, "monotone_failures": [], "leaves": 0})
        row["leaves"] += 1
        for label, new_value in mutations_for(value):
            report = gate.build(mutated_inputs(inputs, target, path, new_value, label == "delete"))
            if signature(report) != baseline:
                row["enforced"] = True
                if kind_of(value) == "number" and label in ("=1e6", "=-1e6"):
                    further = value + (FAR if label == "=1e6" else -FAR)
                    again = gate.build(mutated_inputs(inputs, target, path, further, False))
                    if signature(again) == baseline:
                        row["monotone_failures"].append(f"{label} fails, {further:g} does not")
            elif json.dumps([report["clauses"], report["reported"]], sort_keys=True) != base_clauses:
                row["reported"] = True
    return visited, rows


def leaf_walk(cells: Path, closure: Path, workers: int) -> dict:
    import multiprocessing

    inputs = gate.load_inputs(cells, closure)
    jobs = [(f"cell:{name}", any(f"-{fx}-" in name.replace("cell-", "-") for fx in FULL_FIXTURES))
            for name in sorted(inputs["cells"])]
    jobs += [("closure", True), ("drawn_set", True), ("provenance", True), ("files", True)]
    rows, visited = {}, 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init,
                                                   initargs=(str(cells), str(closure))) as pool:
        for count, part in pool.imap_unordered(walk_target, jobs):
            visited += count
            for key, row in part.items():
                merged = rows.setdefault(key, {"enforced": False, "reported": False, "monotone_failures": [],
                                               "leaves": 0})
                merged["enforced"] |= row["enforced"]
                merged["reported"] |= row["reported"]
                merged["monotone_failures"] += row["monotone_failures"]
                merged["leaves"] += row["leaves"]
    return {"visited": visited, "rows": rows}


def targeted(inputs: dict, burned: bool) -> list[dict]:
    """Each conjunct turned by a named mutation of its INPUTS."""
    first = "cell-20261001-d0-{}.json"
    joints = list(gate.MAPPED_JOINTS)
    cases = []

    base = gate.build(inputs, burned=burned)

    def state(report, conjunct):
        if conjunct is None:
            return report["population_ok"]
        if conjunct == "validity":
            return report["validity"]
        return report["conjuncts"][conjunct]

    def case(name, conjunct, want, mutate):
        """`turned` iff the named conjunct (None: the population precondition) goes from True at the baseline
        to False, the verdict is the one the card gives, and -- for (ii) -- STOP is raised."""
        new = copy.deepcopy({k: v for k, v in inputs.items() if k != "cells"})
        new["cells"] = {k: copy.deepcopy(v) for k, v in inputs["cells"].items()}
        mutate(new)
        report = gate.build(new, burned=burned)
        before, after = state(base, conjunct), state(report, conjunct)
        ok = (before is True and after is False and report["verdict"] == want[0]
              and (want[1] is None or report["STOP"] == want[1]))
        row = {"mutation": name, "conjunct": conjunct, "expected": list(want),
               "got": [report["verdict"], report["STOP"]], "baseline_state": before, "turned": bool(ok)}
        if before is False:
            # Nothing to turn: the conjunct is already FALSE on this population (on the real fresh cells, (ii)
            # under reading B with the trunk unscored). It is turned on the synthetic scored-trunk population
            # in tests/test_d4b_o1.py; here the fuzz records it and requires only that it stays FALSE.
            row["already_false_at_baseline"] = True
            row["turned"] = after is False and report["verdict"] == "FAIL"
        cases.append(row)

    def scale_exact(new):
        cell = new["cells"][first.format("exact_identity")]
        cell["distance_mm"] = (np.asarray(cell["distance_mm"]) * 2.0).tolist()

    def move_wrist(new):
        cell = new["cells"][first.format("oracle")]
        rest = np.asarray(cell["fitted_rest_mapped_cm"])
        rest[joints.index("l_wrist"), 0] += 0.5     # 5 mm on a forearm
        cell["fitted_rest_mapped_cm"] = rest.tolist()

    def closure_out(new):
        new["closure"]["pairs"]["20261001_d0"]["max_abs_m"] = 2e-4

    def mean_is_truth(new):
        for seed_donor in ("20261003-d1",):
            cell = new["cells"][f"cell-{seed_donor}-mean_body.json"]
            cell["fitted_rest_mapped_cm"] = copy.deepcopy(cell["truth_rest_mapped_cm"])

    def spine_is_truth(new):
        cell = new["cells"][first.format("spine_displaced")]
        cell["fitted_rest_mapped_cm"] = copy.deepcopy(cell["truth_rest_mapped_cm"])

    def exact_nonzero(new):
        cell = new["cells"][first.format("exact_identity")]
        rest = np.asarray(cell["fitted_rest_mapped_cm"])
        rest[joints.index("r_foot"), 1] += 1e-6
        cell["fitted_rest_mapped_cm"] = rest.tolist()

    def drop_cell(new):
        new["cells"].pop(first.format("warm"))

    def short_frames(new):
        cell = new["cells"][first.format("oracle")]
        cell["distance_mm"] = cell["distance_mm"][:149]

    def nan_frame(new):
        new["cells"][first.format("converged")]["distance_mm"][70][3] = float("nan")

    def stray_cell(new):
        new["cells"]["cell-20261007-d0-oracle.json"] = copy.deepcopy(new["cells"][first.format("oracle")])

    def fitter_sha(new):
        new["cells"][first.format("mean_body")]["provenance"]["fitter_sha256"] = MISMATCH

    def shared_draw(new):
        source = new["cells"]["cell-20261001-d0-oracle.json"]
        for arm in gate.ARMS:
            target = new["cells"][f"cell-20261001-d1-{arm}.json"]
            target["truth_identity"] = copy.deepcopy(source["truth_identity"])

    def drawn_file(new):
        document = json.loads(new["drawn_set_bytes"])
        document["drawn_set"] = document["drawn_set"][:-1]
        new["drawn_set_bytes"] = json.dumps(document, indent=1).encode()

    def max_iter_banded(new):
        new["cells"][first.format("oracle")]["settings"]["max_iter"] = 300

    def drop_closure(new):
        new["closure"] = None

    def stray_note(new):
        new["cells"][first.format("oracle")]["arm_note"] = {"spine_truth": 0.0}

    def joint_missing(new):
        new["closure"]["pairs"]["20261004_d1"]["joints_missing_from_the_glb"] = ["c_neck"]

    if burned:
        return cases
    case("exact_identity distance x2 on one fixture", "validity", ("INVALID", None), scale_exact)
    case("oracle l_wrist rest moved 5 mm on one fixture", "L", ("FAIL", None), move_wrist)
    case("closure max_abs_m 2e-4 on one oracle cell", "closure", ("FAIL", None), closure_out)
    case("mean_body rest set to the truth rest on one fixture", "must_fail_i_mean_body_misses_L",
         ("FAIL", None), mean_is_truth)
    case("spine_displaced rest set to the truth rest on one fixture",
         "must_fail_ii_spine_displaced_fails_at_the_trunk", ("FAIL", True), spine_is_truth)
    case("exact_identity rest moved 1e-6 cm on one fixture", "must_fail_iii_exact_identity_reads_zero_and_passes",
         ("FAIL", None), exact_nonzero)
    case("population: one cell missing", None, ("FAIL", None), drop_cell)
    case("population: one cell 149 frames", None, ("FAIL", None), short_frames)
    case("population: one non-finite frame", None, ("FAIL", None), nan_frame)
    case("population: a stray cell outside the named set", None, ("FAIL", None), stray_cell)
    case("provenance: a fitter sha256 that is not 3136befb", None, ("FAIL", None), fitter_sha)
    case("fixtures: two fixtures sharing one draw", None, ("FAIL", None), shared_draw)
    case("drawn set: one channel removed from the frozen file", None, ("FAIL", None), drawn_file)
    case("settings: max_iter 300 in a banded arm", None, ("FAIL", None), max_iter_banded)
    case("closure report missing", "closure", ("FAIL", None), drop_closure)
    case("closure: a joint missing from one GLB", "closure", ("FAIL", None), joint_missing)
    case("arm_note on the oracle", None, ("FAIL", None), stray_note)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, default=ROOT / "artifacts/compare/d4b-o1/fresh")
    parser.add_argument("--closure", type=Path, default=ROOT / "artifacts/compare/d4b-o1/closure.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targeted-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    inputs = gate.load_inputs(arguments.cells, arguments.closure)
    baseline_report = gate.build(inputs)
    baseline = signature(baseline_report)
    print("baseline:", baseline, flush=True)
    cases = targeted(inputs, burned=False)
    for c in cases:
        mark = " (already FALSE at the baseline: reading B, the trunk unscored)" if c.get("already_false_at_baseline") else ""
        print(f"  {'TURNED ' if c['turned'] else 'ESCAPED'}  {c['mutation']:60s} -> {c['got']}{mark}", flush=True)
    result = {"baseline": list(baseline), "targeted": cases,
              "all_targeted_turned": all(c["turned"] for c in cases)}
    if not arguments.targeted_only:
        walked = leaf_walk(arguments.cells, arguments.closure, arguments.workers)
        rows = walked["rows"]
        enforced = sorted(k for k, r in rows.items() if r["enforced"])
        reported = sorted(k for k, r in rows.items() if not r["enforced"] and r["reported"])
        unread = sorted(k for k, r in rows.items() if not r["enforced"] and not r["reported"])
        gaps = [k for k in reported + unread if justified(k) is None]
        monotone = {k: r["monotone_failures"] for k, r in rows.items() if r["monotone_failures"]}
        result.update({
            "leaves_visited": walked["visited"], "leaf_classes": len(rows),
            "enforced": len(enforced), "reported": {k: justified(k) for k in reported},
            "unread": {k: justified(k) for k in unread}, "gaps": gaps, "monotone_failures": monotone,
        })
        print(f"leaves visited {walked['visited']} in {len(rows)} classes: {len(enforced)} ENFORCED, "
              f"{len(reported)} REPORTED, {len(unread)} UNREAD; gaps {len(gaps)}; monotone failures {len(monotone)}")
        for k in gaps:
            print("  GAP:", k, "REPORTED" if k in reported else "UNREAD")
        for k, v in monotone.items():
            print("  MONOTONE:", k, v[:2])
        result["clean"] = result["all_targeted_turned"] and not gaps and not monotone
    else:
        result["clean"] = result["all_targeted_turned"]
    arguments.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("FUZZ CLEAN:", result["clean"])
    return 0 if result["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
