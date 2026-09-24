#!/usr/bin/env python3
"""D4b O1, re-registered prospectively: the wired conjunction.

The card is the "D4b O1 re-registered, prospectively" row of `docs/LADDER_EXECUTION_PLAN.md` §2. This gate reads
ONLY arrays the fixture wrote (`tools/fitter/d4b_o1_fixture.py`, one JSON per cell), the frozen drawn set, the
stage-1 provenance record and the closure report, and derives every clause from them. No verdict is a literal:

    verdict = FAIL     if the population, the provenance or an arm's construction is short, missing,
                       non-finite or mismatched (the card: "A missing, short or non-finite cell is FAIL")
            = INVALID  else if the exact_identity arm's pooled statistic exceeds 1.0 mm on any fixture
            = PASS     else iff  L  AND closure  AND must-fail (i)  AND (ii)  AND (iii)
            = FAIL     otherwise

THE BAND L: on each fixture, the rest length of each segment between mapped joints (forward kinematics at
MHR's zero pose with the identity RETAINED), |fitted - truth| in mm, against the sum of its two endpoints'
PAIRED floors -- the per-joint median over frames of the SAME fixture's exact_identity arm. A segment is SCORED
iff a channel of the frozen drawn set moves it (the drawn-set record's `segment_moved_by`, re-derived here);
the rest are reported, not scored. L PASSES iff every scored segment on every fixture is within tolerance.

  (i)   the mean body MISSES L on every fixture: some scored segment beyond its tolerance.
  (ii)  the displaced spine FAILS L AT THE TRUNK on every fixture: the trunk's |error| beyond the trunk's
        tolerance, read WHETHER OR NOT the trunk is scored (the card: exclusion never skips (ii)). If it
        passes on any fixture the band is blind to the defect it exists to see and the step STOPS.
  (iii) the exact_identity arm reads L = 0 on every segment and the gate PASSES it.

`d4_disposition` is the card's own consequence, printed beside the verdict and never folded into it: D4
closes only on PASS with the trunk SCORED ("a trunk that cannot be scored and rejected is not a demonstrated
trunk failure, and such a run cannot close D4").

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate.py \
        --cells artifacts/compare/d4b-o1/fresh --closure artifacts/compare/d4b-o1/closure.json \
        --out artifacts/compare/d4b-o1/gate.json
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4b_o1_gate.py --burned \
        --cells artifacts/compare/d4b-o1/burned --out artifacts/compare/d4b-o1/burned.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fitter"))
from d4b_identifiability import (ARMS, CHANNELS, DONOR_INDICES, FRAMES, MAPPED_JOINTS,  # noqa: E402
                                 SEEDS, SEGMENTS, SPINE_DISPLACEMENT, displaced_spine)

DRAWN_SET = ROOT / "docs/reviews/body-model-o1-records/drawn-set.json"
PROVENANCE = ROOT / "docs/reviews/body-model-o1-records/provenance.json"
SCHEMA = "d4b-o1-cell/1"
CARD_FITTER_SHA256 = "3136befb6cd0415fe746a208f73e15be0f1c8af1ad3594cad277c7f9e5b13535"
D4_SEEDS = (20260922, 20260923, 20260924, 20260925, 20260926, 20260927)
VALIDITY_CEILING_MM = 1.0
CLOSURE_BAND_M = 1e-4
EXACT_ZERO_MM = 1e-9
IDENTITY_CHANNELS = 68
BANDED_ARMS = ("oracle", "exact_identity", "mean_body", "spine_displaced")
SETTINGS = {"calib_frames": 100, "loss_alpha": 2.0, "smoothing": 0.0, "locator_limit_weight": 10.0,
            "freeze_flexible": True}
CALIBRATE_CALLS = {"oracle": 2, "warm": 2, "converged": 2, "exact_identity": 0, "mean_body": 0,
                   "spine_displaced": 0}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------- arithmetic

def finite_array(value, shape) -> np.ndarray | None:
    """`value` as a float array of exactly `shape`, all finite; None otherwise."""
    try:
        array = np.asarray(value, np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        return None
    return array


def pooled_mm(distance: np.ndarray) -> float:
    """D4's pooled statistic: the median over frames of the median over the 17 joints."""
    return float(np.median(np.median(distance, axis=1)))


def floors_mm(distance: np.ndarray) -> np.ndarray:
    """The paired floor: the per-joint median over frames of the exact_identity arm."""
    return np.median(distance, axis=0)


def segment_lengths_mm(rest_cm: np.ndarray) -> dict[str, float]:
    at = {joint: rest_cm[i] for i, joint in enumerate(MAPPED_JOINTS)}
    return {s: float(np.linalg.norm(at[a] - at[b]) * 10.0) for s, (a, b) in SEGMENTS.items()}


def segment_rows(truth_rest: np.ndarray, fitted_rest: np.ndarray, floor: np.ndarray,
                 scored: list[str]) -> dict[str, dict]:
    truth, fitted = segment_lengths_mm(truth_rest), segment_lengths_mm(fitted_rest)
    index = {joint: i for i, joint in enumerate(MAPPED_JOINTS)}
    rows = {}
    for segment, (a, b) in SEGMENTS.items():
        error = abs(fitted[segment] - truth[segment])
        tolerance = float(floor[index[a]] + floor[index[b]])
        rows[segment] = {"truth_mm": round(truth[segment], 4), "fitted_mm": round(fitted[segment], 4),
                         "error_mm": round(error, 6), "tolerance_mm": round(tolerance, 6),
                         "within": bool(error <= tolerance), "scored": segment in scored}
    return rows


def l_passes(rows: dict[str, dict]) -> bool:
    """L on one fixture: every SCORED segment within its tolerance (derived from error and tolerance)."""
    return all(r["error_mm"] <= r["tolerance_mm"] for r in rows.values() if r["scored"])


def misses_l(rows: dict[str, dict]) -> bool:
    """Must-fail (i)'s predicate, the negation of L's: some scored segment beyond its tolerance."""
    return any(r["error_mm"] > r["tolerance_mm"] for r in rows.values() if r["scored"])


# ------------------------------------------------------------------------------------------- the inputs

def load_inputs(cells: Path, closure: Path | None, drawn_set: Path = DRAWN_SET,
                provenance: Path = PROVENANCE) -> dict:
    inputs = {"cells": {}, "cell_dir": str(cells), "closure": None, "files": {}}
    for path in sorted(Path(cells).glob("cell-*.json")):
        inputs["cells"][path.name] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(Path(cells).glob("fixture-*.glb")):
        inputs["files"][path.name] = sha256_bytes(path.read_bytes())
    drawn_bytes = Path(drawn_set).read_bytes()
    inputs["drawn_set"] = json.loads(drawn_bytes)
    inputs["drawn_set_sha256"] = sha256_bytes(drawn_bytes)
    inputs["provenance"] = json.loads(Path(provenance).read_text(encoding="utf-8"))
    if closure is not None and Path(closure).exists():
        inputs["closure"] = json.loads(Path(closure).read_text(encoding="utf-8"))
    return inputs


def expected_provenance(inputs: dict) -> dict:
    stage1 = inputs["provenance"]["sha256"]

    def pick(prefix: str) -> str | None:
        hits = [v for k, v in stage1.items() if k.startswith(prefix)]
        return hits[0] if len(hits) == 1 else None

    return {"fitter_sha256": CARD_FITTER_SHA256,
            "fitter_sha256_stage1": pick("fitter "),
            "donor_sha256": {"0": pick("donor 0 "), "1": pick("donor 1 ")},
            "model_sha256": pick("model "), "fbx_sha256": pick("fbx "),
            "drawn_set_sha256": inputs["drawn_set_sha256"],
            "pymomentum": inputs["provenance"].get("environment", {}).get("pymomentum-cpu")}


def scored_segments(drawn: dict) -> tuple[list[str], list[str], list[str]]:
    """Re-derive the scored segments from the frozen record's own constituents and cross-check them."""
    problems = []
    drawn_set = list(drawn.get("drawn_set") or [])
    moved_by = drawn.get("segment_moved_by") or {}
    if set(moved_by) != set(SEGMENTS):
        problems.append("drawn set: segment_moved_by does not name the card's twelve segments")
    if dict((k, tuple(v)) for k, v in (drawn.get("segments") or {}).items()) != SEGMENTS:
        problems.append("drawn set: its segments are not the card's")
    if not set(drawn_set) <= set(CHANNELS) or len(set(drawn_set)) != len(drawn_set) or not drawn_set:
        problems.append("drawn set: not a non-empty subset of D4's ten channels")
    if drawn.get("frozen_before_any_fit") is not True:
        problems.append("drawn set: not marked frozen before any fit")
    scored = [s for s in SEGMENTS if any(c in drawn_set for c in moved_by.get(s, []))]
    if scored != list(drawn.get("segments_scored") or []):
        problems.append("drawn set: its recorded scored segments are not re-derivable")
    # the rule itself, re-applied to the recorded medians
    rederived = [c for c in CHANNELS if all(
        (drawn.get("per_donor", {}).get(str(d), {}).get("pose_orthogonal_median_mm", {}).get(c, -1.0) or -1.0)
        > drawn.get("frozen", {}).get("threshold_mm", math.inf) for d in DONOR_INDICES)]
    if rederived != drawn_set:
        problems.append("drawn set: the rule re-applied to its own medians does not give its drawn set")
    return drawn_set, scored, problems


# --------------------------------------------------------------------------------------------- the build

def build(inputs: dict, *, burned: bool = False) -> dict:
    problems: list[str] = []
    drawn_set, scored, drawn_problems = scored_segments(inputs["drawn_set"])
    problems += drawn_problems
    expected = expected_provenance(inputs)
    if expected["fitter_sha256_stage1"] != CARD_FITTER_SHA256:
        problems.append("provenance: the stage-1 fitter sha256 is not the card's 3136befb")
    limits = inputs["drawn_set"].get("configured_limits") or {}

    seeds = D4_SEEDS if burned else SEEDS
    donors = (0,) if burned else DONOR_INDICES
    fixtures = [(s, d) for d in donors for s in seeds]
    arms = ARMS
    expected_names = {f"cell-{s}-d{d}-{a}.json" for s, d in fixtures for a in arms}
    present = set(inputs["cells"])
    for name in sorted(expected_names - present):
        problems.append(f"population: {name} missing")
    for name in sorted(present - expected_names):
        problems.append(f"population: {name} is not in the named population")

    canonical_names = None
    cells: dict[tuple, dict] = {}
    for s, d in fixtures:
        for a in arms:
            name = f"cell-{s}-d{d}-{a}.json"
            record = inputs["cells"].get(name)
            if record is None:
                continue
            tag = f"{s}/d{d}/{a}"
            bad = []
            if record.get("schema") != SCHEMA:
                bad.append("schema")
            if (record.get("seed"), record.get("donor"), record.get("arm")) != (s, d, a):
                bad.append("seed/donor/arm do not match the file")
            if bool(record.get("burned")) != burned:
                bad.append("burned flag")
            if record.get("frames") != FRAMES:
                bad.append(f"frames {record.get('frames')} != {FRAMES}")
            if list(record.get("mapped_joints") or []) != list(MAPPED_JOINTS):
                bad.append("mapped joints")
            distance = finite_array(record.get("distance_mm"), (FRAMES, len(MAPPED_JOINTS)))
            truth_rest = finite_array(record.get("truth_rest_mapped_cm"), (len(MAPPED_JOINTS), 3))
            fitted_rest = finite_array(record.get("fitted_rest_mapped_cm"), (len(MAPPED_JOINTS), 3))
            truth_id = finite_array(record.get("truth_identity"), (IDENTITY_CHANNELS,))
            fitted_id = finite_array(record.get("fitted_identity"), (IDENTITY_CHANNELS,))
            names = list(record.get("identity_channel_names") or [])
            for label, value in (("distance_mm", distance), ("truth_rest", truth_rest),
                                 ("fitted_rest", fitted_rest), ("truth_identity", truth_id),
                                 ("fitted_identity", fitted_id)):
                if value is None:
                    bad.append(f"{label} short, missing or non-finite")
            if (len(names) != IDENTITY_CHANNELS or len(set(names)) != IDENTITY_CHANNELS
                    or not all(str(n).startswith("scale_") for n in names) or not set(CHANNELS) <= set(names)):
                bad.append("identity channel names")
            elif canonical_names is None:
                canonical_names = names
            elif names != canonical_names:
                bad.append("identity channel names differ between cells")
            prov = record.get("provenance") or {}
            for key in ("fitter_sha256", "model_sha256", "fbx_sha256", "drawn_set_sha256", "pymomentum"):
                if prov.get(key) != expected[key] or expected[key] is None:
                    bad.append(f"provenance {key}")
            if (prov.get("donor_sha256") or {}).get(str(d)) != expected["donor_sha256"][str(d)] \
                    or expected["donor_sha256"][str(d)] is None:
                bad.append("provenance donor_sha256")
            settings = record.get("settings") or {}
            for key, value in SETTINGS.items():
                if settings.get(key) != value:
                    bad.append(f"settings {key}")
            want_iter = 300 if a == "converged" else 30
            if "retained_from" not in record and settings.get("max_iter") != want_iter:
                bad.append(f"settings max_iter {settings.get('max_iter')} != {want_iter}")
            if settings.get("warm_start") is not (a == "warm"):
                bad.append("settings warm_start")
            if settings.get("calibration_debug_captured") is not (a == "converged"):
                bad.append("settings calibration_debug_captured")
            if "retained_from" not in record and record.get("calibrate_markers_calls") != CALIBRATE_CALLS[a]:
                bad.append("calibrate_markers calls")
            if "retained_from" in record and not burned:
                bad.append("a retained D4 cell in the fresh population")
            if a == "converged" and not (record.get("calibration_iterations") or {}).get("max_iteration_index_reached"):
                bad.append("converged: no iteration record")
            if bad:
                problems.append(f"cell {tag}: " + "; ".join(bad))
                continue
            cells[(s, d, a)] = {"record": record, "distance": distance, "truth_rest": truth_rest,
                                "fitted_rest": fitted_rest, "truth_id": truth_id, "fitted_id": fitted_id,
                                "names": names}

    # fixture identity: one truth per fixture, drawn as declared, twelve distinct draws
    truths = {}
    for s, d in fixtures:
        group = [cells.get((s, d, a)) for a in arms]
        if any(c is None for c in group):
            continue
        first = group[0]
        fixture = first["record"].get("fixture") or {}
        for c in group[1:]:
            if (c["record"].get("fixture") or {}).get("truth_motion_sha256") != fixture.get("truth_motion_sha256") \
                    or not np.array_equal(c["truth_id"], first["truth_id"]) \
                    or not np.array_equal(c["truth_rest"], first["truth_rest"]):
                problems.append(f"fixture {s}/d{d}: the arms do not share one truth")
                break
        want_generator = (f"numpy default_rng({s}) (D4's own)" if burned
                          else f"numpy default_rng([{s}, {d}]) (seeded by the pair)")
        if fixture.get("generator") != want_generator:
            problems.append(f"fixture {s}/d{d}: generator is not {want_generator!r}")
        want_channels = list(CHANNELS) if burned else drawn_set
        if list(fixture.get("drawn_channels") or []) != want_channels:
            problems.append(f"fixture {s}/d{d}: drawn channels are not the declared set")
        names = first["names"]
        for i, n in enumerate(names):
            value = float(first["truth_id"][i])
            if n in want_channels:
                low, high = (limits.get(n) or [math.nan, math.nan])
                if not (low <= value <= high) or (low != high and value == 0.0):
                    problems.append(f"fixture {s}/d{d}: {n} = {value} not a draw within [{low}, {high}]")
            elif value != 0.0:
                problems.append(f"fixture {s}/d{d}: undrawn {n} = {value} is not zero")
        truths[(s, d)] = first["truth_id"]
        # arm construction, read from the arms' own identities
        spine = names.index("scale_spine_length")
        exact = cells[(s, d, "exact_identity")]
        if not np.array_equal(exact["fitted_id"], exact["truth_id"]):
            problems.append(f"fixture {s}/d{d}: exact_identity's identity is not the truth's")
        if np.any(cells[(s, d, "mean_body")]["fitted_id"] != 0.0):
            problems.append(f"fixture {s}/d{d}: mean_body's identity is not zero")
        displaced = cells[(s, d, "spine_displaced")]
        other = np.arange(len(names)) != spine
        if (not np.array_equal(displaced["fitted_id"][other], displaced["truth_id"][other])
                or abs(displaced["fitted_id"][spine] - displaced_spine(float(displaced["truth_id"][spine]))) > 1e-6):
            problems.append(f"fixture {s}/d{d}: spine_displaced is not the truth with the spine moved "
                            f"{SPINE_DISPLACEMENT} toward zero")
    keys = list(truths)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.array_equal(truths[keys[i]], truths[keys[j]]):
                problems.append(f"fixtures {keys[i]} and {keys[j]} share one identity draw")

    complete = [(s, d) for s, d in fixtures if all((s, d, a) in cells for a in arms)]
    per_fixture = {}
    for s, d in complete:
        floor = floors_mm(cells[(s, d, "exact_identity")]["distance"])
        row = {"floor_mm": {j: round(float(floor[i]), 6) for i, j in enumerate(MAPPED_JOINTS)},
               "pooled_mm": {a: round(pooled_mm(cells[(s, d, a)]["distance"]), 6) for a in arms},
               "segments": {}}
        for a in arms:
            c = cells[(s, d, a)]
            row["segments"][a] = segment_rows(c["truth_rest"], c["fitted_rest"], floor, scored)
        oracle_median = np.median(cells[(s, d, "oracle")]["distance"], axis=0)
        row["J_oracle_minus_floor_mm"] = {j: round(float(oracle_median[i] - floor[i]), 6)
                                          for i, j in enumerate(MAPPED_JOINTS)}
        names = cells[(s, d, "oracle")]["names"]
        for a in ("oracle", "warm", "converged"):
            c = cells[(s, d, a)]
            row.setdefault("identity_error", {})[a] = {
                n: round(float(c["fitted_id"][i] - c["truth_id"][i]), 6) for i, n in enumerate(names)
                if n in CHANNELS}
        row["undrawn_drift_oracle"] = {n: round(float(cells[(s, d, "oracle")]["fitted_id"][i]), 6)
                                       for i, n in enumerate(names) if n not in drawn_set
                                       and abs(float(cells[(s, d, "oracle")]["fitted_id"][i])) > 0.0}
        row["converged_iterations"] = cells[(s, d, "converged")]["record"]["calibration_iterations"]
        per_fixture[f"{s}_d{d}"] = row

    # closure on every oracle cell (fresh only), derived from max_abs_m and the frame counts
    closure_rows, closure_problems = {}, []
    if not burned:
        report = inputs.get("closure") or {}
        pairs = report.get("pairs") or {}
        for s, d in fixtures:
            tag = f"{s}_d{d}"
            oracle = cells.get((s, d, "oracle"))
            pair = pairs.get(tag)
            if oracle is None or pair is None:
                closure_problems.append(f"closure {tag}: missing")
                continue
            glb = oracle["record"].get("glb")
            ok = (Path(str(pair.get("glb", ""))).name == glb
                  and inputs["files"].get(glb) == oracle["record"].get("glb_sha256")
                  and inputs["files"].get(glb) is not None
                  and pair.get("frames_in_glb") == FRAMES and pair.get("frames_in_track") == FRAMES
                  and pair.get("joints_missing_from_the_glb") == [] and (pair.get("joints_compared") or 0) >= 127)
            value = pair.get("max_abs_m")
            finite = isinstance(value, (int, float)) and math.isfinite(value)
            closure_rows[tag] = {"max_abs_m": value, "identity_ok": ok,
                                 "within": bool(ok and finite and value <= CLOSURE_BAND_M)}
            if not ok:
                closure_problems.append(f"closure {tag}: not this cell's GLB or not a full 150-frame, 127-joint read")
            if not finite:
                closure_problems.append(f"closure {tag}: max_abs_m non-finite")

    population_ok = not problems and len(complete) == len(fixtures) and not closure_problems
    exact_pooled = {k: v["pooled_mm"]["exact_identity"] for k, v in per_fixture.items()}
    validity = population_ok and all(v <= VALIDITY_CEILING_MM for v in exact_pooled.values())
    l_by_fixture = {k: l_passes(v["segments"]["oracle"]) for k, v in per_fixture.items()}
    mean_misses = {k: misses_l(v["segments"]["mean_body"]) for k, v in per_fixture.items()}
    trunk = {k: v["segments"]["spine_displaced"]["trunk"] for k, v in per_fixture.items()}
    spine_fails_trunk = {k: bool(r["error_mm"] > r["tolerance_mm"]) for k, r in trunk.items()}
    exact_zero = {k: max(r["error_mm"] for r in v["segments"]["exact_identity"].values()) <= EXACT_ZERO_MM
                  and l_passes(v["segments"]["exact_identity"]) for k, v in per_fixture.items()}
    closure_ok = burned or (bool(closure_rows) and all(r["within"] for r in closure_rows.values())
                            and len(closure_rows) == len(fixtures))
    conjuncts = {
        "L": population_ok and all(l_by_fixture.values()),
        "closure": population_ok and closure_ok,
        "must_fail_i_mean_body_misses_L": population_ok and all(mean_misses.values()),
        "must_fail_ii_spine_displaced_fails_at_the_trunk": population_ok and all(spine_fails_trunk.values()),
        "must_fail_iii_exact_identity_reads_zero_and_passes": population_ok and all(exact_zero.values()),
    }
    if not population_ok:
        verdict = "FAIL"
    elif not validity:
        verdict = "INVALID"
    else:
        verdict = "PASS" if all(conjuncts.values()) else "FAIL"
    stop = population_ok and not all(spine_fails_trunk.values())
    trunk_scored = "trunk" in scored
    disposition = ("CLOSES: O1 superseded by D4b PASS" if verdict == "PASS" and trunk_scored else
                   "STAYS OPEN: " + ("the trunk is not scored (the drawn-set rule); the card: such a run cannot "
                                     "close D4" if not trunk_scored else f"verdict {verdict}"))
    if burned:
        disposition = "BURNED: proves the instrument on known data; never evidence, never a disposition"

    def fixture_values(key_fn):
        return {k: key_fn(v) for k, v in per_fixture.items()}

    oracle_segment_errors = {s_: [v["segments"]["oracle"][s_]["error_mm"] for v in per_fixture.values()]
                             for s_ in SEGMENTS}
    oracle_segment_within = {s_: sum(v["segments"]["oracle"][s_]["within"] for v in per_fixture.values())
                             for s_ in SEGMENTS}
    limb_segments = [s_ for s_ in scored if s_ not in ("trunk", "neck_head")]
    clauses = {
        "drawn_set": {
            "predicted": "eight drawn; scale_foot_length UNSEEN, scale_hip_height UNDRAWABLE",
            "measured": {"drawn": drawn_set, "count": len(drawn_set), "scored_segments": scored},
            "verdict": "REPORTED (the rule, frozen before any fit)",
            "prediction": "HELD" if len(drawn_set) == 8 else "FAILED"},
        "validity_exact_identity_pooled_le_1mm": {
            "predicted": "<= 1.0 mm on every fixture (D4 read 0.51-0.76)",
            "measured": {"per_fixture_mm": exact_pooled,
                         "max_mm": max(exact_pooled.values()) if exact_pooled else None},
            "verdict": "PASS" if validity else ("INVALID" if population_ok else "FAIL"),
            "prediction": "HELD" if validity else "FAILED"},
        "L_oracle_rest_segments": {
            "predicted": "FAIL: the trunk fails on most fixtures; limb segments PASS",
            "measured": {"fixtures_passing": sum(l_by_fixture.values()), "fixtures": len(l_by_fixture),
                         "scored_segments_within_per_segment": {s_: oracle_segment_within[s_] for s_ in scored},
                         "max_error_mm_per_segment": {s_: max(v) if v else None
                                                      for s_, v in oracle_segment_errors.items()}},
            "verdict": "PASS" if conjuncts["L"] else "FAIL",
            "prediction": ("HELD" if not conjuncts["L"] else "FAILED")},
        "L_trunk": {
            "predicted": "the trunk segment FAILS on most fixtures",
            "measured": {"scored": trunk_scored,
                         "oracle_trunk_error_mm": fixture_values(lambda v: v["segments"]["oracle"]["trunk"]["error_mm"]),
                         "oracle_trunk_tolerance_mm": fixture_values(
                             lambda v: v["segments"]["oracle"]["trunk"]["tolerance_mm"]),
                         "fixtures_beyond_tolerance": sum(not v["segments"]["oracle"]["trunk"]["within"]
                                                          for v in per_fixture.values())},
            "verdict": ("REPORTED, NOT SCORED (spine below the drawn-set rule)" if not trunk_scored else
                        ("PASS" if all(v["segments"]["oracle"]["trunk"]["within"] for v in per_fixture.values())
                         else "FAIL")),
            "prediction": ("HELD" if sum(not v["segments"]["oracle"]["trunk"]["within"]
                                         for v in per_fixture.values()) > len(per_fixture) / 2 else "FAILED")},
        "L_limbs": {
            "predicted": "limb segments PASS",
            "measured": {"segments": limb_segments,
                         "all_within_on_all_fixtures": all(v["segments"]["oracle"][s_]["within"]
                                                           for v in per_fixture.values() for s_ in limb_segments)},
            "verdict": "PASS" if all(v["segments"]["oracle"][s_]["within"] for v in per_fixture.values()
                                     for s_ in limb_segments) else "FAIL",
            "prediction": "HELD" if all(v["segments"]["oracle"][s_]["within"] for v in per_fixture.values()
                                        for s_ in limb_segments) else "FAILED"},
        "closure_glb_fk_equals_track": {
            "predicted": "<= 1e-4 m on every oracle cell (D3's band)",
            "measured": ({"not read": "burned cells are D4's; D4's own closure is on record"} if burned else
                         {"per_fixture_max_abs_m": {k: r["max_abs_m"] for k, r in closure_rows.items()},
                          "problems": closure_problems}),
            "verdict": "PASS" if conjuncts["closure"] else "FAIL",
            "prediction": "HELD" if conjuncts["closure"] else "FAILED"},
        "must_fail_i_mean_body": {
            "predicted": "misses L on every fixture",
            "measured": {"fixtures_missing_L": sum(mean_misses.values()), "fixtures": len(mean_misses)},
            "verdict": "PASS" if conjuncts["must_fail_i_mean_body_misses_L"] else "FAIL",
            "prediction": "HELD" if conjuncts["must_fail_i_mean_body_misses_L"] else "FAILED"},
        "must_fail_ii_spine_displaced_at_the_trunk": {
            "predicted": "fails L at the trunk on every fixture (unconditional, read whether or not the trunk is scored)",
            "measured": {"trunk_error_mm": {k: r["error_mm"] for k, r in trunk.items()},
                         "trunk_tolerance_mm": {k: r["tolerance_mm"] for k, r in trunk.items()},
                         "fixtures_failing_at_the_trunk": sum(spine_fails_trunk.values()),
                         "trunk_scored_in_L": trunk_scored,
                         "the_band_as_scored_passes_the_displaced_spine_on": sum(
                             l_passes(v["segments"]["spine_displaced"]) for v in per_fixture.values())},
            "verdict": "PASS" if conjuncts["must_fail_ii_spine_displaced_fails_at_the_trunk"] else "FAIL",
            "prediction": "HELD" if conjuncts["must_fail_ii_spine_displaced_fails_at_the_trunk"] else "FAILED",
            "STOP": stop},
        "must_fail_iii_exact_identity": {
            "predicted": "L = 0 on every segment and the gate PASSES it, on every fixture",
            "measured": {"max_error_mm": max((max(r["error_mm"] for r in v["segments"]["exact_identity"].values())
                                              for v in per_fixture.values()), default=None),
                         "fixtures_reading_zero_and_passing": sum(exact_zero.values())},
            "verdict": "PASS" if conjuncts["must_fail_iii_exact_identity_reads_zero_and_passes"] else "FAIL",
            "prediction": "HELD" if conjuncts["must_fail_iii_exact_identity_reads_zero_and_passes"] else "FAILED"},
    }
    clauses["overall"] = {"predicted": "FAIL", "measured": verdict, "verdict": verdict,
                          "prediction": "HELD" if verdict == "FAIL" else "FAILED"}

    # REPORTED, never banded
    def probe(arm: str) -> dict:
        out = {}
        for k, v in per_fixture.items():
            errors = v["identity_error"][arm]
            out[k] = {"spine_error": errors.get("scale_spine_length"),
                      "drawn_channel_max_abs_error": max(abs(errors[c]) for c in drawn_set),
                      "L_passes": l_passes(v["segments"][arm]),
                      "trunk_error_mm": v["segments"][arm]["trunk"]["error_mm"],
                      "pooled_mm": v["pooled_mm"][arm]}
        return out

    reported = {
        "J_oracle_minus_paired_floor_mm": fixture_values(lambda v: v["J_oracle_minus_floor_mm"]),
        "pooled_statistic_mm_per_arm": fixture_values(lambda v: v["pooled_mm"]),
        "oracle_identity_error_on_the_drawn_set": fixture_values(
            lambda v: {c: v["identity_error"]["oracle"][c] for c in drawn_set}),
        "undrawn_identity_channels": [n for n in (canonical_names or []) if n not in drawn_set],
        "undrawn_drift_oracle_nonzero_by_name": fixture_values(lambda v: v["undrawn_drift_oracle"]),
        "WARM_calibration_started_at_the_truth": probe("warm") if per_fixture else {},
        "CONVERGED_max_iter_300": probe("converged") if per_fixture else {},
        "CONVERGED_iterations": fixture_values(lambda v: {
            k: v["converged_iterations"].get(k) for k in ("solves_per_call", "max_iteration_index_reached",
                                                          "solves_stopped_at_the_cap")}),
        "segments_all_arms": fixture_values(lambda v: v["segments"]),
        "floors_mm": fixture_values(lambda v: v["floor_mm"]),
    }
    return {
        "step": "D4b", "burned": burned,
        "verdict": verdict, "d4_disposition": disposition, "STOP": stop,
        "conjuncts": conjuncts, "validity": validity, "population_ok": population_ok,
        "problems": problems + closure_problems,
        "population": {"fixtures": [f"{s}_d{d}" for s, d in fixtures], "arms": list(arms),
                       "frames": FRAMES, "joints": len(MAPPED_JOINTS), "cells_expected": len(expected_names),
                       "cells_valid": len(cells)},
        "drawn_set": drawn_set, "segments_scored": scored,
        "segments_reported_not_scored": [s_ for s_ in SEGMENTS if s_ not in scored],
        "clauses": clauses,
        "reported": reported,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--closure", type=Path)
    parser.add_argument("--burned", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = build(load_inputs(arguments.cells, arguments.closure), burned=arguments.burned)
    arguments.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, clause in report["clauses"].items():
        measured = clause["measured"]
        print(f"  {name:48s} {clause['verdict']:12s} prediction {clause['prediction']}")
    for problem in report["problems"][:20]:
        print("  PROBLEM:", problem)
    print(f"{'BURNED ' if report['burned'] else ''}VERDICT: {report['verdict']}   STOP: {report['STOP']}   "
          f"D4: {report['d4_disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
