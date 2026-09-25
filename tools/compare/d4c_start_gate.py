#!/usr/bin/env python3
"""D4c the calibration's start: ONE verdict, derived from the inputs.

The card is the "D4c the calibration's start" row of `docs/LADDER_EXECUTION_PLAN.md` §2 (copy:
`docs/reviews/body-model-start-card-2026-09-25.md`). This gate reads only what the producers wrote -- the frozen
drawn set, the development JSON and cells, the acceptance cells and their arrays, the closure report, the B1 and
B2 reports, the hygiene and tripwire files themselves and the fitter's source at the card's base commit -- and
derives every clause from them. No verdict is a literal.

    verdict = STOP     if precondition 0 fails (the spine not drawn under the limit-aware rule, or a solver failure)
            = STOP     else if stage 0b fails (the spine control passes L as scored on any development fixture)
            = FAIL     else if the population, provenance or an arm's CONSTRUCTION is short, missing, non-finite
                       or mismatched (the card: "a missing, short or non-finite cell is FAIL")
            = INVALID  else if the exact_identity arm's pooled statistic exceeds 1.0 mm on any acceptance fixture
            = STOP     else if init-only PASSES L on any acceptance fixture
            = PASS     else iff  L  AND closure  AND must-fails i-iv  AND B1  AND B2  AND hygiene
            = FAIL     otherwise

THE BAND L (D4b's, unchanged): per fixture, every segment a drawn channel moves -- the rest length at MHR's zero
pose with the identity retained, |fitted - truth| in mm -- within the sum of its two endpoints' PAIRED floors (the
per-joint median over frames of the same fixture's exact_identity arm). The arithmetic is imported from
`d4b_o1_gate`, never copied.

The gate VERIFIES CONSTRUCTIONS, not labels (Astra's card-review finding 6): it regenerates every acceptance draw
from (seed, donor) with the frozen drawn set, recomputes every landmark start from the consumed landmarks with the
frozen rule (its own momentum-free copy of the arithmetic, fed by the stage-1 record's rest lengths and slopes),
and binds every closure row's measured sha256 to the files on disk.

ONE verdict line is printed. D4's consequence is derived from it in the JSON (`consequence`) and is not a
separate printed line.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate.py --development \
        --out docs/reviews/body-model-start-records/development.json            # stage 3: the freeze
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4c_start_gate.py --out artifacts/compare/d4c-start/gate.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fitter"))
sys.path.insert(0, str(ROOT / "tools/compare"))
import d4b_o1_gate as d4bg  # noqa: E402
from d4b_identifiability import CHANNELS, MAPPED_JOINTS, SEGMENTS, THRESHOLD_MM  # noqa: E402
import d4c_fixture as fx  # noqa: E402

RECORDS = ROOT / "docs/reviews/body-model-start-records"
DRAWN_SET = RECORDS / "drawn-set.json"
DEVELOPMENT = RECORDS / "development.json"
MANIFEST = RECORDS / "acceptance-manifest.json"
TRIPWIRE = RECORDS / "tripwire.json"
OUT_DIR = ROOT / "artifacts/compare/d4c-start"
DEV_CELLS = OUT_DIR / "development"
ACC_CELLS = OUT_DIR / "acceptance"
FITTER = ROOT / "tools/fitter/mhr_delivery.py"
BASE_COMMIT = "a58e447f18fc7492e6d494ee3656720dc5f0086f"        # the card's base; fitter 3136befb there
BASE_FITTER_SHA256 = "3136befb6cd0415fe746a208f73e15be0f1c8af1ad3594cad277c7f9e5b13535"
SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
EIGHT = ("subject-00.glb", "subject-01.glb", "subject-00.body-track.npz", "subject-01.body-track.npz",
         "subject-00.body-track.json", "subject-01.body-track.json", "subject-00.mapping.npz",
         "subject-01.mapping.npz")
D4_DELIVERY = ROOT / "artifacts/compare/d4-body/delivery"
D4_O1 = ROOT / "artifacts/compare/d4-body/o1"
FRAMES = 150
IDENTITY_CHANNELS = 68
VALIDITY_CEILING_MM = 1.0
CLOSURE_BAND_M = 1e-4
EXACT_ZERO_MM = 1e-9
START_AGREEMENT = 1e-4          # |fitter's float32 start - the gate's float64 recomputation|, units
LANDMARK_AGREEMENT_CM = 1e-9    # the consumed landmarks = the truth's mapped joints
LAYOUT_AGREEMENT_CM = 1e-6
TIE_ORDER = ("median", "p90", "p95")
PERCENTILE = {"median": 50.0, "p90": 90.0, "p95": 95.0}
SETTINGS = dict(fx.SETTINGS)
CALIBRATE_CALLS = {"candidate": 2, "candidate_median": 2, "candidate_p90": 2, "candidate_p95": 2, "legacy": 2,
                   "exact_identity": 0, "mean_body": 0, "spine_displaced": 0, "init_only": 0}
B1_CANDIDATE, B1_BASELINE, B1_D4 = "D4c_fitted_MHR_lod2", "baseline_D7c_rig", "D4_fitted_MHR_lod2"
B1_CAMERAS = ("A001", "B001", "C001", "D001")
SUBJECTS = ("subject_00", "subject_01")
# The declared landmark -> joint mapping (mhr_delivery.MAP), carried here so the start is re-derived rather than
# the fitter agreeing with itself; the gate checks it against the delivered track's own declaration.
MAP = {'root': 'root', 'neck': 'c_neck', 'nose': 'c_head', 'left_shoulder': 'l_uparm',
       'right_shoulder': 'r_uparm', 'left_elbow': 'l_lowarm', 'right_elbow': 'r_lowarm',
       'left_wrist': 'l_wrist', 'right_wrist': 'r_wrist', 'left_hip': 'l_upleg', 'right_hip': 'r_upleg',
       'left_knee': 'l_lowleg', 'right_knee': 'r_lowleg', 'left_ankle': 'l_foot', 'right_ankle': 'r_foot',
       'left_eye': 'l_eye', 'right_eye': 'r_eye'}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


# --------------------------------------------------------------------------------- precondition 0

def precondition_0(drawn: dict) -> dict:
    """Re-derive the limit-aware drawn set from its own recorded medians; the spine must be drawn."""
    problems = []
    per = drawn.get("per_donor") or {}
    rederived = []
    for c in CHANNELS:
        values = [(per.get(str(d)) or {}).get("bounded_residual_median_mm", {}).get(c) for d in (0, 1)]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            problems.append(f"drawn set: {c} has no finite bounded median on both donors")
            continue
        if all(v > (drawn.get("frozen") or {}).get("threshold_mm", math.inf) for v in values):
            rederived.append(c)
    recorded = list(drawn.get("drawn_set") or [])
    if rederived != recorded:
        problems.append("drawn set: the rule re-applied to its own medians does not give its drawn set")
    if (drawn.get("frozen") or {}).get("threshold_mm") != THRESHOLD_MM:
        problems.append("drawn set: the threshold is not the card's 2 mm")
    if drawn.get("frozen_before_any_fit") is not True:
        problems.append("drawn set: not marked frozen before any fit")
    failures = (drawn.get("precondition_0") or {}).get("solver_failures")
    if failures is None:
        problems.append("drawn set: the solver-failure record is missing")
    moved_by = drawn.get("segment_moved_by") or {}
    if set(moved_by) != set(SEGMENTS):
        problems.append("drawn set: segment_moved_by does not name the twelve segments")
    scored = [s for s in SEGMENTS if any(c in rederived for c in moved_by.get(s, []))]
    if scored != list(drawn.get("segments_scored") or []):
        problems.append("drawn set: its scored segments are not re-derivable")
    spine = "scale_spine_length" in rederived
    trunk_scored = "trunk" in scored
    stop = (not spine) or (not trunk_scored) or bool(failures) or bool(problems)
    return {"drawn_set": rederived, "scored": scored, "spine_drawn": spine, "trunk_scored": trunk_scored,
            "solver_failures": failures, "problems": problems, "holds": not stop}


# ------------------------------------------------------------------------ the start, re-derived

def calibration_frames(frames: int, calib_frames: int) -> list[int]:
    stride = max(1, (frames - 1) // calib_frames) if calib_frames > 0 and frames > 0 else 1
    return list(range(0, frames, stride))


def recompute_start(array_zup_m: np.ndarray, joint_names: list[str], drawn: dict, statistic: str) -> dict[str, float]:
    """The frozen rule, momentum-free: (l_s - l0_s) / k_c clipped, from the stage-1 record's l0 and slopes."""
    array = np.asarray(array_zup_m, np.float64)
    frames = calibration_frames(array.shape[0], SETTINGS["calib_frames"])
    landmark_of = {joint: landmark for landmark, joint in MAP.items()}
    rest0 = drawn["zero_identity_rest_lengths_mm"]
    slopes = drawn["signed_rest_length_change_mm_per_unit_at_the_upper_limit"]
    limits = drawn["configured_limits"]
    out = {}
    for channel, segments in drawn["start_segments_by_drawn_channel"].items():
        q = PERCENTILE[statistic] if segments == ["trunk"] else 50.0
        observed = []
        for segment in segments:
            a, b = SEGMENTS[segment]
            ia, ib = joint_names.index(landmark_of[a]), joint_names.index(landmark_of[b])
            lengths = np.linalg.norm(array[frames, ia] - array[frames, ib], axis=1) * 1000.0
            observed.append(float(np.nanpercentile(lengths, q, method="linear")))
        length = float(np.mean(observed))
        length0 = float(np.mean([rest0[s] for s in segments]))
        k = float(np.mean([slopes[channel][s] for s in segments]))
        low, high = limits[channel]
        out[channel] = float(np.clip((length - length0) / k, low, high))
    return out


# ------------------------------------------------------------------------------------- the cells

def read_cells(directory: Path, population: list[tuple], arms: tuple) -> tuple[dict, dict, dict]:
    cells, arrays, files = {}, {}, {}
    for path in sorted(Path(directory).glob("cell-*.json")):
        cells[path.name] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(Path(directory).glob("fixture-*")):
        if path.suffix in (".glb", ".npz"):
            files[path.name] = sha256_file(path)
    for s, d in population:
        for a in ("candidate", "init_only", "candidate_median", "candidate_p90", "candidate_p95"):
            if a not in arms:
                continue
            path = Path(directory) / f"fixture-{s}-d{d}-{a}.arrays.npz"
            if path.is_file():
                data = np.load(path)       # no pickle: this file is written with plain arrays
                arrays[path.name] = {"consumed_landmarks_z_up_m": np.asarray(data["consumed_landmarks_z_up_m"]),
                                     "consumed_joint_names": [str(n) for n in data["consumed_joint_names"]],
                                     "truth_mapped_cm": np.asarray(data["truth_mapped_cm"])}
    return cells, arrays, files


def check_cells(inputs: dict, key: str, population: list[tuple], arms: tuple, expected_prov: dict,
                problems: list[str]) -> dict:
    """Population, provenance and construction for one population. Returns the valid cells by (s, d, arm)."""
    cells_in = inputs[key]["cells"]
    arrays = inputs[key]["arrays"]
    files = inputs[key]["files"]
    drawn = inputs["drawn"]
    expected_names = {f"cell-{s}-d{d}-{a}.json" for s, d in population for a in arms}
    for name in sorted(expected_names - set(cells_in)):
        problems.append(f"{key}: {name} missing")
    for name in sorted(set(cells_in) - expected_names):
        problems.append(f"{key}: {name} is not in the named population")
    acceptance = key == "acceptance"
    statistic = inputs.get("chosen_statistic")
    canonical = None
    cells = {}
    for s, d in population:
        for a in arms:
            record = cells_in.get(f"cell-{s}-d{d}-{a}.json")
            if record is None:
                continue
            tag, bad = f"{key} {s}/d{d}/{a}", []
            if record.get("schema") != fx.SCHEMA:
                bad.append("schema")
            if (record.get("seed"), record.get("donor"), record.get("arm")) != (s, d, a):
                bad.append("seed/donor/arm do not match the file")
            want_population = "acceptance" if acceptance else ("d4" if s in fx.D4_SEEDS else "d4b")
            if record.get("population") != want_population:
                bad.append("population label")
            if record.get("lod") != 2 or record.get("frames") != FRAMES:
                bad.append("lod or frames")
            if list(record.get("mapped_joints") or []) != list(MAPPED_JOINTS):
                bad.append("mapped joints")
            distance = d4bg.finite_array(record.get("distance_mm"), (FRAMES, len(MAPPED_JOINTS)))
            truth_rest = d4bg.finite_array(record.get("truth_rest_mapped_cm"), (len(MAPPED_JOINTS), 3))
            fitted_rest = d4bg.finite_array(record.get("fitted_rest_mapped_cm"), (len(MAPPED_JOINTS), 3))
            truth_id = d4bg.finite_array(record.get("truth_identity"), (IDENTITY_CHANNELS,))
            fitted_id = d4bg.finite_array(record.get("fitted_identity"), (IDENTITY_CHANNELS,))
            for label, value in (("distance_mm", distance), ("truth_rest", truth_rest), ("fitted_rest", fitted_rest),
                                 ("truth_identity", truth_id), ("fitted_identity", fitted_id)):
                if value is None:
                    bad.append(f"{label} short, missing or non-finite")
            names = list(record.get("identity_channel_names") or [])
            if (len(names) != IDENTITY_CHANNELS or len(set(names)) != IDENTITY_CHANNELS
                    or not all(str(n).startswith("scale_") for n in names) or not set(CHANNELS) <= set(names)):
                bad.append("identity channel names")
            elif canonical is None:
                canonical = names
            elif names != canonical:
                bad.append("identity channel names differ between cells")
            layout = record.get("truth_rest_full_vs_simplified_character_max_abs_cm")
            if not isinstance(layout, (int, float)) or not math.isfinite(layout) or layout > LAYOUT_AGREEMENT_CM:
                bad.append("the truth rest differs between the 204- and 178-parameter characters")
            prov = record.get("provenance") or {}
            for field, value in expected_prov.items():
                if prov.get(field) != value or (value is None and not (field == "development_json_sha256"
                                                                       and not acceptance)):
                    bad.append(f"provenance {field}")
            settings = record.get("settings") or {}
            for field, value in SETTINGS.items():
                if settings.get(field) != value:
                    bad.append(f"settings {field}")
            if settings.get("calibration_debug_captured") is not (a in fx.CALIBRATING_ARMS):
                bad.append("settings calibration_debug_captured")
            if record.get("calibrate_markers_calls") != CALIBRATE_CALLS[a]:
                bad.append("calibrate_markers calls")
            if a in fx.CALIBRATING_ARMS:
                it = record.get("calibration_iterations") or {}
                per_call = it.get("per_call") or []
                if it.get("calls") != 2 or len(per_call) != 2 or it.get("max_iter") != SETTINGS["max_iter"]:
                    bad.append("the calibration iteration record is not two calls at max_iter")
            if files.get(record.get("arrays")) != record.get("arrays_sha256") or record.get("arrays_sha256") is None:
                bad.append("arrays file hash")
            start = record.get("start") or {}
            want_stat = (statistic if a in ("candidate", "init_only")
                         else a.split("_", 1)[1] if a.startswith("candidate_") else None)
            if start.get("trunk_statistic") != want_stat or (want_stat is None and a in ("candidate", "init_only")):
                bad.append("start: trunk statistic")
            if bool(start.get("zero_start")) is not (a == "legacy") or bool(
                    start.get("held_without_calibration")) is not (a == "init_only"):
                bad.append("start: arm flags")
            if (start.get("start_identity") is None) is not (want_stat is None):
                bad.append("start: identity presence")
            if a != "spine_displaced" and record.get("arm_note") != {}:
                bad.append("arm_note on an arm that carries none")
            if bad:
                problems.append(f"cell {tag}: " + "; ".join(bad))
                continue
            cells[(s, d, a)] = {"record": record, "distance": distance, "truth_rest": truth_rest,
                                "fitted_rest": fitted_rest, "truth_id": truth_id, "fitted_id": fitted_id,
                                "names": names}

    # fixture identity and arm construction, verified
    limits = drawn.get("configured_limits") or {}
    truths, clamps = {}, {}
    for s, d in population:
        group = [cells.get((s, d, a)) for a in arms]
        if any(c is None for c in group):
            continue
        first = group[0]
        names = first["names"]
        fixture = first["record"].get("fixture") or {}
        for c in group[1:]:
            if (c["record"].get("fixture") != first["record"].get("fixture")
                    or not np.array_equal(c["truth_id"], first["truth_id"])
                    or not np.array_equal(c["truth_rest"], first["truth_rest"])):
                problems.append(f"{key} fixture {s}/d{d}: the arms do not share one truth")
                break
        truth = {n: float(first["truth_id"][i]) for i, n in enumerate(names)}
        if acceptance:
            if fixture.get("generator") != f"numpy default_rng([{s}, {d}]) (seeded by the pair)":
                problems.append(f"acceptance fixture {s}/d{d}: generator")
            if list(fixture.get("drawn_channels") or []) != list(inputs["pre0"]["drawn_set"]):
                problems.append(f"acceptance fixture {s}/d{d}: drawn channels are not the frozen drawn set")
            regenerated = fx.acceptance_draw(s, d, inputs["pre0"]["drawn_set"], limits, CHANNELS)
            for n in names:
                want = np.float32(regenerated[n]) if n in regenerated else np.float32(0.0)
                if np.float32(truth[n]) != want:
                    problems.append(f"acceptance fixture {s}/d{d}: {n} is not the regenerated draw")
                    break
        clamps.setdefault(d, set()).add(fixture.get("clamped_parameter_frames"))
        truths[(s, d)] = first["truth_id"]
        spine = names.index("scale_spine_length")
        if (s, d, "exact_identity") in cells and not np.array_equal(
                cells[(s, d, "exact_identity")]["fitted_id"], first["truth_id"]):
            problems.append(f"{key} fixture {s}/d{d}: exact_identity is not the truth")
        if (s, d, "mean_body") in cells and np.any(cells[(s, d, "mean_body")]["fitted_id"] != 0.0):
            problems.append(f"{key} fixture {s}/d{d}: mean_body is not zero")
        if (s, d, "spine_displaced") in cells:
            c = cells[(s, d, "spine_displaced")]
            other = np.arange(len(names)) != spine
            if (not np.array_equal(c["fitted_id"][other], c["truth_id"][other])
                    or abs(c["fitted_id"][spine] - fx.displaced_spine(float(c["truth_id"][spine]))) > 1e-6):
                problems.append(f"{key} fixture {s}/d{d}: spine_displaced is not the truth with the spine moved")
        # the landmark start, recomputed from the consumed landmarks and the frozen rule
        for a in [x for x in arms if x in ("candidate", "init_only") or x.startswith("candidate_")]:
            c = cells[(s, d, a)]
            stat = c["record"]["start"]["trunk_statistic"]
            arr = arrays.get(c["record"].get("arrays"))
            if arr is None or stat not in PERCENTILE:
                problems.append(f"{key} fixture {s}/d{d}/{a}: the consumed landmarks are not readable")
                continue
            consumed, joint_names = arr["consumed_landmarks_z_up_m"], arr["consumed_joint_names"]
            truth_cm = arr["truth_mapped_cm"]
            as_capture = np.stack([truth_cm[..., 0], -truth_cm[..., 2], truth_cm[..., 1]], axis=-1) / 100.0
            columns = [joint_names.index(l) for l in MAP]
            if consumed.shape != (FRAMES, len(joint_names), 3) or np.nanmax(
                    np.abs(consumed[:, columns] - as_capture)) * 100.0 > LANDMARK_AGREEMENT_CM:
                problems.append(f"{key} fixture {s}/d{d}/{a}: the consumed landmarks are not the truth's joints")
                continue
            recomputed = recompute_start(consumed, joint_names, drawn, stat)
            recorded = dict(zip(names, c["record"]["start"]["start_identity"] or []))
            held = {n: float(c["fitted_id"][i]) for i, n in enumerate(names)} if a == "init_only" else recorded
            for n in names:
                want = recomputed.get(n, 0.0)
                if n not in recorded or abs(recorded[n] - want) > START_AGREEMENT or abs(held[n] - want) > START_AGREEMENT:
                    problems.append(f"{key} fixture {s}/d{d}/{a}: the start is not the frozen rule's ({n})")
                    break
    for d, counts in clamps.items():
        if len(counts) != 1 or not all(isinstance(c, int) and c > 0 for c in counts):
            problems.append(f"{key} donor {d}: the fixture clamp is not one donor-determined count")
    keys = list(truths)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.array_equal(truths[keys[i]], truths[keys[j]]):
                problems.append(f"{key}: fixtures {keys[i]} and {keys[j]} share one identity draw")
    return cells


def fixture_rows(cells: dict, population: list[tuple], arms: tuple, scored: list[str]) -> dict:
    rows = {}
    for s, d in population:
        if not all((s, d, a) in cells for a in arms):
            continue
        floor = d4bg.floors_mm(cells[(s, d, "exact_identity")]["distance"])
        row = {"floor_mm": {j: float(floor[i]) for i, j in enumerate(MAPPED_JOINTS)},
               "pooled_mm": {a: d4bg.pooled_mm(cells[(s, d, a)]["distance"]) for a in arms},
               "segments": {a: d4bg.segment_rows(cells[(s, d, a)]["truth_rest"], cells[(s, d, a)]["fitted_rest"],
                                                 floor, scored) for a in arms}}
        names = cells[(s, d, arms[0])]["names"]
        row["identity_error"] = {a: {n: float(cells[(s, d, a)]["fitted_id"][i] - cells[(s, d, a)]["truth_id"][i])
                                     for i, n in enumerate(names) if n in CHANNELS} for a in arms}
        rows[f"{s}_d{d}"] = row
    return rows


def trunk_fails_as_scored(row: dict) -> bool:
    r = row["segments"]["spine_displaced"]["trunk"]
    return bool(r["scored"] and r["error_mm"] > r["tolerance_mm"])


def trunk_ratio(row: dict, arm: str) -> float:
    r = row["segments"][arm]["trunk"]
    return r["error_mm"] / r["tolerance_mm"] if r["tolerance_mm"] > 0 else math.inf


# ------------------------------------------------------------------------------------- development

def development(inputs: dict) -> dict:
    """Stage 0b and the ONE development choice, on the 18 burned fixtures."""
    problems: list[str] = []
    population = fx.POPULATIONS["d4"] + fx.POPULATIONS["d4b"]
    prov = expected_provenance(inputs, development=True)
    ran_on = inputs.get("development_ran_on")
    if ran_on:   # the gate, after the freeze: the cells ran on the stage-2 fitter and fixture, recorded in the JSON
        prov.update(fitter_sha256=ran_on.get("fitter_sha256"), this_file_sha256=ran_on.get("fixture_sha256"))
    cells = check_cells(inputs, "development", population, fx.DEVELOPMENT_ARMS, prov, problems)
    scored = inputs["pre0"]["scored"]
    rows = fixture_rows(cells, population, fx.DEVELOPMENT_ARMS, scored)
    complete = len(rows) == len(population) and not problems
    stage_0b = {k: trunk_fails_as_scored(v) for k, v in rows.items()}
    readings = {}
    for arm in ("legacy",) + tuple(f"candidate_{s}" for s in TIE_ORDER):
        ratios = {k: trunk_ratio(v, arm) for k, v in rows.items()}
        readings[arm] = {
            "worst_trunk_L_ratio": max(ratios.values()) if ratios else None,
            "trunk_L_ratio_per_fixture": ratios,
            "fixtures_with_the_trunk_within": sum(r <= 1.0 for r in ratios.values()),
            "fixtures_passing_L": sum(d4bg.l_passes(v["segments"][arm]) for v in rows.values()),
            "spine_error_per_fixture": {k: v["identity_error"][arm]["scale_spine_length"] for k, v in rows.items()},
            "trunk_error_mm_per_fixture": {k: v["segments"][arm]["trunk"]["error_mm"] for k, v in rows.items()},
            "worst_scored_segment_ratio_per_fixture": {
                k: max(r["error_mm"] / r["tolerance_mm"] for r in v["segments"][arm].values() if r["scored"])
                for k, v in rows.items()},
        }
    stop_0b = complete and not all(stage_0b.values())
    chosen = None
    if complete and not stop_0b:
        best = min(readings[f"candidate_{s}"]["worst_trunk_L_ratio"] for s in TIE_ORDER)
        chosen = next(s for s in TIE_ORDER if readings[f"candidate_{s}"]["worst_trunk_L_ratio"] == best)
    return {
        "step": "D4c", "stage": 3,
        "what": "DEVELOPMENT, burned (D4's six + D4b's twelve): stage 0b, then the ONE development choice -- the "
                "trunk statistic from {median, p90, p95}",
        "frozen_before_any_development_reading": {
            "calibration_frames": "momentum's own selection: stride max(1, (n - 1) // calib_frames), range(0, n, "
                                  "stride) -- 0..149 at 150/100 (calibration-frames.json, read at stage 2)",
            "percentile_interpolation": "numpy linear (nanpercentile method='linear')",
            "tie_order": list(TIE_ORDER),
            "numerator": "the POST-CALIBRATION trunk |fitted - truth| rest length over its paired tolerance (the sum of "
                         "the root and c_neck floors of the same fixture's exact_identity arm)",
            "rule": "the smallest worst-case (max over the 18 fixtures) trunk L-ratio; exact ties go median, then p90, "
                    "then p95",
            "committed_in": "tools/fitter/mhr_delivery.py (landmark_start, calibration_frame_indices) at stage 2",
        },
        "population": [f"{s}_d{d}" for s, d in population],
        "complete": complete,
        "problems": problems,
        "scored_segments": scored,
        "stage_0b": {"spine_displaced_fails_L_at_the_trunk_as_scored": stage_0b,
                     "fixtures_failing": sum(stage_0b.values()), "fixtures": len(stage_0b),
                     "trunk_error_mm": {k: v["segments"]["spine_displaced"]["trunk"]["error_mm"] for k, v in rows.items()},
                     "trunk_tolerance_mm": {k: v["segments"]["spine_displaced"]["trunk"]["tolerance_mm"]
                                            for k, v in rows.items()},
                     "STOP": stop_0b},
        "readings": readings,
        "exact_identity_pooled_mm": {k: v["pooled_mm"]["exact_identity"] for k, v in rows.items()},
        "chosen_trunk_statistic": chosen,
        "development_ran_on": {"fitter_sha256": prov["fitter_sha256"], "fixture_sha256": prov["this_file_sha256"]},
        "cells_sha256": {name: sha256_bytes(json.dumps(rec, sort_keys=True).encode())
                         for name, rec in sorted(inputs["development"]["cells"].items())},
    }


# ---------------------------------------------------------------------------------------- provenance

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def expected_provenance(inputs: dict, development: bool = False) -> dict:
    stage1 = (inputs.get("stage1_provenance") or {}).get("sha256") or {}

    def pick(prefix: str) -> str | None:
        hits = [v for k, v in stage1.items() if k.startswith(prefix)]
        return hits[0] if len(hits) == 1 else None

    return {"fitter_sha256": inputs["fitter_sha256"],
            "d4_fixture_sha256": pick("D4 fixture "), "d4b_fixture_sha256": pick("D4b fixture "),
            "this_file_sha256": inputs["fixture_sha256"],
            "donor_sha256": {"0": pick("donor 0 "), "1": pick("donor 1 ")},
            "model_sha256": pick("model "), "fbx_sha256": pick("fbx "),
            "drawn_set_sha256": sha256_bytes(inputs["drawn_set_bytes"]),
            "d4b_drawn_set_sha256": pick("D4b frozen drawn set "),
            "development_json_sha256": None if development else inputs.get("development_sha256"),
            "pymomentum": (inputs.get("stage1_provenance") or {}).get("environment", {}).get("pymomentum-cpu")}


def source_diff(inputs: dict) -> dict:
    """Attribution (Astra finding 5): fit_one differs from the base ONLY in the two starting identities, each its
    own copy, and in the new keyword and its docstring. Compared as ASTs after undoing exactly those edits."""
    def fit_one(source: str):
        tree = ast.parse(source)
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_one")

    try:
        old, new = fit_one(inputs["fitter_base_source"]), fit_one(inputs["fitter_source"])
    except (StopIteration, SyntaxError, TypeError):
        return {"only_the_two_starts": False, "why": "fit_one not found"}
    new_args = [a.arg for a in new.args.kwonlyargs]
    old_args = [a.arg for a in old.args.kwonlyargs]
    ok = new_args == old_args + ["start_identity"]
    # undo: drop the start_identity kwarg and its default, the docstring, the `start = ...` line
    new.args.kwonlyargs = new.args.kwonlyargs[:-1]
    new.args.kw_defaults = new.args.kw_defaults[:-1]
    new.body = new.body[1:] if isinstance(new.body[0], ast.Expr) else new.body
    old.body = old.body[1:] if isinstance(old.body[0], ast.Expr) else old.body
    starts, calls = 0, 0
    for node in ast.walk(new):
        if isinstance(node, ast.If):
            for field in ("body", "orelse"):
                kept = []
                for statement in getattr(node, field):
                    if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                            and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == "start"
                            and ast.unparse(statement.value) == EXPECTED_START):
                        starts += 1
                        continue
                    kept.append(statement)
                setattr(node, field, kept)
    for node in ast.walk(new):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "copy"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "start"):
            node.func.value.id = "zero"
            calls += 1
    same = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    return {"only_the_two_starts": bool(ok and same and starts == 1 and calls == 2),
            "kwonly_args_added": [a for a in new_args if a not in old_args],
            "start_assignments_removed": starts, "start_copies_mapped_back_to_zero_copies": calls,
            "ast_equal_after_undoing_them": same,
            "base": f"{BASE_COMMIT}:tools/fitter/mhr_delivery.py",
            "base_sha256": sha256_bytes(inputs["fitter_base_source"].encode()) if inputs["fitter_base_source"] else None}


EXPECTED_START = "zero if start_identity is None else np.asarray(start_identity, np.float32)"


def trunk_statistic_in_source(source: str) -> str | None:
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "TRUNK_STATISTIC":
            return getattr(node.value, "value", None)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRUNK_STATISTIC" for t in node.targets):
            return getattr(node.value, "value", None)
    return None


# ------------------------------------------------------------------------------------------ inputs

def load_inputs(dev_cells: Path = DEV_CELLS, acc_cells: Path = ACC_CELLS, out_dir: Path = OUT_DIR,
                development_only: bool = False, fitter_source: str | None = None,
                fitter_sha256: str | None = None, history_ref: str | None = None) -> dict:
    """`fitter_source` / `fitter_sha256` (both or neither): replay the gate against a PINNED fitter -- D4c's, at tag
    ladder/D4c-fail-1a89cc7 -- instead of the working tree's, which later steps change by design (D4d).
    `history_ref`: read the freeze order (which commit ADDED the development JSON and the manifest) from that
    commit's history instead of HEAD's. After D4c's records landed on main alone and the tag was later merged back,
    HEAD's simplified history shows both files added by the one records-only commit, so the order D4c committed them
    in is only visible on the tag's own line."""
    if (fitter_source is None) != (fitter_sha256 is None):
        raise ValueError("pass both the pinned fitter's source and its sha256, or neither")
    if fitter_source is not None and sha256_bytes(fitter_source.encode()) != fitter_sha256:
        raise ValueError("the pinned fitter's source does not hash to the sha256 given for it")
    inputs = {"drawn_set_bytes": DRAWN_SET.read_bytes(),
              "stage1_provenance": json.loads((RECORDS / "provenance.json").read_text(encoding="utf-8")),
              "fitter_source": FITTER.read_text(encoding="utf-8") if fitter_source is None else fitter_source,
              "fitter_sha256": sha256_file(FITTER) if fitter_sha256 is None else fitter_sha256,
              "fixture_sha256": sha256_file(ROOT / "tools/fitter/d4c_fixture.py")}
    cells, arrays, files = read_cells(dev_cells, fx.POPULATIONS["d4"] + fx.POPULATIONS["d4b"], fx.DEVELOPMENT_ARMS)
    inputs["development"] = {"cells": cells, "arrays": arrays, "files": files}
    if development_only:
        return inputs
    inputs["development_json_bytes"] = DEVELOPMENT.read_bytes() if DEVELOPMENT.exists() else b""
    cells, arrays, files = read_cells(acc_cells, fx.POPULATIONS["acceptance"], fx.ACCEPTANCE_ARMS)
    inputs["acceptance"] = {"cells": cells, "arrays": arrays, "files": files}
    inputs["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    inputs["acceptance_cell_sha256"] = {n: sha256_file(Path(acc_cells) / n) for n in sorted(cells)}

    def added(path: Path) -> str:
        lines = git("log", "--diff-filter=A", "--format=%H", *([history_ref] if history_ref else []), "--",
                    str(path.relative_to(ROOT))).splitlines()
        return lines[-1] if lines else ""

    dev_commit, manifest_commit = added(DEVELOPMENT), added(MANIFEST)
    inputs["commits"] = {
        "development_json": dev_commit, "acceptance_manifest": manifest_commit,
        "development_json_time": int(git("show", "-s", "--format=%ct", dev_commit) or 0) if dev_commit else 0,
        "acceptance_manifest_time": int(git("show", "-s", "--format=%ct", manifest_commit) or 0) if manifest_commit else 0,
        "development_is_ancestor": bool(dev_commit and manifest_commit and subprocess.run(
            ["git", "merge-base", "--is-ancestor", dev_commit, manifest_commit], cwd=ROOT).returncode == 0),
        "development_json_committed_bytes_sha256": sha256_bytes(subprocess.run(
            ["git", "show", f"{dev_commit}:docs/reviews/body-model-start-records/development.json"], cwd=ROOT,
            capture_output=True).stdout) if dev_commit else None,
    }
    inputs["fitter_base_source"] = subprocess.run(["git", "show", f"{BASE_COMMIT}:tools/fitter/mhr_delivery.py"],
                                                  cwd=ROOT, capture_output=True, text=True).stdout
    closure = out_dir / "closure.json"
    inputs["closure"] = json.loads(closure.read_text(encoding="utf-8")) if closure.exists() else {}
    for name in ("b1-paired.json", "silhouette-delivery.json", "b2-same-denominator.json"):
        path = out_dir / name
        inputs[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    committed = ROOT / "artifacts/compare/silhouette.json"
    inputs["silhouette_committed"] = json.loads(committed.read_text(encoding="utf-8"))
    inputs["b1_files"] = {"D4c_mesh": sha256_file(out_dir / "work-delivery/delivered-mesh.npz"),
                          "silhouette_mesh": sha256_file(out_dir / "work-silhouette/delivered-mesh.npz"),
                          **{f"delivery/subject-{s:02d}.glb": sha256_file(out_dir / f"delivery/subject-{s:02d}.glb")
                             for s in (0, 1)}}
    inputs["b2_paths"] = {"delivery": str((out_dir / "delivery").resolve()),
                          "rig_build": str((out_dir / "hygiene").resolve())}
    inputs["hygiene"] = {"rebuild": {f: sha256_file(out_dir / "hygiene" / f) for f in EIGHT},
                         "shipped": {f: sha256_file(SHIPPED / f) for f in EIGHT}}
    trip = {}
    for s in ("00", "01"):
        for name in (f"subject-{s}.glb", f"subject-{s}.body-track.npz", f"subject-{s}.markers.npz",
                     f"fit-report-subject-{s}.json"):
            trip[f"delivery/{name}"] = [sha256_file(D4_DELIVERY / name),
                                        sha256_file(out_dir / "tripwire/delivery" / name)]
    for seed in fx.D4_SEEDS:
        for arm in ("oracle", "exact_identity", "mean_body"):
            for suffix in (".glb", ".body-track.npz", ".truth.npz"):
                name = f"seed-{seed}-{arm}{suffix}"
                trip[f"o1/{name}"] = [sha256_file(D4_O1 / name), sha256_file(out_dir / "tripwire/o1" / name)]
    inputs["tripwire_files"] = trip
    inputs["tripwire_record"] = json.loads(TRIPWIRE.read_text(encoding="utf-8")) if TRIPWIRE.exists() else {}
    return inputs


# --------------------------------------------------------------------------------------------- B1/B2

def b1_clauses(inputs: dict) -> dict:
    paired, sil, committed = inputs["b1-paired.json"], inputs["silhouette-delivery.json"], inputs["silhouette_committed"]
    problems = []
    pop = paired.get("population") or {}
    consumed = pop.get("frames_consumed_per_arm") or {}
    arms = (B1_CANDIDATE, B1_BASELINE, B1_D4)
    for arm in arms:
        if any((consumed.get(arm) or {}).get(s) != FRAMES for s in SUBJECTS):
            problems.append(f"B1: arm {arm} did not consume 150 frames on both performers")
    excluded = pop.get("cells_excluded_by_the_mask_cache") or {}
    required = {s: len(B1_CAMERAS) * FRAMES - len(excluded.get(s, [None] * 10**6)) for s in SUBJECTS}
    rows = {}
    for s in SUBJECTS:
        for cand, ref in ((B1_CANDIDATE, B1_BASELINE), (B1_CANDIDATE, B1_D4)):
            row = (paired.get("paired") or {}).get(f"{cand}_minus_{ref}_{s}") or {}
            ci = row.get("ci95_of_the_median_difference")
            if row.get("n") != required[s] or pop.get("cells_required", {}).get(s) != required[s] or required[s] <= 0:
                problems.append(f"B1: {cand} - {ref} {s} not scored on the named population")
            if not (isinstance(ci, list) and len(ci) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v)
                                                                   for v in ci)):
                problems.append(f"B1: {cand} - {ref} {s} carries no finite CI")
                ci = [math.nan, math.nan]
            rows[f"{cand}_minus_{ref}_{s}"] = {"median": row.get("median_difference"), "ci95": ci,
                                               "block_length": row.get("block_length"), "resamples": row.get("resamples")}
            if row.get("block_length") != 15 or row.get("resamples") != 2000:
                problems.append(f"B1: {cand} - {ref} {s} is not block 15 x 2000")
    band = {s: rows[f"{B1_CANDIDATE}_minus_{B1_BASELINE}_{s}"]["ci95"][0] > 0 for s in SUBJECTS}
    # binding to the rebuilt delivery
    files = inputs["b1_files"]
    binding = {
        "paired_candidate_mesh_is_the_file": (paired.get("arm_file_sha256") or {}).get(B1_CANDIDATE) == files["D4c_mesh"]
                                             and files["D4c_mesh"] is not None,
        "silhouette_mesh_is_the_same_file": files["silhouette_mesh"] == files["D4c_mesh"],
        "silhouette_read_the_rebuilt_glbs": all(
            (sil.get("input_sha256") or {}).get(f"subject-{s:02d}.glb") == files[f"delivery/subject-{s:02d}.glb"]
            and files[f"delivery/subject-{s:02d}.glb"] is not None for s in (0, 1)),
        "paired_candidate_medians_equal_the_silhouette_run": all(
            ((paired.get("arms") or {}).get(B1_CANDIDATE) or {}).get(s, {}).get(cam)
            == round(float(((sil.get("arms") or {}).get("ours_delivered") or {}).get(cam, {}).get(s, {}).get(
                "iou", {}).get("median", math.nan)), 4)
            for s in SUBJECTS for cam in B1_CAMERAS),
    }
    ours = (sil.get("arms") or {}).get("ours_delivered") or {}
    frozen = (sil.get("arms") or {}).get("control_frozen_pose_tracked") or {}
    frozen_below = {f"{cam}_{s}": bool((frozen.get(cam, {}).get(s, {}).get("iou", {}).get("median", math.inf))
                                       < (ours.get(cam, {}).get(s, {}).get("iou", {}).get("median", -math.inf)))
                    for cam in B1_CAMERAS for s in SUBJECTS}
    mamma_new = (sil.get("arms") or {}).get("ORACLE_mamma_mesh") or {}
    mamma_old = (committed.get("arms") or {}).get("ORACLE_mamma_mesh") or {}
    mamma_same = {f"{cam}_{s}": bool(mamma_new.get(cam, {}).get(s) is not None
                                     and mamma_new.get(cam, {}).get(s) == mamma_old.get(cam, {}).get(s))
                  for cam in B1_CAMERAS for s in SUBJECTS}
    ok = (not problems and all(band.values()) and all(binding.values()) and all(frozen_below.values())
          and all(mamma_same.values()))
    return {"pass": ok, "band_lower_ci_above_zero": band, "binding": binding, "frozen_pose_below": frozen_below,
            "mamma_bit_identical": mamma_same, "pairs": rows, "problems": problems,
            "arms_pooled_median_iou": {a: {s: ((paired.get("arms") or {}).get(a) or {}).get(s, {}).get("pooled_median_iou")
                                           for s in SUBJECTS} for a in arms},
            "REPORTED_D4c_minus_D4": {s: rows[f"{B1_CANDIDATE}_minus_{B1_D4}_{s}"] for s in SUBJECTS}}


def b2_clauses(inputs: dict) -> dict:
    report = inputs["b2-same-denominator.json"]
    subjects = report.get("subjects") or {}
    checks = {s: {k: v for k, v in (subjects.get(s) or {}).items()
                  if k[:1].isdigit() or k == "marker_names_are_the_declared_map"} for s in SUBJECTS}
    expected_keys = {"1a_handed_array_is_byte_identical_to_the_rig_converter_input_on_this_build",
                     "1b_handed_array_is_byte_identical_to_the_rig_BUILD_s_delivered_array",
                     "1c_the_array_is_the_SMOOTHED_repaired_one", "2a_joint_names_and_order_identical_everywhere",
                     "2b_validity_mask_identical", "3_frame_order_identical",
                     "4a_marker_values_match_the_declared_mapping_and_conversion", "4b_occlusion_flags_match",
                     "marker_names_are_the_declared_map"}
    bound = (report.get("delivery") == inputs["b2_paths"]["delivery"]
             and report.get("rig_build") == inputs["b2_paths"]["rig_build"])
    ok = bound and all(set(checks[s]) == expected_keys and all(v is True for v in checks[s].values()) for s in SUBJECTS)
    return {"pass": bool(ok), "bound_to_the_rebuilt_delivery_and_the_hygiene_rig_build": bound, "checks": checks}


def hygiene_clauses(inputs: dict) -> dict:
    hyg = inputs["hygiene"]
    identical = {f: hyg["rebuild"].get(f) is not None and hyg["rebuild"].get(f) == hyg["shipped"].get(f) for f in EIGHT}
    trip = {k: bool(v[0] is not None and v[0] == v[1]) for k, v in inputs["tripwire_files"].items()}
    record = inputs["tripwire_record"]
    normalised = bool(record.get("all_byte_identical_or_equal_after_normalising"))
    diff = source_diff(inputs)
    base_ok = diff.get("base_sha256") == BASE_FITTER_SHA256
    ok = all(identical.values()) and all(trip.values()) and len(trip) == 8 + 54 and normalised and \
        diff["only_the_two_starts"] and base_ok
    return {"pass": bool(ok), "rig_rebuild_identical": identical, "identical_count": sum(identical.values()),
            "tripwire_byte_identical": trip, "tripwire_files_identical": sum(trip.values()),
            "tripwire_json_equal_after_normalising": normalised, "source_diff": diff,
            "base_fitter_is_the_card_s": base_ok}


# ---------------------------------------------------------------------------------------------- build

def build(inputs: dict) -> dict:
    problems: list[str] = []
    try:
        drawn = json.loads(inputs["drawn_set_bytes"])
    except (ValueError, TypeError):
        drawn = {}
        problems.append("drawn set: unreadable")
    inputs = dict(inputs, drawn=drawn)
    pre0 = precondition_0(drawn)
    inputs["pre0"] = pre0
    try:
        dev_json = json.loads(inputs["development_json_bytes"]) if inputs.get("development_json_bytes") else {}
    except ValueError:
        dev_json = {}
    inputs["development_sha256"] = sha256_bytes(inputs.get("development_json_bytes") or b"")
    inputs["chosen_statistic"] = dev_json.get("chosen_trunk_statistic")
    # stage 0b and the development choice, re-derived from the development cells, which ran on the stage-2 fitter:
    # the current fitter with TRUNK_STATISTIC returned to None must hash to the one the cells record
    ran_on = dev_json.get("development_ran_on") or {}
    unset = "\n".join("TRUNK_STATISTIC: str | None = None" if line.startswith("TRUNK_STATISTIC") else line
                      for line in inputs["fitter_source"].split("\n"))
    if sha256_bytes(unset.encode()) != ran_on.get("fitter_sha256"):
        problems.append("development: the development cells' fitter is not the current fitter with the statistic unset")
    dev = development(dict(inputs, chosen_statistic=None, development_ran_on=ran_on))
    dev_ok = (dev["complete"] and dev["chosen_trunk_statistic"] == dev_json.get("chosen_trunk_statistic")
              and dev["stage_0b"] == dev_json.get("stage_0b") and dev["readings"] == dev_json.get("readings")
              and dev["cells_sha256"] == dev_json.get("cells_sha256"))
    if not dev_ok:
        problems.append("development: the committed development JSON is not re-derivable from the development cells")
    stage_0b_holds = dev["complete"] and not dev["stage_0b"]["STOP"]
    # the freeze order
    commits = inputs.get("commits") or {}
    order_ok = bool(commits.get("development_json") and commits.get("acceptance_manifest")
                    and commits["development_json"] != commits["acceptance_manifest"]
                    and commits.get("development_is_ancestor")
                    and commits["development_json_time"] <= commits["acceptance_manifest_time"]
                    and commits.get("development_json_committed_bytes_sha256") == inputs["development_sha256"])
    if not order_ok:
        problems.append("freeze order: the development JSON's commit is not older than the acceptance cells' commit, "
                        "or the development JSON on disk is not the committed one")
    manifest = inputs.get("manifest") or {}
    if manifest.get("cells_sha256") != inputs.get("acceptance_cell_sha256") or not manifest.get("cells_sha256"):
        problems.append("freeze order: the acceptance manifest does not name the acceptance cells by content")
    if trunk_statistic_in_source(inputs["fitter_source"]) != inputs["chosen_statistic"]:
        problems.append("the fitter's TRUNK_STATISTIC is not the development JSON's choice")

    population = fx.POPULATIONS["acceptance"]
    arms = fx.ACCEPTANCE_ARMS
    cells = check_cells(inputs, "acceptance", population, arms, expected_provenance(inputs), problems)
    scored = pre0["scored"]
    rows = fixture_rows(cells, population, arms, scored)

    # closure on every candidate cell, bound to the measured files by content
    closure_rows, closure_problems = {}, []
    pairs = (inputs.get("closure") or {}).get("pairs") or {}
    files = inputs["acceptance"]["files"]
    for s, d in population:
        tag = f"{s}_d{d}"
        cell = cells.get((s, d, "candidate"))
        pair = pairs.get(tag)
        if cell is None or pair is None:
            closure_problems.append(f"closure {tag}: missing")
            continue
        record = cell["record"]
        ok = (pair.get("glb_sha256") == record.get("glb_sha256") == files.get(record.get("glb"))
              and pair.get("track_sha256") == record.get("track_sha256") == files.get(record.get("track"))
              and files.get(record.get("glb")) is not None and files.get(record.get("track")) is not None
              and pair.get("frames_in_glb") == FRAMES and pair.get("frames_in_track") == FRAMES
              and pair.get("joints_missing_from_the_glb") == [] and (pair.get("joints_compared") or 0) >= 127)
        value = pair.get("max_abs_m")
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        closure_rows[tag] = {"max_abs_m": value, "bound_by_content": ok, "within": bool(ok and finite and value <= CLOSURE_BAND_M)}
        if not ok:
            closure_problems.append(f"closure {tag}: not this cell's GLB and track by content, or not a full read")

    population_ok = not problems and len(rows) == len(population)
    exact_pooled = {k: v["pooled_mm"]["exact_identity"] for k, v in rows.items()}
    validity = population_ok and all(v <= VALIDITY_CEILING_MM for v in exact_pooled.values())
    l_fixture = {k: d4bg.l_passes(v["segments"]["candidate"]) for k, v in rows.items()}
    mean_misses = {k: d4bg.misses_l(v["segments"]["mean_body"]) for k, v in rows.items()}
    spine_fails = {k: trunk_fails_as_scored(v) for k, v in rows.items()}
    exact_zero = {k: max(r["error_mm"] for r in v["segments"]["exact_identity"].values()) <= EXACT_ZERO_MM
                  and d4bg.l_passes(v["segments"]["exact_identity"]) for k, v in rows.items()}
    init_misses = {k: d4bg.misses_l(v["segments"]["init_only"]) for k, v in rows.items()}
    closure_ok = len(closure_rows) == len(population) and all(r["within"] for r in closure_rows.values()) \
        and not closure_problems
    b1, b2, hyg = b1_clauses(inputs), b2_clauses(inputs), hygiene_clauses(inputs)
    conjuncts = {
        "precondition_0": pre0["holds"],
        "stage_0b": stage_0b_holds,
        "validity": validity,
        "L": population_ok and all(l_fixture.values()),
        "closure": population_ok and closure_ok,
        "must_fail_i_mean_body_misses_L": population_ok and all(mean_misses.values()),
        "must_fail_ii_spine_displaced_fails_L_at_the_trunk": population_ok and all(spine_fails.values()),
        "must_fail_iii_exact_identity_reads_zero_and_passes": population_ok and all(exact_zero.values()),
        "must_fail_iv_init_only_misses_L": population_ok and all(init_misses.values()),
        "B1": b1["pass"],
        "B2": b2["pass"],
        "hygiene": hyg["pass"],
    }
    if not pre0["holds"]:
        verdict, reason = "STOP", "precondition 0: the spine is not drawn under the limit-aware rule"
    elif not stage_0b_holds:
        verdict, reason = "STOP", "stage 0b: the spine control passes L as scored on a development fixture"
    elif not population_ok:
        verdict, reason = "FAIL", "population, provenance or construction"
    elif not validity:
        verdict, reason = "INVALID", "the exact_identity floor exceeds 1.0 mm on a fixture"
    elif not all(init_misses.values()):
        verdict, reason = "STOP", "init-only passes L on an acceptance fixture: the oracle cannot score the calibration"
    elif all(conjuncts.values()):
        verdict, reason = "PASS", "every conjunct holds"
    else:
        verdict, reason = "FAIL", "failed: " + ", ".join(k for k, v in conjuncts.items() if not v)
    consequence = {
        "PASS": "D4's O1 is superseded by D4c's fitter with B1 re-shown on it; D4 and D4c close together; --body mhr "
                "carries the new start; the default stays rig",
        "FAIL": "tooling and records merge; the fit_one change stays on the branch; D4 stays open",
        "INVALID": "D4 stays open; INVALID never licenses replacing a fixture",
        "STOP": "D4 stays open; the step stops where the card says; the fit_one change stays on the branch",
    }[verdict]

    # REPORTED, never banded
    def per_fixture(fn):
        return {k: fn(v) for k, v in rows.items()}

    intervals = {}
    for segment in scored:
        lows = [v["segments"]["candidate"][segment]["truth_mm"] - v["segments"]["candidate"][segment]["tolerance_mm"]
                for v in rows.values()]
        highs = [v["segments"]["candidate"][segment]["truth_mm"] + v["segments"]["candidate"][segment]["tolerance_mm"]
                 for v in rows.values()]
        intervals[segment] = {"max_low_mm": max(lows) if lows else None, "min_high_mm": min(highs) if highs else None,
                              "empty": bool(lows and max(lows) > min(highs))}
    caps = {}
    for s, d in population:
        for a in ("candidate", "legacy"):
            c = cells.get((s, d, a))
            if c is not None:
                caps.setdefault(a, {})[f"{s}_d{d}"] = [
                    {k: call.get(k) for k in ("stage", "solves", "stopped_below_the_cap", "stopped_at_the_configured_cap",
                                              "stopped_above_the_configured_cap", "max_index")}
                    for call in c["record"]["calibration_iterations"]["per_call"]]
    starts = {f"{s}_d{d}": {"start": dict(zip(cells[(s, d, "candidate")]["names"],
                                               cells[(s, d, "candidate")]["record"]["start"]["start_identity"])),
                            "truth": {n: float(cells[(s, d, "candidate")]["truth_id"][i])
                                      for i, n in enumerate(cells[(s, d, "candidate")]["names"])}}
              for s, d in population if (s, d, "candidate") in cells}
    start_error = {k: {c: v["start"][c] - v["truth"][c] for c in pre0["drawn_set"]} for k, v in starts.items()}
    reported = {
        "legacy_zero_start_before_arm": per_fixture(lambda v: {
            "L_passes": d4bg.l_passes(v["segments"]["legacy"]), "trunk_L_ratio": trunk_ratio(v, "legacy"),
            "spine_error": v["identity_error"]["legacy"]["scale_spine_length"]}),
        "candidate_trunk_L_ratio": per_fixture(lambda v: trunk_ratio(v, "candidate")),
        "init_only_trunk_L_ratio": per_fixture(lambda v: trunk_ratio(v, "init_only")),
        "worst_scored_segment_ratio": {a: per_fixture(lambda v, a=a: max(
            r["error_mm"] / r["tolerance_mm"] for r in v["segments"][a].values() if r["scored"])) for a in arms},
        "J_candidate_minus_paired_floor_mm": per_fixture(lambda v: None),
        "per_channel_start_minus_truth": start_error,
        "per_channel_recovered_error_candidate": per_fixture(lambda v: {c: v["identity_error"]["candidate"][c]
                                                                        for c in pre0["drawn_set"]}),
        "calibration_cap_exhaustion_per_stage": caps,
        "empty_intersection_of_allowable_rest_length_intervals": intervals,
        "pooled_statistic_mm_per_arm": per_fixture(lambda v: v["pooled_mm"]),
        "segments_all_arms": per_fixture(lambda v: v["segments"]),
        "floors_mm": per_fixture(lambda v: v["floor_mm"]),
        "development_readings": dev["readings"],
        "what_a_PASS_does_not_establish": [
            "the full identity vector: compensating channels can hold the scored lengths with a wrong parameter vector",
            "the eye channels and every undrawn channel: they change anatomy outside the scored segments",
            "the landmark-to-joint convention: every fixture is exact, noiseless, with pinned zero offsets (lane H)"],
    }
    for k in rows:
        s, d = k.split("_d")
        c, e = cells[(int(s), int(d), "candidate")], cells[(int(s), int(d), "exact_identity")]
        reported["J_candidate_minus_paired_floor_mm"][k] = {
            j: float(np.median(c["distance"][:, i]) - np.median(e["distance"][:, i])) for i, j in enumerate(MAPPED_JOINTS)}
    clauses = {
        "precondition_0": {"predicted": "eight drawn, including scale_spine_length and scale_shoulder_width",
                           "measured": {"drawn": pre0["drawn_set"], "count": len(pre0["drawn_set"]),
                                        "scored_segments": scored, "solver_failures": pre0["solver_failures"]},
                           "verdict": "HOLDS" if pre0["holds"] else "STOP",
                           "prediction": "HELD" if len(pre0["drawn_set"]) == 8 and pre0["spine_drawn"]
                           and "scale_shoulder_width" in pre0["drawn_set"] else "FAILED"},
        "stage_0b": {"predicted": "the spine control fails L as scored on all 18 development fixtures",
                     "measured": {"fixtures_failing": dev["stage_0b"]["fixtures_failing"],
                                  "fixtures": dev["stage_0b"]["fixtures"]},
                     "verdict": "HOLDS" if stage_0b_holds else "STOP",
                     "prediction": "HELD" if stage_0b_holds else "FAILED"},
        "validity": {"predicted": "exact_identity pooled <= 1.0 mm on every fixture",
                     "measured": {"per_fixture_mm": exact_pooled, "max_mm": max(exact_pooled.values(), default=None)},
                     "verdict": "PASS" if validity else ("INVALID" if population_ok else "FAIL"),
                     "prediction": "HELD" if validity else "FAILED"},
        "L": {"predicted": "PASS: the trunk within its paired tolerance on most or all fixtures; limbs PASS",
              "measured": {"fixtures_passing": sum(l_fixture.values()), "fixtures": len(l_fixture),
                           "trunk_within": sum(v["segments"]["candidate"]["trunk"]["within"] for v in rows.values()),
                           "worst_trunk_ratio": max((trunk_ratio(v, "candidate") for v in rows.values()), default=None),
                           "limbs_all_within": all(v["segments"]["candidate"][s_]["within"] for v in rows.values()
                                                   for s_ in scored if s_ not in ("trunk", "neck_head")),
                           "segments_failing": sorted({s_ for v in rows.values() for s_, r in v["segments"]["candidate"].items()
                                                       if r["scored"] and not r["within"]})},
              "verdict": "PASS" if conjuncts["L"] else "FAIL", "prediction": "HELD" if conjuncts["L"] else "FAILED"},
        "closure": {"predicted": "<= 1e-4 m on every candidate cell, bound by content",
                    "measured": {"rows": closure_rows, "problems": closure_problems},
                    "verdict": "PASS" if conjuncts["closure"] else "FAIL",
                    "prediction": "HELD" if conjuncts["closure"] else "FAILED"},
        "must_fail_i": {"predicted": "the mean body misses L on every fixture",
                        "measured": {"fixtures_missing": sum(mean_misses.values()), "fixtures": len(mean_misses)},
                        "verdict": "PASS" if conjuncts["must_fail_i_mean_body_misses_L"] else "FAIL",
                        "prediction": "HELD" if conjuncts["must_fail_i_mean_body_misses_L"] else "FAILED"},
        "must_fail_ii": {"predicted": "the spine displaced 0.149 misses L at the trunk (scored) on every fixture",
                         "measured": {"fixtures_failing_at_the_trunk": sum(spine_fails.values()), "fixtures": len(spine_fails),
                                      "trunk_error_mm": per_fixture(lambda v: v["segments"]["spine_displaced"]["trunk"]["error_mm"]),
                                      "trunk_tolerance_mm": per_fixture(lambda v: v["segments"]["spine_displaced"]["trunk"]["tolerance_mm"])},
                         "verdict": "PASS" if conjuncts["must_fail_ii_spine_displaced_fails_L_at_the_trunk"] else "FAIL",
                         "prediction": "HELD" if conjuncts["must_fail_ii_spine_displaced_fails_L_at_the_trunk"] else "FAILED"},
        "must_fail_iii": {"predicted": "exact identity reads L = 0 and PASSES on every fixture",
                          "measured": {"fixtures": sum(exact_zero.values())},
                          "verdict": "PASS" if conjuncts["must_fail_iii_exact_identity_reads_zero_and_passes"] else "FAIL",
                          "prediction": "HELD" if conjuncts["must_fail_iii_exact_identity_reads_zero_and_passes"] else "FAILED"},
        "must_fail_iv": {"predicted": "init-only misses L at the trunk on every fixture (STOP if it passes on any)",
                         "measured": {"fixtures_missing": sum(init_misses.values()), "fixtures": len(init_misses),
                                      "trunk_misses": sum(not v["segments"]["init_only"]["trunk"]["within"] for v in rows.values()),
                                      "worst_trunk_ratio": max((trunk_ratio(v, "init_only") for v in rows.values()), default=None)},
                         "verdict": "PASS" if conjuncts["must_fail_iv_init_only_misses_L"] else "STOP",
                         "prediction": "HELD" if conjuncts["must_fail_iv_init_only_misses_L"] and all(
                             not v["segments"]["init_only"]["trunk"]["within"] for v in rows.values()) else "FAILED"},
        "B1": {"predicted": "D4c - D7c lower CI > 0 on both performers (D4 read +0.156 / +0.115); D4c - D4 within +-0.01",
               "measured": {"pairs": b1["pairs"], "binding": b1["binding"], "problems": b1["problems"],
                            "frozen_pose_below": b1["frozen_pose_below"], "mamma_bit_identical": b1["mamma_bit_identical"],
                            "arms_pooled_median_iou": b1["arms_pooled_median_iou"]},
               "verdict": "PASS" if b1["pass"] else "FAIL",
               "prediction": "HELD" if b1["pass"] and all(
                   abs(r["median"] or math.inf) <= 0.01 for r in b1["REPORTED_D4c_minus_D4"].values()) else "FAILED"},
        "B2": {"predicted": "the consumed array is the rig converter's input; markers re-derive exactly",
               "measured": b2, "verdict": "PASS" if b2["pass"] else "FAIL",
               "prediction": "HELD" if b2["pass"] else "FAILED"},
        "hygiene": {"predicted": "--body rig 8/8; the zero-start tripwire reproduces D4's delivery and O1 cells; the "
                                 "source diff is the two starts only",
                    "measured": {k: hyg[k] for k in ("identical_count", "tripwire_files_identical",
                                                     "tripwire_json_equal_after_normalising", "source_diff",
                                                     "base_fitter_is_the_card_s")},
                    "verdict": "PASS" if hyg["pass"] else "FAIL", "prediction": "HELD" if hyg["pass"] else "FAILED"},
    }
    clauses["overall"] = {"predicted": "PASS", "measured": verdict, "verdict": verdict,
                          "prediction": "HELD" if verdict == "PASS" else "FAILED"}
    return {"step": "D4c", "verdict": verdict, "reason": reason, "consequence": consequence,
            "conjuncts": conjuncts, "population_ok": population_ok, "problems": problems + closure_problems,
            "population": {"fixtures": [f"{s}_d{d}" for s, d in population], "arms": list(arms), "frames": FRAMES,
                           "joints": len(MAPPED_JOINTS), "cells_expected": len(population) * len(arms),
                           "cells_valid": len(cells)},
            "chosen_trunk_statistic": inputs["chosen_statistic"], "drawn_set": pre0["drawn_set"],
            "segments_scored": scored, "development": {k: dev[k] for k in ("stage_0b", "chosen_trunk_statistic", "complete")},
            "clauses": clauses, "B1_REPORTED_D4c_minus_D4": b1["REPORTED_D4c_minus_D4"], "reported": reported}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development", action="store_true",
                        help="stage 3: stage 0b and the development choice, written as the frozen JSON")
    parser.add_argument("--manifest", action="store_true",
                        help="stage 4: name the acceptance cells by content, committed after the development JSON")
    arguments = parser.parse_args()
    if arguments.manifest:
        cells = sorted(ACC_CELLS.glob("cell-*.json"))
        record = {"step": "D4c", "stage": 4,
                  "what": "the acceptance cells, named by content; this file's commit must be younger than the "
                          "development JSON's (the gate checks the git order)",
                  "development_json_sha256": sha256_file(DEVELOPMENT),
                  "cells_sha256": {c.name: sha256_file(c) for c in cells},
                  "cells": len(cells)}
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print("manifest:", len(cells), "cells")
        return 0
    if arguments.development:
        inputs = load_inputs(development_only=True)
        drawn = json.loads(inputs["drawn_set_bytes"])
        inputs.update(drawn=drawn, pre0=precondition_0(drawn), chosen_statistic=None)
        if not inputs["pre0"]["holds"]:
            raise SystemExit("precondition 0 does not hold: STOP, no development runs")
        record = development(inputs)
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print("stage 0b:", record["stage_0b"]["fixtures_failing"], "of", record["stage_0b"]["fixtures"],
              "fail L at the trunk as scored; STOP:", record["stage_0b"]["STOP"])
        for arm, row in record["readings"].items():
            print(f"  {arm:18s} worst trunk ratio {row['worst_trunk_L_ratio']:.4f}  trunk within "
                  f"{row['fixtures_with_the_trunk_within']}/18  L passes {row['fixtures_passing_L']}/18")
        for problem in record["problems"][:20]:
            print("  PROBLEM:", problem)
        print("chosen trunk statistic:", record["chosen_trunk_statistic"])
        return 0
    report = build(load_inputs())
    arguments.out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for name, clause in report["clauses"].items():
        print(f"  {name:16s} {clause['verdict']:8s} prediction {clause['prediction']}")
    for problem in report["problems"][:30]:
        print("  PROBLEM:", problem)
    print(f"VERDICT: {report['verdict']} ({report['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
