"""D4c the calibration's start: the fixtures and their arms, ONE PROCESS PER CELL.

The card is the "D4c the calibration's start" row of `docs/LADDER_EXECUTION_PLAN.md` §2. This file builds the
fixtures and runs the arms; it scores nothing. L, the tolerances, stage 0b, the development choice, the must-fails
and the ONE verdict are computed by `tools/compare/d4c_start_gate.py` on `.venv` from what each cell writes here.

Populations:
  DEVELOPMENT (burned, 18): D4's six (`--population d4`, D4's own generator, donor 0, all ten channels drawn) and
      D4b's twelve (`--population d4b`, seeds 20261001-06 x donors 0/1, D4b's frozen drawn set), both rebuilt by
      `d4b_o1_fixture.build_fixture` (imported, never copied).
  ACCEPTANCE (12): seeds 20261101-06 x donors 0/1 (`--population acceptance`), the identity drawn uniform within
      its configured limit on D4c's FROZEN drawn set (`docs/reviews/body-model-start-records/drawn-set.json`),
      every other channel zero, the generator `default_rng([seed, donor])`, poses clamped to the configured
      limits by D4's own `truth_motion`. The draw is a pure function of (seed, donor, drawn set, limits)
      (`acceptance_draw`), so the gate regenerates it without momentum.

Arms (momentum settings unchanged in every arm: calib_frames 100, loss_alpha 2.0, max_iter 30, offsets pinned at
limit_weight 10, the 26 `*_flexible` channels frozen -- all inside `mhr_delivery.fit_one`):

    candidate          the landmark start (the frozen trunk statistic), both calibration stages from it
    candidate_median   development only: the landmark start with the trunk MEDIAN
    candidate_p90      development only: ... the trunk p90
    candidate_p95      development only: ... the trunk p95
    legacy             the zero start (the D4 fitter, byte for byte)                 (REPORTED, the before arm)
    exact_identity     the truth identity handed in, pose re-solved                   (the paired floor; must-fail iii)
    mean_body          the identity held at zero, pose re-solved                      (must-fail i)
    spine_displaced    the truth, scale_spine_length 0.149 toward zero, pose re-solved (must-fail ii; stage 0b)
    init_only          the landmark start HELD, no calibration at all, pose re-solved (must-fail iv)

The landmark start is `mhr_delivery.landmark_start` -- the fitter's own function -- on the consumed landmark array
and the 178-parameter character the fixture fits with (loaded before any calibration). What L reads is the REST:
forward kinematics at MHR's zero pose with the identity RETAINED (never the track's `rest_positions_z_up_m`).

Calibrating arms capture momentum's own debug log (the calibration config's `debug` only, D4b's proxy, proved
inert by D4b's tripwire) for the cap-exhaustion counts, per stage (A, B) and per solver cap.

    /tmp/momenv/bin/python tools/fitter/d4c_fixture.py --population P --seed S --donor D --arm A --out DIR
    /tmp/momenv/bin/python tools/fitter/d4c_fixture.py --drive --population P --arms ... --out DIR
    /tmp/momenv/bin/python tools/fitter/d4c_fixture.py --tripwire-compare --out JSON
    /tmp/momenv/bin/python tools/fitter/d4c_fixture.py --calibration-frames --out JSON
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
LOD = 2
FRAMES = 150
SCHEMA = "d4c-start-cell/1"
RECORDS = ROOT / "docs/reviews/body-model-start-records"
DRAWN_SET = RECORDS / "drawn-set.json"
DEVELOPMENT = RECORDS / "development.json"
D4B_DRAWN_SET = ROOT / "docs/reviews/body-model-o1-records/drawn-set.json"
SETTINGS = {"calib_frames": 100, "loss_alpha": 2.0, "max_iter": 30, "smoothing": 0.0,
            "locator_limit_weight": 10.0, "freeze_flexible": True}

D4_SEEDS = (20260922, 20260923, 20260924, 20260925, 20260926, 20260927)
D4B_SEEDS = (20261001, 20261002, 20261003, 20261004, 20261005, 20261006)
ACCEPTANCE_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105, 20261106)
DONORS = (0, 1)
POPULATIONS = {
    "d4": [(s, 0) for s in D4_SEEDS],
    "d4b": [(s, d) for d in DONORS for s in D4B_SEEDS],
    "acceptance": [(s, d) for d in DONORS for s in ACCEPTANCE_SEEDS],
}
STATISTICS = ("median", "p90", "p95")
DEVELOPMENT_ARMS = ("legacy", "candidate_median", "candidate_p90", "candidate_p95", "exact_identity",
                    "spine_displaced")
ACCEPTANCE_ARMS = ("candidate", "exact_identity", "mean_body", "spine_displaced", "init_only", "legacy")
CALIBRATING_ARMS = ("candidate", "candidate_median", "candidate_p90", "candidate_p95", "legacy")
SPINE_DISPLACEMENT = 0.149


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def displaced_spine(value: float) -> float:
    """Must-fail (ii): 0.149 toward zero, THROUGH zero if the draw is smaller; -0.149 from an exact zero."""
    direction = -np.sign(value) if value != 0.0 else -1.0
    return float(value + direction * SPINE_DISPLACEMENT)


def acceptance_draw(seed: int, donor: int, drawn: list[str], limits: dict, channels: tuple) -> dict[str, float]:
    """The acceptance identity draw, momentum-free: `d4.truth_motion`'s own loop (one `rng.uniform` per named
    channel, in CHANNELS order, the undrawn collapsed to the point [0, 0]) under `default_rng([seed, donor])`."""
    rng = np.random.default_rng([seed, donor])
    out = {}
    for channel in channels:
        low, high = limits[channel] if channel in drawn else (0.0, 0.0)
        out[channel] = float(rng.uniform(low, high))
    return out


def frozen_statistic() -> str | None:
    if not DEVELOPMENT.exists():
        return None
    return json.loads(DEVELOPMENT.read_text(encoding="utf-8"))["chosen_trunk_statistic"]


# --------------------------------------------------------------------------------------------- one cell

def one_cell(population: str, seed: int, donor: int, arm: str, out: Path) -> dict:
    import pymomentum.geometry as g

    import d4_o1_exactness as d4
    import d4b_o1_fixture as d4bf
    import mhr_delivery as md
    from d4b_identifiability import CHANNELS, MAPPED_JOINTS

    if tuple(md.MAP.values()) != MAPPED_JOINTS:
        raise SystemExit("the mapped joints are not mhr_delivery.MAP's")
    if (seed, donor) not in POPULATIONS[population]:
        raise SystemExit(f"{seed}/d{donor} is not in the {population} population")
    arms = ACCEPTANCE_ARMS if population == "acceptance" else DEVELOPMENT_ARMS
    if arm not in arms:
        raise SystemExit(f"{arm} is not an arm of the {population} population")
    if population == "acceptance" and not DEVELOPMENT.exists():
        raise SystemExit("the development JSON is not frozen: no acceptance fixture may exist before it")
    # Every character this process needs is loaded BEFORE any calibration (mhr_delivery.export_glb).
    truth_character = md.load_character(LOD)
    export_character = md.load_character(LOD, drop_flexible=True)
    limits = d4.configured_limits()
    drawn_record = json.loads(DRAWN_SET.read_text(encoding="utf-8"))
    if population == "acceptance":
        drawn = list(drawn_record["drawn_set"])
        draw_limits = dict(limits)
        for channel in CHANNELS:
            if channel not in drawn:
                draw_limits[channel] = (0.0, 0.0)
        d4.DONOR = ROOT / d4bf.DONOR_FILES[donor]
        motion, drawn_identity, repair = d4.truth_motion(truth_character, np.random.default_rng([seed, donor]),
                                                         draw_limits)
        regenerated = acceptance_draw(seed, donor, drawn, limits, CHANNELS)
        if any(np.float32(regenerated[c]) != np.float32(drawn_identity[c]) for c in CHANNELS):
            raise SystemExit("acceptance_draw does not reproduce truth_motion's draw")
        fixture = {"generator": f"numpy default_rng([{seed}, {donor}]) (seeded by the pair)",
                   "drawn_channels": drawn, "drawn_identity": drawn_identity,
                   "clamped_parameter_frames": repair["clamped_parameter_frames"],
                   "truth_motion_sha256": array_sha256(motion)}
    else:
        motion, fixture = d4bf.build_fixture(truth_character, seed, donor, burned=(population == "d4"))
    full_names = list(truth_character.parameter_transform.names)
    identity_channels = d4bf.scale_names(export_character)
    truth_identity = {n: float(motion[0, full_names.index(n)]) for n in identity_channels}
    truth_cm = d4.mapped_positions_cm(truth_character, motion)
    # allow_pickle: our own build output (object arrays; CLAUDE.md), never a third-party file
    joint_names = [str(n) for n in np.load(d4.JOINT_NAMES_FROM, allow_pickle=True)["joint_names"]]
    array = d4.landmark_array(truth_cm, joint_names)

    truth_start = d4bf.identity_vector(export_character, truth_identity)
    fixed, mean_body, start, statistic, start_record = None, False, None, None, None
    arm_note = {}
    if arm in ("candidate", "init_only"):
        statistic = frozen_statistic()
        if statistic is None:
            raise SystemExit("the trunk statistic is not frozen")
        if md.TRUNK_STATISTIC != statistic:
            raise SystemExit("the fitter's TRUNK_STATISTIC is not the development JSON's choice")
    elif arm.startswith("candidate_"):
        statistic = arm.split("_", 1)[1]
    if statistic is not None:
        start, start_record = md.landmark_start(array, joint_names, export_character, trunk_statistic=statistic)
    if arm == "init_only":
        fixed, start = start.copy(), None
    elif arm == "exact_identity":
        fixed = truth_start.copy()
    elif arm == "mean_body":
        mean_body = True
    elif arm == "spine_displaced":
        displaced = dict(truth_identity)
        displaced["scale_spine_length"] = displaced_spine(truth_identity["scale_spine_length"])
        fixed = d4bf.identity_vector(export_character, displaced)
        arm_note = {"spine_truth": truth_identity["scale_spine_length"],
                    "spine_displaced_to": float(np.float32(displaced["scale_spine_length"]))}

    capture = arm in CALIBRATING_ARMS
    with d4bf.calibration_proxy(start=None, capture=capture) as proxy:
        fit = md.fit_one(array, joint_names, lod=LOD, mean_body=mean_body, free_offsets=False,
                         freeze_flexible=True, fixed_identity=fixed, start_identity=start)
    if list(fit["parameter_names"]) != list(export_character.parameter_transform.names):
        raise SystemExit("the fitted character is not the export character's layout")
    fitted_identity = {n: float(fit["identity"][fit["parameter_names"].index(n)]) for n in identity_channels}
    skeleton = list(fit["character"].skeleton.joint_names)
    rows = [skeleton.index(j) for j in MAPPED_JOINTS]
    fitted_cm = fit["positions_cm"][:, rows]
    distance_mm = np.linalg.norm(fitted_cm - truth_cm, axis=2) * 10.0
    truth_rest = d4bf.rest_mapped_cm(export_character, truth_identity)
    fitted_rest = d4bf.rest_mapped_cm(export_character, fitted_identity)
    truth_rest_full = d4bf.rest_mapped_cm(truth_character, truth_identity)
    start_vector = None
    if statistic is not None:
        vector = (fixed if arm == "init_only" else start)
        start_vector = [float(vector[fit["parameter_names"].index(n)]) for n in identity_channels]

    tag = f"{seed}-d{donor}-{arm}"
    arrays_path = out / f"fixture-{tag}.arrays.npz"
    np.savez(arrays_path, truth_motion=motion, truth_mapped_cm=truth_cm, consumed_landmarks_z_up_m=array,
             consumed_joint_names=np.array(joint_names), fitted_mapped_cm=fitted_cm,
             identity=np.asarray(fit["identity"]), motion=fit["motion"])
    record = {
        "schema": SCHEMA, "population": population, "seed": seed, "donor": donor, "arm": arm, "lod": LOD,
        "provenance": provenance(),
        "fixture": fixture,
        "settings": dict(SETTINGS, calibration_debug_captured=capture),
        "start": {"trunk_statistic": statistic, "start_identity": start_vector, "record": start_record,
                  "held_without_calibration": arm == "init_only",
                  "zero_start": arm == "legacy"},
        "arm_note": arm_note,
        "calibrate_markers_calls": proxy.calls,
        "mapped_joints": list(MAPPED_JOINTS),
        "frames": int(distance_mm.shape[0]),
        "identity_channel_names": identity_channels,
        "truth_identity": [truth_identity[n] for n in identity_channels],
        "fitted_identity": [fitted_identity[n] for n in identity_channels],
        "truth_rest_mapped_cm": truth_rest.tolist(),
        "fitted_rest_mapped_cm": fitted_rest.tolist(),
        "truth_rest_full_vs_simplified_character_max_abs_cm": float(np.abs(truth_rest - truth_rest_full).max()),
        "distance_mm": distance_mm.round(6).tolist(),
        "locator_offset_mm_max": float(max(np.linalg.norm(np.asarray(l.offset, np.float64)) * 10.0
                                           for l in fit["character"].locators)),
        "arrays": arrays_path.name,
        "arrays_sha256": sha256(arrays_path),
    }
    if capture:
        record["calibration_iterations"] = cap_counts(proxy.logs, SETTINGS["max_iter"])
        (out / f"fixture-{tag}.calibration-debug.log").write_text("\n=====CALL=====\n".join(proxy.logs),
                                                                 encoding="utf-8")
    if arm == "candidate":
        prefix = out / f"fixture-{tag}"
        md.export_glb(fit, export_character, prefix.with_suffix(".glb"))
        md.write_track(fit, prefix, subject=0,
                       consumed={"ticks": np.arange(array.shape[0]),
                                 "triangulated_world_positions_z_up_m": array,
                                 "raw_triangulated_world_positions_z_up_m": array},
                       lod=LOD, landmarks="d4c_start_synthetic_truth", settings=SETTINGS,
                       joint_names=joint_names)
        record["glb"] = prefix.with_suffix(".glb").name
        record["track"] = prefix.with_suffix(".body-track.npz").name
        record["glb_sha256"] = sha256(prefix.with_suffix(".glb"))
        record["track_sha256"] = sha256(prefix.with_suffix(".body-track.npz"))
    return record


def cap_counts(logs: list[str], max_iter: int) -> dict:
    """Momentum's debug log -> per calibrate_markers call (stage A, stage B), every solve's last iteration index,
    counted by where it stopped: below max_iter - 1 (the solver's own criterion), exactly max_iter - 1 (the
    configured cap), above it (a solver with its own, larger cap -- momentum's inner sequence solver)."""
    import d4b_o1_fixture as d4bf

    out = {"max_iter": max_iter, "calls": len(logs), "per_call": []}
    for stage, log in zip(("A_locators_only", "B_identity"), logs):
        last = d4bf.iteration_segments(log)
        hist: dict[int, int] = {}
        for k in last:
            hist[k] = hist.get(k, 0) + 1
        out["per_call"].append({
            "stage": stage, "solves": len(last),
            "stopped_below_the_cap": sum(1 for k in last if k < max_iter - 1),
            "stopped_at_the_configured_cap": sum(1 for k in last if k == max_iter - 1),
            "stopped_above_the_configured_cap": sum(1 for k in last if k > max_iter - 1),
            "max_index": max(last) if last else None,
            "last_index_histogram": {str(k): v for k, v in sorted(hist.items())},
        })
    return out


def provenance() -> dict:
    import d4_o1_exactness as d4
    import d4b_o1_fixture as d4bf
    import mhr_delivery as md

    return {
        "fitter_sha256": sha256(Path(md.__file__)),
        "d4_fixture_sha256": sha256(Path(d4.__file__)),
        "d4b_fixture_sha256": sha256(Path(d4bf.__file__)),
        "this_file_sha256": sha256(Path(__file__)),
        "donor_sha256": {str(d): sha256(ROOT / d4bf.DONOR_FILES[d]) for d in DONORS},
        "model_sha256": sha256(md.ASSETS / "compact_v6_1.model"),
        "fbx_sha256": sha256(md.ASSETS / f"lod{LOD}.fbx"),
        "drawn_set_sha256": sha256(DRAWN_SET),
        "d4b_drawn_set_sha256": sha256(D4B_DRAWN_SET),
        "development_json_sha256": sha256(DEVELOPMENT) if DEVELOPMENT.exists() else None,
        "pymomentum": importlib.metadata.version("pymomentum-cpu"),
    }


# ------------------------------------------------------------------------------------------ the tripwire

def tripwire_compare() -> dict:
    """Stage 2: the zero start through the NEW code against D4's retained outputs, file by file.

    Byte-identical: every GLB, body-track.npz, markers.npz, truth.npz and fit report. The JSONs that carry a
    worktree-rooted path are compared field by field after normalising exactly the named path fields
    (`body_model.assets` in a body-track JSON; `glb`, `track` and `fitter_source_sha256` in an O1 cell).
    """
    out = {"delivery": {}, "o1": {}, "normalised_fields": {
        "*.body-track.json": ["body_model.assets"],
        "o1 cell-*.json": ["glb", "track", "fitter_source_sha256"]}}
    d4_delivery, new_delivery = ROOT / "artifacts/compare/d4-body/delivery", \
        ROOT / "artifacts/compare/d4c-start/tripwire/delivery"
    for s in ("00", "01"):
        for name in (f"subject-{s}.glb", f"subject-{s}.body-track.npz", f"subject-{s}.markers.npz",
                     f"fit-report-subject-{s}.json"):
            out["delivery"][name] = {"d4": sha256(d4_delivery / name), "d4c_zero_start": sha256(new_delivery / name)}
            out["delivery"][name]["byte_identical"] = out["delivery"][name]["d4"] == out["delivery"][name]["d4c_zero_start"]
        name = f"subject-{s}.body-track.json"
        a = json.loads((d4_delivery / name).read_text(encoding="utf-8"))
        b = json.loads((new_delivery / name).read_text(encoding="utf-8"))
        a["body_model"]["assets"] = b["body_model"]["assets"] = "<normalised>"
        out["delivery"][name] = {"identical_after_normalising": a == b}
    d4_o1, new_o1 = ROOT / "artifacts/compare/d4-body/o1", ROOT / "artifacts/compare/d4c-start/tripwire/o1"
    for seed in D4_SEEDS:
        for arm in ("oracle", "exact_identity", "mean_body"):
            row = {}
            for suffix in (".glb", ".body-track.npz", ".truth.npz"):
                name = f"seed-{seed}-{arm}{suffix}"
                row[name] = sha256(d4_o1 / name) == sha256(new_o1 / name)
            name = f"seed-{seed}-{arm}.body-track.json"
            a = json.loads((d4_o1 / name).read_text(encoding="utf-8"))
            b = json.loads((new_o1 / name).read_text(encoding="utf-8"))
            a["body_model"]["assets"] = b["body_model"]["assets"] = "<normalised>"
            row[name + " (normalised)"] = a == b
            name = f"cell-{seed}-{arm}.json"
            a = json.loads((d4_o1 / name).read_text(encoding="utf-8"))
            b = json.loads((new_o1 / name).read_text(encoding="utf-8"))
            for key in ("glb", "track", "fitter_source_sha256"):
                a[key] = b[key] = "<normalised>"
            row[name + " (normalised)"] = a == b
            out["o1"][f"{seed}_{arm}"] = row
    out["all_byte_identical_or_equal_after_normalising"] = (
        all(v.get("byte_identical", v.get("identical_after_normalising")) for v in out["delivery"].values())
        and all(all(r.values()) for r in out["o1"].values()))
    out["fitter_sha256_now"] = sha256(ROOT / "tools/fitter/mhr_delivery.py")
    return out


def calibration_frames_probe() -> dict:
    """Stage 2, before any development reading: momentum's OWN calibration-frame selection, read from
    `calibrate_markers`' returned indices on a burned fixture's landmarks (the zero start: the D4 fitter's own
    calibration, so nothing about any start is read), against `mhr_delivery.calibration_frame_indices`."""
    import pymomentum.marker_tracking as mt

    import d4_o1_exactness as d4
    import mhr_delivery as md

    truth_character = md.load_character(LOD)
    motion, _, _ = d4.truth_motion(truth_character, np.random.default_rng(D4_SEEDS[0]), d4.configured_limits())
    truth_cm = d4.mapped_positions_cm(truth_character, motion)
    # allow_pickle: our own build output (object arrays; CLAUDE.md), never a third-party file
    joint_names = [str(n) for n in np.load(d4.JOINT_NAMES_FROM, allow_pickle=True)["joint_names"]]
    array = d4.landmark_array(truth_cm, joint_names)
    character = md.with_pinned_locators(md.load_character(LOD, drop_flexible=True), free=False)
    markers = md.marker_data(md.to_mhr_cm(array), joint_names)
    config = mt.CalibrationConfig()
    config.calib_frames = SETTINGS["calib_frames"]
    config.locators_only = False
    config.global_scale_only = False
    config.loss_alpha = SETTINGS["loss_alpha"]
    config.max_iter = SETTINGS["max_iter"]
    zero = np.zeros(len(character.parameter_transform.names), np.float32)
    _, selected, _ = mt.calibrate_markers(character, zero.copy(), markers, config)
    selected = [int(i) for i in selected]
    ours = md.calibration_frame_indices(array.shape[0], SETTINGS["calib_frames"])
    return {"frames": int(array.shape[0]), "calib_frames": SETTINGS["calib_frames"],
            "greedy_sampling": int(config.greedy_sampling),
            "momentum_selected_frame_indices": selected,
            "mhr_delivery_calibration_frame_indices": ours,
            "equal": selected == ours,
            "rule": "stride = max(1, (n - 1) // calib_frames); frames range(0, n, stride) (momentum computeSampleStride, "
                    "greedy_sampling 0)"}


# -------------------------------------------------------------------------------------------------- CLI

def run_cell(population: str, seed: int, donor: int, arm: str, out: Path) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--population", population, "--seed", str(seed),
               "--donor", str(donor), "--arm", arm, "--out", str(out)]
    print("  ->", population, seed, f"d{donor}", arm, flush=True)
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = completed.stdout.decode("utf-8", "replace").replace("\r", "\n").splitlines()
    for line in [l for l in text if l.strip() and "per-frame" not in l and "Solving sequence" not in l][-3:]:
        print("     ", line, flush=True)
    if completed.returncode:
        raise SystemExit(f"cell {population}/{seed}/d{donor}/{arm} failed ({completed.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--population", choices=tuple(POPULATIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--donor", type=int, default=0)
    parser.add_argument("--arm")
    parser.add_argument("--drive", action="store_true")
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--tripwire-compare", action="store_true")
    parser.add_argument("--calibration-frames", action="store_true")
    arguments = parser.parse_args()
    if arguments.tripwire_compare or arguments.calibration_frames:
        record = tripwire_compare() if arguments.tripwire_compare else calibration_frames_probe()
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in record.items() if not isinstance(v, (dict, list)) or k == "equal"},
                         indent=1))
        return 0
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if arguments.drive:
        arms = arguments.arms or list(ACCEPTANCE_ARMS if arguments.population == "acceptance" else DEVELOPMENT_ARMS)
        for seed, donor in POPULATIONS[arguments.population]:
            for arm in arms:
                run_cell(arguments.population, seed, donor, arm, out)
        return 0
    if arguments.seed is None or arguments.arm is None or arguments.population is None:
        raise SystemExit("--population, --seed and --arm are required without --drive")
    record = one_cell(arguments.population, arguments.seed, arguments.donor, arguments.arm, out)
    path = out / f"cell-{arguments.seed}-d{arguments.donor}-{arguments.arm}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    distance = np.asarray(record["distance_mm"])
    spine = record["identity_channel_names"].index("scale_spine_length")
    print(f"{arguments.seed} d{arguments.donor} {arguments.arm}: pooled "
          f"{float(np.median(np.median(distance, axis=1))):.4f} mm; spine {record['fitted_identity'][spine]:+.4f} "
          f"(truth {record['truth_identity'][spine]:+.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
