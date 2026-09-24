"""D4b O1, re-registered prospectively: the fixture and its arms, ONE PROCESS PER CELL.

The card is the "D4b O1 re-registered, prospectively" row of `docs/LADDER_EXECUTION_PLAN.md` §2. This file
builds the fixtures and runs the arms; it scores nothing. The band L, the tolerances, the must-fails and the
verdict are computed by `tools/compare/d4b_o1_gate.py` on `.venv` from the arrays each cell writes here.

The fixture (every parameter declared by the card): TWO donors, the pre-card's tracked MHR pose for performer 0
and performer 1, each clamped to `compact_v6_1.model`'s configured `limit ... minmax` entries by D4's own
`truth_motion` (imported, never copied); six NEW seeds 20261001-20261006 per donor; the identity drawn uniform
within its configured limit on the FROZEN drawn set (`docs/reviews/body-model-o1-records/drawn-set.json`) and
every other identity channel zero, the generator seeded by the PAIR (seed, donor index) so no two fixtures share
a draw; truth landmarks the 17 mapped locators at pinned zero offsets; the fitter receives landmarks only.

The arms (the momentum settings are the pre-card's in every banded arm -- calib_frames 100, loss_alpha 2.0,
max_iter 30, offsets pinned at limit_weight 10, the 26 `*_flexible` channels frozen -- all inside the unedited
`mhr_delivery.fit_one`):

    oracle            the fitted identity, pose re-solved                       (the candidate)
    exact_identity    the truth identity handed in, pose re-solved              (the paired floor; must-fail iii)
    mean_body         the identity held at zero, pose re-solved                 (must-fail i)
    spine_displaced   the truth identity, scale_spine_length 0.149 toward zero  (must-fail ii)
    warm              the calibration STARTED at the truth identity             (REPORTED probe)
    converged         max_iter 300                                              (REPORTED probe)

WARM and CONVERGED wrap `fit_one`, never re-implement it: `mhr_delivery.mt` is swapped, for one call, for a
proxy whose `calibrate_markers` substitutes the start identity (WARM) or captures momentum's own iteration log
(CONVERGED, debug on the CALIBRATION config only). The tripwire proves the proxy inert: a zero start, and the
debug capture, each reproduce the plain oracle's identity bit for bit.

What L reads is the REST: forward kinematics at MHR's zero pose with the identity RETAINED, both identities on
ONE character object (the 178-parameter export character, loaded before any calibration in the process). Never
the track's `rest_positions_z_up_m`, which `write_track` computes with ALL parameters zero (the mean body).

    /tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --seed S --donor D --arm A --out DIR
    /tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --drive --out DIR [--only SEED:DONOR ...] [--arms ...]
    /tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --burned-read --out DIR       # D4's retained cells
    /tmp/momenv/bin/python tools/fitter/d4b_o1_fixture.py --burned --seed S --arm A --out DIR
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pymomentum.geometry as g

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d4_o1_exactness as d4  # noqa: E402
import mhr_delivery as md  # noqa: E402
from d4b_identifiability import (ARMS, CHANNELS, DONOR_FILES, DONOR_INDICES, FRAMES,  # noqa: E402
                                 MAPPED_JOINTS, SEEDS, displaced_spine)

ROOT = Path(__file__).resolve().parents[2]
LOD = 2
DRAWN_SET = ROOT / "docs/reviews/body-model-o1-records/drawn-set.json"
D4_RETAINED = ROOT / "artifacts/compare/d4-body/o1"
D4_SEEDS = d4.SEEDS
SCHEMA = "d4b-o1-cell/1"
SETTINGS = {"calib_frames": 100, "loss_alpha": 2.0, "max_iter": 30, "smoothing": 0.0,
            "locator_limit_weight": 10.0, "freeze_flexible": True}
CONVERGED_MAX_ITER = 300      # ONLY in the report-only CONVERGED probe
PROBE_ARMS = ("tripwire_zero_start", "tripwire_debug")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def provenance() -> dict:
    return {
        "fitter_sha256": sha256(Path(md.__file__)),
        "d4_fixture_sha256": sha256(Path(d4.__file__)),
        "this_file_sha256": sha256(Path(__file__)),
        "donor_sha256": {str(d): sha256(ROOT / DONOR_FILES[d]) for d in DONOR_INDICES},
        "model_sha256": sha256(md.ASSETS / "compact_v6_1.model"),
        "fbx_sha256": sha256(md.ASSETS / f"lod{LOD}.fbx"),
        "drawn_set_sha256": sha256(DRAWN_SET),
        "pymomentum": importlib.metadata.version("pymomentum-cpu"),
    }


# ---------------------------------------------------------------------------------------------- the proxy

class _CalibrationProxy:
    """Stands in for `pymomentum.marker_tracking` inside `mhr_delivery` for ONE `fit_one` call.

    Every attribute passes through to the real module except `calibrate_markers`, which (a) replaces the start
    identity when `start` is given (WARM) and (b) captures momentum's own log with the CALIBRATION config's
    `debug` on when `capture` is set (CONVERGED), restoring `debug` afterwards so the config handed on to
    `process_markers` is unchanged.
    """

    def __init__(self, real, start: np.ndarray | None, capture: bool):
        self._real = real
        self._start = None if start is None else np.asarray(start, np.float32)
        self._capture = capture
        self.logs: list[str] = []
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def calibrate_markers(self, character, identity, markers, config, *args, **kwargs):
        self.calls += 1
        if self._start is not None:
            identity = self._start.copy()
        if not self._capture:
            return self._real.calibrate_markers(character, identity, markers, config, *args, **kwargs)
        config.debug = True
        sys.stdout.flush()
        sys.stderr.flush()
        saved = os.dup(1), os.dup(2)
        with tempfile.TemporaryFile(mode="w+b") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                result = self._real.calibrate_markers(character, identity, markers, config, *args, **kwargs)
            finally:
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                os.close(saved[0])
                os.close(saved[1])
                config.debug = False
            sink.seek(0)
            self.logs.append(sink.read().decode("utf-8", "replace"))
        return result


@contextlib.contextmanager
def calibration_proxy(start: np.ndarray | None = None, capture: bool = False):
    real = md.mt
    proxy = _CalibrationProxy(real, start, capture)
    md.mt = proxy
    try:
        yield proxy
    finally:
        md.mt = real


def iteration_segments(log: str) -> list[int]:
    """Momentum's debug log -> the last iteration index reached by each solve (a new solve starts when the
    iteration counter does not increase)."""
    segments, previous = [], None
    for match in re.finditer(r"Iteration: (\d+)", log):
        k = int(match.group(1))
        if previous is None or k <= previous:
            segments.append(k)
        else:
            segments[-1] = k
        previous = k
    return segments


# -------------------------------------------------------------------------------------------- the fixture

def drawn_set() -> list[str]:
    record = json.loads(DRAWN_SET.read_text(encoding="utf-8"))
    if not record.get("frozen_before_any_fit"):
        raise SystemExit("the drawn-set record is not the frozen one")
    return list(record["drawn_set"])


def build_fixture(truth_character: g.Character, seed: int, donor: int, burned: bool) -> tuple[np.ndarray, dict]:
    """The truth motion (full 204-parameter layout) and a description of how it was drawn."""
    limits = d4.configured_limits()
    d4.DONOR = ROOT / DONOR_FILES[donor]
    if burned:
        # D4's OWN fixture, reproduced: its generator, donor 0, all ten named channels drawn.
        if donor != 0 or seed not in D4_SEEDS:
            raise SystemExit("a burned fixture is one of D4's six (donor 0)")
        rng = np.random.default_rng(seed)
        draw_limits = limits
        channels = list(CHANNELS)
        generator = f"numpy default_rng({seed}) (D4's own)"
    else:
        channels = drawn_set()
        rng = np.random.default_rng([seed, donor])
        draw_limits = dict(limits)
        for channel in CHANNELS:
            if channel not in channels:
                draw_limits[channel] = (0.0, 0.0)   # every other identity channel zero
        generator = f"numpy default_rng([{seed}, {donor}]) (seeded by the pair)"
    motion, drawn, repair = d4.truth_motion(truth_character, rng, draw_limits)
    names = list(truth_character.parameter_transform.names)
    nonzero_undrawn = [n for i, n in enumerate(names) if n.startswith("scale_") and n not in channels
                       and float(motion[0, i]) != 0.0]
    if nonzero_undrawn:
        raise SystemExit(f"undrawn identity channels are nonzero: {nonzero_undrawn}")
    return motion, {"generator": generator, "drawn_channels": channels, "drawn_identity": drawn,
                    "clamped_parameter_frames": repair["clamped_parameter_frames"],
                    "truth_motion_sha256": array_sha256(motion)}


def scale_names(character: g.Character) -> list[str]:
    return [n for n in character.parameter_transform.names if n.startswith("scale_")]


def identity_vector(character: g.Character, values: dict[str, float]) -> np.ndarray:
    """A full model-parameter vector for `character`: the named identity channels set, every pose zero."""
    names = list(character.parameter_transform.names)
    vector = np.zeros(len(names), np.float32)
    for name, value in values.items():
        vector[names.index(name)] = np.float32(value)
    return vector


def rest_mapped_cm(character: g.Character, values: dict[str, float]) -> np.ndarray:
    """The 17 mapped joints at MHR's zero pose with the identity RETAINED (cm)."""
    skeleton = list(character.skeleton.joint_names)
    rows = [skeleton.index(j) for j in MAPPED_JOINTS]
    state = np.asarray(g.model_parameters_to_skeleton_state(
        character, identity_vector(character, values).astype(np.float64)))
    return state[rows, :3]


# ---------------------------------------------------------------------------------------------- one cell

def one_cell(seed: int, donor: int, arm: str, out: Path, burned: bool = False) -> dict:
    if tuple(md.MAP.values()) != MAPPED_JOINTS:
        raise SystemExit("the mapped joints are not mhr_delivery.MAP's")
    # Every character this process needs is loaded BEFORE any calibration (mhr_delivery.export_glb).
    truth_character = md.load_character(LOD)
    export_character = md.load_character(LOD, drop_flexible=True)
    motion, fixture = build_fixture(truth_character, seed, donor, burned)
    if burned:
        retained = np.load(D4_RETAINED / f"seed-{seed}-oracle.truth.npz")
        if not np.array_equal(retained["truth_motion"], motion):
            raise SystemExit(f"the regenerated D4 fixture {seed} is not the retained one")
        fixture["retained_truth_npz_sha256"] = sha256(D4_RETAINED / f"seed-{seed}-oracle.truth.npz")
    full_names = list(truth_character.parameter_transform.names)
    identity_channels = scale_names(export_character)
    truth_identity = {n: float(motion[0, full_names.index(n)]) for n in identity_channels}
    truth_cm = d4.mapped_positions_cm(truth_character, motion)
    joint_names = [str(n) for n in np.load(d4.JOINT_NAMES_FROM, allow_pickle=True)["joint_names"]]
    array = d4.landmark_array(truth_cm, joint_names)

    truth_start = identity_vector(export_character, truth_identity)
    fixed, mean_body, start, capture, max_iter = None, False, None, False, SETTINGS["max_iter"]
    arm_note = {}
    if arm == "exact_identity":
        fixed = truth_start.copy()
    elif arm == "mean_body":
        mean_body = True
    elif arm == "spine_displaced":
        displaced = dict(truth_identity)
        displaced["scale_spine_length"] = displaced_spine(truth_identity["scale_spine_length"])
        fixed = identity_vector(export_character, displaced)
        arm_note = {"spine_truth": truth_identity["scale_spine_length"],
                    "spine_displaced_to": float(np.float32(displaced["scale_spine_length"]))}
    elif arm == "warm":
        start = truth_start.copy()
    elif arm == "converged":
        capture, max_iter = True, CONVERGED_MAX_ITER
    elif arm == "tripwire_zero_start":
        start = np.zeros_like(truth_start)
    elif arm == "tripwire_debug":
        capture = True
    elif arm != "oracle":
        raise SystemExit(f"unknown arm {arm}")

    with calibration_proxy(start=start, capture=capture) as proxy:
        fit = md.fit_one(array, joint_names, lod=LOD, mean_body=mean_body, free_offsets=False,
                         freeze_flexible=True, fixed_identity=fixed, max_iter=max_iter)
    if list(fit["parameter_names"]) != list(export_character.parameter_transform.names):
        raise SystemExit("the fitted character is not the export character's layout")
    fitted_identity = {n: float(fit["identity"][fit["parameter_names"].index(n)]) for n in identity_channels}
    skeleton = list(fit["character"].skeleton.joint_names)
    rows = [skeleton.index(j) for j in MAPPED_JOINTS]
    fitted_cm = fit["positions_cm"][:, rows]
    distance_mm = np.linalg.norm(fitted_cm - truth_cm, axis=2) * 10.0

    truth_rest = rest_mapped_cm(export_character, truth_identity)
    fitted_rest = rest_mapped_cm(export_character, fitted_identity)
    # the same truth rest on the FULL character (flexible channels zero): the two layouts must agree
    truth_rest_full = rest_mapped_cm(truth_character, truth_identity)

    tag = f"{seed}-d{donor}-{arm}"
    record = {
        "schema": SCHEMA, "burned": burned, "seed": seed, "donor": donor, "arm": arm, "lod": LOD,
        "provenance": provenance(),
        "fixture": fixture,
        "settings": dict(SETTINGS, max_iter=max_iter, warm_start=start is not None and arm == "warm",
                         calibration_debug_captured=capture),
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
    }
    if capture:
        segments = [iteration_segments(log) for log in proxy.logs]
        record["calibration_iterations"] = {
            "max_iter": max_iter,
            "calls": len(segments),
            "solves_per_call": [len(s) for s in segments],
            "max_iteration_index_reached": [max(s) if s else None for s in segments],
            "solves_stopped_at_the_cap": [int(sum(1 for k in s if k >= max_iter - 1)) for s in segments],
            "note": "momentum's own debug log; the last index reached by each solve inside calibrate_markers "
                    "(stage A locators-only, then the full calibration). An index of max_iter-1 is a solve "
                    "stopped by the cap, not converged; below it the solver stopped on its own criterion. "
                    "Tracking ran at the same max_iter (fit_one's single setting) and is not logged.",
        }
    if arm == "oracle":
        prefix = out / f"fixture-{tag}"
        md.export_glb(fit, export_character, prefix.with_suffix(".glb"))
        md.write_track(fit, prefix, subject=0,
                       consumed={"ticks": np.arange(array.shape[0]),
                                 "triangulated_world_positions_z_up_m": array,
                                 "raw_triangulated_world_positions_z_up_m": array},
                       lod=LOD, landmarks="d4b_o1_synthetic_truth", settings=SETTINGS,
                       joint_names=joint_names)
        # basenames: the files sit beside this cell's JSON (a path would name one worktree)
        record["glb"] = prefix.with_suffix(".glb").name
        record["track"] = prefix.with_suffix(".body-track.npz").name
        record["glb_sha256"] = sha256(prefix.with_suffix(".glb"))
        record["track_sha256"] = sha256(prefix.with_suffix(".body-track.npz"))
    np.savez(out / f"fixture-{tag}.arrays.npz", truth_motion=motion, truth_mapped_cm=truth_cm,
             fitted_mapped_cm=fitted_cm, identity=np.asarray(fit["identity"]), motion=fit["motion"])
    return record


# ------------------------------------------------------------------------------------- burned: retained

def burned_read(out: Path) -> list[dict]:
    """D4's retained oracle, exact_identity and mean_body cells, in THIS schema, resolved by content.

    D4's `o1.json` records paths into the ladder-D4 worktree, so nothing here trusts a recorded path: each
    retained file is found by its basename in `artifacts/compare/d4-body/o1/`, its sha256 recorded, and its
    content cross-checked (the regenerated truth motion equals the retained one; the retained cell's pooled
    statistic and recovered channels are reproduced from the retained arrays).
    """
    export_character = md.load_character(LOD, drop_flexible=True)
    truth_character = md.load_character(LOD)
    full_names = list(truth_character.parameter_transform.names)
    identity_channels = scale_names(export_character)
    records = []
    for seed in D4_SEEDS:
        motion, fixture = build_fixture(truth_character, seed, 0, burned=True)
        for arm in ("oracle", "exact_identity", "mean_body"):
            cell_path = D4_RETAINED / f"cell-{seed}-{arm}.json"
            truth_path = D4_RETAINED / f"seed-{seed}-{arm}.truth.npz"
            track_path = D4_RETAINED / f"seed-{seed}-{arm}.body-track.npz"
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            truth = np.load(truth_path)
            track = np.load(track_path, allow_pickle=True)   # D4's own delivery npz (object arrays)
            if not np.array_equal(truth["truth_motion"], motion):
                raise SystemExit(f"retained {truth_path.name} is not D4's fixture {seed}")
            fitted = dict(zip([str(n) for n in track["identity_channel_names"]],
                              [float(v) for v in track["identity_values"]]))
            if sorted(fitted) != sorted(identity_channels):
                raise SystemExit(f"{track_path.name}: identity channels differ")
            for channel, value in cell["recovered_identity"].items():
                if abs(fitted[channel] - value) > 1e-6:
                    raise SystemExit(f"{track_path.name}: {channel} disagrees with {cell_path.name}")
            distance_mm = np.linalg.norm(truth["fitted_positions_mapped_cm"] - truth["truth_positions_mapped_cm"],
                                         axis=2) * 10.0
            pooled = round(float(np.median(np.median(distance_mm, axis=1))), 4)
            if pooled != cell["median_over_frames_of_the_median_over_17_joints_mm"]:
                raise SystemExit(f"{cell_path.name}: pooled statistic not reproduced ({pooled})")
            if [str(j) for j in truth["mapped_joints"]] != list(MAPPED_JOINTS):
                raise SystemExit(f"{truth_path.name}: mapped joints differ")
            truth_identity = {n: float(motion[0, full_names.index(n)]) for n in identity_channels}
            record = {
                "schema": SCHEMA, "burned": True, "seed": seed, "donor": 0, "arm": arm, "lod": LOD,
                "provenance": provenance(),
                "retained_from": {p.name: sha256(p) for p in (cell_path, truth_path, track_path)},
                "retained_fitter_sha256": cell["fitter_source_sha256"],
                "fixture": fixture,
                "settings": dict(SETTINGS, warm_start=False, calibration_debug_captured=False),
                "mapped_joints": list(MAPPED_JOINTS),
                "frames": int(distance_mm.shape[0]),
                "identity_channel_names": identity_channels,
                "truth_identity": [truth_identity[n] for n in identity_channels],
                "fitted_identity": [fitted[n] for n in identity_channels],
                "truth_rest_mapped_cm": rest_mapped_cm(export_character, truth_identity).tolist(),
                "fitted_rest_mapped_cm": rest_mapped_cm(export_character, fitted).tolist(),
                "distance_mm": distance_mm.round(6).tolist(),
                "d4_pooled_statistic_mm_reproduced": pooled,
            }
            path = out / f"cell-{seed}-d0-{arm}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            records.append(record)
            print(f"burned {seed} {arm}: pooled {pooled} mm (retained, reproduced)", flush=True)
    return records


# -------------------------------------------------------------------------------------------------- CLI

def run_cell(seed: int, donor: int, arm: str, out: Path, burned: bool) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--seed", str(seed), "--donor", str(donor),
               "--arm", arm, "--out", str(out)] + (["--burned"] if burned else [])
    print("  ->", seed, f"d{donor}", arm, flush=True)
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = completed.stdout.decode("utf-8", "replace").replace("\r", "\n").splitlines()
    for line in [l for l in tail if l.strip() and "Tracking per-frame" not in l and "Solving sequence" not in l][-4:]:
        print("     ", line, flush=True)
    if completed.returncode:
        raise SystemExit(f"cell {seed}/d{donor}/{arm} failed ({completed.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--donor", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS + PROBE_ARMS)
    parser.add_argument("--burned", action="store_true", help="one of D4's six fixtures, reproduced")
    parser.add_argument("--burned-read", action="store_true", help="D4's retained cells, in this schema")
    parser.add_argument("--drive", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="SEED:DONOR fixtures for --drive")
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    arguments = parser.parse_args()
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if arguments.burned_read:
        burned_read(out)
        return 0
    if arguments.drive:
        fixtures = ([tuple(int(x) for x in item.split(":")) for item in arguments.only]
                    if arguments.only else [(s, d) for d in DONOR_INDICES for s in SEEDS])
        for seed, donor in fixtures:
            for arm in arguments.arms:
                run_cell(seed, donor, arm, out, arguments.burned)
        return 0
    if arguments.seed is None or arguments.arm is None:
        raise SystemExit("--seed and --arm are required without --drive / --burned-read")
    if not arguments.burned and arguments.seed not in SEEDS and arguments.arm not in PROBE_ARMS:
        raise SystemExit(f"{arguments.seed} is not one of the card's seeds")
    record = one_cell(arguments.seed, arguments.donor, arguments.arm, out, burned=arguments.burned)
    path = out / f"cell-{arguments.seed}-d{arguments.donor}-{arguments.arm}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    distance = np.asarray(record["distance_mm"])
    print(f"{arguments.seed} d{arguments.donor} {arguments.arm}: pooled "
          f"{float(np.median(np.median(distance, axis=1))):.4f} mm; spine "
          f"{record['fitted_identity'][record['identity_channel_names'].index('scale_spine_length')]:+.4f} "
          f"(truth {record['truth_identity'][record['identity_channel_names'].index('scale_spine_length')]:+.4f})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
