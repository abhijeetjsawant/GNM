"""D4d a second calibration pass on the landmark start: the fixtures and their arms, ONE PROCESS PER CELL.

The card is the "D4d a second calibration pass" row of `docs/LADDER_EXECUTION_PLAN.md` §2 (copy:
`docs/reviews/body-model-twopass-card-2026-09-25.md`). This file builds the fixtures and runs the arms; it scores
nothing. The Phase-1 decision, L, the tolerances, stage 0b, the must-fails and the ONE verdict are computed by
`tools/compare/d4d_twopass_gate.py` on `.venv` from what each cell writes here.

The candidate is D4c's landmark start (merged from tag `ladder/D4c-fail-1a89cc7`, trunk statistic p90, frozen in
D4c's development JSON) PLUS a second full calibration pass: `mhr_delivery.fit_one(..., passes=2)` repeats stage A
(locators only) and stage B (identity) from its own copy of the first pass's identity, each pass at `max_iter` 30.
Tracking is untouched.

PHASE 1, the diagnostic, on BURNED fixtures only (30):
    d4    D4's six   (seeds 20260922-27, donor 0, D4's own generator, all ten channels drawn)
    d4b   D4b's twelve (seeds 20261001-06 x donors 0/1, D4b's frozen drawn set; the spine never drawn)
    d4c   D4c's twelve (seeds 20261101-06 x donors 0/1, D4c's frozen eight-channel drawn set) -- the fork's population
  New arms (run here):
    warm        calibration started at the TRUTH identity, one pass           (truth-seeded: DECIDES the fork only)
    two_pass    the D4c start, TWO passes                                     (the intervention)
    sw_star     the D4c start with ONLY scale_shoulder_width set to the truth, one pass (truth-seeded: a report)
  Retained arms (D4c's own cells, reused, bound by content hash and fixture identity by the gate):
    d4c_start        D4c's `candidate_p90` (d4, d4b) / `candidate` (d4c): the D4c start, one pass
    exact_identity   the paired floors
    spine_displaced  stage 0b's control

PHASE 2, acceptance, ONLY if the committed Phase-1 decision selects TWO-PASS (`--population phase2`): untouched seeds
20261201-06 x donors 0/1, the identity drawn uniform within its configured limit on D4c's frozen drawn set (D4c's
acceptance generator: `default_rng([seed, donor])`, poses clamped by D4's own `truth_motion`). Arms:
    candidate        the D4c start, TWO passes (GLB and track exported for closure)
    exact_identity   the truth identity handed in, pose re-solved      (the paired floor; must-fail iii)
    mean_body        the identity held at zero, pose re-solved         (must-fail i)
    spine_displaced  the truth, scale_spine_length 0.149 toward zero   (must-fail ii)
    init_only        the D4c start HELD, no calibration                (must-fail iv)
    one_pass         the D4c start, one pass (D4c's fitter)            (REPORTED)
    legacy           the zero start, one pass (D4's fitter)            (REPORTED)
  The generator REFUSES to run unless the Phase-1 decision JSON exists, is byte-identical to its committed copy at
  HEAD, and selects TWO-PASS; every Phase-2 cell carries that JSON's sha256.

Momentum settings are unchanged in every arm (calib_frames 100, loss_alpha 2.0, max_iter 30, offsets pinned at
limit_weight 10, the 26 `*_flexible` channels frozen -- all inside `mhr_delivery.fit_one`). Calibrating arms capture
momentum's debug log per `calibrate_markers` call (D4b's proxy, proved inert by D4b's tripwire) and, fixture-side, the
identity each call was handed and returned plus the locator offsets after it, so stage A and stage B are read
separately, per pass, without widening `fit_one`.

What L reads is the REST: forward kinematics at MHR's zero pose with the identity RETAINED (never the track's
`rest_positions_z_up_m`).

    /tmp/momenv/bin/python tools/fitter/d4d_fixture.py --provenance --out JSON
    /tmp/momenv/bin/python tools/fitter/d4d_fixture.py --population P --seed S --donor D --arm A --out DIR
    /tmp/momenv/bin/python tools/fitter/d4d_fixture.py --drive --population P [--arms ...] --out DIR
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import d4c_fixture as fx  # noqa: E402  (momentum-free at import)

ROOT = Path(__file__).resolve().parents[2]
LOD = 2
FRAMES = 150
SCHEMA = "d4d-twopass-cell/1"
RECORDS = ROOT / "docs/reviews/body-model-twopass-records"
DECISION = RECORDS / "phase1-decision.json"
D4C_RECORDS = ROOT / "docs/reviews/body-model-start-records"
D4C_DRAWN_SET = D4C_RECORDS / "drawn-set.json"
D4C_DEVELOPMENT = D4C_RECORDS / "development.json"
D4C_MANIFEST = D4C_RECORDS / "acceptance-manifest.json"
D4C_CELLS = {"development": ROOT / "artifacts/compare/d4c-start/development",
             "acceptance": ROOT / "artifacts/compare/d4c-start/acceptance"}
OUT = ROOT / "artifacts/compare/d4d-twopass"
SETTINGS = dict(fx.SETTINGS)          # calib_frames 100, loss_alpha 2.0, max_iter 30, smoothing 0, pinned, frozen flex
PASSES = 2                            # the card: fixed at 2, never swept

PHASE2_SEEDS = (20261201, 20261202, 20261203, 20261204, 20261205, 20261206)
DONORS = (0, 1)
POPULATIONS = {
    "d4": list(fx.POPULATIONS["d4"]),
    "d4b": list(fx.POPULATIONS["d4b"]),
    "d4c": list(fx.POPULATIONS["acceptance"]),
    "phase2": [(s, d) for d in DONORS for s in PHASE2_SEEDS],
}
PHASE1 = ("d4", "d4b", "d4c")
PHASE1_NEW_ARMS = ("warm", "two_pass", "sw_star")
PHASE1_RETAINED_ARMS = ("d4c_start", "exact_identity", "spine_displaced")
PHASE1_ARMS = ("d4c_start", "warm", "two_pass", "sw_star", "exact_identity", "spine_displaced")
PHASE2_ARMS = ("candidate", "exact_identity", "mean_body", "spine_displaced", "init_only", "one_pass", "legacy")
# arm -> (start, passes); start in {"landmark", "truth", "sw_star", "zero"}; None = no calibration
CALIBRATING = {"warm": ("truth", 1), "two_pass": ("landmark", 2), "sw_star": ("sw_star", 1),
               "candidate": ("landmark", 2), "one_pass": ("landmark", 1), "legacy": ("zero", 1)}
CALL_LABELS = {1: ("A1_locators_only", "B1_identity"),
               2: ("A1_locators_only", "B1_identity", "A2_locators_only", "B2_identity")}


def retained_cell(population: str, seed: int, donor: int, arm: str) -> Path:
    """Where D4c's retained cell for a Phase-1 retained arm lives (the D4c start is `candidate_p90` in D4c's
    development and `candidate` in its acceptance)."""
    stage = "acceptance" if population == "d4c" else "development"
    name = {"d4c_start": "candidate" if population == "d4c" else "candidate_p90"}.get(arm, arm)
    return D4C_CELLS[stage] / f"cell-{seed}-d{donor}-{name}.json"


# The spine-tercile breakdown, frozen before any Phase-1 reading: the configured limit of scale_spine_length
# ([-1.1, 1.1] in compact_v6_1.model) cut into equal thirds. Population-independent, so the breakdown cannot be
# drawn after the draws are seen. A value on an edge goes to the tercile nearer zero.
def spine_tercile(value: float, limit: tuple[float, float] = (-1.1, 1.1)) -> str:
    low, high = limit
    third = (high - low) / 3.0
    if value < low + third:
        return "short"
    if value > high - third:
        return "long"
    return "middle"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_committed_bytes(path: Path) -> bytes | None:
    completed = subprocess.run(["git", "show", f"HEAD:{Path(path).resolve().relative_to(ROOT)}"], cwd=ROOT,
                               capture_output=True, check=False)
    return completed.stdout if completed.returncode == 0 else None


def decision_guard() -> str:
    """The Phase-2 generator's precondition: the Phase-1 decision exists, is COMMITTED byte for byte at HEAD, and
    selects TWO-PASS. Returns its sha256, which is stamped into every Phase-2 cell."""
    if not DECISION.is_file():
        raise SystemExit("no Phase-1 decision JSON: no Phase-2 fixture may exist before it")
    data = DECISION.read_bytes()
    committed = git_committed_bytes(DECISION)
    if committed is None or committed != data:
        raise SystemExit("the Phase-1 decision JSON is not committed at HEAD byte for byte")
    record = json.loads(data)
    if record.get("selected") != "TWO-PASS" or record.get("phase_2_licensed") is not True:
        raise SystemExit(f"the Phase-1 decision does not license Phase 2 (selected {record.get('selected')!r})")
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------------------- the proxy

class _RecordingProxy:
    """D4b's calibration proxy (log capture on the CALIBRATION config's `debug` only, inert by D4b's tripwire),
    extended fixture-side to record, per `calibrate_markers` call, the identity handed in, the identity returned,
    `locators_only`, and the largest locator offset after the call. It never changes what the call receives."""

    def __init__(self, capture: bool):
        import d4b_o1_fixture as d4bf
        import mhr_delivery as md

        self._inner = d4bf._CalibrationProxy(md.mt, None, capture)
        self.calls_record: list[dict] = []

    @property
    def calls(self) -> int:
        return self._inner.calls

    @property
    def logs(self) -> list[str]:
        return self._inner.logs

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def calibrate_markers(self, character, identity, markers, config, *args, **kwargs):
        handed = np.asarray(identity, np.float32).copy()
        result = self._inner.calibrate_markers(character, identity, markers, config, *args, **kwargs)
        offsets = [float(np.linalg.norm(np.asarray(l.offset, np.float64)) * 10.0) for l in character.locators]
        self.calls_record.append({"locators_only": bool(config.locators_only), "handed": handed,
                                  "returned": np.asarray(result[0], np.float32).copy(),
                                  "locator_offset_mm_max_after": max(offsets) if offsets else None})
        return result


@contextlib.contextmanager
def recording_proxy(capture: bool):
    import mhr_delivery as md

    real = md.mt
    proxy = _RecordingProxy(capture)
    md.mt = proxy
    try:
        yield proxy
    finally:
        md.mt = real


def cap_counts(logs: list[str], passes: int, max_iter: int) -> dict:
    """D4c's split (below / at / above the configured cap, by each solve's last logged iteration index), labelled
    per stage AND per pass: A1, B1 (and A2, B2 on two passes)."""
    import d4b_o1_fixture as d4bf

    labels = CALL_LABELS[passes]
    if len(logs) != len(labels):
        raise SystemExit(f"{len(logs)} calibration logs for {passes} pass(es): expected {len(labels)}")
    out = {"max_iter": max_iter, "passes": passes, "calls": len(logs), "per_call": []}
    for stage, log in zip(labels, logs):
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


# --------------------------------------------------------------------------------------------- one cell

def build_truth(truth_character, population: str, seed: int, donor: int):
    import d4_o1_exactness as d4
    import d4b_o1_fixture as d4bf
    from d4b_identifiability import CHANNELS

    if population in ("d4c", "phase2"):
        limits = d4.configured_limits()
        drawn = list(json.loads(D4C_DRAWN_SET.read_text(encoding="utf-8"))["drawn_set"])
        draw_limits = dict(limits)
        for channel in CHANNELS:
            if channel not in drawn:
                draw_limits[channel] = (0.0, 0.0)
        d4.DONOR = ROOT / d4bf.DONOR_FILES[donor]
        motion, drawn_identity, repair = d4.truth_motion(truth_character, np.random.default_rng([seed, donor]),
                                                         draw_limits)
        regenerated = fx.acceptance_draw(seed, donor, drawn, limits, CHANNELS)
        if any(np.float32(regenerated[c]) != np.float32(drawn_identity[c]) for c in CHANNELS):
            raise SystemExit("acceptance_draw does not reproduce truth_motion's draw")
        return motion, {"generator": f"numpy default_rng([{seed}, {donor}]) (seeded by the pair)",
                        "drawn_channels": drawn, "drawn_identity": drawn_identity,
                        "clamped_parameter_frames": repair["clamped_parameter_frames"],
                        "truth_motion_sha256": fx.array_sha256(motion)}
    return d4bf.build_fixture(truth_character, seed, donor, burned=(population == "d4"))


def one_cell(population: str, seed: int, donor: int, arm: str, out: Path) -> dict:
    import d4_o1_exactness as d4
    import d4b_o1_fixture as d4bf
    import mhr_delivery as md
    from d4b_identifiability import MAPPED_JOINTS

    if tuple(md.MAP.values()) != MAPPED_JOINTS:
        raise SystemExit("the mapped joints are not mhr_delivery.MAP's")
    if (seed, donor) not in POPULATIONS[population]:
        raise SystemExit(f"{seed}/d{donor} is not in the {population} population")
    arms = PHASE2_ARMS if population == "phase2" else PHASE1_NEW_ARMS
    if arm not in arms:
        raise SystemExit(f"{arm} is not an arm run here for the {population} population")
    decision_sha = decision_guard() if population == "phase2" else None
    statistic = fx.frozen_statistic()
    if statistic is None or md.TRUNK_STATISTIC != statistic:
        raise SystemExit("the fitter's TRUNK_STATISTIC is not D4c's frozen development choice")
    # Every character this process needs is loaded BEFORE any calibration (mhr_delivery.export_glb).
    truth_character = md.load_character(LOD)
    export_character = md.load_character(LOD, drop_flexible=True)
    motion, fixture = build_truth(truth_character, population, seed, donor)
    full_names = list(truth_character.parameter_transform.names)
    identity_channels = d4bf.scale_names(export_character)
    truth_identity = {n: float(motion[0, full_names.index(n)]) for n in identity_channels}
    truth_cm = d4.mapped_positions_cm(truth_character, motion)
    # allow_pickle: our own build output (object arrays; CLAUDE.md), never a third-party file
    joint_names = [str(n) for n in np.load(d4.JOINT_NAMES_FROM, allow_pickle=True)["joint_names"]]
    array = d4.landmark_array(truth_cm, joint_names)
    names = list(export_character.parameter_transform.names)

    truth_start = d4bf.identity_vector(export_character, truth_identity)
    landmark, start_record = md.landmark_start(array, joint_names, export_character, trunk_statistic=statistic)
    fixed, mean_body, start, passes, arm_note = None, False, None, 1, {}
    kind = CALIBRATING.get(arm, (None, 1))[0]
    if arm in CALIBRATING:
        passes = CALIBRATING[arm][1]
        if kind == "landmark":
            start = landmark.copy()
        elif kind == "truth":
            start = truth_start.copy()
        elif kind == "sw_star":
            start = landmark.copy()
            start[names.index("scale_shoulder_width")] = np.float32(truth_identity["scale_shoulder_width"])
            arm_note = {"shoulder_width_landmark_start": float(landmark[names.index("scale_shoulder_width")]),
                        "shoulder_width_set_to_truth": float(np.float32(truth_identity["scale_shoulder_width"]))}
        elif kind == "zero":
            start = None                      # fit_one's own zero start: the D4 fitter, byte for byte
    elif arm == "init_only":
        fixed = landmark.copy()
    elif arm == "exact_identity":
        fixed = truth_start.copy()
    elif arm == "mean_body":
        mean_body = True
    elif arm == "spine_displaced":
        displaced = dict(truth_identity)
        displaced["scale_spine_length"] = fx.displaced_spine(truth_identity["scale_spine_length"])
        fixed = d4bf.identity_vector(export_character, displaced)
        arm_note = {"spine_truth": truth_identity["scale_spine_length"],
                    "spine_displaced_to": float(np.float32(displaced["scale_spine_length"]))}
    else:
        raise SystemExit(f"unknown arm {arm}")

    capture = arm in CALIBRATING
    with recording_proxy(capture=capture) as proxy:
        fit = md.fit_one(array, joint_names, lod=LOD, mean_body=mean_body, free_offsets=False,
                         freeze_flexible=True, fixed_identity=fixed, start_identity=start, passes=passes)
    if list(fit["parameter_names"]) != names:
        raise SystemExit("the fitted character is not the export character's layout")
    fitted_identity = {n: float(fit["identity"][names.index(n)]) for n in identity_channels}
    skeleton = list(fit["character"].skeleton.joint_names)
    rows = [skeleton.index(j) for j in MAPPED_JOINTS]
    fitted_cm = fit["positions_cm"][:, rows]
    distance_mm = np.linalg.norm(fitted_cm - truth_cm, axis=2) * 10.0
    truth_rest = d4bf.rest_mapped_cm(export_character, truth_identity)
    fitted_rest = d4bf.rest_mapped_cm(export_character, fitted_identity)
    truth_rest_full = d4bf.rest_mapped_cm(truth_character, truth_identity)

    def scale_values(vector: np.ndarray) -> list[float]:
        return [float(vector[names.index(n)]) for n in identity_channels]

    per_call = []
    labels = CALL_LABELS[passes] if capture else ()
    for label, call in zip(labels, proxy.calls_record):
        returned = {n: float(call["returned"][names.index(n)]) for n in identity_channels}
        per_call.append({"stage": label, "locators_only": call["locators_only"],
                         "identity_handed": scale_values(call["handed"]),
                         "identity_returned": scale_values(call["returned"]),
                         "rest_mapped_cm_after": d4bf.rest_mapped_cm(export_character, returned).tolist(),
                         "locator_offset_mm_max_after": call["locator_offset_mm_max_after"]})

    tag = f"{seed}-d{donor}-{arm}"
    arrays_path = out / f"fixture-{tag}.arrays.npz"
    np.savez(arrays_path, truth_motion=motion, truth_mapped_cm=truth_cm, consumed_landmarks_z_up_m=array,
             consumed_joint_names=np.array(joint_names), fitted_mapped_cm=fitted_cm,
             identity=np.asarray(fit["identity"]), motion=fit["motion"])
    record = {
        "schema": SCHEMA, "population": population, "seed": seed, "donor": donor, "arm": arm, "lod": LOD,
        "provenance": provenance(),
        "phase1_decision_sha256": decision_sha,
        "fixture": fixture,
        "settings": dict(SETTINGS, calibration_debug_captured=capture, passes=passes if capture else 0),
        "start": {"kind": kind if capture else ("landmark_held" if arm == "init_only" else None),
                  "trunk_statistic": statistic,
                  "landmark_start_identity": scale_values(landmark), "landmark_start_record": start_record,
                  "start_identity": None if start is None else scale_values(start),
                  "held_identity": None if fixed is None else scale_values(fixed)},
        "arm_note": arm_note,
        "calibrate_markers_calls": proxy.calls,
        "calibration_calls": per_call,
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
        record["calibration_iterations"] = cap_counts(proxy.logs, passes, SETTINGS["max_iter"])
        (out / f"fixture-{tag}.calibration-debug.log").write_text("\n=====CALL=====\n".join(proxy.logs),
                                                                 encoding="utf-8")
    if arm == "candidate":
        prefix = out / f"fixture-{tag}"
        md.export_glb(fit, export_character, prefix.with_suffix(".glb"))
        md.write_track(fit, prefix, subject=0,
                       consumed={"ticks": np.arange(array.shape[0]),
                                 "triangulated_world_positions_z_up_m": array,
                                 "raw_triangulated_world_positions_z_up_m": array},
                       lod=LOD, landmarks="d4d_twopass_synthetic_truth", settings=SETTINGS,
                       joint_names=joint_names)
        record["glb"] = prefix.with_suffix(".glb").name
        record["track"] = prefix.with_suffix(".body-track.npz").name
        record["glb_sha256"] = sha256(prefix.with_suffix(".glb"))
        record["track_sha256"] = sha256(prefix.with_suffix(".body-track.npz"))
    return record


def provenance() -> dict:
    import d4_o1_exactness as d4
    import d4b_o1_fixture as d4bf
    import mhr_delivery as md

    return {
        "fitter_sha256": sha256(Path(md.__file__)),
        "d4_fixture_sha256": sha256(Path(d4.__file__)),
        "d4b_fixture_sha256": sha256(Path(d4bf.__file__)),
        "d4c_fixture_sha256": sha256(Path(fx.__file__)),
        "this_file_sha256": sha256(Path(__file__)),
        "donor_sha256": {str(d): sha256(ROOT / d4bf.DONOR_FILES[d]) for d in DONORS},
        "model_sha256": sha256(md.ASSETS / "compact_v6_1.model"),
        "fbx_sha256": sha256(md.ASSETS / f"lod{LOD}.fbx"),
        "d4c_drawn_set_sha256": sha256(D4C_DRAWN_SET),
        "d4b_drawn_set_sha256": sha256(fx.D4B_DRAWN_SET),
        "d4c_development_json_sha256": sha256(D4C_DEVELOPMENT),
        "pymomentum": importlib.metadata.version("pymomentum-cpu"),
    }


# ------------------------------------------------------------------------------------ stage 1 provenance

def stage1_provenance() -> dict:
    """Stage 1: the base (the merged D4c tag), by sha256, and precondition 0's input (D4c's frozen drawn set) reused
    by content; the retained D4c cells Phase 1 reuses, checked against D4c's own records; Phase 2 absent on disk."""
    import mhr_delivery as md
    import d4b_o1_fixture as d4bf

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    record = {"step": "D4d", "stage": 1, "date": "2026-09-25", "base_commit": head,
              "base": "main at fedd834 merged with tag ladder/D4c-fail-1a89cc7 (1a89cc73b2ced93543e9cbfb191d79208aef423b)",
              "sha256": {
                  "fitter tools/fitter/mhr_delivery.py (pre-change; the D4c tag's)": sha256(Path(md.__file__)),
                  "D4 fixture tools/fitter/d4_o1_exactness.py": sha256(ROOT / "tools/fitter/d4_o1_exactness.py"),
                  "D4b fixture tools/fitter/d4b_o1_fixture.py": sha256(ROOT / "tools/fitter/d4b_o1_fixture.py"),
                  "D4c fixture tools/fitter/d4c_fixture.py": sha256(Path(fx.__file__)),
                  "D4c gate tools/compare/d4c_start_gate.py": sha256(ROOT / "tools/compare/d4c_start_gate.py"),
                  **{f"donor {d} {d4bf.DONOR_FILES[d]}": sha256(ROOT / d4bf.DONOR_FILES[d]) for d in DONORS},
                  "model .cache/mhr/assets/compact_v6_1.model": sha256(md.ASSETS / "compact_v6_1.model"),
                  "fbx .cache/mhr/assets/lod2.fbx": sha256(md.ASSETS / "lod2.fbx"),
                  "D4c drawn set docs/reviews/body-model-start-records/drawn-set.json": sha256(D4C_DRAWN_SET),
                  "D4c development JSON docs/reviews/body-model-start-records/development.json": sha256(D4C_DEVELOPMENT),
                  "D4c acceptance manifest docs/reviews/body-model-start-records/acceptance-manifest.json":
                      sha256(D4C_MANIFEST),
                  "D4b frozen drawn set docs/reviews/body-model-o1-records/drawn-set.json": sha256(fx.D4B_DRAWN_SET),
              },
              "environment": {"momenv": "/tmp/momenv", "pymomentum-cpu": importlib.metadata.version("pymomentum-cpu"),
                              "numpy": np.__version__, "python": sys.version.split()[0]}}
    # the retained D4c cells, checked against D4c's own content records
    manifest = json.loads(D4C_MANIFEST.read_text(encoding="utf-8"))["cells_sha256"]
    development = json.loads(D4C_DEVELOPMENT.read_text(encoding="utf-8"))["cells_sha256"]
    acc = {name: sha256(D4C_CELLS["acceptance"] / name) for name in manifest}
    dev = {}
    for name in development:
        rec = json.loads((D4C_CELLS["development"] / name).read_text(encoding="utf-8"))
        dev[name] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    record["retained_d4c_cells"] = {
        "acceptance": {"count": len(manifest), "all_equal_to_the_manifest": acc == manifest},
        "development": {"count": len(development), "all_equal_to_the_development_json": dev == development,
                        "hash": "sha256 of json.dumps(record, sort_keys=True) (D4c's development convention)"},
    }
    phase2_on_disk = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "artifacts/compare").rglob("*2026120[1-6]*"))
    record["phase2_population_absent_on_disk_at_stage_1"] = {"paths_with_seeds_2026120x_under_artifacts_compare":
                                                                 len(phase2_on_disk), "examples": phase2_on_disk[:5]}
    return record


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
    parser.add_argument("--provenance", action="store_true")
    arguments = parser.parse_args()
    if arguments.provenance:
        record = stage1_provenance()
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("retained_d4c_cells", "phase2_population_absent_on_disk_at_stage_1")},
                         indent=1))
        return 0
    out = arguments.out.resolve()
    if arguments.population == "phase2":
        decision_guard()                     # before any directory or fixture exists
    out.mkdir(parents=True, exist_ok=True)
    if arguments.drive:
        default = PHASE2_ARMS if arguments.population == "phase2" else PHASE1_NEW_ARMS
        arms = arguments.arms or list(default)
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
