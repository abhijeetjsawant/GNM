"""D4d a second calibration pass: the gate's own arithmetic and guards, on .venv (no pymomentum).

The card is the "D4d a second calibration pass" row of docs/LADDER_EXECUTION_PLAN.md §2. These tests prove the
instruments, not the fitter: the source-diff attribution accepts exactly the second pass, the tripwire's normalisations
are only the named ones, the Phase-2 generator refuses without a committed decision, the spine terciles are the frozen
limit thirds, B2 is re-derived from the files, and (when the artifacts are present) the committed decision JSON is
re-derivable from the Phase-1 cells.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/compare"))
sys.path.insert(0, str(ROOT / "tools/fitter"))

import d4d_twopass_gate as gate  # noqa: E402
import d4d_fixture as fx4  # noqa: E402

BASE = subprocess.run(["git", "show", f"{gate.BASE_COMMIT}:tools/fitter/mhr_delivery.py"], cwd=ROOT,
                      capture_output=True, text=True).stdout
NOW = (ROOT / "tools/fitter/mhr_delivery.py").read_text(encoding="utf-8")


def test_the_base_is_the_d4c_tag_s_fitter():
    import hashlib
    if not BASE:
        pytest.skip("the D4c tag is not in this clone")
    assert hashlib.sha256(BASE.encode()).hexdigest() == gate.BASE_FITTER_SHA256


def test_the_source_diff_accepts_only_the_second_pass():
    if not BASE:
        pytest.skip("the D4c tag is not in this clone")
    diff = gate.source_diff(BASE, NOW)
    assert diff["only_the_second_pass"], diff
    assert diff["kwonly_args_added"] == ["passes"]


@pytest.mark.parametrize("old,new", [
    # pass 2 handed the live identity instead of its own copy
    ("mt.calibrate_markers(character, first.copy(), markers, stage_a)",
     "mt.calibrate_markers(character, first, markers, stage_a)"),
    # the default is not D4c's one pass
    ("start_identity: np.ndarray | None = None, passes: int = 1) -> dict:",
     "start_identity: np.ndarray | None = None, passes: int = 2) -> dict:"),
    # the shared cap moved: tracking must be untouched
    ("    tracking.max_iter = max_iter\n", "    tracking.max_iter = 2 * max_iter\n"),
    # a third pass
    ("        if passes == 2:\n", "        if passes >= 2:\n"),
    # pass 2 restarted from the start instead of pass 1's identity
    ("            first = identity.copy()\n", "            first = start.copy()\n"),
])
def test_the_source_diff_rejects_any_other_change(old, new):
    if not BASE:
        pytest.skip("the D4c tag is not in this clone")
    assert NOW.count(old) == 1, old
    assert not gate.source_diff(BASE, NOW.replace(old, new))["only_the_second_pass"]


def test_the_cli_runs_two_passes_and_the_function_defaults_to_one():
    tree = ast.parse(NOW)
    fit_one = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_one")
    assert fit_one.args.kwonlyargs[-1].arg == "passes" and fit_one.args.kw_defaults[-1].value == 1
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             and n.args and getattr(n.args[0], "value", None) == "--passes"]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    assert keywords["default"].value == 2
    assert [e.value for e in keywords["choices"].elts] == [1, 2]


def test_the_tripwire_normalises_only_the_named_fields():
    log = b"[2026-09-25 01:08:15.298] [info] Iteration: 0, error: 334.9\n"
    same = b"[2026-09-25 12:01:03.069] [info] Iteration: 0, error: 334.9\n"
    moved = b"[2026-09-25 12:01:03.069] [info] Iteration: 0, error: 334.8\n"
    assert gate.normalised_equal("fixture-x.calibration-debug.log", log, same) == (True, "normalised momentum timestamps")
    assert not gate.normalised_equal("fixture-x.calibration-debug.log", log, moved)[0]
    cell = {"provenance": {"fitter_sha256": "a", "this_file_sha256": "b"}, "fitted_identity": [0.1]}
    other_fitter = json.loads(json.dumps(cell))
    other_fitter["provenance"]["fitter_sha256"] = "c"
    other_file = json.loads(json.dumps(cell))
    other_file["provenance"]["this_file_sha256"] = "c"
    other_value = json.loads(json.dumps(other_fitter))
    other_value["fitted_identity"] = [0.2]
    enc = lambda x: json.dumps(x).encode()  # noqa: E731
    assert gate.normalised_equal("cell-1-d0-candidate.json", enc(cell), enc(other_fitter))[0]
    assert not gate.normalised_equal("cell-1-d0-candidate.json", enc(cell), enc(other_file))[0]
    assert not gate.normalised_equal("cell-1-d0-candidate.json", enc(cell), enc(other_value))[0]
    track = {"body_model": {"assets": "/a"}, "x": 1}
    assert gate.normalised_equal("s.body-track.json", enc(track), enc({"body_model": {"assets": "/b"}, "x": 1}))[0]
    assert not gate.normalised_equal("s.body-track.json", enc(track), enc({"body_model": {"assets": "/b"}, "x": 2}))[0]
    assert not gate.normalised_equal("s.glb", b"a", b"b")[0]
    assert gate.normalised_equal("s.glb", b"a", None) == (False, "missing")


def test_the_tripwire_checks_its_sets_by_identity():
    files = {f"delivery/{n}": (b"x", b"x") for n in gate.D4C_DELIVERY_FILES}
    files.update({f"acceptance/f{i}.npz": (b"y", b"y") for i in range(gate.D4C_ACCEPTANCE_FILES)})
    assert gate.tripwire(files)["holds"]
    short = dict(files)
    short.pop(next(k for k in short if k.startswith("acceptance/")))
    assert not gate.tripwire(short)["holds"]
    renamed = dict(files)
    renamed["delivery/subject-02.glb"] = renamed.pop("delivery/subject-00.glb")
    assert not gate.tripwire(renamed)["holds"]


def test_the_spine_terciles_are_the_frozen_limit_thirds():
    assert fx4.spine_tercile(-0.911) == "short"
    assert fx4.spine_tercile(0.0) == "middle"
    assert fx4.spine_tercile(1.0) == "long"
    edge = -1.1 + 2.2 / 3.0
    assert fx4.spine_tercile(edge) == "middle" and fx4.spine_tercile(-edge) == "middle"


def test_the_phase_2_generator_refuses_without_a_committed_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(fx4, "DECISION", tmp_path / "phase1-decision.json")
    with pytest.raises(SystemExit, match="no Phase-1 decision"):
        fx4.decision_guard()
    record = {"selected": "TWO-PASS", "phase_2_licensed": True}
    (tmp_path / "phase1-decision.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(fx4, "git_committed_bytes", lambda path: None)
    with pytest.raises(SystemExit, match="not committed"):
        fx4.decision_guard()
    monkeypatch.setattr(fx4, "git_committed_bytes", lambda path: b"something else")
    with pytest.raises(SystemExit, match="not committed"):
        fx4.decision_guard()
    stop = {"selected": "STOP-WARM-drift", "phase_2_licensed": False}
    (tmp_path / "phase1-decision.json").write_text(json.dumps(stop), encoding="utf-8")
    monkeypatch.setattr(fx4, "git_committed_bytes", lambda path: json.dumps(stop).encode())
    with pytest.raises(SystemExit, match="does not license"):
        fx4.decision_guard()
    (tmp_path / "phase1-decision.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(fx4, "git_committed_bytes", lambda path: json.dumps(record).encode())
    import hashlib
    assert fx4.decision_guard() == hashlib.sha256(json.dumps(record).encode()).hexdigest()


def _synthetic_delivery():
    names = ["root", "neck", "nose"]
    frames = 3
    array = np.arange(frames * 3 * 3, dtype=np.float64).reshape(frames, 3, 3) / 10.0
    array[1, 2] = np.nan
    declared = {"root": "root", "neck": "c_neck"}
    cm = gate._mhr_cm(array)
    positions = np.zeros((frames, 2, 3))
    occluded = np.zeros((frames, 2), bool)
    for c, landmark in enumerate(declared):
        values = cm[:, names.index(landmark)]
        finite = np.isfinite(values).all(axis=1)
        positions[finite, c] = values[finite]
        occluded[:, c] = ~finite
    subject = {"consumed": {"triangulated_world_positions_z_up_m": array, "joint_names": np.array(names),
                            "ticks": np.arange(frames)},
               "rig_track": {"triangulated_world_positions_z_up_m": array.copy(), "ticks": np.arange(frames)},
               "markers": {"source_array_key": np.array("triangulated_world_positions_z_up_m"),
                           "source_joint_names": np.array(names), "marker_positions_mhr_cm": positions,
                           "marker_occluded": occluded, "marker_names": np.array(list(declared))},
               "track": {"consumed_joint_names": np.array(names)},
               "track_json": {"landmark_to_joint": declared}}
    return {"rig_converter_input": {"joint_names": np.array(names),
                                    **{f"{s}_triangulated_world_positions_z_up_m": array.copy() for s in gate.SUBJECTS}},
            "subjects": {s: copy.deepcopy(subject) for s in gate.SUBJECTS}}


def test_b2_is_rederived_from_the_files():
    delivery = _synthetic_delivery()
    assert gate.b2_rederived(delivery)["pass"]
    moved = _synthetic_delivery()
    moved["subjects"]["subject_01"]["markers"]["marker_positions_mhr_cm"] = \
        moved["subjects"]["subject_01"]["markers"]["marker_positions_mhr_cm"] + 1e-6
    assert not gate.b2_rederived(moved)["pass"]
    cast = _synthetic_delivery()
    cast["rig_converter_input"]["subject_00_triangulated_world_positions_z_up_m"] = \
        cast["rig_converter_input"]["subject_00_triangulated_world_positions_z_up_m"].astype(np.float32)
    assert not gate.b2_rederived(cast)["pass"]
    missing = _synthetic_delivery()
    missing["subjects"]["subject_00"]["markers"] = None
    assert not gate.b2_rederived(missing)["pass"]


def test_the_committed_decision_is_rederivable_from_the_phase_1_cells():
    if not gate.DECISION.exists() or not (gate.OUT_DIR / "phase1").is_dir():
        pytest.skip("the Phase-1 cells or the decision are not on this machine")
    inputs = gate.load_phase1()
    derived = json.loads(json.dumps(gate.phase1(inputs), default=float))
    assert derived == json.loads(gate.DECISION.read_text(encoding="utf-8"))


def test_the_gate_reproduces_its_verdict_from_the_artifacts():
    if not (gate.OUT_DIR / "phase2").is_dir() or not gate.DECISION.exists():
        pytest.skip("the D4d artifacts are not on this machine")
    report = gate.build(gate.load_phase1(), gate.load_phase2(), gate.load_hygiene())
    assert report["verdict"] == "PASS", report["reason"]
    assert all(report["conjuncts"].values())


def test_a_malformed_phase_2_cell_fails_closed_rather_than_crashing():
    if not (gate.OUT_DIR / "phase2").is_dir() or not gate.DECISION.exists():
        pytest.skip("the D4d artifacts are not on this machine")
    p1, p2, h = gate.load_phase1(), gate.load_phase2(), gate.load_hygiene()
    cell = p2["cells"]["cell-20261201-d1-candidate.json"]
    cell["calibration_calls"] = cell["calibration_calls"][:1]
    assert gate.build(p1, p2, h)["verdict"] == "FAIL"
