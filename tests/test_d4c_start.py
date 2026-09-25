"""D4c the calibration's start: the gate's arithmetic, its re-derivations and its verdict order.

Momentum-free (the gate runs on .venv). The fitter's own `landmark_start` needs pymomentum; the gate re-derives it
independently and cross-checks it on every candidate and init-only cell, so its arithmetic is tested here.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/compare"))
sys.path.insert(0, str(ROOT / "tools/fitter"))
import d4c_fixture as fx  # noqa: E402
import d4c_start_gate as gate  # noqa: E402
import d4c_identifiability as ident  # noqa: E402

DRAWN = json.loads((ROOT / "docs/reviews/body-model-start-records/drawn-set.json").read_text(encoding="utf-8"))


def test_the_frozen_drawn_set_holds_precondition_0_and_scores_the_trunk():
    pre = gate.precondition_0(DRAWN)
    assert pre["holds"] and pre["spine_drawn"] and pre["trunk_scored"]
    assert pre["drawn_set"] == DRAWN["drawn_set"]
    assert len(pre["drawn_set"]) == 8 and "scale_shoulder_width" in pre["drawn_set"]


def test_precondition_0_stops_when_the_spine_falls_below_the_rule():
    drawn = copy.deepcopy(DRAWN)
    drawn["per_donor"]["1"]["bounded_residual_median_mm"]["scale_spine_length"] = 1.99
    drawn["drawn_set"].remove("scale_spine_length")
    drawn["segments_scored"].remove("trunk")
    pre = gate.precondition_0(drawn)
    assert not pre["spine_drawn"] and not pre["holds"]


def test_precondition_0_stops_on_a_solver_failure_and_on_an_unrederivable_set():
    drawn = copy.deepcopy(DRAWN)
    drawn["precondition_0"]["solver_failures"] = [{"frame": 3}]
    assert not gate.precondition_0(drawn)["holds"]
    drawn = copy.deepcopy(DRAWN)
    drawn["drawn_set"].append("scale_foot_length")
    assert not gate.precondition_0(drawn)["holds"]


def test_the_rule_is_on_both_donors_and_unrounded():
    assert ident.rule({0: 2.0000001, 1: 3.0})
    assert not ident.rule({0: 2.0, 1: 3.0})
    assert not ident.rule({0: 5.0, 1: 1.0})


def test_pose_bounds_remove_fixed_columns_and_leave_unconfigured_unbounded():
    keep, lb, ub = ident.pose_bounds(["a", "b", "c"], np.array([0.2, 0.0, 5.0]),
                                     {"a": (-1.0, 1.0), "b": (0.0, 0.0)})
    assert keep.tolist() == [True, False, True]
    assert lb.tolist() == [-1.2, -np.inf] and ub.tolist() == [0.8, np.inf]


def test_the_bounded_residual_is_zero_inside_the_limits_and_positive_beyond_them():
    jacobian = np.eye(3)[:, :2]
    keep = np.ones(2, bool)
    residual, info = ident.bounded_residual(jacobian, np.array([0.5, -0.5, 0.0]), keep, -np.ones(2), np.ones(2))
    assert info["success"] and np.abs(residual).max() < 1e-9
    residual, _ = ident.bounded_residual(jacobian, np.array([3.0, 0.0, 0.0]), keep, -np.ones(2), np.ones(2))
    assert residual[0] == pytest.approx(2.0)


def test_calibration_frames_follow_momentums_stride():
    assert gate.calibration_frames(150, 100) == list(range(150))
    assert gate.calibration_frames(301, 100) == list(range(0, 301, 3))
    assert gate.calibration_frames(50, 100) == list(range(50))


def _array_with(lengths_mm: dict, frames: int = 150, trunk_frames: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """Landmarks in the capture frame, each segment of DRAWN's start set laid out with the given length."""
    joint_names = list(gate.MAP) + ["left_ear", "right_ear"]
    array = np.full((frames, len(joint_names), 3), np.nan)
    rng = np.random.default_rng(0)
    base = {name: rng.normal(size=3) for name in gate.MAP}
    for name in gate.MAP:
        array[:, joint_names.index(name)] = base[name]
    landmark_of = {j: l for l, j in gate.MAP.items()}
    # place each segment's second endpoint along +x from the first, at the requested length (in metres)
    for channel, segments in DRAWN["start_segments_by_drawn_channel"].items():
        for segment in segments:
            a, b = gate.SEGMENTS[segment]
            length = lengths_mm[segment] / 1000.0
            ia, ib = joint_names.index(landmark_of[a]), joint_names.index(landmark_of[b])
            array[:, ib] = array[:, ia] + np.array([length, 0.0, 0.0])
            if segment == "trunk" and trunk_frames is not None:
                array[:, ib] = array[:, ia] + np.stack([trunk_frames / 1000.0, np.zeros(frames), np.zeros(frames)], 1)
    return array, joint_names


def test_the_start_is_the_frozen_rule_on_a_constructed_body():
    # every segment at its zero-identity rest length: every start channel reads zero
    rest = DRAWN["zero_identity_rest_lengths_mm"]
    # segments are laid out independently, so build them in dependency-free order: lengths only matter per pair
    lengths = dict(rest)
    array, names = _array_with(lengths)
    got = gate.recompute_start(array, names, DRAWN, "median")
    # a segment whose endpoints are shared (shoulder width uses l_uparm/r_uparm) is laid out last-wins; read the
    # channels whose segments were laid out without a later override
    for channel in ("scale_spine_length", "scale_lowlegs", "scale_lowarms"):
        assert got[channel] == pytest.approx(0.0, abs=1e-9)


def test_the_trunk_statistic_selects_the_percentile_and_the_start_clips_to_the_limit():
    rest = DRAWN["zero_identity_rest_lengths_mm"]
    k = DRAWN["signed_rest_length_change_mm_per_unit_at_the_upper_limit"]["scale_spine_length"]["trunk"]
    trunk = np.linspace(rest["trunk"] - 50.0, rest["trunk"] + 50.0, 150)     # a chord that flexion shortens
    array, names = _array_with(dict(rest), trunk_frames=trunk)
    for statistic, q in (("median", 50.0), ("p90", 90.0), ("p95", 95.0)):
        want = (np.percentile(trunk, q, method="linear") - rest["trunk"]) / k
        assert gate.recompute_start(array, names, DRAWN, statistic)["scale_spine_length"] == pytest.approx(want, abs=1e-9)
    far = np.full(150, rest["trunk"] + 10_000.0)
    array, names = _array_with(dict(rest), trunk_frames=far)
    assert gate.recompute_start(array, names, DRAWN, "median")["scale_spine_length"] == \
        DRAWN["configured_limits"]["scale_spine_length"][1]


def test_the_acceptance_draw_is_reproducible_distinct_and_zero_off_the_drawn_set():
    limits = DRAWN["configured_limits"]
    draws = {(s, d): fx.acceptance_draw(s, d, DRAWN["drawn_set"], limits, tuple(limits))
             for s, d in fx.POPULATIONS["acceptance"]}
    assert draws[(20261101, 0)] == fx.acceptance_draw(20261101, 0, DRAWN["drawn_set"], limits, tuple(limits))
    assert len({tuple(v.values()) for v in draws.values()}) == 12
    for v in draws.values():
        assert v["scale_hip_height"] == 0.0 and v["scale_foot_length"] == 0.0
        for c in DRAWN["drawn_set"]:
            low, high = limits[c]
            assert low <= v[c] <= high


def test_the_acceptance_population_is_untouched_by_the_development_one():
    assert not set(fx.POPULATIONS["acceptance"]) & set(fx.POPULATIONS["d4"] + fx.POPULATIONS["d4b"])
    assert fx.ACCEPTANCE_ARMS == ("candidate", "exact_identity", "mean_body", "spine_displaced", "init_only", "legacy")


def test_the_displaced_spine_goes_toward_zero_and_through_it():
    assert fx.displaced_spine(1.0) == pytest.approx(0.851)
    assert fx.displaced_spine(-0.1) == pytest.approx(0.049)
    assert fx.displaced_spine(0.0) == pytest.approx(-0.149)


def _row(error: float, tolerance: float, scored: bool = True) -> dict:
    return {"segments": {"spine_displaced": {"trunk": {"error_mm": error, "tolerance_mm": tolerance, "scored": scored}},
                         "candidate": {"trunk": {"error_mm": error, "tolerance_mm": tolerance, "scored": scored}}}}


def test_the_spine_control_fails_only_when_the_trunk_is_scored_and_beyond_tolerance():
    assert gate.trunk_fails_as_scored(_row(3.0, 2.0))
    assert not gate.trunk_fails_as_scored(_row(3.0, 2.0, scored=False))      # D4b's reading B
    assert not gate.trunk_fails_as_scored(_row(1.0, 2.0))
    assert gate.trunk_ratio(_row(3.0, 2.0), "candidate") == pytest.approx(1.5)


def test_the_source_diff_accepts_only_the_two_starts():
    base = gate.subprocess.run(["git", "show", f"{gate.BASE_COMMIT}:tools/fitter/mhr_delivery.py"], cwd=ROOT,
                               capture_output=True, text=True).stdout
    if not base:
        pytest.skip("the base commit is not reachable from this checkout")
    current = gate.FITTER.read_text(encoding="utf-8")
    assert gate.source_diff({"fitter_base_source": base, "fitter_source": current})["only_the_two_starts"]
    for tamper in (("tracking.max_iter = max_iter", "tracking.max_iter = 300"),
                   ("mt.calibrate_markers(character, start.copy(), markers, stage_a)",
                    "mt.calibrate_markers(character, zero.copy(), markers, stage_a)"),
                   ("calibration.calib_frames = calib_frames", "calibration.calib_frames = 150")):
        assert tamper[0] in current
        changed = current.replace(tamper[0], tamper[1])
        assert not gate.source_diff({"fitter_base_source": base, "fitter_source": changed})["only_the_two_starts"]


def test_the_fitters_statistic_is_read_from_its_source():
    assert gate.trunk_statistic_in_source('TRUNK_STATISTIC: str | None = "p90"\n') == "p90"
    assert gate.trunk_statistic_in_source("TRUNK_STATISTIC = None\n") is None


def test_the_committed_development_record_chose_by_the_frozen_rule():
    record = json.loads((ROOT / "docs/reviews/body-model-start-records/development.json").read_text(encoding="utf-8"))
    worst = {s: record["readings"][f"candidate_{s}"]["worst_trunk_L_ratio"] for s in gate.TIE_ORDER}
    best = min(worst.values())
    assert record["chosen_trunk_statistic"] == next(s for s in gate.TIE_ORDER if worst[s] == best)
    assert record["stage_0b"]["fixtures_failing"] == record["stage_0b"]["fixtures"] == 18
    assert not record["stage_0b"]["STOP"]


GATE_REPORT = ROOT / "artifacts/compare/d4c-start/gate.json"


@pytest.mark.skipif(not GATE_REPORT.exists(), reason="the D4c artifacts are not on this machine")
def test_the_gate_reproduces_its_committed_verdict_from_the_artifacts():
    report = gate.build(gate.load_inputs())
    committed = json.loads((ROOT / "docs/reviews/body-model-start-records/gate.json").read_text(encoding="utf-8"))
    assert report["verdict"] == committed["verdict"]
    assert report["conjuncts"] == committed["conjuncts"]
