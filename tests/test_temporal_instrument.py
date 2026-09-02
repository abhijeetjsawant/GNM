"""Unit tests for I7's temporal instrument and its ladder extractor.

The instrument itself needs the retained MAMMA run, the SOMA-77 observations and a few
minutes; these test the parts that can be wrong *silently* -- the wrappers that ablate a
pipeline stage and must put it back, the masks that decide which cell is starved, the
fixed frame set every lag figure depends on, and the extractor's two-sources contract --
on synthetic data that needs nothing under `artifacts/`.

The one thing worth stating about what is tested here: three of these guard defects that
would have produced a *plausible* report rather than an error. A wrapper that failed to
restore `solve_sequence_positions` would leave every later arm silently ablated; a lag
sweep whose frame set moved with the shift would put the minimum wherever the easy frames
are (the bug I6 hit); and a gap spacing that does not scale with the take length reported
"0 cells" on a 28-frame clip while looking as though the arm had run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    for relative in ("src", "tools/compare", "tools/head", "scripts",
                     "workers/commercial_multiview"):
        sys.path.insert(0, str(ROOT / relative))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


temporal = _module("i7_temporal_instrument", ROOT / "tools/compare/temporal.py")
extractor = _module("i7_temporal", ROOT / "tools/compare/extractors/i7_temporal.py")
cm = temporal.cm


# ----------------------------------------------------- the wrappers: ablate, then restore

def test_sequence_solve_wrapper_calls_the_real_function_and_restores_it():
    original = cm.solve_sequence_positions
    with temporal.sequence_solve("real") as log:
        assert cm.solve_sequence_positions is not original
        assert log.entries == []
    assert cm.solve_sequence_positions is original


def test_sequence_solve_wrapper_restores_even_when_the_body_raises():
    original = cm.solve_sequence_positions
    with pytest.raises(ValueError):
        with temporal.sequence_solve("off"):
            raise ValueError("boom")
    assert cm.solve_sequence_positions is original


def test_sequence_solve_off_is_a_pass_through_and_recovers_nothing():
    """The interpolation-only control must not move a single position."""
    world = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
    world[1, 1] = np.nan
    observations = np.full((2, 1, 3, 3), np.nan)
    with temporal.sequence_solve("off") as log:
        out, recovered = cm.solve_sequence_positions([object()], world, observations)
    assert np.array_equal(out, world, equal_nan=True)
    assert not recovered.any()
    assert log.entries[0]["direct"].tolist() == [[True, True, True], [True, False, True]]


def test_smoothing_window_sets_and_restores_the_shipped_constant():
    original = cm.SMOOTHING_WINDOW_FRAMES
    with temporal.smoothing_window(27):
        assert cm.SMOOTHING_WINDOW_FRAMES == 27
    assert cm.SMOOTHING_WINDOW_FRAMES == original


def test_the_identity_window_really_skips_savitzky_golay():
    """`WINDOW_IDENTITY` must land below the filter's own minimum, or the control
    is not a control -- it is a narrower filter."""
    assert temporal.WINDOW_IDENTITY < 5
    values = np.zeros((30, len(cm.JOINT_NAMES), 3))
    values[:, :, 0] = np.arange(30)[:, None] ** 2      # a curve savgol would alter
    with temporal.smoothing_window(temporal.WINDOW_IDENTITY):
        untouched, _ = cm._fill_and_smooth_positions(values)
    with temporal.smoothing_window(temporal.WINDOW_DEFAULT):
        filtered, _ = cm._fill_and_smooth_positions(values)
    assert np.allclose(untouched, values)
    # polyorder 2 reproduces a quadratic exactly, so assert on a signal it cannot
    values[:, :, 1] = np.sin(np.arange(30) * 0.9)[:, None]
    with temporal.smoothing_window(temporal.WINDOW_IDENTITY):
        untouched, _ = cm._fill_and_smooth_positions(values)
    with temporal.smoothing_window(temporal.WINDOW_DEFAULT):
        filtered, _ = cm._fill_and_smooth_positions(values)
    assert np.allclose(untouched, values)
    assert not np.allclose(filtered, values)


# --------------------------------------------------------------------------- the masks

def _records(frames: int, cameras: int = 4, subjects: int = 2) -> list[list[dict]]:
    names = ["root", "left_wrist", "right_wrist"]
    return [[{"schema_version": "x", "detector": "t", "frame_index": frame,
              "width": 1280, "height": 720, "image_path": "t://",
              "people": [{"index": s, "joints": {n: {"x": 10.0 * s, "y": 1.0 * frame,
                                                     "confidence": 0.95} for n in names}}
                         for s in range(subjects)]}
             for frame in range(frames)] for _ in range(cameras)]


def test_apply_keep_mask_deletes_exactly_the_cells_it_is_given():
    records = _records(5)
    keep = np.ones((2, 5, 4, len(cm.JOINT_NAMES)), dtype=bool)
    keep[1, 3, 2, cm.JOINT_INDEX["left_wrist"]] = False
    out = temporal.apply_keep_mask(records, keep)
    assert "left_wrist" not in out[2][3]["people"][1]["joints"]
    assert "left_wrist" in out[2][3]["people"][0]["joints"]
    assert "left_wrist" in out[1][3]["people"][1]["joints"]
    assert "left_wrist" in out[2][2]["people"][1]["joints"]
    # and the input is untouched, so an arm cannot poison the next one
    assert "left_wrist" in records[2][3]["people"][1]["joints"]


def test_apply_displacements_moves_exactly_the_cells_it_is_given():
    records = _records(4)
    key = (0, 2, 1, cm.JOINT_INDEX["right_wrist"])
    out = temporal.apply_displacements(records, {key: (7.0, -3.0)})
    moved = out[1][2]["people"][0]["joints"]["right_wrist"]
    before = records[1][2]["people"][0]["joints"]["right_wrist"]
    assert moved["x"] == before["x"] + 7.0 and moved["y"] == before["y"] - 3.0
    # the other subject, the other camera and the other frames are all untouched
    assert (out[1][2]["people"][1]["joints"]["right_wrist"]
            == records[1][2]["people"][1]["joints"]["right_wrist"])
    assert out[0][2]["people"][0]["joints"]["right_wrist"] == before
    assert out[1][1]["people"][0]["joints"]["right_wrist"] == records[1][1]["people"][0]["joints"]["right_wrist"]


def test_check_person_order_refuses_a_frame_that_dropped_a_subject():
    records = _records(3)
    records[0][1]["people"] = records[0][1]["people"][:1]
    with pytest.raises(SystemExit, match="no longer means the subject index"):
        temporal.check_person_order(records, 2)


def test_root_is_never_starved_by_any_outage_model():
    """A starved root is whole-frame loss, not single-ray recovery. If root ever
    enters a mask the recovery figures silently become something else."""
    seen = np.ones((2, 40, 4, len(cm.JOINT_NAMES)), dtype=bool)
    seen[:, :, 1:] = False                     # only camera 0 sees anything
    names = ("root", "left_wrist", "right_wrist")
    root = cm.JOINT_INDEX["root"]
    for keep in (temporal.replayed_keep(seen, 20, names, 0),
                 temporal.amplified_keep(seen, 20, names, 0, 2)[0]):
        assert keep[:, :, :, root].all()
    assert temporal.iid_keep(np.ones((2, 20, 4, len(cm.JOINT_NAMES)), dtype=bool),
                             names, seed=1)[:, :, :, root].all()


def test_mask_is_runnable_names_a_joint_that_never_reaches_two_views():
    keep = np.ones((2, 20, 4, len(cm.JOINT_NAMES)), dtype=bool)
    assert temporal.mask_is_runnable(keep, ("left_wrist",)) == []
    keep[0, :, 1:, cm.JOINT_INDEX["left_wrist"]] = False    # one camera, whole take
    assert temporal.mask_is_runnable(keep, ("left_wrist",)) == ["left_wrist"]


def test_iid_arm_matches_the_correlated_arm_s_per_view_miss_rate():
    """Same denominator: the two outage models must differ in STRUCTURE, not rate."""
    rng = np.random.default_rng(3)
    correlated = np.ones((2, 60, 4, len(cm.JOINT_NAMES)), dtype=bool)
    names = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    columns = [cm.JOINT_INDEX[n] for n in names]
    correlated[:, :, :, columns] = rng.random((2, 60, 4, len(columns))) < 0.7
    independent = temporal.iid_keep(correlated, names, seed=5)
    assert abs(float(correlated[:, :, :, columns].mean())
               - float(independent[:, :, :, columns].mean())) < 0.05


# ------------------------------------------------------------------- the lag frame set

def _straight_line(frames: int, joints: int) -> np.ndarray:
    truth = np.zeros((2, frames, len(cm.JOINT_NAMES), 3))
    for index, name in enumerate(joints):
        truth[:, :, cm.JOINT_INDEX[name], 0] = np.arange(frames) * (0.01 * (index + 1))
        truth[:, :, cm.JOINT_INDEX[name], 1] = np.sin(np.arange(frames) * 0.4) * 0.05
    return truth


def test_lag_sweep_scores_one_fixed_frame_set_at_every_shift():
    """The bug I6 hit. If the population moved with the shift the curve would be
    comparing different frames, and the minimum would land on the easy ones."""
    names = ("left_wrist", "right_wrist")
    truth = _straight_line(40, names)
    result = temporal.lag_sweep(truth.copy(), truth, {0: 0, 1: 1}, names)
    span = result["frame_set"]
    assert span == [temporal.LAG_SWEEP, 40 - temporal.LAG_SWEEP]
    assert result["frames_scored"] == span[1] - span[0]


def test_lag_sweep_recovers_a_lag_that_was_injected():
    names = ("left_wrist", "right_wrist")
    truth = _straight_line(60, names)
    lagged = np.roll(truth, 2, axis=1)          # ours[t] = truth[t-2]: a 2-frame lag
    result = temporal.lag_sweep(lagged, truth, {0: 0, 1: 1}, names)
    assert result["argmin_frames"] == 2
    assert result["rms_mm_by_shift"]["2"] < result["rms_mm_by_shift"]["0"]


def test_lag_sweep_time_shifted_truth_control_fails_on_a_lag_free_arm():
    names = ("left_wrist", "right_wrist")
    truth = _straight_line(40, names)
    result = temporal.lag_sweep(truth.copy(), truth, {0: 0, 1: 1}, names)
    control = result["control_time_shifted_truth"]
    assert control["both_shifts_worse_than_zero"] is True
    assert control["zero_rms_mm"] == pytest.approx(0.0, abs=1e-6)


def test_peak_attenuation_is_zero_when_nothing_is_filtered_and_positive_when_flattened():
    names = ("left_wrist", "right_wrist")
    truth = _straight_line(60, names)
    assert temporal.peak_attenuation(truth.copy(), truth, {0: 0, 1: 1},
                                     names)["attenuation_median"] == pytest.approx(0.0, abs=1e-9)
    flattened = truth * 0.5
    assert temporal.peak_attenuation(flattened, truth, {0: 0, 1: 1},
                                     names)["attenuation_median"] > 0.4


# ------------------------------------------------------------- take-length sensitivity

def test_gap_cells_places_gaps_on_a_take_as_short_as_the_fk_clips():
    """A fixed 20-frame stride put ZERO gaps in a 28-frame stride-3 clip and the arm
    reported nothing while looking as though it had run."""
    names = tuple(cm.JOINT_NAMES)
    for frames in (28, 32, 150):
        truth = np.zeros((2, frames, len(cm.JOINT_NAMES), 3))
        for length in temporal.GAP_LENGTHS:
            cells = temporal.gap_cells(truth, names, length)
            assert cells, f"no gap placed at {frames} frames, length {length}"
            assert max(frame for _s, frame, _j in cells) < frames


def test_block_bootstrap_pairs_the_two_arms_on_identical_draws():
    frames = 40
    rng = np.random.default_rng(9)
    a = [rng.normal(10.0, 1.0, 6) for _ in range(frames)]
    b = [values + 5.0 for values in a]          # b is worse on every single cell
    result = temporal.block_bootstrap_pair(a, b)
    assert result["p_candidate_beats_control"] == 1.0
    assert result["blocks_per_draw"] >= 2
    # and identical inputs must not manufacture a margin
    tied = temporal.block_bootstrap_pair(a, [values.copy() for values in a])
    assert tied["p_candidate_beats_control"] == 0.0


def test_lag1_autocorrelation_is_measured_not_assumed():
    assert temporal.lag1_autocorrelation(np.arange(50.0)) > 0.9
    assert abs(temporal.lag1_autocorrelation(
        np.random.default_rng(1).normal(size=4000))) < 0.1
    assert temporal.lag1_autocorrelation(np.array([1.0, 1.0])) is None


def test_truth19_gathers_into_the_pipeline_s_own_joint_order():
    source = np.arange(2 * 4 * 77 * 3, dtype=np.float64).reshape(2, 4, 77, 3)
    out = temporal.truth19_from(source, temporal.SOMA77_TO_AUTOANIM)
    assert out.shape == (2, 4, len(cm.JOINT_NAMES), 3)
    for name, index in temporal.SOMA77_TO_AUTOANIM.items():
        assert np.array_equal(out[:, :, cm.JOINT_INDEX[name]], source[:, :, index])
    # SOMA-77 emits no ears; they must arrive NaN rather than zero
    assert np.isnan(out[:, :, cm.JOINT_INDEX["left_ear"]]).all()


# --------------------------------------------------------------------------- extractor

def _minimal_report() -> dict:
    arm = {"cells": {"total": 100, "single_ray": 10, "single_ray_fraction": 0.1,
                     "no_ray": 5, "no_ray_fraction": 0.05, "recovered": 9,
                     "demoted_to_interpolation": 1},
           "error_on_recovered_cells": {"n": 9, "median_mm": 7.0},
           "control_interpolation_only_on_recovered_cells": {"n": 9, "median_mm": 51.0},
           "margin_on_recovered_cells": {"draws": 1000, "block_frames": 15,
                                         "p_candidate_beats_control": 1.0},
           "lag1_autocorrelation_of_per_frame_recovered_median": 0.74}
    window = {"error": {"median_mm": 1.1},
              "phase_lag": {"argmin_subframe": 0.0, "rms_mm_by_shift": {"0": 1.0}},
              "peak_attenuation": {"attenuation_median": 0.01, "true_peak_speed_m_s": 1.5}}
    source = {
        "reference": "a reference string",
        "oracle_no_drops": {"after_the_temporal_stage": {"median_mm": 1.15},
                            "raw_max_micrometres": 1e-8},
        "recovery_5a": {"arms": {"correlated_amplified": arm, "iid_same_rate": arm},
                        "controls": {"frozen_trajectory": {
                            "error_against_the_real_trajectory": {"median_mm": 694.0}}}},
        "smoothing_5b": {"spike_amplitude_px1280": 24.7,
                         "clean_coverage": {"shipped_window_9": window,
                                            "over_smoothed_window_27": window,
                                            "identity_no_savitzky_golay": window},
                         "spikes": {}, "gaps": {}},
    }
    return {"real_run_occlusion_pattern": {"single_ray_slots": 6,
                                           "camera_support_histogram_0_to_4": [0, 6, 1165,
                                                                               1009, 2920]},
            "sources": {"fk_synthetic": source, "mamma_pred_joints": source},
            "held_out_camera_lag": {"folds": {"A001": {"ran": True, "argmin_subframe": -0.18,
                                                       "scored_slots": 4887},
                                              "B001": {"ran": False, "reason": "stated"}}}}


def test_extractor_returns_the_x_star_shape(tmp_path, monkeypatch):
    path = tmp_path / "artifacts/compare/temporal.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_minimal_report()))
    monkeypatch.setattr(extractor, "ROOT", tmp_path)
    figures, controls = extractor.x_temporal({})
    assert figures and controls
    for row in figures + controls:
        assert set(row) == {"key", "label", "value", "unit", "reference", "better", "note"}
        assert row["better"] in (extractor.LOWER, extractor.HIGHER)
        assert row["reference"]
    assert len({row["key"] for row in figures + controls}) == len(figures + controls)


def test_extractor_keeps_the_two_sources_apart():
    """The MAMMA-derived arm and the MAMMA-free arm must never collide on a key,
    or the ladder could put them on one axis."""
    figures, controls = extractor.x_temporal({})
    keys = [row["key"] for row in figures + controls]
    if not keys:                                   # no report on disk in this checkout
        pytest.skip("artifacts/compare/temporal.json absent")
    assert any(key.startswith("fk_") for key in keys)
    assert any(key.startswith("mamma_") for key in keys)


def test_extractor_is_silent_without_a_report(monkeypatch, tmp_path):
    monkeypatch.setattr(extractor, "ROOT", tmp_path)
    assert extractor.x_temporal({}) == ([], [])
    assert extractor.stamp()["exists"] is False
