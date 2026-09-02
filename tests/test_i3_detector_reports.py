"""Unit tests for I3's three detector instruments and their ladder extractor.

The instruments themselves need MAMMA's retained run and a minute or two each; these
test the parts that can be wrong *silently* -- the preregistered decision rule, the
statistics, the epipolar expression the whole first report rests on, the semantic
exclusions, and the extractor's three-references contract -- on synthetic data and
temporary files that need nothing under `artifacts/`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools" / "swap-harness"))
    sys.path.insert(0, str(ROOT / "tools" / "head"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decision = _module("i3_decision", ROOT / "tools/swap-harness/i3_decision.py")
self_agreement = _module("i3_sam3d_ladder", ROOT / "tools/swap-harness/sam3d_ladder.py")
vs_mamma = _module("i3_mamma_residuals", ROOT / "tools/swap-harness/mamma_residuals.py")
extractor = _module("i3_detector", ROOT / "tools/compare/extractors/i3_detector.py")
cm = decision.cm


# ------------------------------------------------------------------ the decision rule

def test_the_rule_is_declared_preregistered_and_names_three_destinations():
    rule = decision.DECISION_RULE
    assert rule["written_before_any_figure_was_computed"] is True
    assert set(rule["destinations"]) == {
        "a_pseudo_label_campaign", "b_per_camera_offset_fix", "c_neither"}
    # every threshold ships with the reason it is that number, not a bare value
    for key in ("a_pseudo_label_campaign", "b_per_camera_offset_fix"):
        assert rule["thresholds"][key]["why_these_numbers"]
    # the rule must say in so many words that it authorises no shipped constant
    disclaimer = rule["what_this_rule_does_not_authorise"].lower()
    assert "never enters the delivery path" in disclaimer or "mamma-free" in disclaimer


def test_rule_reaches_all_four_verdicts():
    strong_b = dict(g_heldout=0.30, p_heldout=0.99, d_halves=0.10,
                    g_shuffled_cameras=0.01, g_ceiling=0.90)
    assert decision.evaluate_decision_rule(**strong_b)["verdict"] == "b"
    assert decision.evaluate_decision_rule(**{**strong_b, "g_ceiling": 0.30})["verdict"] == "a+b"
    weak = dict(g_heldout=0.02, p_heldout=0.10, d_halves=0.90,
                g_shuffled_cameras=0.02, g_ceiling=0.90)
    assert decision.evaluate_decision_rule(**weak)["verdict"] == "c_neither"
    assert decision.evaluate_decision_rule(**{**weak, "g_ceiling": 0.10})["verdict"] == "a"


def test_a_shuffled_camera_control_that_reproduces_the_gain_fails_clause_four():
    """The case this fixture actually produced: everything else passes, clause 4 does not."""
    result = decision.evaluate_decision_rule(
        g_heldout=0.127, p_heldout=1.0, d_halves=0.48,
        g_shuffled_cameras=0.1147, g_ceiling=0.175)
    clauses = result["b_per_camera_offset_fix"]["clauses"]
    assert clauses["G_heldout >= 0.10"] and clauses["P_heldout >= 0.95"]
    assert clauses["D_halves <= 0.50"]
    assert not clauses["G_shuffled_cameras < 0.5 * G_heldout"]
    assert result["b_per_camera_offset_fix"]["triggered"] is False
    assert result["verdict"] == "a"


def test_missing_variables_do_not_silently_trigger_a_destination():
    result = decision.evaluate_decision_rule(
        g_heldout=None, p_heldout=None, d_halves=None,
        g_shuffled_cameras=None, g_ceiling=None)
    assert result["verdict"] == "c_neither"


# ---------------------------------------------------------------------- the statistics

def test_lag1_autocorrelation_recovers_an_ar1_coefficient():
    rng = np.random.default_rng(0)
    for phi in (0.3, 0.9):
        series = np.zeros(4000)
        for index in range(1, series.size):
            series[index] = phi * series[index - 1] + rng.normal()
        assert abs(decision.lag1_autocorrelation(series) - phi) < 0.05


def test_lag1_autocorrelation_is_none_on_a_constant_or_a_stub():
    assert decision.lag1_autocorrelation(np.full(50, 3.0)) is None
    assert decision.lag1_autocorrelation(np.asarray([1.0, 2.0])) is None
    assert decision.lag1_autocorrelation(np.full(50, np.nan)) is None


def test_block_draws_are_identical_for_every_arm_and_stay_in_range():
    first = decision.block_draws(150, seed=7, draws=20)
    second = decision.block_draws(150, seed=7, draws=20)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert all(index.size == 150 and index.min() >= 0 and index.max() < 150
               for index in first)


def test_paired_bootstrap_is_decisive_when_an_arm_is_uniformly_better_or_equal():
    rng = np.random.default_rng(1)
    baseline = rng.uniform(10.0, 20.0, (150, 8))
    arms = {"base": baseline, "better": baseline - 1.0, "same": baseline.copy()}
    out = decision.paired_block_bootstrap(arms, baseline="base",
                                          candidates=("better", "same"))
    assert out["arms"]["better"]["P_beats_baseline"] == 1.0
    assert out["arms"]["same"]["P_beats_baseline"] == 0.0
    assert out["block_frames"] == decision.BLOCK_FRAMES


# ------------------------------------------------------- the epipolar expression itself

def _ring_rig():
    cameras = []
    for index, (x, y) in enumerate(((0.0, -4.0), (4.0, 0.0), (0.0, 4.0), (-4.0, 0.0))):
        center = np.asarray((x, y, 1.6))
        forward = np.asarray((0.0, 0.0, 1.0)) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        cameras.append(cm.CalibratedCamera(
            name=decision.CAMERAS[index], width=1280, height=720,
            intrinsics=np.asarray(((900.0, 0.0, 640.0), (0.0, 900.0, 360.0),
                                   (0.0, 0.0, 1.0))),
            camera_center_world_m=center,
            camera_to_world_xyzw=Rotation.from_matrix(
                np.column_stack((right, down, forward))).as_quat()))
    return tuple(cameras)


def test_per_joint_expression_medians_to_the_pipelines_own_number():
    """The whole first report rests on expanding `_epipolar_distance_px` per joint."""
    cameras = _ring_rig()
    fundamental = cm._fundamental_matrix(cameras[0], cameras[1])
    rng = np.random.default_rng(4)
    source = np.column_stack((rng.uniform(0, 1280, 19), rng.uniform(0, 720, 19),
                              np.full(19, 0.9)))
    target = np.column_stack((rng.uniform(0, 1280, 19), rng.uniform(0, 720, 19),
                              np.full(19, 0.9)))
    mine, to_target, to_source = self_agreement.symmetric_epipolar(
        fundamental, source, target)
    theirs = cm._epipolar_distance_px(fundamental, source, target,
                                      minimum_confidence=0.25, minimum_shared_joints=4)
    assert np.median(mine) == pytest.approx(theirs, abs=1e-9)
    # and it really is the SUM of two one-sided distances, which is why the report halves it
    assert np.allclose(mine, to_target + to_source)


def test_a_perfectly_consistent_3d_point_has_zero_epipolar_distance():
    """The frozen-skeleton control's premise: any 3D-consistent thing scores 0."""
    cameras = _ring_rig()
    fundamental = cm._fundamental_matrix(cameras[0], cameras[1])
    rng = np.random.default_rng(6)
    points = rng.uniform(-0.5, 0.5, (19, 3)) + np.asarray((0.0, 0.0, 1.2))
    left, _ = cameras[0].project(points)
    right, _ = cameras[1].project(points)
    source = np.column_stack((np.asarray(left), np.full(19, 0.9)))
    target = np.column_stack((np.asarray(right), np.full(19, 0.9)))
    symmetric, _, _ = self_agreement.symmetric_epipolar(fundamental, source, target)
    assert float(np.max(symmetric)) < 1e-6


# ------------------------------------------------------------- the semantic exclusions

def test_five_of_the_nineteen_contract_joints_are_excluded_and_named():
    assert len(decision.PAIRS) == 15
    assert len(decision.SCORED_VS_MAMMA) == 14
    assert "nose" not in decision.SCORED_VS_MAMMA
    assert set(decision.SEMANTIC_EXCLUSIONS) == {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear"}
    assert set(decision.SEMANTIC_EXCLUSIONS) | set(decision.SCORED_VS_MAMMA) == set(
        cm.JOINT_NAMES)
    for reason in decision.SEMANTIC_EXCLUSIONS.values():
        assert len(reason) > 30          # a reason, not a label


def test_left_right_swap_is_an_involution_that_touches_only_lateral_joints():
    names = list(decision.SCORED_VS_MAMMA)
    swapped = vs_mamma.swap_left_right(names)
    assert vs_mamma.swap_left_right(swapped) == names
    for original, crossed in zip(names, swapped):
        if original.startswith(("left_", "right_")):
            assert original != crossed and original.split("_", 1)[1] == crossed.split("_", 1)[1]
        else:
            assert original == crossed


# --------------------------------------------------------------------- the extractor

def _self_agreement_report() -> dict:
    return {
        "headline": {"one_sided_epipolar_median_px1280": 2.768, "shuffled_over_ours": 33.9},
        "units": {"measured_symmetric_over_one_sided_ratio": {"median": 1.983}},
        "gate": {"G1_discrimination": "shuffled / candidate >= 5.0"},
        "arms": {
            "ours": {"one_sided_median_px1280": 2.768, "liveness_px_per_frame": 3.179,
                     "one_sided_quantiles_px1280": {"95": 9.385}},
            "CONTROL_shuffled_cross_view_pairing": {"one_sided_median_px1280": 93.797,
                                                    "liveness_px_per_frame": 3.179},
            "CONTROL_frozen_skeleton_projected": {"one_sided_median_px1280": 0.0,
                                                  "liveness_px_per_frame": 0.0},
            "CONTROL_our_own_3d_reprojected": {"one_sided_median_px1280": 0.0,
                                               "liveness_px_per_frame": 2.638},
        },
        "population": {"what_the_common_denominator_costs": {
            "camera_slots_the_associator_left_unassigned": "144 of 1200 slots"}},
    }


def _common_mode_report() -> dict:
    def arm(median, gain):
        return {"median_mm": median, "gain_vs_as_measured": gain}
    return {
        "why_the_baseline_is_not_rung_7s_number": "raw re-triangulation, not the "
                                                  "delivered smoothed track",
        "are_the_offsets_the_same_size_on_both_halves": "Plainly: NEARLY.",
        "THE_VERDICT": {
            "values": {"G_heldout": 0.127, "P_heldout": 1.0, "D_halves": 0.48,
                       "G_shuffled_cameras": 0.1147, "G_ceiling": 0.1751,
                       "share_surviving_the_ceiling": 0.8249},
            "a_pseudo_label_campaign": {"triggered": True, "clauses": {}},
            "b_per_camera_offset_fix": {"triggered": False, "clauses": {"c4": False}},
            "verdict": "a",
        },
        "3d_error_vs_mamma_projections_mm": {
            "as_measured": arm(43.307, 0.0),
            "SPLIT_HALF_held_out": arm(37.809, 0.127),
            "CONTROL_same_frames_fit_and_score": arm(36.653, 0.1536),
            "CONTROL_interleaved_half_held_out": arm(36.976, 0.1462),
            "CONTROL_shuffled_camera_assignment": arm(38.338, 0.1147),
            "POSTHOC_global_single_offset_held_out": arm(37.285, 0.1391),
            "POSTHOC_CONTROL_global_offset_rotated_90deg": arm(42.399, 0.021),
            "POSTHOC_CONTROL_global_offset_negated": arm(55.957, -0.2921),
            "ORACLE_full_common_mode_removed": arm(35.725, 0.1751),
        },
        "what_clause_4_of_the_rule_revealed": {
            "global_single_offset_px1280": {"held_out_gain": 0.1391,
                                            "four_number_held_out_gain": 0.127},
            "and_how_that_rule_came_out": {"verdict": "NOT MET",
                                           "and_it_is_left_failing_on_purpose": "p95 says "
                                                                                "otherwise"},
        },
    }


def _vs_mamma_report() -> dict:
    return {
        "arms": {
            "ours_vs_mamma_projections": {"median_px1280": 6.138, "n": 13907,
                                          "quantiles_px1280": {"95": 15.642},
                                          "shape": {"better": "two-point mixture"}},
            "MAMMA_own_residual_DEFLATED": {"median_px3840": 4.29,
                                            "cross_check": {"identical": True}},
        },
        "controls": {
            "CONTROL_including_the_semantic_mismatch_nose": {"median_px1280": 6.127},
            "CONTROL_shuffled_subject_pairing": {"median_px1280": 178.642},
            "CONTROL_frozen_frame0_detection": {"median_px1280": 99.455},
            "CONTROL_left_right_swapped": {"median_px1280": 35.315},
        },
        "RETIRED_HEADLINE": {"what_it_actually_compared": {
            "numerator": "our detector against raw triangulation",
            "denominator": "MAMMA against a body-regularised fit that consumed it"}},
        "population": {"scored_against_mamma": 14, "joints_excluded_count": 5},
        "what_the_nose_exclusion_turned_out_to_cost": {
            "finding": "it does not discriminate", "why_it_stays_excluded": "conventions"},
    }


@pytest.fixture
def wired(tmp_path, monkeypatch):
    paths = {}
    for attribute, name, body in (
        ("REPORT_SELF_AGREEMENT", "detector-self-agreement.json", _self_agreement_report()),
        ("REPORT_COMMON_MODE", "detector-common-mode.json", _common_mode_report()),
        ("REPORT_VS_MAMMA", "detector-vs-mamma.json", _vs_mamma_report()),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(body))
        monkeypatch.setattr(extractor, attribute, path)
        paths[attribute] = path
    return paths


def test_extractor_is_empty_when_no_report_is_on_disk(tmp_path, monkeypatch):
    for attribute in ("REPORT_SELF_AGREEMENT", "REPORT_COMMON_MODE", "REPORT_VS_MAMMA"):
        monkeypatch.setattr(extractor, attribute, tmp_path / "absent.json")
    assert extractor.x_detector_reports({}) == ([], [])


def test_extractor_shape_and_three_distinct_references(wired):
    figures, controls = extractor.x_detector_reports({})
    assert figures and controls
    for row in figures + controls:
        assert set(row) == {"key", "label", "value", "unit", "reference", "better", "note"}
    assert len({row["reference"] for row in figures}) == 3
    # every headline figure carries a number; a None would render as a missing instrument
    assert all(row["value"] is not None for row in figures)
    assert len({row["key"] for row in figures + controls}) == len(figures + controls)


def test_the_deflated_mamma_arm_is_never_a_scored_figure(wired):
    figures, controls = extractor.x_detector_reports({})
    deflated = [row for row in figures + controls if "DEFLATED" in row["label"]]
    assert len(deflated) == 1
    assert deflated[0] in controls
    assert deflated[0]["better"] == "exposure count, not a score"
    assert "2.2x" in deflated[0]["note"].lower() or "2.2X" in deflated[0]["note"]


def test_the_two_controls_that_pass_say_so_in_their_notes(wired):
    _figures, controls = extractor.x_detector_reports({})
    by_key = {row["key"]: row for row in controls}
    assert "blind" in by_key["self_own_3d"]["label"].lower() or \
           "PASSES BY DESIGN" in by_key["self_own_3d"]["label"]
    assert by_key["self_own_3d"]["value"] == 0.0
    assert "DOES NOT DISCRIMINATE" in by_key["vs_mamma_with_nose"]["label"]


def test_the_millimetre_and_pixel_mamma_references_are_not_the_same_string(wired):
    figures, _controls = extractor.x_detector_reports({})
    references = {row["key"]: row["reference"] for row in figures}
    assert references["cm_base"] != references["vs_mamma_p50"]
    assert "MILLIMETRES" in references["cm_base"]
    assert "PIXELS" in references["vs_mamma_p50"]
    assert "REFERENCE-FREE" in references["self_agreement_p50"]
