"""D4 -- the body model in the delivery path. Tests that need neither momentum nor artifacts.

The delivery itself is measured by the instruments under `tools/compare/`; what is pinned here is
the part of D4 that is pure code and could rot silently:

  * the capture <-> MHR frame conversion is an exact inverse, and it is written out twice in the
    tree (the fitter under `tools/fitter/`, the B2 instrument under `tools/compare/`, on purpose,
    so B2 is an independent re-derivation) -- the two must agree;
  * `--body` exists on the build script, defaults to `rig`, and the MHR flags exist;
  * the marker derivation turns a non-finite landmark into an OCCLUDED marker at the origin,
    never into a zero the solver would chase;
  * the part partition and the bent-tercile measure used by B1's reported diagnostics;
  * the gate derives its verdicts rather than asserting them: mutating an input turns them.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} is not present")
    sys.path.insert(0, str(ROOT / "tools/compare"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def closure():
    return _load("tools/compare/d4_glb_closure.py", "d4_glb_closure")


@pytest.fixture(scope="module")
def b2():
    return _load("tools/compare/d4_b2_same_denominator.py", "d4_b2_same_denominator")


@pytest.fixture(scope="module")
def gate():
    return _load("tools/compare/d4_body_gate.py", "d4_body_gate")


def test_capture_to_mhr_and_back_is_an_exact_inverse(closure, b2):
    rng = np.random.default_rng(4)
    points = rng.normal(size=(7, 19, 3))
    # (x, z, -y) * 100 metres -> centimetres, and the glTF reader's (x, -z, y) back again. The
    # reader's own scale is carried by the GLB's root node, so `to_capture` is unit-free here.
    there = b2.to_mhr_cm(points)
    back = closure.to_capture(there) / 100.0
    # Not bit-exact: the * 100 and / 100 round in float64. It must be exact to the last bits.
    assert np.allclose(back, points, atol=1e-15, rtol=1e-15)
    assert np.abs(back - points).max() < 1e-15


def test_the_conversion_is_a_rotation_not_a_reflection(b2):
    """A reflection would flip handedness and no joint or silhouette gate could see it."""
    basis = np.eye(3)
    mapped = b2.to_mhr_cm(basis) / 100.0
    assert np.isclose(np.linalg.det(mapped), 1.0)


def test_build_script_exposes_body_rig_by_default():
    source = (ROOT / "scripts/build_commercial_multiview_comparison.py").read_text()
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            keywords = {k.arg: k.value for k in node.keywords}
            found[node.args[0].value] = keywords
    assert "--body" in found
    default = found["--body"].get("default")
    assert isinstance(default, ast.Constant) and default.value == "rig"
    choices = found["--body"].get("choices")
    assert isinstance(choices, ast.Tuple)
    assert {c.value for c in choices.elts} == {"rig", "mhr"}
    for flag in ("--mhr-lod", "--mhr-landmarks", "--mhr-python"):
        assert flag in found


def test_build_script_runs_one_mhr_process_per_performer():
    """A momentum calibration mutates later model loads in the same process (D4 stage 2)."""
    source = (ROOT / "scripts/build_commercial_multiview_comparison.py").read_text()
    assert "--subject" in source
    assert "for subject in range(len(tracks)):" in source


def test_a_non_finite_landmark_becomes_an_occluded_marker():
    fitter = ROOT / "tools/fitter/mhr_delivery.py"
    source = fitter.read_text()
    # The fitter needs pymomentum, so the contract is pinned on its source rather than by import.
    assert "occluded=not finite" in source
    assert "position if finite else np.zeros(3)" in source


def test_part_labels_split_arms_from_torso_and_legs():
    module = _load("tools/compare/d4_silhouette_paired.py", "d4_silhouette_paired")

    class FlatCamera:
        """Orthographic: world (x, y, z) -> pixel (x, z), depth 1. Enough for a partition test."""

        @staticmethod
        def project(points):
            points = np.asarray(points, float)
            return (np.stack([points[:, 0], points[:, 2]], axis=1),
                    np.ones(len(points)))

    landmarks = np.full((len(module.NAMES19), 3), np.nan)
    index = {name: i for i, name in enumerate(module.NAMES19)}
    landmarks[index["root"]] = (0, 0, 0)
    landmarks[index["neck"]] = (0, 0, 50)
    landmarks[index["nose"]] = (0, 0, 60)
    landmarks[index["left_shoulder"]] = (10, 0, 48)
    landmarks[index["left_elbow"]] = (30, 0, 48)
    landmarks[index["left_wrist"]] = (50, 0, 48)
    landmarks[index["right_shoulder"]] = (-10, 0, 48)
    landmarks[index["right_elbow"]] = (-30, 0, 48)
    landmarks[index["right_wrist"]] = (-50, 0, 48)
    landmarks[index["left_hip"]] = (6, 0, 0)
    landmarks[index["right_hip"]] = (-6, 0, 0)
    landmarks[index["left_knee"]] = (6, 0, -30)
    landmarks[index["right_knee"]] = (-6, 0, -30)
    landmarks[index["left_ankle"]] = (6, 0, -60)
    landmarks[index["right_ankle"]] = (-6, 0, -60)

    pixels = np.array([[40.0, 48.0],     # mid forearm
                       [-40.0, 48.0],    # the other forearm
                       [6.0, -45.0],     # shin
                       [0.0, 25.0]])     # trunk
    labels = module.part_labels(landmarks, FlatCamera, pixels)
    assert labels.tolist() == [1, 1, 0, 0]


def test_trunk_lean_is_zero_upright_and_ninety_when_folded():
    module = _load("tools/compare/d4_silhouette_paired.py", "d4_silhouette_paired")
    index = {name: i for i, name in enumerate(module.NAMES19)}
    upright = np.full((len(module.NAMES19), 3), np.nan)
    upright[index["root"]] = (0, 0, 0)
    upright[index["neck"]] = (0, 0, 0.5)
    assert module.trunk_lean_deg(upright) == pytest.approx(0.0, abs=1e-9)
    folded = upright.copy()
    folded[index["neck"]] = (0.5, 0, 0)
    assert module.trunk_lean_deg(folded) == pytest.approx(90.0, abs=1e-9)
    missing = upright.copy()
    missing[index["neck"]] = (np.nan, 0, 0)
    assert np.isnan(module.trunk_lean_deg(missing))


def _minimal_gate_inputs(gate):
    """The smallest input tree `verdicts` reads, all conjuncts passing."""
    eight = gate.EIGHT
    seeds = {"a": {"median_of_per_frame_medians": 0.5}}
    cameras = ("A001", "B001", "C001", "D001")
    silhouette = {name: {cam: {f"subject_{s}": {"iou": value}
                               for s in ("00", "01")} for cam in cameras}
                  for name, value in (("ours_delivered", 0.80), ("control_frozen_pose_tracked", 0.30),
                                      ("ORACLE_mamma_mesh", 0.87))}
    paired = {f"delivered_MHR_lod2_minus_baseline_D7c_rig_subject_{s}":
              {"median_difference": 0.1, "ci95_of_the_median_difference": [0.05, 0.15], "n": 600}
              for s in ("00", "01")}
    arms = {name: {f"subject_{s}": {"pooled_median_iou": 0.8} for s in ("00", "01")}
            for name in ("baseline_D7c_rig", "delivered_MHR_lod2", "control_MHR_mean_body_lod2")}
    return {
        "hygiene": {"rebuild_sha256": {f: "x" for f in eight},
                    "shipped_sha256": {f: "x" for f in eight}},
        "reproduction": {"arms": {"buildscript_MHR_lod6_raw":
                                  {"subject_00": {"pooled_median_iou": gate.PRECARD["subject_00"]},
                                   "subject_01": {"pooled_median_iou": gate.PRECARD["subject_01"]}}}},
        "o1_readings": {"oracle": seeds, "mean_body": {"a": {"median_of_per_frame_medians": 30.0}},
                        "exact_identity": {"a": {"median_of_per_frame_medians": 0.4}}},
        "o1_closure": {"worst_max_abs_m": 1e-6},
        "o1_closure_mutations": {"pairs": {"m": {"max_abs_m": 0.1, "within_band": False}}},
        "b1": {"paired": paired, "arms": arms},
        "b1_silhouette": {"arms": silhouette},
        "b1_mamma": {"cells": {"c": {"identical_all_fields": True}},
                     "bit_identical_on_all_8_cells": True},
        # B2's NUMBERED checks, which is what the gate derives from -- never its verdict leaf
        "b2": {"subjects": {f"subject_{s}": {
            "1a_handed_array_is_byte_identical_to_the_rig_converter_input_on_this_build": True,
            "1c_the_array_is_the_SMOOTHED_repaired_one": True,
            "4a_marker_values_match_the_declared_mapping_and_conversion": True,
            "marker_names_are_the_declared_map": True,
            "verdict": "PASS"} for s in ("00", "01")}},
        "b3": {"subjects": {f"subject_{s}": {"all_landmarks_median_mm": 19.0,
                                             "segment_mean_abs_error_mm": 10.0}
                            for s in ("00", "01")}},
        "b4": {"subjects": {f"subject_{s}": {"absolute_all_joint_median_mm": 35.0,
                                             "absolute_all_joint_bias_median_mm": 40.0,
                                             "absolute_all_joint_spread_median_mm": 30.0}
                            for s in ("00", "01")},
               "subject_to_mamma_body_id": {"0": 1, "1": 0}},
        "b5": {"verdict_bytes": "PASS", "facing_positive_on_every_frame_both_performers": False,
               "subjects": {f"subject_{s}": {
                   "facing_dot_positive_frames": 150, "facing_dot_frames_scored": 150,
                   "sampler_times_are_k_over_30_s": True,
                   "sampler_times_strictly_increasing": True,
                   "joint_names_match_MHR": True, "hierarchy_matches_MHR": True,
                   "rest_translation_max_abs_diff_cm": 2e-6, "mesh_vertices_match_MHR": True,
                   "mesh_all_finite": True, "mesh_frames": 150,
                   "skin_weight_max_abs_diff": 0.0} for s in ("00", "01")}},
        "delivery_closure": {"worst_max_abs_m": 2e-6},
        "free_offsets": {f"subject_{s}": {"locator_offset_mm_median": 120.0,
                                          "nonzero_identity_channels": 13,
                                          "joint_to_landmark_residual_mm_median_over_frames": 140.0}
                         for s in ("00", "01")},
        "b1_parts": {"partition": "every foreground pixel to its nearest segment",
                     "tercile_edges_trunk_lean_deg": {}, "parts": {}, "bent_terciles": {}},
        "o1_prerepair": {"oracle_max_over_seeds_mm": 1.2484,
                         "exact_identity_max_over_seeds_mm": 1.1604},
        "o1": {"fixture_repair": {"clamped_parameter_frames": 1025, "violating_parameters": {}},
               "fitter_source_sha256": "abc"},
        "reproduction_repaired": {
            "arms": {"buildscript_MHR_lod6_raw_oneprocess": {
                "subject_00": {"pooled_median_iou": 0.7894},
                "subject_01": {"pooled_median_iou": 0.7503}}},
            "paired": {f"buildscript_MHR_lod6_raw_oneprocess_minus_precard_fitted_MHR_"
                       f"subject_{s}": {"median_difference": 0.0} for s in ("00", "01")}},
        "source": {"default_body": "rig", "body_choices": ["rig", "mhr"]},
        "fit_report": {"one_process_per_performer": True,
                       "subjects": {f"subject_{s}": {"subjects": {f"subject_{s}": {
                           "mesh_vertices": 10661, "nonzero_identity_channels": 13,
                           "joint_to_landmark_residual_mm_median_over_frames": 16.7,
                           "locator_offset_mm_median": 0.0}}, "lod": 2, "landmarks": "smoothed"}
                           for s in ("00", "01")}},
    }


def test_the_gate_derives_its_verdicts_and_every_conjunct_turns(gate):
    data = _minimal_gate_inputs(gate)
    baseline = gate.verdicts(data)
    assert baseline["D4_acceptance"]["conjuncts"] == {
        "hygiene": "PASS", "reproduction": "PASS", "O1_exactness": "PASS",
        "B1_the_band": "PASS", "B2_same_denominator": "PASS"}

    mutations = {
        "hygiene": ("hygiene/rebuild_sha256/" + gate.EIGHT[0], "moved"),
        "reproduction": ("reproduction/arms/buildscript_MHR_lod6_raw/subject_00/"
                         "pooled_median_iou", 0.5),
        "O1_exactness": ("o1_readings/oracle/a/median_of_per_frame_medians", 2.0),
        "B1_the_band": ("b1/paired/delivered_MHR_lod2_minus_baseline_D7c_rig_subject_01/"
                        "ci95_of_the_median_difference/0", -0.01),
        # an INPUT leaf, never a verdict leaf (D7c's rule, and the gate now derives B2 from
        # its numbered checks)
        "B2_same_denominator": ("b2/subjects/subject_00/"
                                "4a_marker_values_match_the_declared_mapping_and_conversion",
                                False),
    }
    for conjunct, (path, value) in mutations.items():
        mutated = _minimal_gate_inputs(gate)
        gate.put(mutated, path, value)
        moved = gate.verdicts(mutated)
        assert moved[conjunct]["verdict"] == "FAIL", conjunct
        assert conjunct in moved["D4_acceptance"]["failing_conjuncts"]


def test_a_failing_o1_makes_acceptance_fail_and_there_is_no_exception_branch(gate):
    """An exception is an override, and the lane forbids merging on one (Astra, 2026-09-22)."""
    data = _minimal_gate_inputs(gate)
    data["o1_readings"]["oracle"]["a"]["median_of_per_frame_medians"] = 1.03
    report = gate.verdicts(data)
    assert report["O1_exactness"]["verdict"] == "FAIL"
    assert report["D4_acceptance"]["verdict"] == "FAIL"
    assert report["D4_acceptance"]["failing_conjuncts"] == ["O1_exactness"]
    assert report["D4_acceptance"]["line"].startswith("D4 ACCEPTANCE: FAIL")
    assert "1.030 mm > 1 mm" in report["D4_acceptance"]["line"]
    assert "MERGE" not in report["D4_acceptance"]["line"]
    # the whole gate, serialised, must not offer a merge on a failing conjunct anywhere
    assert "MERGE with O1" not in json.dumps(report)


def test_the_opt_in_line_needs_hygiene_and_a_rig_default_in_the_source(gate):
    data = _minimal_gate_inputs(gate)
    data["o1_readings"]["oracle"]["a"]["median_of_per_frame_medians"] = 1.03
    report = gate.verdicts(data)
    optin = report["opt_in_implementation"]
    assert optin["mergeable"] is True
    assert optin["line"].startswith("OPT-IN IMPLEMENTATION: mergeable behind --body rig default")
    assert optin["acceptance"] == "FAIL"          # it carries no acceptance claim
    for path in ("source/default_body", "hygiene/rebuild_sha256/" + gate.EIGHT[0]):
        mutated = _minimal_gate_inputs(gate)
        gate.put(mutated, path, "mhr" if path.endswith("default_body") else "moved")
        assert gate.verdicts(mutated)["opt_in_implementation"]["mergeable"] is False


def test_the_real_build_script_still_defaults_to_rig(gate):
    parsed = gate.build_script_body_argument()
    assert parsed["default_body"] == "rig"
    assert "mhr" in parsed["body_choices"]


def test_b1_lower_bound_exactly_zero_is_not_a_pass(gate):
    data = _minimal_gate_inputs(gate)
    gate.put(data, "b1/paired/delivered_MHR_lod2_minus_baseline_D7c_rig_subject_00/"
                   "ci95_of_the_median_difference/0", 0.0)
    assert gate.verdicts(data)["B1_the_band"]["verdict"] == "FAIL"


def test_the_gate_ignores_another_instruments_verdict_leaf(gate):
    """A gate that reads a verdict can be turned by mutating a verdict. B2's and B5's are inert."""
    data = _minimal_gate_inputs(gate)
    gate.put(data, "b2/subjects/subject_00/verdict", "FAIL")
    gate.put(data, "b5/verdict_bytes", "FAIL")
    report = gate.verdicts(data)
    assert report["B2_same_denominator"]["verdict"] == "PASS"
    assert report["B5_delivered_bytes_REPORTED"]["bytes_verdict"] == "PASS"
