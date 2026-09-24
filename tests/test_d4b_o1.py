"""D4b O1 re-registered: the drawn-set arithmetic, the tolerance arithmetic, and the wired gate.

Runs on `.venv` (no momentum): the gate and the drawn-set rule's top level are momentum-free. The gate is
exercised on a SYNTHETIC population built here to satisfy every identity check, so these tests never read
`artifacts/` (gitignored); the frozen drawn set and the stage-1 provenance are the committed records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/fitter"))
sys.path.insert(0, str(ROOT / "tools/compare"))

import d4b_identifiability as di  # noqa: E402
import d4b_o1_gate as gate  # noqa: E402
import d4b_o1_gate_fuzz as fuzz  # noqa: E402


# ------------------------------------------------------------------------------------ the drawn-set rule

def test_pose_orthogonal_removes_the_span_and_keeps_the_complement():
    rng = np.random.default_rng(0)
    jacobian = rng.normal(size=(9, 4))
    inside = jacobian @ rng.normal(size=4)
    assert np.abs(di.pose_orthogonal(inside, jacobian)).max() < 1e-10
    basis, _ = di.column_basis(jacobian)
    outside = rng.normal(size=9)
    outside -= basis @ (basis.T @ outside)
    np.testing.assert_allclose(di.pose_orthogonal(outside, jacobian), outside, atol=1e-10)


def test_the_rank_convention_drops_a_truncation_scale_direction():
    """A column carrying a 1e-12-relative direction must not absorb that direction (the lstsq artefact)."""
    jacobian = np.zeros((6, 3))
    jacobian[0, 0] = 1.0
    jacobian[1, 1] = 1.0
    jacobian[2, 2] = 1e-12          # a finite-difference truncation direction, far below 1e-6
    displacement = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert di.column_basis(jacobian)[0].shape[1] == 2
    np.testing.assert_allclose(di.pose_orthogonal(displacement, jacobian), displacement)
    lstsq, *_ = np.linalg.lstsq(jacobian, displacement, rcond=None)
    assert np.abs(displacement - jacobian @ lstsq).max() < 1e-9      # what the replaced projection read


def test_the_rule_needs_both_donors_above_2mm():
    assert di.rule({0: 2.5, 1: 2.01})
    assert not di.rule({0: 2.5, 1: 2.0})
    assert not di.rule({0: 0.0, 1: 50.0})


def test_the_endpoint_aggregation_is_the_max_over_joints_in_mm():
    displacement_cm = np.zeros(17 * 3)
    displacement_cm[3:6] = [0.3, 0.4, 0.0]        # one joint 0.5 cm
    assert di.per_joint_max_mm(displacement_cm) == pytest.approx(5.0)


def test_the_spine_displacement_goes_toward_zero_and_through_it():
    assert di.displaced_spine(1.0) == pytest.approx(0.851)
    assert di.displaced_spine(-0.9) == pytest.approx(-0.751)
    assert di.displaced_spine(0.1) == pytest.approx(-0.049)       # through zero
    assert di.displaced_spine(0.0) == pytest.approx(-0.149)       # an undrawn spine: declared direction


def test_the_frozen_drawn_set_is_re_derivable_from_its_own_record():
    record = json.loads(gate.DRAWN_SET.read_text(encoding="utf-8"))
    drawn, scored, problems = gate.scored_segments(record)
    assert problems == []
    assert drawn == record["drawn_set"]
    assert "trunk" not in scored and "shoulder_width" not in scored
    assert record["frozen"]["rank_convention"].startswith("singular values above 1e-06")


# ---------------------------------------------------------------------------- the tolerance arithmetic

def _skeleton_cm() -> np.ndarray:
    points = {
        "root": (0, 0, 0), "c_neck": (0, 50, 0), "c_head": (0, 60, 0),
        "l_uparm": (18, 45, 0), "r_uparm": (-18, 45, 0), "l_lowarm": (45, 45, 0), "r_lowarm": (-45, 45, 0),
        "l_wrist": (70, 45, 0), "r_wrist": (-70, 45, 0), "l_upleg": (10, -5, 0), "r_upleg": (-10, -5, 0),
        "l_lowleg": (10, -50, 0), "r_lowleg": (-10, -50, 0), "l_foot": (10, -92, 0), "r_foot": (-10, -92, 0),
        "l_eye": (3, 65, 8), "r_eye": (-3, 65, 8)}
    return np.asarray([points[j] for j in di.MAPPED_JOINTS], np.float64)


def test_segment_error_and_the_sum_of_paired_floors():
    truth = _skeleton_cm()
    fitted = truth.copy()
    fitted[di.MAPPED_JOINTS.index("l_wrist"), 0] += 0.2       # the left forearm 2 mm long
    floor = np.full(17, 0.4)   # mm
    floor[di.MAPPED_JOINTS.index("l_wrist")] = 0.9
    rows = gate.segment_rows(truth, fitted, floor, scored=["l_forearm"])
    assert rows["l_forearm"]["error_mm"] == pytest.approx(2.0)
    assert rows["l_forearm"]["tolerance_mm"] == pytest.approx(1.3)
    assert rows["l_forearm"]["within"] is False
    assert not gate.l_passes(rows) and gate.misses_l(rows)
    assert rows["r_forearm"]["error_mm"] == 0.0 and not rows["trunk"]["scored"]


def test_an_unscored_segment_can_never_fail_l():
    truth = _skeleton_cm()
    fitted = truth.copy()
    fitted[di.MAPPED_JOINTS.index("c_neck"), 1] -= 5.0
    rows = gate.segment_rows(truth, fitted, np.full(17, 0.5), scored=["l_forearm"])
    assert rows["trunk"]["error_mm"] > rows["trunk"]["tolerance_mm"]
    assert gate.l_passes(rows)            # the band as scored is blind to the trunk


# ----------------------------------------------------------------------- a synthetic, valid population

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _population() -> dict:
    drawn_bytes = gate.DRAWN_SET.read_bytes()
    drawn = json.loads(drawn_bytes)
    provenance = json.loads(gate.PROVENANCE.read_text(encoding="utf-8"))
    stage1 = provenance["sha256"]
    pick = lambda prefix: next(v for k, v in stage1.items() if k.startswith(prefix))  # noqa: E731
    fixture_sha = _sha(gate.FIXTURE_SOURCE.read_bytes())
    names = list(di.CHANNELS) + [f"scale_other_{i:02d}" for i in range(58)]
    spine = names.index("scale_spine_length")
    base_prov = {"fitter_sha256": gate.CARD_FITTER_SHA256, "d4_fixture_sha256": pick("D4 fixture "),
                 "this_file_sha256": fixture_sha,
                 "donor_sha256": {"0": pick("donor 0 "), "1": pick("donor 1 ")},
                 "model_sha256": pick("model "), "fbx_sha256": pick("fbx "),
                 "drawn_set_sha256": _sha(drawn_bytes),
                 "pymomentum": provenance["environment"]["pymomentum-cpu"]}
    skeleton = _skeleton_cm()
    cells, pairs, files = {}, {}, {}
    rng = np.random.default_rng(7)
    for d in di.DONOR_INDICES:
        for s in di.SEEDS:
            truth_id = np.zeros(68)
            drawn_identity = {}
            for c in di.CHANNELS:
                low, high = drawn["configured_limits"][c]
                value = float(np.float32(rng.uniform(low, high))) if c in drawn["drawn_set"] else 0.0
                truth_id[names.index(c)] = value
                drawn_identity[c] = value
            fixture = {"generator": f"numpy default_rng([{s}, {d}]) (seeded by the pair)",
                       "drawn_channels": list(drawn["drawn_set"]), "drawn_identity": drawn_identity,
                       "clamped_parameter_frames": 900 + d, "truth_motion_sha256": f"{s}{d}".ljust(64, "a")}
            for a in di.ARMS:
                fitted_id, fitted_rest, note, distance = truth_id.copy(), skeleton.copy(), {}, 0.6
                if a == "exact_identity":
                    distance = 0.55
                elif a == "mean_body":
                    fitted_id[:] = 0.0
                    fitted_rest[di.MAPPED_JOINTS.index("l_wrist"), 0] += 3.0
                elif a == "spine_displaced":
                    fitted_id[spine] = di.displaced_spine(float(truth_id[spine]))
                    upper = [di.MAPPED_JOINTS.index(j) for j in di.MAPPED_JOINTS
                             if j not in ("root", "l_upleg", "r_upleg", "l_lowleg", "r_lowleg", "l_foot", "r_foot")]
                    fitted_rest[upper, 1] -= 1.5
                    note = {"spine_truth": float(truth_id[spine]), "spine_displaced_to": float(fitted_id[spine])}
                else:
                    fitted_rest[di.MAPPED_JOINTS.index("r_wrist"), 0] -= 0.01
                record = {
                    "schema": gate.SCHEMA, "burned": False, "seed": s, "donor": d, "arm": a, "lod": 2,
                    "provenance": copy.deepcopy(base_prov), "fixture": copy.deepcopy(fixture),
                    "settings": {"calib_frames": 100, "loss_alpha": 2.0, "max_iter": 300 if a == "converged" else 30,
                                 "smoothing": 0.0, "locator_limit_weight": 10.0, "freeze_flexible": True,
                                 "warm_start": a == "warm", "calibration_debug_captured": a == "converged"},
                    "arm_note": note, "calibrate_markers_calls": gate.CALIBRATE_CALLS[a],
                    "mapped_joints": list(di.MAPPED_JOINTS), "frames": 150,
                    "identity_channel_names": names, "truth_identity": truth_id.tolist(),
                    "fitted_identity": fitted_id.tolist(), "truth_rest_mapped_cm": skeleton.tolist(),
                    "fitted_rest_mapped_cm": fitted_rest.tolist(),
                    "truth_rest_full_vs_simplified_character_max_abs_cm": 0.0,
                    "distance_mm": np.full((150, 17), distance).tolist(), "locator_offset_mm_max": 0.0,
                }
                if a == "converged":
                    record["calibration_iterations"] = {"max_iter": 300, "calls": 2, "solves_per_call": [1054, 1056],
                                                        "max_iteration_index_reached": [299, 299],
                                                        "solves_stopped_at_the_cap": [10, 430], "note": "x"}
                if a == "oracle":
                    glb, track = f"fixture-{s}-d{d}-oracle.glb", f"fixture-{s}-d{d}-oracle.body-track.npz"
                    record.update(glb=glb, track=track, glb_sha256=_sha(glb.encode()), track_sha256=_sha(track.encode()))
                    files[glb], files[track] = _sha(glb.encode()), _sha(track.encode())
                    pairs[f"{s}_d{d}"] = {"glb": glb, "track": track, "frames_in_glb": 150, "frames_in_track": 150,
                                          "joints_compared": 127, "joints_missing_from_the_glb": [],
                                          "max_abs_m": 2e-6}
                cells[f"cell-{s}-d{d}-{a}.json"] = record
    return {"cells": cells, "cell_dir": "synthetic", "files": files, "closure": {"pairs": pairs},
            "drawn_set_bytes": drawn_bytes, "fixture_source_sha256": fixture_sha, "provenance": provenance}


@pytest.fixture(scope="module")
def population() -> dict:
    return _population()


def test_a_valid_population_passes_and_the_unscored_trunk_keeps_d4_open(population):
    report = gate.build(population)
    assert report["problems"] == []
    assert report["verdict"] == "PASS" and report["STOP"] is False
    assert report["d4_disposition"].startswith("STAYS OPEN")
    assert report["clauses"]["must_fail_ii_spine_displaced_at_the_trunk"]["measured"][
        "the_band_as_scored_passes_the_displaced_spine_on"] == 12


def _mutated(population, mutate):
    new = dict(population)
    new["cells"] = copy.deepcopy(population["cells"])
    new["closure"] = copy.deepcopy(population["closure"])
    mutate(new)
    return gate.build(new)


def test_a_missing_short_non_finite_or_stray_cell_is_fail(population):
    first = "cell-20261002-d1-{}.json"
    for mutate in (lambda p: p["cells"].pop(first.format("warm")),
                   lambda p: p["cells"][first.format("oracle")].update(distance_mm=p["cells"][first.format("oracle")]["distance_mm"][:149]),
                   lambda p: p["cells"][first.format("mean_body")]["distance_mm"][3].__setitem__(4, float("nan")),
                   lambda p: p["cells"].__setitem__("cell-20261009-d0-oracle.json", p["cells"][first.format("oracle")])):
        assert _mutated(population, mutate)["verdict"] == "FAIL"


def test_a_degraded_floor_is_invalid_never_fail_or_pass(population):
    def degrade(p):
        cell = p["cells"]["cell-20261004-d0-exact_identity.json"]
        cell["distance_mm"] = np.full((150, 17), 1.2).tolist()
    assert _mutated(population, degrade)["verdict"] == "INVALID"


def test_a_spine_the_band_cannot_see_is_a_stop(population):
    def blind(p):
        cell = p["cells"]["cell-20261005-d1-spine_displaced.json"]
        cell["fitted_rest_mapped_cm"] = copy.deepcopy(cell["truth_rest_mapped_cm"])
    report = _mutated(population, blind)
    assert report["verdict"] == "FAIL" and report["STOP"] is True


def test_the_exact_identity_control_must_read_zero(population):
    def nonzero(p):
        cell = p["cells"]["cell-20261003-d0-exact_identity.json"]
        rest = np.asarray(cell["fitted_rest_mapped_cm"])
        rest[0, 1] += 1e-6          # along the trunk: a first-order length change of 1e-5 mm
        cell["fitted_rest_mapped_cm"] = rest.tolist()
    assert _mutated(population, nonzero)["verdict"] == "FAIL"


def test_the_mean_body_must_miss_and_the_closure_must_hold(population):
    def mean_is_truth(p):
        cell = p["cells"]["cell-20261006-d0-mean_body.json"]
        cell["fitted_rest_mapped_cm"] = copy.deepcopy(cell["truth_rest_mapped_cm"])
    assert _mutated(population, mean_is_truth)["verdict"] == "FAIL"
    assert _mutated(population, lambda p: p["closure"]["pairs"]["20261001_d1"].update(max_abs_m=1.1e-4))[
        "verdict"] == "FAIL"
    assert _mutated(population, lambda p: p["closure"]["pairs"]["20261001_d1"].update(frames_in_glb=149))[
        "verdict"] == "FAIL"


def test_provenance_and_fixture_identity_are_checked(population):
    assert _mutated(population, lambda p: p["cells"]["cell-20261001-d0-oracle.json"]["provenance"].update(
        pymomentum="0.0.0"))["verdict"] == "FAIL"

    def shared(p):
        for a in di.ARMS:
            p["cells"][f"cell-20261002-d0-{a}.json"]["truth_identity"] = copy.deepcopy(
                p["cells"]["cell-20261001-d0-oracle.json"]["truth_identity"])
    assert _mutated(population, shared)["verdict"] == "FAIL"
    assert _mutated(population, lambda p: p["cells"]["cell-20261001-d0-oracle.json"]["settings"].update(
        max_iter=300))["verdict"] == "FAIL"


def test_the_fuzz_turns_every_conjunct_on_the_synthetic_population(population):
    cases = fuzz.targeted(population, burned=False)
    assert cases and all(c["turned"] for c in cases), [c for c in cases if not c["turned"]]


def test_the_fuzz_classifies_a_leaf_by_what_it_moves():
    assert fuzz.kind_of(True) == "bool" and fuzz.kind_of(3) == "number" and fuzz.kind_of([]) == "list"
    assert [m[0] for m in fuzz.mutations_for(1.5)] == ["=1e6", "=-1e6", "=0", "delete"]
    assert fuzz.justified("closure.pairs.20261001_d0.within_band") is not None
    assert fuzz.justified("cell(20261001-d0-oracle).truth_rest_mapped_cm.*.*") is None
