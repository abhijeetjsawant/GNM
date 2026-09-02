"""Unit tests for the I2 perfect-2D oracle's helpers.

The instrument itself needs MAMMA's retained run and three and a half minutes; these
test the parts that can be wrong *silently* -- the observation contract, the noise
resampler, the frame shift, and the two error bases -- on synthetic data that needs
nothing on disk. The one test that reads the report skips when it is absent.
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


def _load_module():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools" / "head"))
    spec = importlib.util.spec_from_file_location(
        "oracle_2d", ROOT / "tools" / "compare" / "oracle_2d.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load_module()
cm = oracle.cm


def _rig() -> tuple:
    """Four cameras on a ring, all looking at a point 1 m above the origin."""

    cameras = []
    for index, (x, y) in enumerate(((0.0, -4.0), (4.0, 0.0), (0.0, 4.0), (-4.0, 0.0))):
        center = np.asarray((x, y, 1.6))
        forward = np.asarray((0.0, 0.0, 1.0)) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        cameras.append(cm.CalibratedCamera(
            name=oracle.CAMERAS[index], width=1280, height=720,
            intrinsics=np.asarray(((900.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 1.0))),
            camera_center_world_m=center,
            camera_to_world_xyzw=Rotation.from_matrix(
                np.column_stack((right, down, forward))).as_quat(),
        ))
    return tuple(cameras)


def _subjects(frames: int = 6) -> np.ndarray:
    """Two bodies, 127 SMPL-X joint slots, only the scored ones meaningful."""

    rng = np.random.default_rng(3)
    subjects = np.zeros((2, frames, 127, 3))
    for body in range(2):
        base = rng.uniform(-0.4, 0.4, (127, 3)) + np.asarray((body * 0.8 - 0.4, 0.0, 1.1))
        for frame in range(frames):
            subjects[body, frame] = base + np.asarray((0.01 * frame, 0.0, 0.0))
    return subjects


def test_observations_carry_exactly_the_mapped_joints():
    records = oracle.observations_for(_rig(), _subjects())
    assert len(records) == len(oracle.CAMERAS)
    row = records[0][0]
    assert row["schema_version"] == cm.BODY_OBSERVATION_SCHEMA_VERSION
    assert row["width"] == oracle.WORKING_WIDTH and row["height"] == oracle.WORKING_HEIGHT
    assert len(row["people"]) == 2
    joints = row["people"][0]["joints"]
    assert set(joints) == set(oracle.PAIRS)
    # the four our 19-joint contract carries and PAIRS does not
    assert set(cm.JOINT_NAMES) - set(joints) == {
        "left_eye", "right_eye", "left_ear", "right_ear"}
    assert all(value["confidence"] == 0.95 for value in joints.values())


def test_projection_is_the_pipelines_own_and_round_trips():
    """The observations must be readable by the very code that will consume them."""

    cameras = _rig()
    subjects = _subjects(2)
    records = oracle.observations_for(cameras, subjects)
    for name, joint in oracle.PAIRS.items():
        rows = np.stack([
            cm._person_array(records[c][0]["people"][0])[cm.JOINT_INDEX[name]]
            for c in range(len(cameras))
        ])
        point = cm.triangulate_point(cameras, rows[:, :2], rows[:, 2])
        assert point is not None
        assert np.linalg.norm(point.position_world_m - subjects[0, 0, joint]) < 1e-6


def test_noise_reproduces_the_pool_scaled_to_the_working_width():
    pool = np.abs(np.random.default_rng(0).normal(0.0, 12.0, 200_000))
    noise = oracle.ResidualNoise(pool, seed=7)
    drawn = noise(200_000)
    ratio = oracle.WORKING_WIDTH / oracle.NATIVE_WIDTH
    for quantile in (25, 50, 75, 95):
        expected = np.percentile(pool, quantile) * ratio
        assert abs(np.percentile(drawn, quantile) - expected) < 0.05 * expected
    # and it must actually move the observations
    cameras, subjects = _rig(), _subjects(2)
    clean = oracle.observations_for(cameras, subjects)
    dirty = oracle.observations_for(cameras, subjects, noise=oracle.ResidualNoise(pool, 1))
    a = clean[0][0]["people"][0]["joints"]["root"]
    b = dirty[0][0]["people"][0]["joints"]["root"]
    assert (a["x"], a["y"]) != (b["x"], b["y"])


def test_time_shift_moves_content_and_leaves_the_frame_numbers_alone():
    records = oracle.observations_for(_rig(), _subjects(6))
    shifted = oracle.time_shifted(records, 0, +1)
    # the pipeline refuses streams whose frame numbers differ, so these must match
    assert [r["frame_index"] for r in shifted[0]] == [r["frame_index"] for r in records[0]]
    # frame f now carries what frame f-1 saw
    for frame in range(1, 6):
        assert (shifted[0][frame]["people"][0]["joints"]["root"]["x"]
                == records[0][frame - 1]["people"][0]["joints"]["root"]["x"])
    # the other three cameras are untouched, row object for row object
    for camera in (1, 2, 3):
        assert all(a is b for a, b in zip(shifted[camera], records[camera], strict=True))
    assert oracle.time_shifted(records, 0, -1)[0][0]["people"][0]["joints"]["root"]["x"] == \
        records[0][1]["people"][0]["joints"]["root"]["x"]


def test_root_relative_removes_translation_and_absolute_does_not():
    truth = _subjects(8)
    positions = np.full((2, 8, len(cm.JOINT_NAMES), 3), np.nan)
    for subject, body in ((0, 0), (1, 1)):
        for name, joint in oracle.PAIRS.items():
            positions[subject, :, cm.JOINT_INDEX[name]] = truth[body, :, joint]
    offset = np.asarray((0.05, -0.02, 0.01))
    positions[0] += offset

    matrices = oracle.error_matrices(positions, truth, {0: 0, 1: 1})
    assert np.allclose(matrices[0]["absolute"], np.linalg.norm(offset) * 1000.0)
    assert np.allclose(matrices[0]["relative"], 0.0, atol=1e-9)
    assert np.allclose(matrices[1]["absolute"], 0.0, atol=1e-9)


def test_scoring_against_the_wrong_body_is_the_control_it_claims_to_be():
    truth = _subjects(8)
    positions = np.full((2, 8, len(cm.JOINT_NAMES), 3), np.nan)
    for subject in (0, 1):
        for name, joint in oracle.PAIRS.items():
            positions[subject, :, cm.JOINT_INDEX[name]] = truth[subject, :, joint]
    right = oracle.score_arm(positions, truth, {0: 0, 1: 1})
    crossed = oracle.score_arm(positions, truth, {0: 1, 1: 0})
    assert right["overall"]["absolute"]["median_mm"] < 1e-6
    assert crossed["overall"]["absolute"]["median_mm"] > 100.0


def test_block_bootstrap_brackets_the_median_and_uses_blocks():
    matrix = np.abs(np.random.default_rng(4).normal(0.0, 5.0, (150, 15)))
    result = oracle.block_bootstrap_median(matrix)
    low, high = result["median_ci95_mm"]
    assert result["block_frames"] == oracle.BLOCK_FRAMES > 1
    assert low < float(np.median(matrix)) < high


@pytest.mark.skipif(not (ROOT / "artifacts/compare/oracle-2d.json").exists(),
                    reason="the oracle report needs MAMMA's retained run")
def test_the_report_on_disk_has_a_floor_every_control_fails():
    report = json.loads((ROOT / "artifacts/compare/oracle-2d.json").read_text())
    floor = report["arms"]["exact"]["overall"]["absolute"]["median_mm"]
    assert 0.0 < floor < 5.0
    assert report["arms"]["exact"]["overall"]["absolute"]["coverage"] == 1.0
    assert report["arms"]["noise"]["seeds"] >= 5
    assert len(report["arms"]["noise"]["per_seed"]) >= 5
    assert set(report["controls"]) >= {
        "shuffled_subject_pairing", "frozen_skeleton", "time_shift_+1", "time_shift_-1"}
    for name, control in report["controls"].items():
        assert control["overall"]["absolute"]["median_mm"] > 3.0 * floor, name
    assert report["blind_to"]
    assert report["subject_correspondence"] == {"our_0": "body_id-01", "our_1": "body_id-00"}


@pytest.mark.skipif(not (ROOT / "artifacts/compare/oracle-2d.json").exists(),
                    reason="the oracle report needs MAMMA's retained run")
def test_extractor_yields_figures_and_controls_in_the_ladder_shape():
    sys.path.insert(0, str(ROOT / "tools" / "compare" / "extractors"))
    from i2_oracle import x_oracle_2d  # noqa: PLC0415

    figures, controls = x_oracle_2d({})
    assert figures and controls
    for row in figures + controls:
        assert set(row) == {"key", "label", "value", "unit", "reference", "better", "note"}
        assert row["value"] is not None
        assert row["better"] in ("lower is better", "higher is better",
                                "exposure count, not a score")
        assert row["reference"]
    assert len({row["key"] for row in figures + controls}) == len(figures + controls)
