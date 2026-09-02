"""Unit tests for the I5 hand held-out instrument, on constructed data only.

No artifact is read here and no solve is run, so these do not depend on the
multi-hour leave-one-camera-out grid. Everything is a hand-built hand whose answer
is known -- which is what catches the class of defect this lane keeps producing: a
frame that has silently inverted, a control that is not degenerate, and a held-out
protocol that is not actually held out.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "hands"))
sys.path.insert(0, str(ROOT / "tools" / "compare" / "extractors"))

from autoanim_gnm.commercial_multiview import CalibratedCamera  # noqa: E402


@pytest.fixture(scope="module")
def report():
    import hand_fit_report

    return hand_fit_report


@pytest.fixture(scope="module")
def extractor():
    import i5_hands

    return i5_hands


# ------------------------------------------------------------------ a toy hand

WRIST, INDEX1, MIDDLE1, RING1, PINKY1 = 0, 1, 2, 3, 4
TIP_SLOTS = [5, 6, 7, 8, 9]
KNUCKLES = [INDEX1, MIDDLE1, RING1, PINKY1]


def toy_hand(frames: int = 120, curl: float = 0.03, seed: int | None = None,
             period: float = 60.0):
    """A ten-slot hand: wrist, four knuckles, five tips. Palm along +x, +y across.

    The tips oscillate along the palm normal so the motion is real, low-frequency
    articulation with a known amplitude.
    """
    time = np.arange(frames)
    positions = np.zeros((frames, 10, 3))
    positions[:, WRIST] = 0.0
    for slot, across in zip(KNUCKLES, (0.03, 0.01, -0.01, -0.03)):
        positions[:, slot] = [0.08, across, 0.0]
    swing = curl * np.sin(2.0 * np.pi * time / period)
    for tip, across in zip(TIP_SLOTS, (0.05, 0.02, 0.0, -0.02, -0.04)):
        positions[:, tip, 0] = 0.14
        positions[:, tip, 1] = across
        positions[:, tip, 2] = swing
    if seed is not None:
        positions += np.random.default_rng(seed).normal(0.0, 0.001, positions.shape)
    wrist = np.zeros((frames, 3))
    return positions, wrist


def rigid(positions, wrist, rotation, translation):
    matrix = Rotation.from_rotvec(rotation).as_matrix()
    return positions @ matrix.T + translation, wrist @ matrix.T + translation


def toy_camera(name="X", centre=(0.0, -4.0, 0.0)):
    intrinsics = np.array([[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]])
    # look along +y from -y: camera z-axis is world +y, camera x is world +x.
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return CalibratedCamera(
        name=name, width=1280, height=720, intrinsics=intrinsics,
        camera_center_world_m=np.asarray(centre, dtype=float),
        camera_to_world_xyzw=Rotation.from_matrix(matrix).as_quat())


# ------------------------------------------------- the frame is gauge-invariant

def test_wrist_local_frame_is_invariant_to_rigid_motion(report):
    """Rotate and translate the whole hand: the local coordinates must not move.

    This is the property the whole metric rests on. The angle standard deviation it
    replaces was void precisely because it was NOT invariant -- MHR Euler channels
    and SMPL-X axis-angle triples measure different things.
    """
    positions, wrist = toy_hand()
    before = report.wrist_local_frame(positions, wrist, KNUCKLES, INDEX1, PINKY1)
    moved, moved_wrist = rigid(positions, wrist, [0.7, -0.4, 1.9], [3.0, -1.0, 0.5])
    after = report.wrist_local_frame(moved, moved_wrist, KNUCKLES, INDEX1, PINKY1)
    assert np.allclose(before, after, atol=1e-9)


def test_wrist_local_frame_axes_are_orthonormal_and_right_handed(report):
    positions, wrist = toy_hand(frames=3)
    local = report.wrist_local_frame(positions, wrist, KNUCKLES, INDEX1, PINKY1)
    # the knuckle centroid must sit on the first axis by construction
    centroid = local[:, KNUCKLES].mean(axis=1)
    assert np.allclose(centroid[:, 1:], 0.0, atol=1e-12)
    assert (centroid[:, 0] > 0.0).all()
    # a rigid frame preserves every distance from the origin
    assert np.allclose(np.linalg.norm(local[0, TIP_SLOTS[0]] - local[0, WRIST]),
                       np.linalg.norm(positions[0, TIP_SLOTS[0]] - positions[0, WRIST]))


# ----------------------------------------------------- amplitude and jitter behave

def test_a_constant_hand_has_zero_amplitude_and_suppressed_roughness(report):
    positions, wrist = toy_hand(curl=0.0)
    local = report.wrist_local_frame(positions, wrist, KNUCKLES, INDEX1, PINKY1)
    figures = report.articulation(local[:, TIP_SLOTS], 0.08)
    assert figures["amplitude_mm"] == pytest.approx(0.0, abs=1e-6)
    assert figures["jitter_mm"] == pytest.approx(0.0, abs=1e-6)
    assert figures["roughness"] is None, "0/0 must never be reported as a ratio"
    assert figures["roughness_suppressed_amplitude_below_1mm"] is True


def test_amplitude_matches_the_sinusoid_it_was_given(report):
    """A 30 mm sine has RMS excursion 30/sqrt(2) = 21.2 mm about its own mean."""
    positions, wrist = toy_hand(curl=0.03)
    local = report.wrist_local_frame(positions, wrist, KNUCKLES, INDEX1, PINKY1)
    figures = report.articulation(local[:, TIP_SLOTS], 0.08)
    assert figures["amplitude_mm"] == pytest.approx(30.0 / np.sqrt(2.0), rel=0.05)


def test_jitter_rises_with_added_per_frame_noise_and_amplitude_barely_moves(report):
    smooth, wrist = toy_hand(curl=0.03)
    noisy, _ = toy_hand(curl=0.03, seed=7)
    values = []
    for positions in (smooth, noisy):
        local = report.wrist_local_frame(positions, wrist, KNUCKLES, INDEX1, PINKY1)
        values.append(report.articulation(local[:, TIP_SLOTS], 0.08))
    assert values[1]["jitter_mm"] > 5.0 * values[0]["jitter_mm"]
    assert values[1]["amplitude_mm"] == pytest.approx(values[0]["amplitude_mm"], rel=0.15)


# ------------------------------------------------------- the degenerate arms are

def test_frozen_medoid_is_an_exactly_valid_pose_with_bone_lengths_preserved(report):
    positions, wrist = toy_hand()
    frozen = report.frozen_medoid(positions, wrist)
    assert np.allclose(frozen, frozen[:1], atol=1e-12), "the pose must not move"
    length = np.linalg.norm(positions[:, TIP_SLOTS[0]] - positions[:, MIDDLE1], axis=1)
    frozen_length = np.linalg.norm(frozen[:, TIP_SLOTS[0]] - frozen[:, MIDDLE1], axis=1)
    assert frozen_length.min() == pytest.approx(length.min(), rel=0.05)


def test_frozen_medoid_has_zero_amplitude(report):
    positions, wrist = toy_hand()
    frozen = report.frozen_medoid(positions, wrist)
    local = report.wrist_local_frame(frozen, wrist, KNUCKLES, INDEX1, PINKY1)
    assert report.articulation(local[:, TIP_SLOTS], 0.08)["amplitude_mm"] < 1e-6


def test_lag_shifts_the_configuration_and_keeps_the_true_wrist(report):
    positions, wrist = toy_hand()
    wrist = wrist + np.linspace(0.0, 1.0, len(positions))[:, None] * np.array([0.0, 0.0, 1.0])
    positions = positions + wrist[:, None, :]
    lagged = report.lagged(positions, wrist, 3)
    assert np.allclose(lagged[10] - wrist[10], positions[7] - wrist[7])
    assert np.allclose(lagged[:, WRIST], positions[:, WRIST]), "the anchor is not lagged"


def test_lag_leaves_jitter_exactly_unchanged(report):
    """The demonstration that the thrash figure rejects nothing.

    A rigid time shift of a series has the identical second-difference
    distribution, so a jitter band cannot tell a lagged solution from the
    candidate. Only the held-out camera can.
    """
    positions, wrist = toy_hand(curl=0.03, seed=3)
    figures = []
    for arm in (positions, report.lagged(positions, wrist, 3)):
        local = report.wrist_local_frame(arm, wrist, KNUCKLES, INDEX1, PINKY1)
        figures.append(report.articulation(local[:, TIP_SLOTS], 0.08)["jitter_mm"])
    assert figures[0] == pytest.approx(figures[1], rel=0.05)


def test_duplicated_frames_copies_every_odd_frame_from_its_predecessor(report):
    positions, wrist = toy_hand()
    doubled = report.duplicated(positions, wrist)
    assert np.allclose(doubled[1], doubled[0])
    assert np.allclose(doubled[3], doubled[2])
    assert not np.allclose(doubled[2], doubled[0])


def test_zero_phase_low_pass_does_not_lag_and_the_causal_one_does(report):
    positions, wrist = toy_hand(curl=0.03)
    tip = TIP_SLOTS[0]
    for causal, expected in ((False, 0), (True, report.LOWPASS_FRAMES // 2)):
        smoothed = report.low_passed(positions, wrist, causal=causal)
        a = positions[:, tip, 2] - positions[:, tip, 2].mean()
        b = smoothed[:, tip, 2] - smoothed[:, tip, 2].mean()
        lags = range(-10, 11)
        best = max(lags, key=lambda k: float(np.corrcoef(
            a[10:-10], b[10 + k:len(b) - 10 + k])[0, 1]))
        assert best == pytest.approx(expected, abs=1)


def test_low_pass_attenuates_the_swing(report):
    positions, wrist = toy_hand(curl=0.03)
    smoothed = report.low_passed(positions, wrist, causal=False)
    tip = TIP_SLOTS[0]
    swing = positions[:, tip, 2].std()
    assert 0.3 * swing < smoothed[:, tip, 2].std() < 0.95 * swing


# ------------------------------------------------- the held-out camera sees a lag

def _hand_blob(positions, wrist, camera):
    """A one-camera [F,1,J,3] observation blob from exact projections.

    Only the fingertips are in the test set: in this toy the knuckles are rigid, and
    scoring rigid joints would dilute every arm's median toward zero and make the
    controls look better than they are. The real instrument has the same property and
    handles it the same way -- the test set is the retained gate's surviving
    observations, identical for every arm.
    """
    frames, joints = positions.shape[:2]
    observations = np.zeros((frames, 1, joints, 3))
    for frame in range(frames):
        uv, _ = camera.project(positions[frame])
        observations[frame, 0, :, :2] = uv
        observations[frame, 0, :, 2] = 0.9
    keep = np.zeros((frames, 1, joints), dtype=bool)
    keep[:, 0, TIP_SLOTS] = True
    return {"observations": observations, "wrist": wrist, "keep": keep}


def test_held_out_reprojection_rejects_a_lag_that_jitter_cannot_see(report):
    """The instrument's reason to exist, on data whose answer is known."""
    camera = toy_camera()
    positions, wrist = toy_hand(curl=0.05)
    hand = _hand_blob(positions, wrist, camera)
    truth = report.score_arm(camera, positions, hand, 0)
    lagged = report.score_arm(camera, report.lagged(positions, wrist, 5), hand, 0)
    assert truth["median_px"] < 1e-6
    assert lagged["median_px"] > 5.0
    index = report.block_draws(len(positions), 200, 1)
    assert report.probability_candidate_beats(truth, lagged, index, len(positions)) == 1.0


def test_held_out_reprojection_rejects_a_frozen_hand(report):
    camera = toy_camera()
    positions, wrist = toy_hand(curl=0.05)
    hand = _hand_blob(positions, wrist, camera)
    truth = report.score_arm(camera, positions, hand, 0)
    frozen = report.score_arm(camera, report.frozen_medoid(positions, wrist), hand, 0)
    index = report.block_draws(len(positions), 200, 1)
    assert report.probability_candidate_beats(truth, frozen, index, len(positions)) == 1.0


def test_millimetres_use_each_observation_s_own_depth(report):
    """Two cameras at different ranges must not share a pixels-to-mm constant."""
    positions, wrist = toy_hand()
    hand = _hand_blob(positions, wrist, toy_camera())
    near = report.millimetres_at_the_subject(toy_camera(centre=(0.0, -2.0, 0.0)),
                                             positions, hand, 0)
    far = report.millimetres_at_the_subject(toy_camera(centre=(0.0, -8.0, 0.0)),
                                            positions, hand, 0)
    assert np.median(far) > 3.5 * np.median(near)


# --------------------------------------------------------- the fit is held out

def test_training_weights_zero_the_held_out_camera(report):
    cameras = [toy_camera("A", (0.0, -4.0, 0.0)), toy_camera("B", (3.0, -3.0, 0.5)),
               toy_camera("C", (-3.0, -3.0, 0.5)), toy_camera("D", (0.0, -4.0, 2.0))]
    positions, _ = toy_hand(frames=6)
    observations = np.zeros((6, 4, 10, 3))
    for index, camera in enumerate(cameras):
        for frame in range(6):
            uv, _ = camera.project(positions[frame])
            observations[frame, index, :, :2] = uv
            observations[frame, index, :, 2] = 0.9
    weights = report.training_weights(cameras, observations, held_out=2)
    assert (weights[:, 2] == 0.0).all(), "the held-out camera must carry no weight"
    assert (weights[:, [0, 1, 3]] > 0.0).any()


def test_training_weights_do_not_depend_on_the_held_out_camera_s_pixels(report):
    """The leak this protocol is built to avoid, tested rather than asserted.

    Weighting all four cameras and then zeroing the held-out column would let the
    held-out view set the sigma of every training observation through the median
    epipolar distance. Corrupting the held-out camera's pixels must change nothing.
    """
    cameras = [toy_camera("A", (0.0, -4.0, 0.0)), toy_camera("B", (3.0, -3.0, 0.5)),
               toy_camera("C", (-3.0, -3.0, 0.5)), toy_camera("D", (0.0, -4.0, 2.0))]
    positions, _ = toy_hand(frames=6)
    observations = np.zeros((6, 4, 10, 3))
    for index, camera in enumerate(cameras):
        for frame in range(6):
            uv, _ = camera.project(positions[frame])
            observations[frame, index, :, :2] = uv
            observations[frame, index, :, 2] = 0.9
    clean = report.training_weights(cameras, observations, held_out=2)
    corrupted = observations.copy()
    corrupted[:, 2, :, :2] += 250.0
    assert np.array_equal(clean, report.training_weights(cameras, corrupted, held_out=2))


# --------------------------------------------------------------- the statistics

def test_block_draws_are_reused_and_respect_the_block_length(report):
    index = report.block_draws(150, 20, 5)
    assert index.shape[0] == 20
    assert np.array_equal(index, report.block_draws(150, 20, 5))
    assert index.max() < 150 and index.min() >= 0
    # every block of BLOCK_FRAMES consecutive columns must be consecutive frames
    block = index[0, :report.BLOCK_FRAMES]
    assert np.array_equal(np.diff(block), np.ones(report.BLOCK_FRAMES - 1))


def test_lag_one_autocorrelation_recovers_a_known_series(report):
    rng = np.random.default_rng(0)
    series = np.zeros(400)
    for frame in range(1, 400):
        series[frame] = 0.9 * series[frame - 1] + rng.normal(0.0, 0.2)
    pixels = series - series.min() + 1.0
    frames = np.arange(400)
    assert report.lag_one_autocorrelation(pixels, frames, 400) == pytest.approx(0.9, abs=0.1)


# ------------------------------------------------------------------- the layout

def test_the_smplx_layout_is_derived_and_refuses_a_moved_block(report):
    """The tip permutation is data, not an index convention.

    A synthetic SMPL-X-shaped skeleton whose tips are permuted must come back with
    the permutation recovered rather than assumed.
    """
    joints = np.zeros((20, 127, 3))
    joints[:, :, 0] = 5.0                       # everything far from both wrists
    rng = np.random.default_rng(1)
    for side, wrist, base, tip_base in (("l", 20, 25, 66), ("r", 21, 40, 71)):
        joints[:, wrist] = [0.0, 0.0, 0.0] if side == "l" else [1.0, 0.0, 0.0]
        origin = joints[0, wrist]
        for chain in range(5):
            for step in range(3):
                joints[:, base + 3 * chain + step] = origin + [
                    0.0, 0.02 * chain, 0.03 * (step + 1)]
        # tips deliberately in a scrambled order relative to the chain index
        scramble = [4, 0, 1, 2, 3]
        for chain in range(5):
            joints[:, tip_base + scramble[chain]] = origin + [
                0.0, 0.02 * chain, 0.115] + rng.normal(0.0, 1e-5, 3)
    layout = report.derive_smplx_hand_layout(joints)
    assert layout["l_index"]["tip"] == 66 + 4
    assert layout["l_middle"]["tip"] == 66 + 0
    assert layout["r_thumb"]["tip"] == 71 + 3

    moved = joints.copy()
    moved[:, 30] = [9.0, 9.0, 9.0]              # push one hand joint out of the block
    with pytest.raises(SystemExit):
        report.derive_smplx_hand_layout(moved)


# ------------------------------------------------------------------ the extractor

FIXTURE = {
    "summary": {"held_out_mean_over_hands_mm": 33.5, "hands_measured": 1,
                "controls_rejected": 2, "controls_scored": 2},
    "hands": {"subj0-l": {
        "held_out_summary": {"mean_over_folds_mm": 31.2, "worst_fold_mm": 41.5,
                             "best_fold_mm": 22.4, "folds_measured": 4},
        "folds": {"A001": {
            "test_observations_gated": 365, "test_observations_ungated": 553,
            "lag1_autocorrelation_of_the_candidate_residual_series": 0.71,
            "arms": {
                "candidate": {"median_px": 4.62, "p95_px": 18.0,
                              "median_mm_at_the_subject": 25.8},
                "CONTROL_frozen_rest_hand": {
                    "median_mm_at_the_subject": 122.7, "rejected": True,
                    "P_candidate_beats_this_arm_block_bootstrap": 1.0,
                    "rejected_by": "held-out reprojection AND amplitude"},
                "CONTROL_lag_3_frames": {
                    "median_mm_at_the_subject": 44.0, "rejected": True,
                    "P_candidate_beats_this_arm_block_bootstrap": 0.99,
                    "rejected_by": "held-out reprojection only"},
                "ORACLE_mamma_hand_joints": {
                    "median_mm_at_the_subject": 34.6,
                    "P_candidate_beats_this_arm_block_bootstrap": 0.2},
            }}},
        "articulation": {
            "ours": {"amplitude_mm": 48.8, "amplitude_percent_of_hand_length": 59.5,
                     "hand_length_wrist_to_middle1_mm": 82.1, "jitter_mm": 25.8,
                     "roughness": 0.53,
                     "fingertip_jitter_wrist_translation_removed_only_mm": 50.2,
                     "wrist_anchor_translational_jitter_mm": 8.8},
            "AGREEMENT_mamma": {"amplitude_mm": 22.8, "jitter_mm": 0.18,
                                "roughness": 0.008,
                                "amplitude_percent_of_hand_length": 19.1},
            "agreement_fingertips": {"median_mm": 40.0, "p95_mm": 105.5},
        }}},
}


def test_extractor_is_empty_without_a_report(extractor, tmp_path, monkeypatch):
    monkeypatch.setattr(extractor, "REPORT", tmp_path / "absent.json")
    assert extractor.x_hands({}) == ([], [])


def test_extractor_has_the_x_star_shape(extractor, tmp_path, monkeypatch):
    path = tmp_path / "hand-fit-heldout.json"
    path.write_text(json.dumps(FIXTURE))
    monkeypatch.setattr(extractor, "REPORT", path)
    figures, controls = extractor.x_hands({})
    assert figures and controls
    for row in figures + controls:
        assert set(row) == {"key", "label", "value", "unit", "reference", "better", "note"}
        assert row["reference"], "every figure names its reference"


def test_held_out_and_mamma_figures_never_share_a_reference(extractor, tmp_path,
                                                            monkeypatch):
    path = tmp_path / "hand-fit-heldout.json"
    path.write_text(json.dumps(FIXTURE))
    monkeypatch.setattr(extractor, "REPORT", path)
    figures, controls = extractor.x_hands({})
    held_out = {r["reference"] for r in figures if r["key"].startswith(("heldout", "fold"))}
    mamma = {r["reference"] for r in figures if "mamma" in r["key"]}
    assert held_out and mamma and not (held_out & mamma)
    assert all("pred_joints" not in r for r in held_out)


def test_the_thrash_figure_is_marked_as_a_quantity_the_solver_optimises(extractor,
                                                                        tmp_path,
                                                                        monkeypatch):
    path = tmp_path / "hand-fit-heldout.json"
    path.write_text(json.dumps(FIXTURE))
    monkeypatch.setattr(extractor, "REPORT", path)
    figures, _ = extractor.x_hands({})
    jitter = next(r for r in figures if r["key"].startswith("jitter_subj"))
    assert "SOLVER OPTIMISES" in jitter["reference"]
    assert jitter["better"] == "exposure count, not a score"


def test_every_control_names_the_figure_that_rejects_it(extractor, tmp_path, monkeypatch):
    path = tmp_path / "hand-fit-heldout.json"
    path.write_text(json.dumps(FIXTURE))
    monkeypatch.setattr(extractor, "REPORT", path)
    _, controls = extractor.x_hands({})
    for row in controls:
        if row["key"].startswith("CONTROL"):
            assert "Rejected by" in row["note"]
            assert "jitter" not in row["note"].split("Rejected by")[1].split(".")[0]
