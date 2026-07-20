from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.a2f_v3_postprocess import (
    A2FV3PostprocessError,
    ClaireV3PostprocessChunk,
    ClaireV3Postprocessor,
    EYE_POSTPROCESS_STATUS,
    RAW_PREDICTION_FLOATS,
    _CascadedDegreeTwoInterpolator,
    _EyeAnimator,
    _RegularizedBVLSSolver,
    _SolverConfig,
    _face_mask_lower,
    _jaw_kabsch,
    split_claire_v3_raw_prediction,
)
from autoanim_gnm.a2f_v3_profile import OFFICIAL_V3_ASSET_SHA256
from autoanim_gnm.calibrated_retarget import SourceRigGeometry


REAL_PROFILE = Path(".cache/autoanim_gnm/a2f-v3-claire-profile")


def test_raw_prediction_split_is_exact_and_fails_closed() -> None:
    raw = np.arange(RAW_PREDICTION_FLOATS, dtype=np.float32)
    parts = split_claire_v3_raw_prediction(raw)
    assert parts.skin_deltas.shape == (1, 24_002, 3)
    assert parts.tongue_deltas.shape == (1, 5_602, 3)
    assert parts.jaw_deltas.shape == (1, 5, 3)
    assert parts.eye_rotations_raw_degrees.shape == (1, 4)
    assert parts.skin_deltas[0, 0, 0] == 0.0
    assert parts.tongue_deltas[0, 0, 0] == 72_006.0
    assert parts.jaw_deltas[0, 0, 0] == 88_812.0
    assert parts.eye_rotations_raw_degrees[0, 0] == 88_827.0

    with pytest.raises(A2FV3PostprocessError, match="shape"):
        split_claire_v3_raw_prediction(raw[:-1])
    nonfinite = raw.copy()
    nonfinite[12] = np.nan
    with pytest.raises(A2FV3PostprocessError, match="finite"):
        split_claire_v3_raw_prediction(nonfinite)
    with pytest.raises(A2FV3PostprocessError, match="shape"):
        split_claire_v3_raw_prediction(np.empty((0, RAW_PREDICTION_FLOATS)))


def test_degree_two_interpolator_matches_sdk_recurrence_across_chunks() -> None:
    smoothing = 0.02
    dt = 0.01
    alpha = 1.0 - 0.5 ** (dt / smoothing)
    interpolator = _CascadedDegreeTwoInterpolator(smoothing, (1,))
    first = interpolator.update(np.array([0.0]), dt)
    second = interpolator.update(np.array([1.0]), dt)
    third = interpolator.update(np.array([1.0]), dt)
    np.testing.assert_array_equal(first, [0.0])
    np.testing.assert_allclose(second, [alpha**2], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        third, [3.0 * alpha**2 - 2.0 * alpha**3], rtol=0.0, atol=1e-15
    )

    interpolator.reset()
    np.testing.assert_array_equal(interpolator.update(np.array([7.0]), dt), [7.0])


def test_face_mask_matches_pinned_logistic_lower_upper_composition() -> None:
    neutral = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    level = 0.6
    softness = 0.1
    expected = 1.0 / (
        1.0 + np.exp(-(level - neutral[:, 1]) / softness)
    )
    np.testing.assert_allclose(
        _face_mask_lower(neutral, level, softness), expected, rtol=0.0, atol=1e-15
    )
    assert expected[0] > expected[1] > expected[2]
    with pytest.raises(A2FV3PostprocessError, match="extent"):
        _face_mask_lower(np.zeros((3, 3)), level, softness)


def _synthetic_solver() -> _RegularizedBVLSSolver:
    neutral = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    deltas = np.zeros((5, 4, 3), dtype=np.float64)
    for pose, flat_index in enumerate((0, 1, 3, 4, 6)):
        deltas[pose].reshape(-1)[flat_index] = 1.0
    rig = SourceRigGeometry(
        neutral=neutral,
        deltas=deltas,
        pose_names=("cancelWinner", "cancelLoser", "symL", "symR", "offsetPose"),
    )
    config = _SolverConfig(
        active=np.ones(5, dtype=np.int64),
        cancel_groups=np.array([0, 0, -1, -1, -1]),
        symmetry_groups=np.array([-1, -1, 1, 1, -1]),
        multipliers=np.array([2.0, 1.0, 1.0, 1.0, 1.0]),
        offsets=np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
        l1_regularization=0.05,
        l2_regularization=0.02,
        temporal_regularization=0.1,
        symmetry_regularization=10.0,
        template_bb_size=np.sqrt(3.0),
    )
    return _RegularizedBVLSSolver(rig, config, mask=None, label="synthetic")


def test_regularized_bvls_applies_cancel_symmetry_multiplier_offset_and_state() -> None:
    solver = _synthetic_solver()
    neutral = solver.neutral
    target = neutral.copy()
    target.reshape(-1)[[0, 1, 3, 4, 6]] += [0.8, 0.2, 0.6, 0.6, 0.3]
    first = solver.solve_pose(target)
    assert first[0] == pytest.approx(2.0 * solver._previous[0], abs=1e-8)
    assert first[1] <= 1.0e-8  # Smaller member of the cancel pair is suppressed.
    np.testing.assert_allclose(first[2], first[3], rtol=0.0, atol=1e-7)
    assert first[4] > 0.1  # Published offset is applied after the bounded solve.

    second = solver.solve_pose(neutral)
    assert np.max(second[:4]) > 0.0  # Temporal RHS carries prior solved weights.
    solver.reset()
    reset = solver.solve_pose(neutral)
    np.testing.assert_allclose(reset[:4], 0.0, rtol=0.0, atol=1e-12)
    assert reset[4] == pytest.approx(0.1)


def test_regularized_bvls_rejects_invalid_pair_and_nonfinite_target() -> None:
    solver = _synthetic_solver()
    invalid = _SolverConfig(
        active=np.ones(5, dtype=np.int64),
        cancel_groups=np.array([0, -1, -1, -1, -1]),
        symmetry_groups=np.full(5, -1, dtype=np.int64),
        multipliers=np.ones(5),
        offsets=np.zeros(5),
        l1_regularization=0.0,
        l2_regularization=0.1,
        temporal_regularization=0.0,
        symmetry_regularization=0.0,
        template_bb_size=1.0,
    )
    rig = SourceRigGeometry(
        neutral=solver.neutral,
        deltas=np.zeros((5, 4, 3)),
        pose_names=("a", "b", "c", "d", "e"),
    )
    with pytest.raises(A2FV3PostprocessError, match="must contain two"):
        _RegularizedBVLSSolver(rig, invalid, mask=None, label="invalid")
    target = solver.neutral.copy()
    target[0, 0] = np.inf
    with pytest.raises(A2FV3PostprocessError, match="non-finite"):
        solver.solve_pose(target)


def test_five_point_jaw_kabsch_recovers_proper_transform_and_layouts() -> None:
    neutral = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, -0.5, 0.2],
        ],
        dtype=np.float64,
    )
    angle = np.deg2rad(23.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.25, -0.4, 0.7])
    observed = neutral @ rotation.T + translation
    transform, residual = _jaw_kabsch(
        neutral,
        observed - neutral,
        strength=1.0,
        height_offset=0.0,
        depth_offset=0.0,
    )
    np.testing.assert_allclose(transform[:3, :3], rotation, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(transform[:3, 3], translation, rtol=0.0, atol=2e-7)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0, abs=2e-7)
    assert residual < 1.0e-12
    assert transform.reshape(16)[3] == pytest.approx(translation[0])
    assert transform.reshape(16, order="F")[12] == pytest.approx(translation[0])


def test_eye_animator_repeats_30hz_saccade_at_60hz_and_carries_across_chunks() -> None:
    saccades = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    animator = _EyeAnimator(
        saccades,
        eyeballs_strength=2.0,
        saccade_strength=0.5,
        right_offset=(0.1, -0.2),
        left_offset=(-0.3, 0.4),
        saccade_seed=0.0,
    )
    raw = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    first = animator.update(raw, 1.0 / 60.0)
    second = animator.update(raw, 1.0 / 60.0)
    third = animator.update(raw, 1.0 / 60.0)
    expected_first = np.array([20.6, 40.8, 60.2, 81.4], dtype=np.float32)
    expected_third = np.array([21.6, 41.8, 61.2, 82.4], dtype=np.float32)
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second, expected_first)
    np.testing.assert_array_equal(third, expected_third)
    animator.reset()
    np.testing.assert_array_equal(animator.update(raw, 1.0 / 60.0), expected_first)


@pytest.mark.skipif(not REAL_PROFILE.is_dir(), reason="official public v3 assets not cached")
def test_pinned_real_profile_streaming_matches_whole_sequence_and_sdk_eyes() -> None:
    processor = ClaireV3Postprocessor.from_directory(REAL_PROFILE)
    raw = np.zeros((3, RAW_PREDICTION_FLOATS), dtype=np.float32)
    raw[:, 0] = [0.0, 0.1, -0.05]
    raw[:, 72_006] = [0.0, 0.03, -0.02]
    raw[:, -4:] = [1.0, 2.0, 3.0, 4.0]
    whole = processor.process_sequence(raw, include_geometry=True)

    processor.reset()
    streamed = ClaireV3PostprocessChunk.concatenate(
        [
            processor.process_chunk(raw[:1], include_geometry=True),
            processor.process_chunk(raw[1:], include_geometry=True),
        ]
    )
    for field in (
        "skin_weights",
        "tongue_weights",
        "jaw_transforms",
        "jaw_transform_row_major",
        "jaw_transform_nvidia_column_major",
        "jaw_rms_residual",
        "eye_rotations_degrees",
        "eye_rotations_raw_degrees",
        "skin_geometry",
        "tongue_geometry",
    ):
        np.testing.assert_array_equal(getattr(streamed, field), getattr(whole, field))
    assert whole.eye_postprocess_status == EYE_POSTPROCESS_STATUS
    np.testing.assert_array_equal(whole.eye_rotations_raw_degrees[0], [1.0, 2.0, 3.0, 4.0])
    with np.load(REAL_PROFILE / "model_data_Claire.npz", allow_pickle=False) as values:
        saccade = np.asarray(values["saccade_rot_matrix"], dtype=np.float32)
    expected_eye = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected_eye[:2] += 0.6 * saccade[0]
    expected_eye[2:] += 0.6 * saccade[0]
    np.testing.assert_allclose(whole.eye_rotations_degrees[0], expected_eye, atol=2e-7)
    np.testing.assert_array_equal(whole.eye_rotations_degrees[0], whole.eye_rotations_degrees[1])
    assert whole.tongue_weights[0, 9] == pytest.approx(0.2, abs=1e-6)
    np.testing.assert_allclose(
        whole.jaw_transforms, np.tile(np.eye(4), (3, 1, 1)), atol=1e-6
    )
    assert not whole.skin_weights.flags.writeable


@pytest.mark.skipif(not REAL_PROFILE.is_dir(), reason="official public v3 assets not cached")
def test_profile_tamper_fails_before_postprocessing(tmp_path: Path) -> None:
    for name in OFFICIAL_V3_ASSET_SHA256:
        if name == "network.onnx":
            continue
        (tmp_path / name).symlink_to((REAL_PROFILE / name).resolve())
    config_path = tmp_path / "model_config_Claire.json"
    config_path.unlink()
    config_path.write_text('{"config":{"skin_strength":99}}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        ClaireV3Postprocessor.from_directory(tmp_path)
