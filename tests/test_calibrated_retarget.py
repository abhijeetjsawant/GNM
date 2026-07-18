from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoanim_gnm.a2f import ARKitGNMRetargeter
from autoanim_gnm.calibrated_retarget import (
    CalibratedRetargetError,
    CalibratedRetargeter,
    CalibrationCacheMismatch,
    CalibrationConfig,
    DenseRetargetCalibration,
    RegionSpec,
    SourceRigGeometry,
    build_dense_calibration,
)


REAL_ASSET_DIRECTORY = Path(".cache/autoanim_gnm/a2f-claire")


@pytest.fixture(scope="module")
def synthetic_calibration() -> tuple[DenseRetargetCalibration, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(47)
    target = rng.normal(size=(180, 3)) * np.asarray([1.0, 0.78, 0.52])
    # Break ellipsoid sign symmetries so PCA initialization has one correct
    # proper rotation rather than several equally plausible orientations.
    target[:, 2] += 0.16 * target[:, 0] ** 2 + 0.05 * target[:, 1]
    basis = np.zeros((10, len(target), 3), dtype=np.float64)
    basis[:4, :70] = rng.normal(scale=0.035, size=(4, 70, 3))
    basis[4:8, 70:140] = rng.normal(scale=0.035, size=(4, 70, 3))
    basis[8:, 140:] = rng.normal(scale=0.035, size=(2, 40, 3))

    rotation = Rotation.from_euler("zyx", [31.0, -14.0, 9.0], degrees=True).as_matrix()
    scale = 1.65
    translation = np.asarray([1.7, -0.8, 0.55])
    source_neutral = ((target - translation) @ rotation) / scale
    skin_truth = np.asarray(
        [
            [0.55, -0.18, 0.12, 0.27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.24, 0.48, 0.16, 0.31, 0.0, 0.0],
            [-0.28, 0.11, 0.19, -0.14, 0.21, -0.13, 0.09, 0.18, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    target_skin_deltas = np.einsum("ce,evj->cvj", skin_truth, basis)
    source_skin_deltas = (target_skin_deltas @ rotation) / scale
    skin = SourceRigGeometry(
        source_neutral,
        source_skin_deltas,
        ("leftControl", "rightControl", "depthControl"),
    )

    tongue_indices = np.arange(140, 180)
    tongue_truth = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.61, -0.22],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.17, 0.52],
        ],
        dtype=np.float64,
    )
    target_tongue_deltas = np.einsum(
        "ce,evj->cvj", tongue_truth[:, 8:], basis[8:, tongue_indices]
    )
    source_tongue_deltas = (target_tongue_deltas @ rotation) / scale
    tongue = SourceRigGeometry(
        source_neutral[tongue_indices],
        source_tongue_deltas,
        ("tongueUp", "tongueDown"),
    )
    regions = (
        RegionSpec("left", 0, 4, 1.0),
        RegionSpec("right", 4, 8, 1.0),
        RegionSpec("tongue", 8, 10, 1.0),
    )
    calibration = build_dense_calibration(
        skin,
        target,
        basis,
        tongue_source=tongue,
        regions=regions,
        skin_region_names=("left", "right"),
        tongue_region_names=("tongue",),
        config=CalibrationConfig(
            alignment_max_points=len(target),
            alignment_iterations=30,
            correspondence_neighbors=1,
            ridge_regularization=1e-6,
            coefficient_bound=1.0,
        ),
    )
    return calibration, skin_truth, tongue_truth


def test_synthetic_geometry_calibration_recovers_dense_asymmetric_mapping(
    synthetic_calibration,
) -> None:
    calibration, skin_truth, tongue_truth = synthetic_calibration

    assert calibration.skin_matrix.shape == (3, 10)
    assert calibration.tongue_matrix.shape == (2, 10)
    assert calibration.skin_pose_names == (
        "leftControl",
        "rightControl",
        "depthControl",
    )
    assert calibration.tongue_pose_names == ("tongueUp", "tongueDown")
    np.testing.assert_allclose(calibration.skin_matrix, skin_truth, atol=6e-4)
    np.testing.assert_allclose(calibration.tongue_matrix, tongue_truth, atol=6e-4)
    assert not np.allclose(calibration.skin_matrix[0], calibration.skin_matrix[1])
    assert calibration.metadata["alignment"]["normalized_trimmed_rms"] < 1e-10
    assert len(calibration.calibration_hash) == 64


def test_synthetic_runtime_retarget_is_finite_bounded_and_validated(
    synthetic_calibration,
) -> None:
    calibration, _, _ = synthetic_calibration
    retargeter = CalibratedRetargeter(calibration)
    controls = retargeter.retarget(
        {"leftControl": 4.0, "rightControl": 3.0, "depthControl": 2.0},
        {"tongueUp": 3.0, "tongueDown": 2.0},
    )
    assert controls.shape == (10,)
    assert np.isfinite(controls).all()
    for region in calibration.regions:
        assert np.max(np.abs(controls[region.start : region.stop])) <= region.bound + 1e-7

    sequence = retargeter.retarget_sequence(
        np.asarray([[0.0, 0.0], [0.7, 0.4]], dtype=np.float32),
        ("leftControl", "rightControl"),
        tongue_weights=np.asarray([[0.0], [0.6]], dtype=np.float32),
        tongue_pose_names=("tongueUp",),
    )
    np.testing.assert_allclose(sequence[0], 0.0)
    np.testing.assert_allclose(
        sequence[1],
        retargeter.retarget(
            {"leftControl": 0.7, "rightControl": 0.4}, {"tongueUp": 0.6}
        ),
        atol=1e-7,
    )
    with pytest.raises(CalibratedRetargetError, match="not finite"):
        retargeter.retarget({"leftControl": np.nan})
    with pytest.raises(CalibratedRetargetError, match="Unknown skin controls"):
        retargeter.retarget({"notAControl": 1.0}, strict=True)
    with pytest.raises(CalibratedRetargetError, match="Expected skin weights"):
        retargeter.retarget_sequence(np.zeros((2, 3)), ("leftControl",))


def test_calibration_cache_is_deterministic_pickle_free_and_hash_checked(
    synthetic_calibration, tmp_path: Path
) -> None:
    calibration, _, _ = synthetic_calibration
    first = calibration.save(tmp_path / "first.npz")
    second = calibration.save(tmp_path / "second.npz")
    assert sha256(first.read_bytes()).digest() == sha256(second.read_bytes()).digest()

    loaded = DenseRetargetCalibration.load(
        first, expected_request_hash=calibration.metadata["request_hash"]
    )
    assert loaded.calibration_hash == calibration.calibration_hash
    np.testing.assert_array_equal(loaded.skin_matrix, calibration.skin_matrix)
    np.testing.assert_array_equal(loaded.tongue_matrix, calibration.tongue_matrix)
    np.testing.assert_array_equal(
        CalibratedRetargeter(loaded).retarget({"depthControl": 0.73}),
        CalibratedRetargeter(calibration).retarget({"depthControl": 0.73}),
    )
    with pytest.raises(CalibrationCacheMismatch, match="different assets"):
        DenseRetargetCalibration.load(first, expected_request_hash="0" * 64)


def test_calibration_rejects_invalid_geometry_and_region_contract() -> None:
    neutral = np.arange(36, dtype=np.float64).reshape(12, 3)
    with pytest.raises(CalibratedRetargetError, match="deltas must have shape"):
        build_dense_calibration(
            SourceRigGeometry(neutral, np.zeros((1, 11, 3)), ("pose",)),
            neutral,
            np.zeros((1, 12, 3)),
            regions=(RegionSpec("face", 0, 1),),
            skin_region_names=("face",),
            tongue_region_names=(),
        )
    with pytest.raises(CalibratedRetargetError, match="overlap"):
        build_dense_calibration(
            SourceRigGeometry(neutral, np.zeros((1, 12, 3)), ("pose",)),
            neutral,
            np.ones((2, 12, 3)),
            regions=(RegionSpec("one", 0, 2), RegionSpec("two", 1, 2)),
            skin_region_names=("one",),
            tongue_region_names=(),
        )


@pytest.fixture(scope="module")
def real_retargeter(tmp_path_factory: pytest.TempPathFactory) -> CalibratedRetargeter:
    if not (
        (REAL_ASSET_DIRECTORY / "bs_skin.npz").is_file()
        and (REAL_ASSET_DIRECTORY / "bs_tongue.npz").is_file()
    ):
        pytest.skip("released Claire runtime assets are not available")
    cache = tmp_path_factory.mktemp("calibrated-retarget") / "claire-gnm.npz"
    built = CalibratedRetargeter.from_directory(
        REAL_ASSET_DIRECTORY, cache_path=cache, force_rebuild=True
    )
    # Exercise the fingerprint/config cache gate rather than merely the raw
    # serialization path covered by the synthetic test.
    loaded = CalibratedRetargeter.from_directory(
        REAL_ASSET_DIRECTORY, cache_path=cache
    )
    assert loaded.calibration.calibration_hash == built.calibration.calibration_hash
    return loaded


def test_released_claire_calibration_preserves_controls_and_asymmetry(
    real_retargeter: CalibratedRetargeter,
) -> None:
    calibration = real_retargeter.calibration
    assert calibration.skin_matrix.shape == (52, 383)
    assert calibration.tongue_matrix.shape == (16, 383)
    assert len(calibration.skin_pose_names) == len(set(calibration.skin_pose_names)) == 52
    assert len(calibration.tongue_pose_names) == len(set(calibration.tongue_pose_names)) == 16
    # Claire publishes tongueOut as an all-zero skin delta; every other skin
    # channel and all 16 released tongue controls retain a geometric mapping.
    nonzero_skin = {
        name
        for name, row in zip(
            calibration.skin_pose_names, calibration.skin_matrix, strict=True
        )
        if np.linalg.norm(row) > 1e-5
    }
    assert nonzero_skin == set(calibration.skin_pose_names) - {"tongueOut"}
    assert np.all(np.linalg.norm(calibration.tongue_matrix, axis=1) > 1e-5)

    dimple_left = calibration.skin_matrix[
        calibration.skin_pose_names.index("mouthDimpleLeft")
    ]
    dimple_right = calibration.skin_matrix[
        calibration.skin_pose_names.index("mouthDimpleRight")
    ]
    jaw_forward = calibration.skin_matrix[
        calibration.skin_pose_names.index("jawForward")
    ]
    assert np.linalg.norm(dimple_left) > 0.1
    assert np.linalg.norm(dimple_right) > 0.1
    assert np.linalg.norm(jaw_forward) > 0.1
    assert np.linalg.norm(dimple_left - dimple_right) > 0.1

    controls = real_retargeter.retarget(
        {"mouthDimpleLeft": 1.0, "jawForward": 1.0},
        {"tongueTipUp": 1.0, "tongueLeft": 0.8},
    )
    assert controls.shape == (383,)
    assert np.isfinite(controls).all()
    for region in calibration.regions:
        assert np.max(np.abs(controls[region.start : region.stop])) <= region.bound + 1e-6


def test_dense_geometry_fit_beats_current_semantic_collapse_for_unmapped_controls(
    real_retargeter: CalibratedRetargeter, rig
) -> None:
    semantic = ARKitGNMRetargeter(rig)
    diagnostics = real_retargeter.calibration.metadata["skin_channel_fit"]
    # The current semantic table has no rule for these real ARKit controls, so
    # its prediction is exactly the zero-mapping baseline recorded by the
    # geometry calibration.  The dense solve must materially reduce that same
    # transferred-surface target error.
    for name in ("mouthDimpleLeft", "mouthDimpleRight", "jawForward"):
        np.testing.assert_allclose(semantic.retarget({name: 1.0}), 0.0)
        assert diagnostics[name]["zero_mapping_weighted_error"] > 0
        assert diagnostics[name]["fitted_weighted_error"] < (
            0.75 * diagnostics[name]["zero_mapping_weighted_error"]
        )
