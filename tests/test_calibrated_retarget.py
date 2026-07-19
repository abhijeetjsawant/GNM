from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoanim_gnm.a2f import ARKitGNMRetargeter
import autoanim_gnm.calibrated_retarget as calibrated_retarget_module
from autoanim_gnm.calibrated_retarget import (
    CLAIRE_V3_ASSET_MANIFEST_FILENAME,
    CLAIRE_V3_ASSET_MANIFEST_SCHEMA,
    CLAIRE_V3_HF_REVISION,
    CLAIRE_V3_MODEL_ID,
    CLAIRE_V3_SKIN_ACTIVE,
    CLAIRE_V3_SKIN_POSE_NAMES,
    CLAIRE_V3_TONGUE_MULTIPLIERS,
    CLAIRE_V3_TONGUE_OFFSETS,
    CLAIRE_V3_TONGUE_POSE_NAMES,
    CalibratedRetargetError,
    CalibratedRetargeter,
    CalibrationCacheMismatch,
    CalibrationConfig,
    ClaireV3BlendshapeGeometry,
    DenseRetargetCalibration,
    PostSolverControlRanges,
    RegionSpec,
    SourceRigGeometry,
    build_dense_calibration,
)


REAL_ASSET_DIRECTORY = Path(".cache/autoanim_gnm/a2f-claire")
REAL_V3_PROFILE_DIRECTORY = Path(".cache/autoanim_gnm/a2f-v3-claire-profile")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_synthetic_v3_assets(root: Path) -> None:
    """Write tiny topology, but the exact pinned Claire v3 control schema."""

    root.mkdir(parents=True)
    skin_x = np.linspace(-1.0, 1.0, 24)
    skin_neutral = np.column_stack(
        (skin_x, 0.4 * skin_x**2 + 0.1 * skin_x, np.sin(skin_x))
    ).astype(np.float32)
    tongue_x = np.linspace(-0.5, 0.5, 20)
    tongue_neutral = np.column_stack(
        (tongue_x, 0.2 * tongue_x**2, np.cos(tongue_x))
    ).astype(np.float32)
    rng = np.random.default_rng(20260719)
    skin_payload: dict[str, np.ndarray] = {
        "neutral": skin_neutral,
        "poseNames": np.asarray(("neutral", *CLAIRE_V3_SKIN_POSE_NAMES), dtype="S19"),
        "frontalMask": np.arange(20, dtype=np.int32),
        "rig_version": np.asarray(b"v3.6"),
    }
    for name in CLAIRE_V3_SKIN_POSE_NAMES:
        skin_payload[name] = rng.normal(0.0, 0.01, skin_neutral.shape).astype(np.float32)
    np.savez(root / "bs_skin_Claire.npz", **skin_payload)

    tongue_payload: dict[str, np.ndarray] = {
        "neutral": tongue_neutral,
        "poseNames": np.asarray(
            ("neutral", *CLAIRE_V3_TONGUE_POSE_NAMES), dtype="S15"
        ),
        "rig_version": np.asarray(b"v1.0"),
    }
    for name in CLAIRE_V3_TONGUE_POSE_NAMES:
        tongue_payload[name] = rng.normal(0.0, 0.01, tongue_neutral.shape).astype(
            np.float32
        )
    np.savez(root / "bs_tongue_Claire.npz", **tongue_payload)

    np.savez(
        root / "model_data_Claire.npz",
        neutral_jaw=np.zeros((5, 3), dtype=np.float32),
        neutral_skin=skin_neutral,
        neutral_tongue=tongue_neutral,
        lip_open_pose_delta=np.zeros_like(skin_neutral),
        eye_close_pose_delta=np.zeros_like(skin_neutral),
        saccade_rot_matrix=np.zeros((5_000, 2), dtype=np.float32),
    )
    _write_json(
        root / "network_info.json",
        {
            "id": {
                "type": "diffusion",
                "actor": "multi",
                "version": "3.2",
                "output": "geometry",
            },
            "params": {
                "identities": ["Claire", "James", "Mark"],
                "skin_size": int(skin_neutral.size),
                "tongue_size": int(tongue_neutral.size),
                "jaw_size": 15,
                "eyes_size": 4,
                "num_diffusion_steps": 2,
                "num_gru_layers": 2,
                "gru_latent_dim": 256,
                "num_frames_left_truncate": 15,
                "num_frames_right_truncate": 15,
                "num_frames_center": 30,
            },
            "audio_params": {
                "buffer_len": 16_000,
                "padding_left": 16_000,
                "padding_right": 16_000,
                "samplerate": 16_000,
            },
        },
    )
    _write_json(
        root / "model.json",
        {
            "networkInfoPath": "network_info.json",
            "networkPath": "network.trt",
            "modelConfigPaths": [
                "model_config_Claire.json",
                "model_config_James.json",
                "model_config_Mark.json",
            ],
            "modelDataPaths": [
                "model_data_Claire.npz",
                "model_data_James.npz",
                "model_data_Mark.npz",
            ],
            "blendshapePaths": [
                {
                    part: {
                        "config": f"bs_{part}_config_{identity}.json",
                        "data": f"bs_{part}_{identity}.npz",
                    }
                    for part in ("skin", "tongue")
                }
                for identity in ("Claire", "James", "Mark")
            ],
        },
    )
    _write_json(
        root / "bs_skin_config_Claire.json",
        {
            "blendshape_params": {
                "numPoses": len(CLAIRE_V3_SKIN_POSE_NAMES),
                "bsSolveActivePoses": list(CLAIRE_V3_SKIN_ACTIVE),
                "bsWeightMultipliers": [1.0] * len(CLAIRE_V3_SKIN_POSE_NAMES),
                "bsWeightOffsets": [0.0] * len(CLAIRE_V3_SKIN_POSE_NAMES),
            }
        },
    )
    _write_json(
        root / "bs_tongue_config_Claire.json",
        {
            "blendshape_params": {
                "numPoses": len(CLAIRE_V3_TONGUE_POSE_NAMES),
                "bsSolveActivePoses": [1] * len(CLAIRE_V3_TONGUE_POSE_NAMES),
                "bsWeightMultipliers": list(CLAIRE_V3_TONGUE_MULTIPLIERS),
                "bsWeightOffsets": list(CLAIRE_V3_TONGUE_OFFSETS),
            }
        },
    )
    _write_json(
        root / CLAIRE_V3_ASSET_MANIFEST_FILENAME,
        {
            "schema_version": CLAIRE_V3_ASSET_MANIFEST_SCHEMA,
            "model_id": CLAIRE_V3_MODEL_ID,
            "revision": CLAIRE_V3_HF_REVISION,
        },
    )


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


def test_post_solver_retarget_preserves_valid_above_one_tongue_amplitudes(
    synthetic_calibration,
) -> None:
    calibration, _, _ = synthetic_calibration
    ranges = PostSolverControlRanges(
        skin_pose_names=calibration.skin_pose_names,
        skin_minimum=np.zeros(3),
        skin_maximum=np.ones(3),
        tongue_pose_names=calibration.tongue_pose_names,
        tongue_minimum=np.asarray([0.0, 0.2]),
        tongue_maximum=np.asarray([2.0, 1.2]),
    )
    retargeter = CalibratedRetargeter(
        calibration, post_solver_ranges=ranges
    )
    result = retargeter.retarget_post_solver_sequence(
        np.zeros((1, 3), dtype=np.float32),
        calibration.skin_pose_names,
        tongue_weights=np.asarray([[0.2, 1.5]], dtype=np.float32),
        tongue_pose_names=("tongueDown", "tongueUp"),
    )
    expected_unbounded = (
        1.5 * calibration.tongue_matrix[calibration.tongue_pose_names.index("tongueUp")]
        + 0.2
        * calibration.tongue_matrix[calibration.tongue_pose_names.index("tongueDown")]
    )
    np.testing.assert_allclose(result[0], retargeter._bound(expected_unbounded), atol=1e-7)

    legacy = retargeter.retarget(
        {}, {"tongueUp": 1.5, "tongueDown": 0.2}
    )
    assert not np.allclose(result[0], legacy)
    with pytest.raises(CalibratedRetargetError, match="outside"):
        retargeter.retarget_post_solver_sequence(
            np.zeros((1, 3)),
            calibration.skin_pose_names,
            tongue_weights=np.asarray([[0.2, 2.01]]),
            tongue_pose_names=("tongueDown", "tongueUp"),
        )
    with pytest.raises(CalibratedRetargetError, match="outside"):
        retargeter.retarget_post_solver_sequence(
            np.zeros((1, 3)),
            calibration.skin_pose_names,
            tongue_weights=np.asarray([[0.0, 0.5]]),
            tongue_pose_names=("tongueDown", "tongueUp"),
        )
    with pytest.raises(CalibratedRetargetError, match="schema mismatch"):
        retargeter.retarget_post_solver_sequence(
            np.zeros((1, 2)),
            calibration.skin_pose_names[:2],
            tongue_weights=np.asarray([[0.2, 0.5]]),
            tongue_pose_names=("tongueDown", "tongueUp"),
        )
    with pytest.raises(CalibratedRetargetError, match="non-finite"):
        retargeter.retarget_post_solver_sequence(
            np.asarray([[np.nan, 0.0, 0.0]]),
            calibration.skin_pose_names,
            tongue_weights=np.asarray([[0.2, 0.5]]),
            tongue_pose_names=("tongueDown", "tongueUp"),
        )


def test_from_v3_directory_wires_separate_assets_cache_and_ranges(
    synthetic_calibration,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration, _, _ = synthetic_calibration
    ranges = PostSolverControlRanges(
        skin_pose_names=calibration.skin_pose_names,
        skin_minimum=np.zeros(3),
        skin_maximum=np.ones(3),
        tongue_pose_names=calibration.tongue_pose_names,
        tongue_minimum=np.zeros(2),
        tongue_maximum=np.asarray([2.0, 1.2]),
    )
    neutral = np.column_stack(
        (np.linspace(-1.0, 1.0, 20), np.linspace(0.0, 0.5, 20) ** 2, np.ones(20))
    )
    skin = SourceRigGeometry(
        neutral,
        np.zeros((3, 20, 3)),
        calibration.skin_pose_names,
        np.arange(16),
    )
    tongue = SourceRigGeometry(
        neutral,
        np.zeros((2, 20, 3)),
        calibration.tongue_pose_names,
    )
    assets = SimpleNamespace(
        root=tmp_path,
        source_fingerprint="1" * 64,
        skin=skin,
        tongue=tongue,
        control_ranges=ranges,
    )
    monkeypatch.setattr(
        ClaireV3BlendshapeGeometry,
        "load",
        classmethod(lambda cls, directory, expected_revision: assets),
    )
    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return calibration

    monkeypatch.setattr(calibrated_retarget_module, "build_dense_calibration", fake_build)
    target = np.zeros((20, 3), dtype=np.float32)
    basis = np.zeros((383, 20, 3), dtype=np.float32)

    class FakeAdapter:
        expression_dim = 383
        model = SimpleNamespace(
            version="test",
            variant="test",
            template_vertex_positions=target,
            expression_basis=basis,
            expression_names=tuple(f"expression-{index}" for index in range(383)),
        )

        @staticmethod
        def vertex_group(name: str) -> np.ndarray:
            return np.ones(20, dtype=np.float32)

    result = CalibratedRetargeter.from_v3_directory(
        tmp_path,
        adapter=FakeAdapter(),
        cache_path=tmp_path / "separate-v3-cache.npz",
        force_rebuild=True,
    )
    assert result.post_solver_ranges is ranges
    assert captured["args"][0] is skin
    assert captured["kwargs"]["tongue_source"] is tongue
    assert captured["kwargs"]["source_fingerprint"] == "1" * 64
    assert (tmp_path / "separate-v3-cache.npz").is_file()


def test_synthetic_v3_loader_pins_schema_geometry_and_solver_ranges(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3-profile"
    _write_synthetic_v3_assets(root)
    assets = ClaireV3BlendshapeGeometry.load(root)

    assert assets.revision == CLAIRE_V3_HF_REVISION
    assert assets.network_version == "3.2"
    assert assets.identity == "Claire"
    assert assets.identity_index == 0
    assert assets.skin.pose_names == CLAIRE_V3_SKIN_POSE_NAMES
    assert assets.tongue.pose_names == CLAIRE_V3_TONGUE_POSE_NAMES
    assert len(assets.source_fingerprint) == 64
    ranges = assets.control_ranges
    assert ranges.skin_maximum[CLAIRE_V3_SKIN_POSE_NAMES.index("eyeLookDownLeft")] == 0.0
    assert ranges.tongue_maximum[CLAIRE_V3_TONGUE_POSE_NAMES.index("tongueTipUp")] == 2.0
    assert ranges.tongue_maximum[CLAIRE_V3_TONGUE_POSE_NAMES.index("tongueRollUp")] == 3.0
    assert ranges.tongue_maximum[CLAIRE_V3_TONGUE_POSE_NAMES.index("tongueUp")] == 2.0
    tongue_down = CLAIRE_V3_TONGUE_POSE_NAMES.index("tongueDown")
    assert ranges.tongue_minimum[tongue_down] == pytest.approx(0.2)
    assert ranges.tongue_maximum[tongue_down] == pytest.approx(1.2)

    manifest_path = root / CLAIRE_V3_ASSET_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "not-the-pinned-revision"
    _write_json(manifest_path, manifest)
    with pytest.raises(CalibrationCacheMismatch, match="revision"):
        ClaireV3BlendshapeGeometry.load(root)


def test_synthetic_v3_loader_rejects_modified_post_solver_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v3-profile"
    _write_synthetic_v3_assets(root)
    config_path = root / "bs_tongue_config_Claire.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["blendshape_params"]["bsWeightMultipliers"][0] = 1.0
    _write_json(config_path, config)
    with pytest.raises(CalibrationCacheMismatch, match="pinned release"):
        ClaireV3BlendshapeGeometry.load(root)


def test_official_v3_claire_profile_calibrates_and_reloads_when_available(
    tmp_path: Path,
) -> None:
    if not REAL_V3_PROFILE_DIRECTORY.is_dir():
        pytest.skip("official pinned Claire v3 profile is not available")
    assets = ClaireV3BlendshapeGeometry.load(REAL_V3_PROFILE_DIRECTORY)
    assert assets.skin.neutral.shape == (24_002, 3)
    assert assets.tongue.neutral.shape == (5_602, 3)
    assert assets.control_ranges.tongue_maximum.max() == 3.0
    cache = tmp_path / "official-v3-calibration.npz"
    built = CalibratedRetargeter.from_v3_directory(
        REAL_V3_PROFILE_DIRECTORY,
        cache_path=cache,
        force_rebuild=True,
    )
    loaded = CalibratedRetargeter.from_v3_directory(
        REAL_V3_PROFILE_DIRECTORY,
        cache_path=cache,
    )
    assert built.calibration.skin_matrix.shape == (52, 383)
    assert built.calibration.tongue_matrix.shape == (16, 383)
    assert loaded.calibration.calibration_hash == built.calibration.calibration_hash
    tongue = np.zeros((1, 16), dtype=np.float32)
    tongue[0, 0] = 1.75
    tongue[0, 4] = 2.5
    tongue[0, 9] = 0.2
    controls = loaded.retarget_post_solver_sequence(
        np.zeros((1, 52), dtype=np.float32),
        assets.skin.pose_names,
        tongue_weights=tongue,
        tongue_pose_names=assets.tongue.pose_names,
    )
    assert controls.shape == (1, 383)
    assert np.isfinite(controls).all()


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
