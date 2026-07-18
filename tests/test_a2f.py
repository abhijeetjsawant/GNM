from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.a2f import (
    ARKitGNMRetargeter,
    A2FCoefficientLayout,
    A2FRunnerError,
    A2FValidationError,
    CLAIRE_LAYOUT,
    ClaireSkinAssets,
    ClaireSkinSolver,
    ClaireTongueAssets,
    ClaireTongueSolver,
    parse_a2f_jsonl,
    recover_a2f_auxiliary_track,
    resolve_a2f_runner,
    run_a2f_runner,
)


def _frame(time_seconds: float, coefficients: list[float], layout=CLAIRE_LAYOUT) -> str:
    return json.dumps(
        {
            "timeSeconds": time_seconds,
            "coefficients": coefficients,
            "layout": {
                "skinCount": layout.skin_count,
                "tongueCount": layout.tongue_count,
                "jawCount": layout.jaw_count,
                "eyeCount": layout.eye_count,
            },
        }
    )


def test_parser_matches_swift_codable_frame_and_partitions() -> None:
    first = np.linspace(-0.2, 0.2, CLAIRE_LAYOUT.coefficient_count).tolist()
    second = np.linspace(0.3, -0.3, CLAIRE_LAYOUT.coefficient_count).tolist()

    frames = parse_a2f_jsonl(_frame(0.0, first) + "\n" + _frame(1 / 30, second) + "\n")

    assert len(frames) == 2
    assert frames[0].time_seconds == 0.0
    assert frames[0].coefficients.shape == (169,)
    assert frames[0].skin.shape == (140,)
    assert frames[0].tongue.shape == (10,)
    assert frames[0].jaw.shape == (15,)
    assert frames[0].eyes.shape == (4,)
    assert not frames[0].coefficients.flags.writeable


def test_auxiliary_recovery_fits_jaw_transform_and_preserves_eye_order() -> None:
    neutral = np.asarray(
        [
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.2),
            (-0.6, -0.8, 0.5),
            (0.7, -0.7, -0.4),
        ],
        dtype=np.float64,
    )
    angle = np.deg2rad(12.0)
    rotation = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, np.cos(angle), -np.sin(angle)), (0.0, np.sin(angle), np.cos(angle)))
    )
    translation = np.asarray((0.3, -0.2, 1.1))
    observed = neutral @ rotation.T + translation
    coefficients = np.zeros(CLAIRE_LAYOUT.coefficient_count, dtype=np.float64)
    coefficients[CLAIRE_LAYOUT.jaw_slice] = (observed - neutral).reshape(-1)
    coefficients[CLAIRE_LAYOUT.eye_slice] = (1.0, 2.0, 3.0, 4.0)
    frames = parse_a2f_jsonl(_frame(0.0, coefficients.tolist()))

    track = recover_a2f_auxiliary_track(frames, neutral)

    np.testing.assert_allclose(track.jaw_rotation_matrices[0], rotation, atol=1e-6)
    np.testing.assert_allclose(track.jaw_translations[0], translation, atol=1e-6)
    np.testing.assert_allclose(
        track.jaw_rotation_vectors_degrees[0],
        (12.0, 0.0, 0.0),
        atol=1e-5,
    )
    np.testing.assert_allclose(track.eye_rotations_degrees[0], ((1.0, 2.0), (3.0, 4.0)))
    assert track.jaw_rms_residual[0] < 1e-6
    assert np.linalg.det(track.jaw_rotation_matrices[0]) == pytest.approx(1.0, abs=1e-6)
    assert not track.jaw_points.flags.writeable


@pytest.mark.parametrize(
    "payload,match",
    [
        ("", "no frames"),
        (_frame(0.0, [0.0] * 168), "expected 169 coefficients"),
        (_frame(0.0, [0.0] * 169) + "\n" + _frame(0.0, [0.0] * 169), "strictly increasing"),
        (_frame(0.0, [0.0] * 168 + [float("nan")]), "must be finite"),
        (_frame(0.0, [0.0] * 169, A2FCoefficientLayout(272, 10, 15, 4)), "expected layout"),
    ],
)
def test_parser_rejects_malformed_or_wrong_identity_stream(payload: str, match: str) -> None:
    with pytest.raises(A2FValidationError, match=match):
        parse_a2f_jsonl(payload)


def test_runner_resolver_and_invocation_contract(tmp_path: Path) -> None:
    runner = tmp_path / "fake-a2f-runner"
    runner.write_text(
        """#!/usr/bin/env python3
import json, pathlib, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
assert args["--emotion"] == "joy"
assert args["--emotion-strength"] == "0.4"
layout = {"skinCount": 140, "tongueCount": 10, "jawCount": 15, "eyeCount": 4}
frame = {"timeSeconds": 0.0, "coefficients": [0.0] * 169, "layout": layout}
pathlib.Path(args["--output"]).write_text(json.dumps(frame) + "\\n")
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert resolve_a2f_runner(runner) == runner.resolve()
    frames = run_a2f_runner(
        audio,
        runner=runner,
        model_dir=model_dir,
        emotion="JOY",
        emotion_strength=0.4,
    )
    assert len(frames) == 1
    assert frames[0].layout == CLAIRE_LAYOUT


@pytest.mark.parametrize(
    "emotion,strength,match",
    [("elated", 1.0, "Unsupported"), ("neutral", -0.1, "in \\[0,1\\]"), ("joy", np.nan, "in \\[0,1\\]")],
)
def test_runner_validates_emotion_boundary(
    tmp_path: Path, emotion: str, strength: float, match: str
) -> None:
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(A2FRunnerError, match=match):
        run_a2f_runner(audio, runner=runner, emotion=emotion, emotion_strength=strength)


def test_runner_resolver_rejects_non_executable(tmp_path: Path) -> None:
    target = tmp_path / "runner"
    target.write_text("not executable", encoding="utf-8")
    with pytest.raises(A2FRunnerError, match="unavailable"):
        resolve_a2f_runner(target)


def _write_tiny_assets(root: Path, *, regularization: float = 0.0) -> tuple[np.ndarray, tuple[str, ...]]:
    vertex_count = 6
    pose_names = ("jawOpen", "mouthPucker", "mouthSmileLeft")
    neutral = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 2, 0], [1, 0, 3]],
        dtype=np.float32,
    )
    deltas = np.zeros((3, vertex_count, 3), dtype=np.float32)
    deltas[0, 0, 0] = 1.0
    deltas[1, 1, 1] = 1.0
    deltas[2, 2, 2] = 1.0
    np.savez(
        root / "model_data.npz",
        shapes_matrix_skin=deltas,
        shapes_mean_skin=neutral,
    )
    np.savez(
        root / "bs_skin.npz",
        neutral=neutral,
        poseNames=np.asarray(("neutral",) + pose_names, dtype="S32"),
        frontalMask=np.arange(vertex_count, dtype=np.int32),
        **dict(zip(pose_names, deltas, strict=True)),
    )
    params = {
        "numPoses": 3,
        "bsSolveActivePoses": [1, 1, 1],
        "bsSolveCancelPoses": [-1, -1, -1],
        "bsSolveSymmetryPoses": [-1, -1, -1],
        "bsWeightMultipliers": [1.0, 1.0, 1.0],
        "bsWeightOffsets": [0.0, 0.0, 0.0],
        "strengthL1regularization": regularization,
        "strengthL2regularization": regularization,
        "strengthTemporalSmoothing": regularization,
        "strengthSymmetry": regularization,
        "templateBBSize": float(np.linalg.norm(np.ptp(neutral, axis=0))),
    }
    (root / "bs_skin_config.json").write_text(
        json.dumps({"blendshape_params": params}), encoding="utf-8"
    )
    return deltas, pose_names


def _write_tiny_tongue_assets(root: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    vertex_count = 5
    pose_names = ("tongueTipUp", "tongueUp", "tongueStretch")
    neutral = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1], [1, 2, 1]], dtype=np.float32
    )
    deltas = np.zeros((3, vertex_count, 3), dtype=np.float32)
    deltas[0, 0, 0] = 1.0
    deltas[1, 1, 1] = 1.0
    deltas[2, 2, 2] = 1.0
    np.savez(
        root / "model_data.npz",
        shapes_matrix_tongue=deltas,
        shapes_mean_tongue=neutral,
    )
    np.savez(
        root / "bs_tongue.npz",
        neutral=neutral,
        poseNames=np.asarray(("neutral",) + pose_names, dtype="S32"),
        **dict(zip(pose_names, deltas, strict=True)),
    )
    params = {
        "numPoses": 3,
        "bsSolveActivePoses": [1, 1, 1],
        "bsSolveCancelPoses": [-1, -1, -1],
        "bsSolveSymmetryPoses": [-1, -1, -1],
        "bsWeightMultipliers": [1.0, 1.0, 1.0],
        "bsWeightOffsets": [0.0, 0.0, 0.0],
        "strengthL1regularization": 0.0,
        "strengthL2regularization": 0.0,
        "strengthTemporalSmoothing": 0.0,
        "strengthSymmetry": 0.0,
        "templateBBSize": float(np.linalg.norm(np.ptp(neutral, axis=0))),
    }
    (root / "bs_tongue_config.json").write_text(
        json.dumps({"blendshape_params": params}), encoding="utf-8"
    )
    return deltas, pose_names


def test_reduced_solver_recovers_synthetic_arkit_weights(tmp_path: Path) -> None:
    _, names = _write_tiny_assets(tmp_path)
    assets = ClaireSkinAssets.load(tmp_path)
    solver = ClaireSkinSolver.from_assets(assets)
    expected = np.asarray([[0.15, 0.55, 0.85], [0.8, 0.25, 0.1]], dtype=np.float32)

    actual = solver.solve_coefficients(expected)

    assert solver.pose_names == names
    assert solver.projected_pca.shape == (3, 3)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert np.isfinite(actual).all()
    assert np.all((actual >= 0.0) & (actual <= 1.0))


def test_solver_rejects_nonfinite_and_wrong_width(tmp_path: Path) -> None:
    _write_tiny_assets(tmp_path)
    solver = ClaireSkinSolver.from_directory(tmp_path)
    with pytest.raises(A2FValidationError, match="Expected"):
        solver.solve_coefficients(np.zeros((2, 4)))
    with pytest.raises(A2FValidationError, match="finite"):
        solver.solve_coefficients(np.asarray([[0.0, np.nan, 0.0]]))


def test_reduced_tongue_solver_recovers_synthetic_weights(tmp_path: Path) -> None:
    _, names = _write_tiny_tongue_assets(tmp_path)
    assets = ClaireTongueAssets.load(tmp_path)
    solver = ClaireTongueSolver.from_assets(assets)
    expected = np.asarray([[0.2, 0.6, 0.9], [0.75, 0.1, 0.35]], dtype=np.float32)

    actual = solver.solve_coefficients(expected)

    assert solver.pose_names == names
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert np.isfinite(actual).all()
    assert np.all((actual >= 0.0) & (actual <= 1.0))


def test_arkit_retarget_is_region_scoped_finite_and_bounded(rig) -> None:
    retargeter = ARKitGNMRetargeter(rig)
    weights = {
        "eyeBlinkLeft": 1.0,
        "jawOpen": 0.8,
        "mouthPucker": 0.5,
        "tongueOut": 1.0,
    }

    controls = retargeter.retarget(weights)

    assert controls.shape == (383,)
    assert np.isfinite(controls).all()
    assert float(np.max(np.abs(controls))) <= 3.0
    assert np.any(np.abs(controls[:100]) > 0)
    assert np.allclose(controls[100:200], 0)
    assert np.any(np.abs(controls[200:350]) > 0)
    assert np.any(np.abs(controls[350:382]) > 0)
    assert controls[382] == 0


def test_retarget_sequence_accepts_released_tongue_controls(rig) -> None:
    retargeter = ARKitGNMRetargeter(rig)
    skin = np.zeros((2, 1), dtype=np.float32)
    tongue = np.asarray([[0.0, 0.0], [0.7, 0.9]], dtype=np.float32)

    controls = retargeter.retarget_sequence(
        skin,
        ("jawOpen",),
        tongue_weights=tongue,
        tongue_pose_names=("tongueTipUp", "tongueStretch"),
    )

    assert controls.shape == (2, 383)
    assert np.allclose(controls[0, 350:382], 0)
    assert np.any(np.abs(controls[1, 350:382]) > 0)


@pytest.mark.skipif(
    "AUTOANIM_A2F_ASSET_DIR" not in os.environ,
    reason="set AUTOANIM_A2F_ASSET_DIR to run against released Claire assets",
)
def test_optional_released_claire_assets_solve_one_frame() -> None:
    root = os.environ["AUTOANIM_A2F_ASSET_DIR"]
    skin_solver = ClaireSkinSolver.from_directory(root)
    skin = skin_solver.solve_coefficients(
        np.zeros((1, skin_solver.pca_count), dtype=np.float32)
    )
    tongue_solver = ClaireTongueSolver.from_directory(root)
    tongue = tongue_solver.solve_coefficients(
        np.zeros((1, tongue_solver.pca_count), dtype=np.float32)
    )
    assert skin_solver.pca_count == 140
    assert len(skin_solver.pose_names) == 52
    assert skin.shape == (1, 52)
    assert tongue_solver.pca_count == 10
    assert len(tongue_solver.pose_names) == 16
    assert tongue.shape == (1, 16)
    assert np.isfinite(skin).all() and np.isfinite(tongue).all()
    assert np.all((skin >= 0.0) & (skin <= 1.0))
    assert np.all((tongue >= 0.0) & (tongue <= 1.0))
