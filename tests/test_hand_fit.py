"""Synthetic recovery for the constrained hand chain.

The fixture cannot answer whether the solve is correct -- it has no ground truth.
These tests do: a known pose is projected into the rig, the fit is asked to
recover it, and the answer is checked in position space, which is free of the
Euler-gauge freedom that makes angle-space comparisons misleading.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoanim_gnm.commercial_multiview import CalibratedCamera
from autoanim_gnm.hand_fit import (
    build_hand_chain,
    fit_hand_sequence,
    forward_kinematics,
    load_mhr_skeleton,
)

FRAMES = 10
IDENTITY_QUATERNION = np.asarray([0.0, 0.0, 0.0, 1.0])


def _rig() -> tuple[CalibratedCamera, ...]:
    """Four cameras around a 4.7 m stage, roughly the reference geometry."""

    intrinsics = np.asarray([[2550.0, 0.0, 1920.0], [0.0, 2550.0, 1080.0], [0.0, 0.0, 1.0]])
    cameras = []
    for index, angle in enumerate((0.0, 90.0, 180.0, 270.0)):
        radians = np.radians(angle)
        centre = np.asarray([3.0 * np.sin(radians), -3.0 * np.cos(radians), 1.6])
        forward = np.asarray([0.0, 0.0, 1.25]) - centre
        forward /= np.linalg.norm(forward)
        # Image X right, image Y down, Z along the view direction.
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        matrix = np.column_stack((right, np.cross(forward, right), forward))
        from scipy.spatial.transform import Rotation

        cameras.append(
            CalibratedCamera(
                name=f"C{index}",
                width=3840,
                height=2160,
                intrinsics=intrinsics,
                camera_center_world_m=centre,
                camera_to_world_xyzw=Rotation.from_matrix(matrix).as_quat(),
            )
        )
    return tuple(cameras)


def _truth(chain, wrists):
    """A smooth articulation over the take, inside the anatomical limits."""

    generator = np.random.default_rng(20260829)
    dof = chain.degrees_of_freedom
    base = generator.uniform(-0.2, 0.2, size=dof)
    swing = generator.uniform(-0.15, 0.15, size=dof)
    phase = np.linspace(0.0, 1.5 * np.pi, FRAMES)[:, None]
    angles = base[None, :] + swing[None, :] * np.sin(phase)
    positions = np.stack(
        [
            forward_kinematics(chain, wrists[frame], IDENTITY_QUATERNION, angles[frame])
            for frame in range(FRAMES)
        ]
    )
    return angles, positions


def _observe(cameras, positions):
    values = np.full((FRAMES, len(cameras), positions.shape[1], 3), np.nan)
    for frame in range(FRAMES):
        for index, camera in enumerate(cameras):
            for joint in range(positions.shape[1]):
                uv, depth = camera.project(positions[frame, joint])
                if depth > 0:
                    values[frame, index, joint] = (uv[0], uv[1], 0.9)
    return values


@pytest.fixture(scope="module")
def solved():
    cameras = _rig()
    chain = build_hand_chain("l", load_mhr_skeleton())
    wrists = np.stack(
        [np.asarray([0.15, 0.0, 1.25]) + np.asarray([0.0, 0.02, 0.01]) * frame for frame in range(FRAMES)]
    )
    truth_angles, truth_positions = _truth(chain, wrists)
    observations = _observe(cameras, truth_positions)
    angles, positions, used = fit_hand_sequence(
        cameras, chain, wrists, observations, maximum_evaluations=60
    )
    return truth_positions, positions, used, observations, wrists


def test_recovers_joint_positions(solved):
    truth, positions, _, _, _ = solved
    error = np.linalg.norm(positions - truth, axis=2) * 1000.0
    # Position space, not angle space: a redundant Euler triple reaches the same
    # pose several ways, so angle error overstates pose error badly.
    assert np.median(error) < 5.0, f"median {np.median(error):.2f} mm"
    assert np.percentile(error, 95) < 25.0, f"p95 {np.percentile(error, 95):.2f} mm"


def test_uses_every_visible_observation(solved):
    _, _, used, observations, _ = solved
    visible = np.isfinite(observations[..., :2]).all(axis=3) & (observations[..., 2] > 0.0)
    assert used.sum() == visible.sum()


def test_temporal_prior_holds_the_wrist_block(solved):
    """The failure this prior exists for.

    ``smooth_weight`` covers the joint angles only, so the three wrist
    orientation parameters were unregularised and thrashed harder than the
    fingers did -- 50-120 mm of frame-to-frame acceleration on the reference
    fixture against MAMMA's 1.6-3.2 mm. The wrist-local frame turns with the
    wrist and cannot see it, so this checks wrist-relative positions.
    """

    _, positions, _, _, wrists = solved
    relative = positions - wrists[:, None, :]
    jitter = np.median(
        np.linalg.norm(relative[2:] - 2.0 * relative[1:-1] + relative[:-2], axis=2)
    ) * 1000.0
    assert jitter < 2.0, f"wrist-relative jitter {jitter:.2f} mm"
