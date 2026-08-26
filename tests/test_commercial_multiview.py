from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions
from autoanim_gnm.commercial_multiview import (
    CalibratedCamera,
    CommercialMultiviewError,
    JOINT_INDEX,
    JOINT_NAMES,
    associate_frame,
    positions_to_body_track,
    triangulate_point,
)


def _look_at_camera(name: str, center: tuple[float, float, float]) -> CalibratedCamera:
    center_array = np.asarray(center, dtype=np.float64)
    target = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    forward = target - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_world = np.column_stack((right, down, forward))
    return CalibratedCamera(
        name=name,
        width=1280,
        height=720,
        intrinsics=np.asarray(((900.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 1.0))),
        camera_center_world_m=center_array,
        camera_to_world_xyzw=Rotation.from_matrix(camera_to_world).as_quat(),
    )


def test_weighted_robust_triangulation_rejects_one_bad_camera() -> None:
    cameras = (
        _look_at_camera("front", (0.0, -5.0, 2.0)),
        _look_at_camera("left", (-4.0, -1.0, 2.0)),
        _look_at_camera("right", (4.0, -1.0, 2.0)),
        _look_at_camera("rear", (0.0, 5.0, 2.0)),
    )
    expected = np.asarray((0.25, 0.3, 1.4))
    points = np.asarray([camera.project(expected)[0] for camera in cameras])
    points[2] += (180.0, -120.0)
    result = triangulate_point(
        cameras,
        points,
        np.asarray((0.9, 0.85, 0.7, 0.95)),
        inlier_threshold_px=5.0,
    )
    assert result is not None
    assert 2 not in result.used_camera_indices
    assert np.linalg.norm(result.position_world_m - expected) < 1e-6


def test_association_prefers_four_camera_support_over_two_view_ghost() -> None:
    cameras = (
        _look_at_camera("front", (0.0, -5.0, 2.0)),
        _look_at_camera("left", (-4.0, -1.0, 2.0)),
        _look_at_camera("right", (4.0, -1.0, 2.0)),
        _look_at_camera("rear", (0.0, 5.0, 2.0)),
    )
    roots = (np.asarray((-0.8, 0.1, 1.0)), np.asarray((0.8, 0.2, 1.0)))
    orders = ((1, 0), (0, 1), (1, 0), (0, 1))
    detections: list[list[np.ndarray]] = []
    for camera, order in zip(cameras, orders, strict=True):
        people: list[np.ndarray] = []
        for subject in order:
            joints = np.full((len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
            for name_index, name in enumerate(JOINT_NAMES):
                point = roots[subject] + np.asarray(
                    (0.01 * (name_index % 3), 0.005 * (name_index % 5), 0.02 * (name_index % 4))
                )
                joints[name_index, :2] = camera.project(point)[0]
                joints[name_index, 2] = 0.95
            people.append(joints)
        detections.append(people)
    associated, _ = associate_frame(cameras, detections, subject_count=2)
    reconstructed = []
    for subject in range(2):
        result = triangulate_point(
            cameras,
            associated[subject, :, JOINT_INDEX["root"], :2],
            associated[subject, :, JOINT_INDEX["root"], 2],
            inlier_threshold_px=2.0,
        )
        assert result is not None
        assert len(result.used_camera_indices) == 4
        reconstructed.append(result.position_world_m)
    root_offset = np.asarray((0.02, 0.015, 0.0))
    assert np.allclose(reconstructed, np.asarray(roots) + root_offset, atol=1e-6)


def _synthetic_positions(frame_count: int = 12) -> np.ndarray:
    output = np.zeros((frame_count, len(JOINT_NAMES), 3), dtype=np.float64)
    for frame in range(frame_count):
        phase = frame / (frame_count - 1)
        root = np.asarray((0.0, 0.0, 1.0 + 0.03 * np.sin(phase * np.pi)))
        values = {
            "root": root,
            "neck": root + (0.0, 0.0, 0.62),
            "nose": root + (0.0, -0.08, 0.78),
            "left_eye": root + (-0.035, -0.07, 0.76),
            "right_eye": root + (0.035, -0.07, 0.76),
            "left_ear": root + (-0.09, 0.0, 0.73),
            "right_ear": root + (0.09, 0.0, 0.73),
            "left_shoulder": root + (-0.22, 0.0, 0.52),
            "right_shoulder": root + (0.22, 0.0, 0.52),
            "left_elbow": root + (-0.42, -0.04, 0.34),
            "right_elbow": root + (0.42, 0.04, 0.34),
            "left_wrist": root + (-0.54, -0.04, 0.16 + 0.2 * phase),
            "right_wrist": root + (0.54, 0.04, 0.16),
            "left_hip": root + (-0.09, 0.0, 0.0),
            "right_hip": root + (0.09, 0.0, 0.0),
            "left_knee": root + (-0.09, 0.0, -0.43),
            "right_knee": root + (0.09, 0.0, -0.43),
            "left_ankle": root + (-0.09, 0.0, -0.86),
            "right_ankle": root + (0.09, 0.0, -0.86),
        }
        for name, value in values.items():
            output[frame, JOINT_INDEX[name]] = value
    return output


def test_positions_compile_to_finite_detailed_body_track() -> None:
    track = positions_to_body_track(
        _synthetic_positions(),
        sample_rate_hz=30,
        provenance_sha256=sha256(b"synthetic-commercial-multiview").hexdigest(),
    )
    assert track.joint_names == DETAILED_HUMANOID.names
    assert track.local_rotations_xyzw.shape == (12, 55, 4)
    assert np.isfinite(track.local_rotations_xyzw).all()
    assert np.allclose(np.linalg.norm(track.local_rotations_xyzw, axis=2), 1.0, atol=2e-5)
    positions = forward_kinematics_positions(
        track.root_translation_m,
        track.local_rotations_xyzw,
        skeleton=DETAILED_HUMANOID,
    )
    left_upper_arm = DETAILED_HUMANOID.index("LeftUpperArm")
    left_lower_arm = DETAILED_HUMANOID.index("LeftLowerArm")
    left_hand = DETAILED_HUMANOID.index("LeftHand")
    assert np.linalg.norm(positions[-1, left_hand] - positions[0, left_hand]) > 0.05
    assert np.isfinite(positions[:, left_upper_arm]).all()


def _scaled_camera(camera: CalibratedCamera, factor: int) -> CalibratedCamera:
    return camera.scaled(camera.width * factor, camera.height * factor)


def test_inlier_gate_is_invariant_to_detector_width() -> None:
    """A fixed pixel gate is a shrinking *physical* gate as the detector runs
    wider, so the same observation is accepted at 1280 and rejected at 3840.
    `pixel_scale` must keep the accept/reject decision identical across widths.
    """

    cameras = (
        _look_at_camera("front", (0.0, -5.0, 2.0)),
        _look_at_camera("left", (-4.0, -1.0, 2.0)),
        _look_at_camera("right", (4.0, -1.0, 2.0)),
        _look_at_camera("rear", (0.0, 5.0, 2.0)),
    )
    expected = np.asarray((0.25, 0.3, 1.4))
    points = np.asarray([camera.project(expected)[0] for camera in cameras])
    # Chosen so the residual on this camera sits inside the reference gate but
    # outside a third of it, which is where the two behaviours diverge.
    points[2] += (30.0, 0.0)
    confidence = np.asarray((0.9, 0.85, 0.7, 0.95))

    reference = triangulate_point(cameras, points, confidence)
    assert reference is not None
    assert reference.used_camera_indices == (0, 1, 2, 3)

    for factor in (2, 3):
        wide_cameras = tuple(_scaled_camera(camera, factor) for camera in cameras)
        wide = triangulate_point(
            wide_cameras,
            points * factor,
            confidence,
            pixel_scale=float(factor),
        )
        assert wide is not None
        assert wide.used_camera_indices == reference.used_camera_indices
        assert np.linalg.norm(wide.position_world_m - reference.position_world_m) < 1e-6

    # Without the scale the wider detector silently discards an observation the
    # reference width accepts. That censoring is the defect this prevents.
    unscaled = triangulate_point(
        tuple(_scaled_camera(camera, 3) for camera in cameras),
        points * 3,
        confidence,
    )
    assert unscaled is not None
    assert 2 not in unscaled.used_camera_indices


def test_pixel_scale_rejects_nonpositive_values() -> None:
    cameras = (
        _look_at_camera("front", (0.0, -5.0, 2.0)),
        _look_at_camera("rear", (0.0, 5.0, 2.0)),
    )
    expected = np.asarray((0.0, 0.0, 1.2))
    points = np.asarray([camera.project(expected)[0] for camera in cameras])
    confidence = np.asarray((0.9, 0.9))
    for bad in (0.0, -1.0, float("nan")):
        try:
            triangulate_point(cameras, points, confidence, pixel_scale=bad)
        except CommercialMultiviewError:
            continue
        raise AssertionError(f"pixel_scale={bad!r} was accepted")


def test_association_cost_scales_homogeneously_with_detector_width() -> None:
    """The association objective mixes a pixel median with penalties on
    unitless counts and metre distances. Those weights must scale with the
    detector width or the objective is a different function at each width."""

    cameras = (
        _look_at_camera("front", (0.0, -5.0, 2.0)),
        _look_at_camera("left", (-4.0, -1.0, 2.0)),
        _look_at_camera("right", (4.0, -1.0, 2.0)),
        _look_at_camera("rear", (0.0, 5.0, 2.0)),
    )
    roots = (np.asarray((-0.8, 0.1, 1.0)), np.asarray((0.8, 0.2, 1.0)))

    def scene(factor: int) -> list[list[np.ndarray]]:
        wide = tuple(_scaled_camera(camera, factor) for camera in cameras)
        detections: list[list[np.ndarray]] = []
        for camera_index, camera in enumerate(wide):
            people: list[np.ndarray] = []
            for subject in range(2):
                joints = np.full((len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
                for name_index in range(len(JOINT_NAMES)):
                    point = roots[subject] + np.asarray(
                        (0.01 * (name_index % 3), 0.005 * (name_index % 5), 0.02 * (name_index % 4))
                    )
                    joints[name_index, :2] = camera.project(point)[0]
                    joints[name_index, 2] = 0.95
                # The rear camera loses one subject, which is what makes the
                # support-deficit penalty non-zero. A clean four-camera fixture
                # zeroes that term and would not exercise its scaling at all.
                if camera_index == 3 and subject == 1:
                    joints[:] = np.nan
                people.append(joints)
            detections.append(people)
        return detections

    previous_roots = np.asarray([roots[0] + 0.05, roots[1] - 0.04])
    previous_positions = np.full((2, len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
    for subject in range(2):
        previous_positions[subject, :] = roots[subject] + 0.03

    costs: dict[int, float] = {}
    orders: dict[int, np.ndarray] = {}
    for factor in (1, 2, 3):
        wide = tuple(_scaled_camera(camera, factor) for camera in cameras)
        detections = scene(factor)
        previous_observations = np.full(
            (2, len(cameras), len(JOINT_NAMES), 3), np.nan, dtype=np.float64
        )
        for subject in range(2):
            for camera_index in range(len(cameras)):
                observed = detections[camera_index][subject]
                # Pixel coordinates scale with the detector; confidence does not.
                # The offset is deliberately past the 250 px screen-step clamp
                # at the reference width, so the clamp's scaling is exercised
                # too — a smaller offset leaves that term untested.
                previous_observations[subject, camera_index, :, :2] = observed[:, :2] - 400.0 * factor
                previous_observations[subject, camera_index, :, 2] = observed[:, 2]
        associated, cost = associate_frame(
            wide,
            detections,
            subject_count=2,
            previous_roots_world_m=previous_roots,
            previous_positions_world_m=previous_positions,
            previous_observations_xyc=previous_observations,
            pixel_scale=float(factor),
        )
        costs[factor] = cost
        orders[factor] = associated[:, :, :, :2] / factor

    assert costs[1] > 0.0
    for factor in (2, 3):
        # Homogeneous of degree one: the argmin is invariant, so the chosen
        # association must be identical and only the scale of the cost changes.
        assert costs[factor] == pytest.approx(costs[1] * factor, rel=1e-9)
        assert np.allclose(orders[factor], orders[1], atol=1e-9, equal_nan=True)
