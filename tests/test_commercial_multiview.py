from __future__ import annotations

from hashlib import sha256
import math

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
    associate_frame_graph,
    positions_to_body_track,
    solve_sequence_positions,
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


def _ring(count: int) -> tuple[CalibratedCamera, ...]:
    return tuple(
        _look_at_camera(
            f"cam{index}",
            (4.0 * np.cos(2.0 * np.pi * index / count), 4.0 * np.sin(2.0 * np.pi * index / count), 2.0),
        )
        for index in range(count)
    )


def _people(cameras, roots, *, drop: tuple[int, int] | None = None):
    detections: list[list[np.ndarray]] = []
    for camera_index, camera in enumerate(cameras):
        people: list[np.ndarray] = []
        for subject, root in enumerate(roots):
            joints = np.full((len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
            for name_index in range(len(JOINT_NAMES)):
                point = root + np.asarray(
                    (0.01 * (name_index % 3), 0.005 * (name_index % 5), 0.02 * (name_index % 4))
                )
                joints[name_index, :2] = camera.project(point)[0]
                joints[name_index, 2] = 0.95
            if drop == (camera_index, subject):
                continue
            people.append(joints)
        detections.append(people)
    return detections


def test_graph_association_matches_exhaustive_on_the_ghost_fixture() -> None:
    cameras = (
        _look_at_camera("front", (0.0, -5.0, 2.0)),
        _look_at_camera("left", (-4.0, -1.0, 2.0)),
        _look_at_camera("right", (4.0, -1.0, 2.0)),
        _look_at_camera("rear", (0.0, 5.0, 2.0)),
    )
    roots = (np.asarray((-0.8, 0.1, 1.0)), np.asarray((0.8, 0.2, 1.0)))
    detections = _people(cameras, roots)
    # Shuffle detector ordering per camera: list order is not an identity signal.
    detections = [list(reversed(people)) if index % 2 else people for index, people in enumerate(detections)]

    exhaustive, exhaustive_cost = associate_frame(cameras, detections, subject_count=2)
    graph, graph_cost = associate_frame_graph(cameras, detections, subject_count=2)
    assert np.allclose(graph, exhaustive, equal_nan=True)
    assert graph_cost == pytest.approx(exhaustive_cost, rel=1e-12)


def test_graph_association_survives_a_camera_losing_a_subject() -> None:
    cameras = _ring(4)
    roots = (np.asarray((-0.9, 0.0, 1.0)), np.asarray((0.9, 0.1, 1.0)))
    # The rear camera never sees subject 1, so non-assignment must stay available
    # or the matcher is forced to invent a counterpart.
    detections = _people(cameras, roots, drop=(2, 1))
    assert len(detections[2]) == 1

    graph, _ = associate_frame_graph(cameras, detections, subject_count=2)
    exhaustive, _ = associate_frame(cameras, detections, subject_count=2)
    assert np.allclose(graph, exhaustive, equal_nan=True)
    # The occluded view contributes nothing for that subject and everything for
    # the other, rather than being filled with a wrong person.
    assert not np.isfinite(graph[1, 2]).any()
    assert np.isfinite(graph[0, 2]).all()


def test_graph_association_resolves_three_subjects_the_exhaustive_search_cannot_afford() -> None:
    """With k subjects and c cameras the exhaustive search is (k!)^c: 1,296
    assignments at three subjects and four cameras, each triangulating every
    core joint. The graph path is polynomial and reproduces the same answer."""

    cameras = _ring(4)
    roots = [np.asarray((-1.2 + 0.9 * subject, 0.1 * subject, 1.0)) for subject in range(3)]
    detections = _people(cameras, roots)

    associated, _ = associate_frame_graph(cameras, detections, subject_count=3)
    recovered = []
    for subject in range(3):
        result = triangulate_point(
            cameras,
            associated[subject, :, JOINT_INDEX["root"], :2],
            associated[subject, :, JOINT_INDEX["root"], 2],
            inlier_threshold_px=5.0,
        )
        assert result is not None
        recovered.append(result.position_world_m)
    # Identity slots are ordered by capture world X with no history, which is the
    # determinism contract everything downstream depends on. `_people` offsets
    # every joint, so the root joint sits at root + its own offset.
    root_index = JOINT_INDEX["root"]
    offset = np.asarray(
        (0.01 * (root_index % 3), 0.005 * (root_index % 5), 0.02 * (root_index % 4))
    )
    expected = sorted(roots, key=lambda root: root[0])
    for got, want in zip(recovered, expected, strict=True):
        assert np.linalg.norm(got - (want + offset)) < 1e-6


def test_graph_association_refuses_evidence_from_an_undecided_camera_pair() -> None:
    """Where a camera pair's baseline is near-collinear with the people it sees,
    its epipolar cost matrix cannot separate the pairings. The Hungarian pick is
    then arbitrary, and single linkage would freeze it into a component holding
    detections from two different performers."""

    cameras = _ring(4)
    # Both performers on the axis joining the opposed cameras: the degenerate case.
    roots = (np.asarray((0.22, 0.0, 1.0)), np.asarray((-0.22, 0.0, 1.0)))
    detections = _people(cameras, roots)

    exhaustive, exhaustive_cost = associate_frame(cameras, detections, subject_count=2)
    graph, graph_cost = associate_frame_graph(cameras, detections, subject_count=2)
    assert np.allclose(graph, exhaustive, equal_nan=True)
    assert graph_cost == pytest.approx(exhaustive_cost, abs=1e-9)

    # Without the margin test the arbitrary pick is accepted and the grouping
    # becomes chimeric, which is the defect this guards.
    permissive, permissive_cost = associate_frame_graph(
        cameras, detections, subject_count=2, ambiguity_ratio=1.0, ambiguity_margin_px=0.0
    )
    assert permissive_cost > exhaustive_cost


def test_scoring_survives_a_subject_with_no_history() -> None:
    """`reconstruct_multiview` seeds the previous roots all-NaN and fills a row
    only once a subject is accepted, so a subject missed on the first frame
    leaves a permanent NaN row. That must not NaN out every candidate's cost and
    abort the whole take."""

    cameras = _ring(4)
    roots = (np.asarray((-0.7, 0.0, 1.0)), np.asarray((0.7, 0.1, 1.0)))
    detections = _people(cameras, roots)
    previous = np.asarray([[np.nan, np.nan, np.nan], [0.68, 0.09, 1.0]])

    for associator in (associate_frame, associate_frame_graph):
        associated, cost = associator(
            cameras, detections, subject_count=2, previous_roots_world_m=previous
        )
        assert math.isfinite(cost), associator.__name__
        assert np.isfinite(associated[:, :, JOINT_INDEX["root"], :2]).any()


def test_graph_association_does_not_secretly_delegate(monkeypatch) -> None:
    """The whole deliverable is that the graph path avoids the exhaustive
    search. Every other test compares the two, so all of them would still pass
    if this function quietly degenerated into `return associate_frame(...)` --
    which is its natural failure mode, since it has several fall-through sites.
    Make the exhaustive path explode, and require an answer anyway."""

    cameras = _ring(4)
    roots = [np.asarray((-1.2 + 0.9 * subject, 0.1 * subject, 1.0)) for subject in range(3)]
    detections = _people(cameras, roots)
    assert all(len(people) == 3 for people in detections), "no surplus, so no deferral is expected"

    import autoanim_gnm.commercial_multiview as module

    def explode(*args, **kwargs):
        raise AssertionError("graph association fell through to the exhaustive search")

    monkeypatch.setattr(module, "associate_frame", explode)
    associated, cost = module.associate_frame_graph(cameras, detections, subject_count=3)
    assert math.isfinite(cost)
    assert np.isfinite(associated[:, :, JOINT_INDEX["root"], :2]).all()


def test_exhaustive_fallback_is_refused_rather_than_left_to_stall() -> None:
    """The fallback is a (subjects!)^cameras search. At four subjects on four
    cameras that is 331,776 candidates, hours for a single frame. It must fail
    with the size named, not stall a capture."""

    cameras = _ring(4)
    roots = [np.asarray((-1.5 + 0.9 * subject, 0.1 * subject, 1.0)) for subject in range(4)]
    detections = _people(cameras, roots)
    # One view reports a surplus person, which is what routes the frame to the
    # fallback in the first place.
    detections[0] = detections[0] + [detections[0][0].copy()]

    with pytest.raises(CommercialMultiviewError) as caught:
        associate_frame_graph(cameras, detections, subject_count=4)
    assert "candidates" in str(caught.value)


def _sequence_fixture(frames: int = 30, single_view_span: tuple[int, int] = (10, 20)):
    """A moving subject, with the left wrist visible to only one camera for a
    span in the middle -- the case per-frame triangulation cannot resolve."""

    cameras = _ring(4)
    wrist = JOINT_INDEX["left_wrist"]
    truth = np.zeros((frames, len(JOINT_NAMES), 3), dtype=np.float64)
    for frame in range(frames):
        sway = 0.30 * np.sin(frame * 0.20)
        for joint in range(len(JOINT_NAMES)):
            truth[frame, joint] = (
                0.10 * (joint % 3) + sway,
                0.05 * (joint % 5),
                0.90 + 0.04 * (joint % 4),
            )
        # Keep the forearm a genuinely constant length so the limb prior is real.
        truth[frame, wrist] = truth[frame, JOINT_INDEX["left_elbow"]] + (0.26, 0.0, 0.0)

    observations = np.full((frames, len(cameras), len(JOINT_NAMES), 3), np.nan)
    for frame in range(frames):
        for camera_index, camera in enumerate(cameras):
            for joint in range(len(JOINT_NAMES)):
                if joint == wrist and single_view_span[0] <= frame < single_view_span[1] and camera_index != 0:
                    continue
                observations[frame, camera_index, joint, :2] = camera.project(truth[frame, joint])[0]
                observations[frame, camera_index, joint, 2] = 0.95

    world = np.full((frames, len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
    for frame in range(frames):
        for joint in range(len(JOINT_NAMES)):
            result = triangulate_point(
                cameras, observations[frame, :, joint, :2], observations[frame, :, joint, 2]
            )
            if result is not None:
                world[frame, joint] = result.position_world_m
    return cameras, truth, world, observations


def test_sequence_solve_recovers_a_joint_only_one_camera_can_see() -> None:
    cameras, truth, world, observations = _sequence_fixture()
    wrist = JOINT_INDEX["left_wrist"]
    hidden = slice(10, 20)
    assert not np.isfinite(world[hidden, wrist]).any(), "fixture must leave the wrist unresolved"

    solved, recovered = solve_sequence_positions(cameras, world, observations)
    assert recovered[hidden, wrist].all()
    error = np.linalg.norm(solved[hidden, wrist] - truth[hidden, wrist], axis=1) * 1000.0
    assert error.max() < 40.0, f"recovered wrist is {error.max():.1f} mm from truth"


def test_sequence_solve_leaves_directly_triangulated_slots_alone() -> None:
    """It is a recovery pass for missing evidence, not a re-estimator of present
    evidence. Measured, letting it rewrite good slots moved them up to 700 mm."""

    cameras, _truth, world, observations = _sequence_fixture()
    direct = np.isfinite(world).all(axis=2)
    solved, recovered = solve_sequence_positions(cameras, world, observations)
    assert np.array_equal(solved[direct], world[direct])
    assert not (recovered & direct).any()


def test_sequence_solve_leaves_wholly_unobserved_slots_for_interpolation() -> None:
    """No ray means no evidence. Inventing a position here would be
    interpolation wearing a solver's clothes, so the slot stays NaN and the
    caller's existing fill owns it."""

    cameras, _truth, world, observations = _sequence_fixture()
    wrist = JOINT_INDEX["left_wrist"]
    observations[12:16, :, wrist, :] = np.nan
    solved, recovered = solve_sequence_positions(cameras, world, observations)
    assert not recovered[12:16, wrist].any()
    assert not np.isfinite(solved[12:16, wrist]).any()


def test_sequence_solve_declines_a_genuine_depth_branch_flip() -> None:
    """A ray meets the limb-length sphere in two points. Where the true joint is
    on the *far* branch and the interpolated start sits near the other, the
    temporal term does not rescue it -- measured, not assumed. What matters is
    that it fails safe: the residual gate refuses the slot rather than emitting
    a confident wrong position, and the caller's interpolation takes over."""

    cameras = _ring(4)
    wrist, elbow = JOINT_INDEX["left_wrist"], JOINT_INDEX["left_elbow"]
    frames, hidden = 40, slice(15, 25)
    truth = np.zeros((frames, len(JOINT_NAMES), 3), dtype=np.float64)
    for frame in range(frames):
        for joint in range(len(JOINT_NAMES)):
            truth[frame, joint] = (0.10 * (joint % 3), 0.05 * (joint % 5), 0.90 + 0.04 * (joint % 4))
        # Outside the gap the forearm points along +Y, lateral to camera 0.
        # Inside it swings onto camera 0's depth axis, which is where the
        # two-fold ambiguity actually bites.
        offset = (-0.26, 0.0, 0.0) if hidden.start <= frame < hidden.stop else (0.0, 0.26, 0.0)
        truth[frame, wrist] = truth[frame, elbow] + np.asarray(offset)

    observations = np.full((frames, len(cameras), len(JOINT_NAMES), 3), np.nan)
    for frame in range(frames):
        for camera_index, camera in enumerate(cameras):
            for joint in range(len(JOINT_NAMES)):
                if joint == wrist and hidden.start <= frame < hidden.stop and camera_index != 0:
                    continue
                observations[frame, camera_index, joint, :2] = camera.project(truth[frame, joint])[0]
                observations[frame, camera_index, joint, 2] = 0.95
    world = np.full((frames, len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
    for frame in range(frames):
        for joint in range(len(JOINT_NAMES)):
            result = triangulate_point(
                cameras, observations[frame, :, joint, :2], observations[frame, :, joint, 2]
            )
            if result is not None:
                world[frame, joint] = result.position_world_m

    solved, recovered = solve_sequence_positions(cameras, world, observations)
    assert not recovered[hidden, wrist].any(), "a branch flip must not be reported as recovered"
    assert not np.isfinite(solved[hidden, wrist]).any(), "the slot must be left for interpolation"
