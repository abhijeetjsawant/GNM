"""The head-orientation stage, and the one property that matters: it is not a constant.

`docs/HEAD_ORIENTATION_MEASURED.md` records why these tests are shaped this way. Before
this stage existed, `positions_to_body_track` welded the head to the torso, so every
delivered track carried the identity quaternion on `Head`, `Neck` and both eyes. That
constant scored at parity with a research reference on frame-to-frame jitter -- the metric
anyone reaches for first -- while carrying no head information at all. **So a test that
only checks smoothness would pass on the defect it exists to catch**, and the tests below
are written against the degenerate solution rather than against a tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autoanim_gnm.commercial_multiview import (
    CommercialMultiviewError,
    JOINT_INDEX,
    JOINT_NAMES,
    CalibratedCamera,
    _quat_from_matrix,
    positions_to_body_track,
)
from autoanim_gnm.body import DETAILED_HUMANOID
from autoanim_gnm.head_orientation import (
    MAXIMUM_FRAME_TRAVEL_DEG,
    HeadOrientationError,
    _maximum_frame_travel_deg,
    rodrigues,
    solve_head_orientation,
)

LANDMARKS = ("Head", "HeadEnd", "Jaw", "LeftEye", "RightEye")
# Head-local positions in metres, roughly anatomical: crown 120 mm up, jaw 60 mm down and
# forward, eyes 32 mm either side of centre and 70 mm forward.
TEMPLATE = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.120),
        (0.0, 0.060, -0.060),
        (-0.032, 0.070, 0.020),
        (0.032, 0.070, 0.020),
    ]
)
HEAD_INDEX = DETAILED_HUMANOID.index("Head")
NECK_INDEX = DETAILED_HUMANOID.index("Neck")


def _camera(name: str, centre, look_at) -> CalibratedCamera:
    forward = np.asarray(look_at, float) - np.asarray(centre, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=1)  # camera->world
    # The library's own converter, which branches on the largest diagonal term. A
    # trace-only extraction divides by ~0 for a camera looking back down an axis, and
    # camera C here does exactly that.
    quaternion = _quat_from_matrix(rotation)
    return CalibratedCamera(
        name=name,
        width=1280,
        height=720,
        intrinsics=np.asarray(((850.0, 0.0, 640.0), (0.0, 850.0, 360.0), (0.0, 0.0, 1.0))),
        camera_center_world_m=np.asarray(centre, float),
        camera_to_world_xyzw=quaternion,
    )


def _rig(target=(0.0, 0.0, 1.6)):
    return [
        _camera("A", (0.0, -4.5, 1.6), target),
        _camera("B", (4.5, 0.0, 1.6), target),
        _camera("C", (0.0, 4.5, 1.6), target),
        _camera("D", (-4.5, 0.0, 1.6), target),
    ]


def _synthesise(cameras, rotations, position=(0.0, 0.0, 1.6)):
    """Project a rigid head under known rotations. Truth in, so error is measurable."""
    frames = len(rotations)
    out = np.full((frames, len(cameras), len(LANDMARKS), 3), np.nan)
    for frame in range(frames):
        world = (rotations[frame] @ TEMPLATE.T).T + np.asarray(position, float)
        for index, camera in enumerate(cameras):
            uv, depth = camera.project(world)
            uv = np.asarray(uv).reshape(len(LANDMARKS), 2)
            out[frame, index, :, :2] = uv
            out[frame, index, :, 2] = 0.95
    return out


def _yaw_sequence(frames=40, amplitude_deg=30.0):
    angles = np.radians(amplitude_deg) * np.sin(np.linspace(0.0, 2.0 * np.pi, frames))
    return rodrigues(np.stack([np.zeros(frames), np.zeros(frames), angles], axis=1)), angles


def test_recovers_a_known_yaw_rather_than_averaging_it_away():
    """The estimator must track, not smooth. A fit that returned the mean pose would have
    near-zero error on a jitter metric and be exactly the defect this stage replaces."""
    cameras = _rig()
    truth, angles = _yaw_sequence()
    observations = _synthesise(cameras, truth)

    solved = solve_head_orientation(cameras, observations, LANDMARKS)

    # Compare against truth up to the constant the anatomical gauge fixes -- both are
    # absolute, so the residual rotation should be small and, crucially, CONSTANT.
    relative = np.einsum("fij,fkj->fik", solved.rotations_world, truth)
    offset = relative.mean(axis=0)
    u, _, vt = np.linalg.svd(offset)
    offset = u @ vt
    residual = np.einsum("fij,kj->fik", relative, offset)
    trace = np.clip((np.trace(residual, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    error = np.degrees(np.arccos(trace))
    assert float(np.median(error)) < 2.0, f"tracking error {np.median(error):.2f} deg"

    # And the recovered motion must have the injected amplitude, not a damped one.
    recovered = np.degrees(np.arccos(np.clip(
        (np.trace(np.einsum("fij,kj->fik", solved.rotations_world,
                            solved.rotations_world.mean(axis=0)), axis1=1, axis2=2) - 1.0) / 2.0,
        -1.0, 1.0)))
    assert float(np.max(recovered)) > 0.6 * float(np.max(np.abs(np.degrees(angles)))), (
        "the solve damped the motion it was given"
    )


def test_refuses_when_the_head_is_seen_by_too_few_views():
    """Falling back is correct; guessing is not. The caller reports the fallback."""
    cameras = _rig()
    truth, _ = _yaw_sequence()
    observations = _synthesise(cameras, truth)
    observations[:, 1:, :, 2] = 0.0          # only one camera retains confidence
    with pytest.raises(HeadOrientationError):
        solve_head_orientation(cameras, observations, LANDMARKS)


def test_refuses_to_ship_an_arbitrary_zero():
    """Without the landmarks that define anatomy there is no absolute orientation, and a
    rigid fit is correct only up to a constant. Refuse rather than deliver a head that
    points somewhere smooth and wrong."""
    cameras = _rig()
    truth, _ = _yaw_sequence()
    observations = _synthesise(cameras, truth)
    with pytest.raises(HeadOrientationError, match="anatomical zero"):
        solve_head_orientation(
            cameras, observations[:, :, :3], ("Head", "HeadEnd", "Jaw")
        )


def _standing_pose() -> np.ndarray:
    """One frame of a plausible standing skeleton. Every limb needs a real direction --
    a collapsed pose makes the converter raise on a degenerate bone, which is correct."""
    pose = np.zeros((len(JOINT_NAMES), 3))
    layout = {
        "root": (0.0, 0.0, 0.95), "neck": (0.0, 0.0, 1.45), "nose": (0.0, 0.09, 1.58),
        "left_eye": (0.03, 0.10, 1.60), "right_eye": (-0.03, 0.10, 1.60),
        "left_ear": (0.07, 0.0, 1.59), "right_ear": (-0.07, 0.0, 1.59),
        "left_shoulder": (0.18, 0.0, 1.42), "right_shoulder": (-0.18, 0.0, 1.42),
        "left_elbow": (0.21, 0.0, 1.15), "right_elbow": (-0.21, 0.0, 1.15),
        "left_wrist": (0.23, 0.0, 0.90), "right_wrist": (-0.23, 0.0, 0.90),
        "left_hip": (0.10, 0.0, 0.93), "right_hip": (-0.10, 0.0, 0.93),
        "left_knee": (0.11, 0.0, 0.52), "right_knee": (-0.11, 0.0, 0.52),
        "left_ankle": (0.11, 0.0, 0.09), "right_ankle": (-0.11, 0.0, 0.09),
    }
    for name, value in layout.items():
        pose[JOINT_INDEX[name]] = value
    return pose


def test_body_track_head_is_a_constant_without_a_solve_and_moves_with_one():
    """The regression that matters, stated as the contrast it has to preserve."""
    frames = 24
    positions = np.tile(_standing_pose(), (frames, 1, 1))
    provenance = "0" * 64
    welded = positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256=provenance
    )
    head = welded.local_rotations_xyzw[:, HEAD_INDEX]
    assert np.allclose(head, (0.0, 0.0, 0.0, 1.0), atol=1e-9), (
        "without a solve the head must remain exactly the pre-existing constant"
    )

    angles = np.radians(25.0) * np.sin(np.linspace(0.0, 2.0 * np.pi, frames))
    rotations = rodrigues(np.stack([np.zeros(frames), np.zeros(frames), angles], axis=1))
    solved = positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256=provenance,
        head_world_rotations=rotations,
    )
    moved = solved.local_rotations_xyzw[:, HEAD_INDEX]
    travel = np.degrees(2.0 * np.arccos(np.clip(np.abs(moved[:, 3]), 0.0, 1.0)))
    assert float(np.ptp(travel)) > 20.0, "the supplied head rotation did not reach the rig"
    assert np.allclose(
        solved.local_rotations_xyzw[:, NECK_INDEX], (0.0, 0.0, 0.0, 1.0), atol=1e-9
    ), "the neck must keep the torso frame; splitting the chain is unmeasured"
    assert np.array_equal(welded.root_translation_m, solved.root_translation_m), (
        "supplying a head rotation must not move the root"
    )


def test_body_track_rejects_a_malformed_head_rotation():
    frames = 8
    positions = np.tile(_standing_pose(), (frames, 1, 1))
    with pytest.raises(CommercialMultiviewError):
        positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="0" * 64,
            head_world_rotations=np.zeros((frames, 3, 3)),
        )


def test_frame_travel_is_measured_at_the_neck_not_in_world():
    """A head on a turning body travels in world with the neck perfectly still. Scoring
    world rotation would reject that honest motion and, worse, accept a real neck flip
    that happens while the body is still."""
    frames = 12
    angles = np.linspace(0.0, np.radians(120.0), frames)
    turning_body = rodrigues(np.stack([np.zeros(frames), np.zeros(frames), angles], axis=1))

    in_world = _maximum_frame_travel_deg(turning_body, None)
    at_the_neck = _maximum_frame_travel_deg(turning_body, turning_body)

    assert in_world > 5.0, "the head really is moving in world"
    assert at_the_neck < 1e-6, "but the neck is not, and that is what must be measured"


def test_never_delivers_a_head_that_flips_faster_than_a_neck_can_turn():
    """The reject must FIRE, not merely exist — and this asserts what it guarantees.

    Minimum reprojection has no notion of anatomy: a head can flip between frames and
    still sit on every observation, because much of the flip lies along the viewing rays.
    Fed a sequence containing a step no neck can make, the solver must not hand that step
    on. It is free to smooth it away rather than refuse — an impossible observation is
    evidence the observation is wrong — but it may not deliver it.

    The second assertion is the one that stops this test being vacuous: without the reject
    the very same data yields a solution that DOES contain the impossible step, so the
    filter is demonstrably what changed the outcome.
    """
    cameras = _rig()
    frames = 30
    angles = np.zeros(frames)
    angles[frames // 2 :] = np.radians(110.0)      # one impossible step, mid-take
    truth = rodrigues(np.stack([np.zeros(frames), np.zeros(frames), angles], axis=1))
    observations = _synthesise(cameras, truth)

    assert _maximum_frame_travel_deg(truth, None) > MAXIMUM_FRAME_TRAVEL_DEG, (
        "the fixture must actually contain an impossible step"
    )

    solved = solve_head_orientation(cameras, observations, LANDMARKS)
    delivered = _maximum_frame_travel_deg(solved.rotations_world, None)
    assert delivered <= MAXIMUM_FRAME_TRAVEL_DEG, (
        f"delivered a {delivered:.0f} deg/frame head flip"
    )
    assert solved.temporal_weight > 0.0, (
        "the unsmoothed fit was accepted, so the reject did nothing"
    )

    # And the control: with the bound lifted, the same call reproduces the impossible step.
    import autoanim_gnm.head_orientation as module

    original = module.MAXIMUM_FRAME_TRAVEL_DEG
    module.MAXIMUM_FRAME_TRAVEL_DEG = 1.0e9
    try:
        unfiltered = solve_head_orientation(cameras, observations, LANDMARKS)
    finally:
        module.MAXIMUM_FRAME_TRAVEL_DEG = original
    assert _maximum_frame_travel_deg(unfiltered.rotations_world, None) > MAXIMUM_FRAME_TRAVEL_DEG, (
        "without the reject the solver should have delivered the impossible step; "
        "if it does not, this test proves nothing"
    )
