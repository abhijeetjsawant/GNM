"""D2c: the clavicle's temporal reject, and the three properties that make it safe.

The rule is REACHABILITY, not a step test: a frame is accepted only if its clavicle LOCAL
rotation is within `ceiling * (t - t_last_accepted)` of the last accepted frame's. The
envelope grows with elapsed frames, so a rejected run ends on its own and the constant is
a physical ceiling and nothing else.

Three things are asserted here that no positional score in this lane can see:

  * a rejected frame is REPLACED AND THEN RE-SOLVED -- the elbow and the wrist still lie
    on the directions their landmarks ask for, which a replacement alone would break by
    swinging the whole arm rigidly with the clavicle;
  * a multi-frame excursion is rejected WHOLE, where a step test accepts its plateau;
  * the rule is read at the joint and never in world, because a bone on a turning body
    travels in world with its own joint perfectly still.

docs/reviews/clavicle-origin-2026-09-02.md section 15.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions
from autoanim_gnm.commercial_multiview import (
    CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S,
    JOINT_INDEX,
    JOINT_NAMES,
    _reachable_clavicle_sequence,
    positions_to_body_track,
)

SAMPLE_RATE_HZ = 30
CEILING = CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S / SAMPLE_RATE_HZ
PROVENANCE = "0" * 64
CLAVICLE_CHAIN = (
    "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
    "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand",
)


# ------------------------------------------------------------------ the swappable rules
def passthrough_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """No reject at all. The D2b arm, reached through the identical code path."""
    rotations = np.asarray(local_rotations, dtype=np.float64)
    return rotations.copy(), np.ones(len(rotations), dtype=bool)


def step_test_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """The control the brief names: accept unless the SINGLE step exceeds the ceiling."""
    rotations = np.asarray(local_rotations, dtype=np.float64)
    accepted = np.ones(len(rotations), dtype=bool)
    for frame in range(1, len(rotations)):
        accepted[frame] = (
            cm._quaternion_travel_deg(rotations[frame - 1], rotations[frame])
            <= ceiling_deg_per_frame
        )
    return rotations.copy(), accepted


def world_measured_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """The head lane's lesson as a control: the SAME reject, read in WORLD."""
    local = np.asarray(local_rotations, dtype=np.float64)
    parent = np.asarray(parent_world_rotations, dtype=np.float64)
    world = np.stack([cm._quaternion_multiply(parent[f], local[f]) for f in range(len(local))])
    _replaced, accepted = _reachable_clavicle_sequence(world, parent, ceiling_deg_per_frame)
    return local.copy(), accepted


# ------------------------------------------------------------------------- the fixtures
def _standing_pose() -> np.ndarray:
    """One frame of a plausible standing skeleton, in capture Z-up metres.

    Every landmark is finite on purpose: `positions_to_body_track` rejects a non-finite
    input outright, and SOMA-77's `left_ear`/`right_ear` are populated on zero frames, so
    a truth array taken from the fixture cannot be handed to the converter directly.
    """
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


def _breathing_take(frames: int = 40) -> np.ndarray:
    """A slow, entirely reachable take: the arms swing and the torso sways.

    Nothing here approaches 26.67 deg of clavicle travel in a frame, which is the point --
    a reject ceiling must be inert on motion a body can perform.
    """
    take = np.tile(_standing_pose(), (frames, 1, 1))
    phase = np.linspace(0.0, 2.0 * np.pi, frames)
    for frame in range(frames):
        swing = 0.05 * math.sin(phase[frame])
        for name in ("left_elbow", "left_wrist"):
            take[frame, JOINT_INDEX[name], 1] += 2.0 * swing
        for name in ("right_elbow", "right_wrist"):
            take[frame, JOINT_INDEX[name], 1] -= 2.0 * swing
        for name in ("left_shoulder", "right_shoulder", "neck"):
            take[frame, JOINT_INDEX[name], 1] += 0.4 * swing
        take[frame, JOINT_INDEX["left_shoulder"], 2] += 0.5 * swing
        take[frame, JOINT_INDEX["right_shoulder"], 2] -= 0.5 * swing
    return take


def _solve(positions, rule=None):
    saved = cm._reachable_clavicle_sequence
    if rule is not None:
        cm._reachable_clavicle_sequence = rule
    try:
        return positions_to_body_track(
            positions, sample_rate_hz=SAMPLE_RATE_HZ, provenance_sha256=PROVENANCE
        )
    finally:
        cm._reachable_clavicle_sequence = saved


def _rejected_frames(track_positions, clavicle="LeftShoulder"):
    """Which frames the shipped rule rejects on this take, via the real function."""
    seen: dict[str, np.ndarray] = {}
    real = cm._reachable_clavicle_sequence

    def recording(local, parent_world, ceiling):
        replaced, accepted = real(local, parent_world, ceiling)
        seen[f"call{len(seen)}"] = accepted
        return replaced, accepted

    _solve(track_positions, recording)
    return seen


def _steps_deg(quaternions):
    a, b = quaternions[:-1], quaternions[1:]
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), -1.0, 1.0)))


def _axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(degrees) * 0.5
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


# --------------------------------------------------------------------------- the tests
def test_the_module_under_test_is_this_worktree():
    """The venv's editable install points at the MAIN checkout; a green run there proves
    nothing about this branch."""
    root = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(root)), (
        f"autoanim_gnm resolved to {autoanim_gnm.__file__}, not {root}: re-run with "
        f"PYTHONPATH=$PWD/src"
    )


def test_a_reachable_take_is_accepted_whole_and_the_track_is_untouched():
    """The exactness property: on motion a body can perform the reject is INERT.

    Zero rejected frames on both clavicles, and the track is bit-identical to the same
    solve with the rule swapped for a pass-through -- so no accepted frame is smoothed,
    nudged or renormalised on its way through."""
    take = _breathing_take()
    accepted = _rejected_frames(take)
    assert len(accepted) == 2, "both clavicles must be treated, independently"
    for mask in accepted.values():
        assert bool(mask.all()), (
            f"the ceiling fired on true motion: {int((~mask).sum())} frames rejected"
        )
    shipped = _solve(take)
    no_reject = _solve(take, passthrough_rule)
    assert np.array_equal(
        np.asarray(shipped.local_rotations_xyzw), np.asarray(no_reject.local_rotations_xyzw)
    ), "an inert reject must leave the track bit-identical"
    peak = max(
        float(_steps_deg(np.asarray(shipped.local_rotations_xyzw, dtype=np.float64)[:, DETAILED_HUMANOID.index(name)]).max())
        for name in ("LeftShoulder", "RightShoulder")
    )
    assert peak < CEILING, f"the fixture's own clavicle peak {peak:.2f} is not below {CEILING:.2f}"


def test_a_single_frame_pop_is_rejected_and_the_arm_is_re_solved_onto_its_landmarks():
    """Replace, then RE-SOLVE -- and nothing outside the clavicle chain moves at all."""
    take = _breathing_take()
    popped = take.copy()
    frame = 20
    # Carry the shoulder landmark across the rig's own shoulder origin: the lever inverts
    # and the clavicle direction flips. This is the f48 pop of the delivered take, made.
    popped[frame, JOINT_INDEX["left_shoulder"]] = (-0.05, 0.10, 1.62)

    before = _solve(popped, passthrough_rule)
    local_before = np.asarray(before.local_rotations_xyzw, dtype=np.float64)
    index = DETAILED_HUMANOID.index("LeftShoulder")
    assert float(_steps_deg(local_before[:, index]).max()) > CEILING, (
        "the fixture does not contain a pop: there is nothing for the reject to catch"
    )

    accepted = _rejected_frames(popped)
    left = accepted["call0"]
    assert not bool(left[frame]), "the popped frame must be rejected"
    assert int((~left).sum()) <= 3, "a single-frame pop must not reject a long run"

    after = _solve(popped)
    local_after = np.asarray(after.local_rotations_xyzw, dtype=np.float64)

    # every joint outside the clavicle chain is bit-identical
    for position, joint in enumerate(DETAILED_HUMANOID.joints):
        if joint.name in CLAVICLE_CHAIN:
            continue
        assert np.array_equal(local_after[:, position], local_before[:, position]), (
            f"{joint.name} moved, and it is not on the clavicle chain"
        )

    # and on the rejected frame the elbow and the wrist still lie where the landmarks ask
    fk = forward_kinematics_positions(
        np.asarray(after.root_translation_m, dtype=np.float64),
        np.asarray(after.local_rotations_xyzw, dtype=np.float64),
        skeleton=DETAILED_HUMANOID,
    )
    y_up = np.stack([popped[..., 0], popped[..., 2], -popped[..., 1]], axis=-1)

    def unit(vector):
        return np.asarray(vector, dtype=np.float64) / np.linalg.norm(vector)

    # The control that gives the tolerance a meaning: REPLACE WITHOUT RE-SOLVING -- the
    # replaced clavicle with the arm locals left exactly as they were solved. That is what
    # the brief warns against, and it is the same arithmetic minus pass C.
    swung = local_before.copy()
    swung[:, index] = local_after[:, index]
    fk_swung = forward_kinematics_positions(
        np.asarray(before.root_translation_m, dtype=np.float64), swung,
        skeleton=DETAILED_HUMANOID,
    )

    def offset_deg(positions, parent, child, tail, head):
        rig = unit(positions[frame, DETAILED_HUMANOID.index(child)]
                   - positions[frame, DETAILED_HUMANOID.index(parent)])
        want = unit(y_up[frame, JOINT_INDEX[head]] - y_up[frame, JOINT_INDEX[tail]])
        return float(np.degrees(np.arccos(np.clip(np.dot(rig, want), -1.0, 1.0))))

    for parent, child, tail, head in (
        ("LeftUpperArm", "LeftLowerArm", "left_shoulder", "left_elbow"),
        ("LeftLowerArm", "LeftHand", "left_elbow", "left_wrist"),
    ):
        # 0.05 deg, not zero: `BodyTrack` stores the rotations as float32, and reading a
        # direction back out of a float32 quaternion costs about a hundredth of a degree.
        assert offset_deg(fk, parent, child, tail, head) < 0.05, (
            f"{parent}->{child} left its landmark direction on the replaced frame: the arm "
            "was swung rigidly with the clavicle instead of being re-solved"
        )
        assert offset_deg(fk_swung, parent, child, tail, head) > 5.0, (
            f"{parent}->{child}: replacing WITHOUT re-solving must visibly break this "
            "band, or the band is not testing anything"
        )


def test_a_multi_frame_excursion_is_rejected_whole_where_a_step_test_takes_its_plateau():
    """The reason the rule is reachability. A step test sees the two transitions and
    accepts everything between them -- which is the wrong pose, held."""
    frames = 15
    excursion = _axis_angle((0.0, 0.0, 1.0), 100.0)   # > 3 * 26.67, so unreachable in 3
    sequence = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (frames, 1))
    sequence[5:8] = excursion
    parent = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (frames, 1))

    _replaced, accepted = _reachable_clavicle_sequence(sequence, parent, CEILING)
    assert list(np.flatnonzero(~accepted)) == [5, 6, 7], (
        "reachability must reject the whole excursion, not only its transitions"
    )
    _step_replaced, step_accepted = step_test_rule(sequence, parent, CEILING)
    assert list(np.flatnonzero(~step_accepted)) == [5, 8], (
        "the step test must catch only the two transitions"
    )
    assert bool(step_accepted[6]) and bool(step_accepted[7]), (
        "the control has to ACCEPT the plateau, which is the defect it demonstrates"
    )
    # and the replacement is an interpolation between the accepted neighbours, not a hold
    for frame in (5, 6, 7):
        assert cm._quaternion_travel_deg(_replaced[frame], excursion) > 50.0


def test_an_outlier_first_frame_recovers_inside_the_envelope_bound():
    """Frame 0 is accepted by definition, so a bad frame 0 poisons the reference. The
    growing envelope is what recovers it: it passes 180 deg after ceil(180/26.67) = 7
    frames, so recovery is bounded at 7 whatever the outlier is."""
    frames = 20
    sequence = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (frames, 1))
    sequence[0] = _axis_angle((0.0, 1.0, 0.0), 150.0)
    parent = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (frames, 1))

    _replaced, accepted = _reachable_clavicle_sequence(sequence, parent, CEILING)
    recovered = int(np.flatnonzero(accepted[1:])[0]) + 1
    assert recovered == 6, f"expected recovery on frame 6, got {recovered}"
    bound = math.ceil(180.0 / CEILING)
    assert recovered <= bound, f"recovery must be bounded by the envelope, {bound} frames"
    assert bool(accepted[6:].all()), "everything after recovery is honest and reachable"


def test_the_world_measured_variant_rejects_honest_motion_on_a_turning_body():
    """The head lane's lesson, demonstrated. A clavicle held perfectly still on a body
    that is spinning travels in WORLD at the body's rate; a world-measured ceiling calls
    that a defect and the local one correctly sees nothing."""
    frames = 30
    still = np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (frames, 1))
    for degrees_per_frame, world_should_fire in ((12.0, False), (45.0, True)):
        parent = np.stack(
            [_axis_angle((0.0, 1.0, 0.0), degrees_per_frame * frame) for frame in range(frames)]
        )
        _r, local_accepted = _reachable_clavicle_sequence(still, parent, CEILING)
        _r2, world_accepted = world_measured_rule(still, parent, CEILING)
        assert bool(local_accepted.all()), (
            f"the local rule fired on a body turning at {degrees_per_frame} deg/frame with "
            "its clavicle perfectly still"
        )
        fired = int((~world_accepted).sum())
        if world_should_fire:
            assert fired > frames // 2, (
                f"a scratch spin at {degrees_per_frame} deg/frame "
                f"({degrees_per_frame * SAMPLE_RATE_HZ:.0f} deg/s) must break a "
                f"world-measured ceiling; it rejected {fired} of {frames}"
            )
        else:
            assert fired == 0, (
                f"a pirouette at {degrees_per_frame} deg/frame is under the ceiling and "
                f"neither variant should fire; the world one rejected {fired}"
            )
