"""D9: the arm bones are aimed from their OWN origins, so the elbow and the wrist land.

D2 aimed the clavicle from the rig's own `Shoulder` pivot and that was right, but the
clavicle has a FIXED rest length, so its child `UpperArm`'s origin cannot in general reach
the captured shoulder: on the shipped D8 delivery it sits 10-13 mm off it. Pass C then
aimed every arm bone by a LANDMARK-TO-LANDMARK direction -- `elbow - shoulder`,
`wrist - elbow` -- from that displaced origin, so the whole arm was carried off by the
displacement and the delivered elbow and wrist missed by the same 14 / 15-20 mm. This is
D7b's mechanism one chain further out, and the fix is D7b's fix: aim the bone from the
origin the rotation actually turns about, then RE-SOLVE the chain below it (the D2c
lesson).

WHAT IS ASSERTED HERE, and why each clause exists:

  * the ON-THE-RAY identity -- with the arm aimed from its own origin the delivered elbow
    can only miss by the LENGTH difference, exactly as D7b's neck can only miss by the
    trunk's length error. This test FAILS on the shipped D8 converter, by construction.
  * the chained identity for the wrist, measured from the PLACED elbow and not from the
    captured one, because that is where the delivered elbow is.
  * the ORACLE: on a body whose captured shoulder is exactly reachable -- the clavicle's
    rest length from its own pivot -- the new aim and the old are the SAME direction and
    the two tracks must agree. Not "agree well": the two code paths are handed
    bit-identical direction vectors, so anything but equality means the change did more
    than aiming.
  * the UNTOUCHABLE list: root, `Hips`, the trunk chain, `Neck`, `Head`, both clavicles
    and every LEG local bit-identical to the old path on the same input.

THE `D8` ARM IS THE SHIPPED CODE PATH, NOT A RE-IMPLEMENTATION. `_joint_origin` is module
level and called by bare name precisely so an instrument can substitute it
(`commercial_multiview` says so in its own docstring, and `tests/test_trunk_resolve.py`
uses the same door for D7's arm). Returning the captured `shoulder` landmark for
`*UpperArm` and the captured `elbow` for `*LowerArm` reproduces D8's `elbow - shoulder`
and `wrist - elbow` aims through the identical call site, the identical `_world_for_bone`
and the identical everything else. Nothing here re-derives the converter's arithmetic.

docs/reviews/arm-origin-2026-09-05.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions

from test_pelvis_frame import CAPTURE_FROM_RIG, convert, posed_body  # noqa: F401
from test_trunk_resolve import rig_points, toe_landmarks, world_positions

ARMS = (
    ("LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
     "left_shoulder", "left_elbow", "left_wrist"),
    ("RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
     "right_shoulder", "right_elbow", "right_wrist"),
)
UNTOUCHABLE = (
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftEye", "RightEye",
    "LeftShoulder", "RightShoulder",
    "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
    "LeftFoot", "RightFoot", "LeftToes", "RightToes",
)


def test_the_package_under_test_is_this_worktree() -> None:
    """The venv's editable install points at the MAIN tree; a green test there is a lie."""

    here = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(here))


# --------------------------------------------------------------------------- the fixtures
def to_capture(vector: np.ndarray) -> np.ndarray:
    """Rig Y-up -> capture Z-up. A permutation and a sign: LOSSLESS, and it must be.

    The fixed-point fixture below writes a rig-frame origin back into the capture-frame
    landmark array and then requires the converter to read the SAME float64 out of it. A
    change of basis that rounded would put a floor under the oracle band.
    """

    return np.asarray(vector, dtype=np.float64) @ CAPTURE_FROM_RIG.T


def short_clavicle_skeleton(shortfall_m: float) -> "type(DETAILED_HUMANOID)":
    """The canonical rig with both clavicles `shortfall_m` shorter along their own axis.

    The clavicle BONE is the `Shoulder` joint and its length is the rest translation of
    its child `UpperArm`; shortening that is exactly the mismatch the delivery carries,
    where the captured shoulder is 6-7 mm further from the pivot than the clavicle can
    reach (`docs/LADDER_EXECUTION_PLAN.md`, the D9 card's decomposition). A NEGATIVE
    shortfall lengthens it, which displaces the origin the other way; both are tested,
    because an aim that only works on one sign is not an aim.
    """

    rest = np.array(DETAILED_HUMANOID.rest_translations_m, dtype=np.float64)
    for name in ("LeftUpperArm", "RightUpperArm"):
        index = DETAILED_HUMANOID.index(name)
        length = float(np.linalg.norm(rest[index]))
        rest[index] = rest[index] * ((length - shortfall_m) / length)
    return DETAILED_HUMANOID.with_rest_translations(rest)


def d8_origin_substitute(points: np.ndarray):
    """The SHIPPED D8 aim, through the converter's own substitution point.

    D8 aimed `*UpperArm` along `elbow - shoulder` and `*LowerArm` along `wrist - elbow`,
    both landmark-to-landmark. Hand back the captured `shoulder` where the new code asks
    for `*UpperArm`'s origin and the captured `elbow` where it asks for `*LowerArm`'s, and
    the call site computes D8's directions bit for bit. Every other joint -- the clavicles
    in pass A, `Spine` in the trunk aim -- gets the real origin.
    """

    shipped = cm._joint_origin
    landmark_for = {
        "LeftUpperArm": "left_shoulder", "LeftLowerArm": "left_elbow",
        "RightUpperArm": "right_shoulder", "RightLowerArm": "right_elbow",
    }

    def origin(world, frame, root_translation, rest, joint_name):
        if joint_name in landmark_for:
            return points[frame, cm.JOINT_INDEX[landmark_for[joint_name]]]
        return shipped(world, frame, root_translation, rest, joint_name)

    return origin


def recorded_arm_origins(positions, spine, skeleton, toes):
    """Convert once, recording the float64 `*UpperArm` origins pass C actually used.

    The recorder wraps the shipped `_joint_origin` and changes nothing: it is the only way
    to see the origin at float64, since a track's rotations are stored float32 and forward
    kinematics from them lands ~1e-7 m away -- which would put a floor under the oracle.
    """

    seen: dict[tuple[int, str], np.ndarray] = {}
    shipped = cm._joint_origin

    def recorder(world, frame, root_translation, rest, joint_name):
        value = shipped(world, frame, root_translation, rest, joint_name)
        seen[(frame, joint_name)] = np.asarray(value, dtype=np.float64).copy()
        return value

    cm._joint_origin = recorder
    try:
        track = cm.positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="a" * 64,
            spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    return track, seen


def arm_landmarks_at_rest_length(positions: np.ndarray, skeleton) -> np.ndarray:
    """The fixture's arms rebuilt at the RIG's own bone lengths, directions kept.

    `posed_body`'s arm landmarks are drawn at 280 and 250 mm from a shoulder that is
    itself a construction, so the rig's 286 / 268 mm bones cannot reach them and the
    LENGTH floor swamps the aim -- 151 mm of it, which is a fixture artefact and not
    anything the delivery carries. Putting each landmark at the bone's own length along
    the SAME direction leaves the pose alone and takes the fixture's length error out of
    the measurement, so what remains at the elbow is the aim and the origin displacement
    the clavicle mismatch introduces.
    """

    rest = {joint.name: np.asarray(joint.rest_translation_m, dtype=np.float64)
            for joint in skeleton.joints}
    out = np.array(positions, dtype=np.float64, copy=True)
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    for _clavicle, _upper, lower, hand, shoulder, elbow, wrist in ARMS:
        origin = out[:, cm.JOINT_INDEX[shoulder]]
        new_elbow = origin + float(np.linalg.norm(rest[lower])) * unit(
            out[:, cm.JOINT_INDEX[elbow]] - origin)
        new_wrist = new_elbow + float(np.linalg.norm(rest[hand])) * unit(
            out[:, cm.JOINT_INDEX[wrist]] - out[:, cm.JOINT_INDEX[elbow]])
        out[:, cm.JOINT_INDEX[elbow]] = new_elbow
        out[:, cm.JOINT_INDEX[wrist]] = new_wrist
    return out


def exactly_reachable_body(skeleton, frames: int = 24):
    """A body whose captured ARM sits EXACTLY where this skeleton's bones can put it.

    The oracle's fixture, and it is a FIXED POINT rather than a construction: moving a
    shoulder landmark moves the torso frame (which is built from `shoulder_across`), which
    moves the `Shoulder` pivot, which moves where the clavicle can reach. So the landmark
    is replaced by the origin the converter's own pass C reported for it, the elbow and
    the wrist are put at the rig's own bone lengths along their own directions, and the
    loop runs until nothing moves. Nothing is re-implemented; every origin comes back out
    of `positions_to_body_track` through the recorder.

    ALL THREE landmarks have to be exact, not just the shoulder. With only the shoulder
    reachable the `UpperArm` aims agree and the `LowerArm` aims still do not: D8 measures
    the forearm from the CAPTURED elbow and D9 from the PLACED one, and those differ by
    the upper arm's own length error (0.115 of a quaternion component on this fixture --
    measured, and the reason this helper does the elbow and the wrist too).

    Returns `(positions, spine, toes, iterations, worst_step_m)`.
    """

    positions, spine, _ = posed_body(frames=frames)
    toes = toe_landmarks(positions)
    worst = float("inf")
    for iteration in range(1, 12):
        positions = arm_landmarks_at_rest_length(positions, skeleton)
        _track, seen = recorded_arm_origins(positions, spine, skeleton, toes)
        worst = 0.0
        for _clavicle, upper, _lower, _hand, shoulder, _elbow, _wrist in ARMS:
            for frame in range(frames):
                origin = seen[(frame, upper)]
                landmark = to_capture(origin)
                worst = max(worst, float(np.linalg.norm(
                    landmark - positions[frame, cm.JOINT_INDEX[shoulder]])))
                positions[frame, cm.JOINT_INDEX[shoulder]] = landmark
        if worst < 1.0e-15:
            positions = arm_landmarks_at_rest_length(positions, skeleton)
            return positions, spine, toes, iteration, worst
    raise AssertionError(f"the arm fixed point did not converge (worst {worst:.3e} m)")


# ----------------------------------------------- THE CLAIM: the elbow lands on its ray
@pytest.mark.parametrize("shortfall_mm", (30.0, -30.0))
def test_the_elbow_lands_on_the_ray_from_the_upper_arms_own_origin(
    shortfall_mm: float,
) -> None:
    """FAILS ON THE SHIPPED D8 CONVERTER. The residual is the arm's LENGTH and nothing else.

    `UpperArm` carries one world rotation and `LowerArm`'s rest is a single offset from it,
    so `elbow_origin = upper_origin + R . rest[LowerArm]`. Aim `R` from `upper_origin` at
    the captured elbow and the elbow lands ON the ray: what remains is
    `| ||elbow - upper_origin|| - L_upper |`, the length error a rigid bone cannot remove.
    Under D8's landmark-to-landmark aim `R` points along `elbow - shoulder_landmark`
    instead, and with the clavicle 30 mm short those are different directions.
    """

    skeleton = short_clavicle_skeleton(shortfall_mm / 1000.0)
    positions, spine, _ = posed_body()
    positions = arm_landmarks_at_rest_length(positions, skeleton)
    toes = toe_landmarks(positions)
    track = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    points = rig_points(positions)
    world = forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=skeleton).astype(np.float64)
    rest = {joint.name: np.asarray(joint.rest_translation_m, dtype=np.float64)
            for joint in skeleton.joints}

    for _clavicle, upper, lower, hand, shoulder, elbow, wrist in ARMS:
        origin = world[:, skeleton.index(upper)]
        captured_elbow = points[:, cm.JOINT_INDEX[elbow]]
        delivered_elbow = world[:, skeleton.index(lower)]
        length = float(np.linalg.norm(rest[lower]))
        reach = np.linalg.norm(captured_elbow - origin, axis=1)
        error = np.linalg.norm(delivered_elbow - captured_elbow, axis=1)
        floor = np.abs(reach - length)
        # 1e-6 m: the track's rotations and root are float32 and that quantisation alone
        # is ~1e-7 m on a 280 mm lever. An exact-to-the-bit assertion would be a claim
        # about float64 that the delivered file cannot carry (`test_trunk_resolve.py`
        # bands its own length identity the same way and for the same reason).
        assert np.abs(error - floor).max() < 1.0e-6, (
            f"{lower}: worst departure from the length floor "
            f"{1000 * np.abs(error - floor).max():.6f} mm")
        # ... and the origin really is displaced on this fixture, or the clause above is
        # satisfied by both aims at once and proves nothing.
        displacement = np.linalg.norm(origin - points[:, cm.JOINT_INDEX[shoulder]], axis=1)
        assert np.median(displacement) > 0.010, (
            f"{upper}: the fixture does not separate the two aims "
            f"(median displacement {1000 * np.median(displacement):.3f} mm)")

        # THE WRIST, chained from the PLACED elbow and not from the captured one.
        delivered_wrist = world[:, skeleton.index(hand)]
        captured_wrist = points[:, cm.JOINT_INDEX[wrist]]
        lower_length = float(np.linalg.norm(rest[hand]))
        wrist_reach = np.linalg.norm(captured_wrist - delivered_elbow, axis=1)
        wrist_error = np.linalg.norm(delivered_wrist - captured_wrist, axis=1)
        wrist_floor = np.abs(wrist_reach - lower_length)
        assert np.abs(wrist_error - wrist_floor).max() < 1.0e-6, (
            f"{hand}: worst departure from the chained length floor "
            f"{1000 * np.abs(wrist_error - wrist_floor).max():.6f} mm")


@pytest.mark.parametrize("shortfall_mm", (30.0, -30.0))
def test_the_shipped_d8_aim_misses_the_elbow_by_the_origin_displacement(
    shortfall_mm: float,
) -> None:
    """MUST-FAIL ARM. D8's own construction, through the converter's substitution point."""

    skeleton = short_clavicle_skeleton(shortfall_mm / 1000.0)
    positions, spine, _ = posed_body()
    positions = arm_landmarks_at_rest_length(positions, skeleton)
    toes = toe_landmarks(positions)
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d8_origin_substitute(points)
    try:
        d8 = cm.positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="a" * 64,
            spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    d9 = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)

    def fk(track):
        return forward_kinematics_positions(
            np.asarray(track.root_translation_m, dtype=np.float64),
            np.asarray(track.local_rotations_xyzw, dtype=np.float64),
            skeleton=skeleton).astype(np.float64)

    d8_world, d9_world = fk(d8), fk(d9)
    for _clavicle, upper, lower, _hand, _shoulder, elbow, _wrist in ARMS:
        captured = points[:, cm.JOINT_INDEX[elbow]]
        d8_error = np.linalg.norm(d8_world[:, skeleton.index(lower)] - captured, axis=1)
        d9_error = np.linalg.norm(d9_world[:, skeleton.index(lower)] - captured, axis=1)
        assert np.median(d9_error) < np.median(d8_error), lower
        # The two arms must share an `UpperArm` ORIGIN and differ only in the aim taken
        # from it -- otherwise the comparison is not about the aim at all.
        origin_gap = np.linalg.norm(
            d8_world[:, skeleton.index(upper)] - d9_world[:, skeleton.index(upper)], axis=1)
        assert origin_gap.max() < 1.0e-9, (
            f"{upper}: the arms do not share an origin (worst {origin_gap.max():.3e} m)")


# -------------------------------------------------------------------------- the oracle
def test_with_the_clavicle_exactly_right_the_new_aim_is_the_old_aim() -> None:
    """THE ORACLE. Where the origin IS the landmark the two aims are one direction.

    If this fails the change did more than aiming -- it is not a band on how well the arm
    is placed, it is a statement that the only thing D9 altered is which point the
    direction is measured FROM. The fixture makes the captured shoulder exactly reachable
    (see `exactly_reachable_body`), so the two code paths are handed bit-identical
    direction vectors and the tracks must come out equal.

    The card's band is 1e-9 and it is asserted as written. BIT-IDENTITY is NOT asserted,
    and the reason is recorded rather than discovered later: the fixed point converges to
    about 1e-18 m rather than exactly, and `LeftHand`'s local rotation is a quaternion
    composed with its own inverse, which rounds differently when its parent moves in the
    eighteenth decimal. The measured gap is ~1e-16 of a quaternion component -- nine
    orders below the band and seven below the float32 storage floor the track is written
    in, so the band is met on its own terms and is not a claim the storage cannot carry
    (the D7b precedent: a 1e-9 deg band written below the float32 floor).
    """

    skeleton = DETAILED_HUMANOID
    positions, spine, toes, iterations, worst = exactly_reachable_body(skeleton)
    assert iterations <= 11 and worst < 1.0e-15
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d8_origin_substitute(points)
    try:
        d8 = cm.positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="a" * 64,
            spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    d9 = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)

    assert np.array_equal(d8.root_translation_m, d9.root_translation_m)
    gap = np.abs(np.asarray(d8.local_rotations_xyzw, dtype=np.float64)
                 - np.asarray(d9.local_rotations_xyzw, dtype=np.float64)).max()
    assert gap < 1.0e-9, f"worst quaternion component gap {gap:.3e}"
    # Every joint OUTSIDE the four arm bones is bit-identical even here, so the residual
    # gap above is confined to the joints the step re-aims and is not a global wobble.
    for name in UNTOUCHABLE:
        index = d8.joint_names.index(name)
        assert np.array_equal(d8.local_rotations_xyzw[:, index],
                              d9.local_rotations_xyzw[:, index]), name


def test_the_exactly_reachable_fixture_really_is_reachable() -> None:
    """The oracle's fixture, proved rather than assumed: the origin IS the landmark.

    Without this the oracle above could pass because the fixture collapsed the two aims
    into one for the wrong reason -- a degenerate arm, say -- and nobody would know. The
    band is a femtometre: `exactly_reachable_body` iterates to about 1e-18 m and a
    bit-exact fixed point does not exist, since each pass recomputes the origin from
    landmarks it has just rewritten.
    """

    skeleton = DETAILED_HUMANOID
    positions, spine, toes, _iterations, _worst = exactly_reachable_body(skeleton)
    _track, seen = recorded_arm_origins(positions, spine, skeleton, toes)
    for _clavicle, upper, _lower, _hand, shoulder, _elbow, _wrist in ARMS:
        for frame in range(len(positions)):
            gap = float(np.linalg.norm(to_capture(seen[(frame, upper)])
                                       - positions[frame, cm.JOINT_INDEX[shoulder]]))
            assert gap < 1.0e-15, (upper, frame, gap)
    # and the arm is a real arm on this fixture, not a collapsed one.
    points = rig_points(positions)
    span = np.linalg.norm(points[:, cm.JOINT_INDEX["left_elbow"]]
                          - points[:, cm.JOINT_INDEX["left_shoulder"]], axis=1)
    assert span.min() > 0.1


# ------------------------------------------------- B2: nothing outside the arms may move
@pytest.mark.parametrize("shortfall_mm", (30.0, -30.0))
def test_everything_outside_the_two_arm_bones_is_bit_identical(shortfall_mm: float) -> None:
    """UNTOUCHABLE. The root, the trunk, the head, both clavicles and every leg local.

    With toe landmarks supplied the feet do not read `torso_world` either, so the whole
    lower body is exactly what D8 wrote -- float for float, after the ground projection.
    """

    skeleton = short_clavicle_skeleton(shortfall_mm / 1000.0)
    positions, spine, _ = posed_body()
    toes = toe_landmarks(positions)
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d8_origin_substitute(points)
    try:
        d8 = cm.positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="a" * 64,
            spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    d9 = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)

    assert np.array_equal(d8.root_translation_m, d9.root_translation_m)
    assert np.array_equal(d8.foot_contacts, d9.foot_contacts)
    assert np.array_equal(d8.ticks, d9.ticks)
    assert d8.joint_names == d9.joint_names
    assert np.array_equal(d8.rest_translations_m, d9.rest_translations_m)
    for name in UNTOUCHABLE:
        index = d8.joint_names.index(name)
        assert np.array_equal(d8.local_rotations_xyzw[:, index],
                              d9.local_rotations_xyzw[:, index]), name
    # ... and the four arm bones DID move, or the list above proves nothing.
    for _clavicle, upper, lower, _hand, _shoulder, _elbow, _wrist in ARMS:
        for name in (upper, lower):
            index = d8.joint_names.index(name)
            assert not np.array_equal(d8.local_rotations_xyzw[:, index],
                                      d9.local_rotations_xyzw[:, index]), name


def test_the_fingers_inherit_and_carry_no_solve_of_their_own(shortfall_mm: float = 30.0
                                                             ) -> None:
    """The hands and fingers follow the re-solved forearm; their LOCALS do not move.

    Every finger takes `_finger_rest_local`, a constant, so its local is bit-identical.
    `LeftHand` takes its PARENT's world, and its local is therefore that world composed
    with its own inverse -- an identity that rounds differently when the parent moves, so
    it is banded at 1e-6 rather than asserted bitwise (the measured departure is ~1e-16,
    and the delivered rotation is the identity to within it). That is the "hands and
    fingers inherit" clause in the card, asserted rather than assumed.
    """

    skeleton = short_clavicle_skeleton(shortfall_mm / 1000.0)
    positions, spine, _ = posed_body()
    toes = toe_landmarks(positions)
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d8_origin_substitute(points)
    try:
        d8 = cm.positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="a" * 64,
            spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    d9 = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    fingers = [i for i, name in enumerate(d8.joint_names)
               if name.startswith(("LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing",
                                   "LeftLittle", "RightThumb", "RightIndex",
                                   "RightMiddle", "RightRing", "RightLittle"))]
    assert fingers
    for index in fingers:
        assert np.array_equal(d8.local_rotations_xyzw[:, index],
                              d9.local_rotations_xyzw[:, index]), d8.joint_names[index]
    for hand in ("LeftHand", "RightHand"):
        index = d8.joint_names.index(hand)
        gap = np.abs(np.asarray(d8.local_rotations_xyzw[:, index], dtype=np.float64)
                     - np.asarray(d9.local_rotations_xyzw[:, index], dtype=np.float64)).max()
        assert gap < 1.0e-6, (hand, gap)
        identity = np.abs(np.asarray(d9.local_rotations_xyzw[:, index], dtype=np.float64)
                          - np.asarray((0.0, 0.0, 0.0, 1.0))).max()
        assert identity < 1.0e-6, (hand, identity)


def test_the_legs_are_still_aimed_landmark_to_landmark() -> None:
    """D9 is ARMS ONLY. The thigh's world +Y must still be parallel to `knee - hip`.

    The card's leg dry run is REPORTED and not shipped, and this is the assertion that
    keeps it that way: if a later pass moves the legs to their own origins this test goes
    red and the change cannot arrive unnoticed.
    """

    skeleton = short_clavicle_skeleton(0.030)
    positions, spine, _ = posed_body()
    toes = toe_landmarks(positions)
    track = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="a" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton, toe_world_z_up_m=toes)
    points = rig_points(positions)
    rotations = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    world = forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64), rotations,
        skeleton=skeleton).astype(np.float64)
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    # ARCSIN OF THE CROSS PRODUCT, never arccos of the dot: `arccos` near 1 is
    # ill-conditioned and a float32 quaternion's norm alone reads as 0.036 deg through it
    # (`tests/test_trunk_resolve.py` carries the same note).
    angle = lambda a, b: np.degrees(np.arcsin(
        np.clip(np.linalg.norm(np.cross(a, b), axis=1), 0.0, 1.0)))
    for thigh, shin, hip, knee in (("LeftUpperLeg", "LeftLowerLeg", "left_hip", "left_knee"),
                                   ("RightUpperLeg", "RightLowerLeg", "right_hip",
                                    "right_knee")):
        axis = unit(world[:, skeleton.index(shin)] - world[:, skeleton.index(thigh)])
        landmark = unit(points[:, cm.JOINT_INDEX[knee]] - points[:, cm.JOINT_INDEX[hip]])
        assert angle(axis, landmark).max() < 1.0e-3, thigh
