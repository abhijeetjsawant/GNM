"""D2: the clavicle direction is measured from the joint the rotation turns about.

`positions_to_body_track` used to measure the clavicle direction from `pelvis + 0.72 *
torso_up` -- a synthetic anchor on the torso axis -- while `_world_for_bone` applied the
result as a rotation about the rig's own Shoulder origin, 110 mm out and 110 mm higher.
These tests pin the three claims that matter:

  1. `_joint_origin` IS `forward_kinematics_positions`, for the two clavicles, on the
     canonical and on a sized skeleton;
  2. on a synthetic canonical body the six arm landmarks round-trip, and the legacy anchor
     swapped back in FAILS (the must-fail control lives inside the test);
  3. nothing outside the clavicle chain moves.

WHAT TEST 2 HAD TO BE CHANGED TO SAY, and it is the step's finding. The round trip built
the way `retarget_cost.landmarks_from_fk` builds it CANNOT reach a few millimetres, and
not because of the clavicle: it writes the rig's UpperLeg origins into the hip slots
(anatomically right -- they are the femoral joint centres) while the converter puts `Hips`
on that midpoint, 80 mm higher. So the second solve places the rig 80 mm lower against the
same landmarks, and the clavicle -- the first direction ever measured in the RIG's frame
rather than between two landmarks -- is the first thing that can see it. Take the rig's own
hip drop out of the re-solve's origin and the round trip closes to under a millimetre, which
is what `test_the_roundtrip_residual_is_root_placement_not_the_clavicle` asserts. The legs
reading 0.00 through all of it was never evidence the root placement is right; it is
evidence that a translation-invariant chain scored root-relatively cannot see a translation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import (
    DETAILED_HUMANOID,
    _rotate_vector,
    forward_kinematics_positions,
)

SHA = "0" * 64
# The real helper, bound at import: every control below swaps `cm._joint_origin`, and a
# variant that reached for it through the module would call itself.
TRUE_ORIGIN = cm._joint_origin
CLAVICLES = ("LeftShoulder", "RightShoulder")
# D9 (2026-09-05): the four arm bones are aimed from their own origin too, the same idiom.
ARM_BONES = ("LeftUpperArm", "LeftLowerArm", "RightUpperArm", "RightLowerArm")
ARM_LANDMARKS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist")
LEG_LANDMARKS = ("left_hip", "right_hip", "left_knee", "right_knee",
                 "left_ankle", "right_ankle")
# "exactly 0.00 mm", in the only sense available: `forward_kinematics_positions` returns
# float32, so a metre-scale coordinate is quantised at ~1e-4 mm. A micron is three orders
# below anything this step moves and two below that quantisation's reach over a chain.
EXACT_MM = 1e-3
# `retarget_cost.RIG_FOR`, copied rather than imported: `tools/swap-harness` is not on the
# test path, and a rig joint moving under one of these names should break this file too.
RIG_FOR = {
    "neck": "Neck",
    "left_shoulder": "LeftUpperArm", "right_shoulder": "RightUpperArm",
    "left_elbow": "LeftLowerArm", "right_elbow": "RightLowerArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_hip": "LeftUpperLeg", "right_hip": "RightUpperLeg",
    "left_knee": "LeftLowerLeg", "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
}


# --------------------------------------------------------------------------- the trap
def test_autoanim_gnm_is_imported_from_this_worktree():
    """The venv's editable install points at the MAIN tree's `src`.

    Every measurement in this step is worthless if the tests exercise another checkout's
    converter, and the failure is silent. `PYTHONPATH=$PWD/src` is what makes it right.
    """
    import autoanim_gnm

    resolved = Path(autoanim_gnm.__file__).resolve()
    root = Path(__file__).resolve().parents[1]
    assert str(resolved).startswith(str(root)), (
        f"autoanim_gnm resolved to {resolved}, outside this worktree ({root}). "
        f"Run pytest with PYTHONPATH=$PWD/src."
    )


# --------------------------------------------------------------------------- helpers
def _pose(frames: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Non-trivial local rotations for the canonical rig, and a moving root."""
    rng = np.random.default_rng(20260902)
    joints = len(DETAILED_HUMANOID.joints)
    local = np.zeros((frames, joints, 4))
    local[..., 3] = 1.0
    # A handful of joints actually turn, by an amount a body could hold, and the amount
    # changes over the take so a constant cannot pass anything below.
    driven = ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
              "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
              "LeftLowerArm", "RightLowerArm", "LeftUpperLeg", "RightUpperLeg",
              "LeftLowerLeg", "RightLowerLeg")
    for name in driven:
        index = DETAILED_HUMANOID.index(name)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(18.0) * np.sin(np.linspace(0.0, 2.4, frames) + rng.uniform(0, 3))
        local[:, index, :3] = axis * np.sin(angle / 2.0)[:, None]
        local[:, index, 3] = np.cos(angle / 2.0)
    root = np.zeros((frames, 3))
    root[:, 0] = 0.01 * np.arange(frames)
    root[:, 1] = 0.9 + 0.005 * np.sin(np.linspace(0.0, 3.0, frames))
    root[:, 2] = -4.0 + 0.02 * np.arange(frames)
    return root, local


def _landmarks_from_fk(fk: np.ndarray, skeleton=DETAILED_HUMANOID) -> np.ndarray:
    """`retarget_cost.landmarks_from_fk`, reproduced so the test needs no swap harness."""
    out = np.zeros((len(fk), len(cm.JOINT_NAMES), 3))
    for name, rig in RIG_FOR.items():
        out[:, cm.JOINT_INDEX[name]] = fk[:, skeleton.index(rig)]
    head = fk[:, skeleton.index("Head")]
    for name in cm.JOINT_NAMES:
        index = cm.JOINT_INDEX[name]
        if not out[:, index].any():
            out[:, index] = head
    out[:, cm.JOINT_INDEX["root"]] = 0.5 * (
        out[:, cm.JOINT_INDEX["left_hip"]] + out[:, cm.JOINT_INDEX["right_hip"]])
    return out


def _to_z_up(y_up: np.ndarray) -> np.ndarray:
    return np.stack([y_up[..., 0], -y_up[..., 2], y_up[..., 1]], axis=-1)


def _from_z_up(z_up: np.ndarray) -> np.ndarray:
    """The converter's own change of basis -- what `points` is inside it."""
    return np.stack([z_up[..., 0], z_up[..., 2], -z_up[..., 1]], axis=-1)


def _solve(source_z_up: np.ndarray, skeleton=None):
    saved = cm.DETAILED_HUMANOID
    if skeleton is not None:
        cm.DETAILED_HUMANOID = skeleton
    try:
        return cm.positions_to_body_track(
            source_z_up, sample_rate_hz=30, provenance_sha256=SHA)
    finally:
        cm.DETAILED_HUMANOID = saved


def _fk(track, skeleton=DETAILED_HUMANOID) -> np.ndarray:
    return forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=skeleton).astype(np.float64)


def _score(fk: np.ndarray, reference_y_up: np.ndarray, skeleton=DETAILED_HUMANOID) -> dict:
    """Root-relative per-landmark error in mm -- `retarget_cost.score`'s convention."""
    origin_fk = 0.5 * (fk[:, skeleton.index("LeftUpperLeg")]
                       + fk[:, skeleton.index("RightUpperLeg")])
    origin_ref = 0.5 * (reference_y_up[:, cm.JOINT_INDEX["left_hip"]]
                        + reference_y_up[:, cm.JOINT_INDEX["right_hip"]])
    return {
        name: float(np.median(np.linalg.norm(
            (fk[:, skeleton.index(rig)] - origin_fk)
            - (reference_y_up[:, cm.JOINT_INDEX[name]] - origin_ref), axis=1)) * 1000.0)
        for name, rig in RIG_FOR.items()
    }


def _sized(skeleton, factors: dict):
    import dataclasses

    joints = list(skeleton.joints)
    for name, factor in factors.items():
        index = skeleton.index(name)
        joints[index] = dataclasses.replace(
            joints[index],
            rest_translation_m=tuple(
                np.asarray(joints[index].rest_translation_m) * factor))
    return dataclasses.replace(skeleton, joints=tuple(joints))


SIZING = {"LeftShoulder": 1.19, "RightShoulder": 1.19, "LeftUpperArm": 0.88,
          "RightUpperArm": 0.88, "LeftLowerArm": 1.07, "RightLowerArm": 1.07,
          "LeftHand": 0.93, "RightHand": 0.93, "Spine": 1.11, "Chest": 0.94,
          "LeftLowerLeg": 1.05, "RightLowerLeg": 1.05}


# --------------------------------------------------------------------------- 1. the helper
@pytest.mark.parametrize("label", ("canonical", "sized"))
def test_joint_origin_is_forward_kinematics_for_both_clavicles(label):
    """The helper must BE the FK recursion, not merely resemble it.

    Tolerance, and why it is not the 1e-12 the plan asked for: `forward_kinematics_
    positions` returns `positions.astype(np.float32)`, so at metre scale its own answer is
    quantised at ~1e-7 m. The helper is exact in float64 and is therefore compared against
    FK at float32 resolution, and against a float64 walk written out longhand here at
    1e-12. Deviation from the plan, recorded rather than worked around.
    """
    skeleton = DETAILED_HUMANOID if label == "canonical" else _sized(DETAILED_HUMANOID, SIZING)
    root, local = _pose()
    track_root = root.copy()
    fk = forward_kinematics_positions(track_root, local, skeleton=skeleton).astype(np.float64)

    # world rotations, rebuilt from the locals exactly as FK does
    world = np.zeros_like(local)
    for index, joint in enumerate(skeleton.joints):
        world[:, index] = (
            local[:, index] if joint.parent == -1
            else cm._quaternion_multiply(world[:, joint.parent], local[:, index]))
    rest = {j.name: np.asarray(j.rest_translation_m, dtype=np.float64) for j in skeleton.joints}

    saved = cm.DETAILED_HUMANOID
    cm.DETAILED_HUMANOID = skeleton
    try:
        for name in CLAVICLES:
            for frame in (0, 5, len(root) - 1):
                got = cm._joint_origin(world, frame, track_root, rest, name)
                # (a) against FK's own output, at FK's float32 resolution
                assert np.allclose(got, fk[frame, skeleton.index(name)], atol=1e-6), name
                # (b) against a float64 walk written out here, at 1e-12
                chain, index = [], skeleton.index(name)
                while index != -1:
                    chain.append(index)
                    index = skeleton.joints[index].parent
                chain.reverse()
                want = track_root[frame] + rest[skeleton.joints[chain[0]].name]
                for parent, child in zip(chain, chain[1:]):
                    want = want + _rotate_vector(
                        world[frame, parent], rest[skeleton.joints[child].name])
                assert np.allclose(got, want, atol=1e-12), name
    finally:
        cm.DETAILED_HUMANOID = saved


def test_joint_origin_as_the_converter_actually_calls_it():
    """Wrap the real converter, do not re-implement it: record every call and check it.

    The recorded origin is compared against FK of the FINISHED track ROOT-RELATIVELY,
    because `positions_to_body_track` ends in `project_generated_foot_contacts`, which
    shifts the root by ~140 mm after these calls are made. The relative geometry is what
    the direction depends on and is what must match.
    """
    root, local = _pose()
    fk_reference = forward_kinematics_positions(root, local, skeleton=DETAILED_HUMANOID)
    source = _to_z_up(_landmarks_from_fk(fk_reference.astype(np.float64)))

    recorded: dict = {}
    real = cm._joint_origin

    def recording(world, frame, root_translation, rest, joint_name):
        value = real(world, frame, root_translation, rest, joint_name)
        recorded.setdefault(joint_name, {})[frame] = (
            np.asarray(value).copy(), np.asarray(root_translation[frame]).copy())
        return value

    cm._joint_origin = recording
    try:
        track = _solve(source)
    finally:
        cm._joint_origin = real

    # D2 asked for the clavicles alone; D9 aims the arm bones from their own origin as well.
    assert set(recorded) == set(CLAVICLES) | set(ARM_BONES), sorted(recorded)
    assert len(recorded["LeftShoulder"]) == len(source)
    fk = forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=DETAILED_HUMANOID).astype(np.float64)
    hips = DETAILED_HUMANOID.index("Hips")
    for name in CLAVICLES:
        index = DETAILED_HUMANOID.index(name)
        for frame, (value, _) in recorded[name].items():
            # relative to Hips, which removes the later foot-contact root correction
            got = value - (recorded["LeftShoulder"][frame][1]
                           + np.asarray(DETAILED_HUMANOID.joints[hips].rest_translation_m))
            want = fk[frame, index] - fk[frame, hips]
            assert np.allclose(got, want, atol=1e-6), (name, frame, got, want)


# --------------------------------------------------------------------------- 2. round trip
def _legacy_anchor(points_y_up: np.ndarray, scalar: float = 0.72):
    """The pre-D2 origin, as a helper. It cannot be written against `_joint_origin`'s
    signature -- the anchor needs the LENGTH of `torso_up` and the signature carries the
    rig, not the landmarks -- so it closes over the converter's own Y-up landmark array.
    """
    def helper(world, frame, root_translation, rest, joint_name):
        p = points_y_up[frame]
        pelvis = 0.5 * (p[cm.JOINT_INDEX["left_hip"]] + p[cm.JOINT_INDEX["right_hip"]])
        return pelvis + scalar * (p[cm.JOINT_INDEX["neck"]] - pelvis)
    return helper


def _hip_drop_removed(world, frame, root_translation, rest, joint_name):
    """`_joint_origin` with the rig's own UpperLeg drop taken back out -- the origin the
    rig would have if its LEG ROOTS, not `Hips`, sat on the hip-landmark midpoint. No
    constant: the offset is the skeleton's own rest translation."""
    skeleton = cm.DETAILED_HUMANOID
    drop = 0.5 * (np.asarray(rest["LeftUpperLeg"]) + np.asarray(rest["RightUpperLeg"]))
    return TRUE_ORIGIN(world, frame, root_translation, rest, joint_name) - _rotate_vector(
        world[frame, skeleton.index("Hips")], drop)


def _shipped(_points_y_up):
    return TRUE_ORIGIN


def _make_legacy(scalar: float = 0.72):
    """A FACTORY, not a helper, and that distinction is the whole point.

    The legacy anchor is expressed in the LANDMARK frame, so it must read the landmarks of
    the pass it runs in. Pass 2's input is the synthetic body, not the capture; a helper
    closed over pass 1's array would make the "before" arm something the pre-D2 converter
    never computed. That bug was live in this file and in the gate, and in the gate it
    showed up as the legacy control missing the committed 41.57/47.05 mm by 20 mm.
    """
    def factory(points_y_up):
        return _legacy_anchor(points_y_up, scalar)
    return factory


def _make_hip_drop_removed(_points_y_up):
    return _hip_drop_removed


def _round_trip(make_pass1=None, make_pass2=None):
    make_pass1 = make_pass1 or _shipped
    make_pass2 = make_pass2 or make_pass1
    root, local = _pose(frames=14)
    fk0 = forward_kinematics_positions(root, local, skeleton=DETAILED_HUMANOID).astype(np.float64)
    source = _to_z_up(_landmarks_from_fk(fk0))
    real = cm._joint_origin
    try:
        cm._joint_origin = make_pass1(_from_z_up(source))
        fk1 = _fk(_solve(source))
        synth = _landmarks_from_fk(fk1)
        cm._joint_origin = make_pass2(synth)
        fk2 = _fk(_solve(_to_z_up(synth)))
    finally:
        cm._joint_origin = real
    return _score(fk2, synth)


def test_legs_and_neck_round_trip_exactly_and_the_clavicle_chain_is_the_only_thing_that_moves():
    err = _round_trip()
    for name in LEG_LANDMARKS + ("neck",):
        assert err[name] == pytest.approx(0.0, abs=EXACT_MM), (name, err[name])


def test_the_roundtrip_residual_is_root_placement_not_the_clavicle():
    """The step's finding, as an assertion.

    As the instrument builds it, the round trip cannot reach the pre-registered 5 mm --
    the hip landmarks it feeds back are the rig's UpperLeg origins while the converter
    puts `Hips` on their midpoint, 80 mm higher, so the re-solve sits 80 mm low against
    the same landmarks. Remove the rig's own hip drop from the RE-SOLVE's origin only --
    the delivered path untouched -- and the six arm landmarks close to under a
    millimetre. That is the attribution: all of the residual is root placement.
    """
    shipped = _round_trip()
    corrected = _round_trip(make_pass2=_make_hip_drop_removed)
    # D2b (2026-09-03) placed the root on the captured hips, so the SHIPPED round trip now
    # closes, and the control that once attributed the residual to root placement inverts:
    # removing the hip drop a SECOND time un-corrects the round trip by the same 60-80 mm.
    # The attribution stands, read the other way round -- the control is now a must-fail.
    assert max(shipped[n] for n in ARM_LANDMARKS) <= 1.0, shipped
    assert max(corrected[n] for n in ARM_LANDMARKS) > 5.0, corrected
    for name in LEG_LANDMARKS + ("neck",):
        assert corrected[name] == pytest.approx(0.0, abs=EXACT_MM), name


def test_the_legacy_anchor_fails_the_same_round_trip():
    """MUST FAIL, on the identical code path, with only the origin helper swapped.

    This is the pre-D2 converter exactly: the anchor reads each pass's OWN landmarks, so
    the arm is what that code actually computed and not an artefact of a stale closure.
    Both arms carry the root-placement contamination of section 4 equally -- same
    denominator -- so the gap between them is the origin and nothing else.
    """
    legacy = _round_trip(make_pass1=_make_legacy())
    assert max(legacy[n] for n in ARM_LANDMARKS) > 5.0, legacy
    # and it does not touch the legs either -- the legs never read an origin
    for name in LEG_LANDMARKS:
        assert legacy[name] == pytest.approx(0.0, abs=EXACT_MM), name


def test_no_scalar_on_the_torso_axis_reaches_the_band():
    """No gate a constant can pass: the true origin is 110 mm OFF the torso axis."""
    best = min(
        max(_round_trip(make_pass1=_make_legacy(scalar))[n] for n in ARM_LANDMARKS)
        for scalar in (0.40, 0.55, 0.72, 0.85, 1.00, 1.20))
    assert best > 5.0, best


# --------------------------------------------------------------------------- 3. blast radius
def test_only_the_clavicle_chain_and_the_two_hands_can_change():
    """Everything outside the clavicle chain is bit-identical under the origin swap.

    The fingers are named explicitly: their locals are `_finger_rest_local(name)`, a
    constant per joint name, so they CANNOT move however far the arm turns -- only their
    world rotations do. A test that let the fingers into the differing set would be
    asserting nothing.
    """
    root, local = _pose(frames=14)
    fk0 = forward_kinematics_positions(root, local, skeleton=DETAILED_HUMANOID).astype(np.float64)
    source = _to_z_up(_landmarks_from_fk(fk0))
    fixed = _solve(source)
    real = cm._joint_origin
    cm._joint_origin = _legacy_anchor(_from_z_up(source))
    try:
        legacy = _solve(source)
    finally:
        cm._joint_origin = real

    assert np.array_equal(np.asarray(fixed.root_translation_m),
                          np.asarray(legacy.root_translation_m))
    a = np.asarray(fixed.local_rotations_xyzw)
    b = np.asarray(legacy.local_rotations_xyzw)
    differ = {joint.name for index, joint in enumerate(DETAILED_HUMANOID.joints)
              if not np.array_equal(a[:, index], b[:, index])}
    chain = {"LeftShoulder", "LeftUpperArm", "LeftLowerArm",
             "RightShoulder", "RightUpperArm", "RightLowerArm"}
    assert chain <= differ, sorted(chain - differ)
    assert differ <= chain | {"LeftHand", "RightHand"}, sorted(differ - chain)
    fingers = tuple(j.name for j in DETAILED_HUMANOID.joints if j.name.startswith(
        ("LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing", "LeftLittle",
         "RightThumb", "RightIndex", "RightMiddle", "RightRing", "RightLittle")))
    assert fingers, "the detailed rig is supposed to have fingers"
    for name in fingers + ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
                           "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
                           "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"):
        index = DETAILED_HUMANOID.index(name)
        assert np.array_equal(a[:, index], b[:, index]), name


def test_no_constant_arrived_with_the_fix():
    """The 0.72 anchor leaves and nothing replaces it, in the shipped source."""
    source = Path(cm.__file__).read_text(encoding="utf-8")
    body = source[source.index("def positions_to_body_track"):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    live = "\n".join(line for line in body.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "0.72 * torso_up" not in live
    assert "torso_up" in live          # still the torso axis for the frames that need it
    # D9b (2026-09-07): the aims that call `_joint_origin` were lifted verbatim out of
    # `positions_to_body_track` into two module-level helpers so the re-solve after the
    # foot-contact hoist runs the same code; the pin follows them.
    helpers = ""
    for name in ("def _aim_trunk_neck_and_clavicles", "def _aim_arms_and_hands"):
        start = source.index(name)
        chunk = source[start:]
        chunk = chunk[:chunk.index("\ndef ", 1)] if "\ndef " in chunk[1:] else chunk
        helpers += "\n".join(line for line in chunk.splitlines()
                              if not line.lstrip().startswith("#"))
    assert "_joint_origin(" in helpers
    module_live = "\n".join(line for line in source.splitlines()
                            if not line.lstrip().startswith("#"))
    assert "0.72 * torso_up" not in module_live
