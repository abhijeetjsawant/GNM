"""D9b: the foot-contact hoist re-aim, and the properties that make it safe.

`positions_to_body_track` ends by calling `project_generated_foot_contacts`, which
TRANSLATES the root -- a per-run correction that plants a slow, low foot, plus a whole-take
lift by the deepest penetration. A translation leaves every rotation alone, so before this
step every root-dependent bone still pointed along `target - origin_BEFORE_the_hoist` while
its origin had moved. The fix is the D2c standing rule -- a placed parent, then the chains
below it re-solved -- with the root as the parent.

Six things are asserted here, and three of them no positional score in this lane can see:

  * every root-dependent bone lies on its ray FROM THE DELIVERED ORIGIN, and misses from
    the pre-hoist origin. Both halves: the first says the fix landed, the second says the
    fixture actually exercises it and the bones are not merely unmoved;
  * the projection's output is READ-ONLY afterwards -- the root, the contacts and the foot,
    toe, leg, `Hips` and `Root` rotations come back bit-identical, which is what keeps the
    1e-5 m contact lock true and what a re-run of pass C would break;
  * with the hoist forced to zero the whole solve is bit-identical, so the re-solve does
    not reorder a float operation on a frame it should not touch (the D7b
    seven-normalisations lesson);
  * the projection runs exactly ONCE -- planted feet read zero speed on the runs they were
    planted on, so a second pass is not a fixed point;
  * pass B is rerun over the whole take, because reachability is a property of a sequence
    and a re-aimed clavicle is a different sequence;
  * each rewritten local agrees in SIGN with the projection's own local on its own frame,
    which is what replaces the global quaternion hemisphere walk. Re-running that walk
    could flip the bits of an unhoisted frame that follows a hoisted one.

IT ALSO RE-PINS TWO CLAUSES THIS STEP MOVED, in the file that owns them rather than by
editing the file that pinned them:

  * `test_clavicle_temporal.py` asserts `_reachable_clavicle_sequence` is called exactly
    TWICE. It is now called four times: pass B, then pass B again over the re-aimed
    clavicles. The property that test guards -- nothing rejected on reachable motion -- is
    asserted here on all four calls.
  * `test_clavicle_origin.py` asserts the literal text `_joint_origin(` inside
    `positions_to_body_track`'s own source slice. The aims that call it now live in two
    module-level helpers, and the guarded property -- the clavicle aimed from its own
    origin, the 0.72 anchor gone and nothing replacing it -- is asserted here against those.

docs/reviews/hoist-reaim-2026-09-07.md.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions
from autoanim_gnm.commercial_multiview import JOINT_INDEX, positions_to_body_track

from test_clavicle_temporal import _breathing_take

SAMPLE_RATE_HZ = 30
PROVENANCE = "0" * 64
FRAMES = 40
# 3 mm per frame at 30 fps is 0.09 m/s, comfortably inside the capture path's own
# 0.30 m/s contact envelope, so the feet stay ELIGIBLE while the body travels. That is
# what makes the per-run root correction fire; a static fixture would hoist by nothing and
# every assertion below would pass vacuously, which is why `test_the_fixture_hoists`
# exists and runs first.
DRIFT_M_PER_FRAME = 0.003

# (the joint whose rotation aims the bone, the child on the ray, the landmark it aims at)
RAYS = (
    ("LeftUpperArm", "LeftLowerArm", "left_elbow"),
    ("LeftLowerArm", "LeftHand", "left_wrist"),
    ("RightUpperArm", "RightLowerArm", "right_elbow"),
    ("RightLowerArm", "RightHand", "right_wrist"),
    ("LeftShoulder", "LeftUpperArm", "left_shoulder"),
    ("RightShoulder", "RightUpperArm", "right_shoulder"),
    ("Spine", "Neck", "neck"),
)
FROZEN = ("Root", "Hips", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
          "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes")


# ------------------------------------------------------------------------- the fixture
def _travelling_take(frames: int = FRAMES):
    """A reachable take that TRAVELS, so the contact projection plants a foot and hoists.

    Landmarks in capture Z-up metres, plus the ball-of-foot and `Spine1` positions the
    delivery supplies -- the toes so the feet are solved from the toes as the delivery's
    are, and the spine so the D7b pelvis frame runs and the trunk aim is root-DEPENDENT.
    Without the spine the trunk takes the legacy landmark-only line and the `Spine -> neck`
    ray could not see this defect at all.
    """
    take = _breathing_take(frames)
    drift = np.arange(frames)[:, None] * np.asarray([DRIFT_M_PER_FRAME, 0.0, 0.0])
    take = take + drift[:, None, :]
    pelvis = 0.5 * (take[:, JOINT_INDEX["left_hip"]] + take[:, JOINT_INDEX["right_hip"]])
    neck = take[:, JOINT_INDEX["neck"]]
    spine = pelvis + 0.25 * (neck - pelvis)
    ball = np.asarray([0.0, 0.14, -0.06])
    toes = np.stack([take[:, JOINT_INDEX["left_ankle"]] + ball,
                     take[:, JOINT_INDEX["right_ankle"]] + ball], axis=1)
    return take, toes, spine


def _solve(zero_hoist: bool = False):
    """The real converter, with WATCHERS on the projection and the clavicle reject.

    Returns `(track, records)`. `zero_hoist` replaces the root the projection returns with
    the root that went in and clears `foot_contacts` -- the refactor tripwire's arm. The
    contacts must be cleared because `validate_body_track` asserts a contact-locked foot
    has not moved to within 1e-5 m, and a foot locked but not translated has.
    """
    take, toes, spine = _travelling_take()
    records: dict = {"projection": [], "reject": []}
    real_projection = cm.project_generated_foot_contacts
    real_reject = cm._reachable_clavicle_sequence

    def projection(track, **kwargs):
        projected, diagnostics = real_projection(track, **kwargs)
        if zero_hoist:
            from dataclasses import replace

            projected = replace(
                projected,
                root_translation_m=np.array(track.root_translation_m, copy=True),
                foot_contacts=np.zeros_like(np.asarray(projected.foot_contacts)))
        records["projection"].append({"pre": track, "post": projected,
                                      "diagnostics": diagnostics})
        return projected, diagnostics

    def reject(local_rotations, parent_world_rotations, ceiling):
        replaced, accepted = real_reject(
            local_rotations, parent_world_rotations, ceiling)
        records["reject"].append({"local": np.array(local_rotations, copy=True),
                                  "accepted": accepted})
        return replaced, accepted

    cm.project_generated_foot_contacts = projection
    cm._reachable_clavicle_sequence = reject
    try:
        track = positions_to_body_track(
            take, sample_rate_hz=SAMPLE_RATE_HZ, provenance_sha256=PROVENANCE,
            toe_world_z_up_m=toes, spine_world_z_up_m=spine)
    finally:
        cm.project_generated_foot_contacts = real_projection
        cm._reachable_clavicle_sequence = real_reject
    records["landmarks"] = take
    return track, records


@pytest.fixture(scope="module")
def solved():
    return _solve()


def _hoist(records) -> np.ndarray:
    record = records["projection"][0]
    return (np.asarray(record["post"].root_translation_m, dtype=np.float64)
            - np.asarray(record["pre"].root_translation_m, dtype=np.float64))


def _fk(track) -> np.ndarray:
    """Delivered joint positions in the RIG's Y-up world, from the track's own arrays."""
    return forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=DETAILED_HUMANOID).astype(np.float64)


def _rig(points_z_up: np.ndarray) -> np.ndarray:
    """The converter's own change of basis: capture (x, y, z) -> rig (x, z, -y)."""
    out = points_z_up[..., (0, 2, 1)].copy()
    out[..., 2] *= -1.0
    return out


def _perpendicular_miss(fk, landmarks, parent, child, name, origin_shift=None):
    origin = fk[:, DETAILED_HUMANOID.index(parent)]
    if origin_shift is not None:
        origin = origin - origin_shift
    axis = fk[:, DETAILED_HUMANOID.index(child)] - fk[:, DETAILED_HUMANOID.index(parent)]
    axis = axis / np.linalg.norm(axis, axis=1)[:, None]
    delta = landmarks[:, JOINT_INDEX[name]] - origin
    return np.linalg.norm(delta - np.sum(delta * axis, axis=1)[:, None] * axis, axis=1)


# --------------------------------------------------------------------------- the tests
def test_the_module_under_test_is_this_worktree():
    """The venv's editable install points at the MAIN checkout; a green run there proves
    nothing about this branch."""
    root = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(root)), (
        f"autoanim_gnm resolved to {autoanim_gnm.__file__}, not {root}: re-run with "
        f"PYTHONPATH=$PWD/src")


def test_the_fixture_hoists(solved):
    """NO ASSERTION BELOW MEANS ANYTHING IF THE ROOT DOES NOT MOVE. This runs first."""
    _track, records = solved
    assert len(records["projection"]) == 1
    contacts = records["projection"][0]["diagnostics"].contact_frames
    assert sum(contacts) > 0, f"the fixture asserts no foot contact at all: {contacts}"
    magnitude = 1000.0 * np.linalg.norm(_hoist(records), axis=1)
    assert magnitude.max() > 5.0, (
        f"the fixture's root moves at most {magnitude.max():.3f} mm; every ray assertion "
        f"below would pass on a build that did nothing")
    assert (magnitude < 1e-3).sum() > 0, "the fixture has no unhoisted frame to compare"


def test_every_root_dependent_bone_lies_on_its_ray_from_the_DELIVERED_origin(solved):
    """The fix. A bone must point at its landmark from where the delivered file puts it."""
    track, records = solved
    fk = _fk(track)
    landmarks = _rig(records["landmarks"])
    for parent, child, name in RAYS:
        miss = _perpendicular_miss(fk, landmarks, parent, child, name)
        assert miss.max() < 1.0e-5, (
            f"{parent} -> {name} misses its ray by {1000.0 * miss.max():.4f} mm from the "
            f"origin the delivered file gives it")


def test_and_MISSES_from_the_pre_hoist_origin(solved):
    """The dual, and the reason the test above is not vacuous.

    A build that never re-aimed would read zero from the PRE-hoist origin and several
    millimetres from the delivered one. This asserts the opposite arrangement, so the two
    tests together can only be satisfied by a body aimed from the hoisted root.
    """
    track, records = solved
    fk = _fk(track)
    landmarks = _rig(records["landmarks"])
    hoist = _hoist(records)
    hoisted = 1000.0 * np.linalg.norm(hoist, axis=1) > 0.5
    for parent, child, name in RAYS:
        miss = _perpendicular_miss(fk, landmarks, parent, child, name,
                                   origin_shift=hoist)
        assert miss[hoisted].max() > 1.0e-4, (
            f"{parent} -> {name} lies on its ray from the PRE-hoist origin too, which "
            f"means the root did not move for this bone and nothing here is tested")


def test_the_projections_output_is_read_only_afterwards(solved):
    """Root, contacts and every frozen joint come back bit-identical.

    This is what a re-run of pass C would break: it rewrites the legs, the feet and the
    toes, overwriting `candidate_local`, and the delivered file would then carry a
    translated root with an unplanted foot.
    """
    track, records = solved
    projected = records["projection"][0]["post"]
    assert np.array_equal(np.asarray(track.root_translation_m),
                          np.asarray(projected.root_translation_m))
    assert np.array_equal(np.asarray(track.foot_contacts),
                          np.asarray(projected.foot_contacts))
    delivered = np.asarray(track.local_rotations_xyzw)
    before = np.asarray(projected.local_rotations_xyzw)
    for name in FROZEN:
        index = DETAILED_HUMANOID.index(name)
        assert np.array_equal(delivered[:, index], before[:, index]), name
    fingers = [i for i, joint in enumerate(DETAILED_HUMANOID.joints)
               if joint.name.startswith(("LeftThumb", "LeftIndex", "LeftMiddle",
                                         "LeftRing", "LeftLittle", "RightThumb",
                                         "RightIndex", "RightMiddle", "RightRing",
                                         "RightLittle"))]
    assert fingers, "the detailed rig is supposed to have fingers"
    for index in fingers:
        assert np.array_equal(delivered[:, index], before[:, index])


def test_the_moved_joints_are_exactly_the_declared_set(solved):
    """Nothing outside `_ROOT_DEPENDENT_JOINTS` moves, and the frozen list is disjoint."""
    track, records = solved
    delivered = np.asarray(track.local_rotations_xyzw)
    before = np.asarray(records["projection"][0]["post"].local_rotations_xyzw)
    moved = {DETAILED_HUMANOID.joints[index].name
             for index in range(delivered.shape[1])
             if not np.array_equal(delivered[:, index], before[:, index])}
    assert moved, "nothing moved at all: the re-solve did not run"
    assert moved <= set(cm._ROOT_DEPENDENT_JOINTS), sorted(moved - set(cm._ROOT_DEPENDENT_JOINTS))
    assert not set(FROZEN) & set(cm._ROOT_DEPENDENT_JOINTS)


def test_the_unhoisted_frames_are_bit_identical(solved):
    """A frame the projection did not move must come out of the re-solve unchanged."""
    track, records = solved
    magnitude = 1000.0 * np.linalg.norm(_hoist(records), axis=1)
    still = magnitude < 1e-3
    assert still.sum() > 0
    delivered = np.asarray(track.local_rotations_xyzw)
    before = np.asarray(records["projection"][0]["post"].local_rotations_xyzw)
    assert np.array_equal(delivered[still], before[still])


def test_the_refactor_tripwire_with_the_hoist_forced_to_zero():
    """The D7b seven-normalisations lesson, in miniature.

    Pass A and the re-solve share `_aim_trunk_neck_and_clavicles`, so a refactor that
    changed that helper's arithmetic would move BOTH and no comparison inside one build
    could see it. With the hoist forced to zero the re-solve must reproduce the
    projection's own output BIT FOR BIT -- rotations, root and contacts -- because with no
    root move the same float operations run on the same inputs in the same order.
    """
    track, records = _solve(zero_hoist=True)
    projected = records["projection"][0]["post"]
    assert np.array_equal(np.asarray(track.local_rotations_xyzw),
                          np.asarray(projected.local_rotations_xyzw))
    assert np.array_equal(np.asarray(track.root_translation_m),
                          np.asarray(projected.root_translation_m))
    assert np.array_equal(np.asarray(track.foot_contacts),
                          np.asarray(projected.foot_contacts))


def test_the_projection_runs_exactly_once(solved):
    """Planted feet read zero speed on the runs they were planted on and the floor
    percentile moves elsewhere, so a second pass is not a fixed point."""
    _track, records = solved
    assert len(records["projection"]) == 1


def test_pass_B_is_rerun_over_the_whole_take_and_stays_inert(solved):
    """The re-pin of `test_clavicle_temporal`'s call-count clause, and its real property.

    Reachability is a property of a SEQUENCE. A re-aimed clavicle is a different sequence,
    so the rule is evaluated again over the whole take rather than left on the locals it
    scored before the root moved. Four calls: left, right, then left and right again. On
    reachable motion every one of them must accept every frame -- that is the clause the
    older test carries and it is asserted here on all four.
    """
    _track, records = solved
    assert len(records["reject"]) == 4, (
        "two clavicles in pass B and two in the re-solve's rerun")
    for call in records["reject"]:
        assert bool(call["accepted"].all()), (
            f"the ceiling fired on reachable motion: "
            f"{int((~call['accepted']).sum())} frames rejected")
    # And the rerun genuinely saw different locals: the clavicles were re-aimed first.
    assert not np.array_equal(records["reject"][0]["local"], records["reject"][2]["local"]), (
        "the second pass saw the same clavicle locals as the first, so either the "
        "clavicles were not re-aimed or pass B ran on stale rotations")


def test_each_rewritten_local_agrees_in_sign_with_the_projections_own(solved):
    """What replaces the global hemisphere walk.

    `local` has already been walked from frame 0 to keep the quaternion path continuous.
    Walking it AGAIN over rewritten frames could flip the bits of an unhoisted frame that
    follows a hoisted one -- the D7b lesson across time. Each rewritten local is instead
    signed against the DELIVERED local of its OWN frame, which cannot reach across frames.
    """
    track, records = solved
    delivered = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    before = np.asarray(records["projection"][0]["post"].local_rotations_xyzw,
                        dtype=np.float64)
    for name in cm._ROOT_DEPENDENT_JOINTS:
        index = DETAILED_HUMANOID.index(name)
        dots = np.sum(delivered[:, index] * before[:, index], axis=1)
        assert (dots >= 0.0).all(), (
            f"{name} flipped hemisphere against the projection's own local on "
            f"{int((dots < 0.0).sum())} frames")


def test_no_constant_arrived_with_the_hoist_re_aim():
    """The re-pin of `test_clavicle_origin.py::test_no_constant_arrived_with_the_fix`.

    That test slices `positions_to_body_track`'s own source and looks for the literal
    `_joint_origin(`. The aims that call it now live in the two helpers, so the guarded
    property is asserted against those instead -- and the 0.72 anchor is asserted gone from
    the WHOLE module, which is stronger than the slice.
    """
    for helper in (cm._aim_trunk_neck_and_clavicles, cm._aim_arms_and_hands):
        live = "\n".join(line for line in inspect.getsource(helper).splitlines()
                         if not line.lstrip().startswith("#"))
        assert "_joint_origin(" in live, helper.__name__
    module = Path(cm.__file__).read_text(encoding="utf-8")
    assert "0.72 * torso_up" not in "\n".join(
        line for line in module.splitlines() if not line.lstrip().startswith("#"))
    # And the re-solve reaches the helpers by BARE MODULE NAME, so an instrument that
    # substitutes `_joint_origin` or `_reachable_clavicle_sequence` runs its controls
    # through the identical code path.
    body = inspect.getsource(positions_to_body_track)
    assert "_aim_trunk_neck_and_clavicles(" in body
    assert "_aim_arms_and_hands(" in body
    assert "_reachable_clavicle_sequence(" in body
