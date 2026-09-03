"""D2b: the rig's hip joints are placed on the captured hips.

`positions_to_body_track` used to write ``root_translation = pelvis - rest["Hips"]``, which
puts the rig's **`Hips` joint** on the captured hip-landmark midpoint. The rig's leg roots
-- which is what a ``left_hip`` landmark actually IS, the femoral joint centre -- hang
``rest["LeftUpperLeg"] = (0.09, -0.08, 0)`` below and beside it, so the whole skeleton sat
80 mm low on its own hips, on every frame, and the ground projection then hoisted it
~140 mm to keep it out of the floor. D2b subtracts ``_leg_root_offset(hips_world, rest)``
= ``R_hips . mid(rest[LeftUpperLeg], rest[RightUpperLeg])`` so FK's UpperLeg midpoint lands
on the captured midpoint instead.

**No constant arrives.** ``mid`` is read from the caller's own ``rest`` dict; the 0.08 is
never written down, and the tests below pin that by reading it from the skeleton too.

WHAT THESE TESTS PIN:

  1. FK's UpperLeg midpoint IS the captured hip midpoint, canonical and sized;
  2. the synthetic canonical round trip now lands the arms within 1 mm and the legs at 0,
     and the zero-offset variant -- D2 alone, the same code path -- fails it;
  3. torso, head, leg, foot and toe LOCAL rotations are bit-identical between the fixed
     and the zero-offset variant;
  4. the world-vertical shortcut differs from the derivation on a tilted-pelvis frame;
  5. `autoanim_gnm` is imported from this worktree.

**A NOTE ON TEST 2 AND `tests/test_clavicle_origin.py`.** That file's
``test_the_roundtrip_residual_is_root_placement_not_the_clavicle`` asserts the *inverse* of
test 2 here -- that the shipped round trip exceeds 5 mm and only closes when the hip drop
is removed from the re-solve. That was true of D2 and is false of D2b, by design and as
that test's own docstring predicted ("Fix the root/hip placement convention first"). It is
deliberately NOT edited here; see the D2b report.
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
# Bound at import: every variant below swaps `cm._leg_root_offset`, and one that reached
# for it through the module would call itself.
SHIPPED_OFFSET = cm._leg_root_offset
ARM_LANDMARKS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist")
LEG_LANDMARKS = ("left_hip", "right_hip", "left_knee", "right_knee",
                 "left_ankle", "right_ankle")
# "exactly 0.00 mm" in the only sense available: `forward_kinematics_positions` returns
# float32, so a metre-scale coordinate is quantised at ~1e-4 mm.
EXACT_MM = 1e-3
# The band for test 1. FK's own float32 output is the floor, not slack: over the
# Root->Hips->UpperLeg chain a micron is roughly ten times its quantisation.
FK_BAND_M = 1e-6
RIG_FOR = {
    "neck": "Neck",
    "left_shoulder": "LeftUpperArm", "right_shoulder": "RightUpperArm",
    "left_elbow": "LeftLowerArm", "right_elbow": "RightLowerArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_hip": "LeftUpperLeg", "right_hip": "RightUpperLeg",
    "left_knee": "LeftLowerLeg", "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
}
UNTOUCHED = ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftEye", "RightEye",
             "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
             "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes")


# --------------------------------------------------------------------------- 5. the trap
def test_autoanim_gnm_is_imported_from_this_worktree():
    """The venv's editable install points at the MAIN tree's `src`, and the failure is
    silent: every measurement in this step would be about another checkout's converter.
    `PYTHONPATH=$PWD/src` is what makes it right."""
    import autoanim_gnm

    resolved = Path(autoanim_gnm.__file__).resolve()
    root = Path(__file__).resolve().parents[1]
    assert str(resolved).startswith(str(root)), (
        f"autoanim_gnm resolved to {resolved}, outside this worktree ({root}). "
        f"Run pytest with PYTHONPATH=$PWD/src."
    )


# --------------------------------------------------------------------------- the variants
def zero_offset(hips_world, rest):
    """D2 alone: `Hips` back on the captured hip midpoint. The must-fail control."""
    return np.zeros(3)


def world_vertical(metres):
    """The plausible shortcut: the same 80 mm applied as a WORLD vertical.

    It agrees with the derivation exactly while the pelvis is upright and diverges as the
    subject leans, which is what test 4 measures. A reader who accepts "lift the rig 80 mm"
    without asking *in which frame* writes this.
    """
    def helper(hips_world, rest):
        return np.asarray((0.0, -float(metres), 0.0))
    return helper


class offset(object):
    """`with offset(helper):` -- swap the module attribute the converter calls by name."""

    def __init__(self, helper):
        self.helper = helper

    def __enter__(self):
        self.saved = cm._leg_root_offset
        cm._leg_root_offset = self.helper

    def __exit__(self, *exc):
        cm._leg_root_offset = self.saved
        return False


# --------------------------------------------------------------------------- helpers
def _pose(frames: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Non-trivial local rotations for the canonical rig, and a moving root.

    `tests/test_clavicle_origin.py::_pose`, reproduced rather than imported: the two files
    must stay independently readable, and a joint moving under one of these names should
    break both. The Hips turn, which is what makes the world-vertical control in test 4
    separable from the derivation at all.
    """
    rng = np.random.default_rng(20260902)
    joints = len(DETAILED_HUMANOID.joints)
    local = np.zeros((frames, joints, 4))
    local[..., 3] = 1.0
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
    """`retarget_cost.landmarks_from_fk`, reproduced so the test needs no swap harness.

    It writes the rig's **UpperLeg origins** into the `left_hip` / `right_hip` slots, which
    is anatomically right -- those are the femoral joint centres. Before D2b the converter
    then put `Hips` on their midpoint, 80 mm higher, and the round trip could not close.
    """
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


# Deliberately scales the two UpperLeg offsets, which `tools/head/sized_skeleton.py` does
# NOT do in Y. That makes test 1 an actual test of "the offset is read from `rest`": a
# converter that had the canonical 0.08 written into it would pass on the canonical rig
# and fail here.
SIZING = {"LeftShoulder": 1.19, "RightShoulder": 1.19, "LeftUpperArm": 0.88,
          "RightUpperArm": 0.88, "LeftLowerArm": 1.07, "RightLowerArm": 1.07,
          "LeftHand": 0.93, "RightHand": 0.93, "Spine": 1.11, "Chest": 0.94,
          "LeftLowerLeg": 1.05, "RightLowerLeg": 1.05,
          "LeftUpperLeg": 1.35, "RightUpperLeg": 1.35, "Hips": 0.92}


def _synthetic_capture(skeleton=DETAILED_HUMANOID, frames: int = 14):
    """A landmark cloud with canonical proportions BY CONSTRUCTION, in capture Z-up."""
    root, local = _pose(frames)
    fk0 = forward_kinematics_positions(root, local, skeleton=skeleton).astype(np.float64)
    return _to_z_up(_landmarks_from_fk(fk0, skeleton))


# ------------------------------------------------------- 1. the placement, as a theorem
@pytest.mark.parametrize("label", ("canonical", "sized"))
def test_fk_upperleg_midpoint_lands_on_the_captured_hip_midpoint(label):
    """The step, stated as the identity it is, on a rig whose hip drop is NOT canonical.

    The comparison is against the track handed to `project_generated_foot_contacts`, not
    the delivered one: the projection adds its own vertical afterwards (~83 mm here), which
    is a separate, deliberate stage and is what the D2b gate's absolute-placement figures
    measure. The projection is WRAPPED to get at that track, never re-implemented.
    """
    skeleton = DETAILED_HUMANOID if label == "canonical" else _sized(DETAILED_HUMANOID, SIZING)
    source = _synthetic_capture(skeleton)
    points_y = _from_z_up(source)

    captured: list = []
    real = cm.project_generated_foot_contacts

    def wrapper(track, **kwargs):
        projected, diagnostics = real(track, **kwargs)
        captured.append(track)
        return projected, diagnostics

    cm.project_generated_foot_contacts = wrapper
    try:
        _solve(source, skeleton)
    finally:
        cm.project_generated_foot_contacts = real

    assert len(captured) == 1
    fk = _fk(captured[0], skeleton)
    leg_mid = 0.5 * (fk[:, skeleton.index("LeftUpperLeg")]
                     + fk[:, skeleton.index("RightUpperLeg")])
    hip_mid = 0.5 * (points_y[:, cm.JOINT_INDEX["left_hip"]]
                     + points_y[:, cm.JOINT_INDEX["right_hip"]])
    error = np.linalg.norm(leg_mid - hip_mid, axis=1)
    assert error.max() <= FK_BAND_M, (label, float(error.max()))

    # and the control on the identical code path: with the offset zeroed, the rig sits
    # exactly the skeleton's own hip drop low. Not a band -- an identity.
    with offset(zero_offset):
        captured.clear()
        cm.project_generated_foot_contacts = wrapper
        try:
            _solve(source, skeleton)
        finally:
            cm.project_generated_foot_contacts = real
    fk_zero = _fk(captured[0], skeleton)
    leg_mid_zero = 0.5 * (fk_zero[:, skeleton.index("LeftUpperLeg")]
                          + fk_zero[:, skeleton.index("RightUpperLeg")])
    drop = 0.5 * (np.asarray(skeleton.joints[skeleton.index("LeftUpperLeg")].rest_translation_m)
                  + np.asarray(skeleton.joints[skeleton.index("RightUpperLeg")].rest_translation_m))
    assert abs(np.linalg.norm(drop)) > 0.05, "the fixture must have a hip drop to detect"
    assert np.median(np.linalg.norm(leg_mid_zero - hip_mid, axis=1)) == pytest.approx(
        float(np.linalg.norm(drop)), abs=1e-4)


# ------------------------------------------------------------------- 2. the round trip
def _round_trip(skeleton=None, variant=None):
    """solve -> FK -> landmarks -> solve, with the SAME offset variant on both passes."""
    skel = skeleton or DETAILED_HUMANOID
    source = _synthetic_capture(skel)
    with offset(variant if variant is not None else SHIPPED_OFFSET):
        fk1 = _fk(_solve(source, skeleton), skel)
        synth = _landmarks_from_fk(fk1, skel)
        fk2 = _fk(_solve(_to_z_up(synth), skeleton), skel)
    return _score(fk2, synth, skel)


def test_the_canonical_round_trip_now_lands_the_arms_within_a_millimetre():
    """D2's own pre-registered band, met once its false premise is removed.

    D2 measured 67.25 / 79.32 mm here and attributed 100 % of it to root placement with a
    control that took the rig's hip drop out of the re-solve. This is that attribution
    shipped: the arms close, and the legs and the neck stay exactly where they were.
    """
    err = _round_trip()
    assert max(err[n] for n in ARM_LANDMARKS) <= 1.0, err
    for name in LEG_LANDMARKS + ("neck",):
        assert err[name] == pytest.approx(0.0, abs=EXACT_MM), (name, err[name])


def test_the_zero_offset_variant_fails_the_same_round_trip():
    """MUST FAIL, on the identical code path, with only the offset helper swapped.

    This is D2 alone -- the converter as it stood one commit ago. Both arms carry the same
    clavicle origin, the same landmarks and the same score, so the gap between them is the
    root placement and nothing else. Same denominator.
    """
    err = _round_trip(variant=zero_offset)
    assert max(err[n] for n in ARM_LANDMARKS) > 5.0, err
    # and it does not touch the legs, which is exactly why the legs were never evidence:
    # a translation-invariant chain scored root-relatively cannot see a translation.
    for name in LEG_LANDMARKS:
        assert err[name] == pytest.approx(0.0, abs=EXACT_MM), name


def test_the_sized_round_trip_closes_too():
    """The offset is read from `rest`, so a rig with a different hip drop closes as well.

    A converter carrying the canonical 0.08 as a literal would pass the canonical test
    above and fail this one; `SIZING` scales the UpperLeg offsets by 1.35 for that reason.
    """
    err = _round_trip(skeleton=_sized(DETAILED_HUMANOID, SIZING))
    assert max(err[n] for n in ARM_LANDMARKS) <= 1.0, err
    for name in LEG_LANDMARKS + ("neck",):
        assert err[name] == pytest.approx(0.0, abs=EXACT_MM), (name, err[name])


# ------------------------------------------------------------------- 3. the blast radius
def test_only_the_clavicle_chain_and_the_hands_move_before_the_projection():
    """Everything else is bit-identical between the fixed and the zero-offset variant.

    PRE-PROJECTION, and the distinction is load-bearing: on the DELIVERED track the FOOT
    locals differ too, because `project_generated_foot_contacts` rewrites them inside
    contact runs and the runs move when the root does. The facing instrument still cannot
    move under this change -- it reads the Hips, chest, Neck and Head rotations and the
    mesh nose, not one of which is a foot.
    """
    source = _synthetic_capture()
    captured: list = []
    real = cm.project_generated_foot_contacts

    def wrapper(track, **kwargs):
        projected, diagnostics = real(track, **kwargs)
        captured.append(track)
        return projected, diagnostics

    cm.project_generated_foot_contacts = wrapper
    try:
        _solve(source)
        with offset(zero_offset):
            _solve(source)
    finally:
        cm.project_generated_foot_contacts = real

    fixed, zero = captured
    a = np.asarray(fixed.local_rotations_xyzw)
    b = np.asarray(zero.local_rotations_xyzw)
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
    for name in fingers + UNTOUCHED:
        index = DETAILED_HUMANOID.index(name)
        assert np.array_equal(a[:, index], b[:, index]), name

    # the root itself moved, and by exactly the skeleton's own rotated hip drop
    rest = {j.name: np.asarray(j.rest_translation_m, dtype=np.float64)
            for j in DETAILED_HUMANOID.joints}
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    hips = np.asarray(zero.local_rotations_xyzw, dtype=np.float64)[
        :, DETAILED_HUMANOID.index("Hips")]
    predicted = -_rotate_vector(hips, np.broadcast_to(mid, (len(hips), 3)))
    moved = (np.asarray(fixed.root_translation_m, dtype=np.float64)
             - np.asarray(zero.root_translation_m, dtype=np.float64))
    assert np.abs(moved - predicted).max() <= 1e-6, float(np.abs(moved - predicted).max())


# ------------------------------------------------- 4. the world-vertical shortcut fails
def test_a_world_vertical_lift_is_not_the_derivation_on_a_tilted_pelvis():
    """The control that discriminates the derivation from a number.

    A world-vertical lift of the same 80 mm is *identical* to the derivation while the
    pelvis is upright; it diverges as the subject leans, by `2*sin(tilt/2)*80 mm`. So the
    test asserts both halves: agreement at zero tilt would make the control vacuous, and
    a divergence that did not track the tilt would mean the two differ for some other
    reason. Measured on the pre-projection root, where the term is unmixed with the
    projection's own vertical.
    """
    source = _synthetic_capture()
    rest = {j.name: np.asarray(j.rest_translation_m, dtype=np.float64)
            for j in DETAILED_HUMANOID.joints}
    lift = float(-0.5 * (rest["LeftUpperLeg"][1] + rest["RightUpperLeg"][1]))
    assert lift == pytest.approx(0.08, abs=1e-9)

    captured: list = []
    real = cm.project_generated_foot_contacts

    def wrapper(track, **kwargs):
        projected, diagnostics = real(track, **kwargs)
        captured.append(track)
        return projected, diagnostics

    cm.project_generated_foot_contacts = wrapper
    try:
        _solve(source)
        with offset(world_vertical(lift)):
            _solve(source)
    finally:
        cm.project_generated_foot_contacts = real

    derived, shortcut = captured
    delta = np.linalg.norm(
        np.asarray(derived.root_translation_m, dtype=np.float64)
        - np.asarray(shortcut.root_translation_m, dtype=np.float64), axis=1)

    hips = np.asarray(derived.local_rotations_xyzw, dtype=np.float64)[
        :, DETAILED_HUMANOID.index("Hips")]
    up = _rotate_vector(hips, np.broadcast_to(np.asarray((0.0, 1.0, 0.0)), (len(hips), 3)))
    tilt = np.arccos(np.clip(up[:, 1], -1.0, 1.0))
    predicted = 2.0 * np.sin(tilt / 2.0) * lift

    assert tilt.max() > np.radians(5.0), "the fixture must tilt the pelvis at all"
    assert delta.max() > 1e-3, ("the shortcut is indistinguishable from the derivation on "
                                "this fixture, so the control proves nothing", float(delta.max()))
    assert np.abs(delta - predicted).max() <= 1e-6, float(np.abs(delta - predicted).max())
    # and it is not a constant offset: it tracks the tilt, which is the whole argument
    assert np.corrcoef(delta, tilt)[0, 1] > 0.99


def test_no_constant_arrived_with_the_root_fix():
    """The offset is the skeleton's own rest geometry; 0.08 is never written down.

    Parsed rather than grepped: the helper's docstring and the converter's NOTE both say
    "0.08" in prose, deliberately, and a substring search over the raw source would score
    the comments instead of the code. `ast` is what separates the two.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(cm._leg_root_offset))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    if (function.body and isinstance(function.body[0], ast.Expr)
            and isinstance(function.body[0].value, ast.Constant)
            and isinstance(function.body[0].value.value, str)):
        function.body = function.body[1:]          # drop the docstring
    live = ast.unparse(ast.Module(body=function.body, type_ignores=[]))
    for literal in ast.walk(ast.parse(live)):
        if isinstance(literal, ast.Constant) and isinstance(literal.value, (int, float)):
            assert literal.value in (0, 0.5), (
                f"a numeric literal {literal.value!r} appears in the shipped offset; the "
                f"only ones allowed are the 0.5 of a midpoint and an index")
    assert 'rest[\'LeftUpperLeg\']' in live and 'rest[\'RightUpperLeg\']' in live, live
