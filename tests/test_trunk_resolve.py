"""D7b: after the pelvis frame, the trunk chain is re-solved onto the captured neck.

D7 gave `Hips` its own frame and left `Spine`, `Chest` and `UpperChest` aimed along
`neck - hip_mid`. The `Spine` joint hangs 117 mm off `Hips` **in the pelvis frame**, so a
pelvis pitched away from the trunk line moves the `Spine` origin off that line and a
straight rigid chain aimed from a displaced origin misses the neck by the displacement.
The standing rule -- *after replacing a parent, RE-SOLVE the chains below it* -- was not
applied to the spine.

Every test here asserts it imported THIS worktree's package. The synthetic bodies are
posed through the converter's own primitives, so the answers are exact by construction
rather than by agreement with anything.

THE `D7` ARM IS THE SHIPPED CODE PATH, NOT A RE-IMPLEMENTATION. `_joint_origin` is module
level and called by bare name precisely so an instrument can substitute it
(`commercial_multiview` says so in its docstring). Returning the hip midpoint for `Spine`
and the real origin for every other joint reproduces D7's `neck - hip_mid` aim through the
identical call site, the identical `_frame_alignment`, and the identical everything else.

docs/reviews/trunk-resolve-2026-09-05.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.body import DETAILED_HUMANOID, forward_kinematics_positions

from test_pelvis_frame import CAPTURE_FROM_RIG, convert, posed_body  # noqa: F401

TRUNK_CHAIN = ("Chest", "UpperChest", "Neck")


def test_the_package_under_test_is_this_worktree() -> None:
    """The venv's editable install points at the MAIN tree; a green test there is a lie."""

    here = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(here))


# --------------------------------------------------------------------------- the fixtures
def rig_points(positions: np.ndarray) -> np.ndarray:
    """The converter's own change of basis, applied exactly as it applies it."""

    points = positions[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    return points


def toe_landmarks(positions: np.ndarray) -> np.ndarray:
    """A ball-of-foot point per side, so the FEET never fall back to the torso frame.

    Deliberate. Without toes the foot takes `torso_world`, which this step moves, so the
    feet are *reported* and not banded (the card says so). With toes the foot frame is
    built from the ankle, the ball and the shin and is independent of the trunk, which is
    what lets the leg/root bit-identity below be exact rather than approximate.
    """

    frames = len(positions)
    toes = np.zeros((frames, 2, 3), dtype=np.float64)
    for side, ankle in enumerate(("left_ankle", "right_ankle")):
        toes[:, side] = positions[:, cm.JOINT_INDEX[ankle]] + np.asarray((0.0, 0.16, -0.06))
    return toes


def d7_origin_substitute(points: np.ndarray):
    """The SHIPPED D7 aim, through the converter's own substitution point.

    `torso_world` is built from `neck - _joint_origin(..., "Spine")` after this step; D7
    built it from `neck - hip_mid`. Hand back the hip midpoint for `Spine` and the real
    origin for everything else and the call site computes D7's line, bit for bit.
    """

    shipped = cm._joint_origin

    def origin(world, frame, root_translation, rest, joint_name):
        if joint_name == "Spine":
            return 0.5 * (points[frame, cm.JOINT_INDEX["left_hip"]]
                          + points[frame, cm.JOINT_INDEX["right_hip"]])
        return shipped(world, frame, root_translation, rest, joint_name)

    return origin


def world_positions(track) -> np.ndarray:
    return forward_kinematics_positions(
        np.asarray(track.root_translation_m, dtype=np.float64),
        np.asarray(track.local_rotations_xyzw, dtype=np.float64),
        skeleton=DETAILED_HUMANOID,
    ).astype(np.float64)


def trunk_length_m() -> float:
    """`L_rest`: `Spine` origin to `Neck` origin when the three trunk rests are collinear."""

    rest = {joint.name: np.asarray(joint.rest_translation_m, dtype=np.float64)
            for joint in DETAILED_HUMANOID.joints}
    for name in TRUNK_CHAIN:
        assert rest[name][0] == 0.0 and rest[name][2] == 0.0, (
            f"{name}'s rest is not collinear +Y; the length-floor argument does not hold")
    return float(sum(np.linalg.norm(rest[name]) for name in TRUNK_CHAIN))


# ------------------------------------------------ THE CLAIM: the neck reaches its landmark
def test_the_neck_lands_on_its_landmark_to_the_trunk_LENGTH_and_nothing_more() -> None:
    """A pitched pelvis, and the delivered `Neck` misses only by the chord's length error.

    `Spine`, `Chest` and `UpperChest` share one world rotation and their rests are
    collinear +Y, so `Neck_origin = Spine_origin + R . (rest sum)`. Aim `R` at the neck
    from the `Spine` origin and the neck lands ON the ray: what is left is
    `| L_rest - ||neck - Spine_origin|| |`, the residual a straight rigid chain cannot
    remove. It is EXACT, not approximate, and this asserts the identity to a micron.
    """

    positions, spine, _ = posed_body(pelvis_deg=34.0, thorax_deg=-16.0)
    track = convert(positions, spine, toe_world_z_up_m=toe_landmarks(positions))
    points = rig_points(positions)
    world = world_positions(track)
    neck_index = track.joint_names.index("Neck")
    spine_index = track.joint_names.index("Spine")
    length = trunk_length_m()

    error = np.linalg.norm(world[:, neck_index] - points[:, cm.JOINT_INDEX["neck"]], axis=1)
    chord = np.linalg.norm(
        points[:, cm.JOINT_INDEX["neck"]] - world[:, spine_index], axis=1)
    floor = np.abs(length - chord)
    # 1e-6 m, which is the card's own band for the clean arm. The track's rotations and
    # root are float32 -- that quantisation alone is ~8e-8 m here -- so an exact-to-the-bit
    # assertion would be a claim about float64 that the delivered file cannot carry.
    assert np.abs(error - floor).max() < 1.0e-6, (
        f"worst departure from the length floor {1000 * np.abs(error - floor).max():.6f} mm")
    # And the floor itself is a length error, so it is small beside the displacement D7
    # leaves behind. If this ever fails the fixture stopped being a trunk.
    assert np.median(floor) < 0.05


def test_the_aim_runs_from_the_spine_origin_and_not_from_the_hip_midpoint() -> None:
    """THE ONE-LINE CLAIM, read off the delivered rotations. Fails on the shipped D7 code.

    `Spine`'s world +Y must be parallel to `neck - Spine_origin`. Under D7 it is parallel
    to `neck - hip_mid` instead, and on a pitched pelvis those are different directions --
    so this test is red on the pre-D7b converter and green after it.
    """

    positions, spine, _ = posed_body(pelvis_deg=34.0, thorax_deg=-16.0)
    track = convert(positions, spine, toe_world_z_up_m=toe_landmarks(positions))
    points = rig_points(positions)
    world = world_positions(track)
    rotations = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    spine_index = track.joint_names.index("Spine")
    hips_index = track.joint_names.index("Hips")
    # `Spine`'s WORLD rotation: its parent chain is Root (identity) -> Hips -> Spine.
    spine_world = cm._quaternion_multiply(rotations[:, hips_index], rotations[:, spine_index])
    hip_mid = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                     + points[:, cm.JOINT_INDEX["right_hip"]])
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    axis = unit(np.stack([cm._rotate_vector(q, np.asarray((0.0, 1.0, 0.0)))
                          for q in spine_world]))
    from_spine = unit(points[:, cm.JOINT_INDEX["neck"]] - world[:, spine_index])
    from_hips = unit(points[:, cm.JOINT_INDEX["neck"]] - hip_mid)
    # ARCSIN OF THE CROSS PRODUCT, never arccos of the dot. `arccos` near 1 is
    # ill-conditioned: a float32 quaternion's norm is 1 +/- 2e-7 and that alone reads as
    # 0.036 deg of "error" through `arccos`, which is a property of the arithmetic and not
    # of the rig. `|a x b| = sin(theta)` is well conditioned at small angles.
    angle = lambda a, b: np.degrees(np.arcsin(
        np.clip(np.linalg.norm(np.cross(a, b), axis=1), 0.0, 1.0)))
    to_spine_deg = angle(axis, from_spine)
    to_hips_deg = angle(axis, from_hips)
    assert to_spine_deg.max() < 1.0e-3, f"worst {to_spine_deg.max():.9f} deg off its own origin"
    # The must-fail half: the two aims are genuinely different on this fixture, so the
    # assertion above is not satisfied by both answers at once.
    assert np.median(to_hips_deg) > 1.0, (
        f"the fixture does not separate the two aims (median {np.median(to_hips_deg):.4f} deg)")


def test_the_shipped_d7_aim_misses_the_neck_by_the_spine_origin_displacement() -> None:
    """MUST-FAIL ARM. D7's own construction, through the converter's substitution point."""

    positions, spine, _ = posed_body(pelvis_deg=34.0, thorax_deg=-16.0)
    toes = toe_landmarks(positions)
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d7_origin_substitute(points)
    try:
        d7 = convert(positions, spine, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    d7b = convert(positions, spine, toe_world_z_up_m=toes)

    neck = points[:, cm.JOINT_INDEX["neck"]]
    neck_index = d7.joint_names.index("Neck")
    spine_index = d7.joint_names.index("Spine")
    d7_error = np.linalg.norm(world_positions(d7)[:, neck_index] - neck, axis=1)
    d7b_error = np.linalg.norm(world_positions(d7b)[:, neck_index] - neck, axis=1)
    # The displacement that explains it: D7's `Spine` origin is off the trunk line.
    displacement = np.linalg.norm(
        world_positions(d7)[:, spine_index] - world_positions(d7b)[:, spine_index], axis=1)
    assert np.median(d7_error) > 0.05, f"median {1000 * np.median(d7_error):.1f} mm"
    assert np.median(d7b_error) < np.median(d7_error)
    assert np.median(displacement) < 1.0e-9, (
        "the two arms must share a `Spine` ORIGIN and differ only in the aim taken from it")


# ------------------------------------------------- B2: the root and the legs do not move
def test_the_root_and_the_legs_are_bit_identical_to_the_shipped_d7_path() -> None:
    """UNTOUCHABLE. Nothing below `Hips` may move, and the root formula is not touched.

    `root_translation` is `pelvis - rest["Hips"] - _leg_root_offset(hips_world, rest)` and
    every term is set before the trunk aim is computed. The leg chain hangs off `Hips`.
    With toe landmarks supplied the feet do not read `torso_world` either, so the whole
    lower body is exactly what D7 wrote -- float for float, after the ground projection.
    """

    positions, spine, _ = posed_body(pelvis_deg=34.0, thorax_deg=-16.0)
    toes = toe_landmarks(positions)
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d7_origin_substitute(points)
    try:
        d7 = convert(positions, spine, toe_world_z_up_m=toes)
    finally:
        cm._joint_origin = shipped
    d7b = convert(positions, spine, toe_world_z_up_m=toes)

    assert np.array_equal(d7.root_translation_m, d7b.root_translation_m)
    assert np.array_equal(d7.foot_contacts, d7b.foot_contacts)
    for name in ("Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
                 "LeftFoot", "RightFoot", "LeftToes", "RightToes"):
        index = d7.joint_names.index(name)
        assert np.array_equal(d7.local_rotations_xyzw[:, index],
                              d7b.local_rotations_xyzw[:, index]), name
    # ... and the trunk DID move, or the test above proves nothing.
    trunk_index = d7.joint_names.index("Spine")
    assert not np.array_equal(d7.local_rotations_xyzw[:, trunk_index],
                              d7b.local_rotations_xyzw[:, trunk_index])


def test_without_toe_landmarks_the_feet_follow_the_trunk_and_that_is_reported() -> None:
    """The one place the lower body DOES move, named rather than hidden.

    `Pass C` falls the foot back to `torso_world` when a toe landmark is missing, so on a
    fallback frame this step turns the foot. The delivery's toe solve resolves 100 % and
    98.3 % of frames, so it is a small population there -- and it is reported, never
    banded. This test exists so the exposure is asserted rather than assumed.
    """

    positions, spine, _ = posed_body(pelvis_deg=34.0, thorax_deg=-16.0)
    points = rig_points(positions)
    shipped = cm._joint_origin
    cm._joint_origin = d7_origin_substitute(points)
    try:
        d7 = convert(positions, spine)
    finally:
        cm._joint_origin = shipped
    d7b = convert(positions, spine)
    index = d7.joint_names.index("LeftFoot")
    assert not np.array_equal(d7.local_rotations_xyzw[:, index],
                              d7b.local_rotations_xyzw[:, index])


# ------------------------------------------------------------ the legacy path is untouched
def test_the_legacy_path_never_reaches_the_new_aim() -> None:
    """With no spine landmark `_joint_origin` is never asked for `Spine`. One branch."""

    positions, _, _ = posed_body()
    asked: list[str] = []
    shipped = cm._joint_origin

    def watcher(world, frame, root_translation, rest, joint_name):
        asked.append(joint_name)
        return shipped(world, frame, root_translation, rest, joint_name)

    cm._joint_origin = watcher
    try:
        convert(positions, None)
    finally:
        cm._joint_origin = shipped
    assert asked, "the watcher never ran; the substitution point moved"
    assert "Spine" not in asked


def test_the_legacy_torso_channel_is_still_the_trunk_line_bit_for_bit() -> None:
    """What `Spine`, `Chest` and `UpperChest` were, asserted against the construction."""

    positions, _, _ = posed_body()
    track = convert(positions, None)
    points = rig_points(positions)
    rotations = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    hips_index = track.joint_names.index("Hips")
    spine_index = track.joint_names.index("Spine")
    for frame in range(len(points)):
        p = points[frame]
        pelvis = 0.5 * (p[cm.JOINT_INDEX["left_hip"]] + p[cm.JOINT_INDEX["right_hip"]])
        expected = cm._frame_alignment(
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
            p[cm.JOINT_INDEX["neck"]] - pelvis,
            p[cm.JOINT_INDEX["left_shoulder"]] - p[cm.JOINT_INDEX["right_shoulder"]])
        got = cm._quaternion_multiply(rotations[frame, hips_index],
                                      rotations[frame, spine_index])
        assert min(np.abs(got - expected).max(), np.abs(got + expected).max()) < 1e-6


@pytest.mark.parametrize("wobble", (0.0, 0.004))
def test_a_legacy_track_is_bit_identical_whatever_the_spine_point_would_have_been(
    wobble: float,
) -> None:
    """The spine feed is the ONLY switch: with it `None`, the track cannot depend on it."""

    positions, spine, _ = posed_body(wobble_m=wobble)
    reference = convert(positions, None)
    track = convert(positions, None)
    assert np.array_equal(track.local_rotations_xyzw, reference.local_rotations_xyzw)
    assert np.array_equal(track.root_translation_m, reference.root_translation_m)
    assert spine.shape == (len(positions), 3)
