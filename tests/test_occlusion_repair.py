"""D8 -- the occlusion repair: the conditioning gate, the reachability reject, the gap hold.

A NEW file. No existing test is edited; `tests/test_commercial_multiview.py`,
`tests/test_pelvis_frame.py` and `tests/test_trunk_resolve.py` stay as they are and stay
green, which is itself part of the claim -- the repair must not move a legacy result that
never had an occluded slot.

What is asserted here, and why each one exists:

* the conditioning gate fires on a near-opposite camera pair and NOT on an orthogonal one,
  because the whole rule is a statement about geometry and a rule that fired on every pair
  would be a smoother;
* three or more supporting views are never demoted, because the third view fixes the depth
  the pair cannot;
* reachability accepts a fast-but-possible move and rejects an impossible one;
* reachability rejects the wrong plateau between two big steps that a STEP TEST accepts,
  and accepts a legitimate travel across a long gap that a step test rejects -- the two
  defects that make a step test the control and not the rule (CLAUDE.md);
* the ORACLE: on clean fully-seen input the repair does nothing at all and the smoothed
  output is bit-identical to the pre-D8 path;
* the RAW array is bit-identical whatever the rules do, because it is captured before them
  and is D8's one unchanged reference;
* no LEG is ever demoted on the clean fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import autoanim_gnm
import autoanim_gnm.commercial_multiview as cm
from autoanim_gnm.commercial_multiview import (
    CalibratedCamera,
    JOINT_INDEX,
    JOINT_NAMES,
)

SAMPLE_RATE_HZ = 30


def test_the_package_under_test_is_this_worktree() -> None:
    """The venv's editable install points at the MAIN tree; a green test there is a lie."""

    here = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(here))


# ------------------------------------------------------------------------------ fixtures
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


# front and rear sit on opposite sides of the subject: their rays to a point near the
# origin are nearly anti-parallel, which is the reference rig's A001/C001 relationship.
# left and front are ~90 degrees apart, which is a well-conditioned pair.
FRONT, REAR, LEFT, RIGHT = (
    _look_at_camera("front", (0.0, -5.0, 2.0)),
    _look_at_camera("rear", (0.0, 5.0, 2.0)),
    _look_at_camera("left", (-4.0, -1.0, 2.0)),
    _look_at_camera("right", (4.0, -1.0, 2.0)),
)
CAMERAS = (FRONT, LEFT, REAR, RIGHT)


def _body(frame: int, offset: np.ndarray) -> np.ndarray:
    """One plausible 19-joint body, drifting slowly with the frame. Never degenerate."""

    base = {
        "root": (0.0, 0.0, 1.0), "neck": (0.0, 0.0, 1.5), "nose": (0.0, 0.06, 1.65),
        "left_eye": (0.03, 0.07, 1.68), "right_eye": (-0.03, 0.07, 1.68),
        "left_ear": (0.07, 0.0, 1.67), "right_ear": (-0.07, 0.0, 1.67),
        "left_shoulder": (0.18, 0.0, 1.45), "right_shoulder": (-0.18, 0.0, 1.45),
        "left_elbow": (0.22, 0.02, 1.18), "right_elbow": (-0.22, 0.02, 1.18),
        "left_wrist": (0.24, 0.04, 0.92), "right_wrist": (-0.24, 0.04, 0.92),
        "left_hip": (0.10, 0.0, 0.98), "right_hip": (-0.10, 0.0, 0.98),
        "left_knee": (0.11, 0.01, 0.55), "right_knee": (-0.11, 0.01, 0.55),
        "left_ankle": (0.12, 0.02, 0.10), "right_ankle": (-0.12, 0.02, 0.10),
    }
    points = np.zeros((len(JOINT_NAMES), 3), dtype=np.float64)
    drift = np.asarray((0.004 * frame, 0.003 * frame, 0.0))
    swing = np.asarray((0.0, 0.0, 0.02 * np.sin(frame * 0.4)))
    for name, value in base.items():
        points[JOINT_INDEX[name]] = np.asarray(value) + offset + drift
        if name.endswith(("wrist", "elbow")):
            points[JOINT_INDEX[name]] = points[JOINT_INDEX[name]] + swing
    return points


def _records(frames: int = 14, subjects: int = 2) -> tuple[list[list[dict]], np.ndarray]:
    """Four cameras, every joint seen by every camera on every frame. The clean fixture."""

    offsets = [np.asarray((-0.65, 0.0, 0.0)), np.asarray((0.65, 0.0, 0.0))]
    truth = np.stack([
        np.stack([_body(frame, offsets[subject]) for frame in range(frames)])
        for subject in range(subjects)])
    out: list[list[dict]] = []
    for camera in CAMERAS:
        rows = []
        for frame in range(frames):
            people = []
            for subject in range(subjects):
                uv, depth = camera.project(truth[subject, frame])
                joints = {
                    name: {"x": float(uv[JOINT_INDEX[name]][0]),
                           "y": float(uv[JOINT_INDEX[name]][1]),
                           "confidence": 0.95}
                    for name in JOINT_NAMES
                    if depth[JOINT_INDEX[name]] > 0.0}
                people.append({"index": subject, "joints": joints})
            rows.append({"frame_index": frame, "width": 1280, "height": 720,
                         "people": people})
        out.append(rows)
    return out, truth


# --------------------------------------------------------------- the conditioning gate
def test_the_conditioning_gate_fires_on_an_opposite_pair_and_not_on_an_orthogonal_one() -> None:
    """The rule is a statement about geometry, and it has to discriminate on geometry.

    One point, one ceiling, two different supporting pairs. Nothing else differs, so the
    only thing that can move the verdict is the angle between the two rays.
    """

    point = np.asarray((0.0, 0.0, 1.2))
    world = np.full((1, len(JOINT_NAMES), 3), np.nan)
    world[0, JOINT_INDEX["left_shoulder"]] = point

    opposite = np.zeros((1, len(JOINT_NAMES), len(CAMERAS)), dtype=bool)
    opposite[0, JOINT_INDEX["left_shoulder"], [0, 2]] = True      # front + rear
    orthogonal = np.zeros_like(opposite)
    orthogonal[0, JOINT_INDEX["left_shoulder"], [0, 1]] = True    # front + left

    assert cm._ray_pair_angles_deg(CAMERAS, point, [0, 2]) > 160.0
    assert 60.0 < cm._ray_pair_angles_deg(CAMERAS, point, [0, 1]) < 120.0

    demoted, report = cm._repair_occluded_slots(
        CAMERAS, world, opposite, sample_rate_hz=SAMPLE_RATE_HZ,
        ray_pair_ceiling_deg=cm.RAY_PAIR_CONDITIONING_CEILING_DEG, reachability=False)
    assert report["demoted_slots"] == 1
    assert not np.isfinite(demoted[0, JOINT_INDEX["left_shoulder"]]).all()

    kept, report = cm._repair_occluded_slots(
        CAMERAS, world, orthogonal, sample_rate_hz=SAMPLE_RATE_HZ,
        ray_pair_ceiling_deg=cm.RAY_PAIR_CONDITIONING_CEILING_DEG, reachability=False)
    assert report["demoted_slots"] == 0
    assert np.allclose(kept[0, JOINT_INDEX["left_shoulder"]], point)


def test_a_third_supporting_view_is_never_demoted() -> None:
    """A pair cannot fix depth along its axis; a third view can, so the rule must not fire."""

    point = np.asarray((0.0, 0.0, 1.2))
    world = np.full((1, len(JOINT_NAMES), 3), np.nan)
    world[0, JOINT_INDEX["left_shoulder"]] = point
    views = np.zeros((1, len(JOINT_NAMES), len(CAMERAS)), dtype=bool)
    views[0, JOINT_INDEX["left_shoulder"], [0, 2, 1]] = True      # front + rear + left

    _values, report = cm._repair_occluded_slots(
        CAMERAS, world, views, sample_rate_hz=SAMPLE_RATE_HZ,
        ray_pair_ceiling_deg=cm.RAY_PAIR_CONDITIONING_CEILING_DEG, reachability=False)
    assert report["demoted_slots"] == 0


def test_the_conditioning_gate_never_writes_into_the_array_it_is_given() -> None:
    """The raw array is D8's fixed reference and the repair takes a copy, not a slice."""

    point = np.asarray((0.0, 0.0, 1.2))
    world = np.full((1, len(JOINT_NAMES), 3), np.nan)
    world[0, JOINT_INDEX["left_shoulder"]] = point
    before = world.copy()
    views = np.zeros((1, len(JOINT_NAMES), len(CAMERAS)), dtype=bool)
    views[0, JOINT_INDEX["left_shoulder"], [0, 2]] = True

    cm._repair_occluded_slots(
        CAMERAS, world, views, sample_rate_hz=SAMPLE_RATE_HZ,
        ray_pair_ceiling_deg=cm.RAY_PAIR_CONDITIONING_CEILING_DEG, reachability=True)
    assert np.array_equal(world, before, equal_nan=True)


# ------------------------------------------------------------------- the reachability rule
def test_reachability_accepts_a_fast_but_possible_move_and_rejects_an_impossible_one() -> None:
    """34 m/s is 1.13 m per frame at 30 fps. Half of that is fast; ten times it is not."""

    ceiling = cm.REACHABILITY_SPEED_CEILING_M_S["left_wrist"]
    points = np.zeros((3, 3))
    points[1] = (0.0, 0.0, 0.5)                    # 0.5 m in one frame = 15 m/s
    points[2] = (0.0, 0.0, 11.0)                   # 10.5 m in one frame = 315 m/s
    accepted = cm._reachable_landmark_sequence(
        points, ceiling, cm.REACHABILITY_SLACK_M, SAMPLE_RATE_HZ)
    assert accepted.tolist() == [True, True, False]


def test_a_non_finite_frame_is_never_accepted_and_never_becomes_the_anchor() -> None:
    points = np.zeros((4, 3))
    points[1] = np.nan
    points[2] = (0.0, 0.0, 0.2)
    points[3] = (0.0, 0.0, 0.4)
    accepted = cm._reachable_landmark_sequence(
        points, cm.REACHABILITY_SPEED_CEILING_M_S["left_wrist"],
        cm.REACHABILITY_SLACK_M, SAMPLE_RATE_HZ)
    assert accepted[1] == np.False_
    assert accepted[0] and accepted[2] and accepted[3]


def test_reachability_rejects_the_wrong_plateau_a_step_test_accepts() -> None:
    """CLAUDE.md: a step test accepts the wrong plateau between two big steps.

    Frames 0-2 sit at A, frames 3-7 sit at B a long way off, frames 8-10 are back at A.
    Every step INSIDE the plateau is zero, so a step test waves 4-7 through once it has
    taken 3 as its reference. Reachability keeps its anchor at the last ACCEPTED frame --
    still at A -- and refuses the whole plateau.
    """

    ceiling, slack = 1.0, 0.05
    points = np.zeros((11, 3))
    points[3:8] = (0.0, 0.0, 2.0)
    reach = cm._reachable_landmark_sequence(points, ceiling, slack, SAMPLE_RATE_HZ)
    step = cm._reachable_landmark_sequence(points, ceiling, slack, SAMPLE_RATE_HZ,
                                           rule="step")
    assert not reach[3:8].any(), "reachability must refuse every frame of the plateau"
    assert step[4:8].all(), "the step test accepts the plateau -- that is why it is the control"
    assert reach[8:].all() and reach[:3].all()


def test_reachability_accepts_a_travel_across_a_long_gap_a_step_test_rejects() -> None:
    """The envelope widens with the frames elapsed; a one-frame budget does not.

    The landmark is seen at frame 0, unobserved for twenty frames, and seen again 0.5 m
    away. At 1 m/s that is comfortably reachable in twenty frames and impossible in one.
    """

    ceiling, slack = 1.0, 0.05
    points = np.full((22, 3), np.nan)
    points[0] = (0.0, 0.0, 0.0)
    points[21] = (0.0, 0.0, 0.5)
    reach = cm._reachable_landmark_sequence(points, ceiling, slack, SAMPLE_RATE_HZ)
    step = cm._reachable_landmark_sequence(points, ceiling, slack, SAMPLE_RATE_HZ,
                                           rule="step")
    assert reach[21], "a body may travel while it is unobserved"
    assert not step[21], "the step test rejects it for having moved -- the second defect"


# ----------------------------------------------------------------------- the gap clause
def test_a_long_gap_is_held_on_the_parent_and_a_short_one_is_left_to_the_fill() -> None:
    frames = 20
    values = np.full((frames, len(JOINT_NAMES), 3), np.nan)
    elbow, wrist = JOINT_INDEX["left_elbow"], JOINT_INDEX["left_wrist"]
    for frame in range(frames):
        values[frame, elbow] = (0.0, 0.05 * frame, 1.0)
        values[frame, wrist] = (0.0, 0.05 * frame, 0.7)
    values[3:5, wrist] = np.nan                 # a 2-frame gap: short
    values[8:17, wrist] = np.nan                # a 9-frame gap: long

    filled, held = cm._hold_long_gaps_on_parent(values, maximum_gap_frames=6)
    assert not np.isfinite(filled[3:5, wrist]).any(), "a short gap is the fill's job"
    assert np.isfinite(filled[8:17, wrist]).all(), "a long gap is held, not interpolated"
    # Held means carried on the parent: the offset from the elbow is the one it had at the
    # last known frame, translated by the elbow's own motion.
    offset = values[7, wrist] - values[7, elbow]
    for frame in range(8, 17):
        assert np.allclose(filled[frame, wrist] - filled[frame, elbow], offset)
    assert held > 0.0


def test_the_gap_clause_does_nothing_when_nothing_is_missing() -> None:
    values = np.zeros((10, len(JOINT_NAMES), 3))
    filled, held = cm._hold_long_gaps_on_parent(values, maximum_gap_frames=6)
    assert np.array_equal(values, filled)
    assert held == 0.0


# ---------------------------------------------------------------------------- the oracle
@pytest.fixture(scope="module")
def clean_runs():
    records, truth = _records()
    off = cm.reconstruct_multiview(
        CAMERAS, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ,
        ray_pair_conditioning_ceiling_deg=None, reachability_reject=False,
        segment_length_ceiling_fraction=None,  # D8b (2026-09-06): the "off" arm is pre-D8 code again
        maximum_interpolated_gap_frames=None)
    on = cm.reconstruct_multiview(
        CAMERAS, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ)
    return off, on, truth


def test_the_oracle_clean_input_is_bit_identical_and_fires_nothing(clean_runs) -> None:
    """The dual of 'no gate a constant can pass': a rule that fires on clean input is a
    smoother, not a reject. Nothing is occluded here, so nothing may be demoted, rejected
    or held, and the delivered array must come out unchanged to the last bit."""

    off, on, _truth = clean_runs
    _tracks_off, diagnostics_off, smoothed_off, _raw_off = off
    _tracks_on, diagnostics_on, smoothed_on, _raw_on = on

    payload = diagnostics_on.as_dict()
    assert payload["held_joint_fraction"] == 0.0
    assert [row["demoted_slots"] for row in payload["occlusion_repair"]] == [0, 0]
    assert [row["rejected_slots"] for row in payload["occlusion_repair"]] == [0, 0]
    assert np.array_equal(smoothed_off, smoothed_on)
    assert diagnostics_off.interpolated_joint_fraction == \
        diagnostics_on.interpolated_joint_fraction


def test_the_raw_array_is_bit_identical_whatever_the_rules_do(clean_runs) -> None:
    """The raw array is captured before the repair and is D8's one unchanged reference."""

    off, on, _truth = clean_runs
    assert np.array_equal(off[3], on[3], equal_nan=True)


def test_no_leg_is_ever_demoted_or_rejected_on_the_clean_fixture(clean_runs) -> None:
    """B3's prediction, asserted where it can be asserted exactly."""

    _off, on, _truth = clean_runs
    payload = on[1].as_dict()
    legs = {"left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"}
    for row in payload["occlusion_repair"]:
        assert not legs & set(row["demoted_by_joint"])
        assert not legs & set(row["rejected_by_joint"])


def test_turning_the_rules_on_changes_the_smoothed_array_when_a_view_is_lost() -> None:
    """The complement of the oracle: if nothing ever changed, the step would be inert.

    Two cameras are stripped of one performer's arm landmarks over a run of frames, which
    is the reference take's own failure, and the two remaining cameras are the opposite
    pair. The repair must then do something, and the raw array must still not move.
    """

    records, _truth = _records()
    stripped = [[dict(row, people=[dict(person, joints={
        name: value for name, value in person["joints"].items()
        if not (subject == 0 and camera in (1, 3) and 4 <= row["frame_index"] <= 11
                and name in ("left_shoulder", "left_elbow", "left_wrist",
                             "right_shoulder", "right_elbow", "right_wrist"))})
        for subject, person in enumerate(row["people"])]) for row in rows]
        for camera, rows in enumerate(records)]

    off = cm.reconstruct_multiview(
        CAMERAS, stripped, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ,
        ray_pair_conditioning_ceiling_deg=None, reachability_reject=False,
        segment_length_ceiling_fraction=None,  # D8b (2026-09-06): the "off" arm is pre-D8 code again
        maximum_interpolated_gap_frames=None)
    on = cm.reconstruct_multiview(
        CAMERAS, stripped, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ)

    assert np.array_equal(off[3], on[3], equal_nan=True), "the raw array must not move"
    payload = on[1].as_dict()
    fired = sum(row["demoted_slots"] + row["rejected_slots"]
                for row in payload["occlusion_repair"])
    assert fired > 0, "with only the opposite pair left the conditioning gate must fire"
    assert not np.array_equal(off[2], on[2]), "and the delivered array must follow it"
