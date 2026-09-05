"""D8b -- the segment-length consistency reject on the captured points.

A NEW file. No existing test is edited: `tests/test_occlusion_repair.py`,
`tests/test_commercial_multiview.py`, `tests/test_pelvis_frame.py` and
`tests/test_trunk_resolve.py` stay as they are and stay green, which is part of the claim --
a rule that only ever withholds an impossible frame must not move a legacy result whose
frames are all possible.

What is asserted here, and why each one exists:

* the rule FIRES on a consistent shoulder collapse -- the defect measured on the reference
  take, where three cameras agree with a point that puts the performer's shoulders 122 mm
  apart when his own take median is 364 -- and charges it to the CHILD landmarks;
* it does NOT fire on honest motion inside the ceiling, because a rule that fired on real
  movement would be a smoother wearing a reject's clothes;
* it never fires on the LEGS of a clean body, which is the must-fail the card names;
* the RAW array is untouched -- both the array handed to the function and the one
  `reconstruct_multiview` returns;
* the ORACLE: on clean fully-seen input the rule does nothing and the smoothed output is
  bit-identical to the same run with the rule off;
* the three modes do three different things to the RAYS, and are distinguishable at the
  pipeline level rather than only in a docstring;
* the two blindnesses that matter are asserted rather than described: a take-long collapse
  is invisible, and a segment with too few measured frames is never judged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm
import autoanim_gnm.commercial_multiview as cm
from autoanim_gnm.commercial_multiview import JOINT_INDEX, JOINT_NAMES

import test_occlusion_repair as d8

SAMPLE_RATE_HZ = 30
CAMERAS = d8.CAMERAS


def test_the_package_under_test_is_this_worktree() -> None:
    """The venv's editable install points at the MAIN tree; a green test there is a lie."""

    here = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(here))


# ------------------------------------------------------------------------------ fixtures
def _rigid_take(frames: int = 40) -> np.ndarray:
    """One performer's triangulated points, `[frame, joint, 3]`, with rigid segments.

    `test_occlusion_repair._body` is reused rather than re-typed: it is a plausible
    19-joint body whose elbows and wrists swing, so the honest motion here is real motion
    and not a frozen skeleton the rule could never fire on.
    """

    return np.stack([d8._body(frame, np.zeros(3)) for frame in range(frames)])


def _collapse_shoulders(values: np.ndarray, frames: slice, factor: float) -> np.ndarray:
    """Move both shoulders toward the neck by `factor`. The measured failure mode."""

    out = values.copy()
    neck = out[frames, JOINT_INDEX["neck"]]
    for name in ("left_shoulder", "right_shoulder"):
        joint = JOINT_INDEX[name]
        out[frames, joint] = neck + factor * (out[frames, joint] - neck)
    return out


# ------------------------------------------------------------------------------ it fires
def test_it_fires_on_a_consistent_shoulder_collapse_and_charges_the_child() -> None:
    """The defect this step exists for, reduced to its arithmetic.

    Both shoulders are pulled halfway to the neck for eight frames. The shoulder line
    halves, which is 50 % off the take median and far outside any ceiling, and the shoulder
    line has no parent -- so BOTH of its endpoints are marked. Nothing else in the body has
    moved, and the frames outside the run must be untouched.
    """

    values = _rigid_take()
    collapsed = _collapse_shoulders(values, slice(10, 18), 0.5)
    withheld, actions, report = cm._reject_inconsistent_segments(
        collapsed, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)

    assert report["length_rejected_by_segment"].get("shoulder_line") == 8
    assert report["length_rejected_frames_by_segment"]["shoulder_line"] == list(range(10, 18))
    for name in ("left_shoulder", "right_shoulder"):
        joint = JOINT_INDEX[name]
        assert not np.isfinite(withheld[10:18, joint]).any(), f"{name} must be withheld"
        assert np.isfinite(withheld[:10, joint]).all(), "the honest frames stay"
        assert np.isfinite(withheld[18:, joint]).all()
    assert actions["marked"][10:18, JOINT_INDEX["left_shoulder"]].all()


def test_the_collapse_is_invisible_to_every_geometric_gate_that_precedes_it() -> None:
    """Why the rule has to exist at all, asserted rather than asserted-in-prose.

    A collapse applied identically to every view triangulates to a well-conditioned point:
    the supporting views are three or four, so D8's conditioning gate never fires, and the
    landmark's frame-to-frame travel is a few centimetres, so the reachability reject never
    fires either. With D8's two rules on and D8b's off, nothing is withheld.
    """

    records, _truth = d8._records(frames=24)
    collapsed = []
    for rows in records:
        new_rows = []
        for row in rows:
            people = []
            for subject, person in enumerate(row["people"]):
                joints = dict(person["joints"])
                if subject == 0 and 8 <= row["frame_index"] < 18 and "neck" in joints:
                    neck = joints["neck"]
                    for name in ("left_shoulder", "right_shoulder"):
                        if name not in joints:
                            continue
                        joints[name] = dict(
                            joints[name],
                            x=neck["x"] + 0.5 * (joints[name]["x"] - neck["x"]),
                            y=neck["y"] + 0.5 * (joints[name]["y"] - neck["y"]))
                people.append({"index": person["index"], "joints": joints})
            new_rows.append(dict(row, people=people))
        collapsed.append(new_rows)

    _tracks, diagnostics, _smoothed, _raw = cm.reconstruct_multiview(
        CAMERAS, collapsed, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ,
        segment_length_ceiling_fraction=None)
    payload = diagnostics.as_dict()
    assert [row["demoted_slots"] for row in payload["occlusion_repair"]] == [0, 0], \
        "the conditioning gate cannot see a collapse every camera agrees on"
    assert [row["rejected_slots"] for row in payload["occlusion_repair"]] == [0, 0], \
        "and neither can a reachability envelope"

    _tracks, diagnostics, _smoothed, _raw = cm.reconstruct_multiview(
        CAMERAS, collapsed, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ)
    payload = diagnostics.as_dict()
    assert sum(row["length_rejected_slots"] for row in payload["occlusion_repair"]) > 0, \
        "the performer's own segment lengths are the only evidence that can"


# -------------------------------------------------------------------------- it does not
def test_it_does_not_fire_on_honest_motion() -> None:
    """The dual of the clause above. The fixture's elbows and wrists swing every frame."""

    values = _rigid_take()
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_slots"] == 0
    assert report["length_rejected_by_segment"] == {}


def test_it_never_fires_on_a_leg_of_a_clean_body() -> None:
    """The card's must-fail, asserted through the real pipeline on the clean fixture."""

    records, _truth = d8._records(frames=24)
    _tracks, diagnostics, _smoothed, _raw = cm.reconstruct_multiview(
        CAMERAS, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ)
    legs = {"left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle"}
    for row in diagnostics.as_dict()["occlusion_repair"]:
        assert not legs & set(row["length_rejected_by_joint"] or {})


def test_a_ceiling_tight_enough_to_eat_the_legs_is_what_a_bad_ceiling_looks_like() -> None:
    """No gate a constant can pass, from the other side: the band has to be losable.

    At a ceiling of half a per cent the fixture's own honest motion breaks every segment,
    which is what a ceiling chosen without a measured seed would do.
    """

    values = _rigid_take()
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=0.005)
    assert report["length_rejected_slots"] > 0


# ------------------------------------------------------------------- the raw array
def test_the_rule_never_writes_into_the_array_it_is_given() -> None:
    values = _collapse_shoulders(_rigid_take(), slice(10, 18), 0.5)
    before = values.copy()
    cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert np.array_equal(values, before, equal_nan=True)


@pytest.fixture(scope="module")
def clean_runs():
    records, truth = d8._records(frames=24)
    off = cm.reconstruct_multiview(
        CAMERAS, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ,
        segment_length_ceiling_fraction=None)
    on = cm.reconstruct_multiview(
        CAMERAS, records, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ)
    return off, on, truth


def test_the_oracle_clean_input_is_bit_identical_and_fires_nothing(clean_runs) -> None:
    """A rule that fires on clean input is a smoother, not a reject."""

    off, on, _truth = clean_runs
    payload = on[1].as_dict()
    assert [row["length_rejected_slots"] for row in payload["occlusion_repair"]] == [0, 0]
    assert np.array_equal(off[2], on[2]), "the delivered array must not move"


def test_the_raw_array_is_bit_identical_whether_the_rule_runs_or_not(clean_runs) -> None:
    """D8b's reject sits after `world` is captured; the raw array is the fixed reference."""

    off, on, _truth = clean_runs
    assert np.array_equal(off[3], on[3], equal_nan=True)


# ---------------------------------------------------------------------------- the modes
def test_the_three_modes_do_three_different_things_to_the_rays() -> None:
    values = _collapse_shoulders(_rigid_take(), slice(10, 18), 0.5)
    marks = {}
    for mode in cm.SEGMENT_LENGTH_MODES:
        _withheld, actions, _report = cm._reject_inconsistent_segments(
            values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION, mode=mode)
        marks[mode] = actions
    assert not marks["demote"]["clear_rays"].any()
    assert not marks["demote"]["keep_best_ray"].any()
    assert marks["reject"]["clear_rays"].any()
    assert not marks["reject"]["keep_best_ray"].any()
    assert marks["best_ray"]["keep_best_ray"].any()
    assert not marks["best_ray"]["clear_rays"].any()
    # All three withhold the same POINTS; only the rays differ.
    for mode in ("reject", "best_ray"):
        assert np.array_equal(marks[mode]["marked"], marks["demote"]["marked"])


def test_the_modes_are_distinguishable_at_the_pipeline_level() -> None:
    """Three recovery mechanisms, not three names for one. If two of these agreed bit for
    bit the selector between them would be measuring nothing."""

    records, _truth = d8._records(frames=30)
    stripped = [[dict(row, people=[dict(person, joints={
        name: value for name, value in person["joints"].items()
        if not (subject == 0 and camera == 3 and 6 <= row["frame_index"] <= 19
                and name in ("left_shoulder", "right_shoulder"))})
        for subject, person in enumerate(row["people"])]) for row in rows]
        for camera, rows in enumerate(records)]
    collapsed = []
    for camera, rows in enumerate(stripped):
        new_rows = []
        for row in rows:
            people = []
            for subject, person in enumerate(row["people"]):
                joints = dict(person["joints"])
                if subject == 0 and 8 <= row["frame_index"] < 16 and "neck" in joints:
                    neck = joints["neck"]
                    for name in ("left_shoulder", "right_shoulder"):
                        if name not in joints:
                            continue
                        joints[name] = dict(
                            joints[name],
                            x=neck["x"] + 0.45 * (joints[name]["x"] - neck["x"]),
                            y=neck["y"] + 0.45 * (joints[name]["y"] - neck["y"]))
                people.append({"index": person["index"], "joints": joints})
            new_rows.append(dict(row, people=people))
        collapsed.append(new_rows)

    outputs = {}
    for mode in cm.SEGMENT_LENGTH_MODES:
        _tracks, _diagnostics, smoothed, _raw = cm.reconstruct_multiview(
            CAMERAS, collapsed, subject_count=2, sample_rate_hz=SAMPLE_RATE_HZ,
            segment_length_mode=mode)
        outputs[mode] = smoothed
    assert not np.array_equal(outputs["demote"], outputs["reject"])
    assert not np.array_equal(outputs["demote"], outputs["best_ray"])
    assert not np.array_equal(outputs["reject"], outputs["best_ray"])


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(cm.CommercialMultiviewError):
        cm._reject_inconsistent_segments(
            _rigid_take(), ceiling_fraction=0.15, mode="whatever")


def test_a_non_positive_ceiling_is_refused() -> None:
    with pytest.raises(cm.CommercialMultiviewError):
        cm._reject_inconsistent_segments(_rigid_take(), ceiling_fraction=0.0)


# ------------------------------------------------------------------- the two blindnesses
def test_the_fault_must_be_a_minority_of_the_take_or_the_reference_is_corrupt() -> None:
    """The reference is a MEDIAN, so it survives a minority and not a majority.

    Found while building the pipeline-level mode test: with exactly half the frames
    collapsed the median lands between the honest and the collapsed value and EVERY frame
    reads off by more than the ceiling -- the rule then withholds the whole take, which is a
    worse failure than the one it is fixing. A minority fault is what it is for, and this is
    the boundary written down rather than discovered again.
    """

    values = _collapse_shoulders(_rigid_take(frames=40), slice(0, 20), 0.5)
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_by_segment"]["shoulder_line"] == 40, \
        "at a 50/50 split the median is between the two values and every frame fires"

    minority = _collapse_shoulders(_rigid_take(frames=40), slice(0, 12), 0.5)
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        minority, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_by_segment"]["shoulder_line"] == 12, \
        "a minority fault leaves the median where it belongs"


def test_a_take_long_collapse_is_invisible_and_that_is_stated_as_a_test() -> None:
    """The reference is the performer's OWN median, so a collapse on every frame moves the
    median with it and nothing fires. This is the rule's founding limitation and it is
    asserted here so it can never be forgotten in a summary."""

    values = _collapse_shoulders(_rigid_take(), slice(None), 0.5)
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_slots"] == 0


def test_a_segment_with_too_few_measured_frames_is_never_judged() -> None:
    """A median over a handful of frames is not a reference, and a rule with no reference
    does not get to reject anything."""

    values = _rigid_take()
    values[5:, JOINT_INDEX["left_wrist"]] = np.nan
    values[:5, JOINT_INDEX["left_wrist"]] = values[:5, JOINT_INDEX["left_elbow"]] * 3.0
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["segment_frames_measured"]["left_forearm"] == 5
    assert report["length_rejected_by_segment"].get("left_forearm") is None
    assert report["segment_median_m"]["left_forearm"] is None


def test_the_child_cascade_is_counted_not_hidden() -> None:
    """A pure shoulder collapse makes the UPPER ARM read long too, so the elbow is marked
    with the shoulder and a good point goes with a bad one. The report counts those cells;
    it does not pretend they are not there."""

    values = _collapse_shoulders(_rigid_take(), slice(10, 18), 0.5)
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    fired = report["length_rejected_by_segment"]
    assert "shoulder_line" in fired
    assert "left_upper_arm" in fired and "right_upper_arm" in fired, \
        "the upper arm reads long when its parent shoulder has moved"
    # EXACTLY the two elbows on each of the eight collapsed frames, and nothing else: the
    # shoulder line is skipped in the count (it charges its own endpoints, so its child is
    # its parent) and the forearms do not fire, because the elbow and wrist both moved with
    # nothing between them. A `> 0` here would have passed the tautological version of this
    # counter that shipped in the first draft.
    assert report["children_marked_under_a_marked_parent_segment"] == 16
    assert set(report["length_rejected_by_joint"]) >= {
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"}
    assert not ({"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
                 "right_ankle"} & set(report["length_rejected_by_joint"]))


def test_every_joint_name_the_rule_speaks_of_exists() -> None:
    for _name, parent, child, charged in cm.SEGMENT_LENGTH_RULES:
        assert parent in JOINT_INDEX and child in JOINT_INDEX
        for landmark in charged:
            assert landmark in JOINT_NAMES
