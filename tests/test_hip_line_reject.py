"""D8c -- the HIP LINE row in `SEGMENT_LENGTH_RULES`, and what it costs.

A NEW file. No existing test is edited. `tests/test_segment_length_reject.py` stays exactly
as it is and stays green, which is part of the claim: D8b's nine rows behave identically
with a tenth beside them, because every length is computed on the array as it arrived and
the rule is order-independent within itself.

What is asserted here, and why each one exists:

* THE ROW IS THERE, exactly once, at the END of the tuple, charging BOTH endpoints. The
  position matters only because every report keys its per-segment blocks in tuple order and
  a reader comparing two builds should not have to sort them.
* it FIRES on the measured mode -- both hips moved along their own line toward its midpoint,
  which is what the reference take does on frames 110-119 -- and charges both hips;
* it does NOT fire when the hips are honest, and it does NOT fire on the LEGS, the ARMS or
  the shoulder line of the same clean body;
* THE KNEES ARE NOT DRAGGED IN. On a hip collapse the thighs read LONG, and if the thigh
  rows fired the knees would be withheld with the hips -- a cascade that the real take does
  not show and that an inward-collapse injection must not manufacture. The thighs' own
  ceiling is what stops it and the test says so with numbers.
* THE COST OF CHARGING BOTH ENDPOINTS is asserted, not described: on a ONE-HIP frame -- the
  reference take's frames 84-86, where one femoral head moves and the other matches the
  take -- the rule withholds the good hip with the bad one. That is the card's registered
  KNOWN OVER-CHARGE and it is here as a test so it can never quietly stop being true.
* the RAW array is untouched;
* a hip line that is wrong on EVERY frame is invisible, because the reference is the
  performer's own median. The blindness the whole rule is built on, asserted rather than
  written down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import autoanim_gnm
import autoanim_gnm.commercial_multiview as cm
from autoanim_gnm.commercial_multiview import JOINT_INDEX

import test_occlusion_repair as d8

HIP_ROW = ("hip_line", "left_hip", "right_hip", ("left_hip", "right_hip"))


def test_the_package_under_test_is_this_worktree() -> None:
    """The venv's editable install points at the MAIN tree; a green test there is a lie."""

    here = Path(__file__).resolve().parents[1]
    assert str(Path(autoanim_gnm.__file__).resolve()).startswith(str(here))


# ------------------------------------------------------------------------------ fixtures
def _take(frames: int = 40) -> np.ndarray:
    """One performer's triangulated points, `[frame, joint, 3]`, with rigid segments."""

    return np.stack([d8._body(frame, np.zeros(3)) for frame in range(frames)])


def _collapse_hips(values: np.ndarray, frames: slice, factor: float,
                   sides: tuple[str, ...] = ("left_hip", "right_hip")) -> np.ndarray:
    """Move the hips toward the MIDPOINT OF THE TWO HIPS, along the line between them.

    The mode measured on the reference take, and NOT a collapse toward the pelvis landmark:
    on frames 110-119 the hip line halves while |hip_mid - root| holds (68.0-80.7 mm against
    a 76.5 mm median) and both thighs hold. Passing one side is the 84-86 mode, where a
    single femoral head moves and the midpoint therefore moves with it.
    """

    out = values.copy()
    left = out[frames, JOINT_INDEX["left_hip"]]
    right = out[frames, JOINT_INDEX["right_hip"]]
    midpoint = 0.5 * (left + right)
    for name in sides:
        joint = JOINT_INDEX[name]
        out[frames, joint] = midpoint + factor * (out[frames, joint] - midpoint)
    return out


# ------------------------------------------------------------------------------ the row
def test_the_hip_row_is_in_the_table_once_last_and_charges_both_endpoints() -> None:
    rows = [row for row in cm.SEGMENT_LENGTH_RULES if row[0] == "hip_line"]
    assert rows == [HIP_ROW], "exactly one hip_line row, charging both endpoints"
    assert cm.SEGMENT_LENGTH_RULES[-1] == HIP_ROW, (
        "the hip row is appended LAST, so every report's per-segment key order is the "
        "D8b order with one entry added rather than reshuffled")
    assert len(cm.SEGMENT_LENGTH_RULES) == 10
    # The ceiling and the modes are D8b's and D8c re-selected neither.
    assert cm.SEGMENT_LENGTH_CEILING_FRACTION == 0.15
    assert cm.SEGMENT_LENGTH_MODES == ("demote", "reject", "best_ray")


def test_the_nine_D8b_rows_are_untouched() -> None:
    """A tenth row must not disturb the nine. The whole B2 band, in one assertion."""

    assert tuple(row for row in cm.SEGMENT_LENGTH_RULES if row[0] != "hip_line") == (
        ("shoulder_line", "left_shoulder", "right_shoulder",
         ("left_shoulder", "right_shoulder")),
        ("left_upper_arm", "left_shoulder", "left_elbow", ("left_elbow",)),
        ("right_upper_arm", "right_shoulder", "right_elbow", ("right_elbow",)),
        ("left_forearm", "left_elbow", "left_wrist", ("left_wrist",)),
        ("right_forearm", "right_elbow", "right_wrist", ("right_wrist",)),
        ("left_thigh", "left_hip", "left_knee", ("left_knee",)),
        ("right_thigh", "right_hip", "right_knee", ("right_knee",)),
        ("left_shin", "left_knee", "left_ankle", ("left_ankle",)),
        ("right_shin", "right_knee", "right_ankle", ("right_ankle",)),
    )


# ------------------------------------------------------------------------------ it fires
def test_it_fires_on_a_symmetric_hip_collapse_and_charges_both_hips() -> None:
    """The defect this row exists for, reduced to its arithmetic.

    Both hips are pulled halfway to their own midpoint for eight frames. The hip line
    halves -- 50 % off the take median, far outside any ceiling -- and the hip line's two
    endpoints are both marked, exactly as the shoulder line's are.
    """

    collapsed = _collapse_hips(_take(), slice(10, 18), 0.5)
    withheld, actions, report = cm._reject_inconsistent_segments(
        collapsed, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)

    assert report["length_rejected_by_segment"].get("hip_line") == 8
    assert report["length_rejected_frames_by_segment"]["hip_line"] == list(range(10, 18))
    for name in ("left_hip", "right_hip"):
        joint = JOINT_INDEX[name]
        assert not np.isfinite(withheld[10:18, joint]).any(), f"{name} must be withheld"
        assert np.isfinite(withheld[:10, joint]).all(), "the honest frames stay"
        assert np.isfinite(withheld[18:, joint]).all(), "the honest frames stay"
    # `demote` is the shipped mode and it touches no ray.
    assert not actions["clear_rays"].any()
    assert not actions["keep_best_ray"].any()


def test_the_knees_are_not_dragged_in_by_a_hip_collapse() -> None:
    """A hip collapse makes both thighs read LONG. They must not read long ENOUGH.

    This is the cascade the real take does not show -- on frames 110-119 the thighs hold at
    391.9-421.8 mm against their own 401.8 / 406.2 medians -- and it is the reason the
    card's synthetic injection moves the hips along their own line rather than toward a
    landmark off it. The numbers are asserted so a future injection that lengthens the
    thighs cannot pass as this one.
    """

    values = _take()
    collapsed = _collapse_hips(values, slice(10, 18), 0.5)
    _withheld, _actions, report = cm._reject_inconsistent_segments(
        collapsed, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)

    fired = report["length_rejected_by_segment"]
    assert set(fired) == {"hip_line"}, f"only the hip line may fire, got {fired}"
    assert report["children_marked_under_a_marked_parent_segment"] == 0
    assert not ({"left_knee", "right_knee", "left_ankle", "right_ankle"}
                & set(report["length_rejected_by_joint"]))

    # And the reason, stated as a number rather than as a hope: the thighs DO grow, and by
    # how much is what keeps them under the ceiling.
    for parent, child, name in (("left_hip", "left_knee", "left_thigh"),
                                ("right_hip", "right_knee", "right_thigh")):
        before = np.linalg.norm(values[10:18, JOINT_INDEX[parent]]
                                - values[10:18, JOINT_INDEX[child]], axis=1)
        after = np.linalg.norm(collapsed[10:18, JOINT_INDEX[parent]]
                               - collapsed[10:18, JOINT_INDEX[child]], axis=1)
        growth = float(np.max(after / before - 1.0))
        assert growth > 0.0, f"{name} must read longer when its hip moves inward"
        assert growth < cm.SEGMENT_LENGTH_CEILING_FRACTION, (
            f"{name} grows {growth:.4f}, which would cascade at "
            f"{cm.SEGMENT_LENGTH_CEILING_FRACTION}")


# ---------------------------------------------------------------- the registered cost
def test_charging_both_endpoints_over_charges_a_one_hip_frame() -> None:
    """THE KNOWN OVER-CHARGE, asserted so it can never quietly stop being true.

    On the reference take's frames 84-86 one femoral head moves per frame and the other
    matches the take: frame 84 reads 84 mm from the root against a 127 mm right, frame 85
    reads 138 against 103. The hip line fires on those frames and, charging both endpoints,
    withholds the hip that was RIGHT along with the one that was wrong. Three frames on that
    take. The card registers it, `SEGMENT_LENGTH_RULES`' provenance entry registers it, and
    this is the test that keeps it honest.

    The alternative -- a per-hip `root->hip` rule, which would mark only the hip that
    moved -- was measured and refused on data: on the same honest mask `root->left_hip`
    spreads -9.0 %/+25.5 % at p5-p95 on the falling performer, so the pelvis landmark is too
    loose to be a length reference at this ceiling.
    """

    values = _take()
    one_hip = _collapse_hips(values, slice(10, 18), 0.5, sides=("left_hip",))
    withheld, _actions, report = cm._reject_inconsistent_segments(
        one_hip, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)

    assert report["length_rejected_by_segment"].get("hip_line") == 8
    right = JOINT_INDEX["right_hip"]
    # The right hip never moved -- and it is withheld anyway. That is the cost.
    assert np.allclose(one_hip[10:18, right], values[10:18, right])
    assert not np.isfinite(withheld[10:18, right]).any(), (
        "the both-endpoints convention withholds the hip that matched the take")
    assert not np.isfinite(withheld[10:18, JOINT_INDEX["left_hip"]]).any()


# --------------------------------------------------------------------------- it holds
def test_it_does_not_fire_on_an_honest_body() -> None:
    """No hip, leg, arm or shoulder fire on a clean take. The card's S3 must-fail."""

    _withheld, _actions, report = cm._reject_inconsistent_segments(
        _take(), ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_slots"] == 0
    assert report["length_rejected_by_segment"] == {}
    assert report["segment_median_m"]["hip_line"] is not None


def test_a_hip_line_inside_the_ceiling_does_not_fire() -> None:
    """The boundary, from both sides, on the segment the row adds.

    A rule that fired at 10 % would be a smoother wearing a reject's clothes, and one that
    did not fire at 20 % would not see the take's own defect.
    """

    inside = _collapse_hips(_take(), slice(10, 18), 0.90)     # 10 % narrower
    _w, _a, report = cm._reject_inconsistent_segments(
        inside, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert "hip_line" not in report["length_rejected_by_segment"]

    outside = _collapse_hips(_take(), slice(10, 18), 0.80)    # 20 % narrower
    _w, _a, report = cm._reject_inconsistent_segments(
        outside, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_by_segment"].get("hip_line") == 8


def test_it_fires_on_an_outward_stretch_too() -> None:
    """The rule is `|L - median| / median` and is symmetric by construction.

    The reference take needs both signs: frames 110-119 collapse inward and frame 113 spikes
    outward at 355 mm against a 215 mm median, and the whole of run 158-168 is an outward
    stretch along the A-C baseline.
    """

    stretched = _collapse_hips(_take(), slice(10, 18), 1.25)
    _w, _a, report = cm._reject_inconsistent_segments(
        stretched, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert report["length_rejected_by_segment"].get("hip_line") == 8


# -------------------------------------------------------------------- the blindnesses
def test_a_take_long_hip_collapse_is_invisible() -> None:
    """The reference is the performer's OWN median, so a take-long error moves it too.

    Not a defect to be fixed here: it is the founding blindness of a self-referential rule
    and it is why the card's own classification insists the fault is a MINORITY of the take.
    """

    every_frame = _collapse_hips(_take(), slice(None), 0.5)
    _w, _a, report = cm._reject_inconsistent_segments(
        every_frame, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert "hip_line" not in report["length_rejected_by_segment"]
    assert report["segment_median_m"]["hip_line"] is not None


def test_the_rule_cannot_score_direction_of_the_hip_line() -> None:
    """A LENGTH INVARIANT CANNOT SCORE DIRECTION (CLAUDE.md), on this segment too.

    Both hips are rotated 40 degrees about the vertical through their own midpoint. The
    pelvis now faces somewhere else entirely and the hip line is exactly as long as it was,
    so nothing fires. Nothing about D7's pelvis frame may be read off this row.
    """

    values = _take()
    turned = values.copy()
    angle = np.radians(40.0)
    rotation = np.asarray(((np.cos(angle), -np.sin(angle), 0.0),
                           (np.sin(angle), np.cos(angle), 0.0),
                           (0.0, 0.0, 1.0)))
    frames = slice(10, 18)
    midpoint = 0.5 * (values[frames, JOINT_INDEX["left_hip"]]
                      + values[frames, JOINT_INDEX["right_hip"]])
    for name in ("left_hip", "right_hip"):
        joint = JOINT_INDEX[name]
        turned[frames, joint] = midpoint + (values[frames, joint] - midpoint) @ rotation.T

    lengths_before = np.linalg.norm(values[frames, JOINT_INDEX["left_hip"]]
                                    - values[frames, JOINT_INDEX["right_hip"]], axis=1)
    lengths_after = np.linalg.norm(turned[frames, JOINT_INDEX["left_hip"]]
                                   - turned[frames, JOINT_INDEX["right_hip"]], axis=1)
    assert np.allclose(lengths_before, lengths_after), "the rotation preserves the length"
    assert not np.allclose(turned[frames, JOINT_INDEX["left_hip"]],
                           values[frames, JOINT_INDEX["left_hip"]]), "the hips did move"

    _w, _a, report = cm._reject_inconsistent_segments(
        turned, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert "hip_line" not in report["length_rejected_by_segment"]


def test_the_raw_array_is_never_modified() -> None:
    collapsed = _collapse_hips(_take(), slice(10, 18), 0.5)
    reference = collapsed.copy()
    cm._reject_inconsistent_segments(
        collapsed, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION)
    assert np.array_equal(collapsed, reference, equal_nan=True)


def test_too_few_measured_frames_means_no_judgement() -> None:
    """A median over a handful of frames is not a reference, and a rule with no reference
    does not get to reject anything -- on the hip line as on every other segment."""

    values = _take()
    values[5:, JOINT_INDEX["left_hip"]] = np.nan
    _w, _a, report = cm._reject_inconsistent_segments(
        values, ceiling_fraction=cm.SEGMENT_LENGTH_CEILING_FRACTION,
        minimum_samples=cm.MINIMUM_LIMB_SAMPLES)
    assert report["segment_frames_measured"]["hip_line"] == 5
    assert report["segment_median_m"]["hip_line"] is None
    assert "hip_line" not in report["length_rejected_by_segment"]
