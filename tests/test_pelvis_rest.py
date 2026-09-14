"""D7c -- the pelvis on the rig's own rest. NEW FILE; no existing test is edited.

Four things are pinned here, and each exists because something could otherwise pass while
being wrong:

  * BIT-PARITY between the shipped branch and the instrument-side estimator the selector
    actually evaluated. Without it S selects one estimator and production ships another --
    a defect no band in this step could see, because every band scores the delivery.
  * CONTAINMENT, proved and not asserted: the four `SOMA77_REST_*` constants are DELETED
    from the module and the rig modes rebuild bit-identically, while mode C RAISES. The
    second half is the positive control -- a containment test that only shows "still works"
    cannot tell an unused constant from a constant on a path the test never took.
  * the GUARD's scope: it runs under the two rig modes and never under A/B/C, which is what
    makes the refactor tripwire's bit-identity claim true.
  * the guard's FROZEN denominator, including the NaN trap: `np.median` over an array
    carrying NaN returns NaN, which would make the ceiling infinite and the mask empty -- a
    silent pass.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _relative in ("src", "tools/compare", "tools/swap-harness"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import DETAILED_HUMANOID  # noqa: E402

SOMA_CONSTANTS = ("SOMA77_REST_PELVIS_UP", "SOMA77_REST_HIPMID_TO_SPINE1",
                  "SOMA77_REST_HIP_ACROSS", "SOMA77_REST_PELVIS_TEMPLATE_M")
FRAMES = 40


def _rest() -> dict[str, np.ndarray]:
    return {joint.name: np.asarray(joint.rest_translation_m, dtype=np.float64)
            for joint in DETAILED_HUMANOID.joints}


def _observation(seed: int = 20260914):
    """A moving pelvis with noise on the three points every arm reads.

    Not the selector's fixture -- that one needs the cameras and the real triangulator. This
    is a self-contained stand-in whose only job is to exercise both branches, the guard and
    the interpolation on data that is not degenerate.
    """
    rng = np.random.default_rng(seed)
    rest = _rest()
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    points = np.zeros((FRAMES, len(cm.JOINT_NAMES), 3))
    spine = np.zeros((FRAMES, 3))
    for frame in range(FRAMES):
        angle = 0.6 * np.sin(frame / 7.0)
        lean = 0.35 * np.cos(frame / 5.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle) * np.sin(lean), np.sin(angle) * np.cos(lean)],
            [0.0, np.cos(lean), np.sin(lean)],
            [-np.sin(angle), -np.cos(angle) * np.sin(lean), np.cos(angle) * np.cos(lean)]])
        origin = np.array([0.02 * frame, 0.9, 0.01 * frame])
        noise = lambda: rng.normal(0.0, 0.004, 3)  # noqa: E731
        points[frame, cm.JOINT_INDEX["left_hip"]] = (
            origin + rotation @ (rest["LeftUpperLeg"] - mid) + noise())
        points[frame, cm.JOINT_INDEX["right_hip"]] = (
            origin + rotation @ (rest["RightUpperLeg"] - mid) + noise())
        points[frame, cm.JOINT_INDEX["root"]] = origin + noise()
        points[frame, cm.JOINT_INDEX["neck"]] = origin + rotation @ np.array([0.0, 0.5, 0.0])
        spine[frame] = origin + rotation @ (rest["Spine"] - mid) + noise()
    return points, spine, rest


# ------------------------------------------------------------------------- bit-parity
@pytest.mark.parametrize("mode", cm.RIG_REST_PELVIS_MODES)
def test_shipped_branch_is_bit_identical_to_the_estimator_the_selector_evaluated(mode):
    """S chose between two estimators written under `tools/compare/`. Production must run
    the same arithmetic, to the bit, or the selection means nothing."""
    import d7c_pelvis_estimators as est

    points, spine, rest = _observation()
    shipped, shipped_report = cm._pelvis_world_frames(points, spine, rest=rest, mode=mode)
    instrument, instrument_report = est.rig_rest_pelvis_frames(
        points, spine, rest, mode=mode, guard=True)
    assert np.array_equal(shipped, instrument)
    assert (shipped_report["lever_guard"]["demoted_frames"]
            == instrument_report["lever_guard"]["demoted_frames"])


# ------------------------------------------------------------------------- containment
def test_the_four_soma_constants_are_absent_from_the_rig_modes_data_flow():
    """DELETE them and rebuild. Bit-identical under the rig modes; mode C must RAISE.

    The second assertion is the positive control: without it this test cannot distinguish
    a constant that is genuinely unused from one on a path it never exercises.
    """
    points, spine, rest = _observation()
    before = {mode: cm._pelvis_world_frames(points, spine, rest=rest, mode=mode)[0]
              for mode in cm.RIG_REST_PELVIS_MODES}
    saved = {name: getattr(cm, name) for name in SOMA_CONSTANTS}
    try:
        for name in SOMA_CONSTANTS:
            delattr(cm, name)
        for mode in cm.RIG_REST_PELVIS_MODES:
            after, _ = cm._pelvis_world_frames(points, spine, rest=rest, mode=mode)
            assert np.array_equal(before[mode], after), mode
        with pytest.raises(NameError):
            cm._pelvis_world_frames(points, spine, rest=rest, mode="C_kabsch_pelvis")
    finally:
        for name, value in saved.items():
            setattr(cm, name, value)


def test_the_missing_data_path_also_avoids_the_constants():
    """Containment must hold on the gap path too, not only on the fully-resolved one."""
    points, spine, rest = _observation()
    spine = spine.copy()
    spine[[3, 4, 11, 27]] = np.nan
    before, _ = cm._pelvis_world_frames(points, spine, rest=rest,
                                        mode="E_rig_rest_kabsch")
    saved = {name: getattr(cm, name) for name in SOMA_CONSTANTS}
    try:
        for name in SOMA_CONSTANTS:
            delattr(cm, name)
        after, report = cm._pelvis_world_frames(points, spine, rest=rest,
                                                mode="E_rig_rest_kabsch")
    finally:
        for name, value in saved.items():
            setattr(cm, name, value)
    assert np.array_equal(before, after)
    assert report["interpolated_frames"] >= 4


# ------------------------------------------------------------------------- the guard
def test_the_guard_runs_under_both_rig_modes_and_never_under_a_b_or_c():
    """Its scope is what makes the refactor tripwire's bit-identity claim true."""
    points, spine, rest = _observation()
    spine = spine.copy()
    spine[9] *= 0.5                      # a lever far off the subject's own median
    for mode in cm.RIG_REST_PELVIS_MODES:
        _, report = cm._pelvis_world_frames(points, spine, rest=rest, mode=mode)
        assert report["lever_guard"]["applied"] is True
        assert 9 in report["lever_guard"]["demoted_frames"]
    for mode in ("A_root_to_spine1", "B_hipmid_to_spine1", "C_kabsch_pelvis"):
        _, report = cm._pelvis_world_frames(points, spine, rest=rest, mode=mode)
        assert report["lever_guard"]["applied"] is False
        assert report["lever_guard"]["demoted_frames"] == []


def test_a_rig_mode_without_a_rest_skeleton_raises_rather_than_inventing_one():
    points, spine, _ = _observation()
    with pytest.raises(cm.CommercialMultiviewError):
        cm._pelvis_world_frames(points, spine, mode="E_rig_rest_kabsch")


def test_the_guards_denominator_is_frozen_and_survives_nan():
    """`np.median` over an array carrying NaN returns NaN -- an infinite ceiling and an
    EMPTY mask, which is a silent pass. The finite mask is explicit, and this pins it."""
    points, spine, _ = _observation()
    hip_mid = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                     + points[:, cm.JOINT_INDEX["right_hip"]])
    reference = float(np.median(np.linalg.norm(spine - hip_mid, axis=1)))
    holed = spine.copy()
    holed[[1, 2, 3]] = np.nan
    _, report = cm._pelvis_lever_guard(holed, hip_mid)
    assert report["pre_guard_median_mm"] is not None
    assert np.isfinite(report["pre_guard_median_mm"])
    assert abs(report["pre_guard_median_mm"] - 1e3 * reference) < 2.0
    assert report["finite_frames_before_the_guard"] == len(spine) - 3


def test_the_guard_never_writes_the_callers_array():
    points, spine, _ = _observation()
    spine = spine.copy()
    spine[5] *= 0.4
    original = spine.copy()
    hip_mid = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                     + points[:, cm.JOINT_INDEX["right_hip"]])
    masked, report = cm._pelvis_lever_guard(spine, hip_mid)
    assert np.array_equal(spine, original)
    assert not np.isfinite(masked[5]).all()
    assert 5 in report["demoted_frames"]


def test_the_shipping_mode_is_the_one_the_selector_chose():
    assert cm.PELVIS_FRAME_SOURCE == "E_rig_rest_kabsch"
    assert cm.PELVIS_FRAME_SOURCE in cm.RIG_REST_PELVIS_MODES


# ---------------------------------------------------------------------------------------
# THE FOUR STRUCTURAL PINS D7c SUPERSEDES, RE-PINNED HERE.
#
# `tests/test_pelvis_frame.py` is D7's record and is NOT edited (the D9b precedent: two pins
# it superseded were re-pinned in a new file, never in the file they came from). Four of its
# tests fail under D7c, in two distinct classes, and the distinction is the whole point:
#
#   ONE IS MOVED BY DESIGN, and the card pre-registered it. `round_trips_the_pelvis_frame`
#   asserts the converter recovers the posed pelvis to 0.01 deg. Its body is SOMASKEL77
#   POSED, so its truth pelvis IS SOMA-77's convention -- the very constant D7c removes --
#   and under `E_rig_rest_kabsch` it reads 7.568 deg. That is the card's "D7's own
#   instruments will read this WORSE by ~7 deg by construction", realised to the degree. It
#   is NOT a regression, it is NOT a band, and the re-pin below asserts the SAME exactness
#   against the reference D7c is actually built on: the RIG's own rest.
#
#   THREE ARE SIGNATURE PINS AND NOTHING MORE. They call `_pelvis_world_frames` by bare name
#   with no `rest`, and the default mode is now a rig mode, which raises rather than
#   inventing a rest. The BEHAVIOUR they pin -- gap interpolation, the whole-subject
#   fallback, the smoothing knob's effect -- is unchanged, and each is re-pinned below with
#   the rest supplied.
# ---------------------------------------------------------------------------------------
def _posed_rig_body(frames: int = 30, wobble_m: float = 0.0, seed: int = 7):
    """A rig body posed by a known rotation: the reference D7c is built on.

    The difference from `test_pelvis_frame.py`'s `posed_body` is the whole subject of this
    step -- that fixture poses SOMA-77's rest pelvis, this one poses the RIG's.
    """
    rng = np.random.default_rng(seed)
    rest = _rest()
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    points = np.zeros((frames, len(cm.JOINT_NAMES), 3))
    spine = np.zeros((frames, 3))
    truth = np.zeros((frames, 3, 3))
    for frame in range(frames):
        yaw, pitch = 0.5 * np.sin(frame / 6.0), 0.4 * np.cos(frame / 4.0)
        rotation = (np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0],
                              [-np.sin(yaw), 0.0, np.cos(yaw)]])
                    @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)],
                                [0.0, np.sin(pitch), np.cos(pitch)]]))
        truth[frame] = rotation
        origin = np.array([0.015 * frame, 0.95, -0.01 * frame])
        jitter = (lambda: rng.normal(0.0, wobble_m, 3)) if wobble_m else (lambda: 0.0)
        points[frame, cm.JOINT_INDEX["left_hip"]] = (
            origin + rotation @ (rest["LeftUpperLeg"] - mid) + jitter())
        points[frame, cm.JOINT_INDEX["right_hip"]] = (
            origin + rotation @ (rest["RightUpperLeg"] - mid) + jitter())
        points[frame, cm.JOINT_INDEX["root"]] = origin
        points[frame, cm.JOINT_INDEX["neck"]] = origin + rotation @ np.array([0.0, 0.5, 0.0])
        spine[frame] = origin + rotation @ (rest["Spine"] - mid) + jitter()
    return points, spine, truth, rest


def _geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,nkj->nik", a, b)
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def test_repin_the_exact_recovery_oracle_against_the_rigs_own_rest():
    """D7's oracle, moved by design onto the reference this step is built on.

    D7's version poses SOMA-77's rest and reads 7.568 deg here, exactly as the card
    pre-registered. Posed on the RIG's rest, BOTH rig modes are exact.
    """
    from scipy.spatial.transform import Rotation

    points, spine, truth, rest = _posed_rig_body()
    for mode in cm.RIG_REST_PELVIS_MODES:
        quaternions, _ = cm._pelvis_world_frames(points, spine, rest=rest, mode=mode)
        error = _geodesic_deg(Rotation.from_quat(quaternions).as_matrix(), truth)
        assert error.max() < 0.01, f"{mode}: worst {error.max():.6f} deg"


def test_repin_the_soma_posed_oracle_reads_the_conventions_own_size_and_that_is_expected():
    """The SAME body D7's oracle poses, scored under the shipping mode: MOVED BY DESIGN.

    Pinned as a NUMBER so a future change that moves it for a different reason is visible.
    The SOMA convention's own pelvis sits ~7 deg off the rig's `Hips`->`Spine` axis, and
    that is the whole quantity D7c removes from the delivery.
    """
    from scipy.spatial.transform import Rotation

    points, spine, truth, rest = _posed_rig_body()
    # pose the SPINE point by SOMA's convention instead of the rig's: the same 6.87 deg
    # tilt the shipped template carries, applied about the hip line.
    tilt = np.radians(6.87)
    about_hip_line = np.array([[1.0, 0.0, 0.0],
                               [0.0, np.cos(tilt), -np.sin(tilt)],
                               [0.0, np.sin(tilt), np.cos(tilt)]])
    mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])
    soma_spine = np.stack([
        points[f, cm.JOINT_INDEX["root"]]
        + truth[f] @ about_hip_line @ (rest["Spine"] - mid) for f in range(len(spine))])
    quaternions, _ = cm._pelvis_world_frames(points, soma_spine, rest=rest,
                                             mode=cm.PELVIS_FRAME_SOURCE)
    error = _geodesic_deg(Rotation.from_quat(quaternions).as_matrix(), truth)
    assert 5.0 < np.median(error) < 9.0, f"median {np.median(error):.3f} deg"


def test_repin_a_gap_is_interpolated_and_the_definition_never_switches_per_frame():
    """D7's pin, unchanged in substance; only the `rest` argument is new."""
    from scipy.spatial.transform import Rotation

    points, spine, _, rest = _posed_rig_body()
    holed = spine.copy()
    holed[5:9] = np.nan
    quaternions, report = cm._pelvis_world_frames(points, holed, rest=rest)
    assert report["status"] == "solved"
    assert report["interpolated_frames"] == 4
    assert quaternions is not None and np.isfinite(quaternions).all()
    steps = _geodesic_deg(Rotation.from_quat(quaternions[1:]).as_matrix(),
                          Rotation.from_quat(quaternions[:-1]).as_matrix())
    assert steps.max() < 10.0, f"largest step {steps.max():.3f} deg"


def test_repin_too_few_resolved_frames_falls_the_WHOLE_subject_back_with_a_reason():
    """D7's pin, unchanged in substance; only the `rest` argument is new."""
    points, spine, _, rest = _posed_rig_body()
    holed = spine.copy()
    holed[: int(0.8 * len(holed))] = np.nan
    quaternions, report = cm._pelvis_world_frames(points, holed, rest=rest)
    assert quaternions is None
    assert report["status"] == "fell_back_to_torso_frame"
    assert "PELVIS_MINIMUM_RESOLVED_FRACTION" in report["reason"]
    assert report["lever_guard"]["applied"] is True


def test_repin_a_smoothing_window_is_a_knob_and_it_moves_the_answer():
    """D7's pin, unchanged in substance; only the `rest` argument is new."""
    from scipy.spatial.transform import Rotation

    points, spine, _, rest = _posed_rig_body(wobble_m=0.01)
    base, _ = cm._pelvis_world_frames(points, spine, rest=rest, smoothing_frames=0)
    wide, _ = cm._pelvis_world_frames(points, spine, rest=rest, smoothing_frames=9)
    moved = _geodesic_deg(Rotation.from_quat(wide).as_matrix(),
                          Rotation.from_quat(base).as_matrix())
    assert moved.max() > 0.05
