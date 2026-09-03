"""D3 -- one rest skeleton per performer, serialised on the track.

Schema must-fails and the propagation contract.  NEW file: the brief for D3 forbids
editing an existing test.  Every test asserts it is importing the worktree's package,
because the venv's editable install points at the main tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import autoanim_gnm
from autoanim_gnm.body import (
    DETAILED_HUMANOID,
    BodyTrack,
    BodyValidationError,
    forward_kinematics_positions,
    skeleton_for_joint_names,
    skeleton_for_track,
    skeleton_for_track_dict,
    validate_body_track,
)
from autoanim_gnm.performer_skeleton import performer_skeleton

TICKS = 48_000


def _track(rest=None, frames: int = 3) -> BodyTrack:
    joints = len(DETAILED_HUMANOID.joints)
    rotations = np.zeros((frames, joints, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    eyes = np.zeros((frames, 2, 4), dtype=np.float32)
    eyes[..., 3] = 1.0
    ticks = np.arange(frames, dtype=np.int64) * (TICKS // 30)
    return BodyTrack(
        duration_ticks=int(ticks[-1]),
        ticks_per_second=TICKS,
        sample_rate_hz=30,
        joint_names=DETAILED_HUMANOID.names,
        ticks=ticks,
        root_translation_m=np.zeros((frames, 3), dtype=np.float32),
        local_rotations_xyzw=rotations,
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=np.broadcast_to(np.asarray((0.0, 0.0, 1.0), np.float32), (frames, 3)),
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=eyes,
        source_plan_sha256="0" * 64,
        **({} if rest is None else {"rest_translations_m": rest}),
    )


def _sized_rest() -> np.ndarray:
    rest = np.array(DETAILED_HUMANOID.rest_translations_m, dtype=np.float64)
    for name, factor in (("LeftLowerArm", 0.8), ("RightLowerArm", 1.2), ("LeftLowerLeg", 0.9),
                         ("RightLowerLeg", 1.1), ("Spine", 0.85), ("LeftUpperLeg", 1.15)):
        rest[DETAILED_HUMANOID.index(name)] *= factor
    return rest


def test_the_tests_run_against_the_worktree_package():
    here = Path(__file__).resolve().parents[1]
    assert Path(autoanim_gnm.__file__).resolve().is_relative_to(here / "src"), autoanim_gnm.__file__


def test_a_track_without_a_rest_is_the_canonical_body_by_identity():
    track = _track()
    assert skeleton_for_track(track) is DETAILED_HUMANOID
    assert np.array_equal(track.rest_translations_m, DETAILED_HUMANOID.rest_translations_m)
    assert track.rest_translations_m.dtype == np.float64


def test_a_sized_track_carries_its_own_rest_bit_for_bit():
    rest = _sized_rest()
    track = _track(rest)
    own = skeleton_for_track(track)
    assert own is not DETAILED_HUMANOID
    assert own.names == DETAILED_HUMANOID.names
    assert [j.parent for j in own.joints] == [j.parent for j in DETAILED_HUMANOID.joints]
    assert np.array_equal(own.rest_translations_m, rest)          # float64, no rounding


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 3)),                                              # wrong shape
        np.full((len(DETAILED_HUMANOID.joints), 3), np.nan),            # non-finite
        np.array(DETAILED_HUMANOID.rest_translations_m) + np.array([0.0, 0.01, 0.0]),  # root moved
    ],
    ids=["wrong-shape", "non-finite", "non-zero-root"],
)
def test_schema_must_fails_are_rejected(bad):
    with pytest.raises(BodyValidationError):
        _track(bad)


def test_serialisation_round_trips_bit_identically_for_canonical_and_sized_tracks():
    for rest in (None, _sized_rest()):
        track = _track(rest)
        doc = json.loads(track.canonical_json_bytes().decode("utf-8"))
        back = BodyTrack.from_dict(doc)
        assert np.array_equal(back.rest_translations_m, track.rest_translations_m)
        assert back.canonical_json_bytes() == track.canonical_json_bytes()
        assert skeleton_for_track_dict(doc).names == DETAILED_HUMANOID.names
        assert np.array_equal(skeleton_for_track_dict(doc).rest_translations_m,
                              track.rest_translations_m)
    sized_doc = _track(_sized_rest()).as_dict()
    assert "rest_translations_m" in sized_doc
    assert sized_doc["schema_version"] != _track().as_dict()["schema_version"]


def test_legacy_json_without_a_rest_loads_as_the_canonical_body():
    doc = _track().as_dict()
    doc.pop("rest_translations_m", None)
    back = BodyTrack.from_dict(doc)
    assert skeleton_for_track(back) is DETAILED_HUMANOID
    assert skeleton_for_track_dict(doc) is DETAILED_HUMANOID


def test_a_sized_track_is_validated_on_its_own_rest_not_the_canonical_one():
    """The contact-anchor check runs forward kinematics; a foot planted on one body
    is not planted on another, so validating a sized track against the canonical
    skeleton must be a contract violation, not a silent pass."""

    rest = _sized_rest()
    track = _track(rest)
    validate_body_track(track)                                         # its own rest: fine
    with pytest.raises(BodyValidationError):
        validate_body_track(track, skeleton=DETAILED_HUMANOID)         # a different body


def test_the_defect_pattern_still_returns_canonical_and_is_therefore_wrong_for_a_sized_track():
    track = _track(_sized_rest())
    canonical = skeleton_for_joint_names(track.joint_names)
    assert canonical is DETAILED_HUMANOID                              # the old pattern
    fk_wrong = forward_kinematics_positions(track.root_translation_m, track.local_rotations_xyzw,
                                            skeleton=canonical)
    fk_right = forward_kinematics_positions(track.root_translation_m, track.local_rotations_xyzw,
                                            skeleton=skeleton_for_track(track))
    assert np.abs(fk_wrong - fk_right).max() > 0.01                    # tens of mm, not noise


def test_performer_skeleton_is_a_derivation_the_root_and_hierarchy_untouched():
    """Bone lengths come from the positions handed in; Root stays zero; names and
    parents stay the contract.  The torso chain sums to the measured hip-midpoint->neck
    span MINUS the skeleton's own leg-root drop (read from rest, no constant)."""

    from autoanim_gnm.commercial_multiview import JOINT_INDEX, JOINT_NAMES

    rng = np.random.default_rng(20260903)
    frames = 40
    pos = np.zeros((frames, len(JOINT_NAMES), 3))
    # a T-posed body in capture Z-up metres, 0.9x canonical, with 5 mm jitter
    layout = {
        "left_hip": (0.09, 0.0, 0.90), "right_hip": (-0.09, 0.0, 0.90),
        "left_knee": (0.09, 0.0, 0.51), "right_knee": (-0.09, 0.0, 0.51),
        "left_ankle": (0.09, 0.0, 0.13), "right_ankle": (-0.09, 0.0, 0.13),
        "neck": (0.0, 0.0, 1.45),
        "left_shoulder": (0.17, 0.0, 1.43), "right_shoulder": (-0.17, 0.0, 1.43),
        "left_elbow": (0.42, 0.0, 1.43), "right_elbow": (-0.42, 0.0, 1.43),
        "left_wrist": (0.66, 0.0, 1.43), "right_wrist": (-0.66, 0.0, 1.43),
    }
    for name in JOINT_NAMES:
        base = layout.get(name, (0.0, 0.0, 1.55))
        pos[:, JOINT_INDEX[name]] = np.asarray(base) + rng.normal(0.0, 0.005, (frames, 3))
    sized, report = performer_skeleton(DETAILED_HUMANOID, pos)
    assert sized.names == DETAILED_HUMANOID.names
    assert [j.parent for j in sized.joints] == [j.parent for j in DETAILED_HUMANOID.joints]
    rest = sized.rest_translations_m
    assert np.all(rest[0] == 0.0)
    shin = np.linalg.norm(rest[DETAILED_HUMANOID.index("LeftFoot")])
    assert abs(shin - 0.38) < 0.02                                    # knee->ankle, 0.38 m
    torso = sum(np.linalg.norm(rest[DETAILED_HUMANOID.index(n)])
                for n in ("Spine", "Chest", "UpperChest", "Neck"))
    drop = -0.5 * (rest[DETAILED_HUMANOID.index("LeftUpperLeg")][1]
                   + rest[DETAILED_HUMANOID.index("RightUpperLeg")][1])
    assert abs(torso + drop - 0.55) < 0.02                            # hip-mid->neck, 0.55 m
    assert isinstance(report, dict)
