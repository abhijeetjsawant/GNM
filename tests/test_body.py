from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from autoanim_gnm.acting import TICKS_PER_SECOND
from autoanim_gnm.body import (
    ATTACHMENT_SCHEMA_VERSION,
    BODY_TRACK_SCHEMA_VERSION,
    CANONICAL_HUMANOID,
    SKELETON_SCHEMA_VERSION,
    BodyTrack,
    BodyValidationError,
    HumanoidSkeleton,
    JointSpec,
    attachment_contract,
    canonical_inverse_bind_matrices,
    compile_body_track,
    forward_kinematics_positions,
    validate_skeleton,
)


def _beat(
    identifier: str,
    start: int,
    end: int,
    *,
    stance: str,
    gestures: list[str],
    energy: float,
    gaze: str,
    gaze_strength: float,
    preserve_feet: bool = True,
) -> dict:
    return {
        "id": identifier,
        "start_tick": start,
        "end_tick": end,
        "intent": "reassure",
        "valence": 0.25,
        "arousal": 0.55,
        "body": {
            "stance": stance,
            "gesture_tags": gestures,
            "energy": energy,
        },
        "face": {"expression_tags": ["restrained"], "intensity": 0.3},
        "gaze": {"target": gaze, "strength": gaze_strength},
        "constraints": {
            "preserve_lipsync": True,
            "preserve_foot_contacts": preserve_feet,
        },
    }


def _plan(*, preserve_second: bool = True) -> dict:
    return {
        "schema_version": "autoanim.acting-plan/1.0",
        "status": "ok",
        "summary": "Open reassurance, followed by a forward nod.",
        "beats": [
            _beat(
                "beat_0001",
                0,
                TICKS_PER_SECOND,
                stance="open",
                gestures=["open_palm", "small"],
                energy=0.4,
                gaze="away_left",
                gaze_strength=0.8,
            ),
            _beat(
                "beat_0002",
                TICKS_PER_SECOND,
                2 * TICKS_PER_SECOND,
                stance="forward",
                gestures=["head_nod"],
                energy=0.75,
                gaze="up",
                gaze_strength=0.6,
                preserve_feet=preserve_second,
            ),
        ],
        "diagnostics": [],
    }


def test_canonical_skeleton_is_parent_ordered_and_interchange_ready() -> None:
    validate_skeleton(CANONICAL_HUMANOID)
    contract = CANONICAL_HUMANOID.as_dict()

    assert contract["schema_version"] == SKELETON_SCHEMA_VERSION
    assert contract["interchange"]["master"] == "OpenUSD UsdSkel"
    assert contract["interchange"]["runtime"] == ["glTF 2.0 skin", "VRM 1.0 humanoid"]
    assert contract["coordinate_system"] == {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "+Z",
        "linear_unit": "meter",
        "rotation": "local quaternion [x,y,z,w]",
    }
    assert len(CANONICAL_HUMANOID.joints) == 25
    for index, joint in enumerate(CANONICAL_HUMANOID.joints):
        assert joint.parent < index
        if joint.parent >= 0:
            parent_path = CANONICAL_HUMANOID.joints[joint.parent].usd_path
            assert joint.usd_path == f"{parent_path}/{joint.name}"

    rest_rotations = np.zeros((25, 4), dtype=np.float32)
    rest_rotations[:, 3] = 1.0
    rest_positions = forward_kinematics_positions(np.zeros(3), rest_rotations)
    np.testing.assert_allclose(
        rest_positions[CANONICAL_HUMANOID.index("LeftFoot")],
        [-0.09, 0.05, 0.02],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        rest_positions[CANONICAL_HUMANOID.index("RightToes")],
        [0.09, 0.0, 0.16],
        atol=1e-7,
    )
    inverse_bind = canonical_inverse_bind_matrices()
    assert inverse_bind.shape == (25, 4, 4)
    assert not inverse_bind.flags.writeable
    bind = np.broadcast_to(np.eye(4), inverse_bind.shape).copy()
    bind[:, :3, 3] = rest_positions
    np.testing.assert_allclose(
        bind @ inverse_bind,
        np.broadcast_to(np.eye(4), inverse_bind.shape),
        atol=1e-7,
    )


def test_skeleton_validation_rejects_nonhierarchical_usd_path() -> None:
    joints = list(CANONICAL_HUMANOID.joints)
    joints[6] = JointSpec(
        name="Head",
        parent=5,
        rest_translation_m=(0.0, 0.10, 0.0),
        usd_path="Root/DetachedHead",
        vrm_humanoid="head",
    )
    with pytest.raises(BodyValidationError, match="UsdSkel"):
        validate_skeleton(HumanoidSkeleton(tuple(joints)))


def test_attachment_contract_makes_gnm_ownership_and_composition_explicit() -> None:
    contract = attachment_contract()
    assert contract["schema_version"] == ATTACHMENT_SCHEMA_VERSION
    assert contract["composition_order"] == [
        "body_local_base",
        "gnm_neck_head_additive",
        "gnm_eye_local",
        "animator_override_additive",
    ]
    assert contract["mappings"]["gnm.neck"]["translation"] == "body only"
    assert contract["mappings"]["gnm.head"]["rotation"] == "GNM additive after body base"
    assert contract["mappings"]["gnm.left_eye"]["rotation"].startswith("GNM only")
    assert contract["rules"]["lipsync_owner"].startswith("GNM speech pipeline")

    # Callers cannot mutate the process-global contract by editing a returned value.
    contract["rules"]["eye_owner"] = "someone else"
    assert attachment_contract()["rules"]["eye_owner"] == "GNM"


def test_compile_body_track_is_deterministic_numeric_and_tick_exact() -> None:
    first = compile_body_track(_plan(), duration_ticks=2 * TICKS_PER_SECOND)
    second = compile_body_track(_plan(), duration_ticks=2 * TICKS_PER_SECOND)

    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.ticks.dtype == np.int64
    assert first.ticks[0] == 0
    assert first.ticks[-1] == 2 * TICKS_PER_SECOND
    np.testing.assert_array_equal(np.diff(first.ticks), np.full(60, 1600, dtype=np.int64))
    assert first.local_rotations_xyzw.shape == (61, 25, 4)
    assert first.gnm_eye_rotations_xyzw.shape == (61, 2, 4)
    np.testing.assert_allclose(
        np.linalg.norm(first.local_rotations_xyzw, axis=2), 1.0, atol=2e-6
    )

    # Halfway through beat one, the open-palm arms and leftward gaze are active.
    midpoint = int(np.flatnonzero(first.ticks == TICKS_PER_SECOND // 2)[0])
    right_arm = CANONICAL_HUMANOID.index("RightUpperArm")
    assert not np.allclose(first.local_rotations_xyzw[midpoint, right_arm], [0, 0, 0, 1])
    assert first.gaze_direction_body[midpoint, 0] < -0.30
    assert first.gaze_strength[midpoint] == pytest.approx(0.8)
    assert not np.allclose(first.gnm_eye_rotations_xyzw[midpoint, 0], [0, 0, 0, 1])

    # Beat two produces a forward spine and a time-varying nod, not a held mock pose.
    spine = CANONICAL_HUMANOID.index("Spine")
    head = CANONICAL_HUMANOID.index("Head")
    second_frames = np.flatnonzero(
        (first.ticks >= TICKS_PER_SECOND + 8_000)
        & (first.ticks <= 2 * TICKS_PER_SECOND - 8_000)
    )
    assert np.max(np.abs(first.local_rotations_xyzw[second_frames, spine, 0])) > 0.02
    assert np.ptp(first.local_rotations_xyzw[second_frames, head, 0]) > 0.03
    assert np.max(first.gaze_direction_body[second_frames, 1]) > 0.10


def test_compiler_preserves_real_foot_and_toe_world_contacts() -> None:
    track = compile_body_track(_plan(), duration_ticks=2 * TICKS_PER_SECOND)
    positions = forward_kinematics_positions(
        track.root_translation_m,
        track.local_rotations_xyzw,
    )
    indices = [
        CANONICAL_HUMANOID.index(name)
        for name in ("LeftFoot", "LeftToes", "RightFoot", "RightToes")
    ]
    displacement = np.linalg.norm(positions[:, indices] - positions[0, indices], axis=2)

    assert track.foot_contacts.all()
    assert float(np.max(displacement)) < 1e-8
    lower_body = [
        CANONICAL_HUMANOID.index(name)
        for name in (
            "Hips",
            "LeftUpperLeg",
            "LeftLowerLeg",
            "LeftFoot",
            "LeftToes",
            "RightUpperLeg",
            "RightLowerLeg",
            "RightFoot",
            "RightToes",
        )
    ]
    expected = np.zeros((track.ticks.size, len(lower_body), 4), dtype=np.float32)
    expected[:, :, 3] = 1.0
    np.testing.assert_allclose(track.local_rotations_xyzw[:, lower_body], expected, atol=0.0)


def test_false_contact_intent_is_serialized_without_inventing_locomotion() -> None:
    track = compile_body_track(
        _plan(preserve_second=False), duration_ticks=2 * TICKS_PER_SECOND
    )
    second_beat = (track.ticks >= TICKS_PER_SECOND) & (
        track.ticks < 2 * TICKS_PER_SECOND
    )
    assert not track.foot_contacts[second_beat].any()
    assert track.foot_contacts[track.ticks < TICKS_PER_SECOND].all()
    assert np.all(track.root_translation_m == 0.0)


def test_track_serialization_round_trips_without_numeric_drift() -> None:
    track = compile_body_track(_plan(), duration_ticks=2 * TICKS_PER_SECOND)
    payload = track.as_dict()
    assert payload["schema_version"] == BODY_TRACK_SCHEMA_VERSION
    assert payload["source_plan_sha256"] == track.source_plan_sha256
    assert "no motion-capture reconstruction" in payload["limitations"]
    restored = BodyTrack.from_dict(json.loads(json.dumps(payload, allow_nan=False)))

    assert restored.canonical_json_bytes() == track.canonical_json_bytes()
    assert not restored.ticks.flags.writeable
    assert not restored.local_rotations_xyzw.flags.writeable

    payload["limitations"] = []
    with pytest.raises(BodyValidationError, match="limitations"):
        BodyTrack.from_dict(payload)


def test_track_validation_rejects_fractional_ticks_and_contact_sliding() -> None:
    track = compile_body_track(_plan(), duration_ticks=2 * TICKS_PER_SECOND)
    payload = track.as_dict()
    payload["ticks"][1] = 1600.5
    with pytest.raises(BodyValidationError, match="integer"):
        BodyTrack.from_dict(payload)

    rotations = track.local_rotations_xyzw.copy()
    hips = CANONICAL_HUMANOID.index("Hips")
    angle = 0.08
    rotations[15, hips] = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
    with pytest.raises(BodyValidationError, match="foot contact moved"):
        replace(track, local_rotations_xyzw=rotations)


@pytest.mark.parametrize("sample_rate", [0, 29, 1200])
def test_compiler_rejects_invalid_or_nonintegral_timebases(sample_rate: int) -> None:
    with pytest.raises(BodyValidationError, match="sample_rate"):
        compile_body_track(
            _plan(),
            duration_ticks=2 * TICKS_PER_SECOND,
            sample_rate_hz=sample_rate,
        )
