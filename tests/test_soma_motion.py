from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from autoanim_gnm.body import CANONICAL_HUMANOID, forward_kinematics_positions
from autoanim_gnm.soma_motion import (
    GEM_X_CONTACT_NAMES,
    GEM_X_CONTACT_SCHEMA_ID,
    SOMASKEL77_NAMES,
    SOMASKEL77_PARENTS,
    SomaMotion,
    SomaMotionValidationError,
    canonicalize_soma_arrays,
    project_soma_to_body_track,
    soma_forward_kinematics,
)


GEM_COMMIT = "32992550dba114c62243fb55e361311972dce8f9"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _rest_positions() -> np.ndarray:
    positions = np.zeros((77, 3), dtype=np.float32)
    offsets = np.zeros((77, 3), dtype=np.float32)
    offsets[:, 1] = 0.04
    offsets[0] = 0.0
    offsets[SOMASKEL77_NAMES.index("LeftShoulder")] = (-0.10, 0.05, 0.0)
    offsets[SOMASKEL77_NAMES.index("RightShoulder")] = (0.10, 0.05, 0.0)
    offsets[SOMASKEL77_NAMES.index("LeftLeg")] = (-0.08, -0.10, 0.0)
    offsets[SOMASKEL77_NAMES.index("RightLeg")] = (0.08, -0.10, 0.0)
    for index, parent in enumerate(SOMASKEL77_PARENTS):
        if parent >= 0:
            positions[index] = positions[parent] + offsets[index]
    return positions


def _motion(*, frames: int = 3) -> SomaMotion:
    ticks = np.linspace(0, 3200, frames, dtype=np.int64)
    rotations = np.zeros((frames, 77, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    roots = np.zeros((frames, 3), dtype=np.float32)
    roots[:, 0] = np.linspace(0.0, 0.1, frames)
    rest = _rest_positions()
    rest_world_rotations = np.zeros((77, 4), dtype=np.float32)
    rest_world_rotations[:, 3] = 1.0
    joints = soma_forward_kinematics(roots, rotations, rest, rest_world_rotations)
    return SomaMotion(
        provider_id="nvidia_gem_x",
        provider_git_commit_oid=GEM_COMMIT,
        operation="video_capture",
        motion_kind="observed",
        duration_ticks=3200,
        sample_rate_hz=30,
        ticks=ticks,
        source_pts=np.arange(frames, dtype=np.int64),
        root_translation_m=roots,
        local_rotations_xyzw=rotations,
        rest_joint_positions_m=rest,
        rest_world_rotations_xyzw=rest_world_rotations,
        joint_positions_m=joints,
        contacts=np.zeros((frames, 6), dtype=np.bool_),
        contact_schema_id=GEM_X_CONTACT_SCHEMA_ID,
        contact_names=GEM_X_CONTACT_NAMES,
        source_handedness="right",
        source_up_axis="+Y",
        source_forward_axis="+Z",
        source_linear_unit_in_meters=1.0,
        source_to_canonical_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_time_base_numerator=1,
        source_time_base_denominator=30,
        input_sha256=SHA_A,
        provider_raw_motion_sha256=SHA_B,
    )


def test_somaskel77_contract_is_exact_and_parent_ordered() -> None:
    assert len(SOMASKEL77_NAMES) == 77
    assert len(set(SOMASKEL77_NAMES)) == 77
    assert SOMASKEL77_NAMES[0] == "Hips"
    assert SOMASKEL77_NAMES[69:77] == (
        "LeftFoot",
        "LeftToeBase",
        "LeftToeEnd",
        "RightLeg",
        "RightShin",
        "RightFoot",
        "RightToeBase",
        "RightToeEnd",
    )
    assert SOMASKEL77_PARENTS[0] == -1
    assert all(parent < index for index, parent in enumerate(SOMASKEL77_PARENTS))


def test_soma_motion_is_immutable_deterministic_and_fk_consistent() -> None:
    track = _motion()
    assert track.manifest_dict()["production_validated"] is False
    assert track.manifest_dict()["coordinate_system"]["up_axis"] == "+Y"
    assert track.content_sha256() == _motion().content_sha256()
    assert not track.local_rotations_xyzw.flags.writeable
    with pytest.raises(ValueError):
        track.root_translation_m[0, 0] = 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider_git_commit_oid", "main", "Git object ID"),
        ("provider_git_commit_oid", "0" * 40, "allowlisted"),
        ("input_sha256", "bad", "SHA-256"),
        ("operation", "acting_generation", "inconsistent"),
        ("motion_kind", "generated", "inconsistent"),
        ("contact_names", ("left",) * 6, "contact schema"),
        ("duration_ticks", 30 * 60 * 48_000 + 1, "30-minute"),
        ("source_time_base_denominator", 0, "timebase"),
        ("source_handedness", "left", "right-handed"),
        ("source_up_axis", "+Z", "right-handed"),
        ("source_linear_unit_in_meters", 0.01, "meters"),
        (
            "source_to_canonical_rotation_xyzw",
            (0.0, 0.0, 1.0, 0.0),
            "identity canonicalization",
        ),
    ],
)
def test_soma_motion_rejects_provenance_and_semantic_confusion(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(SomaMotionValidationError, match=message):
        replace(_motion(), **{field: value})


def test_soma_motion_rejects_quaternion_sign_flips_and_bad_fk() -> None:
    source = _motion()
    rotations = source.local_rotations_xyzw.copy()
    rotations[1, 3] *= -1.0
    with pytest.raises(SomaMotionValidationError, match="signs"):
        replace(source, local_rotations_xyzw=rotations)

    positions = source.joint_positions_m.copy()
    positions[1, 42, 0] += 0.01
    with pytest.raises(SomaMotionValidationError, match="FK"):
        replace(source, joint_positions_m=positions)

    with pytest.raises(SomaMotionValidationError, match="source PTS"):
        replace(source, source_pts=np.array((0, 1, 3), dtype=np.int64))


def test_source_pts_support_nonzero_start_and_exact_ntsc_timing() -> None:
    source = _motion()
    pts = np.array((90_000, 93_003, 96_006), dtype=np.int64)
    ticks = np.array((0, 1602, 3203), dtype=np.int64)
    track = replace(
        source,
        source_pts=pts,
        source_time_base_numerator=1,
        source_time_base_denominator=90_000,
        ticks=ticks,
        duration_ticks=3203,
    )
    np.testing.assert_array_equal(track.ticks, ticks)


def test_projection_preserves_body_motion_but_not_face_eye_or_raw_contacts() -> None:
    source = _motion()
    rotations = source.local_rotations_xyzw.copy()
    angle = np.deg2rad(20.0)
    rotations[:, SOMASKEL77_NAMES.index("LeftArm"), 2] = np.sin(angle / 2.0)
    rotations[:, SOMASKEL77_NAMES.index("LeftArm"), 3] = np.cos(angle / 2.0)
    rotations[:, SOMASKEL77_NAMES.index("LeftEye"), 1] = np.sin(angle / 2.0)
    rotations[:, SOMASKEL77_NAMES.index("LeftEye"), 3] = np.cos(angle / 2.0)
    joints = soma_forward_kinematics(
        source.root_translation_m,
        rotations,
        source.rest_joint_positions_m,
        source.rest_world_rotations_xyzw,
    )
    contacts = source.contacts.copy()
    contacts[:, :] = True
    source = replace(
        source,
        local_rotations_xyzw=rotations,
        joint_positions_m=joints,
        contacts=contacts,
    )

    projected = project_soma_to_body_track(source)
    body_left_arm = CANONICAL_HUMANOID.index("LeftUpperArm")
    np.testing.assert_allclose(
        projected.local_rotations_xyzw[:, body_left_arm],
        rotations[:, SOMASKEL77_NAMES.index("LeftArm")],
        atol=1e-7,
    )
    assert not projected.foot_contacts.any()
    np.testing.assert_allclose(projected.gnm_eye_rotations_xyzw[..., :3], 0.0)
    np.testing.assert_allclose(projected.gnm_eye_rotations_xyzw[..., 3], 1.0)
    assert projected.source_plan_sha256 == source.content_sha256()


def test_projection_matches_source_hips_world_transform_with_heading() -> None:
    source = _motion()
    roots = source.root_translation_m.copy()
    roots[:] = (1.25, 0.92, -2.0)
    rotations = source.local_rotations_xyzw.copy()
    half_yaw = np.deg2rad(35.0) / 2.0
    rotations[:, 0] = (0.0, np.sin(half_yaw), 0.0, np.cos(half_yaw))
    joints = soma_forward_kinematics(
        roots,
        rotations,
        source.rest_joint_positions_m,
        source.rest_world_rotations_xyzw,
    )
    source = replace(
        source,
        root_translation_m=roots,
        local_rotations_xyzw=rotations,
        joint_positions_m=joints,
    )
    projected = project_soma_to_body_track(source)
    target_positions = forward_kinematics_positions(
        projected.root_translation_m, projected.local_rotations_xyzw
    )
    np.testing.assert_allclose(
        target_positions[:, CANONICAL_HUMANOID.index("Hips")], roots, atol=1e-6
    )
    np.testing.assert_allclose(
        projected.local_rotations_xyzw[:, CANONICAL_HUMANOID.index("Root")],
        rotations[:, 0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        projected.local_rotations_xyzw[:, CANONICAL_HUMANOID.index("Hips")],
        np.broadcast_to((0.0, 0.0, 0.0, 1.0), (source.ticks.size, 4)),
        atol=1e-7,
    )


def test_explicit_axis_and_unit_conversion_canonicalizes_arrays() -> None:
    source = _motion()
    # Source is centimeters, Z-up. -90 degrees about X maps +Z to +Y.
    half = -np.pi / 4.0
    conversion = np.array((np.sin(half), 0.0, 0.0, np.cos(half)), dtype=np.float32)
    roots = np.array([[100.0, 200.0, 90.0]], dtype=np.float32)
    local = np.zeros((1, 77, 4), dtype=np.float32)
    local[..., 3] = 1.0
    local[:, SOMASKEL77_NAMES.index("LeftArm")] = (
        0.0,
        np.sin(np.deg2rad(15.0) / 2.0),
        0.0,
        np.cos(np.deg2rad(15.0) / 2.0),
    )
    rest = np.zeros((77, 3), dtype=np.float32)
    rest_world = np.zeros((77, 4), dtype=np.float32)
    rest_world[:, 3] = 1.0
    joints = np.broadcast_to(roots[:, None, :], (1, 77, 3))
    (
        canonical_root,
        canonical_local,
        canonical_rest,
        canonical_rest_world,
        canonical_joints,
    ) = (
        canonicalize_soma_arrays(
            root_translation=roots,
            local_rotations_xyzw=local,
            rest_joint_positions=rest,
            rest_world_rotations_xyzw=rest_world,
            joint_positions=joints,
            source_to_canonical_rotation_xyzw=conversion,
            source_linear_unit_in_meters=0.01,
        )
    )
    np.testing.assert_allclose(canonical_root, [[1.0, 0.9, -2.0]], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(canonical_local, axis=2), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(canonical_rest, 0.0, atol=1e-7)
    np.testing.assert_allclose(canonical_rest_world[..., 3], 1.0, atol=1e-6)
    np.testing.assert_allclose(canonical_joints[:, 0], canonical_root, atol=1e-6)


def test_basis_conversion_preserves_articulated_fk() -> None:
    source = _motion()
    local = source.local_rotations_xyzw.copy()
    half_motion = np.deg2rad(25.0) / 2.0
    local[:, SOMASKEL77_NAMES.index("LeftArm")] = (
        0.0,
        np.sin(half_motion),
        0.0,
        np.cos(half_motion),
    )
    source_joints = soma_forward_kinematics(
        source.root_translation_m,
        local,
        source.rest_joint_positions_m,
        source.rest_world_rotations_xyzw,
    )
    half_basis = -np.pi / 4.0
    conversion = np.array(
        (np.sin(half_basis), 0.0, 0.0, np.cos(half_basis)), dtype=np.float32
    )
    (
        canonical_root,
        canonical_local,
        canonical_rest,
        canonical_rest_world,
        canonical_joints,
    ) = canonicalize_soma_arrays(
        root_translation=source.root_translation_m,
        local_rotations_xyzw=local,
        rest_joint_positions=source.rest_joint_positions_m,
        rest_world_rotations_xyzw=source.rest_world_rotations_xyzw,
        joint_positions=source_joints,
        source_to_canonical_rotation_xyzw=conversion,
        source_linear_unit_in_meters=1.0,
    )
    recomputed = soma_forward_kinematics(
        canonical_root,
        canonical_local,
        canonical_rest,
        canonical_rest_world,
    )
    np.testing.assert_allclose(recomputed, canonical_joints, atol=1e-6)


def test_projection_composes_two_soma_neck_joints_once() -> None:
    source = _motion()
    rotations = source.local_rotations_xyzw.copy()
    half = np.deg2rad(10.0) / 2.0
    for name in ("Neck1", "Neck2"):
        rotations[:, SOMASKEL77_NAMES.index(name), 1] = np.sin(half)
        rotations[:, SOMASKEL77_NAMES.index(name), 3] = np.cos(half)
    joints = soma_forward_kinematics(
        source.root_translation_m,
        rotations,
        source.rest_joint_positions_m,
        source.rest_world_rotations_xyzw,
    )
    projected = project_soma_to_body_track(
        replace(source, local_rotations_xyzw=rotations, joint_positions_m=joints)
    )
    neck = projected.local_rotations_xyzw[:, CANONICAL_HUMANOID.index("Neck")]
    expected_half = np.deg2rad(20.0) / 2.0
    np.testing.assert_allclose(neck[:, 1], np.sin(expected_half), atol=1e-6)
    np.testing.assert_allclose(neck[:, 3], np.cos(expected_half), atol=1e-6)


def test_projection_removes_nonidentity_source_rest_joint_bases() -> None:
    source = _motion()
    rest_world = source.rest_world_rotations_xyzw.copy()
    half = np.deg2rad(30.0) / 2.0
    left_arm = SOMASKEL77_NAMES.index("LeftArm")
    rest_world[left_arm] = (np.sin(half), 0.0, 0.0, np.cos(half))
    joints = soma_forward_kinematics(
        source.root_translation_m,
        source.local_rotations_xyzw,
        source.rest_joint_positions_m,
        rest_world,
    )
    source = replace(
        source,
        rest_world_rotations_xyzw=rest_world,
        joint_positions_m=joints,
    )
    projected = project_soma_to_body_track(source)
    identities = np.zeros_like(projected.local_rotations_xyzw)
    identities[..., 3] = 1.0
    np.testing.assert_allclose(projected.local_rotations_xyzw, identities, atol=1e-6)
