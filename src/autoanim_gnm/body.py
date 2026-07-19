"""Dependency-free humanoid body contract and deterministic acting compiler.

This module is intentionally a foundation, not a motion-capture system.  It
defines an editable skeleton which can be translated to UsdSkel and glTF/VRM,
then compiles the declarative body and gaze portion of an acting plan into a
bounded upper-body track.  Feet and the lower-body chain remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Sequence

import numpy as np

from .acting import TICKS_PER_SECOND, validate_acting_plan


SKELETON_SCHEMA_VERSION = "autoanim.humanoid-skeleton/1.0"
BODY_TRACK_SCHEMA_VERSION = "autoanim.body-track/1.0"
ATTACHMENT_SCHEMA_VERSION = "autoanim.gnm-body-attachment/1.0"
DEFAULT_SAMPLE_RATE_HZ = 30
MAX_SAMPLE_RATE_HZ = 120
MAX_DURATION_TICKS = 30 * 60 * TICKS_PER_SECOND
CONTACT_TOLERANCE_M = 1e-5
BODY_TRACK_LIMITATIONS = (
    "declarative upper-body synthesis only",
    "no motion-capture reconstruction",
    "no proprietary body model",
    "lower-body locomotion is not generated",
)


class BodyValidationError(ValueError):
    """A body skeleton or animation track violated its fail-closed contract."""


@dataclass(frozen=True, slots=True)
class JointSpec:
    """One parent-before-child joint in the canonical humanoid skeleton."""

    name: str
    parent: int
    rest_translation_m: tuple[float, float, float]
    usd_path: str
    vrm_humanoid: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "rest_translation_m": list(self.rest_translation_m),
            "rest_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "rest_scale": [1.0, 1.0, 1.0],
            "usd_path": self.usd_path,
            "gltf_node": self.name,
            "vrm_humanoid": self.vrm_humanoid,
        }


@dataclass(frozen=True, slots=True)
class HumanoidSkeleton:
    """Canonical ordered skeleton, expressed in meters and local transforms."""

    joints: tuple[JointSpec, ...]
    schema_version: str = SKELETON_SCHEMA_VERSION

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise BodyValidationError(f"Unknown humanoid joint: {name}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "+Z",
                "linear_unit": "meter",
                "rotation": "local quaternion [x,y,z,w]",
            },
            "interchange": {
                "master": "OpenUSD UsdSkel",
                "runtime": ["glTF 2.0 skin", "VRM 1.0 humanoid"],
                "usd_joint_order": "parent-before-child; usd_path is the UsdSkel joint token",
                "bind_matrices": "canonical_inverse_bind_matrices() returns glTF-ready values",
            },
            "joints": [joint.as_dict() for joint in self.joints],
        }


def _joint(
    name: str,
    parent: int,
    translation: tuple[float, float, float],
    path: str,
    vrm: str | None = None,
) -> JointSpec:
    return JointSpec(name, parent, translation, path, vrm)


# Anthropometric values are a neutral 1.70 m authoring template, not an
# identity/body estimator.  Identity-specific proportions can replace the rest
# translations in a future immutable character-body revision.
CANONICAL_HUMANOID = HumanoidSkeleton(
    (
        _joint("Root", -1, (0.0, 0.0, 0.0), "Root"),
        _joint("Hips", 0, (0.0, 0.98, 0.0), "Root/Hips", "hips"),
        _joint("Spine", 1, (0.0, 0.12, 0.0), "Root/Hips/Spine", "spine"),
        _joint("Chest", 2, (0.0, 0.16, 0.0), "Root/Hips/Spine/Chest", "chest"),
        _joint(
            "UpperChest",
            3,
            (0.0, 0.15, 0.0),
            "Root/Hips/Spine/Chest/UpperChest",
            "upperChest",
        ),
        _joint(
            "Neck",
            4,
            (0.0, 0.15, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/Neck",
            "neck",
        ),
        _joint(
            "Head",
            5,
            (0.0, 0.10, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/Neck/Head",
            "head",
        ),
        _joint(
            "LeftEye",
            6,
            (-0.032, 0.045, 0.075),
            "Root/Hips/Spine/Chest/UpperChest/Neck/Head/LeftEye",
            "leftEye",
        ),
        _joint(
            "RightEye",
            6,
            (0.032, 0.045, 0.075),
            "Root/Hips/Spine/Chest/UpperChest/Neck/Head/RightEye",
            "rightEye",
        ),
        _joint(
            "LeftShoulder",
            4,
            (-0.11, 0.10, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/LeftShoulder",
            "leftShoulder",
        ),
        _joint(
            "LeftUpperArm",
            9,
            (-0.16, 0.0, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/LeftShoulder/LeftUpperArm",
            "leftUpperArm",
        ),
        _joint(
            "LeftLowerArm",
            10,
            (-0.26, 0.0, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/LeftShoulder/LeftUpperArm/LeftLowerArm",
            "leftLowerArm",
        ),
        _joint(
            "LeftHand",
            11,
            (-0.24, 0.0, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/LeftShoulder/LeftUpperArm/LeftLowerArm/LeftHand",
            "leftHand",
        ),
        _joint(
            "RightShoulder",
            4,
            (0.11, 0.10, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/RightShoulder",
            "rightShoulder",
        ),
        _joint(
            "RightUpperArm",
            13,
            (0.16, 0.0, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/RightShoulder/RightUpperArm",
            "rightUpperArm",
        ),
        _joint(
            "RightLowerArm",
            14,
            (0.26, 0.0, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/RightShoulder/RightUpperArm/RightLowerArm",
            "rightLowerArm",
        ),
        _joint(
            "RightHand",
            15,
            (0.24, 0.0, 0.0),
            "Root/Hips/Spine/Chest/UpperChest/RightShoulder/RightUpperArm/RightLowerArm/RightHand",
            "rightHand",
        ),
        _joint(
            "LeftUpperLeg",
            1,
            (-0.09, -0.08, 0.0),
            "Root/Hips/LeftUpperLeg",
            "leftUpperLeg",
        ),
        _joint(
            "LeftLowerLeg",
            17,
            (0.0, -0.43, 0.0),
            "Root/Hips/LeftUpperLeg/LeftLowerLeg",
            "leftLowerLeg",
        ),
        _joint(
            "LeftFoot",
            18,
            (0.0, -0.42, 0.02),
            "Root/Hips/LeftUpperLeg/LeftLowerLeg/LeftFoot",
            "leftFoot",
        ),
        _joint(
            "LeftToes",
            19,
            (0.0, -0.05, 0.14),
            "Root/Hips/LeftUpperLeg/LeftLowerLeg/LeftFoot/LeftToes",
            "leftToes",
        ),
        _joint(
            "RightUpperLeg",
            1,
            (0.09, -0.08, 0.0),
            "Root/Hips/RightUpperLeg",
            "rightUpperLeg",
        ),
        _joint(
            "RightLowerLeg",
            21,
            (0.0, -0.43, 0.0),
            "Root/Hips/RightUpperLeg/RightLowerLeg",
            "rightLowerLeg",
        ),
        _joint(
            "RightFoot",
            22,
            (0.0, -0.42, 0.02),
            "Root/Hips/RightUpperLeg/RightLowerLeg/RightFoot",
            "rightFoot",
        ),
        _joint(
            "RightToes",
            23,
            (0.0, -0.05, 0.14),
            "Root/Hips/RightUpperLeg/RightLowerLeg/RightFoot/RightToes",
            "rightToes",
        ),
    )
)


def validate_skeleton(skeleton: HumanoidSkeleton = CANONICAL_HUMANOID) -> None:
    if skeleton.schema_version != SKELETON_SCHEMA_VERSION:
        raise BodyValidationError("Unsupported humanoid skeleton schema")
    if not skeleton.joints or skeleton.joints[0].parent != -1:
        raise BodyValidationError("Humanoid skeleton must begin with one root")
    names: set[str] = set()
    paths: set[str] = set()
    vrm_roles: set[str] = set()
    for index, joint in enumerate(skeleton.joints):
        if not joint.name or joint.name in names:
            raise BodyValidationError("Humanoid joint names must be non-empty and unique")
        if not joint.usd_path or joint.usd_path in paths:
            raise BodyValidationError("UsdSkel joint paths must be non-empty and unique")
        if index and not 0 <= joint.parent < index:
            raise BodyValidationError("Humanoid joints must be ordered parent-before-child")
        if index == 0 and joint.parent != -1:
            raise BodyValidationError("Only the first humanoid joint can be the root")
        if index and joint.usd_path != f"{skeleton.joints[joint.parent].usd_path}/{joint.name}":
            raise BodyValidationError("UsdSkel joint paths must preserve the skeleton hierarchy")
        translation = np.asarray(joint.rest_translation_m, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise BodyValidationError("Humanoid rest translations must be finite 3-vectors")
        if joint.vrm_humanoid is not None:
            if joint.vrm_humanoid in vrm_roles:
                raise BodyValidationError("VRM humanoid roles must be unique")
            vrm_roles.add(joint.vrm_humanoid)
        names.add(joint.name)
        paths.add(joint.usd_path)
    required = {
        "hips",
        "spine",
        "head",
        "leftUpperArm",
        "rightUpperArm",
        "leftUpperLeg",
        "rightUpperLeg",
        "leftFoot",
        "rightFoot",
    }
    if not required.issubset(vrm_roles):
        raise BodyValidationError("Canonical skeleton is missing required VRM humanoid roles")


_ATTACHMENT_CONTRACT: dict[str, Any] = {
    "schema_version": ATTACHMENT_SCHEMA_VERSION,
    "composition_order": [
        "body_local_base",
        "gnm_neck_head_additive",
        "gnm_eye_local",
        "animator_override_additive",
    ],
    "mappings": {
        "gnm.neck": {
            "body_joint": "Neck",
            "rotation": "GNM additive after body base",
            "translation": "body only",
            "scale": "body only",
        },
        "gnm.head": {
            "body_joint": "Head",
            "rotation": "GNM additive after body base",
            "translation": "body only",
            "scale": "body only",
        },
        "gnm.left_eye": {
            "body_joint": "LeftEye",
            "rotation": "GNM only; body compiler emits desired GNM eye-local rotation",
            "translation": "character attachment calibration only",
            "scale": "character attachment calibration only",
        },
        "gnm.right_eye": {
            "body_joint": "RightEye",
            "rotation": "GNM only; body compiler emits desired GNM eye-local rotation",
            "translation": "character attachment calibration only",
            "scale": "character attachment calibration only",
        },
    },
    "rules": {
        "face_expression_owner": "GNM facial coefficients",
        "lipsync_owner": "GNM speech pipeline; body/acting compilation cannot modify visemes",
        "body_base_owner": "canonical humanoid Root through Head",
        "eye_owner": "GNM",
        "neck_head_additive_default": "identity; downstream GNM micro-motion may populate it",
        "no_double_drive": "Never copy GNM neck/head rotations into body base and additive layers",
    },
}


def attachment_contract() -> dict[str, Any]:
    """Return a detached JSON-compatible copy of the ownership contract."""

    return json.loads(json.dumps(_ATTACHMENT_CONTRACT, sort_keys=True))


def _readonly_array(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.array(value, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class BodyTrack:
    """A sampled, deterministic local-joint track on an integer timebase."""

    duration_ticks: int
    ticks_per_second: int
    sample_rate_hz: int
    joint_names: tuple[str, ...]
    ticks: np.ndarray
    root_translation_m: np.ndarray
    local_rotations_xyzw: np.ndarray
    foot_contacts: np.ndarray
    gaze_direction_body: np.ndarray
    gaze_strength: np.ndarray
    gnm_eye_rotations_xyzw: np.ndarray
    source_plan_sha256: str

    def __post_init__(self) -> None:
        raw_ticks = np.asarray(self.ticks)
        if raw_ticks.dtype.kind not in "iu" or raw_ticks.dtype.kind == "b":
            raise BodyValidationError("Body track ticks must be integers")
        raw_contacts = np.asarray(self.foot_contacts)
        if raw_contacts.dtype.kind != "b":
            raise BodyValidationError("Body track foot contacts must be boolean")
        object.__setattr__(self, "ticks", _readonly_array(raw_ticks, np.int64))
        object.__setattr__(
            self,
            "root_translation_m",
            _readonly_array(self.root_translation_m, np.float32),
        )
        object.__setattr__(
            self,
            "local_rotations_xyzw",
            _readonly_array(self.local_rotations_xyzw, np.float32),
        )
        object.__setattr__(
            self,
            "foot_contacts",
            _readonly_array(raw_contacts, np.bool_),
        )
        object.__setattr__(
            self,
            "gaze_direction_body",
            _readonly_array(self.gaze_direction_body, np.float32),
        )
        object.__setattr__(
            self,
            "gaze_strength",
            _readonly_array(self.gaze_strength, np.float32),
        )
        object.__setattr__(
            self,
            "gnm_eye_rotations_xyzw",
            _readonly_array(self.gnm_eye_rotations_xyzw, np.float32),
        )
        validate_body_track(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BODY_TRACK_SCHEMA_VERSION,
            "skeleton_schema_version": SKELETON_SCHEMA_VERSION,
            "timebase": {
                "ticks_per_second": self.ticks_per_second,
                "duration_ticks": self.duration_ticks,
                "sample_rate_hz": self.sample_rate_hz,
            },
            "joint_names": list(self.joint_names),
            "ticks": self.ticks.tolist(),
            "root_translation_m": self.root_translation_m.tolist(),
            "local_rotations_xyzw": self.local_rotations_xyzw.tolist(),
            "foot_contacts": self.foot_contacts.tolist(),
            "gaze": {
                "direction_body": self.gaze_direction_body.tolist(),
                "strength": self.gaze_strength.tolist(),
                "gnm_eye_rotations_xyzw": self.gnm_eye_rotations_xyzw.tolist(),
            },
            "attachment_schema_version": ATTACHMENT_SCHEMA_VERSION,
            "source_plan_sha256": self.source_plan_sha256,
            "limitations": list(BODY_TRACK_LIMITATIONS),
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Any) -> BodyTrack:
        if not isinstance(value, dict):
            raise BodyValidationError("Serialized body track must be an object")
        expected = {
            "schema_version",
            "skeleton_schema_version",
            "timebase",
            "joint_names",
            "ticks",
            "root_translation_m",
            "local_rotations_xyzw",
            "foot_contacts",
            "gaze",
            "attachment_schema_version",
            "source_plan_sha256",
            "limitations",
        }
        if set(value) != expected:
            raise BodyValidationError("Serialized body track fields are missing or unknown")
        if value["schema_version"] != BODY_TRACK_SCHEMA_VERSION:
            raise BodyValidationError("Unsupported body track schema")
        if value["skeleton_schema_version"] != SKELETON_SCHEMA_VERSION:
            raise BodyValidationError("Unsupported body-track skeleton schema")
        if value["attachment_schema_version"] != ATTACHMENT_SCHEMA_VERSION:
            raise BodyValidationError("Unsupported GNM attachment schema")
        if value["limitations"] != list(BODY_TRACK_LIMITATIONS):
            raise BodyValidationError("Serialized body-track limitations are invalid")
        timebase = value["timebase"]
        if not isinstance(timebase, dict) or set(timebase) != {
            "ticks_per_second",
            "duration_ticks",
            "sample_rate_hz",
        }:
            raise BodyValidationError("Serialized body timebase is invalid")
        gaze = value["gaze"]
        if not isinstance(gaze, dict) or set(gaze) != {
            "direction_body",
            "strength",
            "gnm_eye_rotations_xyzw",
        }:
            raise BodyValidationError("Serialized body gaze is invalid")
        ticks = value["ticks"]
        if not isinstance(ticks, list) or any(type(tick) is not int for tick in ticks):
            raise BodyValidationError("Serialized body ticks must be integer values")
        contacts = value["foot_contacts"]
        if (
            not isinstance(contacts, list)
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(type(item) is not bool for item in row)
                for row in contacts
            )
        ):
            raise BodyValidationError("Serialized foot contacts must be boolean pairs")
        joint_names = value["joint_names"]
        if not isinstance(joint_names, list) or any(type(name) is not str for name in joint_names):
            raise BodyValidationError("Serialized joint names are invalid")
        return cls(
            duration_ticks=timebase["duration_ticks"],
            ticks_per_second=timebase["ticks_per_second"],
            sample_rate_hz=timebase["sample_rate_hz"],
            joint_names=tuple(joint_names),
            ticks=np.asarray(ticks, dtype=np.int64),
            root_translation_m=value["root_translation_m"],
            local_rotations_xyzw=value["local_rotations_xyzw"],
            foot_contacts=contacts,
            gaze_direction_body=gaze["direction_body"],
            gaze_strength=gaze["strength"],
            gnm_eye_rotations_xyzw=gaze["gnm_eye_rotations_xyzw"],
            source_plan_sha256=value["source_plan_sha256"],
        )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(np.asarray(left), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _axis_quaternion(axis: int, angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    output = np.zeros(angle.shape + (4,), dtype=np.float64)
    output[..., axis] = np.sin(angle * 0.5)
    output[..., 3] = np.cos(angle * 0.5)
    return output


def _euler_xyz_quaternion(angles: np.ndarray) -> np.ndarray:
    """Convert local XYZ Euler radians to normalized [x,y,z,w] quaternions."""

    angles = np.asarray(angles, dtype=np.float64)
    qx = _axis_quaternion(0, angles[..., 0])
    qy = _axis_quaternion(1, angles[..., 1])
    qz = _axis_quaternion(2, angles[..., 2])
    output = _quaternion_multiply(qz, _quaternion_multiply(qy, qx))
    output /= np.linalg.norm(output, axis=-1, keepdims=True)
    return output.astype(np.float32)


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    xyz = q[..., :3]
    uv = np.cross(xyz, vector)
    uuv = np.cross(xyz, uv)
    return vector + 2.0 * (q[..., 3, None] * uv + uuv)


def forward_kinematics_positions(
    root_translation_m: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    *,
    skeleton: HumanoidSkeleton = CANONICAL_HUMANOID,
) -> np.ndarray:
    """Return joint origins in body world space for one or many frames."""

    roots = np.asarray(root_translation_m, dtype=np.float64)
    rotations = np.asarray(local_rotations_xyzw, dtype=np.float64)
    single = roots.ndim == 1
    if single:
        roots = roots[None, :]
        rotations = rotations[None, :, :]
    frame_count = roots.shape[0]
    if roots.shape != (frame_count, 3) or rotations.shape != (
        frame_count,
        len(skeleton.joints),
        4,
    ):
        raise BodyValidationError("Forward-kinematics arrays do not match the skeleton")
    positions = np.zeros((frame_count, len(skeleton.joints), 3), dtype=np.float64)
    world_rotations = np.zeros((frame_count, len(skeleton.joints), 4), dtype=np.float64)
    for index, joint in enumerate(skeleton.joints):
        local_translation = np.asarray(joint.rest_translation_m, dtype=np.float64)
        if joint.parent == -1:
            positions[:, index] = roots + local_translation
            world_rotations[:, index] = rotations[:, index]
        else:
            parent = joint.parent
            positions[:, index] = positions[:, parent] + _rotate_vector(
                world_rotations[:, parent],
                np.broadcast_to(local_translation, (frame_count, 3)),
            )
            world_rotations[:, index] = _quaternion_multiply(
                world_rotations[:, parent], rotations[:, index]
            )
    output = positions.astype(np.float32)
    return output[0] if single else output


def canonical_inverse_bind_matrices(
    skeleton: HumanoidSkeleton = CANONICAL_HUMANOID,
) -> np.ndarray:
    """Return inverse bind matrices for the canonical neutral rest skeleton.

    Matrices use the conventional translation-in-the-last-column layout.
    A glTF writer must flatten each matrix column-major, as required by glTF.
    """

    validate_skeleton(skeleton)
    identity_rotations = np.zeros((len(skeleton.joints), 4), dtype=np.float32)
    identity_rotations[:, 3] = 1.0
    positions = forward_kinematics_positions(
        np.zeros(3, dtype=np.float32), identity_rotations, skeleton=skeleton
    )
    inverse = np.broadcast_to(
        np.eye(4, dtype=np.float32), (len(skeleton.joints), 4, 4)
    ).copy()
    inverse[:, :3, 3] = -positions
    inverse.setflags(write=False)
    return inverse


def _validate_integer(value: Any, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        condition = "positive integer" if positive else "integer"
        raise BodyValidationError(f"{label} must be a {condition}")
    return value


def validate_body_track(
    track: BodyTrack,
    *,
    skeleton: HumanoidSkeleton = CANONICAL_HUMANOID,
    contact_tolerance_m: float = CONTACT_TOLERANCE_M,
) -> None:
    validate_skeleton(skeleton)
    duration = _validate_integer(track.duration_ticks, "duration_ticks", positive=True)
    ticks_per_second = _validate_integer(
        track.ticks_per_second, "ticks_per_second", positive=True
    )
    sample_rate = _validate_integer(track.sample_rate_hz, "sample_rate_hz", positive=True)
    if ticks_per_second != TICKS_PER_SECOND:
        raise BodyValidationError("Body track must use the acting-plan 48 kHz integer timebase")
    if duration > MAX_DURATION_TICKS:
        raise BodyValidationError("Body track duration exceeds the 30-minute phase-1 limit")
    if sample_rate > MAX_SAMPLE_RATE_HZ or ticks_per_second % sample_rate:
        raise BodyValidationError(
            "sample_rate_hz must be at most 120 and divide ticks_per_second exactly"
        )
    if tuple(track.joint_names) != skeleton.names:
        raise BodyValidationError("Body track joint order does not match the canonical skeleton")
    ticks = np.asarray(track.ticks)
    if ticks.ndim != 1 or ticks.size < 2:
        raise BodyValidationError("Body track requires at least two sample ticks")
    if ticks[0] != 0 or ticks[-1] != duration or np.any(np.diff(ticks) <= 0):
        raise BodyValidationError("Body ticks must increase strictly from zero through duration")
    frame_count = ticks.size
    roots = np.asarray(track.root_translation_m)
    rotations = np.asarray(track.local_rotations_xyzw)
    contacts = np.asarray(track.foot_contacts)
    directions = np.asarray(track.gaze_direction_body)
    strengths = np.asarray(track.gaze_strength)
    eyes = np.asarray(track.gnm_eye_rotations_xyzw)
    if roots.shape != (frame_count, 3):
        raise BodyValidationError("Body root translations have invalid shape")
    if rotations.shape != (frame_count, len(skeleton.joints), 4):
        raise BodyValidationError("Body joint rotations have invalid shape")
    if contacts.shape != (frame_count, 2) or contacts.dtype.kind != "b":
        raise BodyValidationError("Foot contacts must be a boolean [frame, left/right] array")
    if directions.shape != (frame_count, 3) or strengths.shape != (frame_count,):
        raise BodyValidationError("Body gaze channels have invalid shape")
    if eyes.shape != (frame_count, 2, 4):
        raise BodyValidationError("GNM eye rotations have invalid shape")
    numeric = (roots, rotations, directions, strengths, eyes)
    if any(not np.isfinite(item).all() for item in numeric):
        raise BodyValidationError("Body track numeric channels must be finite")
    rotation_norms = np.linalg.norm(rotations, axis=2)
    eye_norms = np.linalg.norm(eyes, axis=2)
    if not np.allclose(rotation_norms, 1.0, rtol=0.0, atol=2e-5):
        raise BodyValidationError("Body joint quaternions must be normalized")
    if not np.allclose(eye_norms, 1.0, rtol=0.0, atol=2e-5):
        raise BodyValidationError("GNM eye quaternions must be normalized")
    if np.any(strengths < 0.0) or np.any(strengths > 1.0):
        raise BodyValidationError("Body gaze strength must remain in [0,1]")
    direction_norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(direction_norms, 1.0, rtol=0.0, atol=2e-5):
        raise BodyValidationError("Body gaze directions must be normalized")
    if (
        not isinstance(track.source_plan_sha256, str)
        or len(track.source_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in track.source_plan_sha256)
    ):
        raise BodyValidationError("Body track source plan hash is invalid")
    if not math.isfinite(contact_tolerance_m) or contact_tolerance_m < 0.0:
        raise BodyValidationError("Foot contact tolerance must be finite and non-negative")

    positions = forward_kinematics_positions(roots, rotations, skeleton=skeleton)
    contact_joints = (
        (skeleton.index("LeftFoot"), skeleton.index("LeftToes")),
        (skeleton.index("RightFoot"), skeleton.index("RightToes")),
    )
    for side, joint_indices in enumerate(contact_joints):
        active = contacts[:, side]
        run_start: int | None = None
        for frame in range(frame_count + 1):
            is_active = frame < frame_count and bool(active[frame])
            if is_active and run_start is None:
                run_start = frame
            if not is_active and run_start is not None:
                segment = positions[run_start:frame, joint_indices, :]
                anchor = segment[0]
                error = float(np.max(np.linalg.norm(segment - anchor, axis=2), initial=0.0))
                if error > contact_tolerance_m:
                    label = "left" if side == 0 else "right"
                    raise BodyValidationError(
                        f"{label} foot contact moved {error:.8f} m "
                        f"(limit {contact_tolerance_m:.8f} m)"
                    )
                run_start = None


def _pose_for_beat(
    beat: dict[str, Any],
    skeleton: HumanoidSkeleton,
) -> tuple[np.ndarray, float, float, float]:
    """Return local Euler pose, gaze yaw/pitch, and gaze strength."""

    angles = np.zeros((len(skeleton.joints), 3), dtype=np.float64)
    body = beat["body"]
    energy = float(body["energy"])
    amplitude = 0.35 + 0.65 * energy

    def add(joint: str, xyz_degrees: Sequence[float], scale: float = 1.0) -> None:
        angles[skeleton.index(joint)] += np.deg2rad(xyz_degrees) * amplitude * scale

    stance = body["stance"]
    if stance == "grounded":
        add("Chest", (1.0, 0.0, 0.0), 0.5)
    elif stance == "open":
        add("UpperChest", (-2.0, 0.0, 0.0))
        add("LeftShoulder", (0.0, -4.0, -5.0))
        add("RightShoulder", (0.0, 4.0, 5.0))
    elif stance == "guarded":
        add("UpperChest", (3.0, 0.0, 0.0))
        add("LeftUpperArm", (0.0, -12.0, -9.0))
        add("RightUpperArm", (0.0, 12.0, 9.0))
    elif stance == "forward":
        add("Spine", (5.0, 0.0, 0.0))
        add("Chest", (3.0, 0.0, 0.0))
    elif stance == "withdrawn":
        add("Spine", (-4.0, 0.0, 0.0))
        add("UpperChest", (2.0, 0.0, 0.0))

    tags = body["gesture_tags"]
    gesture_scale = 1.0
    if "small" in tags:
        gesture_scale *= 0.65
    if "broad" in tags:
        gesture_scale *= 1.30
    for tag in tags:
        if tag == "open_palm":
            add("LeftUpperArm", (-8.0, 12.0, 26.0), gesture_scale)
            add("RightUpperArm", (-8.0, -12.0, -26.0), gesture_scale)
            add("LeftLowerArm", (0.0, 18.0, -24.0), gesture_scale)
            add("RightLowerArm", (0.0, -18.0, 24.0), gesture_scale)
            add("LeftHand", (0.0, 0.0, -10.0), gesture_scale)
            add("RightHand", (0.0, 0.0, 10.0), gesture_scale)
        elif tag == "point":
            add("RightUpperArm", (-18.0, -34.0, 12.0), gesture_scale)
            add("RightLowerArm", (0.0, -8.0, -28.0), gesture_scale)
        elif tag == "count":
            add("RightUpperArm", (-10.0, -22.0, -8.0), gesture_scale)
            add("RightLowerArm", (0.0, -10.0, -48.0), gesture_scale)
            add("RightHand", (0.0, 0.0, 12.0), gesture_scale)
        elif tag == "shrug":
            add("LeftShoulder", (0.0, 0.0, -12.0), gesture_scale)
            add("RightShoulder", (0.0, 0.0, 12.0), gesture_scale)
            add("Head", (0.0, 3.0, 0.0), gesture_scale)
        elif tag == "hand_to_chest":
            add("RightUpperArm", (-16.0, 34.0, -18.0), gesture_scale)
            add("RightLowerArm", (0.0, 10.0, 62.0), gesture_scale)

    target = beat["gaze"]["target"]
    strength = float(beat["gaze"]["strength"])
    gaze_degrees = {
        "camera": (0.0, 0.0),
        "listener": (4.0, 0.0),
        "away_left": (-24.0, 1.0),
        "away_right": (24.0, 1.0),
        "down": (0.0, -14.0),
        "up": (0.0, 14.0),
        "unspecified": (0.0, 0.0),
    }
    yaw, pitch = gaze_degrees[target]
    if target == "unspecified":
        strength = 0.0
    yaw = math.radians(yaw) * strength
    pitch = math.radians(pitch) * strength
    # The body owns these base turns; GNM receives the residual ocular turn.
    angles[skeleton.index("Neck"), 0] += pitch * 0.12
    angles[skeleton.index("Neck"), 1] += yaw * 0.12
    angles[skeleton.index("Head"), 0] += pitch * 0.26
    angles[skeleton.index("Head"), 1] += yaw * 0.26
    return angles, yaw, pitch, strength


def _smoothstep(value: float) -> float:
    clipped = min(max(value, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def compile_body_track(
    acting_plan: Any,
    *,
    duration_ticks: int,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    skeleton: HumanoidSkeleton = CANONICAL_HUMANOID,
) -> BodyTrack:
    """Compile validated acting body/gaze intent to an editable numeric track.

    The compiler is deterministic: there is no randomness, inference, wall
    clock, or platform-dependent optimizer.  Beat edges use a 150 ms bounded
    smoothstep transition.  This phase does not synthesize locomotion.
    """

    validate_skeleton(skeleton)
    duration = _validate_integer(duration_ticks, "duration_ticks", positive=True)
    if duration > MAX_DURATION_TICKS:
        raise BodyValidationError("duration_ticks exceeds the 30-minute phase-1 limit")
    rate = _validate_integer(sample_rate_hz, "sample_rate_hz", positive=True)
    if rate > MAX_SAMPLE_RATE_HZ or TICKS_PER_SECOND % rate:
        raise BodyValidationError(
            "sample_rate_hz must be at most 120 and divide 48,000 ticks exactly"
        )
    plan = validate_acting_plan(acting_plan, duration_ticks=duration)
    beats = plan["beats"]
    step = TICKS_PER_SECOND // rate
    ticks = np.arange(0, duration + 1, step, dtype=np.int64)
    if ticks[-1] != duration:
        ticks = np.concatenate((ticks, np.asarray([duration], dtype=np.int64)))
    frame_count = ticks.size
    joint_count = len(skeleton.joints)
    local_angles = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    gaze_yaw_pitch = np.zeros((frame_count, 2), dtype=np.float64)
    gaze_strength = np.zeros(frame_count, dtype=np.float64)
    foot_contacts = np.ones((frame_count, 2), dtype=np.bool_)

    neutral_pose = np.zeros((joint_count, 3), dtype=np.float64)
    neutral_gaze = (0.0, 0.0, 0.0)
    targets = [_pose_for_beat(beat, skeleton) for beat in beats]
    transition_cap = int(round(0.150 * TICKS_PER_SECOND))

    for frame, tick_value in enumerate(ticks.tolist()):
        active_index = next(
            (
                index
                for index, beat in enumerate(beats)
                if beat["start_tick"] <= tick_value < beat["end_tick"]
            ),
            None,
        )
        if active_index is None:
            continue
        beat = beats[active_index]
        target_pose, target_yaw, target_pitch, target_strength = targets[active_index]
        length = beat["end_tick"] - beat["start_tick"]
        transition = min(transition_cap, max(1, length // 3))
        contiguous_before = (
            active_index > 0 and beats[active_index - 1]["end_tick"] == beat["start_tick"]
        )
        contiguous_after = (
            active_index + 1 < len(beats)
            and beats[active_index + 1]["start_tick"] == beat["end_tick"]
        )
        incoming_pose, incoming_yaw, incoming_pitch, incoming_strength = (
            targets[active_index - 1] if contiguous_before else (neutral_pose, *neutral_gaze)
        )
        outgoing_pose, outgoing_yaw, outgoing_pitch, outgoing_strength = (
            targets[active_index + 1] if contiguous_after else (neutral_pose, *neutral_gaze)
        )

        if tick_value < beat["start_tick"] + transition:
            blend = _smoothstep((tick_value - beat["start_tick"]) / transition)
            pose = incoming_pose + blend * (target_pose - incoming_pose)
            yaw = incoming_yaw + blend * (target_yaw - incoming_yaw)
            pitch = incoming_pitch + blend * (target_pitch - incoming_pitch)
            strength = incoming_strength + blend * (target_strength - incoming_strength)
        elif tick_value > beat["end_tick"] - transition:
            blend = _smoothstep(
                (tick_value - (beat["end_tick"] - transition)) / transition
            )
            pose = target_pose + blend * (outgoing_pose - target_pose)
            yaw = target_yaw + blend * (outgoing_yaw - target_yaw)
            pitch = target_pitch + blend * (outgoing_pitch - target_pitch)
            strength = target_strength + blend * (outgoing_strength - target_strength)
        else:
            pose = target_pose.copy()
            yaw, pitch, strength = target_yaw, target_pitch, target_strength

        # Nods and shakes are deterministic beat-local acting accents.  Their
        # sinusoid begins/ends at zero and is attenuated by the same edge ramp.
        phase = (tick_value - beat["start_tick"]) / length
        edge = min(1.0, (tick_value - beat["start_tick"]) / transition)
        edge = min(edge, (beat["end_tick"] - tick_value) / transition)
        accent_envelope = _smoothstep(edge)
        gesture_tags = beat["body"]["gesture_tags"]
        accent_amplitude = (0.35 + 0.65 * float(beat["body"]["energy"])) * accent_envelope
        if "small" in gesture_tags:
            accent_amplitude *= 0.65
        if "broad" in gesture_tags:
            accent_amplitude *= 1.30
        if "head_nod" in gesture_tags:
            pose[skeleton.index("Head"), 0] += (
                math.radians(7.0) * accent_amplitude * math.sin(4.0 * math.pi * phase)
            )
        if "head_shake" in gesture_tags:
            pose[skeleton.index("Head"), 1] += (
                math.radians(9.0) * accent_amplitude * math.sin(4.0 * math.pi * phase)
            )

        local_angles[frame] = pose
        gaze_yaw_pitch[frame] = (yaw, pitch)
        gaze_strength[frame] = strength
        preserve = bool(beat["constraints"]["preserve_foot_contacts"])
        foot_contacts[frame] = (preserve, preserve)

    rotations = _euler_xyz_quaternion(local_angles)
    root_translation = np.zeros((frame_count, 3), dtype=np.float32)
    directions = np.column_stack(
        (
            np.tan(gaze_yaw_pitch[:, 0]),
            np.tan(gaze_yaw_pitch[:, 1]),
            np.ones(frame_count, dtype=np.float64),
        )
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # Eyes own the residual gaze after the body-base neck/head split.
    eye_angles = np.zeros((frame_count, 2, 3), dtype=np.float64)
    eye_angles[:, :, 0] = gaze_yaw_pitch[:, None, 1] * 0.62
    eye_angles[:, :, 1] = gaze_yaw_pitch[:, None, 0] * 0.62
    eye_rotations = _euler_xyz_quaternion(eye_angles)
    return BodyTrack(
        duration_ticks=duration,
        ticks_per_second=TICKS_PER_SECOND,
        sample_rate_hz=rate,
        joint_names=skeleton.names,
        ticks=ticks,
        root_translation_m=root_translation,
        local_rotations_xyzw=rotations,
        foot_contacts=foot_contacts,
        gaze_direction_body=directions,
        gaze_strength=gaze_strength,
        gnm_eye_rotations_xyzw=eye_rotations,
        source_plan_sha256=_canonical_hash(plan),
    )


validate_skeleton(CANONICAL_HUMANOID)
