"""Validated SOMA-77 source motion and an explicit canonical-25 projection.

The full SOMA track is the lossless source artifact.  Projection to AutoAnim's
current 25-joint body contract is intentionally lossy: fingers and source face
joints stay in the SOMA artifact, eyes remain GNM-owned, and provider contacts
are not promoted to hard foot plants without a separate contact solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .acting import TICKS_PER_SECOND
from .body import BodyTrack, CANONICAL_HUMANOID, DETAILED_HUMANOID, HumanoidSkeleton


SOMA_MOTION_SCHEMA_VERSION = "autoanim.soma-motion/1.0"
SOMASKEL77_SCHEMA_VERSION = "autoanim.somaskel77/1.0"
SOMASKEL77_PATH = Path(__file__).with_name("data") / "somaskel77-v1.json"
MAX_SOMA_FRAMES = 30 * 60 * 120 + 1
MAX_SOMA_DURATION_TICKS = 30 * 60 * TICKS_PER_SECOND
FK_TOLERANCE_M = 1e-3
QUATERNION_TOLERANCE = 2e-5
SOMASKEL77_SEMANTIC_SHA256 = (
    "3625d93a8e8be26f79f0333dd42f3d1bf5269fc4ca668175e5f0bfb3cd01adab"
)
ALLOWED_PROVIDER_COMMITS = {
    "nvidia_gem_x": "32992550dba114c62243fb55e361311972dce8f9",
    "nvidia_kimodo_soma": "1aece8c124d73d255ceff5086d983b844c9f4e94",
}


class SomaMotionValidationError(ValueError):
    """A SOMA source-motion artifact violated its fail-closed contract."""


def _load_skeleton_contract() -> dict[str, Any]:
    value = json.loads(SOMASKEL77_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != SOMASKEL77_SCHEMA_VERSION:
        raise SomaMotionValidationError("Unexpected somaskel77 schema version")
    if value.get("joint_count") != 77:
        raise SomaMotionValidationError("somaskel77 must contain exactly 77 joints")
    names = value.get("joint_names")
    parents = value.get("parents")
    if not isinstance(names, list) or len(names) != 77 or len(set(names)) != 77:
        raise SomaMotionValidationError("somaskel77 names must be 77 unique strings")
    if not isinstance(parents, list) or len(parents) != 77:
        raise SomaMotionValidationError("somaskel77 parents must contain 77 indices")
    for index, parent in enumerate(parents):
        if type(parent) is not int or parent >= index or parent < -1:
            raise SomaMotionValidationError("somaskel77 hierarchy is not parent-before-child")
        if index == 0 and parent != -1:
            raise SomaMotionValidationError("somaskel77 Hips must be the only root")
        if index > 0 and parent < 0:
            raise SomaMotionValidationError("somaskel77 contains an unexpected extra root")
    semantic_value = {
        key: value[key]
        for key in (
            "schema_version",
            "name",
            "joint_count",
            "joint_names",
            "parents",
            "coordinate_system",
        )
    }
    observed_digest = sha256(
        json.dumps(semantic_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_digest != SOMASKEL77_SEMANTIC_SHA256:
        raise SomaMotionValidationError("somaskel77 semantic digest is not the reviewed schema")
    return value


_SKELETON = _load_skeleton_contract()
SOMASKEL77_NAMES = tuple(_SKELETON["joint_names"])
SOMASKEL77_PARENTS = tuple(_SKELETON["parents"])
SOMASKEL77_PROVENANCE_SHA256 = sha256(
    json.dumps(_SKELETON["upstream"], sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()

GEM_X_CONTACT_SCHEMA_ID = "gem_x_somaskel77_contacts/1.0"
GEM_X_CONTACT_NAMES = (
    "LeftFoot",
    "LeftToeBase",
    "RightFoot",
    "RightToeBase",
    "LeftHand",
    "RightHand",
)
KIMODO_CONTACT_SCHEMA_ID = "kimodo_somaskel77_heel_toe/1.0"
KIMODO_CONTACT_NAMES = ("left_heel", "left_toe", "right_heel", "right_toe")
CONTACT_SCHEMAS = {
    GEM_X_CONTACT_SCHEMA_ID: GEM_X_CONTACT_NAMES,
    KIMODO_CONTACT_SCHEMA_ID: KIMODO_CONTACT_NAMES,
}


def _readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.array(value, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    xyz = quaternion[..., :3]
    uv = np.cross(xyz, vector)
    uuv = np.cross(xyz, uv)
    return vector + 2.0 * (quaternion[..., 3, None] * uv + uuv)


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    inverse = value.copy()
    inverse[..., :3] *= -1.0
    return inverse / np.sum(value * value, axis=-1, keepdims=True)


def canonicalize_soma_arrays(
    *,
    root_translation: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    rest_joint_positions: np.ndarray,
    rest_world_rotations_xyzw: np.ndarray,
    joint_positions: np.ndarray,
    source_to_canonical_rotation_xyzw: np.ndarray,
    source_linear_unit_in_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply an explicit source-world basis and unit conversion.

    This utility does not infer axes.  A provider adapter must obtain the
    source convention from its pinned output contract and pass the reviewed
    source-to-canonical rotation.
    """

    conversion = np.asarray(source_to_canonical_rotation_xyzw, dtype=np.float64)
    if conversion.shape != (4,) or not np.isfinite(conversion).all():
        raise SomaMotionValidationError("Source-to-canonical rotation must be finite [4]")
    conversion_norm = float(np.linalg.norm(conversion))
    if abs(conversion_norm - 1.0) > QUATERNION_TOLERANCE:
        raise SomaMotionValidationError("Source-to-canonical rotation must be normalized")
    if (
        not isinstance(source_linear_unit_in_meters, (int, float))
        or isinstance(source_linear_unit_in_meters, bool)
        or not np.isfinite(source_linear_unit_in_meters)
        or source_linear_unit_in_meters <= 0.0
    ):
        raise SomaMotionValidationError("Source linear-unit scale must be finite and positive")
    roots_source = np.asarray(root_translation)
    local_source = np.asarray(local_rotations_xyzw)
    rest_source = np.asarray(rest_joint_positions)
    rest_world_source = np.asarray(rest_world_rotations_xyzw)
    joints_source = np.asarray(joint_positions)
    if (
        roots_source.ndim != 2
        or roots_source.shape[1:] != (3,)
        or local_source.shape != (roots_source.shape[0], 77, 4)
        or rest_source.shape != (77, 3)
        or rest_world_source.shape != (77, 4)
        or joints_source.shape != (roots_source.shape[0], 77, 3)
    ):
        raise SomaMotionValidationError("Source SOMA arrays have invalid shapes")
    if any(
        not np.isfinite(value).all()
        for value in (
            roots_source,
            local_source,
            rest_source,
            rest_world_source,
            joints_source,
        )
    ):
        raise SomaMotionValidationError("Source SOMA arrays must be finite")
    if np.any(np.linalg.norm(local_source, axis=2) < QUATERNION_TOLERANCE) or np.any(
        np.linalg.norm(rest_world_source, axis=1) < QUATERNION_TOLERANCE
    ):
        raise SomaMotionValidationError("Source SOMA quaternions cannot be zero")
    inverse = conversion.copy()
    inverse[:3] *= -1.0
    source_local = np.asarray(local_rotations_xyzw, dtype=np.float64)
    converted_local = _quaternion_multiply(
        _quaternion_multiply(conversion, source_local), inverse
    )
    converted_local /= np.linalg.norm(converted_local, axis=-1, keepdims=True)
    source_rest_world = np.asarray(rest_world_rotations_xyzw, dtype=np.float64)
    converted_rest_world = _quaternion_multiply(
        _quaternion_multiply(conversion, source_rest_world), inverse
    )
    converted_rest_world /= np.linalg.norm(
        converted_rest_world, axis=-1, keepdims=True
    )

    def convert_points(value: np.ndarray) -> np.ndarray:
        source = np.asarray(value, dtype=np.float64) * source_linear_unit_in_meters
        rotation = np.broadcast_to(conversion, source.shape[:-1] + (4,))
        return _rotate_vector(rotation, source).astype(np.float32)

    return (
        convert_points(root_translation),
        converted_local.astype(np.float32),
        convert_points(rest_joint_positions),
        converted_rest_world.astype(np.float32),
        convert_points(joint_positions),
    )


def soma_forward_kinematics(
    root_translation_m: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    rest_joint_positions_m: np.ndarray,
    rest_world_rotations_xyzw: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate somaskel77 joint origins in world space."""

    roots = np.asarray(root_translation_m, dtype=np.float64)
    local = np.asarray(local_rotations_xyzw, dtype=np.float64)
    rest = np.asarray(rest_joint_positions_m, dtype=np.float64)
    if rest_world_rotations_xyzw is None:
        rest_world = np.zeros((77, 4), dtype=np.float64)
        rest_world[:, 3] = 1.0
    else:
        rest_world = np.asarray(rest_world_rotations_xyzw, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1:] != (3,):
        raise SomaMotionValidationError("SOMA root translation must be [frame,3]")
    frame_count = roots.shape[0]
    if (
        local.shape != (frame_count, 77, 4)
        or rest.shape != (77, 3)
        or rest_world.shape != (77, 4)
    ):
        raise SomaMotionValidationError("SOMA FK arrays have invalid shape")
    positions = np.zeros((frame_count, 77, 3), dtype=np.float64)
    world_rotations = np.zeros((frame_count, 77, 4), dtype=np.float64)
    for index, parent in enumerate(SOMASKEL77_PARENTS):
        if parent == -1:
            positions[:, index] = roots
            world_rotations[:, index] = _quaternion_multiply(
                rest_world[index], local[:, index]
            )
        else:
            offset = _rotate_vector(
                _quaternion_inverse(rest_world[parent]),
                rest[index] - rest[parent],
            )
            offset = np.broadcast_to(offset, (frame_count, 3))
            positions[:, index] = positions[:, parent] + _rotate_vector(
                world_rotations[:, parent], offset
            )
            rest_local = _quaternion_multiply(
                _quaternion_inverse(rest_world[parent]), rest_world[index]
            )
            world_rotations[:, index] = _quaternion_multiply(
                _quaternion_multiply(world_rotations[:, parent], rest_local),
                local[:, index],
            )
    return positions.astype(np.float32)


def _soma_world_rotations(
    local_rotations_xyzw: np.ndarray,
    rest_world_rotations_xyzw: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_rotations_xyzw, dtype=np.float64)
    rest_world = np.asarray(rest_world_rotations_xyzw, dtype=np.float64)
    frame_count = local.shape[0]
    output = np.zeros((frame_count, 77, 4), dtype=np.float64)
    for index, parent in enumerate(SOMASKEL77_PARENTS):
        if parent == -1:
            output[:, index] = _quaternion_multiply(
                rest_world[index], local[:, index]
            )
        else:
            rest_local = _quaternion_multiply(
                _quaternion_inverse(rest_world[parent]), rest_world[index]
            )
            output[:, index] = _quaternion_multiply(
                _quaternion_multiply(output[:, parent], rest_local),
                local[:, index],
            )
    output /= np.linalg.norm(output, axis=2, keepdims=True)
    return output


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SomaMotionValidationError(f"{label} must be a lowercase SHA-256")


def _validate_git_oid(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SomaMotionValidationError(f"{label} must be a 40-character Git object ID")


@dataclass(frozen=True, slots=True)
class SomaMotion:
    """Immutable common motion returned by an attested GEM-X/Kimodo adapter."""

    provider_id: str
    provider_git_commit_oid: str
    operation: str
    motion_kind: str
    duration_ticks: int
    sample_rate_hz: int
    ticks: np.ndarray
    source_pts: np.ndarray
    root_translation_m: np.ndarray
    local_rotations_xyzw: np.ndarray
    rest_joint_positions_m: np.ndarray
    rest_world_rotations_xyzw: np.ndarray
    joint_positions_m: np.ndarray
    contacts: np.ndarray
    contact_schema_id: str
    contact_names: tuple[str, ...]
    source_handedness: str
    source_up_axis: str
    source_forward_axis: str
    source_linear_unit_in_meters: float
    source_to_canonical_rotation_xyzw: tuple[float, float, float, float]
    source_time_base_numerator: int
    source_time_base_denominator: int
    input_sha256: str
    provider_raw_motion_sha256: str

    def __post_init__(self) -> None:
        raw_ticks = np.asarray(self.ticks)
        raw_pts = np.asarray(self.source_pts)
        raw_contacts = np.asarray(self.contacts)
        if raw_ticks.dtype.kind not in "iu" or raw_ticks.dtype.kind == "b":
            raise SomaMotionValidationError("SOMA ticks must be integers")
        if raw_pts.dtype.kind not in "iu" or raw_pts.dtype.kind == "b":
            raise SomaMotionValidationError("SOMA source PTS must be integers")
        if raw_contacts.dtype.kind != "b":
            raise SomaMotionValidationError("SOMA contacts must be boolean")
        object.__setattr__(self, "ticks", _readonly(raw_ticks, np.int64))
        object.__setattr__(self, "source_pts", _readonly(raw_pts, np.int64))
        object.__setattr__(
            self, "root_translation_m", _readonly(self.root_translation_m, np.float32)
        )
        object.__setattr__(
            self,
            "local_rotations_xyzw",
            _readonly(self.local_rotations_xyzw, np.float32),
        )
        object.__setattr__(
            self,
            "rest_joint_positions_m",
            _readonly(self.rest_joint_positions_m, np.float32),
        )
        object.__setattr__(
            self,
            "rest_world_rotations_xyzw",
            _readonly(self.rest_world_rotations_xyzw, np.float32),
        )
        object.__setattr__(
            self, "joint_positions_m", _readonly(self.joint_positions_m, np.float32)
        )
        object.__setattr__(self, "contacts", _readonly(raw_contacts, np.bool_))
        object.__setattr__(self, "contact_names", tuple(self.contact_names))
        object.__setattr__(
            self,
            "source_to_canonical_rotation_xyzw",
            tuple(float(value) for value in self.source_to_canonical_rotation_xyzw),
        )
        object.__setattr__(
            self,
            "source_linear_unit_in_meters",
            float(self.source_linear_unit_in_meters),
        )
        validate_soma_motion(self)

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOMA_MOTION_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "provider_git_commit_oid": self.provider_git_commit_oid,
            "operation": self.operation,
            "motion_kind": self.motion_kind,
            "skeleton": {
                "schema_version": SOMASKEL77_SCHEMA_VERSION,
                "semantic_sha256": SOMASKEL77_SEMANTIC_SHA256,
                "provenance_sha256": SOMASKEL77_PROVENANCE_SHA256,
                "joint_count": 77,
            },
            "coordinate_system": _SKELETON["coordinate_system"],
            "source_coordinate_system": {
                "handedness": self.source_handedness,
                "up_axis": self.source_up_axis,
                "forward_axis": self.source_forward_axis,
                "linear_unit_in_meters": self.source_linear_unit_in_meters,
                "source_to_canonical_rotation_xyzw": list(
                    self.source_to_canonical_rotation_xyzw
                ),
                "canonicalization_applied": True,
            },
            "timebase": {
                "ticks_per_second": TICKS_PER_SECOND,
                "duration_ticks": self.duration_ticks,
                "sample_rate_hz": self.sample_rate_hz,
                "sample_rate_semantics": "nominal; ticks are exact source PTS projection",
                "source_time_base": {
                    "numerator": self.source_time_base_numerator,
                    "denominator": self.source_time_base_denominator,
                },
            },
            "contact_schema": {
                "id": self.contact_schema_id,
                "contact_names": list(self.contact_names),
            },
            "source": {
                "input_sha256": self.input_sha256,
                "provider_raw_motion_sha256": self.provider_raw_motion_sha256,
            },
            "frame_count": int(self.ticks.size),
            "production_validated": False,
        }

    def content_sha256(self) -> str:
        digest = sha256(
            json.dumps(
                self.manifest_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name in (
            "ticks",
            "source_pts",
            "root_translation_m",
            "local_rotations_xyzw",
            "rest_joint_positions_m",
            "rest_world_rotations_xyzw",
            "joint_positions_m",
            "contacts",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()


def validate_soma_motion(track: SomaMotion) -> None:
    if track.provider_id not in {"nvidia_gem_x", "nvidia_kimodo_soma"}:
        raise SomaMotionValidationError("Unsupported SOMA motion provider")
    _validate_git_oid(track.provider_git_commit_oid, "provider_git_commit_oid")
    if track.provider_git_commit_oid != ALLOWED_PROVIDER_COMMITS[track.provider_id]:
        raise SomaMotionValidationError("Provider Git object ID is not allowlisted")
    expected_operation = {
        "nvidia_gem_x": ("video_capture", "observed"),
        "nvidia_kimodo_soma": ("acting_generation", "generated"),
    }[track.provider_id]
    if (track.operation, track.motion_kind) != expected_operation:
        raise SomaMotionValidationError("Provider operation/motion kind is inconsistent")
    if (
        type(track.duration_ticks) is not int
        or track.duration_ticks <= 0
        or track.duration_ticks > MAX_SOMA_DURATION_TICKS
    ):
        raise SomaMotionValidationError(
            "duration_ticks must be a positive integer within the 30-minute limit"
        )
    if (
        type(track.sample_rate_hz) is not int
        or track.sample_rate_hz <= 0
        or track.sample_rate_hz > 120
        or TICKS_PER_SECOND % track.sample_rate_hz
    ):
        raise SomaMotionValidationError("sample_rate_hz must divide 48 kHz and be <=120")
    if (
        track.source_handedness,
        track.source_up_axis,
        track.source_forward_axis,
    ) != ("right", "+Y", "+Z"):
        raise SomaMotionValidationError(
            "Pinned SOMA providers must attest right-handed +Y-up +Z-forward output"
        )
    if not np.isclose(
        track.source_linear_unit_in_meters, 1.0, rtol=0.0, atol=1e-12
    ):
        raise SomaMotionValidationError("Pinned SOMA provider output must be in meters")
    source_conversion = np.asarray(
        track.source_to_canonical_rotation_xyzw, dtype=np.float64
    )
    if source_conversion.shape != (4,) or not np.isfinite(source_conversion).all():
        raise SomaMotionValidationError("Source-to-canonical rotation must be finite [4]")
    if not np.isclose(
        np.linalg.norm(source_conversion), 1.0, rtol=0.0, atol=QUATERNION_TOLERANCE
    ):
        raise SomaMotionValidationError("Source-to-canonical rotation must be normalized")
    if not np.allclose(
        source_conversion, (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-7
    ):
        raise SomaMotionValidationError(
            "Pinned SOMA provider output must use the identity canonicalization"
        )
    if (
        type(track.source_time_base_numerator) is not int
        or type(track.source_time_base_denominator) is not int
        or track.source_time_base_numerator <= 0
        or track.source_time_base_denominator <= 0
    ):
        raise SomaMotionValidationError("Source timebase must be a positive rational")
    frame_count = track.ticks.size
    if frame_count < 2 or frame_count > MAX_SOMA_FRAMES:
        raise SomaMotionValidationError("SOMA track frame count is outside supported limits")
    if track.ticks.shape != (frame_count,) or track.source_pts.shape != (frame_count,):
        raise SomaMotionValidationError("SOMA ticks/source PTS must be one-dimensional")
    if (
        track.ticks[0] != 0
        or track.ticks[-1] != track.duration_ticks
        or np.any(track.ticks[1:] <= track.ticks[:-1])
    ):
        raise SomaMotionValidationError(
            "SOMA ticks must increase strictly from zero through duration"
        )
    if np.any(track.source_pts[1:] <= track.source_pts[:-1]):
        raise SomaMotionValidationError("SOMA source PTS must increase strictly")
    first_source_pts = int(track.source_pts[0])
    tick_numerators = [
        (int(value) - first_source_pts)
        * track.source_time_base_numerator
        * TICKS_PER_SECOND
        for value in track.source_pts
    ]
    tick_denominator = track.source_time_base_denominator
    expected_ticks = np.asarray(
        [
            (2 * int(numerator) + tick_denominator) // (2 * tick_denominator)
            for numerator in tick_numerators
        ],
        dtype=np.int64,
    )
    if not np.array_equal(track.ticks, expected_ticks):
        raise SomaMotionValidationError(
            "SOMA ticks do not match source PTS under exact rational conversion"
        )
    expected_shapes = {
        "root_translation_m": (frame_count, 3),
        "local_rotations_xyzw": (frame_count, 77, 4),
        "rest_joint_positions_m": (77, 3),
        "rest_world_rotations_xyzw": (77, 4),
        "joint_positions_m": (frame_count, 77, 3),
    }
    for name, shape in expected_shapes.items():
        value = getattr(track, name)
        if value.shape != shape:
            raise SomaMotionValidationError(f"{name} has invalid shape")
        if not np.isfinite(value).all():
            raise SomaMotionValidationError(f"{name} must contain only finite values")
    norms = np.linalg.norm(track.local_rotations_xyzw, axis=2)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=QUATERNION_TOLERANCE):
        raise SomaMotionValidationError("SOMA quaternions must be normalized")
    rest_rotation_norms = np.linalg.norm(track.rest_world_rotations_xyzw, axis=1)
    if not np.allclose(
        rest_rotation_norms, 1.0, rtol=0.0, atol=QUATERNION_TOLERANCE
    ):
        raise SomaMotionValidationError("SOMA rest-world quaternions must be normalized")
    adjacent_dots = np.sum(
        track.local_rotations_xyzw[1:] * track.local_rotations_xyzw[:-1], axis=2
    )
    if np.any(adjacent_dots < -QUATERNION_TOLERANCE):
        raise SomaMotionValidationError("SOMA quaternion signs must be continuous")
    expected_contacts = CONTACT_SCHEMAS.get(track.contact_schema_id)
    if expected_contacts is None or tuple(track.contact_names) != expected_contacts:
        raise SomaMotionValidationError("Unknown or mismatched SOMA contact schema")
    if track.contacts.shape != (frame_count, len(expected_contacts)):
        raise SomaMotionValidationError("SOMA contacts have invalid shape")
    provider_contact_schema = {
        "nvidia_gem_x": GEM_X_CONTACT_SCHEMA_ID,
        "nvidia_kimodo_soma": KIMODO_CONTACT_SCHEMA_ID,
    }[track.provider_id]
    if track.contact_schema_id != provider_contact_schema:
        raise SomaMotionValidationError("Contact schema is not valid for this provider")
    if not np.allclose(track.rest_joint_positions_m[0], 0.0, rtol=0.0, atol=1e-7):
        raise SomaMotionValidationError("SOMA rest joints must be centered on Hips")
    _validate_sha256(track.input_sha256, "input_sha256")
    _validate_sha256(track.provider_raw_motion_sha256, "provider_raw_motion_sha256")
    fk_positions = soma_forward_kinematics(
        track.root_translation_m,
        track.local_rotations_xyzw,
        track.rest_joint_positions_m,
        track.rest_world_rotations_xyzw,
    )
    error = float(
        np.max(np.linalg.norm(fk_positions - track.joint_positions_m, axis=2), initial=0.0)
    )
    if error > FK_TOLERANCE_M:
        raise SomaMotionValidationError(
            f"SOMA joint positions disagree with FK by {error:.6f} m"
        )


_DELTA_MAPPING = {
    "Root": "Hips",
    "Hips": "Hips",
    "Spine": "Spine1",
    "Chest": "Spine2",
    "UpperChest": "Chest",
    "Neck": "Neck2",
    "Head": "Head",
    # SOMA anatomical left is positive source X, and so is AutoAnim's since 2026-09-02:
    # both declare right-handed, up +Y, forward +Z, in which the left is `up x forward`.
    # Each side therefore maps to its own side.
    #
    # THIS BOUNDARY USED TO SWAP THEM, and the reason was written beside it -- "AutoAnim's
    # reviewed canonical skeleton defines Left on negative X". That was true, and it was
    # itself the defect: the swap here was a COMPENSATION for the rig's mirrored naming,
    # so removing the mirror without removing the compensation would have shipped every
    # SOMA performer with their arms and legs exchanged. Nothing on this lane has a
    # capture fixture that would have caught it.
    # docs/reviews/facing-fix-2026-09-02.md; tests/test_facing_fix.py.
    "LeftShoulder": "LeftShoulder",
    "LeftUpperArm": "LeftArm",
    "LeftLowerArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightUpperArm": "RightArm",
    "RightLowerArm": "RightForeArm",
    "RightHand": "RightHand",
    "LeftUpperLeg": "LeftLeg",
    "LeftLowerLeg": "LeftShin",
    "LeftFoot": "LeftFoot",
    "LeftToes": "LeftToeBase",
    "RightUpperLeg": "RightLeg",
    "RightLowerLeg": "RightShin",
    "RightFoot": "RightFoot",
    "RightToes": "RightToeBase",
}


def _detailed_delta_mapping() -> dict[str, str]:
    mapping = dict(_DELTA_MAPPING)
    for target_side, source_side in (("Left", "Left"), ("Right", "Right")):
        mapping.update(
            {
                f"{target_side}ThumbMetacarpal": f"{source_side}HandThumb1",
                f"{target_side}ThumbProximal": f"{source_side}HandThumb2",
                f"{target_side}ThumbDistal": f"{source_side}HandThumb3",
            }
        )
        for target_finger, source_finger in (
            ("Index", "Index"),
            ("Middle", "Middle"),
            ("Ring", "Ring"),
            ("Little", "Pinky"),
        ):
            # SOMA has a metacarpal-like first segment plus three phalanges for
            # non-thumb fingers. VRM/MPFB has only the three deforming
            # phalanges. Selecting SOMA 2/3/4 in world-delta space folds the
            # upstream SOMA segment into the target proximal rotation.
            mapping.update(
                {
                    f"{target_side}{target_finger}Proximal": (
                        f"{source_side}Hand{source_finger}2"
                    ),
                    f"{target_side}{target_finger}Intermediate": (
                        f"{source_side}Hand{source_finger}3"
                    ),
                    f"{target_side}{target_finger}Distal": (
                        f"{source_side}Hand{source_finger}4"
                    ),
                }
            )
    return mapping


_DETAILED_DELTA_MAPPING = _detailed_delta_mapping()


def _project_soma_to_skeleton(
    track: SomaMotion,
    *,
    skeleton: HumanoidSkeleton,
    mapping: dict[str, str],
) -> BodyTrack:
    """Project one validated SOMA track into an exact supported skeleton.

    The source Hips world transform is transferred to AutoAnim's artificial
    Root, and the target Hips offset is removed from root translation so target
    FK reproduces the source pelvis. Source bind-world rotations remove local
    joint-basis differences before mapped world-space deltas are solved back
    into the target hierarchy. The 77-to-25 joint reduction and different body
    proportions still make this a preview artifact, not a production retarget.
    """

    validate_soma_motion(track)
    frame_count = track.ticks.size
    rotations = np.zeros((frame_count, len(skeleton.joints), 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    soma_index = {name: index for index, name in enumerate(SOMASKEL77_NAMES)}
    body_index = {name: index for index, name in enumerate(skeleton.names)}
    source_world = _soma_world_rotations(
        track.local_rotations_xyzw, track.rest_world_rotations_xyzw
    )
    source_delta_world = _quaternion_multiply(
        source_world,
        _quaternion_inverse(track.rest_world_rotations_xyzw)[None, :, :],
    )
    target_world = np.zeros_like(rotations, dtype=np.float64)
    target_world[..., 3] = 1.0
    for target_index, joint in enumerate(skeleton.joints):
        source_name = mapping.get(joint.name)
        if source_name is None:
            if joint.parent >= 0:
                target_world[:, target_index] = target_world[:, joint.parent]
            continue
        desired_world = source_delta_world[:, soma_index[source_name]]
        if joint.parent == -1:
            target_local = desired_world
        else:
            target_local = _quaternion_multiply(
                _quaternion_inverse(target_world[:, joint.parent]), desired_world
            )
        target_local /= np.linalg.norm(target_local, axis=1, keepdims=True)
        rotations[:, target_index] = target_local.astype(np.float32)
        if joint.parent == -1:
            target_world[:, target_index] = target_local
        else:
            target_world[:, target_index] = _quaternion_multiply(
                target_world[:, joint.parent], target_local
            )
    source_hips_rotation = target_world[:, body_index["Root"]]

    gaze_direction = np.zeros((frame_count, 3), dtype=np.float32)
    gaze_direction[:, 2] = 1.0
    eye_rotations = np.zeros((frame_count, 2, 4), dtype=np.float32)
    eye_rotations[..., 3] = 1.0
    # BodyTrack v1 contacts are hard, stationary constraints. Provider labels
    # remain in SomaMotion until a contact solve has anchored the target feet.
    foot_contacts = np.zeros((frame_count, 2), dtype=np.bool_)
    hips_offset = np.asarray(
        skeleton.joints[body_index["Hips"]].rest_translation_m,
        dtype=np.float32,
    )
    root_translation = track.root_translation_m - _rotate_vector(
        source_hips_rotation,
        np.broadcast_to(hips_offset, (frame_count, 3)),
    ).astype(np.float32)
    return BodyTrack(
        duration_ticks=track.duration_ticks,
        ticks_per_second=TICKS_PER_SECOND,
        sample_rate_hz=track.sample_rate_hz,
        joint_names=skeleton.names,
        ticks=track.ticks,
        root_translation_m=root_translation,
        local_rotations_xyzw=rotations,
        foot_contacts=foot_contacts,
        gaze_direction_body=gaze_direction,
        gaze_strength=np.zeros(frame_count, dtype=np.float32),
        gnm_eye_rotations_xyzw=eye_rotations,
        source_plan_sha256=track.content_sha256(),
    )


def project_soma_to_body_track(track: SomaMotion) -> BodyTrack:
    """Create the legacy preview-only canonical-25 projection."""

    return _project_soma_to_skeleton(
        track,
        skeleton=CANONICAL_HUMANOID,
        mapping=_DELTA_MAPPING,
    )


def project_soma_to_detailed_body_track(track: SomaMotion) -> BodyTrack:
    """Preserve SOMA hand articulation in the append-only 55-joint runtime."""

    return _project_soma_to_skeleton(
        track,
        skeleton=DETAILED_HUMANOID,
        mapping=_DETAILED_DELTA_MAPPING,
    )
