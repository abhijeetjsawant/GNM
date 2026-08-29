"""Validated native speech motion and direct SMPL-X55 body retargeting.

The learned generator is an untrusted external provider.  AutoAnim consumes a
small numeric contract, independently validates its timing and kinematics, and
then transfers rest-relative world-space rotation deltas directly into the
existing detailed 55-joint body.  GNM-owned jaw, eye, and facial channels are
never represented in the resulting :class:`BodyTrack`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
import json
import math
from typing import Any

import numpy as np

from .acting import TICKS_PER_SECOND
from .body import (
    DETAILED_HUMANOID,
    MAX_DURATION_TICKS,
    MAX_SAMPLE_RATE_HZ,
    BodyTrack,
    _quaternion_multiply,
    _rotate_vector,
)


SMPLX55_SCHEMA_VERSION = "autoanim.smplx55-skeleton/1.0"
NATIVE_SPEECH_MOTION_SCHEMA_VERSION = "autoanim.native-speech-motion/1.0"
GESTURELSM_PROVIDER_ID = "gesturelsm_shortcut_reflow"
MAX_NATIVE_FRAMES = 216_001
QUATERNION_TOLERANCE = 2e-5
FK_TOLERANCE_M = 0.002


class SpeechMotionValidationError(ValueError):
    """A generated motion artifact violated the trusted application contract."""


def _load_smplx55_contract() -> dict[str, Any]:
    path = resources.files("autoanim_gnm").joinpath("data/smplx55-v1.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "coordinate_system",
        "joint_names",
        "parents",
        "target_to_source",
        "gnm_owned_source_joints",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("Bundled SMPL-X55 contract fields differ")
    if value["schema_version"] != SMPLX55_SCHEMA_VERSION:
        raise RuntimeError("Bundled SMPL-X55 contract version differs")
    names = value["joint_names"]
    parents = value["parents"]
    if (
        not isinstance(names, list)
        or len(names) != 55
        or len(set(names)) != 55
        or any(type(name) is not str or not name for name in names)
        or not isinstance(parents, list)
        or len(parents) != len(names)
        or parents[0] != -1
        or any(type(parent) is not int for parent in parents)
        or any(
            parent < 0 or parent >= index
            for index, parent in enumerate(parents[1:], start=1)
        )
    ):
        raise RuntimeError("Bundled SMPL-X55 hierarchy is invalid")
    mapping = value["target_to_source"]
    if (
        not isinstance(mapping, dict)
        or not set(mapping).issubset(DETAILED_HUMANOID.names)
        or not set(mapping.values()).issubset(names)
        or "Root" in mapping
        or mapping.get("Hips") != "pelvis"
    ):
        raise RuntimeError("Bundled SMPL-X55 mapping is invalid")
    gnm_owned = value["gnm_owned_source_joints"]
    if set(gnm_owned) != {"jaw", "left_eye_smplhf", "right_eye_smplhf"}:
        raise RuntimeError("Bundled SMPL-X55 GNM ownership is invalid")
    return value


_SMPLX55_CONTRACT = _load_smplx55_contract()
SMPLX55_NAMES = tuple(_SMPLX55_CONTRACT["joint_names"])
SMPLX55_PARENTS = tuple(int(value) for value in _SMPLX55_CONTRACT["parents"])
SMPLX55_TARGET_TO_SOURCE = dict(_SMPLX55_CONTRACT["target_to_source"])
SMPLX55_GNM_OWNED = tuple(_SMPLX55_CONTRACT["gnm_owned_source_joints"])
SMPLX55_SEMANTIC_SHA256 = sha256(
    json.dumps(
        _SMPLX55_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.array(value, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpeechMotionValidationError(f"{label} must be a lowercase SHA-256")


def _validate_git_oid(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpeechMotionValidationError(
            f"{label} must be a 40-character Git object ID"
        )


def axis_angle_to_quaternion(axis_angle_rad: np.ndarray) -> np.ndarray:
    """Convert finite local axis-angle radians to continuous XYZW quaternions."""

    values = np.asarray(axis_angle_rad, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise SpeechMotionValidationError("Axis-angle rotations must be finite [...,3]")
    angles = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(angles > 8.0 * math.pi):
        raise SpeechMotionValidationError("Axis-angle rotations exceed the safety limit")
    half = 0.5 * angles
    scale = np.empty_like(angles)
    small = angles < 1e-8
    scale[small] = 0.5
    scale[~small] = np.sin(half[~small]) / angles[~small]
    output = np.concatenate((values * scale, np.cos(half)), axis=-1)
    output /= np.linalg.norm(output, axis=-1, keepdims=True)
    if output.ndim >= 3:
        for frame in range(1, output.shape[0]):
            dots = np.sum(output[frame] * output[frame - 1], axis=-1)
            output[frame, dots < 0.0] *= -1.0
    return output.astype(np.float32)


def _quaternion_inverse(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    inverse = quaternion.copy()
    inverse[..., :3] *= -1.0
    return inverse / np.sum(quaternion * quaternion, axis=-1, keepdims=True)


def _smplx_world_rotations(
    local_rotations_xyzw: np.ndarray,
    rest_world_rotations_xyzw: np.ndarray,
) -> np.ndarray:
    local = np.asarray(local_rotations_xyzw, dtype=np.float64)
    rest_world = np.asarray(rest_world_rotations_xyzw, dtype=np.float64)
    frames = local.shape[0]
    output = np.zeros((frames, len(SMPLX55_NAMES), 4), dtype=np.float64)
    output[..., 3] = 1.0
    for index, parent in enumerate(SMPLX55_PARENTS):
        if parent == -1:
            output[:, index] = _quaternion_multiply(rest_world[index], local[:, index])
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


def smplx55_forward_kinematics(
    root_translation_m: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    rest_joint_positions_m: np.ndarray,
    rest_world_rotations_xyzw: np.ndarray,
) -> np.ndarray:
    """Evaluate the declared SMPL-X55 skeleton independently of the worker."""

    roots = np.asarray(root_translation_m, dtype=np.float64)
    local = np.asarray(local_rotations_xyzw, dtype=np.float64)
    rest_positions = np.asarray(rest_joint_positions_m, dtype=np.float64)
    rest_world = np.asarray(rest_world_rotations_xyzw, dtype=np.float64)
    frames = roots.shape[0]
    if (
        roots.shape != (frames, 3)
        or local.shape != (frames, 55, 4)
        or rest_positions.shape != (55, 3)
        or rest_world.shape != (55, 4)
    ):
        raise SpeechMotionValidationError("SMPL-X55 FK arrays have invalid shapes")
    positions = np.zeros((frames, 55, 3), dtype=np.float64)
    world = np.zeros((frames, 55, 4), dtype=np.float64)
    world[..., 3] = 1.0
    for index, parent in enumerate(SMPLX55_PARENTS):
        if parent == -1:
            positions[:, index] = roots
            world[:, index] = _quaternion_multiply(rest_world[index], local[:, index])
        else:
            rest_offset = _rotate_vector(
                _quaternion_inverse(rest_world[parent]),
                rest_positions[index] - rest_positions[parent],
            )
            positions[:, index] = positions[:, parent] + _rotate_vector(
                world[:, parent], np.broadcast_to(rest_offset, (frames, 3))
            )
            rest_local = _quaternion_multiply(
                _quaternion_inverse(rest_world[parent]), rest_world[index]
            )
            world[:, index] = _quaternion_multiply(
                _quaternion_multiply(world[:, parent], rest_local), local[:, index]
            )
    return positions.astype(np.float32)


@dataclass(frozen=True, slots=True)
class NativeSpeechMotion:
    """Immutable motion generated from one audio/transcript candidate seed."""

    provider_id: str
    provider_git_commit_oid: str
    model_revision: str
    model_artifact_sha256: str
    candidate_seed: int
    duration_ticks: int
    sample_rate_hz: int
    ticks: np.ndarray
    root_translation_m: np.ndarray
    local_rotations_xyzw: np.ndarray
    rest_joint_positions_m: np.ndarray
    rest_world_rotations_xyzw: np.ndarray
    joint_positions_m: np.ndarray
    audio_sha256: str
    transcript_sha256: str
    alignment_sha256: str

    def __post_init__(self) -> None:
        raw_ticks = np.asarray(self.ticks)
        if raw_ticks.dtype.kind not in "iu" or raw_ticks.dtype.kind == "b":
            raise SpeechMotionValidationError("Native speech-motion ticks must be integers")
        object.__setattr__(self, "ticks", _readonly(raw_ticks, np.int64))
        for name in (
            "root_translation_m",
            "local_rotations_xyzw",
            "rest_joint_positions_m",
            "rest_world_rotations_xyzw",
            "joint_positions_m",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.float32))
        validate_native_speech_motion(self)

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_SPEECH_MOTION_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "provider_git_commit_oid": self.provider_git_commit_oid,
            "model_revision": self.model_revision,
            "model_artifact_sha256": self.model_artifact_sha256,
            "candidate_seed": self.candidate_seed,
            "duration_ticks": self.duration_ticks,
            "sample_rate_hz": self.sample_rate_hz,
            "frame_count": int(self.ticks.size),
            "skeleton": {
                "schema_version": SMPLX55_SCHEMA_VERSION,
                "semantic_sha256": SMPLX55_SEMANTIC_SHA256,
                "joint_names": list(SMPLX55_NAMES),
            },
            "source": {
                "audio_sha256": self.audio_sha256,
                "transcript_sha256": self.transcript_sha256,
                "alignment_sha256": self.alignment_sha256,
            },
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
            "root_translation_m",
            "local_rotations_xyzw",
            "rest_joint_positions_m",
            "rest_world_rotations_xyzw",
            "joint_positions_m",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()


def validate_native_speech_motion(track: NativeSpeechMotion) -> None:
    if track.provider_id != GESTURELSM_PROVIDER_ID:
        raise SpeechMotionValidationError("Unsupported speech-motion provider")
    _validate_git_oid(track.provider_git_commit_oid, "provider_git_commit_oid")
    _validate_git_oid(track.model_revision, "model_revision")
    _validate_sha256(track.model_artifact_sha256, "model_artifact_sha256")
    _validate_sha256(track.audio_sha256, "audio_sha256")
    _validate_sha256(track.transcript_sha256, "transcript_sha256")
    _validate_sha256(track.alignment_sha256, "alignment_sha256")
    if (
        type(track.candidate_seed) is not int
        or track.candidate_seed < 0
        or track.candidate_seed > 2**63 - 1
    ):
        raise SpeechMotionValidationError("candidate_seed is outside the uint63 range")
    if (
        type(track.duration_ticks) is not int
        or track.duration_ticks <= 0
        or track.duration_ticks > MAX_DURATION_TICKS
    ):
        raise SpeechMotionValidationError("duration_ticks is outside supported limits")
    if (
        type(track.sample_rate_hz) is not int
        or track.sample_rate_hz <= 0
        or track.sample_rate_hz > MAX_SAMPLE_RATE_HZ
        or TICKS_PER_SECOND % track.sample_rate_hz
    ):
        raise SpeechMotionValidationError(
            "sample_rate_hz must divide 48 kHz and be at most 120"
        )
    frames = track.ticks.size
    if frames < 2 or frames > MAX_NATIVE_FRAMES:
        raise SpeechMotionValidationError("Native speech-motion frame count is unsupported")
    if (
        track.ticks.shape != (frames,)
        or track.ticks[0] != 0
        or track.ticks[-1] != track.duration_ticks
        or np.any(track.ticks[1:] <= track.ticks[:-1])
    ):
        raise SpeechMotionValidationError(
            "Native ticks must increase strictly from zero through duration"
        )
    expected_shapes = {
        "root_translation_m": (frames, 3),
        "local_rotations_xyzw": (frames, 55, 4),
        "rest_joint_positions_m": (55, 3),
        "rest_world_rotations_xyzw": (55, 4),
        "joint_positions_m": (frames, 55, 3),
    }
    for name, shape in expected_shapes.items():
        value = getattr(track, name)
        if value.shape != shape or not np.isfinite(value).all():
            raise SpeechMotionValidationError(f"{name} has invalid shape or values")
    if not np.allclose(
        np.linalg.norm(track.local_rotations_xyzw, axis=2),
        1.0,
        rtol=0.0,
        atol=QUATERNION_TOLERANCE,
    ):
        raise SpeechMotionValidationError("Native joint quaternions must be normalized")
    if not np.allclose(
        np.linalg.norm(track.rest_world_rotations_xyzw, axis=1),
        1.0,
        rtol=0.0,
        atol=QUATERNION_TOLERANCE,
    ):
        raise SpeechMotionValidationError("Native rest quaternions must be normalized")
    adjacent_dots = np.sum(
        track.local_rotations_xyzw[1:] * track.local_rotations_xyzw[:-1], axis=2
    )
    if np.any(adjacent_dots < -QUATERNION_TOLERANCE):
        raise SpeechMotionValidationError("Native quaternion signs must be continuous")
    if not np.allclose(track.rest_joint_positions_m[0], 0.0, rtol=0.0, atol=1e-6):
        raise SpeechMotionValidationError("SMPL-X rest pelvis must be at the origin")
    fk = smplx55_forward_kinematics(
        track.root_translation_m,
        track.local_rotations_xyzw,
        track.rest_joint_positions_m,
        track.rest_world_rotations_xyzw,
    )
    error = float(
        np.max(np.linalg.norm(fk - track.joint_positions_m, axis=2), initial=0.0)
    )
    if error > FK_TOLERANCE_M:
        raise SpeechMotionValidationError(
            f"SMPL-X joint positions disagree with FK by {error:.6f} m"
        )


def retarget_smplx55_samples_to_autoanim55(
    *,
    root_translation_m: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    rest_world_rotations_xyzw: np.ndarray,
    ticks: np.ndarray,
    duration_ticks: int,
    sample_rate_hz: int,
    source_sha256: str,
    source_to_canonical_rotation_xyzw: np.ndarray | None = None,
    source_orientation_mode: str = "conjugate",
) -> BodyTrack:
    """Transfer already-validated SMPL-X55 samples into AutoAnim-55.

    Provider adapters own their source schema validation. This shared boundary
    validates the exact numeric/timing subset used by retargeting so video and
    audio motion providers cannot silently diverge in coordinate handling.

    ``conjugate`` is appropriate when source and target body-local axes use the
    same world-basis change.  ``world_pre_rotate`` is for captures such as
    MAMMA where SMPL-X's root orientation maps body-local axes into a different
    calibrated capture world: the world basis is applied on the left only,
    before the shared left/right anatomical reflection is applied.
    """

    roots = np.asarray(root_translation_m, dtype=np.float32)
    local_rotations = np.asarray(local_rotations_xyzw, dtype=np.float32)
    rest_world = np.asarray(rest_world_rotations_xyzw, dtype=np.float32)
    basis = np.asarray(
        (0.0, 0.0, 0.0, 1.0)
        if source_to_canonical_rotation_xyzw is None
        else source_to_canonical_rotation_xyzw,
        dtype=np.float64,
    )
    tick_values = np.asarray(ticks)
    frames = tick_values.size
    _validate_sha256(source_sha256, "source_sha256")
    if (
        frames < 2
        or roots.shape != (frames, 3)
        or local_rotations.shape != (frames, 55, 4)
        or rest_world.shape != (55, 4)
        or basis.shape != (4,)
        or tick_values.shape != (frames,)
        or tick_values.dtype.kind not in "iu"
        or not np.isfinite(roots).all()
        or not np.isfinite(local_rotations).all()
        or not np.isfinite(rest_world).all()
        or not np.isfinite(basis).all()
        or source_orientation_mode not in {"conjugate", "world_pre_rotate"}
    ):
        raise SpeechMotionValidationError("Retarget SMPL-X55 samples are invalid")
    basis_norm = float(np.linalg.norm(basis))
    if abs(basis_norm - 1.0) > QUATERNION_TOLERANCE:
        raise SpeechMotionValidationError("Source-to-canonical rotation is not normalized")
    if not np.allclose(
        np.linalg.norm(local_rotations, axis=2),
        1.0,
        rtol=0.0,
        atol=QUATERNION_TOLERANCE,
    ) or not np.allclose(
        np.linalg.norm(rest_world, axis=1),
        1.0,
        rtol=0.0,
        atol=QUATERNION_TOLERANCE,
    ):
        raise SpeechMotionValidationError("Retarget SMPL-X55 quaternions are not normalized")
    source_index = {name: index for index, name in enumerate(SMPLX55_NAMES)}
    target_index = {name: index for index, name in enumerate(DETAILED_HUMANOID.names)}
    source_world = _smplx_world_rotations(
        local_rotations, rest_world
    )
    source_delta = _quaternion_multiply(
        source_world,
        _quaternion_inverse(rest_world)[None, :, :],
    )
    if source_orientation_mode == "conjugate":
        source_delta = _quaternion_multiply(
            _quaternion_multiply(basis, source_delta), _quaternion_inverse(basis)
        )
    else:
        source_delta = _quaternion_multiply(basis, source_delta)
    # SMPL-X uses anatomical-left at +X; AutoAnim's right-handed character
    # contract uses anatomical-left at -X. Reflect rotations through the YZ
    # plane before applying them to the target hierarchy. A rotation is an
    # axial vector, so R_target = M R_source M for M=diag(-1,1,1), which maps
    # quaternion [x,y,z,w] to [x,-y,-z,w]. Omitting this turns native downward
    # shoulder motion into raised arms while still producing normalized quats.
    source_delta = source_delta.copy()
    source_delta[..., 1:3] *= -1.0
    target_local = np.zeros((frames, 55, 4), dtype=np.float64)
    target_local[..., 3] = 1.0
    target_world = np.zeros_like(target_local)
    target_world[..., 3] = 1.0
    for index, joint in enumerate(DETAILED_HUMANOID.joints):
        source_name = SMPLX55_TARGET_TO_SOURCE.get(joint.name)
        if source_name is None:
            if joint.parent >= 0:
                target_world[:, index] = target_world[:, joint.parent]
            continue
        desired_world = source_delta[:, source_index[source_name]]
        if joint.parent == -1:
            local = desired_world
        else:
            local = _quaternion_multiply(
                _quaternion_inverse(target_world[:, joint.parent]), desired_world
            )
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        for frame in range(1, frames):
            if np.dot(local[frame], local[frame - 1]) < 0.0:
                local[frame] *= -1.0
        target_local[:, index] = local
        if joint.parent == -1:
            target_world[:, index] = local
        else:
            target_world[:, index] = _quaternion_multiply(
                target_world[:, joint.parent], local
            )
    hips_offset = np.asarray(
        DETAILED_HUMANOID.joints[target_index["Hips"]].rest_translation_m,
        dtype=np.float64,
    )
    reflected_root_translation = _rotate_vector(
        np.broadcast_to(basis, (frames, 4)), roots
    ).astype(np.float32)
    reflected_root_translation[:, 0] *= -1.0
    root_translation = reflected_root_translation - _rotate_vector(
        target_world[:, target_index["Root"]],
        np.broadcast_to(hips_offset, (frames, 3)),
    ).astype(np.float32)
    gaze_direction = np.zeros((frames, 3), dtype=np.float32)
    gaze_direction[:, 2] = 1.0
    eyes = np.zeros((frames, 2, 4), dtype=np.float32)
    eyes[..., 3] = 1.0
    return BodyTrack(
        duration_ticks=duration_ticks,
        ticks_per_second=TICKS_PER_SECOND,
        sample_rate_hz=sample_rate_hz,
        joint_names=DETAILED_HUMANOID.names,
        ticks=tick_values,
        root_translation_m=root_translation,
        local_rotations_xyzw=target_local.astype(np.float32),
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=gaze_direction,
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=eyes,
        source_plan_sha256=source_sha256,
    )


def retarget_smplx55_to_autoanim55(track: NativeSpeechMotion) -> BodyTrack:
    """Transfer one validated native candidate directly into AutoAnim-55."""

    validate_native_speech_motion(track)
    return retarget_smplx55_samples_to_autoanim55(
        root_translation_m=track.root_translation_m,
        local_rotations_xyzw=track.local_rotations_xyzw,
        rest_world_rotations_xyzw=track.rest_world_rotations_xyzw,
        ticks=track.ticks,
        duration_ticks=track.duration_ticks,
        sample_rate_hz=track.sample_rate_hz,
        source_sha256=track.content_sha256(),
    )


__all__ = [
    "FK_TOLERANCE_M",
    "GESTURELSM_PROVIDER_ID",
    "NATIVE_SPEECH_MOTION_SCHEMA_VERSION",
    "NativeSpeechMotion",
    "SMPLX55_GNM_OWNED",
    "SMPLX55_NAMES",
    "SMPLX55_PARENTS",
    "SMPLX55_SCHEMA_VERSION",
    "SMPLX55_SEMANTIC_SHA256",
    "SMPLX55_TARGET_TO_SOURCE",
    "SpeechMotionValidationError",
    "axis_angle_to_quaternion",
    "retarget_smplx55_to_autoanim55",
    "retarget_smplx55_samples_to_autoanim55",
    "smplx55_forward_kinematics",
    "validate_native_speech_motion",
]
