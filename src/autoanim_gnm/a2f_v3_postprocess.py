"""Stateful Claire-v3 geometry postprocessing for local Audio2Face inference.

The equations in this module are a NumPy/SciPy port of the pinned MIT-licensed
NVIDIA Audio2Face-3D SDK (Copyright (c) 2025 NVIDIA Corporation & Affiliates),
specifically ``animator_cuda.cu``, ``animator.cpp``, ``math_utils.cpp``, and
the CPU blendshape solver.  It accepts the raw 88,831-value network prediction
and produces Claire skin/tongue controls plus the physical jaw transform.

Only the exact hash-verified public Claire-v3 profile is accepted.  The skin
interpolators and blendshape-solver temporal weights are deliberately stateful
so callers can process retained inference chunks without storing a whole clip.

Eye rotations apply NVIDIA's eyeball strength and per-eye offsets plus the
shared seeded saccade table.  The saccade clock advances at 30 Hz regardless
of output rate, matching the SDK's multi-track animator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from .a2f_v3_profile import OFFICIAL_V3_ASSET_SHA256, load_official_v3_claire_profile
from .calibrated_retarget import ClaireV3BlendshapeGeometry, SourceRigGeometry


RAW_SKIN_FLOATS = 72_006
RAW_TONGUE_FLOATS = 16_806
RAW_JAW_FLOATS = 15
RAW_EYE_FLOATS = 4
RAW_PREDICTION_FLOATS = (
    RAW_SKIN_FLOATS + RAW_TONGUE_FLOATS + RAW_JAW_FLOATS + RAW_EYE_FLOATS
)
SKIN_VERTEX_COUNT = RAW_SKIN_FLOATS // 3
TONGUE_VERTEX_COUNT = RAW_TONGUE_FLOATS // 3
JAW_POINT_COUNT = RAW_JAW_FLOATS // 3
EYE_POSTPROCESS_STATUS = "pinned_sdk_eye_animator_with_seeded_30hz_saccades"


class A2FV3PostprocessError(ValueError):
    """Raised when raw geometry, profile data, or solver state is invalid."""


def _verified_interpretation_snapshot(
    source_root: Path,
) -> tempfile.TemporaryDirectory[str]:
    """Copy exact verified Claire interpretation bytes from stable descriptors.

    The small interpretation bundle is snapshotted before any NPZ/JSON parser
    reopens it.  This prevents a path replacement from changing the bytes after
    provenance verification.  The large ONNX model has its own descriptor-bound
    verification in :mod:`a2f_v3_local`.
    """

    snapshot = tempfile.TemporaryDirectory(prefix="autoanim-a2f-v3-profile-")
    destination_root = Path(snapshot.name)
    try:
        for name, expected_sha256 in OFFICIAL_V3_ASSET_SHA256.items():
            if name == "network.onnx":
                continue
            source = source_root / name
            descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            try:
                digest = hashlib.sha256()
                with os.fdopen(os.dup(descriptor), "rb") as input_handle, (
                    destination_root / name
                ).open("xb") as output_handle:
                    while block := input_handle.read(1024 * 1024):
                        digest.update(block)
                        output_handle.write(block)
                if digest.hexdigest() != expected_sha256:
                    raise A2FV3PostprocessError(
                        f"Official v3 asset hash differs: {name}"
                    )
            finally:
                os.close(descriptor)
    except Exception:
        snapshot.cleanup()
        raise
    return snapshot


@dataclass(frozen=True, slots=True)
class ClaireV3RawPrediction:
    """Zero-copy, shape-checked views into a raw network prediction batch."""

    skin_deltas: np.ndarray
    tongue_deltas: np.ndarray
    jaw_deltas: np.ndarray
    eye_rotations_raw_degrees: np.ndarray


@dataclass(frozen=True, slots=True)
class ClaireV3PostprocessChunk:
    """One bounded chunk of postprocessed Claire-v3 output.

    ``jaw_transforms`` contains conventional 4x4 matrices.  The row-major and
    column-major flattened representations are both explicit because NVIDIA's
    Eigen implementation writes column-major memory while AutoAnim's sequence
    provider names its 16-value field as row-major.
    """

    skin_weights: np.ndarray
    tongue_weights: np.ndarray
    jaw_transforms: np.ndarray
    jaw_transform_row_major: np.ndarray
    jaw_transform_nvidia_column_major: np.ndarray
    jaw_rms_residual: np.ndarray
    eye_rotations_degrees: np.ndarray
    eye_rotations_raw_degrees: np.ndarray
    skin_geometry: np.ndarray | None
    tongue_geometry: np.ndarray | None
    eye_postprocess_status: str = EYE_POSTPROCESS_STATUS

    @classmethod
    def concatenate(
        cls, chunks: list[ClaireV3PostprocessChunk]
    ) -> ClaireV3PostprocessChunk:
        if not chunks:
            raise A2FV3PostprocessError("Cannot concatenate an empty chunk list")
        include_skin = chunks[0].skin_geometry is not None
        include_tongue = chunks[0].tongue_geometry is not None
        if any((item.skin_geometry is not None) != include_skin for item in chunks):
            raise A2FV3PostprocessError("Skin geometry retention changed between chunks")
        if any((item.tongue_geometry is not None) != include_tongue for item in chunks):
            raise A2FV3PostprocessError("Tongue geometry retention changed between chunks")

        def joined(name: str) -> np.ndarray:
            result = np.concatenate([getattr(item, name) for item in chunks], axis=0)
            result.setflags(write=False)
            return result

        skin_geometry = None
        if include_skin:
            skin_geometry = np.concatenate(
                [item.skin_geometry for item in chunks if item.skin_geometry is not None],
                axis=0,
            )
            skin_geometry.setflags(write=False)
        tongue_geometry = None
        if include_tongue:
            tongue_geometry = np.concatenate(
                [item.tongue_geometry for item in chunks if item.tongue_geometry is not None],
                axis=0,
            )
            tongue_geometry.setflags(write=False)
        return cls(
            skin_weights=joined("skin_weights"),
            tongue_weights=joined("tongue_weights"),
            jaw_transforms=joined("jaw_transforms"),
            jaw_transform_row_major=joined("jaw_transform_row_major"),
            jaw_transform_nvidia_column_major=joined(
                "jaw_transform_nvidia_column_major"
            ),
            jaw_rms_residual=joined("jaw_rms_residual"),
            eye_rotations_degrees=joined("eye_rotations_degrees"),
            eye_rotations_raw_degrees=joined("eye_rotations_raw_degrees"),
            skin_geometry=skin_geometry,
            tongue_geometry=tongue_geometry,
        )


@dataclass(frozen=True, slots=True)
class _AnimatorConfig:
    upper_face_smoothing: float
    lower_face_smoothing: float
    upper_face_strength: float
    lower_face_strength: float
    face_mask_level: float
    face_mask_softness: float
    skin_strength: float
    blink_strength: float
    lower_teeth_strength: float
    lower_teeth_height_offset: float
    lower_teeth_depth_offset: float
    lip_open_offset: float
    tongue_strength: float
    tongue_height_offset: float
    tongue_depth_offset: float
    eyelid_open_offset: float
    eyeballs_strength: float
    saccade_strength: float
    right_eye_rot_x_offset: float
    right_eye_rot_y_offset: float
    left_eye_rot_x_offset: float
    left_eye_rot_y_offset: float
    eye_saccade_seed: float


@dataclass(frozen=True, slots=True)
class _SolverConfig:
    active: np.ndarray
    cancel_groups: np.ndarray
    symmetry_groups: np.ndarray
    multipliers: np.ndarray
    offsets: np.ndarray
    l1_regularization: float
    l2_regularization: float
    temporal_regularization: float
    symmetry_regularization: float
    template_bb_size: float
    tolerance: float = 1.0e-10


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise A2FV3PostprocessError(
            f"{label} must have shape {shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise A2FV3PostprocessError(f"{label} contains non-finite values")
    return array


def split_claire_v3_raw_prediction(raw_prediction: np.ndarray) -> ClaireV3RawPrediction:
    """Validate and split ``[frames, 88831]`` network output into geometry views."""

    raw = np.asarray(raw_prediction)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2 or raw.shape[1] != RAW_PREDICTION_FLOATS or raw.shape[0] == 0:
        raise A2FV3PostprocessError(
            "Claire-v3 raw prediction must have shape "
            f"[frames,{RAW_PREDICTION_FLOATS}], got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise A2FV3PostprocessError(
            "Claire-v3 raw prediction must contain only finite numeric values"
        )
    skin_stop = RAW_SKIN_FLOATS
    tongue_stop = skin_stop + RAW_TONGUE_FLOATS
    jaw_stop = tongue_stop + RAW_JAW_FLOATS
    return ClaireV3RawPrediction(
        skin_deltas=raw[:, :skin_stop].reshape(-1, SKIN_VERTEX_COUNT, 3),
        tongue_deltas=raw[:, skin_stop:tongue_stop].reshape(
            -1, TONGUE_VERTEX_COUNT, 3
        ),
        jaw_deltas=raw[:, tongue_stop:jaw_stop].reshape(-1, JAW_POINT_COUNT, 3),
        eye_rotations_raw_degrees=raw[:, jaw_stop:],
    )


class _CascadedDegreeTwoInterpolator:
    """Stateful port of NVIDIA ``Interpolator`` with its fixed degree of two."""

    def __init__(self, smoothing_seconds: float, shape: tuple[int, ...]) -> None:
        if not np.isfinite(smoothing_seconds) or smoothing_seconds < 0.0:
            raise A2FV3PostprocessError("Interpolator smoothing must be finite and non-negative")
        if not shape or any(item <= 0 for item in shape):
            raise A2FV3PostprocessError("Interpolator shape must be non-empty and positive")
        self.smoothing_seconds = float(smoothing_seconds)
        self.shape = shape
        self._stage_one: np.ndarray | None = None
        self._stage_two: np.ndarray | None = None

    def reset(self) -> None:
        self._stage_one = None
        self._stage_two = None

    def update(self, raw: np.ndarray, dt_seconds: float) -> np.ndarray:
        value = _finite_array(raw, self.shape, "Interpolator input")
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise A2FV3PostprocessError("Interpolator dt must be finite and positive")
        if self._stage_one is None:
            # The CUDA implementation initializes every one of degree+1 states
            # to the first raw value before executing the ordinary update.
            self._stage_one = value.copy()
            self._stage_two = value.copy()
        assert self._stage_two is not None
        if self.smoothing_seconds > 0.0:
            alpha = 1.0 - 0.5 ** (float(dt_seconds) / self.smoothing_seconds)
            self._stage_one += (value - self._stage_one) * alpha
            self._stage_two += (self._stage_one - self._stage_two) * alpha
        else:
            self._stage_one[...] = value
            self._stage_two[...] = self._stage_one
        return self._stage_two.copy()


def _pair_indices(groups: np.ndarray, active_indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    active_groups = np.asarray(groups, dtype=np.int64)[active_indices]
    pairs: list[tuple[int, int]] = []
    for group in np.unique(active_groups):
        indices = np.flatnonzero(active_groups == group)
        if len(indices) == 2:
            pairs.append((int(indices[0]), int(indices[1])))
    return tuple(pairs)


class _RegularizedBVLSSolver:
    """Stateful CPU port of NVIDIA's regularized normal-equation BVLS solve."""

    def __init__(
        self,
        rig: SourceRigGeometry,
        config: _SolverConfig,
        *,
        mask: np.ndarray | None,
        label: str,
    ) -> None:
        validated = rig.validated(label)
        pose_count = len(validated.pose_names)
        active = np.asarray(config.active)
        cancel = np.asarray(config.cancel_groups)
        symmetry = np.asarray(config.symmetry_groups)
        multipliers = np.asarray(config.multipliers, dtype=np.float64)
        offsets = np.asarray(config.offsets, dtype=np.float64)
        expected = (pose_count,)
        solver_vectors = (active, cancel, symmetry, multipliers, offsets)
        if any(value.shape != expected for value in solver_vectors):
            raise A2FV3PostprocessError(
                f"{label} solver vectors must all have shape {expected}"
            )
        if np.any((active != 0) & (active != 1)) or not np.any(active):
            raise A2FV3PostprocessError(
                f"{label} active flags must be zero/one with at least one active pose"
            )
        if not np.isfinite(multipliers).all() or not np.isfinite(offsets).all():
            raise A2FV3PostprocessError(f"{label} weight transforms must be finite")
        scalars = (
            config.l1_regularization,
            config.l2_regularization,
            config.temporal_regularization,
            config.symmetry_regularization,
            config.template_bb_size,
            config.tolerance,
        )
        if not all(np.isfinite(value) for value in scalars) or any(
            value < 0.0 for value in scalars
        ):
            raise A2FV3PostprocessError(
                f"{label} solver scalars must be finite and non-negative"
            )
        if config.template_bb_size <= 0.0 or config.tolerance <= 0.0:
            raise A2FV3PostprocessError(
                f"{label} template size and tolerance must be positive"
            )

        for group_label, groups in (("cancel", cancel), ("symmetry", symmetry)):
            for group in np.unique(groups[groups >= 0]):
                if np.count_nonzero(groups == group) != 2:
                    raise A2FV3PostprocessError(
                        f"{label} {group_label} group {int(group)} must contain two poses"
                    )

        if mask is None:
            indices = np.arange(len(validated.neutral), dtype=np.int64)
        else:
            indices = np.asarray(mask, dtype=np.int64)
            if (
                indices.ndim != 1
                or indices.size == 0
                or np.any(indices < 0)
                or np.any(indices >= len(validated.neutral))
                or len(np.unique(indices)) != len(indices)
            ):
                raise A2FV3PostprocessError(f"{label} solver mask is invalid")

        self.label = label
        self.pose_names = validated.pose_names
        self.neutral = validated.neutral.astype(np.float64, copy=True)
        self.mask = indices.copy()
        self.active_indices = np.flatnonzero(active).astype(np.int64)
        self.multipliers = multipliers.copy()
        self.offsets = offsets.copy()
        self.temporal_regularization = float(config.temporal_regularization)
        self.tolerance = float(config.tolerance)
        active_deltas = validated.deltas[self.active_indices][:, self.mask, :]
        self._a_matrix = active_deltas.reshape(len(self.active_indices), -1).T
        if not np.isfinite(self._a_matrix).all():
            raise A2FV3PostprocessError(f"{label} active rig deltas are non-finite")

        extent = np.max(self.neutral, axis=0) - np.min(self.neutral, axis=0)
        self.scale_factor = float(
            (np.linalg.norm(extent) / config.template_bb_size) ** 2
        )
        if not np.isfinite(self.scale_factor) or self.scale_factor <= 0.0:
            raise A2FV3PostprocessError(f"{label} rig bounding box is degenerate")

        symmetry_pairs = _pair_indices(symmetry, self.active_indices)
        symmetry_matrix = np.zeros(
            (len(symmetry_pairs), len(self.active_indices)), dtype=np.float64
        )
        for row, (left, right) in enumerate(symmetry_pairs):
            symmetry_matrix[row, left] = 1.0
            symmetry_matrix[row, right] = -1.0

        count = len(self.active_indices)
        normal = self._a_matrix.T @ self._a_matrix
        normal += (
            config.l1_regularization**2
            * 0.25
            * self.scale_factor
            * np.ones((count, count), dtype=np.float64)
        )
        normal += (
            config.l2_regularization * 10.0 * self.scale_factor
            + config.temporal_regularization * 100.0 * self.scale_factor
        ) * np.eye(count, dtype=np.float64)
        if len(symmetry_matrix):
            normal += (
                config.symmetry_regularization
                * 10.0
                * self.scale_factor
                * (symmetry_matrix.T @ symmetry_matrix)
            )
        normal = 0.5 * (normal + normal.T)
        try:
            cholesky = np.linalg.cholesky(normal)
        except np.linalg.LinAlgError as exc:
            raise A2FV3PostprocessError(
                f"{label} regularized normal matrix is not positive definite"
            ) from exc
        self.normal_matrix = normal
        self._least_squares_matrix = cholesky.T
        self.cancel_pairs = _pair_indices(cancel, self.active_indices)
        self._previous = np.zeros(count, dtype=np.float64)

    def reset(self) -> None:
        self._previous.fill(0.0)

    def _bounded_solve(self, right_hand_side: np.ndarray, upper: np.ndarray) -> np.ndarray:
        target = np.linalg.solve(self._least_squares_matrix.T, right_hand_side)
        result = lsq_linear(
            self._least_squares_matrix,
            target,
            bounds=(np.zeros_like(upper), upper),
            method="bvls",
            tol=max(self.tolerance, 1.0e-12),
        )
        if not result.success or not np.isfinite(result.x).all():
            raise A2FV3PostprocessError(
                f"{self.label} bounded blendshape solve failed: {result.message}"
            )
        return np.asarray(result.x, dtype=np.float64)

    def solve_pose(self, target_pose: np.ndarray) -> np.ndarray:
        target = _finite_array(
            target_pose, self.neutral.shape, f"{self.label} target pose"
        )
        target_delta = (target[self.mask] - self.neutral[self.mask]).reshape(-1)
        right_hand_side = self._a_matrix.T @ target_delta
        right_hand_side += (
            self.temporal_regularization * self.scale_factor * self._previous
        )
        upper = np.ones(len(self.active_indices), dtype=np.float64)
        solved = self._bounded_solve(right_hand_side, upper)
        for left, right in self.cancel_pairs:
            upper[right if solved[left] >= solved[right] else left] = 1.0e-10
        if self.cancel_pairs:
            solved = self._bounded_solve(right_hand_side, upper)
        self._previous = solved
        full = np.zeros(len(self.pose_names), dtype=np.float64)
        full[self.active_indices] = solved
        full = full * self.multipliers + self.offsets
        if not np.isfinite(full).all():
            raise A2FV3PostprocessError(
                f"{self.label} post-solver weights contain non-finite values"
            )
        return full.astype(np.float32)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A2FV3PostprocessError(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise A2FV3PostprocessError(f"{label} must contain a JSON object")
    return value


def _load_animator_config(path: Path) -> _AnimatorConfig:
    document = _read_json(path, "Claire-v3 animator config")
    config = document.get("config")
    expected_keys = {
        "input_strength",
        "upper_face_smoothing",
        "lower_face_smoothing",
        "upper_face_strength",
        "lower_face_strength",
        "face_mask_level",
        "face_mask_softness",
        "skin_strength",
        "blink_strength",
        "lower_teeth_strength",
        "lower_teeth_height_offset",
        "lower_teeth_depth_offset",
        "lip_open_offset",
        "tongue_strength",
        "tongue_height_offset",
        "tongue_depth_offset",
        "eyeballs_strength",
        "saccade_strength",
        "right_eye_rot_x_offset",
        "right_eye_rot_y_offset",
        "left_eye_rot_x_offset",
        "left_eye_rot_y_offset",
        "eyelid_open_offset",
        "eye_saccade_seed",
    }
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise A2FV3PostprocessError(
            "Claire-v3 animator config keys differ from the pinned SDK schema"
        )
    try:
        numeric = {key: float(value) for key, value in config.items()}
    except (TypeError, ValueError) as exc:
        raise A2FV3PostprocessError(
            "Claire-v3 animator config values must be numeric"
        ) from exc
    if not all(np.isfinite(value) for value in numeric.values()):
        raise A2FV3PostprocessError("Claire-v3 animator config must be finite")
    if numeric["input_strength"] != 1.0:
        raise A2FV3PostprocessError(
            "Claire-v3 input strength differs from the pinned profile"
        )
    if numeric["face_mask_softness"] <= 0.0:
        raise A2FV3PostprocessError("Claire-v3 face-mask softness must be positive")
    if numeric["upper_face_smoothing"] < 0.0 or numeric["lower_face_smoothing"] < 0.0:
        raise A2FV3PostprocessError("Claire-v3 face smoothing cannot be negative")
    return _AnimatorConfig(
        upper_face_smoothing=numeric["upper_face_smoothing"],
        lower_face_smoothing=numeric["lower_face_smoothing"],
        upper_face_strength=numeric["upper_face_strength"],
        lower_face_strength=numeric["lower_face_strength"],
        face_mask_level=numeric["face_mask_level"],
        face_mask_softness=numeric["face_mask_softness"],
        skin_strength=numeric["skin_strength"],
        blink_strength=numeric["blink_strength"],
        lower_teeth_strength=numeric["lower_teeth_strength"],
        lower_teeth_height_offset=numeric["lower_teeth_height_offset"],
        lower_teeth_depth_offset=numeric["lower_teeth_depth_offset"],
        lip_open_offset=numeric["lip_open_offset"],
        tongue_strength=numeric["tongue_strength"],
        tongue_height_offset=numeric["tongue_height_offset"],
        tongue_depth_offset=numeric["tongue_depth_offset"],
        eyelid_open_offset=numeric["eyelid_open_offset"],
        eyeballs_strength=numeric["eyeballs_strength"],
        saccade_strength=numeric["saccade_strength"],
        right_eye_rot_x_offset=numeric["right_eye_rot_x_offset"],
        right_eye_rot_y_offset=numeric["right_eye_rot_y_offset"],
        left_eye_rot_x_offset=numeric["left_eye_rot_x_offset"],
        left_eye_rot_y_offset=numeric["left_eye_rot_y_offset"],
        eye_saccade_seed=numeric["eye_saccade_seed"],
    )


def _load_solver_config(path: Path, pose_count: int, label: str) -> _SolverConfig:
    document = _read_json(path, f"Claire-v3 {label} solver config")
    params = document.get("blendshape_params")
    expected_keys = {
        "strengthL2regularization",
        "strengthTemporalSmoothing",
        "strengthL1regularization",
        "strengthSymmetry",
        "numPoses",
        "bsSolveActivePoses",
        "bsSolveCancelPoses",
        "bsSolveSymmetryPoses",
        "bsWeightMultipliers",
        "bsWeightOffsets",
        "templateBBSize",
    }
    if not isinstance(params, dict) or set(params) != expected_keys:
        raise A2FV3PostprocessError(
            f"Claire-v3 {label} solver config differs from the pinned SDK schema"
        )
    if params.get("numPoses") != pose_count:
        raise A2FV3PostprocessError(
            f"Claire-v3 {label} solver must contain {pose_count} poses"
        )
    try:
        return _SolverConfig(
            active=np.asarray(params["bsSolveActivePoses"], dtype=np.int64),
            cancel_groups=np.asarray(params["bsSolveCancelPoses"], dtype=np.int64),
            symmetry_groups=np.asarray(params["bsSolveSymmetryPoses"], dtype=np.int64),
            multipliers=np.asarray(params["bsWeightMultipliers"], dtype=np.float64),
            offsets=np.asarray(params["bsWeightOffsets"], dtype=np.float64),
            l1_regularization=float(params["strengthL1regularization"]),
            l2_regularization=float(params["strengthL2regularization"]),
            temporal_regularization=float(params["strengthTemporalSmoothing"]),
            symmetry_regularization=float(params["strengthSymmetry"]),
            template_bb_size=float(params["templateBBSize"]),
        )
    except (TypeError, ValueError) as exc:
        raise A2FV3PostprocessError(
            f"Claire-v3 {label} solver config contains invalid values"
        ) from exc


def _face_mask_lower(neutral_skin: np.ndarray, level: float, softness: float) -> np.ndarray:
    neutral = np.asarray(neutral_skin, dtype=np.float64)
    if neutral.ndim != 2 or neutral.shape[1] != 3 or not np.isfinite(neutral).all():
        raise A2FV3PostprocessError("Skin neutral geometry must be finite [vertices,3]")
    y = neutral[:, 1]
    minimum = float(np.min(y))
    extent = float(np.max(y) - minimum)
    if extent <= 0.0 or not np.isfinite(extent):
        raise A2FV3PostprocessError("Skin neutral Y extent must be positive")
    if not np.isfinite(softness) or softness <= 0.0:
        raise A2FV3PostprocessError("Face-mask softness must be finite and positive")
    exponent = -(level - (y - minimum) / extent) / softness
    return 1.0 / (1.0 + np.exp(exponent))


def _jaw_kabsch(
    neutral_jaw: np.ndarray,
    jaw_delta: np.ndarray,
    *,
    strength: float,
    height_offset: float,
    depth_offset: float,
) -> tuple[np.ndarray, float]:
    neutral = _finite_array(neutral_jaw, (JAW_POINT_COUNT, 3), "Neutral jaw")
    delta = _finite_array(jaw_delta, (JAW_POINT_COUNT, 3), "Jaw delta")
    observed = neutral + delta * strength
    observed[:, 1] += height_offset
    observed[:, 2] += depth_offset
    a_mean = np.mean(observed, axis=0)
    b_mean = np.mean(neutral, axis=0)
    a_delta = observed - a_mean
    b_delta = neutral - b_mean
    try:
        left, _, right_t = np.linalg.svd(b_delta.T @ a_delta, full_matrices=True)
    except np.linalg.LinAlgError as exc:
        raise A2FV3PostprocessError("Jaw rigid transform SVD failed") from exc
    right = right_t.T
    handedness = np.eye(3, dtype=np.float64)
    handedness[2, 2] = np.linalg.det(right @ left.T)
    rotation = right @ handedness @ left.T
    translation = a_mean - b_mean @ rotation.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    predicted = neutral @ rotation.T + translation
    residual = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    if not np.isfinite(transform).all() or not np.isfinite(residual):
        raise A2FV3PostprocessError("Jaw rigid transform contains non-finite values")
    return transform.astype(np.float32), residual


class _EyeAnimator:
    """Stateful port of NVIDIA's four-channel eye/saccade animator."""

    def __init__(
        self,
        saccade_rotations: np.ndarray,
        *,
        eyeballs_strength: float,
        saccade_strength: float,
        right_offset: tuple[float, float],
        left_offset: tuple[float, float],
        saccade_seed: float,
    ) -> None:
        saccades = np.asarray(saccade_rotations, dtype=np.float32)
        if (
            saccades.ndim != 2
            or saccades.shape[1] != 2
            or len(saccades) == 0
            or not np.isfinite(saccades).all()
        ):
            raise A2FV3PostprocessError(
                "Eye saccade rotations must be a non-empty finite [frames,2] array"
            )
        parameters = np.asarray(
            [
                eyeballs_strength,
                saccade_strength,
                *right_offset,
                *left_offset,
                saccade_seed,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(parameters).all():
            raise A2FV3PostprocessError("Eye animator parameters must be finite")
        self.saccades = saccades.copy()
        self.eyeballs_strength = parameters[0]
        self.saccade_strength = parameters[1]
        self.offsets = parameters[2:6].copy()
        self.saccade_seed = parameters[6]
        self._live_time = np.float32(0.0)

    def reset(self) -> None:
        self._live_time = np.float32(0.0)

    def update(self, raw_rotations: np.ndarray, dt_seconds: float) -> np.ndarray:
        raw = np.asarray(raw_rotations, dtype=np.float32)
        if raw.shape != (4,) or not np.isfinite(raw).all():
            raise A2FV3PostprocessError(
                "Raw eye rotations must be one finite rightXY/leftXY vector"
            )
        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise A2FV3PostprocessError("Eye animator dt must be finite and positive")
        max_frames = np.float32(len(self.saccades))
        total_time = np.fmod(self.saccade_seed + self._live_time, max_frames)
        if total_time < 0.0:
            total_time += max_frames
        frame_index = int(total_time)
        shared_saccade = self.saccades[frame_index]
        output = self.offsets + raw * self.eyeballs_strength
        output[:2] += shared_saccade * self.saccade_strength
        output[2:] += shared_saccade * self.saccade_strength
        increment = np.float32(dt_seconds) * np.float32(30.0)
        self._live_time = np.fmod(self._live_time + increment, max_frames)
        if self._live_time < 0.0:
            self._live_time += max_frames
        if not np.isfinite(output).all():
            raise A2FV3PostprocessError("Final eye rotations contain non-finite values")
        return output.astype(np.float32)


class ClaireV3Postprocessor:
    """Hash-pinned, streamable Claire-v3 raw-geometry postprocessor."""

    def __init__(
        self,
        *,
        root: Path,
        geometry: ClaireV3BlendshapeGeometry,
        animator: _AnimatorConfig,
        skin_solver: _RegularizedBVLSSolver,
        tongue_solver: _RegularizedBVLSSolver,
        neutral_skin: np.ndarray,
        neutral_tongue: np.ndarray,
        neutral_jaw: np.ndarray,
        lip_open_delta: np.ndarray,
        eye_close_delta: np.ndarray,
        saccade_rotations: np.ndarray,
        profile_snapshot: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.root = root
        self.geometry = geometry
        self.animator = animator
        # Keep the verified snapshot alive for the processor lifetime so its
        # geometry root never aliases a removed/reused temporary path.
        self._profile_snapshot = profile_snapshot
        self.skin_solver = skin_solver
        self.tongue_solver = tongue_solver
        self.neutral_skin = neutral_skin
        self.neutral_tongue = neutral_tongue
        self.neutral_jaw = neutral_jaw
        self.lip_open_delta = lip_open_delta
        self.eye_close_delta = eye_close_delta
        self.face_mask_lower = _face_mask_lower(
            neutral_skin, animator.face_mask_level, animator.face_mask_softness
        )
        self._lower_interpolator = _CascadedDegreeTwoInterpolator(
            animator.lower_face_smoothing, neutral_skin.shape
        )
        self._upper_interpolator = _CascadedDegreeTwoInterpolator(
            animator.upper_face_smoothing, neutral_skin.shape
        )
        self._eye_animator = _EyeAnimator(
            saccade_rotations,
            eyeballs_strength=animator.eyeballs_strength,
            saccade_strength=animator.saccade_strength,
            right_offset=(
                animator.right_eye_rot_x_offset,
                animator.right_eye_rot_y_offset,
            ),
            left_offset=(
                animator.left_eye_rot_x_offset,
                animator.left_eye_rot_y_offset,
            ),
            saccade_seed=animator.eye_saccade_seed,
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> ClaireV3Postprocessor:
        source_root = Path(directory).expanduser().resolve()
        snapshot = _verified_interpretation_snapshot(source_root)
        root = Path(snapshot.name)
        # Every parser below reads only the descriptor-copied verified snapshot.
        profile = load_official_v3_claire_profile(root, verify_network=False)
        geometry = ClaireV3BlendshapeGeometry.load(root)
        if (
            profile.root != geometry.root
            or profile.skin_pose_names != geometry.skin.pose_names
            or profile.tongue_pose_names != geometry.tongue.pose_names
        ):
            raise A2FV3PostprocessError(
                "Claire-v3 profile and geometry identities do not match"
            )
        animator = _load_animator_config(root / "model_config_Claire.json")
        skin_config = _load_solver_config(
            root / "bs_skin_config_Claire.json", len(geometry.skin.pose_names), "skin"
        )
        tongue_config = _load_solver_config(
            root / "bs_tongue_config_Claire.json",
            len(geometry.tongue.pose_names),
            "tongue",
        )
        try:
            with np.load(root / "model_data_Claire.npz", allow_pickle=False) as values:
                neutral_skin = _finite_array(
                    values["neutral_skin"], (SKIN_VERTEX_COUNT, 3), "Neutral skin"
                ).copy()
                neutral_tongue = _finite_array(
                    values["neutral_tongue"],
                    (TONGUE_VERTEX_COUNT, 3),
                    "Neutral tongue",
                ).copy()
                neutral_jaw = _finite_array(
                    values["neutral_jaw"], (JAW_POINT_COUNT, 3), "Neutral jaw"
                ).copy()
                lip_open_delta = _finite_array(
                    values["lip_open_pose_delta"],
                    (SKIN_VERTEX_COUNT, 3),
                    "Lip-open helper delta",
                ).copy()
                eye_close_delta = _finite_array(
                    values["eye_close_pose_delta"],
                    (SKIN_VERTEX_COUNT, 3),
                    "Eye-close helper delta",
                ).copy()
                saccade = _finite_array(
                    values["saccade_rot_matrix"], (5_000, 2), "Saccade matrix"
                ).copy()
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, A2FV3PostprocessError):
                raise
            raise A2FV3PostprocessError(
                f"Could not load Claire-v3 model data: {exc}"
            ) from exc
        skin_solver = _RegularizedBVLSSolver(
            geometry.skin,
            skin_config,
            mask=geometry.skin.alignment_indices,
            label="Claire-v3 skin",
        )
        tongue_solver = _RegularizedBVLSSolver(
            geometry.tongue,
            tongue_config,
            mask=None,
            label="Claire-v3 tongue",
        )
        return cls(
            root=source_root,
            geometry=geometry,
            animator=animator,
            skin_solver=skin_solver,
            tongue_solver=tongue_solver,
            neutral_skin=neutral_skin,
            neutral_tongue=neutral_tongue,
            neutral_jaw=neutral_jaw,
            lip_open_delta=lip_open_delta,
            eye_close_delta=eye_close_delta,
            saccade_rotations=saccade,
            profile_snapshot=snapshot,
        )

    def reset(self) -> None:
        """Reset interpolation and solver temporal history for a new clip."""

        self._lower_interpolator.reset()
        self._upper_interpolator.reset()
        self.skin_solver.reset()
        self.tongue_solver.reset()
        self._eye_animator.reset()

    def process_chunk(
        self,
        raw_prediction: np.ndarray,
        *,
        dt_seconds: float = 1.0 / 60.0,
        include_geometry: bool = False,
    ) -> ClaireV3PostprocessChunk:
        """Process a retained frame chunk while carrying all temporal state."""

        if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise A2FV3PostprocessError("Postprocess dt must be finite and positive")
        raw = split_claire_v3_raw_prediction(raw_prediction)
        frame_count = len(raw.skin_deltas)
        skin_weights = np.empty(
            (frame_count, len(self.geometry.skin.pose_names)), dtype=np.float32
        )
        tongue_weights = np.empty(
            (frame_count, len(self.geometry.tongue.pose_names)), dtype=np.float32
        )
        jaw_transforms = np.empty((frame_count, 4, 4), dtype=np.float32)
        jaw_residual = np.empty(frame_count, dtype=np.float32)
        eye_rotations = np.empty((frame_count, RAW_EYE_FLOATS), dtype=np.float32)
        skin_geometry = (
            np.empty((frame_count, SKIN_VERTEX_COUNT, 3), dtype=np.float32)
            if include_geometry
            else None
        )
        tongue_geometry = (
            np.empty((frame_count, TONGUE_VERTEX_COUNT, 3), dtype=np.float32)
            if include_geometry
            else None
        )

        mask = self.face_mask_lower[:, None]
        for index in range(frame_count):
            skin_delta = (
                np.asarray(raw.skin_deltas[index], dtype=np.float64)
                * self.animator.skin_strength
            )
            skin_delta += self.eye_close_delta * (
                -self.animator.eyelid_open_offset
            )
            # The pinned profile's blink offset is zero; therefore the SDK's
            # blinkOffset*blinkStrength helper term vanishes exactly.
            skin_delta += self.lip_open_delta * self.animator.lip_open_offset
            lower = self._lower_interpolator.update(skin_delta, dt_seconds)
            upper = self._upper_interpolator.update(skin_delta, dt_seconds)
            skin_pose = self.neutral_skin + (
                upper * self.animator.upper_face_strength * (1.0 - mask)
                + lower * self.animator.lower_face_strength * mask
            )
            tongue_pose = (
                self.neutral_tongue
                + np.asarray(raw.tongue_deltas[index], dtype=np.float64)
                * self.animator.tongue_strength
            )
            tongue_pose[:, 1] += self.animator.tongue_height_offset
            tongue_pose[:, 2] += self.animator.tongue_depth_offset
            if not np.isfinite(skin_pose).all() or not np.isfinite(tongue_pose).all():
                raise A2FV3PostprocessError(
                    "Claire-v3 animated geometry contains non-finite values"
                )
            skin_weights[index] = self.skin_solver.solve_pose(skin_pose)
            tongue_weights[index] = self.tongue_solver.solve_pose(tongue_pose)
            jaw_transforms[index], jaw_residual[index] = _jaw_kabsch(
                self.neutral_jaw,
                raw.jaw_deltas[index],
                strength=self.animator.lower_teeth_strength,
                height_offset=self.animator.lower_teeth_height_offset,
                depth_offset=self.animator.lower_teeth_depth_offset,
            )
            eye_rotations[index] = self._eye_animator.update(
                raw.eye_rotations_raw_degrees[index], dt_seconds
            )
            if skin_geometry is not None:
                skin_geometry[index] = skin_pose.astype(np.float32)
            if tongue_geometry is not None:
                tongue_geometry[index] = tongue_pose.astype(np.float32)

        row_major = jaw_transforms.reshape(frame_count, 16).copy()
        column_major = np.stack(
            [matrix.reshape(16, order="F") for matrix in jaw_transforms], axis=0
        ).astype(np.float32)
        eyes = np.asarray(raw.eye_rotations_raw_degrees, dtype=np.float32).copy()
        values = (
            skin_weights,
            tongue_weights,
            jaw_transforms,
            row_major,
            column_major,
            jaw_residual,
            eye_rotations,
            eyes,
        )
        if not all(np.isfinite(value).all() for value in values):
            raise A2FV3PostprocessError("Postprocessed Claire-v3 output is non-finite")
        for value in values:
            value.setflags(write=False)
        if skin_geometry is not None:
            skin_geometry.setflags(write=False)
        if tongue_geometry is not None:
            tongue_geometry.setflags(write=False)
        return ClaireV3PostprocessChunk(
            skin_weights=skin_weights,
            tongue_weights=tongue_weights,
            jaw_transforms=jaw_transforms,
            jaw_transform_row_major=row_major,
            jaw_transform_nvidia_column_major=column_major,
            jaw_rms_residual=jaw_residual,
            eye_rotations_degrees=eye_rotations,
            eye_rotations_raw_degrees=eyes,
            skin_geometry=skin_geometry,
            tongue_geometry=tongue_geometry,
        )

    def process_sequence(
        self,
        raw_prediction: np.ndarray,
        *,
        dt_seconds: float = 1.0 / 60.0,
        include_geometry: bool = False,
        reset: bool = True,
    ) -> ClaireV3PostprocessChunk:
        """Convenience whole-sequence wrapper, primarily for qualification tests."""

        if reset:
            self.reset()
        return self.process_chunk(
            raw_prediction,
            dt_seconds=dt_seconds,
            include_geometry=include_geometry,
        )


__all__ = [
    "A2FV3PostprocessError",
    "ClaireV3PostprocessChunk",
    "ClaireV3Postprocessor",
    "ClaireV3RawPrediction",
    "EYE_POSTPROCESS_STATUS",
    "RAW_EYE_FLOATS",
    "RAW_JAW_FLOATS",
    "RAW_PREDICTION_FLOATS",
    "RAW_SKIN_FLOATS",
    "RAW_TONGUE_FLOATS",
    "split_claire_v3_raw_prediction",
]
