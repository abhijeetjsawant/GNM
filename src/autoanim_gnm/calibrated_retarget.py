"""Geometry-calibrated Claire/ARKit to GNM expression retargeting.

The semantic retargeter in :mod:`autoanim_gnm.a2f` deliberately maps named
ARKit controls to a small set of human-readable expression prototypes.  That
is useful as a fallback, but it discards controls that do not have an authored
rule and collapses asymmetric controls into the same prototype.  This module
builds a dense, deterministic mapping from every released Claire skin and
tongue control into GNM's native expression basis.

Calibration has three stages:

* robust similarity alignment of the two neutral point clouds;
* confidence-weighted k-nearest surface correspondence (topology independent);
* per-GNM-region bounded ridge least squares against the actual vertex basis.

The result is a small versioned NPZ cache.  Runtime retargeting is only a pair
of matrix multiplies followed by the same per-region magnitude contract used
by :class:`autoanim_gnm.rig.ControlRig`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from .a2f import ClaireSkinAssets, ClaireTongueAssets
from .gnm_adapter import GNMAdapter


CALIBRATION_FORMAT_VERSION = 1
CALIBRATION_ALGORITHM = "claire-arkit-to-gnm-dense-v1"
EXPECTED_GNM_EXPRESSION_DIM = 383

# Audio2Face-3D v3 does not embed its Hugging Face revision in the runtime
# files.  The loader below pins provenance through exact published hashes, an
# immutable Hugging Face snapshot directory, or the small manifest below.
CLAIRE_V3_MODEL_ID = "nvidia/Audio2Face-3D-v3.0"
CLAIRE_V3_HF_REVISION = "b74132732fd9a9d29b237bec193ded64c9745e91"
CLAIRE_V3_NETWORK_VERSION = "3.2"
CLAIRE_V3_SKIN_RIG_VERSION = "v3.6"
CLAIRE_V3_TONGUE_RIG_VERSION = "v1.0"
CLAIRE_V3_ASSET_MANIFEST_SCHEMA = "autoanim.a2f-v3-assets/1.0"
CLAIRE_V3_ASSET_MANIFEST_FILENAME = "autoanim-a2f-v3-assets.json"

# Exact files published at CLAIRE_V3_HF_REVISION.  This permits a deliberately
# minimal Claire-only runtime profile (without model.json or the 2.4 GB TRT
# engine) to retain cryptographically verifiable provenance.
CLAIRE_V3_PROFILE_SHA256: dict[str, str] = {
    "network_info.json": "5524cdbe96a6bc89c78f06f32ae959e2302c50c663f407cb2b392c0ecac5975d",
    "model_data_Claire.npz": "4f05331263fa609321335e55c20922f4d6709d33160d368c3b537f019429ea4f",
    "bs_skin_Claire.npz": "bcb1fde2c7384fe9ec3cf9932b0fdeeda01fe4a1e42bba3817bba14e7f1716d3",
    "bs_tongue_Claire.npz": "812f10c34edb6ab6f36aedfe1d59a79d8190a5a8ee0a6071382f6bae9e3413b6",
    "bs_skin_config_Claire.json": "e2b508c5d17f1fb01c3a5b0292072d09e66e8c55bc23fcbe0c9aee8f8eae1713",
    "bs_tongue_config_Claire.json": "ace4b0b6b9be280f96a66568bd13ac4ea1fddf9c690464ab450fe339d9752e98",
}

CLAIRE_V3_SKIN_POSE_NAMES: tuple[str, ...] = (
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
)

CLAIRE_V3_TONGUE_POSE_NAMES: tuple[str, ...] = (
    "tongueTipUp",
    "tongueTipDown",
    "tongueTipLeft",
    "tongueTipRight",
    "tongueRollUp",
    "tongueRollDown",
    "tongueRollLeft",
    "tongueRollRight",
    "tongueUp",
    "tongueDown",
    "tongueLeft",
    "tongueRight",
    "tongueIn",
    "tongueStretch",
    "tongueWide",
    "tongueNarrow",
)

CLAIRE_V3_SKIN_ACTIVE: tuple[int, ...] = (
    1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
    1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
)
CLAIRE_V3_TONGUE_MULTIPLIERS: tuple[float, ...] = (
    2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0,
    2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
)
CLAIRE_V3_TONGUE_OFFSETS: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)


class CalibratedRetargetError(ValueError):
    """Raised when calibration inputs, a cache, or runtime weights are invalid."""


class CalibrationCacheMismatch(CalibratedRetargetError):
    """Raised when a valid cache was built for different assets or settings."""


@dataclass(frozen=True)
class RegionSpec:
    """One independently bounded GNM expression coefficient region."""

    name: str
    start: int
    stop: int
    bound: float = 3.0

    def __post_init__(self) -> None:
        if not self.name:
            raise CalibratedRetargetError("A calibration region needs a name")
        if self.start < 0 or self.stop <= self.start:
            raise CalibratedRetargetError(
                f"Invalid coefficient range for region {self.name!r}: "
                f"[{self.start}, {self.stop})"
            )
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise CalibratedRetargetError(
                f"Region {self.name!r} needs a finite positive coefficient bound"
            )


GNM_REGION_SPECS: tuple[RegionSpec, ...] = (
    RegionSpec("left_eye", 0, 100),
    RegionSpec("right_eye", 100, 200),
    RegionSpec("lower_face", 200, 350),
    RegionSpec("tongue", 350, 382),
    RegionSpec("pupils", 382, 383),
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Deterministic numerical settings included in the cache identity."""

    alignment_max_points: int = 6_000
    alignment_iterations: int = 24
    alignment_trim_fraction: float = 0.78
    correspondence_neighbors: int = 4
    correspondence_distance_quantile: float = 0.985
    ridge_regularization: float = 5.0e-3
    coefficient_bound: float = 3.0

    def __post_init__(self) -> None:
        if self.alignment_max_points < 32:
            raise CalibratedRetargetError("alignment_max_points must be at least 32")
        if self.alignment_iterations < 1:
            raise CalibratedRetargetError("alignment_iterations must be positive")
        if not 0.5 <= self.alignment_trim_fraction <= 1.0:
            raise CalibratedRetargetError("alignment_trim_fraction must be in [0.5, 1]")
        if self.correspondence_neighbors < 1:
            raise CalibratedRetargetError("correspondence_neighbors must be positive")
        if not 0.8 <= self.correspondence_distance_quantile <= 1.0:
            raise CalibratedRetargetError(
                "correspondence_distance_quantile must be in [0.8, 1]"
            )
        if not np.isfinite(self.ridge_regularization) or self.ridge_regularization < 0:
            raise CalibratedRetargetError(
                "ridge_regularization must be finite and non-negative"
            )
        if not np.isfinite(self.coefficient_bound) or self.coefficient_bound <= 0:
            raise CalibratedRetargetError("coefficient_bound must be finite and positive")


@dataclass(frozen=True)
class SimilarityTransform:
    """A proper 3D similarity transform using column-vector rotation semantics."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise CalibratedRetargetError("Similarity scale must be finite and positive")
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise CalibratedRetargetError("Similarity rotation/translation have invalid shapes")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise CalibratedRetargetError("Similarity transform contains non-finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
            raise CalibratedRetargetError("Similarity rotation is not orthonormal")
        if np.linalg.det(rotation) < 0.999:
            raise CalibratedRetargetError("Similarity rotation must preserve handedness")
        rotation = rotation.copy()
        translation = translation.copy()
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        values = _points(points, "points")
        return self.scale * (values @ self.rotation.T) + self.translation

    def apply_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        if values.shape[-1:] != (3,) or not np.isfinite(values).all():
            raise CalibratedRetargetError("Vectors must be finite with final dimension 3")
        return self.scale * (values @ self.rotation.T)


@dataclass(frozen=True)
class SourceRigGeometry:
    """Neutral mesh and named one-unit source blendshape displacements."""

    neutral: np.ndarray
    deltas: np.ndarray
    pose_names: tuple[str, ...]
    alignment_indices: np.ndarray | None = None

    def validated(self, label: str) -> SourceRigGeometry:
        neutral = _points(self.neutral, f"{label} neutral")
        deltas = np.asarray(self.deltas, dtype=np.float64)
        names = tuple(str(name) for name in self.pose_names)
        if deltas.shape != (len(names), len(neutral), 3):
            raise CalibratedRetargetError(
                f"{label} deltas must have shape [{len(names)},{len(neutral)},3], "
                f"got {deltas.shape}"
            )
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise CalibratedRetargetError(f"{label} pose names must be non-empty and unique")
        if not np.isfinite(deltas).all():
            raise CalibratedRetargetError(f"{label} deltas contain non-finite values")
        indices: np.ndarray | None = None
        if self.alignment_indices is not None:
            indices = np.asarray(self.alignment_indices, dtype=np.int64)
            if (
                indices.ndim != 1
                or len(indices) < 16
                or np.any(indices < 0)
                or np.any(indices >= len(neutral))
                or len(np.unique(indices)) != len(indices)
            ):
                raise CalibratedRetargetError(
                    f"{label} alignment_indices must contain at least 16 unique valid indices"
                )
        return SourceRigGeometry(
            neutral=neutral.copy(),
            deltas=deltas.copy(),
            pose_names=names,
            alignment_indices=None if indices is None else indices.copy(),
        )


@dataclass(frozen=True)
class PostSolverControlRanges:
    """Allowed values after Audio2Face's multiplier/offset postprocessor.

    These are deliberately separate from the legacy normalized-weight API.
    Claire v3 has valid tongue outputs above one, so silently applying a
    generic ``[0, 1]`` clamp destroys the published rig's intended motion.
    """

    skin_pose_names: tuple[str, ...]
    skin_minimum: np.ndarray
    skin_maximum: np.ndarray
    tongue_pose_names: tuple[str, ...]
    tongue_minimum: np.ndarray
    tongue_maximum: np.ndarray

    def __post_init__(self) -> None:
        for prefix in ("skin", "tongue"):
            names = tuple(str(name) for name in getattr(self, f"{prefix}_pose_names"))
            minimum = np.asarray(getattr(self, f"{prefix}_minimum"), dtype=np.float32)
            maximum = np.asarray(getattr(self, f"{prefix}_maximum"), dtype=np.float32)
            if len(set(names)) != len(names) or any(not name for name in names):
                raise CalibratedRetargetError(
                    f"Post-solver {prefix} pose names must be non-empty and unique"
                )
            if minimum.shape != (len(names),) or maximum.shape != (len(names),):
                raise CalibratedRetargetError(
                    f"Post-solver {prefix} ranges must match {len(names)} pose names"
                )
            if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
                raise CalibratedRetargetError(
                    f"Post-solver {prefix} ranges must be finite"
                )
            if np.any(minimum > maximum):
                raise CalibratedRetargetError(
                    f"Post-solver {prefix} minimum exceeds maximum"
                )
            minimum = minimum.copy()
            maximum = maximum.copy()
            minimum.setflags(write=False)
            maximum.setflags(write=False)
            object.__setattr__(self, f"{prefix}_pose_names", names)
            object.__setattr__(self, f"{prefix}_minimum", minimum)
            object.__setattr__(self, f"{prefix}_maximum", maximum)


@dataclass(frozen=True)
class ClaireV3BlendshapeGeometry:
    """Validated, revision-pinned Claire geometry from Audio2Face-3D v3."""

    root: Path
    revision: str
    network_version: str
    identity: str
    identity_index: int
    skin: SourceRigGeometry
    tongue: SourceRigGeometry
    control_ranges: PostSolverControlRanges
    source_fingerprint: str

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_revision: str = CLAIRE_V3_HF_REVISION,
    ) -> ClaireV3BlendshapeGeometry:
        """Load Claire v3 assets only when their immutable revision is known.

        ``snapshot_download(..., revision=CLAIRE_V3_HF_REVISION)`` directories
        and exact official Claire-only profile files are recognized directly.
        Synthetic or transformed assets must include
        :data:`CLAIRE_V3_ASSET_MANIFEST_FILENAME` with the model id, schema,
        and exact revision.
        """

        root = Path(directory).expanduser().resolve()
        if expected_revision != CLAIRE_V3_HF_REVISION:
            raise CalibrationCacheMismatch(
                f"Unsupported Claire v3 revision {expected_revision!r}; this loader is "
                f"pinned to {CLAIRE_V3_HF_REVISION}"
            )
        if not root.is_dir():
            raise CalibratedRetargetError(
                f"Claire v3 asset directory does not exist: {root}"
            )
        manifest = root / CLAIRE_V3_ASSET_MANIFEST_FILENAME
        fingerprint_files = list(CLAIRE_V3_PROFILE_SHA256)
        if manifest.is_file():
            provenance = _read_json(manifest, "Claire v3 provenance manifest")
            expected_manifest = {
                "schema_version": CLAIRE_V3_ASSET_MANIFEST_SCHEMA,
                "model_id": CLAIRE_V3_MODEL_ID,
                "revision": expected_revision,
            }
            for key, expected in expected_manifest.items():
                if provenance.get(key) != expected:
                    raise CalibrationCacheMismatch(
                        f"Claire v3 manifest {key!r} must be {expected!r}, "
                        f"got {provenance.get(key)!r}"
                    )
            fingerprint_files.insert(0, CLAIRE_V3_ASSET_MANIFEST_FILENAME)
        elif not (root.name == expected_revision and root.parent.name == "snapshots"):
            for name, expected_digest in CLAIRE_V3_PROFILE_SHA256.items():
                path = root / name
                if not path.is_file():
                    raise CalibrationCacheMismatch(
                        "Claire v3 assets have no verifiable revision; use the exact "
                        f"Hugging Face snapshot {expected_revision}, add "
                        f"{CLAIRE_V3_ASSET_MANIFEST_FILENAME}, or retain the complete "
                        "official Claire profile"
                    )
                try:
                    actual_digest = sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    raise CalibratedRetargetError(
                        f"Could not hash Claire v3 profile asset {path}: {exc}"
                    ) from exc
                if actual_digest != expected_digest:
                    raise CalibrationCacheMismatch(
                        f"Claire v3 profile asset {name} does not match pinned revision "
                        f"{expected_revision}"
                    )

        network = _read_json(root / "network_info.json", "Claire v3 network info")
        _validate_claire_v3_network(network)
        model_path = root / "model.json"
        if model_path.is_file():
            _validate_claire_v3_model(
                _read_json(model_path, "Claire v3 model manifest")
            )
            fingerprint_files.append("model.json")

        skin = _load_claire_v3_blendshapes(
            root / "bs_skin_Claire.npz",
            "skin",
            CLAIRE_V3_SKIN_POSE_NAMES,
            CLAIRE_V3_SKIN_RIG_VERSION,
            require_frontal_mask=True,
        )
        tongue = _load_claire_v3_blendshapes(
            root / "bs_tongue_Claire.npz",
            "tongue",
            CLAIRE_V3_TONGUE_POSE_NAMES,
            CLAIRE_V3_TONGUE_RIG_VERSION,
            require_frontal_mask=False,
        )
        _validate_claire_v3_model_data(root / "model_data_Claire.npz", skin, tongue)

        params = network["params"]
        if int(params["skin_size"]) != skin.neutral.size:
            raise CalibrationCacheMismatch(
                "Claire v3 skin_size does not match bs_skin_Claire geometry"
            )
        if int(params["tongue_size"]) != tongue.neutral.size:
            raise CalibrationCacheMismatch(
                "Claire v3 tongue_size does not match bs_tongue_Claire geometry"
            )
        skin_minimum, skin_maximum = _load_solver_control_ranges(
            root / "bs_skin_config_Claire.json", skin.pose_names, "skin"
        )
        tongue_minimum, tongue_maximum = _load_solver_control_ranges(
            root / "bs_tongue_config_Claire.json", tongue.pose_names, "tongue"
        )
        ranges = PostSolverControlRanges(
            skin_pose_names=skin.pose_names,
            skin_minimum=skin_minimum,
            skin_maximum=skin_maximum,
            tongue_pose_names=tongue.pose_names,
            tongue_minimum=tongue_minimum,
            tongue_maximum=tongue_maximum,
        )
        return cls(
            root=root,
            revision=expected_revision,
            network_version=CLAIRE_V3_NETWORK_VERSION,
            identity="Claire",
            identity_index=0,
            skin=skin,
            tongue=tongue,
            control_ranges=ranges,
            source_fingerprint=_file_set_fingerprint(root, fingerprint_files),
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CalibratedRetargetError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibratedRetargetError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibratedRetargetError(f"{label} must contain a JSON object")
    return value


def _validate_claire_v3_network(network: Mapping[str, Any]) -> None:
    identity = network.get("id")
    params = network.get("params")
    audio = network.get("audio_params")
    if not isinstance(identity, Mapping) or not isinstance(params, Mapping) or not isinstance(
        audio, Mapping
    ):
        raise CalibrationCacheMismatch("Claire v3 network metadata is incomplete")
    expected_identity = {
        "type": "diffusion",
        "actor": "multi",
        "version": CLAIRE_V3_NETWORK_VERSION,
        "output": "geometry",
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise CalibrationCacheMismatch(
                f"Claire v3 network id.{key} must be {expected!r}, got {identity.get(key)!r}"
            )
    expected_params: dict[str, Any] = {
        "identities": ["Claire", "James", "Mark"],
        "jaw_size": 15,
        "eyes_size": 4,
        "num_diffusion_steps": 2,
        "num_gru_layers": 2,
        "gru_latent_dim": 256,
        "num_frames_left_truncate": 15,
        "num_frames_right_truncate": 15,
        "num_frames_center": 30,
    }
    for key, expected in expected_params.items():
        if params.get(key) != expected:
            raise CalibrationCacheMismatch(
                f"Claire v3 network params.{key} must be {expected!r}, got {params.get(key)!r}"
            )
    for key in ("skin_size", "tongue_size"):
        value = params.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CalibrationCacheMismatch(
                f"Claire v3 network params.{key} must be a positive integer"
            )
    for key in ("buffer_len", "padding_left", "padding_right", "samplerate"):
        if audio.get(key) != 16_000:
            raise CalibrationCacheMismatch(
                f"Claire v3 audio_params.{key} must be 16000, got {audio.get(key)!r}"
            )


def _validate_claire_v3_model(model: Mapping[str, Any]) -> None:
    expected_scalar = {
        "networkInfoPath": "network_info.json",
        "networkPath": "network.trt",
    }
    for key, expected in expected_scalar.items():
        if model.get(key) != expected:
            raise CalibrationCacheMismatch(
                f"Claire v3 model {key} must be {expected!r}, got {model.get(key)!r}"
            )
    if model.get("modelConfigPaths") != [
        "model_config_Claire.json",
        "model_config_James.json",
        "model_config_Mark.json",
    ] or model.get("modelDataPaths") != [
        "model_data_Claire.npz",
        "model_data_James.npz",
        "model_data_Mark.npz",
    ]:
        raise CalibrationCacheMismatch("Claire v3 model identity paths do not match v3.0")
    expected_blendshapes = []
    for identity in ("Claire", "James", "Mark"):
        expected_blendshapes.append(
            {
                "skin": {
                    "config": f"bs_skin_config_{identity}.json",
                    "data": f"bs_skin_{identity}.npz",
                },
                "tongue": {
                    "config": f"bs_tongue_config_{identity}.json",
                    "data": f"bs_tongue_{identity}.npz",
                },
            }
        )
    if model.get("blendshapePaths") != expected_blendshapes:
        raise CalibrationCacheMismatch(
            "Claire v3 model blendshape paths do not match the pinned release"
        )


def _npz_text(values: np.ndarray, label: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise CalibrationCacheMismatch(f"{label} must be a one-dimensional string array")
    output: list[str] = []
    for value in array.tolist():
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CalibrationCacheMismatch(f"{label} is not valid UTF-8") from exc
        if not isinstance(value, str) or not value:
            raise CalibrationCacheMismatch(f"{label} contains an invalid name")
        output.append(value)
    return tuple(output)


def _npz_scalar_text(values: np.ndarray, label: str) -> str:
    array = np.asarray(values)
    if array.shape != ():
        raise CalibrationCacheMismatch(f"{label} must be a scalar string")
    value = array.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CalibrationCacheMismatch(f"{label} is not valid UTF-8") from exc
    if not isinstance(value, str):
        raise CalibrationCacheMismatch(f"{label} must be a scalar string")
    return value


def _load_claire_v3_blendshapes(
    path: Path,
    label: str,
    expected_pose_names: tuple[str, ...],
    expected_rig_version: str,
    *,
    require_frontal_mask: bool,
) -> SourceRigGeometry:
    if not path.is_file():
        raise CalibratedRetargetError(f"Claire v3 {label} geometry is missing: {path}")
    expected_keys = {
        "neutral",
        "poseNames",
        "rig_version",
        *expected_pose_names,
    }
    if require_frontal_mask:
        expected_keys.add("frontalMask")
    try:
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != expected_keys:
                missing = sorted(expected_keys - set(values.files))
                extra = sorted(set(values.files) - expected_keys)
                raise CalibrationCacheMismatch(
                    f"Claire v3 {label} geometry keys differ from the pinned release; "
                    f"missing={missing}, extra={extra}"
                )
            pose_names = _npz_text(values["poseNames"], f"Claire v3 {label} poseNames")
            if pose_names != ("neutral", *expected_pose_names):
                raise CalibrationCacheMismatch(
                    f"Claire v3 {label} poseNames differ from the pinned release"
                )
            rig_version = _npz_scalar_text(
                values["rig_version"], f"Claire v3 {label} rig_version"
            )
            if rig_version != expected_rig_version:
                raise CalibrationCacheMismatch(
                    f"Claire v3 {label} rig_version must be {expected_rig_version!r}, "
                    f"got {rig_version!r}"
                )
            neutral = np.asarray(values["neutral"], dtype=np.float64)
            deltas = np.stack(
                [np.asarray(values[name], dtype=np.float64) for name in expected_pose_names]
            )
            frontal_mask = (
                np.asarray(values["frontalMask"], dtype=np.int64)
                if require_frontal_mask
                else None
            )
    except CalibratedRetargetError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CalibratedRetargetError(
            f"Could not load Claire v3 {label} geometry {path}: {exc}"
        ) from exc
    return SourceRigGeometry(
        neutral=neutral,
        deltas=deltas,
        pose_names=expected_pose_names,
        alignment_indices=frontal_mask,
    ).validated(f"Claire v3 {label}")


def _validate_claire_v3_model_data(
    path: Path, skin: SourceRigGeometry, tongue: SourceRigGeometry
) -> None:
    expected_keys = {
        "neutral_jaw",
        "neutral_skin",
        "neutral_tongue",
        "lip_open_pose_delta",
        "eye_close_pose_delta",
        "saccade_rot_matrix",
    }
    if not path.is_file():
        raise CalibratedRetargetError(f"Claire v3 model data is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != expected_keys:
                raise CalibrationCacheMismatch(
                    "Claire v3 model_data_Claire keys differ from the pinned release"
                )
            neutral_skin = np.asarray(values["neutral_skin"], dtype=np.float64)
            neutral_tongue = np.asarray(values["neutral_tongue"], dtype=np.float64)
            neutral_jaw = np.asarray(values["neutral_jaw"], dtype=np.float64)
            lip_open = np.asarray(values["lip_open_pose_delta"], dtype=np.float64)
            eye_close = np.asarray(values["eye_close_pose_delta"], dtype=np.float64)
            saccade = np.asarray(values["saccade_rot_matrix"], dtype=np.float64)
    except CalibratedRetargetError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CalibratedRetargetError(f"Could not load Claire v3 model data {path}: {exc}") from exc
    if neutral_jaw.shape != (5, 3):
        raise CalibrationCacheMismatch("Claire v3 neutral_jaw must have shape [5,3]")
    if lip_open.shape != skin.neutral.shape or eye_close.shape != skin.neutral.shape:
        raise CalibrationCacheMismatch(
            "Claire v3 skin helper deltas do not match the skin neutral geometry"
        )
    if saccade.shape != (5_000, 2):
        raise CalibrationCacheMismatch(
            "Claire v3 saccade_rot_matrix must have shape [5000,2]"
        )
    if not all(
        np.isfinite(value).all()
        for value in (neutral_skin, neutral_tongue, neutral_jaw, lip_open, eye_close, saccade)
    ):
        raise CalibratedRetargetError("Claire v3 model data contains non-finite values")
    if neutral_skin.shape != skin.neutral.shape or neutral_tongue.shape != tongue.neutral.shape:
        raise CalibrationCacheMismatch(
            "Claire v3 model-data neutral geometry does not match the blendshape topology"
        )


def _load_solver_control_ranges(
    path: Path, pose_names: tuple[str, ...], label: str
) -> tuple[np.ndarray, np.ndarray]:
    config = _read_json(path, f"Claire v3 {label} solver config")
    params = config.get("blendshape_params")
    if not isinstance(params, Mapping):
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} solver config lacks blendshape_params"
        )
    if params.get("numPoses") != len(pose_names):
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} solver numPoses must be {len(pose_names)}"
        )
    try:
        active = np.asarray(params["bsSolveActivePoses"], dtype=np.int8)
        multipliers = np.asarray(params["bsWeightMultipliers"], dtype=np.float64)
        offsets = np.asarray(params["bsWeightOffsets"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} solver ranges are incomplete"
        ) from exc
    expected_shape = (len(pose_names),)
    if active.shape != expected_shape or multipliers.shape != expected_shape or offsets.shape != expected_shape:
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} solver arrays must have shape {expected_shape}"
        )
    if np.any((active != 0) & (active != 1)):
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} active pose flags must be zero or one"
        )
    if not np.isfinite(multipliers).all() or np.any(multipliers <= 0):
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} weight multipliers must be finite and positive"
        )
    if not np.isfinite(offsets).all():
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} weight offsets must be finite"
        )
    expected_active = np.asarray(
        CLAIRE_V3_SKIN_ACTIVE
        if label == "skin"
        else np.ones(len(CLAIRE_V3_TONGUE_POSE_NAMES), dtype=np.int8),
        dtype=np.int8,
    )
    expected_multipliers = np.asarray(
        np.ones(len(CLAIRE_V3_SKIN_POSE_NAMES), dtype=np.float64)
        if label == "skin"
        else CLAIRE_V3_TONGUE_MULTIPLIERS,
        dtype=np.float64,
    )
    expected_offsets = np.asarray(
        np.zeros(len(CLAIRE_V3_SKIN_POSE_NAMES), dtype=np.float64)
        if label == "skin"
        else CLAIRE_V3_TONGUE_OFFSETS,
        dtype=np.float64,
    )
    if (
        not np.array_equal(active, expected_active)
        or not np.array_equal(multipliers, expected_multipliers)
        or not np.array_equal(offsets, expected_offsets)
    ):
        raise CalibrationCacheMismatch(
            f"Claire v3 {label} post-solver controls differ from the pinned release"
        )
    maximum = offsets + multipliers * active
    return offsets.astype(np.float32), maximum.astype(np.float32)


def _points(values: np.ndarray, label: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    if output.ndim != 2 or output.shape[1] != 3 or len(output) < 4:
        raise CalibratedRetargetError(f"{label} must have shape [vertices,3]")
    if not np.isfinite(output).all():
        raise CalibratedRetargetError(f"{label} contains non-finite values")
    if np.max(np.ptp(output, axis=0)) <= 1e-12:
        raise CalibratedRetargetError(f"{label} is degenerate")
    return output


def _deterministic_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    # A fixed generator keeps calibration bit-stable while avoiding topology-
    # ordered sampling bias (for example, selecting only the first face part).
    indices = np.random.default_rng(0x474E4D).choice(len(points), maximum, replace=False)
    return points[np.sort(indices)]


def _axis_maps() -> tuple[np.ndarray, ...]:
    maps: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            # PCA eigenvector frames have arbitrary handedness.  Keep both
            # parities here and reject improper *world* rotations only after
            # composing the source and target PCA frames.
            maps.append(candidate)
    return tuple(maps)


def _robust_radius(points: np.ndarray) -> float:
    # The centroid is rotation equivariant.  A coordinate-wise median is not,
    # and would bias PCA initialization whenever the source axes differ.
    centered = points - np.mean(points, axis=0)
    radii = np.linalg.norm(centered, axis=1)
    value = float(np.quantile(radii, 0.8))
    if not np.isfinite(value) or value <= 1e-12:
        raise CalibratedRetargetError("Point cloud has no usable spatial extent")
    return value


def _umeyama(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise CalibratedRetargetError("Similarity pairs must have matching [points,3] shapes")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0:
        signs[-1] = -1.0
    rotation = u_matrix @ np.diag(signs) @ vt_matrix
    variance = float(np.sum(source_centered * source_centered) / len(source))
    if variance <= 1e-18:
        raise CalibratedRetargetError("Similarity source pairs are degenerate")
    scale = float(np.sum(singular_values * signs) / variance)
    if scale <= 1e-12:
        raise CalibratedRetargetError("Similarity solve produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def estimate_similarity_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    config: CalibrationConfig | None = None,
) -> tuple[SimilarityTransform, dict[str, float | int]]:
    """Estimate a robust topology-independent source-to-target alignment.

    Initial hypotheses include identity and all proper PCA-axis assignments.
    Trimmed point-to-surface ICP then refines the best hypothesis.  Searching
    PCA assignments makes the method tolerate unknown source coordinate axes;
    rejecting reflections ensures that left and right are never silently
    swapped by a handedness change.
    """

    settings = config or CalibrationConfig()
    source_points = _deterministic_subsample(_points(source, "alignment source"), settings.alignment_max_points)
    target_points = _deterministic_subsample(_points(target, "alignment target"), settings.alignment_max_points)
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    initial_scale = _robust_radius(target_points) / _robust_radius(source_points)
    source_cov = np.cov(source_points - source_center, rowvar=False)
    target_cov = np.cov(target_points - target_center, rowvar=False)
    _, source_axes = np.linalg.eigh(source_cov)
    _, target_axes = np.linalg.eigh(target_cov)
    source_axes = source_axes[:, ::-1]
    target_axes = target_axes[:, ::-1]
    target_tree = cKDTree(target_points)

    candidates = [np.eye(3, dtype=np.float64)]
    candidates.extend(target_axes @ axis_map @ source_axes.T for axis_map in _axis_maps())
    best_score = np.inf
    best_transform: SimilarityTransform | None = None
    trim_count = max(16, int(np.ceil(settings.alignment_trim_fraction * len(source_points))))
    for rotation in candidates:
        if np.linalg.det(rotation) < 0.999:
            continue
        translation = target_center - initial_scale * (rotation @ source_center)
        transform = SimilarityTransform(initial_scale, rotation, translation)
        distances, _ = target_tree.query(transform.apply_points(source_points), k=1, workers=1)
        score = float(np.mean(np.partition(distances, trim_count - 1)[:trim_count] ** 2))
        if score < best_score:
            best_score = score
            best_transform = transform
    if best_transform is None:
        raise CalibratedRetargetError("Could not initialize neutral point-cloud alignment")

    previous_score = np.inf
    iterations = 0
    for iterations in range(1, settings.alignment_iterations + 1):
        transformed = best_transform.apply_points(source_points)
        distances, nearest = target_tree.query(transformed, k=1, workers=1)
        keep = np.argpartition(distances, trim_count - 1)[:trim_count]
        candidate = _umeyama(source_points[keep], target_points[nearest[keep]])
        candidate_scale = candidate.scale
        # Partial face clouds can make one-way ICP shrink toward a feature.
        # Keep scale inside a broad but safe range around the robust estimate.
        lower_scale, upper_scale = initial_scale * 0.55, initial_scale * 1.8
        if not lower_scale <= candidate_scale <= upper_scale:
            candidate_scale = float(np.clip(candidate_scale, lower_scale, upper_scale))
            matched_center = np.mean(target_points[nearest[keep]], axis=0)
            source_matched_center = np.mean(source_points[keep], axis=0)
            candidate = SimilarityTransform(
                candidate_scale,
                candidate.rotation,
                matched_center - candidate_scale * (candidate.rotation @ source_matched_center),
            )
        transformed_candidate = candidate.apply_points(source_points)
        candidate_distances, _ = target_tree.query(transformed_candidate, k=1, workers=1)
        score = float(
            np.mean(np.partition(candidate_distances, trim_count - 1)[:trim_count] ** 2)
        )
        best_transform = candidate
        if abs(previous_score - score) <= max(score, 1e-18) * 1e-8:
            previous_score = score
            break
        previous_score = score

    normalized_rms = float(np.sqrt(previous_score) / _robust_radius(target_points))
    if not np.isfinite(normalized_rms) or normalized_rms > 0.35:
        raise CalibratedRetargetError(
            "Neutral alignment is too poor for calibration "
            f"(normalized trimmed RMS {normalized_rms:.4f})"
        )
    return best_transform, {
        "source_points": len(source_points),
        "target_points": len(target_points),
        "iterations": iterations,
        "normalized_trimmed_rms": normalized_rms,
    }


def _refine_translation(
    source: np.ndarray,
    target: np.ndarray,
    transform: SimilarityTransform,
    iterations: int = 8,
) -> SimilarityTransform:
    """Refine only neutral correspondence translation, preserving vector scale/axes."""

    source_points = _deterministic_subsample(_points(source, "translation source"), 4_000)
    target_points = _deterministic_subsample(_points(target, "translation target"), 4_000)
    translation = transform.translation + (
        np.median(target_points, axis=0) - np.median(transform.apply_points(source_points), axis=0)
    )
    tree = cKDTree(target_points)
    for _ in range(iterations):
        aligned = transform.scale * (source_points @ transform.rotation.T) + translation
        distances, nearest = tree.query(aligned, k=1, workers=1)
        keep_count = max(16, int(0.8 * len(source_points)))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        correction = np.median(target_points[nearest[keep]] - aligned[keep], axis=0)
        translation = translation + correction
        if np.linalg.norm(correction) < 1e-10:
            break
    return SimilarityTransform(transform.scale, transform.rotation, translation)


def _region_vertices(
    basis: np.ndarray,
    spec: RegionSpec,
    explicit_mask: np.ndarray | None,
) -> np.ndarray:
    components = basis[spec.start : spec.stop]
    energy = np.linalg.norm(components, axis=(0, 2))
    threshold = max(float(np.max(energy, initial=0.0)) * 1e-9, 1e-14)
    active = energy > threshold
    if explicit_mask is not None:
        mask = np.asarray(explicit_mask)
        if mask.shape != (basis.shape[1],):
            raise CalibratedRetargetError(
                f"Vertex mask for region {spec.name!r} must have shape [{basis.shape[1]}]"
            )
        active &= mask.astype(bool)
    indices = np.flatnonzero(active)
    if len(indices) < 8:
        raise CalibratedRetargetError(
            f"Region {spec.name!r} has too few active target vertices ({len(indices)})"
        )
    return indices


def _transferred_region_deltas(
    source: SourceRigGeometry,
    target_neutral: np.ndarray,
    target_indices: np.ndarray,
    point_transform: SimilarityTransform,
    vector_transform: SimilarityTransform,
    config: CalibrationConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    source_points = point_transform.apply_points(source.neutral)
    target_points = target_neutral[target_indices]
    neighbors = min(config.correspondence_neighbors, len(source_points))
    distances, nearest = cKDTree(source_points).query(
        target_points, k=neighbors, workers=1
    )
    if neighbors == 1:
        distances = distances[:, None]
        nearest = nearest[:, None]
    nearest_distance = distances[:, 0]
    cutoff = float(np.quantile(nearest_distance, config.correspondence_distance_quantile))
    robust_scale = max(float(np.median(nearest_distance)) * 3.0, _robust_radius(target_points) * 1e-4)
    confidence = 1.0 / (1.0 + (nearest_distance / robust_scale) ** 2)
    confidence[nearest_distance > max(cutoff, robust_scale)] *= 0.05
    if np.count_nonzero(confidence > 1e-4) < 8:
        raise CalibratedRetargetError("Too few reliable neutral surface correspondences")
    inverse = 1.0 / np.maximum(distances, robust_scale * 1e-4) ** 2
    interpolation = inverse / np.sum(inverse, axis=1, keepdims=True)
    transformed_deltas = vector_transform.apply_vectors(source.deltas)
    gathered = transformed_deltas[:, nearest, :]
    transferred = np.einsum("vk,cvkj->cvj", interpolation, gathered, optimize=True)
    return transferred, confidence, {
        "vertices": len(target_indices),
        "median_distance": float(np.median(nearest_distance)),
        "p95_distance": float(np.quantile(nearest_distance, 0.95)),
        "effective_vertices": int(np.count_nonzero(confidence > 0.1)),
    }


def _bounded_region_solve(
    design: np.ndarray,
    targets: np.ndarray,
    vertex_confidence: np.ndarray,
    *,
    bound: float,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve all channels through a compact normal-equation BVLS system."""

    coefficient_count = design.shape[1]
    weights = np.repeat(np.sqrt(vertex_confidence), 3)
    weighted_design = design * weights[:, None]
    weighted_targets = targets.reshape(len(targets), -1).T * weights[:, None]
    normal = weighted_design.T @ weighted_design
    diagonal_scale = max(float(np.trace(normal)) / max(coefficient_count, 1), 1e-16)
    normal += regularization * diagonal_scale * np.eye(coefficient_count)
    normal = 0.5 * (normal + normal.T)
    try:
        cholesky = np.linalg.cholesky(normal)
    except np.linalg.LinAlgError:
        jitter = max(float(np.linalg.norm(normal, ord=2)) * 1e-10, 1e-14)
        try:
            cholesky = np.linalg.cholesky(normal + jitter * np.eye(coefficient_count))
        except np.linalg.LinAlgError as exc:
            raise CalibratedRetargetError(
                "GNM calibration normal matrix is not positive definite"
            ) from exc
    least_squares_matrix = cholesky.T
    right_hand_sides = weighted_design.T @ weighted_targets
    compact_targets = np.linalg.solve(cholesky, right_hand_sides)
    coefficients = np.empty((targets.shape[0], coefficient_count), dtype=np.float64)
    for channel in range(targets.shape[0]):
        solved = lsq_linear(
            least_squares_matrix,
            compact_targets[:, channel],
            bounds=(-bound, bound),
            method="bvls",
            tol=1e-11,
            lsmr_tol=None,
            max_iter=coefficient_count * 4,
        )
        if not solved.success or not np.isfinite(solved.x).all():
            raise CalibratedRetargetError(
                f"Bounded GNM calibration solve failed: {solved.message}"
            )
        coefficients[channel] = solved.x
    predictions = design @ coefficients.T
    residual = (predictions - targets.reshape(len(targets), -1).T) * weights[:, None]
    baseline = targets.reshape(len(targets), -1).T * weights[:, None]
    residual_energy = np.sum(residual * residual, axis=0)
    baseline_energy = np.sum(baseline * baseline, axis=0)
    return coefficients, residual_energy, baseline_energy


def _fit_source_mapping(
    source: SourceRigGeometry,
    target_neutral: np.ndarray,
    target_basis: np.ndarray,
    regions: Sequence[RegionSpec],
    region_masks: Mapping[str, np.ndarray],
    *,
    point_transform: SimilarityTransform,
    vector_transform: SimilarityTransform,
    config: CalibrationConfig,
) -> tuple[np.ndarray, dict[str, dict[str, float | int]], dict[str, dict[str, float | int]]]:
    mapping = np.zeros((len(source.pose_names), len(target_basis)), dtype=np.float64)
    residual_energy = np.zeros(len(source.pose_names), dtype=np.float64)
    baseline_energy = np.zeros(len(source.pose_names), dtype=np.float64)
    correspondence_diagnostics: dict[str, dict[str, float | int]] = {}
    for spec in regions:
        vertices = _region_vertices(target_basis, spec, region_masks.get(spec.name))
        transferred, confidence, correspondence = _transferred_region_deltas(
            source,
            target_neutral,
            vertices,
            point_transform,
            vector_transform,
            config,
        )
        components = target_basis[spec.start : spec.stop, vertices, :]
        design = components.transpose(1, 2, 0).reshape(-1, spec.stop - spec.start)
        solved, region_residual, region_baseline = _bounded_region_solve(
            design,
            transferred,
            confidence,
            bound=spec.bound,
            regularization=config.ridge_regularization,
        )
        mapping[:, spec.start : spec.stop] = solved
        residual_energy += region_residual
        baseline_energy += region_baseline
        correspondence_diagnostics[spec.name] = correspondence

    channel_diagnostics: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(source.pose_names):
        baseline = float(np.sqrt(max(baseline_energy[index], 0.0)))
        residual = float(np.sqrt(max(residual_energy[index], 0.0)))
        channel_diagnostics[name] = {
            "coefficient_l2": float(np.linalg.norm(mapping[index])),
            "nonzero_coefficients": int(np.count_nonzero(np.abs(mapping[index]) > 1e-7)),
            "fitted_weighted_error": residual,
            "zero_mapping_weighted_error": baseline,
            "relative_fit_error": residual / baseline if baseline > 1e-14 else 0.0,
        }
    if not np.isfinite(mapping).all():
        raise CalibratedRetargetError("Dense calibration produced non-finite coefficients")
    return mapping.astype(np.float32), channel_diagnostics, correspondence_diagnostics


def _validate_regions(regions: Sequence[RegionSpec], expression_dim: int) -> tuple[RegionSpec, ...]:
    output = tuple(regions)
    if not output:
        raise CalibratedRetargetError("At least one GNM coefficient region is required")
    names = [region.name for region in output]
    if len(set(names)) != len(names):
        raise CalibratedRetargetError("GNM calibration region names must be unique")
    occupied = np.zeros(expression_dim, dtype=bool)
    for region in output:
        if region.stop > expression_dim:
            raise CalibratedRetargetError(
                f"Region {region.name!r} exceeds expression dimension {expression_dim}"
            )
        if np.any(occupied[region.start : region.stop]):
            raise CalibratedRetargetError("GNM calibration coefficient regions overlap")
        occupied[region.start : region.stop] = True
    return output


def _hash_array(hasher: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    hasher.update(name.encode("utf-8") + b"\0")
    hasher.update(array.dtype.str.encode("ascii") + b"\0")
    hasher.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
    if array.size:
        hasher.update(memoryview(array).cast("B"))


def _geometry_fingerprint(
    sources: Sequence[tuple[str, SourceRigGeometry]],
    target_neutral: np.ndarray,
    target_basis: np.ndarray,
) -> tuple[str, str]:
    source_hasher = sha256()
    for label, source in sources:
        _hash_array(source_hasher, f"{label}.neutral", source.neutral)
        _hash_array(source_hasher, f"{label}.deltas", source.deltas)
        source_hasher.update("\0".join(source.pose_names).encode("utf-8"))
    target_hasher = sha256()
    _hash_array(target_hasher, "target.neutral", target_neutral)
    _hash_array(target_hasher, "target.basis", target_basis)
    return source_hasher.hexdigest(), target_hasher.hexdigest()


def _request_hash(
    source_fingerprint: str,
    target_fingerprint: str,
    config: CalibrationConfig,
    regions: Sequence[RegionSpec],
    skin_region_names: Sequence[str],
    tongue_region_names: Sequence[str],
) -> str:
    payload = {
        "algorithm": CALIBRATION_ALGORITHM,
        "format_version": CALIBRATION_FORMAT_VERSION,
        "source_fingerprint": source_fingerprint,
        "target_fingerprint": target_fingerprint,
        "config": asdict(config),
        "regions": [asdict(region) for region in regions],
        "skin_regions": list(skin_region_names),
        "tongue_regions": list(tongue_region_names),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _calibration_hash(
    skin_matrix: np.ndarray,
    tongue_matrix: np.ndarray,
    skin_transform: SimilarityTransform,
    tongue_transform: SimilarityTransform,
    metadata: Mapping[str, Any],
) -> str:
    hasher = sha256()
    _hash_array(hasher, "skin_matrix", skin_matrix)
    _hash_array(hasher, "tongue_matrix", tongue_matrix)
    _hash_array(hasher, "skin_rotation", skin_transform.rotation)
    _hash_array(hasher, "skin_translation", skin_transform.translation)
    _hash_array(hasher, "tongue_rotation", tongue_transform.rotation)
    _hash_array(hasher, "tongue_translation", tongue_transform.translation)
    hasher.update(np.float64(skin_transform.scale).tobytes())
    hasher.update(np.float64(tongue_transform.scale).tobytes())
    clean_metadata = dict(metadata)
    clean_metadata.pop("calibration_hash", None)
    hasher.update(
        json.dumps(clean_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return hasher.hexdigest()


@dataclass(frozen=True)
class DenseRetargetCalibration:
    """Serializable dense source-control to GNM-coefficient calibration."""

    skin_pose_names: tuple[str, ...]
    tongue_pose_names: tuple[str, ...]
    skin_matrix: np.ndarray
    tongue_matrix: np.ndarray
    regions: tuple[RegionSpec, ...]
    skin_transform: SimilarityTransform
    tongue_transform: SimilarityTransform
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        skin_names = tuple(self.skin_pose_names)
        tongue_names = tuple(self.tongue_pose_names)
        skin_matrix = np.asarray(self.skin_matrix, dtype=np.float32)
        tongue_matrix = np.asarray(self.tongue_matrix, dtype=np.float32)
        if not skin_names or len(set(skin_names)) != len(skin_names):
            raise CalibratedRetargetError("Skin pose names must be non-empty and unique")
        if len(set(tongue_names)) != len(tongue_names):
            raise CalibratedRetargetError("Tongue pose names must be unique")
        if skin_matrix.ndim != 2 or skin_matrix.shape[0] != len(skin_names):
            raise CalibratedRetargetError("Skin calibration matrix has an invalid shape")
        if tongue_matrix.shape != (len(tongue_names), skin_matrix.shape[1]):
            raise CalibratedRetargetError("Tongue calibration matrix has an invalid shape")
        if not np.isfinite(skin_matrix).all() or not np.isfinite(tongue_matrix).all():
            raise CalibratedRetargetError("Calibration matrices contain non-finite values")
        regions = _validate_regions(self.regions, skin_matrix.shape[1])
        metadata = dict(self.metadata)
        required_metadata = {
            "algorithm",
            "format_version",
            "request_hash",
            "source_fingerprint",
            "target_fingerprint",
            "calibration_hash",
        }
        missing = sorted(required_metadata - metadata.keys())
        if missing:
            raise CalibratedRetargetError(
                f"Calibration metadata is missing: {', '.join(missing)}"
            )
        if metadata["algorithm"] != CALIBRATION_ALGORITHM:
            raise CalibrationCacheMismatch(
                f"Unsupported calibration algorithm {metadata['algorithm']!r}"
            )
        if int(metadata["format_version"]) != CALIBRATION_FORMAT_VERSION:
            raise CalibrationCacheMismatch(
                f"Unsupported calibration format {metadata['format_version']!r}"
            )
        expected_hash = _calibration_hash(
            skin_matrix,
            tongue_matrix,
            self.skin_transform,
            self.tongue_transform,
            metadata,
        )
        if metadata["calibration_hash"] != expected_hash:
            raise CalibratedRetargetError("Calibration content hash does not match its payload")
        skin_matrix = skin_matrix.copy()
        tongue_matrix = tongue_matrix.copy()
        skin_matrix.setflags(write=False)
        tongue_matrix.setflags(write=False)
        object.__setattr__(self, "skin_pose_names", skin_names)
        object.__setattr__(self, "tongue_pose_names", tongue_names)
        object.__setattr__(self, "skin_matrix", skin_matrix)
        object.__setattr__(self, "tongue_matrix", tongue_matrix)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "metadata", metadata)

    @property
    def expression_dim(self) -> int:
        return self.skin_matrix.shape[1]

    @property
    def calibration_hash(self) -> str:
        return str(self.metadata["calibration_hash"])

    def save(self, path: str | Path) -> Path:
        """Atomically write a deterministic, pickle-free NPZ cache."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "skin_pose_names": np.asarray(self.skin_pose_names, dtype=np.str_),
            "tongue_pose_names": np.asarray(self.tongue_pose_names, dtype=np.str_),
            "skin_matrix": self.skin_matrix,
            "tongue_matrix": self.tongue_matrix,
            "region_names": np.asarray([region.name for region in self.regions], dtype=np.str_),
            "region_ranges": np.asarray(
                [[region.start, region.stop] for region in self.regions], dtype=np.int32
            ),
            "region_bounds": np.asarray([region.bound for region in self.regions], dtype=np.float64),
            "skin_scale": np.asarray(self.skin_transform.scale, dtype=np.float64),
            "skin_rotation": self.skin_transform.rotation,
            "skin_translation": self.skin_transform.translation,
            "tongue_scale": np.asarray(self.tongue_transform.scale, dtype=np.float64),
            "tongue_rotation": self.tongue_transform.rotation,
            "tongue_translation": self.tongue_transform.translation,
            "metadata_json": np.asarray(
                json.dumps(self.metadata, sort_keys=True, separators=(",", ":")), dtype=np.str_
            ),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
            with zipfile.ZipFile(
                temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                for name in sorted(arrays):
                    buffer = BytesIO()
                    np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
            os.replace(temporary, destination)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise CalibratedRetargetError(
                f"Could not save calibration cache {destination}: {exc}"
            ) from exc
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_request_hash: str | None = None,
    ) -> DenseRetargetCalibration:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise CalibratedRetargetError(f"Calibration cache does not exist: {source}")
        required = {
            "skin_pose_names",
            "tongue_pose_names",
            "skin_matrix",
            "tongue_matrix",
            "region_names",
            "region_ranges",
            "region_bounds",
            "skin_scale",
            "skin_rotation",
            "skin_translation",
            "tongue_scale",
            "tongue_rotation",
            "tongue_translation",
            "metadata_json",
        }
        try:
            with np.load(source, allow_pickle=False) as values:
                missing = sorted(required - set(values.files))
                if missing:
                    raise CalibratedRetargetError(
                        f"Calibration cache is missing: {', '.join(missing)}"
                    )
                skin_names = tuple(str(value) for value in values["skin_pose_names"].tolist())
                tongue_names = tuple(str(value) for value in values["tongue_pose_names"].tolist())
                region_names = tuple(str(value) for value in values["region_names"].tolist())
                region_ranges = np.asarray(values["region_ranges"], dtype=np.int64)
                region_bounds = np.asarray(values["region_bounds"], dtype=np.float64)
                if region_ranges.shape != (len(region_names), 2) or region_bounds.shape != (
                    len(region_names),
                ):
                    raise CalibratedRetargetError("Calibration region arrays have invalid shapes")
                regions = tuple(
                    RegionSpec(name, int(bounds[0]), int(bounds[1]), float(bound))
                    for name, bounds, bound in zip(
                        region_names, region_ranges, region_bounds, strict=True
                    )
                )
                metadata = json.loads(str(values["metadata_json"].item()))
                calibration = cls(
                    skin_pose_names=skin_names,
                    tongue_pose_names=tongue_names,
                    skin_matrix=np.asarray(values["skin_matrix"], dtype=np.float32).copy(),
                    tongue_matrix=np.asarray(values["tongue_matrix"], dtype=np.float32).copy(),
                    regions=regions,
                    skin_transform=SimilarityTransform(
                        float(values["skin_scale"].item()),
                        np.asarray(values["skin_rotation"], dtype=np.float64).copy(),
                        np.asarray(values["skin_translation"], dtype=np.float64).copy(),
                    ),
                    tongue_transform=SimilarityTransform(
                        float(values["tongue_scale"].item()),
                        np.asarray(values["tongue_rotation"], dtype=np.float64).copy(),
                        np.asarray(values["tongue_translation"], dtype=np.float64).copy(),
                    ),
                    metadata=metadata,
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, CalibratedRetargetError):
                raise
            raise CalibratedRetargetError(
                f"Could not load calibration cache {source}: {exc}"
            ) from exc
        if expected_request_hash is not None and (
            calibration.metadata["request_hash"] != expected_request_hash
        ):
            raise CalibrationCacheMismatch(
                "Calibration cache was built for different assets or numerical settings"
            )
        return calibration


def build_dense_calibration(
    skin_source: SourceRigGeometry,
    target_neutral: np.ndarray,
    target_basis: np.ndarray,
    *,
    tongue_source: SourceRigGeometry | None = None,
    regions: Sequence[RegionSpec] = GNM_REGION_SPECS,
    skin_region_names: Sequence[str] = ("left_eye", "right_eye", "lower_face", "pupils"),
    tongue_region_names: Sequence[str] = ("tongue",),
    target_alignment_mask: np.ndarray | None = None,
    target_region_masks: Mapping[str, np.ndarray] | None = None,
    config: CalibrationConfig | None = None,
    source_fingerprint: str | None = None,
    target_fingerprint: str | None = None,
) -> DenseRetargetCalibration:
    """Build a dense calibration from arbitrary source and target geometry."""

    settings = config or CalibrationConfig()
    skin = skin_source.validated("skin")
    tongue = tongue_source.validated("tongue") if tongue_source is not None else None
    target = _points(target_neutral, "target neutral")
    basis = np.asarray(target_basis, dtype=np.float64)
    if basis.ndim != 3 or basis.shape[1:] != target.shape:
        raise CalibratedRetargetError(
            f"Target basis must have shape [expressions,{len(target)},3], got {basis.shape}"
        )
    if not np.isfinite(basis).all() or len(basis) < 1:
        raise CalibratedRetargetError("Target expression basis must be non-empty and finite")
    region_specs = _validate_regions(
        tuple(
            RegionSpec(
                region.name,
                region.start,
                region.stop,
                min(region.bound, settings.coefficient_bound),
            )
            for region in regions
        ),
        len(basis),
    )
    by_name = {region.name: region for region in region_specs}
    try:
        skin_regions = tuple(by_name[name] for name in skin_region_names)
        tongue_regions = tuple(by_name[name] for name in tongue_region_names)
    except KeyError as exc:
        raise CalibratedRetargetError(f"Unknown calibration region: {exc.args[0]}") from exc
    if set(skin_region_names) & set(tongue_region_names):
        raise CalibratedRetargetError("Skin and tongue calibration regions must not overlap")
    masks = dict(target_region_masks or {})

    source_alignment = (
        skin.neutral[skin.alignment_indices]
        if skin.alignment_indices is not None
        else skin.neutral
    )
    if target_alignment_mask is None:
        target_alignment = target
    else:
        alignment_mask = np.asarray(target_alignment_mask)
        if alignment_mask.shape != (len(target),):
            raise CalibratedRetargetError(
                f"target_alignment_mask must have shape [{len(target)}]"
            )
        target_alignment = target[alignment_mask.astype(bool)]
        if len(target_alignment) < 16:
            raise CalibratedRetargetError("target_alignment_mask selects too few vertices")
    skin_transform, alignment_diagnostics = estimate_similarity_alignment(
        source_alignment, target_alignment, config=settings
    )

    if tongue is None:
        tongue_transform = skin_transform
    else:
        tongue_mask = masks.get("tongue")
        if tongue_mask is None:
            tongue_vertices = np.flatnonzero(
                np.linalg.norm(
                    basis[min(region.start for region in tongue_regions) : max(region.stop for region in tongue_regions)],
                    axis=(0, 2),
                )
                > 1e-14
            )
        else:
            tongue_vertices = np.flatnonzero(np.asarray(tongue_mask).astype(bool))
        if len(tongue_vertices) < 8:
            raise CalibratedRetargetError("Too few target tongue vertices for correspondence")
        tongue_transform = _refine_translation(
            tongue.neutral, target[tongue_vertices], skin_transform
        )

    skin_mapping, skin_channels, skin_correspondence = _fit_source_mapping(
        skin,
        target,
        basis,
        skin_regions,
        masks,
        point_transform=skin_transform,
        vector_transform=skin_transform,
        config=settings,
    )
    if tongue is None:
        tongue_mapping = np.zeros((0, len(basis)), dtype=np.float32)
        tongue_channels: dict[str, dict[str, float | int]] = {}
        tongue_correspondence: dict[str, dict[str, float | int]] = {}
    else:
        tongue_mapping, tongue_channels, tongue_correspondence = _fit_source_mapping(
            tongue,
            target,
            basis,
            tongue_regions,
            masks,
            point_transform=tongue_transform,
            vector_transform=skin_transform,
            config=settings,
        )

    if source_fingerprint is None or target_fingerprint is None:
        computed_source, computed_target = _geometry_fingerprint(
            [("skin", skin)] + ([] if tongue is None else [("tongue", tongue)]),
            target,
            basis,
        )
        source_fingerprint = source_fingerprint or computed_source
        target_fingerprint = target_fingerprint or computed_target
    request_hash = _request_hash(
        source_fingerprint,
        target_fingerprint,
        settings,
        region_specs,
        skin_region_names,
        tongue_region_names if tongue is not None else (),
    )
    metadata: dict[str, Any] = {
        "algorithm": CALIBRATION_ALGORITHM,
        "format_version": CALIBRATION_FORMAT_VERSION,
        "request_hash": request_hash,
        "source_fingerprint": source_fingerprint,
        "target_fingerprint": target_fingerprint,
        "config": asdict(settings),
        "alignment": alignment_diagnostics,
        "skin_correspondence": skin_correspondence,
        "tongue_correspondence": tongue_correspondence,
        "skin_channel_fit": skin_channels,
        "tongue_channel_fit": tongue_channels,
    }
    metadata["calibration_hash"] = _calibration_hash(
        skin_mapping,
        tongue_mapping,
        skin_transform,
        tongue_transform,
        metadata,
    )
    return DenseRetargetCalibration(
        skin_pose_names=skin.pose_names,
        tongue_pose_names=() if tongue is None else tongue.pose_names,
        skin_matrix=skin_mapping,
        tongue_matrix=tongue_mapping,
        regions=region_specs,
        skin_transform=skin_transform,
        tongue_transform=tongue_transform,
        metadata=metadata,
    )


def _file_set_fingerprint(root: Path, names: Sequence[str]) -> str:
    hasher = sha256()
    for name in names:
        path = root / name
        if not path.is_file():
            raise CalibratedRetargetError(f"Calibration asset is missing: {path}")
        hasher.update(name.encode("utf-8") + b"\0")
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    hasher.update(chunk)
        except OSError as exc:
            raise CalibratedRetargetError(f"Could not hash calibration asset {path}: {exc}") from exc
    return hasher.hexdigest()


def _gnm_fingerprint(adapter: GNMAdapter) -> str:
    hasher = sha256()
    hasher.update(str(adapter.model.version).encode("utf-8") + b"\0")
    hasher.update(str(adapter.model.variant).encode("utf-8") + b"\0")
    _hash_array(hasher, "template", np.asarray(adapter.model.template_vertex_positions))
    _hash_array(hasher, "expression_basis", np.asarray(adapter.model.expression_basis))
    hasher.update("\0".join(str(name) for name in adapter.model.expression_names).encode("utf-8"))
    return hasher.hexdigest()


def _gnm_region_masks(adapter: GNMAdapter) -> dict[str, np.ndarray]:
    return {
        "left_eye": adapter.vertex_group("expression_basis_left_eye") > 0,
        "right_eye": adapter.vertex_group("expression_basis_right_eye") > 0,
        "lower_face": adapter.vertex_group("expression_basis_mouth_nose_ears") > 0,
        "tongue": adapter.vertex_group("tongue") > 0,
        "pupils": adapter.vertex_group("eye_interiors") > 0,
    }


class CalibratedRetargeter:
    """Fast runtime facade over a :class:`DenseRetargetCalibration`."""

    def __init__(
        self,
        calibration: DenseRetargetCalibration,
        *,
        post_solver_ranges: PostSolverControlRanges | None = None,
    ):
        self.calibration = calibration
        if post_solver_ranges is not None and (
            post_solver_ranges.skin_pose_names != calibration.skin_pose_names
            or post_solver_ranges.tongue_pose_names != calibration.tongue_pose_names
        ):
            raise CalibratedRetargetError(
                "Post-solver control ranges do not match the calibration pose schema"
            )
        self.post_solver_ranges = post_solver_ranges
        self._skin_lookup = {
            name: index for index, name in enumerate(calibration.skin_pose_names)
        }
        self._tongue_lookup = {
            name: index for index, name in enumerate(calibration.tongue_pose_names)
        }

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        adapter: GNMAdapter | None = None,
        cache_path: str | Path | None = None,
        force_rebuild: bool = False,
        config: CalibrationConfig | None = None,
    ) -> CalibratedRetargeter:
        """Load or build a calibration for released Claire runtime assets."""

        root = Path(directory).expanduser().resolve()
        settings = config or CalibrationConfig()
        gnm = adapter or GNMAdapter()
        if gnm.expression_dim != EXPECTED_GNM_EXPRESSION_DIM:
            raise CalibratedRetargetError(
                f"Expected GNM expression dimension {EXPECTED_GNM_EXPRESSION_DIM}, "
                f"got {gnm.expression_dim}"
            )
        source_fingerprint = _file_set_fingerprint(
            root, ("bs_skin.npz", "bs_tongue.npz")
        )
        target_fingerprint = _gnm_fingerprint(gnm)
        effective_regions = tuple(
            RegionSpec(
                region.name,
                region.start,
                region.stop,
                min(region.bound, settings.coefficient_bound),
            )
            for region in GNM_REGION_SPECS
        )
        request_hash = _request_hash(
            source_fingerprint,
            target_fingerprint,
            settings,
            effective_regions,
            ("left_eye", "right_eye", "lower_face", "pupils"),
            ("tongue",),
        )
        cache = (
            Path(cache_path).expanduser().resolve()
            if cache_path is not None
            else root / f"calibrated_retarget_{CALIBRATION_FORMAT_VERSION}.npz"
        )
        if cache.is_file() and not force_rebuild:
            try:
                return cls(
                    DenseRetargetCalibration.load(
                        cache, expected_request_hash=request_hash
                    )
                )
            except CalibrationCacheMismatch:
                pass

        skin_assets = ClaireSkinAssets.load(root)
        tongue_assets = ClaireTongueAssets.load(root)
        skin = SourceRigGeometry(
            neutral=skin_assets.neutral,
            deltas=skin_assets.pose_deltas,
            pose_names=skin_assets.pose_names,
            alignment_indices=skin_assets.mask,
        )
        tongue = SourceRigGeometry(
            neutral=tongue_assets.neutral,
            deltas=tongue_assets.pose_deltas,
            pose_names=tongue_assets.pose_names,
        )
        masks = _gnm_region_masks(gnm)
        calibration = build_dense_calibration(
            skin,
            np.asarray(gnm.model.template_vertex_positions),
            np.asarray(gnm.model.expression_basis),
            tongue_source=tongue,
            target_alignment_mask=gnm.vertex_group("hockey_mask") > 0,
            target_region_masks=masks,
            config=settings,
            source_fingerprint=source_fingerprint,
            target_fingerprint=target_fingerprint,
        )
        calibration.save(cache)
        return cls(calibration)

    @classmethod
    def from_v3_directory(
        cls,
        directory: str | Path,
        *,
        adapter: GNMAdapter | None = None,
        cache_path: str | Path | None = None,
        force_rebuild: bool = False,
        config: CalibrationConfig | None = None,
        expected_revision: str = CLAIRE_V3_HF_REVISION,
    ) -> CalibratedRetargeter:
        """Load or build a calibration for pinned Claire Audio2Face v3 assets.

        This constructor has a separate cache identity and preserves the v3
        solver's published postprocessing ranges.  It must not be pointed at
        the topologically incompatible Claire v2.3 asset bundle.
        """

        assets = ClaireV3BlendshapeGeometry.load(
            directory, expected_revision=expected_revision
        )
        root = assets.root
        settings = config or CalibrationConfig()
        gnm = adapter or GNMAdapter()
        if gnm.expression_dim != EXPECTED_GNM_EXPRESSION_DIM:
            raise CalibratedRetargetError(
                f"Expected GNM expression dimension {EXPECTED_GNM_EXPRESSION_DIM}, "
                f"got {gnm.expression_dim}"
            )
        target_fingerprint = _gnm_fingerprint(gnm)
        effective_regions = tuple(
            RegionSpec(
                region.name,
                region.start,
                region.stop,
                min(region.bound, settings.coefficient_bound),
            )
            for region in GNM_REGION_SPECS
        )
        request_hash = _request_hash(
            assets.source_fingerprint,
            target_fingerprint,
            settings,
            effective_regions,
            ("left_eye", "right_eye", "lower_face", "pupils"),
            ("tongue",),
        )
        cache = (
            Path(cache_path).expanduser().resolve()
            if cache_path is not None
            else root
            / f"calibrated_retarget_a2f_v3_claire_{CALIBRATION_FORMAT_VERSION}.npz"
        )
        if cache.is_file() and not force_rebuild:
            try:
                calibration = DenseRetargetCalibration.load(
                    cache, expected_request_hash=request_hash
                )
                return cls(
                    calibration, post_solver_ranges=assets.control_ranges
                )
            except CalibrationCacheMismatch:
                pass

        calibration = build_dense_calibration(
            assets.skin,
            np.asarray(gnm.model.template_vertex_positions),
            np.asarray(gnm.model.expression_basis),
            tongue_source=assets.tongue,
            target_alignment_mask=gnm.vertex_group("hockey_mask") > 0,
            target_region_masks=_gnm_region_masks(gnm),
            config=settings,
            source_fingerprint=assets.source_fingerprint,
            target_fingerprint=target_fingerprint,
        )
        calibration.save(cache)
        return cls(calibration, post_solver_ranges=assets.control_ranges)

    def _weights(
        self,
        values: Mapping[str, float],
        lookup: Mapping[str, int],
        label: str,
        *,
        strict: bool,
    ) -> np.ndarray:
        output = np.zeros(len(lookup), dtype=np.float32)
        unknown = sorted(set(values) - set(lookup))
        if strict and unknown:
            raise CalibratedRetargetError(
                f"Unknown {label} controls: {', '.join(unknown)}"
            )
        for name, value in values.items():
            if name not in lookup:
                continue
            number = float(value)
            if not np.isfinite(number):
                raise CalibratedRetargetError(f"{label} weight {name!r} is not finite")
            output[lookup[name]] = np.float32(np.clip(number, 0.0, 1.0))
        return output

    def _bound(self, controls: np.ndarray) -> np.ndarray:
        result = np.asarray(controls, dtype=np.float32).copy()
        for region in self.calibration.regions:
            region_values = result[..., region.start : region.stop]
            maximum = np.max(np.abs(region_values), axis=-1, keepdims=True, initial=0.0)
            scale = np.minimum(1.0, region.bound / np.maximum(maximum, 1e-12))
            result[..., region.start : region.stop] = region_values * scale
        if not np.isfinite(result).all():
            raise CalibratedRetargetError("Dense retarget produced non-finite controls")
        return result

    def retarget(
        self,
        weights: Mapping[str, float],
        tongue_weights: Mapping[str, float] | None = None,
        *,
        strict: bool = False,
    ) -> np.ndarray:
        skin = self._weights(weights, self._skin_lookup, "skin", strict=strict)
        tongue = self._weights(
            tongue_weights or {}, self._tongue_lookup, "tongue", strict=strict
        )
        controls = skin @ self.calibration.skin_matrix
        if len(tongue):
            controls = controls + tongue @ self.calibration.tongue_matrix
        return self._bound(controls)

    def retarget_sequence(
        self,
        weights: np.ndarray,
        pose_names: Sequence[str],
        *,
        tongue_weights: np.ndarray | None = None,
        tongue_pose_names: Sequence[str] | None = None,
        strict: bool = False,
    ) -> np.ndarray:
        values = np.asarray(weights, dtype=np.float32)
        names = tuple(pose_names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise CalibratedRetargetError(
                f"Expected skin weights [frames,{len(names)}], got {values.shape}"
            )
        if len(set(names)) != len(names) or not np.isfinite(values).all():
            raise CalibratedRetargetError("Skin pose names must be unique and weights finite")
        unknown = sorted(set(names) - set(self._skin_lookup))
        if strict and unknown:
            raise CalibratedRetargetError(f"Unknown skin controls: {', '.join(unknown)}")
        skin_matrix = np.zeros((len(names), self.calibration.expression_dim), dtype=np.float32)
        for index, name in enumerate(names):
            if name in self._skin_lookup:
                skin_matrix[index] = self.calibration.skin_matrix[self._skin_lookup[name]]
        controls = np.clip(values, 0.0, 1.0) @ skin_matrix

        if tongue_weights is not None:
            if tongue_pose_names is None:
                raise CalibratedRetargetError(
                    "tongue_pose_names are required with tongue_weights"
                )
            tongue_values = np.asarray(tongue_weights, dtype=np.float32)
            tongue_names = tuple(tongue_pose_names)
            if tongue_values.shape != (len(values), len(tongue_names)):
                raise CalibratedRetargetError(
                    f"Expected tongue weights [{len(values)},{len(tongue_names)}], "
                    f"got {tongue_values.shape}"
                )
            if len(set(tongue_names)) != len(tongue_names) or not np.isfinite(
                tongue_values
            ).all():
                raise CalibratedRetargetError(
                    "Tongue pose names must be unique and weights finite"
                )
            unknown_tongue = sorted(set(tongue_names) - set(self._tongue_lookup))
            if strict and unknown_tongue:
                raise CalibratedRetargetError(
                    f"Unknown tongue controls: {', '.join(unknown_tongue)}"
                )
            tongue_matrix = np.zeros(
                (len(tongue_names), self.calibration.expression_dim), dtype=np.float32
            )
            for index, name in enumerate(tongue_names):
                if name in self._tongue_lookup:
                    tongue_matrix[index] = self.calibration.tongue_matrix[
                        self._tongue_lookup[name]
                    ]
            controls += np.clip(tongue_values, 0.0, 1.0) @ tongue_matrix
        elif tongue_pose_names is not None:
            raise CalibratedRetargetError(
                "tongue_weights are required with tongue_pose_names"
            )
        return self._bound(controls)

    def _validated_post_solver_values(
        self,
        weights: np.ndarray,
        pose_names: Sequence[str],
        *,
        expected_names: tuple[str, ...],
        minimum: np.ndarray,
        maximum: np.ndarray,
        label: str,
        frames: int | None,
        tolerance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(weights, dtype=np.float32)
        names = tuple(str(name) for name in pose_names)
        expected_shape = (
            (values.shape[0], len(names))
            if values.ndim == 2
            else (0, len(names))
        )
        if values.ndim != 2 or values.shape != expected_shape:
            raise CalibratedRetargetError(
                f"Expected post-solver {label} weights [frames,{len(names)}], "
                f"got {values.shape}"
            )
        if frames is not None and len(values) != frames:
            raise CalibratedRetargetError(
                f"Expected {frames} post-solver {label} frames, got {len(values)}"
            )
        if len(set(names)) != len(names):
            raise CalibratedRetargetError(
                f"Post-solver {label} pose names must be unique"
            )
        missing = sorted(set(expected_names) - set(names))
        unknown = sorted(set(names) - set(expected_names))
        if missing or unknown:
            raise CalibratedRetargetError(
                f"Post-solver {label} schema mismatch; missing={missing}, unknown={unknown}"
            )
        if not np.isfinite(values).all():
            raise CalibratedRetargetError(
                f"Post-solver {label} weights contain non-finite values"
            )
        order = np.asarray([expected_names.index(name) for name in names], dtype=np.int64)
        ordered_minimum = minimum[order]
        ordered_maximum = maximum[order]
        outside = (values < ordered_minimum - tolerance) | (
            values > ordered_maximum + tolerance
        )
        if np.any(outside):
            frame_index, pose_index = np.argwhere(outside)[0]
            raise CalibratedRetargetError(
                f"Post-solver {label} weight {names[int(pose_index)]!r} at frame "
                f"{int(frame_index)} is {float(values[frame_index, pose_index]):.7g}, "
                f"outside [{float(ordered_minimum[pose_index]):.7g}, "
                f"{float(ordered_maximum[pose_index]):.7g}]"
            )
        return values, order

    def retarget_post_solver_sequence(
        self,
        skin_weights: np.ndarray,
        skin_pose_names: Sequence[str],
        *,
        tongue_weights: np.ndarray,
        tongue_pose_names: Sequence[str],
        tolerance: float = 1.0e-5,
    ) -> np.ndarray:
        """Retarget already-postprocessed v3 solver output without clipping it.

        The complete pinned schema is required so a missing offset-bearing or
        inactive channel cannot silently change the solver contract.
        """

        ranges = self.post_solver_ranges
        if ranges is None:
            raise CalibratedRetargetError(
                "This calibration has no post-solver control-range contract"
            )
        if not np.isfinite(tolerance) or tolerance < 0:
            raise CalibratedRetargetError(
                "Post-solver validation tolerance must be finite and non-negative"
            )
        skin, skin_order = self._validated_post_solver_values(
            skin_weights,
            skin_pose_names,
            expected_names=ranges.skin_pose_names,
            minimum=ranges.skin_minimum,
            maximum=ranges.skin_maximum,
            label="skin",
            frames=None,
            tolerance=tolerance,
        )
        tongue, tongue_order = self._validated_post_solver_values(
            tongue_weights,
            tongue_pose_names,
            expected_names=ranges.tongue_pose_names,
            minimum=ranges.tongue_minimum,
            maximum=ranges.tongue_maximum,
            label="tongue",
            frames=len(skin),
            tolerance=tolerance,
        )
        skin_matrix = self.calibration.skin_matrix[skin_order]
        tongue_matrix = self.calibration.tongue_matrix[tongue_order]
        # Do not clamp here: values above one and non-zero offsets are part of
        # Claire v3's published post-solver rig.  GNM's independent regional
        # safety bounds are still applied to the final native coefficients.
        return self._bound(skin @ skin_matrix + tongue @ tongue_matrix)

    def retarget_post_solver(
        self,
        skin_weights: Mapping[str, float],
        tongue_weights: Mapping[str, float],
        *,
        tolerance: float = 1.0e-5,
    ) -> np.ndarray:
        """Single-frame mapping variant of :meth:`retarget_post_solver_sequence`."""

        ranges = self.post_solver_ranges
        if ranges is None:
            raise CalibratedRetargetError(
                "This calibration has no post-solver control-range contract"
            )
        missing_skin = sorted(set(ranges.skin_pose_names) - set(skin_weights))
        unknown_skin = sorted(set(skin_weights) - set(ranges.skin_pose_names))
        missing_tongue = sorted(set(ranges.tongue_pose_names) - set(tongue_weights))
        unknown_tongue = sorted(set(tongue_weights) - set(ranges.tongue_pose_names))
        if missing_skin or unknown_skin:
            raise CalibratedRetargetError(
                "Post-solver skin schema mismatch; "
                f"missing={missing_skin}, unknown={unknown_skin}"
            )
        if missing_tongue or unknown_tongue:
            raise CalibratedRetargetError(
                "Post-solver tongue schema mismatch; "
                f"missing={missing_tongue}, unknown={unknown_tongue}"
            )
        return self.retarget_post_solver_sequence(
            np.asarray(
                [[skin_weights[name] for name in ranges.skin_pose_names]],
                dtype=np.float32,
            ),
            ranges.skin_pose_names,
            tongue_weights=np.asarray(
                [[tongue_weights[name] for name in ranges.tongue_pose_names]],
                dtype=np.float32,
            ),
            tongue_pose_names=ranges.tongue_pose_names,
            tolerance=tolerance,
        )[0]
