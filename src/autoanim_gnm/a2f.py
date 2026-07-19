"""NVIDIA Audio2Face-3D Claire inference adapter and GNM retargeting.

The native runner emits Claire's identity-specific PCA coefficients, not
ARKit weights.  This module validates that stream, projects the released
Claire PCA representation directly into the normal equations of NVIDIA's
released bounded blendshape solve, and then maps the resulting named ARKit
controls into GNM's semantic expression prototypes.

No full Claire mesh is reconstructed per frame.  The expensive terms
``A.T @ A``, ``A.T @ (mean - neutral)``, and ``A.T @ PCA.T`` are prepared
once when :class:`ClaireSkinSolver` is created.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Literal

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .rig import ControlRig


DEFAULT_CLAIRE_MODEL = "aufklarer/Audio2Face-3D-v2.3.1-Claire-MLX"
RUNNER_ENVIRONMENT_VARIABLE = "AUTOANIM_A2F_RUNNER"
MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "AUTOANIM_A2F_MODEL_DIR"
_A2F_MODEL_REQUIRED_FILES = frozenset(
    {
        "audio2face3d.safetensors",
        "default_emotion.f32",
        "model_config.json",
        "network_info.json",
    }
)
SUPPORTED_A2F_EMOTIONS = frozenset(
    {
        "neutral",
        "surprise",
        "anger",
        "contempt",
        "disgust",
        "fear",
        "grief",
        "joy",
        "outofbreath",
        "pain",
        "sad",
    }
)


class A2FValidationError(ValueError):
    """Raised when runner output or released assets violate their contract."""


class A2FRunnerError(RuntimeError):
    """Raised when the native Audio2Face runner cannot be resolved or fails."""


@dataclass(frozen=True)
class A2FCoefficientLayout:
    """The coefficient partitions encoded by speech-swift's Codable frame."""

    skin_count: int
    tongue_count: int
    jaw_count: int
    eye_count: int

    @property
    def coefficient_count(self) -> int:
        return self.skin_count + self.tongue_count + self.jaw_count + self.eye_count

    @property
    def skin_slice(self) -> slice:
        return slice(0, self.skin_count)

    @property
    def tongue_slice(self) -> slice:
        return slice(self.skin_count, self.skin_count + self.tongue_count)

    @property
    def jaw_slice(self) -> slice:
        start = self.tongue_slice.stop
        return slice(start, start + self.jaw_count)

    @property
    def eye_slice(self) -> slice:
        start = self.jaw_slice.stop
        return slice(start, start + self.eye_count)

    @classmethod
    def from_json(cls, value: object, *, line_number: int) -> A2FCoefficientLayout:
        if not isinstance(value, Mapping):
            raise A2FValidationError(f"JSONL line {line_number}: layout must be an object")
        keys = ("skinCount", "tongueCount", "jawCount", "eyeCount")
        counts: list[int] = []
        for key in keys:
            item = value.get(key)
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise A2FValidationError(
                    f"JSONL line {line_number}: layout.{key} must be a non-negative integer"
                )
            counts.append(item)
        layout = cls(*counts)
        if layout.coefficient_count <= 0:
            raise A2FValidationError(f"JSONL line {line_number}: layout is empty")
        return layout


CLAIRE_LAYOUT = A2FCoefficientLayout(140, 10, 15, 4)


@dataclass(frozen=True)
class A2FFrame:
    """One validated, timestamped Audio2Face coefficient frame."""

    time_seconds: float
    coefficients: np.ndarray
    layout: A2FCoefficientLayout

    @property
    def skin(self) -> np.ndarray:
        return self.coefficients[self.layout.skin_slice]

    @property
    def tongue(self) -> np.ndarray:
        return self.coefficients[self.layout.tongue_slice]

    @property
    def jaw(self) -> np.ndarray:
        return self.coefficients[self.layout.jaw_slice]

    @property
    def eyes(self) -> np.ndarray:
        return self.coefficients[self.layout.eye_slice]


@dataclass(frozen=True)
class A2FAuxiliaryTrack:
    """Recovered physical jaw observations and native eye rotations.

    Claire's 15 jaw outputs are five XYZ point displacements rather than a
    named blendshape.  ``jaw_rotation_vectors_degrees`` and
    ``jaw_translations`` are the best-fit rigid transform from the released
    neutral jaw points to those observations.  The four eye values are the
    two Maya X/Y rotation channels for the right eye followed by the left eye,
    matching NVIDIA's Claire deployment configuration.
    """

    jaw_points: np.ndarray
    jaw_rotation_matrices: np.ndarray
    jaw_rotation_vectors_degrees: np.ndarray
    jaw_translations: np.ndarray
    jaw_rms_residual: np.ndarray
    eye_rotations_degrees: np.ndarray


def recover_a2f_auxiliary_track(
    frames: Sequence[A2FFrame],
    neutral_jaw: np.ndarray,
) -> A2FAuxiliaryTrack:
    """Recover Claire jaw transforms and preserve its native eye channels."""

    if not frames:
        raise A2FValidationError("Auxiliary recovery requires at least one Audio2Face frame")
    neutral = np.asarray(neutral_jaw, dtype=np.float64)
    if neutral.shape != (5, 3) or not np.isfinite(neutral).all():
        raise A2FValidationError("Claire neutral_jaw must be a finite [5,3] array")
    if any(frame.layout.jaw_count != 15 or frame.layout.eye_count != 4 for frame in frames):
        raise A2FValidationError("Auxiliary recovery requires Claire's 15-jaw/4-eye layout")

    neutral_center = np.mean(neutral, axis=0)
    centered_neutral = neutral - neutral_center
    points = np.empty((len(frames), 5, 3), dtype=np.float64)
    matrices = np.empty((len(frames), 3, 3), dtype=np.float64)
    translations = np.empty((len(frames), 3), dtype=np.float64)
    residuals = np.empty(len(frames), dtype=np.float64)
    eyes = np.empty((len(frames), 2, 2), dtype=np.float64)

    for index, frame in enumerate(frames):
        observed = neutral + np.asarray(frame.jaw, dtype=np.float64).reshape(5, 3)
        observed_center = np.mean(observed, axis=0)
        centered_observed = observed - observed_center
        left, _, right_t = np.linalg.svd(centered_neutral.T @ centered_observed)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
        translation = observed_center - neutral_center @ rotation.T
        predicted = neutral @ rotation.T + translation
        points[index] = observed
        matrices[index] = rotation
        translations[index] = translation
        residuals[index] = np.sqrt(np.mean((predicted - observed) ** 2))
        # NVIDIA's Claire inference configuration lists right-eye offsets
        # before left-eye offsets; preserve that order in the artifact.
        eyes[index] = np.asarray(frame.eyes, dtype=np.float64).reshape(2, 2)

    rotations = Rotation.from_matrix(matrices).as_rotvec()
    result = A2FAuxiliaryTrack(
        jaw_points=points.astype(np.float32),
        jaw_rotation_matrices=matrices.astype(np.float32),
        jaw_rotation_vectors_degrees=np.rad2deg(rotations).astype(np.float32),
        jaw_translations=translations.astype(np.float32),
        jaw_rms_residual=residuals.astype(np.float32),
        eye_rotations_degrees=eyes.astype(np.float32),
    )
    for value in (
        result.jaw_points,
        result.jaw_rotation_matrices,
        result.jaw_rotation_vectors_degrees,
        result.jaw_translations,
        result.jaw_rms_residual,
        result.eye_rotations_degrees,
    ):
        if not np.isfinite(value).all():
            raise A2FValidationError("Recovered jaw/eye track contains non-finite values")
        value.setflags(write=False)
    return result


def _jsonl_text(source: str | bytes | Path | Iterable[str]) -> str:
    if isinstance(source, Path):
        try:
            return source.read_text(encoding="utf-8")
        except OSError as exc:
            raise A2FValidationError(f"Could not read Audio2Face JSONL {source}: {exc}") from exc
    if isinstance(source, bytes):
        try:
            return source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise A2FValidationError("Audio2Face JSONL is not valid UTF-8") from exc
    if isinstance(source, str):
        return source
    return "\n".join(line.rstrip("\r\n") for line in source)


def parse_a2f_jsonl(
    source: str | bytes | Path | Iterable[str],
    *,
    expected_layout: A2FCoefficientLayout | None = CLAIRE_LAYOUT,
) -> tuple[A2FFrame, ...]:
    """Parse and strictly validate speech-swift ``Audio2Face3DFrame`` JSONL.

    Timestamps must be finite, non-negative, and strictly increasing.  Every
    frame must carry the same explicit layout and exactly that many finite
    coefficients.  By default, the required layout is Claire/James's 169
    coefficient layout; pass ``None`` only for an explicitly generic caller.
    """

    frames: list[A2FFrame] = []
    previous_time = -np.inf
    stream_layout: A2FCoefficientLayout | None = None
    for line_number, raw_line in enumerate(_jsonl_text(source).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise A2FValidationError(
                f"JSONL line {line_number}: invalid JSON ({exc.msg})"
            ) from exc
        if not isinstance(value, Mapping):
            raise A2FValidationError(f"JSONL line {line_number}: frame must be an object")

        raw_time = value.get("timeSeconds")
        if (
            not isinstance(raw_time, (int, float))
            or isinstance(raw_time, bool)
            or not np.isfinite(raw_time)
            or raw_time < 0
        ):
            raise A2FValidationError(
                f"JSONL line {line_number}: timeSeconds must be finite and non-negative"
            )
        time_seconds = float(raw_time)
        if time_seconds <= previous_time:
            raise A2FValidationError(
                f"JSONL line {line_number}: timestamps must be strictly increasing"
            )

        layout = A2FCoefficientLayout.from_json(value.get("layout"), line_number=line_number)
        if expected_layout is not None and layout != expected_layout:
            raise A2FValidationError(
                f"JSONL line {line_number}: expected layout {expected_layout}, got {layout}"
            )
        if stream_layout is not None and layout != stream_layout:
            raise A2FValidationError(f"JSONL line {line_number}: coefficient layout changed")

        raw_coefficients = value.get("coefficients")
        if not isinstance(raw_coefficients, list):
            raise A2FValidationError(f"JSONL line {line_number}: coefficients must be an array")
        if len(raw_coefficients) != layout.coefficient_count:
            raise A2FValidationError(
                f"JSONL line {line_number}: expected {layout.coefficient_count} coefficients, "
                f"got {len(raw_coefficients)}"
            )
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw_coefficients):
            raise A2FValidationError(
                f"JSONL line {line_number}: coefficients must contain only numbers"
            )
        coefficients = np.asarray(raw_coefficients, dtype=np.float32)
        if not np.isfinite(coefficients).all():
            raise A2FValidationError(f"JSONL line {line_number}: coefficients must be finite")
        coefficients.setflags(write=False)
        frames.append(A2FFrame(time_seconds, coefficients, layout))
        previous_time = time_seconds
        stream_layout = layout

    if not frames:
        raise A2FValidationError("Audio2Face JSONL contains no frames")
    return tuple(frames)


def resolve_a2f_runner(explicit: str | Path | None = None) -> Path:
    """Resolve an executable native runner without mutating or building it."""

    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    else:
        configured = os.environ.get(RUNNER_ENVIRONMENT_VARIABLE)
        if configured:
            candidates.append(Path(configured).expanduser())
        build = root / "native" / "a2f-runner" / ".build"
        candidates.extend(
            (
                build / "release" / "a2f-runner",
                build / "arm64-apple-macosx" / "release" / "a2f-runner",
                build / "debug" / "a2f-runner",
                build / "arm64-apple-macosx" / "debug" / "a2f-runner",
            )
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    searched = ", ".join(str(candidate) for candidate in candidates) or "<none>"
    raise A2FRunnerError(
        f"Audio2Face runner is unavailable (searched: {searched}). "
        f"Set {RUNNER_ENVIRONMENT_VARIABLE} or build native/a2f-runner."
    )


def resolve_a2f_model_directory(
    explicit: str | Path | None = None,
    *,
    model: str = DEFAULT_CLAIRE_MODEL,
) -> Path:
    """Resolve the exact local Audio2Face bundle used by speech-swift.

    The native runner otherwise resolves a model ID through speech-swift's
    cache at inference time. Production evidence needs a concrete directory
    whose bytes can be hashed and passed with ``--model-dir``. This resolver
    mirrors speech-swift's current flat-cache compatibility and Hub-style
    cache paths without downloading or changing either location.
    """

    if explicit is not None:
        candidates = [Path(explicit).expanduser()]
    else:
        configured = os.environ.get(MODEL_DIRECTORY_ENVIRONMENT_VARIABLE)
        if configured:
            candidates = [Path(configured).expanduser()]
        else:
            if not isinstance(model, str) or not model.strip():
                raise A2FRunnerError("Audio2Face model ID cannot be empty")
            components = model.split("/")
            if (
                len(components) != 2
                or any(not component or component in {".", ".."} for component in components)
                or any("/" in component or "\\" in component for component in components)
            ):
                raise A2FRunnerError(
                    "Audio2Face model ID must be one safe organization/repository pair"
                )
            cache_override = os.environ.get("QWEN3_CACHE_DIR") or os.environ.get(
                "QWEN3_ASR_CACHE_DIR"
            )
            cache_root = (
                Path(cache_override).expanduser()
                if cache_override and cache_override.strip()
                else Path.home() / "Library" / "Caches"
            )
            speech_cache = cache_root / "qwen3-speech"
            flat_key = "".join(
                character if character.isalnum() or character in ".-_" else "_"
                for character in model.replace("/", "_")
            ).strip("._") or "model"
            candidates = [
                speech_cache / flat_key,
                speech_cache / "models" / components[0] / components[1],
            ]

    diagnostics: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        missing = sorted(
            name for name in _A2F_MODEL_REQUIRED_FILES if not (resolved / name).is_file()
        )
        if resolved.is_dir() and not missing:
            return resolved
        diagnostics.append(
            f"{candidate} (missing: {', '.join(missing) if missing else 'directory'})"
        )
    searched = "; ".join(diagnostics) or "<none>"
    raise A2FRunnerError(
        "Audio2Face model bundle is unavailable or incomplete "
        f"(searched: {searched}). Set {MODEL_DIRECTORY_ENVIRONMENT_VARIABLE} "
        "to a complete local Claire MLX bundle."
    )


def run_a2f_runner(
    audio_path: str | Path,
    *,
    runner: str | Path | None = None,
    output_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    model: str = DEFAULT_CLAIRE_MODEL,
    offline: bool = False,
    emotion: str = "neutral",
    emotion_strength: float = 1.0,
    timeout_seconds: float = 900.0,
    expected_layout: A2FCoefficientLayout | None = CLAIRE_LAYOUT,
) -> tuple[A2FFrame, ...]:
    """Run the native MLX adapter and return its validated coefficient frames."""

    audio = Path(audio_path).expanduser().resolve()
    if not audio.is_file():
        raise A2FRunnerError(f"Input audio does not exist: {audio}")
    if audio.suffix.lower() != ".wav":
        raise A2FRunnerError(f"Native Audio2Face input must be WAV: {audio}")
    executable = resolve_a2f_runner(runner)
    selected_emotion = emotion.strip().lower() if isinstance(emotion, str) else ""
    if selected_emotion not in SUPPORTED_A2F_EMOTIONS:
        choices = ", ".join(sorted(SUPPORTED_A2F_EMOTIONS))
        raise A2FRunnerError(f"Unsupported Audio2Face emotion {emotion!r}; choose one of: {choices}")
    if (
        isinstance(emotion_strength, bool)
        or not isinstance(emotion_strength, (int, float))
        or not np.isfinite(emotion_strength)
        or not 0.0 <= float(emotion_strength) <= 1.0
    ):
        raise A2FRunnerError("Audio2Face emotion_strength must be finite and in [0,1]")

    with tempfile.TemporaryDirectory(prefix="autoanim-a2f-") as temporary:
        output = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else Path(temporary) / "motion.jsonl"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(executable), "--input", str(audio), "--output", str(output)]
        if model_dir is not None:
            local_model = Path(model_dir).expanduser().resolve()
            if not local_model.is_dir():
                raise A2FRunnerError(f"Audio2Face model directory does not exist: {local_model}")
            command.extend(("--model-dir", str(local_model)))
        else:
            if not model.strip():
                raise A2FRunnerError("Audio2Face model ID cannot be empty")
            command.extend(("--model", model))
        if offline:
            command.append("--offline")
        command.extend(
            ("--emotion", selected_emotion, "--emotion-strength", str(float(emotion_strength)))
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise A2FRunnerError(f"Audio2Face runner could not complete: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostics"
            raise A2FRunnerError(
                f"Audio2Face runner exited with status {completed.returncode}: {detail}"
            )
        if not output.is_file():
            raise A2FRunnerError(f"Audio2Face runner did not create {output}")
        return parse_a2f_jsonl(output, expected_layout=expected_layout)


def _require_array(
    values: Mapping[str, np.ndarray],
    key: str,
    *,
    ndim: int,
) -> np.ndarray:
    if key not in values:
        raise A2FValidationError(f"Claire asset is missing {key!r}")
    output = np.asarray(values[key])
    if output.ndim != ndim or not np.issubdtype(output.dtype, np.number):
        raise A2FValidationError(f"Claire asset {key!r} has invalid shape or dtype: {output.shape}")
    return output


@dataclass(frozen=True)
class ClaireSkinAssets:
    """Validated released Claire PCA and ARKit skin-rig arrays."""

    root: Path
    pca_basis: np.ndarray
    pca_mean: np.ndarray
    neutral: np.ndarray
    pose_names: tuple[str, ...]
    pose_deltas: np.ndarray
    mask: np.ndarray
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
    tolerance: float = 1e-10

    @classmethod
    def load(cls, directory: str | Path) -> ClaireSkinAssets:
        """Load ``model_data.npz``, ``bs_skin.npz``, and its solver config."""

        root = Path(directory).expanduser().resolve()
        required = ("model_data.npz", "bs_skin.npz", "bs_skin_config.json")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise A2FValidationError(f"Claire asset directory is missing: {', '.join(missing)}")
        try:
            with np.load(root / "model_data.npz", allow_pickle=False) as model_data:
                pca_basis = _require_array(model_data, "shapes_matrix_skin", ndim=3).astype(
                    np.float32, copy=True
                )
                pca_mean = _require_array(model_data, "shapes_mean_skin", ndim=2).astype(
                    np.float32, copy=True
                )
            with np.load(root / "bs_skin.npz", allow_pickle=False) as skin_data:
                neutral = _require_array(skin_data, "neutral", ndim=2).astype(np.float32, copy=True)
                raw_names = np.asarray(skin_data["poseNames"])
                if raw_names.ndim != 1 or len(raw_names) < 2:
                    raise A2FValidationError("Claire poseNames must contain neutral and blendshapes")
                decoded = tuple(
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in raw_names.tolist()
                )
                if decoded[0] != "neutral":
                    raise A2FValidationError("Claire poseNames must begin with neutral")
                pose_names = decoded[1:]
                missing_poses = [name for name in pose_names if name not in skin_data]
                if missing_poses:
                    raise A2FValidationError(
                        f"Claire skin data is missing pose deltas: {', '.join(missing_poses)}"
                    )
                pose_deltas = np.stack(
                    [np.asarray(skin_data[name], dtype=np.float32) for name in pose_names], axis=0
                )
                mask = np.asarray(skin_data["frontalMask"], dtype=np.int64).copy()
            config = json.loads((root / "bs_skin_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, A2FValidationError):
                raise
            raise A2FValidationError(f"Could not load Claire skin assets: {exc}") from exc

        if pca_basis.shape[1:] != pca_mean.shape or neutral.shape != pca_mean.shape:
            raise A2FValidationError(
                "Claire PCA basis, mean, and neutral geometry have inconsistent shapes"
            )
        if pose_deltas.shape != (len(pose_names),) + neutral.shape:
            raise A2FValidationError("Claire blendshape deltas have inconsistent shapes")
        if mask.ndim != 1 or mask.size == 0 or np.any(mask < 0) or np.any(mask >= len(neutral)):
            raise A2FValidationError("Claire frontalMask contains invalid vertex indices")
        if len(np.unique(mask)) != len(mask):
            raise A2FValidationError("Claire frontalMask contains duplicate vertex indices")

        try:
            params = config["blendshape_params"]
            num_poses = int(params["numPoses"])
            active = np.asarray(params["bsSolveActivePoses"], dtype=np.int8)
            cancel = np.asarray(params["bsSolveCancelPoses"], dtype=np.int32)
            symmetry = np.asarray(params["bsSolveSymmetryPoses"], dtype=np.int32)
            multipliers = np.asarray(params["bsWeightMultipliers"], dtype=np.float64)
            offsets = np.asarray(params["bsWeightOffsets"], dtype=np.float64)
            l1 = float(params["strengthL1regularization"])
            l2 = float(params["strengthL2regularization"])
            temporal = float(params["strengthTemporalSmoothing"])
            symmetry_strength = float(params["strengthSymmetry"])
            template_bb_size = float(params["templateBBSize"])
            tolerance = float(params.get("tolerance", 1e-10))
        except (KeyError, TypeError, ValueError) as exc:
            raise A2FValidationError(f"Claire solver config is invalid: {exc}") from exc
        vectors = (active, cancel, symmetry, multipliers, offsets)
        if num_poses != len(pose_names) or any(len(vector) != num_poses for vector in vectors):
            raise A2FValidationError("Claire solver config pose counts do not match poseNames")
        if not np.any(active):
            raise A2FValidationError("Claire solver config has no active poses")
        scalars = (l1, l2, temporal, symmetry_strength, template_bb_size, tolerance)
        if not all(np.isfinite(value) for value in scalars) or any(value < 0 for value in scalars):
            raise A2FValidationError("Claire solver regularization values must be finite and non-negative")
        if template_bb_size <= 0 or tolerance <= 0:
            raise A2FValidationError("Claire templateBBSize and tolerance must be positive")
        if not np.isfinite(multipliers).all() or not np.isfinite(offsets).all():
            raise A2FValidationError("Claire weight transforms must be finite")

        return cls(
            root=root,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            neutral=neutral,
            pose_names=pose_names,
            pose_deltas=pose_deltas,
            mask=mask,
            active=active.astype(bool),
            cancel_groups=cancel,
            symmetry_groups=symmetry,
            multipliers=multipliers,
            offsets=offsets,
            l1_regularization=l1,
            l2_regularization=l2,
            temporal_regularization=temporal,
            symmetry_regularization=symmetry_strength,
            template_bb_size=template_bb_size,
            tolerance=tolerance,
        )


@dataclass(frozen=True)
class ClaireTongueAssets(ClaireSkinAssets):
    """Validated released Claire tongue PCA and 16-control rig arrays."""

    @classmethod
    def load(cls, directory: str | Path) -> ClaireTongueAssets:
        root = Path(directory).expanduser().resolve()
        required = ("model_data.npz", "bs_tongue.npz", "bs_tongue_config.json")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise A2FValidationError(f"Claire asset directory is missing: {', '.join(missing)}")
        try:
            with np.load(root / "model_data.npz", allow_pickle=False) as model_data:
                pca_basis = _require_array(model_data, "shapes_matrix_tongue", ndim=3).astype(
                    np.float32, copy=True
                )
                pca_mean = _require_array(model_data, "shapes_mean_tongue", ndim=2).astype(
                    np.float32, copy=True
                )
            with np.load(root / "bs_tongue.npz", allow_pickle=False) as tongue_data:
                neutral = _require_array(tongue_data, "neutral", ndim=2).astype(
                    np.float32, copy=True
                )
                raw_names = np.asarray(tongue_data["poseNames"])
                if raw_names.ndim != 1 or len(raw_names) < 2:
                    raise A2FValidationError("Claire tongue poseNames must contain neutral and controls")
                decoded = tuple(
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in raw_names.tolist()
                )
                if decoded[0] != "neutral":
                    raise A2FValidationError("Claire tongue poseNames must begin with neutral")
                pose_names = decoded[1:]
                missing_poses = [name for name in pose_names if name not in tongue_data]
                if missing_poses:
                    raise A2FValidationError(
                        f"Claire tongue data is missing pose deltas: {', '.join(missing_poses)}"
                    )
                pose_deltas = np.stack(
                    [np.asarray(tongue_data[name], dtype=np.float32) for name in pose_names], axis=0
                )
            config = json.loads((root / "bs_tongue_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if isinstance(exc, A2FValidationError):
                raise
            raise A2FValidationError(f"Could not load Claire tongue assets: {exc}") from exc

        if pca_basis.shape[1:] != pca_mean.shape or neutral.shape != pca_mean.shape:
            raise A2FValidationError(
                "Claire tongue PCA basis, mean, and neutral geometry have inconsistent shapes"
            )
        if pose_deltas.shape != (len(pose_names),) + neutral.shape:
            raise A2FValidationError("Claire tongue control deltas have inconsistent shapes")
        try:
            params = config["blendshape_params"]
            num_poses = int(params["numPoses"])
            active = np.asarray(params["bsSolveActivePoses"], dtype=np.int8)
            cancel = np.asarray(params["bsSolveCancelPoses"], dtype=np.int32)
            symmetry = np.asarray(params["bsSolveSymmetryPoses"], dtype=np.int32)
            multipliers = np.asarray(params["bsWeightMultipliers"], dtype=np.float64)
            offsets = np.asarray(params["bsWeightOffsets"], dtype=np.float64)
            l1 = float(params["strengthL1regularization"])
            l2 = float(params["strengthL2regularization"])
            temporal = float(params["strengthTemporalSmoothing"])
            symmetry_strength = float(params["strengthSymmetry"])
            template_bb_size = float(params["templateBBSize"])
            tolerance = float(params.get("tolerance", 1e-10))
        except (KeyError, TypeError, ValueError) as exc:
            raise A2FValidationError(f"Claire tongue solver config is invalid: {exc}") from exc
        vectors = (active, cancel, symmetry, multipliers, offsets)
        if num_poses != len(pose_names) or any(len(vector) != num_poses for vector in vectors):
            raise A2FValidationError("Claire tongue solver config pose counts do not match poseNames")
        if not np.any(active):
            raise A2FValidationError("Claire tongue solver config has no active controls")
        scalars = (l1, l2, temporal, symmetry_strength, template_bb_size, tolerance)
        if not all(np.isfinite(value) for value in scalars) or any(value < 0 for value in scalars):
            raise A2FValidationError(
                "Claire tongue solver regularization values must be finite and non-negative"
            )
        if template_bb_size <= 0 or tolerance <= 0:
            raise A2FValidationError("Claire tongue templateBBSize and tolerance must be positive")
        if not np.isfinite(multipliers).all() or not np.isfinite(offsets).all():
            raise A2FValidationError("Claire tongue weight transforms must be finite")

        return cls(
            root=root,
            pca_basis=pca_basis,
            pca_mean=pca_mean,
            neutral=neutral,
            pose_names=pose_names,
            pose_deltas=pose_deltas,
            mask=np.arange(len(neutral), dtype=np.int64),
            active=active.astype(bool),
            cancel_groups=cancel,
            symmetry_groups=symmetry,
            multipliers=multipliers,
            offsets=offsets,
            l1_regularization=l1,
            l2_regularization=l2,
            temporal_regularization=temporal,
            symmetry_regularization=symmetry_strength,
            template_bb_size=template_bb_size,
            tolerance=tolerance,
        )


def _pair_indices(groups: np.ndarray, active_indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    active_groups = groups[active_indices]
    pairs: list[tuple[int, int]] = []
    for group in sorted(set(int(item) for item in active_groups if item >= 0)):
        indices = np.flatnonzero(active_groups == group)
        if len(indices) == 2:
            pairs.append((int(indices[0]), int(indices[1])))
    return tuple(pairs)


@dataclass
class ClaireSkinSolver:
    """Reduced bounded least-squares solver for Claire skin PCA frames."""

    pose_names: tuple[str, ...]
    pca_count: int
    active_indices: np.ndarray
    multipliers: np.ndarray
    offsets: np.ndarray
    projected_mean_delta: np.ndarray
    projected_pca: np.ndarray
    normal_matrix: np.ndarray
    least_squares_matrix: np.ndarray
    scale_factor: float
    temporal_regularization: float
    tolerance: float
    cancel_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_directory(cls, directory: str | Path) -> ClaireSkinSolver:
        return cls.from_assets(ClaireSkinAssets.load(directory))

    @classmethod
    def from_assets(cls, assets: ClaireSkinAssets) -> ClaireSkinSolver:
        active_indices = np.flatnonzero(assets.active)
        mask = assets.mask
        # A is [masked xyz positions, active ARKit controls].  The PCA basis
        # remains in coefficient space; only these reduced products survive.
        active_deltas = assets.pose_deltas[active_indices][:, mask, :]
        a_matrix = active_deltas.reshape(len(active_indices), -1).T.astype(np.float64)
        masked_pca = assets.pca_basis[:, mask, :].reshape(len(assets.pca_basis), -1)
        mean_delta = (assets.pca_mean[mask] - assets.neutral[mask]).reshape(-1)
        if not (
            np.isfinite(a_matrix).all()
            and np.isfinite(masked_pca).all()
            and np.isfinite(mean_delta).all()
        ):
            raise A2FValidationError("Claire reduced skin arrays contain non-finite values")

        projected_mean_delta = a_matrix.T @ mean_delta.astype(np.float64)
        projected_pca = a_matrix.T @ masked_pca.astype(np.float64).T
        normal = a_matrix.T @ a_matrix

        extent = np.max(assets.neutral, axis=0) - np.min(assets.neutral, axis=0)
        scale_factor = float((np.linalg.norm(extent) / assets.template_bb_size) ** 2)
        symmetry_pairs = _pair_indices(assets.symmetry_groups, active_indices)
        symmetry_matrix = np.zeros((len(symmetry_pairs), len(active_indices)), dtype=np.float64)
        for row, (left, right) in enumerate(symmetry_pairs):
            symmetry_matrix[row, left] = 1.0
            symmetry_matrix[row, right] = -1.0

        count = len(active_indices)
        normal += (
            assets.l1_regularization**2
            * 0.25
            * scale_factor
            * np.ones((count, count), dtype=np.float64)
        )
        normal += (
            assets.l2_regularization * 10.0 * scale_factor
            + assets.temporal_regularization * 100.0 * scale_factor
        ) * np.eye(count, dtype=np.float64)
        if len(symmetry_matrix):
            normal += (
                assets.symmetry_regularization
                * 10.0
                * scale_factor
                * (symmetry_matrix.T @ symmetry_matrix)
            )
        normal = 0.5 * (normal + normal.T)
        try:
            # scipy's BVLS works on Cx ~= y.  C=L.T and L@y=b preserve
            # C.T@C=normal and C.T@y=b for NVIDIA's normal equation.
            cholesky = np.linalg.cholesky(normal)
        except np.linalg.LinAlgError:
            jitter = max(float(np.linalg.norm(normal, ord=2)) * 1e-10, 1e-12)
            normal += jitter * np.eye(count, dtype=np.float64)
            try:
                cholesky = np.linalg.cholesky(normal)
            except np.linalg.LinAlgError as exc:
                raise A2FValidationError("Claire blendshape normal matrix is not positive definite") from exc

        return cls(
            pose_names=assets.pose_names,
            pca_count=len(assets.pca_basis),
            active_indices=active_indices,
            multipliers=assets.multipliers.copy(),
            offsets=assets.offsets.copy(),
            projected_mean_delta=projected_mean_delta,
            projected_pca=projected_pca,
            normal_matrix=normal,
            least_squares_matrix=cholesky.T,
            scale_factor=scale_factor,
            temporal_regularization=assets.temporal_regularization,
            tolerance=assets.tolerance,
            cancel_pairs=_pair_indices(assets.cancel_groups, active_indices),
        )

    def _bounded_solve(self, right_hand_side: np.ndarray, upper: np.ndarray) -> np.ndarray:
        cholesky = self.least_squares_matrix.T
        target = np.linalg.solve(cholesky, right_hand_side)
        result = lsq_linear(
            self.least_squares_matrix,
            target,
            bounds=(np.zeros_like(upper), upper),
            method="bvls",
            tol=max(self.tolerance, 1e-12),
        )
        if not result.success or not np.isfinite(result.x).all():
            raise A2FValidationError(f"Claire bounded blendshape solve failed: {result.message}")
        return np.asarray(result.x, dtype=np.float64)

    def solve_coefficients(self, skin_coefficients: np.ndarray) -> np.ndarray:
        """Convert ``[frames, PCA]`` Claire skin coefficients to named ARKit weights."""

        coefficients = np.asarray(skin_coefficients, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        if coefficients.ndim != 2 or coefficients.shape[1] != self.pca_count:
            raise A2FValidationError(
                f"Expected [frames,{self.pca_count}] Claire skin coefficients, got {coefficients.shape}"
            )
        if not np.isfinite(coefficients).all():
            raise A2FValidationError("Claire skin coefficients must be finite")

        output = np.zeros((len(coefficients), len(self.pose_names)), dtype=np.float32)
        previous = np.zeros(len(self.active_indices), dtype=np.float64)
        for frame_index, frame in enumerate(coefficients):
            right_hand_side = self.projected_mean_delta + self.projected_pca @ frame
            # This mirrors NVIDIA's released CPU solver: the temporal term in
            # b is TemporalReg*scale*prev, while the normal matrix uses its
            # separately published 100*scale factor.
            right_hand_side += (
                self.temporal_regularization * self.scale_factor * previous
            )
            upper = np.ones(len(self.active_indices), dtype=np.float64)
            solved = self._bounded_solve(right_hand_side, upper)
            if self.cancel_pairs:
                for left, right in self.cancel_pairs:
                    upper[right if solved[left] >= solved[right] else left] = 1e-10
                solved = self._bounded_solve(right_hand_side, upper)
            previous = solved
            full = np.zeros(len(self.pose_names), dtype=np.float64)
            full[self.active_indices] = solved
            full = full * self.multipliers + self.offsets
            # Released Claire's transforms are identity.  Clipping here keeps
            # a stable ARKit [0,1] contract if a custom config changes them.
            output[frame_index] = np.clip(full, 0.0, 1.0).astype(np.float32)
        return output

    def solve_frames(self, frames: Sequence[A2FFrame]) -> np.ndarray:
        if not frames:
            raise A2FValidationError("Cannot solve an empty Audio2Face frame sequence")
        if any(frame.layout.skin_count != self.pca_count for frame in frames):
            raise A2FValidationError("Audio2Face frame skin layout does not match Claire assets")
        return self.solve_coefficients(np.stack([frame.skin for frame in frames], axis=0))


class ClaireTongueSolver(ClaireSkinSolver):
    """Reduced bounded solve for Claire's 10 PCA to 16 tongue controls."""

    @classmethod
    def from_directory(cls, directory: str | Path) -> ClaireTongueSolver:
        return cls.from_assets(ClaireTongueAssets.load(directory))

    def solve_frames(self, frames: Sequence[A2FFrame]) -> np.ndarray:
        if not frames:
            raise A2FValidationError("Cannot solve an empty Audio2Face frame sequence")
        if any(frame.layout.tongue_count != self.pca_count for frame in frames):
            raise A2FValidationError("Audio2Face frame tongue layout does not match Claire assets")
        return self.solve_coefficients(np.stack([frame.tongue for frame in frames], axis=0))


RegionName = Literal["left_eye", "right_eye", "lower_face", "tongue"]


@dataclass(frozen=True)
class RetargetRule:
    """One inspectable semantic approximation from ARKit into GNM."""

    target_prototype: str
    sources: tuple[str, ...]
    gain: float
    regions: tuple[RegionName, ...]
    reduction: Literal["mean", "max"] = "mean"


# GNM is not an ARKit rig and has no jaw joint.  These rules intentionally
# expose every approximation instead of hiding it in an opaque matrix.
ARKIT_TO_GNM_RULES: tuple[RetargetRule, ...] = (
    RetargetRule("wink_left", ("eyeBlinkLeft",), 1.0, ("left_eye",)),
    RetargetRule("wink_right", ("eyeBlinkRight",), 1.0, ("right_eye",)),
    RetargetRule("squint", ("eyeSquintLeft", "cheekSquintLeft"), 0.70, ("left_eye",), "max"),
    RetargetRule("squint", ("eyeSquintRight", "cheekSquintRight"), 0.70, ("right_eye",), "max"),
    RetargetRule("surprise", ("eyeWideLeft", "browOuterUpLeft", "browInnerUp"), 0.38, ("left_eye",), "max"),
    RetargetRule("surprise", ("eyeWideRight", "browOuterUpRight", "browInnerUp"), 0.38, ("right_eye",), "max"),
    RetargetRule("snarl", ("browDownLeft", "noseSneerLeft"), 0.42, ("left_eye",), "max"),
    RetargetRule("snarl", ("browDownRight", "noseSneerRight"), 0.42, ("right_eye",), "max"),
    RetargetRule("stretch_face", ("jawOpen",), 0.90, ("lower_face",)),
    RetargetRule("compress_face", ("mouthClose",), -0.55, ("lower_face",)),
    RetargetRule("compress_face", ("mouthPressLeft", "mouthPressRight"), -0.30, ("lower_face",)),
    RetargetRule("funneler", ("mouthFunnel",), 0.90, ("lower_face",)),
    RetargetRule("pucker", ("mouthPucker",), 0.90, ("lower_face",)),
    RetargetRule("mouth_left", ("mouthLeft", "jawLeft"), 0.70, ("lower_face",), "max"),
    RetargetRule("mouth_right", ("mouthRight", "jawRight"), 0.70, ("lower_face",), "max"),
    RetargetRule("smile_wide", ("mouthSmileLeft", "mouthSmileRight"), 0.80, ("lower_face",), "mean"),
    RetargetRule("corners_down", ("mouthFrownLeft", "mouthFrownRight"), 0.80, ("lower_face",), "mean"),
    RetargetRule("stretch_face", ("mouthStretchLeft", "mouthStretchRight"), 0.50, ("lower_face",), "mean"),
    RetargetRule("lips_roll_in", ("mouthRollLower", "mouthRollUpper"), -0.55, ("lower_face",), "mean"),
    RetargetRule("suck", ("mouthShrugLower", "mouthShrugUpper"), 0.35, ("lower_face",), "mean"),
    RetargetRule("snarl", ("mouthUpperUpLeft", "mouthUpperUpRight"), 0.35, ("lower_face",), "mean"),
    RetargetRule("stretch_face", ("mouthLowerDownLeft", "mouthLowerDownRight"), 0.22, ("lower_face",), "mean"),
    RetargetRule("blow", ("cheekPuff",), 0.60, ("lower_face",)),
    RetargetRule("tongue_center", ("tongueOut",), 0.80, ("tongue",)),
)


TONGUE_TO_GNM_RULES: tuple[RetargetRule, ...] = (
    # GNM exposes one semantic tongue prototype, so directional Claire
    # controls can only be reduced to a visible central tongue gesture.
    RetargetRule(
        "tongue_center",
        ("tongueTipUp", "tongueRollUp", "tongueUp", "tongueStretch"),
        0.85,
        ("tongue",),
        "max",
    ),
)


_REGION_SLICES: dict[RegionName, slice] = {
    "left_eye": slice(0, 100),
    "right_eye": slice(100, 200),
    "lower_face": slice(200, 350),
    "tongue": slice(350, 382),
}


class ARKitGNMRetargeter:
    """Apply the documented semantic approximation to a :class:`ControlRig`."""

    def __init__(
        self,
        rig: ControlRig,
        rules: Sequence[RetargetRule] = ARKIT_TO_GNM_RULES,
        tongue_rules: Sequence[RetargetRule] = TONGUE_TO_GNM_RULES,
    ):
        self.rig = rig
        self.rules = tuple(rules)
        self.tongue_rules = tuple(tongue_rules)
        self._prototypes = {
            rule.target_prototype: rig.decoder.prototype(rule.target_prototype)
            for rule in self.rules + self.tongue_rules
        }

    def retarget(
        self,
        weights: Mapping[str, float],
        tongue_weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        tongue_weights = tongue_weights or {}
        for name, value in tuple(weights.items()) + tuple(tongue_weights.items()):
            if not np.isfinite(value):
                raise A2FValidationError(f"ARKit weight {name!r} is not finite")
        result = np.zeros(self.rig.adapter.expression_dim, dtype=np.float32)
        for rule, source_weights in (
            *((rule, weights) for rule in self.rules),
            *((rule, tongue_weights) for rule in self.tongue_rules),
        ):
            values = np.asarray(
                [
                    np.clip(float(source_weights.get(source, 0.0)), 0.0, 1.0)
                    for source in rule.sources
                ],
                dtype=np.float32,
            )
            strength = float(np.max(values) if rule.reduction == "max" else np.mean(values))
            if strength <= 0:
                continue
            prototype = self._prototypes[rule.target_prototype]
            for region in rule.regions:
                region_slice = _REGION_SLICES[region]
                result[region_slice] += (
                    np.float32(rule.gain * strength) * prototype[region_slice]
                )
        # Match ControlRig's per-region magnitude contract without depending
        # on its private implementation.
        for region_slice in _REGION_SLICES.values():
            maximum = float(np.max(np.abs(result[region_slice]), initial=0.0))
            if maximum > 3.0:
                result[region_slice] *= np.float32(3.0 / maximum)
        if not np.isfinite(result).all():
            raise A2FValidationError("ARKit-to-GNM retarget produced non-finite controls")
        return result

    def retarget_sequence(
        self,
        weights: np.ndarray,
        pose_names: Sequence[str],
        *,
        tongue_weights: np.ndarray | None = None,
        tongue_pose_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        values = np.asarray(weights, dtype=np.float32)
        names = tuple(pose_names)
        if values.ndim != 2 or values.shape[1] != len(names):
            raise A2FValidationError(
                f"Expected [frames,{len(names)}] ARKit weights, got {values.shape}"
            )
        if len(set(names)) != len(names):
            raise A2FValidationError("ARKit pose names must be unique")
        if not np.isfinite(values).all():
            raise A2FValidationError("ARKit weights must be finite")
        tongue_values: np.ndarray | None = None
        tongue_names: tuple[str, ...] = ()
        if tongue_weights is not None:
            if tongue_pose_names is None:
                raise A2FValidationError("tongue_pose_names are required with tongue_weights")
            tongue_values = np.asarray(tongue_weights, dtype=np.float32)
            tongue_names = tuple(tongue_pose_names)
            if tongue_values.ndim != 2 or tongue_values.shape != (
                len(values),
                len(tongue_names),
            ):
                raise A2FValidationError(
                    f"Expected tongue weights [{len(values)},{len(tongue_names)}], "
                    f"got {tongue_values.shape}"
                )
            if len(set(tongue_names)) != len(tongue_names):
                raise A2FValidationError("Tongue pose names must be unique")
            if not np.isfinite(tongue_values).all():
                raise A2FValidationError("Tongue weights must be finite")
        return np.stack(
            [
                self.retarget(
                    dict(zip(names, frame, strict=True)),
                    (
                        dict(zip(tongue_names, tongue_values[index], strict=True))
                        if tongue_values is not None
                        else None
                    ),
                )
                for index, frame in enumerate(values)
            ],
            axis=0,
        )
