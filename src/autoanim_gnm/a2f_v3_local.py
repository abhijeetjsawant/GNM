"""Bounded local ONNX execution for the pinned Audio2Face v3 Claire network.

This is an experimental raw-geometry inference boundary, not NVIDIA's SDK.
It verifies the public profile and ONNX ABI, reproduces the repository's
official window schedule, and records enough evidence to audit every model
execution.  Post-processing, blendshape solving, retargeting, and perceptual
qualification intentionally live outside this module.

Long clips should use :func:`consume_local_v3_raw`: the callback receives one
immutable retained chunk at a time, and generated 88,831-value frames are
released before the next execution.  :func:`run_local_v3_raw` is a deliberately
bounded convenience collector for tests and short diagnostic clips.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Protocol
import wave

import numpy as np

from .a2f_v3_profile import (
    OfficialV3ClaireProfile,
    load_official_v3_claire_profile,
)
from .sequence_provider import (
    A2F_V3_DISCARDED_LEFT_FRAMES,
    A2F_V3_GENERATED_FRAMES_PER_INFERENCE,
    A2F_V3_IDENTITY,
    A2F_V3_IDENTITY_INDEX,
    A2F_V3_INFERENCE_WINDOW_SAMPLES,
    A2F_V3_MODEL_REVISION,
    A2F_V3_NETWORK_VERSION,
    A2F_V3_OUTPUT_FPS,
    A2F_V3_PADDING_LEFT_SAMPLES,
    A2F_V3_PUBLIC_MODEL_VERSION,
    A2F_V3_RETAINED_FRAMES_PER_STEP,
    A2F_V3_RETAINED_STEP_SAMPLES,
    A2F_V3_SAMPLE_RATE_HZ,
    A2F_V3_TARGET_OFFSET_SAMPLES,
    AudioSampleClock,
    SequenceChunkPlan,
    SequenceOutputTimebase,
    SequenceProviderError,
    build_official_v3_inference_plan,
    inspect_bound_pcm_audio,
)


QUALITY_A2F_V3_LOCAL_RAW_CANDIDATE = "a2f_v3_local_raw_candidate_unqualified"
LOCAL_V3_OUTPUT_WIDTH = 88_831
LOCAL_V3_NOISE_SHAPE = (1, 3, 60, LOCAL_V3_OUTPUT_WIDTH)
LOCAL_V3_LATENT_SHAPE = (2, 2, 1, 256)
LOCAL_V3_IDENTITY_SHAPE = (1, 3)
LOCAL_V3_EMOTION_SHAPE = (1, 30, 10)
LOCAL_V3_EMOTION_NAMES = (
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "outofbreath",
    "pain",
    "sadness",
)

_PROCESS_INFERENCE_LOCK = threading.Lock()


def default_local_v3_profile_directory() -> Path:
    """Return the configured or repository-cached public Claire-v3 profile."""

    import os

    configured = os.environ.get("AUTOANIM_A2F_V3_PROFILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2]
        / ".cache"
        / "autoanim_gnm"
        / "a2f-v3-claire-profile"
    )


class _NodeArgument(Protocol):
    name: str
    type: str
    shape: list[Any]


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_NodeArgument]: ...
    def get_outputs(self) -> list[_NodeArgument]: ...
    def get_providers(self) -> list[str]: ...
    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


SessionFactory = Callable[[Path, tuple[str, ...]], _InferenceSession]
RawChunkConsumer = Callable[["LocalV3RawChunk"], None]


@dataclass(frozen=True, slots=True)
class LocalV3TensorSignature:
    name: str
    tensor_type: str
    shape: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalV3ChunkEvidence:
    chunk_index: int
    emitted_frame_count: int
    input_pcm_s16le_sha256: str
    input_float32_sha256: str
    identity_sha256: str
    emotion_sha256: str
    input_latents_sha256: str
    noise_sha256: str
    generated_prediction_sha256: str
    retained_prediction_sha256: str
    output_latents_sha256: str
    inference_seconds: float


@dataclass(frozen=True, slots=True)
class LocalV3RawChunk:
    """One model execution's retained raw geometry and audit evidence.

    Warmup executions are delivered with a zero-row ``prediction`` matrix so
    callers can audit them without mistaking generated boundary frames for
    valid output.
    """

    plan: SequenceChunkPlan
    target_samples: tuple[int, ...]
    timestamps_seconds: np.ndarray
    prediction: np.ndarray
    evidence: LocalV3ChunkEvidence

    def __post_init__(self) -> None:
        timestamps = _readonly_array(
            self.timestamps_seconds,
            dtype=np.float64,
            shape=(self.plan.output_frame_count,),
            label="timestamps_seconds",
        )
        prediction = _readonly_array(
            self.prediction,
            dtype=np.float32,
            shape=(self.plan.output_frame_count, LOCAL_V3_OUTPUT_WIDTH),
            label="prediction",
        )
        if self.target_samples != self.plan.output_target_samples:
            raise SequenceProviderError(
                "LOCAL_V3_RETAIN_MISMATCH",
                "Retained target samples differ from the official v3 plan",
            )
        object.__setattr__(self, "timestamps_seconds", timestamps)
        object.__setattr__(self, "prediction", prediction)


@dataclass(frozen=True, slots=True)
class LocalV3RunEvidence:
    quality_label: str
    production_qualified: bool
    sdk_runtime_parity_verified: bool
    public_model_version: str
    network_version: str
    model_revision: str
    identity: str
    identity_index: int
    model_sha256: str
    pinned_model_descriptor_verified: bool
    default_onnxruntime_boundary: bool
    profile_asset_sha256: tuple[tuple[str, str], ...]
    source_artifact_sha256: str
    source_pcm_s16le_sha256: str
    runtime_name: str
    runtime_version: str
    requested_providers: tuple[str, ...]
    available_providers: tuple[str, ...]
    active_providers: tuple[str, ...]
    model_inputs: tuple[LocalV3TensorSignature, ...]
    model_outputs: tuple[LocalV3TensorSignature, ...]
    noise_seed: int
    noise_algorithm: str
    emotion_names: tuple[str, ...]
    emotion_vector: tuple[float, ...]
    numpy_version: str
    session_creation_seconds: float
    inference_seconds: float
    total_seconds: float
    chunks: tuple[LocalV3ChunkEvidence, ...]


@dataclass(frozen=True, slots=True)
class LocalV3RawExecution:
    audio: AudioSampleClock
    output_timebase: SequenceOutputTimebase
    evidence: LocalV3RunEvidence


@dataclass(frozen=True, slots=True)
class LocalV3RawResult:
    """Bounded, immutable collection of retained raw geometry."""

    execution: LocalV3RawExecution
    target_samples: tuple[int, ...]
    timestamps_seconds: np.ndarray
    prediction: np.ndarray

    def __post_init__(self) -> None:
        frames = self.execution.output_timebase.frame_count
        timestamps = _readonly_array(
            self.timestamps_seconds,
            dtype=np.float64,
            shape=(frames,),
            label="timestamps_seconds",
        )
        prediction = _readonly_array(
            self.prediction,
            dtype=np.float32,
            shape=(frames, LOCAL_V3_OUTPUT_WIDTH),
            label="prediction",
        )
        if len(self.target_samples) != frames:
            raise SequenceProviderError(
                "LOCAL_V3_RETAIN_MISMATCH",
                "Collected target-sample count differs from output timebase",
            )
        object.__setattr__(self, "timestamps_seconds", timestamps)
        object.__setattr__(self, "prediction", prediction)


def _readonly_array(
    value: Any, *, dtype: np.dtype[Any] | type[Any], shape: tuple[int, ...], label: str
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.isfinite(array).all():
        raise SequenceProviderError(
            "LOCAL_V3_INVALID_OUTPUT", f"{label} must be finite with shape {shape}"
        )
    output = np.ascontiguousarray(array).copy()
    output.setflags(write=False)
    return output


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    if contiguous.size == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _read_exact_pcm_s16le(path: Path, expected: AudioSampleClock) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            payload = handle.readframes(expected.sample_count)
            trailing = handle.readframes(1)
    except (OSError, EOFError, wave.Error) as exc:
        raise SequenceProviderError(
            "INVALID_AUDIO_CLOCK", "Local v3 audio must remain readable PCM WAV"
        ) from exc
    expected_bytes = expected.sample_count * expected.sample_width_bytes
    if (
        len(payload) != expected_bytes
        or trailing
        or hashlib.sha256(payload).hexdigest() != expected.pcm_s16le_sha256
    ):
        raise SequenceProviderError(
            "INVALID_AUDIO_CLOCK", "Local v3 PCM payload changed while being read"
        )
    samples = np.frombuffer(payload, dtype="<i2").copy()
    if samples.shape != (expected.sample_count,):
        raise SequenceProviderError(
            "INVALID_AUDIO_CLOCK", "Local v3 PCM sample count differs"
        )
    return samples


class _ORTSessionAdapter:
    def __init__(self, session: Any, runtime_version: str, available: tuple[str, ...]):
        self._session = session
        self._autoanim_runtime_version = runtime_version
        self._autoanim_runtime_name = "onnxruntime"
        self._autoanim_available_providers = available

    def get_inputs(self) -> list[_NodeArgument]:
        return self._session.get_inputs()

    def get_outputs(self) -> list[_NodeArgument]:
        return self._session.get_outputs()

    def get_providers(self) -> list[str]:
        return self._session.get_providers()

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        return self._session.run(output_names, input_feed)


def _default_session_factory(
    model_path: Path,
    providers: tuple[str, ...],
    *,
    expected_model_sha256: str,
) -> _InferenceSession:
    # Optional dependency by design: importing this module remains safe on
    # hosts that only validate or import externally generated v3 controls.
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SequenceProviderError(
            "LOCAL_V3_RUNTIME_MISSING",
            "onnxruntime is required for experimental local Audio2Face v3 inference",
        ) from exc
    available = tuple(str(value) for value in ort.get_available_providers())
    missing = tuple(provider for provider in providers if provider not in available)
    if missing:
        raise SequenceProviderError(
            "LOCAL_V3_PROVIDER_UNAVAILABLE",
            f"Requested ONNX Runtime providers are unavailable: {missing}",
        )
    descriptor = -1
    try:
        descriptor = os.open(model_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size <= 0:
            raise SequenceProviderError(
                "LOCAL_V3_MODEL_INVALID", "Pinned local v3 model is not a regular file"
            )
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected_model_sha256:
            raise SequenceProviderError(
                "OFFICIAL_ASSET_HASH_MISMATCH",
                "Pinned local v3 model descriptor hash differs",
            )
        # os.dup shares the open-file description and therefore its offset.
        # Reset the stable descriptor before ORT reopens it through /dev/fd.
        os.lseek(descriptor, 0, os.SEEK_SET)
        descriptor_root = Path("/proc/self/fd")
        if not descriptor_root.is_dir():
            descriptor_root = Path("/dev/fd")
        descriptor_path = descriptor_root / str(descriptor)
        session = ort.InferenceSession(
            str(descriptor_path), providers=list(providers)
        )
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise SequenceProviderError(
                "LOCAL_V3_MODEL_CHANGED",
                "Pinned local v3 model descriptor changed during session construction",
            )
    except SequenceProviderError:
        raise
    except Exception as exc:  # ORT exposes backend-specific exception classes.
        raise SequenceProviderError(
            "LOCAL_V3_SESSION_FAILED", "Unable to create local v3 ONNX session"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    adapter = _ORTSessionAdapter(session, str(ort.__version__), available)
    adapter._autoanim_model_descriptor_hash_verified = True
    adapter._autoanim_default_onnxruntime_boundary = True
    return adapter


def _signature(node: _NodeArgument) -> LocalV3TensorSignature:
    return LocalV3TensorSignature(
        name=str(node.name),
        tensor_type=str(node.type),
        shape=tuple("*" if dim is None or isinstance(dim, str) else str(dim) for dim in node.shape),
    )


_EXPECTED_INPUTS = (
    LocalV3TensorSignature("window", "tensor(float)", ("*", "16000")),
    LocalV3TensorSignature("identity", "tensor(float)", ("*", "3")),
    LocalV3TensorSignature("emotion", "tensor(float)", ("*", "30", "10")),
    LocalV3TensorSignature("input_latents", "tensor(float)", ("2", "2", "*", "256")),
    LocalV3TensorSignature("noise", "tensor(float)", ("*", "3", "60", "88831")),
)
_EXPECTED_OUTPUTS = (
    LocalV3TensorSignature("prediction", "tensor(float)", ("*", "60", "88831")),
    LocalV3TensorSignature("output_latents", "tensor(float)", ("2", "2", "*", "256")),
)


def _verify_model_signature(
    session: _InferenceSession,
) -> tuple[tuple[LocalV3TensorSignature, ...], tuple[LocalV3TensorSignature, ...]]:
    inputs = tuple(_signature(node) for node in session.get_inputs())
    outputs = tuple(_signature(node) for node in session.get_outputs())
    if inputs != _EXPECTED_INPUTS or outputs != _EXPECTED_OUTPUTS:
        raise SequenceProviderError(
            "LOCAL_V3_MODEL_SIGNATURE_MISMATCH",
            "Pinned Audio2Face v3 ONNX input/output signatures differ",
        )
    return inputs, outputs


def _window_for_plan(samples: np.ndarray, plan: SequenceChunkPlan) -> np.ndarray:
    pcm = np.zeros(A2F_V3_INFERENCE_WINDOW_SAMPLES, dtype="<i2")
    source_start = plan.source_intersection_start_sample
    source_end = source_start + plan.source_intersection_sample_count
    destination_start = source_start - plan.audio_start_sample
    destination_end = destination_start + plan.source_intersection_sample_count
    if (
        destination_start != plan.audio_padding_left_samples
        or destination_start < 0
        or destination_end > len(pcm)
        or source_start < 0
        or source_end > len(samples)
    ):
        raise SequenceProviderError(
            "LOCAL_V3_WINDOW_MISMATCH",
            "Official v3 plan cannot be represented by the bound PCM samples",
        )
    pcm[destination_start:destination_end] = samples[source_start:source_end]
    return pcm


def _valid_retained_offsets(plan: SequenceChunkPlan, sample_count: int) -> tuple[int, ...]:
    first_model_frame = plan.chunk_index * A2F_V3_RETAINED_FRAMES_PER_STEP
    offsets: list[int] = []
    targets: list[int] = []
    for offset in range(A2F_V3_RETAINED_FRAMES_PER_STEP):
        frame_window_start = (
            (first_model_frame + offset)
            * A2F_V3_RETAINED_STEP_SAMPLES
            // A2F_V3_RETAINED_FRAMES_PER_STEP
            - A2F_V3_PADDING_LEFT_SAMPLES
        )
        target = frame_window_start + A2F_V3_TARGET_OFFSET_SAMPLES
        if 0 <= target < sample_count:
            offsets.append(offset)
            targets.append(target)
    if tuple(targets) != plan.output_target_samples:
        raise SequenceProviderError(
            "LOCAL_V3_RETAIN_MISMATCH",
            "Retained-frame selection differs from the official v3 plan",
        )
    return tuple(offsets)


def consume_local_v3_raw(
    audio_path: str | Path,
    profile_directory: str | Path,
    consumer: RawChunkConsumer,
    *,
    noise_seed: int = 0,
    emotion_vector: Sequence[float] | np.ndarray | None = None,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
    session_factory: SessionFactory | None = None,
) -> LocalV3RawExecution:
    """Execute the pinned network and stream retained raw geometry to ``consumer``.

    A process-wide lock covers session construction, all recurrent executions,
    and every callback.  This keeps memory bounded and prevents concurrent
    requests from oversubscribing the large model.  The callback must therefore
    return promptly and must not recursively invoke this function.
    """

    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int) or not (
        0 <= noise_seed < 2**64
    ):
        raise SequenceProviderError(
            "LOCAL_V3_INVALID_SEED", "noise_seed must be an unsigned 64-bit integer"
        )
    if not providers or any(not isinstance(item, str) or not item for item in providers):
        raise SequenceProviderError(
            "LOCAL_V3_INVALID_PROVIDER", "At least one named runtime provider is required"
        )
    if emotion_vector is None:
        emotion_values = np.zeros(10, dtype=np.float32)
    else:
        try:
            emotion_values = np.asarray(emotion_vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise SequenceProviderError(
                "LOCAL_V3_INVALID_EMOTION",
                "emotion_vector must contain exactly 10 finite values",
            ) from exc
        if emotion_values.shape != (10,) or not np.isfinite(emotion_values).all():
            raise SequenceProviderError(
                "LOCAL_V3_INVALID_EMOTION",
                "emotion_vector must contain exactly 10 finite values",
            )
        emotion_values = emotion_values.copy()
    if not callable(consumer):
        raise SequenceProviderError(
            "LOCAL_V3_INVALID_CONSUMER", "Raw chunk consumer must be callable"
        )

    audio_source = Path(audio_path).expanduser().resolve()
    audio = inspect_bound_pcm_audio(audio_source)
    samples = _read_exact_pcm_s16le(audio_source, audio)
    frame_count = (
        audio.sample_count * A2F_V3_OUTPUT_FPS + A2F_V3_SAMPLE_RATE_HZ - 1
    ) // A2F_V3_SAMPLE_RATE_HZ
    timebase = SequenceOutputTimebase(
        units="seconds",
        fps_numerator=A2F_V3_OUTPUT_FPS,
        fps_denominator=1,
        frame_count=frame_count,
        timestamp_origin_seconds=0.0,
    )
    plans = build_official_v3_inference_plan(audio, timebase)

    started = time.perf_counter()
    with _PROCESS_INFERENCE_LOCK:
        profile: OfficialV3ClaireProfile = load_official_v3_claire_profile(
            profile_directory, verify_network=session_factory is not None
        )
        model_path = profile.root / "network.onnx"
        session_started = time.perf_counter()
        if session_factory is None:
            session = _default_session_factory(
                model_path,
                providers,
                expected_model_sha256=profile.network_sha256,
            )
        else:
            session = session_factory(model_path, providers)
        session_creation_seconds = time.perf_counter() - session_started
        model_inputs, model_outputs = _verify_model_signature(session)
        active_providers = tuple(str(value) for value in session.get_providers())
        if not active_providers or active_providers[0] != providers[0]:
            raise SequenceProviderError(
                "LOCAL_V3_PROVIDER_MISMATCH",
                "ONNX session did not activate the requested primary provider",
            )
        available_providers = tuple(
            str(value)
            for value in getattr(session, "_autoanim_available_providers", active_providers)
        )
        runtime_version = str(
            getattr(session, "_autoanim_runtime_version", "injected-test-session")
        )

        identity = np.zeros(LOCAL_V3_IDENTITY_SHAPE, dtype=np.float32)
        identity[0, A2F_V3_IDENTITY_INDEX] = 1.0
        emotion = np.broadcast_to(
            emotion_values[None, None, :], LOCAL_V3_EMOTION_SHAPE
        ).copy()
        latents = np.zeros(LOCAL_V3_LATENT_SHAPE, dtype=np.float32)
        identity_hash = _sha256_array(identity)
        emotion_hash = _sha256_array(emotion)
        rng = np.random.Generator(np.random.PCG64(noise_seed))
        chunk_evidence: list[LocalV3ChunkEvidence] = []
        inference_seconds = 0.0

        for plan in plans:
            pcm = _window_for_plan(samples, plan)
            window = (pcm.astype(np.float32) / np.float32(32768.0))[None, :]
            noise = rng.standard_normal(LOCAL_V3_NOISE_SHAPE, dtype=np.float32)
            input_latents_hash = _sha256_array(latents)
            inputs = {
                "window": window,
                "identity": identity,
                "emotion": emotion,
                "input_latents": latents,
                "noise": noise,
            }
            inference_started = time.perf_counter()
            try:
                raw_outputs = session.run(
                    ["prediction", "output_latents"], inputs
                )
            except Exception as exc:
                raise SequenceProviderError(
                    "LOCAL_V3_INFERENCE_FAILED",
                    f"Local v3 inference failed at chunk {plan.chunk_index}",
                ) from exc
            elapsed = time.perf_counter() - inference_started
            inference_seconds += elapsed
            if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != 2:
                raise SequenceProviderError(
                    "LOCAL_V3_INVALID_OUTPUT", "Local v3 runtime returned wrong outputs"
                )
            prediction = np.asarray(raw_outputs[0], dtype=np.float32)
            output_latents = np.asarray(raw_outputs[1], dtype=np.float32)
            if (
                prediction.shape
                != (1, A2F_V3_GENERATED_FRAMES_PER_INFERENCE, LOCAL_V3_OUTPUT_WIDTH)
                or output_latents.shape != LOCAL_V3_LATENT_SHAPE
                or not np.isfinite(prediction).all()
                or not np.isfinite(output_latents).all()
            ):
                raise SequenceProviderError(
                    "LOCAL_V3_INVALID_OUTPUT",
                    "Local v3 outputs have wrong shape or non-finite values",
                )

            offsets = _valid_retained_offsets(plan, audio.sample_count)
            retained_indices = tuple(
                A2F_V3_DISCARDED_LEFT_FRAMES + offset for offset in offsets
            )
            retained = np.ascontiguousarray(
                prediction[0, list(retained_indices), :]
                if retained_indices
                else np.empty((0, LOCAL_V3_OUTPUT_WIDTH), dtype=np.float32)
            )
            timestamps = (
                np.arange(
                    plan.output_start_frame,
                    plan.output_start_frame + plan.output_frame_count,
                    dtype=np.float64,
                )
                / A2F_V3_OUTPUT_FPS
            )
            evidence = LocalV3ChunkEvidence(
                chunk_index=plan.chunk_index,
                emitted_frame_count=plan.output_frame_count,
                input_pcm_s16le_sha256=_sha256_array(pcm),
                input_float32_sha256=_sha256_array(window),
                identity_sha256=identity_hash,
                emotion_sha256=emotion_hash,
                input_latents_sha256=input_latents_hash,
                noise_sha256=_sha256_array(noise),
                generated_prediction_sha256=_sha256_array(prediction),
                retained_prediction_sha256=_sha256_array(retained),
                output_latents_sha256=_sha256_array(output_latents),
                inference_seconds=elapsed,
            )
            chunk = LocalV3RawChunk(
                plan=plan,
                target_samples=plan.output_target_samples,
                timestamps_seconds=timestamps,
                prediction=retained,
                evidence=evidence,
            )
            chunk_evidence.append(evidence)
            # State changes are committed before the callback, including for a
            # zero-output warmup execution.
            latents = np.ascontiguousarray(output_latents).copy()
            consumer(chunk)

        total_seconds = time.perf_counter() - started
        evidence = LocalV3RunEvidence(
            quality_label=QUALITY_A2F_V3_LOCAL_RAW_CANDIDATE,
            production_qualified=False,
            sdk_runtime_parity_verified=False,
            public_model_version=A2F_V3_PUBLIC_MODEL_VERSION,
            network_version=A2F_V3_NETWORK_VERSION,
            model_revision=A2F_V3_MODEL_REVISION,
            identity=A2F_V3_IDENTITY,
            identity_index=A2F_V3_IDENTITY_INDEX,
            model_sha256=profile.network_sha256,
            pinned_model_descriptor_verified=bool(
                getattr(
                    session, "_autoanim_model_descriptor_hash_verified", False
                )
            ),
            default_onnxruntime_boundary=bool(
                getattr(session, "_autoanim_default_onnxruntime_boundary", False)
            ),
            profile_asset_sha256=tuple(sorted(profile.interpretation_asset_sha256.items())),
            source_artifact_sha256=audio.artifact_sha256,
            source_pcm_s16le_sha256=audio.pcm_s16le_sha256,
            runtime_name=str(
                getattr(session, "_autoanim_runtime_name", "injected-session")
            ),
            runtime_version=runtime_version,
            requested_providers=providers,
            available_providers=available_providers,
            active_providers=active_providers,
            model_inputs=model_inputs,
            model_outputs=model_outputs,
            noise_seed=noise_seed,
            noise_algorithm="numpy.random.Generator(PCG64).standard_normal(float32)",
            emotion_names=LOCAL_V3_EMOTION_NAMES,
            emotion_vector=tuple(float(value) for value in emotion_values),
            numpy_version=np.__version__,
            session_creation_seconds=session_creation_seconds,
            inference_seconds=inference_seconds,
            total_seconds=total_seconds,
            chunks=tuple(chunk_evidence),
        )
        return LocalV3RawExecution(audio=audio, output_timebase=timebase, evidence=evidence)


def run_local_v3_raw(
    audio_path: str | Path,
    profile_directory: str | Path,
    *,
    max_collect_frames: int = 120,
    noise_seed: int = 0,
    emotion_vector: Sequence[float] | np.ndarray | None = None,
    providers: tuple[str, ...] = ("CPUExecutionProvider",),
    session_factory: SessionFactory | None = None,
) -> LocalV3RawResult:
    """Collect a short raw-geometry result with an explicit memory bound."""

    if isinstance(max_collect_frames, bool) or not isinstance(max_collect_frames, int) or max_collect_frames <= 0:
        raise SequenceProviderError(
            "LOCAL_V3_INVALID_LIMIT", "max_collect_frames must be a positive integer"
        )
    audio = inspect_bound_pcm_audio(audio_path)
    frame_count = (
        audio.sample_count * A2F_V3_OUTPUT_FPS + A2F_V3_SAMPLE_RATE_HZ - 1
    ) // A2F_V3_SAMPLE_RATE_HZ
    if frame_count > max_collect_frames:
        raise SequenceProviderError(
            "LOCAL_V3_COLLECTION_LIMIT",
            f"Raw result has {frame_count} frames; bounded collector allows {max_collect_frames}",
        )

    target_samples: list[int] = []
    timestamps: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    def collect(chunk: LocalV3RawChunk) -> None:
        target_samples.extend(chunk.target_samples)
        timestamps.append(chunk.timestamps_seconds)
        predictions.append(chunk.prediction)

    execution = consume_local_v3_raw(
        audio_path,
        profile_directory,
        collect,
        noise_seed=noise_seed,
        emotion_vector=emotion_vector,
        providers=providers,
        session_factory=session_factory,
    )
    joined_timestamps = (
        np.concatenate(timestamps)
        if timestamps
        else np.empty((0,), dtype=np.float64)
    )
    joined_prediction = (
        np.concatenate(predictions, axis=0)
        if predictions
        else np.empty((0, LOCAL_V3_OUTPUT_WIDTH), dtype=np.float32)
    )
    return LocalV3RawResult(
        execution=execution,
        target_samples=tuple(target_samples),
        timestamps_seconds=joined_timestamps,
        prediction=joined_prediction,
    )


__all__ = [
    "LOCAL_V3_OUTPUT_WIDTH",
    "LOCAL_V3_EMOTION_NAMES",
    "QUALITY_A2F_V3_LOCAL_RAW_CANDIDATE",
    "LocalV3ChunkEvidence",
    "LocalV3RawChunk",
    "LocalV3RawExecution",
    "LocalV3RawResult",
    "LocalV3RunEvidence",
    "LocalV3TensorSignature",
    "consume_local_v3_raw",
    "default_local_v3_profile_directory",
    "run_local_v3_raw",
]
