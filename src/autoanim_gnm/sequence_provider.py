"""Source-agnostic sequence controls and fail-closed A2F-v3 worker envelopes.

This module is a validation boundary only.  It does not contact a worker, run
Audio2Face, synthesize controls, alter animation, or promote a validated
envelope to production quality.  The v3 quality label explicitly remains an
unqualified sequence-model candidate until the independent lipsync
qualification path approves a retained controls track.  Envelope hashes bind
content but are not worker-authentication signatures; transport identity and
signature verification remain external deployment responsibilities.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence
import wave

import numpy as np

from .artifacts import sha256


CONTROL_SCHEMA_VERSION = "autoanim.sequence-control-schema/1.0"
V3_REQUEST_SCHEMA_VERSION = "autoanim.a2f-v3-worker-request/1.0"
V3_RESPONSE_SCHEMA_VERSION = "autoanim.a2f-v3-worker-response/1.0"
QUALITY_A2F_V3_SEQUENCE_CANDIDATE = "a2f_v3_sequence_candidate_unqualified"
QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW = "a2f_v2_3_framewise_preview"
ZERO_STATE_SHA256 = "0" * 64

_PROVIDER_ID = "nvidia.audio2face-3d"
_V3_MODEL_VERSION = "3.0"
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_BYTES = 256 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


class SequenceProviderError(ValueError):
    """A worker contract, artifact binding, or sequence payload is invalid."""

    def __init__(self, code: str, message: str, *, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True, slots=True)
class SequenceControlNames:
    skin: tuple[str, ...]
    tongue: tuple[str, ...]
    jaw: tuple[str, ...]
    eye: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "skin": list(self.skin),
            "tongue": list(self.tongue),
            "jaw": list(self.jaw),
            "eye": list(self.eye),
        }


@dataclass(frozen=True, slots=True)
class AudioSampleClock:
    artifact_sha256: str
    pcm_s16le_sha256: str
    sample_rate_hz: int
    sample_count: int
    channel_count: int
    sample_width_bytes: int


@dataclass(frozen=True, slots=True)
class SequenceArtifactBindings:
    model_sha256: str
    runtime_sha256: str
    identity_sha256: str
    blendshape_schema_sha256: str


@dataclass(frozen=True, slots=True)
class SequenceOutputTimebase:
    units: str
    fps_numerator: int
    fps_denominator: int
    frame_count: int
    timestamp_origin_seconds: float

    @property
    def fps(self) -> float:
        return self.fps_numerator / self.fps_denominator


@dataclass(frozen=True, slots=True)
class SequenceChunkPlan:
    chunk_index: int
    audio_start_sample: int
    audio_sample_count: int
    audio_overlap_previous_samples: int
    output_start_frame: int
    output_frame_count: int


@dataclass(frozen=True, slots=True)
class V3WorkerRequest:
    provider_id: str
    model_version: str
    quality_label: str
    bindings: SequenceArtifactBindings
    audio: AudioSampleClock
    output_timebase: SequenceOutputTimebase
    chunks: tuple[SequenceChunkPlan, ...]
    control_names: SequenceControlNames
    request_sha256: str


@dataclass(frozen=True, slots=True)
class SequenceChunkProvenance:
    chunk_index: int
    audio_start_sample: int
    audio_sample_count: int
    audio_overlap_previous_samples: int
    output_start_frame: int
    output_frame_count: int
    state_in_sha256: str
    state_out_sha256: str
    chunk_payload_sha256: str


@dataclass(frozen=True, slots=True)
class SequenceProviderTrack:
    """Immutable controls emitted by any validated sequence provider.

    The generic contract supports both the v3 sequence-candidate and the v2.3
    framewise-preview labels.  The A2F-v3 response validator below accepts only
    the former, preventing a framewise response from being relabeled as v3.
    """

    provider_id: str
    model_version: str
    quality_label: str
    bindings: SequenceArtifactBindings
    source_audio_sha256: str
    audio_sample_rate_hz: int
    audio_sample_count: int
    output_timebase: SequenceOutputTimebase
    timestamps: np.ndarray
    control_names: SequenceControlNames
    skin: np.ndarray
    tongue: np.ndarray
    jaw: np.ndarray
    eye: np.ndarray
    chunks: tuple[SequenceChunkProvenance, ...]
    request_sha256: str
    response_sha256: str

    def __post_init__(self) -> None:
        if self.quality_label not in {
            QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
            QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW,
        }:
            raise SequenceProviderError(
                "INVALID_QUALITY_LABEL", "Sequence quality label is unsupported"
            )
        if (
            self.quality_label == QUALITY_A2F_V3_SEQUENCE_CANDIDATE
            and self.model_version != _V3_MODEL_VERSION
        ) or (
            self.quality_label == QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW
            and not self.model_version.startswith("2.3")
        ):
            raise SequenceProviderError(
                "INVALID_QUALITY_LABEL",
                "Sequence quality label and model version disagree",
            )
        for value, field in (
            (self.bindings.model_sha256, "track.bindings.model_sha256"),
            (self.bindings.runtime_sha256, "track.bindings.runtime_sha256"),
            (self.bindings.identity_sha256, "track.bindings.identity_sha256"),
            (
                self.bindings.blendshape_schema_sha256,
                "track.bindings.blendshape_schema_sha256",
            ),
            (self.source_audio_sha256, "track.source_audio_sha256"),
            (self.request_sha256, "track.request_sha256"),
            (self.response_sha256, "track.response_sha256"),
        ):
            _expect_sha(value, field)
        if self.audio_sample_rate_hz <= 0 or self.audio_sample_count <= 0:
            raise SequenceProviderError(
                "INVALID_AUDIO_CLOCK", "Track audio sample clock must be positive"
            )
        if _parse_control_names(self.control_names.as_dict()) != self.control_names:
            raise SequenceProviderError(
                "INVALID_CONTROL_SCHEMA", "Track control names are invalid"
            )
        timestamps = _readonly_numeric_vector(self.timestamps, "track.timestamps")
        expected_frames = (
            self.audio_sample_count * self.output_timebase.fps_numerator
            + self.audio_sample_rate_hz * self.output_timebase.fps_denominator
            - 1
        ) // (
            self.audio_sample_rate_hz * self.output_timebase.fps_denominator
        )
        if (
            self.output_timebase.units != "seconds"
            or self.output_timebase.timestamp_origin_seconds != 0.0
            or self.output_timebase.fps_numerator <= 0
            or self.output_timebase.fps_denominator <= 0
            or self.output_timebase.frame_count != len(timestamps)
            or self.output_timebase.frame_count != expected_frames
            or not 12.0 <= self.output_timebase.fps <= 60.0
            or not np.allclose(
                timestamps,
                np.arange(len(timestamps), dtype=np.float64)
                / self.output_timebase.fps,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise SequenceProviderError(
                "TIMESTAMP_MISMATCH",
                "Track timestamps and rational output clock must match the audio duration",
            )
        matrices = (
            ("skin", self.skin, len(self.control_names.skin)),
            ("tongue", self.tongue, len(self.control_names.tongue)),
            ("jaw", self.jaw, len(self.control_names.jaw)),
            ("eye", self.eye, len(self.control_names.eye)),
        )
        object.__setattr__(self, "timestamps", timestamps)
        for name, value, width in matrices:
            matrix = _readonly_numeric_matrix(
                value, len(timestamps), width, f"track.{name}"
            )
            object.__setattr__(self, name, matrix)
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise SequenceProviderError(
                "TIMESTAMP_MISMATCH", "Sequence timestamps must increase strictly"
            )
        if not self.chunks:
            raise SequenceProviderError(
                "CHUNK_MISMATCH", "Sequence track must retain chunk provenance"
            )
        next_frame = 0
        prior_audio_end = 0
        expected_state_in = ZERO_STATE_SHA256
        for index, chunk in enumerate(self.chunks):
            audio_end = chunk.audio_start_sample + chunk.audio_sample_count
            expected_overlap = (
                0 if index == 0 else prior_audio_end - chunk.audio_start_sample
            )
            if (
                chunk.chunk_index != index
                or chunk.output_start_frame != next_frame
                or chunk.output_frame_count <= 0
                or chunk.state_in_sha256 != expected_state_in
                or chunk.audio_start_sample < 0
                or chunk.audio_sample_count <= 0
                or audio_end > self.audio_sample_count
                or (index == 0 and chunk.audio_start_sample != 0)
                or (index > 0 and expected_overlap <= 0)
                or (index > 0 and audio_end <= prior_audio_end)
                or chunk.audio_overlap_previous_samples != expected_overlap
            ):
                raise SequenceProviderError(
                    "CHUNK_MISMATCH", "Track chunk/state provenance is discontinuous"
                )
            for value, field in (
                (chunk.state_in_sha256, "track.chunk.state_in_sha256"),
                (chunk.state_out_sha256, "track.chunk.state_out_sha256"),
                (chunk.chunk_payload_sha256, "track.chunk.chunk_payload_sha256"),
            ):
                _expect_sha(value, field)
            next_frame += chunk.output_frame_count
            prior_audio_end = audio_end
            expected_state_in = chunk.state_out_sha256
        if next_frame != len(timestamps) or prior_audio_end != self.audio_sample_count:
            raise SequenceProviderError(
                "DURATION_MISMATCH", "Track chunks do not cover all output frames"
            )


@dataclass(frozen=True, slots=True)
class A2FV3WorkerPreflight:
    system: str
    machine: str
    validated_local_runtime: bool
    can_execute_locally: bool
    quality_label: str
    blocker_code: str
    blocker: str
    required_external_capabilities: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def local_a2f_v3_worker_preflight() -> A2FV3WorkerPreflight:
    """Report local capability without probing a network or claiming a runtime."""

    system = platform.system() or "unknown"
    machine = platform.machine() or "unknown"
    if system == "Darwin":
        code = "NVIDIA_V3_EXTERNAL_WORKER_REQUIRED"
        blocker = (
            "No validated NVIDIA Audio2Face-3D v3 runtime is installed in this "
            "macOS process. A separately provisioned Linux or Windows NVIDIA GPU "
            "worker with pinned model, runtime, identity, and blendshape-schema "
            "artifacts is required."
        )
    else:
        code = "NVIDIA_V3_RUNTIME_NOT_VALIDATED"
        blocker = (
            "This process has no validated, version-pinned NVIDIA Audio2Face-3D "
            "v3 runtime. Supply an externally provisioned NVIDIA worker and verify "
            "all request/response artifact bindings before use."
        )
    return A2FV3WorkerPreflight(
        system=system,
        machine=machine,
        validated_local_runtime=False,
        can_execute_locally=False,
        quality_label=QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
        blocker_code=code,
        blocker=blocker,
        required_external_capabilities=(
            "official Audio2Face-3D v3 runtime",
            "supported NVIDIA GPU",
            "Linux or Windows worker",
            "pinned model/runtime/identity/blendshape-schema artifacts",
        ),
    )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SequenceProviderError(
            "INVALID_ENVELOPE", "Worker envelope is not canonical JSON"
        ) from exc


def _payload_sha256(document: Mapping[str, Any], digest_field: str) -> str:
    payload = deepcopy(dict(document))
    payload.pop(digest_field, None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def seal_v3_worker_request_document(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = deepcopy(dict(document))
    sealed.pop("request_sha256", None)
    sealed["request_sha256"] = _payload_sha256(sealed, "request_sha256")
    return sealed


def seal_v3_worker_response_document(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = deepcopy(dict(document))
    sealed.pop("response_sha256", None)
    sealed["response_sha256"] = _payload_sha256(sealed, "response_sha256")
    return sealed


def sequence_chunk_payload_sha256(chunk: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(chunk))
    payload.pop("chunk_payload_sha256", None)
    payload.pop("state_out_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def sequence_state_out_sha256(
    *,
    request_sha256: str,
    chunk_index: int,
    state_in_sha256: str,
    chunk_payload_sha256: str,
) -> str:
    for value, field in (
        (request_sha256, "request_sha256"),
        (state_in_sha256, "state_in_sha256"),
        (chunk_payload_sha256, "chunk_payload_sha256"),
    ):
        _expect_sha(value, field)
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise SequenceProviderError(
            "INVALID_STATE", "chunk_index must be a nonnegative integer"
        )
    return hashlib.sha256(
        _canonical_json(
            {
                "domain": "autoanim.sequence-state-provenance/1.0",
                "request_sha256": request_sha256,
                "chunk_index": chunk_index,
                "state_in_sha256": state_in_sha256,
                "chunk_payload_sha256": chunk_payload_sha256,
            }
        )
    ).hexdigest()


def _pairs_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SequenceProviderError(
                "DUPLICATE_KEY", f"Duplicate JSON key {key!r}", field=key
            )
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise SequenceProviderError("INVALID_NUMBER", f"JSON constant {value!r} is forbidden")


def _load_json_source(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    maximum_bytes: int,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        document = deepcopy(dict(source))
    else:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise SequenceProviderError("ENVELOPE_MISSING", "Worker envelope is missing")
            if path.stat().st_size > maximum_bytes:
                raise SequenceProviderError("ENVELOPE_TOO_LARGE", "Worker envelope is too large")
            payload = path.read_bytes()
        else:
            payload = bytes(source)
            if len(payload) > maximum_bytes:
                raise SequenceProviderError("ENVELOPE_TOO_LARGE", "Worker envelope is too large")
        try:
            document = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_constant,
            )
        except SequenceProviderError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SequenceProviderError(
                "INVALID_JSON", "Worker envelope is not strict UTF-8 JSON"
            ) from exc
    if not isinstance(document, dict):
        raise SequenceProviderError("INVALID_ENVELOPE", "Worker envelope must be an object")
    return document


def _expect_object(value: Any, field: str, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SequenceProviderError(
            "INVALID_ENVELOPE", f"{field} must be an object", field=field
        )
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise SequenceProviderError(
            "INVALID_ENVELOPE",
            f"{field} keys differ (missing={sorted(expected-actual)}, extra={sorted(actual-expected)})",
            field=field,
        )
    return value


def _expect_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SequenceProviderError(
            "INVALID_ENVELOPE", f"{field} must be an array", field=field
        )
    return value


def _expect_string(value: Any, field: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SequenceProviderError(
            "INVALID_ENVELOPE", f"{field} must be a nonempty string", field=field
        )
    return value


def _expect_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SequenceProviderError(
            "INVALID_HASH", f"{field} must be a lowercase SHA-256", field=field
        )
    return value


def _expect_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SequenceProviderError(
            "INVALID_ENVELOPE",
            f"{field} must be an integer >= {minimum}",
            field=field,
        )
    return value


def _expect_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SequenceProviderError(
            "INVALID_ENVELOPE", f"{field} must be numeric", field=field
        )
    number = float(value)
    if not np.isfinite(number):
        raise SequenceProviderError(
            "INVALID_NUMBER", f"{field} must be finite", field=field
        )
    return number


def _readonly_numeric_vector(value: Any, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SequenceProviderError(
            "INVALID_CONTROLS", f"{field} must be numeric", field=field
        ) from exc
    if array.ndim != 1 or not np.isfinite(array).all():
        raise SequenceProviderError(
            "INVALID_CONTROLS", f"{field} must be one finite vector", field=field
        )
    output = array.astype(np.float64, copy=True)
    output.setflags(write=False)
    return output


def _readonly_numeric_matrix(value: Any, rows: int, columns: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SequenceProviderError(
            "INVALID_CONTROLS", f"{field} must be numeric", field=field
        ) from exc
    if array.shape != (rows, columns) or not np.isfinite(array).all():
        raise SequenceProviderError(
            "INVALID_CONTROLS",
            f"{field} must be finite [{rows},{columns}]",
            field=field,
        )
    output = array.astype(np.float32)
    output.setflags(write=False)
    return output


def _parse_control_names(value: Any, field: str = "control_names") -> SequenceControlNames:
    source = _expect_object(value, field, ("skin", "tongue", "jaw", "eye"))
    parsed: dict[str, tuple[str, ...]] = {}
    all_names: set[str] = set()
    for group in ("skin", "tongue", "jaw", "eye"):
        items = _expect_list(source[group], f"{field}.{group}")
        if not items or len(items) > 512:
            raise SequenceProviderError(
                "INVALID_CONTROL_SCHEMA",
                f"{field}.{group} must contain 1-512 names",
                field=f"{field}.{group}",
            )
        names = tuple(
            _expect_string(item, f"{field}.{group}.{index}", maximum=128)
            for index, item in enumerate(items)
        )
        if len(names) != len(set(names)) or any(
            _CONTROL_NAME.fullmatch(name) is None for name in names
        ):
            raise SequenceProviderError(
                "INVALID_CONTROL_SCHEMA",
                f"{field}.{group} names must be unique and machine-safe",
                field=f"{field}.{group}",
            )
        overlap = all_names.intersection(names)
        if overlap:
            raise SequenceProviderError(
                "INVALID_CONTROL_SCHEMA",
                f"Control names must be globally unique: {sorted(overlap)}",
                field=field,
            )
        all_names.update(names)
        parsed[group] = names
    return SequenceControlNames(**parsed)


def parse_sequence_control_schema(source: str | Path | Mapping[str, Any]) -> SequenceControlNames:
    document = _load_json_source(source, maximum_bytes=_MAX_REQUEST_BYTES)
    root = _expect_object(
        document,
        "control_schema",
        ("schema_version", "skin", "tongue", "jaw", "eye"),
    )
    if root["schema_version"] != CONTROL_SCHEMA_VERSION:
        raise SequenceProviderError(
            "UNSUPPORTED_SCHEMA", "Unsupported sequence control schema"
        )
    return _parse_control_names(
        {group: root[group] for group in ("skin", "tongue", "jaw", "eye")}
    )


def inspect_bound_pcm_audio(path: str | Path) -> AudioSampleClock:
    """Hash a normalized mono 16-kHz PCM16 WAV and its exact sample clock."""

    source = Path(path)
    if not source.is_file():
        raise SequenceProviderError("AUDIO_MISSING", "Bound worker audio is missing")
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            declared_frames = handle.getnframes()
            compression = handle.getcomptype()
            digest = hashlib.sha256()
            consumed_frames = 0
            while consumed_frames < declared_frames:
                requested = min(65536, declared_frames - consumed_frames)
                block = handle.readframes(requested)
                frame_bytes = channels * width
                if frame_bytes <= 0 or len(block) % frame_bytes:
                    raise SequenceProviderError(
                        "INVALID_AUDIO_CLOCK", "PCM frame bytes are truncated"
                    )
                actual_frames = len(block) // frame_bytes
                if actual_frames == 0:
                    break
                digest.update(block)
                consumed_frames += actual_frames
    except SequenceProviderError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise SequenceProviderError(
            "INVALID_AUDIO_CLOCK", "Worker audio must be a valid PCM WAV"
        ) from exc
    if (
        channels != 1
        or width != 2
        or rate != 16000
        or compression != "NONE"
        or consumed_frames != declared_frames
        or declared_frames <= 0
    ):
        raise SequenceProviderError(
            "INVALID_AUDIO_CLOCK",
            "Audio2Face v3 worker input must be nonempty mono 16-kHz PCM16",
        )
    return AudioSampleClock(
        artifact_sha256=sha256(source),
        pcm_s16le_sha256=digest.hexdigest(),
        sample_rate_hz=rate,
        sample_count=declared_frames,
        channel_count=channels,
        sample_width_bytes=width,
    )


def _verify_bound_file(path: str | Path, expected: str, field: str) -> None:
    artifact = Path(path)
    if not artifact.is_file():
        raise SequenceProviderError(
            "BINDING_MISSING", f"{field} artifact is missing", field=field
        )
    if sha256(artifact) != expected:
        raise SequenceProviderError(
            "BINDING_MISMATCH", f"{field} artifact hash differs", field=field
        )


def _parse_provider(value: Any, field: str = "provider") -> tuple[str, str, str]:
    source = _expect_object(
        value, field, ("provider_id", "model_version", "quality_label")
    )
    provider_id = _expect_string(source["provider_id"], f"{field}.provider_id")
    model_version = _expect_string(source["model_version"], f"{field}.model_version")
    quality_label = _expect_string(source["quality_label"], f"{field}.quality_label")
    if (
        provider_id != _PROVIDER_ID
        or model_version != _V3_MODEL_VERSION
        or quality_label != QUALITY_A2F_V3_SEQUENCE_CANDIDATE
    ):
        raise SequenceProviderError(
            "PROVIDER_SUBSTITUTION",
            "A v3 worker envelope must explicitly identify the v3 sequence-candidate provider",
            field=field,
        )
    return provider_id, model_version, quality_label


def _parse_bindings(value: Any, field: str = "bindings") -> SequenceArtifactBindings:
    source = _expect_object(
        value,
        field,
        ("model_sha256", "runtime_sha256", "identity_sha256", "blendshape_schema_sha256"),
    )
    return SequenceArtifactBindings(
        model_sha256=_expect_sha(source["model_sha256"], f"{field}.model_sha256"),
        runtime_sha256=_expect_sha(source["runtime_sha256"], f"{field}.runtime_sha256"),
        identity_sha256=_expect_sha(source["identity_sha256"], f"{field}.identity_sha256"),
        blendshape_schema_sha256=_expect_sha(
            source["blendshape_schema_sha256"], f"{field}.blendshape_schema_sha256"
        ),
    )


def _parse_audio(value: Any, field: str = "audio") -> AudioSampleClock:
    source = _expect_object(
        value,
        field,
        (
            "artifact_sha256",
            "pcm_s16le_sha256",
            "sample_rate_hz",
            "sample_count",
            "channel_count",
            "sample_width_bytes",
        ),
    )
    return AudioSampleClock(
        artifact_sha256=_expect_sha(source["artifact_sha256"], f"{field}.artifact_sha256"),
        pcm_s16le_sha256=_expect_sha(
            source["pcm_s16le_sha256"], f"{field}.pcm_s16le_sha256"
        ),
        sample_rate_hz=_expect_integer(source["sample_rate_hz"], f"{field}.sample_rate_hz", minimum=1),
        sample_count=_expect_integer(source["sample_count"], f"{field}.sample_count", minimum=1),
        channel_count=_expect_integer(source["channel_count"], f"{field}.channel_count", minimum=1),
        sample_width_bytes=_expect_integer(
            source["sample_width_bytes"], f"{field}.sample_width_bytes", minimum=1
        ),
    )


def _parse_timebase(value: Any, audio: AudioSampleClock) -> SequenceOutputTimebase:
    field = "output_timebase"
    source = _expect_object(
        value,
        field,
        ("units", "fps_numerator", "fps_denominator", "frame_count", "timestamp_origin_seconds"),
    )
    units = _expect_string(source["units"], f"{field}.units")
    numerator = _expect_integer(source["fps_numerator"], f"{field}.fps_numerator", minimum=1)
    denominator = _expect_integer(source["fps_denominator"], f"{field}.fps_denominator", minimum=1)
    frame_count = _expect_integer(source["frame_count"], f"{field}.frame_count", minimum=2)
    origin = _expect_number(source["timestamp_origin_seconds"], f"{field}.timestamp_origin_seconds")
    fps = numerator / denominator
    expected_frames = (
        audio.sample_count * numerator + audio.sample_rate_hz * denominator - 1
    ) // (audio.sample_rate_hz * denominator)
    if (
        units != "seconds"
        or origin != 0.0
        or not 12.0 <= fps <= 60.0
        or frame_count != expected_frames
    ):
        raise SequenceProviderError(
            "DURATION_MISMATCH",
            "Output clock must start at zero and exactly ceil the bound audio duration",
            field=field,
        )
    return SequenceOutputTimebase(units, numerator, denominator, frame_count, origin)


def _parse_chunk_plan(
    value: Any,
    audio: AudioSampleClock,
    timebase: SequenceOutputTimebase,
) -> tuple[SequenceChunkPlan, ...]:
    items = _expect_list(value, "chunks")
    if not items:
        raise SequenceProviderError("CHUNK_PLAN_INVALID", "At least one sequence chunk is required")
    chunks: list[SequenceChunkPlan] = []
    prior_audio_end = 0
    next_output_frame = 0
    for index, item in enumerate(items):
        field = f"chunks.{index}"
        source = _expect_object(
            item,
            field,
            (
                "chunk_index",
                "audio_start_sample",
                "audio_sample_count",
                "audio_overlap_previous_samples",
                "output_start_frame",
                "output_frame_count",
            ),
        )
        chunk = SequenceChunkPlan(
            chunk_index=_expect_integer(source["chunk_index"], f"{field}.chunk_index"),
            audio_start_sample=_expect_integer(
                source["audio_start_sample"], f"{field}.audio_start_sample"
            ),
            audio_sample_count=_expect_integer(
                source["audio_sample_count"], f"{field}.audio_sample_count", minimum=1
            ),
            audio_overlap_previous_samples=_expect_integer(
                source["audio_overlap_previous_samples"],
                f"{field}.audio_overlap_previous_samples",
            ),
            output_start_frame=_expect_integer(
                source["output_start_frame"], f"{field}.output_start_frame"
            ),
            output_frame_count=_expect_integer(
                source["output_frame_count"], f"{field}.output_frame_count", minimum=1
            ),
        )
        audio_end = chunk.audio_start_sample + chunk.audio_sample_count
        expected_overlap = 0 if index == 0 else prior_audio_end - chunk.audio_start_sample
        if (
            chunk.chunk_index != index
            or chunk.output_start_frame != next_output_frame
            or audio_end > audio.sample_count
            or (index == 0 and chunk.audio_start_sample != 0)
            or (index > 0 and expected_overlap <= 0)
            or (index > 0 and audio_end <= prior_audio_end)
            or chunk.audio_overlap_previous_samples != expected_overlap
        ):
            raise SequenceProviderError(
                "CHUNK_PLAN_INVALID",
                "Chunk indices, output coverage, audio coverage, or overlap provenance differ",
                field=field,
            )
        chunks.append(chunk)
        prior_audio_end = audio_end
        next_output_frame += chunk.output_frame_count
    if prior_audio_end != audio.sample_count or next_output_frame != timebase.frame_count:
        raise SequenceProviderError(
            "CHUNK_PLAN_INVALID",
            "Chunks must cover the complete bound audio and output frame sequence",
        )
    return tuple(chunks)


def validate_v3_worker_request(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    audio_path: str | Path,
    model_path: str | Path,
    runtime_path: str | Path,
    identity_path: str | Path,
    blendshape_schema_path: str | Path,
) -> V3WorkerRequest:
    """Validate a sealed request and every concrete external-worker binding."""

    document = _load_json_source(source, maximum_bytes=_MAX_REQUEST_BYTES)
    root = _expect_object(
        document,
        "request",
        (
            "schema_version",
            "provider",
            "bindings",
            "audio",
            "output_timebase",
            "chunks",
            "request_sha256",
        ),
    )
    if root["schema_version"] != V3_REQUEST_SCHEMA_VERSION:
        raise SequenceProviderError("UNSUPPORTED_SCHEMA", "Unsupported v3 request schema")
    declared_hash = _expect_sha(root["request_sha256"], "request_sha256")
    if declared_hash != _payload_sha256(root, "request_sha256"):
        raise SequenceProviderError("REQUEST_HASH_MISMATCH", "v3 request payload hash differs")
    provider_id, model_version, quality_label = _parse_provider(root["provider"])
    bindings = _parse_bindings(root["bindings"])
    actual_audio = inspect_bound_pcm_audio(audio_path)
    declared_audio = _parse_audio(root["audio"])
    if declared_audio != actual_audio:
        raise SequenceProviderError(
            "AUDIO_BINDING_MISMATCH", "Audio artifact or exact sample clock differs"
        )
    _verify_bound_file(model_path, bindings.model_sha256, "model")
    _verify_bound_file(runtime_path, bindings.runtime_sha256, "runtime")
    _verify_bound_file(identity_path, bindings.identity_sha256, "identity")
    _verify_bound_file(
        blendshape_schema_path, bindings.blendshape_schema_sha256, "blendshape_schema"
    )
    control_names = parse_sequence_control_schema(blendshape_schema_path)
    timebase = _parse_timebase(root["output_timebase"], actual_audio)
    chunks = _parse_chunk_plan(root["chunks"], actual_audio, timebase)
    return V3WorkerRequest(
        provider_id=provider_id,
        model_version=model_version,
        quality_label=quality_label,
        bindings=bindings,
        audio=actual_audio,
        output_timebase=timebase,
        chunks=chunks,
        control_names=control_names,
        request_sha256=declared_hash,
    )


def _provider_dict(request: V3WorkerRequest) -> dict[str, str]:
    return {
        "provider_id": request.provider_id,
        "model_version": request.model_version,
        "quality_label": request.quality_label,
    }


def _parse_response_chunk(
    value: Any,
    *,
    plan: SequenceChunkPlan,
    request: V3WorkerRequest,
    expected_state_in: str,
) -> tuple[SequenceChunkProvenance, np.ndarray, dict[str, np.ndarray]]:
    field = f"chunks.{plan.chunk_index}"
    source = _expect_object(
        value,
        field,
        (
            "chunk_index",
            "audio_start_sample",
            "audio_sample_count",
            "audio_overlap_previous_samples",
            "output_start_frame",
            "output_frame_count",
            "state_in_sha256",
            "state_out_sha256",
            "timestamps_seconds",
            "controls",
            "chunk_payload_sha256",
        ),
    )
    plan_fields = (
        "chunk_index",
        "audio_start_sample",
        "audio_sample_count",
        "audio_overlap_previous_samples",
        "output_start_frame",
        "output_frame_count",
    )
    for name in plan_fields:
        actual = _expect_integer(source[name], f"{field}.{name}")
        if actual != getattr(plan, name):
            raise SequenceProviderError(
                "CHUNK_MISMATCH", f"Response {field}.{name} differs from request", field=field
            )
    state_in = _expect_sha(source["state_in_sha256"], f"{field}.state_in_sha256")
    state_out = _expect_sha(source["state_out_sha256"], f"{field}.state_out_sha256")
    if state_in != expected_state_in:
        raise SequenceProviderError(
            "STATE_CHAIN_MISMATCH", "Sequence state input does not continue the prior chunk", field=field
        )
    timestamps = _readonly_numeric_vector(source["timestamps_seconds"], f"{field}.timestamps_seconds")
    if len(timestamps) != plan.output_frame_count:
        raise SequenceProviderError(
            "TIMESTAMP_MISMATCH", "Chunk timestamp count differs from its output frame count", field=field
        )
    expected_timestamps = (
        np.arange(plan.output_start_frame, plan.output_start_frame + plan.output_frame_count)
        / request.output_timebase.fps
    )
    if not np.allclose(timestamps.astype(np.float64), expected_timestamps, rtol=0.0, atol=2e-6):
        raise SequenceProviderError(
            "TIMESTAMP_MISMATCH", "Chunk timestamps differ from the exact request clock", field=field
        )
    controls_source = _expect_object(source["controls"], f"{field}.controls", ("skin", "tongue", "jaw", "eye"))
    matrices = {
        group: _readonly_numeric_matrix(
            controls_source[group],
            plan.output_frame_count,
            len(getattr(request.control_names, group)),
            f"{field}.controls.{group}",
        )
        for group in ("skin", "tongue", "jaw", "eye")
    }
    declared_chunk_hash = _expect_sha(
        source["chunk_payload_sha256"], f"{field}.chunk_payload_sha256"
    )
    actual_chunk_hash = sequence_chunk_payload_sha256(source)
    if declared_chunk_hash != actual_chunk_hash:
        raise SequenceProviderError(
            "CHUNK_HASH_MISMATCH", "Sequence chunk payload hash differs", field=field
        )
    expected_state_out = sequence_state_out_sha256(
        request_sha256=request.request_sha256,
        chunk_index=plan.chunk_index,
        state_in_sha256=state_in,
        chunk_payload_sha256=declared_chunk_hash,
    )
    if state_out != expected_state_out:
        raise SequenceProviderError(
            "STATE_CHAIN_MISMATCH", "Sequence state output provenance hash differs", field=field
        )
    provenance = SequenceChunkProvenance(
        **{name: getattr(plan, name) for name in plan_fields},
        state_in_sha256=state_in,
        state_out_sha256=state_out,
        chunk_payload_sha256=declared_chunk_hash,
    )
    return provenance, timestamps, matrices


def validate_v3_worker_response(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    request: V3WorkerRequest,
) -> SequenceProviderTrack:
    """Validate and stitch a sealed v3 response into the generic track contract."""

    document = _load_json_source(source, maximum_bytes=_MAX_RESPONSE_BYTES)
    root = _expect_object(
        document,
        "response",
        (
            "schema_version",
            "request_sha256",
            "provider",
            "bindings",
            "audio",
            "output_timebase",
            "control_names",
            "chunks",
            "response_sha256",
        ),
    )
    if root["schema_version"] != V3_RESPONSE_SCHEMA_VERSION:
        raise SequenceProviderError("UNSUPPORTED_SCHEMA", "Unsupported v3 response schema")
    declared_response_hash = _expect_sha(root["response_sha256"], "response_sha256")
    if declared_response_hash != _payload_sha256(root, "response_sha256"):
        raise SequenceProviderError("RESPONSE_HASH_MISMATCH", "v3 response payload hash differs")
    if _expect_sha(root["request_sha256"], "request_sha256") != request.request_sha256:
        raise SequenceProviderError(
            "RESPONSE_BINDING_MISMATCH", "Response references a different request"
        )
    provider_id, model_version, quality_label = _parse_provider(root["provider"])
    bindings = _parse_bindings(root["bindings"])
    audio = _parse_audio(root["audio"])
    timebase = _parse_timebase(root["output_timebase"], audio)
    control_names = _parse_control_names(root["control_names"])
    if (
        _provider_dict(request)
        != {
            "provider_id": provider_id,
            "model_version": model_version,
            "quality_label": quality_label,
        }
        or bindings != request.bindings
        or audio != request.audio
        or timebase != request.output_timebase
        or control_names != request.control_names
    ):
        raise SequenceProviderError(
            "RESPONSE_BINDING_MISMATCH",
            "Response provider, artifacts, audio clock, output clock, or controls schema differs",
        )
    chunk_values = _expect_list(root["chunks"], "chunks")
    if len(chunk_values) != len(request.chunks):
        raise SequenceProviderError(
            "CHUNK_MISMATCH", "Response chunk count differs from request"
        )
    provenance: list[SequenceChunkProvenance] = []
    timestamp_parts: list[np.ndarray] = []
    control_parts: dict[str, list[np.ndarray]] = {
        "skin": [],
        "tongue": [],
        "jaw": [],
        "eye": [],
    }
    expected_state_in = ZERO_STATE_SHA256
    for plan, value in zip(request.chunks, chunk_values, strict=True):
        chunk, timestamps, matrices = _parse_response_chunk(
            value,
            plan=plan,
            request=request,
            expected_state_in=expected_state_in,
        )
        provenance.append(chunk)
        timestamp_parts.append(timestamps)
        for group, matrix in matrices.items():
            control_parts[group].append(matrix)
        expected_state_in = chunk.state_out_sha256
    timestamps = np.concatenate(timestamp_parts)
    if len(timestamps) != request.output_timebase.frame_count:
        raise SequenceProviderError(
            "DURATION_MISMATCH", "Stitched response frame count differs from request"
        )
    return SequenceProviderTrack(
        provider_id=provider_id,
        model_version=model_version,
        quality_label=quality_label,
        bindings=bindings,
        source_audio_sha256=audio.artifact_sha256,
        audio_sample_rate_hz=audio.sample_rate_hz,
        audio_sample_count=audio.sample_count,
        output_timebase=timebase,
        timestamps=timestamps,
        control_names=control_names,
        skin=np.concatenate(control_parts["skin"]),
        tongue=np.concatenate(control_parts["tongue"]),
        jaw=np.concatenate(control_parts["jaw"]),
        eye=np.concatenate(control_parts["eye"]),
        chunks=tuple(provenance),
        request_sha256=request.request_sha256,
        response_sha256=declared_response_hash,
    )
