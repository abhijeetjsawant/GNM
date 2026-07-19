from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import wave

import numpy as np
import pytest

from autoanim_gnm.artifacts import sha256
from autoanim_gnm.sequence_provider import (
    CONTROL_SCHEMA_VERSION,
    QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW,
    QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
    V3_REQUEST_SCHEMA_VERSION,
    V3_RESPONSE_SCHEMA_VERSION,
    EXECUTION_CHAIN_ROOT_SHA256,
    AudioSampleClock,
    SequenceArtifactBindings,
    SequenceChunkProvenance,
    SequenceControlNames,
    SequenceProviderError,
    SequenceProviderTrack,
    SequenceOutputTimebase,
    build_official_v3_inference_plan,
    inspect_bound_pcm_audio,
    local_a2f_v3_worker_preflight,
    seal_v3_worker_request_document,
    seal_v3_worker_response_document,
    sequence_chunk_payload_sha256,
    sequence_execution_chain_out_sha256,
    validate_v3_worker_request,
    validate_v3_worker_response,
)
from autoanim_gnm.serialization import write_json


@dataclass
class WorkerCase:
    audio_path: Path
    model_path: Path
    runtime_path: Path
    identity_path: Path
    schema_path: Path
    request_document: dict[str, object]
    request_path: Path
    request: object
    response_document: dict[str, object]
    response_path: Path


def _write_pcm_wave(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> Path:
    values = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes())
    return path


def _request_arguments(case: WorkerCase) -> dict[str, Path]:
    return {
        "audio_path": case.audio_path,
        "model_path": case.model_path,
        "runtime_path": case.runtime_path,
        "identity_path": case.identity_path,
        "blendshape_schema_path": case.schema_path,
    }


def _provider() -> dict[str, str]:
    return {
        "provider_id": "nvidia.audio2face-3d",
        "model_version": "3.0",
        "quality_label": QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
    }


def _control_values(start_frame: int, frame_count: int) -> dict[str, list[list[float]]]:
    frames = np.arange(start_frame, start_frame + frame_count, dtype=np.float64)
    return {
        "skin": np.column_stack((0.5 + 0.1 * np.sin(frames / 5.0), 0.2 + 0.05 * np.cos(frames / 4.0))).tolist(),
        "tongue": (0.1 + 0.02 * np.sin(frames / 3.0))[:, None].tolist(),
        "jaw": np.column_stack((2.0 * np.sin(frames / 6.0), 0.2 * np.cos(frames / 7.0), 0.1 * np.sin(frames / 8.0))).tolist(),
        "eye": np.column_stack((0.5 * np.sin(frames / 9.0), 0.4 * np.cos(frames / 10.0), 0.3 * np.sin(frames / 11.0), 0.2 * np.cos(frames / 12.0))).tolist(),
    }


def _seal_chunk(chunk: dict[str, object], request_sha256: str) -> dict[str, object]:
    output = deepcopy(chunk)
    output.pop("chunk_payload_sha256", None)
    output.pop("execution_chain_out_sha256", None)
    chunk_hash = sequence_chunk_payload_sha256(output)
    output["chunk_payload_sha256"] = chunk_hash
    output["execution_chain_out_sha256"] = sequence_execution_chain_out_sha256(
        request_sha256=request_sha256,
        chunk_index=int(output["chunk_index"]),
        execution_chain_in_sha256=str(output["execution_chain_in_sha256"]),
        chunk_payload_sha256=chunk_hash,
    )
    return output


def _case(tmp_path: Path, *, sample_count: int = 16_000) -> WorkerCase:
    samples = np.round(
        12000.0
        * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_count) / 16000.0)
    ).astype(np.int16)
    audio_path = _write_pcm_wave(tmp_path / "normalized.wav", samples)
    model_path = tmp_path / "a2f-v3.onnx"
    model_path.write_bytes(b"pinned synthetic model artifact for envelope tests\n")
    runtime_path = tmp_path / "runtime-container.digest"
    runtime_path.write_bytes(b"sha256:synthetic-runtime-image\n")
    identity_path = tmp_path / "v3-identity.npz"
    identity_path.write_bytes(b"pinned v3 identity geometry\n")
    schema_path = write_json(
        tmp_path / "blendshape-schema.json",
        {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "skin": ["jawOpen", "mouthClose"],
            "tongue": ["tongueOut"],
            "jaw": ["jawRxDegrees", "jawRyDegrees", "jawRzDegrees"],
            "eye": ["leftEyePitch", "leftEyeYaw", "rightEyePitch", "rightEyeYaw"],
        },
    )
    audio = inspect_bound_pcm_audio(audio_path)
    bindings = {
        "model_sha256": sha256(model_path),
        "runtime_sha256": sha256(runtime_path),
        "identity_sha256": sha256(identity_path),
        "blendshape_schema_sha256": sha256(schema_path),
    }
    frame_count = (sample_count * 60 + 15_999) // 16_000
    output_timebase = {
        "units": "seconds",
        "fps_numerator": 60,
        "fps_denominator": 1,
        "frame_count": frame_count,
        "timestamp_origin_seconds": 0.0,
    }
    chunks: list[dict[str, object]] = []
    for plan in build_official_v3_inference_plan(
        audio,
        SequenceOutputTimebase("seconds", 60, 1, frame_count, 0.0),
    ):
        document = asdict(plan)
        document["output_target_samples"] = list(plan.output_target_samples)
        chunks.append(document)
    request_document = seal_v3_worker_request_document(
        {
            "schema_version": V3_REQUEST_SCHEMA_VERSION,
            "provider": _provider(),
            "bindings": bindings,
            "audio": asdict(audio),
            "output_timebase": output_timebase,
            "chunks": chunks,
        }
    )
    request_path = write_json(tmp_path / "worker-request.json", request_document)
    request = validate_v3_worker_request(
        request_path,
        audio_path=audio_path,
        model_path=model_path,
        runtime_path=runtime_path,
        identity_path=identity_path,
        blendshape_schema_path=schema_path,
    )

    response_chunks: list[dict[str, object]] = []
    chain_in = EXECUTION_CHAIN_ROOT_SHA256
    for plan in chunks:
        start = int(plan["output_start_frame"])
        count = int(plan["output_frame_count"])
        chunk = {
            **plan,
            "execution_chain_in_sha256": chain_in,
            "timestamps_seconds": (np.arange(start, start + count) / 60.0).tolist(),
            "controls": _control_values(start, count),
        }
        sealed_chunk = _seal_chunk(chunk, request.request_sha256)
        response_chunks.append(sealed_chunk)
        chain_in = str(sealed_chunk["execution_chain_out_sha256"])
    response_document = seal_v3_worker_response_document(
        {
            "schema_version": V3_RESPONSE_SCHEMA_VERSION,
            "request_sha256": request.request_sha256,
            "provider": _provider(),
            "bindings": bindings,
            "audio": asdict(audio),
            "output_timebase": output_timebase,
            "control_names": {
                "skin": ["jawOpen", "mouthClose"],
                "tongue": ["tongueOut"],
                "jaw": ["jawRxDegrees", "jawRyDegrees", "jawRzDegrees"],
                "eye": ["leftEyePitch", "leftEyeYaw", "rightEyePitch", "rightEyeYaw"],
            },
            "chunks": response_chunks,
        }
    )
    response_path = write_json(tmp_path / "worker-response.json", response_document)
    return WorkerCase(
        audio_path=audio_path,
        model_path=model_path,
        runtime_path=runtime_path,
        identity_path=identity_path,
        schema_path=schema_path,
        request_document=request_document,
        request_path=request_path,
        request=request,
        response_document=response_document,
        response_path=response_path,
    )


def test_valid_synthetic_sequence_envelope_is_deterministic_and_immutable(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    first_request = validate_v3_worker_request(case.request_path, **_request_arguments(case))
    second_request = validate_v3_worker_request(case.request_path, **_request_arguments(case))
    first = validate_v3_worker_response(case.response_path, request=first_request)
    second = validate_v3_worker_response(case.response_path, request=second_request)

    assert first_request == second_request
    assert first.quality_label == QUALITY_A2F_V3_SEQUENCE_CANDIDATE
    assert first.model_version == "3.0"
    assert first.response_sha256 == second.response_sha256
    assert first.output_timebase == first_request.output_timebase
    np.testing.assert_array_equal(
        first.timestamps, np.arange(60, dtype=np.float64) / 60.0
    )
    np.testing.assert_array_equal(first.skin, second.skin)
    np.testing.assert_array_equal(first.tongue, second.tongue)
    np.testing.assert_array_equal(first.jaw, second.jaw)
    np.testing.assert_array_equal(first.eye, second.eye)
    assert first.skin.shape == (60, 2)
    assert first.tongue.shape == (60, 1)
    assert first.jaw.shape == (60, 3)
    assert first.eye.shape == (60, 4)
    assert not first.timestamps.flags.writeable
    assert not first.skin.flags.writeable
    assert first.chunks[0].execution_chain_in_sha256 == EXECUTION_CHAIN_ROOT_SHA256
    assert first.chunks[0].audio_start_sample == -16000
    assert first.chunks[0].audio_padding_left_samples == 16000
    assert first.chunks[0].output_frame_count == 0
    assert (
        first.chunks[1].execution_chain_in_sha256
        == first.chunks[0].execution_chain_out_sha256
    )
    with pytest.raises(SequenceProviderError) as caught:
        replace(first, quality_label=QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW)
    assert caught.value.code == "INVALID_QUALITY_LABEL"
    with pytest.raises(SequenceProviderError) as caught:
        replace(
            first,
            output_timebase=replace(first.output_timebase, frame_count=61),
        )
    assert caught.value.code == "TIMESTAMP_MISMATCH"
    shortened_chunks = list(first.chunks)
    shortened_chunks[-1] = replace(
        shortened_chunks[-1],
        audio_sample_count=shortened_chunks[-1].audio_sample_count - 1,
    )
    with pytest.raises(SequenceProviderError) as caught:
        replace(first, chunks=tuple(shortened_chunks))
    assert caught.value.code == "CHUNK_MISMATCH"


def test_valid_v2_3_framewise_preview_track_retains_generic_chunk_contract() -> None:
    frame_count = 30
    sample_count = 16_000
    timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
    frames = np.arange(frame_count, dtype=np.float64)
    first_targets = tuple(frame * sample_count // frame_count for frame in range(15))
    second_targets = tuple(
        frame * sample_count // frame_count for frame in range(15, frame_count)
    )
    first = SequenceChunkProvenance(
        chunk_index=0,
        audio_start_sample=0,
        audio_sample_count=12_000,
        audio_overlap_previous_samples=0,
        audio_padding_left_samples=0,
        audio_padding_right_samples=0,
        source_intersection_start_sample=0,
        source_intersection_sample_count=12_000,
        generated_frame_count=15,
        discarded_left_frame_count=0,
        discarded_right_frame_count=0,
        output_start_frame=0,
        output_frame_count=15,
        output_target_samples=first_targets,
        execution_chain_in_sha256=EXECUTION_CHAIN_ROOT_SHA256,
        execution_chain_out_sha256="6" * 64,
        chunk_payload_sha256="7" * 64,
    )
    second = SequenceChunkProvenance(
        chunk_index=1,
        audio_start_sample=8_000,
        audio_sample_count=8_000,
        audio_overlap_previous_samples=4_000,
        audio_padding_left_samples=0,
        audio_padding_right_samples=0,
        source_intersection_start_sample=8_000,
        source_intersection_sample_count=8_000,
        generated_frame_count=15,
        discarded_left_frame_count=0,
        discarded_right_frame_count=0,
        output_start_frame=15,
        output_frame_count=15,
        output_target_samples=second_targets,
        execution_chain_in_sha256=first.execution_chain_out_sha256,
        execution_chain_out_sha256="8" * 64,
        chunk_payload_sha256="9" * 64,
    )

    track = SequenceProviderTrack(
        provider_id="nvidia.audio2face-3d",
        model_version="2.3.1",
        quality_label=QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW,
        bindings=SequenceArtifactBindings("1" * 64, "2" * 64, "3" * 64, "4" * 64),
        source_audio_sha256="5" * 64,
        audio_sample_rate_hz=16_000,
        audio_sample_count=sample_count,
        output_timebase=SequenceOutputTimebase(
            "seconds", 30, 1, frame_count, 0.0
        ),
        timestamps=timestamps,
        control_names=SequenceControlNames(
            skin=("jawOpen", "mouthClose"),
            tongue=("tongueOut",),
            jaw=("jawRxDegrees",),
            eye=("leftEyePitch",),
        ),
        skin=np.column_stack(
            (0.5 + 0.1 * np.sin(frames / 5.0), 0.2 + 0.05 * np.cos(frames / 4.0))
        ),
        tongue=(0.1 + 0.02 * np.sin(frames / 3.0))[:, None],
        jaw=(2.0 * np.sin(frames / 6.0))[:, None],
        eye=(0.5 * np.cos(frames / 9.0))[:, None],
        chunks=(first, second),
        request_sha256="a" * 64,
        response_sha256="b" * 64,
    )

    assert track.quality_label == QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW
    assert track.model_version == "2.3.1"
    assert track.output_timebase.fps == 30.0
    np.testing.assert_array_equal(track.timestamps, timestamps)
    assert not track.skin.flags.writeable
    with pytest.raises(SequenceProviderError) as caught:
        replace(
            track,
            chunks=(replace(first, audio_padding_left_samples=1), second),
        )
    assert caught.value.code == "CHUNK_MISMATCH"


def test_one_frame_official_v3_envelope_roundtrips_with_warmup_execution(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, sample_count=1)

    track = validate_v3_worker_response(case.response_path, request=case.request)

    assert track.output_timebase.frame_count == 1
    np.testing.assert_array_equal(track.timestamps, np.asarray((0.0,)))
    assert tuple(chunk.output_frame_count for chunk in track.chunks) == (0, 1)
    assert track.chunks[0].execution_chain_in_sha256 == EXECUTION_CHAIN_ROOT_SHA256
    assert (
        track.chunks[1].execution_chain_in_sha256
        == track.chunks[0].execution_chain_out_sha256
    )


def test_mac_preflight_fails_closed_without_claiming_a_local_v3_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("autoanim_gnm.sequence_provider.platform.system", lambda: "Darwin")
    monkeypatch.setattr("autoanim_gnm.sequence_provider.platform.machine", lambda: "arm64")

    report = local_a2f_v3_worker_preflight()

    assert report.system == "Darwin"
    assert report.machine == "arm64"
    assert not report.validated_local_runtime
    assert not report.can_execute_locally
    assert report.blocker_code == "NVIDIA_V3_EXTERNAL_WORKER_REQUIRED"
    assert "Linux or Windows NVIDIA GPU worker" in report.blocker
    assert report.quality_label == QUALITY_A2F_V3_SEQUENCE_CANDIDATE


@pytest.mark.parametrize(
    ("sample_count", "frame_count", "emitted_counts"),
    (
        (1, 1, (0, 1)),
        (266, 1, (0, 1)),
        (267, 2, (0, 2)),
        (3_999, 15, (0, 15)),
        (4_000, 15, (0, 15)),
        (4_001, 16, (0, 15, 1)),
        (15_999, 60, (0, 15, 30, 15)),
        (16_000, 60, (0, 15, 30, 15)),
        (16_001, 61, (0, 15, 30, 16)),
    ),
)
def test_official_v3_plan_preserves_signed_padding_partial_blocks_and_sample_ticks(
    sample_count: int,
    frame_count: int,
    emitted_counts: tuple[int, ...],
) -> None:
    audio = AudioSampleClock("a" * 64, "b" * 64, 16_000, sample_count, 1, 2)
    timebase = SequenceOutputTimebase("seconds", 60, 1, frame_count, 0.0)

    plan = build_official_v3_inference_plan(audio, timebase)

    assert tuple(chunk.output_frame_count for chunk in plan) == emitted_counts
    assert plan[0].audio_start_sample == -16_000
    assert plan[0].audio_padding_left_samples == 16_000
    assert plan[0].output_frame_count == 0
    assert all(chunk.generated_frame_count == 60 for chunk in plan)
    assert all(chunk.discarded_left_frame_count == 15 for chunk in plan)
    assert all(chunk.discarded_right_frame_count == 15 for chunk in plan)
    target_samples = tuple(
        target for chunk in plan for target in chunk.output_target_samples
    )
    assert target_samples == tuple(
        frame * 16_000 // 60 for frame in range(frame_count)
    )
    for chunk in plan:
        assert (
            chunk.audio_padding_left_samples
            + chunk.source_intersection_sample_count
            + chunk.audio_padding_right_samples
            == 16_000
        )
        assert chunk.source_intersection_start_sample == max(
            0, chunk.audio_start_sample
        )


def test_official_v3_plan_rejects_direct_caller_clock_and_pcm_substitution() -> None:
    audio = AudioSampleClock("a" * 64, "b" * 64, 16_000, 16_000, 1, 2)
    timebase = SequenceOutputTimebase("seconds", 60, 1, 60, 0.0)

    for invalid_audio, invalid_timebase in (
        (replace(audio, channel_count=2), timebase),
        (replace(audio, sample_width_bytes=4), timebase),
        (audio, replace(timebase, frame_count=59)),
        (audio, replace(timebase, units="milliseconds")),
        (audio, replace(timebase, timestamp_origin_seconds=1.0)),
    ):
        with pytest.raises(SequenceProviderError) as caught:
            build_official_v3_inference_plan(invalid_audio, invalid_timebase)
        assert caught.value.code == "MODEL_CLOCK_MISMATCH"


@pytest.mark.parametrize(
    ("field", "argument"),
    (
        ("model", "model_path"),
        ("runtime", "runtime_path"),
        ("identity", "identity_path"),
        ("blendshape_schema", "blendshape_schema_path"),
    ),
)
def test_request_rejects_model_runtime_identity_and_schema_artifact_substitution(
    tmp_path: Path,
    field: str,
    argument: str,
) -> None:
    case = _case(tmp_path)
    substitute = tmp_path / f"substitute-{field}.bin"
    substitute.write_bytes(b"substituted artifact\n")
    arguments = _request_arguments(case)
    arguments[argument] = substitute

    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(case.request_path, **arguments)

    assert caught.value.code == "BINDING_MISMATCH"
    assert caught.value.field == field


def test_request_rejects_audio_artifact_and_sample_clock_substitution(tmp_path: Path) -> None:
    case = _case(tmp_path)
    other_samples = np.zeros(16000, dtype=np.int16)
    substitute = _write_pcm_wave(tmp_path / "substitute.wav", other_samples)
    arguments = _request_arguments(case)
    arguments["audio_path"] = substitute
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(case.request_path, **arguments)
    assert caught.value.code == "AUDIO_BINDING_MISMATCH"


def test_request_rejects_legacy_30_fps_v3_clock_and_schema(tmp_path: Path) -> None:
    case = _case(tmp_path)
    legacy_clock = deepcopy(case.request_document)
    legacy_clock["output_timebase"]["fps_numerator"] = 30  # type: ignore[index]
    legacy_clock["output_timebase"]["frame_count"] = 30  # type: ignore[index]
    legacy_clock["chunks"][0]["output_frame_count"] = 15  # type: ignore[index]
    legacy_clock["chunks"][1]["output_start_frame"] = 15  # type: ignore[index]
    legacy_clock["chunks"][1]["output_frame_count"] = 15  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(legacy_clock),
            **_request_arguments(case),
        )
    assert caught.value.code == "MODEL_CLOCK_MISMATCH"

    legacy_schema = deepcopy(case.request_document)
    legacy_schema["schema_version"] = "autoanim.a2f-v3-worker-request/1.0"
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(legacy_schema),
            **_request_arguments(case),
        )
    assert caught.value.code == "UNSUPPORTED_SCHEMA"

    clock_tamper = deepcopy(case.request_document)
    clock_tamper["audio"]["sample_count"] = 15999  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(clock_tamper),
            **_request_arguments(case),
        )
    assert caught.value.code == "AUDIO_BINDING_MISMATCH"


def test_request_rejects_hash_tampering_and_v2_preview_relabel(tmp_path: Path) -> None:
    case = _case(tmp_path)
    tampered = deepcopy(case.request_document)
    tampered["bindings"]["model_sha256"] = "f" * 64  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(tampered, **_request_arguments(case))
    assert caught.value.code == "REQUEST_HASH_MISMATCH"

    relabeled = deepcopy(case.request_document)
    relabeled["provider"]["quality_label"] = QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(relabeled),
            **_request_arguments(case),
        )
    assert caught.value.code == "PROVIDER_SUBSTITUTION"


def test_request_rejects_missing_chunk_bad_overlap_and_duration(tmp_path: Path) -> None:
    case = _case(tmp_path)
    missing = deepcopy(case.request_document)
    missing["chunks"] = []
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(missing), **_request_arguments(case)
        )
    assert caught.value.code == "CHUNK_PLAN_INVALID"

    overlap = deepcopy(case.request_document)
    overlap["chunks"][1]["audio_overlap_previous_samples"] = 3999  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(overlap), **_request_arguments(case)
        )
    assert caught.value.code == "CHUNK_PLAN_INVALID"

    duration = deepcopy(case.request_document)
    duration["output_timebase"]["frame_count"] = 59  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(duration), **_request_arguments(case)
        )
    assert caught.value.code == "DURATION_MISMATCH"


def test_response_rejects_missing_chunk_and_missing_execution_chain(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    missing_chunk = deepcopy(case.response_document)
    missing_chunk["chunks"] = missing_chunk["chunks"][:-1]  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(missing_chunk), request=case.request
        )
    assert caught.value.code == "CHUNK_MISMATCH"

    missing_execution_chain = deepcopy(case.response_document)
    del missing_execution_chain["chunks"][0]["execution_chain_in_sha256"]  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(missing_execution_chain),
            request=case.request,
        )
    assert caught.value.code == "INVALID_ENVELOPE"


def test_response_rejects_legacy_schema_and_substituted_output_clock(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    legacy = deepcopy(case.response_document)
    legacy["schema_version"] = "autoanim.a2f-v3-worker-response/1.0"
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(legacy), request=case.request
        )
    assert caught.value.code == "UNSUPPORTED_SCHEMA"

    substituted = deepcopy(case.response_document)
    substituted["output_timebase"]["fps_numerator"] = 30  # type: ignore[index]
    substituted["output_timebase"]["frame_count"] = 30  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(substituted), request=case.request
        )
    assert caught.value.code == "RESPONSE_BINDING_MISMATCH"


def test_response_rejects_resealed_execution_chain_substitution(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    response = deepcopy(case.response_document)
    second = response["chunks"][1]  # type: ignore[index]
    second["execution_chain_in_sha256"] = "e" * 64
    response["chunks"][1] = _seal_chunk(second, case.request.request_sha256)  # type: ignore[index]

    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(response), request=case.request
        )

    assert caught.value.code == "EXECUTION_CHAIN_MISMATCH"


def test_response_rejects_timestamp_duration_and_control_shape_errors(tmp_path: Path) -> None:
    case = _case(tmp_path)
    timestamp = deepcopy(case.response_document)
    last = timestamp["chunks"][1]  # type: ignore[index]
    last["timestamps_seconds"][0] += 0.01
    timestamp["chunks"][1] = _seal_chunk(last, case.request.request_sha256)  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(timestamp), request=case.request
        )
    assert caught.value.code == "TIMESTAMP_MISMATCH"

    controls = deepcopy(case.response_document)
    last = controls["chunks"][1]  # type: ignore[index]
    last["controls"]["jaw"] = last["controls"]["jaw"][:-1]
    controls["chunks"][1] = _seal_chunk(last, case.request.request_sha256)  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(controls), request=case.request
        )
    assert caught.value.code == "INVALID_CONTROLS"


def test_response_rejects_nonfinite_or_unnamed_controls_and_artifact_substitution(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    nonnumeric = deepcopy(case.response_document)
    last = nonnumeric["chunks"][1]  # type: ignore[index]
    last["controls"]["skin"][0][0] = None
    nonnumeric["chunks"][1] = _seal_chunk(last, case.request.request_sha256)  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(nonnumeric), request=case.request
        )
    assert caught.value.code == "INVALID_CONTROLS"

    unnamed = deepcopy(case.response_document)
    unnamed["control_names"]["skin"][0] = "differentJawOpen"  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(unnamed), request=case.request
        )
    assert caught.value.code == "RESPONSE_BINDING_MISMATCH"

    substitution = deepcopy(case.response_document)
    substitution["bindings"]["runtime_sha256"] = "d" * 64  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(substitution), request=case.request
        )
    assert caught.value.code == "RESPONSE_BINDING_MISMATCH"


def test_response_rejects_chunk_payload_and_execution_chain_output_tampering(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    payload = deepcopy(case.response_document)
    payload["chunks"][1]["controls"]["eye"][0][0] += 0.1  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(payload), request=case.request
        )
    assert caught.value.code == "CHUNK_HASH_MISMATCH"

    state = deepcopy(case.response_document)
    state["chunks"][1]["execution_chain_out_sha256"] = "c" * 64  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(state), request=case.request
        )
    assert caught.value.code == "EXECUTION_CHAIN_MISMATCH"
