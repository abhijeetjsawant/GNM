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
    ZERO_STATE_SHA256,
    SequenceProviderError,
    inspect_bound_pcm_audio,
    local_a2f_v3_worker_preflight,
    seal_v3_worker_request_document,
    seal_v3_worker_response_document,
    sequence_chunk_payload_sha256,
    sequence_state_out_sha256,
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
    output.pop("state_out_sha256", None)
    chunk_hash = sequence_chunk_payload_sha256(output)
    output["chunk_payload_sha256"] = chunk_hash
    output["state_out_sha256"] = sequence_state_out_sha256(
        request_sha256=request_sha256,
        chunk_index=int(output["chunk_index"]),
        state_in_sha256=str(output["state_in_sha256"]),
        chunk_payload_sha256=chunk_hash,
    )
    return output


def _case(tmp_path: Path) -> WorkerCase:
    samples = np.round(
        12000.0 * np.sin(2.0 * np.pi * 220.0 * np.arange(16000) / 16000.0)
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
    chunks = [
        {
            "chunk_index": 0,
            "audio_start_sample": 0,
            "audio_sample_count": 12000,
            "audio_overlap_previous_samples": 0,
            "output_start_frame": 0,
            "output_frame_count": 15,
        },
        {
            "chunk_index": 1,
            "audio_start_sample": 8000,
            "audio_sample_count": 8000,
            "audio_overlap_previous_samples": 4000,
            "output_start_frame": 15,
            "output_frame_count": 15,
        },
    ]
    output_timebase = {
        "units": "seconds",
        "fps_numerator": 30,
        "fps_denominator": 1,
        "frame_count": 30,
        "timestamp_origin_seconds": 0.0,
    }
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
    state_in = ZERO_STATE_SHA256
    for plan in chunks:
        start = int(plan["output_start_frame"])
        count = int(plan["output_frame_count"])
        chunk = {
            **plan,
            "state_in_sha256": state_in,
            "timestamps_seconds": (np.arange(start, start + count) / 30.0).tolist(),
            "controls": _control_values(start, count),
        }
        sealed_chunk = _seal_chunk(chunk, request.request_sha256)
        response_chunks.append(sealed_chunk)
        state_in = str(sealed_chunk["state_out_sha256"])
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
        first.timestamps, np.arange(30, dtype=np.float64) / 30.0
    )
    np.testing.assert_array_equal(first.skin, second.skin)
    np.testing.assert_array_equal(first.tongue, second.tongue)
    np.testing.assert_array_equal(first.jaw, second.jaw)
    np.testing.assert_array_equal(first.eye, second.eye)
    assert first.skin.shape == (30, 2)
    assert first.tongue.shape == (30, 1)
    assert first.jaw.shape == (30, 3)
    assert first.eye.shape == (30, 4)
    assert not first.timestamps.flags.writeable
    assert not first.skin.flags.writeable
    assert first.chunks[0].state_in_sha256 == ZERO_STATE_SHA256
    assert first.chunks[1].state_in_sha256 == first.chunks[0].state_out_sha256
    with pytest.raises(SequenceProviderError) as caught:
        replace(first, quality_label=QUALITY_A2F_V2_3_FRAMEWISE_PREVIEW)
    assert caught.value.code == "INVALID_QUALITY_LABEL"
    with pytest.raises(SequenceProviderError) as caught:
        replace(
            first,
            output_timebase=replace(first.output_timebase, frame_count=31),
        )
    assert caught.value.code == "TIMESTAMP_MISMATCH"
    shortened_chunks = list(first.chunks)
    shortened_chunks[-1] = replace(
        shortened_chunks[-1],
        audio_sample_count=shortened_chunks[-1].audio_sample_count - 1,
    )
    with pytest.raises(SequenceProviderError) as caught:
        replace(first, chunks=tuple(shortened_chunks))
    assert caught.value.code == "DURATION_MISMATCH"


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
    duration["output_timebase"]["frame_count"] = 29  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_request(
            seal_v3_worker_request_document(duration), **_request_arguments(case)
        )
    assert caught.value.code == "DURATION_MISMATCH"


def test_response_rejects_missing_chunk_and_missing_state(tmp_path: Path) -> None:
    case = _case(tmp_path)
    missing_chunk = deepcopy(case.response_document)
    missing_chunk["chunks"] = missing_chunk["chunks"][:-1]  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(missing_chunk), request=case.request
        )
    assert caught.value.code == "CHUNK_MISMATCH"

    missing_state = deepcopy(case.response_document)
    del missing_state["chunks"][0]["state_in_sha256"]  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(missing_state), request=case.request
        )
    assert caught.value.code == "INVALID_ENVELOPE"


def test_response_rejects_resealed_state_chain_substitution(tmp_path: Path) -> None:
    case = _case(tmp_path)
    response = deepcopy(case.response_document)
    second = response["chunks"][1]  # type: ignore[index]
    second["state_in_sha256"] = "e" * 64
    response["chunks"][1] = _seal_chunk(second, case.request.request_sha256)  # type: ignore[index]

    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(response), request=case.request
        )

    assert caught.value.code == "STATE_CHAIN_MISMATCH"


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


def test_response_rejects_chunk_payload_and_state_output_tampering(tmp_path: Path) -> None:
    case = _case(tmp_path)
    payload = deepcopy(case.response_document)
    payload["chunks"][1]["controls"]["eye"][0][0] += 0.1  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(payload), request=case.request
        )
    assert caught.value.code == "CHUNK_HASH_MISMATCH"

    state = deepcopy(case.response_document)
    state["chunks"][1]["state_out_sha256"] = "c" * 64  # type: ignore[index]
    with pytest.raises(SequenceProviderError) as caught:
        validate_v3_worker_response(
            seal_v3_worker_response_document(state), request=case.request
        )
    assert caught.value.code == "STATE_CHAIN_MISMATCH"
