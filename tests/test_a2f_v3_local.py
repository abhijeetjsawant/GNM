from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading
import wave

import numpy as np
import pytest

import autoanim_gnm.a2f_v3_local as local_v3
from autoanim_gnm.a2f_v3_local import (
    LOCAL_V3_OUTPUT_WIDTH,
    QUALITY_A2F_V3_LOCAL_RAW_CANDIDATE,
    consume_local_v3_raw,
    run_local_v3_raw,
)
from autoanim_gnm.a2f_v3_profile import OfficialV3ClaireProfile
from autoanim_gnm.sequence_provider import (
    A2F_V3_MODEL_REVISION,
    A2F_V3_SDK_REVISION,
    SequenceProviderError,
)


class FakeNode:
    def __init__(self, name: str, shape: list[object]):
        self.name = name
        self.type = "tensor(float)"
        self.shape = shape


def _inputs() -> list[FakeNode]:
    return [
        FakeNode("window", ["batch", 16000]),
        FakeNode("identity", ["batch", 3]),
        FakeNode("emotion", ["batch", 30, 10]),
        FakeNode("input_latents", [2, 2, "batch", 256]),
        FakeNode("noise", ["batch", 3, 60, 88831]),
    ]


def _outputs() -> list[FakeNode]:
    return [
        FakeNode("prediction", ["batch", 60, 88831]),
        FakeNode("output_latents", [2, 2, "batch", 256]),
    ]


class FakeSession:
    def __init__(self, *, inputs: list[FakeNode] | None = None):
        self.inputs = inputs or _inputs()
        self.calls: list[dict[str, np.ndarray]] = []
        self._autoanim_runtime_name = "fake-session"
        self._autoanim_runtime_version = "fake-1"
        self._autoanim_available_providers = ("CPUExecutionProvider",)

    def get_inputs(self) -> list[FakeNode]:
        return self.inputs

    def get_outputs(self) -> list[FakeNode]:
        return _outputs()

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["prediction", "output_latents"]
        self.calls.append({name: value.copy() for name, value in input_feed.items()})
        value = np.float32(len(self.calls))
        prediction = np.full((1, 60, LOCAL_V3_OUTPUT_WIDTH), value, dtype=np.float32)
        return [prediction, input_feed["input_latents"] + np.float32(1.0)]


def _write_wave(path: Path, samples: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return path


def _patch_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "profile"
    root.mkdir()
    network = root / "network.onnx"
    network.write_bytes(b"fake onnx for injected-session tests")
    network_sha = hashlib.sha256(network.read_bytes()).hexdigest()
    profile = OfficialV3ClaireProfile(
        root=root,
        public_model_version="3.0",
        network_version="3.2",
        identity="Claire",
        identity_index=0,
        model_revision=A2F_V3_MODEL_REVISION,
        required_sdk_revision=A2F_V3_SDK_REVISION,
        skin_pose_names=tuple(f"skin{index}" for index in range(52)),
        tongue_pose_names=tuple(f"tongue{index}" for index in range(16)),
        skin_minimums=(0.0,) * 52,
        skin_maximums=(1.0,) * 52,
        tongue_minimums=(0.0,) * 16,
        tongue_maximums=(1.0,) * 16,
        interpretation_asset_sha256={"network_info.json": "1" * 64},
        network_sha256=network_sha,
    )

    def load(directory: str | Path, *, verify_network: bool = False):
        assert Path(directory) == root
        assert verify_network is True
        return profile

    monkeypatch.setattr(local_v3, "load_official_v3_claire_profile", load)
    return root


def test_fake_session_executes_warmup_chains_state_and_retains_exact_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _patch_profile(monkeypatch, tmp_path)
    audio = _write_wave(tmp_path / "one-sample.wav", np.array([16384], dtype=np.int16))
    sessions: list[FakeSession] = []

    def factory(model_path: Path, providers: tuple[str, ...]) -> FakeSession:
        assert model_path == profile / "network.onnx"
        assert providers == ("CPUExecutionProvider",)
        session = FakeSession()
        sessions.append(session)
        return session

    first = run_local_v3_raw(
        audio, profile, max_collect_frames=1, noise_seed=1729, session_factory=factory
    )
    observed_chunks = []

    def observe(chunk: local_v3.LocalV3RawChunk) -> None:
        assert local_v3._PROCESS_INFERENCE_LOCK.locked()
        observed_chunks.append(chunk)

    second = consume_local_v3_raw(
        audio,
        profile,
        observe,
        noise_seed=1729,
        emotion_vector=[0.25] + [0.0] * 9,
        session_factory=factory,
    )

    assert first.execution.evidence.quality_label == QUALITY_A2F_V3_LOCAL_RAW_CANDIDATE
    assert first.execution.evidence.production_qualified is False
    assert first.execution.evidence.sdk_runtime_parity_verified is False
    assert first.execution.evidence.runtime_name == "fake-session"
    assert first.execution.evidence.runtime_version == "fake-1"
    assert first.execution.evidence.active_providers == ("CPUExecutionProvider",)
    assert first.target_samples == (0,)
    assert first.timestamps_seconds.tolist() == [0.0]
    assert first.prediction.shape == (1, LOCAL_V3_OUTPUT_WIDTH)
    assert np.all(first.prediction == 2.0)
    assert first.prediction.flags.writeable is False

    # Chunk 0 emits nothing but still executes and advances recurrent state.
    assert [chunk.emitted_frame_count for chunk in first.execution.evidence.chunks] == [0, 1]
    assert len(sessions[0].calls) == 2
    assert np.all(sessions[0].calls[0]["input_latents"] == 0.0)
    assert np.all(sessions[0].calls[1]["input_latents"] == 1.0)
    assert np.count_nonzero(sessions[0].calls[0]["window"]) == 0
    assert sessions[0].calls[1]["window"][0, 8000] == pytest.approx(0.5)
    assert np.count_nonzero(sessions[0].calls[1]["window"]) == 1

    first_noise = [chunk.noise_sha256 for chunk in first.execution.evidence.chunks]
    second_noise = [chunk.noise_sha256 for chunk in second.evidence.chunks]
    assert first_noise == second_noise
    assert first_noise[0] != first_noise[1]
    assert first.execution.evidence.emotion_vector == (0.0,) * 10
    assert second.evidence.emotion_vector == (0.25,) + (0.0,) * 9
    assert (
        first.execution.evidence.chunks[0].emotion_sha256
        != second.evidence.chunks[0].emotion_sha256
    )
    assert [chunk.plan.output_frame_count for chunk in observed_chunks] == [0, 1]
    assert np.all(sessions[1].calls[0]["emotion"][0, :, 0] == 0.25)
    assert first.execution.evidence.chunks[0].output_latents_sha256 == (
        first.execution.evidence.chunks[1].input_latents_sha256
    )


def test_model_signature_and_collection_bound_fail_closed_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _patch_profile(monkeypatch, tmp_path)
    audio = _write_wave(tmp_path / "audio.wav", np.zeros(16_000, dtype=np.int16))
    factory_called = False

    def must_not_run(model_path: Path, providers: tuple[str, ...]) -> FakeSession:
        nonlocal factory_called
        factory_called = True
        return FakeSession()

    with pytest.raises(SequenceProviderError) as limit:
        run_local_v3_raw(
            audio,
            profile,
            max_collect_frames=59,
            session_factory=must_not_run,
        )
    assert limit.value.code == "LOCAL_V3_COLLECTION_LIMIT"
    assert factory_called is False

    bad = _inputs()
    bad[4] = FakeNode("noise", ["batch", 3, 60, 88830])
    session = FakeSession(inputs=bad)
    one_sample = _write_wave(tmp_path / "short.wav", np.zeros(1, dtype=np.int16))
    with pytest.raises(SequenceProviderError) as signature:
        run_local_v3_raw(
            one_sample,
            profile,
            max_collect_frames=1,
            session_factory=lambda model, providers: session,
        )
    assert signature.value.code == "LOCAL_V3_MODEL_SIGNATURE_MISMATCH"
    assert session.calls == []

    for invalid in ([0.0] * 9, [0.0] * 9 + [float("nan")]):
        with pytest.raises(SequenceProviderError) as emotion:
            run_local_v3_raw(
                one_sample,
                profile,
                max_collect_frames=1,
                emotion_vector=invalid,
                session_factory=lambda model, providers: session,
            )
        assert emotion.value.code == "LOCAL_V3_INVALID_EMOTION"


class LightweightSession(FakeSession):
    """Signature-faithful fake whose runtime payload follows patched test widths."""

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["prediction", "output_latents"]
        self.calls.append({})
        prediction = np.zeros(
            (1, 60, local_v3.LOCAL_V3_OUTPUT_WIDTH), dtype=np.float32
        )
        return [prediction, input_feed["input_latents"] + np.float32(1.0)]


def test_process_single_flight_serializes_concurrent_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _patch_profile(monkeypatch, tmp_path)
    audio = _write_wave(tmp_path / "concurrent.wav", np.zeros(1, dtype=np.int16))
    monkeypatch.setattr(local_v3, "LOCAL_V3_OUTPUT_WIDTH", 1)
    monkeypatch.setattr(local_v3, "LOCAL_V3_NOISE_SHAPE", (1, 1))

    state_lock = threading.Lock()
    first_inference_entered = threading.Event()
    allow_first_inference_to_finish = threading.Event()
    second_factory_entered = threading.Event()
    factory_calls = 0
    active_inferences = 0
    max_active_inferences = 0

    class BlockingSession(LightweightSession):
        def run(
            self, output_names: list[str], input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            nonlocal active_inferences, max_active_inferences
            with state_lock:
                active_inferences += 1
                max_active_inferences = max(max_active_inferences, active_inferences)
                should_block = not first_inference_entered.is_set()
                if should_block:
                    first_inference_entered.set()
            try:
                if should_block:
                    assert allow_first_inference_to_finish.wait(timeout=5.0)
                return super().run(output_names, input_feed)
            finally:
                with state_lock:
                    active_inferences -= 1

    def factory(model_path: Path, providers: tuple[str, ...]) -> BlockingSession:
        nonlocal factory_calls
        with state_lock:
            factory_calls += 1
            if factory_calls == 2:
                second_factory_entered.set()
        return BlockingSession()

    failures: list[BaseException] = []
    callback_counts = [0, 0]

    def worker(index: int) -> None:
        try:
            def consume(chunk: local_v3.LocalV3RawChunk) -> None:
                callback_counts[index] += 1

            consume_local_v3_raw(
                audio,
                profile,
                consume,
                session_factory=factory,
            )
        except BaseException as exc:  # Preserve thread failures for the test thread.
            failures.append(exc)

    first = threading.Thread(target=worker, args=(0,), daemon=True)
    second = threading.Thread(target=worker, args=(1,), daemon=True)
    first.start()
    assert first_inference_entered.wait(timeout=5.0)
    second.start()

    # The second request has begun, but session construction is also protected
    # by the same process lock held by the first blocked inference.
    assert not second_factory_entered.wait(timeout=0.15)
    with state_lock:
        assert factory_calls == 1
        assert active_inferences == 1
        assert max_active_inferences == 1

    allow_first_inference_to_finish.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert second_factory_entered.is_set()
    assert factory_calls == 2
    assert max_active_inferences == 1
    assert callback_counts == [2, 2]


def test_ten_minute_schedule_streams_exact_frames_without_raw_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _patch_profile(monkeypatch, tmp_path)
    sample_count = 600 * 16_000
    audio = _write_wave(
        tmp_path / "ten-minutes.wav", np.zeros(sample_count, dtype=np.int16)
    )
    # Preserve the actual audio clock, official scheduling, window extraction,
    # state chaining, and callback cadence. Only large fake model payloads are
    # reduced; the signature still advertises the exact pinned 88,831-wide ABI.
    monkeypatch.setattr(local_v3, "LOCAL_V3_OUTPUT_WIDTH", 1)
    monkeypatch.setattr(local_v3, "LOCAL_V3_NOISE_SHAPE", (1, 1))
    session = LightweightSession()
    callback_count = 0
    emitted_frames = 0
    maximum_chunk_frames = 0
    retained_payload_bytes = 0

    def consume(chunk: local_v3.LocalV3RawChunk) -> None:
        nonlocal callback_count, emitted_frames, maximum_chunk_frames
        nonlocal retained_payload_bytes
        callback_count += 1
        emitted_frames += chunk.plan.output_frame_count
        maximum_chunk_frames = max(maximum_chunk_frames, len(chunk.prediction))
        # Deliberately retain only scalar accounting, never a chunk/array.
        retained_payload_bytes += chunk.prediction.nbytes

    execution = consume_local_v3_raw(
        audio,
        profile,
        consume,
        noise_seed=600,
        session_factory=lambda model, providers: session,
    )

    assert execution.audio.sample_count == sample_count
    assert execution.output_timebase.frame_count == 36_000
    assert emitted_frames == 36_000
    assert callback_count == 1_202
    assert len(execution.evidence.chunks) == callback_count
    assert len(session.calls) == callback_count
    assert maximum_chunk_frames == 30
    assert retained_payload_bytes == 36_000 * np.dtype(np.float32).itemsize
    assert execution.evidence.chunks[0].emitted_frame_count == 0
    assert execution.evidence.chunks[-1].emitted_frame_count == 15


REAL_PROFILE = Path(".cache/autoanim_gnm/a2f-v3-claire-profile")
RUN_REAL = os.environ.get("AUTOANIM_RUN_A2F_V3_LOCAL_REAL") == "1"


@pytest.mark.skipif(
    not RUN_REAL or not (REAL_PROFILE / "network.onnx").is_file(),
    reason="set AUTOANIM_RUN_A2F_V3_LOCAL_REAL=1 with the cached public v3 profile",
)
def test_cached_real_model_runs_one_retained_frame(tmp_path: Path) -> None:
    audio = _write_wave(tmp_path / "one-sample.wav", np.zeros(1, dtype=np.int16))
    result = run_local_v3_raw(
        audio,
        REAL_PROFILE,
        max_collect_frames=1,
        noise_seed=20260720,
    )
    assert result.prediction.shape == (1, LOCAL_V3_OUTPUT_WIDTH)
    assert np.isfinite(result.prediction).all()
    assert result.execution.evidence.runtime_name == "onnxruntime"
    assert result.execution.evidence.model_sha256 == (
        "db47c2701ca849de443c9e9f25657210f829a74fc458ee6fed603a8a501253a8"
    )
    assert result.execution.evidence.production_qualified is False
    assert result.execution.evidence.sdk_runtime_parity_verified is False
