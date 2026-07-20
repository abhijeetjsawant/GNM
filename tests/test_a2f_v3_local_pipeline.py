"""Focused contract tests for the local Audio2Face-v3 pipeline boundary.

These tests intentionally keep the 691 MB ONNX model out of the fast suite.
The runtime and postprocessor have their own numerical tests; this module owns
pipeline selection, input validation, evidence honesty, and artifact wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from autoanim_gnm.a2f_v3_local import LOCAL_V3_EMOTION_NAMES
from autoanim_gnm.a2f_v3_postprocess import ClaireV3PostprocessChunk
from autoanim_gnm.audio import EmotionAnalysis, ProsodyTrack
from autoanim_gnm.audio_pipeline import (
    _local_v3_emotion_vector,
    _local_v3_evidence_document,
    run_audio_pipeline,
)
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.sequence_provider import AudioSampleClock, SequenceOutputTimebase


class _ReachedMedia(RuntimeError):
    """Sentinel proving that argument validation accepted the request."""


def _stop_at_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reached_media(*_args: object, **_kwargs: object) -> float:
        raise _ReachedMedia

    monkeypatch.setattr("autoanim_gnm.audio_pipeline.normalize_audio", reached_media)


def test_local_v3_requires_native_60_fps_before_media_work(tmp_path: Path) -> None:
    with pytest.raises(AutoAnimError, match="60") as error:
        run_audio_pipeline(
            tmp_path / "not-read.wav",
            tmp_path / "output",
            fps=30,
            backend="a2f-v3-local",
        )
    assert error.value.code == "INPUT_INVALID"


def test_local_v3_application_caps_full_track_materialization_at_ten_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.normalize_audio",
        lambda *_args, **_kwargs: 10.001,
    )
    with pytest.raises(AutoAnimError, match="10 seconds|full track") as error:
        run_audio_pipeline(
            tmp_path / "source.wav",
            tmp_path / "output",
            fps=60,
            backend="a2f-v3-local",
        )
    assert error.value.code == "LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    "binding_name",
    (
        "a2f_v3_request_path",
        "a2f_v3_response_path",
        "a2f_v3_model_path",
        "a2f_v3_runtime_path",
        "a2f_v3_identity_path",
        "a2f_v3_schema_path",
    ),
)
def test_local_v3_rejects_external_worker_bindings(
    tmp_path: Path,
    binding_name: str,
) -> None:
    with pytest.raises(AutoAnimError, match="external|binding|import") as error:
        run_audio_pipeline(
            tmp_path / "not-read.wav",
            tmp_path / "output",
            fps=60,
            backend="a2f-v3-local",
            **{binding_name: tmp_path / "untrusted-binding"},
        )
    assert error.value.code == "INPUT_INVALID"


def test_local_v3_profile_is_optional_and_seed_zero_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stop_at_media(monkeypatch)
    with pytest.raises(_ReachedMedia):
        run_audio_pipeline(
            tmp_path / "source.wav",
            tmp_path / "output",
            fps=60,
            backend="a2f-v3-local",
            a2f_v3_local_seed=0,
        )


def test_local_v3_explicit_profile_and_maximum_uint64_seed_are_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stop_at_media(monkeypatch)
    with pytest.raises(_ReachedMedia):
        run_audio_pipeline(
            tmp_path / "source.wav",
            tmp_path / "output",
            fps=60,
            backend="a2f-v3-local",
            a2f_v3_profile_dir=tmp_path / "profile",
            a2f_v3_local_seed=(1 << 64) - 1,
        )


@pytest.mark.parametrize("seed", (-1, 1 << 64, True, 1.5, "1"))
def test_local_v3_seed_must_be_a_uint64(
    tmp_path: Path,
    seed: object,
) -> None:
    with pytest.raises(AutoAnimError, match="seed|uint64") as error:
        run_audio_pipeline(
            tmp_path / "not-read.wav",
            tmp_path / "output",
            fps=60,
            backend="a2f-v3-local",
            a2f_v3_local_seed=seed,  # type: ignore[arg-type]
        )
    assert error.value.code == "INPUT_INVALID"


def test_local_v3_profile_is_not_an_external_worker_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the profile is valid locally but forbidden on fallbacks."""

    _stop_at_media(monkeypatch)
    with pytest.raises(_ReachedMedia):
        run_audio_pipeline(
            tmp_path / "source.wav",
            tmp_path / "output",
            fps=60,
            backend="a2f-v3-local",
            a2f_v3_profile_dir=tmp_path / "profile",
        )

    with pytest.raises(AutoAnimError, match="backend"):
        run_audio_pipeline(
            tmp_path / "not-read.wav",
            tmp_path / "fallback-output",
            backend="fallback",
            a2f_v3_profile_dir=tmp_path / "profile",
        )


@pytest.mark.parametrize(
    ("autoanim_emotion", "v3_emotion"),
    (
        ("surprise", "amazement"),
        ("anger", "anger"),
        ("contempt", "cheekiness"),
        ("disgust", "disgust"),
        ("fear", "fear"),
        ("joy", "joy"),
        ("sad", "sadness"),
    ),
)
def test_validated_emotion_maps_to_exact_public_v3_channel_order(
    autoanim_emotion: str,
    v3_emotion: str,
) -> None:
    vector = _local_v3_emotion_vector(
        autoanim_emotion,
        validated=True,
        strength=0.625,
    )
    assert tuple(LOCAL_V3_EMOTION_NAMES) == (
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
    expected = np.zeros(10, dtype=np.float32)
    expected[LOCAL_V3_EMOTION_NAMES.index(v3_emotion)] = np.float32(0.625)
    np.testing.assert_array_equal(vector, expected)


@pytest.mark.parametrize(
    "emotion",
    ("neutral", "surprise", "anger", "contempt", "disgust", "fear", "joy", "sad"),
)
def test_unvalidated_or_neutral_emotion_fails_closed_to_zero_vector(
    emotion: str,
) -> None:
    vector = _local_v3_emotion_vector(
        emotion,
        validated=emotion == "neutral",
        strength=1.0,
    )
    np.testing.assert_array_equal(vector, np.zeros(10, dtype=np.float32))


def test_dialog_heuristic_cannot_author_local_v3_emotion() -> None:
    vector = _local_v3_emotion_vector(
        "joy", validated=True, strength=1.0, source="dialog_heuristic"
    )
    np.testing.assert_array_equal(vector, np.zeros(10, dtype=np.float32))


@dataclass(frozen=True)
class _EvidenceFixture:
    quality_label: str = "a2f_v3_local_raw_candidate_unqualified"
    production_qualified: bool = False
    sdk_runtime_parity_verified: bool = False
    noise_seed: int = 17
    emotion_names: tuple[str, ...] = LOCAL_V3_EMOTION_NAMES
    emotion_vector: tuple[float, ...] = (0.0,) * 10
    inference_seconds: float = 0.012
    session_creation_seconds: float = 0.034
    runtime_name: str = "injected-session"
    pinned_model_descriptor_verified: bool = False
    default_onnxruntime_boundary: bool = False


@dataclass(frozen=True)
class _ExecutionFixture:
    evidence: _EvidenceFixture
    audio: AudioSampleClock
    output_timebase: SequenceOutputTimebase


def test_injected_execution_cannot_claim_genuine_model_inference() -> None:
    execution = _ExecutionFixture(
        evidence=_EvidenceFixture(),
        audio=AudioSampleClock("a" * 64, "b" * 64, 16_000, 16_000, 1, 2),
        output_timebase=SequenceOutputTimebase("seconds", 60, 1, 60, 0.0),
    )

    document = _local_v3_evidence_document(execution)  # type: ignore[arg-type]

    assert document["schema_version"] == "autoanim.a2f-v3-local-run/1.0"
    assert document["genuine_v3_onnx_inference"] is False
    assert document["quality_label"].endswith("candidate_unqualified")
    assert document["official_sdk_runtime"] is False
    assert document["official_runtime_parity"] is False
    assert document["official_postprocess_parity"] is False
    assert document["sdk_runtime_parity_verified"] is False
    assert document["production_qualified"] is False
    assert document["jaw_matrix_applied"] is False
    assert document["audio"]["sample_rate_hz"] == 16_000
    assert document["output_timebase"]["fps_numerator"] == 60


def test_descriptor_verified_default_ort_execution_can_claim_genuine_model() -> None:
    execution = _ExecutionFixture(
        evidence=_EvidenceFixture(
            runtime_name="onnxruntime",
            pinned_model_descriptor_verified=True,
            default_onnxruntime_boundary=True,
        ),
        audio=AudioSampleClock("a" * 64, "b" * 64, 16_000, 16_000, 1, 2),
        output_timebase=SequenceOutputTimebase("seconds", 60, 1, 60, 0.0),
    )
    document = _local_v3_evidence_document(execution)  # type: ignore[arg-type]
    assert document["genuine_v3_onnx_inference"] is True


def test_local_v3_fast_pipeline_writes_bound_controls_and_honest_run_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise integration without loading the large ONNX model.

    The runtime and postprocessor numerical contracts are tested separately;
    these fakes retain their real shapes and clocks so this test catches broken
    data flow, emotion ownership, artifact names, and claim inflation.
    """

    frame_count = 30
    source_timestamps = np.arange(frame_count, dtype=np.float64) / 60.0
    observed: dict[str, object] = {}

    def fake_normalize(_source: Path, destination: Path) -> float:
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
        return 0.5

    prosody = ProsodyTrack(
        timestamps=source_timestamps.astype(np.float32),
        rms_dbfs=np.full(frame_count, -30.0, dtype=np.float32),
        energy=np.full(frame_count, 0.5, dtype=np.float32),
        speech_activity=np.ones(frame_count, dtype=np.float32),
        pitch_semitones=np.zeros(frame_count, dtype=np.float32),
        accent=np.zeros(frame_count, dtype=np.float32),
        phrase_id=np.zeros(frame_count, dtype=np.int32),
    )

    skin_pose_names = (
        "mouthClose",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthRollLower",
        "mouthRollUpper",
        "jawOpen",
    )
    profile = SimpleNamespace(
        skin_pose_names=skin_pose_names,
        tongue_pose_names=("tongueOut",),
        as_dict=lambda: {"identity": "Claire", "hash_verified": True},
    )

    class FakePostprocessor:
        @classmethod
        def from_directory(cls, profile_directory: Path) -> "FakePostprocessor":
            observed["postprocess_profile"] = Path(profile_directory)
            return cls()

        def process_chunk(
            self,
            prediction: np.ndarray,
            *,
            dt_seconds: float,
            include_geometry: bool,
        ) -> ClaireV3PostprocessChunk:
            assert prediction.shape == (frame_count, 88_831)
            assert dt_seconds == pytest.approx(1.0 / 60.0)
            assert include_geometry is False
            identity_matrices = np.repeat(
                np.eye(4, dtype=np.float32)[None, :, :], frame_count, axis=0
            )
            row_major = identity_matrices.reshape(frame_count, 16)
            return ClaireV3PostprocessChunk(
                skin_weights=np.zeros(
                    (frame_count, len(skin_pose_names)), dtype=np.float32
                ),
                tongue_weights=np.zeros((frame_count, 1), dtype=np.float32),
                jaw_transforms=identity_matrices,
                jaw_transform_row_major=row_major,
                jaw_transform_nvidia_column_major=row_major.copy(),
                jaw_rms_residual=np.zeros(frame_count, dtype=np.float32),
                eye_rotations_degrees=np.zeros((frame_count, 4), dtype=np.float32),
                eye_rotations_raw_degrees=np.zeros(
                    (frame_count, 4), dtype=np.float32
                ),
                skin_geometry=None,
                tongue_geometry=None,
            )

    class FakeCalibration:
        calibration_hash = "c" * 64

        def save(self, destination: Path) -> None:
            np.savez_compressed(destination, calibration_hash=self.calibration_hash)

    class FakeRetargeter:
        calibration = FakeCalibration()

        def retarget_post_solver_sequence(
            self,
            skin: np.ndarray,
            skin_names: tuple[str, ...],
            *,
            tongue_weights: np.ndarray,
            tongue_pose_names: tuple[str, ...],
        ) -> np.ndarray:
            assert skin.shape == (frame_count, len(skin_pose_names))
            assert skin_names == profile.skin_pose_names
            assert tongue_weights.shape == (frame_count, 1)
            assert tongue_pose_names == profile.tongue_pose_names
            return np.zeros((frame_count, 383), dtype=np.float32)

    def fake_consume(
        audio_path: Path,
        profile_directory: Path,
        consumer: object,
        **kwargs: object,
    ) -> _ExecutionFixture:
        observed["audio_path"] = Path(audio_path)
        observed["runtime_profile"] = Path(profile_directory)
        observed.update(kwargs)
        chunk = SimpleNamespace(
            plan=SimpleNamespace(output_frame_count=frame_count),
            prediction=np.zeros((frame_count, 88_831), dtype=np.float32),
            timestamps_seconds=source_timestamps,
        )
        consumer(chunk)  # type: ignore[operator]
        return _ExecutionFixture(
            evidence=_EvidenceFixture(
                noise_seed=int(kwargs["noise_seed"]),
                emotion_vector=tuple(
                    float(value)
                    for value in np.asarray(kwargs["emotion_vector"]).tolist()
                ),
            ),
            audio=AudioSampleClock("a" * 64, "b" * 64, 16_000, 8_000, 1, 2),
            output_timebase=SequenceOutputTimebase(
                "seconds", 60, 1, frame_count, 0.0
            ),
        )

    monkeypatch.setattr("autoanim_gnm.audio_pipeline.normalize_audio", fake_normalize)
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.run_rhubarb", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.analyze_emotion",
        lambda *_args, **_kwargs: EmotionAnalysis("joy", 1.0, True, "manual", {}),
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.extract_prosody",
        lambda *_args, **_kwargs: prosody,
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.load_official_v3_claire_profile",
        lambda *_args, **_kwargs: profile,
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.ClaireV3Postprocessor", FakePostprocessor
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.CalibratedRetargeter.from_v3_directory",
        lambda *_args, **_kwargs: FakeRetargeter(),
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.consume_local_v3_raw", fake_consume
    )

    profile_directory = tmp_path / "profile"
    output = tmp_path / "output"
    result = run_audio_pipeline(
        tmp_path / "source.wav",
        output,
        fps=60,
        backend="a2f-v3-local",
        emotion="joy",
        emotion_strength=0.4,
        a2f_v3_profile_dir=profile_directory,
        a2f_v3_local_seed=73,
    )

    expected_emotion = np.zeros(10, dtype=np.float32)
    expected_emotion[LOCAL_V3_EMOTION_NAMES.index("joy")] = np.float32(0.4)
    np.testing.assert_array_equal(observed["emotion_vector"], expected_emotion)
    assert observed["noise_seed"] == 73
    assert observed["providers"] == ("CPUExecutionProvider",)
    assert observed["runtime_profile"] == profile_directory.resolve()
    assert observed["postprocess_profile"] == profile_directory.resolve()

    assert result["analysis"]["motion_backend"] == (
        "local_a2f_v3_candidate_unqualified"
    )
    assert result["analysis"]["backend"].endswith("candidate-unqualified")
    assert result["analysis"]["emotion_applied"] == "joy"
    assert result["analysis"]["emotion_decomposition"] == (
        "validated_v3_model_conditioning_only"
    )
    assert result["analysis"]["emotion_strength"] == pytest.approx(0.4)
    local_run = result["analysis"]["local_sequence_run"]
    assert local_run["genuine_v3_onnx_inference"] is False
    assert local_run["official_sdk_runtime"] is False
    assert local_run["official_runtime_parity"] is False
    assert local_run["official_postprocess_parity"] is False
    assert local_run["production_qualified"] is False
    assert result["animation"]["production_validated"] is False

    run_path = output / "a2f-v3-local-run.json"
    controls_path = output / "arkit_controls.npz"
    assert run_path.is_file()
    assert controls_path.is_file()
    retained_run = json.loads(run_path.read_text(encoding="utf-8"))
    assert retained_run["schema_version"] == local_run["schema_version"]
    assert retained_run["noise_seed"] == local_run["noise_seed"]
    assert retained_run["emotion_names"] == list(local_run["emotion_names"])
    assert retained_run["emotion_vector"] == list(local_run["emotion_vector"])
    assert retained_run["production_qualified"] is False
    assert retained_run["emotion_conditioning"] == {
        "analysis_source": "manual",
        "analysis_validated": True,
        "applied_label": "joy",
        "applied_vector": expected_emotion.tolist(),
        "authorized_for_model": True,
        "requested_label": "joy",
        "requested_strength": 0.4,
    }
    assert retained_run["artifacts"]["controls_sha256"]
    assert result["artifacts"]["a2f_v3_local_run"] == run_path.name
    with np.load(controls_path, allow_pickle=False) as controls:
        np.testing.assert_array_equal(controls["timestamps"], source_timestamps)
        assert controls["skin_weights"].shape == (
            frame_count,
            len(skin_pose_names),
        )
        assert controls["tongue_weights"].shape == (frame_count, 1)
        assert controls["jaw_transform_row_major"].shape == (frame_count, 16)
        assert controls["eye_rotations_degrees"].shape == (frame_count, 2, 2)


def test_local_v3_profile_tamper_fails_closed_without_fallback_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = np.arange(30, dtype=np.float32) / 60.0

    def fake_normalize(_source: Path, destination: Path) -> float:
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
        return 0.5

    prosody = ProsodyTrack(
        timestamps=timestamps,
        rms_dbfs=np.full(30, -80.0, dtype=np.float32),
        energy=np.zeros(30, dtype=np.float32),
        speech_activity=np.zeros(30, dtype=np.float32),
        pitch_semitones=np.zeros(30, dtype=np.float32),
        accent=np.zeros(30, dtype=np.float32),
        phrase_id=np.zeros(30, dtype=np.int32),
    )

    monkeypatch.setattr("autoanim_gnm.audio_pipeline.normalize_audio", fake_normalize)
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.run_rhubarb", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.analyze_emotion",
        lambda *_args, **_kwargs: EmotionAnalysis("neutral", 1.0, True, "manual", {}),
    )
    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.extract_prosody",
        lambda *_args, **_kwargs: prosody,
    )

    def reject_tampered_profile(*_args: object, **_kwargs: object) -> object:
        raise ValueError("network.onnx SHA-256 does not match the pinned profile")

    monkeypatch.setattr(
        "autoanim_gnm.audio_pipeline.load_official_v3_claire_profile",
        reject_tampered_profile,
    )
    output = tmp_path / "output"

    with pytest.raises(AutoAnimError, match="SHA-256|pinned profile") as error:
        run_audio_pipeline(
            tmp_path / "source.wav",
            output,
            fps=60,
            backend="a2f-v3-local",
            a2f_v3_profile_dir=tmp_path / "tampered-profile",
        )

    assert error.value.code == "INPUT_INVALID"
    assert not (output / "a2f-v3-local-run.json").exists()
    assert not (output / "arkit_controls.npz").exists()
    assert not (output / "controls.npz").exists()
