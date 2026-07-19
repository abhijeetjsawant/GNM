"""Real-audio E2E for the external v3 import/retarget boundary.

The local machine cannot execute NVIDIA v3.  This test deliberately adapts a
retained real v2.3 control take into the *published v3 post-solver ranges* to
exercise transport, profile validation, the separate v3 geometry calibration,
GNM retarget, oral validation and animated export.  It is not evidence of v3
inference quality and the resulting metadata must keep that claim false.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.a2f_v3_profile import load_official_v3_claire_profile
from autoanim_gnm.artifacts import sha256
from autoanim_gnm.audio import normalize_audio
from autoanim_gnm.audio_pipeline import run_audio_pipeline
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.sequence_provider import (
    A2F_V3_EYE_CONTROL_NAMES,
    A2F_V3_JAW_CONTROL_NAMES,
    CONTROL_SCHEMA_VERSION,
    QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
    V3_REQUEST_SCHEMA_VERSION,
    V3_RESPONSE_SCHEMA_VERSION,
    ZERO_STATE_SHA256,
    inspect_bound_pcm_audio,
    seal_v3_worker_request_document,
    seal_v3_worker_response_document,
    sequence_chunk_payload_sha256,
    sequence_state_out_sha256,
    validate_v3_worker_request,
)
from autoanim_gnm.serialization import write_json


PROFILE = Path(".cache/autoanim_gnm/a2f-v3-claire-profile")
SOURCE_JOB = Path("artifacts/jobs/01kxtqqjmhzpx4fdygvjs5xjta")
RHUBARB = Path(".cache/autoanim_gnm/rhubarb/rhubarb")
REQUIRED = (
    PROFILE / "network.onnx",
    PROFILE / "worker-runtime-attestation.json",
    SOURCE_JOB / "input.wav",
    SOURCE_JOB / "arkit_controls.npz",
    RHUBARB,
)


def _fixture_available() -> bool:
    return all(path.is_file() for path in REQUIRED)


def test_explicit_v3_import_fails_before_media_work_when_bindings_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(AutoAnimError, match="requires: identity, model"):
        run_audio_pipeline(
            tmp_path / "not-read.wav",
            tmp_path / "output",
            backend="a2f-v3",
            a2f_v3_request_path=tmp_path / "request.json",
            a2f_v3_response_path=tmp_path / "response.json",
        )


def test_v3_bindings_cannot_be_supplied_to_a_fallback_backend(tmp_path: Path) -> None:
    with pytest.raises(AutoAnimError, match="require --backend a2f-v3"):
        run_audio_pipeline(
            tmp_path / "not-read.wav",
            tmp_path / "output",
            backend="fallback",
            a2f_v3_request_path=tmp_path / "request.json",
        )


@pytest.mark.skipif(
    not _fixture_available(), reason="real audio and pinned public v3 profile not cached"
)
def test_real_audio_sequence_import_reaches_animated_gnm_without_v3_quality_claim(
    tmp_path: Path,
) -> None:
    profile = load_official_v3_claire_profile(PROFILE, verify_network=True)
    source = SOURCE_JOB / "input.wav"
    normalized = tmp_path / "normalized.wav"
    normalize_audio(source, normalized)
    with np.load(SOURCE_JOB / "arkit_controls.npz", allow_pickle=False) as values:
        raw_skin = np.asarray(values["skin_weights"], dtype=np.float32)
        raw_tongue = np.asarray(values["tongue_weights"], dtype=np.float32)
        skin_names = tuple(str(value) for value in values["skin_pose_names"].tolist())
        tongue_names = tuple(
            str(value) for value in values["tongue_pose_names"].tolist()
        )
    assert skin_names == profile.skin_pose_names
    assert tongue_names == profile.tongue_pose_names
    skin_min = np.asarray(profile.skin_minimums, dtype=np.float32)
    skin_max = np.asarray(profile.skin_maximums, dtype=np.float32)
    tongue_min = np.asarray(profile.tongue_minimums, dtype=np.float32)
    tongue_max = np.asarray(profile.tongue_maximums, dtype=np.float32)
    skin = skin_min[None, :] + np.clip(raw_skin, 0.0, 1.0) * (
        skin_max - skin_min
    )[None, :]
    tongue = tongue_min[None, :] + np.clip(raw_tongue, 0.0, 1.0) * (
        tongue_max - tongue_min
    )[None, :]

    clock = inspect_bound_pcm_audio(normalized)
    frame_count = (
        clock.sample_count * 30 + clock.sample_rate_hz - 1
    ) // clock.sample_rate_hz
    assert frame_count == len(skin)
    schema_path = write_json(
        tmp_path / "control-schema.json",
        {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "skin": list(skin_names),
            "tongue": list(tongue_names),
            "jaw": list(A2F_V3_JAW_CONTROL_NAMES),
            "eye": list(A2F_V3_EYE_CONTROL_NAMES),
        },
    )
    runtime_path = PROFILE / "worker-runtime-attestation.json"
    identity_path = PROFILE / "model_data_Claire.npz"
    model_path = PROFILE / "network.onnx"
    bindings = {
        "model_sha256": sha256(model_path),
        "runtime_sha256": sha256(runtime_path),
        "identity_sha256": sha256(identity_path),
        "blendshape_schema_sha256": sha256(schema_path),
    }
    output_timebase = {
        "units": "seconds",
        "fps_numerator": 30,
        "fps_denominator": 1,
        "frame_count": frame_count,
        "timestamp_origin_seconds": 0.0,
    }
    chunk_plan = {
        "chunk_index": 0,
        "audio_start_sample": 0,
        "audio_sample_count": clock.sample_count,
        "audio_overlap_previous_samples": 0,
        "output_start_frame": 0,
        "output_frame_count": frame_count,
    }
    provider = {
        "provider_id": "nvidia.audio2face-3d",
        "model_version": "3.0",
        "quality_label": QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
    }
    request_path = write_json(
        tmp_path / "request.json",
        seal_v3_worker_request_document(
            {
                "schema_version": V3_REQUEST_SCHEMA_VERSION,
                "provider": provider,
                "bindings": bindings,
                "audio": asdict(clock),
                "output_timebase": output_timebase,
                "chunks": [chunk_plan],
            }
        ),
    )
    request = validate_v3_worker_request(
        request_path,
        audio_path=normalized,
        model_path=model_path,
        runtime_path=runtime_path,
        identity_path=identity_path,
        blendshape_schema_path=schema_path,
    )
    jaw = np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (frame_count, 1))
    chunk = {
        **chunk_plan,
        "state_in_sha256": ZERO_STATE_SHA256,
        "timestamps_seconds": (np.arange(frame_count) / 30.0).tolist(),
        "controls": {
            "skin": skin.tolist(),
            "tongue": tongue.tolist(),
            "jaw": jaw.tolist(),
            "eye": np.zeros((frame_count, 4), dtype=np.float32).tolist(),
        },
    }
    chunk_hash = sequence_chunk_payload_sha256(chunk)
    chunk["chunk_payload_sha256"] = chunk_hash
    chunk["state_out_sha256"] = sequence_state_out_sha256(
        request_sha256=request.request_sha256,
        chunk_index=0,
        state_in_sha256=ZERO_STATE_SHA256,
        chunk_payload_sha256=chunk_hash,
    )
    response_path = write_json(
        tmp_path / "response.json",
        seal_v3_worker_response_document(
            {
                "schema_version": V3_RESPONSE_SCHEMA_VERSION,
                "request_sha256": request.request_sha256,
                "provider": provider,
                "bindings": bindings,
                "audio": asdict(clock),
                "output_timebase": output_timebase,
                "control_names": {
                    "skin": list(skin_names),
                    "tongue": list(tongue_names),
                    "jaw": list(A2F_V3_JAW_CONTROL_NAMES),
                    "eye": list(A2F_V3_EYE_CONTROL_NAMES),
                },
                "chunks": [chunk],
            }
        ),
    )

    output = tmp_path / "output"
    result = run_audio_pipeline(
        source,
        output,
        fps=30,
        backend="a2f-v3",
        rhubarb_bin=RHUBARB,
        a2f_v3_request_path=request_path,
        a2f_v3_response_path=response_path,
        a2f_v3_model_path=model_path,
        a2f_v3_runtime_path=runtime_path,
        a2f_v3_identity_path=identity_path,
        a2f_v3_schema_path=schema_path,
        a2f_v3_profile_dir=PROFILE,
    )
    assert result["analysis"]["motion_backend"] == (
        "unverified_external_sequence_controls_candidate"
    )
    assert result["analysis"]["sequence_import"]["production_qualified"] is False
    assert result["analysis"]["sequence_import"][
        "worker_authentication_verified"
    ] is False
    assert result["viewer"]["status"] == "ready"
    assert result["viewer"]["glb_covers_full_track"] is True
    assert result["metrics"]["mouth_step_max_interocular"] <= 0.0401
    assert result["oral_validation"]["lip_order_inversion_risk_frames"] == 0
    assert result["oral_validation"]["tongue_teeth_collision_risk_frames"] == 0
    assert (output / "animation.glb").is_file()
    assert (output / "a2f-v3-import.json").is_file()
