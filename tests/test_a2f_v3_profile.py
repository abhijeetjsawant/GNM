from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from autoanim_gnm.a2f_v3_profile import (
    OFFICIAL_V3_ASSET_SHA256,
    OFFICIAL_V3_TONGUE_CONTROL_NAMES,
    load_official_v3_claire_profile,
)
from autoanim_gnm.sequence_provider import (
    A2F_V3_EYE_CONTROL_NAMES,
    A2F_V3_JAW_CONTROL_NAMES,
    A2F_V3_MODEL_REVISION,
    A2F_V3_SDK_REVISION,
    AudioSampleClock,
    QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
    SequenceArtifactBindings,
    SequenceChunkProvenance,
    SequenceControlNames,
    SequenceOutputTimebase,
    SequenceProviderError,
    SequenceProviderTrack,
    EXECUTION_CHAIN_ROOT_SHA256,
    build_official_v3_inference_plan,
    validate_official_v3_sequence_track,
)


SKIN_NAMES = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "browDownLeft",
    "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft",
    "noseSneerRight", "tongueOut",
)


def _track() -> SequenceProviderTrack:
    frames = 120
    tongue = np.zeros((frames, 16), dtype=np.float32)
    tongue[:, 0] = 1.75
    tongue[:, 4] = 2.50
    tongue[:, 9] = 0.20
    hashes = SequenceArtifactBindings(
        OFFICIAL_V3_ASSET_SHA256["network.onnx"],
        "2" * 64,
        OFFICIAL_V3_ASSET_SHA256["model_data_Claire.npz"],
        "4" * 64,
    )
    audio_clock = AudioSampleClock("5" * 64, "5" * 64, 16_000, 32_000, 1, 2)
    output_timebase = SequenceOutputTimebase("seconds", 60, 1, frames, 0.0)
    chain_in = EXECUTION_CHAIN_ROOT_SHA256
    provenance: list[SequenceChunkProvenance] = []
    for plan in build_official_v3_inference_plan(audio_clock, output_timebase):
        chain_out = "6" * 64
        provenance.append(
            SequenceChunkProvenance(
                **asdict(plan),
                execution_chain_in_sha256=chain_in,
                execution_chain_out_sha256=chain_out,
                chunk_payload_sha256="7" * 64,
            )
        )
        chain_in = chain_out
    return SequenceProviderTrack(
        provider_id="nvidia.audio2face-3d",
        model_version="3.0",
        quality_label=QUALITY_A2F_V3_SEQUENCE_CANDIDATE,
        bindings=hashes,
        source_audio_sha256="5" * 64,
        audio_sample_rate_hz=16_000,
        audio_sample_count=32_000,
        output_timebase=output_timebase,
        timestamps=np.arange(frames, dtype=np.float64) / 60.0,
        control_names=SequenceControlNames(
            SKIN_NAMES,
            OFFICIAL_V3_TONGUE_CONTROL_NAMES,
            A2F_V3_JAW_CONTROL_NAMES,
            A2F_V3_EYE_CONTROL_NAMES,
        ),
        skin=np.zeros((frames, 52), dtype=np.float32),
        tongue=tongue,
        jaw=np.zeros((frames, 16), dtype=np.float32),
        eye=np.zeros((frames, 4), dtype=np.float32),
        chunks=tuple(provenance),
        request_sha256="8" * 64,
        response_sha256="9" * 64,
    )


def _validate(track: SequenceProviderTrack):
    minimums = np.zeros(16, dtype=np.float64)
    minimums[9] = 0.2
    maximums = np.ones(16, dtype=np.float64)
    maximums[[0, 8, 13]] = 2.0
    maximums[4] = 3.0
    maximums[9] = 1.2
    return validate_official_v3_sequence_track(
        track,
        skin_pose_names=SKIN_NAMES,
        skin_minimums=np.zeros(52, dtype=np.float64),
        skin_maximums=np.ones(52, dtype=np.float64),
        tongue_pose_names=OFFICIAL_V3_TONGUE_CONTROL_NAMES,
        tongue_minimums=minimums,
        tongue_maximums=maximums,
        public_model_version="3.0",
        network_version="3.2",
        identity="Claire",
        identity_index=0,
        model_revision=A2F_V3_MODEL_REVISION,
        sdk_revision=A2F_V3_SDK_REVISION,
    )


def test_official_v3_track_accepts_post_solver_tongue_values_above_one() -> None:
    validation = _validate(_track())
    assert validation.tongue_control_count == 16
    assert validation.production_qualified is False
    assert validation.quality_label == QUALITY_A2F_V3_SEQUENCE_CANDIDATE


def test_official_v3_track_rejects_clipped_offset_or_out_of_range_controls() -> None:
    track = _track()
    clipped = track.tongue.copy()
    clipped[:, 9] = 0.0
    with pytest.raises(SequenceProviderError, match="tongue controls exceed"):
        _validate(replace(track, tongue=clipped))


def test_official_v3_track_rejects_v2_calibration_schema_substitution() -> None:
    track = _track()
    names = SequenceControlNames(
        track.control_names.skin[:-1] + ("notTongueOut",),
        track.control_names.tongue,
        track.control_names.jaw,
        track.control_names.eye,
    )
    with pytest.raises(SequenceProviderError, match="exact Claire"):
        _validate(replace(track, control_names=names))


REAL_PROFILE = Path(".cache/autoanim_gnm/a2f-v3-claire-profile")


@pytest.mark.skipif(not REAL_PROFILE.is_dir(), reason="official public v3 assets not cached")
def test_pinned_public_v3_claire_assets_load_with_official_ranges() -> None:
    profile = load_official_v3_claire_profile(REAL_PROFILE)
    assert profile.skin_pose_names == SKIN_NAMES
    assert profile.tongue_pose_names == OFFICIAL_V3_TONGUE_CONTROL_NAMES
    assert profile.tongue_maximums[4] == 3.0
    assert profile.tongue_minimums[9] == 0.2
    assert profile.validate_track(_track()).production_qualified is False
