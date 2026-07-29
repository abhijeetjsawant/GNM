from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoanim_gnm.body import BodyTrack, CANONICAL_HUMANOID
from autoanim_gnm.body_compositor import (
    BodyCompositionError,
    compose_unified_performance,
    write_unified_preview_html,
)


def _body_track(*, eyes_active: bool = False) -> BodyTrack:
    frames = 3
    rotations = np.zeros((frames, 25, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    eyes = np.zeros((frames, 2, 4), dtype=np.float32)
    eyes[..., 3] = 1.0
    if eyes_active:
        eyes[1, 0, 1] = 0.1
        eyes[1, 0, 3] = np.sqrt(0.99)
    gaze = np.zeros((frames, 3), dtype=np.float32)
    gaze[:, 2] = 1.0
    return BodyTrack(
        duration_ticks=3200,
        ticks_per_second=48_000,
        sample_rate_hz=30,
        joint_names=CANONICAL_HUMANOID.names,
        ticks=np.asarray([0, 1600, 3200], dtype=np.int64),
        root_translation_m=np.zeros((frames, 3), dtype=np.float32),
        local_rotations_xyzw=rotations,
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=gaze,
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=eyes,
        source_plan_sha256="2" * 64,
    )


def _setup(tmp_path: Path) -> dict:
    canonical = tmp_path / "source.mp4"
    canonical.write_bytes(b"canonical-source")
    face_source = tmp_path / "face-crop.mp4"
    face_source.write_bytes(b"derived-face-source")
    face_performance = tmp_path / "performance.npz"
    expression = np.arange(3 * 383, dtype=np.float32).reshape(3, 383) / 1000.0
    rotations = np.zeros((3, 4, 3), dtype=np.float32)
    rotations[:, 0, 1] = [0.0, 0.1, 0.2]
    rotations[:, 2, 1] = [0.0, 0.05, 0.1]
    translation = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.02, 0.0], [0.02, 0.03, 0.0]],
        dtype=np.float32,
    )
    np.savez_compressed(
        face_performance,
        identity=np.zeros(253, dtype=np.float32),
        expression=expression,
        rotations=rotations,
        translation=translation,
        timestamps_seconds=np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0]),
        source_pts=np.asarray([0, 512, 1024], dtype=np.int64),
    )
    face_glb = tmp_path / "face.glb"
    face_glb.write_bytes(b"face")
    body_glb = tmp_path / "body.glb"
    body_glb.write_bytes(b"body")
    body_manifest = tmp_path / "body.json"
    body_manifest.write_text(
        json.dumps({"gnm_head_socket": {"attachment_calibrated": False}}),
        encoding="utf-8",
    )
    body_asset = tmp_path / "body.npz"
    body_asset.write_bytes(b"asset")
    canonical_sha = sha256(canonical.read_bytes()).hexdigest()
    face_sha = sha256(face_source.read_bytes()).hexdigest()
    soma = SimpleNamespace(
        input_sha256=canonical_sha,
        source_pts=np.asarray([0, 512, 1024], dtype=np.int64),
        ticks=np.asarray([0, 1600, 3200], dtype=np.int64),
        source_time_base_numerator=1,
        source_time_base_denominator=15360,
    )
    return {
        "canonical_source_path": canonical,
        "face_source_path": face_source,
        "face_source_derivation": {
            "schema_version": "autoanim.source-derivation/1.0",
            "operation": "spatial_crop_and_scale",
            "parent_source_sha256": canonical_sha,
            "derived_source_sha256": face_sha,
            "crop_ltrb_pixels": [0, 150, 320, 630],
            "output_size_pixels": [640, 960],
            "timing_policy": "exact_source_pts_preserved",
        },
        "face_performance_path": face_performance,
        "face_glb_path": face_glb,
        "soma_track": soma,
        "body_track": _body_track(),
        "body_glb_path": body_glb,
        "body_manifest_path": body_manifest,
        "body_asset_path": body_asset,
    }


def test_composition_preserves_face_authority_and_reports_publish_blockers(
    tmp_path: Path,
) -> None:
    arguments = _setup(tmp_path)
    output = tmp_path / "composition.json"
    result = compose_unified_performance(output, **arguments)

    assert result["timeline"]["face_body_exact_pts_match"] is True
    assert result["ownership"]["authoritative_face_arrays_byte_identical"] is True
    assert (
        result["ownership"]["gnm_expression_input_sha256"]
        == result["ownership"]["gnm_expression_output_sha256"]
    )
    assert (
        result["ownership"]["gnm_eye_input_sha256"]
        == result["ownership"]["gnm_eye_output_sha256"]
    )
    assert result["ownership"]["base_head_applied_once"] is True
    assert result["attachment"]["single_connected_character_mesh"] is False
    assert "GNM_HEAD_SOCKET_UNCALIBRATED" in result["publish_blockers"]
    assert result["production_validated"] is False
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["gnm_expression"],
            np.load(arguments["face_performance_path"])["expression"],
        )
        assert np.count_nonzero(arrays["gnm_owned_rotations"][:, :2]) == 0
        assert np.count_nonzero(arrays["gnm_owned_translation"]) == 0
        assert np.count_nonzero(arrays["gnm_owned_rotations"][:, 2:]) > 0


def test_composition_rejects_nonidentical_face_and_body_pts(tmp_path: Path) -> None:
    arguments = _setup(tmp_path)
    arguments["soma_track"].source_pts = np.asarray([0, 512, 1536], dtype=np.int64)
    with pytest.raises(BodyCompositionError, match="PTS"):
        compose_unified_performance(tmp_path / "composition.json", **arguments)


def test_composition_rejects_second_eye_owner(tmp_path: Path) -> None:
    arguments = _setup(tmp_path)
    arguments["body_track"] = _body_track(eyes_active=True)
    with pytest.raises(BodyCompositionError, match="ownership conflict"):
        compose_unified_performance(tmp_path / "composition.json", **arguments)


def test_composition_rejects_unbound_face_derivation(tmp_path: Path) -> None:
    arguments = _setup(tmp_path)
    arguments["face_source_derivation"]["parent_source_sha256"] = "0" * 64
    with pytest.raises(BodyCompositionError, match="sealed source"):
        compose_unified_performance(tmp_path / "composition.json", **arguments)


def test_unified_preview_is_explicitly_diagnostic_and_uses_one_video_clock(
    tmp_path: Path,
) -> None:
    output = write_unified_preview_html(
        tmp_path / "review.html",
        source_video_url="/source.mp4",
        body_glb_url="/body.glb",
        face_glb_url="/face.glb",
        manifest_url="/composition.json",
        vendor_base_url="/vendor",
    )
    html = output.read_text(encoding="utf-8")
    assert "NOT PUBLISHABLE · UNCALIBRATED ATTACHMENT" in html
    assert "single_connected_character_mesh" in html
    assert "mixer.setTime(source.currentTime)" in html
    assert "action.setLoop(THREE.LoopOnce,1)" in html
    assert "action.clampWhenFinished=true" in html
    assert 'src="/source.mp4"' in html
    assert "new GLTFLoader().load(url" in html
