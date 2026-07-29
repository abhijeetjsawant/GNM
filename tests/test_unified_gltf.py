from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from autoanim_gnm.body import BodyTrack, CANONICAL_HUMANOID
from autoanim_gnm.body_provider import sha256_file
from autoanim_gnm.serialization import write_json, write_npz
from autoanim_gnm.unified_gltf import (
    UnifiedGLTFError,
    _boundary_edge_count,
    _composition_array_sha256,
    _load_and_validate_composition,
    _trajectory_project_four_weights,
    _zipper_loops,
)


def _ring(count: int, y: float, radius_x: float, radius_z: float) -> np.ndarray:
    angle = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    return np.column_stack(
        (radius_x * np.cos(angle), np.full(count, y), radius_z * np.sin(angle))
    ).astype(np.float32)


def test_unequal_loop_zipper_closes_both_cut_boundaries() -> None:
    body = _ring(8, 0.0, 1.0, 0.8)
    head = _ring(13, 0.2, 0.92, 0.75)
    vertices = np.concatenate((body, head))
    body_loop = np.arange(len(body), dtype=np.int32)
    head_loop = np.arange(len(head), dtype=np.int32) + len(body)
    bridge = _zipper_loops(body_loop, head_loop, vertices)
    assert bridge.shape == (len(body) + len(head), 3)
    assert _boundary_edge_count(
        bridge, np.concatenate((body_loop, head_loop))
    ) == len(body) + len(head)
    p0, p1, p2 = (vertices[bridge[:, corner]] for corner in range(3))
    assert np.all(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) > 0.0)


def test_four_weight_projection_is_exact_when_four_joints_suffice() -> None:
    vertices = np.asarray([[0.2, 0.4, 0.1], [-0.2, 0.3, 0.0]], dtype=np.float32)
    dense = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.0],
            [0.0, 0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    matrices = np.repeat(np.eye(4)[None, None], 3 * 5, axis=0).reshape(
        3, 5, 4, 4
    )
    for frame in range(3):
        for joint in range(5):
            matrices[frame, joint, :3, 3] = (
                frame * 0.01 * joint,
                frame * -0.005 * joint,
                0.0,
            )
    indices, weights, metrics = _trajectory_project_four_weights(
        vertices, dense, matrices
    )
    assert indices.shape == (2, 4)
    assert weights.shape == (2, 4)
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1.0e-6)
    assert metrics["rms_m"] <= 1.0e-12
    assert metrics["max_m"] <= 1.0e-12


def test_sealed_composition_rejects_same_time_but_different_source_pts(
    tmp_path,
) -> None:
    frames = 2
    rotations = np.zeros((frames, 25, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    eye_rotations = np.zeros((frames, 2, 4), dtype=np.float32)
    eye_rotations[..., 3] = 1.0
    gaze = np.zeros((frames, 3), dtype=np.float32)
    gaze[:, 2] = 1.0
    track = BodyTrack(
        duration_ticks=1600,
        ticks_per_second=48_000,
        sample_rate_hz=30,
        joint_names=CANONICAL_HUMANOID.names,
        ticks=np.asarray([0, 1600], dtype=np.int64),
        root_translation_m=np.zeros((frames, 3), dtype=np.float32),
        local_rotations_xyzw=rotations,
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=gaze,
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=eye_rotations,
        source_plan_sha256="1" * 64,
    )
    performance = {
        "source_pts": np.asarray([0, 513], dtype=np.int64),
        "identity": np.zeros(253, dtype=np.float32),
        "expression": np.zeros((frames, 383), dtype=np.float32),
        "rotations": np.zeros((frames, 4, 3), dtype=np.float32),
        "translation": np.zeros((frames, 3), dtype=np.float32),
        "timestamps_seconds": track.ticks.astype(np.float64) / 48_000,
    }
    body_asset = tmp_path / "body.npz"
    body_manifest = tmp_path / "body.json"
    face_performance = tmp_path / "face.npz"
    body_asset.write_bytes(b"body")
    body_manifest.write_text("{}", encoding="utf-8")
    write_npz(face_performance, **performance)
    arrays_path = tmp_path / "composition.npz"
    sealed_pts = np.asarray([0, 512], dtype=np.int64)
    arrays = {
        "schema_version": np.asarray("autoanim.unified-performance-arrays/1.0"),
        "ticks": track.ticks,
        "source_pts": sealed_pts,
        "timestamps_seconds": performance["timestamps_seconds"],
        "gnm_identity": performance["identity"],
        "gnm_expression": performance["expression"],
        "gnm_owned_rotations": performance["rotations"],
        "gnm_owned_translation": performance["translation"],
        "body_root_translation_m": track.root_translation_m,
        "body_local_rotations_xyzw": track.local_rotations_xyzw,
        "body_foot_contacts": track.foot_contacts,
    }
    write_npz(arrays_path, **arrays)
    manifest = {
        "schema_version": "autoanim.unified-performance/1.0",
        "production_validated": False,
        "artifacts": {
            "unified_arrays": {"sha256": sha256_file(arrays_path)},
            "body_asset": {"sha256": sha256_file(body_asset)},
            "body_manifest": {"sha256": sha256_file(body_manifest)},
            "face_performance": {"sha256": sha256_file(face_performance)},
        },
        "timeline": {
            "ticks_sha256": _composition_array_sha256(track.ticks),
            "source_pts_sha256": _composition_array_sha256(sealed_pts),
            "face_body_exact_pts_match": True,
            "face_body_exact_ticks_match": True,
        },
    }
    manifest["payload_sha256"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = tmp_path / "composition.json"
    write_json(manifest_path, manifest)
    with pytest.raises(UnifiedGLTFError, match="sealed composition"):
        _load_and_validate_composition(
            manifest_path,
            arrays_path,
            body_asset_path=body_asset,
            body_manifest_path=body_manifest,
            face_performance_path=face_performance,
            performance=performance,
            track=track,
        )
