from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest
import trimesh

from autoanim_gnm.animated_gltf import (
    AnimationCompressionError,
    export_animated_gnm_glb,
    factor_vertex_animation,
)
from autoanim_gnm.gnm_adapter import GNMAdapter


def _json_chunk(path: Path) -> dict:
    payload = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF" and version == 2 and length == len(payload)
    chunk_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    assert chunk_type == b"JSON"
    return json.loads(payload[20 : 20 + chunk_length])


def test_low_rank_factor_selects_smallest_passing_rank():
    rng = np.random.default_rng(44)
    base = rng.normal(size=(12, 3)).astype(np.float32)
    basis = rng.normal(scale=0.002, size=(2, 12, 3)).astype(np.float32)
    weights = rng.normal(size=(18, 2)).astype(np.float32)
    weights[0] = 0
    frames = base + np.einsum("fk,kvj->fvj", weights, basis)

    factor = factor_vertex_animation(
        frames,
        max_targets=4,
        mesh_p95_limit_m=1e-6,
        mesh_max_limit_m=2e-6,
    )
    assert factor.rank == 2
    reconstructed = factor.base_vertices + np.einsum(
        "fk,kvj->fvj", factor.weights, factor.morph_positions
    )
    np.testing.assert_allclose(reconstructed, frames, atol=2e-6)


def test_low_rank_factor_fails_closed_when_cap_is_too_small():
    rng = np.random.default_rng(9)
    frames = rng.normal(scale=0.01, size=(8, 10, 3)).astype(np.float32)
    with pytest.raises(AnimationCompressionError) as caught:
        factor_vertex_animation(
            frames,
            max_targets=1,
            mesh_p95_limit_m=1e-9,
            mesh_max_limit_m=1e-9,
        )
    assert caught.value.metrics["rank"] == 1


def test_real_gnm_animation_exports_standard_morph_track(tmp_path: Path):
    adapter = GNMAdapter()
    expression = np.zeros((12, adapter.expression_dim), dtype=np.float32)
    phase = np.sin(np.linspace(0, np.pi, len(expression))).astype(np.float32)
    expression[:, 205] = 0.35 * phase
    expression[:, 217] = -0.20 * phase
    rotations = np.zeros((len(expression), adapter.model.num_joints, 3), dtype=np.float32)
    rotations[:, 1, 1] = np.deg2rad(0.8) * phase
    frames = np.stack(
        [adapter.mesh(expression=e, rotations=r) for e, r in zip(expression, rotations, strict=True)]
    )
    timestamps = np.arange(len(frames), dtype=np.float32) / 30.0
    output = tmp_path / "animation.glb"

    exported = export_animated_gnm_glb(output, adapter, frames, timestamps)
    document = _json_chunk(output)
    assert exported.rank <= 3
    assert exported.mesh_p95_mm <= 0.10
    assert exported.mesh_max_mm <= 0.50
    assert exported.landmark_p95_mm <= 0.25
    assert exported.landmark_max_mm <= 1.0
    assert len(document["meshes"][0]["primitives"][0]["targets"]) == exported.rank
    assert document["animations"][0]["name"] == "autoanim"
    assert document["animations"][0]["channels"][0]["target"]["path"] == "weights"
    assert document["accessors"][document["animations"][0]["samplers"][0]["output"]][
        "count"
    ] == len(frames) * exported.rank
    mapping = np.load(exported.mapping_path)
    assert mapping["morph_weights"].shape == (len(frames), exported.rank)
    loaded = trimesh.load(output, force="scene", process=False)
    geometry = next(iter(loaded.geometry.values()))
    assert geometry.vertices.shape == (18_437, 3)
    assert geometry.faces.shape == (35_324, 3)


def test_static_track_emits_valid_glb_without_animation(tmp_path: Path):
    adapter = GNMAdapter()
    frames = np.repeat(adapter.mesh()[None], 2, axis=0)
    output = tmp_path / "static.glb"
    exported = export_animated_gnm_glb(
        output, adapter, frames, np.asarray((0.0, 1.0 / 30.0), dtype=np.float32)
    )
    document = _json_chunk(output)
    assert exported.rank == 0
    assert "animations" not in document
