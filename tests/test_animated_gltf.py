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


def _tongue_teeth_corrective_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small track whose rank-one projection creates one false collision."""

    frame_count = 7
    frames = np.zeros((frame_count, 70, 3), dtype=np.float32)
    # Stable face-local semantic axes: 100 mm IOD and +Y face-up.
    frames[:, 36] = (-0.05, 0.05, 0.0)
    frames[:, 45] = (0.05, 0.05, 0.0)
    frames[:, 8] = (0.0, -0.1, 0.0)
    frames[:, 27] = (0.0, 0.1, 0.0)
    for upper, lower, x in ((61, 67, -0.01), (62, 66, 0.0), (63, 65, 0.01)):
        frames[:, upper] = (x, 0.002, 0.0)
        frames[:, lower] = (x, -0.002, 0.0)

    # Three ordinary mesh modes dominate the SVD. The smaller tongue/teeth
    # motion remains safely separated in every source frame, but its rank-one
    # projection crosses the 0.1 mm collision threshold on the last frame.
    dominant_motion = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (-0.003014009, 0.001476804, -0.003650071),
            (0.006801811, 0.003822097, -0.001231958),
            (-0.004395719, 0.002986669, 0.002946450),
            (0.001254191, 0.008330657, -0.006406277),
            (0.003545476, -0.002389227, -0.002603547),
            (-0.001644438, 0.008547962, 0.003563887),
        ],
        dtype=np.float32,
    )
    frames[:, 0, 0] = dominant_motion[:, 0]
    frames[:, 1, 1] = dominant_motion[:, 1]
    frames[:, 2, 2] = dominant_motion[:, 2]
    frames[:, 68] = np.asarray(
        [
            (0.0, 0.0, 0.000500000),
            (0.000011437, 0.000914011, -0.000511490),
            (-0.000432746, 0.000031701, -0.000054514),
            (-0.000010571, 0.000041325, -0.000492894),
            (0.000079128, 0.000189455, -0.000696545),
            (-0.000242736, 0.000030398, -0.000118995),
            (-0.000818525, -0.001050079, 0.000203522),
        ],
        dtype=np.float32,
    )
    frames[:, 69] = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.000215130, 0.000674241, -0.000579942),
            (-0.000370352, -0.000501333, 0.000116780),
            (-0.000415816, 0.000112825, -0.000458266),
            (0.000107853, 0.000336371, -0.000873785),
            (-0.000632431, -0.000016070, -0.000159415),
            (-0.000998955, -0.001132892, 0.000268087),
        ],
        dtype=np.float32,
    )
    landmark_indices = np.arange(68, dtype=np.int64)[:, None]
    landmark_weights = np.ones((68, 1), dtype=np.float64)
    return frames, landmark_indices, landmark_weights


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


def test_oral_corrective_prevents_rank_one_tongue_teeth_collision():
    frames, landmark_indices, landmark_weights = _tongue_teeth_corrective_fixture()
    tongue = np.asarray((68,), dtype=np.int64)
    teeth = np.asarray((69,), dtype=np.int64)
    collision_distance_m = 0.001 * 0.1  # threshold ratio times the 100 mm IOD

    factor = factor_vertex_animation(
        frames,
        max_targets=2,
        mesh_p95_limit_m=0.01,
        mesh_max_limit_m=0.01,
        landmark_indices=landmark_indices,
        landmark_weights=landmark_weights,
        landmark_p95_limit_m=0.01,
        landmark_max_limit_m=0.01,
        preserve_oral_semantics=True,
        tongue_vertex_indices=tongue,
        teeth_vertex_indices=teeth,
        tongue_teeth_collision_risk_interocular=0.001,
    )

    assert factor.rank == 2
    assert factor.oral_corrective_targets == 1
    ordinary_rank = factor.rank - factor.oral_corrective_targets
    rank_one_frames = factor.base_vertices + np.einsum(
        "fk,kvj->fvj",
        factor.weights[:, :ordinary_rank],
        factor.morph_positions[:ordinary_rank],
    )
    corrected_frames = factor.base_vertices + np.einsum(
        "fk,kvj->fvj", factor.weights, factor.morph_positions
    )
    source_distance = np.linalg.norm(frames[:, 68] - frames[:, 69], axis=1)
    rank_one_distance = np.linalg.norm(
        rank_one_frames[:, 68] - rank_one_frames[:, 69], axis=1
    )
    corrected_distance = np.linalg.norm(
        corrected_frames[:, 68] - corrected_frames[:, 69], axis=1
    )

    assert np.all(source_distance > collision_distance_m)
    assert np.any(rank_one_distance <= collision_distance_m)
    assert np.all(corrected_distance > collision_distance_m)
    assert np.count_nonzero(factor.weights[:, ordinary_rank:]) == 1
    np.testing.assert_allclose(corrected_frames[-1], frames[-1], atol=1.0e-9)


@pytest.mark.parametrize(
    ("tongue", "teeth"),
    (
        (np.asarray((68,), dtype=np.int64), None),
        (None, np.asarray((69,), dtype=np.int64)),
    ),
)
def test_oral_semantics_rejects_one_sided_tongue_teeth_groups(
    tongue: np.ndarray | None,
    teeth: np.ndarray | None,
):
    frames, landmark_indices, landmark_weights = _tongue_teeth_corrective_fixture()

    with pytest.raises(ValueError, match="supplied together"):
        factor_vertex_animation(
            frames,
            max_targets=2,
            landmark_indices=landmark_indices,
            landmark_weights=landmark_weights,
            preserve_oral_semantics=True,
            tongue_vertex_indices=tongue,
            teeth_vertex_indices=teeth,
        )


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
    assert int(mapping["oral_corrective_targets"]) == exported.oral_corrective_targets
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
