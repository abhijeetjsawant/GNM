from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import trimesh

from autoanim_gnm.gltf_export import export_gnm_glb, split_triangle_corner_uvs
from autoanim_gnm.gnm_adapter import GNMAdapter


@pytest.fixture(scope="module")
def adapter() -> GNMAdapter:
    return GNMAdapter()


def test_seam_split_preserves_every_gnm_triangle_corner(adapter: GNMAdapter):
    vertices = adapter.mesh()
    source_uvs = np.asarray(adapter.model.triangle_uvs, dtype=np.float32)
    split = split_triangle_corner_uvs(vertices, adapter.triangles, source_uvs)

    assert split.positions.shape == (18_437, 3)
    assert split.triangles.shape == (35_324, 3)
    assert len(split.positions) - len(np.unique(split.source_vertices)) == 616
    np.testing.assert_array_equal(
        split.source_vertices[split.triangles], adapter.triangles
    )
    np.testing.assert_array_equal(split.uvs[split.triangles], source_uvs)
    np.testing.assert_allclose(split.positions, vertices[split.source_vertices])


def test_exported_glb_and_animation_mapping_roundtrip(tmp_path, adapter: GNMAdapter):
    output = tmp_path / "head.glb"
    exported = export_gnm_glb(output, adapter, adapter.mesh())

    assert output.read_bytes()[:4] == b"glTF"
    assert exported.vertex_count == 18_437
    assert exported.triangle_count == 35_324
    assert exported.seam_duplicates == 616
    mapping = np.load(exported.mapping_path)
    assert mapping["glb_vertex_to_gnm_vertex"].shape == (18_437,)
    np.testing.assert_array_equal(
        mapping["uvs"][mapping["triangles"]],
        np.asarray(adapter.model.triangle_uvs, dtype=np.float32),
    )

    loaded = trimesh.load(output, force="scene", process=False)
    assert len(loaded.geometry) == 1
    geometry = next(iter(loaded.geometry.values()))
    assert geometry.vertices.shape == (18_437, 3)
    assert geometry.faces.shape == (35_324, 3)
    assert geometry.visual.kind == "vertex"


def test_texture_is_embedded_without_changing_topology(tmp_path, adapter: GNMAdapter):
    checker = np.zeros((8, 8, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    texture = tmp_path / "checker.png"
    Image.fromarray(checker).save(texture)
    output = tmp_path / "textured.glb"

    export_gnm_glb(output, adapter, adapter.mesh(), texture_path=texture)
    loaded = trimesh.load(output, force="scene", process=False)
    geometry = next(iter(loaded.geometry.values()))
    assert geometry.vertices.shape == (18_437, 3)
    assert geometry.visual.kind == "texture"
    assert geometry.visual.material.baseColorTexture.size == (8, 8)


def test_seam_split_rejects_mismatched_uvs():
    with pytest.raises(ValueError, match="triangle_uvs"):
        split_triangle_corner_uvs(
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
            np.zeros((2, 3, 2), dtype=np.float32),
        )
