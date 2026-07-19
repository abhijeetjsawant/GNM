from __future__ import annotations

import json
import numpy as np
from PIL import Image
import pytest
import struct
import trimesh

from autoanim_gnm.animated_gltf import export_animated_gnm_glb
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


def test_animated_glb_embeds_character_texture(tmp_path, adapter: GNMAdapter):
    texture = tmp_path / "character.png"
    vertical = np.zeros((8, 16, 3), dtype=np.uint8)
    vertical[..., 0] = np.arange(8, dtype=np.uint8)[:, None] * 30
    vertical[..., 1] = 120
    Image.fromarray(vertical).save(texture)
    neutral = adapter.mesh()
    frames = np.stack((neutral, neutral.copy()))
    frames[1, 0, 0] += 1e-3
    output = tmp_path / "animated-textured.glb"
    packed_uvs = np.asarray(adapter.model.triangle_uvs, dtype=np.float32) * 0.5 + 0.1

    exported = export_animated_gnm_glb(
        output,
        adapter,
        frames,
        np.asarray((0.0, 1.0), dtype=np.float32),
        texture_path=texture,
        triangle_uvs=packed_uvs,
    )
    payload = output.read_bytes()
    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    assert json_type == b"JSON"
    document = json.loads(payload[20 : 20 + json_length].decode("utf-8"))
    assert document["images"][0]["mimeType"] == "image/png"
    assert document["textures"][0]["source"] == 0
    assert document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] == {
        "index": 0
    }
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    assert "COLOR_0" not in attributes

    binary_header = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<I4s", payload, binary_header)
    assert binary_type == b"BIN\0"
    binary = payload[binary_header + 8 : binary_header + 8 + binary_length]
    uv_accessor = document["accessors"][attributes["TEXCOORD_0"]]
    uv_view = document["bufferViews"][uv_accessor["bufferView"]]
    byte_offset = int(uv_view.get("byteOffset", 0)) + int(uv_accessor.get("byteOffset", 0))
    gltf_uvs = np.frombuffer(
        binary,
        dtype="<f4",
        count=int(uv_accessor["count"]) * 2,
        offset=byte_offset,
    ).reshape(-1, 2)
    with np.load(exported.mapping_path) as mapping:
        internal = mapping["internal_uvs_lower_left"]
        np.testing.assert_allclose(mapping["gltf_uvs_upper_left"], gltf_uvs)
        np.testing.assert_allclose(internal[mapping["triangles"]], packed_uvs)
    np.testing.assert_allclose(gltf_uvs[:, 0], internal[:, 0])
    np.testing.assert_allclose(gltf_uvs[:, 1], 1.0 - internal[:, 1])


def test_seam_split_rejects_mismatched_uvs():
    with pytest.raises(ValueError, match="triangle_uvs"):
        split_triangle_corner_uvs(
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
            np.zeros((2, 3, 2), dtype=np.float32),
        )
