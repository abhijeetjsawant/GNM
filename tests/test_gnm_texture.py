from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import trimesh

from autoanim_gnm.gltf_export import export_gnm_glb, split_triangle_corner_uvs
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.gnm_texture import (
    GNM_TEXTURE_COMPONENTS,
    build_gnm_texture_atlas,
)
from autoanim_gnm.multiview import CameraIntrinsics, PerspectiveCamera
from autoanim_gnm.multiview_pipeline import (
    _component_texture_metrics,
    texture_camera_from_fit,
)
from autoanim_gnm.texture_baker import bake_multiview_texture


@pytest.fixture(scope="module")
def adapter() -> GNMAdapter:
    return GNMAdapter()


def test_every_gnm_triangle_has_exactly_one_texture_component(
    adapter: GNMAdapter,
) -> None:
    atlas = build_gnm_texture_atlas(adapter, 256)

    assert atlas.component_names == GNM_TEXTURE_COMPONENTS
    assert atlas.triangle_components.shape == (35_324,)
    np.testing.assert_array_equal(
        np.bincount(atlas.triangle_components, minlength=6),
        np.asarray((24_820, 1_512, 1_512, 2_828, 2_828, 1_824)),
    )
    for component_index, name in enumerate(atlas.component_names):
        official = np.asarray(
            adapter.model.triangle_indices_for_group(name), dtype=np.int32
        )
        np.testing.assert_array_equal(
            np.flatnonzero(atlas.triangle_components == component_index), official
        )


def test_repacked_component_uv_domains_are_disjoint_and_padded(
    adapter: GNMAdapter,
) -> None:
    atlas = build_gnm_texture_atlas(adapter, (256, 512))
    pad_u = atlas.padding_texels / (atlas.width - 1)
    pad_v = atlas.padding_texels / (atlas.height - 1)

    for component_index, name in enumerate(atlas.component_names):
        selected = atlas.triangle_components == component_index
        uvs = atlas.triangle_uvs[selected].reshape(-1, 2)
        left, bottom, right, top = atlas.component_bounds[name]
        assert float(np.min(uvs[:, 0])) >= left + pad_u - 1.0e-7
        assert float(np.max(uvs[:, 0])) <= right - pad_u + 1.0e-7
        assert float(np.min(uvs[:, 1])) >= bottom + pad_v - 1.0e-7
        assert float(np.max(uvs[:, 1])) <= top - pad_v + 1.0e-7

    # Component tile interiors cannot intersect, even before considering that
    # actual GNM islands occupy only part of each tile.
    for left_index, left_name in enumerate(atlas.component_names):
        l0, b0, r0, t0 = atlas.component_bounds[left_name]
        for right_name in atlas.component_names[left_index + 1 :]:
            l1, b1, r1, t1 = atlas.component_bounds[right_name]
            positive_area_overlap = min(r0, r1) > max(l0, l1) and min(
                t0, t1
            ) > max(b0, b1)
            assert not positive_area_overlap, (left_name, right_name)


def test_unseen_anatomy_has_distinct_nonzero_fallbacks(adapter: GNMAdapter) -> None:
    atlas = build_gnm_texture_atlas(adapter, 256)
    colors = atlas.generic_vertex_colors

    assert colors.shape == (17_821, 3)
    assert colors.dtype == np.uint8
    assert not np.any(np.all(colors == 0, axis=1))
    for group, expected in {
        "skin_exterior": (178, 142, 126),
        "scleras": (224, 219, 207),
        "irises": (91, 60, 42),
        "teeth": (229, 223, 207),
        "tongue": (154, 69, 82),
    }.items():
        indices = np.asarray(adapter.model.vertex_group_indices(group), dtype=np.int32)
        assert len(indices) > 0
        np.testing.assert_array_equal(
            np.unique(colors[indices], axis=0),
            np.asarray((expected,), dtype=np.uint8),
        )
    # GNM's gums and teeth groups overlap at their boundary; teeth take visual
    # precedence there, while gum-only vertices retain the gum fallback.
    gum_only = np.setdiff1d(
        adapter.model.vertex_group_indices("gums"),
        adapter.model.vertex_group_indices("teeth"),
    )
    np.testing.assert_array_equal(
        np.unique(colors[gum_only], axis=0),
        np.asarray(((164, 92, 96),), dtype=np.uint8),
    )


def test_static_glb_uses_repacked_uvs_and_preserves_vertical_orientation(
    tmp_path, adapter: GNMAdapter
) -> None:
    atlas = build_gnm_texture_atlas(adapter, 64)
    image = np.empty((64, 64, 4), dtype=np.uint8)
    image[:32] = (245, 35, 35, 255)  # top
    image[32:] = (35, 35, 245, 255)  # bottom
    texture = tmp_path / "vertical.png"
    Image.fromarray(image, mode="RGBA").save(texture)
    output = tmp_path / "repacked.glb"

    exported = export_gnm_glb(
        output,
        adapter,
        adapter.mesh(),
        texture_path=texture,
        triangle_uvs=atlas.triangle_uvs,
    )
    mapping = np.load(exported.mapping_path)
    split = split_triangle_corner_uvs(
        adapter.mesh(), adapter.triangles, atlas.triangle_uvs
    )
    np.testing.assert_array_equal(mapping["uvs_lower_left"], split.uvs)

    loaded = trimesh.load(output, force="scene", process=False)
    geometry = next(iter(loaded.geometry.values()))
    np.testing.assert_allclose(geometry.visual.uv, split.uvs, atol=1.0e-7)
    sampled = geometry.visual.to_color().vertex_colors[:, :3]
    high_v = geometry.visual.uv[:, 1] > 0.75
    low_v = geometry.visual.uv[:, 1] < 0.25
    assert np.any(high_v) and np.any(low_v)
    np.testing.assert_array_equal(
        np.unique(sampled[high_v], axis=0),
        np.asarray(((245, 35, 35),), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.unique(sampled[low_v], axis=0),
        np.asarray(((35, 35, 245),), dtype=np.uint8),
    )


def test_physical_front_camera_observes_both_eye_atlas_tiles(
    adapter: GNMAdapter,
) -> None:
    atlas = build_gnm_texture_atlas(adapter, 128)
    intrinsics = CameraIntrinsics(500.0, 500.0, 127.5, 127.5)
    camera = texture_camera_from_fit(
        PerspectiveCamera(0.0, 0.0, 0.0, 0.0, -0.28, 0.72, intrinsics)
    )
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    baked = bake_multiview_texture(
        adapter.mesh(),
        adapter.triangles,
        atlas.triangle_uvs,
        (image,),
        (camera,),
        masks=(np.ones((256, 256), dtype=bool),),
        generic_vertex_colors=atlas.generic_vertex_colors,
        texture_size=128,
        inpaint=False,
    )
    _, components = _component_texture_metrics(baked, atlas)

    assert components["skin"]["observed_texels"] > 1_000
    assert components["left_eye"]["observed_texels"] > 0
    assert components["right_eye"]["observed_texels"] > 0
