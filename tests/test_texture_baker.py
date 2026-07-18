from __future__ import annotations

import numpy as np
import pytest

from autoanim_gnm.texture_baker import PerspectiveCamera, bake_multiview_texture


def _plane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        (
            (-1.0, -1.0, 2.0),
            (1.0, -1.0, 2.0),
            (1.0, 1.0, 2.0),
            (-1.0, 1.0, 2.0),
        ),
        dtype=np.float64,
    )
    # Clockwise as seen by the camera, so normals face camera (-Z).
    triangles = np.asarray(((0, 2, 1), (0, 3, 2)), dtype=np.int64)
    vertex_uvs = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    triangle_uvs = vertex_uvs[triangles]
    return vertices, triangles, triangle_uvs


def _camera(*, tx: float = 0.0) -> PerspectiveCamera:
    intrinsics = np.asarray(
        ((32.0, 0.0, 32.0), (0.0, 32.0, 32.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[0, 3] = tx
    return PerspectiveCamera(intrinsics, world_to_camera)


def _gradient_image(*, exposure: float = 1.0, bias: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[:65, :65]
    u = np.clip((x - 16.0) / 32.0, 0.0, 1.0)
    v = np.clip((y - 16.0) / 32.0, 0.0, 1.0)
    checker = ((np.floor(u * 8) + np.floor(v * 8)) % 2) * 0.3 + 0.2
    image = np.stack((u, v, checker), axis=-1)
    return np.clip(image * exposure + bias, 0.0, 1.0)


def test_triangle_corner_uvs_preserve_gradient_and_checker() -> None:
    vertices, triangles, triangle_uvs = _plane()
    result = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [_gradient_image()],
        [_camera()],
        texture_size=17,
        inpaint=False,
    )

    y, x = np.mgrid[:17, :17]
    expected_u = x / 16.0
    expected_v = 1.0 - y / 16.0
    assert np.max(np.abs(result.rgba[..., 0] / 255.0 - expected_u)) <= 1.0 / 255.0
    assert np.max(np.abs(result.rgba[..., 1] / 255.0 - expected_v)) <= 1.0 / 255.0
    expected_checker = (
        (np.floor(expected_u * 8) + np.floor(expected_v * 8)) % 2
    ) * 0.3 + 0.2
    # Avoid checker discontinuities where bilinear sampling is intentionally soft.
    interiors = ((x % 2) == 1) & ((y % 2) == 1)
    assert np.max(
        np.abs(result.rgba[..., 2][interiors] / 255.0 - expected_checker[interiors])
    ) <= 1.0 / 255.0
    assert result.metrics["observed_fraction"] == pytest.approx(1.0)


def test_occluded_texels_and_backfaces_are_rejected() -> None:
    vertices, triangles, triangle_uvs = _plane()
    # A front-facing occluder has degenerate UVs, so it participates in the
    # camera z-buffer without claiming atlas texels.
    occluder_vertices = np.asarray(
        ((-0.45, -0.45, 1.5), (0.45, 0.45, 1.5), (0.45, -0.45, 1.5))
    )
    vertices_with_occluder = np.vstack((vertices, occluder_vertices))
    triangles_with_occluder = np.vstack((triangles, (4, 5, 6)))
    uvs_with_occluder = np.concatenate(
        (triangle_uvs, np.zeros((1, 3, 2), dtype=np.float64)), axis=0
    )
    occluded = bake_multiview_texture(
        vertices_with_occluder,
        triangles_with_occluder,
        uvs_with_occluder,
        [_gradient_image()],
        [_camera()],
        texture_size=33,
        inpaint=False,
    )
    assert not occluded.observed[18:23, 18:23].all()
    assert occluded.generic[20, 20]

    backfacing = bake_multiview_texture(
        vertices,
        triangles[:, ::-1],
        triangle_uvs[:, ::-1],
        [_gradient_image()],
        [_camera()],
        texture_size=17,
        inpaint=False,
    )
    assert not np.any(backfacing.observed)
    assert np.all(backfacing.generic)


def test_triangle_crossing_near_plane_cannot_bypass_missing_z_buffer() -> None:
    # The conservative rasterizer drops the whole triangle when one vertex is
    # behind the near plane.  Its surviving world-space samples must not be
    # accepted merely because the missing z-buffer depth is infinity.
    vertices = np.asarray(
        ((-0.2, -0.2, 0.5), (0.8, -0.8, 2.0), (0.0, 0.8, 2.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 2, 1),), dtype=np.int64)
    triangle_uvs = np.asarray((((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)),))
    camera = _camera()
    camera = PerspectiveCamera(camera.intrinsics, camera.world_to_camera, near=1.0)

    result = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [_gradient_image()],
        [camera],
        texture_size=33,
        inpaint=False,
    )

    assert np.any(result.atlas_mask)
    assert not np.any(result.observed)
    assert np.all(result.generic)


def test_generic_vertex_colors_are_interpolated_and_remain_generic() -> None:
    vertices, triangles, triangle_uvs = _plane()
    fallback = np.asarray(
        ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)),
        dtype=np.uint8,
    )
    empty_mask = np.zeros((65, 65), dtype=np.float64)
    result = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [_gradient_image()],
        [_camera()],
        masks=[empty_mask],
        generic_vertex_colors=fallback,
        texture_size=17,
        inpaint=False,
    )

    # Internal v=0 is the bottom row of the top-to-bottom output image.
    np.testing.assert_allclose(result.rgba[-1, 0, :3], fallback[0], atol=1)
    np.testing.assert_allclose(result.rgba[-1, -1, :3], fallback[1], atol=1)
    np.testing.assert_allclose(result.rgba[0, -1, :3], fallback[2], atol=1)
    np.testing.assert_allclose(result.rgba[0, 0, :3], fallback[3], atol=1)
    assert np.all(result.generic)
    assert not np.any(result.observed | result.mirrored | result.inpainted)


def test_multiple_masked_views_increase_direct_coverage() -> None:
    vertices, triangles, triangle_uvs = _plane()
    image = _gradient_image()
    left_mask = np.zeros((65, 65), dtype=np.float64)
    left_mask[:, :34] = 1.0
    right_mask = np.zeros((65, 65), dtype=np.float64)
    right_mask[:, 31:] = 1.0
    left = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [image],
        [_camera()],
        masks=[left_mask],
        texture_size=33,
        inpaint=False,
    )
    combined = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [image, image],
        [_camera(), _camera()],
        masks=[left_mask, right_mask],
        texture_size=33,
        inpaint=False,
    )
    assert combined.metrics["observed_fraction"] > left.metrics["observed_fraction"] + 0.35
    assert combined.metrics["observed_fraction"] == pytest.approx(1.0)
    assert combined.metrics["overlap_fraction"] > 0.0


def test_overlap_harmonization_recovers_exposure_and_is_deterministic() -> None:
    vertices, triangles, triangle_uvs = _plane()
    reference = _gradient_image()
    dark = _gradient_image(exposure=0.6, bias=0.1)
    arguments = dict(
        vertices=vertices,
        triangles=triangles,
        triangle_uvs=triangle_uvs,
        images=[reference, dark],
        cameras=[_camera(), _camera()],
        texture_size=33,
        inpaint=False,
    )
    first = bake_multiview_texture(**arguments)
    second = bake_multiview_texture(**arguments)

    assert first.color_gain[1] == pytest.approx(np.full(3, 1.0 / 0.6), abs=0.035)
    assert first.color_bias[1] == pytest.approx(np.full(3, -0.1 / 0.6), abs=0.025)
    expected = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [reference],
        [_camera()],
        texture_size=33,
        inpaint=False,
    )
    assert np.max(np.abs(first.rgba.astype(int) - expected.rgba.astype(int))) <= 2
    for name in (
        "rgba",
        "confidence",
        "source_view",
        "observed",
        "mirrored",
        "inpainted",
        "generic",
        "atlas_mask",
        "triangle_index",
        "overlap_count",
        "color_gain",
        "color_bias",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert first.metrics == second.metrics


def test_every_texel_has_explicit_provenance() -> None:
    vertices, triangles, triangle_uvs = _plane()
    mask = np.zeros((65, 65), dtype=np.float64)
    mask[:, 16:33] = 1.0  # Observe left atlas half only.
    mask[31:34] = 0.0  # A small horizontal gap exercises labeled inpainting.
    result = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [_gradient_image()],
        [_camera()],
        masks=[mask],
        texture_size=(35, 41),
        mirror_fill=True,
        inpaint=True,
        maximum_inpaint_distance=2.0,
    )
    count = (
        result.observed.astype(np.uint8)
        + result.mirrored.astype(np.uint8)
        + result.inpainted.astype(np.uint8)
        + result.generic.astype(np.uint8)
    )
    assert np.all(count == 1)
    assert np.any(result.observed)
    assert np.any(result.mirrored)
    assert np.any(result.inpainted | result.generic)
    assert np.all(result.source_view[result.inpainted | result.generic] == -1)
    assert sum(
        float(result.metrics[name])
        for name in (
            "observed_fraction",
            "mirrored_fraction",
            "inpainted_fraction",
            "generic_fraction",
        )
    ) == pytest.approx(1.0)


def test_inpainting_does_not_cross_a_uv_seam() -> None:
    # Two disconnected mesh triangles are packed against each other in UV
    # space.  Only the left one passes the face mask; even an unlimited fill
    # radius must not propagate its color into the right UV island.
    vertices = np.asarray(
        (
            (-1.0, -1.0, 2.0),
            (0.0, -1.0, 2.0),
            (-1.0, 1.0, 2.0),
            (0.1, -1.0, 2.0),
            (1.0, -1.0, 2.0),
            (1.0, 1.0, 2.0),
        )
    )
    triangles = np.asarray(((0, 2, 1), (3, 5, 4)))
    triangle_uvs = np.asarray(
        (
            ((0.0, 0.0), (0.0, 1.0), (0.5, 0.0)),
            ((0.5, 0.0), (1.0, 1.0), (1.0, 0.0)),
        )
    )
    mask = np.zeros((65, 65), dtype=np.float64)
    mask[:, :33] = 1.0
    result = bake_multiview_texture(
        vertices,
        triangles,
        triangle_uvs,
        [_gradient_image()],
        [_camera()],
        masks=[mask],
        texture_size=33,
        maximum_inpaint_distance=100.0,
    )
    right_island = result.atlas_mask & (np.indices(result.atlas_mask.shape)[1] > 17)
    assert np.any(right_island)
    assert np.all(result.generic[right_island])
    assert not np.any(result.inpainted[right_island])


@pytest.mark.parametrize(
    ("change", "match"),
    (
        (lambda values: values | {"vertices": np.zeros((4, 2))}, "vertices"),
        (lambda values: values | {"triangles": np.asarray(((0, 1, 9),))}, "out-of-range"),
        (lambda values: values | {"triangle_uvs": np.zeros((1, 2, 2))}, "triangle_uvs"),
        (lambda values: values | {"triangle_uvs": np.full((2, 3, 2), 2.0)}, "in \\[0,1\\]"),
        (lambda values: values | {"images": []}, "at least one"),
        (lambda values: values | {"cameras": []}, "same length"),
        (lambda values: values | {"images": [np.zeros((4, 4))]}, "shape"),
        (lambda values: values | {"texture_size": 1}, "dimensions"),
    ),
)
def test_invalid_inputs_are_rejected(change, match: str) -> None:
    vertices, triangles, triangle_uvs = _plane()
    values = {
        "vertices": vertices,
        "triangles": triangles,
        "triangle_uvs": triangle_uvs,
        "images": [_gradient_image()],
        "cameras": [_camera()],
        "texture_size": 17,
    }
    with pytest.raises((TypeError, ValueError), match=match):
        bake_multiview_texture(**change(values))


def test_invalid_camera_and_auxiliary_fields_are_rejected() -> None:
    vertices, triangles, triangle_uvs = _plane()
    invalid_camera = PerspectiveCamera(np.eye(2), np.eye(4))
    with pytest.raises(ValueError, match="intrinsics"):
        bake_multiview_texture(
            vertices,
            triangles,
            triangle_uvs,
            [_gradient_image()],
            [invalid_camera],
            texture_size=17,
        )
    singular_camera = PerspectiveCamera(
        np.eye(3), np.zeros((4, 4), dtype=np.float64)
    )
    with pytest.raises(ValueError, match="last row|invertible"):
        bake_multiview_texture(
            vertices,
            triangles,
            triangle_uvs,
            [_gradient_image()],
            [singular_camera],
            texture_size=17,
        )
    with pytest.raises(ValueError, match="masks"):
        bake_multiview_texture(
            vertices,
            triangles,
            triangle_uvs,
            [_gradient_image()],
            [_camera()],
            masks=[np.ones((3, 3))],
            texture_size=17,
        )
