from __future__ import annotations

import numpy as np

from autoanim_gnm.body_binding import (
    clip_triangle_mesh,
    interpolate_clipped_attribute,
)


def _cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, 1, -1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def test_plane_clip_produces_one_exact_loop_and_interpolation() -> None:
    vertices, triangles = _cube()
    clipped = clip_triangle_mesh(
        vertices,
        triangles,
        plane_point=np.zeros(3),
        plane_normal=np.asarray((0.0, 1.0, 0.0)),
        keep_positive=False,
    )
    assert len(clipped.cut_loop) == 8
    assert np.max(np.abs(clipped.vertices[clipped.cut_loop, 1])) <= 1.0e-7
    reconstructed = interpolate_clipped_attribute(
        vertices, clipped.source_indices, clipped.source_weights
    )
    np.testing.assert_allclose(reconstructed, clipped.vertices, atol=1.0e-7)
    np.testing.assert_allclose(
        np.sum(clipped.source_weights, axis=1), 1.0, atol=1.0e-7
    )


def test_plane_clip_is_deterministic_and_keeps_requested_halfspace() -> None:
    vertices, triangles = _cube()
    arguments = {
        "plane_point": np.asarray((0.0, 0.2, 0.0)),
        "plane_normal": np.asarray((0.0, 1.0, 0.0)),
        "keep_positive": True,
    }
    first = clip_triangle_mesh(vertices, triangles, **arguments)
    second = clip_triangle_mesh(vertices, triangles, **arguments)
    assert np.array_equal(first.vertices, second.vertices)
    assert np.array_equal(first.triangles, second.triangles)
    assert np.array_equal(first.source_indices, second.source_indices)
    assert np.array_equal(first.source_weights, second.source_weights)
    assert float(np.min(first.vertices[:, 1])) >= 0.2 - 1.0e-6
