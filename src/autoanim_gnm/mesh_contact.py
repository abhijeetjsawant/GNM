"""Deterministic surface-contact checks for separate GNM mesh components."""

from __future__ import annotations

import numpy as np


def _segment_triangle_intersections(
    start: np.ndarray,
    stop: np.ndarray,
    triangles: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Vectorized Möller–Trumbore tests for paired segments and triangles."""

    direction = stop - start
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    cross_direction = np.cross(direction, edge_2)
    determinant = np.einsum("ij,ij->i", edge_1, cross_direction)
    valid = np.abs(determinant) > tolerance
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    offset = start - triangles[:, 0]
    barycentric_u = inverse * np.einsum("ij,ij->i", offset, cross_direction)
    cross_offset = np.cross(offset, edge_1)
    barycentric_v = inverse * np.einsum("ij,ij->i", direction, cross_offset)
    distance = inverse * np.einsum("ij,ij->i", edge_2, cross_offset)
    return (
        valid
        & (barycentric_u >= -tolerance)
        & (barycentric_v >= -tolerance)
        & (barycentric_u + barycentric_v <= 1.0 + tolerance)
        & (distance >= -tolerance)
        & (distance <= 1.0 + tolerance)
    )


def count_triangle_intersection_pairs(
    vertices: np.ndarray,
    triangles_a: np.ndarray,
    triangles_b: np.ndarray,
    *,
    tolerance: float = 1.0e-8,
) -> int:
    """Count triangle pairs whose surfaces cross.

    GNM's tongue and lips are independent open surfaces, so an inside/outside
    test is not defined. Triangle crossings are the useful failure signal:
    they catch visible interpenetration without rejecting close, non-crossing
    contact.
    """

    points = np.asarray(vertices, dtype=np.float64)
    first_indices = np.asarray(triangles_a, dtype=np.int64)
    second_indices = np.asarray(triangles_b, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("vertices must have shape (vertex_count, 3) and be finite")
    for name, indices in (
        ("triangles_a", first_indices),
        ("triangles_b", second_indices),
    ):
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError(f"{name} must have shape (triangle_count, 3)")
        if len(indices) and (
            int(np.min(indices)) < 0 or int(np.max(indices)) >= len(points)
        ):
            raise ValueError(f"{name} contains an out-of-range vertex index")
    if (
        not np.isfinite(tolerance)
        or isinstance(tolerance, bool)
        or tolerance <= 0.0
    ):
        raise ValueError("tolerance must be a finite positive number")
    if not len(first_indices) or not len(second_indices):
        return 0

    first = points[first_indices]
    second = points[second_indices]
    first_minimum = np.min(first, axis=1)
    first_maximum = np.max(first, axis=1)
    second_minimum = np.min(second, axis=1)
    second_maximum = np.max(second, axis=1)
    bounds_overlap = np.all(
        first_maximum[:, None, :] + tolerance
        >= second_minimum[None, :, :],
        axis=2,
    ) & np.all(
        second_maximum[None, :, :] + tolerance
        >= first_minimum[:, None, :],
        axis=2,
    )
    first_pair_indices, second_pair_indices = np.nonzero(bounds_overlap)
    if not len(first_pair_indices):
        return 0

    first_pairs = first[first_pair_indices]
    second_pairs = second[second_pair_indices]
    intersects = np.zeros(len(first_pairs), dtype=bool)
    for start, stop in ((0, 1), (1, 2), (2, 0)):
        intersects |= _segment_triangle_intersections(
            first_pairs[:, start],
            first_pairs[:, stop],
            second_pairs,
            tolerance=tolerance,
        )
        intersects |= _segment_triangle_intersections(
            second_pairs[:, start],
            second_pairs[:, stop],
            first_pairs,
            tolerance=tolerance,
        )
    return int(np.count_nonzero(intersects))


__all__ = ["count_triangle_intersection_pairs"]
