"""Deterministic, provenance-preserving multi-view texture baking.

The baker deliberately separates *measurement* from *hole handling*.  Pixels
projected from a visible mesh surface are marked ``observed``.  Optional atlas
mirroring and small-hole propagation are marked ``mirrored`` and ``inpainted``
respectively.  Everything else is ``generic``.  The four maps are exhaustive
and mutually exclusive, so downstream code cannot accidentally present an
unseen part of a head as photographic evidence.

Camera convention
-----------------
``world_to_camera`` maps homogeneous world points to a camera whose positive Z
axis points forward.  ``intrinsics`` maps camera X/Z and Y/Z to image pixel
coordinates.  Integer image coordinates denote pixel centres.  Input UVs use
the module's lower-left mesh convention (V=0 at the bottom); returned image
arrays are conventional top-to-bottom arrays.  The GLB exporter flips V once
when converting this representation to glTF's top-left texture coordinates.

This module is CPU-only and intentionally contains no GNM-specific imports.  It
can therefore be tested with small synthetic meshes and reused by the image
pipeline without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class PerspectiveCamera:
    """A calibrated pinhole camera."""

    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    near: float = 1.0e-5


@dataclass(frozen=True)
class TextureBakeResult:
    """Texture atlas plus confidence and audit maps.

    ``rgba`` is directly consumable by Pillow/OpenCV (uint8, top-to-bottom).
    ``source_view`` is the strongest direct source for observed texels, the
    inherited source for mirrored texels, and -1 for inpainted/generic texels.
    """

    rgba: np.ndarray
    confidence: np.ndarray
    source_view: np.ndarray
    observed: np.ndarray
    mirrored: np.ndarray
    inpainted: np.ndarray
    generic: np.ndarray
    atlas_mask: np.ndarray
    triangle_index: np.ndarray
    overlap_count: np.ndarray
    color_gain: np.ndarray
    color_bias: np.ndarray
    metrics: Mapping[str, float | int]


def _cross2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _triangle_barycentrics(
    points: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """Returns affine barycentrics for 2D ``points`` against ``triangle``."""

    denominator = _cross2(triangle[1] - triangle[0], triangle[2] - triangle[0])
    if abs(float(denominator)) <= 1.0e-14:
        return np.full((len(points), 3), np.nan, dtype=np.float64)
    first = _cross2(triangle[1] - points, triangle[2] - points) / denominator
    second = _cross2(triangle[2] - points, triangle[0] - points) / denominator
    return np.column_stack((first, second, 1.0 - first - second))


def _pixel_grid_for_triangle(
    triangle: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns integer X/Y coordinates and barycentrics inside a 2D triangle."""

    minimum = np.ceil(np.min(triangle, axis=0) - 1.0e-9).astype(np.int64)
    maximum = np.floor(np.max(triangle, axis=0) + 1.0e-9).astype(np.int64)
    x0 = max(int(minimum[0]), 0)
    y0 = max(int(minimum[1]), 0)
    x1 = min(int(maximum[0]), width - 1)
    y1 = min(int(maximum[1]), height - 1)
    if x0 > x1 or y0 > y1:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 3), dtype=np.float64)
    xs, ys = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.int64),
        np.arange(y0, y1 + 1, dtype=np.int64),
    )
    flat_x = xs.ravel()
    flat_y = ys.ravel()
    barycentrics = _triangle_barycentrics(
        np.column_stack((flat_x, flat_y)).astype(np.float64), triangle
    )
    inside = np.all(barycentrics >= -1.0e-8, axis=1)
    return flat_x[inside], flat_y[inside], barycentrics[inside]


def _as_rgb(image: np.ndarray, index: int) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"images[{index}] must have shape [height,width,3]")
    if value.shape[0] < 2 or value.shape[1] < 2:
        raise ValueError(f"images[{index}] must be at least 2x2 pixels")
    if value.dtype == np.bool_:
        raise ValueError(f"images[{index}] must contain RGB intensities")
    if np.issubdtype(value.dtype, np.integer):
        if np.min(value) < 0 or np.max(value) > 255:
            raise ValueError(f"images[{index}] integer values must be in [0,255]")
        result = value.astype(np.float64) / 255.0
    elif np.issubdtype(value.dtype, np.floating):
        result = value.astype(np.float64)
        if not np.all(np.isfinite(result)) or np.min(result) < 0.0 or np.max(result) > 1.0:
            raise ValueError(f"images[{index}] float values must be finite and in [0,1]")
    else:
        raise ValueError(f"images[{index}] has an unsupported dtype")
    return result


def _as_camera(camera: PerspectiveCamera, index: int) -> PerspectiveCamera:
    if not isinstance(camera, PerspectiveCamera):
        raise TypeError(f"cameras[{index}] must be a PerspectiveCamera")
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    transform = np.asarray(camera.world_to_camera, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"cameras[{index}].intrinsics must have shape [3,3]")
    if transform.shape != (4, 4):
        raise ValueError(f"cameras[{index}].world_to_camera must have shape [4,4]")
    if not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(transform)):
        raise ValueError(f"cameras[{index}] must contain finite values")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError(f"cameras[{index}] focal lengths must be positive")
    if abs(float(np.linalg.det(intrinsics))) <= 1.0e-12:
        raise ValueError(f"cameras[{index}].intrinsics must be invertible")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-8):
        raise ValueError(f"cameras[{index}].world_to_camera has an invalid last row")
    if abs(float(np.linalg.det(transform))) <= 1.0e-12:
        raise ValueError(f"cameras[{index}].world_to_camera must be invertible")
    if not np.isfinite(camera.near) or camera.near <= 0.0:
        raise ValueError(f"cameras[{index}].near must be positive")
    return PerspectiveCamera(intrinsics, transform, float(camera.near))


def _as_field(
    field: np.ndarray | float | None,
    shape: tuple[int, int],
    name: str,
    index: int,
    *,
    binary: bool,
) -> np.ndarray:
    if field is None:
        return np.ones(shape, dtype=np.float64)
    value = np.asarray(field, dtype=np.float64)
    if value.ndim == 0:
        value = np.full(shape, float(value), dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{name}[{index}] must be scalar or match the image height/width")
    if not np.all(np.isfinite(value)) or np.min(value) < 0.0 or np.max(value) > 1.0:
        raise ValueError(f"{name}[{index}] values must be finite and in [0,1]")
    if binary:
        return (value >= 0.5).astype(np.float64)
    return value


def _bilinear(field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Bilinearly samples a 2D or 3D field at pixel-centre coordinates."""

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, field.shape[1] - 1)
    y1 = np.minimum(y0 + 1, field.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    if field.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]
    return (
        field[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + field[y0, x1] * wx * (1.0 - wy)
        + field[y1, x0] * (1.0 - wx) * wy
        + field[y1, x1] * wx * wy
    )


def _project(
    points: np.ndarray, camera: PerspectiveCamera
) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    camera_points = (camera.world_to_camera @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = camera_points[:, :2] / depth[:, None]
    pixel_h = (
        camera.intrinsics
        @ np.column_stack((normalized, np.ones(len(points), dtype=np.float64))).T
    ).T
    return pixel_h[:, :2] / pixel_h[:, 2:3], depth


def _camera_center(camera: PerspectiveCamera) -> np.ndarray:
    inverse = np.linalg.inv(camera.world_to_camera)
    return inverse[:3, 3] / inverse[3, 3]


def _triangle_island_ids(
    triangle_uvs: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Finds UV islands from shared, equal triangle-corner UV edges.

    Vertex indices cannot identify a UV island because UV seams deliberately
    duplicate coordinates per corner.  Conversely, pixel connected components
    can merge two packed islands that merely touch.  Exact UV-edge connectivity
    (with deterministic 12-decimal canonicalization) captures the intended
    topology and keeps inpainting from crossing a seam.
    """

    parent = np.arange(len(triangle_uvs), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    edges: dict[
        tuple[
            tuple[int, int],
            tuple[tuple[float, float], tuple[float, float]],
        ],
        int,
    ] = {}
    for triangle_index, (triangle, vertex_indices) in enumerate(
        zip(triangle_uvs, triangles, strict=True)
    ):
        rounded = np.round(triangle, decimals=12)
        for first, second in ((0, 1), (1, 2), (2, 0)):
            endpoint_a = tuple(float(value) for value in rounded[first])
            endpoint_b = tuple(float(value) for value in rounded[second])
            vertex_edge = tuple(
                sorted((int(vertex_indices[first]), int(vertex_indices[second])))
            )
            uv_edge = tuple(sorted((endpoint_a, endpoint_b)))
            key = (vertex_edge, uv_edge)
            previous = edges.get(key)
            if previous is None:
                edges[key] = triangle_index
            else:
                union(previous, triangle_index)
    roots = np.asarray([find(index) for index in range(len(triangle_uvs))])
    unique_roots = {root: index for index, root in enumerate(sorted(set(roots.tolist())))}
    return np.asarray([unique_roots[int(root)] for root in roots], dtype=np.int32)


def _uv_raster(
    triangle_uvs: np.ndarray, triangles: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangle_map = np.full((height, width), -1, dtype=np.int32)
    barycentric_map = np.zeros((height, width, 3), dtype=np.float64)
    for triangle_index, uv in enumerate(triangle_uvs):
        # Internal mesh UVs are bottom-up; image arrays are top-down.
        pixel_uv = np.column_stack(
            (uv[:, 0] * (width - 1), (1.0 - uv[:, 1]) * (height - 1))
        )
        xs, ys, barycentrics = _pixel_grid_for_triangle(pixel_uv, width, height)
        if len(xs) == 0:
            continue
        unassigned = triangle_map[ys, xs] < 0
        if np.any(unassigned):
            selected_x = xs[unassigned]
            selected_y = ys[unassigned]
            triangle_map[selected_y, selected_x] = triangle_index
            barycentric_map[selected_y, selected_x] = barycentrics[unassigned]
    triangle_islands = _triangle_island_ids(triangle_uvs, triangles)
    island_map = np.full((height, width), -1, dtype=np.int32)
    occupied = triangle_map >= 0
    island_map[occupied] = triangle_islands[triangle_map[occupied]]
    return triangle_map, barycentric_map, island_map


def _depth_raster(
    projected: np.ndarray,
    depths: np.ndarray,
    triangles: np.ndarray,
    width: int,
    height: int,
    near: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    triangle_buffer = np.full((height, width), -1, dtype=np.int32)
    for triangle_index, vertex_indices in enumerate(triangles):
        triangle_depth = depths[vertex_indices]
        if np.any(triangle_depth <= near):
            # Near-plane clipping is intentionally conservative.  A calibrated
            # face capture should never intersect the near plane.
            continue
        triangle_2d = projected[vertex_indices]
        xs, ys, screen_barycentrics = _pixel_grid_for_triangle(
            triangle_2d, width, height
        )
        if len(xs) == 0:
            continue
        inverse_depth = np.sum(
            screen_barycentrics / triangle_depth[None, :], axis=1
        )
        valid = inverse_depth > 0.0
        if not np.any(valid):
            continue
        xs = xs[valid]
        ys = ys[valid]
        candidate_depth = 1.0 / inverse_depth[valid]
        old_depth = depth_buffer[ys, xs]
        old_triangle = triangle_buffer[ys, xs]
        closer = candidate_depth < old_depth - 1.0e-12
        tie = (np.abs(candidate_depth - old_depth) <= 1.0e-12) & (
            (old_triangle < 0) | (triangle_index < old_triangle)
        )
        update = closer | tie
        if np.any(update):
            update_x = xs[update]
            update_y = ys[update]
            depth_buffer[update_y, update_x] = candidate_depth[update]
            triangle_buffer[update_y, update_x] = triangle_index
    return depth_buffer, triangle_buffer


def _robust_affine(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fits target ~= gain*source+bias using deterministic Huber IRLS."""

    gains = np.ones(3, dtype=np.float64)
    biases = np.zeros(3, dtype=np.float64)
    base_weights = np.maximum(weights.astype(np.float64), 1.0e-8)
    for channel in range(3):
        x = source[:, channel]
        y = target[:, channel]
        design = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
        robust = np.ones(len(x), dtype=np.float64)
        solution = np.asarray((1.0, 0.0), dtype=np.float64)
        for _ in range(10):
            combined = base_weights * robust
            normal = design.T @ (combined[:, None] * design)
            rhs = design.T @ (combined * y)
            # A weak identity prior handles nearly uniform overlap without
            # inventing an extreme gain/bias pair.
            normal += np.diag((1.0e-5, 1.0e-7))
            rhs += np.asarray((1.0e-5, 0.0))
            try:
                updated = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                break
            residual = y - design @ updated
            centre = np.median(residual)
            scale = 1.4826 * np.median(np.abs(residual - centre)) + 1.0e-6
            magnitude = np.abs(residual - centre)
            robust = np.minimum(1.0, (1.5 * scale) / np.maximum(magnitude, 1.0e-12))
            if np.max(np.abs(updated - solution)) < 1.0e-9:
                solution = updated
                break
            solution = updated
        gains[channel] = np.clip(solution[0], 0.25, 4.0)
        biases[channel] = np.clip(solution[1], -0.5, 0.5)
    return gains, biases


def _harmonize_views(
    samples: np.ndarray,
    weights: np.ndarray,
    minimum_overlap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    view_count = len(samples)
    gains = np.ones((view_count, 3), dtype=np.float64)
    biases = np.zeros((view_count, 3), dtype=np.float64)
    if view_count <= 1:
        return gains, biases, view_count

    coverage_weight = np.sum(weights, axis=(1, 2))
    anchor = int(np.argmax(coverage_weight))
    calibrated = {anchor}
    remaining = set(range(view_count)) - calibrated
    while remaining:
        best: tuple[int, int, int] | None = None
        for source_index in sorted(remaining):
            for target_index in sorted(calibrated):
                overlap = (weights[source_index] > 0.0) & (weights[target_index] > 0.0)
                count = int(np.count_nonzero(overlap))
                candidate = (count, -source_index, -target_index)
                if best is None or candidate > best:
                    best = candidate
        assert best is not None
        overlap_count, negative_source, negative_target = best
        if overlap_count < minimum_overlap:
            break
        source_index = -negative_source
        target_index = -negative_target
        overlap = (weights[source_index] > 0.0) & (weights[target_index] > 0.0)
        source_values = samples[source_index][overlap]
        target_values = (
            samples[target_index][overlap] * gains[target_index]
            + biases[target_index]
        )
        overlap_weights = np.sqrt(
            weights[source_index][overlap] * weights[target_index][overlap]
        )
        gains[source_index], biases[source_index] = _robust_affine(
            source_values, target_values, overlap_weights
        )
        calibrated.add(source_index)
        remaining.remove(source_index)
    return gains, biases, len(calibrated)


def _nearest_depth(
    depth_buffer: np.ndarray,
    triangle_buffer: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples the closest of the four neighbouring z-buffer pixels."""

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, depth_buffer.shape[1] - 1)
    y1 = np.minimum(y0 + 1, depth_buffer.shape[0] - 1)
    all_depths = np.column_stack(
        (
            depth_buffer[y0, x0],
            depth_buffer[y0, x1],
            depth_buffer[y1, x0],
            depth_buffer[y1, x1],
        )
    )
    all_triangles = np.column_stack(
        (
            triangle_buffer[y0, x0],
            triangle_buffer[y0, x1],
            triangle_buffer[y1, x0],
            triangle_buffer[y1, x1],
        )
    )
    selection = np.argmin(all_depths, axis=1)
    row = np.arange(len(coordinates))
    return all_depths[row, selection], all_triangles[row, selection]


def _fill_small_holes(
    color: np.ndarray,
    confidence: np.ndarray,
    source_view: np.ndarray,
    island_map: np.ndarray,
    known: np.ndarray,
    maximum_distance: float,
) -> np.ndarray:
    """Nearest-boundary fills small holes independently within each UV island."""

    filled = np.zeros_like(island_map, dtype=bool)
    for component_index in np.unique(island_map[island_map >= 0]):
        component = island_map == component_index
        component_known = component & known
        component_missing = component & ~known
        if not np.any(component_known) or not np.any(component_missing):
            continue
        # Zeros are the only possible sources.  Pixels outside this component
        # stay non-zero, preventing one UV island from bleeding into another.
        distance_input = np.ones_like(component, dtype=np.uint8)
        distance_input[component_known] = 0
        distances, indices = distance_transform_edt(
            distance_input, return_indices=True
        )
        selected = component_missing & (distances <= maximum_distance)
        if not np.any(selected):
            continue
        source_y = indices[0][selected]
        source_x = indices[1][selected]
        color[selected] = color[source_y, source_x]
        confidence[selected] = confidence[source_y, source_x] * np.exp(
            -distances[selected] / max(maximum_distance, 1.0)
        )
        source_view[selected] = -1
        filled[selected] = True
        known[selected] = True
    return filled


def bake_multiview_texture(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_uvs: np.ndarray,
    images: Sequence[np.ndarray],
    cameras: Sequence[PerspectiveCamera],
    *,
    texture_size: int | tuple[int, int] = 1024,
    masks: Sequence[np.ndarray | float | None] | None = None,
    confidences: Sequence[np.ndarray | float | None] | None = None,
    generic_color: tuple[int, int, int] = (178, 142, 126),
    generic_vertex_colors: np.ndarray | None = None,
    mirror_fill: bool = False,
    inpaint: bool = True,
    maximum_inpaint_distance: float = 8.0,
    minimum_color_overlap: int = 16,
    visibility_epsilon: float = 1.0e-3,
    incidence_power: float = 2.0,
) -> TextureBakeResult:
    """Bakes calibrated RGB views into a triangle-corner UV atlas.

    Args:
      vertices: World-space mesh vertices, shape ``[V,3]``.
      triangles: Vertex indices, shape ``[T,3]``.
      triangle_uvs: Independent UV coordinate for every triangle corner,
        shape ``[T,3,2]``.  This preserves GNM UV seams.
      images: Calibrated RGB images (uint8 [0,255] or float [0,1]).
      cameras: One pinhole camera per image.
      masks: Optional per-view face/skin masks.  Pixels outside a mask are never
        baked; callers should use these to reject hair, hands, and occluders.
      confidences: Optional scalar or per-pixel measurement confidences.
      texture_size: Integer square size or ``(height, width)``.
      generic_color: Explicit fallback RGB used where no evidence/fill exists.
      generic_vertex_colors: Optional uint8 per-vertex anatomical fallback,
        shape ``[V,3]``.  Values are barycentrically interpolated in atlas
        space and take precedence over ``generic_color``.  They remain labeled
        ``generic`` in the provenance maps.
      mirror_fill: Fill a missing texel from its horizontally mirrored atlas
        location only when that location was directly observed.
      inpaint: Propagate nearest observed/mirrored color into *small* holes in
        the same connected UV island.  These texels remain labeled inpainted.

    Returns:
      A :class:`TextureBakeResult` whose provenance maps are exhaustive.
    """

    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles)
    uv_array = np.asarray(triangle_uvs, dtype=np.float64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or len(vertex_array) < 3:
        raise ValueError("vertices must have shape [vertex_count,3]")
    if not np.all(np.isfinite(vertex_array)):
        raise ValueError("vertices must contain finite values")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3 or len(triangle_array) < 1:
        raise ValueError("triangles must have shape [triangle_count,3]")
    if not np.issubdtype(triangle_array.dtype, np.integer):
        if not np.all(np.equal(triangle_array, np.floor(triangle_array))):
            raise ValueError("triangles must contain integer vertex indices")
    triangle_array = triangle_array.astype(np.int64)
    if np.min(triangle_array) < 0 or np.max(triangle_array) >= len(vertex_array):
        raise ValueError("triangles contain an out-of-range vertex index")
    if uv_array.shape != (len(triangle_array), 3, 2):
        raise ValueError("triangle_uvs must have shape [triangle_count,3,2]")
    if (
        not np.all(np.isfinite(uv_array))
        or np.min(uv_array) < 0.0
        or np.max(uv_array) > 1.0
    ):
        raise ValueError("triangle_uvs must be finite and in [0,1]")

    if isinstance(texture_size, int):
        texture_height = texture_width = texture_size
    else:
        if len(texture_size) != 2:
            raise ValueError("texture_size must be an integer or (height,width)")
        texture_height, texture_width = texture_size
    if (
        not isinstance(texture_height, (int, np.integer))
        or not isinstance(texture_width, (int, np.integer))
        or texture_height < 2
        or texture_width < 2
    ):
        raise ValueError("texture dimensions must be integers >= 2")
    texture_height = int(texture_height)
    texture_width = int(texture_width)

    if len(images) == 0:
        raise ValueError("at least one image and camera are required")
    if len(images) != len(cameras):
        raise ValueError("images and cameras must have the same length")
    if masks is not None and len(masks) != len(images):
        raise ValueError("masks must have one entry per image")
    if confidences is not None and len(confidences) != len(images):
        raise ValueError("confidences must have one entry per image")
    if len(generic_color) != 3 or any(
        not isinstance(value, (int, np.integer)) or value < 0 or value > 255
        for value in generic_color
    ):
        raise ValueError("generic_color must contain three integers in [0,255]")
    generic_vertices: np.ndarray | None = None
    if generic_vertex_colors is not None:
        generic_vertices = np.asarray(generic_vertex_colors)
        if generic_vertices.shape != (len(vertex_array), 3):
            raise ValueError("generic_vertex_colors must have shape [vertex_count,3]")
        if not np.issubdtype(generic_vertices.dtype, np.integer):
            raise ValueError("generic_vertex_colors must contain integer RGB values")
        if np.min(generic_vertices) < 0 or np.max(generic_vertices) > 255:
            raise ValueError("generic_vertex_colors values must be in [0,255]")
        generic_vertices = generic_vertices.astype(np.float64) / 255.0
    if maximum_inpaint_distance < 0.0 or not np.isfinite(maximum_inpaint_distance):
        raise ValueError("maximum_inpaint_distance must be finite and non-negative")
    if minimum_color_overlap < 1:
        raise ValueError("minimum_color_overlap must be positive")
    if visibility_epsilon <= 0.0 or not np.isfinite(visibility_epsilon):
        raise ValueError("visibility_epsilon must be finite and positive")
    if incidence_power < 0.0 or not np.isfinite(incidence_power):
        raise ValueError("incidence_power must be finite and non-negative")

    rgb_images = [_as_rgb(image, index) for index, image in enumerate(images)]
    camera_values = [_as_camera(camera, index) for index, camera in enumerate(cameras)]
    mask_values = [
        _as_field(
            None if masks is None else masks[index],
            rgb.shape[:2],
            "masks",
            index,
            binary=True,
        )
        for index, rgb in enumerate(rgb_images)
    ]
    confidence_values = [
        _as_field(
            None if confidences is None else confidences[index],
            rgb.shape[:2],
            "confidences",
            index,
            binary=False,
        )
        for index, rgb in enumerate(rgb_images)
    ]

    triangle_map, barycentric_map, island_map = _uv_raster(
        uv_array, triangle_array, texture_width, texture_height
    )
    atlas_mask = triangle_map >= 0
    view_count = len(rgb_images)
    samples = np.zeros(
        (view_count, texture_height, texture_width, 3), dtype=np.float64
    )
    sample_weights = np.zeros(
        (view_count, texture_height, texture_width), dtype=np.float64
    )

    triangle_vertices = vertex_array[triangle_array]
    face_normals_area = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    normal_lengths = np.linalg.norm(face_normals_area, axis=1)
    valid_normals = normal_lengths > 1.0e-14
    face_normals = np.zeros_like(face_normals_area)
    face_normals[valid_normals] = (
        face_normals_area[valid_normals] / normal_lengths[valid_normals, None]
    )
    # Area-weighted vertex normals avoid visible confidence seams between
    # adjacent triangles on smooth cheeks, eyes, and forehead.  The face
    # normal remains a defensive fallback for a zero-length interpolant.
    vertex_normals = np.zeros_like(vertex_array)
    for corner in range(3):
        np.add.at(vertex_normals, triangle_array[:, corner], face_normals_area)
    vertex_normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    valid_vertex_normals = vertex_normal_lengths > 1.0e-14
    vertex_normals[valid_vertex_normals] /= vertex_normal_lengths[
        valid_vertex_normals, None
    ]

    # Group occupied atlas pixels by triangle once.  Re-scanning the full atlas
    # for every GNM triangle would be O(texture_pixels * triangle_count).
    flat_triangles = triangle_map.ravel()
    occupied_flat = np.flatnonzero(flat_triangles >= 0)
    occupied_order = occupied_flat[
        np.argsort(flat_triangles[occupied_flat], kind="stable")
    ]
    triangle_counts = np.bincount(
        flat_triangles[occupied_flat], minlength=len(triangle_array)
    )
    triangle_offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(triangle_counts))
    )

    for view_index, (image, camera, mask, confidence_field) in enumerate(
        zip(rgb_images, camera_values, mask_values, confidence_values, strict=True)
    ):
        projected_vertices, vertex_depths = _project(vertex_array, camera)
        depth_buffer, triangle_buffer = _depth_raster(
            projected_vertices,
            vertex_depths,
            triangle_array,
            image.shape[1],
            image.shape[0],
            camera.near,
        )
        centre = _camera_center(camera)
        for triangle_index, vertex_indices in enumerate(triangle_array):
            if not valid_normals[triangle_index]:
                continue
            start = triangle_offsets[triangle_index]
            stop = triangle_offsets[triangle_index + 1]
            if start == stop:
                continue
            flat_pixels = occupied_order[start:stop]
            ys, xs = np.divmod(flat_pixels, texture_width)
            barycentrics = barycentric_map[ys, xs]
            world_points = barycentrics @ vertex_array[vertex_indices]
            pixel_points, point_depths = _project(world_points, camera)
            within = (
                (point_depths > camera.near)
                & (pixel_points[:, 0] >= 0.0)
                & (pixel_points[:, 0] <= image.shape[1] - 1)
                & (pixel_points[:, 1] >= 0.0)
                & (pixel_points[:, 1] <= image.shape[0] - 1)
            )
            if not np.any(within):
                continue

            directions = centre[None, :] - world_points
            direction_norm = np.linalg.norm(directions, axis=1)
            incidence = np.zeros(len(world_points), dtype=np.float64)
            direction_valid = direction_norm > 1.0e-14
            surface_normals = barycentrics @ vertex_normals[vertex_indices]
            surface_lengths = np.linalg.norm(surface_normals, axis=1)
            invalid_surface = surface_lengths <= 1.0e-14
            surface_normals[~invalid_surface] /= surface_lengths[~invalid_surface, None]
            surface_normals[invalid_surface] = face_normals[triangle_index]
            incidence[direction_valid] = np.sum(
                directions[direction_valid] * surface_normals[direction_valid], axis=1
            ) / direction_norm[direction_valid]
            within &= incidence > 1.0e-6
            if not np.any(within):
                continue

            selected = np.flatnonzero(within)
            selected_pixels = pixel_points[selected]
            nearest_depth, nearest_triangle = _nearest_depth(
                depth_buffer, triangle_buffer, selected_pixels
            )
            tolerance = visibility_epsilon * np.maximum(1.0, point_depths[selected])
            # A triangle crossing the near plane is conservatively omitted by
            # the z-buffer.  Missing depth used to compare as ``point <= inf``
            # and was therefore incorrectly treated as visible.  A finite,
            # assigned z-buffer sample is now a mandatory visibility witness.
            rasterized = (nearest_triangle >= 0) & np.isfinite(nearest_depth)
            visible = rasterized & (
                (nearest_triangle == triangle_index)
                | (point_depths[selected] <= nearest_depth + tolerance)
            )
            if not np.any(visible):
                continue
            selected = selected[visible]
            selected_pixels = pixel_points[selected]

            projected_triangle = projected_vertices[vertex_indices]
            projected_area = 0.5 * abs(
                float(
                    _cross2(
                        projected_triangle[1] - projected_triangle[0],
                        projected_triangle[2] - projected_triangle[0],
                    )
                )
            )
            uv_pixels = np.column_stack(
                (
                    uv_array[triangle_index, :, 0] * (texture_width - 1),
                    (1.0 - uv_array[triangle_index, :, 1]) * (texture_height - 1),
                )
            )
            texture_area = 0.5 * abs(
                float(_cross2(uv_pixels[1] - uv_pixels[0], uv_pixels[2] - uv_pixels[0]))
            )
            resolution = np.sqrt(projected_area / max(texture_area, 1.0e-12))
            resolution_weight = resolution / (1.0 + resolution)
            mask_weight = _bilinear(mask, selected_pixels)
            measurement_confidence = _bilinear(confidence_field, selected_pixels)
            weight = (
                np.power(np.clip(incidence[selected], 0.0, 1.0), incidence_power)
                * resolution_weight
                * mask_weight
                * measurement_confidence
            )
            accepted = weight > 1.0e-12
            if not np.any(accepted):
                continue
            selected = selected[accepted]
            selected_pixels = selected_pixels[accepted]
            ys_selected = ys[selected]
            xs_selected = xs[selected]
            samples[view_index, ys_selected, xs_selected] = _bilinear(
                image, selected_pixels
            )
            sample_weights[view_index, ys_selected, xs_selected] = weight[accepted]

    gains, biases, harmonized_views = _harmonize_views(
        samples, sample_weights, minimum_color_overlap
    )
    corrected = np.clip(
        samples * gains[:, None, None, :] + biases[:, None, None, :], 0.0, 1.0
    )
    total_weight = np.sum(sample_weights, axis=0)
    observed = total_weight > 0.0
    color = np.empty((texture_height, texture_width, 3), dtype=np.float64)
    color[:] = np.asarray(generic_color, dtype=np.float64) / 255.0
    if generic_vertices is not None and np.any(atlas_mask):
        generic_triangles = triangle_array[triangle_map[atlas_mask]]
        generic_barycentrics = barycentric_map[atlas_mask]
        color[atlas_mask] = np.sum(
            generic_vertices[generic_triangles]
            * generic_barycentrics[:, :, None],
            axis=1,
        )
    if np.any(observed):
        color[observed] = (
            np.sum(corrected * sample_weights[..., None], axis=0)[observed]
            / total_weight[observed, None]
        )
    confidence_map = 1.0 - np.prod(1.0 - np.clip(sample_weights, 0.0, 1.0), axis=0)
    source_view = np.full((texture_height, texture_width), -1, dtype=np.int16)
    if np.any(observed):
        strongest = np.argmax(sample_weights, axis=0).astype(np.int16)
        source_view[observed] = strongest[observed]
    overlap_count = np.count_nonzero(sample_weights > 0.0, axis=0).astype(np.uint16)

    mirrored = np.zeros_like(atlas_mask)
    known = observed.copy()
    if mirror_fill:
        source_x = np.arange(texture_width - 1, -1, -1)
        mirrored_observed = observed[:, source_x]
        targets = atlas_mask & ~known & mirrored_observed
        if np.any(targets):
            mirrored_color = color[:, source_x]
            mirrored_confidence = confidence_map[:, source_x]
            mirrored_source = source_view[:, source_x]
            color[targets] = mirrored_color[targets]
            confidence_map[targets] = mirrored_confidence[targets] * 0.5
            source_view[targets] = mirrored_source[targets]
            mirrored[targets] = True
            known[targets] = True

    inpainted = np.zeros_like(atlas_mask)
    if inpaint and maximum_inpaint_distance > 0.0:
        inpainted = _fill_small_holes(
            color,
            confidence_map,
            source_view,
            island_map,
            known,
            maximum_inpaint_distance,
        )
    generic = ~(observed | mirrored | inpainted)
    # Exhaustive, exclusive provenance is an invariant—not a best effort.
    provenance_sum = (
        observed.astype(np.uint8)
        + mirrored.astype(np.uint8)
        + inpainted.astype(np.uint8)
        + generic.astype(np.uint8)
    )
    if not np.all(provenance_sum == 1):  # pragma: no cover - defensive invariant
        raise RuntimeError("texture provenance maps are not exhaustive and exclusive")

    rgba = np.empty((texture_height, texture_width, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    atlas_count = int(np.count_nonzero(atlas_mask))
    denominator = max(atlas_count, 1)
    metrics: dict[str, float | int] = {
        "texture_width": texture_width,
        "texture_height": texture_height,
        "view_count": view_count,
        "harmonized_view_count": harmonized_views,
        "atlas_texels": atlas_count,
        "observed_texels": int(np.count_nonzero(observed & atlas_mask)),
        "mirrored_texels": int(np.count_nonzero(mirrored & atlas_mask)),
        "inpainted_texels": int(np.count_nonzero(inpainted & atlas_mask)),
        "generic_atlas_texels": int(np.count_nonzero(generic & atlas_mask)),
        "observed_fraction": float(np.count_nonzero(observed & atlas_mask) / denominator),
        "mirrored_fraction": float(np.count_nonzero(mirrored & atlas_mask) / denominator),
        "inpainted_fraction": float(np.count_nonzero(inpainted & atlas_mask) / denominator),
        "generic_fraction": float(np.count_nonzero(generic & atlas_mask) / denominator),
        "overlap_fraction": float(
            np.count_nonzero((overlap_count > 1) & atlas_mask) / denominator
        ),
        "mean_observed_confidence": float(
            np.mean(confidence_map[observed]) if np.any(observed) else 0.0
        ),
    }
    return TextureBakeResult(
        rgba=rgba,
        confidence=confidence_map.astype(np.float32),
        source_view=source_view,
        observed=observed,
        mirrored=mirrored,
        inpainted=inpainted,
        generic=generic,
        atlas_mask=atlas_mask,
        triangle_index=triangle_map,
        overlap_count=overlap_count,
        color_gain=gains.astype(np.float32),
        color_bias=biases.astype(np.float32),
        metrics=metrics,
    )


__all__ = [
    "PerspectiveCamera",
    "TextureBakeResult",
    "bake_multiview_texture",
]
