"""Topology-bound measurements of GNM's actual exterior mouth opening.

Sparse face landmarks are useful semantic controls, but they do not lie on the
rendered mouth-hole boundary.  This module discovers the exact 58-vertex
boundary loop from GNM Head v3 topology and measures its central signed gap and
projected opening area in a face-local, interocular-normalized frame.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .gnm_adapter import GNMAdapter


MOUTH_BOUNDARY_SET_SHA256 = (
    "bb7e41031e160dac968e98265c87b42463036b883920e66a5e8623836954e695"
)
MOUTH_BOUNDARY_CYCLE_SHA256 = (
    "25002f4e5651f0d54fbdaebbd7173114005da92d9bba728974b81f33c8e818d0"
)


class MouthGeometryError(ValueError):
    """The checked-in GNM topology or evaluated geometry is incompatible."""


@dataclass(frozen=True, slots=True)
class MouthBoundaryTopology:
    cycle: np.ndarray
    upper_path_positions: np.ndarray
    lower_path_positions: np.ndarray
    right_corner_position: int
    left_corner_position: int
    set_sha256: str
    cycle_sha256: str


@dataclass(frozen=True, slots=True)
class MouthBoundaryMeasurement:
    signed_central_gap_interocular: float
    opening_area_interocular_squared: float
    signed_central_gap_m: float
    opening_area_m2: float
    interocular_m: float
    upper_center_m: float
    lower_center_m: float


def _readonly(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


def _digest_i32(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype="<i4").tobytes()).hexdigest()


def discover_mouth_boundary(adapter: GNMAdapter) -> MouthBoundaryTopology:
    """Discover and fail-closed validate the GNM Head v3 exterior mouth loop."""

    triangles = np.asarray(adapter.triangles, dtype=np.int32)
    skin = np.asarray(adapter.vertex_group("skin_exterior"), dtype=np.float32) > 0.5
    skin_faces = triangles[np.all(skin[triangles], axis=1)]
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in skin_faces:
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_counts[tuple(sorted((int(left), int(right))))] += 1
    adjacency: dict[int, list[int]] = defaultdict(list)
    for (left, right), count in edge_counts.items():
        if count != 1:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)

    remaining = set(adjacency)
    selected: set[int] | None = None
    while remaining:
        seed = min(remaining)
        component = {seed}
        stack = [seed]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency[vertex]:
                if neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)
        remaining.difference_update(component)
        component_digest = _digest_i32(np.asarray(sorted(component), dtype=np.int32))
        if component_digest == MOUTH_BOUNDARY_SET_SHA256:
            selected = component
            break
    if selected is None or len(selected) != 58:
        raise MouthGeometryError(
            "GNM Head v3 exterior mouth-boundary topology was not found"
        )
    if any(len(adjacency[index]) != 2 for index in selected):
        raise MouthGeometryError("GNM mouth boundary is not one degree-two cycle")

    start = min(selected)
    first = min(adjacency[start])
    cycle = [start, first]
    previous, current = start, first
    while True:
        candidates = [value for value in adjacency[current] if value != previous]
        if len(candidates) != 1:
            raise MouthGeometryError("GNM mouth boundary traversal is ambiguous")
        following = candidates[0]
        if following == start:
            break
        if following in cycle or following not in selected:
            raise MouthGeometryError("GNM mouth boundary cycle is malformed")
        cycle.append(following)
        previous, current = current, following
    cycle_array = np.asarray(cycle, dtype=np.int32)
    cycle_digest = _digest_i32(cycle_array)
    if len(cycle_array) != 58 or cycle_digest != MOUTH_BOUNDARY_CYCLE_SHA256:
        raise MouthGeometryError("GNM mouth-boundary cycle ordering changed")

    neutral = np.asarray(
        adapter.model.template_vertex_positions,
        dtype=np.float64,
    )[cycle_array]
    right_corner_position = int(np.argmax(neutral[:, 0]))
    left_corner_position = int(np.argmin(neutral[:, 0]))
    if not (
        right_corner_position == 2
        and left_corner_position == 30
        and right_corner_position < left_corner_position
    ):
        raise MouthGeometryError("GNM mouth-corner topology changed")
    first_path = np.arange(
        right_corner_position, left_corner_position + 1, dtype=np.int32
    )
    second_path = np.concatenate(
        (
            np.arange(left_corner_position, len(cycle_array), dtype=np.int32),
            np.arange(0, right_corner_position + 1, dtype=np.int32),
        )
    )
    upper_lip = np.asarray(adapter.vertex_group("upper_lip"), dtype=np.float32) > 0.5
    lower_lip = np.asarray(adapter.vertex_group("lower_lip"), dtype=np.float32) > 0.5
    first_vertices = cycle_array[first_path]
    second_vertices = cycle_array[second_path]
    first_upper = int(np.count_nonzero(upper_lip[first_vertices]))
    first_lower = int(np.count_nonzero(lower_lip[first_vertices]))
    second_upper = int(np.count_nonzero(upper_lip[second_vertices]))
    second_lower = int(np.count_nonzero(lower_lip[second_vertices]))
    if first_upper > second_upper and first_lower < second_lower:
        upper_path, lower_path = first_path, second_path
    elif second_upper > first_upper and second_lower < first_lower:
        upper_path, lower_path = second_path, first_path
    else:
        raise MouthGeometryError(
            "GNM mouth-boundary paths do not match upper/lower lip ownership"
        )
    if (
        not np.all(upper_lip[cycle_array[upper_path[1:-1]]])
        or np.any(lower_lip[cycle_array[upper_path[1:-1]]])
        or not np.all(lower_lip[cycle_array[lower_path[1:-1]]])
        or np.any(upper_lip[cycle_array[lower_path[1:-1]]])
    ):
        raise MouthGeometryError(
            "GNM mouth-boundary interior ownership is not exclusive"
        )
    return MouthBoundaryTopology(
        cycle=_readonly(cycle_array),
        upper_path_positions=_readonly(upper_path),
        lower_path_positions=_readonly(lower_path),
        right_corner_position=right_corner_position,
        left_corner_position=left_corner_position,
        set_sha256=MOUTH_BOUNDARY_SET_SHA256,
        cycle_sha256=cycle_digest,
    )


def _face_coordinates(
    loop: np.ndarray,
    landmarks: np.ndarray,
    topology: MouthBoundaryTopology,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(loop, dtype=np.float64)
    compact = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (58, 3) or compact.shape != (68, 3):
        raise MouthGeometryError("Mouth measurement geometry has an invalid shape")
    if not np.isfinite(points).all() or not np.isfinite(compact).all():
        raise MouthGeometryError("Mouth measurement geometry is non-finite")
    left = points[topology.left_corner_position]
    right = points[topology.right_corner_position]
    right_axis = right - left
    width = float(np.linalg.norm(right_axis))
    if width <= 1.0e-8:
        raise MouthGeometryError("Mouth-corner axis is degenerate")
    right_axis /= width
    up_axis = compact[27] - compact[8]
    up_axis -= np.dot(up_axis, right_axis) * right_axis
    up_length = float(np.linalg.norm(up_axis))
    interocular = float(np.linalg.norm(compact[36] - compact[45]))
    if up_length <= 1.0e-8 or interocular <= 1.0e-8:
        raise MouthGeometryError("Face-local mouth frame is degenerate")
    up_axis /= up_length
    center = np.float64(0.5) * (left + right)
    offsets = points - center
    coordinates = np.column_stack((offsets @ right_axis, offsets @ up_axis))
    return coordinates, up_axis, interocular


def _center_intersection(coordinates: np.ndarray, path: np.ndarray) -> float:
    hits: list[float] = []
    for first, second in zip(path[:-1], path[1:], strict=True):
        x1, y1 = coordinates[int(first)]
        x2, y2 = coordinates[int(second)]
        if (x1 <= 0.0 < x2) or (x2 <= 0.0 < x1):
            hits.append(float(y1 + (y2 - y1) * (-x1) / (x2 - x1)))
    if len(hits) != 1:
        raise MouthGeometryError(
            f"Mouth boundary path has {len(hits)} central intersections, expected one"
        )
    return hits[0]


def measure_mouth_boundary(
    vertices: np.ndarray,
    landmarks: np.ndarray,
    topology: MouthBoundaryTopology,
) -> MouthBoundaryMeasurement:
    """Measure the rendered skin opening, not the sparse landmark proxy."""

    mesh = np.asarray(vertices, dtype=np.float64)
    if mesh.ndim != 2 or mesh.shape[1:] != (3,):
        raise MouthGeometryError("Mouth measurement requires a vertex-by-3 mesh")
    loop = (
        mesh
        if mesh.shape == (len(topology.cycle), 3)
        else mesh[np.asarray(topology.cycle, dtype=np.int64)]
    )
    coordinates, _, interocular = _face_coordinates(loop, landmarks, topology)
    upper = _center_intersection(coordinates, topology.upper_path_positions)
    lower = _center_intersection(coordinates, topology.lower_path_positions)
    signed_gap = upper - lower
    next_coordinates = np.roll(coordinates, -1, axis=0)
    signed_twice_area = np.sum(
        coordinates[:, 0] * next_coordinates[:, 1]
        - next_coordinates[:, 0] * coordinates[:, 1]
    )
    area = float(0.5 * abs(signed_twice_area))
    return MouthBoundaryMeasurement(
        signed_central_gap_interocular=float(signed_gap / interocular),
        opening_area_interocular_squared=float(area / (interocular * interocular)),
        signed_central_gap_m=float(signed_gap),
        opening_area_m2=area,
        interocular_m=interocular,
        upper_center_m=float(upper),
        lower_center_m=float(lower),
    )


def boundary_opening_displacement(
    vertices: np.ndarray,
    landmarks: np.ndarray,
    topology: MouthBoundaryTopology,
    *,
    lift_m: float,
) -> np.ndarray:
    """Return tapered upper/lower boundary motion for a requested central lift."""

    if not np.isfinite(lift_m) or lift_m < 0.0:
        raise MouthGeometryError("Boundary opening lift must be finite and non-negative")
    mesh = np.asarray(vertices, dtype=np.float64)
    loop = (
        mesh
        if mesh.shape == (len(topology.cycle), 3)
        else mesh[np.asarray(topology.cycle, dtype=np.int64)]
    )
    coordinates, up_axis, _ = _face_coordinates(loop, landmarks, topology)
    half_width = max(float(np.max(np.abs(coordinates[:, 0]))), 1.0e-8)
    taper = np.clip(1.0 - np.abs(coordinates[:, 0]) / half_width, 0.0, 1.0)
    desired = np.zeros((len(loop), 3), dtype=np.float64)
    upper = np.asarray(topology.upper_path_positions, dtype=np.int64)
    lower = np.asarray(topology.lower_path_positions, dtype=np.int64)
    desired[upper] += (
        np.float64(0.5 * lift_m) * taper[upper, None] * up_axis[None, :]
    )
    desired[lower] -= (
        np.float64(0.5 * lift_m) * taper[lower, None] * up_axis[None, :]
    )
    return np.asarray(desired, dtype=np.float32)


__all__ = [
    "MOUTH_BOUNDARY_CYCLE_SHA256",
    "MOUTH_BOUNDARY_SET_SHA256",
    "MouthBoundaryMeasurement",
    "MouthBoundaryTopology",
    "MouthGeometryError",
    "boundary_opening_displacement",
    "discover_mouth_boundary",
    "measure_mouth_boundary",
]
