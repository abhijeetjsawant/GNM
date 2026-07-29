"""Immutable geometric binding between one GNM identity and one MPFB body.

The provider's ``neck_seam_vertex_indices`` are an unordered search slab, not
an anatomical boundary.  This module therefore intersects both neutral meshes
with semantic neck planes, records the exact clipping interpolation, and
computes a bounded similarity transform between the resulting loops.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
from scipy.spatial import cKDTree

from .body import CANONICAL_HUMANOID
from .body_provider import load_and_validate_body_asset, sha256_file
from .serialization import write_json, write_npz


GNM_BODY_BINDING_SCHEMA_VERSION = "autoanim.gnm-body-binding/2.0"
GNM_BODY_BINDING_ARRAYS_SCHEMA_VERSION = "autoanim.gnm-body-binding-arrays/2.0"
MAX_BINDING_NPZ_BYTES = 64 * 1024 * 1024
_PLANE_EPSILON_M = 2.0e-7
GNM_TO_AUTOANIM_BASIS = np.diag((-1.0, 1.0, 1.0)).astype(np.float32)


class BodyBindingError(ValueError):
    """A body/head binding could not be constructed or verified safely."""


@dataclass(frozen=True, slots=True)
class PlaneClippedMesh:
    """A triangle mesh and exact two-source interpolation for each vertex."""

    vertices: np.ndarray
    triangles: np.ndarray
    source_indices: np.ndarray
    source_weights: np.ndarray
    cut_loop: np.ndarray


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    faces = np.asarray(triangles, dtype=np.int64)
    edges = np.sort(
        np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
            axis=0,
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise BodyBindingError("Mesh contains a non-manifold edge")
    return unique[counts == 1]


def _ordered_loops_from_edges(edges: np.ndarray) -> list[np.ndarray]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in np.asarray(edges, dtype=np.int64).tolist():
        adjacency[first].append(second)
        adjacency[second].append(first)
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise BodyBindingError("Boundary is not a collection of closed manifold loops")
    remaining = set(adjacency)
    loops: list[np.ndarray] = []
    while remaining:
        start = min(remaining)
        loop = [start]
        previous = -1
        current = start
        while True:
            neighbors = sorted(adjacency[current])
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if following == start:
                break
            if following in loop:
                raise BodyBindingError("Boundary loop self-intersects topologically")
            loop.append(following)
            previous, current = current, following
        remaining.difference_update(loop)
        loops.append(np.asarray(loop, dtype=np.int32))
    return loops


def _canonicalize_loop(indices: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    loop = np.asarray(indices, dtype=np.int32)
    points = np.asarray(vertices, dtype=np.float64)[loop]
    # The cut plane is near-horizontal. Keep a deterministic X/Z winding.
    signed_area = 0.5 * np.sum(
        points[:, 0] * np.roll(points[:, 2], -1)
        - np.roll(points[:, 0], -1) * points[:, 2]
    )
    if signed_area < 0.0:
        loop = loop[::-1]
        points = points[::-1]
    order = np.lexsort((loop, -points[:, 2], -points[:, 0]))
    return np.roll(loop, -int(order[0]))


def clip_triangle_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    keep_positive: bool,
    epsilon_m: float = _PLANE_EPSILON_M,
) -> PlaneClippedMesh:
    """Clip a triangle mesh and retain an exact interpolation provenance map.

    A retained original vertex maps to itself with weight one. A plane
    intersection maps to the two endpoints of the crossed original edge.
    """

    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    origin = np.asarray(plane_point, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or origin.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite(points).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(normal).all()
        or not np.isfinite(epsilon_m)
        or epsilon_m <= 0.0
    ):
        raise BodyBindingError("Plane clipping inputs are invalid")
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-9:
        raise BodyBindingError("Clipping plane normal is degenerate")
    normal /= length
    signed = (points - origin) @ normal
    if not keep_positive:
        signed *= -1.0

    output_vertices: list[np.ndarray] = []
    output_sources: list[tuple[int, int]] = []
    output_weights: list[tuple[float, float]] = []
    original_map: dict[int, int] = {}
    intersection_map: dict[tuple[int, int], int] = {}

    def original(index: int) -> int:
        if index not in original_map:
            original_map[index] = len(output_vertices)
            output_vertices.append(points[index])
            output_sources.append((index, index))
            output_weights.append((1.0, 0.0))
        return original_map[index]

    def intersection(first: int, second: int) -> int:
        edge = (min(first, second), max(first, second))
        if edge in intersection_map:
            return intersection_map[edge]
        denominator = signed[first] - signed[second]
        if abs(float(denominator)) <= 1.0e-15:
            raise BodyBindingError("Clipping encountered a coplanar crossing edge")
        t = float(signed[first] / denominator)
        t = min(1.0, max(0.0, t))
        index = len(output_vertices)
        output_vertices.append(points[first] * (1.0 - t) + points[second] * t)
        output_sources.append((first, second))
        output_weights.append((1.0 - t, t))
        intersection_map[edge] = index
        return index

    output_faces: list[tuple[int, int, int]] = []
    for triangle in faces.tolist():
        polygon: list[tuple[int, bool]] = [
            (int(index), bool(signed[index] >= -epsilon_m)) for index in triangle
        ]
        clipped: list[int] = []
        for edge_index in range(3):
            first, first_inside = polygon[edge_index]
            second, second_inside = polygon[(edge_index + 1) % 3]
            if first_inside:
                clipped.append(original(first))
            if first_inside != second_inside:
                clipped.append(intersection(first, second))
        compact: list[int] = []
        for index in clipped:
            if not compact or compact[-1] != index:
                compact.append(index)
        if len(compact) > 1 and compact[0] == compact[-1]:
            compact.pop()
        for corner in range(1, len(compact) - 1):
            candidate = (compact[0], compact[corner], compact[corner + 1])
            tri_points = np.asarray([output_vertices[index] for index in candidate])
            if np.linalg.norm(
                np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
            ) > 1.0e-14:
                output_faces.append(candidate)

    clipped_vertices = np.asarray(output_vertices, dtype=np.float32)
    clipped_faces = np.asarray(output_faces, dtype=np.int32)
    if len(clipped_vertices) < 3 or len(clipped_faces) < 1:
        raise BodyBindingError("Plane clipping removed the complete mesh")
    boundary = _boundary_edges(clipped_faces)
    cut_distance = np.abs((clipped_vertices - origin) @ normal)
    cut_edges = boundary[
        np.logical_and(
            cut_distance[boundary[:, 0]] <= epsilon_m * 4.0,
            cut_distance[boundary[:, 1]] <= epsilon_m * 4.0,
        )
    ]
    loops = _ordered_loops_from_edges(cut_edges)
    if len(loops) != 1 or len(loops[0]) < 8:
        raise BodyBindingError("Clipping did not produce exactly one neck loop")
    loop = _canonicalize_loop(loops[0], clipped_vertices)
    return PlaneClippedMesh(
        vertices=clipped_vertices,
        triangles=clipped_faces,
        source_indices=np.asarray(output_sources, dtype=np.int32),
        source_weights=np.asarray(output_weights, dtype=np.float32),
        cut_loop=loop,
    )


def interpolate_clipped_attribute(
    values: np.ndarray, source_indices: np.ndarray, source_weights: np.ndarray
) -> np.ndarray:
    """Interpolate any vertex-leading array using a sealed clip map."""

    source = np.asarray(values)
    indices = np.asarray(source_indices, dtype=np.int64)
    weights = np.asarray(source_weights, dtype=np.float64)
    if (
        source.ndim < 1
        or indices.ndim != 2
        or indices.shape[1] != 2
        or weights.shape != indices.shape
        or np.any(indices < 0)
        or np.any(indices >= len(source))
    ):
        raise BodyBindingError("Clipped attribute interpolation inputs are invalid")
    extra = (1,) * (source.ndim - 1)
    result = (
        source[indices[:, 0]] * weights[:, 0].reshape((-1, *extra))
        + source[indices[:, 1]] * weights[:, 1].reshape((-1, *extra))
    )
    return result.astype(source.dtype, copy=False)


def _global_rest_matrices(local_matrices: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.asarray(local_matrices, dtype=np.float64)
    hierarchy = np.asarray(parents, dtype=np.int64)
    world = np.empty_like(local)
    for index, parent in enumerate(hierarchy.tolist()):
        world[index] = local[index] if parent == -1 else world[parent] @ local[index]
    return world


def _loop_perimeter(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1)))


def _semantic_frame(axis_y: np.ndarray) -> np.ndarray:
    y_axis = np.asarray(axis_y, dtype=np.float64)
    y_axis /= np.linalg.norm(y_axis)
    forward_hint = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    x_axis = np.cross(y_axis, forward_hint)
    if np.linalg.norm(x_axis) <= 1.0e-6:
        raise BodyBindingError("Semantic neck axis is parallel to the forward axis")
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    frame = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(frame) < 0.999:
        raise BodyBindingError("Semantic attachment frame contains a reflection")
    return frame


def _loop_residuals(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    forward = cKDTree(first).query(second, workers=1)[0]
    reverse = cKDTree(second).query(first, workers=1)[0]
    return np.concatenate((forward, reverse))


def calibrate_gnm_body_binding(
    manifest_path: str | Path,
    arrays_path: str | Path,
    *,
    body_manifest_path: str | Path,
    body_asset_path: str | Path,
    gnm_neutral_vertices: np.ndarray,
    gnm_triangles: np.ndarray,
    gnm_identity: np.ndarray,
    gnm_neck_position: np.ndarray,
    gnm_head_position: np.ndarray,
    gnm_cut_fraction: float = 0.35,
    collar_height_m: float = 0.015,
) -> dict[str, Any]:
    """Create one exact, immutable geometric preview binding."""

    provider_manifest = load_and_validate_body_asset(
        body_manifest_path, body_asset_path
    )
    with np.load(body_asset_path, allow_pickle=False) as archive:
        body = {name: np.array(archive[name], copy=True) for name in archive.files}
    gnm_vertices = np.asarray(gnm_neutral_vertices, dtype=np.float32)
    gnm_faces = np.asarray(gnm_triangles, dtype=np.int32)
    identity = np.asarray(gnm_identity, dtype=np.float32)
    gnm_neck = np.asarray(gnm_neck_position, dtype=np.float64)
    gnm_head = np.asarray(gnm_head_position, dtype=np.float64)
    if (
        gnm_vertices.ndim != 2
        or gnm_vertices.shape[1] != 3
        or gnm_faces.ndim != 2
        or gnm_faces.shape[1] != 3
        or identity.shape != (253,)
        or gnm_neck.shape != (3,)
        or gnm_head.shape != (3,)
        or not np.isfinite(gnm_vertices).all()
        or not np.isfinite(identity).all()
        or not 0.25 <= gnm_cut_fraction <= 0.50
        or not 0.010 <= collar_height_m <= 0.030
    ):
        raise BodyBindingError("GNM binding inputs have invalid shape or values")

    # GNM names subject-left as +X while the canonical body names anatomical
    # left as -X. Bake that reflection into positions/bases and reverse winding;
    # the runtime socket itself must remain a proper rotation.
    baked_gnm_vertices = gnm_vertices @ GNM_TO_AUTOANIM_BASIS.T
    baked_gnm_faces = gnm_faces[:, [0, 2, 1]]
    baked_gnm_neck = gnm_neck @ GNM_TO_AUTOANIM_BASIS.T
    baked_gnm_head = gnm_head @ GNM_TO_AUTOANIM_BASIS.T

    world_rest = _global_rest_matrices(body["local_rest_matrices"], body["parents"])
    neck_index = CANONICAL_HUMANOID.index("Neck")
    head_index = CANONICAL_HUMANOID.index("Head")
    body_neck = world_rest[neck_index, :3, 3]
    body_head = world_rest[head_index, :3, 3]
    body_axis = body_head - body_neck
    gnm_axis = baked_gnm_head - baked_gnm_neck
    body_clip = clip_triangle_mesh(
        body["vertices_m"],
        body["triangles"],
        plane_point=body_neck,
        plane_normal=body_axis,
        keep_positive=False,
    )
    gnm_plane = baked_gnm_neck + gnm_cut_fraction * gnm_axis
    gnm_clip = clip_triangle_mesh(
        baked_gnm_vertices,
        baked_gnm_faces,
        plane_point=gnm_plane,
        plane_normal=gnm_axis,
        keep_positive=True,
    )

    body_loop_points = body_clip.vertices[body_clip.cut_loop].astype(np.float64)
    gnm_loop_points = gnm_clip.vertices[gnm_clip.cut_loop].astype(np.float64)
    body_perimeter = _loop_perimeter(body_loop_points)
    gnm_perimeter = _loop_perimeter(gnm_loop_points)
    scale = body_perimeter / gnm_perimeter
    if not 0.95 <= scale <= 1.05:
        raise BodyBindingError("GNM/body neck scale exceeds the bounded calibration range")
    rotation = _semantic_frame(body_axis) @ _semantic_frame(gnm_axis).T
    body_axis_unit = body_axis / np.linalg.norm(body_axis)
    translation = (
        np.mean(body_loop_points, axis=0) + collar_height_m * body_axis_unit
    ) - (
        scale * (rotation @ np.mean(gnm_loop_points, axis=0))
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    transformed_loop = (
        gnm_loop_points @ transform[:3, :3].T + transform[:3, 3]
    )
    body_frame = _semantic_frame(body_axis)
    body_planar = (body_loop_points - np.mean(body_loop_points, axis=0)) @ body_frame
    gnm_planar = (transformed_loop - np.mean(transformed_loop, axis=0)) @ body_frame
    body_planar[:, 1] = 0.0
    gnm_planar[:, 1] = 0.0
    residuals = _loop_residuals(body_planar, gnm_planar)
    metrics = {
        "uniform_scale": float(scale),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "collar_height_m": float(collar_height_m),
        "body_loop_perimeter_m": body_perimeter,
        "gnm_loop_perimeter_m": gnm_perimeter,
        "centered_planar_shape_rms_m": float(
            np.sqrt(np.mean(residuals * residuals))
        ),
        "centered_planar_shape_p95_m": float(np.percentile(residuals, 95)),
        "centered_planar_shape_max_m": float(np.max(residuals)),
        "body_neck_loop_vertices": int(len(body_clip.cut_loop)),
        "gnm_neck_loop_vertices": int(len(gnm_clip.cut_loop)),
    }

    arrays = {
        "schema_version": np.asarray(GNM_BODY_BINDING_ARRAYS_SCHEMA_VERSION),
        "gnm_to_body_world_matrix": transform.astype(np.float32),
        "body_cut_plane_point": body_neck.astype(np.float32),
        "body_cut_plane_normal": (body_axis / np.linalg.norm(body_axis)).astype(
            np.float32
        ),
        "gnm_cut_plane_point": gnm_plane.astype(np.float32),
        "gnm_cut_plane_normal": (gnm_axis / np.linalg.norm(gnm_axis)).astype(
            np.float32
        ),
        "body_vertices": body_clip.vertices,
        "body_triangles": body_clip.triangles,
        "body_source_indices": body_clip.source_indices,
        "body_source_weights": body_clip.source_weights,
        "body_neck_loop_indices": body_clip.cut_loop,
        "gnm_vertices": gnm_clip.vertices,
        "gnm_triangles": gnm_clip.triangles,
        "gnm_source_indices": gnm_clip.source_indices,
        "gnm_source_weights": gnm_clip.source_weights,
        "gnm_neck_loop_indices": gnm_clip.cut_loop,
    }
    write_npz(arrays_path, **arrays)
    body_digest = sha256_file(body_asset_path)
    identity_digest = array_sha256(identity)
    arrays_digest = sha256_file(arrays_path)
    blockers = [
        "CHARACTER_BODY_IDENTITY_NOT_CALIBRATED",
        "BRIDGE_LOOKDEV_NOT_ARTIST_APPROVED",
        "SOMA_25_PROJECTION_PREVIEW_ONLY",
    ]
    if metrics["centered_planar_shape_p95_m"] > 0.005:
        blockers.append("NECK_CENTERED_PLANAR_SHAPE_RESIDUAL_ABOVE_5MM")
    manifest: dict[str, Any] = {
        "schema_version": GNM_BODY_BINDING_SCHEMA_VERSION,
        "status": "geometry_calibrated_preview",
        "production_validated": False,
        "publish_ready": False,
        "body": {
            "asset_sha256": body_digest,
            "manifest_sha256": sha256_file(body_manifest_path),
            "provider": provider_manifest["provider"],
            "provider_attachment_calibrated": provider_manifest[
                "gnm_head_socket"
            ]["attachment_calibrated"],
        },
        "gnm": {
            "version": "3.0",
            "variant": "head",
            "identity_sha256": identity_digest,
            "neutral_vertices_sha256": array_sha256(gnm_vertices),
            "triangles_sha256": array_sha256(gnm_faces),
            "source_basis_matrix": GNM_TO_AUTOANIM_BASIS.astype(float).tolist(),
            "triangle_winding_reversed": True,
        },
        "socket": {
            "type": "gnm_model_to_body_world_rest_similarity",
            "skin_joints": ["Neck", "Head"],
            "matrix_semantics": "GNM model to MPFB body world rest transform",
            "uniform_similarity_only": True,
            "no_reflection": True,
            "gnm_to_body_world_matrix": transform.astype(float).tolist(),
        },
        "cut": {
            "method": "semantic_joint_axis_triangle_plane_intersection",
            "provider_seam_indices_used_as_topology": False,
            "provider_head_removed": True,
            "gnm_cut_fraction": float(gnm_cut_fraction),
            "collar_height_m": float(collar_height_m),
            "body_kept_triangle_count": int(len(body_clip.triangles)),
            "gnm_kept_triangle_count": int(len(gnm_clip.triangles)),
        },
        "metrics": metrics,
        "arrays": {
            "name": Path(arrays_path).name,
            "bytes": Path(arrays_path).stat().st_size,
            "sha256": arrays_digest,
            "field_sha256": {
                name: array_sha256(value)
                for name, value in arrays.items()
                if name != "schema_version"
            },
        },
        "publish_blockers": blockers,
    }
    manifest["payload_sha256"] = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(manifest_path, manifest)
    return manifest


def load_gnm_body_binding(
    manifest_path: str | Path,
    arrays_path: str | Path,
    *,
    expected_body_asset_sha256: str,
    expected_identity_sha256: str,
    expected_body_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_file = Path(manifest_path)
    arrays_file = Path(arrays_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BodyBindingError("Binding manifest is unreadable") from exc
    if (
        manifest.get("schema_version") != GNM_BODY_BINDING_SCHEMA_VERSION
        or manifest.get("body", {}).get("asset_sha256")
        != expected_body_asset_sha256
        or manifest.get("gnm", {}).get("identity_sha256")
        != expected_identity_sha256
        or manifest.get("body", {}).get("manifest_sha256")
        != expected_body_manifest_sha256
        or manifest.get("arrays", {}).get("sha256") != sha256_file(arrays_file)
        or arrays_file.stat().st_size > MAX_BINDING_NPZ_BYTES
    ):
        raise BodyBindingError("Binding identity, body, or array digest is invalid")
    try:
        with zipfile.ZipFile(arrays_file) as archive:
            uncompressed_bytes = sum(info.file_size for info in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise BodyBindingError("Binding array archive is invalid") from exc
    if uncompressed_bytes > MAX_BINDING_NPZ_BYTES:
        raise BodyBindingError("Binding arrays exceed the uncompressed size limit")
    payload = dict(manifest)
    claimed = payload.pop("payload_sha256", None)
    observed = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed != observed:
        raise BodyBindingError("Binding manifest payload digest is invalid")
    try:
        with np.load(arrays_file, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise BodyBindingError("Binding arrays are unreadable") from exc
    expected = {
        "schema_version",
        "gnm_to_body_world_matrix",
        "body_cut_plane_point",
        "body_cut_plane_normal",
        "gnm_cut_plane_point",
        "gnm_cut_plane_normal",
        "body_vertices",
        "body_triangles",
        "body_source_indices",
        "body_source_weights",
        "body_neck_loop_indices",
        "gnm_vertices",
        "gnm_triangles",
        "gnm_source_indices",
        "gnm_source_weights",
        "gnm_neck_loop_indices",
    }
    if set(arrays) != expected or str(arrays["schema_version"]) != (
        GNM_BODY_BINDING_ARRAYS_SCHEMA_VERSION
    ):
        raise BodyBindingError("Binding arrays have missing or unknown fields")
    field_hashes = manifest.get("arrays", {}).get("field_sha256")
    expected_hashed_fields = expected - {"schema_version"}
    if not isinstance(field_hashes, dict) or set(field_hashes) != expected_hashed_fields:
        raise BodyBindingError("Binding field-hash schema is invalid")
    for name, digest in manifest["arrays"]["field_sha256"].items():
        if name not in arrays or array_sha256(arrays[name]) != digest:
            raise BodyBindingError(f"Binding array field is invalid: {name}")
    for prefix in ("body", "gnm"):
        vertices = arrays[f"{prefix}_vertices"]
        triangles = arrays[f"{prefix}_triangles"]
        source_indices = arrays[f"{prefix}_source_indices"]
        source_weights = arrays[f"{prefix}_source_weights"]
        loop = arrays[f"{prefix}_neck_loop_indices"]
        source_limit = 13_380 if prefix == "body" else 17_821
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or triangles.ndim != 2
            or triangles.shape[1] != 3
            or source_indices.shape != (len(vertices), 2)
            or source_weights.shape != (len(vertices), 2)
            or loop.ndim != 1
            or len(loop) < 8
            or len(np.unique(loop)) != len(loop)
            or not np.isfinite(vertices).all()
            or not np.isfinite(source_weights).all()
            or np.any(triangles < 0)
            or np.any(triangles >= len(vertices))
            or np.any(source_indices < 0)
            or np.any(source_indices >= source_limit)
            or np.any(loop < 0)
            or np.any(loop >= len(vertices))
            or np.any(source_weights < 0.0)
            or not np.allclose(
                np.sum(source_weights, axis=1), 1.0, atol=1.0e-6, rtol=0.0
            )
        ):
            raise BodyBindingError(f"Binding {prefix} topology is invalid")
    transform = np.asarray(arrays["gnm_to_body_world_matrix"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise BodyBindingError("Binding socket transform is invalid")
    linear = transform[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    if (
        not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7)
        or np.linalg.det(linear) <= 0.0
        or float(np.max(singular) - np.min(singular)) > 1.0e-5
    ):
        raise BodyBindingError("Binding socket is not a proper uniform similarity")
    if (
        manifest.get("production_validated") is not False
        or manifest.get("publish_ready") is not False
        or not manifest.get("publish_blockers")
    ):
        raise BodyBindingError("Preview binding readiness claims are invalid")
    return manifest, arrays


__all__ = [
    "BodyBindingError",
    "GNM_BODY_BINDING_ARRAYS_SCHEMA_VERSION",
    "GNM_BODY_BINDING_SCHEMA_VERSION",
    "PlaneClippedMesh",
    "array_sha256",
    "calibrate_gnm_body_binding",
    "clip_triangle_mesh",
    "interpolate_clipped_attribute",
    "load_gnm_body_binding",
]
