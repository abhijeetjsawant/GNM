"""Volumetric GNM tongue/lip soft contact through the native Rust core."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import trimesh

from .gnm_adapter import GNMAdapter
from .physics import (
    PhysicsError,
    PhysicsInputError,
    PhysicsLibraryError,
    _Bindings,
    _float_pointer,
    _require_array,
    _resolve_library_path,
)


class _AaSoftContactConfig(ctypes.Structure):
    _fields_ = (
        ("frames_per_second", ctypes.c_float),
        ("substeps", ctypes.c_uint32),
        ("iterations", ctypes.c_uint32),
        ("edge_compliance", ctypes.c_float),
        ("volume_compliance", ctypes.c_float),
        ("tether_compliance", ctypes.c_float),
        ("contact_compliance", ctypes.c_float),
        ("contact_thickness_m", ctypes.c_float),
        ("contact_activation_distance_m", ctypes.c_float),
        ("max_displacement_m", ctypes.c_float),
    )


@dataclass(frozen=True, slots=True)
class SoftContactConfig:
    frames_per_second: float = 30.0
    substeps: int = 8
    iterations: int = 16
    edge_compliance: float = 2.0e-7
    volume_compliance: float = 0.0
    tether_compliance: float = 2.0e-5
    contact_compliance: float = 1.0e-9
    contact_thickness_m: float = 0.001
    contact_activation_distance_m: float = 0.006
    max_displacement_m: float = 0.012

    def _as_native(self) -> _AaSoftContactConfig:
        try:
            return _AaSoftContactConfig(
                self.frames_per_second,
                self.substeps,
                self.iterations,
                self.edge_compliance,
                self.volume_compliance,
                self.tether_compliance,
                self.contact_compliance,
                self.contact_thickness_m,
                self.contact_activation_distance_m,
                self.max_displacement_m,
            )
        except (OverflowError, TypeError, ValueError) as exc:
            raise PhysicsInputError(
                "SoftContactConfig contains an ABI-incompatible value"
            ) from exc

    @classmethod
    def _from_native(cls, value: _AaSoftContactConfig) -> SoftContactConfig:
        return cls(
            frames_per_second=float(value.frames_per_second),
            substeps=int(value.substeps),
            iterations=int(value.iterations),
            edge_compliance=float(value.edge_compliance),
            volume_compliance=float(value.volume_compliance),
            tether_compliance=float(value.tether_compliance),
            contact_compliance=float(value.contact_compliance),
            contact_thickness_m=float(value.contact_thickness_m),
            contact_activation_distance_m=float(
                value.contact_activation_distance_m
            ),
            max_displacement_m=float(value.max_displacement_m),
        )


@dataclass(frozen=True, slots=True)
class GNMSoftContactTopology:
    global_vertex_indices: np.ndarray
    tongue_vertex_count: int
    lower_lip_vertex_count: int
    upper_lip_vertex_count: int
    rest_positions: np.ndarray
    surface_triangles: np.ndarray
    tetrahedra: np.ndarray
    contact_pairs: np.ndarray
    inverse_masses: np.ndarray
    volume_coverage: float
    boundary_vertex_count: int

def _readonly_contiguous(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    output = np.ascontiguousarray(value, dtype=dtype)
    output.setflags(write=False)
    return output


def _local_triangles(
    global_triangles: np.ndarray,
    global_indices: np.ndarray,
) -> np.ndarray:
    size = int(global_indices.max(initial=-1)) + 1
    lookup = np.full(size, -1, dtype=np.int32)
    lookup[global_indices] = np.arange(global_indices.size, dtype=np.int32)
    local = lookup[np.asarray(global_triangles, dtype=np.int32)]
    if np.any(local < 0):
        raise RuntimeError("GNM component triangle references a vertex outside its group")
    return local


def _boundary_directed_edges(triangles: np.ndarray) -> np.ndarray:
    directed = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    return directed[counts[inverse] == 1]


def _tetrahedralize_open_tongue(
    positions: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    boundary = _boundary_directed_edges(triangles)
    boundary_vertices = np.unique(boundary)
    if boundary.size == 0:
        capped_positions = positions
        capped_triangles = triangles
    else:
        cap_index = positions.shape[0]
        cap = np.mean(positions[boundary_vertices], axis=0, keepdims=True)
        capped_positions = np.concatenate((positions, cap), axis=0)
        # Reverse the oriented boundary edge to close the surface consistently.
        cap_faces = np.column_stack(
            (
                boundary[:, 1],
                boundary[:, 0],
                np.full(boundary.shape[0], cap_index, dtype=np.int32),
            )
        )
        capped_triangles = np.concatenate((triangles, cap_faces), axis=0)
    classifier = trimesh.Trimesh(
        vertices=capped_positions,
        faces=capped_triangles,
        process=False,
        validate=False,
    )
    if not classifier.is_watertight or not classifier.is_winding_consistent:
        raise RuntimeError("GNM tongue cap did not form a consistent closed classifier")
    candidates = np.asarray(Delaunay(positions).simplices, dtype=np.int32)
    centers = np.mean(positions[candidates], axis=1)
    inside = np.asarray(classifier.contains(centers), dtype=bool)
    tetrahedra = candidates[inside]
    vertices = positions[tetrahedra]
    signed_six = np.einsum(
        "ij,ij->i",
        vertices[:, 1] - vertices[:, 0],
        np.cross(
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
    )
    tetrahedra = tetrahedra[np.abs(signed_six) > 6.0e-13]
    vertices = positions[tetrahedra]
    volume = np.sum(
        np.abs(
            np.einsum(
                "ij,ij->i",
                vertices[:, 1] - vertices[:, 0],
                np.cross(
                    vertices[:, 2] - vertices[:, 0],
                    vertices[:, 3] - vertices[:, 0],
                ),
            )
        )
        / 6.0
    )
    coverage = float(volume / classifier.volume)
    tetrahedra = np.sort(tetrahedra, axis=1)
    order = np.lexsort(tetrahedra.T[::-1])
    tetrahedra = np.unique(tetrahedra[order], axis=0)
    return tetrahedra, coverage, int(boundary_vertices.size)


def _sequence_contact_pairs(
    local_frames: np.ndarray,
    *,
    tongue_vertex_count: int,
    tongue_triangles: np.ndarray,
    lip_triangles: np.ndarray,
    maximum_distance_m: float,
    sample_stride: int,
    neighbors: int = 8,
) -> np.ndarray:
    contacts: set[tuple[int, int, int, int]] = set()
    tongue_indices = np.arange(tongue_vertex_count, dtype=np.int32)
    lip_indices = np.arange(
        tongue_vertex_count, local_frames.shape[1], dtype=np.int32
    )
    sampled = sorted(
        set(range(0, local_frames.shape[0], sample_stride))
        | {local_frames.shape[0] - 1}
    )
    for point_indices, obstacle_triangles in (
        (tongue_indices, lip_triangles),
        (lip_indices, tongue_triangles),
    ):
        for frame_index in sampled:
            frame = local_frames[frame_index]
            triangles = frame[obstacle_triangles]
            centers = np.mean(triangles, axis=1)
            tree = cKDTree(centers)
            _, nearest = tree.query(
                frame[point_indices],
                k=min(neighbors, obstacle_triangles.shape[0]),
            )
            if nearest.ndim == 1:
                nearest = nearest[:, None]
            points = np.repeat(
                frame[point_indices, None, :], nearest.shape[1], axis=1
            ).reshape(-1, 3)
            candidate_triangles = triangles[nearest.reshape(-1)]
            closest = trimesh.triangles.closest_point(candidate_triangles, points)
            distance = np.linalg.norm(points - closest, axis=1).reshape(nearest.shape)
            rows, columns = np.nonzero(distance <= maximum_distance_m)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                triangle = obstacle_triangles[int(nearest[row, column])]
                contacts.add(
                    (
                        int(point_indices[row]),
                        int(triangle[0]),
                        int(triangle[1]),
                        int(triangle[2]),
                    )
                )
    if not contacts:
        raise RuntimeError("No tongue/lip soft-contact candidates were discovered")
    return np.asarray(sorted(contacts), dtype=np.uint32)


def _filter_tetrahedra_for_sequence(
    tetrahedra: np.ndarray,
    tongue_frames: np.ndarray,
    rest_positions: np.ndarray,
    *,
    minimum_ratio: float = 0.10,
    maximum_ratio: float = 10.0,
) -> np.ndarray:
    rest = rest_positions[tetrahedra]
    rest_six = np.einsum(
        "ij,ij->i",
        rest[:, 1] - rest[:, 0],
        np.cross(rest[:, 2] - rest[:, 0], rest[:, 3] - rest[:, 0]),
    )
    keep = np.ones(tetrahedra.shape[0], dtype=bool)
    for frame in tongue_frames:
        vertices = frame[tetrahedra]
        signed_six = np.einsum(
            "ij,ij->i",
            vertices[:, 1] - vertices[:, 0],
            np.cross(
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
        )
        ratio = signed_six / rest_six
        keep &= np.isfinite(ratio)
        keep &= ratio >= minimum_ratio
        keep &= ratio <= maximum_ratio
    filtered = tetrahedra[keep]
    if filtered.shape[0] < 1_000:
        raise RuntimeError(
            "GNM target sequence leaves too few consistently oriented tongue tetrahedra"
        )
    return filtered


def build_gnm_soft_contact_topology(
    adapter: GNMAdapter,
    target_frames: np.ndarray,
    *,
    contact_search_distance_m: float = 0.008,
    contact_sample_stride: int = 3,
) -> GNMSoftContactTopology:
    """Recover a tongue volume and conservative contacts for one GNM sequence."""
    frames = np.asarray(target_frames)
    expected = (adapter.model.num_vertices, 3)
    if (
        frames.dtype != np.float32
        or frames.ndim != 3
        or frames.shape[0] == 0
        or frames.shape[1:] != expected
        or not frames.flags.c_contiguous
        or not np.isfinite(frames).all()
    ):
        raise PhysicsInputError(
            f"target_frames must be finite C-contiguous float32 [F,{expected[0]},3]"
        )
    tongue_indices = np.flatnonzero(adapter.vertex_group("tongue") > 0.5).astype(
        np.int32
    )
    lip_indices = np.flatnonzero(adapter.vertex_group("lower_lip") > 0.5).astype(
        np.int32
    )
    upper_lip_indices = np.flatnonzero(
        adapter.vertex_group("upper_lip") > 0.5
    ).astype(np.int32)
    global_indices = np.concatenate(
        (tongue_indices, lip_indices, upper_lip_indices)
    )
    tongue_triangles = _local_triangles(
        np.asarray(adapter.model.triangles_group("tongue"), dtype=np.int32),
        tongue_indices,
    )
    lip_triangles = _local_triangles(
        np.asarray(adapter.model.triangles_group("lower_lip"), dtype=np.int32),
        lip_indices,
    )
    lip_triangles += tongue_indices.size
    upper_lip_triangles = _local_triangles(
        np.asarray(adapter.model.triangles_group("upper_lip"), dtype=np.int32),
        upper_lip_indices,
    )
    upper_lip_triangles += tongue_indices.size + lip_indices.size
    all_lip_triangles = np.concatenate(
        (lip_triangles, upper_lip_triangles), axis=0
    )
    neutral = np.ascontiguousarray(adapter.mesh()[global_indices], dtype=np.float32)
    tetrahedra, coverage, boundary_count = _tetrahedralize_open_tongue(
        neutral[: tongue_indices.size],
        tongue_triangles,
    )
    surface_triangles = np.concatenate(
        (tongue_triangles, all_lip_triangles), axis=0
    )
    local_frames = np.ascontiguousarray(frames[:, global_indices], dtype=np.float32)
    unfiltered_tetrahedra = tetrahedra
    tetrahedra = _filter_tetrahedra_for_sequence(
        unfiltered_tetrahedra,
        local_frames[:, : tongue_indices.size],
        neutral[: tongue_indices.size],
    )
    def tetra_volume_sum(indices: np.ndarray) -> float:
        vertices = neutral[indices]
        return float(
            np.sum(
                np.abs(
                    np.einsum(
                        "ij,ij->i",
                        vertices[:, 1] - vertices[:, 0],
                        np.cross(
                            vertices[:, 2] - vertices[:, 0],
                            vertices[:, 3] - vertices[:, 0],
                        ),
                    )
                )
            )
        )

    coverage *= tetra_volume_sum(tetrahedra) / tetra_volume_sum(
        unfiltered_tetrahedra
    )
    contact_pairs = _sequence_contact_pairs(
        local_frames,
        tongue_vertex_count=int(tongue_indices.size),
        tongue_triangles=tongue_triangles,
        lip_triangles=all_lip_triangles,
        maximum_distance_m=contact_search_distance_m,
        sample_stride=contact_sample_stride,
    )

    inverse_masses = np.ones(global_indices.size, dtype=np.float32)
    boundary_vertices = np.unique(_boundary_directed_edges(tongue_triangles))
    inverse_masses[boundary_vertices] = np.float32(0.08)
    lip_x = neutral[tongue_indices.size :, 0]
    horizontal = np.abs(lip_x - np.median(lip_x))
    normalized = horizontal / max(float(np.max(horizontal)), 1.0e-6)
    inverse_masses[tongue_indices.size :] = np.asarray(
        0.55 - 0.43 * normalized**1.5,
        dtype=np.float32,
    )
    return GNMSoftContactTopology(
        global_vertex_indices=_readonly_contiguous(global_indices, np.dtype(np.int32)),
        tongue_vertex_count=int(tongue_indices.size),
        lower_lip_vertex_count=int(lip_indices.size),
        upper_lip_vertex_count=int(upper_lip_indices.size),
        rest_positions=_readonly_contiguous(neutral, np.dtype(np.float32)),
        surface_triangles=_readonly_contiguous(
            surface_triangles, np.dtype(np.uint32)
        ),
        tetrahedra=_readonly_contiguous(tetrahedra, np.dtype(np.uint32)),
        contact_pairs=_readonly_contiguous(contact_pairs, np.dtype(np.uint32)),
        inverse_masses=_readonly_contiguous(inverse_masses, np.dtype(np.float32)),
        volume_coverage=coverage,
        boundary_vertex_count=boundary_count,
    )


class SoftContactSimulator:
    """Own a native volumetric tongue/lip solver."""

    def __init__(
        self,
        topology: GNMSoftContactTopology,
        *,
        library_path: str | Path | None = None,
        config: SoftContactConfig | None = None,
    ) -> None:
        if not isinstance(topology, GNMSoftContactTopology):
            raise PhysicsInputError("topology must be a GNMSoftContactTopology")
        if config is not None and not isinstance(config, SoftContactConfig):
            raise PhysicsInputError("config must be a SoftContactConfig or None")
        self._lock = threading.RLock()
        self._bindings = _Bindings(_resolve_library_path(library_path))
        library = self._bindings.library
        try:
            library.aa_soft_contact_default_config.argtypes = ()
            library.aa_soft_contact_default_config.restype = _AaSoftContactConfig
            library.aa_soft_contact_simulator_create.argtypes = (
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                _AaSoftContactConfig,
            )
            library.aa_soft_contact_simulator_create.restype = ctypes.c_void_p
            library.aa_soft_contact_simulator_destroy.argtypes = (ctypes.c_void_p,)
            library.aa_soft_contact_simulator_destroy.restype = None
            library.aa_soft_contact_simulate_chunk.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
            )
            library.aa_soft_contact_simulate_chunk.restype = ctypes.c_int32
            library.aa_soft_contact_report_json.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_size_t,
            )
            library.aa_soft_contact_report_json.restype = ctypes.c_size_t
        except AttributeError as exc:
            raise PhysicsLibraryError(
                f"Physics release library lacks the soft-contact ABI: {self._bindings.path}"
            ) from exc
        native_config = (
            library.aa_soft_contact_default_config()
            if config is None
            else config._as_native()
        )
        self.config = SoftContactConfig._from_native(native_config)
        self.topology = topology
        self._simulator: int | None = library.aa_soft_contact_simulator_create(
            _float_pointer(topology.rest_positions),
            topology.rest_positions.size,
            topology.surface_triangles.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint32)
            ),
            topology.surface_triangles.shape[0],
            topology.tetrahedra.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            topology.tetrahedra.shape[0],
            topology.contact_pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            topology.contact_pairs.shape[0],
            _float_pointer(topology.inverse_masses),
            topology.inverse_masses.size,
            native_config,
        )
        if not self._simulator:
            raise PhysicsInputError(self._bindings.last_error())

    @property
    def closed(self) -> bool:
        return self._simulator is None

    def _require_open(self) -> int:
        if self._simulator is None:
            raise PhysicsError("SoftContactSimulator is closed")
        return self._simulator

    def simulate(self, targets: np.ndarray) -> np.ndarray:
        targets = _require_array(
            targets,
            name="targets",
            dtype=np.dtype(np.float32),
            ndim=3,
        )
        expected = (targets.shape[0], self.topology.rest_positions.shape[0], 3)
        if targets.shape != expected or targets.shape[0] == 0:
            raise PhysicsInputError(f"targets must have nonempty shape {expected}")
        if not np.isfinite(targets).all():
            raise PhysicsInputError("targets must contain only finite values")
        output = np.empty_like(targets)
        with self._lock:
            status = self._bindings.library.aa_soft_contact_simulate_chunk(
                self._require_open(),
                _float_pointer(targets),
                targets.size,
                _float_pointer(output),
                output.size,
            )
            if status != 0:
                raise PhysicsError(self._bindings.last_error())
        return output

    def report(self) -> dict[str, Any]:
        with self._lock:
            simulator = self._require_open()
            required = self._bindings.library.aa_soft_contact_report_json(
                simulator, None, 0
            )
            if required == 0:
                raise PhysicsError(self._bindings.last_error())
            buffer = ctypes.create_string_buffer(required)
            written = self._bindings.library.aa_soft_contact_report_json(
                simulator, buffer, required
            )
            if written != required:
                raise PhysicsError(self._bindings.last_error())
        try:
            report = json.loads(buffer.value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PhysicsLibraryError(
                "Native soft-contact report is not valid JSON"
            ) from exc
        if not isinstance(report, dict):
            raise PhysicsLibraryError("Native soft-contact report must be a JSON object")
        report.update(
            {
                "tongue_vertex_count": self.topology.tongue_vertex_count,
                "lower_lip_vertex_count": self.topology.lower_lip_vertex_count,
                "upper_lip_vertex_count": self.topology.upper_lip_vertex_count,
                "tetrahedral_volume_coverage": self.topology.volume_coverage,
                "tongue_boundary_vertex_count": self.topology.boundary_vertex_count,
            }
        )
        return report

    def close(self) -> None:
        with self._lock:
            if self._simulator is not None:
                self._bindings.library.aa_soft_contact_simulator_destroy(
                    self._simulator
                )
                self._simulator = None

    def __enter__(self) -> SoftContactSimulator:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def simulate_gnm_tongue_lip_soft_contact(
    adapter: GNMAdapter,
    target_frames: np.ndarray,
    *,
    library_path: str | Path | None = None,
    config: SoftContactConfig | None = None,
) -> tuple[np.ndarray, GNMSoftContactTopology, dict[str, Any]]:
    """Solve local oral tissue and map it back into complete GNM frames."""
    topology = build_gnm_soft_contact_topology(adapter, target_frames)
    local_targets = np.ascontiguousarray(
        target_frames[:, topology.global_vertex_indices],
        dtype=np.float32,
    )
    with SoftContactSimulator(
        topology,
        library_path=library_path,
        config=config,
    ) as simulator:
        solved = simulator.simulate(local_targets)
        report = simulator.report()
    output = np.ascontiguousarray(target_frames.copy(), dtype=np.float32)
    output[:, topology.global_vertex_indices] = solved
    return output, topology, report


__all__ = [
    "GNMSoftContactTopology",
    "SoftContactConfig",
    "SoftContactSimulator",
    "build_gnm_soft_contact_topology",
    "simulate_gnm_tongue_lip_soft_contact",
]
