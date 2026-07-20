"""Standard glTF morph animation for verified GNM vertex tracks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gnm.shape.visualization import vertex_colors as vertex_colors_module

from .gltf_export import split_triangle_corner_uvs
from .gnm_adapter import GNMAdapter
from .oral_validation import classify_lip_landmarks
from .runtime_material import (
    load_runtime_material_derivatives,
    prepare_runtime_material,
)
from .serialization import write_npz


class AnimationCompressionError(ValueError):
    """The target cap cannot reconstruct a vertex animation faithfully."""

    def __init__(self, message: str, metrics: dict[str, float | int]):
        super().__init__(message)
        self.metrics = metrics


_ORAL_PRIORITY_LANDMARKS = (8, 27, 36, 45, 61, 62, 63, 65, 66, 67)
_ORAL_PRIORITY_SCALE = 32.0


@dataclass(frozen=True, slots=True)
class LowRankVertexAnimation:
    base_vertices: np.ndarray
    morph_positions: np.ndarray
    weights: np.ndarray
    mesh_p95_m: float
    mesh_max_m: float
    landmark_p95_m: float
    landmark_max_m: float
    oral_corrective_targets: int = 0

    @property
    def rank(self) -> int:
        return int(len(self.morph_positions))


@dataclass(frozen=True, slots=True)
class AnimatedGLBExport:
    path: Path
    mapping_path: Path
    rank: int
    frame_count: int
    vertex_count: int
    triangle_count: int
    mesh_p95_mm: float
    mesh_max_mm: float
    landmark_p95_mm: float
    landmark_max_mm: float
    oral_corrective_targets: int


def _landmarks(
    vertices: np.ndarray,
    indices: np.ndarray | None,
    weights: np.ndarray | None,
) -> np.ndarray | None:
    if indices is None or weights is None:
        return None
    selected = vertices[..., indices, :]
    return np.sum(selected * weights[..., None], axis=-2)


def _tongue_teeth_collision_frames(
    frames: np.ndarray,
    landmarks: np.ndarray,
    tongue_indices: np.ndarray,
    teeth_indices: np.ndarray,
    *,
    threshold_interocular: float,
) -> np.ndarray:
    """Match the viewer gate's nearest-vertex tongue/teeth risk classifier."""

    interocular = np.linalg.norm(landmarks[:, 36] - landmarks[:, 45], axis=1)
    if not np.isfinite(interocular).all() or np.any(interocular <= 1.0e-6):
        raise ValueError("Tongue/teeth semantics require valid interocular scale")
    collision = np.empty(len(frames), dtype=bool)
    for frame_index, frame in enumerate(frames):
        distances, _ = cKDTree(frame[teeth_indices]).query(
            frame[tongue_indices], k=1, workers=1
        )
        collision[frame_index] = (
            float(np.min(distances)) / interocular[frame_index]
            <= threshold_interocular
        )
    return collision


def factor_vertex_animation(
    frames: np.ndarray,
    *,
    max_targets: int = 32,
    mesh_p95_limit_m: float = 0.00010,
    mesh_max_limit_m: float = 0.00050,
    landmark_indices: np.ndarray | None = None,
    landmark_weights: np.ndarray | None = None,
    landmark_p95_limit_m: float = 0.00025,
    landmark_max_limit_m: float = 0.00100,
    preserve_oral_semantics: bool = False,
    lip_contact_gap_interocular: float = 0.006,
    lip_order_inversion_tolerance_interocular: float = 0.0005,
    tongue_vertex_indices: np.ndarray | None = None,
    teeth_vertex_indices: np.ndarray | None = None,
    tongue_teeth_collision_risk_interocular: float = 0.001,
) -> LowRankVertexAnimation:
    """Find the smallest deterministic morph basis that passes geometry gates.

    When ``preserve_oral_semantics`` is enabled, factorization prioritizes the
    landmark-support vertices used for inner-lip contact and signed ordering.
    A rank is accepted only when it preserves the source contact classification
    and introduces no new lip-order inversion risks.
    """

    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3 or values.shape[2:] != (3,) or len(values) < 1:
        raise ValueError("frames must have shape [frames,vertices,3]")
    if not np.isfinite(values).all():
        raise ValueError("vertex animation contains nonfinite values")
    if not isinstance(max_targets, int) or isinstance(max_targets, bool) or max_targets < 0:
        raise ValueError("max_targets must be a non-negative integer")
    if any(
        not np.isfinite(limit) or limit < 0
        for limit in (
            mesh_p95_limit_m,
            mesh_max_limit_m,
            landmark_p95_limit_m,
            landmark_max_limit_m,
        )
    ):
        raise ValueError("animation reconstruction limits must be finite and non-negative")
    if (
        not np.isfinite(lip_contact_gap_interocular)
        or lip_contact_gap_interocular <= 0.0
        or not np.isfinite(lip_order_inversion_tolerance_interocular)
        or lip_order_inversion_tolerance_interocular <= 0.0
        or not np.isfinite(tongue_teeth_collision_risk_interocular)
        or tongue_teeth_collision_risk_interocular <= 0.0
    ):
        raise ValueError("oral semantic thresholds must be finite and positive")

    observed_landmarks = _landmarks(values, landmark_indices, landmark_weights)
    coordinate_scale = np.ones((values.shape[1], 3), dtype=np.float32)
    observed_contact: np.ndarray | None = None
    observed_order_risk: np.ndarray | None = None
    observed_tongue_teeth_collision: np.ndarray | None = None
    tongue_indices: np.ndarray | None = None
    teeth_indices: np.ndarray | None = None
    if preserve_oral_semantics:
        if observed_landmarks is None:
            raise ValueError(
                "preserve_oral_semantics requires landmark indices and weights"
            )
        landmark_index_values = np.asarray(landmark_indices, dtype=np.int64)
        if landmark_index_values.ndim < 2 or landmark_index_values.shape[0] < 68:
            raise ValueError(
                "preserve_oral_semantics requires a 68-landmark vertex regressor"
            )
        priority_vertices = np.unique(
            landmark_index_values[np.asarray(_ORAL_PRIORITY_LANDMARKS)].reshape(-1)
        )
        if (
            np.any(priority_vertices < 0)
            or np.any(priority_vertices >= values.shape[1])
        ):
            raise ValueError("oral landmark regressor contains an invalid vertex index")
        coordinate_scale[priority_vertices] = np.float32(_ORAL_PRIORITY_SCALE)
        semantic_weights = np.asarray(landmark_weights, dtype=np.float64)
        observed_semantic_landmarks = _landmarks(
            values,
            landmark_indices,
            semantic_weights,
        )
        if observed_semantic_landmarks is None:
            raise AssertionError("oral semantic landmark evaluation unexpectedly failed")
        _, observed_contact, observed_order_risk = classify_lip_landmarks(
            observed_semantic_landmarks,
            contact_gap_interocular=lip_contact_gap_interocular,
            order_tolerance_interocular=(
                lip_order_inversion_tolerance_interocular
            ),
        )
        if (tongue_vertex_indices is None) != (teeth_vertex_indices is None):
            raise ValueError(
                "tongue and teeth vertex indices must be supplied together"
            )
        if tongue_vertex_indices is not None and teeth_vertex_indices is not None:
            tongue_indices = np.asarray(tongue_vertex_indices, dtype=np.int64)
            teeth_indices = np.asarray(teeth_vertex_indices, dtype=np.int64)
            for label, indices in (
                ("tongue", tongue_indices),
                ("teeth", teeth_indices),
            ):
                if (
                    indices.ndim != 1
                    or len(indices) == 0
                    or len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0)
                    or np.any(indices >= values.shape[1])
                ):
                    raise ValueError(f"{label} semantic vertex indices are invalid")
            observed_tongue_teeth_collision = _tongue_teeth_collision_frames(
                values,
                observed_semantic_landmarks,
                tongue_indices,
                teeth_indices,
                threshold_interocular=tongue_teeth_collision_risk_interocular,
            )

    base = values[0].copy()
    flattened = ((values - base) * coordinate_scale).reshape(len(values), -1)
    if float(np.max(np.abs(flattened), initial=0.0)) <= 1e-12:
        return LowRankVertexAnimation(
            base,
            np.zeros((0, values.shape[1], 3), dtype=np.float32),
            np.zeros((len(values), 0), dtype=np.float32),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    available = min(max_targets, len(values) - 1, min(flattened.shape))
    if available <= 0:
        metrics = {"available_rank": int(available), "frame_count": int(len(values))}
        raise AnimationCompressionError("Animated track has no permitted morph targets", metrics)

    # The number of frames is much smaller than the coordinate dimension for
    # normal clips.  Eigendecomposing X X^T is both exact and inexpensive.
    gram = flattened @ flattened.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    singular = np.sqrt(eigenvalues)
    numerical = singular > max(float(singular[0]) * 1e-7, 1e-10)
    available = min(available, int(np.count_nonzero(numerical)))
    if available <= 0:
        return LowRankVertexAnimation(
            base,
            np.zeros((0, values.shape[1], 3), dtype=np.float32),
            np.zeros((len(values), 0), dtype=np.float32),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    right = (left[:, :available].T @ flattened) / singular[:available, None]
    track_weights = left[:, :available] * singular[:available]

    # Remove SVD's sign ambiguity so repeated exports remain byte-stable.
    for component in range(available):
        pivot = int(np.argmax(np.abs(right[component])))
        if right[component, pivot] < 0:
            right[component] *= -1.0
            track_weights[:, component] *= -1.0
    track_weights[0] = 0.0

    reconstruction = np.zeros_like(flattened)
    final_metrics: dict[str, float | int] = {}
    best_oral_corrective: LowRankVertexAnimation | None = None
    for rank in range(1, available + 1):
        reconstruction += np.outer(track_weights[:, rank - 1], right[rank - 1])
        reconstructed_frames = (
            reconstruction.reshape(values.shape) / coordinate_scale + base
        )
        vertex_error = np.linalg.norm(reconstructed_frames - values, axis=2)
        mesh_p95 = float(np.percentile(vertex_error, 95))
        mesh_max = float(np.max(vertex_error, initial=0.0))
        predicted_landmarks = _landmarks(
            reconstructed_frames, landmark_indices, landmark_weights
        )
        if observed_landmarks is None or predicted_landmarks is None:
            landmark_p95 = 0.0
            landmark_max = 0.0
        else:
            landmark_error = np.linalg.norm(
                predicted_landmarks - observed_landmarks, axis=2
            )
            landmark_p95 = float(np.percentile(landmark_error, 95))
            landmark_max = float(np.max(landmark_error, initial=0.0))
        final_metrics = {
            "rank": rank,
            "available_rank": available,
            "mesh_p95_m": mesh_p95,
            "mesh_max_m": mesh_max,
            "landmark_p95_m": landmark_p95,
            "landmark_max_m": landmark_max,
        }
        geometry_passed = bool(
            mesh_p95 <= mesh_p95_limit_m
            and mesh_max <= mesh_max_limit_m
            and landmark_p95 <= landmark_p95_limit_m
            and landmark_max <= landmark_max_limit_m
        )
        oral_semantics_passed = True
        if preserve_oral_semantics:
            if (
                predicted_landmarks is None
                or observed_contact is None
                or observed_order_risk is None
            ):
                raise AssertionError("oral semantic inputs disappeared during factorization")
            predicted_semantic_landmarks = _landmarks(
                reconstructed_frames,
                landmark_indices,
                semantic_weights,
            )
            if predicted_semantic_landmarks is None:
                raise AssertionError(
                    "oral semantic landmark evaluation unexpectedly failed"
                )
            _, predicted_contact, predicted_order_risk = classify_lip_landmarks(
                predicted_semantic_landmarks,
                contact_gap_interocular=lip_contact_gap_interocular,
                order_tolerance_interocular=(
                    lip_order_inversion_tolerance_interocular
                ),
            )
            contact_changes = int(np.count_nonzero(predicted_contact != observed_contact))
            introduced_order_risks = int(
                np.count_nonzero(predicted_order_risk & ~observed_order_risk)
            )
            introduced_tongue_teeth_collision = np.zeros(len(values), dtype=bool)
            if geometry_passed and observed_tongue_teeth_collision is not None:
                assert tongue_indices is not None and teeth_indices is not None
                predicted_collision = _tongue_teeth_collision_frames(
                    reconstructed_frames,
                    predicted_semantic_landmarks,
                    tongue_indices,
                    teeth_indices,
                    threshold_interocular=(
                        tongue_teeth_collision_risk_interocular
                    ),
                )
                introduced_tongue_teeth_collision = (
                    predicted_collision & ~observed_tongue_teeth_collision
                )
            introduced_collision_count = int(
                np.count_nonzero(introduced_tongue_teeth_collision)
            )
            final_metrics.update(
                {
                    "lip_contact_classification_changed_frames": contact_changes,
                    "introduced_lip_order_risk_frames": introduced_order_risks,
                    "introduced_tongue_teeth_collision_frames": (
                        introduced_collision_count
                    ),
                }
            )
            oral_semantics_passed = (
                contact_changes == 0
                and introduced_order_risks == 0
                and introduced_collision_count == 0
            )
            failed_semantic_frames = (
                (predicted_contact != observed_contact)
                | (predicted_order_risk & ~observed_order_risk)
                | introduced_tongue_teeth_collision
            )
            corrective_count = int(np.count_nonzero(failed_semantic_frames))
            # A numerically tiny oral displacement can sit below the global
            # SVD threshold while still crossing a signed lip-order/contact
            # boundary. If the geometry tolerances already pass, retain one
            # exact sparse residual target per affected sampled frame. This is
            # fail-closed and bounded by the caller's target cap; it does not
            # relax or relabel the oral semantic gate.
            if (
                geometry_passed
                and not oral_semantics_passed
                and corrective_count > 0
                and rank + corrective_count <= max_targets
            ):
                failed_indices = np.flatnonzero(failed_semantic_frames)
                svd_morph = (
                    right[:rank].reshape(rank, values.shape[1], 3)
                    / coordinate_scale[None, :, :]
                ).astype(np.float32)
                svd_weights = track_weights[:, :rank].astype(np.float32)
                svd_frames = base + np.einsum(
                    "fk,kvj->fvj", svd_weights, svd_morph
                )
                residual_morph = np.asarray(
                    values[failed_indices] - svd_frames[failed_indices],
                    dtype=np.float32,
                )
                residual_weights = np.zeros(
                    (len(values), corrective_count), dtype=np.float32
                )
                residual_weights[
                    failed_indices, np.arange(corrective_count)
                ] = 1.0
                candidate_morph = np.concatenate(
                    (svd_morph, residual_morph), axis=0
                )
                candidate_weights = np.concatenate(
                    (svd_weights, residual_weights), axis=1
                )
                candidate_frames = base + np.einsum(
                    "fk,kvj->fvj", candidate_weights, candidate_morph
                )
                candidate_vertex_error = np.linalg.norm(
                    candidate_frames - values, axis=2
                )
                candidate_mesh_p95 = float(
                    np.percentile(candidate_vertex_error, 95)
                )
                candidate_mesh_max = float(
                    np.max(candidate_vertex_error, initial=0.0)
                )
                candidate_landmarks = _landmarks(
                    candidate_frames, landmark_indices, landmark_weights
                )
                if candidate_landmarks is None:
                    raise AssertionError(
                        "oral corrective landmark evaluation unexpectedly failed"
                    )
                candidate_landmark_error = np.linalg.norm(
                    candidate_landmarks - observed_landmarks, axis=2
                )
                candidate_landmark_p95 = float(
                    np.percentile(candidate_landmark_error, 95)
                )
                candidate_landmark_max = float(
                    np.max(candidate_landmark_error, initial=0.0)
                )
                candidate_semantic_landmarks = _landmarks(
                    candidate_frames, landmark_indices, semantic_weights
                )
                if candidate_semantic_landmarks is None:
                    raise AssertionError(
                        "oral corrective semantic evaluation unexpectedly failed"
                    )
                _, candidate_contact, candidate_order_risk = classify_lip_landmarks(
                    candidate_semantic_landmarks,
                    contact_gap_interocular=lip_contact_gap_interocular,
                    order_tolerance_interocular=(
                        lip_order_inversion_tolerance_interocular
                    ),
                )
                candidate_collision_passed = True
                if observed_tongue_teeth_collision is not None:
                    assert tongue_indices is not None and teeth_indices is not None
                    candidate_collision = _tongue_teeth_collision_frames(
                        candidate_frames,
                        candidate_semantic_landmarks,
                        tongue_indices,
                        teeth_indices,
                        threshold_interocular=(
                            tongue_teeth_collision_risk_interocular
                        ),
                    )
                    candidate_collision_passed = not np.any(
                        candidate_collision & ~observed_tongue_teeth_collision
                    )
                candidate_passed = bool(
                    candidate_mesh_p95 <= mesh_p95_limit_m
                    and candidate_mesh_max <= mesh_max_limit_m
                    and candidate_landmark_p95 <= landmark_p95_limit_m
                    and candidate_landmark_max <= landmark_max_limit_m
                    and np.array_equal(candidate_contact, observed_contact)
                    and not np.any(candidate_order_risk & ~observed_order_risk)
                    and candidate_collision_passed
                )
                if candidate_passed:
                    corrective = LowRankVertexAnimation(
                        base_vertices=base,
                        morph_positions=candidate_morph,
                        weights=candidate_weights,
                        mesh_p95_m=candidate_mesh_p95,
                        mesh_max_m=candidate_mesh_max,
                        landmark_p95_m=candidate_landmark_p95,
                        landmark_max_m=candidate_landmark_max,
                        oral_corrective_targets=corrective_count,
                    )
                    if (
                        best_oral_corrective is None
                        or corrective.rank < best_oral_corrective.rank
                    ):
                        best_oral_corrective = corrective
        if (
            mesh_p95 <= mesh_p95_limit_m
            and mesh_max <= mesh_max_limit_m
            and landmark_p95 <= landmark_p95_limit_m
            and landmark_max <= landmark_max_limit_m
            and oral_semantics_passed
        ):
            direct = LowRankVertexAnimation(
                base_vertices=base,
                morph_positions=(
                    right[:rank].reshape(rank, values.shape[1], 3)
                    / coordinate_scale[None, :, :]
                ).astype(np.float32),
                weights=track_weights[:, :rank].astype(np.float32),
                mesh_p95_m=mesh_p95,
                mesh_max_m=mesh_max,
                landmark_p95_m=landmark_p95,
                landmark_max_m=landmark_max,
            )
            return (
                best_oral_corrective
                if best_oral_corrective is not None
                and best_oral_corrective.rank <= direct.rank
                else direct
            )
    if best_oral_corrective is not None:
        return best_oral_corrective
    raise AnimationCompressionError(
        f"Animation needs more than {max_targets} morph targets to pass reconstruction "
        "and oral-semantic gates",
        final_metrics,
    )


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    positions = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    face_normals = np.cross(
        positions[faces[:, 1]] - positions[faces[:, 0]],
        positions[faces[:, 2]] - positions[faces[:, 0]],
    )
    normals = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = length[:, 0] > 1e-12
    normals[valid] /= length[valid]
    normals[~valid] = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    return normals


def _vertex_tangents(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    """Compute deterministic glTF-compatible accumulated tangent frames.

    The accumulation and Gram-Schmidt step follow glTF's tangent convention;
    the fourth component stores bitangent handedness.  Degenerate UV faces get
    a stable normal-orthogonal fallback rather than NaN tangents.
    """

    positions = np.asarray(vertices, dtype=np.float32)
    normal_values = np.asarray(normals, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    coordinates = np.asarray(uvs, dtype=np.float32)
    tangent_sum = np.zeros_like(positions)
    bitangent_sum = np.zeros_like(positions)
    p0, p1, p2 = (positions[faces[:, corner]] for corner in range(3))
    uv0, uv1, uv2 = (coordinates[faces[:, corner]] for corner in range(3))
    edge1 = p1 - p0
    edge2 = p2 - p0
    delta1 = uv1 - uv0
    delta2 = uv2 - uv0
    determinant = delta1[:, 0] * delta2[:, 1] - delta1[:, 1] * delta2[:, 0]
    valid = np.abs(determinant) > 1e-12
    reciprocal = np.zeros_like(determinant)
    reciprocal[valid] = 1.0 / determinant[valid]
    face_tangent = (
        edge1 * delta2[:, 1, None] - edge2 * delta1[:, 1, None]
    ) * reciprocal[:, None]
    face_bitangent = (
        edge2 * delta1[:, 0, None] - edge1 * delta2[:, 0, None]
    ) * reciprocal[:, None]
    face_tangent[~valid] = 0.0
    face_bitangent[~valid] = 0.0
    for corner in range(3):
        np.add.at(tangent_sum, faces[:, corner], face_tangent)
        np.add.at(bitangent_sum, faces[:, corner], face_bitangent)

    tangent_xyz = tangent_sum - normal_values * np.sum(
        normal_values * tangent_sum, axis=1, keepdims=True
    )
    length = np.linalg.norm(tangent_xyz, axis=1)
    fallback = length <= 1e-12
    if np.any(fallback):
        normals_fallback = normal_values[fallback]
        axis = np.zeros_like(normals_fallback)
        choose_x = np.abs(normals_fallback[:, 0]) < 0.9
        axis[choose_x, 0] = 1.0
        axis[~choose_x, 1] = 1.0
        tangent_xyz[fallback] = np.cross(axis, normals_fallback)
        length[fallback] = np.linalg.norm(tangent_xyz[fallback], axis=1)
    tangent_xyz /= np.maximum(length[:, None], 1e-12)
    handedness = np.where(
        np.sum(np.cross(normal_values, tangent_xyz) * bitangent_sum, axis=1) < 0.0,
        -1.0,
        1.0,
    ).astype(np.float32)
    return np.column_stack((tangent_xyz, handedness)).astype(np.float32)


class _GLBBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: list[dict[str, int]] = []
        self.accessors: list[dict[str, object]] = []

    def blob(self, payload: bytes) -> int:
        """Append an untyped binary payload and return its buffer-view index."""

        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(payload)
        index = len(self.buffer_views)
        self.buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        )
        return index

    def accessor(
        self,
        array: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int | None = None,
        normalized: bool = False,
        bounds: bool = False,
    ) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes(order="C")
        self.data.extend(payload)
        view: dict[str, int] = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        view_index = len(self.buffer_views)
        self.buffer_views.append(view)
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor_type]
        if contiguous.size % components:
            raise ValueError("Accessor array size does not match its declared type")
        accessor: dict[str, object] = {
            "bufferView": view_index,
            "componentType": component_type,
            "count": int(contiguous.size // components),
            "type": accessor_type,
        }
        if normalized:
            accessor["normalized"] = True
        if bounds:
            rows = contiguous.reshape(-1, components)
            accessor["min"] = rows.min(axis=0).astype(float).tolist()
            accessor["max"] = rows.max(axis=0).astype(float).tolist()
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index


def _glb_bytes(document: dict[str, object], binary: bytes) -> bytes:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    padded_binary = binary + b"\0" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(padded_binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(encoded), b"JSON"),
            encoded,
            struct.pack("<I4s", len(padded_binary), b"BIN\0"),
            padded_binary,
        )
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def export_animated_gnm_glb(
    path: str | Path,
    adapter: GNMAdapter,
    frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_targets: int = 32,
    mapping_path: str | Path | None = None,
    texture_path: str | Path | None = None,
    triangle_uvs: np.ndarray | None = None,
    material_paths: Mapping[str, str | Path] | None = None,
    material_normal_encoding: str | None = None,
    runtime_material_paths: Mapping[str, str | Path] | None = None,
) -> AnimatedGLBExport:
    """Compress and export exact evaluated GNM frames as glTF morph animation."""

    times = np.asarray(timestamps, dtype=np.float32)
    values = np.asarray(frames, dtype=np.float32)
    if times.shape != (len(values),) or not np.isfinite(times).all():
        raise ValueError("timestamps must be one finite value per frame")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    selected_material_paths = dict(material_paths or {})
    selected_runtime_paths = dict(runtime_material_paths or {})
    if selected_runtime_paths and (selected_material_paths or texture_path is not None):
        raise ValueError(
            "runtime_material_paths cannot be combined with source material paths"
        )
    if texture_path is not None:
        existing_base = selected_material_paths.get("base_color")
        if existing_base is not None and Path(existing_base) != Path(texture_path):
            raise ValueError("texture_path conflicts with material_paths.base_color")
        selected_material_paths["base_color"] = texture_path
    if "normal" in selected_material_paths:
        if material_normal_encoding not in {"unorm", "signed_float"}:
            raise ValueError(
                "material_normal_encoding must explicitly be 'unorm' or "
                "'signed_float' when material_paths includes a normal map"
            )
    elif material_normal_encoding is not None:
        raise ValueError(
            "material_normal_encoding requires a normal source material path"
        )
    has_material = bool(selected_material_paths or selected_runtime_paths)
    has_normal_texture = (
        "normal" in selected_material_paths or "normal" in selected_runtime_paths
    )
    factor = factor_vertex_animation(
        values,
        max_targets=max_targets,
        landmark_indices=adapter.landmark_indices,
        landmark_weights=adapter.landmark_weights,
        preserve_oral_semantics=True,
        tongue_vertex_indices=np.flatnonzero(
            adapter.vertex_group("tongue") > 0.5
        ),
        teeth_vertex_indices=np.flatnonzero(
            adapter.vertex_group("teeth") > 0.5
        ),
    )
    selected_uvs = (
        np.asarray(adapter.model.triangle_uvs, dtype=np.float32)
        if triangle_uvs is None
        else np.asarray(triangle_uvs, dtype=np.float32)
    )
    split = split_triangle_corner_uvs(
        factor.base_vertices,
        adapter.triangles,
        selected_uvs,
    )
    source_map = split.source_vertices
    base_normals = _vertex_normals(factor.base_vertices, adapter.triangles)
    split_normals = base_normals[source_map]
    morph_positions = factor.morph_positions[:, source_map]
    # GNM stores triangle-corner UVs in lower-left convention. glTF samples
    # image rows from the upper-left, so flip V exactly once at export.
    gltf_uvs = split.uvs.copy()
    gltf_uvs[:, 1] = 1.0 - gltf_uvs[:, 1]
    base_tangents = (
        _vertex_tangents(split.positions, split_normals, split.triangles, gltf_uvs)
        if has_normal_texture
        else None
    )
    morph_normals = np.empty_like(morph_positions)
    morph_tangents = np.empty_like(morph_positions) if has_normal_texture else None
    for index, direction in enumerate(factor.morph_positions):
        target_normals = _vertex_normals(
            factor.base_vertices + direction, adapter.triangles
        )
        split_target_normals = target_normals[source_map]
        morph_normals[index] = split_target_normals - split_normals
        if morph_tangents is not None and base_tangents is not None:
            target_tangents = _vertex_tangents(
                split.positions + morph_positions[index],
                split_target_normals,
                split.triangles,
                gltf_uvs,
            )
            morph_tangents[index] = target_tangents[:, :3] - base_tangents[:, :3]
    builder = _GLBBuilder()
    position_accessor = builder.accessor(
        split.positions,
        component_type=5126,
        accessor_type="VEC3",
        target=34962,
        bounds=True,
    )
    normal_accessor = builder.accessor(
        split_normals,
        component_type=5126,
        accessor_type="VEC3",
        target=34962,
    )
    uv_accessor = builder.accessor(
        gltf_uvs,
        component_type=5126,
        accessor_type="VEC2",
        target=34962,
    )
    tangent_accessor = (
        builder.accessor(
            base_tangents,
            component_type=5126,
            accessor_type="VEC4",
            target=34962,
        )
        if base_tangents is not None
        else None
    )
    index_type = np.uint16 if len(split.positions) <= np.iinfo(np.uint16).max else np.uint32
    index_component = 5123 if index_type is np.uint16 else 5125
    index_accessor = builder.accessor(
        split.triangles.astype(index_type),
        component_type=index_component,
        accessor_type="SCALAR",
        target=34963,
        bounds=True,
    )
    targets: list[dict[str, int]] = []
    for index, (positions, normals) in enumerate(
        zip(morph_positions, morph_normals, strict=True)
    ):
        target = {
            "POSITION": builder.accessor(
                positions,
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
                bounds=True,
            ),
            "NORMAL": builder.accessor(
                normals,
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
            ),
        }
        if morph_tangents is not None:
            target["TANGENT"] = builder.accessor(
                morph_tangents[index],
                component_type=5126,
                accessor_type="VEC3",
                target=34962,
            )
        targets.append(target)
    time_accessor: int | None = None
    weight_accessor: int | None = None
    if factor.rank:
        time_accessor = builder.accessor(
            times,
            component_type=5126,
            accessor_type="SCALAR",
            bounds=True,
        )
        weight_accessor = builder.accessor(
            factor.weights,
            component_type=5126,
            accessor_type="SCALAR",
        )

    attributes: dict[str, int] = {
        "POSITION": position_accessor,
        "NORMAL": normal_accessor,
        "TEXCOORD_0": uv_accessor,
    }
    if tangent_accessor is not None:
        attributes["TANGENT"] = tangent_accessor
    if not has_material:
        colors = np.asarray(
            vertex_colors_module.get_vertex_colors(gnm_np=adapter.model),
            dtype=np.float32,
        )[source_map]
        rgba = np.column_stack(
            (
                np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8),
                np.full(len(colors), 255, dtype=np.uint8),
            )
        )
        attributes["COLOR_0"] = builder.accessor(
            rgba,
            component_type=5121,
            accessor_type="VEC4",
            target=34962,
            normalized=True,
        )
    primitive: dict[str, object] = {
        "attributes": attributes,
        "indices": index_accessor,
        "material": 0,
        "mode": 4,
    }
    if targets:
        primitive["targets"] = targets
    mesh: dict[str, object] = {
        "name": "GNM_Head_3_0",
        "primitives": [primitive],
        "extras": {
            "targetNames": [f"autoanim_{index:02d}" for index in range(factor.rank)]
        },
    }
    if factor.rank:
        mesh["weights"] = [0.0] * factor.rank
    material: dict[str, object] = {
        "name": "GNM character material" if has_material else "GNM anatomical preview",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.72,
        },
        "doubleSided": False,
    }
    images: list[dict[str, object]] | None = None
    textures: list[dict[str, object]] | None = None
    samplers: list[dict[str, object]] | None = None
    extensions_used: list[str] = []
    if has_material:
        runtime = (
            load_runtime_material_derivatives(selected_runtime_paths)
            if selected_runtime_paths
            else prepare_runtime_material(
                selected_material_paths,
                normal_encoding=material_normal_encoding or "unorm",
            )
        )
        images = []
        # Facial atlases do not tile. Clamping prevents linear/mipmap samples
        # near an island or atlas boundary from bleeding the opposite edge.
        samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        textures = []

        def add_texture(name: str, image: Image.Image) -> int:
            encoded = BytesIO()
            image.save(encoded, format="PNG", optimize=False, compress_level=9)
            image_view = builder.blob(encoded.getvalue())
            source_index = len(images)
            images.append(
                {
                    "name": name,
                    "bufferView": image_view,
                    "mimeType": "image/png",
                }
            )
            texture_index = len(textures)
            textures.append({"name": name, "sampler": 0, "source": source_index})
            return texture_index

        base_index = add_texture("Character base color", runtime.base_color)
        material["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": base_index
        }
        if runtime.normal is not None:
            material["normalTexture"] = {
                "index": add_texture("Character tangent-space normal", runtime.normal)
            }
        if runtime.metallic_roughness is not None:
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {
                "index": add_texture(
                    "Character metallic-roughness", runtime.metallic_roughness
                )
            }
            material["pbrMetallicRoughness"]["roughnessFactor"] = 1.0
        if runtime.specular_color is not None:
            specular_index = add_texture(
                "Character specular color", runtime.specular_color
            )
            material["extensions"] = {
                "KHR_materials_specular": {
                    "specularFactor": 1.0,
                    "specularColorFactor": [1.0, 1.0, 1.0],
                    "specularColorTexture": {"index": specular_index},
                }
            }
            extensions_used.append("KHR_materials_specular")
        material["extras"] = {
            "autoanim_runtime_projection": {
                "source_map_semantics": sorted(selected_material_paths),
                "runtime_derivative_semantics": sorted(selected_runtime_paths),
                "rendered_map_semantics": (
                    sorted(selected_runtime_paths)
                    if selected_runtime_paths
                    else sorted(
                        name
                        for name in selected_material_paths
                        if name
                        in {"base_color", "normal", "roughness", "specular_color"}
                    )
                ),
                "preserved_but_not_rendered": (
                    []
                    if selected_runtime_paths
                    else sorted(
                        set(selected_material_paths)
                        - {"base_color", "normal", "roughness", "specular_color"}
                    )
                ),
                "runtime_resolution": list(runtime.runtime_size),
                "source_normal_encoding": material_normal_encoding,
            }
        }

    document: dict[str, object] = {
        "asset": {
            "version": "2.0",
            "generator": "AutoAnim GNM animated exporter 1.0",
            "extras": {
                "gnm_version": "3.0",
                "coordinate_system": "+Y_up_+Z_forward_meters",
                "reconstruction": {
                    "rank": factor.rank,
                    "oral_corrective_targets": factor.oral_corrective_targets,
                    "mesh_p95_mm": factor.mesh_p95_m * 1000.0,
                    "mesh_max_mm": factor.mesh_max_m * 1000.0,
                    "landmark_p95_mm": factor.landmark_p95_m * 1000.0,
                    "landmark_max_mm": factor.landmark_max_m * 1000.0,
                },
            },
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "AutoAnim_GNM_Head_v3", "mesh": 0}],
        "meshes": [mesh],
        "materials": [material],
        "buffers": [{"byteLength": len(builder.data)}],
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
    }
    if images is not None and textures is not None and samplers is not None:
        document["images"] = images
        document["textures"] = textures
        document["samplers"] = samplers
    if extensions_used:
        document["extensionsUsed"] = extensions_used
    if factor.rank and time_accessor is not None and weight_accessor is not None:
        document["animations"] = [
            {
                "name": "autoanim",
                "samplers": [
                    {
                        "input": time_accessor,
                        "output": weight_accessor,
                        "interpolation": "LINEAR",
                    }
                ],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}],
            }
        ]
    output = Path(path)
    _atomic_bytes(output, _glb_bytes(document, bytes(builder.data)))
    mapping = Path(mapping_path) if mapping_path is not None else output.with_name(
        f"{output.stem}-mapping.npz"
    )
    write_npz(
        mapping,
        glb_vertex_to_gnm_vertex=source_map,
        triangles=split.triangles,
        uvs=split.uvs,
        uvs_lower_left=split.uvs,
        internal_uvs_lower_left=split.uvs,
        gltf_uvs_upper_left=gltf_uvs,
        timestamps=times,
        morph_weights=factor.weights,
        oral_corrective_targets=np.asarray(
            factor.oral_corrective_targets, dtype=np.int32
        ),
    )
    return AnimatedGLBExport(
        path=output,
        mapping_path=mapping,
        rank=factor.rank,
        frame_count=len(values),
        vertex_count=len(split.positions),
        triangle_count=len(split.triangles),
        mesh_p95_mm=factor.mesh_p95_m * 1000.0,
        mesh_max_mm=factor.mesh_max_m * 1000.0,
        landmark_p95_mm=factor.landmark_p95_m * 1000.0,
        landmark_max_mm=factor.landmark_max_m * 1000.0,
        oral_corrective_targets=factor.oral_corrective_targets,
    )
