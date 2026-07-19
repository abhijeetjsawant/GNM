"""Standard glTF morph animation for verified GNM vertex tracks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from PIL import Image

from gnm.shape.visualization import vertex_colors as vertex_colors_module

from .gltf_export import split_triangle_corner_uvs
from .gnm_adapter import GNMAdapter
from .serialization import write_npz


class AnimationCompressionError(ValueError):
    """The target cap cannot reconstruct a vertex animation faithfully."""

    def __init__(self, message: str, metrics: dict[str, float | int]):
        super().__init__(message)
        self.metrics = metrics


@dataclass(frozen=True, slots=True)
class LowRankVertexAnimation:
    base_vertices: np.ndarray
    morph_positions: np.ndarray
    weights: np.ndarray
    mesh_p95_m: float
    mesh_max_m: float
    landmark_p95_m: float
    landmark_max_m: float

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


def _landmarks(
    vertices: np.ndarray,
    indices: np.ndarray | None,
    weights: np.ndarray | None,
) -> np.ndarray | None:
    if indices is None or weights is None:
        return None
    selected = vertices[..., indices, :]
    return np.sum(selected * weights[..., None], axis=-2)


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
) -> LowRankVertexAnimation:
    """Find the smallest deterministic morph basis that passes geometry gates."""

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

    base = values[0].copy()
    flattened = (values - base).reshape(len(values), -1)
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

    observed_landmarks = _landmarks(values, landmark_indices, landmark_weights)
    reconstruction = np.zeros_like(flattened)
    final_metrics: dict[str, float | int] = {}
    for rank in range(1, available + 1):
        reconstruction += np.outer(track_weights[:, rank - 1], right[rank - 1])
        reconstructed_frames = reconstruction.reshape(values.shape) + base
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
        if (
            mesh_p95 <= mesh_p95_limit_m
            and mesh_max <= mesh_max_limit_m
            and landmark_p95 <= landmark_p95_limit_m
            and landmark_max <= landmark_max_limit_m
        ):
            return LowRankVertexAnimation(
                base_vertices=base,
                morph_positions=right[:rank].reshape(rank, values.shape[1], 3).astype(
                    np.float32
                ),
                weights=track_weights[:, :rank].astype(np.float32),
                mesh_p95_m=mesh_p95,
                mesh_max_m=mesh_max,
                landmark_p95_m=landmark_p95,
                landmark_max_m=landmark_max,
            )
    raise AnimationCompressionError(
        f"Animation needs more than {max_targets} morph targets to pass reconstruction gates",
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
) -> AnimatedGLBExport:
    """Compress and export exact evaluated GNM frames as glTF morph animation."""

    times = np.asarray(timestamps, dtype=np.float32)
    values = np.asarray(frames, dtype=np.float32)
    if times.shape != (len(values),) or not np.isfinite(times).all():
        raise ValueError("timestamps must be one finite value per frame")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    factor = factor_vertex_animation(
        values,
        max_targets=max_targets,
        landmark_indices=adapter.landmark_indices,
        landmark_weights=adapter.landmark_weights,
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
    morph_normals = np.empty_like(morph_positions)
    for index, direction in enumerate(factor.morph_positions):
        target_normals = _vertex_normals(
            factor.base_vertices + direction, adapter.triangles
        )
        morph_normals[index] = target_normals[source_map] - split_normals

    # GNM stores triangle-corner UVs in lower-left convention. glTF samples
    # image rows from the upper-left, so flip V exactly once at export.
    gltf_uvs = split.uvs.copy()
    gltf_uvs[:, 1] = 1.0 - gltf_uvs[:, 1]
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
    for positions, normals in zip(morph_positions, morph_normals, strict=True):
        targets.append(
            {
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
        )
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
    if texture_path is None:
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
        "name": "GNM character material" if texture_path is not None else "GNM anatomical preview",
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
    if texture_path is not None:
        source_texture = Path(texture_path)
        if not source_texture.is_file():
            raise FileNotFoundError(source_texture)
        with Image.open(source_texture) as opened:
            image = opened.convert("RGBA")
            encoded = BytesIO()
            image.save(encoded, format="PNG", optimize=False, compress_level=9)
        image_view = builder.blob(encoded.getvalue())
        images = [{"name": "Character base color", "bufferView": image_view, "mimeType": "image/png"}]
        samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        textures = [{"name": "Character base color", "sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    document: dict[str, object] = {
        "asset": {
            "version": "2.0",
            "generator": "AutoAnim GNM animated exporter 1.0",
            "extras": {
                "gnm_version": "3.0",
                "coordinate_system": "+Y_up_+Z_forward_meters",
                "reconstruction": {
                    "rank": factor.rank,
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
        internal_uvs_lower_left=split.uvs,
        gltf_uvs_upper_left=gltf_uvs,
        timestamps=times,
        morph_weights=factor.weights,
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
    )
