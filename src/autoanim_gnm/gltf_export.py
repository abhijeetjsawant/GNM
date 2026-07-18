"""Seam-correct GLB export for GNM meshes.

GNM stores UVs per triangle corner while glTF stores attributes per vertex.
Vertices at UV seams therefore have to be duplicated.  This module keeps an
explicit mapping back to the original GNM vertex so the browser can update all
duplicates identically during animation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import trimesh

from gnm.shape.visualization import vertex_colors as vertex_colors_module

from .gnm_adapter import GNMAdapter
from .serialization import write_npz


@dataclass(frozen=True, slots=True)
class SeamSplitMesh:
    """One indexed mesh whose UVs are valid glTF per-vertex attributes."""

    positions: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray
    source_vertices: np.ndarray

    def validate(self, *, source_vertex_count: int) -> None:
        vertex_count = len(self.positions)
        if self.positions.shape != (vertex_count, 3):
            raise ValueError("positions must have shape [vertices,3]")
        if self.uvs.shape != (vertex_count, 2):
            raise ValueError("uvs must have shape [vertices,2]")
        if self.source_vertices.shape != (vertex_count,):
            raise ValueError("source_vertices must have shape [vertices]")
        if self.triangles.ndim != 2 or self.triangles.shape[1:] != (3,):
            raise ValueError("triangles must have shape [triangles,3]")
        arrays = (self.positions, self.uvs)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("seam-split geometry contains nonfinite values")
        if len(self.triangles) and (
            int(self.triangles.min()) < 0 or int(self.triangles.max()) >= vertex_count
        ):
            raise ValueError("triangle index is outside the seam-split vertex array")
        if len(self.source_vertices) and (
            int(self.source_vertices.min()) < 0
            or int(self.source_vertices.max()) >= source_vertex_count
        ):
            raise ValueError("source vertex mapping is outside the GNM mesh")
        if np.any(self.uvs < 0.0) or np.any(self.uvs > 1.0):
            raise ValueError("UV coordinates must be in [0,1]")


@dataclass(frozen=True, slots=True)
class GLBExport:
    path: Path
    mapping_path: Path
    vertex_count: int
    triangle_count: int
    seam_duplicates: int


def split_triangle_corner_uvs(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_uvs: np.ndarray,
) -> SeamSplitMesh:
    """Duplicate only vertices whose exact triangle-corner UVs differ.

    The first occurrence order is retained, making the result deterministic.
    UV float bit patterns are used in the key rather than a tolerance: the
    checked-in GNM atlas is authoritative and must survive export exactly.
    """

    source = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    corner_uvs = np.asarray(triangle_uvs, dtype=np.float32)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("vertices must have shape [vertices,3]")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("triangles must have shape [triangles,3]")
    if corner_uvs.shape != (len(faces), 3, 2):
        raise ValueError("triangle_uvs must have shape [triangles,3,2]")
    if not np.isfinite(source).all() or not np.isfinite(corner_uvs).all():
        raise ValueError("vertices and UVs must be finite")
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(source)):
        raise ValueError("triangle index is outside the source vertex array")

    lookup: dict[tuple[int, int, int], int] = {}
    split_positions: list[np.ndarray] = []
    split_uvs: list[np.ndarray] = []
    source_vertices: list[int] = []
    split_indices = np.empty(faces.size, dtype=np.int32)
    flattened_uvs = corner_uvs.reshape(-1, 2)
    for corner, (source_index, uv) in enumerate(
        zip(faces.reshape(-1), flattened_uvs, strict=True)
    ):
        # NumPy scalar .view preserves the exact checked-in float32 bits.
        key = (int(source_index), int(uv[0].view(np.uint32)), int(uv[1].view(np.uint32)))
        split_index = lookup.get(key)
        if split_index is None:
            split_index = len(split_positions)
            lookup[key] = split_index
            split_positions.append(source[int(source_index)])
            split_uvs.append(uv)
            source_vertices.append(int(source_index))
        split_indices[corner] = split_index

    result = SeamSplitMesh(
        positions=np.asarray(split_positions, dtype=np.float32),
        triangles=split_indices.reshape(-1, 3),
        uvs=np.asarray(split_uvs, dtype=np.float32),
        source_vertices=np.asarray(source_vertices, dtype=np.int32),
    )
    result.validate(source_vertex_count=len(source))
    return result


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


def export_gnm_glb(
    path: str | Path,
    adapter: GNMAdapter,
    vertices: np.ndarray,
    *,
    texture_path: str | Path | None = None,
    triangle_uvs: np.ndarray | None = None,
    mapping_path: str | Path | None = None,
) -> GLBExport:
    """Export one static GNM state as a validated, seam-correct GLB.

    When ``texture_path`` is absent, GNM's anatomical debug colors are used.
    The sidecar mapping is part of the contract for vertex-cache animation.
    """

    output = Path(path)
    source = np.asarray(vertices, dtype=np.float32)
    if source.shape != (adapter.model.num_vertices, 3):
        raise ValueError(
            f"Expected [{adapter.model.num_vertices},3] GNM vertices, got {source.shape}"
        )
    selected_uvs = (
        np.asarray(adapter.model.triangle_uvs, dtype=np.float32)
        if triangle_uvs is None
        else np.asarray(triangle_uvs, dtype=np.float32)
    )
    split = split_triangle_corner_uvs(
        source,
        adapter.triangles,
        selected_uvs,
    )
    if texture_path is None:
        colors = np.asarray(
            vertex_colors_module.get_vertex_colors(gnm_np=adapter.model),
            dtype=np.float32,
        )[split.source_vertices]
        rgba = np.column_stack(
            (
                np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8),
                np.full(len(colors), 255, dtype=np.uint8),
            )
        )
        visual: trimesh.visual.base.Visuals = trimesh.visual.ColorVisuals(
            vertex_colors=rgba
        )
    else:
        texture = Path(texture_path)
        if not texture.is_file():
            raise FileNotFoundError(texture)
        with Image.open(texture) as opened:
            image = opened.convert("RGBA").copy()
        visual = trimesh.visual.TextureVisuals(uv=split.uvs, image=image)

    mesh = trimesh.Trimesh(
        vertices=split.positions,
        faces=split.triangles,
        visual=visual,
        process=False,
        validate=False,
    )
    mesh.metadata.update(
        {
            "gnm_version": "3.0",
            "source_vertex_count": int(adapter.model.num_vertices),
            "seam_split_vertex_count": int(len(split.positions)),
        }
    )
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="GNM_Head_3_0", geom_name="GNM_Head_3_0")
    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    _atomic_bytes(output, payload)

    mapping = Path(mapping_path) if mapping_path is not None else output.with_name(
        f"{output.stem}-mapping.npz"
    )
    write_npz(
        mapping,
        glb_vertex_to_gnm_vertex=split.source_vertices,
        triangles=split.triangles,
        # Internal UVs are lower-left/Pillow coordinates.  Trimesh flips V
        # exactly once on GLB export to glTF's top-left convention.
        uvs=split.uvs,
        uvs_lower_left=split.uvs,
    )
    return GLBExport(
        path=output,
        mapping_path=mapping,
        vertex_count=len(split.positions),
        triangle_count=len(split.triangles),
        seam_duplicates=len(split.positions) - len(np.unique(split.source_vertices)),
    )
