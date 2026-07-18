"""GNM-specific UV packing and anatomical texture fallbacks.

GNM Head 3.0 ships six independent mesh components.  Each component owns a
complete local ``[0, 1]`` UV domain, so the raw ``triangle_uvs`` array is not a
single atlas: skin, both eyes, both teeth/gum meshes, and tongue overlap one
another.  This module turns those six local domains into one deterministic,
non-overlapping atlas suitable for a single glTF material.

The UVs returned here use the conventional mesh/Pillow convention used by
``texture_baker`` and trimesh: ``v=0`` addresses the bottom of a top-to-bottom
image array.  Trimesh converts that convention to glTF's top-left texture
coordinate convention while writing the GLB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .gnm_adapter import GNMAdapter


GNM_TEXTURE_COMPONENTS = (
    "skin",
    "left_eye",
    "right_eye",
    "upper_teeth_and_gums",
    "lower_teeth_and_gums",
    "tongue",
)

# The skin receives 72% of the atlas.  The remaining strip is split in
# proportion to component triangle count, retaining useful detail for teeth
# while keeping every anatomical part in an isolated tile.
_SKIN_WIDTH = 0.72


@dataclass(frozen=True, slots=True)
class GNMTextureAtlas:
    """A deterministic single-material atlas derived from GNM local UVs."""

    triangle_uvs: np.ndarray
    triangle_components: np.ndarray
    component_names: tuple[str, ...]
    component_bounds: Mapping[str, tuple[float, float, float, float]]
    generic_vertex_colors: np.ndarray
    padding_texels: float
    width: int
    height: int

    @property
    def layout_id(self) -> str:
        return "gnm-head-v3-repacked-1"


def _triangle_component_partition(
    adapter: GNMAdapter,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return and strictly validate GNM's official triangle partition."""

    actual_names = tuple(str(name) for name in adapter.model.mesh_component_names)
    if actual_names != GNM_TEXTURE_COMPONENTS:
        raise RuntimeError(
            "Unexpected GNM mesh component order: "
            f"expected {GNM_TEXTURE_COMPONENTS}, got {actual_names}"
        )
    triangle_count = len(adapter.triangles)
    labels = np.full(triangle_count, -1, dtype=np.int16)
    groups: list[np.ndarray] = []
    for component_index, name in enumerate(GNM_TEXTURE_COMPONENTS):
        indices = np.asarray(
            adapter.model.triangle_indices_for_group(name), dtype=np.int32
        )
        if indices.ndim != 1 or len(indices) == 0:
            raise RuntimeError(f"GNM component {name!r} has no triangle indices")
        if np.any(labels[indices] >= 0):
            duplicate = indices[labels[indices] >= 0]
            raise RuntimeError(
                f"GNM component triangles overlap at triangle {int(duplicate[0])}"
            )
        labels[indices] = component_index
        groups.append(indices)
    missing = np.flatnonzero(labels < 0)
    if len(missing):
        raise RuntimeError(
            f"GNM mesh component partition omits triangle {int(missing[0])}"
        )
    return labels, tuple(groups)


def _component_bounds(
    groups: tuple[np.ndarray, ...],
) -> dict[str, tuple[float, float, float, float]]:
    bounds: dict[str, tuple[float, float, float, float]] = {
        "skin": (0.0, 0.0, _SKIN_WIDTH, 1.0)
    }
    remaining_counts = np.asarray(
        [len(group) for group in groups[1:]], dtype=np.float64
    )
    heights = remaining_counts / float(np.sum(remaining_counts))
    # Pack from the top down in the declared component order so the two eyes
    # remain adjacent in the atlas and the tongue occupies the bottom tile.
    top = 1.0
    for name, height in zip(GNM_TEXTURE_COMPONENTS[1:], heights, strict=True):
        bottom = top - float(height)
        bounds[name] = (_SKIN_WIDTH, bottom, 1.0, top)
        top = bottom
    # Avoid accumulated float error at the bottom boundary.
    tongue = bounds["tongue"]
    bounds["tongue"] = (tongue[0], 0.0, tongue[2], tongue[3])
    return bounds


def _anatomical_vertex_colors(adapter: GNMAdapter) -> np.ndarray:
    """Return honest non-photographic colors for unseen GNM anatomy."""

    model = adapter.model
    colors = np.zeros((model.num_vertices, 3), dtype=np.uint8)

    # Component defaults make every vertex safe even when it is not part of a
    # more specific semantic group (notably pupils/inner eye geometry).
    defaults = {
        "skin": (178, 142, 126),
        "left_eye": (48, 35, 30),
        "right_eye": (48, 35, 30),
        "upper_teeth_and_gums": (164, 92, 96),
        "lower_teeth_and_gums": (164, 92, 96),
        "tongue": (154, 69, 82),
    }
    for name, value in defaults.items():
        colors[np.asarray(model.vertex_group_indices(name), dtype=np.int32)] = value

    # Finer official groups distinguish the anatomy contained inside a mesh
    # component.  These colors are purposefully plausible but generic; they do
    # not imply that an unobserved iris, tooth, or gum came from a photograph.
    overrides = {
        "mouth_sock": (92, 47, 50),
        "scleras": (224, 219, 207),
        "irises": (91, 60, 42),
        "gums": (164, 92, 96),
        "teeth": (229, 223, 207),
        "tongue": (154, 69, 82),
    }
    available = set(str(name) for name in model.vertex_group_names)
    for name, value in overrides.items():
        if name in available:
            colors[np.asarray(model.vertex_group_indices(name), dtype=np.int32)] = value

    if np.any(np.all(colors == 0, axis=1)):
        first = int(np.flatnonzero(np.all(colors == 0, axis=1))[0])
        raise RuntimeError(f"GNM anatomical fallback omits vertex {first}")
    return colors


def build_gnm_texture_atlas(
    adapter: GNMAdapter,
    texture_size: int | tuple[int, int],
    *,
    padding_texels: float = 1.5,
) -> GNMTextureAtlas:
    """Repack all GNM components into a non-overlapping single atlas.

    Padding is measured in output texels rather than UV units.  This ensures
    bilinear filtering never reaches a neighbouring component, including at
    the smallest supported 128px diagnostic atlas.
    """

    if isinstance(texture_size, int):
        height = width = texture_size
    else:
        if len(texture_size) != 2:
            raise ValueError("texture_size must be an integer or (height,width)")
        height, width = texture_size
    if (
        not isinstance(height, (int, np.integer))
        or not isinstance(width, (int, np.integer))
        or int(height) < 2
        or int(width) < 2
    ):
        raise ValueError("texture dimensions must be integers >= 2")
    height, width = int(height), int(width)
    if not np.isfinite(padding_texels) or padding_texels < 0.5:
        raise ValueError("padding_texels must be finite and at least 0.5")

    labels, groups = _triangle_component_partition(adapter)
    bounds = _component_bounds(groups)
    source_uvs = np.asarray(adapter.model.triangle_uvs, dtype=np.float64)
    if source_uvs.shape != (len(adapter.triangles), 3, 2):
        raise RuntimeError("GNM triangle UV array does not match its topology")
    if (
        not np.all(np.isfinite(source_uvs))
        or np.any(source_uvs < 0.0)
        or np.any(source_uvs > 1.0)
    ):
        raise RuntimeError("GNM local triangle UVs must be finite and in [0,1]")

    packed = np.empty_like(source_uvs)
    pad_u = float(padding_texels) / max(width - 1, 1)
    pad_v = float(padding_texels) / max(height - 1, 1)
    for name, indices in zip(GNM_TEXTURE_COMPONENTS, groups, strict=True):
        left, bottom, right, top = bounds[name]
        usable_width = right - left - 2.0 * pad_u
        usable_height = top - bottom - 2.0 * pad_v
        if usable_width <= 0.0 or usable_height <= 0.0:
            raise ValueError(
                f"texture_size is too small for padded GNM component {name!r}"
            )
        local = source_uvs[indices]
        packed[indices, :, 0] = left + pad_u + local[:, :, 0] * usable_width
        packed[indices, :, 1] = bottom + pad_v + local[:, :, 1] * usable_height

    return GNMTextureAtlas(
        triangle_uvs=packed.astype(np.float32),
        triangle_components=labels,
        component_names=GNM_TEXTURE_COMPONENTS,
        component_bounds=bounds,
        generic_vertex_colors=_anatomical_vertex_colors(adapter),
        padding_texels=float(padding_texels),
        width=width,
        height=height,
    )


__all__ = [
    "GNM_TEXTURE_COMPONENTS",
    "GNMTextureAtlas",
    "build_gnm_texture_atlas",
]
