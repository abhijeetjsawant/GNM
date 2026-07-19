"""Runtime-safe projection of a validated facial material package.

The production package retains every measured map at source precision.  glTF's
core real-time material can consume only a subset directly, so this module
builds deterministic PIL images for that subset without changing the retained
source assets or overstating renderer support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from PIL import Image
from tifffile import TiffFile, TiffFileError


RUNTIME_SEMANTICS = frozenset(
    {"base_color", "normal", "roughness", "specular_color"}
)
PRESERVED_SEMANTICS = frozenset(
    {
        "base_color",
        "normal",
        "displacement",
        "specular_color",
        "roughness",
        "subsurface_color",
        "subsurface_radius",
        "confidence",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialImages:
    """Decoded images ready for glTF material encoding."""

    base_color: Image.Image
    normal: Image.Image | None
    metallic_roughness: Image.Image | None
    specular_color: Image.Image | None
    runtime_size: tuple[int, int]


RUNTIME_DERIVATIVE_KEYS = frozenset(
    {"base_color", "normal", "metallic_roughness", "specular_color"}
)
MAX_RUNTIME_SOURCE_DIMENSION = 8_192
MAX_RUNTIME_TEXTURE_DIMENSION = 4_096


def _source_size(path: Path, *, semantic: str) -> tuple[int, int]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            with TiffFile(path) as tiff:
                if len(tiff.pages) != 1:
                    raise ValueError("Runtime TIFF must contain exactly one image page")
                width = int(tiff.pages[0].imagewidth)
                height = int(tiff.pages[0].imagelength)
        else:
            with Image.open(path) as opened:
                width, height = opened.size
    except (TiffFileError, OSError, ValueError) as exc:
        raise ValueError(f"Could not inspect runtime {semantic} map: {path}") from exc
    if (
        width < 1
        or height < 1
        or width > MAX_RUNTIME_SOURCE_DIMENSION
        or height > MAX_RUNTIME_SOURCE_DIMENSION
    ):
        raise ValueError(
            f"Runtime {semantic} source dimensions {(width, height)} exceed the "
            f"{MAX_RUNTIME_SOURCE_DIMENSION}px attachment limit; bake an offline LOD"
        )
    return int(width), int(height)


def _decode_unit(
    path: Path,
    *,
    semantic: str,
    target_size: tuple[int, int],
    normal_encoding: str,
) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"Could not decode runtime {semantic} map: {path}")
    if decoded.ndim == 2:
        decoded = decoded[..., None]
    if decoded.ndim != 3 or decoded.shape[2] not in {1, 3, 4}:
        raise ValueError(f"Unsupported runtime {semantic} image shape: {decoded.shape}")
    if decoded.shape[2] == 3:
        decoded = decoded[..., ::-1]
    elif decoded.shape[2] == 4:
        decoded = decoded[..., [2, 1, 0, 3]]
    if (decoded.shape[1], decoded.shape[0]) != target_size:
        decoded = cv2.resize(decoded, target_size, interpolation=cv2.INTER_AREA)
        if decoded.ndim == 2:
            decoded = decoded[..., None]
    if np.issubdtype(decoded.dtype, np.integer):
        values = decoded.astype(np.float32) / float(np.iinfo(decoded.dtype).max)
    elif np.issubdtype(decoded.dtype, np.floating):
        values = decoded.astype(np.float32)
    else:
        raise ValueError(f"Unsupported runtime {semantic} dtype: {decoded.dtype}")
    if semantic == "normal" and normal_encoding == "signed_float":
        if not np.issubdtype(decoded.dtype, np.floating):
            raise ValueError("signed_float normal encoding requires floating-point pixels")
        values = values * 0.5 + 0.5
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"Runtime {semantic} map contains nonfinite values")
    return np.clip(values, 0.0, 1.0)


def _quantize(values: np.ndarray, mode: str) -> Image.Image:
    encoded = np.clip(np.rint(values * 255.0), 0.0, 255.0).astype(np.uint8)
    if mode == "L":
        encoded = encoded[..., 0]
    return Image.fromarray(encoded, mode=mode)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def prepare_runtime_material(
    paths: Mapping[str, str | Path],
    *,
    normal_convention: str = "gnm_lower_left_opengl",
    normal_encoding: str = "unorm",
) -> RuntimeMaterialImages:
    """Decode the glTF-supported material subset with aligned dimensions.

    Roughness is packed into the green channel required by glTF's
    metallic-roughness texture.  Blue is zero because skin is a dielectric;
    red is set to white and intentionally unused (there is no occlusion map in
    the current validated facial package schema).
    """

    unknown = set(paths) - PRESERVED_SEMANTICS
    if unknown:
        raise ValueError(f"Unknown material semantics: {sorted(unknown)}")
    if "base_color" not in paths:
        raise ValueError("A runtime material requires base_color")

    if normal_convention != "gnm_lower_left_opengl":
        raise ValueError(
            "normal_convention must be 'gnm_lower_left_opengl' for the current exporter"
        )
    if normal_encoding not in {"unorm", "signed_float"}:
        raise ValueError("normal_encoding must be 'unorm' or 'signed_float'")
    resolved = {name: Path(value) for name, value in paths.items()}
    source_sizes = {
        name: _source_size(path, semantic=name) for name, path in resolved.items()
    }
    original_size = source_sizes["base_color"]
    mismatched_sizes = {
        name: size for name, size in source_sizes.items() if size != original_size
    }
    if mismatched_sizes:
        raise ValueError(
            f"Runtime material source maps are not dimension-aligned: {mismatched_sizes}"
        )
    scale = min(
        1.0,
        MAX_RUNTIME_TEXTURE_DIMENSION / float(max(original_size)),
    )
    target_size = (
        max(1, int(round(original_size[0] * scale))),
        max(1, int(round(original_size[1] * scale))),
    )
    base_values = _decode_unit(
        resolved["base_color"],
        semantic="base_color",
        target_size=target_size,
        normal_encoding=normal_encoding,
    )
    if base_values.shape[2] == 3:
        base_values = np.concatenate(
            (base_values, np.ones((*base_values.shape[:2], 1), dtype=np.float32)),
            axis=2,
        )
    base = _quantize(base_values, "RGBA")
    del base_values
    size = base.size

    normal: Image.Image | None = None
    if "normal" in resolved:
        normal_values = _decode_unit(
            resolved["normal"],
            semantic="normal",
            target_size=target_size,
            normal_encoding=normal_encoding,
        )
        if normal_values.shape[2] != 3:
            raise ValueError("Runtime normal map must have three channels")
        vectors = normal_values * 2.0 - 1.0
        lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
        if np.any(lengths <= 1.0e-6):
            raise ValueError("Runtime normal map contains a degenerate vector")
        normal_values = vectors / lengths * 0.5 + 0.5
        # GNM package UVs use a lower-left V axis while glTF uses top-left
        # texture coordinates.  The exporter flips V, so the tangent-space Y
        # component must also be reflected to preserve the measured direction.
        normal_values = normal_values.copy()
        normal_values[..., 1] = 1.0 - normal_values[..., 1]
        normal = _quantize(normal_values, "RGB")
        del normal_values
    specular: Image.Image | None = None
    if "specular_color" in resolved:
        specular_values = _decode_unit(
            resolved["specular_color"],
            semantic="specular_color",
            target_size=target_size,
            normal_encoding=normal_encoding,
        )
        if specular_values.shape[2] != 3:
            raise ValueError("Runtime specular color map must have three channels")
        # KHR_materials_specular defines specularColorTexture as an sRGB
        # multiplier over the dielectric F0, whereas the retained package
        # deliberately stores that multiplier in linear space.  Absolute F0
        # maps require an explicit IOR conversion and are rejected by the
        # attachment semantic contract rather than silently misrendered.
        specular = _quantize(_linear_to_srgb(specular_values), "RGB")
        del specular_values
    roughness: Image.Image | None = None
    if "roughness" in resolved:
        roughness_values = _decode_unit(
            resolved["roughness"],
            semantic="roughness",
            target_size=target_size,
            normal_encoding=normal_encoding,
        )
        if roughness_values.shape[2] != 1:
            raise ValueError("Runtime roughness map must have one channel")
        roughness = _quantize(roughness_values, "L")
        del roughness_values
    for semantic, image in (
        ("normal", normal),
        ("specular_color", specular),
        ("roughness", roughness),
    ):
        if image is not None and image.size != size:
            raise ValueError(
                f"Runtime {semantic} dimensions {image.size} do not match base color {size}"
            )

    packed: Image.Image | None = None
    if roughness is not None:
        white = Image.new("L", size, 255)
        black = Image.new("L", size, 0)
        packed = Image.merge("RGB", (white, roughness, black))

    return RuntimeMaterialImages(
        base_color=base,
        normal=normal,
        metallic_roughness=packed,
        specular_color=specular,
        runtime_size=(int(size[0]), int(size[1])),
    )


def write_runtime_material_derivatives(
    source_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    normal_encoding: str = "unorm",
) -> dict[str, Path]:
    """Seal deterministic 8-bit glTF derivatives beside retained source maps."""

    runtime = prepare_runtime_material(
        source_paths, normal_encoding=normal_encoding
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    images = {
        "base_color": runtime.base_color,
        "normal": runtime.normal,
        "metallic_roughness": runtime.metallic_roughness,
        "specular_color": runtime.specular_color,
    }
    written: dict[str, Path] = {}
    for semantic, image in images.items():
        if image is None:
            continue
        path = destination / f"runtime-{semantic.replace('_', '-')}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        written[semantic] = path
    return written


def load_runtime_material_derivatives(
    paths: Mapping[str, str | Path],
) -> RuntimeMaterialImages:
    """Load already-derived, hash-verified glTF textures without reprocessing."""

    unknown = set(paths) - RUNTIME_DERIVATIVE_KEYS
    if unknown or "base_color" not in paths:
        raise ValueError(
            f"Invalid runtime material derivative keys: {sorted(unknown)}"
        )
    expected_modes = {
        "base_color": "RGBA",
        "normal": "RGB",
        "metallic_roughness": "RGB",
        "specular_color": "RGB",
    }
    loaded: dict[str, Image.Image] = {}
    size: tuple[int, int] | None = None
    for semantic, value in paths.items():
        path = Path(value)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.mode != expected_modes[semantic]:
                raise ValueError(
                    f"Runtime {semantic} must be a {expected_modes[semantic]} PNG"
                )
            image = opened.copy()
            image.load()
        if size is None:
            size = image.size
        elif image.size != size:
            raise ValueError("Runtime material derivatives are not dimension-aligned")
        loaded[semantic] = image
    assert size is not None
    return RuntimeMaterialImages(
        base_color=loaded["base_color"],
        normal=loaded.get("normal"),
        metallic_roughness=loaded.get("metallic_roughness"),
        specular_color=loaded.get("specular_color"),
        runtime_size=(int(size[0]), int(size[1])),
    )


__all__ = [
    "PRESERVED_SEMANTICS",
    "MAX_RUNTIME_SOURCE_DIMENSION",
    "MAX_RUNTIME_TEXTURE_DIMENSION",
    "RUNTIME_DERIVATIVE_KEYS",
    "RUNTIME_SEMANTICS",
    "RuntimeMaterialImages",
    "load_runtime_material_derivatives",
    "prepare_runtime_material",
    "write_runtime_material_derivatives",
]
