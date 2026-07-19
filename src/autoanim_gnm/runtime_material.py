"""Bounded projection of validated source-precision facial materials.

Source maps remain immutable and retain their native dtype. Browser derivatives
are deliberately bounded, 8-bit glTF textures. Native high-resolution sources
must use TIFF strips/tiles whose individual decoded segments fit the configured
budget; they are decoded to disk-backed memmaps and processed in row chunks.
PNG remains supported only while its complete decoded image fits the resident
memory budget.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Iterator, Mapping

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
RUNTIME_DERIVATIVE_KEYS = frozenset(
    {"base_color", "normal", "metallic_roughness", "specular_color"}
)

MAX_RUNTIME_SOURCE_DIMENSION = 8_192
MAX_RUNTIME_TEXTURE_DIMENSION = 4_096
MAX_RUNTIME_SOURCE_DECODED_BYTES = 1024 * 1024 * 1024
MAX_RUNTIME_RESIDENT_SOURCE_BYTES = 128 * 1024 * 1024
MAX_RUNTIME_TIFF_SEGMENT_BYTES = 64 * 1024 * 1024
RUNTIME_WORKING_CHUNK_BYTES = 8 * 1024 * 1024
RUNTIME_PROJECTION_PROFILE = "autoanim.browser-material-lod.v1"

_LOSSLESS_TIFF_COMPRESSIONS = frozenset(
    {
        "NONE",
        "ADOBE_DEFLATE",
        "DEFLATE",
        "PACKBITS",
    }
)
_SOURCE_CHANNELS = {
    "base_color": frozenset({3, 4}),
    "normal": frozenset({3}),
    "roughness": frozenset({1}),
    "specular_color": frozenset({3}),
}


@dataclass(frozen=True, slots=True)
class RuntimeMaterialImages:
    """Decoded browser images plus their bounded source-to-LOD contract."""

    base_color: Image.Image
    normal: Image.Image | None
    metallic_roughness: Image.Image | None
    specular_color: Image.Image | None
    runtime_size: tuple[int, int]
    source_size: tuple[int, int]
    lod_scale_factor: int
    projection_profile: str


@dataclass(frozen=True, slots=True)
class _RuntimeSource:
    path: Path
    semantic: str
    image_format: str
    width: int
    height: int
    channels: int
    dtype: np.dtype
    decoded_bytes: int
    maximum_segment_bytes: int


def _open_regular_nofollow(path: Path) -> tuple[int, os.stat_result]:
    if path.is_symlink():
        raise ValueError(f"Runtime material source may not be a symlink: {path}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError("This platform cannot safely open runtime material sources")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        )
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ValueError(f"Runtime material source is not a regular file: {path}")
    return descriptor, info


def _inspect_source(path: Path, *, semantic: str) -> _RuntimeSource:
    descriptor, _ = _open_regular_nofollow(path)
    try:
        suffix = path.suffix.lower()
        if suffix in {".tif", ".tiff"}:
            with os.fdopen(os.dup(descriptor), "rb") as image_file:
                with TiffFile(image_file, name=path.name) as tiff:
                    if len(tiff.pages) != 1:
                        raise ValueError(
                            "Runtime TIFF must contain exactly one image page"
                        )
                    page = tiff.pages[0]
                    width = int(page.imagewidth)
                    height = int(page.imagelength)
                    channels = int(page.samplesperpixel or 1)
                    dtype = np.dtype(page.dtype)
                    expected_axes = "YXS" if channels > 1 else "YX"
                    if (
                        int(page.imagedepth or 1) != 1
                        or int(page.tiledepth or 1) != 1
                        or str(page.axes) != expected_axes
                        or tuple(page.shape)
                        != (
                            (height, width, channels)
                            if channels > 1
                            else (height, width)
                        )
                        or bool(page.subifds)
                    ):
                        raise ValueError(
                            "Runtime TIFF must be one 2D YX/YXS image without SubIFDs"
                        )
                    if np.issubdtype(dtype, np.signedinteger):
                        raise ValueError(
                            "Runtime integer maps must use unsigned samples"
                        )
                    if channels > 1 and int(page.planarconfig or 1) != 1:
                        raise ValueError(
                            "Runtime TIFF must use contiguous interleaved samples"
                        )
                    orientation = page.tags.get("Orientation")
                    if orientation is not None and int(orientation.value) != 1:
                        raise ValueError(
                            "Runtime TIFF orientation transforms are unsupported"
                        )
                    compression = getattr(
                        page.compression, "name", str(page.compression)
                    )
                    if compression not in _LOSSLESS_TIFF_COMPRESSIONS:
                        raise ValueError(
                            "Runtime TIFF compression must be lossless and supported"
                        )
                    photometric = getattr(
                        page.photometric, "name", str(page.photometric)
                    )
                    expected_photometric = (
                        "RGB" if channels in {3, 4} else "MINISBLACK"
                    )
                    if photometric != expected_photometric:
                        raise ValueError(
                            f"Runtime {semantic} TIFF photometric "
                            f"{photometric!r} is unsupported"
                        )
                    if page.is_tiled:
                        segment_width = min(width, int(page.tilewidth or width))
                        segment_height = min(
                            height, int(page.tilelength or height)
                        )
                    else:
                        segment_width = width
                        segment_height = min(
                            height, int(page.rowsperstrip or height)
                        )
                    maximum_segment_bytes = int(
                        segment_width
                        * segment_height
                        * channels
                        * dtype.itemsize
                    )
                    if not 1 <= len(page.dataoffsets) <= 16_384:
                        raise ValueError(
                            "Runtime TIFF strip/tile count exceeds the bounded profile"
                        )
            image_format = "TIFF"
        elif suffix == ".png":
            with os.fdopen(os.dup(descriptor), "rb") as image_file:
                with Image.open(image_file) as opened:
                    if (
                        opened.format != "PNG"
                        or int(getattr(opened, "n_frames", 1)) != 1
                    ):
                        raise ValueError(
                            "Runtime PNG must contain exactly one PNG image"
                        )
                    width, height = opened.size
                    mode_channels = {
                        "L": 1,
                        "I;16": 1,
                        "I;16B": 1,
                        "RGB": 3,
                        "RGBA": 4,
                    }
                    channels = mode_channels.get(opened.mode, 4)
            # Budget against 16-bit samples; the exact dtype is verified after
            # bounded decode. This intentionally errs on the safe side.
            dtype = np.dtype(np.uint16)
            maximum_segment_bytes = int(
                width * height * channels * dtype.itemsize
            )
            image_format = "PNG"
        else:
            raise ValueError("Runtime material maps must be PNG or TIFF")
    except (TiffFileError, OSError, ValueError) as exc:
        raise ValueError(
            f"Could not inspect runtime {semantic} map: {path}"
        ) from exc
    finally:
        os.close(descriptor)

    decoded_bytes = int(width * height * channels * dtype.itemsize)
    if (
        width < 1
        or height < 1
        or width > MAX_RUNTIME_SOURCE_DIMENSION
        or height > MAX_RUNTIME_SOURCE_DIMENSION
        or decoded_bytes > MAX_RUNTIME_SOURCE_DECODED_BYTES
    ):
        raise ValueError(
            f"Runtime {semantic} source dimensions {(width, height)} or decoded "
            f"bytes exceed the bounded {MAX_RUNTIME_SOURCE_DIMENSION}px attachment limit"
        )
    if semantic in _SOURCE_CHANNELS and channels not in _SOURCE_CHANNELS[semantic]:
        raise ValueError(
            f"Runtime {semantic} has {channels} channels; expected "
            f"{sorted(_SOURCE_CHANNELS[semantic])}"
        )
    if image_format == "PNG" and decoded_bytes > MAX_RUNTIME_RESIDENT_SOURCE_BYTES:
        raise ValueError(
            f"Runtime {semantic} PNG exceeds the resident decode budget; use a "
            "bounded-strip/tiled TIFF for native high-resolution sources"
        )
    if image_format == "TIFF" and maximum_segment_bytes > MAX_RUNTIME_TIFF_SEGMENT_BYTES:
        raise ValueError(
            f"Runtime {semantic} source has an unsafe TIFF strip/tile size"
        )
    return _RuntimeSource(
        path=path,
        semantic=semantic,
        image_format=image_format,
        width=int(width),
        height=int(height),
        channels=int(channels),
        dtype=dtype,
        decoded_bytes=decoded_bytes,
        maximum_segment_bytes=maximum_segment_bytes,
    )


@contextmanager
def _decoded_source(source: _RuntimeSource) -> Iterator[np.ndarray]:
    descriptor, before = _open_regular_nofollow(source.path)
    decoded: np.ndarray | None = None
    mapped: np.memmap | None = None
    image_file = None
    tiff = None
    decode_scratch = None
    try:
        if source.image_format == "TIFF":
            image_file = os.fdopen(os.dup(descriptor), "rb")
            tiff = TiffFile(image_file, name=source.path.name)
            page = tiff.pages[0]
            decode_scratch = tempfile.TemporaryFile(
                prefix="autoanim-runtime-material-"
            )
            filesystem = os.fstatvfs(decode_scratch.fileno())
            available_scratch_bytes = int(
                filesystem.f_bavail * filesystem.f_frsize
            )
            if available_scratch_bytes < (
                source.decoded_bytes + 128 * 1024 * 1024
            ):
                raise ValueError(
                    f"Insufficient scratch space to project {source.semantic} safely"
                )
            mapped = page.asarray(
                out=decode_scratch,
                maxworkers=1,
                buffersize=RUNTIME_WORKING_CHUNK_BYTES,
            )
            decoded = mapped
        else:
            chunks: list[bytes] = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(
                    descriptor,
                    min(8 * 1024 * 1024, before.st_size - offset),
                    offset,
                )
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
            if offset != before.st_size:
                raise ValueError(
                    f"Runtime source changed while reading: {source.path}"
                )
            encoded = np.frombuffer(b"".join(chunks), dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if decoded is not None and decoded.ndim == 3:
                if decoded.shape[2] == 3:
                    decoded = decoded[..., ::-1]
                elif decoded.shape[2] == 4:
                    decoded = decoded[..., [2, 1, 0, 3]]
        if decoded is None or decoded.size == 0:
            raise ValueError(
                f"Could not decode runtime {source.semantic}: {source.path}"
            )
        if decoded.ndim == 2:
            decoded = decoded[..., None]
        if (
            decoded.ndim != 3
            or decoded.shape[:2] != (source.height, source.width)
            or decoded.shape[2] not in {1, 3, 4}
        ):
            raise ValueError(
                f"Runtime {source.semantic} decode does not match inspected metadata"
            )
        if np.issubdtype(decoded.dtype, np.signedinteger):
            raise ValueError(
                f"Runtime {source.semantic} integer maps must use unsigned samples"
            )
        yield decoded
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError(
                f"Runtime source changed during projection: {source.path}"
            )
    finally:
        if mapped is not None:
            mapping = getattr(mapped, "_mmap", None)
            if mapping is not None:
                mapping.close()
        if tiff is not None:
            tiff.close()
        if image_file is not None and not image_file.closed:
            image_file.close()
        if decode_scratch is not None:
            decode_scratch.close()
        os.close(descriptor)


def _lod_factor(size: tuple[int, int]) -> int:
    factor = 1
    while max(
        math.ceil(size[0] / factor), math.ceil(size[1] / factor)
    ) > int(MAX_RUNTIME_TEXTURE_DIMENSION):
        factor *= 2
    return factor


def _unit_values(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.unsignedinteger):
        return values.astype(np.float32) / float(np.iinfo(values.dtype).max)
    if np.issubdtype(values.dtype, np.floating):
        result = values.astype(np.float32)
        if not bool(np.isfinite(result).all()):
            raise ValueError("Runtime material source contains nonfinite values")
        return result
    raise ValueError(f"Unsupported runtime material dtype: {values.dtype}")


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.04045,
        clipped / 12.92,
        np.power((clipped + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def _block_average(values: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return values
    height, width, channels = values.shape
    output_height = math.ceil(height / factor)
    output_width = math.ceil(width / factor)
    padded = np.zeros(
        (output_height * factor, output_width * factor, channels),
        dtype=np.float32,
    )
    padded[:height, :width] = values
    reduced = padded.reshape(
        output_height, factor, output_width, factor, channels
    ).sum(axis=(1, 3))
    y_counts = np.minimum(
        factor, height - np.arange(output_height, dtype=np.int64) * factor
    )
    x_counts = np.minimum(
        factor, width - np.arange(output_width, dtype=np.int64) * factor
    )
    reduced /= (y_counts[:, None] * x_counts[None, :])[..., None]
    return reduced


def _project_source(
    source: _RuntimeSource,
    *,
    factor: int,
    normal_encoding: str,
) -> np.ndarray:
    target_height = math.ceil(source.height / factor)
    target_width = math.ceil(source.width / factor)
    output_channels = 4 if source.semantic == "base_color" else 3
    if source.semantic == "roughness":
        output_channels = 1
    output = np.empty(
        (target_height, target_width, output_channels), dtype=np.uint8
    )
    bytes_per_target_row = max(
        1,
        factor
        * source.width
        * max(1, source.channels)
        * np.dtype(np.float32).itemsize
        * 2,
    )
    target_rows_per_chunk = max(
        1, RUNTIME_WORKING_CHUNK_BYTES // bytes_per_target_row
    )

    with _decoded_source(source) as decoded:
        if source.semantic == "normal" and normal_encoding == "signed_float":
            if not np.issubdtype(decoded.dtype, np.floating):
                raise ValueError(
                    "signed_float normal encoding requires floating-point pixels"
                )
        for target_start in range(0, target_height, target_rows_per_chunk):
            target_stop = min(
                target_height, target_start + target_rows_per_chunk
            )
            source_start = target_start * factor
            source_stop = min(source.height, target_stop * factor)
            values = _unit_values(
                np.asarray(decoded[source_start:source_stop])
            )
            source_minimum = float(np.min(values))
            source_maximum = float(np.max(values))
            expected_minimum = (
                -1.0
                if source.semantic == "normal"
                and normal_encoding == "signed_float"
                else 0.0
            )
            if source_minimum < expected_minimum or source_maximum > 1.0:
                raise ValueError(
                    f"Runtime {source.semantic} source values are outside the "
                    f"declared [{expected_minimum}, 1] encoding"
                )

            if source.semantic == "base_color":
                if values.shape[2] not in {3, 4}:
                    raise ValueError(
                        "Runtime base color must have three or four channels"
                    )
                rgb = _block_average(
                    _srgb_to_linear(values[..., :3]), factor
                )
                rgb = _linear_to_srgb(rgb)
                alpha = (
                    _block_average(values[..., 3:4], factor)
                    if values.shape[2] == 4
                    else np.ones((*rgb.shape[:2], 1), dtype=np.float32)
                )
                projected = np.concatenate((rgb, alpha), axis=2)
            elif source.semantic == "normal":
                if values.shape[2] != 3:
                    raise ValueError(
                        "Runtime normal map must have three channels"
                    )
                vectors = (
                    values
                    if normal_encoding == "signed_float"
                    else values * 2.0 - 1.0
                )
                vectors = _block_average(vectors, factor)
                lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
                if bool(np.any(lengths <= 1.0e-6)):
                    raise ValueError(
                        "Runtime normal map contains a degenerate vector"
                    )
                projected = vectors / lengths * 0.5 + 0.5
                # GNM source uses lower-left V; glTF pixels use top-left row
                # order, therefore tangent-space Y is reflected exactly once.
                projected[..., 1] = 1.0 - projected[..., 1]
            elif source.semantic == "specular_color":
                if values.shape[2] != 3:
                    raise ValueError(
                        "Runtime specular color must have three channels"
                    )
                projected = _linear_to_srgb(
                    _block_average(values, factor)
                )
            elif source.semantic == "roughness":
                if values.shape[2] != 1:
                    raise ValueError(
                        "Runtime roughness must have one channel"
                    )
                projected = _block_average(values, factor)
            else:
                raise ValueError(
                    f"Unsupported runtime projection semantic {source.semantic}"
                )

            if not bool(np.isfinite(projected).all()):
                raise ValueError(
                    f"Runtime {source.semantic} projection contains nonfinite values"
                )
            output[target_start:target_stop] = np.clip(
                np.rint(projected * 255.0), 0.0, 255.0
            ).astype(np.uint8)
    return output


def _image_from_array(values: np.ndarray) -> Image.Image:
    if values.ndim == 3 and values.shape[2] == 1:
        return Image.fromarray(values[..., 0]).copy()
    return Image.fromarray(values).copy()


def _aligned_sources(
    paths: Mapping[str, str | Path],
) -> tuple[dict[str, _RuntimeSource], tuple[int, int], int]:
    unknown = set(paths) - PRESERVED_SEMANTICS
    if unknown:
        raise ValueError(f"Unknown material semantics: {sorted(unknown)}")
    if "base_color" not in paths:
        raise ValueError("A runtime material requires base_color")
    resolved = {name: Path(value) for name, value in paths.items()}
    sources = {
        name: _inspect_source(path, semantic=name)
        for name, path in resolved.items()
    }
    original_size = (
        sources["base_color"].width,
        sources["base_color"].height,
    )
    mismatched_sizes = {
        name: (source.width, source.height)
        for name, source in sources.items()
        if (source.width, source.height) != original_size
    }
    if mismatched_sizes:
        raise ValueError(
            f"Runtime material source maps are not dimension-aligned: {mismatched_sizes}"
        )
    return sources, original_size, _lod_factor(original_size)


def prepare_runtime_material(
    paths: Mapping[str, str | Path],
    *,
    normal_convention: str = "gnm_lower_left_opengl",
    normal_encoding: str = "unorm",
) -> RuntimeMaterialImages:
    """Derive a deterministic, bounded browser LOD from source maps.

    Downsampling uses power-of-two linear-light box filtering. Tangent normals
    are filtered as vectors and renormalized; roughness stays linear; linear
    specular multipliers are encoded to sRGB for ``KHR_materials_specular``.
    """

    if normal_convention != "gnm_lower_left_opengl":
        raise ValueError(
            "normal_convention must be 'gnm_lower_left_opengl' for the current exporter"
        )
    if normal_encoding not in {"unorm", "signed_float"}:
        raise ValueError("normal_encoding must be 'unorm' or 'signed_float'")

    sources, original_size, factor = _aligned_sources(paths)
    expected_size = (
        math.ceil(original_size[0] / factor),
        math.ceil(original_size[1] / factor),
    )

    base_array = _project_source(
        sources["base_color"], factor=factor, normal_encoding=normal_encoding
    )
    base = _image_from_array(base_array)
    del base_array
    if base.mode != "RGBA" or base.size != expected_size:
        raise ValueError("Runtime base-color projection has an invalid result")

    normal: Image.Image | None = None
    if "normal" in sources:
        normal_array = _project_source(
            sources["normal"], factor=factor, normal_encoding=normal_encoding
        )
        normal = _image_from_array(normal_array)
        del normal_array

    specular: Image.Image | None = None
    if "specular_color" in sources:
        specular_array = _project_source(
            sources["specular_color"],
            factor=factor,
            normal_encoding=normal_encoding,
        )
        specular = _image_from_array(specular_array)
        del specular_array

    roughness: Image.Image | None = None
    if "roughness" in sources:
        roughness_array = _project_source(
            sources["roughness"],
            factor=factor,
            normal_encoding=normal_encoding,
        )
        roughness = _image_from_array(roughness_array)
        del roughness_array

    for semantic, image in (
        ("normal", normal),
        ("specular_color", specular),
        ("roughness", roughness),
    ):
        if image is not None and image.size != base.size:
            raise ValueError(
                f"Runtime {semantic} dimensions {image.size} do not match base color {base.size}"
            )

    packed: Image.Image | None = None
    if roughness is not None:
        white = Image.new("L", base.size, 255)
        black = Image.new("L", base.size, 0)
        packed = Image.merge("RGB", (white, roughness, black))

    return RuntimeMaterialImages(
        base_color=base,
        normal=normal,
        metallic_roughness=packed,
        specular_color=specular,
        runtime_size=(int(base.size[0]), int(base.size[1])),
        source_size=(int(original_size[0]), int(original_size[1])),
        lod_scale_factor=int(factor),
        projection_profile=RUNTIME_PROJECTION_PROFILE,
    )


def write_runtime_material_derivatives(
    source_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    normal_encoding: str = "unorm",
) -> dict[str, Path]:
    """Seal deterministic 8-bit glTF derivatives beside retained sources."""

    if normal_encoding not in {"unorm", "signed_float"}:
        raise ValueError("normal_encoding must be 'unorm' or 'signed_float'")
    sources, _, factor = _aligned_sources(source_paths)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for source_semantic, output_semantic in (
        ("base_color", "base_color"),
        ("normal", "normal"),
        ("roughness", "metallic_roughness"),
        ("specular_color", "specular_color"),
    ):
        source = sources.get(source_semantic)
        if source is None:
            continue
        projected = _project_source(
            source, factor=factor, normal_encoding=normal_encoding
        )
        if source_semantic == "roughness":
            height, width = projected.shape[:2]
            packed = np.empty((height, width, 3), dtype=np.uint8)
            packed[..., 0] = 255
            packed[..., 1] = projected[..., 0]
            packed[..., 2] = 0
            del projected
            projected = packed
        image = _image_from_array(projected)
        del projected
        path = destination / f"runtime-{output_semantic.replace('_', '-')}.png"
        try:
            image.save(path, format="PNG", optimize=False, compress_level=9)
        finally:
            image.close()
        written[output_semantic] = path
    return written


def load_runtime_material_derivatives(
    paths: Mapping[str, str | Path],
) -> RuntimeMaterialImages:
    """Load already hash-verified bounded glTF textures without reprocessing."""

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
            if max(opened.size) > MAX_RUNTIME_TEXTURE_DIMENSION:
                raise ValueError(
                    f"Runtime {semantic} exceeds the browser LOD dimension limit"
                )
            image = opened.copy()
            image.load()
        if size is None:
            size = image.size
        elif image.size != size:
            raise ValueError(
                "Runtime material derivatives are not dimension-aligned"
            )
        loaded[semantic] = image
    assert size is not None
    return RuntimeMaterialImages(
        base_color=loaded["base_color"],
        normal=loaded.get("normal"),
        metallic_roughness=loaded.get("metallic_roughness"),
        specular_color=loaded.get("specular_color"),
        runtime_size=(int(size[0]), int(size[1])),
        source_size=(int(size[0]), int(size[1])),
        lod_scale_factor=1,
        projection_profile=RUNTIME_PROJECTION_PROFILE,
    )


__all__ = [
    "PRESERVED_SEMANTICS",
    "MAX_RUNTIME_RESIDENT_SOURCE_BYTES",
    "MAX_RUNTIME_SOURCE_DECODED_BYTES",
    "MAX_RUNTIME_SOURCE_DIMENSION",
    "MAX_RUNTIME_TEXTURE_DIMENSION",
    "MAX_RUNTIME_TIFF_SEGMENT_BYTES",
    "RUNTIME_DERIVATIVE_KEYS",
    "RUNTIME_PROJECTION_PROFILE",
    "RUNTIME_SEMANTICS",
    "RuntimeMaterialImages",
    "load_runtime_material_derivatives",
    "prepare_runtime_material",
    "write_runtime_material_derivatives",
]
