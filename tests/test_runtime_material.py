from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

import autoanim_gnm.runtime_material as runtime_material
from autoanim_gnm.runtime_material import (
    load_runtime_material_derivatives,
    prepare_runtime_material,
    write_runtime_material_derivatives,
)


def _sources(root: Path) -> dict[str, Path]:
    Image.new("RGB", (3, 1), (64, 96, 128)).save(root / "base.png")
    Image.new("RGB", (3, 1), (128, 64, 255)).save(root / "normal.png")
    roughness = np.asarray([[0, 32768, 65535]], dtype=np.uint16)
    assert cv2.imwrite(str(root / "roughness.png"), roughness)
    specular = np.full((1, 3, 3), 32768, dtype=np.uint16)
    assert cv2.imwrite(str(root / "specular.png"), specular)
    return {
        "base_color": root / "base.png",
        "normal": root / "normal.png",
        "roughness": root / "roughness.png",
        "specular_color": root / "specular.png",
    }


def test_runtime_projection_preserves_16_bit_range_and_color_contract(
    tmp_path: Path,
) -> None:
    runtime = prepare_runtime_material(_sources(tmp_path))

    packed = np.asarray(runtime.metallic_roughness)
    np.testing.assert_array_equal(packed[0, :, 0], np.asarray([255, 255, 255]))
    np.testing.assert_allclose(packed[0, :, 1], np.asarray([0, 128, 255]), atol=1)
    np.testing.assert_array_equal(packed[0, :, 2], np.asarray([0, 0, 0]))
    # The package is lower-left/OpenGL while glTF UVs are upper-left.  The
    # tangent-space green channel is reflected exactly once.
    encoded_normal = np.asarray(runtime.normal, dtype=np.float32) / 255.0 * 2.0 - 1.0
    np.testing.assert_allclose(
        np.linalg.norm(encoded_normal, axis=2), np.ones((1, 3)), atol=0.02
    )
    assert np.all(np.asarray(runtime.normal)[0, :, 1] > 128)
    # A linear value of 0.5 encodes to approximately 0.735 in sRGB.
    np.testing.assert_allclose(
        np.asarray(runtime.specular_color), np.full((1, 3, 3), 188), atol=1
    )


def test_runtime_derivatives_are_deterministic_and_loaded_without_reprocessing(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path)
    first = write_runtime_material_derivatives(sources, tmp_path / "first")
    second = write_runtime_material_derivatives(sources, tmp_path / "second")

    assert set(first) == {
        "base_color",
        "normal",
        "metallic_roughness",
        "specular_color",
    }
    for semantic in first:
        assert first[semantic].read_bytes() == second[semantic].read_bytes()
    loaded = load_runtime_material_derivatives(first)
    assert loaded.base_color.mode == "RGBA"
    assert loaded.normal is not None and loaded.normal.mode == "RGB"
    assert loaded.metallic_roughness is not None
    assert loaded.specular_color is not None
    assert loaded.runtime_size == (3, 1)


def test_runtime_projection_has_explicit_source_and_viewer_lod_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    monkeypatch.setattr(runtime_material, "MAX_RUNTIME_TEXTURE_DIMENSION", 2)
    projected = prepare_runtime_material(sources)
    assert projected.runtime_size == (2, 1)

    monkeypatch.setattr(runtime_material, "MAX_RUNTIME_SOURCE_DIMENSION", 2)
    with pytest.raises(ValueError, match="attachment limit"):
        prepare_runtime_material(sources)


def test_signed_float_normal_encoding_is_never_guessed_from_pixel_minimum(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (2, 2), (64, 96, 128)).save(tmp_path / "base.png")
    signed = np.empty((2, 2, 3), dtype=np.float32)
    signed[..., 0] = 0.99
    signed[..., 1] = 0.10
    signed[..., 2] = 0.10
    assert cv2.imwrite(str(tmp_path / "normal.tiff"), signed)
    paths = {"base_color": tmp_path / "base.png", "normal": tmp_path / "normal.tiff"}

    projected = prepare_runtime_material(paths, normal_encoding="signed_float")
    incorrectly_unorm = prepare_runtime_material(paths, normal_encoding="unorm")

    assert projected.normal is not None and incorrectly_unorm.normal is not None
    assert not np.array_equal(
        np.asarray(projected.normal), np.asarray(incorrectly_unorm.normal)
    )

    with pytest.raises(ValueError, match="floating-point"):
        prepare_runtime_material(_sources(tmp_path), normal_encoding="signed_float")
