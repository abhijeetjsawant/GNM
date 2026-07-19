from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
import tifffile

from autoanim_gnm.cli import build_parser
from autoanim_gnm.materials import MaterialValidationError, validate_material_package
import autoanim_gnm.materials as materials_module
from autoanim_gnm.runtime_material import write_runtime_material_derivatives


NOW = datetime(2026, 7, 19, tzinfo=timezone.utc)
SOURCE_HASH = "1" * 64


def test_material_validator_is_exposed_by_cli() -> None:
    parsed = build_parser().parse_args(
        ["material", "/capture/package", "--spec", "/capture/material.json"]
    )
    assert parsed.command == "material"
    assert parsed.package_root == Path("/capture/package")


def _write_rgb(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    Image.new("RGB", size, (96, 128, 160)).save(path)


def _write_gray(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    Image.new("L", size, 127).save(path)


def _write_displacement(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    pixels = np.full((size[1], size[0]), 32768, dtype=np.uint16)
    Image.fromarray(pixels).save(path)


def _entry(
    path: str,
    color_space: str,
    *,
    source_resolution: list[int] | None = None,
    resampling: str = "none",
) -> dict[str, object]:
    return {
        "layout": "atlas",
        "path": path,
        "color_space": color_space,
        "source_resolution": source_resolution or [16, 16],
        "resampling": resampling,
    }


def _package(tmp_path: Path) -> dict[str, object]:
    rgb = {
        "base_color": "srgb",
        "normal": "linear",
        "specular_color": "linear",
        "subsurface_color": "srgb",
        "subsurface_radius": "linear",
    }
    gray = {"roughness": "linear", "confidence": "linear"}
    inventory: dict[str, object] = {}
    for semantic, color_space in rgb.items():
        filename = f"{semantic}.png"
        if semantic == "normal":
            Image.new("RGB", (16, 16), (128, 128, 255)).save(tmp_path / filename)
        else:
            _write_rgb(tmp_path / filename)
        inventory[semantic] = _entry(filename, color_space)
        if semantic == "normal":
            inventory[semantic]["normal_encoding"] = "unorm"
    for semantic, color_space in gray.items():
        filename = f"{semantic}.png"
        _write_gray(tmp_path / filename)
        inventory[semantic] = _entry(filename, color_space)
    _write_displacement(tmp_path / "displacement.png")
    inventory["displacement"] = _entry("displacement.png", "linear")
    _write_gray(tmp_path / "mask_skin.png")
    inventory["masks"] = {"skin": _entry("mask_skin.png", "linear")}

    map_names = [
        "base_color",
        "normal",
        "displacement",
        "specular_color",
        "roughness",
        "subsurface_color",
        "subsurface_radius",
        "confidence",
        "masks.skin",
    ]
    lineage = {
        name: {"operation": "derived", "source_sha256s": [SOURCE_HASH]}
        for name in map_names
    }
    return {
        "package_root": tmp_path,
        "package_id": "actor-alex-v001",
        "inventory": inventory,
        "capture": {
            "capture_id": "capture-001",
            "captured_at": "2026-07-01T12:00:00Z",
            "method": "multiview_passive",
            "devices": ["calibrated-camera-a"],
            "polarized": False,
            "spatial_resolution_mm_per_pixel": 0.25,
            "calibration_sha256": "2" * 64,
        },
        "provenance": {
            "producer": "AutoAnim test fixture",
            "pipeline": "fixture-baker",
            "pipeline_version": "1.0.0",
            "created_at": "2026-07-02T12:00:00+00:00",
            "source_sha256s": [SOURCE_HASH],
            "processing_log_sha256": "3" * 64,
            "map_lineage": lineage,
        },
        "rights": {
            "status": "cleared",
            "commercial_allowed": True,
            "subject_consent_attested": True,
            "scope": "commercial",
            "evidence_ref": "release://actor-alex/2026-07-01",
            "evidence_sha256": "4" * 64,
            "expires_at": "2027-07-01T00:00:00Z",
        },
        "claims": {
            "resolution_label": "unclaimed",
            "native_resolution": True,
            "pore_resolved": False,
            "relightable": False,
        },
        "now": NOW,
    }


def _validate(package: dict[str, object]) -> dict[str, object]:
    return validate_material_package(**package)  # type: ignore[arg-type]


def test_valid_package_is_deterministic_strict_json_with_file_evidence(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    first = _validate(package)
    second = _validate(package)

    assert first == second
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert first["schema_version"] == "autoanim.material-package.v2"
    assert first["totals"]["file_count"] == 9
    assert first["totals"]["bytes"] > 0
    base = first["maps"]["base_color"]["files"]["atlas"]
    assert base["path"] == "base_color.png"
    assert base["width"] == 16
    assert base["height"] == 16
    assert base["channels"] == 3
    assert base["sha256"] == hashlib.sha256(
        (tmp_path / "base_color.png").read_bytes()
    ).hexdigest()
    assert base["bytes"] == (tmp_path / "base_color.png").stat().st_size
    assert base["decoded_bytes"] == 16 * 16 * 3
    assert base["decode_strategy"] == "bounded_resident_png"
    assert first["totals"]["decoded_bytes"] > 0
    assert first["quality_evidence"]["pore_claim_gate_passed"] is False
    assert first["quality_evidence"]["relightable_claim_gate_passed"] is False
    assert first["quality_evidence"]["pore_frequency_validation_performed"] is False
    assert first["quality_evidence"]["unseen_light_validation_performed"] is False


def test_normal_encoding_is_explicit_and_signed_float_is_not_inferred(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    normal_entry = package["inventory"]["normal"]
    del normal_entry["normal_encoding"]
    with pytest.raises(MaterialValidationError) as missing:
        _validate(package)
    assert missing.value.code == "INVALID_SCHEMA"

    signed = np.empty((16, 16, 3), dtype=np.float32)
    signed[..., 0] = 0.99
    signed[..., 1] = 0.10
    signed[..., 2] = 0.10
    assert cv2.imwrite(str(tmp_path / "normal.tiff"), signed)
    normal_entry.update(
        {
            "path": "normal.tiff",
            "normal_encoding": "signed_float",
        }
    )
    manifest = _validate(package)
    assert manifest["maps"]["normal"]["normal_encoding"] == "signed_float"

    normal_entry["path"] = "normal.png"
    with pytest.raises(MaterialValidationError) as integer_signed:
        _validate(package)
    assert integer_signed.value.code == "NORMAL_ENCODING_DTYPE_MISMATCH"


def test_symlink_asset_is_rejected_even_when_target_is_a_valid_image(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    target = tmp_path / "outside.png"
    _write_gray(target)
    link = tmp_path / "roughness.png"
    link.unlink()
    link.symlink_to(target)

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "SYMLINK_FORBIDDEN"


def test_aligned_atlas_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    _write_gray(tmp_path / "roughness.png", (12, 16))
    package["inventory"]["roughness"]["source_resolution"] = [12, 16]

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "DIMENSION_MISMATCH"


def test_image_magic_must_match_extension(tmp_path: Path) -> None:
    package = _package(tmp_path)
    Image.new("RGB", (16, 16), (0, 0, 0)).save(
        tmp_path / "base_color.jpg", format="PNG"
    )
    package["inventory"]["base_color"]["path"] = "base_color.jpg"

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "IMAGE_FORMAT_MISMATCH"


@pytest.mark.parametrize(
    ("status", "commercial", "expected"),
    [
        ("unknown", True, "RIGHTS_NOT_CLEARED"),
        ("cleared", False, "COMMERCIAL_RIGHTS_REQUIRED"),
    ],
)
def test_unknown_or_noncommercial_rights_fail_closed(
    tmp_path: Path, status: str, commercial: bool, expected: str
) -> None:
    package = _package(tmp_path)
    package["rights"]["status"] = status
    package["rights"]["commercial_allowed"] = commercial

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == expected


def test_8k_claim_is_rejected_when_inventory_was_upsampled(tmp_path: Path) -> None:
    package = _package(tmp_path)
    for name, entry in package["inventory"].items():
        values = entry.values() if name == "masks" else [entry]
        for value in values:
            value["source_resolution"] = [8, 8]
            value["resampling"] = "upsampled"
    package["claims"]["native_resolution"] = False
    package["claims"]["resolution_label"] = "8k"

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "UPSAMPLED_RESOLUTION_CLAIM"


def test_pore_claim_rejects_inferred_geometry_provenance(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package["capture"]["method"] = "polarized_multilight"
    package["capture"]["polarized"] = True
    package["capture"]["spatial_resolution_mm_per_pixel"] = 0.04
    package["provenance"]["map_lineage"]["normal"]["operation"] = "inferred"
    package["claims"]["resolution_label"] = "8k"
    package["claims"]["pore_resolved"] = True

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "FALSE_PORE_CLAIM"
    assert "Inferred" in str(raised.value)


def test_pore_claim_rejects_4k_even_with_high_detail_attestation(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package["capture"]["method"] = "polarized_multilight"
    package["capture"]["polarized"] = True
    package["capture"]["spatial_resolution_mm_per_pixel"] = 0.04
    package["claims"]["resolution_label"] = "4k"
    package["claims"]["pore_resolved"] = True
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "FALSE_PORE_CLAIM"
    assert "8K" in str(raised.value)


def test_relightable_claim_requires_polarized_capture_evidence(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package["claims"]["relightable"] = True

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "FALSE_RELIGHTABLE_CLAIM"


def test_normal_map_must_contain_plausible_encoded_vectors(tmp_path: Path) -> None:
    package = _package(tmp_path)
    Image.new("RGB", (16, 16), (128, 128, 128)).save(tmp_path / "normal.png")

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "NORMAL_VECTOR_MISMATCH"


def test_float_map_with_nan_pixels_is_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    pixels = np.full((16, 16), 0.5, dtype=np.float32)
    pixels[4, 7] = np.nan
    assert cv2.imwrite(str(tmp_path / "roughness.tiff"), pixels)
    package["inventory"]["roughness"]["path"] = "roughness.tiff"

    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "NONFINITE_PIXELS"


def test_nonfinite_json_metadata_and_unknown_keys_are_rejected(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    package["capture"]["spatial_resolution_mm_per_pixel"] = float("nan")
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "NONFINITE_JSON"

    package = _package(tmp_path)
    package["claims"]["marketing_copy"] = "cinema ready"
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "INVALID_SCHEMA"


def test_parent_traversal_and_mixed_layout_are_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    package["inventory"]["base_color"]["path"] = "../base_color.png"
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "UNSAFE_PATH"

    package = _package(tmp_path)
    normal = package["inventory"]["normal"]
    normal.pop("path")
    normal["layout"] = "udim"
    normal["tiles"] = {"1001": "normal.png"}
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "LAYOUT_MISMATCH"


def test_udim_sets_must_cover_identical_tiles(tmp_path: Path) -> None:
    package = _package(tmp_path)
    for name, entry in package["inventory"].items():
        values = entry.values() if name == "masks" else [entry]
        for value in values:
            path = value.pop("path")
            value["layout"] = "udim"
            value["tiles"] = {"1001": path}
    valid = _validate(package)
    assert valid["quality_evidence"]["layout"] == "udim"

    _write_gray(tmp_path / "roughness.1002.png")
    package["inventory"]["roughness"]["tiles"]["1002"] = "roughness.1002.png"
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "UDIM_TILE_MISMATCH"


def test_decode_bomb_is_rejected_from_tiff_metadata_before_pixel_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    bomb = np.zeros((1024, 1024), dtype=np.uint16)
    tifffile.imwrite(
        tmp_path / "roughness.tiff",
        bomb,
        compression="deflate",
        tile=(256, 256),
        photometric="minisblack",
    )
    package["inventory"]["roughness"].update(
        {"path": "roughness.tiff", "source_resolution": [1024, 1024]}
    )
    monkeypatch.setattr(
        materials_module, "MAX_MATERIAL_DECODED_BYTES_PER_FILE", 512 * 1024
    )
    called = False

    def forbidden_asarray(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pixel decoder must not run")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden_asarray)
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert called is False


def test_tiff_volume_and_oversized_strip_are_rejected(tmp_path: Path) -> None:
    package = _package(tmp_path)
    tifffile.imwrite(
        tmp_path / "roughness.tiff",
        np.zeros((2, 16, 16), dtype=np.uint16),
        photometric="minisblack",
        metadata={"axes": "ZYX"},
        volumetric=True,
    )
    package["inventory"]["roughness"]["path"] = "roughness.tiff"
    with pytest.raises(MaterialValidationError) as volume:
        _validate(package)
    assert volume.value.code == "TIFF_DIMENSIONAL_LAYOUT_UNSUPPORTED"

    package = _package(tmp_path)
    tifffile.imwrite(
        tmp_path / "roughness.tiff",
        np.zeros((16, 16), dtype=np.uint16),
        photometric="minisblack",
        rowsperstrip=16,
    )
    package["inventory"]["roughness"]["path"] = "roughness.tiff"
    previous = materials_module.MAX_TIFF_SEGMENT_DECODED_BYTES
    materials_module.MAX_TIFF_SEGMENT_DECODED_BYTES = 64
    try:
        with pytest.raises(MaterialValidationError) as strip:
            _validate(package)
    finally:
        materials_module.MAX_TIFF_SEGMENT_DECODED_BYTES = previous
    assert strip.value.code == "TIFF_SEGMENT_LIMIT_EXCEEDED"


def test_signed_integer_and_nonopaque_base_alpha_fail_closed(tmp_path: Path) -> None:
    package = _package(tmp_path)
    tifffile.imwrite(
        tmp_path / "roughness.tiff",
        np.zeros((16, 16), dtype=np.int16),
        photometric="minisblack",
    )
    package["inventory"]["roughness"]["path"] = "roughness.tiff"
    with pytest.raises(MaterialValidationError) as signed:
        _validate(package)
    assert signed.value.code == "PIXEL_DTYPE_MISMATCH"

    package = _package(tmp_path)
    rgba = np.full((16, 16, 4), 255, dtype=np.uint8)
    rgba[7, 9, 3] = 128
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "base_color.png")
    with pytest.raises(MaterialValidationError) as alpha:
        _validate(package)
    assert alpha.value.code == "BASE_COLOR_ALPHA_MISMATCH"


def test_partial_png_and_nested_symlink_fail_closed(tmp_path: Path) -> None:
    package = _package(tmp_path)
    normal = tmp_path / "normal.png"
    normal.write_bytes(normal.read_bytes()[:24])
    with pytest.raises(MaterialValidationError) as partial:
        _validate(package)
    assert partial.value.code == "IMAGE_DECODE_FAILED"

    package = _package(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    target = nested / "base.png"
    _write_rgb(target)
    link = tmp_path / "linked"
    link.symlink_to(nested, target_is_directory=True)
    package["inventory"]["base_color"]["path"] = "linked/base.png"
    with pytest.raises(MaterialValidationError) as symlink:
        _validate(package)
    assert symlink.value.code == "SYMLINK_FORBIDDEN"


def _write_uniform_tiled_tiff(
    path: Path,
    *,
    channels: int,
    value: int | tuple[int, int, int],
    dtype: np.dtype = np.dtype(np.uint8),
) -> None:
    shape = (8192, 8192, channels) if channels > 1 else (8192, 8192)
    tile_shape = (256, 256, channels) if channels > 1 else (256, 256)
    tile = np.empty(tile_shape, dtype=dtype)
    tile[...] = value
    tifffile.imwrite(
        path,
        data=(tile for _ in range(32 * 32)),
        shape=shape,
        dtype=dtype,
        tile=(256, 256),
        compression="deflate",
        photometric="rgb" if channels > 1 else "minisblack",
    )


def test_native_8192_complete_package_validates_and_derives_bounded_browser_lod(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    rgb_values = {
        "base_color": (96, 128, 160),
        "normal": (128, 128, 255),
        "specular_color": (64, 64, 64),
        "subsurface_color": (120, 80, 64),
        "subsurface_radius": (128, 96, 64),
    }
    for semantic, value in rgb_values.items():
        filename = f"{semantic}.tiff"
        _write_uniform_tiled_tiff(
            tmp_path / filename, channels=3, value=value
        )
        package["inventory"][semantic].update(
            {"path": filename, "source_resolution": [8192, 8192]}
        )
    for semantic, value in {
        "roughness": 127,
        "confidence": 255,
    }.items():
        filename = f"{semantic}.tiff"
        _write_uniform_tiled_tiff(
            tmp_path / filename, channels=1, value=value
        )
        package["inventory"][semantic].update(
            {"path": filename, "source_resolution": [8192, 8192]}
        )
    _write_uniform_tiled_tiff(
        tmp_path / "displacement.tiff",
        channels=1,
        value=32768,
        dtype=np.dtype(np.uint16),
    )
    package["inventory"]["displacement"].update(
        {"path": "displacement.tiff", "source_resolution": [8192, 8192]}
    )
    _write_uniform_tiled_tiff(
        tmp_path / "mask_skin.tiff", channels=1, value=255
    )
    package["inventory"]["masks"]["skin"].update(
        {"path": "mask_skin.tiff", "source_resolution": [8192, 8192]}
    )
    package["claims"].update(
        {
            "resolution_label": "8k",
            "native_resolution": True,
            "pore_resolved": False,
            "relightable": False,
        }
    )

    manifest = _validate(package)
    assert manifest["quality_evidence"]["minimum_map_width"] == 8192
    assert manifest["quality_evidence"]["all_maps_native"] is True
    assert manifest["claims"]["pore_resolved"] is False
    assert {
        entry["decode_strategy"]
        for material in manifest["maps"].values()
        for entry in material["files"].values()
    } == {"disk_memmap_tiff_segments"}

    sources = {
        semantic: tmp_path / manifest["maps"][semantic]["files"]["atlas"]["path"]
        for semantic in (
            "base_color",
            "normal",
            "roughness",
            "specular_color",
        )
    }
    derivatives = write_runtime_material_derivatives(
        sources, tmp_path / "runtime", normal_encoding="unorm"
    )
    assert set(derivatives) == {
        "base_color",
        "normal",
        "metallic_roughness",
        "specular_color",
    }
    for path in derivatives.values():
        with Image.open(path) as image:
            assert image.size == (4096, 4096)


def test_material_package_resource_limit_fails_before_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    monkeypatch.setattr("autoanim_gnm.materials.MAX_MATERIAL_FILES", 8)
    with pytest.raises(MaterialValidationError) as raised:
        _validate(package)
    assert raised.value.code == "RESOURCE_LIMIT_EXCEEDED"
