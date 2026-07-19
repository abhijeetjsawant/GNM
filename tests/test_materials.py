from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from autoanim_gnm.cli import build_parser
from autoanim_gnm.materials import MaterialValidationError, validate_material_package


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
    assert first["schema_version"] == "autoanim.material-package.v1"
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
    assert first["quality_evidence"]["pore_claim_gate_passed"] is False
    assert first["quality_evidence"]["relightable_claim_gate_passed"] is False
    assert first["quality_evidence"]["pore_frequency_validation_performed"] is False
    assert first["quality_evidence"]["unseen_light_validation_performed"] is False


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
