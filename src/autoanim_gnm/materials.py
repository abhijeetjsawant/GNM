"""Validation for production facial-material packages.

The validator deliberately does not synthesize missing maps or infer provenance.
It turns a caller-supplied, JSON-compatible inventory into a deterministic manifest
only after the material files and the accompanying capture, lineage, and rights
attestations pass fail-closed checks.

This module supports lossless PNG and TIFF atlases or UDIM sets.  OpenEXR is not
accepted because the project's current pinned image stack cannot portably inspect
its channel names and metadata; accepting it without that inspection would make the
validation claim stronger than the evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from tifffile import TiffFile, TiffFileError


SCHEMA_VERSION = "autoanim.material-package.v2"
ATTACHMENT_SCHEMA_VERSION = "autoanim.material-attachment.v1"

MAX_MATERIAL_FILES = 128
MAX_MATERIAL_MASKS = 32
MAX_UDIM_TILES_PER_MAP = 100
MAX_MATERIAL_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MATERIAL_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
MAX_MATERIAL_DIMENSION = 16_384
MAX_MATERIAL_PIXELS_PER_FILE = 268_435_456
MAX_MATERIAL_AGGREGATE_PIXELS = 1_200_000_000
MAX_MATERIAL_PATH_DEPTH = 8
MAX_MATERIAL_DECODED_BYTES_PER_FILE = 1024 * 1024 * 1024
MAX_MATERIAL_AGGREGATE_DECODED_BYTES = 6 * 1024 * 1024 * 1024
MAX_RESIDENT_IMAGE_BYTES = 128 * 1024 * 1024
MAX_RESIDENT_ENCODED_BYTES = 256 * 1024 * 1024
MAX_TIFF_SEGMENT_DECODED_BYTES = 64 * 1024 * 1024
MAX_TIFF_SEGMENTS = 16_384
MAX_MATERIAL_AGGREGATE_TIFF_SEGMENTS = 65_536
PIXEL_SCAN_CHUNK_BYTES = 8 * 1024 * 1024

_MATERIAL_SEMANTICS = (
    "base_color",
    "normal",
    "displacement",
    "specular_color",
    "roughness",
    "subsurface_color",
    "subsurface_radius",
    "confidence",
)
_INVENTORY_KEYS = frozenset((*_MATERIAL_SEMANTICS, "masks"))
_COLOR_SPACE = {
    "base_color": "srgb",
    "normal": "linear",
    "displacement": "linear",
    "specular_color": "linear",
    "roughness": "linear",
    "subsurface_color": "srgb",
    "subsurface_radius": "linear",
    "confidence": "linear",
    "mask": "linear",
}
_CHANNELS = {
    "base_color": frozenset((3, 4)),
    "normal": frozenset((3,)),
    "displacement": frozenset((1,)),
    "specular_color": frozenset((3,)),
    "roughness": frozenset((1,)),
    "subsurface_color": frozenset((3,)),
    "subsurface_radius": frozenset((3,)),
    "confidence": frozenset((1,)),
    "mask": frozenset((1,)),
}
_ALLOWED_DEPTHS = {
    "base_color": frozenset((8, 16, 32)),
    "normal": frozenset((8, 16, 32)),
    "displacement": frozenset((16, 32)),
    "specular_color": frozenset((8, 16, 32)),
    "roughness": frozenset((8, 16, 32)),
    "subsurface_color": frozenset((8, 16, 32)),
    "subsurface_radius": frozenset((8, 16, 32)),
    "confidence": frozenset((8, 16, 32)),
    "mask": frozenset((8, 16, 32)),
}
_NORMALIZED_FLOAT_SEMANTICS = frozenset(
    (
        "base_color",
        "specular_color",
        "roughness",
        "subsurface_color",
        "subsurface_radius",
        "confidence",
        "mask",
    )
)
_EXPECTED_FORMAT = {
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    # JPEG is listed so a PNG renamed to .jpg produces an explicit format error.
    # A genuine JPEG is subsequently rejected because this is a lossless package.
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
}
_LOSSLESS_FORMATS = frozenset(("PNG", "TIFF"))
_LOSSLESS_TIFF_COMPRESSIONS = frozenset(
    (
        "NONE",
        "ADOBE_DEFLATE",
        "DEFLATE",
        "PACKBITS",
    )
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HIGH_DETAIL_CAPTURE = frozenset(
    ("cross_polarized_multiview", "polarized_multilight", "photometric_stereo")
)
_RELIGHTABLE_CAPTURE = frozenset(
    ("cross_polarized_multiview", "polarized_multilight")
)
_CAPTURE_METHODS = frozenset(
    (
        "single_image",
        "multiview_passive",
        "cross_polarized_multiview",
        "polarized_multilight",
        "photometric_stereo",
        "synthetic",
    )
)


class MaterialValidationError(ValueError):
    """A material package is unsafe, inconsistent, or over-claimed."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True, slots=True)
class _DecodedFile:
    path: str
    sha256: str
    byte_count: int
    image_format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    dtype: str
    minimum: float
    maximum: float
    decoded_bytes: int
    decode_strategy: str
    segment_count: int
    maximum_decoded_segment_bytes: int

    def as_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "format": self.image_format,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "dtype": self.dtype,
            "value_range": [self.minimum, self.maximum],
            "decoded_bytes": self.decoded_bytes,
            "decode_strategy": self.decode_strategy,
            "segment_count": self.segment_count,
            "maximum_decoded_segment_bytes": self.maximum_decoded_segment_bytes,
        }


def validate_material_package(
    package_root: str | os.PathLike[str],
    *,
    package_id: str,
    inventory: Mapping[str, Any],
    capture: Mapping[str, Any],
    provenance: Mapping[str, Any],
    rights: Mapping[str, Any],
    claims: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a material inventory and return a deterministic JSON manifest.

    ``inventory`` must contain every core semantic plus a non-empty ``masks``
    object.  An entry has one of these two exact forms::

        {"layout": "atlas", "path": "base.png", "color_space": "srgb",
         "source_resolution": [4096, 4096], "resampling": "none"}

    The normal entry additionally requires ``"normal_encoding": "unorm"``
    for integer or normalized-float pixels, or ``"signed_float"`` for a
    floating-point ``[-1, 1]`` representation.  The representation is never
    inferred from pixel values.

        {"layout": "udim", "tiles": {"1001": "base.1001.png"},
         "color_space": "srgb", "source_resolution": [4096, 4096],
         "resampling": "none"}

    Paths are always relative to ``package_root``.  The function never modifies
    package bytes and never follows symlinks. TIFF pixels are decoded into an
    unlinked temporary scratch file so compressed or contiguous 8K sources do
    not become multi-gigabyte resident arrays or mutable-source memory maps.
    """

    validator = MaterialPackageValidator(package_root, now=now)
    return validator.validate(
        package_id=package_id,
        inventory=inventory,
        capture=capture,
        provenance=provenance,
        rights=rights,
        claims=claims,
    )


def validate_material_attachment(
    attachment: Mapping[str, Any],
    *,
    material_manifest: Mapping[str, Any],
    character_id: str,
    revision_id: str,
    revision_manifest_sha256: str,
    identity_sha256: str,
    triangle_corner_uv_f32le_sha256: str,
    character_subject: str,
) -> dict[str, Any]:
    """Validate the subject and exact-UV envelope for package attachment."""

    if not isinstance(attachment, Mapping):
        raise MaterialValidationError(
            "INVALID_ATTACHMENT", "Material attachment must be a JSON object."
        )
    _require_json_value(attachment, "attachment")
    required = frozenset(
        {
            "schema_version",
            "package_id",
            "material_manifest_payload_sha256",
            "authored_for",
            "subject_binding",
            "material_semantics",
        }
    )
    actual_attachment = set(attachment)
    if (
        actual_attachment != set(required)
        and actual_attachment != set(required) | {"attachment_payload_sha256"}
    ):
        missing = sorted(set(required) - actual_attachment)
        extra = sorted(actual_attachment - set(required) - {"attachment_payload_sha256"})
        raise MaterialValidationError(
            "INVALID_SCHEMA",
            f"attachment keys do not match schema; missing={missing}, extra={extra}.",
            field="attachment",
        )
    if attachment["schema_version"] != ATTACHMENT_SCHEMA_VERSION:
        raise MaterialValidationError(
            "INVALID_ATTACHMENT",
            f"Attachment schema must be {ATTACHMENT_SCHEMA_VERSION!r}.",
            field="attachment.schema_version",
        )
    if attachment["package_id"] != material_manifest.get("package_id"):
        raise MaterialValidationError(
            "ATTACHMENT_PACKAGE_MISMATCH",
            "Attachment package_id does not match the validated material package.",
            field="attachment.package_id",
        )
    if attachment["material_manifest_payload_sha256"] != material_manifest.get(
        "manifest_payload_sha256"
    ):
        raise MaterialValidationError(
            "ATTACHMENT_PACKAGE_MISMATCH",
            "Attachment package manifest digest does not match the validated package.",
            field="attachment.material_manifest_payload_sha256",
        )

    authored = attachment["authored_for"]
    if not isinstance(authored, Mapping):
        raise MaterialValidationError(
            "INVALID_ATTACHMENT", "authored_for must be an object.", field="attachment.authored_for"
        )
    authored_fields = frozenset(
        {
            "character_id",
            "revision_id",
            "revision_manifest_sha256",
            "identity_sha256",
            "gnm_version",
            "topology",
            "triangle_count",
            "uv_layout",
            "uv_origin",
            "triangle_corner_uv_f32le_sha256",
            "normal_space",
            "normal_y",
            "tangent_basis",
            "authored_for_attested",
        }
    )
    _expect_keys(authored, required=authored_fields, field="attachment.authored_for")
    expected_authored = {
        "character_id": character_id,
        "revision_id": revision_id,
        "revision_manifest_sha256": revision_manifest_sha256,
        "identity_sha256": identity_sha256,
        "gnm_version": "3.0",
        "topology": "GNM_Head_3_0",
        "triangle_count": 35_324,
        "uv_layout": "atlas",
        "uv_origin": "lower_left",
        "triangle_corner_uv_f32le_sha256": triangle_corner_uv_f32le_sha256,
        "normal_space": "tangent",
        "normal_y": "positive",
        "tangent_basis": "autoanim_gltf_tangent_v1",
        "authored_for_attested": True,
    }
    mismatched = [
        key for key, expected in expected_authored.items() if authored.get(key) != expected
    ]
    if mismatched:
        raise MaterialValidationError(
            "MATERIAL_BINDING_MISMATCH",
            f"Material was not authored for the selected character revision: {mismatched}.",
            field="attachment.authored_for",
        )

    subject = attachment["subject_binding"]
    if not isinstance(subject, Mapping):
        raise MaterialValidationError(
            "INVALID_ATTACHMENT",
            "subject_binding must be an object.",
            field="attachment.subject_binding",
        )
    subject_fields = frozenset(
        {
            "package_subject",
            "character_subject",
            "same_subject_attested",
            "attester",
            "evidence_ref",
            "evidence_sha256",
        }
    )
    _expect_keys(subject, required=subject_fields, field="attachment.subject_binding")
    if (
        subject.get("same_subject_attested") is not True
        or subject.get("package_subject") != character_subject
        or subject.get("character_subject") != character_subject
    ):
        raise MaterialValidationError(
            "MATERIAL_SUBJECT_MISMATCH",
            "Material and character must have an explicit same-subject attestation.",
            field="attachment.subject_binding",
        )
    for key in ("attester", "evidence_ref"):
        if not isinstance(subject.get(key), str) or not subject[key].strip():
            raise MaterialValidationError(
                "INVALID_ATTACHMENT",
                f"subject_binding.{key} is required.",
                field=f"attachment.subject_binding.{key}",
            )
    _require_sha(subject.get("evidence_sha256"), "attachment.subject_binding.evidence_sha256")

    semantics = attachment["material_semantics"]
    if not isinstance(semantics, Mapping):
        raise MaterialValidationError(
            "INVALID_ATTACHMENT",
            "material_semantics must be an object.",
            field="attachment.material_semantics",
        )
    semantic_fields = frozenset(
        {
            "specular_model",
            "normal_encoding",
            "displacement_unit",
            "displacement_midpoint",
            "displacement_scale_m",
            "subsurface_radius_unit",
            "base_color_alpha",
        }
    )
    _expect_keys(
        semantics, required=semantic_fields, field="attachment.material_semantics"
    )
    expected_values = {
        "specular_model": "gltf_dielectric_f0_multiplier_rgb_linear",
        "normal_encoding": material_manifest.get("maps", {})
        .get("normal", {})
        .get("normal_encoding"),
        "displacement_unit": "meters",
        "subsurface_radius_unit": "millimeters",
        "base_color_alpha": "unused_opaque",
    }
    bad_semantics = [
        key for key, expected in expected_values.items() if semantics.get(key) != expected
    ]
    if expected_values["normal_encoding"] not in {"unorm", "signed_float"}:
        bad_semantics.append("normal_encoding")
    midpoint = semantics.get("displacement_midpoint")
    scale = semantics.get("displacement_scale_m")
    if (
        isinstance(midpoint, bool)
        or not isinstance(midpoint, (int, float))
        or not math.isfinite(float(midpoint))
        or not 0.0 <= float(midpoint) <= 1.0
    ):
        bad_semantics.append("displacement_midpoint")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        bad_semantics.append("displacement_scale_m")
    if bad_semantics:
        raise MaterialValidationError(
            "UNSUPPORTED_MATERIAL_SEMANTICS",
            f"Material semantics are unsupported or incomplete: {sorted(set(bad_semantics))}.",
            field="attachment.material_semantics",
        )

    validated = copy.deepcopy(dict(attachment))
    supplied_payload_sha256 = validated.pop("attachment_payload_sha256", None)
    attachment_payload_sha256 = hashlib.sha256(
        _canonical_json(validated)
    ).hexdigest()
    if (
        supplied_payload_sha256 is not None
        and supplied_payload_sha256 != attachment_payload_sha256
    ):
        raise MaterialValidationError(
            "ATTACHMENT_INTEGRITY_FAILED",
            "Attachment payload digest does not match its content.",
            field="attachment.attachment_payload_sha256",
        )
    validated["attachment_payload_sha256"] = attachment_payload_sha256
    _canonical_json(validated)
    return validated


class MaterialPackageValidator:
    """Fail-closed validator for a single package root."""

    def __init__(
        self,
        package_root: str | os.PathLike[str],
        *,
        now: datetime | None = None,
    ) -> None:
        raw_root = Path(package_root)
        if raw_root.is_symlink():
            raise MaterialValidationError(
                "UNSAFE_ROOT", "Material package root may not be a symlink."
            )
        try:
            root = raw_root.resolve(strict=True)
        except OSError as exc:
            raise MaterialValidationError(
                "MISSING_ROOT", "Material package root does not exist."
            ) from exc
        if not root.is_dir():
            raise MaterialValidationError(
                "INVALID_ROOT", "Material package root must be a directory."
            )
        self.root = root
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise MaterialValidationError(
                "INVALID_TIME", "Validation time must be timezone-aware."
            )
        self.now = current.astimezone(timezone.utc)
        self._decoded_file_count = 0
        self._encoded_bytes = 0
        self._declared_pixels = 0
        self._declared_decoded_bytes = 0
        self._tiff_segments = 0

    def validate(
        self,
        *,
        package_id: str,
        inventory: Mapping[str, Any],
        capture: Mapping[str, Any],
        provenance: Mapping[str, Any],
        rights: Mapping[str, Any],
        claims: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Validator instances are reusable; every invocation receives a fresh
        # streaming resource budget and aborts before decoding the first file
        # that would exceed it.
        self._decoded_file_count = 0
        self._encoded_bytes = 0
        self._declared_pixels = 0
        self._declared_decoded_bytes = 0
        self._tiff_segments = 0
        if not isinstance(package_id, str) or not _PACKAGE_ID_RE.fullmatch(package_id):
            raise MaterialValidationError(
                "INVALID_PACKAGE_ID", "package_id contains unsupported characters.", field="package_id"
            )
        for field, value in (
            ("inventory", inventory),
            ("capture", capture),
            ("provenance", provenance),
            ("rights", rights),
            ("claims", claims),
        ):
            _require_json_value(value, field)
            if not isinstance(value, Mapping):
                raise MaterialValidationError(
                    "INVALID_SCHEMA", f"{field} must be a JSON object.", field=field
                )

        capture_doc = self._validate_capture(capture)
        provenance_doc = self._validate_provenance(provenance)
        if _parse_time(
            provenance_doc["created_at"], field="provenance.created_at"
        ) < _parse_time(capture_doc["captured_at"], field="capture.captured_at"):
            raise MaterialValidationError(
                "PROVENANCE_PRECEDES_CAPTURE",
                "Material provenance cannot predate its capture.",
                field="provenance.created_at",
            )
        rights_doc = self._validate_rights(rights)
        claims_doc = self._validate_claims_schema(claims)
        map_entries = self._validate_inventory(inventory)
        self._validate_lineage(map_entries, provenance_doc)
        self._validate_alignment(map_entries)
        quality = self._validate_claim_evidence(
            map_entries, capture_doc, provenance_doc, claims_doc
        )

        maps_manifest: dict[str, Any] = {}
        for name in sorted(map_entries):
            entry = map_entries[name]
            maps_manifest[name] = {
                "semantic": entry["semantic"],
                "layout": entry["layout"],
                "color_space": entry["color_space"],
                "source_resolution": list(entry["source_resolution"]),
                "resampling": entry["resampling"],
                "files": {
                    tile: decoded.as_manifest()
                    for tile, decoded in sorted(entry["files"].items())
                },
            }
            if entry["semantic"] == "normal":
                maps_manifest[name]["normal_encoding"] = entry["normal_encoding"]

        unique_files = {
            decoded.path: decoded
            for entry in map_entries.values()
            for decoded in entry["files"].values()
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "package_id": package_id,
            "maps": maps_manifest,
            "capture": capture_doc,
            "provenance": provenance_doc,
            "rights": rights_doc,
            "claims": claims_doc,
            "quality_evidence": quality,
            "totals": {
                "file_count": len(unique_files),
                "bytes": sum(item.byte_count for item in unique_files.values()),
                "decoded_bytes": sum(
                    item.decoded_bytes for item in unique_files.values()
                ),
            },
        }
        canonical = _canonical_json(manifest)
        manifest["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        # This is a final assertion, not a best-effort conversion.  NaN, Paths,
        # tuples, ndarray scalars, and any other non-JSON values are forbidden.
        _canonical_json(manifest)
        return manifest

    def _validate_inventory(
        self, inventory: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        _expect_keys(inventory, required=_INVENTORY_KEYS, field="inventory")
        masks = inventory["masks"]
        if not isinstance(masks, Mapping) or not masks:
            raise MaterialValidationError(
                "INVALID_MASKS", "inventory.masks must be a non-empty JSON object.", field="inventory.masks"
            )
        if len(masks) > MAX_MATERIAL_MASKS:
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                f"Material packages may contain at most {MAX_MATERIAL_MASKS} masks.",
                field="inventory.masks",
            )

        flattened: list[tuple[str, str, Any]] = [
            (semantic, semantic, inventory[semantic]) for semantic in _MATERIAL_SEMANTICS
        ]
        for mask_name, entry in masks.items():
            if not isinstance(mask_name, str) or not _NAME_RE.fullmatch(mask_name):
                raise MaterialValidationError(
                    "INVALID_MASK_NAME",
                    "Mask names must be lower snake_case identifiers.",
                    field="inventory.masks",
                )
            flattened.append((f"masks.{mask_name}", "mask", entry))

        output: dict[str, dict[str, Any]] = {}
        used_paths: set[str] = set()
        for name, semantic, entry in flattened:
            output[name] = self._validate_map_entry(
                name, semantic, entry, used_paths=used_paths
            )
        decoded = [
            item for entry in output.values() for item in entry["files"].values()
        ]
        if len(decoded) > MAX_MATERIAL_FILES:
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                f"Material packages may contain at most {MAX_MATERIAL_FILES} files.",
                field="inventory",
            )
        if sum(item.byte_count for item in decoded) > MAX_MATERIAL_TOTAL_BYTES:
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                "Material package encoded bytes exceed the import limit.",
                field="inventory",
            )
        if sum(item.width * item.height for item in decoded) > MAX_MATERIAL_AGGREGATE_PIXELS:
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                "Material package decoded pixels exceed the import limit.",
                field="inventory",
            )
        if (
            sum(item.decoded_bytes for item in decoded)
            > MAX_MATERIAL_AGGREGATE_DECODED_BYTES
        ):
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                "Material package decoded bytes exceed the import limit.",
                field="inventory",
            )
        return output

    def _validate_map_entry(
        self,
        name: str,
        semantic: str,
        entry: Any,
        *,
        used_paths: set[str],
    ) -> dict[str, Any]:
        field = f"inventory.{name}"
        if not isinstance(entry, Mapping):
            raise MaterialValidationError(
                "INVALID_MAP_ENTRY", f"{field} must be a JSON object.", field=field
            )
        common = frozenset(("layout", "color_space", "source_resolution", "resampling"))
        semantic_fields = common | ({"normal_encoding"} if semantic == "normal" else set())
        layout = entry.get("layout")
        if layout == "atlas":
            _expect_keys(entry, required=semantic_fields | {"path"}, field=field)
            raw_files = {"atlas": entry["path"]}
        elif layout == "udim":
            _expect_keys(entry, required=semantic_fields | {"tiles"}, field=field)
            raw_tiles = entry["tiles"]
            if not isinstance(raw_tiles, Mapping) or not raw_tiles:
                raise MaterialValidationError(
                    "INVALID_UDIM", f"{field}.tiles must be a non-empty object.", field=f"{field}.tiles"
                )
            if len(raw_tiles) > MAX_UDIM_TILES_PER_MAP:
                raise MaterialValidationError(
                    "RESOURCE_LIMIT_EXCEEDED",
                    f"A material semantic may contain at most {MAX_UDIM_TILES_PER_MAP} UDIM tiles.",
                    field=f"{field}.tiles",
                )
            raw_files = {}
            for tile, path in raw_tiles.items():
                if not isinstance(tile, str) or not tile.isdigit() or not 1001 <= int(tile) <= 1999:
                    raise MaterialValidationError(
                        "INVALID_UDIM", f"{field} contains invalid UDIM tile {tile!r}.", field=f"{field}.tiles"
                    )
                raw_files[tile] = path
        else:
            raise MaterialValidationError(
                "INVALID_LAYOUT", f"{field}.layout must be 'atlas' or 'udim'.", field=f"{field}.layout"
            )

        expected_space = _COLOR_SPACE[semantic]
        if entry.get("color_space") != expected_space:
            raise MaterialValidationError(
                "COLOR_SPACE_MISMATCH",
                f"{name} must declare {expected_space!r} color space.",
                field=f"{field}.color_space",
            )
        normal_encoding: str | None = None
        if semantic == "normal":
            normal_encoding = entry.get("normal_encoding")
            if normal_encoding not in {"unorm", "signed_float"}:
                raise MaterialValidationError(
                    "INVALID_NORMAL_ENCODING",
                    f"{field}.normal_encoding must be 'unorm' or 'signed_float'.",
                    field=f"{field}.normal_encoding",
                )
        source_resolution = entry.get("source_resolution")
        if (
            not isinstance(source_resolution, list)
            or len(source_resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in source_resolution)
        ):
            raise MaterialValidationError(
                "INVALID_SOURCE_RESOLUTION",
                f"{field}.source_resolution must be [positive width, positive height].",
                field=f"{field}.source_resolution",
            )
        resampling = entry.get("resampling")
        if not isinstance(resampling, str) or resampling not in {"none", "downsampled", "upsampled"}:
            raise MaterialValidationError(
                "INVALID_RESAMPLING",
                f"{field}.resampling must be none, downsampled, or upsampled.",
                field=f"{field}.resampling",
            )

        decoded_files: dict[str, _DecodedFile] = {}
        for tile, raw_path in raw_files.items():
            safe_path, relative = self._safe_file(raw_path, field=f"{field}.{tile}")
            if relative in used_paths:
                raise MaterialValidationError(
                    "DUPLICATE_MAP_FILE",
                    f"{relative} is assigned to more than one material semantic.",
                    field=field,
                )
            used_paths.add(relative)
            decoded_files[tile] = self._decode_image(
                safe_path,
                relative=relative,
                semantic=semantic,
                field=field,
                normal_encoding=normal_encoding,
            )

        source_size = tuple(source_resolution)
        for decoded in decoded_files.values():
            actual = (decoded.width, decoded.height)
            if resampling == "none" and actual != source_size:
                raise MaterialValidationError(
                    "NATIVE_RESOLUTION_MISMATCH",
                    f"{name} declares no resampling, but {actual} != source {source_size}.",
                    field=f"{field}.source_resolution",
                )
            if resampling == "downsampled" and not (
                actual[0] <= source_size[0]
                and actual[1] <= source_size[1]
                and actual != source_size
            ):
                raise MaterialValidationError(
                    "RESAMPLING_MISMATCH",
                    f"{name} is not smaller than its declared source resolution.",
                    field=f"{field}.resampling",
                )
            if resampling == "upsampled" and not (
                actual[0] >= source_size[0]
                and actual[1] >= source_size[1]
                and actual != source_size
            ):
                raise MaterialValidationError(
                    "RESAMPLING_MISMATCH",
                    f"{name} is not larger than its declared source resolution.",
                    field=f"{field}.resampling",
                )

        validated_entry = {
            "semantic": semantic,
            "layout": layout,
            "color_space": expected_space,
            "source_resolution": source_size,
            "resampling": resampling,
            "files": decoded_files,
        }
        if semantic == "normal":
            validated_entry["normal_encoding"] = normal_encoding
        return validated_entry

    def _safe_file(self, raw_path: Any, *, field: str) -> tuple[Path, str]:
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or "\x00" in raw_path:
            raise MaterialValidationError(
                "UNSAFE_PATH", "Material paths must be non-empty POSIX relative paths.", field=field
            )
        pure = PurePosixPath(raw_path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise MaterialValidationError(
                "UNSAFE_PATH", f"Unsafe material path {raw_path!r}.", field=field
            )
        if len(pure.parts) > MAX_MATERIAL_PATH_DEPTH:
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                f"Material paths may contain at most {MAX_MATERIAL_PATH_DEPTH} components.",
                field=field,
            )

        cursor = self.root
        try:
            for index, part in enumerate(pure.parts):
                cursor = cursor / part
                info = cursor.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise MaterialValidationError(
                        "SYMLINK_FORBIDDEN",
                        f"Material path {raw_path!r} contains a symlink.",
                        field=field,
                    )
                if index < len(pure.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                    raise MaterialValidationError(
                        "UNSAFE_PATH", f"Material parent is not a directory: {raw_path!r}.", field=field
                    )
            if not stat.S_ISREG(cursor.lstat().st_mode):
                raise MaterialValidationError(
                    "NOT_REGULAR_FILE", f"Material asset is not a regular file: {raw_path!r}.", field=field
                )
        except FileNotFoundError as exc:
            raise MaterialValidationError(
                "MISSING_MAP", f"Material asset does not exist: {raw_path!r}.", field=field
            ) from exc
        try:
            resolved = cursor.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise MaterialValidationError(
                "UNSAFE_PATH", f"Material asset escapes package root: {raw_path!r}.", field=field
            ) from exc
        return resolved, pure.as_posix()

    def _decode_image(
        self,
        path: Path,
        *,
        relative: str,
        semantic: str,
        field: str,
        normal_encoding: str | None,
    ) -> _DecodedFile:
        suffix = path.suffix.lower()
        expected_format = _EXPECTED_FORMAT.get(suffix)
        if expected_format is None:
            raise MaterialValidationError(
                "UNSUPPORTED_IMAGE_FORMAT",
                f"{relative} must use PNG or TIFF.",
                field=field,
            )
        file_descriptor, file_info = self._open_regular_file(relative, field=field)
        decoded: np.ndarray | None = None
        decode_scratch = None
        decode_strategy = ""
        segment_count = 1
        maximum_decoded_segment_bytes = 0
        try:
            if file_info.st_size > MAX_MATERIAL_FILE_BYTES:
                raise MaterialValidationError(
                    "RESOURCE_LIMIT_EXCEEDED",
                    f"{relative} exceeds the per-file encoded byte limit.",
                    field=field,
                )
            self._decoded_file_count += 1
            self._encoded_bytes += int(file_info.st_size)
            if self._decoded_file_count > MAX_MATERIAL_FILES:
                raise MaterialValidationError(
                    "RESOURCE_LIMIT_EXCEEDED",
                    f"Material packages may contain at most {MAX_MATERIAL_FILES} files.",
                    field="inventory",
                )
            if self._encoded_bytes > MAX_MATERIAL_TOTAL_BYTES:
                raise MaterialValidationError(
                    "RESOURCE_LIMIT_EXCEEDED",
                    "Material package encoded bytes exceed the import limit.",
                    field="inventory",
                )
            if expected_format == "TIFF":
                with os.fdopen(os.dup(file_descriptor), "rb") as image_file:
                    with TiffFile(image_file, name=relative) as tiff:
                        frame_count = len(tiff.pages)
                        if frame_count < 1:
                            raise TiffFileError("TIFF has no image pages")
                        page = tiff.pages[0]
                        actual_format = "TIFF"
                        declared_width = int(page.imagewidth)
                        declared_height = int(page.imagelength)
                        declared_channels = int(page.samplesperpixel or 1)
                        dtype = np.dtype(page.dtype)
                        bit_depth = _dtype_depth(dtype)
                        expected_axes = "YXS" if declared_channels > 1 else "YX"
                        if (
                            int(page.imagedepth or 1) != 1
                            or int(page.tiledepth or 1) != 1
                            or str(page.axes) != expected_axes
                            or tuple(page.shape)
                            != (
                                (declared_height, declared_width, declared_channels)
                                if declared_channels > 1
                                else (declared_height, declared_width)
                            )
                            or bool(page.subifds)
                        ):
                            raise MaterialValidationError(
                                "TIFF_DIMENSIONAL_LAYOUT_UNSUPPORTED",
                                f"{relative} must be a single 2D YX/YXS image without depth or SubIFDs.",
                                field=field,
                            )
                        declared_decoded_bytes = self._record_decoded_budget(
                            relative=relative,
                            width=declared_width,
                            height=declared_height,
                            channels=declared_channels,
                            itemsize=dtype.itemsize,
                            field=field,
                        )
                        if frame_count != 1:
                            raise MaterialValidationError(
                                "MULTIFRAME_IMAGE_FORBIDDEN",
                                f"{relative} contains {frame_count} frames; material maps must contain exactly one image.",
                                field=field,
                            )
                        compression = getattr(page.compression, "name", str(page.compression))
                        if compression not in _LOSSLESS_TIFF_COMPRESSIONS:
                            raise MaterialValidationError(
                                "LOSSY_IMAGE_FORBIDDEN",
                                f"{relative} uses unsupported or lossy TIFF compression {compression!r}.",
                                field=field,
                            )
                        orientation_tag = page.tags.get("Orientation")
                        if orientation_tag is not None and int(orientation_tag.value) != 1:
                            raise MaterialValidationError(
                                "TIFF_ORIENTATION_UNSUPPORTED",
                                f"{relative} must store pixels in top-left row order; orientation transforms are not inferred.",
                                field=field,
                            )
                        planar_value = int(page.planarconfig or 1)
                        if declared_channels > 1 and planar_value != 1:
                            raise MaterialValidationError(
                                "TIFF_PLANAR_LAYOUT_UNSUPPORTED",
                                f"{relative} must use contiguous interleaved samples.",
                                field=field,
                            )
                        photometric = getattr(page.photometric, "name", str(page.photometric))
                        expected_photometric = (
                            "RGB" if declared_channels in {3, 4} else "MINISBLACK"
                        )
                        if photometric != expected_photometric:
                            raise MaterialValidationError(
                                "TIFF_PHOTOMETRIC_MISMATCH",
                                f"{relative} uses {photometric!r}; {semantic} requires {expected_photometric!r} sample semantics.",
                                field=field,
                            )
                        if page.is_tiled:
                            segment_width = min(
                                declared_width, int(page.tilewidth or declared_width)
                            )
                            segment_height = min(
                                declared_height, int(page.tilelength or declared_height)
                            )
                        else:
                            segment_width = declared_width
                            segment_height = min(
                                declared_height,
                                int(page.rowsperstrip or declared_height),
                            )
                        segment_count = len(page.dataoffsets)
                        self._tiff_segments += segment_count
                        maximum_decoded_segment_bytes = int(
                            segment_width
                            * segment_height
                            * declared_channels
                            * dtype.itemsize
                        )
                        if (
                            segment_count < 1
                            or segment_count > MAX_TIFF_SEGMENTS
                            or self._tiff_segments
                            > MAX_MATERIAL_AGGREGATE_TIFF_SEGMENTS
                            or maximum_decoded_segment_bytes
                            > MAX_TIFF_SEGMENT_DECODED_BYTES
                        ):
                            raise MaterialValidationError(
                                "TIFF_SEGMENT_LIMIT_EXCEEDED",
                                f"{relative} has an unsafe TIFF strip/tile layout; use bounded tiles or strips.",
                                field=field,
                            )
                        decode_scratch = tempfile.TemporaryFile(
                            prefix="autoanim-material-decode-"
                        )
                        filesystem = os.fstatvfs(decode_scratch.fileno())
                        available_scratch_bytes = int(
                            filesystem.f_bavail * filesystem.f_frsize
                        )
                        if available_scratch_bytes < (
                            declared_decoded_bytes + 128 * 1024 * 1024
                        ):
                            raise MaterialValidationError(
                                "TEMP_STORAGE_LIMIT_EXCEEDED",
                                f"Insufficient scratch space to validate {relative} safely.",
                                field=field,
                            )
                        decoded = page.asarray(
                            out=decode_scratch,
                            maxworkers=1,
                            buffersize=PIXEL_SCAN_CHUNK_BYTES,
                        )
                        decode_strategy = "disk_memmap_tiff_segments"
                        (
                            width,
                            height,
                            channels,
                            bit_depth,
                            minimum,
                            maximum,
                        ) = _scan_decoded_pixels(
                            decoded,
                            semantic=semantic,
                            normal_encoding=normal_encoding,
                            relative=relative,
                            field=field,
                        )
                        if (
                            (width, height, channels)
                            != (
                                declared_width,
                                declared_height,
                                declared_channels,
                            )
                            or np.dtype(decoded.dtype) != dtype
                        ):
                            raise MaterialValidationError(
                                "IMAGE_HEADER_MISMATCH",
                                f"{relative} decoded pixels do not match TIFF metadata.",
                                field=field,
                            )
            else:
                if file_info.st_size > MAX_RESIDENT_ENCODED_BYTES:
                    raise MaterialValidationError(
                        "SCALABLE_SOURCE_REQUIRED",
                        f"{relative} encoded bytes exceed the resident parser budget; use a bounded-strip/tiled TIFF.",
                        field=field,
                    )
                is_png = os.pread(file_descriptor, 8, 0) == b"\x89PNG\r\n\x1a\n"
                decoded_bytes = 0
                declared_channels = 0
                png_bit_depth = 0
                png_width = png_height = 0
                if is_png:
                    (
                        png_width,
                        png_height,
                        declared_channels,
                        png_bit_depth,
                    ) = _inspect_png_header(
                        file_descriptor, relative=relative, field=field
                    )
                    decoded_bytes = self._record_decoded_budget(
                        relative=relative,
                        width=png_width,
                        height=png_height,
                        channels=declared_channels,
                        itemsize=png_bit_depth // 8,
                        field=field,
                    )
                    maximum_decoded_segment_bytes = decoded_bytes
                    if decoded_bytes > MAX_RESIDENT_IMAGE_BYTES:
                        raise MaterialValidationError(
                            "SCALABLE_SOURCE_REQUIRED",
                            f"{relative} cannot be decoded within the resident-memory budget; store native high-resolution maps as bounded-strip/tiled TIFF.",
                            field=field,
                        )
                with os.fdopen(os.dup(file_descriptor), "rb") as image_file:
                    with Image.open(image_file) as image:
                        actual_format = image.format
                        frame_count = int(getattr(image, "n_frames", 1))
                        declared_width, declared_height = image.size
                        image.verify()
                if actual_format != expected_format:
                    raise MaterialValidationError(
                        "IMAGE_FORMAT_MISMATCH",
                        f"{relative} extension declares {expected_format}, bytes are {actual_format}.",
                        field=field,
                    )
                if frame_count != 1:
                    raise MaterialValidationError(
                        "MULTIFRAME_IMAGE_FORBIDDEN",
                        f"{relative} contains {frame_count} frames; material maps must contain exactly one image.",
                        field=field,
                    )
                if actual_format not in _LOSSLESS_FORMATS:
                    raise MaterialValidationError(
                        "LOSSY_IMAGE_FORBIDDEN",
                        f"{relative} is lossy; production material maps must be PNG or TIFF.",
                        field=field,
                    )
                if (png_width, png_height) != (
                    declared_width,
                    declared_height,
                ):
                    raise MaterialValidationError(
                        "IMAGE_HEADER_MISMATCH",
                        f"{relative} PNG metadata is internally inconsistent.",
                        field=field,
                    )
                payload = _read_file_descriptor(
                    file_descriptor,
                    int(file_info.st_size),
                    maximum=MAX_RESIDENT_ENCODED_BYTES,
                )
                if len(payload) != file_info.st_size:
                    raise MaterialValidationError(
                        "IMAGE_DECODE_FAILED",
                        f"Could not read exactly {file_info.st_size} encoded bytes from {relative}.",
                        field=field,
                    )
                encoded = np.frombuffer(payload, dtype=np.uint8)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                file_sha256 = hashlib.sha256(payload).hexdigest()
                del encoded, payload
                decode_strategy = "bounded_resident_png"
                (
                    width,
                    height,
                    channels,
                    bit_depth,
                    minimum,
                    maximum,
                ) = _scan_decoded_pixels(
                    decoded,
                    semantic=semantic,
                    normal_encoding=normal_encoding,
                    relative=relative,
                    field=field,
                )
                if (
                    (width, height, channels, bit_depth)
                    != (
                        declared_width,
                        declared_height,
                        declared_channels,
                        png_bit_depth,
                    )
                ):
                    raise MaterialValidationError(
                        "IMAGE_HEADER_MISMATCH",
                        f"{relative} decoded pixels do not match PNG metadata.",
                        field=field,
                    )
            if expected_format == "TIFF":
                file_sha256 = _sha256_file_descriptor(
                    file_descriptor, int(file_info.st_size), relative=relative, field=field
                )
            final_info = os.fstat(file_descriptor)
            if _stat_identity(file_info) != _stat_identity(final_info):
                raise MaterialValidationError(
                    "SOURCE_CHANGED_DURING_VALIDATION",
                    f"{relative} changed while it was being validated.",
                    field=field,
                )
        except MaterialValidationError:
            raise
        except (
            TiffFileError,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
            MemoryError,
        ) as exc:
            raise MaterialValidationError(
                "IMAGE_DECODE_FAILED", f"Image decode failed for {relative}.", field=field
            ) from exc
        finally:
            if isinstance(decoded, np.memmap):
                mapping = getattr(decoded, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            if decode_scratch is not None:
                decode_scratch.close()
            os.close(file_descriptor)
        if actual_format != expected_format:
            raise MaterialValidationError(
                "IMAGE_FORMAT_MISMATCH",
                f"{relative} extension declares {expected_format}, bytes are {actual_format}.",
                field=field,
            )
        decoded_bytes = int(width * height * channels * (bit_depth // 8))
        return _DecodedFile(
            path=relative,
            sha256=file_sha256,
            byte_count=file_info.st_size,
            image_format=actual_format,
            width=int(width),
            height=int(height),
            channels=int(channels),
            bit_depth=bit_depth,
            dtype=str(decoded.dtype),
            minimum=minimum,
            maximum=maximum,
            decoded_bytes=decoded_bytes,
            decode_strategy=decode_strategy,
            segment_count=segment_count,
            maximum_decoded_segment_bytes=maximum_decoded_segment_bytes,
        )

    def _record_decoded_budget(
        self,
        *,
        relative: str,
        width: int,
        height: int,
        channels: int,
        itemsize: int,
        field: str,
    ) -> int:
        pixels = int(width * height)
        decoded_bytes = int(pixels * channels * itemsize)
        if (
            width < 1
            or height < 1
            or channels < 1
            or width > MAX_MATERIAL_DIMENSION
            or height > MAX_MATERIAL_DIMENSION
            or pixels > MAX_MATERIAL_PIXELS_PER_FILE
            or decoded_bytes > MAX_MATERIAL_DECODED_BYTES_PER_FILE
        ):
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                f"{relative} exceeds the decoded dimension, pixel, or byte limit.",
                field=field,
            )
        self._declared_pixels += pixels
        self._declared_decoded_bytes += decoded_bytes
        if (
            self._declared_pixels > MAX_MATERIAL_AGGREGATE_PIXELS
            or self._declared_decoded_bytes
            > MAX_MATERIAL_AGGREGATE_DECODED_BYTES
        ):
            raise MaterialValidationError(
                "RESOURCE_LIMIT_EXCEEDED",
                "Material package decoded pixels or bytes exceed the import limit.",
                field="inventory",
            )
        return decoded_bytes

    def _open_regular_file(self, relative: str, *, field: str) -> tuple[int, os.stat_result]:
        """Open a package file without following any path component symlink."""

        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory_flag is None:
            raise MaterialValidationError(
                "SAFE_OPEN_UNAVAILABLE",
                "This platform cannot guarantee symlink-safe package validation.",
                field=field,
            )
        flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        descriptors: list[int] = []
        asset: int | None = None
        try:
            current = os.open(self.root, os.O_RDONLY | directory_flag | nofollow)
            descriptors.append(current)
            parts = PurePosixPath(relative).parts
            for component in parts[:-1]:
                current = os.open(
                    component,
                    os.O_RDONLY | directory_flag | nofollow,
                    dir_fd=current,
                )
                descriptors.append(current)
            asset = os.open(parts[-1], flags, dir_fd=current)
            info = os.fstat(asset)
            if not stat.S_ISREG(info.st_mode):
                os.close(asset)
                asset = None
                raise MaterialValidationError(
                    "NOT_REGULAR_FILE",
                    f"Material asset is not a regular file: {relative!r}.",
                    field=field,
                )
            return asset, info
        except MaterialValidationError:
            if asset is not None:
                os.close(asset)
            raise
        except OSError as exc:
            if asset is not None:
                os.close(asset)
            raise MaterialValidationError(
                "SAFE_OPEN_FAILED",
                f"Could not safely open material asset {relative!r}.",
                field=field,
            ) from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    @staticmethod
    def _validate_alignment(map_entries: Mapping[str, Mapping[str, Any]]) -> None:
        layouts = {entry["layout"] for entry in map_entries.values()}
        if len(layouts) != 1:
            raise MaterialValidationError(
                "LAYOUT_MISMATCH", "All material maps must use one common atlas or UDIM layout."
            )
        layout = next(iter(layouts))
        reference_tiles: set[str] | None = None
        reference_sizes: dict[str, tuple[int, int]] | None = None
        for name, entry in map_entries.items():
            files = entry["files"]
            tiles = set(files)
            sizes = {tile: (item.width, item.height) for tile, item in files.items()}
            if reference_tiles is None:
                reference_tiles = tiles
                reference_sizes = sizes
                continue
            if tiles != reference_tiles:
                raise MaterialValidationError(
                    "UDIM_TILE_MISMATCH",
                    f"{name} does not cover the same UDIM tiles as the other maps.",
                )
            if sizes != reference_sizes:
                label = "atlas" if layout == "atlas" else "UDIM tile"
                raise MaterialValidationError(
                    "DIMENSION_MISMATCH",
                    f"{name} {label} dimensions are not aligned with the other maps.",
                )

    def _validate_capture(self, capture: Mapping[str, Any]) -> dict[str, Any]:
        required = frozenset(
            (
                "capture_id",
                "captured_at",
                "method",
                "devices",
                "polarized",
                "spatial_resolution_mm_per_pixel",
                "calibration_sha256",
            )
        )
        _expect_keys(capture, required=required, field="capture")
        if not isinstance(capture["capture_id"], str) or not capture["capture_id"].strip():
            raise MaterialValidationError("INVALID_CAPTURE", "capture_id is required.", field="capture.capture_id")
        captured_at = _parse_time(capture["captured_at"], field="capture.captured_at")
        if captured_at > self.now:
            raise MaterialValidationError(
                "FUTURE_CAPTURE", "Capture time may not be in the future.", field="capture.captured_at"
            )
        if not isinstance(capture["method"], str) or capture["method"] not in _CAPTURE_METHODS:
            raise MaterialValidationError("INVALID_CAPTURE", "Unsupported capture method.", field="capture.method")
        devices = capture["devices"]
        if not isinstance(devices, list) or not devices or any(not isinstance(item, str) or not item.strip() for item in devices):
            raise MaterialValidationError("INVALID_CAPTURE", "devices must be a non-empty string array.", field="capture.devices")
        if type(capture["polarized"]) is not bool:
            raise MaterialValidationError("INVALID_CAPTURE", "polarized must be a boolean.", field="capture.polarized")
        spatial = capture["spatial_resolution_mm_per_pixel"]
        if isinstance(spatial, bool) or not isinstance(spatial, (int, float)) or not math.isfinite(float(spatial)) or spatial <= 0:
            raise MaterialValidationError(
                "INVALID_CAPTURE", "spatial_resolution_mm_per_pixel must be finite and positive.", field="capture.spatial_resolution_mm_per_pixel"
            )
        _require_sha(capture["calibration_sha256"], "capture.calibration_sha256")
        return copy.deepcopy(dict(capture))

    def _validate_provenance(self, provenance: Mapping[str, Any]) -> dict[str, Any]:
        required = frozenset(
            (
                "producer",
                "pipeline",
                "pipeline_version",
                "created_at",
                "source_sha256s",
                "processing_log_sha256",
                "map_lineage",
            )
        )
        _expect_keys(provenance, required=required, field="provenance")
        for key in ("producer", "pipeline", "pipeline_version"):
            if not isinstance(provenance[key], str) or not provenance[key].strip():
                raise MaterialValidationError("INVALID_PROVENANCE", f"provenance.{key} is required.", field=f"provenance.{key}")
        created_at = _parse_time(provenance["created_at"], field="provenance.created_at")
        if created_at > self.now:
            raise MaterialValidationError(
                "FUTURE_PROVENANCE", "Provenance time may not be in the future.", field="provenance.created_at"
            )
        sources = provenance["source_sha256s"]
        if not isinstance(sources, list) or not sources:
            raise MaterialValidationError("INVALID_PROVENANCE", "At least one source hash is required.", field="provenance.source_sha256s")
        for index, value in enumerate(sources):
            _require_sha(value, f"provenance.source_sha256s.{index}")
        if len(set(sources)) != len(sources):
            raise MaterialValidationError("INVALID_PROVENANCE", "Source hashes must be unique.", field="provenance.source_sha256s")
        _require_sha(provenance["processing_log_sha256"], "provenance.processing_log_sha256")
        if not isinstance(provenance["map_lineage"], Mapping):
            raise MaterialValidationError("INVALID_PROVENANCE", "map_lineage must be an object.", field="provenance.map_lineage")
        return copy.deepcopy(dict(provenance))

    def _validate_rights(self, rights: Mapping[str, Any]) -> dict[str, Any]:
        required = frozenset(
            (
                "status",
                "commercial_allowed",
                "subject_consent_attested",
                "scope",
                "evidence_ref",
                "evidence_sha256",
                "expires_at",
            )
        )
        _expect_keys(rights, required=required, field="rights")
        # Unknown, pending, absent, or merely non-commercial rights all fail.
        if rights["status"] != "cleared":
            raise MaterialValidationError("RIGHTS_NOT_CLEARED", "Rights status must be 'cleared'.", field="rights.status")
        if rights["commercial_allowed"] is not True:
            raise MaterialValidationError("COMMERCIAL_RIGHTS_REQUIRED", "commercial_allowed must be true.", field="rights.commercial_allowed")
        if rights["subject_consent_attested"] is not True:
            raise MaterialValidationError("CONSENT_REQUIRED", "Subject consent must be attested.", field="rights.subject_consent_attested")
        if rights["scope"] != "commercial":
            raise MaterialValidationError("COMMERCIAL_SCOPE_REQUIRED", "Rights scope must be commercial.", field="rights.scope")
        if not isinstance(rights["evidence_ref"], str) or not rights["evidence_ref"].strip():
            raise MaterialValidationError("RIGHTS_EVIDENCE_REQUIRED", "A rights evidence reference is required.", field="rights.evidence_ref")
        _require_sha(rights["evidence_sha256"], "rights.evidence_sha256")
        expires = rights["expires_at"]
        if expires is not None:
            expiry = _parse_time(expires, field="rights.expires_at")
            if expiry <= self.now:
                raise MaterialValidationError("RIGHTS_EXPIRED", "Rights evidence has expired.", field="rights.expires_at")
        return copy.deepcopy(dict(rights))

    @staticmethod
    def _validate_claims_schema(claims: Mapping[str, Any]) -> dict[str, Any]:
        required = frozenset(("resolution_label", "native_resolution", "pore_resolved", "relightable"))
        _expect_keys(claims, required=required, field="claims")
        if not isinstance(claims["resolution_label"], str) or claims["resolution_label"] not in {"unclaimed", "2k", "4k", "8k"}:
            raise MaterialValidationError("INVALID_CLAIMS", "resolution_label must be unclaimed, 2k, 4k, or 8k.", field="claims.resolution_label")
        for key in ("native_resolution", "pore_resolved", "relightable"):
            if type(claims[key]) is not bool:
                raise MaterialValidationError("INVALID_CLAIMS", f"claims.{key} must be a boolean.", field=f"claims.{key}")
        if claims["pore_resolved"] and claims["resolution_label"] != "8k":
            raise MaterialValidationError(
                "FALSE_PORE_CLAIM",
                "A pore-resolved claim requires native 8K maps; 4K is not sufficient.",
                field="claims.pore_resolved",
            )
        return copy.deepcopy(dict(claims))

    @staticmethod
    def _validate_lineage(
        map_entries: Mapping[str, Mapping[str, Any]], provenance: Mapping[str, Any]
    ) -> None:
        lineage = provenance["map_lineage"]
        expected = set(map_entries)
        if set(lineage) != expected:
            missing = sorted(expected - set(lineage))
            extra = sorted(set(lineage) - expected)
            raise MaterialValidationError(
                "LINEAGE_MISMATCH",
                f"map_lineage must exactly cover inventory maps; missing={missing}, extra={extra}.",
                field="provenance.map_lineage",
            )
        sources = set(provenance["source_sha256s"])
        for name, value in lineage.items():
            if not isinstance(value, Mapping):
                raise MaterialValidationError("INVALID_LINEAGE", f"Lineage for {name} must be an object.", field=f"provenance.map_lineage.{name}")
            _expect_keys(value, required=frozenset(("operation", "source_sha256s")), field=f"provenance.map_lineage.{name}")
            if not isinstance(value["operation"], str) or value["operation"] not in {"captured", "derived", "inferred"}:
                raise MaterialValidationError("INVALID_LINEAGE", f"Invalid lineage operation for {name}.", field=f"provenance.map_lineage.{name}.operation")
            refs = value["source_sha256s"]
            if not isinstance(refs, list) or not refs:
                raise MaterialValidationError("INVALID_LINEAGE", f"Lineage for {name} needs source hashes.", field=f"provenance.map_lineage.{name}.source_sha256s")
            for ref in refs:
                _require_sha(ref, f"provenance.map_lineage.{name}.source_sha256s")
            if not set(refs) <= sources:
                raise MaterialValidationError("UNKNOWN_LINEAGE_SOURCE", f"Lineage for {name} references an undeclared source.", field=f"provenance.map_lineage.{name}.source_sha256s")

    @staticmethod
    def _validate_claim_evidence(
        map_entries: Mapping[str, Mapping[str, Any]],
        capture: Mapping[str, Any],
        provenance: Mapping[str, Any],
        claims: Mapping[str, Any],
    ) -> dict[str, Any]:
        all_decoded = [item for entry in map_entries.values() for item in entry["files"].values()]
        contains_upsampling = any(entry["resampling"] == "upsampled" for entry in map_entries.values())
        all_native = all(entry["resampling"] == "none" for entry in map_entries.values())
        minimum_width = min(item.width for item in all_decoded)
        minimum_height = min(item.height for item in all_decoded)
        minimum_depth = min(item.bit_depth for item in all_decoded)
        physical_minimum_depth = min(
            item.bit_depth
            for name, entry in map_entries.items()
            if not name.startswith("masks.")
            for item in entry["files"].values()
        )
        if claims["native_resolution"] and not all_native:
            raise MaterialValidationError(
                "FALSE_NATIVE_CLAIM", "native_resolution cannot be true when any map was resampled.", field="claims.native_resolution"
            )
        resolution_floor = {"2k": 2048, "4k": 4096, "8k": 8192}
        label = claims["resolution_label"]
        lineage = provenance["map_lineage"]
        core_inferred = [
            name
            for name in _MATERIAL_SEMANTICS
            if lineage[name]["operation"] == "inferred"
        ]
        if claims["pore_resolved"]:
            if contains_upsampling or not all_native:
                raise MaterialValidationError("FALSE_PORE_CLAIM", "Pore-resolved maps must be native, not resampled.", field="claims.pore_resolved")
            if capture["method"] not in _HIGH_DETAIL_CAPTURE:
                raise MaterialValidationError("FALSE_PORE_CLAIM", "Capture method does not support a pore-resolved claim.", field="claims.pore_resolved")
            if float(capture["spatial_resolution_mm_per_pixel"]) > 0.08:
                raise MaterialValidationError("FALSE_PORE_CLAIM", "Capture sampling is too coarse to support a pore-resolved claim.", field="claims.pore_resolved")
            if any(lineage[name]["operation"] == "inferred" for name in ("normal", "displacement")):
                raise MaterialValidationError("FALSE_PORE_CLAIM", "Inferred normal/displacement cannot substantiate captured pores.", field="claims.pore_resolved")
            if label != "8k":
                raise MaterialValidationError("FALSE_PORE_CLAIM", "Pore-resolved claims require native measured 8K map resolution.", field="claims.pore_resolved")
            if min(
                item.bit_depth
                for name in ("normal", "displacement")
                for item in map_entries[name]["files"].values()
            ) < 16:
                raise MaterialValidationError("FALSE_PORE_CLAIM", "Pore normal/displacement maps require at least 16-bit precision.", field="claims.pore_resolved")
        if claims["relightable"]:
            if contains_upsampling:
                raise MaterialValidationError("FALSE_RELIGHTABLE_CLAIM", "Relightable maps may not be upsampled.", field="claims.relightable")
            if capture["method"] not in _RELIGHTABLE_CAPTURE or capture["polarized"] is not True:
                raise MaterialValidationError("FALSE_RELIGHTABLE_CLAIM", "Relightable material separation requires a declared polarized multiview/multilight capture.", field="claims.relightable")
            if core_inferred:
                raise MaterialValidationError("FALSE_RELIGHTABLE_CLAIM", f"Inferred maps cannot substantiate relightability: {core_inferred}.", field="claims.relightable")
            if physical_minimum_depth < 16:
                raise MaterialValidationError("FALSE_RELIGHTABLE_CLAIM", "Relightable physical maps require at least 16-bit precision.", field="claims.relightable")
        if label != "unclaimed":
            floor = resolution_floor[label]
            if contains_upsampling:
                raise MaterialValidationError(
                    "UPSAMPLED_RESOLUTION_CLAIM",
                    f"{label.upper()} may not be claimed from upsampled maps.",
                    field="claims.resolution_label",
                )
            if minimum_width < floor or minimum_height < floor:
                raise MaterialValidationError(
                    "FALSE_RESOLUTION_CLAIM",
                    f"{label.upper()} requires every aligned map/tile to be at least {floor}x{floor}.",
                    field="claims.resolution_label",
                )

        return {
            "layout": next(iter({entry["layout"] for entry in map_entries.values()})),
            "minimum_map_width": minimum_width,
            "minimum_map_height": minimum_height,
            "minimum_bit_depth": minimum_depth,
            "physical_minimum_bit_depth": physical_minimum_depth,
            "all_maps_native": all_native,
            "contains_upsampling": contains_upsampling,
            "core_inferred_maps": core_inferred,
            # These are structural/attestation gates, not perceptual validation.
            # Independent frequency and unseen-light artifacts are not part of v1.
            "pore_claim_gate_passed": claims["pore_resolved"],
            "relightable_claim_gate_passed": claims["relightable"],
            "pore_frequency_validation_performed": False,
            "unseen_light_validation_performed": False,
        }


def _expect_keys(
    value: Mapping[str, Any], *, required: frozenset[str], field: str
) -> None:
    actual = set(value)
    if actual != set(required):
        missing = sorted(set(required) - actual)
        extra = sorted(actual - set(required))
        raise MaterialValidationError(
            "INVALID_SCHEMA",
            f"{field} keys do not match schema; missing={missing}, extra={extra}.",
            field=field,
        )


def _require_sha(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MaterialValidationError(
            "INVALID_SHA256", f"{field} must be a lowercase SHA-256 hex digest.", field=field
        )


def _parse_time(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise MaterialValidationError("INVALID_TIMESTAMP", f"{field} must be an RFC 3339 timestamp.", field=field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaterialValidationError("INVALID_TIMESTAMP", f"{field} must be an RFC 3339 timestamp.", field=field) from exc
    if parsed.tzinfo is None:
        raise MaterialValidationError("INVALID_TIMESTAMP", f"{field} must include a timezone.", field=field)
    return parsed.astimezone(timezone.utc)


def _inspect_png_header(
    file_descriptor: int, *, relative: str, field: str
) -> tuple[int, int, int, int]:
    """Read the fixed PNG IHDR without allocating for declared image size."""

    header = os.pread(file_descriptor, 33, 0)
    if (
        len(header) != 33
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[8:12] != b"\x00\x00\x00\r"
        or header[12:16] != b"IHDR"
    ):
        raise MaterialValidationError(
            "IMAGE_DECODE_FAILED",
            f"{relative} has no valid PNG IHDR.",
            field=field,
        )
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth = int(header[24])
    color_type = int(header[25])
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    channels = channels_by_type.get(color_type)
    valid_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        4: {8, 16},
        6: {8, 16},
    }
    if channels is None or bit_depth not in valid_depths.get(color_type, set()):
        raise MaterialValidationError(
            "UNSUPPORTED_PIXEL_TYPE",
            f"{relative} uses unsupported PNG color type/depth {color_type}/{bit_depth}.",
            field=field,
        )
    # Packed sub-byte images are never valid production material precision.
    if bit_depth not in {8, 16}:
        raise MaterialValidationError(
            "BIT_DEPTH_MISMATCH",
            f"{relative} is {bit_depth}-bit; production material maps require byte-aligned samples.",
            field=field,
        )
    return width, height, channels, bit_depth


def _sha256_file_descriptor(
    file_descriptor: int,
    byte_count: int,
    *,
    relative: str,
    field: str,
) -> str:
    """Hash an already-safe-open file in bounded chunks without path races."""

    digest = hashlib.sha256()
    offset = 0
    chunk_size = 8 * 1024 * 1024
    while offset < byte_count:
        chunk = os.pread(file_descriptor, min(chunk_size, byte_count - offset), offset)
        if not chunk:
            raise MaterialValidationError(
                "SOURCE_CHANGED_DURING_VALIDATION",
                f"{relative} became shorter while it was being hashed.",
                field=field,
            )
        digest.update(chunk)
        offset += len(chunk)
    if offset != byte_count:
        raise MaterialValidationError(
            "SOURCE_CHANGED_DURING_VALIDATION",
            f"{relative} byte count changed while it was being hashed.",
            field=field,
        )
    return digest.hexdigest()


def _read_file_descriptor(
    file_descriptor: int, byte_count: int, *, maximum: int
) -> bytearray:
    if byte_count > maximum:
        return bytearray()
    payload = bytearray(byte_count)
    view = memoryview(payload)
    offset = 0
    chunk_size = 8 * 1024 * 1024
    while offset < byte_count:
        stop = min(byte_count, offset + chunk_size)
        read = os.preadv(file_descriptor, [view[offset:stop]], offset)
        if read < 1:
            break
        offset += int(read)
    if offset != byte_count:
        return bytearray()
    return payload


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _scan_decoded_pixels(
    decoded: np.ndarray,
    *,
    semantic: str,
    normal_encoding: str | None,
    relative: str,
    field: str,
) -> tuple[int, int, int, int, float, float]:
    """Validate pixels in bounded row chunks; source may be a disk memmap."""

    if decoded is None or decoded.size == 0:
        raise MaterialValidationError(
            "IMAGE_DECODE_FAILED", f"Image decoder returned no pixels for {relative}.", field=field
        )
    if decoded.ndim == 2:
        height, width = decoded.shape
        channels = 1
    elif decoded.ndim == 3:
        height, width, channels = decoded.shape
    else:
        raise MaterialValidationError(
            "INVALID_IMAGE_SHAPE",
            f"Unsupported image shape for {relative}: {decoded.shape}.",
            field=field,
        )
    if width <= 0 or height <= 0 or channels not in _CHANNELS[semantic]:
        expected = sorted(_CHANNELS[semantic])
        raise MaterialValidationError(
            "CHANNEL_MISMATCH",
            f"{relative} has {channels} channels; {semantic} requires {expected}.",
            field=field,
        )
    bit_depth = _dtype_depth(decoded.dtype)
    if bit_depth not in _ALLOWED_DEPTHS[semantic]:
        raise MaterialValidationError(
            "BIT_DEPTH_MISMATCH",
            f"{relative} is {bit_depth}-bit; {semantic} accepts {sorted(_ALLOWED_DEPTHS[semantic])}.",
            field=field,
        )
    if np.issubdtype(decoded.dtype, np.signedinteger):
        raise MaterialValidationError(
            "PIXEL_DTYPE_MISMATCH",
            f"{relative} uses signed integer samples; material integer maps must be unsigned.",
            field=field,
        )
    if (
        semantic == "normal"
        and normal_encoding == "signed_float"
        and not np.issubdtype(decoded.dtype, np.floating)
    ):
        raise MaterialValidationError(
            "NORMAL_ENCODING_DTYPE_MISMATCH",
            f"{relative} declares signed_float but has integer pixels.",
            field=field,
        )

    row_bytes = max(1, int(width * channels * decoded.dtype.itemsize))
    rows_per_chunk = max(1, PIXEL_SCAN_CHUNK_BYTES // row_bytes)
    minimum = math.inf
    maximum = -math.inf
    plausible_normal_count = 0
    inspected_normal_count = 0
    for row_start in range(0, int(height), rows_per_chunk):
        chunk = np.asarray(decoded[row_start : row_start + rows_per_chunk])
        if np.issubdtype(chunk.dtype, np.floating) and not bool(
            np.isfinite(chunk).all()
        ):
            raise MaterialValidationError(
                "NONFINITE_PIXELS",
                f"{relative} contains NaN or infinite pixels.",
                field=field,
            )
        minimum = min(minimum, float(np.min(chunk)))
        maximum = max(maximum, float(np.max(chunk)))
        if semantic == "base_color" and channels == 4:
            opaque = (
                1.0
                if np.issubdtype(chunk.dtype, np.floating)
                else float(np.iinfo(chunk.dtype).max)
            )
            if not bool(np.all(chunk[..., 3] == opaque)):
                raise MaterialValidationError(
                    "BASE_COLOR_ALPHA_MISMATCH",
                    f"{relative} declares unused opaque alpha but contains non-opaque samples.",
                    field=field,
                )
        if semantic == "normal":
            vectors = chunk[..., :3].astype(np.float32)
            if np.issubdtype(chunk.dtype, np.unsignedinteger):
                vectors /= float(np.iinfo(chunk.dtype).max)
                vectors = vectors * 2.0 - 1.0
            elif normal_encoding == "unorm":
                vectors = vectors * 2.0 - 1.0
            lengths = np.linalg.norm(vectors, axis=2)
            plausible_normal_count += int(
                np.count_nonzero(
                    np.logical_and(lengths >= 0.5, lengths <= 1.5)
                )
            )
            inspected_normal_count += int(lengths.size)

    if np.issubdtype(decoded.dtype, np.floating):
        if semantic in _NORMALIZED_FLOAT_SEMANTICS and (
            minimum < 0.0 or maximum > 1.0
        ):
            raise MaterialValidationError(
                "PIXEL_RANGE_MISMATCH",
                f"{relative} must be normalized to [0, 1].",
                field=field,
            )
        if semantic == "normal":
            expected_range = (
                (0.0, 1.0) if normal_encoding == "unorm" else (-1.0, 1.0)
            )
            if minimum < expected_range[0] or maximum > expected_range[1]:
                raise MaterialValidationError(
                    "PIXEL_RANGE_MISMATCH",
                    f"Float normal map {relative} does not match declared "
                    f"{normal_encoding!r} range {list(expected_range)}.",
                    field=field,
                )

    if semantic == "normal":
        if (
            inspected_normal_count < 1
            or plausible_normal_count / inspected_normal_count < 0.99
        ):
            raise MaterialValidationError(
                "NORMAL_VECTOR_MISMATCH",
                f"{relative} does not contain plausible encoded tangent-space normal vectors.",
                field=field,
            )

    return (
        int(width),
        int(height),
        int(channels),
        int(bit_depth),
        float(minimum),
        float(maximum),
    )


def _dtype_depth(dtype: np.dtype[Any]) -> int:
    if dtype == np.dtype(np.uint8):
        return 8
    if dtype in (np.dtype(np.uint16), np.dtype(np.int16)):
        return 16
    if dtype in (np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.uint32)):
        return 32
    raise MaterialValidationError(
        "UNSUPPORTED_PIXEL_TYPE", f"Unsupported decoded pixel type {dtype}."
    )


def _require_json_value(value: Any, field: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise MaterialValidationError("NONFINITE_JSON", f"{field} contains NaN or infinity.", field=field)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, f"{field}.{index}")
        return
    if isinstance(value, Mapping) and type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise MaterialValidationError("NON_JSON_VALUE", f"{field} contains a non-string key.", field=field)
            _require_json_value(item, f"{field}.{key}")
        return
    raise MaterialValidationError(
        "NON_JSON_VALUE", f"{field} contains non-JSON value {type(value).__name__}.", field=field
    )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterialValidationError(
            "NON_JSON_MANIFEST", "Material manifest is not strict JSON."
        ) from exc


__all__ = [
    "ATTACHMENT_SCHEMA_VERSION",
    "MaterialPackageValidator",
    "MaterialValidationError",
    "SCHEMA_VERSION",
    "validate_material_attachment",
    "validate_material_package",
]
