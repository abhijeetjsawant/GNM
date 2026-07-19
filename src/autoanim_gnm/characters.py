"""Versioned, consent-audited character assets promoted from reconstruction jobs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .animated_gltf import export_animated_gnm_glb
from .artifacts import JobStore, new_ulid, sha256, utc_now
from .errors import AutoAnimError
from .gnm_adapter import GNMAdapter
from .materials import (
    MaterialValidationError,
    validate_material_attachment,
    validate_material_package,
)
from .runtime_material import (
    MAX_RUNTIME_TEXTURE_DIMENSION,
    PRESERVED_SEMANTICS,
    RUNTIME_DERIVATIVE_KEYS,
    RUNTIME_PROJECTION_PROFILE,
    write_runtime_material_derivatives,
)
from .serialization import write_json, write_npz


_ULID_ALPHABET = frozenset("0123456789abcdefghjkmnpqrstvwxyz")
_PROMOTABLE_KINDS = frozenset({"image_fit", "multiview_reconstruction"})
_CONSENT_SCOPES = frozenset({"personal", "production", "commercial", "research"})
_SCOPE_GRANTS = {
    "personal": frozenset({"personal"}),
    "research": frozenset({"research"}),
    "production": frozenset({"personal", "production"}),
    "commercial": frozenset({"personal", "production", "commercial"}),
}
_GNM_TRIANGLE_COUNT = 35_324
_MATERIAL_SPEC_FIELDS = frozenset(
    {"package_id", "inventory", "capture", "provenance", "rights", "claims"}
)


def _valid_ulid(value: str) -> bool:
    return len(value) == 26 and all(character in _ULID_ALPHABET for character in value)


def _clean_name(value: str) -> str:
    name = " ".join(str(value).split())
    if not name or len(name) > 120 or any(ord(character) < 32 for character in name):
        raise AutoAnimError(
            "INPUT_INVALID", "Character name must contain 1-120 printable characters"
        )
    return name


def _clean_required(value: str, label: str, *, maximum: int = 160) -> str:
    cleaned = " ".join(str(value).split())
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 for character in cleaned)
    ):
        raise AutoAnimError(
            "INPUT_INVALID",
            f"{label} must contain 1-{maximum} printable characters",
        )
    return cleaned


def _parse_expiry(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AutoAnimError(
            "INPUT_INVALID", "Consent expiry must be an ISO-8601 timestamp with timezone"
        ) from exc
    if parsed.tzinfo is None:
        raise AutoAnimError(
            "INPUT_INVALID", "Consent expiry must include an explicit timezone"
        )
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= datetime.now(timezone.utc):
        raise AutoAnimError("CONSENT_EXPIRED", "Consent expiry must be in the future")
    return parsed.isoformat().replace("+00:00", "Z")


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copy2(source, destination)
    return {
        "name": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def _uv_array_sha256(value: np.ndarray) -> str:
    """Canonical digest for the GNM triangle-corner UV value array."""

    array = np.ascontiguousarray(value, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"autoanim.gnm-v3.triangle-uvs.f32le\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_validated_material_asset(
    package_root: Path,
    relative: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    """Copy one validator-approved asset without following a replaced symlink."""

    root = package_root.resolve(strict=True)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise AutoAnimError("MATERIAL_INVALID", "Validated material path became unsafe")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise AutoAnimError(
            "DEPENDENCY_MISSING",
            "This platform cannot safely import material assets without following symlinks",
        )
    descriptors: list[int] = []
    source_descriptor: int | None = None
    try:
        current = os.open(root, os.O_RDONLY | directory | nofollow)
        descriptors.append(current)
        for component in pure.parts[:-1]:
            current = os.open(
                component,
                os.O_RDONLY | directory | nofollow,
                dir_fd=current,
            )
            descriptors.append(current)
        source_descriptor = os.open(
            pure.parts[-1],
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
        info = os.fstat(source_descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Material asset changed after validation; import was aborted",
            )
        digest = hashlib.sha256()
        written = 0
        with os.fdopen(os.dup(source_descriptor), "rb") as source, destination.open(
            "xb"
        ) as target:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if written != expected_bytes or digest.hexdigest() != expected_sha256:
            destination.unlink(missing_ok=True)
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Material asset changed after validation; import was aborted",
            )
        return {
            "name": destination.name,
            "bytes": written,
            "sha256": digest.hexdigest(),
        }
    except AutoAnimError:
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise AutoAnimError(
            "INTEGRITY_FAILED", "Material asset could not be copied safely"
        ) from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class CharacterRevision:
    character_id: str
    revision_id: str
    name: str
    identity: np.ndarray
    texture_path: Path | None
    triangle_uvs: np.ndarray | None
    preview_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    identity_sha256: str
    texture_sha256: str | None
    texture_uvs_sha256: str | None
    texture_uvs_array_sha256: str | None
    material_paths: Mapping[str, Path]
    material_asset_paths: Mapping[str, Path]
    material_sha256s: Mapping[str, str]
    runtime_material_paths: Mapping[str, Path]
    runtime_material_sha256s: Mapping[str, str]
    material_manifest_sha256: str


class CharacterStore:
    """Immutable character revisions with a small mutable current-revision pointer."""

    def __init__(self, root: str | Path, jobs: JobStore):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs

    def _character_dir(self, character_id: str) -> Path:
        if not _valid_ulid(character_id):
            raise FileNotFoundError(character_id)
        path = (self.root / character_id).resolve()
        if path.parent != self.root:
            raise FileNotFoundError(character_id)
        return path

    def _revision_dir(self, character_id: str, revision_id: str) -> Path:
        if not _valid_ulid(revision_id):
            raise FileNotFoundError(revision_id)
        character_dir = self._character_dir(character_id)
        path = (character_dir / "revisions" / revision_id).resolve()
        if path.parent != (character_dir / "revisions").resolve():
            raise FileNotFoundError(revision_id)
        return path

    @contextmanager
    def _character_lock(self, character_id: str):
        character_dir = self._character_dir(character_id)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            character_dir / ".lock",
            os.O_RDWR | os.O_CREAT | nofollow | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Character mutation lock is not a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def promote(
        self,
        job_id: str,
        *,
        name: str,
        consent_attested: bool,
        consent_subject: str,
        consent_attester: str,
        consent_scope: str,
        consent_evidence_ref: str,
        consent_evidence_sha256: str,
        consent_expires_at: str | None = None,
        consent_note: str | None = None,
    ) -> dict[str, Any]:
        """Create a reusable character from a successful identity-fit job.

        Promotion copies only allowlisted derived artifacts. Original biometric
        input images remain in the job ledger and are never copied into the
        character library.
        """

        if consent_attested is not True:
            raise AutoAnimError(
                "CONSENT_REQUIRED",
                "Character promotion requires an explicit performer/rights-holder consent attestation.",
            )
        clean_name = _clean_name(name)
        subject = _clean_required(consent_subject, "Consent subject")
        attester = _clean_required(consent_attester, "Consent attester")
        scope = consent_scope.strip().lower()
        if scope not in _CONSENT_SCOPES:
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Consent scope must be one of: {', '.join(sorted(_CONSENT_SCOPES))}",
            )
        evidence_ref = _clean_required(
            consent_evidence_ref, "Consent evidence reference", maximum=300
        )
        evidence_sha256 = consent_evidence_sha256.strip().lower()
        if len(evidence_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in evidence_sha256
        ):
            raise AutoAnimError(
                "INPUT_INVALID", "Consent evidence must include its SHA-256 content digest"
            )
        expires_at = _parse_expiry(consent_expires_at)
        note = " ".join((consent_note or "").split())
        if len(note) > 500:
            raise AutoAnimError("INPUT_INVALID", "Consent note must be at most 500 characters")
        try:
            job = self.jobs.require_sealed(job_id)
        except FileNotFoundError as exc:
            raise AutoAnimError("JOB_NOT_FOUND", "Source reconstruction job was not found") from exc
        kind = job.get("kind")
        if job.get("status") != "succeeded" or kind not in _PROMOTABLE_KINDS:
            raise AutoAnimError(
                "INPUT_INVALID",
                "Only successful image-fit or multiview-reconstruction jobs can become characters.",
            )
        artifacts = job.get("artifacts", {})
        parameters = artifacts.get("parameters") or artifacts.get("fit")
        preview = artifacts.get("textured_glb") or artifacts.get("glb")
        if not isinstance(parameters, dict) or not isinstance(preview, dict):
            raise AutoAnimError(
                "INTERNAL_ERROR", "Source job is missing its allowlisted identity or preview artifact"
            )
        try:
            parameters_path = self.jobs.artifact(job_id, str(parameters["name"]))
            preview_path = self.jobs.artifact(job_id, str(preview["name"]))
        except (FileNotFoundError, KeyError, TypeError) as exc:
            raise AutoAnimError("INTERNAL_ERROR", "Source job artifacts failed integrity lookup") from exc
        try:
            with np.load(parameters_path, allow_pickle=False) as values:
                identity_key = "identity" if "identity" in values.files else "fitted_identity"
                identity = np.asarray(values[identity_key], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise AutoAnimError("INTERNAL_ERROR", "Source identity artifact is unreadable") from exc
        if identity.shape != (253,) or not np.isfinite(identity).all():
            raise AutoAnimError(
                "INTERNAL_ERROR", "Source identity must be one finite GNM v3 (253,) vector"
            )

        texture_path: Path | None = None
        texture_triangle_uvs: np.ndarray | None = None
        texture = artifacts.get("texture")
        if isinstance(texture, dict) and isinstance(texture.get("name"), str):
            try:
                texture_path = self.jobs.artifact(job_id, texture["name"])
            except FileNotFoundError as exc:
                raise AutoAnimError("INTERNAL_ERROR", "Source texture failed integrity lookup") from exc
            texture_maps = artifacts.get("texture_maps")
            if not isinstance(texture_maps, dict) or not isinstance(
                texture_maps.get("name"), str
            ):
                raise AutoAnimError(
                    "INTERNAL_ERROR",
                    "Textured source job is missing its sealed texture UV layout",
                )
            try:
                texture_maps_path = self.jobs.artifact(job_id, texture_maps["name"])
                with np.load(texture_maps_path, allow_pickle=False) as values:
                    texture_triangle_uvs = np.asarray(
                        values["triangle_uvs"], dtype=np.float32
                    )
            except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                raise AutoAnimError(
                    "INTERNAL_ERROR", "Source texture UV layout is unreadable"
                ) from exc
            if (
                texture_triangle_uvs.shape != (_GNM_TRIANGLE_COUNT, 3, 2)
                or not np.isfinite(texture_triangle_uvs).all()
                or np.min(texture_triangle_uvs) < 0.0
                or np.max(texture_triangle_uvs) > 1.0
            ):
                raise AutoAnimError(
                    "INTERNAL_ERROR", "Source texture UV layout is invalid for GNM Head 3.0"
                )
        if texture_triangle_uvs is None:
            texture_triangle_uvs = np.asarray(
                GNMAdapter().model.triangle_uvs, dtype=np.float32
            )
        if (
            texture_triangle_uvs.shape != (_GNM_TRIANGLE_COUNT, 3, 2)
            or not np.isfinite(texture_triangle_uvs).all()
            or np.min(texture_triangle_uvs) < 0.0
            or np.max(texture_triangle_uvs) > 1.0
        ):
            raise AutoAnimError(
                "INTERNAL_ERROR", "Character UV layout is invalid for GNM Head 3.0"
            )

        character_id = new_ulid()
        revision_id = new_ulid()
        destination = self._character_dir(character_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{character_id}.", suffix=".tmp", dir=destination.parent)
        )
        revision_dir = temporary / "revisions" / revision_id
        revision_dir.mkdir(parents=True)
        created = utc_now()
        try:
            write_npz(revision_dir / "identity.npz", identity=identity)
            copied: dict[str, dict[str, Any]] = {
                "identity": {
                    "name": "identity.npz",
                    "bytes": (revision_dir / "identity.npz").stat().st_size,
                    "sha256": sha256(revision_dir / "identity.npz"),
                },
                "preview": _copy_verified(preview_path, revision_dir / "preview.glb"),
            }
            write_npz(
                revision_dir / "texture-uvs.npz",
                triangle_uvs=texture_triangle_uvs,
            )
            copied["texture_uvs"] = {
                "name": "texture-uvs.npz",
                "bytes": (revision_dir / "texture-uvs.npz").stat().st_size,
                "sha256": sha256(revision_dir / "texture-uvs.npz"),
                "array_sha256": _uv_array_sha256(texture_triangle_uvs),
            }
            texture_size: list[int] | None = None
            if texture_path is not None:
                with Image.open(texture_path) as image:
                    image.verify()
                with Image.open(texture_path) as image:
                    texture_size = [int(image.width), int(image.height)]
                suffix = texture_path.suffix.lower()
                texture_name = f"base-color{suffix}"
                copied["base_color"] = _copy_verified(
                    texture_path, revision_dir / texture_name
                )
            material = {
                "schema_version": "1.0",
                "model": "OpenPBR-compatible capture inventory",
                "maps": {
                    "base_color": copied.get("base_color"),
                    "normal": None,
                    "displacement": None,
                    "specular_color": None,
                    "roughness": None,
                    "subsurface_color": None,
                    "subsurface_radius": None,
                },
                "resolution": texture_size,
                "capture_class": (
                    "unpolarized_multiview_rgb" if texture_path is not None else "no_measured_texture"
                ),
                "uv_layout_asset": "texture_uvs",
                "uv_convention": "gnm_triangle_corner_lower_left_v3",
                "pore_detail_validated": False,
                "relightable": False,
                "production_validated": False,
            }
            write_json(revision_dir / "material.json", material)
            copied["material"] = {
                "name": "material.json",
                "bytes": (revision_dir / "material.json").stat().st_size,
                "sha256": sha256(revision_dir / "material.json"),
            }
            source_texture = job.get("texture", {}) if isinstance(job.get("texture"), dict) else {}
            revision_manifest = {
                "schema_version": "1.0",
                "character_id": character_id,
                "revision_id": revision_id,
                "created_at": created,
                "gnm": {
                    "version": "3.0",
                    "identity_dim": 253,
                    "identity_sha256": copied["identity"]["sha256"],
                    "texture_uvs_array_sha256": copied["texture_uvs"][
                        "array_sha256"
                    ],
                },
                "source": {
                    "job_id": job_id,
                    "job_kind": kind,
                    "job_input_sha256": job.get("input", {}).get("sha256"),
                    "fit_production_validated": bool(
                        job.get("fit", {}).get("production_validated", False)
                    ),
                },
                "consent": {
                    "attested": True,
                    "attested_at": created,
                    "subject": subject,
                    "attester": attester,
                    "scope": scope,
                    "evidence_ref": evidence_ref,
                    "evidence_sha256": evidence_sha256,
                    "expires_at": expires_at,
                    "note": note or None,
                },
                "appearance": {
                    "material_artifact": "material",
                    "base_color_observed_fraction": source_texture.get("observed_fraction"),
                    "capture_class": material["capture_class"],
                    "production_validated": False,
                },
                "oral": {
                    "character_calibration_required": True,
                    "tongue_visibility_validated": False,
                    "teeth_collision_validated": False,
                    "production_validated": False,
                },
                "body": {
                    "status": "not_attached",
                    "rig_standard": None,
                    "head_attachment_validated": False,
                },
                "assets": copied,
                "production_validated": False,
            }
            write_json(revision_dir / "manifest.json", revision_manifest)
            revision_manifest_sha256 = sha256(revision_dir / "manifest.json")
            top_manifest = {
                "schema_version": "1.0",
                "character_id": character_id,
                "name": clean_name,
                "created_at": created,
                "updated_at": created,
                "current_revision_id": revision_id,
                "current_revision_manifest_sha256": revision_manifest_sha256,
                "revision_count": 1,
                "revisions": {
                    revision_id: {
                        "manifest_sha256": revision_manifest_sha256,
                        "created_at": created,
                        "identity_sha256": copied["identity"]["sha256"],
                        "texture_uvs_array_sha256": copied["texture_uvs"][
                            "array_sha256"
                        ],
                    }
                },
                "source_job_id": job_id,
                "consent_attested": True,
                "consent_status": "active",
                "consent_scope": scope,
                "consent_expires_at": expires_at,
                "appearance_status": (
                    "rgb_atlas_unvalidated" if texture_path is not None else "geometry_only"
                ),
                "current_identity_sha256": copied["identity"]["sha256"],
                "current_texture_uvs_array_sha256": copied["texture_uvs"][
                    "array_sha256"
                ],
                "current_material_rights_expires_at": None,
                "body_status": "not_attached",
                "production_validated": False,
            }
            write_json(temporary / "manifest.json", self.jobs.signer.sign(top_manifest))
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.read(character_id)

    def prepare_material_attachment(
        self,
        character_id: str,
        package_root: str | Path,
        *,
        specification: Mapping[str, Any],
        base_revision_id: str,
        usage_scope: str,
        attester: str,
        evidence_ref: str,
        evidence_sha256: str,
        package_subject: str,
        same_subject_attested: bool,
        authored_for_attested: bool,
        displacement_midpoint: float,
        displacement_scale_m: float,
    ) -> dict[str, Any]:
        """Validate a package and emit its exact revision/subject envelope."""

        clean_attester = _clean_required(attester, "Material binding attester")
        clean_evidence_ref = _clean_required(
            evidence_ref, "Material binding evidence reference", maximum=300
        )
        clean_package_subject = _clean_required(
            package_subject, "Material package subject"
        )
        if same_subject_attested is not True or authored_for_attested is not True:
            raise AutoAnimError(
                "MATERIAL_INVALID",
                "Material attachment requires explicit same-subject and exact-revision authorship attestations",
            )
        if not isinstance(specification, Mapping) or set(specification) != set(
            _MATERIAL_SPEC_FIELDS
        ):
            raise AutoAnimError(
                "MATERIAL_INVALID",
                "Material specification fields are missing or unknown",
            )
        base = self.resolve(
            character_id, base_revision_id, usage_scope=usage_scope
        )
        character = self.read(character_id)
        if character.get("current_revision_id") != base_revision_id:
            raise AutoAnimError(
                "REVISION_CONFLICT",
                "Material attachment template must target the exact current revision",
            )
        if base.texture_uvs_array_sha256 is None:
            raise AutoAnimError(
                "MATERIAL_BINDING_REQUIRED",
                "Selected character revision has no canonical UV value digest",
            )
        try:
            material_manifest = validate_material_package(
                package_root,
                package_id=specification["package_id"],
                inventory=specification["inventory"],
                capture=specification["capture"],
                provenance=specification["provenance"],
                rights=specification["rights"],
                claims=specification["claims"],
            )
            if material_manifest["quality_evidence"]["layout"] != "atlas":
                raise MaterialValidationError(
                    "MATERIAL_LAYOUT_UNSUPPORTED",
                    "Runtime attachment requires one aligned atlas.",
                    field="inventory",
                )
            consent = base.manifest.get("consent", {})
            subject = consent.get("subject") if isinstance(consent, dict) else None
            if not isinstance(subject, str) or not subject:
                raise MaterialValidationError(
                    "MATERIAL_SUBJECT_MISMATCH",
                    "Character revision has no consent subject.",
                )
            attachment = {
                "schema_version": "autoanim.material-attachment.v1",
                "package_id": material_manifest["package_id"],
                "material_manifest_payload_sha256": material_manifest[
                    "manifest_payload_sha256"
                ],
                "authored_for": {
                    "character_id": character_id,
                    "revision_id": base.revision_id,
                    "revision_manifest_sha256": base.manifest_sha256,
                    "identity_sha256": base.identity_sha256,
                    "gnm_version": "3.0",
                    "topology": "GNM_Head_3_0",
                    "triangle_count": _GNM_TRIANGLE_COUNT,
                    "uv_layout": "atlas",
                    "uv_origin": "lower_left",
                    "triangle_corner_uv_f32le_sha256": base.texture_uvs_array_sha256,
                    "normal_space": "tangent",
                    "normal_y": "positive",
                    "tangent_basis": "autoanim_gltf_tangent_v1",
                    "authored_for_attested": True,
                },
                "subject_binding": {
                    "package_subject": clean_package_subject,
                    "character_subject": subject,
                    "same_subject_attested": same_subject_attested,
                    "attester": clean_attester,
                    "evidence_ref": clean_evidence_ref,
                    "evidence_sha256": evidence_sha256,
                },
                "material_semantics": {
                    "specular_model": "gltf_dielectric_f0_multiplier_rgb_linear",
                    "normal_encoding": material_manifest["maps"]["normal"][
                        "normal_encoding"
                    ],
                    "displacement_unit": "meters",
                    "displacement_midpoint": displacement_midpoint,
                    "displacement_scale_m": displacement_scale_m,
                    "subsurface_radius_unit": "millimeters",
                    "base_color_alpha": "unused_opaque",
                },
            }
            validated_attachment = validate_material_attachment(
                attachment,
                material_manifest=material_manifest,
                character_id=character_id,
                revision_id=base.revision_id,
                revision_manifest_sha256=base.manifest_sha256,
                identity_sha256=base.identity_sha256,
                triangle_corner_uv_f32le_sha256=base.texture_uvs_array_sha256,
                character_subject=subject,
            )
        except MaterialValidationError as exc:
            raise AutoAnimError(
                "MATERIAL_INVALID",
                str(exc),
                {"material_code": exc.code, "field": exc.field},
            ) from exc
        return {
            "material_manifest": material_manifest,
            "attachment": validated_attachment,
        }

    def attach_material(
        self,
        character_id: str,
        package_root: str | Path,
        *,
        specification: Mapping[str, Any],
        attachment: Mapping[str, Any],
        base_revision_id: str,
        usage_scope: str = "production",
    ) -> dict[str, Any]:
        with self._character_lock(character_id):
            return self._attach_material_locked(
                character_id,
                package_root,
                specification=specification,
                attachment=attachment,
                base_revision_id=base_revision_id,
                usage_scope=usage_scope,
            )

    def _attach_material_locked(
        self,
        character_id: str,
        package_root: str | Path,
        *,
        specification: Mapping[str, Any],
        attachment: Mapping[str, Any],
        base_revision_id: str,
        usage_scope: str = "production",
    ) -> dict[str, Any]:
        """Attach a validated atlas as a new immutable character revision.

        Map-to-UV semantic correctness cannot be inferred from file dimensions,
        so the caller must explicitly attest the exact sealed character UV
        binding.  The recorded identity and UV hashes make that claim auditable.
        """

        if not isinstance(specification, Mapping) or set(specification) != set(
            _MATERIAL_SPEC_FIELDS
        ):
            raise AutoAnimError(
                "MATERIAL_INVALID",
                "Material specification fields are missing or unknown",
            )
        base = self.resolve(
            character_id, base_revision_id, usage_scope=usage_scope
        )
        character = self.read(character_id)
        if character.get("current_revision_id") != base_revision_id:
            raise AutoAnimError(
                "REVISION_CONFLICT",
                "Material import must target the exact current character revision",
                {
                    "expected_current_revision_id": base_revision_id,
                    "actual_current_revision_id": character.get("current_revision_id"),
                },
            )
        if base.triangle_uvs is None or base.texture_uvs_sha256 is None:
            raise AutoAnimError(
                "MATERIAL_BINDING_REQUIRED",
                "Selected character revision has no sealed GNM UV layout",
            )
        package = Path(package_root)
        try:
            validated = validate_material_package(
                package,
                package_id=specification["package_id"],
                inventory=specification["inventory"],
                capture=specification["capture"],
                provenance=specification["provenance"],
                rights=specification["rights"],
                claims=specification["claims"],
            )
        except MaterialValidationError as exc:
            raise AutoAnimError(
                "MATERIAL_INVALID",
                str(exc),
                {"material_code": exc.code, "field": exc.field},
            ) from exc
        consent = base.manifest.get("consent", {})
        character_subject = (
            consent.get("subject") if isinstance(consent, dict) else None
        )
        if not isinstance(character_subject, str) or not character_subject:
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Character revision has no consent subject binding"
            )
        try:
            validated_attachment = validate_material_attachment(
                attachment,
                material_manifest=validated,
                character_id=character_id,
                revision_id=base.revision_id,
                revision_manifest_sha256=base.manifest_sha256,
                identity_sha256=base.identity_sha256,
                triangle_corner_uv_f32le_sha256=(
                    base.texture_uvs_array_sha256 or ""
                ),
                character_subject=character_subject,
            )
        except MaterialValidationError as exc:
            raise AutoAnimError(
                "MATERIAL_BINDING_MISMATCH",
                str(exc),
                {"material_code": exc.code, "field": exc.field},
            ) from exc
        quality = validated.get("quality_evidence", {})
        if quality.get("layout") != "atlas":
            raise AutoAnimError(
                "MATERIAL_LAYOUT_UNSUPPORTED",
                "The current GNM real-time runtime accepts one aligned atlas; validated UDIM packages require an explicit atlas bake before attachment",
            )

        character_dir = self._character_dir(character_id)
        revisions_dir = character_dir / "revisions"
        revisions_dir.mkdir(parents=True, exist_ok=True)
        revision_id = new_ulid()
        final_revision = self._revision_dir(character_id, revision_id)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{revision_id}.", suffix=".tmp", dir=revisions_dir
            )
        )
        created = utc_now()
        published = False
        top_published = False
        try:
            source_revision_dir = self._revision_dir(
                character_id, base.revision_id
            )
            source_assets = base.manifest.get("assets", {})
            identity_source = self._verified_asset(
                source_revision_dir, source_assets, "identity"
            )
            uv_source = self._verified_asset(
                source_revision_dir, source_assets, "texture_uvs"
            )
            assert identity_source is not None and uv_source is not None
            copied: dict[str, dict[str, Any]] = {
                "identity": _copy_verified(
                    identity_source, temporary / "identity.npz"
                ),
                "texture_uvs": _copy_verified(
                    uv_source, temporary / "texture-uvs.npz"
                ),
            }
            copied["texture_uvs"]["array_sha256"] = (
                base.texture_uvs_array_sha256
            )
            if copied["identity"]["sha256"] != base.identity_sha256:
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Copied character identity hash changed"
                )
            if copied["texture_uvs"]["sha256"] != base.texture_uvs_sha256:
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Copied character UV hash changed"
                )

            material_asset_keys: dict[str, str] = {}
            material_paths: dict[str, Path] = {}
            for semantic, entry in sorted(validated["maps"].items()):
                file_entry = entry["files"]["atlas"]
                relative = str(file_entry["path"])
                suffix = Path(relative).suffix.lower()
                if semantic.startswith("masks."):
                    logical = f"mask__{semantic.split('.', 1)[1]}"
                else:
                    logical = semantic
                filename = f"material-{logical.replace('__', '-')}{suffix}"
                copied[logical] = _copy_validated_material_asset(
                    package,
                    relative,
                    temporary / filename,
                    expected_sha256=str(file_entry["sha256"]),
                    expected_bytes=int(file_entry["bytes"]),
                )
                material_asset_keys[semantic] = logical
                if semantic in PRESERVED_SEMANTICS:
                    material_paths[semantic] = temporary / filename

            try:
                runtime_material_paths = write_runtime_material_derivatives(
                    material_paths,
                    temporary,
                    normal_encoding=str(
                        validated["maps"]["normal"]["normal_encoding"]
                    ),
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise AutoAnimError(
                    "MATERIAL_RUNTIME_UNSUPPORTED",
                    "Material package is valid for retention but cannot be projected safely into the bounded realtime glTF profile",
                ) from exc
            runtime_asset_keys: dict[str, str] = {}
            for semantic, path in sorted(runtime_material_paths.items()):
                logical = f"runtime__{semantic}"
                copied[logical] = {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                runtime_asset_keys[semantic] = logical
            with Image.open(runtime_material_paths["base_color"]) as runtime_base:
                runtime_resolution = [
                    int(runtime_base.size[0]),
                    int(runtime_base.size[1]),
                ]
            source_base = validated["maps"]["base_color"]["files"]["atlas"]
            source_resolution = [
                int(source_base["width"]),
                int(source_base["height"]),
            ]
            source_runtime_bindings = {
                semantic: {
                    "sha256": str(entry["files"]["atlas"]["sha256"]),
                    "width": int(entry["files"]["atlas"]["width"]),
                    "height": int(entry["files"]["atlas"]["height"]),
                    "dtype": str(entry["files"]["atlas"]["dtype"]),
                    "color_space": str(entry["color_space"]),
                    "resampling": str(entry["resampling"]),
                }
                for semantic, entry in sorted(validated["maps"].items())
                if semantic in {"base_color", "normal", "roughness", "specular_color"}
            }
            derivative_bindings = {
                semantic: {
                    "sha256": str(copied[logical]["sha256"]),
                    "bytes": int(copied[logical]["bytes"]),
                    "width": runtime_resolution[0],
                    "height": runtime_resolution[1],
                    "format": "PNG",
                    "bit_depth": 8,
                }
                for semantic, logical in sorted(runtime_asset_keys.items())
            }

            binding = validated_attachment
            write_json(temporary / "material-package.json", validated)
            copied["material_package"] = {
                "name": "material-package.json",
                "bytes": (temporary / "material-package.json").stat().st_size,
                "sha256": sha256(temporary / "material-package.json"),
            }
            material_descriptor = {
                "schema_version": "autoanim.character-material.v3",
                "model": "OpenPBR-compatible source package with glTF runtime projection",
                "package_id": validated["package_id"],
                "package_manifest_payload_sha256": validated[
                    "manifest_payload_sha256"
                ],
                "package_artifact": "material_package",
                "maps": material_asset_keys,
                "binding": binding,
                "claims": validated["claims"],
                "quality_evidence": validated["quality_evidence"],
                "runtime_projection": {
                    "schema_version": RUNTIME_PROJECTION_PROFILE,
                    "source_package_manifest_payload_sha256": validated[
                        "manifest_payload_sha256"
                    ],
                    "assets": runtime_asset_keys,
                    "source_bindings": source_runtime_bindings,
                    "derivative_bindings": derivative_bindings,
                    "rendered": list(runtime_asset_keys),
                    "preserved_not_rendered": [
                        name
                        for name in (
                            "displacement",
                            "subsurface_color",
                            "subsurface_radius",
                            "confidence",
                        )
                        if name in material_paths
                    ],
                    "specular_linear_to_srgb": True,
                    "normal_green_reflected_for_gltf_v_flip": True,
                    "source_normal_encoding": validated["maps"]["normal"][
                        "normal_encoding"
                    ],
                    "runtime_bit_depth": 8,
                    "source_precision_preserved": True,
                    "source_resolution": source_resolution,
                    "runtime_resolution": runtime_resolution,
                    "maximum_runtime_dimension": MAX_RUNTIME_TEXTURE_DIMENSION,
                    "downsample_filter": "power_of_two_box_linear_light_v1",
                    "normal_filter": "vector_average_renormalize_v1",
                    "roughness_filter": "linear_box_v1",
                    "source_decode": "bounded_tiff_scratch_chunks_or_resident_png_v1",
                },
                "production_validated": False,
            }
            write_json(temporary / "material.json", material_descriptor)
            copied["material"] = {
                "name": "material.json",
                "bytes": (temporary / "material.json").stat().st_size,
                "sha256": sha256(temporary / "material.json"),
            }

            adapter = GNMAdapter()
            neutral_vertices = adapter.mesh(identity=base.identity)
            export_animated_gnm_glb(
                temporary / "preview.glb",
                adapter,
                neutral_vertices[None, ...],
                np.asarray([0.0], dtype=np.float32),
                mapping_path=temporary / "preview-mapping.npz",
                triangle_uvs=base.triangle_uvs,
                runtime_material_paths=runtime_material_paths,
            )
            copied["preview"] = {
                "name": "preview.glb",
                "bytes": (temporary / "preview.glb").stat().st_size,
                "sha256": sha256(temporary / "preview.glb"),
            }
            copied["preview_mapping"] = {
                "name": "preview-mapping.npz",
                "bytes": (temporary / "preview-mapping.npz").stat().st_size,
                "sha256": sha256(temporary / "preview-mapping.npz"),
            }

            revision_manifest = {
                "schema_version": "1.1",
                "character_id": character_id,
                "revision_id": revision_id,
                "created_at": created,
                "gnm": {
                    "version": "3.0",
                    "identity_dim": 253,
                    "identity_sha256": copied["identity"]["sha256"],
                    "texture_uvs_array_sha256": base.texture_uvs_array_sha256,
                },
                "source": {
                    "parent_revision_id": base.revision_id,
                    "parent_revision_manifest_sha256": base.manifest_sha256,
                    "reconstruction": base.manifest.get("source"),
                },
                "consent": base.manifest.get("consent"),
                "appearance": {
                    "material_artifact": "material",
                    "material_package_artifact": "material_package",
                    "material_package_id": validated["package_id"],
                    "material_manifest_payload_sha256": validated[
                        "manifest_payload_sha256"
                    ],
                    "capture_class": validated["capture"]["method"],
                    "resolution_label": validated["claims"]["resolution_label"],
                    "layout": quality["layout"],
                    "material_rights_expires_at": validated["rights"]["expires_at"],
                    "uv_binding": binding,
                    "pore_claim_gate_passed": quality[
                        "pore_claim_gate_passed"
                    ],
                    "relightable_claim_gate_passed": quality[
                        "relightable_claim_gate_passed"
                    ],
                    "pore_frequency_validation_performed": False,
                    "unseen_light_validation_performed": False,
                    "production_validated": False,
                },
                "oral": base.manifest.get("oral"),
                "body": base.manifest.get("body"),
                "assets": copied,
                "production_validated": False,
            }
            write_json(temporary / "manifest.json", revision_manifest)
            revision_manifest_sha256 = sha256(temporary / "manifest.json")

            latest = self.read(character_id)
            if (
                latest.get("updated_at") != character.get("updated_at")
                or latest.get("current_revision_id")
                != character.get("current_revision_id")
            ):
                raise AutoAnimError(
                    "BUSY", "Character changed during material import; retry from the latest revision"
                )
            os.replace(temporary, final_revision)
            published = True
            _fsync_directory(revisions_dir)
            revisions = dict(latest.get("revisions", {}))
            revisions[revision_id] = {
                "manifest_sha256": revision_manifest_sha256,
                "created_at": created,
                "parent_revision_id": base.revision_id,
                "identity_sha256": copied["identity"]["sha256"],
                "texture_uvs_array_sha256": base.texture_uvs_array_sha256,
            }
            resolution = str(validated["claims"]["resolution_label"])
            updated = dict(latest)
            updated.update(
                {
                    "updated_at": created,
                    "current_revision_id": revision_id,
                    "current_revision_manifest_sha256": revision_manifest_sha256,
                    "revision_count": len(revisions),
                    "revisions": revisions,
                    "appearance_status": f"pbr_atlas_attached_{resolution}_claim_gated",
                    "current_identity_sha256": copied["identity"]["sha256"],
                    "current_texture_uvs_array_sha256": base.texture_uvs_array_sha256,
                    "current_material_rights_expires_at": validated["rights"][
                        "expires_at"
                    ],
                    "production_validated": False,
                }
            )
            write_json(
                character_dir / "manifest.json", self.jobs.signer.sign(updated)
            )
            top_published = True
            _fsync_directory(character_dir)
        except Exception:
            # Once the signed top pointer has been atomically replaced, the
            # new revision must remain present even if the subsequent
            # directory fsync reports an error. Deleting it would leave a
            # cryptographically valid top manifest pointing at missing data.
            if published and not top_published:
                shutil.rmtree(final_revision, ignore_errors=True)
                _fsync_directory(revisions_dir)
            elif not published:
                shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.read(character_id)

    def revoke(self, character_id: str, *, reason: str, revoked_by: str) -> dict[str, Any]:
        with self._character_lock(character_id):
            return self._revoke_locked(
                character_id, reason=reason, revoked_by=revoked_by
            )

    def _revoke_locked(
        self, character_id: str, *, reason: str, revoked_by: str
    ) -> dict[str, Any]:
        """Revoke future use while retaining the immutable audit history."""

        clean_reason = _clean_required(reason, "Revocation reason", maximum=500)
        clean_actor = _clean_required(revoked_by, "Revoked by")
        character_dir = self._character_dir(character_id)
        character = self.read(character_id)
        if character.get("consent_status") == "revoked":
            return character
        now = utc_now()
        updated = dict(character)
        updated.update(
            {
                "updated_at": now,
                "consent_status": "revoked",
                "revocation": {
                    "revoked_at": now,
                    "revoked_by": clean_actor,
                    "reason": clean_reason,
                },
            }
        )
        write_json(character_dir / "manifest.json", self.jobs.signer.sign(updated))
        return self.read(character_id)

    def read(self, character_id: str) -> dict[str, Any]:
        path = self._character_dir(character_id) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(character_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(character_id) from exc
        if value.get("character_id") != character_id:
            raise FileNotFoundError(character_id)
        if not self.jobs.signer.verify(value):
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Character manifest failed its cryptographic integrity check",
            )
        return value

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        characters: list[dict[str, Any]] = []
        for path in self.root.glob("*/manifest.json"):
            try:
                character_id = path.parent.name
                value = self.read(character_id)
                status = self._effective_consent_status(value)
                material_rights_status = self._effective_material_rights_status(
                    value
                )
                item = {
                    key: (status if key == "consent_status" else value.get(key))
                    for key in (
                        "character_id",
                        "name",
                        "created_at",
                        "updated_at",
                        "current_revision_id",
                        "consent_status",
                        "consent_scope",
                        "appearance_status",
                        "body_status",
                        "production_validated",
                    )
                }
                item["material_rights_status"] = material_rights_status
                item["current_material_rights_expires_at"] = value.get(
                    "current_material_rights_expires_at"
                )
                characters.append(item)
            except (FileNotFoundError, AutoAnimError):
                continue
        characters.sort(
            key=lambda value: (str(value.get("updated_at") or ""), value["character_id"]),
            reverse=True,
        )
        return characters[: max(0, limit)]

    def resolve(
        self,
        character_id: str,
        revision_id: str | None = None,
        *,
        usage_scope: str = "personal",
    ) -> CharacterRevision:
        character = self.read(character_id)
        consent_status = self._effective_consent_status(character)
        if consent_status == "revoked":
            raise AutoAnimError(
                "CONSENT_REVOKED", "Character consent has been revoked; reuse is blocked"
            )
        if consent_status == "expired":
            raise AutoAnimError(
                "CONSENT_EXPIRED", "Character consent has expired; reuse is blocked"
            )
        if consent_status != "active":
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Character consent state is invalid; reuse is blocked"
            )
        requested_scope = usage_scope.strip().lower()
        granted_scope = str(character.get("consent_scope") or "")
        if requested_scope not in _CONSENT_SCOPES:
            raise AutoAnimError("INPUT_INVALID", "Requested character usage scope is invalid")
        if requested_scope not in _SCOPE_GRANTS.get(granted_scope, frozenset()):
            raise AutoAnimError(
                "CONSENT_SCOPE_DENIED",
                f"Character consent scope {granted_scope!r} does not authorize {requested_scope!r} use",
            )
        selected_revision = revision_id or character.get("current_revision_id")
        if not isinstance(selected_revision, str):
            raise FileNotFoundError(character_id)
        revision_dir = self._revision_dir(character_id, selected_revision)
        manifest_path = revision_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(selected_revision)
        revision_index = character.get("revisions", {})
        anchored = revision_index.get(selected_revision) if isinstance(revision_index, dict) else None
        expected_manifest_sha256 = (
            anchored.get("manifest_sha256") if isinstance(anchored, dict) else None
        )
        if not isinstance(expected_manifest_sha256, str) or sha256(manifest_path) != expected_manifest_sha256:
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Character revision manifest failed its integrity check"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(selected_revision) from exc
        if (
            manifest.get("character_id") != character_id
            or manifest.get("revision_id") != selected_revision
        ):
            raise FileNotFoundError(selected_revision)
        assets = manifest.get("assets", {})

        consent = manifest.get("consent", {})
        expires_at = consent.get("expires_at") if isinstance(consent, dict) else None
        if isinstance(expires_at, str):
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise AutoAnimError(
                    "INTERNAL_ERROR", "Character consent expiry is invalid"
                ) from exc
            if expiry.tzinfo is None:
                raise AutoAnimError("INTERNAL_ERROR", "Character consent expiry has no timezone")
            if expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
                raise AutoAnimError(
                    "CONSENT_EXPIRED", "Character consent has expired; reuse is blocked"
                )
        appearance = manifest.get("appearance", {})
        material_rights_expiry = (
            appearance.get("material_rights_expires_at")
            if isinstance(appearance, dict)
            else None
        )
        if isinstance(material_rights_expiry, str):
            try:
                rights_expiry = datetime.fromisoformat(
                    material_rights_expiry.replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Character material rights expiry is invalid"
                ) from exc
            if rights_expiry.tzinfo is None:
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Character material rights expiry has no timezone"
                )
            if rights_expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
                raise AutoAnimError(
                    "RIGHTS_EXPIRED", "Character material rights have expired; reuse is blocked"
                )

        def verified_asset(logical: str, *, required: bool = True) -> Path | None:
            return self._verified_asset(revision_dir, assets, logical, required=required)

        identity_path = verified_asset("identity")
        assert identity_path is not None
        try:
            with np.load(identity_path, allow_pickle=False) as values:
                identity = np.asarray(values["identity"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise AutoAnimError("INTERNAL_ERROR", "Character identity is unreadable") from exc
        if identity.shape != (253,) or not np.isfinite(identity).all():
            raise AutoAnimError("INTERNAL_ERROR", "Character identity is invalid")
        identity.setflags(write=False)
        preview = verified_asset("preview")
        assert preview is not None
        material_descriptor = verified_asset("material")
        assert material_descriptor is not None
        try:
            material_document = json.loads(
                material_descriptor.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Character material descriptor is unreadable"
            ) from exc
        if not isinstance(material_document, dict):
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Character material descriptor is invalid"
            )
        material_paths: dict[str, Path] = {}
        material_asset_paths: dict[str, Path] = {}
        material_sha256s: dict[str, str] = {}
        runtime_material_paths: dict[str, Path] = {}
        runtime_material_sha256s: dict[str, str] = {}
        descriptor_maps = material_document.get("maps", {})
        material_schema = material_document.get("schema_version")
        if material_schema in {
            "autoanim.character-material.v2",
            "autoanim.character-material.v3",
        }:
            package_logical = material_document.get("package_artifact")
            if not isinstance(package_logical, str):
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character material descriptor has no package artifact",
                )
            package_document_path = verified_asset(package_logical)
            assert package_document_path is not None
            try:
                package_document = json.loads(
                    package_document_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Character material package is unreadable"
                ) from exc
            if not isinstance(package_document, dict):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Character material package is invalid"
                )
            package_payload = dict(package_document)
            supplied_package_digest = package_payload.pop(
                "manifest_payload_sha256", None
            )
            try:
                calculated_package_digest = hashlib.sha256(
                    json.dumps(
                        package_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            except (TypeError, ValueError) as exc:
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character material package is not strict JSON",
                ) from exc
            binding = material_document.get("binding")
            expected_package_digest = material_document.get(
                "package_manifest_payload_sha256"
            )
            if (
                supplied_package_digest != calculated_package_digest
                or supplied_package_digest != expected_package_digest
                or not isinstance(binding, dict)
                or binding.get("material_manifest_payload_sha256")
                != supplied_package_digest
            ):
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character material package digest or attachment binding is invalid",
                )
            if not isinstance(descriptor_maps, dict):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Character material map index is invalid"
                )
            for semantic, logical in sorted(descriptor_maps.items()):
                if not isinstance(semantic, str) or not isinstance(logical, str):
                    raise AutoAnimError(
                        "INTEGRITY_FAILED", "Character material map index is invalid"
                    )
                material_path = verified_asset(logical)
                assert material_path is not None
                material_asset_paths[semantic] = material_path
                material_sha256s[semantic] = str(assets[logical]["sha256"])
                if semantic in PRESERVED_SEMANTICS:
                    material_paths[semantic] = material_path
            runtime_projection = material_document.get("runtime_projection", {})
            runtime_assets = (
                runtime_projection.get("assets")
                if isinstance(runtime_projection, dict)
                else None
            )
            if not isinstance(runtime_assets, dict):
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character runtime material asset index is invalid",
                )
            if material_schema == "autoanim.character-material.v3":
                if (
                    package_document.get("schema_version")
                    != "autoanim.material-package.v2"
                    or runtime_projection.get("schema_version")
                    != RUNTIME_PROJECTION_PROFILE
                    or runtime_projection.get(
                        "source_package_manifest_payload_sha256"
                    )
                    != supplied_package_digest
                    or runtime_projection.get("maximum_runtime_dimension")
                    != MAX_RUNTIME_TEXTURE_DIMENSION
                    or runtime_projection.get("source_decode")
                    != "bounded_tiff_scratch_chunks_or_resident_png_v1"
                    or runtime_projection.get("downsample_filter")
                    != "power_of_two_box_linear_light_v1"
                    or runtime_projection.get("normal_filter")
                    != "vector_average_renormalize_v1"
                    or runtime_projection.get("roughness_filter")
                    != "linear_box_v1"
                ):
                    raise AutoAnimError(
                        "INTEGRITY_FAILED",
                        "Character runtime projection provenance is invalid",
                    )
                source_bindings = runtime_projection.get("source_bindings")
                package_maps = package_document.get("maps")
                if (
                    not isinstance(source_bindings, dict)
                    or set(source_bindings)
                    != {"base_color", "normal", "roughness", "specular_color"}
                    or not isinstance(package_maps, dict)
                ):
                    raise AutoAnimError(
                        "INTEGRITY_FAILED",
                        "Character runtime source bindings are invalid",
                    )
                for semantic, source_binding in source_bindings.items():
                    package_map = package_maps.get(semantic)
                    if (
                        not isinstance(source_binding, dict)
                        or not isinstance(package_map, dict)
                        or not isinstance(package_map.get("files"), dict)
                        or not isinstance(
                            package_map["files"].get("atlas"), dict
                        )
                    ):
                        raise AutoAnimError(
                            "INTEGRITY_FAILED",
                            "Character runtime source binding is malformed",
                        )
                    source_file = package_map["files"]["atlas"]
                    expected_source_binding = {
                        "sha256": source_file.get("sha256"),
                        "width": source_file.get("width"),
                        "height": source_file.get("height"),
                        "dtype": source_file.get("dtype"),
                        "color_space": package_map.get("color_space"),
                        "resampling": package_map.get("resampling"),
                    }
                    if source_binding != expected_source_binding:
                        raise AutoAnimError(
                            "INTEGRITY_FAILED",
                            "Character runtime source binding does not match the sealed package",
                        )
            if not set(runtime_assets) <= RUNTIME_DERIVATIVE_KEYS:
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character runtime material contains unknown semantics",
                )
            if material_schema == "autoanim.character-material.v3":
                derivative_bindings = runtime_projection.get(
                    "derivative_bindings"
                )
                if (
                    not isinstance(derivative_bindings, dict)
                    or set(derivative_bindings) != set(runtime_assets)
                ):
                    raise AutoAnimError(
                        "INTEGRITY_FAILED",
                        "Character runtime derivative bindings are invalid",
                    )
            for semantic, logical in sorted(runtime_assets.items()):
                if not isinstance(logical, str):
                    raise AutoAnimError(
                        "INTEGRITY_FAILED",
                        "Character runtime material asset index is invalid",
                    )
                runtime_path = verified_asset(logical)
                assert runtime_path is not None
                runtime_material_paths[semantic] = runtime_path
                runtime_material_sha256s[semantic] = str(
                    assets[logical]["sha256"]
                )
                if material_schema == "autoanim.character-material.v3":
                    derivative_bindings = runtime_projection.get(
                        "derivative_bindings"
                    )
                    derivative = (
                        derivative_bindings.get(semantic)
                        if isinstance(derivative_bindings, dict)
                        else None
                    )
                    if not isinstance(derivative, dict):
                        raise AutoAnimError(
                            "INTEGRITY_FAILED",
                            "Character runtime derivative binding is missing",
                        )
                    with Image.open(runtime_path) as runtime_image:
                        actual_size = runtime_image.size
                        actual_format = runtime_image.format
                    expected_derivative = {
                        "sha256": assets[logical]["sha256"],
                        "bytes": assets[logical]["bytes"],
                        "width": int(actual_size[0]),
                        "height": int(actual_size[1]),
                        "format": actual_format,
                        "bit_depth": 8,
                    }
                    if derivative != expected_derivative:
                        raise AutoAnimError(
                            "INTEGRITY_FAILED",
                            "Character runtime derivative binding is invalid",
                        )
            if "base_color" not in runtime_material_paths:
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character runtime material has no base-color derivative",
                )
        else:
            for semantic in sorted(PRESERVED_SEMANTICS):
                material_path = verified_asset(semantic, required=False)
                if material_path is not None:
                    material_paths[semantic] = material_path
                    material_asset_paths[semantic] = material_path
                    material_sha256s[semantic] = str(assets[semantic]["sha256"])
        texture = material_paths.get("base_color")
        texture_uvs_path = verified_asset("texture_uvs", required=False)
        triangle_uvs: np.ndarray | None = None
        triangle_uvs_array_sha256: str | None = None
        if texture_uvs_path is not None:
            try:
                with np.load(texture_uvs_path, allow_pickle=False) as values:
                    triangle_uvs = np.asarray(values["triangle_uvs"], dtype=np.float32)
            except (OSError, KeyError, ValueError) as exc:
                raise AutoAnimError(
                    "INTERNAL_ERROR", "Character texture UV layout is unreadable"
                ) from exc
            if (
                triangle_uvs.shape != (_GNM_TRIANGLE_COUNT, 3, 2)
                or not np.isfinite(triangle_uvs).all()
                or np.min(triangle_uvs) < 0.0
                or np.max(triangle_uvs) > 1.0
            ):
                raise AutoAnimError(
                    "INTERNAL_ERROR", "Character texture UV layout is invalid"
                )
            triangle_uvs_array_sha256 = _uv_array_sha256(triangle_uvs)
            expected_array_sha256 = (
                assets.get("texture_uvs", {}).get("array_sha256")
                if isinstance(assets.get("texture_uvs"), dict)
                else None
            )
            if (
                expected_array_sha256 is not None
                and triangle_uvs_array_sha256 != expected_array_sha256
            ):
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Character UV value array failed its canonical integrity check",
                )
            triangle_uvs.setflags(write=False)
        elif texture is not None:
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Character texture has no sealed UV layout"
            )
        return CharacterRevision(
            character_id=character_id,
            revision_id=selected_revision,
            name=str(character.get("name")),
            identity=identity,
            texture_path=texture,
            triangle_uvs=triangle_uvs,
            preview_path=preview,
            manifest=manifest,
            manifest_sha256=expected_manifest_sha256,
            identity_sha256=str(assets["identity"]["sha256"]),
            texture_sha256=(
                str(assets["base_color"]["sha256"])
                if isinstance(assets.get("base_color"), dict)
                else None
            ),
            texture_uvs_sha256=(
                str(assets["texture_uvs"]["sha256"])
                if isinstance(assets.get("texture_uvs"), dict)
                else None
            ),
            texture_uvs_array_sha256=triangle_uvs_array_sha256,
            material_paths=material_paths,
            material_asset_paths=material_asset_paths,
            material_sha256s=material_sha256s,
            runtime_material_paths=runtime_material_paths,
            runtime_material_sha256s=runtime_material_sha256s,
            material_manifest_sha256=str(assets["material"]["sha256"]),
        )

    @staticmethod
    def _effective_consent_status(character: dict[str, Any]) -> str:
        status = str(character.get("consent_status") or "unknown")
        if status == "revoked":
            return "revoked"
        expires_at = character.get("consent_expires_at")
        if isinstance(expires_at, str):
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            except ValueError:
                return "invalid"
            if expiry.tzinfo is None:
                return "invalid"
            if expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
                return "expired"
        return status

    @staticmethod
    def _effective_material_rights_status(character: dict[str, Any]) -> str:
        expires_at = character.get("current_material_rights_expires_at")
        if expires_at is None:
            return "not_applicable"
        if not isinstance(expires_at, str):
            return "invalid"
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return "invalid"
        if expiry.tzinfo is None:
            return "invalid"
        return (
            "expired"
            if expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc)
            else "active"
        )

    @staticmethod
    def _verified_asset(
        revision_dir: Path,
        assets: Any,
        logical: str,
        *,
        required: bool = True,
    ) -> Path | None:
        entry = assets.get(logical) if isinstance(assets, dict) else None
        if entry is None and not required:
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise AutoAnimError("INTEGRITY_FAILED", f"Character {logical} asset is missing")
        name = entry["name"]
        if Path(name).name != name or Path(name).is_absolute():
            raise AutoAnimError(
                "INTEGRITY_FAILED", f"Character {logical} asset path is unsafe"
            )
        unresolved = revision_dir / name
        if unresolved.is_symlink():
            raise AutoAnimError(
                "INTEGRITY_FAILED", f"Character {logical} asset cannot be a symlink"
            )
        path = unresolved.resolve()
        if path.parent != revision_dir.resolve() or not path.is_file():
            raise AutoAnimError("INTEGRITY_FAILED", f"Character {logical} asset is missing")
        if (
            not isinstance(entry.get("bytes"), int)
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry.get("sha256")
        ):
            raise AutoAnimError(
                "INTEGRITY_FAILED", f"Character {logical} asset failed its integrity check"
            )
        return path

    def asset(
        self,
        character_id: str,
        revision_id: str,
        logical_name: str,
        *,
        usage_scope: str = "personal",
        _resolved: CharacterRevision | None = None,
    ) -> Path:
        try:
            revision = _resolved or self.resolve(
                character_id, revision_id, usage_scope=usage_scope
            )
            if (
                revision.character_id != character_id
                or revision.revision_id != revision_id
            ):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Resolved character revision does not match asset request"
                )
            path = self._verified_asset(
                self._revision_dir(character_id, revision_id),
                revision.manifest.get("assets", {}),
                logical_name,
            )
        except AutoAnimError as exc:
            raise FileNotFoundError(logical_name) from exc
        if path is None:
            raise FileNotFoundError(logical_name)
        return path


__all__ = ["CharacterRevision", "CharacterStore"]
