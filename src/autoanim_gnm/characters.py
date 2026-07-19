"""Versioned, consent-audited character assets promoted from reconstruction jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from .artifacts import JobStore, new_ulid, sha256, utc_now
from .errors import AutoAnimError
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
                assert texture_triangle_uvs is not None
                write_npz(
                    revision_dir / "texture-uvs.npz",
                    triangle_uvs=texture_triangle_uvs,
                )
                copied["texture_uvs"] = {
                    "name": "texture-uvs.npz",
                    "bytes": (revision_dir / "texture-uvs.npz").stat().st_size,
                    "sha256": sha256(revision_dir / "texture-uvs.npz"),
                }
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
                "uv_layout_asset": (
                    "texture_uvs" if texture_path is not None else None
                ),
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
                "body_status": "not_attached",
                "production_validated": False,
            }
            write_json(temporary / "manifest.json", self.jobs.signer.sign(top_manifest))
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.read(character_id)

    def revoke(self, character_id: str, *, reason: str, revoked_by: str) -> dict[str, Any]:
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
                characters.append(
                    {
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
                )
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
        texture = verified_asset("base_color", required=False)
        texture_uvs_path = verified_asset("texture_uvs", required=False)
        triangle_uvs: np.ndarray | None = None
        if texture is not None:
            if texture_uvs_path is None:
                raise AutoAnimError(
                    "INTERNAL_ERROR", "Character texture has no sealed UV layout"
                )
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
            triangle_uvs.setflags(write=False)
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
    ) -> Path:
        revision = self.resolve(
            character_id, revision_id, usage_scope=usage_scope
        )
        try:
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
