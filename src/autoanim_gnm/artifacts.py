"""Versioned local job artifacts and terminal manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
from typing import Any

from .serialization import write_json
from .errors import AutoAnimError
from .integrity import IntegritySigner


_ULID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_ulid() -> str:
    milliseconds = int(datetime.now(timezone.utc).timestamp() * 1000)
    value = (milliseconds << 80) | secrets.randbits(80)
    characters = []
    for _ in range(26):
        characters.append(_ULID_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(characters))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_input_name(name: str) -> str:
    basename = Path(name).name
    clean = "".join(character if character.isalnum() or character in ".-_" else "_" for character in basename)
    return clean[:200] or "input.bin"


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.signer = IntegritySigner(
            self.root.parent / ".autoanim-integrity" / "hmac.key"
        )
        self.recover_interrupted()

    def _write_manifest(self, path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        signed = self.signer.sign(manifest)
        write_json(path, signed)
        return signed

    def recover_interrupted(self) -> None:
        for manifest_path in self.root.glob("*/result.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if "integrity" in manifest and not self.signer.verify(manifest):
                    continue
                if manifest.get("status") == "running":
                    manifest["status"] = "failed"
                    manifest["updated_at"] = utc_now()
                    manifest["error"] = {
                        "code": "PROCESS_INTERRUPTED",
                        "message": "The prior application process stopped before this job completed.",
                        "details": {},
                        "retryable": True,
                    }
                    self._write_manifest(manifest_path, manifest)
            except Exception:
                continue

    def start(
        self,
        kind: str,
        input_path: str | Path,
        configuration: dict[str, Any],
        *,
        original_name: str | None = None,
    ) -> tuple[str, Path, Path, dict]:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        job_id = new_ulid()
        job_dir = self.root / job_id
        job_dir.mkdir(mode=0o700)
        sanitized = safe_input_name(original_name or source.name)
        suffix = Path(sanitized).suffix.lower() or ".bin"
        retained_input = job_dir / f"input{suffix}"
        shutil.copy2(source, retained_input)
        created = utc_now()
        manifest = {
            "schema_version": "1.0",
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "created_at": created,
            "updated_at": created,
            "input": {
                "name": sanitized,
                "sha256": sha256(retained_input),
                "bytes": retained_input.stat().st_size,
                "media_type": _media_type(retained_input),
            },
            "configuration": configuration,
            "versions": {},
            "metrics": {},
            "warnings": [],
            "artifacts": {},
            "error": None,
        }
        manifest = self._write_manifest(job_dir / "result.json", manifest)
        return job_id, job_dir, retained_input, manifest

    def start_many(
        self,
        kind: str,
        input_paths: list[str | Path] | tuple[str | Path, ...],
        configuration: dict[str, Any],
        *,
        original_names: list[str] | tuple[str, ...] | None = None,
        attachments: dict[str, str | Path] | None = None,
    ) -> tuple[str, Path, tuple[Path, ...], dict]:
        """Start one job with an ordered, hash-audited set of retained inputs."""

        sources = tuple(Path(path) for path in input_paths)
        if not sources or any(not source.is_file() for source in sources):
            raise FileNotFoundError("Every multi-input source must be a file")
        if original_names is not None and len(original_names) != len(sources):
            raise ValueError("original_names must match input_paths")
        job_id = new_ulid()
        job_dir = self.root / job_id
        job_dir.mkdir(mode=0o700)
        retained: list[Path] = []
        inputs: list[dict[str, Any]] = []
        retained_attachments: list[dict[str, Any]] = []
        aggregate = hashlib.sha256()
        total_bytes = 0
        try:
            for index, source in enumerate(sources):
                supplied_name = original_names[index] if original_names is not None else source.name
                sanitized = safe_input_name(supplied_name)
                suffix = Path(sanitized).suffix.lower() or ".bin"
                destination = job_dir / f"input-{index + 1:02d}{suffix}"
                shutil.copy2(source, destination)
                digest = sha256(destination)
                size = destination.stat().st_size
                aggregate.update(index.to_bytes(4, "big"))
                aggregate.update(bytes.fromhex(digest))
                retained.append(destination)
                total_bytes += size
                inputs.append(
                    {
                        "index": index,
                        "name": sanitized,
                        "sha256": digest,
                        "bytes": size,
                        "media_type": _media_type(destination),
                    }
                )
            for attachment_index, (logical_name, attachment_path) in enumerate(
                sorted((attachments or {}).items())
            ):
                source = Path(attachment_path)
                if not source.is_file():
                    raise FileNotFoundError(f"Attachment {logical_name} must be a file")
                safe_logical = safe_input_name(logical_name)
                suffix = source.suffix.lower() or ".bin"
                retained_name = f"attachment-{attachment_index + 1:02d}-{safe_logical}{suffix}"
                destination = job_dir / retained_name
                shutil.copy2(source, destination)
                digest = sha256(destination)
                size = destination.stat().st_size
                aggregate.update(b"attachment")
                aggregate.update(attachment_index.to_bytes(4, "big"))
                aggregate.update(logical_name.encode("utf-8"))
                aggregate.update(bytes.fromhex(digest))
                total_bytes += size
                retained_attachments.append(
                    {
                        "logical_name": logical_name,
                        "retained_name": retained_name,
                        "sha256": digest,
                        "bytes": size,
                        "media_type": _media_type(destination),
                    }
                )
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        created = utc_now()
        manifest = {
            "schema_version": "1.0",
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "created_at": created,
            "updated_at": created,
            "input": {
                "name": f"{len(inputs)} ordered images",
                "sha256": aggregate.hexdigest(),
                "bytes": total_bytes,
                "media_type": "multipart/mixed",
            },
            "inputs": inputs,
            **({"attachments": retained_attachments} if retained_attachments else {}),
            "configuration": configuration,
            "versions": {},
            "metrics": {},
            "warnings": [],
            "artifacts": {},
            "error": None,
        }
        manifest = self._write_manifest(job_dir / "result.json", manifest)
        return job_id, job_dir, tuple(retained), manifest

    def finish(
        self,
        manifest: dict,
        job_dir: Path,
        pipeline_result: dict,
        versions: dict[str, str],
    ) -> dict:
        detailed_artifacts: dict[str, dict[str, Any]] = {}
        for logical, name in pipeline_result.get("artifacts", {}).items():
            path = job_dir / name
            if path.is_file():
                detailed_artifacts[logical] = {
                    "name": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "media_type": _media_type(path),
                }
        final = dict(pipeline_result)
        final.update(
            {
                "schema_version": "1.0",
                "job_id": manifest["job_id"],
                "status": "succeeded",
                "created_at": manifest["created_at"],
                "updated_at": utc_now(),
                "input": manifest["input"],
                "configuration": manifest["configuration"],
                "versions": versions,
                "artifacts": detailed_artifacts,
                "error": None,
            }
        )
        if "inputs" in manifest:
            final["inputs"] = manifest["inputs"]
        if "attachments" in manifest:
            final["attachments"] = manifest["attachments"]
        return self._write_manifest(job_dir / "result.json", final)

    def fail(self, manifest: dict, job_dir: Path, error: dict, versions: dict[str, str]) -> dict:
        manifest.update(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "versions": versions,
                "error": error,
            }
        )
        return self._write_manifest(job_dir / "result.json", manifest)

    def read(self, job_id: str) -> dict:
        path = self.job_dir(job_id) / "result.json"
        if not path.is_file():
            raise FileNotFoundError(job_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if "integrity" in value and not self.signer.verify(value):
            raise AutoAnimError(
                "INTEGRITY_FAILED", "Job manifest failed its cryptographic integrity check"
            )
        return value

    def require_sealed(self, job_id: str) -> dict[str, Any]:
        value = self.read(job_id)
        if not self.signer.verify(value):
            raise AutoAnimError(
                "INTEGRITY_UNSEALED",
                "Legacy job has no trusted manifest seal; explicitly attest and seal it before reuse",
            )
        return value

    def seal_legacy(
        self,
        job_id: str,
        *,
        attested_by: str,
        reason: str,
    ) -> dict[str, Any]:
        """Explicitly adopt a pre-sealing job after verifying its current bytes."""

        actor = " ".join(attested_by.split())
        explanation = " ".join(reason.split())
        if not actor or len(actor) > 160 or not explanation or len(explanation) > 500:
            raise AutoAnimError(
                "INPUT_INVALID", "Legacy seal requires an attester and a concise reason"
            )
        path = self.job_dir(job_id) / "result.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(job_id) from exc
        if "integrity" in value:
            if not self.signer.verify(value):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Existing job seal is invalid and cannot be replaced"
                )
            return value
        if value.get("status") not in {"succeeded", "failed"}:
            raise AutoAnimError("INPUT_INVALID", "Only terminal legacy jobs can be sealed")
        job_dir = self.job_dir(job_id)
        for entry in value.get("artifacts", {}).values():
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise AutoAnimError("INTEGRITY_FAILED", "Legacy artifact ledger is invalid")
            name = entry["name"]
            candidate = job_dir / name
            if (
                Path(name).name != name
                or not candidate.is_file()
                or candidate.stat().st_size != entry.get("bytes")
                or sha256(candidate) != entry.get("sha256")
            ):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", f"Legacy artifact {name} does not match its ledger"
                )
        inputs = value.get("inputs")
        if isinstance(inputs, list):
            for entry in inputs:
                index = entry.get("index") if isinstance(entry, dict) else None
                matches = (
                    list(job_dir.glob(f"input-{index + 1:02d}.*"))
                    if isinstance(index, int)
                    else []
                )
                if (
                    len(matches) != 1
                    or matches[0].stat().st_size != entry.get("bytes")
                    or sha256(matches[0]) != entry.get("sha256")
                ):
                    raise AutoAnimError(
                        "INTEGRITY_FAILED", "Legacy input ledger does not match"
                    )
        else:
            entry = value.get("input", {})
            matches = [item for item in job_dir.glob("input.*") if item.is_file()]
            if (
                len(matches) != 1
                or matches[0].stat().st_size != entry.get("bytes")
                or sha256(matches[0]) != entry.get("sha256")
            ):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Legacy input ledger does not match"
                )
        for entry in value.get("attachments", []):
            name = entry.get("retained_name") if isinstance(entry, dict) else None
            candidate = job_dir / str(name)
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not candidate.is_file()
                or candidate.stat().st_size != entry.get("bytes")
                or sha256(candidate) != entry.get("sha256")
            ):
                raise AutoAnimError(
                    "INTEGRITY_FAILED", "Legacy attachment ledger does not match"
                )
        adopted = dict(value)
        adopted["integrity_migration"] = {
            "sealed_at": utc_now(),
            "attested_by": actor,
            "reason": explanation,
            "preexisting_provenance_not_cryptographically_proven": True,
        }
        return self._write_manifest(path, adopted)

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return privacy-minimized summaries in reverse chronological order."""

        summaries: list[dict[str, Any]] = []
        for manifest_path in self.root.glob("*/result.json"):
            try:
                job_id = manifest_path.parent.name
                manifest = self.read(job_id)
                artifacts = manifest.get("artifacts", {})
                viewable = isinstance(
                    artifacts.get("textured_glb") or artifacts.get("glb"), dict
                )
                input_summary = manifest.get("input", {})
                summaries.append(
                    {
                        "job_id": job_id,
                        "kind": manifest.get("kind", "unknown"),
                        "status": manifest.get("status", "unknown"),
                        "created_at": manifest.get("created_at"),
                        "updated_at": manifest.get("updated_at"),
                        "input": {
                            "name": input_summary.get("name", "input"),
                            "media_type": input_summary.get("media_type"),
                        },
                        "warning_count": len(manifest.get("warnings", [])),
                        "viewable": viewable,
                    }
                )
            except Exception:
                continue
        summaries.sort(
            key=lambda summary: (str(summary.get("created_at") or ""), summary["job_id"]),
            reverse=True,
        )
        return summaries[: max(0, limit)]

    def job_dir(self, job_id: str) -> Path:
        if len(job_id) != 26 or any(character not in _ULID_ALPHABET for character in job_id):
            raise FileNotFoundError(job_id)
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise FileNotFoundError(job_id)
        return path

    def artifact(self, job_id: str, name: str) -> Path:
        manifest = self.read(job_id)
        entry = next(
            (
                value
                for value in manifest.get("artifacts", {}).values()
                if isinstance(value, dict) and value.get("name") == name
            ),
            None,
        )
        if Path(name).name != name or entry is None:
            raise FileNotFoundError(name)
        path = self.job_dir(job_id) / name
        if not path.is_file():
            raise FileNotFoundError(name)
        if (
            path.stat().st_size != entry.get("bytes")
            or sha256(path) != entry.get("sha256")
        ):
            raise FileNotFoundError(name)
        return path


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".npz": "application/octet-stream",
        ".obj": "text/plain",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".png": "image/png",
        ".mp4": "video/mp4",
        ".flv": "video/x-flv",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aiff": "audio/aiff",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
