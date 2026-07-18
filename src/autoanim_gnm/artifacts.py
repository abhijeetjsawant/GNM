"""Versioned local job artifacts and terminal manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import secrets
import shutil
from typing import Any

from .serialization import write_json


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
        self.recover_interrupted()

    def recover_interrupted(self) -> None:
        for manifest_path in self.root.glob("*/result.json"):
            try:
                import json

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "running":
                    manifest["status"] = "failed"
                    manifest["updated_at"] = utc_now()
                    manifest["error"] = {
                        "code": "PROCESS_INTERRUPTED",
                        "message": "The prior application process stopped before this job completed.",
                        "details": {},
                        "retryable": True,
                    }
                    write_json(manifest_path, manifest)
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
        write_json(job_dir / "result.json", manifest)
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
        write_json(job_dir / "result.json", manifest)
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
        write_json(job_dir / "result.json", final)
        return final

    def fail(self, manifest: dict, job_dir: Path, error: dict, versions: dict[str, str]) -> dict:
        manifest.update(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "versions": versions,
                "error": error,
            }
        )
        write_json(job_dir / "result.json", manifest)
        return manifest

    def read(self, job_id: str) -> dict:
        import json

        path = self.job_dir(job_id) / "result.json"
        if not path.is_file():
            raise FileNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return privacy-minimized summaries in reverse chronological order."""

        import json

        summaries: list[dict[str, Any]] = []
        for manifest_path in self.root.glob("*/result.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                job_id = manifest_path.parent.name
                self.job_dir(job_id)
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
        allowed = {entry["name"] for entry in manifest.get("artifacts", {}).values()}
        if Path(name).name != name or name not in allowed:
            raise FileNotFoundError(name)
        path = self.job_dir(job_id) / name
        if not path.is_file():
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
