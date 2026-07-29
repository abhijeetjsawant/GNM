"""Durable local production hierarchy for projects, shots, takes, and versions.

This store deliberately sits beside the existing job and character stores.
It references their immutable records but never mutates or reinterprets them.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterator

from .artifacts import new_ulid, sha256, utc_now
from .characters import CharacterStore
from .errors import AutoAnimError
from .serialization import write_json


SCHEMA_VERSION = "autoanim.production-workspace.v1"
_ULID_ALPHABET = frozenset("0123456789abcdefghjkmnpqrstvwxyz")
_MEDIA_KINDS = frozenset(("audio", "video"))
_JOB_KIND_BY_MEDIA = {
    "audio": "audio_animation",
    "video": "video_performance",
}


def _valid_ulid(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 26
        and all(character in _ULID_ALPHABET for character in value)
    )


def _clean_text(
    value: str | None,
    label: str,
    *,
    maximum: int,
    required: bool = True,
) -> str | None:
    cleaned = " ".join(str(value or "").split())
    if (
        (required and not cleaned)
        or len(cleaned) > maximum
        or any(ord(character) < 32 for character in cleaned)
    ):
        qualifier = f"1-{maximum}" if required else f"at most {maximum}"
        raise AutoAnimError(
            "INPUT_INVALID",
            f"{label} must contain {qualifier} printable characters",
        )
    return cleaned or None


def _clean_sha256(value: str, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AutoAnimError("INPUT_INVALID", f"{label} must be one SHA-256 digest")
    return digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ProductionStore:
    """File-backed production records with immutable character-revision pins."""

    def __init__(self, root: str | Path, characters: CharacterStore):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.characters = characters
        self.jobs = characters.jobs
        self.signer = self.jobs.signer

    def _project_dir(self, project_id: str) -> Path:
        if not _valid_ulid(project_id):
            raise FileNotFoundError(project_id)
        path = (self.root / project_id).resolve()
        if path.parent != self.root:
            raise FileNotFoundError(project_id)
        return path

    def _shot_dir(self, project_id: str, shot_id: str) -> Path:
        if not _valid_ulid(shot_id):
            raise FileNotFoundError(shot_id)
        parent = (self._project_dir(project_id) / "shots").resolve()
        path = (parent / shot_id).resolve()
        if path.parent != parent:
            raise FileNotFoundError(shot_id)
        return path

    @staticmethod
    def _record_path(parent: Path, record_id: str) -> Path:
        if not _valid_ulid(record_id):
            raise FileNotFoundError(record_id)
        path = (parent / f"{record_id}.json").resolve()
        if path.parent != parent.resolve():
            raise FileNotFoundError(record_id)
        return path

    @contextmanager
    def _lock(self, directory: Path) -> Iterator[None]:
        if not directory.is_dir():
            raise FileNotFoundError(directory.name)
        descriptor = os.open(
            directory / ".lock",
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AutoAnimError(
                    "INTEGRITY_FAILED",
                    "Production mutation lock is not a regular file",
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _write_record(self, path: Path, value: dict[str, Any]) -> dict[str, Any]:
        signed = self.signer.sign(value)
        write_json(path, signed)
        return signed

    def _read_record(
        self,
        path: Path,
        *,
        id_field: str,
        expected_id: str,
    ) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(expected_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Production record is unreadable",
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get(id_field) != expected_id
        ):
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Production record identity or schema is invalid",
            )
        if not self.signer.verify(value):
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Production record failed its cryptographic integrity check",
            )
        return value

    def create_project(
        self,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        clean_name = _clean_text(name, "Project name", maximum=120)
        clean_description = _clean_text(
            description, "Project description", maximum=1000, required=False
        )
        project_id = new_ulid()
        created = utc_now()
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "project",
            "project_id": project_id,
            "name": clean_name,
            "description": clean_description,
            "lifecycle": "active",
            "created_at": created,
            "updated_at": created,
        }
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{project_id}.",
                suffix=".tmp",
                dir=self.root,
            )
        )
        try:
            (temporary / "shots").mkdir()
            self._write_record(temporary / "manifest.json", record)
            _fsync_directory(temporary)
            destination = self._project_dir(project_id)
            os.replace(temporary, destination)
            _fsync_directory(self.root)
        except Exception:
            if temporary.exists():
                import shutil

                shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.get_project(project_id)

    def get_project(self, project_id: str) -> dict[str, Any]:
        project_dir = self._project_dir(project_id)
        value = self._read_record(
            project_dir / "manifest.json",
            id_field="project_id",
            expected_id=project_id,
        )
        output = dict(value)
        shots = self.list_shots(project_id)
        output["shot_count"] = len(shots)
        output["shot_ids"] = [shot["shot_id"] for shot in shots]
        return output

    def list_projects(self, limit: int = 100) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit < 0 or limit > 1000:
            raise AutoAnimError("INPUT_INVALID", "Project list limit must be in [0,1000]")
        values: list[dict[str, Any]] = []
        for path in self.root.glob("*/manifest.json"):
            try:
                values.append(self.get_project(path.parent.name))
            except FileNotFoundError:
                continue
        values.sort(key=lambda value: (value["created_at"], value["project_id"]), reverse=True)
        return values[:limit]

    def create_shot(
        self,
        project_id: str,
        *,
        name: str,
        character_id: str,
        character_revision_id: str,
        description: str | None = None,
        usage_scope: str = "production",
    ) -> dict[str, Any]:
        project = self.get_project(project_id)
        if project.get("lifecycle") != "active":
            raise AutoAnimError("LIFECYCLE_INVALID", "Shots require an active project")
        clean_name = _clean_text(name, "Shot name", maximum=120)
        clean_description = _clean_text(
            description, "Shot description", maximum=1000, required=False
        )
        if not _valid_ulid(character_id) or not _valid_ulid(character_revision_id):
            raise AutoAnimError(
                "INPUT_INVALID",
                "Shot character and revision IDs must be exact ULIDs",
            )
        try:
            revision = self.characters.resolve(
                character_id,
                character_revision_id,
                usage_scope=usage_scope,
            )
        except FileNotFoundError as exc:
            raise AutoAnimError(
                "CHARACTER_REVISION_MISMATCH",
                "Selected character revision does not exist",
            ) from exc
        if (
            revision.character_id != character_id
            or revision.revision_id != character_revision_id
        ):
            raise AutoAnimError(
                "CHARACTER_REVISION_MISMATCH",
                "Selected character revision does not belong to the character",
            )
        pin = {
            "character_id": character_id,
            "revision_id": character_revision_id,
            "revision_manifest_sha256": revision.manifest_sha256,
            "identity_sha256": revision.identity_sha256,
            "texture_sha256": revision.texture_sha256,
            "texture_uvs_array_sha256": revision.texture_uvs_array_sha256,
            "material_manifest_sha256": revision.material_manifest_sha256,
            "usage_scope": usage_scope,
        }
        shot_id = new_ulid()
        created = utc_now()
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "shot",
            "project_id": project_id,
            "shot_id": shot_id,
            "name": clean_name,
            "description": clean_description,
            "lifecycle": "setup",
            "character_revision": pin,
            "created_at": created,
            "updated_at": created,
        }
        project_dir = self._project_dir(project_id)
        with self._lock(project_dir):
            destination = self._shot_dir(project_id, shot_id)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{shot_id}.",
                    suffix=".tmp",
                    dir=project_dir / "shots",
                )
            )
            try:
                (temporary / "takes").mkdir()
                (temporary / "versions").mkdir()
                self._write_record(temporary / "manifest.json", record)
                _fsync_directory(temporary)
                os.replace(temporary, destination)
                _fsync_directory(project_dir / "shots")
            except Exception:
                if temporary.exists():
                    import shutil

                    shutil.rmtree(temporary, ignore_errors=True)
                raise
        return self.get_shot(project_id, shot_id)

    def _take_records(self, project_id: str, shot_id: str) -> list[dict[str, Any]]:
        directory = self._shot_dir(project_id, shot_id) / "takes"
        output: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                output.append(
                    self._read_record(
                        path,
                        id_field="take_id",
                        expected_id=path.stem,
                    )
                )
            except FileNotFoundError:
                continue
        output.sort(key=lambda value: (value["created_at"], value["take_id"]))
        return output

    def _version_records(self, project_id: str, shot_id: str) -> list[dict[str, Any]]:
        directory = self._shot_dir(project_id, shot_id) / "versions"
        output: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                output.append(
                    self._read_record(
                        path,
                        id_field="shot_version_id",
                        expected_id=path.stem,
                    )
                )
            except FileNotFoundError:
                continue
        output.sort(
            key=lambda value: (value["created_at"], value["shot_version_id"])
        )
        return output

    def get_shot(self, project_id: str, shot_id: str) -> dict[str, Any]:
        project_dir = self._project_dir(project_id)
        self._read_record(
            project_dir / "manifest.json",
            id_field="project_id",
            expected_id=project_id,
        )
        shot_dir = self._shot_dir(project_id, shot_id)
        value = self._read_record(
            shot_dir / "manifest.json",
            id_field="shot_id",
            expected_id=shot_id,
        )
        if value.get("project_id") != project_id:
            raise FileNotFoundError(shot_id)
        takes = self._take_records(project_id, shot_id)
        versions = self._version_records(project_id, shot_id)
        lifecycle = "performance" if versions else ("ready" if takes else "setup")
        output = dict(value)
        output.update(
            {
                "lifecycle": lifecycle,
                "take_count": len(takes),
                "take_ids": [take["take_id"] for take in takes],
                "version_count": len(versions),
                "version_ids": [
                    version["shot_version_id"] for version in versions
                ],
                "latest_version_id": (
                    versions[-1]["shot_version_id"] if versions else None
                ),
            }
        )
        return output

    def list_shots(
        self, project_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit < 0 or limit > 1000:
            raise AutoAnimError("INPUT_INVALID", "Shot list limit must be in [0,1000]")
        project_dir = self._project_dir(project_id)
        self._read_record(
            project_dir / "manifest.json",
            id_field="project_id",
            expected_id=project_id,
        )
        output: list[dict[str, Any]] = []
        for path in (project_dir / "shots").glob("*/manifest.json"):
            try:
                output.append(self.get_shot(project_id, path.parent.name))
            except FileNotFoundError:
                continue
        output.sort(key=lambda value: (value["created_at"], value["shot_id"]), reverse=True)
        return output[:limit]

    def create_take(
        self,
        project_id: str,
        shot_id: str,
        *,
        name: str,
        media_kind: str,
        source_name: str,
        source_sha256: str,
        source_bytes: int,
        source_media_type: str | None = None,
    ) -> dict[str, Any]:
        shot = self.get_shot(project_id, shot_id)
        if shot["lifecycle"] not in {"setup", "ready", "performance"}:
            raise AutoAnimError("LIFECYCLE_INVALID", "Shot cannot accept a take")
        clean_name = _clean_text(name, "Take name", maximum=120)
        kind = str(media_kind).strip().lower()
        if kind not in _MEDIA_KINDS:
            raise AutoAnimError("INPUT_INVALID", "Take media kind must be audio or video")
        clean_source_name = _clean_text(source_name, "Source name", maximum=200)
        digest = _clean_sha256(source_sha256, "Source media digest")
        if (
            not isinstance(source_bytes, int)
            or isinstance(source_bytes, bool)
            or source_bytes < 0
        ):
            raise AutoAnimError("INPUT_INVALID", "Source byte count must be non-negative")
        media_type = _clean_text(
            source_media_type,
            "Source media type",
            maximum=160,
            required=False,
        )
        take_id = new_ulid()
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "take",
            "project_id": project_id,
            "shot_id": shot_id,
            "take_id": take_id,
            "name": clean_name,
            "media_kind": kind,
            "source": {
                "name": clean_source_name,
                "sha256": digest,
                "bytes": source_bytes,
                "media_type": media_type,
            },
            "character_revision_manifest_sha256": shot["character_revision"][
                "revision_manifest_sha256"
            ],
            "lifecycle": "ready",
            "created_at": utc_now(),
        }
        shot_dir = self._shot_dir(project_id, shot_id)
        with self._lock(shot_dir):
            path = self._record_path(shot_dir / "takes", take_id)
            self._write_record(path, record)
            _fsync_directory(path.parent)
        return self.get_take(project_id, shot_id, take_id)

    def get_take(
        self, project_id: str, shot_id: str, take_id: str
    ) -> dict[str, Any]:
        shot_dir = self._shot_dir(project_id, shot_id)
        self.get_shot(project_id, shot_id)
        path = self._record_path(shot_dir / "takes", take_id)
        value = self._read_record(path, id_field="take_id", expected_id=take_id)
        if value.get("project_id") != project_id or value.get("shot_id") != shot_id:
            raise FileNotFoundError(take_id)
        linked = [
            version["shot_version_id"]
            for version in self._version_records(project_id, shot_id)
            if version.get("take_id") == take_id
        ]
        output = dict(value)
        output["lifecycle"] = "linked" if linked else "ready"
        output["shot_version_ids"] = linked
        return output

    def link_job(
        self,
        project_id: str,
        shot_id: str,
        take_id: str,
        *,
        job_id: str,
    ) -> dict[str, Any]:
        shot_dir = self._shot_dir(project_id, shot_id)
        with self._lock(shot_dir):
            shot = self.get_shot(project_id, shot_id)
            take = self.get_take(project_id, shot_id, take_id)
            try:
                job = self.jobs.require_sealed(job_id)
            except FileNotFoundError as exc:
                raise AutoAnimError("JOB_NOT_FOUND", "Performance job was not found") from exc
            if job.get("status") != "succeeded":
                raise AutoAnimError(
                    "LIFECYCLE_INVALID",
                    "Only completed successful jobs can become shot versions",
                )
            expected_kind = _JOB_KIND_BY_MEDIA[take["media_kind"]]
            if job.get("kind") != expected_kind:
                raise AutoAnimError(
                    "JOB_INCOMPATIBLE",
                    f"{take['media_kind'].title()} take requires a {expected_kind} job",
                )
            job_input = job.get("input") if isinstance(job.get("input"), dict) else {}
            if job_input.get("sha256") != take["source"]["sha256"]:
                raise AutoAnimError(
                    "JOB_INCOMPATIBLE",
                    "Performance job input does not match the take's sealed source media",
                )
            configuration = (
                job.get("configuration")
                if isinstance(job.get("configuration"), dict)
                else {}
            )
            pin = shot["character_revision"]
            try:
                resolved_revision = self.characters.resolve(
                    pin["character_id"],
                    pin["revision_id"],
                    usage_scope=pin["usage_scope"],
                )
            except FileNotFoundError as exc:
                raise AutoAnimError(
                    "CHARACTER_REVISION_MISMATCH",
                    "The shot's pinned character revision is no longer available",
                ) from exc
            if (
                resolved_revision.manifest_sha256 != pin["revision_manifest_sha256"]
                or resolved_revision.identity_sha256 != pin["identity_sha256"]
                or resolved_revision.material_manifest_sha256
                != pin["material_manifest_sha256"]
            ):
                raise AutoAnimError(
                    "CHARACTER_REVISION_MISMATCH",
                    "The shot's pinned character revision no longer matches its sealed hashes",
                )
            if (
                configuration.get("character_id") != pin["character_id"]
                or configuration.get("character_revision_id") != pin["revision_id"]
            ):
                raise AutoAnimError(
                    "CHARACTER_REVISION_MISMATCH",
                    "Performance job does not use the shot's pinned character revision",
                )
            existing = self._version_records(project_id, shot_id)
            if any(version.get("job_id") == job_id for version in existing):
                raise AutoAnimError(
                    "LIFECYCLE_INVALID",
                    "Performance job is already linked to this shot",
                )
            job_manifest = self.jobs.job_dir(job_id) / "result.json"
            version_id = new_ulid()
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "shot_version",
                "project_id": project_id,
                "shot_id": shot_id,
                "take_id": take_id,
                "shot_version_id": version_id,
                "ordinal": len(existing) + 1,
                "parent_shot_version_id": (
                    existing[-1]["shot_version_id"] if existing else None
                ),
                "job_id": job_id,
                "job_kind": expected_kind,
                "job_manifest_sha256": sha256(job_manifest),
                "character_revision": dict(pin),
                "lifecycle": "linked",
                "created_at": utc_now(),
            }
            path = self._record_path(shot_dir / "versions", version_id)
            self._write_record(path, record)
            _fsync_directory(path.parent)
        return self.get_shot_version(project_id, shot_id, version_id)

    def get_shot_version(
        self, project_id: str, shot_id: str, shot_version_id: str
    ) -> dict[str, Any]:
        shot_dir = self._shot_dir(project_id, shot_id)
        self.get_shot(project_id, shot_id)
        path = self._record_path(
            shot_dir / "versions",
            shot_version_id,
        )
        value = self._read_record(
            path,
            id_field="shot_version_id",
            expected_id=shot_version_id,
        )
        if value.get("project_id") != project_id or value.get("shot_id") != shot_id:
            raise FileNotFoundError(shot_version_id)
        return value


__all__ = ["ProductionStore", "SCHEMA_VERSION"]
