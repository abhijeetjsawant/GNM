from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoanim_gnm.artifacts import JobStore, new_ulid
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.production import ProductionStore


class _Characters:
    def __init__(self, jobs: JobStore):
        self.jobs = jobs
        self.character_id = new_ulid()
        self.revision_id = new_ulid()
        self.revision_manifest_sha256 = "1" * 64
        self.resolve_calls: list[tuple[str, str | None, str]] = []
        self.reuse_allowed = True

    def resolve(
        self,
        character_id: str,
        revision_id: str | None = None,
        *,
        usage_scope: str = "personal",
    ) -> SimpleNamespace:
        self.resolve_calls.append((character_id, revision_id, usage_scope))
        if not self.reuse_allowed:
            raise AutoAnimError("CONSENT_REVOKED", "Character reuse has been revoked")
        if character_id != self.character_id or revision_id != self.revision_id:
            raise FileNotFoundError(revision_id or character_id)
        return SimpleNamespace(
            character_id=character_id,
            revision_id=revision_id,
            manifest_sha256=self.revision_manifest_sha256,
            identity_sha256="2" * 64,
            texture_sha256="3" * 64,
            texture_uvs_array_sha256="4" * 64,
            material_manifest_sha256="5" * 64,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[ProductionStore, _Characters]:
    jobs = JobStore(tmp_path / "jobs")
    characters = _Characters(jobs)
    return ProductionStore(tmp_path / "production", characters), characters


def _project_and_shot(
    store: ProductionStore,
    characters: _Characters,
) -> tuple[dict, dict]:
    project = store.create_project("Film", "A durable local production")
    shot = store.create_shot(
        project["project_id"],
        name="Close up",
        description="Opening line",
        character_id=characters.character_id,
        character_revision_id=characters.revision_id,
    )
    return project, shot


def _take(
    store: ProductionStore,
    project: dict,
    shot: dict,
    *,
    media_kind: str = "audio",
    source_sha256: str = "a" * 64,
) -> dict:
    return store.create_take(
        project["project_id"],
        shot["shot_id"],
        name="Performance 1",
        media_kind=media_kind,
        source_name=f"take.{ 'wav' if media_kind == 'audio' else 'mov' }",
        source_sha256=source_sha256,
        source_bytes=1024,
        source_media_type="audio/wav" if media_kind == "audio" else "video/quicktime",
    )


def _finished_job(
    jobs: JobStore,
    tmp_path: Path,
    *,
    kind: str,
    character_id: str,
    revision_id: str,
) -> tuple[str, str]:
    source = tmp_path / f"{new_ulid()}.bin"
    source.write_bytes(b"source")
    job_id, job_dir, _, manifest = jobs.start(
        kind,
        source,
        {
            "character_id": character_id,
            "character_revision_id": revision_id,
        },
    )
    jobs.finish(
        manifest,
        job_dir,
        {"kind": kind, "warnings": [], "artifacts": {}},
        {},
    )
    return job_id, jobs.read(job_id)["input"]["sha256"]


def test_project_and_shot_survive_restart_and_list(
    workspace: tuple[ProductionStore, _Characters],
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)

    restarted = ProductionStore(store.root, characters)
    assert restarted.get_project(project["project_id"])["shot_ids"] == [
        shot["shot_id"]
    ]
    assert [value["project_id"] for value in restarted.list_projects()] == [
        project["project_id"]
    ]
    assert [
        value["shot_id"] for value in restarted.list_shots(project["project_id"])
    ] == [shot["shot_id"]]


def test_shot_pins_exact_revision_when_character_current_changes(
    workspace: tuple[ProductionStore, _Characters],
) -> None:
    store, characters = workspace
    _, shot = _project_and_shot(store, characters)
    original_pin = dict(shot["character_revision"])

    characters.revision_id = new_ulid()
    characters.revision_manifest_sha256 = "9" * 64

    persisted = store.get_shot(shot["project_id"], shot["shot_id"])
    assert persisted["character_revision"] == original_pin
    assert characters.resolve_calls == [
        (
            original_pin["character_id"],
            original_pin["revision_id"],
            "production",
        )
    ]


def test_shot_rejects_unknown_or_mismatched_revision(
    workspace: tuple[ProductionStore, _Characters],
) -> None:
    store, characters = workspace
    project = store.create_project("Film")

    with pytest.raises(AutoAnimError, match="does not exist") as failure:
        store.create_shot(
            project["project_id"],
            name="Bad pin",
            character_id=characters.character_id,
            character_revision_id=new_ulid(),
        )
    assert failure.value.code == "CHARACTER_REVISION_MISMATCH"
    assert store.list_shots(project["project_id"]) == []


def test_take_advances_shot_to_ready_and_validates_source(
    workspace: tuple[ProductionStore, _Characters],
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)
    assert shot["lifecycle"] == "setup"

    take = _take(store, project, shot)
    assert take["lifecycle"] == "ready"
    ready = store.get_shot(project["project_id"], shot["shot_id"])
    assert ready["lifecycle"] == "ready"
    assert ready["take_ids"] == [take["take_id"]]

    with pytest.raises(AutoAnimError) as failure:
        store.create_take(
            project["project_id"],
            shot["shot_id"],
            name="Bad",
            media_kind="image",
            source_name="portrait.png",
            source_sha256="not-a-digest",
            source_bytes=-1,
        )
    assert failure.value.code == "INPUT_INVALID"


def test_compatible_completed_job_creates_immutable_shot_version(
    workspace: tuple[ProductionStore, _Characters],
    tmp_path: Path,
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)
    job_id, source_sha256 = _finished_job(
        characters.jobs,
        tmp_path,
        kind="audio_animation",
        character_id=characters.character_id,
        revision_id=characters.revision_id,
    )
    take = _take(store, project, shot, source_sha256=source_sha256)

    version = store.link_job(
        project["project_id"],
        shot["shot_id"],
        take["take_id"],
        job_id=job_id,
    )
    assert version["ordinal"] == 1
    assert version["job_id"] == job_id
    assert version["character_revision"] == shot["character_revision"]
    assert len(version["job_manifest_sha256"]) == 64
    assert store.get_shot(project["project_id"], shot["shot_id"])[
        "lifecycle"
    ] == "performance"
    assert store.get_take(project["project_id"], shot["shot_id"], take["take_id"])[
        "lifecycle"
    ] == "linked"

    restarted = ProductionStore(store.root, characters)
    assert restarted.get_shot_version(
        project["project_id"],
        shot["shot_id"],
        version["shot_version_id"],
    ) == version


def test_link_rejects_wrong_media_revision_and_duplicate_job(
    workspace: tuple[ProductionStore, _Characters],
    tmp_path: Path,
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)
    take = _take(store, project, shot)

    video_job, video_digest = _finished_job(
        characters.jobs,
        tmp_path,
        kind="video_performance",
        character_id=characters.character_id,
        revision_id=characters.revision_id,
    )
    take = _take(store, project, shot, source_sha256=video_digest)
    with pytest.raises(AutoAnimError) as incompatible:
        store.link_job(
            project["project_id"],
            shot["shot_id"],
            take["take_id"],
            job_id=video_job,
        )
    assert incompatible.value.code == "JOB_INCOMPATIBLE"

    wrong_revision_job, wrong_revision_digest = _finished_job(
        characters.jobs,
        tmp_path,
        kind="audio_animation",
        character_id=characters.character_id,
        revision_id=new_ulid(),
    )
    wrong_revision_take = _take(
        store, project, shot, source_sha256=wrong_revision_digest
    )
    with pytest.raises(AutoAnimError) as mismatch:
        store.link_job(
            project["project_id"],
            shot["shot_id"],
            wrong_revision_take["take_id"],
            job_id=wrong_revision_job,
        )
    assert mismatch.value.code == "CHARACTER_REVISION_MISMATCH"

    valid_job, valid_digest = _finished_job(
        characters.jobs,
        tmp_path,
        kind="audio_animation",
        character_id=characters.character_id,
        revision_id=characters.revision_id,
    )
    # A take's source provenance cannot be rewritten. Create a fresh take for
    # the separate valid source rather than treating it as the failed video take.
    valid_take = _take(store, project, shot, source_sha256=valid_digest)
    store.link_job(
        project["project_id"],
        shot["shot_id"],
        valid_take["take_id"],
        job_id=valid_job,
    )
    with pytest.raises(AutoAnimError) as duplicate:
        store.link_job(
            project["project_id"],
            shot["shot_id"],
            valid_take["take_id"],
            job_id=valid_job,
        )
    assert duplicate.value.code == "LIFECYCLE_INVALID"


def test_link_rejects_a_job_from_different_source_media(
    workspace: tuple[ProductionStore, _Characters],
    tmp_path: Path,
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)
    job_id, _ = _finished_job(
        characters.jobs,
        tmp_path,
        kind="audio_animation",
        character_id=characters.character_id,
        revision_id=characters.revision_id,
    )
    take = _take(store, project, shot, source_sha256="b" * 64)

    with pytest.raises(AutoAnimError) as mismatch:
        store.link_job(
            project["project_id"],
            shot["shot_id"],
            take["take_id"],
            job_id=job_id,
        )

    assert mismatch.value.code == "JOB_INCOMPATIBLE"


def test_link_rechecks_pinned_character_consent(
    workspace: tuple[ProductionStore, _Characters],
    tmp_path: Path,
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)
    job_id, source_sha256 = _finished_job(
        characters.jobs,
        tmp_path,
        kind="audio_animation",
        character_id=characters.character_id,
        revision_id=characters.revision_id,
    )
    take = _take(store, project, shot, source_sha256=source_sha256)
    characters.reuse_allowed = False

    with pytest.raises(AutoAnimError) as blocked:
        store.link_job(
            project["project_id"],
            shot["shot_id"],
            take["take_id"],
            job_id=job_id,
        )

    assert blocked.value.code == "CONSENT_REVOKED"


def test_tampered_shot_record_fails_closed(
    workspace: tuple[ProductionStore, _Characters],
) -> None:
    store, characters = workspace
    project, shot = _project_and_shot(store, characters)
    path = (
        store.root
        / project["project_id"]
        / "shots"
        / shot["shot_id"]
        / "manifest.json"
    )
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(AutoAnimError) as failure:
        store.get_shot(project["project_id"], shot["shot_id"])
    assert failure.value.code == "INTEGRITY_FAILED"
