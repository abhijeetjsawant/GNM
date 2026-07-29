from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.artifacts import new_ulid
from autoanim_gnm.errors import AutoAnimError


class _RecordingProductionStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def list_projects(self, *, limit: int) -> list[dict[str, Any]]:
        self.calls.append(("list_projects", (), {"limit": limit}))
        return [{"project_id": "project-1", "name": "Film"}]

    def create_project(
        self, name: str, *, description: str | None
    ) -> dict[str, Any]:
        self.calls.append(
            ("create_project", (name,), {"description": description})
        )
        return {
            "project_id": "project-1",
            "name": name,
            "description": description,
        }

    def get_project(self, project_id: str) -> dict[str, Any]:
        self.calls.append(("get_project", (project_id,), {}))
        if project_id == "missing":
            raise FileNotFoundError(project_id)
        return {"project_id": project_id, "name": "Film"}

    def list_shots(
        self, project_id: str, *, limit: int
    ) -> list[dict[str, Any]]:
        self.calls.append(
            ("list_shots", (project_id,), {"limit": limit})
        )
        if project_id == "missing":
            raise FileNotFoundError(project_id)
        return [
            {
                "project_id": project_id,
                "shot_id": "shot-1",
                "name": "Close-up",
            }
        ]

    def create_shot(
        self,
        project_id: str,
        *,
        name: str,
        character_id: str,
        character_revision_id: str,
        description: str | None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "create_shot",
                (project_id,),
                {
                    "name": name,
                    "character_id": character_id,
                    "character_revision_id": character_revision_id,
                    "description": description,
                },
            )
        )
        if character_revision_id == "wrong-revision":
            raise AutoAnimError(
                "CHARACTER_REVISION_MISMATCH",
                "Character revision does not belong to the selected character",
            )
        return {
            "project_id": project_id,
            "shot_id": "shot-1",
            "name": name,
            "character_id": character_id,
            "character_revision_id": character_revision_id,
            "description": description,
        }

    def get_shot(self, project_id: str, shot_id: str) -> dict[str, Any]:
        self.calls.append(("get_shot", (project_id, shot_id), {}))
        if shot_id == "missing":
            raise FileNotFoundError(shot_id)
        return {
            "project_id": project_id,
            "shot_id": shot_id,
            "name": "Close-up",
        }

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
        source_media_type: str | None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "create_take",
                (project_id, shot_id),
                {
                    "name": name,
                    "media_kind": media_kind,
                    "source_name": source_name,
                    "source_sha256": source_sha256,
                    "source_bytes": source_bytes,
                    "source_media_type": source_media_type,
                },
            )
        )
        return {
            "project_id": project_id,
            "shot_id": shot_id,
            "take_id": "take-1",
            "name": name,
            "media_kind": media_kind,
        }

    def link_job(
        self,
        project_id: str,
        shot_id: str,
        take_id: str,
        *,
        job_id: str,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "link_job",
                (project_id, shot_id, take_id),
                {"job_id": job_id},
            )
        )
        if take_id == "missing":
            raise FileNotFoundError(take_id)
        return {
            "project_id": project_id,
            "shot_id": shot_id,
            "take_id": take_id,
            "job_id": job_id,
            "version_id": "version-1",
        }


def _client(tmp_path: Path) -> tuple[TestClient, _RecordingProductionStore]:
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        production_root=tmp_path / "production",
    )
    production = _RecordingProductionStore()
    app.state.production = production
    return TestClient(app), production


def test_project_and_shot_json_routes_are_native_friendly(tmp_path: Path) -> None:
    client, production = _client(tmp_path)

    created_project = client.post(
        "/api/projects",
        json={"name": "  Film  ", "description": "Hero sequence"},
    )
    assert created_project.status_code == 201
    assert created_project.json() == {
        "project_id": "project-1",
        "name": "Film",
        "description": "Hero sequence",
    }
    assert client.get("/api/projects?limit=500").json() == {
        "projects": [{"project_id": "project-1", "name": "Film"}]
    }
    assert client.get("/api/projects/project-1").json()["project_id"] == "project-1"

    created_shot = client.post(
        "/api/projects/project-1/shots",
        json={
            "name": "Close-up",
            "character_id": "character-1",
            "character_revision_id": "revision-1",
            "description": "Opening line",
        },
    )
    assert created_shot.status_code == 201
    assert created_shot.json()["character_revision_id"] == "revision-1"
    assert client.get("/api/projects/project-1/shots?limit=0").json() == {
        "shots": [
            {
                "project_id": "project-1",
                "shot_id": "shot-1",
                "name": "Close-up",
            }
        ]
    }
    shot = client.get("/api/projects/project-1/shots/shot-1")
    assert shot.status_code == 200
    assert shot.json()["shot_id"] == "shot-1"

    assert (
        "list_projects",
        (),
        {"limit": 200},
    ) in production.calls
    assert (
        "list_shots",
        ("project-1",),
        {"limit": 1},
    ) in production.calls


def test_take_and_job_link_routes_transport_exact_provenance(
    tmp_path: Path,
) -> None:
    client, production = _client(tmp_path)
    digest = "a" * 64

    take = client.post(
        "/api/projects/project-1/shots/shot-1/takes",
        json={
            "name": "Take 03",
            "media_kind": "video",
            "source_name": "performance.mov",
            "source_sha256": digest,
            "source_bytes": 12_345,
            "source_media_type": "video/quicktime",
        },
    )
    assert take.status_code == 201
    assert take.json()["take_id"] == "take-1"
    linked = client.post(
        "/api/projects/project-1/shots/shot-1/takes/take-1/jobs",
        json={"job_id": "job-1"},
    )
    assert linked.status_code == 201
    assert linked.json() == {
        "project_id": "project-1",
        "shot_id": "shot-1",
        "take_id": "take-1",
        "job_id": "job-1",
        "version_id": "version-1",
    }
    assert (
        "create_take",
        ("project-1", "shot-1"),
        {
            "name": "Take 03",
            "media_kind": "video",
            "source_name": "performance.mov",
            "source_sha256": digest,
            "source_bytes": 12_345,
            "source_media_type": "video/quicktime",
        },
    ) in production.calls
    assert (
        "link_job",
        ("project-1", "shot-1", "take-1"),
        {"job_id": "job-1"},
    ) in production.calls


def test_production_routes_fail_closed_for_missing_or_invalid_records(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path)

    missing_project = client.get("/api/projects/missing")
    assert missing_project.status_code == 404
    assert missing_project.json()["code"] == "PROJECT_NOT_FOUND"
    missing_shots = client.get("/api/projects/missing/shots")
    assert missing_shots.status_code == 404
    assert missing_shots.json()["code"] == "PROJECT_NOT_FOUND"
    missing_shot = client.get("/api/projects/project-1/shots/missing")
    assert missing_shot.status_code == 404
    assert missing_shot.json()["code"] == "SHOT_NOT_FOUND"
    project_missing_for_shot = client.get(
        "/api/projects/missing/shots/shot-1"
    )
    assert project_missing_for_shot.status_code == 404
    assert project_missing_for_shot.json()["code"] == "PROJECT_NOT_FOUND"
    missing_take = client.post(
        "/api/projects/project-1/shots/shot-1/takes/missing/jobs",
        json={"job_id": "job-1"},
    )
    assert missing_take.status_code == 404
    assert missing_take.json()["code"] == "TAKE_NOT_FOUND"
    shot_missing_for_link = client.post(
        "/api/projects/project-1/shots/missing/takes/take-1/jobs",
        json={"job_id": "job-1"},
    )
    assert shot_missing_for_link.status_code == 404
    assert shot_missing_for_link.json()["code"] == "SHOT_NOT_FOUND"
    project_missing_for_link = client.post(
        "/api/projects/missing/shots/shot-1/takes/take-1/jobs",
        json={"job_id": "job-1"},
    )
    assert project_missing_for_link.status_code == 404
    assert project_missing_for_link.json()["code"] == "PROJECT_NOT_FOUND"

    mismatch = client.post(
        "/api/projects/project-1/shots",
        json={
            "name": "Close-up",
            "character_id": "character-1",
            "character_revision_id": "wrong-revision",
        },
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["code"] == "CHARACTER_REVISION_MISMATCH"

    bad_hash = client.post(
        "/api/projects/project-1/shots/shot-1/takes",
        json={
            "name": "Take",
            "media_kind": "video",
            "source_name": "source.mov",
            "source_sha256": "not-a-sha",
            "source_bytes": 12,
        },
    )
    assert bad_hash.status_code == 422
    extra_field = client.post(
        "/api/projects",
        json={"name": "Film", "unexpected": True},
    )
    assert extra_field.status_code == 422


def test_production_routes_report_missing_store_dependency(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    client.app.state.production = None

    response = client.get("/api/projects")

    assert response.status_code == 424
    assert response.json()["code"] == "DEPENDENCY_MISSING"


def test_routes_wire_real_production_store_and_sealed_job(
    tmp_path: Path,
) -> None:
    app = create_app(
        tmp_path / "jobs",
        model_path=tmp_path / "missing.task",
        production_root=tmp_path / "production",
    )
    client = TestClient(app)
    character_id = new_ulid()
    revision_id = new_ulid()

    class _Characters:
        def resolve(
            self,
            selected_character_id: str,
            selected_revision_id: str | None = None,
            *,
            usage_scope: str = "personal",
        ) -> SimpleNamespace:
            assert usage_scope == "production"
            if (
                selected_character_id != character_id
                or selected_revision_id != revision_id
            ):
                raise FileNotFoundError(selected_revision_id)
            return SimpleNamespace(
                character_id=character_id,
                revision_id=revision_id,
                manifest_sha256="1" * 64,
                identity_sha256="2" * 64,
                texture_sha256="3" * 64,
                texture_uvs_array_sha256="4" * 64,
                material_manifest_sha256="5" * 64,
            )

    app.state.production.characters = _Characters()
    project_response = client.post(
        "/api/projects",
        json={"name": "Film", "description": "Persistent workspace"},
    )
    assert project_response.status_code == 201, project_response.text
    project_id = project_response.json()["project_id"]
    shot_response = client.post(
        f"/api/projects/{project_id}/shots",
        json={
            "name": "Close-up",
            "character_id": character_id,
            "character_revision_id": revision_id,
        },
    )
    assert shot_response.status_code == 201, shot_response.text
    shot_id = shot_response.json()["shot_id"]

    source = tmp_path / "voice.wav"
    source.write_bytes(b"voice")
    job_id, job_dir, _, manifest = app.state.service.store.start(
        "audio_animation",
        source,
        {
            "character_id": character_id,
            "character_revision_id": revision_id,
        },
    )
    app.state.service.store.finish(
        manifest,
        job_dir,
        {
            "kind": "audio_animation",
            "warnings": [],
            "artifacts": {},
        },
        {},
    )
    job_input = app.state.service.store.read(job_id)["input"]
    take_response = client.post(
        f"/api/projects/{project_id}/shots/{shot_id}/takes",
        json={
            "name": "Take 01",
            "media_kind": "audio",
            "source_name": "voice.wav",
            "source_sha256": job_input["sha256"],
            "source_bytes": job_input["bytes"],
            "source_media_type": "audio/wav",
        },
    )
    assert take_response.status_code == 201, take_response.text
    take_id = take_response.json()["take_id"]
    linked = client.post(
        (
            f"/api/projects/{project_id}/shots/{shot_id}/takes/"
            f"{take_id}/jobs"
        ),
        json={"job_id": job_id},
    )
    assert linked.status_code == 201, linked.text
    assert linked.json()["job_id"] == job_id
    assert linked.json()["ordinal"] == 1

    listed = client.get("/api/projects").json()["projects"]
    assert [project["project_id"] for project in listed] == [project_id]
    persisted = client.get(f"/api/projects/{project_id}/shots/{shot_id}")
    assert persisted.status_code == 200
    assert persisted.json()["latest_version_id"] == linked.json()["shot_version_id"]
