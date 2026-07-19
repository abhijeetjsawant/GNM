from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.animation import calibrate_lip_contact
from autoanim_gnm.artifacts import JobStore, sha256
from autoanim_gnm.characters import CharacterStore
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.semantic_decoder import ExpressionDecoder
from autoanim_gnm.serialization import write_npz
from autoanim_gnm.service import ApplicationService


def _consent() -> dict[str, str]:
    return {
        "consent_subject": "Test Performer",
        "consent_attester": "Test Producer",
        "consent_scope": "production",
        "consent_evidence_ref": "release-test-001",
        "consent_evidence_sha256": "a" * 64,
    }


def _reanchor_revision(characters: CharacterStore, character_id: str) -> Path:
    top_path = characters.root / character_id / "manifest.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    revision_id = top["current_revision_id"]
    revision_manifest = (
        characters.root / character_id / "revisions" / revision_id / "manifest.json"
    )
    digest = sha256(revision_manifest)
    top["current_revision_manifest_sha256"] = digest
    top["revisions"][revision_id]["manifest_sha256"] = digest
    top_path.write_text(
        json.dumps(characters.jobs.signer.sign(top)), encoding="utf-8"
    )
    return revision_manifest.parent


def _successful_identity_job(
    store: JobStore,
    root: Path,
    *,
    textured: bool = True,
) -> tuple[str, np.ndarray]:
    source = root / "private-source.png"
    Image.new("RGB", (8, 8), (90, 70, 55)).save(source)
    job_id, job_dir, _, manifest = store.start(
        "multiview_reconstruction" if textured else "image_fit",
        source,
        {},
    )
    identity = np.linspace(-0.35, 0.35, 253, dtype=np.float32)
    write_npz(
        job_dir / "fit.npz",
        **({"fitted_identity": identity} if textured else {"identity": identity}),
    )
    (job_dir / "fitted.glb").write_bytes(b"glTF-preview")
    artifacts = {
        "fit" if textured else "parameters": "fit.npz",
        "textured_glb" if textured else "glb": "fitted.glb",
    }
    pipeline: dict = {
        "kind": "multiview_reconstruction" if textured else "image_fit",
        "fit": {"production_validated": False},
        "warnings": [],
        "artifacts": artifacts,
    }
    if textured:
        Image.new("RGB", (16, 8), (120, 85, 65)).save(job_dir / "texture.png")
        packed_uvs = np.full((35_324, 3, 2), 0.25, dtype=np.float32)
        write_npz(job_dir / "texture-maps.npz", triangle_uvs=packed_uvs)
        artifacts["texture"] = "texture.png"
        artifacts["texture_maps"] = "texture-maps.npz"
        pipeline["texture"] = {"observed_fraction": 0.72}
    store.finish(manifest, job_dir, pipeline, {"gnm": "3.0"})
    return job_id, identity


def test_character_promotion_is_immutable_consent_audited_and_resolvable(
    tmp_path: Path,
) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, expected_identity = _successful_identity_job(jobs, tmp_path)
    characters = CharacterStore(tmp_path / "characters", jobs)

    with pytest.raises(AutoAnimError) as missing_consent:
        characters.promote(
            job_id, name="Hero", consent_attested=False, **_consent()
        )
    assert missing_consent.value.code == "CONSENT_REQUIRED"

    manifest = characters.promote(
        job_id,
        name="  Hero   One  ",
        consent_attested=True,
        consent_note="release-42",
        **_consent(),
    )
    assert manifest["name"] == "Hero One"
    assert manifest["consent_attested"] is True
    assert manifest["appearance_status"] == "rgb_atlas_unvalidated"
    revision = characters.resolve(manifest["character_id"])
    np.testing.assert_array_equal(revision.identity, expected_identity)
    assert revision.texture_path is not None
    assert revision.texture_path.name == "base-color.png"
    assert revision.triangle_uvs is not None
    np.testing.assert_array_equal(
        revision.triangle_uvs,
        np.full((35_324, 3, 2), 0.25, dtype=np.float32),
    )
    assert revision.texture_uvs_sha256 is not None
    assert revision.preview_path.read_bytes() == b"glTF-preview"
    assert revision.manifest["consent"]["note"] == "release-42"
    assert revision.manifest["appearance"]["production_validated"] is False
    assert revision.manifest["oral"]["tongue_visibility_validated"] is False
    assert revision.manifest["body"]["status"] == "not_attached"
    material = json.loads(
        (revision.preview_path.parent / "material.json").read_text(encoding="utf-8")
    )
    assert material["resolution"] == [16, 8]
    assert material["maps"]["normal"] is None
    assert material["pore_detail_validated"] is False
    assert not any(path.name.startswith("input") for path in revision.preview_path.parent.iterdir())
    assert [item["character_id"] for item in characters.list()] == [manifest["character_id"]]


def test_character_integrity_failure_is_fail_closed(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    created = characters.promote(
        job_id, name="Geometry", consent_attested=True, **_consent()
    )
    revision = characters.resolve(created["character_id"])
    (revision.preview_path.parent / "identity.npz").write_bytes(b"tampered")
    with pytest.raises(AutoAnimError, match="integrity"):
        characters.resolve(created["character_id"])


def test_character_promotion_rejects_tampered_source_job_artifact(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    (jobs.job_dir(job_id) / "fit.npz").write_bytes(b"replaced after completion")

    with pytest.raises(AutoAnimError, match="integrity lookup"):
        characters.promote(
            job_id,
            name="Tampered",
            consent_attested=True,
            **_consent(),
        )


def test_character_promotion_rejects_rewritten_source_manifest_root(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    artifact = jobs.job_dir(job_id) / "fit.npz"
    replacement = np.full(253, 0.25, dtype=np.float32)
    write_npz(artifact, identity=replacement)
    result_path = jobs.job_dir(job_id) / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["artifacts"]["parameters"]["bytes"] = artifact.stat().st_size
    result["artifacts"]["parameters"]["sha256"] = sha256(artifact)
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(AutoAnimError) as error:
        characters.promote(
            job_id,
            name="Rewritten",
            consent_attested=True,
            **_consent(),
        )
    assert error.value.code == "INTEGRITY_FAILED"


def test_legacy_job_requires_explicit_verified_seal(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    result_path = jobs.job_dir(job_id) / "result.json"
    legacy = json.loads(result_path.read_text(encoding="utf-8"))
    legacy.pop("integrity")
    result_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(AutoAnimError) as unsealed:
        jobs.require_sealed(job_id)
    assert unsealed.value.code == "INTEGRITY_UNSEALED"
    sealed = jobs.seal_legacy(
        job_id,
        attested_by="migration operator",
        reason="current files independently reviewed",
    )
    assert jobs.signer.verify(sealed)
    assert sealed["integrity_migration"][
        "preexisting_provenance_not_cryptographically_proven"
    ] is True
    jobs.require_sealed(job_id)


def test_revision_manifest_digest_is_anchored_by_character_manifest(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    created = characters.promote(
        job_id, name="Anchored", consent_attested=True, **_consent()
    )
    revision = characters.resolve(created["character_id"])
    manifest_path = revision.preview_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["production_validated"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AutoAnimError, match="manifest failed its integrity"):
        characters.resolve(created["character_id"])


def test_character_top_manifest_seal_prevents_anchor_and_revocation_rewrite(
    tmp_path: Path,
) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    created = characters.promote(
        job_id, name="Sealed", consent_attested=True, **_consent()
    )
    characters.revoke(
        created["character_id"], reason="withdrawn", revoked_by="performer"
    )
    top_path = characters.root / created["character_id"] / "manifest.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    top["consent_status"] = "active"
    top.pop("revocation", None)
    top_path.write_text(json.dumps(top), encoding="utf-8")

    with pytest.raises(AutoAnimError) as error:
        characters.read(created["character_id"])
    assert error.value.code == "INTEGRITY_FAILED"


@pytest.mark.parametrize("unsafe_name", ["../outside.txt", "/tmp/outside.txt"])
def test_character_assets_reject_manifest_path_escape(
    tmp_path: Path, unsafe_name: str
) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    created = characters.promote(
        job_id, name="Contained", consent_attested=True, **_consent()
    )
    revision = characters.resolve(created["character_id"])
    manifest_path = revision.preview_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["material"] = {
        "name": unsafe_name,
        "bytes": 0,
        "sha256": "0" * 64,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reanchor_revision(characters, created["character_id"])

    with pytest.raises(FileNotFoundError):
        characters.asset(
            created["character_id"], created["current_revision_id"], "material"
        )


def test_character_assets_reject_external_symlink(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    created = characters.promote(
        job_id, name="No Symlink", consent_attested=True, **_consent()
    )
    revision = characters.resolve(created["character_id"])
    external = tmp_path / "external.txt"
    external.write_text("private", encoding="utf-8")
    linked = revision.preview_path.parent / "linked.txt"
    linked.symlink_to(external)
    manifest_path = revision.preview_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["material"] = {
        "name": linked.name,
        "bytes": external.stat().st_size,
        "sha256": sha256(external),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reanchor_revision(characters, created["character_id"])

    with pytest.raises(FileNotFoundError):
        characters.asset(
            created["character_id"], created["current_revision_id"], "material"
        )


def test_character_revocation_blocks_reuse(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    created = characters.promote(
        job_id, name="Revocable", consent_attested=True, **_consent()
    )
    revoked = characters.revoke(
        created["character_id"], reason="performer withdrew", revoked_by="producer"
    )
    assert revoked["consent_status"] == "revoked"
    with pytest.raises(AutoAnimError) as error:
        characters.resolve(created["character_id"])
    assert error.value.code == "CONSENT_REVOKED"


def test_character_expiry_blocks_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    expiry = datetime.now(timezone.utc) + timedelta(days=1)
    created = characters.promote(
        job_id,
        name="Expiring",
        consent_attested=True,
        consent_expires_at=expiry.isoformat(),
        **_consent(),
    )

    class AfterExpiry(datetime):
        @classmethod
        def now(cls, tz=None):
            value = expiry + timedelta(days=1)
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr("autoanim_gnm.characters.datetime", AfterExpiry)
    with pytest.raises(AutoAnimError) as error:
        characters.resolve(created["character_id"])
    assert error.value.code == "CONSENT_EXPIRED"
    assert characters.list()[0]["consent_status"] == "expired"


def test_character_scope_is_enforced(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    job_id, _ = _successful_identity_job(jobs, tmp_path, textured=False)
    characters = CharacterStore(tmp_path / "characters", jobs)
    consent = _consent()
    consent["consent_scope"] = "research"
    created = characters.promote(
        job_id, name="Research Only", consent_attested=True, **consent
    )
    characters.resolve(created["character_id"], usage_scope="research")
    with pytest.raises(AutoAnimError) as error:
        characters.resolve(created["character_id"], usage_scope="production")
    assert error.value.code == "CONSENT_SCOPE_DENIED"


def test_research_character_viewer_and_asset_propagate_scope(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    job_id, _ = _successful_identity_job(app.state.service.store, tmp_path, textured=False)
    consent = _consent()
    consent["consent_scope"] = "research"
    created = app.state.service.promote_character(
        job_id,
        name="Research Only",
        consent_attested=True,
        **consent,
    )
    client = TestClient(app)
    character_id = created["character_id"]
    revision_id = created["current_revision_id"]
    denied = client.get(f"/api/characters/{character_id}/viewer")
    assert denied.status_code == 403
    assert denied.json()["code"] == "CONSENT_SCOPE_DENIED"
    viewer = client.get(
        f"/api/characters/{character_id}/viewer?usage_scope=research"
    )
    assert viewer.status_code == 200
    assert "usage_scope=research" in viewer.text
    preview = client.get(
        f"/api/characters/{character_id}/revisions/{revision_id}/files/preview"
        "?usage_scope=research"
    )
    assert preview.status_code == 200


def test_api_reports_typed_integrity_failures(tmp_path: Path) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    job_id, _ = _successful_identity_job(app.state.service.store, tmp_path, textured=False)
    client = TestClient(app)
    job_result = app.state.service.store.job_dir(job_id) / "result.json"
    value = json.loads(job_result.read_text(encoding="utf-8"))
    value["kind"] = "rewritten"
    job_result.write_text(json.dumps(value), encoding="utf-8")
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 409
    assert response.json()["code"] == "INTEGRITY_FAILED"

    clean_job_id, _ = _successful_identity_job(
        app.state.service.store, tmp_path, textured=False
    )
    created = app.state.service.promote_character(
        clean_job_id,
        name="Tamper Target",
        consent_attested=True,
        **_consent(),
    )
    top = app.state.service.characters.root / created["character_id"] / "manifest.json"
    character = json.loads(top.read_text(encoding="utf-8"))
    character["name"] = "Rewritten"
    top.write_text(json.dumps(character), encoding="utf-8")
    response = client.get(f"/api/characters/{created['character_id']}")
    assert response.status_code == 409
    assert response.json()["code"] == "INTEGRITY_FAILED"

    revision_job_id, _ = _successful_identity_job(
        app.state.service.store, tmp_path, textured=False
    )
    revision_character = app.state.service.promote_character(
        revision_job_id,
        name="Revision Tamper",
        consent_attested=True,
        **_consent(),
    )
    revision_id = revision_character["current_revision_id"]
    revision_manifest = (
        app.state.service.characters.root
        / revision_character["character_id"]
        / "revisions"
        / revision_id
        / "manifest.json"
    )
    revision_manifest.write_text("{}", encoding="utf-8")
    response = client.get(
        f"/api/characters/{revision_character['character_id']}/viewer"
    )
    assert response.status_code == 409
    assert response.json()["code"] == "INTEGRITY_FAILED"

    asset_job_id, _ = _successful_identity_job(
        app.state.service.store, tmp_path, textured=False
    )
    asset_character = app.state.service.promote_character(
        asset_job_id,
        name="Asset Tamper",
        consent_attested=True,
        **_consent(),
    )
    asset_revision_id = asset_character["current_revision_id"]
    preview_path = (
        app.state.service.characters.root
        / asset_character["character_id"]
        / "revisions"
        / asset_revision_id
        / "preview.glb"
    )
    preview_path.write_bytes(b"changed")
    response = client.get(
        f"/api/characters/{asset_character['character_id']}/revisions/"
        f"{asset_revision_id}/files/preview"
    )
    assert response.status_code == 409
    assert response.json()["code"] == "INTEGRITY_FAILED"


def test_control_rig_and_lip_contact_are_identity_specific() -> None:
    adapter = GNMAdapter()
    decoder = ExpressionDecoder(
        "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
    )
    identity = np.zeros(adapter.identity_dim, dtype=np.float32)
    identity[:40] = np.linspace(-0.35, 0.35, 40, dtype=np.float32)
    neutral_rig = ControlRig(adapter, decoder)
    character_rig = ControlRig(adapter, decoder, identity=identity)
    np.testing.assert_allclose(
        character_rig.compact_landmarks(np.zeros(adapter.expression_dim, dtype=np.float32)),
        adapter.landmarks(identity=identity),
        atol=2e-6,
    )
    assert not np.allclose(character_rig.neutral_landmarks, neutral_rig.neutral_landmarks)
    neutral_contact = calibrate_lip_contact(neutral_rig)
    character_contact = calibrate_lip_contact(character_rig)
    assert character_contact.calibration_hash != neutral_contact.calibration_hash
    assert character_contact.neutral_gap_interocular != pytest.approx(
        neutral_contact.neutral_gap_interocular,
        abs=1e-7,
    )


def test_character_api_promotes_lists_views_and_rejects_missing_consent(
    tmp_path: Path,
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    job_id, _ = _successful_identity_job(app.state.service.store, tmp_path)
    client = TestClient(app)

    rejected = client.post(
        "/api/characters/from-job",
        data={
            "job_id": job_id,
            "name": "Hero",
            **{
                key: value
                for key, value in _consent().items()
                if key != "consent_evidence_sha256"
            },
        },
        files={"consent_evidence": ("release.txt", b"signed release")},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "CONSENT_REQUIRED"
    promoted = client.post(
        "/api/characters/from-job",
        data={
            "job_id": job_id,
            "name": "Hero",
            "consent_attested": "true",
            **{
                key: value
                for key, value in _consent().items()
                if key != "consent_evidence_sha256"
            },
        },
        files={"consent_evidence": ("release.txt", b"signed release")},
    )
    assert promoted.status_code == 201, promoted.text
    character_id = promoted.json()["character_id"]
    revision = app.state.service.characters.resolve(character_id)
    assert revision.manifest["consent"]["evidence_sha256"] == hashlib.sha256(
        b"signed release"
    ).hexdigest()
    listed = client.get("/api/characters").json()["characters"]
    assert [item["character_id"] for item in listed] == [character_id]
    assert client.get(f"/api/characters/{character_id}").status_code == 200
    viewer = client.get(f"/api/characters/{character_id}/viewer")
    assert viewer.status_code == 200
    assert f"/api/characters/{character_id}/revisions/" in viewer.text
    revision_id = promoted.json()["current_revision_id"]
    preview = client.get(
        f"/api/characters/{character_id}/revisions/{revision_id}/files/preview"
    )
    assert preview.status_code == 200
    assert preview.content == b"glTF-preview"


@pytest.mark.parametrize("route", ["audio", "video"])
def test_character_api_transports_exact_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "missing.task")
    captured: dict[str, str | None] = {}

    def fake_media(input_path, **kwargs):
        captured["character_id"] = kwargs.get("character_id")
        captured["character_revision_id"] = kwargs.get("character_revision_id")
        return {"kind": route, "status": "succeeded"}

    monkeypatch.setattr(app.state.service, route, fake_media)
    response = TestClient(app).post(
        f"/api/{route}",
        files={"file": (f"source.{ 'wav' if route == 'audio' else 'mp4' }", b"media")},
        data={"character_id": "01" + "a" * 24, "character_revision_id": "01" + "b" * 24},
    )
    assert response.status_code == 201
    assert captured == {
        "character_id": "01" + "a" * 24,
        "character_revision_id": "01" + "b" * 24,
    }


@pytest.mark.parametrize("method_name", ["audio", "video"])
def test_service_applies_exact_character_revision_to_performance_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    service = ApplicationService(tmp_path / "jobs", model_path=tmp_path / "model.task")
    job_id, expected_identity = _successful_identity_job(service.store, tmp_path)
    character = service.promote_character(
        job_id,
        name="Hero",
        consent_attested=True,
        **_consent(),
    )
    source = tmp_path / ("performance.mp4" if method_name == "video" else "voice.wav")
    source.write_bytes(b"media")
    captured: dict = {}

    def fake_pipeline(input_path, output_dir, **kwargs):
        captured.update(kwargs)
        return {"kind": f"{method_name}_fake", "warnings": [], "artifacts": {}}

    monkeypatch.setattr(
        f"autoanim_gnm.service.run_{method_name}_pipeline",
        fake_pipeline,
    )
    result = getattr(service, method_name)(
        source,
        character_id=character["character_id"],
    )
    np.testing.assert_array_equal(captured["identity"], expected_identity)
    assert Path(captured["texture_path"]).name == "base-color.png"
    assert captured["texture_triangle_uvs"].shape == (35_324, 3, 2)
    assert np.all(captured["texture_triangle_uvs"] == 0.25)
    assert captured["character_ref"]["revision_id"] == character["current_revision_id"]
    assert captured["character_ref"]["texture_uvs_sha256"]
    assert result["configuration"]["character_id"] == character["character_id"]
