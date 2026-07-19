from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

import autoanim_gnm.cli as cli_module
from autoanim_gnm.artifacts import JobStore
from autoanim_gnm.characters import CharacterStore
from autoanim_gnm.cli import build_parser
from autoanim_gnm.errors import AutoAnimError
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


def _character(store: JobStore, root: Path) -> tuple[CharacterStore, dict]:
    source = root / "source.png"
    Image.new("RGB", (8, 8), (90, 70, 55)).save(source)
    job_id, job_dir, _, manifest = store.start("image_fit", source, {})
    identity = np.linspace(-0.25, 0.25, 253, dtype=np.float32)
    write_npz(job_dir / "fit.npz", identity=identity)
    (job_dir / "fitted.glb").write_bytes(b"glTF-preview")
    store.finish(
        manifest,
        job_dir,
        {
            "kind": "image_fit",
            "fit": {"production_validated": False},
            "warnings": [],
            "artifacts": {"parameters": "fit.npz", "glb": "fitted.glb"},
        },
        {"gnm": "3.0"},
    )
    characters = CharacterStore(root / "characters", store)
    created = characters.promote(
        job_id,
        name="PBR Actor",
        consent_attested=True,
        **_consent(),
    )
    return characters, created


def _entry(path: str, color_space: str) -> dict:
    return {
        "layout": "atlas",
        "path": path,
        "color_space": color_space,
        "source_resolution": [16, 16],
        "resampling": "none",
    }


def _material(root: Path) -> dict:
    inventory: dict[str, object] = {}
    for semantic, color_space in {
        "base_color": "srgb",
        "normal": "linear",
        "specular_color": "linear",
        "subsurface_color": "srgb",
        "subsurface_radius": "linear",
    }.items():
        color = (128, 128, 255) if semantic == "normal" else (96, 128, 160)
        Image.new("RGB", (16, 16), color).save(root / f"{semantic}.png")
        inventory[semantic] = _entry(f"{semantic}.png", color_space)
        if semantic == "normal":
            inventory[semantic]["normal_encoding"] = "unorm"
    for semantic in ("roughness", "confidence"):
        Image.new("L", (16, 16), 127).save(root / f"{semantic}.png")
        inventory[semantic] = _entry(f"{semantic}.png", "linear")
    Image.fromarray(np.full((16, 16), 32768, dtype=np.uint16)).save(
        root / "displacement.png"
    )
    inventory["displacement"] = _entry("displacement.png", "linear")
    Image.new("L", (16, 16), 255).save(root / "skin-mask.png")
    inventory["masks"] = {"skin": _entry("skin-mask.png", "linear")}
    source_hash = "1" * 64
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
    return {
        "package_id": "actor-pbr-v001",
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
            "producer": "AutoAnim fixture",
            "pipeline": "fixture-baker",
            "pipeline_version": "1.0.0",
            "created_at": "2026-07-02T12:00:00+00:00",
            "source_sha256s": [source_hash],
            "processing_log_sha256": "3" * 64,
            "map_lineage": {
                name: {"operation": "derived", "source_sha256s": [source_hash]}
                for name in map_names
            },
        },
        "rights": {
            "status": "cleared",
            "commercial_allowed": True,
            "subject_consent_attested": True,
            "scope": "commercial",
            "evidence_ref": "release://actor/2026-07-01",
            "evidence_sha256": "4" * 64,
            "expires_at": "2027-07-01T00:00:00Z",
        },
        "claims": {
            "resolution_label": "unclaimed",
            "native_resolution": True,
            "pore_resolved": False,
            "relightable": False,
        },
    }


def _glb_json(path: Path) -> dict:
    payload = path.read_bytes()
    assert payload[:4] == b"glTF"
    json_length, json_kind = struct.unpack_from("<II", payload, 12)
    assert json_kind == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def test_character_material_cli_requires_exact_revision_and_local_package() -> None:
    parser = build_parser()
    template = parser.parse_args(
        [
            "character",
            "material-template",
            "character-id",
            "--character-revision",
            "revision-id",
            "--package-root",
            "/capture/package",
            "--spec",
            "/capture/spec.json",
            "--attester",
            "Lookdev",
            "--evidence-ref",
            "release://binding",
            "--evidence",
            "/capture/release.pdf",
            "--package-subject",
            "Test Performer",
            "--same-subject-attested",
            "--authored-for-attested",
            "--displacement-midpoint",
            "0.5",
            "--displacement-scale-m",
            "0.002",
            "--out",
            "/capture/attachment.json",
        ]
    )
    assert template.character_command == "material-template"
    assert template.character_revision == "revision-id"
    assert template.package_root == Path("/capture/package")
    imported = parser.parse_args(
        [
            "character",
            "import-material",
            "character-id",
            "--character-revision",
            "revision-id",
            "--package-root",
            "/capture/package",
            "--spec",
            "/capture/spec.json",
            "--attachment",
            "/capture/attachment.json",
        ]
    )
    assert imported.character_command == "import-material"
    assert imported.attachment == Path("/capture/attachment.json")


def test_material_template_cli_keeps_attachment_output_separate_from_job_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakeService:
        def __init__(self, artifact_root, **kwargs):
            captured["artifact_root"] = Path(artifact_root)

        def prepare_character_material_attachment(self, *args, **kwargs):
            captured["prepare"] = kwargs
            return {
                "attachment": {
                    "schema_version": "autoanim.material-attachment.v1",
                    "attachment_payload_sha256": "a" * 64,
                },
                "material_manifest": {"manifest_payload_sha256": "b" * 64},
            }

    monkeypatch.setattr(cli_module, "ApplicationService", FakeService)
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "binding.txt"
    evidence.write_text("signed", encoding="utf-8")
    output = tmp_path / "material-attachment.json"
    jobs = tmp_path / "jobs"
    exit_code = cli_module.main(
        [
            "character",
            "--artifacts",
            str(jobs),
            "material-template",
            "character-id",
            "--character-revision",
            "revision-id",
            "--package-root",
            str(tmp_path),
            "--spec",
            str(spec),
            "--attester",
            "Lookdev",
            "--evidence-ref",
            "release://binding",
            "--evidence",
            str(evidence),
            "--package-subject",
            "Test Performer",
            "--same-subject-attested",
            "--authored-for-attested",
            "--displacement-midpoint",
            "0.5",
            "--displacement-scale-m",
            "0.002",
            "--out",
            str(output),
        ]
    )
    assert exit_code == 0
    assert captured["artifact_root"] == jobs
    assert output.is_file()
    assert not output.is_dir()


def test_material_attachment_is_exact_immutable_and_runtime_renderable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    characters, created = _character(JobStore(tmp_path / "jobs"), tmp_path)
    character_id = created["character_id"]
    base_id = created["current_revision_id"]
    base = characters.resolve(character_id, base_id)
    package = tmp_path / "material"
    package.mkdir()
    spec = _material(package)
    with pytest.raises(AutoAnimError) as missing_attestation:
        characters.prepare_material_attachment(
            character_id,
            package,
            specification=spec,
            base_revision_id=base_id,
            usage_scope="production",
            attester="Lookdev Supervisor",
            evidence_ref="lookdev://binding/001",
            evidence_sha256="b" * 64,
            package_subject="Test Performer",
            same_subject_attested=True,
            authored_for_attested=False,
            displacement_midpoint=0.5,
            displacement_scale_m=0.002,
        )
    assert missing_attestation.value.code == "MATERIAL_INVALID"
    prepared = characters.prepare_material_attachment(
        character_id,
        package,
        specification=spec,
        base_revision_id=base_id,
        usage_scope="production",
        attester="Lookdev Supervisor",
        evidence_ref="lookdev://binding/001",
        evidence_sha256="b" * 64,
        package_subject="Test Performer",
        same_subject_attested=True,
        authored_for_attested=True,
        displacement_midpoint=0.5,
        displacement_scale_m=0.002,
    )

    top_before = characters.read(character_id)
    wrong_uv = copy.deepcopy(prepared["attachment"])
    wrong_uv.pop("attachment_payload_sha256")
    wrong_uv["authored_for"]["triangle_corner_uv_f32le_sha256"] = "0" * 64
    with pytest.raises(AutoAnimError) as uv_error:
        characters.attach_material(
            character_id,
            package,
            specification=spec,
            attachment=wrong_uv,
            base_revision_id=base_id,
        )
    assert uv_error.value.code == "MATERIAL_BINDING_MISMATCH"
    assert uv_error.value.details["material_code"] == "MATERIAL_BINDING_MISMATCH"
    assert characters.read(character_id) == top_before

    wrong_subject = copy.deepcopy(prepared["attachment"])
    wrong_subject.pop("attachment_payload_sha256")
    wrong_subject["subject_binding"]["package_subject"] = "Different Performer"
    with pytest.raises(AutoAnimError) as subject_error:
        characters.attach_material(
            character_id,
            package,
            specification=spec,
            attachment=wrong_subject,
            base_revision_id=base_id,
        )
    assert subject_error.value.code == "MATERIAL_BINDING_MISMATCH"
    assert subject_error.value.details["material_code"] == "MATERIAL_SUBJECT_MISMATCH"
    assert characters.read(character_id) == top_before

    updated = characters.attach_material(
        character_id,
        package,
        specification=spec,
        attachment=prepared["attachment"],
        base_revision_id=base_id,
    )
    assert updated["revision_count"] == 2
    assert updated["current_revision_id"] != base_id
    current = characters.resolve(character_id)
    old = characters.resolve(character_id, base_id)
    assert current.manifest["source"]["parent_revision_id"] == base_id
    assert current.identity_sha256 == old.identity_sha256 == base.identity_sha256
    assert current.texture_uvs_sha256 == old.texture_uvs_sha256
    assert current.texture_uvs_array_sha256 == old.texture_uvs_array_sha256
    assert set(current.material_asset_paths) == {
        "base_color",
        "confidence",
        "displacement",
        "masks.skin",
        "normal",
        "roughness",
        "specular_color",
        "subsurface_color",
        "subsurface_radius",
    }
    assert set(current.runtime_material_paths) == {
        "base_color",
        "normal",
        "metallic_roughness",
        "specular_color",
    }
    assert current.manifest["appearance"]["production_validated"] is False
    assert current.manifest["appearance"]["uv_binding"][
        "attachment_payload_sha256"
    ] == prepared["attachment"]["attachment_payload_sha256"]
    document = _glb_json(current.preview_path)
    primitive = document["meshes"][0]["primitives"][0]
    material = document["materials"][0]
    assert "TANGENT" in primitive["attributes"]
    assert "normalTexture" in material
    assert "metallicRoughnessTexture" in material["pbrMetallicRoughness"]
    assert "KHR_materials_specular" in material["extensions"]
    assert "KHR_materials_specular" in document["extensionsUsed"]

    service = ApplicationService(
        tmp_path / "jobs", character_root=tmp_path / "characters"
    )
    captured: dict[str, dict] = {}
    for method in ("audio", "video"):
        source = tmp_path / ("voice.wav" if method == "audio" else "take.mp4")
        source.write_bytes(b"media")

        def fake_pipeline(input_path, output_dir, *, _method=method, **kwargs):
            captured[_method] = kwargs
            return {"kind": _method, "warnings": [], "artifacts": {}}

        monkeypatch.setattr(
            f"autoanim_gnm.service.run_{method}_pipeline", fake_pipeline
        )
        getattr(service, method)(source, character_id=character_id)
    for kwargs in captured.values():
        assert kwargs["texture_path"] is None
        assert set(kwargs["runtime_material_paths"]) == set(
            current.runtime_material_paths
        )
        assert kwargs["character_ref"]["revision_id"] == current.revision_id
        assert kwargs["character_ref"]["material_map_sha256s"]
        assert kwargs["character_ref"]["runtime_material_sha256s"]
        assert "uv_binding" not in kwargs["character_ref"]["appearance"]

    with pytest.raises(AutoAnimError) as stale:
        characters.attach_material(
            character_id,
            package,
            specification=spec,
            attachment=prepared["attachment"],
            base_revision_id=base_id,
        )
    assert stale.value.code == "REVISION_CONFLICT"

    package_artifact = current.preview_path.parent / "material-package.json"
    package_bytes = package_artifact.read_bytes()
    package_artifact.unlink()
    with pytest.raises(AutoAnimError) as missing_package:
        characters.resolve(character_id, current.revision_id)
    assert missing_package.value.code == "INTEGRITY_FAILED"
    package_artifact.write_bytes(package_bytes)

    class AfterMaterialExpiry(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2027, 7, 2, tzinfo=timezone.utc)
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr("autoanim_gnm.characters.datetime", AfterMaterialExpiry)
    with pytest.raises(AutoAnimError) as expired:
        characters.resolve(character_id, current.revision_id)
    assert expired.value.code == "RIGHTS_EXPIRED"
    assert characters.list()[0]["material_rights_status"] == "expired"
    # Rights expiry belongs to the appearance revision and does not rewrite or
    # invalidate its geometry-only parent.
    characters.resolve(character_id, base_id)


def test_material_source_mutation_after_attachment_template_fails_atomically(
    tmp_path: Path,
) -> None:
    characters, created = _character(JobStore(tmp_path / "jobs"), tmp_path)
    character_id = created["character_id"]
    base_id = created["current_revision_id"]
    package = tmp_path / "material"
    package.mkdir()
    spec = _material(package)
    prepared = characters.prepare_material_attachment(
        character_id,
        package,
        specification=spec,
        base_revision_id=base_id,
        usage_scope="production",
        attester="Lookdev Supervisor",
        evidence_ref="lookdev://binding/002",
        evidence_sha256=hashlib.sha256(b"binding evidence").hexdigest(),
        package_subject="Test Performer",
        same_subject_attested=True,
        authored_for_attested=True,
        displacement_midpoint=0.5,
        displacement_scale_m=0.002,
    )
    Image.new("L", (16, 16), 240).save(package / "roughness.png")
    before = characters.read(character_id)

    with pytest.raises(AutoAnimError) as changed:
        characters.attach_material(
            character_id,
            package,
            specification=spec,
            attachment=prepared["attachment"],
            base_revision_id=base_id,
        )
    assert changed.value.code == "MATERIAL_BINDING_MISMATCH"
    assert characters.read(character_id) == before
    revisions = characters.root / character_id / "revisions"
    assert sorted(path.name for path in revisions.iterdir()) == [base_id]


def test_material_publish_fsync_failure_never_leaves_dangling_top_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    characters, created = _character(JobStore(tmp_path / "jobs"), tmp_path)
    character_id = created["character_id"]
    base_id = created["current_revision_id"]
    package = tmp_path / "material"
    package.mkdir()
    spec = _material(package)
    prepared = characters.prepare_material_attachment(
        character_id,
        package,
        specification=spec,
        base_revision_id=base_id,
        usage_scope="production",
        attester="Lookdev Supervisor",
        evidence_ref="lookdev://binding/fsync",
        evidence_sha256="c" * 64,
        package_subject="Test Performer",
        same_subject_attested=True,
        authored_for_attested=True,
        displacement_midpoint=0.5,
        displacement_scale_m=0.002,
    )
    character_dir = characters.root / character_id
    from autoanim_gnm import characters as characters_module

    original_fsync = characters_module._fsync_directory

    def fail_top_fsync(path: Path) -> None:
        if Path(path) == character_dir:
            raise OSError("injected directory fsync failure")
        original_fsync(path)

    monkeypatch.setattr(characters_module, "_fsync_directory", fail_top_fsync)
    with pytest.raises(OSError, match="injected"):
        characters.attach_material(
            character_id,
            package,
            specification=spec,
            attachment=prepared["attachment"],
            base_revision_id=base_id,
        )
    top = characters.read(character_id)
    assert top["current_revision_id"] != base_id
    assert top["revision_count"] == 2
    resolved = characters.resolve(character_id, top["current_revision_id"])
    assert resolved.preview_path.is_file()
