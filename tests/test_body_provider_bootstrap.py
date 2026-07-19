from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import pytest

import autoanim_gnm.body_provider as body_provider
from autoanim_gnm.body_provider import (
    BodyProviderError,
    audit_makehuman_system_assets_archive,
    default_body_provider_request,
    digest_body_profile_tree,
    validate_body_profile_attestation,
    write_body_provider_json,
)


def _bootstrap_module():
    path = Path(__file__).parents[1] / "scripts" / "bootstrap_body_provider.py"
    spec = importlib.util.spec_from_file_location("autoanim_test_body_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _system_pack(path: Path, *, license_name: str = "CC0", unsafe: bool = False) -> str:
    metadata = {
        "brown": {
            "author": "makehuman_system",
            "license": license_name,
            "type": "eyes",
        }
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("packs/makehuman_system_assets.json", json.dumps(metadata))
        archive.writestr("eyes/materials/brown.mhmat", "name brown\n")
        if unsafe:
            archive.writestr("../escape", "must never be extracted")
    return sha256(path.read_bytes()).hexdigest()


def test_system_pack_audit_requires_safe_uniformly_cc0_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "system.zip"
    expected = _system_pack(archive)
    monkeypatch.setattr(
        body_provider,
        "CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256",
        expected,
    )

    audit = audit_makehuman_system_assets_archive(archive, expected_sha256=expected)

    assert audit["asset_records"] == 1
    assert audit["asset_spdx"] == "CC0-1.0"
    assert audit["sha256"] == expected


def test_system_pack_audit_rejects_traversal_even_with_matching_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "system.zip"
    expected = _system_pack(archive, unsafe=True)
    monkeypatch.setattr(
        body_provider,
        "CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256",
        expected,
    )

    with pytest.raises(BodyProviderError, match="unsafe entry"):
        audit_makehuman_system_assets_archive(archive, expected_sha256=expected)


def test_system_pack_audit_rejects_non_cc0_asset_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "system.zip"
    expected = _system_pack(archive, license_name="CC-BY")
    monkeypatch.setattr(
        body_provider,
        "CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256",
        expected,
    )

    with pytest.raises(BodyProviderError, match="uniformly CC0"):
        audit_makehuman_system_assets_archive(archive, expected_sha256=expected)


def test_production_request_rejects_any_caller_selected_system_pack() -> None:
    with pytest.raises(BodyProviderError, match="corroborated official-mirror lock"):
        default_body_provider_request(
            "unlocked-pack",
            system_assets_sha256="7" * 64,
        )


def test_loaded_profile_attestation_binds_complete_installed_trees(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "extension"
    assets = tmp_path / "assets"
    extension.mkdir()
    assets.mkdir()
    (extension / "module.py").write_bytes(b"reviewed extension")
    (assets / "brown.mhmat").write_bytes(b"reviewed asset")
    attestation = tmp_path / "attestation.json"
    write_body_provider_json(
        attestation,
        {
            "schema_version": body_provider.BODY_PROFILE_ATTESTATION_SCHEMA,
            "mpfb_archive_sha256": body_provider.PINNED_MPFB_EXTENSION_SHA256,
            "system_assets_archive_sha256": (
                body_provider.CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256
            ),
            "extension_tree": digest_body_profile_tree(extension),
            "system_assets_tree": digest_body_profile_tree(assets),
        },
    )
    validate_body_profile_attestation(
        attestation,
        extension_root=extension,
        system_assets_root=assets,
        expected_mpfb_archive_sha256=body_provider.PINNED_MPFB_EXTENSION_SHA256,
        expected_system_assets_archive_sha256=(
            body_provider.CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256
        ),
    )

    (assets / "brown.mhmat").write_bytes(b"tampered asset")
    with pytest.raises(BodyProviderError, match="system_assets_tree digest"):
        validate_body_profile_attestation(
            attestation,
            extension_root=extension,
            system_assets_root=assets,
            expected_mpfb_archive_sha256=body_provider.PINNED_MPFB_EXTENSION_SHA256,
            expected_system_assets_archive_sha256=(
                body_provider.CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256
            ),
        )


def test_bootstrap_profile_environment_is_project_local_without_home_override(
    tmp_path: Path,
) -> None:
    bootstrap = _bootstrap_module()
    environment = bootstrap._profile_environment(tmp_path)

    assert environment.get("HOME") == os.environ.get("HOME")
    for key in (
        "BLENDER_USER_RESOURCES",
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_EXTENSIONS",
        "BLENDER_USER_DATAFILES",
    ):
        assert Path(environment[key]).is_relative_to(tmp_path)


def test_cached_download_is_accepted_only_when_size_and_hash_match(tmp_path: Path) -> None:
    bootstrap = _bootstrap_module()
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"pinned bytes")
    expected = sha256(destination.read_bytes()).hexdigest()

    bootstrap._download(
        "https://invalid.example/not-used",
        destination,
        expected_sha256=expected,
        expected_size=destination.stat().st_size,
    )
    with pytest.raises(RuntimeError, match="fails its pin"):
        bootstrap._download(
            "https://invalid.example/not-used",
            destination,
            expected_sha256="0" * 64,
            expected_size=destination.stat().st_size,
        )
