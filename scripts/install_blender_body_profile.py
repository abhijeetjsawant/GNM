#!/usr/bin/env python3
"""Install audited MPFB inputs into the active isolated Blender profile.

This script must run inside Blender.  The host bootstrap sets every
``BLENDER_USER_*`` directory to a project-local path before invoking it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import bpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoanim_gnm.body_provider import (  # noqa: E402
    BODY_PROFILE_ATTESTATION_SCHEMA,
    digest_body_profile_tree,
    sha256_file,
    write_body_provider_json,
)


def _paths() -> tuple[Path, Path, Path]:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 3:
        raise RuntimeError(
            "Expected audited MPFB ZIP, system-assets ZIP, and attestation paths after --"
        )
    archives = tuple(Path(value).resolve() for value in args[:2])
    if not all(path.is_file() for path in archives):
        raise RuntimeError("An audited provider archive is missing")
    return archives[0], archives[1], Path(args[2]).resolve()


def main() -> int:
    extension_zip, system_assets_zip, attestation_path = _paths()
    result = bpy.ops.extensions.package_install_files(
        filepath=str(extension_zip),
        repo="user_default",
        enable_on_install=True,
        overwrite=True,
    )
    if result != {"FINISHED"}:
        raise RuntimeError(f"MPFB extension installation failed: {result}")

    from bl_ext.user_default.mpfb.services import AssetService, LocationService

    archive_issue = AssetService.check_asset_pack_zip(str(system_assets_zip))
    if archive_issue is not None:
        raise RuntimeError(f"MPFB rejected the system-assets archive: {archive_issue}")
    result = bpy.ops.mpfb.load_pack(filepath=str(system_assets_zip))
    if result != {"FINISHED"}:
        raise RuntimeError(f"System-assets installation failed: {result}")
    installed = AssetService.check_if_modern_makehuman_system_assets_installed()
    if installed != (True, True) or not AssetService.system_assets_pack_is_installed():
        raise RuntimeError(f"MPFB did not recognize the installed system pack: {installed}")
    extension_module = importlib.import_module("bl_ext.user_default.mpfb")
    extension_root = Path(extension_module.__file__).resolve().parent
    system_assets_root = Path(LocationService.get_user_home()).resolve() / "data"
    write_body_provider_json(
        attestation_path,
        {
            "schema_version": BODY_PROFILE_ATTESTATION_SCHEMA,
            "mpfb_archive_sha256": sha256_file(extension_zip),
            "system_assets_archive_sha256": sha256_file(system_assets_zip),
            "extension_tree": digest_body_profile_tree(extension_root),
            "system_assets_tree": digest_body_profile_tree(system_assets_root),
        },
    )
    if bpy.ops.wm.save_userpref() != {"FINISHED"}:
        raise RuntimeError("Could not persist the isolated Blender preferences")
    print(f"AUTOANIM_BODY_PROFILE={LocationService.get_user_home()}")
    print("AUTOANIM_BODY_PROFILE_READY=1")
    print(f"AUTOANIM_BODY_PROFILE_ATTESTATION={attestation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
