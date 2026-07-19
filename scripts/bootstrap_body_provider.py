#!/usr/bin/env python3
"""Reproducibly audit and bootstrap the pinned macOS body provider.

The bootstrap never writes to Blender's global user directories.  Downloads,
the extension repository, preferences, MPFB data and smoke-test output all
live below ``--cache-dir``.  The MakeHuman archive is fetched from both
official mirrors because its publisher does not publish a signed checksum;
the mirrors must be byte-identical to the checked-in corroborated lock.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autoanim_gnm.body_provider import (  # noqa: E402
    CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
    MAKEHUMAN_LICENSE_URL,
    MAKEHUMAN_SYSTEM_ASSETS_MIRROR_URL,
    MAKEHUMAN_SYSTEM_ASSETS_URL,
    PINNED_BLENDER_SHA256,
    PINNED_BLENDER_URL,
    PINNED_BLENDER_VERSION,
    PINNED_MPFB_EXTENSION_SHA256,
    PINNED_MPFB_EXTENSION_URL,
    PINNED_MPFB_GIT_COMMIT,
    PINNED_MPFB_VERSION,
    audit_makehuman_system_assets_archive,
    audit_mpfb_extension_archive,
    default_body_provider_request,
    load_and_validate_body_provider_result,
    sha256_file,
    write_body_provider_json,
)


BLENDER_CHECKSUM_URL = (
    "https://download.blender.org/release/Blender4.5/blender-4.5.11.sha256"
)
EXPECTED_SIZES = {
    "blender": 308_255_028,
    "mpfb": 44_978_070,
    "system_assets": 280_737_770,
}


def _download(url: str, destination: Path, *, expected_sha256: str, expected_size: int) -> None:
    if destination.is_file():
        if destination.stat().st_size != expected_size or sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"Existing cached artifact fails its pin: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "AutoAnim-body-bootstrap/1.0"})
    digest = sha256()
    size = 0
    with urlopen(request, timeout=60) as source, partial.open("wb") as target:
        while block := source.read(1024 * 1024):
            target.write(block)
            digest.update(block)
            size += len(block)
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"Downloaded artifact fails its pin: {destination.name}; "
            f"bytes={size}, sha256={digest.hexdigest()}"
        )
    partial.replace(destination)


def _verify_blender_publisher_checksum() -> str:
    request = Request(
        BLENDER_CHECKSUM_URL,
        headers={"User-Agent": "AutoAnim-body-bootstrap/1.0"},
    )
    with urlopen(request, timeout=30) as response:
        checksum_text = response.read().decode("ascii")
    expected_line = f"{PINNED_BLENDER_SHA256}  blender-4.5.11-macos-arm64.dmg"
    if expected_line not in checksum_text.splitlines():
        raise RuntimeError("Blender's publisher checksum file does not contain the pinned DMG")
    return expected_line


def _profile_environment(cache: Path) -> dict[str, str]:
    profile = cache / "profile"
    environment = dict(os.environ)
    environment.update(
        {
            "BLENDER_USER_RESOURCES": str(profile / "resources"),
            "BLENDER_USER_CONFIG": str(profile / "config"),
            "BLENDER_USER_SCRIPTS": str(profile / "scripts"),
            "BLENDER_USER_EXTENSIONS": str(profile / "extensions"),
            "BLENDER_USER_DATAFILES": str(profile / "datafiles"),
        }
    )
    return environment


def _mounted_blender(cache: Path, dmg: Path) -> tuple[Path, bool]:
    mount = cache / "mount"
    executable = mount / "Blender.app" / "Contents" / "MacOS" / "Blender"
    if executable.is_file():
        return executable, False
    mount.mkdir(parents=True, exist_ok=True)
    if any(mount.iterdir()):
        raise RuntimeError(f"Blender mount directory is not empty: {mount}")
    subprocess.run(
        [
            "hdiutil",
            "attach",
            "-nobrowse",
            "-readonly",
            "-mountpoint",
            str(mount),
            str(dmg),
        ],
        check=True,
    )
    if not executable.is_file():
        raise RuntimeError("Pinned Blender DMG did not contain Blender.app")
    return executable, True


def _verify_blender_runtime(executable: Path, environment: dict[str, str]) -> dict[str, object]:
    completed = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    first_line = completed.stdout.splitlines()[0].strip()
    if first_line != f"Blender {PINNED_BLENDER_VERSION} LTS":
        raise RuntimeError(f"Unexpected Blender runtime: {first_line}")
    app = executable.parents[2]
    subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        ["spctl", "--assess", "--type", "execute", str(app)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    license_path = executable.parents[1] / "Resources" / "text" / "license" / "license.md"
    license_text = license_path.read_text(encoding="utf-8")
    if "GPL-3.0-or-later" not in license_text:
        raise RuntimeError("Pinned Blender bundle does not contain the expected GPL license notice")
    return {
        "version": first_line,
        "code_spdx": "GPL-3.0-or-later",
        "codesign_verified": True,
        "gatekeeper_notarized": True,
    }


def _install_profile(
    executable: Path,
    environment: dict[str, str],
    extension_zip: Path,
    system_assets_zip: Path,
    attestation_path: Path,
) -> None:
    subprocess.run(
        [
            str(executable),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "31",
            "--python",
            str(PROJECT_ROOT / "scripts" / "install_blender_body_profile.py"),
            "--",
            str(extension_zip),
            str(system_assets_zip),
            str(attestation_path),
        ],
        check=True,
        env=environment,
    )


def _smoke_test(
    cache: Path,
    executable: Path,
    environment: dict[str, str],
    extension_zip: Path,
    system_assets_zip: Path,
    attestation_path: Path,
) -> dict[str, object]:
    output = cache / "run" / "bootstrap-smoke"
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    response_path = output / "response.json"
    request = default_body_provider_request(
        "bootstrap-real-mpfb",
        system_assets_sha256=CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
    )
    write_body_provider_json(request_path, request)
    worker_environment = dict(environment)
    worker_environment.update(
        {
            "AUTOANIM_MPFB_EXTENSION_ZIP": str(extension_zip),
            "AUTOANIM_MAKEHUMAN_SYSTEM_ASSETS_ZIP": str(system_assets_zip),
            "AUTOANIM_BODY_PROFILE_ATTESTATION": str(attestation_path),
        }
    )
    subprocess.run(
        [
            str(executable),
            "--background",
            "--python-exit-code",
            "31",
            "--python",
            str(PROJECT_ROOT / "scripts" / "blender_body_worker.py"),
            "--",
            str(request_path),
            str(response_path),
        ],
        check=True,
        env=worker_environment,
    )
    response = load_and_validate_body_provider_result(request_path, response_path)
    if response["status"] != "succeeded":
        raise RuntimeError(f"Body-provider smoke test did not succeed: {response}")
    artifacts = response["artifacts"]
    return {
        "request": str(request_path),
        "response": str(response_path),
        "manifest": str(output / artifacts["manifest_json"]),
        "asset": str(output / artifacts["asset_npz"]),
        "asset_sha256": artifacts["asset_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "autoanim_gnm" / "body-provider",
    )
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and not args.install:
        parser.error("--smoke requires --install")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("This lock is specifically for pinned Blender macOS arm64")

    cache = args.cache_dir.resolve()
    downloads = cache / "downloads"
    blender_dmg = downloads / "blender-4.5.11-macos-arm64.dmg"
    extension_zip = downloads / "add-on-mpfb-v2.0.16.zip"
    system_primary = downloads / "makehuman_system_assets_cc0-mirror1.zip"
    system_mirror = downloads / "makehuman_system_assets_cc0-mirror2.zip"
    profile_attestation = cache / "profile-attestation.json"

    publisher_checksum = _verify_blender_publisher_checksum()
    _download(
        PINNED_BLENDER_URL,
        blender_dmg,
        expected_sha256=PINNED_BLENDER_SHA256,
        expected_size=EXPECTED_SIZES["blender"],
    )
    _download(
        PINNED_MPFB_EXTENSION_URL,
        extension_zip,
        expected_sha256=PINNED_MPFB_EXTENSION_SHA256,
        expected_size=EXPECTED_SIZES["mpfb"],
    )
    for url, path in (
        (MAKEHUMAN_SYSTEM_ASSETS_URL, system_primary),
        (MAKEHUMAN_SYSTEM_ASSETS_MIRROR_URL, system_mirror),
    ):
        _download(
            url,
            path,
            expected_sha256=CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
            expected_size=EXPECTED_SIZES["system_assets"],
        )
    if sha256_file(system_primary) != sha256_file(system_mirror):
        raise RuntimeError("Official MakeHuman system-asset mirrors are not byte-identical")

    mpfb_audit = audit_mpfb_extension_archive(extension_zip)
    system_audit = audit_makehuman_system_assets_archive(
        system_primary,
        expected_sha256=CORROBORATED_MAKEHUMAN_SYSTEM_ASSETS_SHA256,
    )
    environment = _profile_environment(cache)
    executable, mounted_here = _mounted_blender(cache, blender_dmg)
    try:
        runtime_audit = _verify_blender_runtime(executable, environment)
        smoke = None
        if args.install:
            _install_profile(
                executable,
                environment,
                extension_zip,
                system_primary,
                profile_attestation,
            )
        if args.smoke:
            smoke = _smoke_test(
                cache,
                executable,
                environment,
                extension_zip,
                system_primary,
                profile_attestation,
            )
    finally:
        if mounted_here:
            subprocess.run(["hdiutil", "detach", str(cache / "mount")], check=True)

    report = {
        "schema_version": "autoanim.body-provider-bootstrap-audit/1.0",
        "blender": {
            "runtime": runtime_audit,
            "url": PINNED_BLENDER_URL,
            "sha256": PINNED_BLENDER_SHA256,
            "bytes": blender_dmg.stat().st_size,
            "publisher_checksum": publisher_checksum,
        },
        "mpfb": {
            "url": PINNED_MPFB_EXTENSION_URL,
            "git_commit": PINNED_MPFB_GIT_COMMIT,
            "git_commit_relationship_verified": False,
            **mpfb_audit,
        },
        "makehuman_system_assets": {
            "urls": [MAKEHUMAN_SYSTEM_ASSETS_URL, MAKEHUMAN_SYSTEM_ASSETS_MIRROR_URL],
            "mirrors_byte_identical": True,
            "publisher_signed_checksum_available": False,
            "license_url": MAKEHUMAN_LICENSE_URL,
            **system_audit,
        },
        "profile": {key: environment[key] for key in sorted(environment) if key.startswith("BLENDER_USER_")},
        "profile_attestation": str(profile_attestation) if args.install else None,
        "smoke": smoke,
    }
    report_path = cache / "audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"AUTOANIM_BODY_AUDIT={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
