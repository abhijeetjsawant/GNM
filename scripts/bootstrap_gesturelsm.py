#!/usr/bin/env python3
"""Install or verify the pinned GestureLSM provider on a Linux CUDA host."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "workers" / "gesturelsm"
LOCK = json.loads((WORKER / "provider-lock.json").read_text(encoding="utf-8"))


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_ok(path: Path, *, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == expected_size
        and _sha(path) == expected_sha256
    )


def verify(install_root: Path) -> dict[str, object]:
    provider = install_root / "GestureLSM"
    checkpoint = provider / "ckpt" / LOCK["model"]["checkpoint"]
    config = provider / LOCK["model"]["config"]
    observed_commit = None
    if provider.is_dir():
        observed_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=provider,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or None
    decoder_results = {
        relative: _artifact_ok(
            provider / relative,
            expected_size=specification["bytes"],
            expected_sha256=specification["sha256"],
        )
        for relative, specification in LOCK["model"]["decoder_artifacts"].items()
    }
    result = {
        "provider_root": str(provider),
        "provider_commit_expected": LOCK["provider"]["git_commit_oid"],
        "provider_commit_observed": observed_commit,
        "provider_commit_ok": observed_commit == LOCK["provider"]["git_commit_oid"],
        "checkpoint": str(checkpoint),
        "checkpoint_ok": _artifact_ok(
            checkpoint,
            expected_size=LOCK["model"]["checkpoint_bytes"],
            expected_sha256=LOCK["model"]["checkpoint_sha256"],
        ),
        "config_ok": config.is_file()
        and not config.is_symlink()
        and _sha(config) == LOCK["model"]["config_sha256"],
        "decoder_artifacts": decoder_results,
        "decoder_artifacts_ok": all(decoder_results.values()),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "conda_available": shutil.which("conda") is not None,
    }
    result["ready_for_inference"] = all(
        result[key]
        for key in (
            "provider_commit_ok",
            "checkpoint_ok",
            "config_ok",
            "decoder_artifacts_ok",
            "nvidia_smi_available",
            "conda_available",
        )
    )
    return result


def install(install_root: Path) -> None:
    install_root.mkdir(parents=True, exist_ok=True)
    provider = install_root / "GestureLSM"
    if not provider.exists():
        _run(["git", "clone", LOCK["provider"]["repository"], str(provider)])
    _run(["git", "fetch", "origin", "main"], cwd=provider)
    _run(["git", "checkout", "--detach", LOCK["provider"]["git_commit_oid"]], cwd=provider)
    patch = WORKER / "upstream-supplied-transcript.patch"
    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)], cwd=provider
    )
    if reverse_check.returncode:
        _run(["git", "apply", str(patch)], cwd=provider)
    environment = "autoanim-gesturelsm"
    environments = json.loads(
        subprocess.run(
            ["conda", "env", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["envs"]
    if not any(Path(value).name == environment for value in environments):
        _run(["conda", "create", "-n", environment, "python=3.12", "-y"])
    _run(
        [
            "conda",
            "install",
            "-n",
            environment,
            "pytorch==2.1.2",
            "torchvision==0.16.2",
            "torchaudio==2.1.2",
            "pytorch-cuda=11.8",
            "-c",
            "pytorch",
            "-c",
            "nvidia",
            "-y",
        ]
    )
    _run(
        ["conda", "run", "-n", environment, "pip", "install", "-r", "requirements.txt"],
        cwd=provider,
    )
    _run(
        [
            "conda",
            "run",
            "-n",
            environment,
            "huggingface-cli",
            "download",
            LOCK["model"]["repository"],
            "--revision",
            LOCK["model"]["revision"],
            "--local-dir",
            str(provider / "ckpt"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    arguments = parser.parse_args()
    if arguments.install:
        install(arguments.install_root)
    report = verify(arguments.install_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_inference"] else 2


if __name__ == "__main__":
    sys.exit(main())
