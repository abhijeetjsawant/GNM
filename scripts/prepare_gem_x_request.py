#!/usr/bin/env python3
"""Seal exact video/preprocessing evidence for the Modal GEM-X worker."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from workers.gem_x.infer_cuda import GEM_X_COMMIT, REQUEST_SCHEMA


REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--bounding-boxes", type=Path, required=True)
    parser.add_argument("--vitpose", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    if not REQUEST_ID.fullmatch(arguments.request_id):
        raise ValueError("Unsafe GEM-X request_id")
    inputs = {
        "video": arguments.video.resolve(strict=True),
        "timing": arguments.timing.resolve(strict=True),
        "bounding_boxes": arguments.bounding_boxes.resolve(strict=True),
        "vitpose": arguments.vitpose.resolve(strict=True),
    }
    output = arguments.output_root.resolve() / arguments.request_id
    output.mkdir(parents=True, exist_ok=False)
    declarations = {}
    for label, source in inputs.items():
        destination = output / source.name
        if destination.name in {value["file_name"] for value in declarations.values()}:
            destination = output / f"{label}-{source.name}"
        shutil.copy2(source, destination)
        declarations[label] = {
            "file_name": destination.name,
            "sha256": _sha256(destination),
        }
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": arguments.request_id,
        "provider_git_commit_oid": GEM_X_COMMIT,
        "sampling_mode": "official_regression",
        "static_camera": True,
        "source": declarations,
    }
    (output / "request.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
