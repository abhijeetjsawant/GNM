"""Small deterministic, atomic artifact writers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _atomic_path(path: Path) -> tuple[object, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    return handle, Path(handle.name)


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    handle, temporary = _atomic_path(path)
    try:
        payload = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    handle, temporary = _atomic_path(path)
    try:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    except Exception:
        handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return path
