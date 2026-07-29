"""Provider-internal exact frame/PTS identity for GEM-X preprocessing."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


TIMING_SCHEMA = "autoanim.gem-x-frame-timing/1.0"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decode_video_with_timing(
    video_path: Path,
    *,
    av_module: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if av_module is None:
        import av as av_module

    frames: list[np.ndarray] = []
    points: list[int] = []
    frame_hashes: list[str] = []
    with av_module.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for frame in container.decode(stream):
            if frame.pts is None:
                raise RuntimeError("GEM-X preview requires a PTS on every decoded frame")
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(rgb)
            points.append(int(frame.pts))
            frame_hashes.append(sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest())
    if len(frames) < 2 or any(right <= left for left, right in zip(points, points[1:])):
        raise RuntimeError("GEM-X preview requires strictly increasing frame PTS")
    result = np.stack(frames)
    return result, {
        "schema_version": TIMING_SCHEMA,
        "input_sha256": file_sha256(video_path),
        "frame_count": len(frames),
        "source_pts": points,
        "source_time_base": {
            "numerator": int(time_base.numerator),
            "denominator": int(time_base.denominator),
        },
        "rgb_frame_sha256": frame_hashes,
    }


def write_timing(path: Path, timing: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial")
    temporary.write_text(
        json.dumps(timing, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_timing(path: Path, video_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frame timing sidecar does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "input_sha256",
        "frame_count",
        "source_pts",
        "source_time_base",
        "rgb_frame_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("Frame timing sidecar has an unexpected schema")
    if value["schema_version"] != TIMING_SCHEMA or value["input_sha256"] != file_sha256(
        video_path
    ):
        raise RuntimeError("Frame timing sidecar is not bound to the input video")
    frame_count = value["frame_count"]
    points = value["source_pts"]
    hashes = value["rgb_frame_sha256"]
    time_base = value["source_time_base"]
    if (
        type(frame_count) is not int
        or frame_count < 2
        or not isinstance(points, list)
        or not isinstance(hashes, list)
        or len(points) != frame_count
        or len(hashes) != frame_count
        or any(type(point) is not int for point in points)
        or any(right <= left for left, right in zip(points, points[1:]))
        or not isinstance(time_base, dict)
        or set(time_base) != {"numerator", "denominator"}
        or any(type(value) is not int or value <= 0 for value in time_base.values())
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        )
    ):
        raise RuntimeError("Frame timing sidecar contains invalid timing")
    return value


def verify_decoded_frames(frames: np.ndarray, timing: dict[str, Any]) -> None:
    if len(frames) != timing["frame_count"]:
        raise RuntimeError("Decoded frame count differs from the timing sidecar")
    observed = [
        sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames
    ]
    if observed != timing["rgb_frame_sha256"]:
        raise RuntimeError("Decoded RGB frame identity differs from preprocessing")
