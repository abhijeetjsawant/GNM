"""Render a source/safe-SOMA side-by-side diagnostic preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from autoanim_gnm.nvidia_body_provider import (
    load_gem_x_cuda_response,
    load_gem_x_preview_response,
)
from autoanim_gnm.soma_motion import SOMASKEL77_PARENTS


def render_preview(
    video_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    cuda_soma: bool = False,
) -> None:
    motion = (
        load_gem_x_cuda_response(manifest_path)
        if cuda_soma
        else load_gem_x_preview_response(manifest_path)
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create preview: {output_path}")

    positions = motion.joint_positions_m
    x_min, y_min = positions[..., :2].min(axis=(0, 1))
    x_max, y_max = positions[..., :2].max(axis=(0, 1))
    span = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    center_x = float(x_min + x_max) * 0.5
    center_y = float(y_min + y_max) * 0.5
    scale = 0.82 * min(width, height) / span
    frame_index = 0
    try:
        while frame_index < len(positions):
            available, source = capture.read()
            if not available:
                raise RuntimeError("Source video ended before the SOMA track")
            panel = np.full((height, width, 3), (24, 24, 27), dtype=np.uint8)
            joints = positions[frame_index]
            pixels = np.empty((77, 2), dtype=np.int32)
            pixels[:, 0] = np.rint((joints[:, 0] - center_x) * scale + width / 2).astype(
                np.int32
            )
            pixels[:, 1] = np.rint(
                height / 2 - (joints[:, 1] - center_y) * scale
            ).astype(np.int32)
            for joint, parent in enumerate(SOMASKEL77_PARENTS):
                if parent >= 0:
                    cv2.line(
                        panel,
                        tuple(pixels[parent]),
                        tuple(pixels[joint]),
                        (125, 205, 255),
                        max(1, width // 320),
                        cv2.LINE_AA,
                    )
            for point in pixels:
                cv2.circle(panel, tuple(point), max(2, width // 180), (245, 245, 245), -1)
            cv2.putText(
                panel,
                (
                    "GEM-X / SOMA-77  |  CUDA image-conditioned"
                    if cuda_soma
                    else "GEM-X / SOMA-77  |  Apple CPU preview"
                ),
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "Research-only - no production accuracy claim",
                (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (110, 170, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(np.concatenate((source, panel), axis=1))
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cuda-soma",
        action="store_true",
        help="Load a strict full-image-conditioned CUDA GEM-X response.",
    )
    arguments = parser.parse_args()
    render_preview(
        arguments.video.resolve(),
        arguments.manifest.resolve(),
        arguments.output.resolve(),
        cuda_soma=arguments.cuda_soma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
