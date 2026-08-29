#!/usr/bin/env python3
"""Render a source-free 3D skeleton preview from a MAMMA joints archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SOURCE_Z_UP_TO_Y_UP = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)), dtype=np.float64
)


def _rotation(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    y = np.asarray(
        ((np.cos(yaw), 0.0, np.sin(yaw)), (0.0, 1.0, 0.0), (-np.sin(yaw), 0.0, np.cos(yaw))),
        dtype=np.float64,
    )
    x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, np.cos(pitch), -np.sin(pitch)), (0.0, np.sin(pitch), np.cos(pitch))),
        dtype=np.float64,
    )
    return x @ y


def render(vertices_path: Path, skeleton_path: Path, output_path: Path) -> None:
    with np.load(vertices_path, allow_pickle=False) as values:
        joints = np.asarray(values["pred_joints"], dtype=np.float64)[:, :55]
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    parents = np.asarray(skeleton["parents"], dtype=np.int64)
    if joints.ndim != 3 or joints.shape[1:] != (55, 3) or len(parents) != 55:
        raise ValueError("Expected a [frames,55,3] MAMMA joints archive and SMPL-X-55 skeleton")
    if not np.isfinite(joints).all():
        raise ValueError("MAMMA joints must be finite")

    # Canonical AutoAnim space: +Y up, root stable at its initial origin.
    positions = np.einsum("ab,fjb->fja", SOURCE_Z_UP_TO_Y_UP, joints)
    positions -= positions[0, 0]
    camera_space = np.einsum("ab,fjb->fja", _rotation(18.0, -8.0), positions)
    low = camera_space.min(axis=(0, 1))
    high = camera_space.max(axis=(0, 1))
    center = (low + high) * 0.5
    span = max(float(high[0] - low[0]), float(high[1] - low[1]), 1e-3)

    width, height, fps = 720, 720, 30.0
    scale = 0.82 * min(width, height) / span
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {output_path}")
    try:
        for frame in camera_space:
            image = np.full((height, width, 3), (16, 19, 25), dtype=np.uint8)
            pixels = np.empty((55, 2), dtype=np.int32)
            pixels[:, 0] = np.rint((frame[:, 0] - center[0]) * scale + width / 2).astype(np.int32)
            pixels[:, 1] = np.rint(height / 2 - (frame[:, 1] - center[1]) * scale).astype(np.int32)
            # Draw farther joints first so the nearest limbs remain legible.
            for joint in np.argsort(frame[:, 2]):
                parent = int(parents[joint])
                if parent >= 0:
                    cv2.line(image, tuple(pixels[parent]), tuple(pixels[joint]), (111, 205, 255), 4, cv2.LINE_AA)
            for point in pixels:
                cv2.circle(image, tuple(point), 5, (246, 248, 252), -1, cv2.LINE_AA)
            cv2.putText(image, "MAMMA two-view body fit", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (237, 241, 247), 2, cv2.LINE_AA)
            cv2.putText(image, "Experimental calibration", (28, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (134, 172, 210), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.vertices.resolve(strict=True), arguments.skeleton.resolve(strict=True), arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
