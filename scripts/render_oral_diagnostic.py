#!/usr/bin/env python3
"""Render a close-up contact sheet of actual GNM oral geometry.

This is a geometry-inspection aid, not a perceptual, dental-contact, or
collision validator.  It deliberately removes head pose so selected frames
stay registered while the lips, mouth sock, teeth/gums, and tongue deform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# GNM is intentionally vendored as a namespace package at the repository root.
# Direct script execution otherwise places only ``scripts/`` on sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from autoanim_gnm.gnm_adapter import GNMAdapter


SURFACES = (
    ("skin exterior", "skin_exterior", (178, 142, 126), False),
    ("mouth sock", "mouth_sock", (92, 47, 50), True),
    ("upper teeth/gums", "upper_teeth_and_gums", (164, 92, 96), True),
    ("lower teeth/gums", "lower_teeth_and_gums", (164, 92, 96), True),
    ("tongue", "tongue", (154, 69, 82), True),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Succeeded audio job")
    parser.add_argument(
        "--output", "--out", type=Path, required=True, help="Output PNG path"
    )
    parser.add_argument("--frames", type=int, default=6, help="Panels (2-12)")
    parser.add_argument("--size", type=int, default=320, help="Panel size (160-960)")
    args = parser.parse_args()
    if not 2 <= args.frames <= 12:
        parser.error("--frames must be in [2,12]")
    if not 160 <= args.size <= 960:
        parser.error("--size must be in [160,960]")
    return args


def _greedy_frames(score: np.ndarray, count: int) -> list[int]:
    """Keep frame zero plus separated, high-activity oral frames."""

    selected = [0]
    minimum_separation = max(2, len(score) // max(4 * count, 1))
    for candidate in np.argsort(score)[::-1]:
        index = int(candidate)
        if all(abs(index - existing) >= minimum_separation for existing in selected):
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.linspace(0, len(score) - 1, count, dtype=np.int32):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return selected


class OralRenderer:
    """Small deterministic painter for GNM's official oral surface groups."""

    def __init__(self, adapter: GNMAdapter, size: int):
        self.adapter = adapter
        self.size = size
        model = adapter.model
        neutral_landmarks = adapter.landmarks()
        mouth = neutral_landmarks[48:68, :2]
        self.center = (mouth.min(axis=0) + mouth.max(axis=0)) * 0.5
        mouth_width = float(np.ptp(mouth[:, 0]))
        self.scale = 0.70 * size / mouth_width
        self.light = np.asarray((-0.25, 0.45, 1.0), dtype=np.float32)
        self.light /= np.linalg.norm(self.light)

        teeth = np.zeros(model.num_vertices, dtype=bool)
        teeth[np.asarray(model.vertex_group_indices("teeth"), dtype=np.int32)] = True
        triangles: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        double_sided: list[np.ndarray] = []
        for label, group_name, rgb, is_double_sided in SURFACES:
            indices = np.asarray(model.triangle_indices_for_group(group_name), dtype=np.int32)
            group_triangles = adapter.triangles[indices]
            vertex_rgb = np.broadcast_to(
                np.asarray(rgb, dtype=np.float32),
                (model.num_vertices, 3),
            ).copy()
            if "teeth/gums" in label:
                vertex_rgb[teeth] = np.asarray((229, 223, 207), dtype=np.float32)
            triangle_rgb = vertex_rgb[group_triangles].mean(axis=1)
            triangles.append(group_triangles)
            colors.append(triangle_rgb[:, ::-1])  # OpenCV BGR
            double_sided.append(
                np.full(len(group_triangles), is_double_sided, dtype=bool)
            )
        self.triangles = np.concatenate(triangles)
        self.base_bgr = np.concatenate(colors)
        self.double_sided = np.concatenate(double_sided)

    def _project(self, vertices: np.ndarray) -> np.ndarray:
        output = np.empty((len(vertices), 2), dtype=np.float32)
        output[:, 0] = (
            (vertices[:, 0] - self.center[0]) * self.scale + self.size / 2
        )
        output[:, 1] = (
            -(vertices[:, 1] - self.center[1]) * self.scale + self.size / 2
        )
        return output

    def render(self, vertices: np.ndarray) -> np.ndarray:
        projected = self._project(vertices)
        points = projected[self.triangles]
        area = (
            (points[:, 1, 0] - points[:, 0, 0])
            * (points[:, 2, 1] - points[:, 0, 1])
            - (points[:, 1, 1] - points[:, 0, 1])
            * (points[:, 2, 0] - points[:, 0, 0])
        )
        keep = (area < 0.0) | self.double_sided
        triangles = self.triangles[keep]
        points = points[keep]
        base_bgr = self.base_bgr[keep]
        edges_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        edges_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        normals = np.cross(edges_a, edges_b)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        diffuse = np.abs(normals @ self.light)
        colors = np.clip(
            base_bgr * (0.38 + 0.62 * diffuse[:, None]), 0, 255
        ).astype(np.uint8)
        depth = vertices[triangles].mean(axis=1)[:, 2]
        canvas = np.full((self.size, self.size, 3), (22, 20, 24), dtype=np.uint8)
        for index in np.argsort(depth):
            polygon = np.rint(points[index]).astype(np.int32)
            cv2.fillConvexPoly(
                canvas, polygon, colors[index].tolist(), lineType=cv2.LINE_AA
            )
        return canvas


def _oral_metrics(
    adapter: GNMAdapter, expression: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tongue_mask = adapter.vertex_group("tongue") > 0.5
    tongue_basis = np.asarray(
        adapter.model.expression_basis[350:382, tongue_mask], dtype=np.float32
    )
    tongue_delta = np.einsum(
        "tc,cvj->tvj", expression[:, 350:382], tongue_basis, optimize=True
    )
    tongue_max_mm = np.max(np.linalg.norm(tongue_delta, axis=3 - 1), axis=1) * 1000.0

    pairs = np.asarray(((61, 67), (62, 66), (63, 65)), dtype=np.int32)
    landmark_indices = np.unique(pairs)
    landmarks = adapter.compact_template[landmark_indices] + np.einsum(
        "tc,cvj->tvj",
        expression,
        adapter.compact_expression_basis[:, landmark_indices],
        optimize=True,
    )
    remap = {int(value): index for index, value in enumerate(landmark_indices)}
    upper = landmarks[:, [remap[int(pair[0])] for pair in pairs]]
    lower = landmarks[:, [remap[int(pair[1])] for pair in pairs]]
    lip_gap_mm = np.mean(np.linalg.norm(upper - lower, axis=2), axis=1) * 1000.0
    return tongue_max_mm.astype(np.float32), lip_gap_mm.astype(np.float32)


def _normalized(values: np.ndarray) -> np.ndarray:
    span = float(np.ptp(values))
    if span <= 1e-8:
        return np.zeros_like(values)
    return (values - float(values.min())) / span


def main() -> int:
    args = _parse_args()
    controls_path = args.job / "controls.npz"
    if not controls_path.is_file():
        raise FileNotFoundError(f"missing retained controls: {controls_path}")
    with np.load(controls_path, allow_pickle=False) as controls:
        expression = np.asarray(controls["expression"], dtype=np.float32)
        timestamps = np.asarray(controls["timestamps"], dtype=np.float32)
    if expression.shape != (len(timestamps), 383) or not np.isfinite(expression).all():
        raise ValueError(f"invalid GNM expression track: {expression.shape}")

    adapter = GNMAdapter()
    tongue_max_mm, lip_gap_mm = _oral_metrics(adapter, expression)
    score = 0.60 * _normalized(tongue_max_mm) + 0.40 * _normalized(lip_gap_mm)
    frames = _greedy_frames(score, args.frames)
    renderer = OralRenderer(adapter, args.size)

    columns = min(3, len(frames))
    rows = (len(frames) + columns - 1) // columns
    header_height = 74
    sheet = np.full(
        (header_height + rows * args.size, columns * args.size, 3),
        (26, 24, 28),
        dtype=np.uint8,
    )
    cv2.putText(
        sheet,
        "GNM ORAL GEOMETRY - DIAGNOSTIC ONLY",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (242, 238, 232),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "Actual lips / mouth sock / teeth+gums / tongue. No collision or perceptual validation.",
        (14, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (177, 169, 182),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "Head pose removed for registered comparison",
        (14, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (137, 130, 143),
        1,
        cv2.LINE_AA,
    )

    selected_metrics: list[dict[str, float | int]] = []
    for panel_index, frame in enumerate(frames):
        vertices = adapter.mesh(expression=expression[frame])
        panel = renderer.render(vertices)
        label = (
            f"f{frame}  {timestamps[frame]:.2f}s  "
            f"tongue {tongue_max_mm[frame]:.2f}mm  gap {lip_gap_mm[frame]:.2f}mm"
        )
        cv2.rectangle(panel, (0, 0), (args.size, 28), (16, 15, 18), -1)
        cv2.putText(
            panel,
            label,
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.30, args.size / 900.0),
            (242, 238, 232),
            1,
            cv2.LINE_AA,
        )
        row, column = divmod(panel_index, columns)
        y = header_height + row * args.size
        x = column * args.size
        sheet[y : y + args.size, x : x + args.size] = panel
        selected_metrics.append(
            {
                "frame": frame,
                "timestamp_seconds": float(timestamps[frame]),
                "isolated_tongue_max_mm": float(tongue_max_mm[frame]),
                "inner_lip_pair_gap_mm": float(lip_gap_mm[frame]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise OSError(f"could not write {args.output}")
    print(
        "DIAGNOSTIC ONLY - no collision, occlusion correctness, or perceptual "
        "quality validation."
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "job": str(args.job.resolve()),
                "surfaces": [surface[0] for surface in SURFACES],
                "selected": selected_metrics,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
