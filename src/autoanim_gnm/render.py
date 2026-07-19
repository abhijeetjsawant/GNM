"""Deterministic CPU previews for the untextured GNM mesh."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .gnm_adapter import GNMAdapter


class MeshRenderer:
    def __init__(
        self,
        adapter: GNMAdapter,
        size: int = 640,
        *,
        identity: np.ndarray | None = None,
    ):
        self.adapter = adapter
        self.size = int(size)
        self.identity = None if identity is None else np.asarray(identity, dtype=np.float32)
        neutral = adapter.mesh(identity=self.identity)
        skin = adapter.vertex_group("skin_exterior") > 0.5
        xy = neutral[skin, :2]
        self.center = (xy.min(axis=0) + xy.max(axis=0)) / 2
        self.scale = 0.85 * self.size / float(np.ptp(xy[:, 1]))
        triangles = adapter.triangles
        self.triangles = triangles[np.all(skin[triangles], axis=1)]
        self.light = np.asarray((-0.3, 0.5, 1.0), dtype=np.float32)
        self.light /= np.linalg.norm(self.light)
        self.base_bgr = np.asarray((190, 180, 170), dtype=np.float32)

    def project(self, vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float32)
        output = np.empty((len(vertices), 2), dtype=np.float32)
        output[:, 0] = (vertices[:, 0] - self.center[0]) * self.scale + self.size / 2
        output[:, 1] = -(vertices[:, 1] - self.center[1]) * self.scale + self.size / 2
        return output

    def render(self, vertices: np.ndarray, landmarks: np.ndarray | None = None) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float32)
        projected = self.project(vertices)
        triangles = self.triangles
        points = projected[triangles]
        area = (
            (points[:, 1, 0] - points[:, 0, 0])
            * (points[:, 2, 1] - points[:, 0, 1])
            - (points[:, 1, 1] - points[:, 0, 1])
            * (points[:, 2, 0] - points[:, 0, 0])
        )
        keep = area < 0
        triangles = triangles[keep]
        points = points[keep]
        edges_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
        edges_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        normals = np.cross(edges_a, edges_b)
        normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normal_length, 1e-8)
        diffuse = np.maximum(normals @ self.light, 0)
        intensity = 0.35 + 0.65 * diffuse
        colors = np.clip(self.base_bgr[None, :] * intensity[:, None], 0, 255).astype(np.uint8)
        depth = vertices[triangles].mean(axis=1)[:, 2]
        order = np.argsort(depth)
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for index in order:
            polygon = np.rint(points[index]).astype(np.int32)
            cv2.fillConvexPoly(canvas, polygon, colors[index].tolist(), lineType=cv2.LINE_AA)
        if landmarks is not None:
            landmark_pixels = np.rint(self.project(np.asarray(landmarks))).astype(np.int32)
            for x, y in landmark_pixels:
                if 0 <= x < self.size and 0 <= y < self.size:
                    cv2.circle(canvas, (int(x), int(y)), 2, (45, 45, 45), -1, cv2.LINE_AA)
        return canvas

    def save_png(
        self,
        path: str | Path,
        vertices: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.render(vertices, landmarks)):
            raise OSError(f"Could not write preview: {path}")
        return path
