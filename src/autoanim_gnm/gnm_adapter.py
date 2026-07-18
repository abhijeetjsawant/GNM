"""Narrow, validated adapter around Google's official GNM NumPy API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gnm.shape import gnm_landmarks, gnm_numpy
from gnm.shape.data.versions import gnm_specs


EXPECTED = {
    "vertices": 17_821,
    "triangles": 35_324,
    "quads": 17_662,
    "identity": 253,
    "expression": 383,
    "joints": 4,
}


class GNMAdapter:
    """Loads GNM Head v3 and exposes compact landmarks and safe exports."""

    def __init__(self) -> None:
        self.model = gnm_numpy.GNM.from_local(
            gnm_specs.GNMMajorVersion.V3,
            gnm_specs.GNMVariant.HEAD,
        )
        actual = {
            "vertices": self.model.num_vertices,
            "triangles": len(self.model.triangles),
            "quads": len(self.model.quads),
            "identity": self.model.identity_dim,
            "expression": self.model.expression_dim,
            "joints": self.model.num_joints,
        }
        if actual != EXPECTED or self.model.version.value != "3.0":
            raise RuntimeError(f"Unexpected GNM asset: {actual}, {self.model.version}")
        config = gnm_landmarks.load_landmarks(
            gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68
        )
        self.landmark_indices = np.asarray(config.indices, dtype=np.int32)
        self.landmark_weights = np.asarray(config.weights, dtype=np.float32)
        self._compact_template = self._apply_landmark_regressor(
            np.asarray(self.model.template_vertex_positions)
        )
        self._compact_identity = self._basis_landmarks(
            np.asarray(self.model.vertex_identity_basis)
        )
        self._compact_expression = self._basis_landmarks(
            np.asarray(self.model.expression_basis)
        )

    @property
    def identity_dim(self) -> int:
        return self.model.identity_dim

    @property
    def expression_dim(self) -> int:
        return self.model.expression_dim

    @property
    def triangles(self) -> np.ndarray:
        return np.asarray(self.model.triangles, dtype=np.int32)

    @property
    def compact_template(self) -> np.ndarray:
        return self._compact_template.copy()

    @property
    def compact_identity_basis(self) -> np.ndarray:
        return self._compact_identity.copy()

    @property
    def compact_expression_basis(self) -> np.ndarray:
        return self._compact_expression.copy()

    def _apply_landmark_regressor(self, vertices: np.ndarray) -> np.ndarray:
        selected = vertices[..., self.landmark_indices, :]
        return np.sum(selected * self.landmark_weights[..., None], axis=-2)

    def _basis_landmarks(self, basis: np.ndarray) -> np.ndarray:
        selected = basis[:, self.landmark_indices, :]
        return np.sum(
            selected * self.landmark_weights[None, ..., None], axis=-2
        ).astype(np.float32)

    def mesh(
        self,
        identity: np.ndarray | None = None,
        expression: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> np.ndarray:
        vertices = self.model(
            identity=identity,
            expression=expression,
            rotations=rotations,
            translation=translation,
        )
        vertices = np.asarray(vertices, dtype=np.float32)
        if not np.isfinite(vertices).all():
            raise ValueError("GNM generated nonfinite vertices")
        return vertices

    def landmarks(
        self,
        identity: np.ndarray | None = None,
        expression: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> np.ndarray:
        _, landmarks = self.model.vertices_and_landmarks(
            gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68,
            identity=identity,
            expression=expression,
            rotations=rotations,
            translation=translation,
        )
        return np.asarray(landmarks, dtype=np.float32)

    def vertex_group(self, name: str) -> np.ndarray:
        try:
            index = list(self.model.vertex_group_names).index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return np.asarray(self.model.vertex_groups[index], dtype=np.float32)

    def export_obj(self, path: str | Path, vertices: np.ndarray) -> Path:
        path = Path(path)
        vertices = np.asarray(vertices)
        if vertices.shape != (self.model.num_vertices, 3):
            raise ValueError(f"Unexpected vertex shape: {vertices.shape}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# AutoAnim GNM Head 3.0\n")
            for x, y, z in vertices:
                handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
            for a, b, c in self.triangles + 1:
                handle.write(f"f {a} {b} {c}\n")
        return path
