"""TensorFlow-free evaluation of GNM's checked-in expression decoder."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np


EXPRESSION_NAMES = (
    "surprise",
    "disgust",
    "suck",
    "compress_face",
    "stretch_face",
    "happy",
    "squint",
    "platysma",
    "blow",
    "funneler",
    "smile_wide",
    "corners_down",
    "pucker",
    "wink_left",
    "wink_right",
    "mouth_left",
    "mouth_right",
    "lips_roll_in",
    "snarl",
    "tongue_center",
)

_LAYERS = ("dense_13", "dense_14", "dense_15", "dense_16", "dense_17")


class ExpressionDecoder:
    """Evaluates the 64-latent/20-class Keras decoder with NumPy."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._weights: list[tuple[np.ndarray, np.ndarray]] = []
        with h5py.File(self.model_path, "r") as handle:
            for layer in _LAYERS:
                group = handle[f"model_weights/{layer}/{layer}"]
                kernel = np.asarray(group["kernel:0"], dtype=np.float32)
                bias = np.asarray(group["bias:0"], dtype=np.float32)
                self._weights.append((kernel, bias))
        widths = [self._weights[0][0].shape[0]]
        widths.extend(kernel.shape[1] for kernel, _ in self._weights)
        if widths != [84, 64, 128, 256, 512, 383]:
            raise ValueError(f"Unexpected expression decoder graph: {widths}")

    @property
    def output_dim(self) -> int:
        return 383

    def decode(
        self,
        latent: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        latent = np.asarray(latent, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.float32)
        if latent.shape[-1:] != (64,) or classes.shape[-1:] != (20,):
            raise ValueError(
                f"Expected [...,64] latent and [...,20] classes; got "
                f"{latent.shape} and {classes.shape}"
            )
        batch_shape = np.broadcast_shapes(latent.shape[:-1], classes.shape[:-1])
        latent = np.broadcast_to(latent, batch_shape + (64,))
        classes = np.broadcast_to(classes, batch_shape + (20,))
        value = np.concatenate([latent, classes], axis=-1)
        for index, (kernel, bias) in enumerate(self._weights):
            value = value @ kernel + bias
            if index < len(self._weights) - 1:
                value = np.maximum(value, 0)
        return np.asarray(value, dtype=np.float32)

    def prototype(self, name: str) -> np.ndarray:
        try:
            index = EXPRESSION_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown expression prototype: {name}") from exc
        classes = np.zeros(20, dtype=np.float32)
        classes[index] = 1.0
        return self.decode(np.zeros(64, dtype=np.float32), classes)

    def blend(self, weights: Mapping[str, float]) -> np.ndarray:
        result = np.zeros(self.output_dim, dtype=np.float32)
        for name, weight in weights.items():
            result += np.float32(weight) * self.prototype(name)
        return result
