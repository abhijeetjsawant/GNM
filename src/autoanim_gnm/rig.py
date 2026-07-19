"""Deterministic semantic controls constrained to GNM expression regions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .gnm_adapter import GNMAdapter
from .semantic_decoder import ExpressionDecoder


VISEME_DEFINITIONS: dict[str, dict[str, float]] = {
    "X": {},
    "A": {"compress_face": -0.35},
    "B": {"stretch_face": 0.25, "smile_wide": 0.15},
    "C": {"stretch_face": 0.60},
    "D": {"stretch_face": 1.00},
    "E": {"funneler": 0.65},
    "F": {"pucker": 0.70, "funneler": 0.30},
    "G": {"lips_roll_in": -0.35},
    "H": {"stretch_face": 0.50, "tongue_center": 0.35},
}

EMOTION_DEFINITIONS: dict[str, dict[str, float]] = {
    "neutral": {},
    "joy": {"happy": 0.70, "smile_wide": 0.30},
    "surprise": {"surprise": 1.0},
    "disgust": {"disgust": 1.0},
    "sad": {"corners_down": 0.75, "compress_face": 0.25},
    "anger": {"snarl": 0.50, "platysma": 0.30, "compress_face": 0.20},
    "fear": {"surprise": 0.55, "compress_face": 0.45},
    "contempt": {"snarl": 0.60, "mouth_left": 0.40},
}


class ControlRig:
    def __init__(
        self,
        adapter: GNMAdapter,
        decoder: ExpressionDecoder,
        *,
        identity: np.ndarray | None = None,
    ):
        self.adapter = adapter
        self.decoder = decoder
        if decoder.output_dim != adapter.expression_dim:
            raise ValueError("Decoder and GNM expression dimensions differ")
        identity_value = (
            np.zeros(adapter.identity_dim, dtype=np.float32)
            if identity is None
            else np.asarray(identity, dtype=np.float32).copy()
        )
        if identity_value.shape != (adapter.identity_dim,) or not np.isfinite(identity_value).all():
            raise ValueError(f"identity must be one finite ({adapter.identity_dim},) vector")
        identity_value.setflags(write=False)
        self.identity = identity_value
        self._prototype_cache: dict[str, np.ndarray] = {}
        neutral_landmarks = self.adapter.compact_template + np.einsum(
            "i,ijk->jk",
            self.identity,
            self.adapter.compact_identity_basis,
            optimize=True,
        )
        neutral_landmarks = np.asarray(neutral_landmarks, dtype=np.float32)
        neutral_landmarks.setflags(write=False)
        self.neutral_landmarks = neutral_landmarks
        self._interocular_distance = float(np.linalg.norm(neutral_landmarks[36] - neutral_landmarks[45]))
        if self._interocular_distance <= 0:
            raise ValueError("GNM interocular distance is invalid")

    def _blend(self, definition: Mapping[str, float]) -> np.ndarray:
        output = np.zeros(self.adapter.expression_dim, dtype=np.float32)
        for name, weight in definition.items():
            if name not in self._prototype_cache:
                self._prototype_cache[name] = self.decoder.prototype(name)
            output += np.float32(weight) * self._prototype_cache[name]
        return np.clip(output, -3.0, 3.0)

    def viseme(self, cue: str) -> np.ndarray:
        if cue not in VISEME_DEFINITIONS:
            raise KeyError(f"Unknown Rhubarb cue: {cue}")
        output = self._blend(VISEME_DEFINITIONS[cue])
        masked = np.zeros_like(output)
        masked[200:350] = output[200:350]
        if cue == "H":
            tongue = self.decoder.prototype("tongue_center")
            masked[350:382] = np.clip(0.70 * tongue[350:382], -3.0, 3.0)
        return masked

    def emotion(self, name: str) -> np.ndarray:
        if name not in EMOTION_DEFINITIONS:
            raise KeyError(f"Unknown emotion: {name}")
        output = self._blend(EMOTION_DEFINITIONS[name])
        masked = np.zeros_like(output)
        masked[:350] = output[:350]
        return masked

    def blink(self) -> np.ndarray:
        """Deterministic bilateral blink isolated to the two eye regions."""

        left = self.decoder.prototype("wink_left")
        right = self.decoder.prototype("wink_right")
        output = np.zeros(self.adapter.expression_dim, dtype=np.float32)
        output[:100] = left[:100]
        output[100:200] = right[100:200]
        return self._bound_regions(output)[0]

    def _bound_regions(self, value: np.ndarray) -> tuple[np.ndarray, bool]:
        output = np.asarray(value, dtype=np.float32).copy()
        saturated = False
        for start, end in ((0, 100), (100, 200), (200, 350), (350, 382), (382, 383)):
            maximum = float(np.max(np.abs(output[start:end]), initial=0.0))
            if maximum > 3.0:
                output[start:end] *= np.float32(3.0 / maximum)
                saturated = True
        return output, saturated

    def compose(
        self,
        viseme: np.ndarray,
        emotion: np.ndarray,
        *,
        mouth_activity: float | None = None,
        emotion_strength: float = 1.0,
    ) -> tuple[np.ndarray, bool]:
        speech = np.asarray(viseme, dtype=np.float32)
        affect = np.asarray(emotion, dtype=np.float32) * np.float32(emotion_strength)
        if mouth_activity is None:
            raw = speech + affect
        else:
            activity = float(np.clip(mouth_activity, 0.0, 1.0))
            raw = speech.copy()
            # Upper face keeps the emotional performance. Speech-critical
            # lower face receives only a restrained residual and the tongue
            # remains wholly owned by speech.
            raw[:200] += affect[:200]
            raw[200:350] += affect[200:350] * np.float32(0.14 * (1.0 - 0.65 * activity))
        return self._bound_regions(raw)

    def compact_landmarks(self, expression: np.ndarray) -> np.ndarray:
        expression = np.asarray(expression, dtype=np.float32)
        return self.neutral_landmarks + np.einsum(
            "i,ijk->jk", expression, self.adapter.compact_expression_basis, optimize=True
        )

    def mouth_step_ratio(self, left: np.ndarray, right: np.ndarray) -> float:
        before = self.compact_landmarks(left)[48:68]
        after = self.compact_landmarks(right)[48:68]
        return float(np.max(np.linalg.norm(after - before, axis=1)) / self._interocular_distance)

    def limit_mouth_step(
        self,
        previous: np.ndarray,
        target: np.ndarray,
        maximum_ratio: float = 0.04,
    ) -> tuple[np.ndarray, bool]:
        """Cap lower-face/tongue motion in a perceptual landmark scale."""

        previous = np.asarray(previous, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        candidate = previous.copy()
        candidate[:200] = target[:200]
        candidate[200:382] = target[200:382]
        distance = self.mouth_step_ratio(previous, candidate)
        if distance <= maximum_ratio or distance <= 1e-12:
            return target.copy(), False
        alpha = np.float32(maximum_ratio / distance)
        output = target.copy()
        output[200:382] = previous[200:382] + alpha * (target[200:382] - previous[200:382])
        return output, True

    def geometry_metrics(self, expression: np.ndarray) -> dict[str, float]:
        landmarks = self.adapter.landmarks(identity=self.identity, expression=expression)
        aperture_pairs = ((61, 67), (62, 66), (63, 65))
        aperture = float(
            np.mean([np.linalg.norm(landmarks[a] - landmarks[b]) for a, b in aperture_pairs])
        )
        width = float(np.linalg.norm(landmarks[48] - landmarks[54]))
        neutral = self.adapter.mesh(identity=self.identity)
        posed = self.adapter.mesh(identity=self.identity, expression=expression)
        tongue = self.adapter.vertex_group("tongue") > 0
        tongue_motion = float(np.mean(np.linalg.norm(posed[tongue] - neutral[tongue], axis=1)))
        return {
            "mouth_aperture": aperture,
            "mouth_width": width,
            "tongue_motion": tongue_motion,
        }
