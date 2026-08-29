"""Bounded, source-landmark-driven correction for a GNM face performance.

The semantic video retargeter is a useful prior, but it cannot preserve a
performer's exact visible mouth, eyelid, and brow geometry.  This module keeps
that prior and solves a small per-frame inverse problem against the observed
GNM-68 landmarks.  Identity remains fixed; only GNM expression coefficients
and a review-only weak-perspective camera are estimated.

The result is intentionally a 2D visual match.  A monocular RGB track does not
provide ground-truth facial depth, tongue contact, or occluded skin motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .fitting import project_landmarks, rotation_matrix
from .gnm_adapter import GNMAdapter


@dataclass(frozen=True, slots=True)
class VisualFaceRetargetConfig:
    coefficient_bound: float = 3.0
    base_prior: float = 0.75
    temporal_prior: float = 0.35
    camera_iterations: int = 2
    camera_max_evaluations: int = 80

    def __post_init__(self) -> None:
        if not np.isfinite(self.coefficient_bound) or self.coefficient_bound <= 0:
            raise ValueError("coefficient_bound must be finite and positive")
        if not np.isfinite(self.base_prior) or self.base_prior <= 0:
            raise ValueError("base_prior must be finite and positive")
        if not np.isfinite(self.temporal_prior) or self.temporal_prior < 0:
            raise ValueError("temporal_prior must be finite and non-negative")
        if self.camera_iterations < 1 or self.camera_max_evaluations < 1:
            raise ValueError("camera solve limits must be positive")


@dataclass(frozen=True, slots=True)
class VisualFaceRetargetResult:
    expression: np.ndarray
    cameras: np.ndarray
    base_nme: np.ndarray
    corrected_nme: np.ndarray
    base_region_nme: np.ndarray
    corrected_region_nme: np.ndarray
    bound_fraction: np.ndarray
    report: dict[str, Any]


REGION_NAMES = ("brows", "eyes", "outer_mouth", "inner_mouth")
REGION_SLICES = (slice(17, 27), slice(36, 48), slice(48, 60), slice(60, 68))


def _camera_weights() -> np.ndarray:
    weights = np.zeros(68, dtype=np.float64)
    weights[:17] = 0.30
    weights[27:36] = 3.00
    weights[36:48] = 2.00
    return weights


def _expression_weights() -> np.ndarray:
    weights = np.ones(68, dtype=np.float64)
    weights[:17] = 0.50
    weights[17:27] = 3.00
    weights[27:36] = 1.50
    weights[36:48] = 4.00
    weights[48:60] = 5.00
    weights[60:68] = 10.00
    return weights


def _initial_camera(points: np.ndarray, observed: np.ndarray) -> np.ndarray:
    observed_eye = float(np.linalg.norm(observed[36] - observed[45]))
    model_eye = float(np.linalg.norm(points[36, :2] - points[45, :2]))
    scale = observed_eye / max(model_eye, 1.0e-8)
    camera = np.asarray(
        (0.0, 0.0, 0.0, np.log(max(scale, 1.0e-8)), 0.0, 0.0),
        dtype=np.float64,
    )
    camera[4:] = observed[27:36].mean(axis=0) - project_landmarks(
        points[27:36], camera
    ).mean(axis=0)
    return camera


def _fit_camera(
    points: np.ndarray,
    observed: np.ndarray,
    image_size: tuple[int, int],
    initial: np.ndarray | None,
    *,
    maximum_evaluations: int,
) -> np.ndarray:
    height, width = image_size
    interocular = max(float(np.linalg.norm(observed[36] - observed[45])), 1.0)
    weights = _camera_weights()
    start = _initial_camera(points, observed) if initial is None else initial
    low = np.asarray(
        (-0.8, -0.8, -0.6, np.log(1.0e-3), -2.0 * width, -2.0 * height),
        dtype=np.float64,
    )
    high = np.asarray(
        (0.8, 0.8, 0.6, np.log(1.0e5), 3.0 * width, 3.0 * height),
        dtype=np.float64,
    )

    def residual(camera: np.ndarray) -> np.ndarray:
        return (
            np.sqrt(weights[:, None])
            * (project_landmarks(points, camera) - observed)
            / interocular
        ).ravel()

    return least_squares(
        residual,
        np.clip(start, low + 1.0e-9, high - 1.0e-9),
        bounds=(low, high),
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=maximum_evaluations,
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
    ).x


def _projected_expression_basis(
    basis: np.ndarray, camera: np.ndarray
) -> np.ndarray:
    rotation = rotation_matrix(*camera[:3])
    scale = float(np.exp(camera[3]))
    rotated = np.einsum("ijk,lk->ijl", basis, rotation, optimize=True)
    projected = np.stack(
        (scale * rotated[:, :, 0], -scale * rotated[:, :, 1]), axis=2
    )
    return projected.transpose(1, 2, 0).reshape(136, basis.shape[0])


def _nme(
    projected: np.ndarray, observed: np.ndarray
) -> tuple[float, np.ndarray]:
    interocular = max(float(np.linalg.norm(observed[36] - observed[45])), 1.0)
    distance = np.linalg.norm(projected - observed, axis=1) / interocular
    regions = np.asarray(
        [float(np.mean(distance[region])) for region in REGION_SLICES],
        dtype=np.float32,
    )
    return float(np.mean(distance)), regions


def solve_visual_face_track(
    adapter: GNMAdapter,
    *,
    identity: np.ndarray,
    base_expression: np.ndarray,
    observed_landmarks: np.ndarray,
    image_size: tuple[int, int],
    detected: np.ndarray | None = None,
    config: VisualFaceRetargetConfig = VisualFaceRetargetConfig(),
) -> VisualFaceRetargetResult:
    """Correct a semantic GNM track toward exact observed 2D face geometry."""

    identity = np.asarray(identity, dtype=np.float32)
    base = np.asarray(base_expression, dtype=np.float32)
    observed = np.asarray(observed_landmarks, dtype=np.float64)
    if identity.shape != (adapter.identity_dim,) or not np.isfinite(identity).all():
        raise ValueError("identity must be one finite GNM identity vector")
    if base.ndim != 2 or base.shape[1] != adapter.expression_dim:
        raise ValueError("base_expression must have shape [frames,expression_dim]")
    if observed.shape != (len(base), 68, 2) or not np.isfinite(observed).all():
        raise ValueError("observed_landmarks must be finite with shape [frames,68,2]")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must be positive (height,width)")
    observed_mask = (
        np.ones(len(base), dtype=bool)
        if detected is None
        else np.asarray(detected, dtype=bool)
    )
    if observed_mask.shape != (len(base),):
        raise ValueError("detected must have one value per frame")

    neutral = adapter.compact_template.astype(np.float64) + np.einsum(
        "i,ijk->jk",
        identity.astype(np.float64),
        adapter.compact_identity_basis.astype(np.float64),
        optimize=True,
    )
    basis = adapter.compact_expression_basis.astype(np.float64)
    output = base.astype(np.float64).copy()
    cameras = np.zeros((len(base), 6), dtype=np.float64)
    base_nme = np.full(len(base), np.nan, dtype=np.float32)
    corrected_nme = np.full(len(base), np.nan, dtype=np.float32)
    base_regions = np.full((len(base), len(REGION_NAMES)), np.nan, dtype=np.float32)
    corrected_regions = np.full_like(base_regions, np.nan)
    bound_fraction = np.zeros(len(base), dtype=np.float32)
    expression_weights = np.repeat(np.sqrt(_expression_weights()), 2)
    identity_matrix = np.eye(adapter.expression_dim, dtype=np.float64)
    previous_expression: np.ndarray | None = None
    previous_camera: np.ndarray | None = None

    for frame in range(len(base)):
        if not observed_mask[frame]:
            if previous_expression is not None:
                output[frame] = previous_expression
            if previous_camera is not None:
                cameras[frame] = previous_camera
            continue
        target = observed[frame]
        base_points = neutral + np.einsum(
            "i,ijk->jk", base[frame], basis, optimize=True
        )
        base_camera = _fit_camera(
            base_points,
            target,
            image_size,
            previous_camera,
            maximum_evaluations=config.camera_max_evaluations,
        )
        base_nme[frame], base_regions[frame] = _nme(
            project_landmarks(base_points, base_camera), target
        )
        expression = base[frame].astype(np.float64).copy()
        camera = base_camera
        for _ in range(config.camera_iterations):
            design = _projected_expression_basis(basis, camera)
            target_delta = (target - project_landmarks(neutral, camera)).reshape(-1)
            matrices = [expression_weights[:, None] * design]
            values = [expression_weights * target_delta]
            matrices.append(np.sqrt(config.base_prior) * identity_matrix)
            values.append(np.sqrt(config.base_prior) * base[frame])
            if previous_expression is not None and config.temporal_prior > 0.0:
                matrices.append(np.sqrt(config.temporal_prior) * identity_matrix)
                values.append(np.sqrt(config.temporal_prior) * previous_expression)
            expression = lsq_linear(
                np.vstack(matrices),
                np.concatenate(values),
                bounds=(-config.coefficient_bound, config.coefficient_bound),
                method="trf",
                lsmr_tol="auto",
                max_iter=120,
            ).x
            points = neutral + np.einsum(
                "i,ijk->jk", expression, basis, optimize=True
            )
            camera = _fit_camera(
                points,
                target,
                image_size,
                camera,
                maximum_evaluations=config.camera_max_evaluations,
            )
        output[frame] = expression
        cameras[frame] = camera
        corrected_nme[frame], corrected_regions[frame] = _nme(
            project_landmarks(points, camera), target
        )
        bound_fraction[frame] = float(
            np.mean(np.abs(expression) >= config.coefficient_bound - 1.0e-4)
        )
        previous_expression = expression
        previous_camera = camera

    valid = observed_mask & np.isfinite(corrected_nme)
    if not np.any(valid):
        raise ValueError("visual face solve needs at least one detected frame")
    report: dict[str, Any] = {
        "schema_version": "autoanim.visual-face-retarget/1.0",
        "status": "research_preview",
        "match_space": "mediapipe478_to_gnm68_source_pixels_weak_perspective",
        "frame_count": int(len(base)),
        "observed_frame_count": int(np.count_nonzero(valid)),
        "identity_fixed": True,
        "base_nme_mean": float(np.mean(base_nme[valid])),
        "corrected_nme_mean": float(np.mean(corrected_nme[valid])),
        "corrected_nme_p95": float(np.percentile(corrected_nme[valid], 95)),
        "improvement_fraction": float(
            1.0 - np.mean(corrected_nme[valid]) / max(np.mean(base_nme[valid]), 1.0e-8)
        ),
        "regions": {
            name: {
                "base_mean": float(np.mean(base_regions[valid, index])),
                "corrected_mean": float(np.mean(corrected_regions[valid, index])),
                "corrected_p95": float(
                    np.percentile(corrected_regions[valid, index], 95)
                ),
            }
            for index, name in enumerate(REGION_NAMES)
        },
        "coefficient_bound": config.coefficient_bound,
        "maximum_frame_bound_fraction": float(np.max(bound_fraction[valid])),
        "limitations": [
            "This is a measured 2D landmark match, not an exact 3D facial-surface reconstruction.",
            "Monocular MediaPipe observations do not measure tongue contact, occluded skin, or metric depth.",
            "Only the GNM-68 landmark locations are constrained; between-landmark surface fidelity needs separate validation.",
        ],
    }
    return VisualFaceRetargetResult(
        expression=output.astype(np.float32),
        cameras=cameras.astype(np.float32),
        base_nme=base_nme,
        corrected_nme=corrected_nme,
        base_region_nme=base_regions,
        corrected_region_nme=corrected_regions,
        bound_fraction=bound_fraction,
        report=report,
    )


__all__ = [
    "REGION_NAMES",
    "VisualFaceRetargetConfig",
    "VisualFaceRetargetResult",
    "solve_visual_face_track",
]
