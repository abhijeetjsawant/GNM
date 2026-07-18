"""Robust weak-perspective fitting of visible GNM identity modes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from .gnm_adapter import GNMAdapter
from .rig import ControlRig


@dataclass(frozen=True, slots=True)
class FitResult:
    identity: np.ndarray
    nuisance: np.ndarray
    camera: np.ndarray
    fitted_landmarks: np.ndarray
    nme: float
    pixel_error: float
    saturation_fraction: float
    stability_rms: float
    confidence: str
    confidence_reasons: tuple[str, ...]


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rx = np.asarray(((1, 0, 0), (0, cp, -sp), (0, sp, cp)), dtype=np.float64)
    rz = np.asarray(((cr, -sr, 0), (sr, cr, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ rx @ ry


def project_landmarks(points: np.ndarray, camera: np.ndarray) -> np.ndarray:
    yaw, pitch, roll, log_scale, tx, ty = camera
    rotated = np.asarray(points, dtype=np.float64) @ rotation_matrix(yaw, pitch, roll).T
    scale = math.exp(float(log_scale))
    return np.column_stack((scale * rotated[:, 0] + tx, -scale * rotated[:, 1] + ty))


class IdentityFitter:
    def __init__(self, adapter: GNMAdapter, rig: ControlRig):
        self.adapter = adapter
        self.rig = rig
        self.template = adapter.compact_template.astype(np.float64)
        self.identity_basis = adapter.compact_identity_basis[:20].astype(np.float64)
        anchors = ["happy", "surprise", "pucker", "corners_down"]
        self.nuisance_coefficients = np.stack([rig.decoder.prototype(name) for name in anchors])
        self.nuisance_basis = np.einsum(
            "ae,elc->alc",
            self.nuisance_coefficients,
            adapter.compact_expression_basis,
        ).astype(np.float64)
        self.weights = np.ones(68, dtype=np.float64)
        self.weights[:17] = 0.5
        self.weights[27:48] = 2.0

    def _landmarks(self, beta: np.ndarray, nuisance: np.ndarray, modes: int) -> np.ndarray:
        return (
            self.template
            + np.einsum("i,ilc->lc", beta[:modes], self.identity_basis[:modes])
            + np.einsum("i,ilc->lc", nuisance, self.nuisance_basis)
        )

    def _initial_camera(self, observed: np.ndarray) -> np.ndarray:
        observed_eye = np.linalg.norm(observed[36] - observed[45])
        model_eye = np.linalg.norm(self.template[36, :2] - self.template[45, :2])
        scale = observed_eye / max(model_eye, 1e-8)
        camera = np.asarray((0.0, 0.0, 0.0, math.log(scale), 0.0, 0.0), dtype=np.float64)
        model_nose = project_landmarks(self.template[27:36], camera).mean(axis=0)
        observed_nose = observed[27:36].mean(axis=0)
        camera[4:] = observed_nose - model_nose
        return camera

    def _bounds(self, width: int, height: int, modes: int) -> tuple[np.ndarray, np.ndarray]:
        low = np.concatenate(
            [
                np.asarray((-.8, -.8, -.6, math.log(1e-3), -2 * width, -2 * width)),
                np.full(modes, -3.0),
                np.full(4, -1.0),
            ]
        )
        high = np.concatenate(
            [
                np.asarray((.8, .8, .6, math.log(1e5), 3 * width, 3 * width)),
                np.full(modes, 3.0),
                np.full(4, 1.0),
            ]
        )
        return low, high

    def _solve(
        self,
        observed: np.ndarray,
        width: int,
        height: int,
        modes: int,
        initial: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        interocular = float(np.linalg.norm(observed[36] - observed[45]))
        camera0 = self._initial_camera(observed) if initial is None else initial[0]

        def camera_residual(camera: np.ndarray) -> np.ndarray:
            predicted = project_landmarks(self.template, camera)
            selection = slice(27, 48)
            return (
                np.sqrt(self.weights[selection, None])
                * (predicted[selection] - observed[selection])
                / interocular
            ).ravel()

        camera_low = np.asarray((-.8, -.8, -.6, math.log(1e-3), -2 * width, -2 * width))
        camera_high = np.asarray((.8, .8, .6, math.log(1e5), 3 * width, 3 * width))
        if initial is None:
            camera_fit = least_squares(
                camera_residual,
                camera0,
                bounds=(camera_low, camera_high),
                loss="soft_l1",
                f_scale=.01,
                max_nfev=300,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            beta0 = np.zeros(modes)
            nuisance0 = np.zeros(4)
            camera0 = camera_fit.x
        else:
            beta0 = np.zeros(modes)
            beta0[: min(modes, len(initial[1]))] = initial[1][:modes]
            nuisance0 = initial[2]
        x0 = np.concatenate((camera0, beta0, nuisance0))

        def residual(values: np.ndarray) -> np.ndarray:
            camera = values[:6]
            beta = values[6 : 6 + modes]
            nuisance = values[6 + modes :]
            predicted = project_landmarks(self._landmarks(beta, nuisance, modes), camera)
            reprojection = (
                np.sqrt(self.weights[:, None]) * (predicted - observed) / interocular
            ).ravel()
            return np.concatenate((reprojection, np.sqrt(.0003) * beta, np.sqrt(.003) * nuisance))

        low, high = self._bounds(width, height, modes)
        fitted = least_squares(
            residual,
            x0,
            bounds=(low, high),
            loss="soft_l1",
            f_scale=.01,
            max_nfev=300,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        camera = fitted.x[:6]
        beta = fitted.x[6 : 6 + modes]
        nuisance = fitted.x[6 + modes :]
        return camera, beta, nuisance

    def fit(
        self,
        observed: np.ndarray,
        image_shape: tuple[int, int],
        *,
        modes: int = 20,
        face_width: float | None = None,
        compute_stability: bool = True,
    ) -> FitResult:
        observed = np.asarray(observed, dtype=np.float64)
        if observed.shape != (68, 2) or not np.isfinite(observed).all():
            raise ValueError(f"Expected finite [68,2] observations; got {observed.shape}")
        if modes not in (10, 20):
            raise ValueError("modes must be 10 or 20")
        height, width = image_shape
        camera, beta, nuisance = self._solve(observed, width, height, 10)
        if modes == 20:
            camera, beta, nuisance = self._solve(
                observed, width, height, 20, (camera, beta, nuisance)
            )
        fitted_points = project_landmarks(self._landmarks(beta, nuisance, modes), camera)
        distances = np.linalg.norm(fitted_points - observed, axis=1)
        interocular = float(np.linalg.norm(observed[36] - observed[45]))
        nme = float(np.mean(distances) / interocular)
        saturation = float(np.mean(np.abs(beta) >= 2.999))
        stability = 0.0
        if compute_stability:
            perturbations = []
            for seed in range(4):
                noisy = observed + np.random.default_rng(seed).normal(0, 0.5, observed.shape)
                _, noisy_beta, _ = self._solve(noisy, width, height, modes)
                perturbations.append(float(np.sqrt(np.mean((noisy_beta - beta) ** 2))))
            stability = float(np.mean(perturbations))
        face_width = float(face_width if face_width is not None else np.ptp(observed[:, 0]))
        yaw, pitch = np.degrees(camera[:2])
        reasons: list[str] = []
        rejected = (
            face_width < 64
            or nme > .080
            or abs(yaw) > 45
            or abs(pitch) > 35
            or saturation > .30
            or stability > 1.25
        )
        if rejected:
            confidence = "rejected"
        elif (
            face_width >= 128
            and nme <= .035
            and abs(yaw) <= 25
            and abs(pitch) <= 20
            and saturation <= .10
            and stability <= .35
        ):
            confidence = "high"
        elif (
            face_width >= 80
            and nme <= .060
            and abs(yaw) <= 40
            and abs(pitch) <= 30
            and saturation <= .25
            and stability <= .75
        ):
            confidence = "medium"
        else:
            confidence = "low"
        if face_width < 128:
            reasons.append("SMALL_FACE")
        if nme > .035:
            reasons.append("REPROJECTION_ERROR")
        if abs(yaw) > 25 or abs(pitch) > 20:
            reasons.append("EXTREME_POSE")
        if saturation > .10:
            reasons.append("COEFFICIENT_SATURATION")
        if stability > .35:
            reasons.append("UNSTABLE_IDENTITY")
        identity = np.zeros(self.adapter.identity_dim, dtype=np.float32)
        identity[:modes] = beta.astype(np.float32)
        return FitResult(
            identity=identity,
            nuisance=nuisance.astype(np.float32),
            camera=camera.astype(np.float32),
            fitted_landmarks=fitted_points.astype(np.float32),
            nme=nme,
            pixel_error=float(np.mean(distances)),
            saturation_fraction=saturation,
            stability_rms=stability,
            confidence=confidence,
            confidence_reasons=tuple(reasons),
        )
