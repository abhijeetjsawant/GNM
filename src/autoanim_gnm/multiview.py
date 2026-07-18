"""Shared-identity fitting from guided multi-view GNM-68 observations.

This module deliberately fits only the portion of GNM identity that is visible
to the sparse 68-landmark regressor.  In GNM Head 3.0, identity dimensions
170:253 control eye/teeth geometry and have exactly zero support at those
landmarks.  Returning values for them would therefore be invented rather than
estimated.

The solver alternates robust per-view camera refinement with a shared linearized
shape solve.  The latter is performed in an SVD-observable subspace, with PCA
and nuisance-expression priors.  It is intended as a deterministic geometric
initialization for dense/photometric refinement, not as a claim that sparse
landmarks alone recover a metrically exact head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, least_squares, lsq_linear, minimize

from .gnm_adapter import GNMAdapter


CameraKind = Literal["perspective", "weak_perspective"]

LANDMARK_COUNT = 68
GNM_IDENTITY_DIM = 253
GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM = 170
PERSPECTIVE_CAMERA_CONVENTION = "autoanim.gnm_front_opencv.v2"
IDENTITY_COEFFICIENT_BOUND = 3.0
NUISANCE_COEFFICIENT_BOUND = 0.35
IDENTITY_SOLVER = "observable_subspace_constrained_least_squares_v1"

METRIC_SCALE_CAVEAT = (
    "Sparse 2D landmarks do not determine absolute physical head scale. "
    "Weak-perspective views additionally confound camera depth and scale; calibrated "
    "intrinsics recover depth only in GNM model units, not a guaranteed real-world size."
)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = np.asarray((self.fx, self.fy, self.cx, self.cy), dtype=np.float64)
        if not np.isfinite(values).all() or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera intrinsics must be finite with positive fx and fy")


@dataclass(frozen=True, slots=True)
class MultiViewObservation:
    """One image's GNM sparse-68 correspondences and measurement metadata.

    ``image_size`` is ``(height, width)``.  Confidence can be a scalar view
    confidence or one value per landmark.  Visibility can be boolean or a
    continuous [0, 1] visibility weight.
    """

    landmarks: np.ndarray
    image_size: tuple[int, int]
    intrinsics: CameraIntrinsics | None = None
    role: str | None = None
    confidence: float | np.ndarray = 1.0
    visibility: np.ndarray | None = None
    initial_camera: CameraEstimate | None = None
    lock_camera: bool = False


@dataclass(frozen=True, slots=True)
class WeakPerspectiveCamera:
    yaw: float
    pitch: float
    roll: float
    scale: float
    tx: float
    ty: float

    kind: CameraKind = "weak_perspective"


@dataclass(frozen=True, slots=True)
class PerspectiveCamera:
    yaw: float
    pitch: float
    roll: float
    tx: float
    ty: float
    tz: float
    intrinsics: CameraIntrinsics

    kind: CameraKind = "perspective"


CameraEstimate = WeakPerspectiveCamera | PerspectiveCamera


@dataclass(frozen=True, slots=True)
class ViewFitReport:
    index: int
    role: str
    camera_kind: CameraKind
    accepted: bool
    rejection_reason: str | None
    visible_landmarks: int
    nme: float
    mean_pixel_error: float
    median_pixel_error: float
    robust_view_weight: float


@dataclass(frozen=True, slots=True)
class MultiViewFitReport:
    accepted: bool
    nme: float
    mean_pixel_error: float
    per_view: tuple[ViewFitReport, ...]
    accepted_view_indices: tuple[int, ...]
    rejected_view_indices: tuple[int, ...]
    unlocked_stages: tuple[int, ...]
    observable_rank: int
    active_identity_modes: int
    weakly_observable_directions: int
    condition_number: float
    observability_ratio: float
    saturation_fraction: float
    identity_solver: str
    identity_coefficient_bound: float
    nuisance_coefficient_bound: float
    identity_consistency_matrix: np.ndarray
    leave_one_out_nme: np.ndarray
    metric_scale_caveat: str


@dataclass(frozen=True, slots=True)
class MultiViewFitResult:
    identity: np.ndarray
    nuisance: tuple[np.ndarray, ...]
    cameras: tuple[CameraEstimate, ...]
    fitted_landmarks: tuple[np.ndarray, ...]
    report: MultiViewFitReport


@dataclass(slots=True)
class _PreparedView:
    landmarks: np.ndarray
    height: int
    width: int
    intrinsics: CameraIntrinsics | None
    role: str
    base_weights: np.ndarray
    visibility: np.ndarray
    normalizer: float
    initial_camera: CameraEstimate | None
    lock_camera: bool


@dataclass(slots=True)
class _CoreFit:
    beta: np.ndarray
    nuisance: list[np.ndarray]
    cameras: list[CameraEstimate]
    fitted: list[np.ndarray]
    robust_weights: list[np.ndarray]
    view_weights: np.ndarray
    stages: tuple[int, ...]
    singular_values: np.ndarray
    observable_rank: int


def _bounded_observable_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    observable_vectors: np.ndarray,
    nuisance_columns: int,
) -> np.ndarray:
    """Solve a bounded least-squares problem without leaving the evidence subspace.

    The first variables are coordinates in the right-singular-vector basis of
    the observable identity Jacobian.  Identity coefficient bounds therefore
    become linear constraints, not box constraints on those coordinates.  A
    solve-then-clip implementation is invalid here: component-wise clipping can
    rotate a coefficient vector out of the measured subspace and manufacture
    shape along a landmark-null direction.

    The remaining variables are direct per-view nuisance-expression
    coefficients and use ordinary box bounds.  The unconstrained solution is
    returned exactly when it is feasible, preserving the fast path and numeric
    behavior for ordinary captures.
    """

    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    vectors = np.asarray(observable_vectors, dtype=np.float64)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("design and target must have compatible least-squares shapes")
    if vectors.ndim != 2:
        raise ValueError("observable_vectors must be a rank-by-mode matrix")
    rank, active_modes = vectors.shape
    if nuisance_columns < 0 or matrix.shape[1] != rank + nuisance_columns:
        raise ValueError("design columns do not match identity and nuisance variables")
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(values).all()
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("bounded least-squares inputs must be finite")

    unconstrained = np.linalg.lstsq(matrix, values, rcond=None)[0]
    identity = vectors.T @ unconstrained[:rank]
    nuisance = unconstrained[rank:]
    tolerance = 1.0e-9
    if (
        np.max(np.abs(identity), initial=0.0) <= IDENTITY_COEFFICIENT_BOUND + tolerance
        and np.max(np.abs(nuisance), initial=0.0) <= NUISANCE_COEFFICIENT_BOUND + tolerance
    ):
        return unconstrained

    # Once every active mode is observable, the right-singular vectors form a
    # square orthogonal change of coordinates.  Solve directly in coefficient
    # space with a stable bounded least-squares routine, then rotate back to
    # observable coordinates.  This is both exact and materially faster than a
    # general linear-constraint optimizer for the high-rank stages.
    if rank == active_modes and rank:
        coefficient_design = matrix.copy()
        coefficient_design[:, :rank] = matrix[:, :rank] @ vectors
        lower = np.full(matrix.shape[1], -np.inf, dtype=np.float64)
        upper = np.full(matrix.shape[1], np.inf, dtype=np.float64)
        lower[:rank] = -IDENTITY_COEFFICIENT_BOUND
        upper[:rank] = IDENTITY_COEFFICIENT_BOUND
        if nuisance_columns:
            lower[rank:] = -NUISANCE_COEFFICIENT_BOUND
            upper[rank:] = NUISANCE_COEFFICIENT_BOUND
        bounded = lsq_linear(
            coefficient_design,
            values,
            bounds=(lower, upper),
            method="trf",
            tol=1.0e-12,
            lsmr_tol=None,
            max_iter=300,
        )
        if not bounded.success or not np.isfinite(bounded.x).all():
            raise ValueError(
                "Full-rank bounded identity solve failed closed: "
                f"status={bounded.status}, message={bounded.message}"
            )
        solution = np.asarray(bounded.x, dtype=np.float64)
        solution[:rank] = vectors @ solution[:rank]
        return solution

    # Start from a feasible point on the ray toward the unconstrained optimum.
    initial = unconstrained.copy()
    identity_peak = float(np.max(np.abs(identity), initial=0.0))
    if identity_peak > IDENTITY_COEFFICIENT_BOUND:
        initial[:rank] *= IDENTITY_COEFFICIENT_BOUND / identity_peak
    if nuisance_columns:
        initial[rank:] = np.clip(
            initial[rank:], -NUISANCE_COEFFICIENT_BOUND, NUISANCE_COEFFICIENT_BOUND
        )

    gram = matrix.T @ matrix
    rhs = matrix.T @ values

    def objective(solution: np.ndarray) -> float:
        return float(0.5 * solution @ gram @ solution - rhs @ solution)

    def gradient(solution: np.ndarray) -> np.ndarray:
        return gram @ solution - rhs

    lower = np.full(matrix.shape[1], -np.inf, dtype=np.float64)
    upper = np.full(matrix.shape[1], np.inf, dtype=np.float64)
    if nuisance_columns:
        lower[rank:] = -NUISANCE_COEFFICIENT_BOUND
        upper[rank:] = NUISANCE_COEFFICIENT_BOUND
    constraints: tuple[LinearConstraint, ...] = ()
    if rank and active_modes:
        constraint_matrix = np.zeros(
            (active_modes, matrix.shape[1]), dtype=np.float64
        )
        constraint_matrix[:, :rank] = vectors.T
        constraints = (
            LinearConstraint(
                constraint_matrix,
                -IDENTITY_COEFFICIENT_BOUND,
                IDENTITY_COEFFICIENT_BOUND,
            ),
        )
    fitted = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"ftol": 1.0e-12, "maxiter": 300, "disp": False},
    )
    solution = np.asarray(fitted.x, dtype=np.float64)
    solved_identity = vectors.T @ solution[:rank]
    solved_nuisance = solution[rank:]
    feasible = (
        np.isfinite(solution).all()
        and np.max(np.abs(solved_identity), initial=0.0)
        <= IDENTITY_COEFFICIENT_BOUND + 1.0e-7
        and np.max(np.abs(solved_nuisance), initial=0.0)
        <= NUISANCE_COEFFICIENT_BOUND + 1.0e-7
    )
    if not fitted.success or not feasible:
        raise ValueError(
            "Constrained observable identity solve failed closed: "
            f"status={fitted.status}, message={fitted.message}, feasible={feasible}"
        )
    return solution


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return the deterministic Y-X-Z Euler rotation used by the fitter."""

    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rx = np.asarray(((1, 0, 0), (0, cp, -sp), (0, sp, cp)), dtype=np.float64)
    rz = np.asarray(((cr, -sr, 0), (sr, cr, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ rx @ ry


def project_points(points: np.ndarray, camera: CameraEstimate) -> np.ndarray:
    """Project ``[N,3]`` GNM points through a fitted or synthetic camera."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"Expected finite [N,3] points; got {points.shape}")
    rotation = rotation_matrix(camera.yaw, camera.pitch, camera.roll)
    rotated = points @ rotation.T
    if isinstance(camera, WeakPerspectiveCamera):
        if not np.isfinite(camera.scale) or camera.scale <= 0:
            raise ValueError("Weak-perspective scale must be finite and positive")
        return np.column_stack(
            (camera.scale * rotated[:, 0] + camera.tx, -camera.scale * rotated[:, 1] + camera.ty)
        )
    # GNM faces +Z.  A physical frontal camera therefore sits on +Z and looks
    # toward -Z.  Camera X stays image-right, camera Y is image-down, and
    # positive camera Z points into the scene (OpenCV convention).
    translated_x = rotated[:, 0] + camera.tx
    translated_y_up = rotated[:, 1] + camera.ty
    depth = camera.tz - rotated[:, 2]
    if not np.isfinite(depth).all() or np.any(depth <= 1e-6):
        raise ValueError("Perspective projection contains points behind the camera")
    intrinsics = camera.intrinsics
    return np.column_stack(
        (
            intrinsics.fx * translated_x / depth + intrinsics.cx,
            intrinsics.cy - intrinsics.fy * translated_y_up / depth,
        )
    )


class MultiViewIdentityFitter:
    """Fit one GNM identity to two or more guided views.

    Parameters are intentionally conservative: eye/teeth modes remain neutral,
    nuisance expression is bounded, and mode prefixes are unlocked only in the
    configured deterministic order.
    """

    def __init__(
        self,
        adapter: GNMAdapter,
        rig: object | None = None,
        *,
        stages: Sequence[int] = (20, 40, 80, 120, 170),
        observability_rtol: float = 1e-4,
        identity_prior: float = 2e-5,
        nuisance_prior: float = 2e-3,
        max_outer_iterations: int = 3,
    ) -> None:
        self.adapter = adapter
        self.template = np.asarray(adapter.compact_template, dtype=np.float64)
        self.identity_basis = np.asarray(
            adapter.compact_identity_basis[:GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM],
            dtype=np.float64,
        )
        if self.template.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"Unexpected compact template shape: {self.template.shape}")
        if self.identity_basis.shape != (
            GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM,
            LANDMARK_COUNT,
            3,
        ):
            raise ValueError(f"Unexpected compact identity basis: {self.identity_basis.shape}")
        tail = np.asarray(adapter.compact_identity_basis[GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM:])
        if np.any(tail):
            raise ValueError("GNM sparse-68 observability boundary changed; update the fitter")

        cleaned_stages = tuple(dict.fromkeys(int(value) for value in stages))
        if (
            not cleaned_stages
            or any(value <= 0 or value > GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM for value in cleaned_stages)
            or tuple(sorted(cleaned_stages)) != cleaned_stages
        ):
            raise ValueError("stages must be strictly increasing values in [1, 170]")
        if not np.isfinite(observability_rtol) or not 0 < observability_rtol < 1:
            raise ValueError("observability_rtol must be in (0, 1)")
        if identity_prior < 0 or nuisance_prior < 0:
            raise ValueError("coefficient priors must be non-negative")
        if max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be positive")
        self.stages = cleaned_stages
        self.observability_rtol = float(observability_rtol)
        self.identity_prior = float(identity_prior)
        self.nuisance_prior = float(nuisance_prior)
        self.max_outer_iterations = int(max_outer_iterations)
        self.nuisance_basis = self._make_nuisance_basis(rig)

        self.landmark_weights = np.ones(LANDMARK_COUNT, dtype=np.float64)
        self.landmark_weights[:17] = 0.55
        self.landmark_weights[27:48] = 1.6

    def _make_nuisance_basis(self, rig: object | None) -> np.ndarray:
        expression_basis = np.asarray(self.adapter.compact_expression_basis, dtype=np.float64)
        if rig is not None:
            try:
                anchors = ("happy", "surprise", "pucker", "corners_down")
                coefficients = np.stack([rig.decoder.prototype(name) for name in anchors])
                result = np.einsum("ae,elc->alc", coefficients, expression_basis)
                if result.shape == (4, LANDMARK_COUNT, 3) and np.isfinite(result).all():
                    return result
            except (AttributeError, KeyError, ValueError):
                pass
        # GNM expression PCA is region ordered.  Four low-order skin directions
        # provide neutral nuisance slack when the semantic decoder is unavailable.
        return expression_basis[:4].copy()

    @property
    def nuisance_dim(self) -> int:
        return int(self.nuisance_basis.shape[0])

    def fit(self, observations: Sequence[MultiViewObservation]) -> MultiViewFitResult:
        """Fit observations and refit after excluding a detectable mixed identity."""

        views = self._prepare_views(observations)
        if len(views) < 2:
            raise ValueError("Multi-view identity fitting requires at least two views")

        first = self._fit_core(views)
        consistency = self._identity_consistency(views, first)
        leave_one_out_nme = self._leave_one_out_nme(views)
        rejected = self._detect_inconsistent_views(
            views, first, consistency, leave_one_out_nme
        )
        accepted = [index for index in range(len(views)) if index not in rejected]
        if rejected and len(accepted) >= 2:
            reduced = self._fit_core([views[index] for index in accepted])
            final = self._expand_core_to_all_views(views, accepted, reduced)
        else:
            final = first
            rejected = set()
            accepted = list(range(len(views)))

        return self._build_result(
            views, final, accepted, rejected, consistency, leave_one_out_nme
        )

    def _prepare_views(self, observations: Sequence[MultiViewObservation]) -> list[_PreparedView]:
        if isinstance(observations, (str, bytes)):
            raise ValueError("observations must be a sequence of MultiViewObservation")
        prepared: list[_PreparedView] = []
        for index, observation in enumerate(observations):
            if not isinstance(observation, MultiViewObservation):
                raise ValueError(f"View {index} is not a MultiViewObservation")
            if observation.intrinsics is not None and not isinstance(
                observation.intrinsics, CameraIntrinsics
            ):
                raise ValueError(f"View {index}: intrinsics must be CameraIntrinsics or None")
            if observation.initial_camera is not None and not isinstance(
                observation.initial_camera, (WeakPerspectiveCamera, PerspectiveCamera)
            ):
                raise ValueError(
                    f"View {index}: initial_camera must be a supported camera or None"
                )
            if observation.lock_camera and observation.initial_camera is None:
                raise ValueError(f"View {index}: lock_camera requires initial_camera")
            if isinstance(observation.initial_camera, PerspectiveCamera):
                if observation.intrinsics is None:
                    raise ValueError(
                        f"View {index}: perspective initial_camera requires intrinsics"
                    )
                if observation.initial_camera.intrinsics != observation.intrinsics:
                    raise ValueError(
                        f"View {index}: initial camera intrinsics do not match observation"
                    )
            if isinstance(observation.initial_camera, WeakPerspectiveCamera) and (
                observation.intrinsics is not None
            ):
                raise ValueError(
                    f"View {index}: weak-perspective initial_camera cannot use intrinsics"
                )
            points = np.asarray(observation.landmarks, dtype=np.float64)
            if points.shape != (LANDMARK_COUNT, 2) or not np.isfinite(points).all():
                raise ValueError(f"View {index}: expected finite [68,2] landmarks; got {points.shape}")
            if len(observation.image_size) != 2:
                raise ValueError(f"View {index}: image_size must be (height, width)")
            height, width = observation.image_size
            if (
                isinstance(height, bool)
                or isinstance(width, bool)
                or int(height) != height
                or int(width) != width
                or height <= 0
                or width <= 0
            ):
                raise ValueError(f"View {index}: image dimensions must be positive integers")

            confidence = np.asarray(observation.confidence, dtype=np.float64)
            if confidence.ndim == 0:
                confidence = np.full(LANDMARK_COUNT, float(confidence), dtype=np.float64)
            if confidence.shape != (LANDMARK_COUNT,) or not np.isfinite(confidence).all():
                raise ValueError(f"View {index}: confidence must be a scalar or finite [68]")
            if np.any((confidence < 0) | (confidence > 1)):
                raise ValueError(f"View {index}: confidence must lie in [0,1]")

            if observation.visibility is None:
                visibility = np.ones(LANDMARK_COUNT, dtype=np.float64)
            else:
                visibility = np.asarray(observation.visibility, dtype=np.float64)
                if visibility.shape != (LANDMARK_COUNT,) or not np.isfinite(visibility).all():
                    raise ValueError(f"View {index}: visibility must be finite [68]")
                if np.any((visibility < 0) | (visibility > 1)):
                    raise ValueError(f"View {index}: visibility must lie in [0,1]")
            base_weights = confidence * visibility * self.landmark_weights
            visible = base_weights > 1e-8
            if int(np.count_nonzero(visible)) < 24:
                raise ValueError(f"View {index}: at least 24 confident visible landmarks are required")
            visible_points = points[visible]
            extent = max(
                float(np.ptp(visible_points[:, 0])),
                float(np.ptp(visible_points[:, 1])),
            )
            eye_distance = (
                float(np.linalg.norm(points[36] - points[45]))
                if visible[36] and visible[45]
                else 0.0
            )
            normalizer = max(eye_distance, 0.35 * extent, 1.0)
            prepared.append(
                _PreparedView(
                    landmarks=points,
                    height=int(height),
                    width=int(width),
                    intrinsics=observation.intrinsics,
                    role=(observation.role or "unspecified").strip().lower(),
                    base_weights=base_weights,
                    visibility=visible,
                    normalizer=normalizer,
                    initial_camera=observation.initial_camera,
                    lock_camera=bool(observation.lock_camera),
                )
            )
        return prepared

    def _shape(self, beta: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
        return (
            self.template
            + np.einsum("i,ilc->lc", beta, self.identity_basis)
            + np.einsum("i,ilc->lc", nuisance, self.nuisance_basis)
        )

    def _fit_core(self, views: list[_PreparedView]) -> _CoreFit:
        beta = np.zeros(GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM, dtype=np.float64)
        nuisance = [np.zeros(self.nuisance_dim, dtype=np.float64) for _ in views]
        cameras = [self._initial_camera(view) for view in views]
        robust = [np.ones(LANDMARK_COUNT, dtype=np.float64) for _ in views]
        view_weights = np.ones(len(views), dtype=np.float64)
        singular_values = np.empty(0, dtype=np.float64)
        rank = 0
        completed_stages: list[int] = []

        for stage in self.stages:
            for _ in range(self.max_outer_iterations):
                for index, view in enumerate(views):
                    shape = self._shape(beta, nuisance[index])
                    cameras[index] = self._refine_camera(
                        view, shape, cameras[index], robust[index] * view_weights[index]
                    )
                beta, nuisance, singular_values, rank = self._solve_shape(
                    views,
                    cameras,
                    beta,
                    nuisance,
                    robust,
                    view_weights,
                    stage,
                )
                fitted = [
                    project_points(self._shape(beta, nuisance[index]), camera)
                    for index, camera in enumerate(cameras)
                ]
                robust, view_weights = self._irls_weights(views, fitted)
            completed_stages.append(stage)

        fitted = [
            project_points(self._shape(beta, nuisance[index]), camera)
            for index, camera in enumerate(cameras)
        ]
        return _CoreFit(
            beta=beta,
            nuisance=nuisance,
            cameras=cameras,
            fitted=fitted,
            robust_weights=robust,
            view_weights=view_weights,
            stages=tuple(completed_stages),
            singular_values=singular_values,
            observable_rank=rank,
        )

    def _initial_camera(self, view: _PreparedView) -> CameraEstimate:
        if view.initial_camera is not None:
            return view.initial_camera
        candidates = self._yaw_candidates(view.role)
        best_camera: CameraEstimate | None = None
        best_cost = math.inf
        neutral_weights = np.ones(LANDMARK_COUNT, dtype=np.float64)
        for yaw in candidates:
            camera = self._camera_from_yaw(view, yaw)
            refined = self._refine_camera(view, self.template, camera, neutral_weights)
            projected = project_points(self.template, refined)
            residual = np.linalg.norm(projected - view.landmarks, axis=1) / view.normalizer
            cost = float(np.median(residual[view.visibility]))
            if cost < best_cost:
                best_camera, best_cost = refined, cost
        assert best_camera is not None
        return best_camera

    @staticmethod
    def _yaw_candidates(role: str) -> tuple[float, ...]:
        degrees: tuple[float, ...]
        normalized = role.replace("-", "_").replace(" ", "_")
        if "front" in normalized:
            degrees = (0.0, -20.0, 20.0)
        elif "left" in normalized and ("profile" in normalized or "side" in normalized):
            degrees = (75.0, 90.0, 55.0)
        elif "right" in normalized and ("profile" in normalized or "side" in normalized):
            degrees = (-75.0, -90.0, -55.0)
        elif "left" in normalized:
            degrees = (40.0, 55.0, 25.0)
        elif "right" in normalized:
            degrees = (-40.0, -55.0, -25.0)
        else:
            degrees = (0.0, -40.0, 40.0, -75.0, 75.0)
        return tuple(math.radians(value) for value in degrees)

    def _camera_from_yaw(self, view: _PreparedView, yaw: float) -> CameraEstimate:
        rotation = rotation_matrix(yaw, 0.0, 0.0)
        rotated = self.template @ rotation.T
        # Camera initialization must be invariant to coordinates whose
        # confidence/visibility is zero.  Prefer stable central-face anchors,
        # then fall back to every visible landmark if profile occlusion leaves
        # too few of those anchors.
        anchor_mask = view.visibility.copy()
        central = np.zeros(LANDMARK_COUNT, dtype=bool)
        central[27:48] = True
        if int(np.count_nonzero(anchor_mask & central)) >= 8:
            anchor_mask &= central
        selected = np.flatnonzero(anchor_mask)
        weights = np.maximum(view.base_weights[selected], 1.0e-12)
        weights /= float(np.sum(weights))

        observed = view.landmarks[selected]
        model = rotated[selected]
        if view.intrinsics is None:
            model_2d = np.column_stack((model[:, 0], -model[:, 1]))
            observed_center = np.sum(observed * weights[:, None], axis=0)
            model_center = np.sum(model_2d * weights[:, None], axis=0)
            centered_observed = observed - observed_center
            centered_model = model_2d - model_center
            denominator = float(np.sum(weights[:, None] * centered_model**2))
            numerator = float(
                np.sum(weights[:, None] * centered_model * centered_observed)
            )
            if numerator > 1.0e-12 and denominator > 1.0e-12:
                scale = numerator / denominator
            else:
                observed_spread = float(
                    np.sum(weights[:, None] * centered_observed**2)
                )
                scale = math.sqrt(observed_spread / max(denominator, 1.0e-12))
            scale = max(scale, 1.0e-3)
            return WeakPerspectiveCamera(
                yaw=yaw,
                pitch=0.0,
                roll=0.0,
                scale=scale,
                tx=float(observed_center[0] - scale * model_center[0]),
                ty=float(observed_center[1] - scale * model_center[1]),
            )
        intrinsics = view.intrinsics
        observed_normalized = np.column_stack(
            (
                (observed[:, 0] - intrinsics.cx) / intrinsics.fx,
                -(observed[:, 1] - intrinsics.cy) / intrinsics.fy,
            )
        )
        observed_center = np.sum(observed_normalized * weights[:, None], axis=0)
        model_center_2d = np.sum(model[:, :2] * weights[:, None], axis=0)
        model_center_z = float(np.sum(model[:, 2] * weights))
        centered_observed = observed_normalized - observed_center
        centered_model = model[:, :2] - model_center_2d
        denominator = float(np.sum(weights[:, None] * centered_model**2))
        numerator = float(
            np.sum(weights[:, None] * centered_model * centered_observed)
        )
        if numerator > 1.0e-12 and denominator > 1.0e-12:
            inverse_depth = numerator / denominator
        else:
            observed_spread = float(
                np.sum(weights[:, None] * centered_observed**2)
            )
            inverse_depth = math.sqrt(observed_spread / max(denominator, 1.0e-12))
        depth = 1.0 / max(inverse_depth, 1.0e-6)
        tz = max(depth + model_center_z, 0.15)
        center_depth = tz - model_center_z
        tx = observed_center[0] * center_depth - model_center_2d[0]
        ty = observed_center[1] * center_depth - model_center_2d[1]
        return PerspectiveCamera(yaw, 0.0, 0.0, float(tx), float(ty), float(tz), intrinsics)

    @staticmethod
    def _encode_camera(camera: CameraEstimate) -> np.ndarray:
        if isinstance(camera, WeakPerspectiveCamera):
            return np.asarray(
                (camera.yaw, camera.pitch, camera.roll, math.log(camera.scale), camera.tx, camera.ty),
                dtype=np.float64,
            )
        return np.asarray(
            (camera.yaw, camera.pitch, camera.roll, camera.tx, camera.ty, math.log(camera.tz)),
            dtype=np.float64,
        )

    @staticmethod
    def _decode_camera(values: np.ndarray, view: _PreparedView) -> CameraEstimate:
        if view.intrinsics is None:
            return WeakPerspectiveCamera(
                float(values[0]),
                float(values[1]),
                float(values[2]),
                float(math.exp(values[3])),
                float(values[4]),
                float(values[5]),
            )
        return PerspectiveCamera(
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(math.exp(values[5])),
            view.intrinsics,
        )

    def _refine_camera(
        self,
        view: _PreparedView,
        shape: np.ndarray,
        initial: CameraEstimate,
        robust_weights: np.ndarray,
    ) -> CameraEstimate:
        if view.lock_camera:
            return initial
        x0 = self._encode_camera(initial)
        if view.intrinsics is None:
            low = np.asarray(
                (-1.75, -0.9, -0.8, math.log(1e-3), -2 * view.width, -2 * view.height)
            )
            high = np.asarray(
                (1.75, 0.9, 0.8, math.log(1e5), 3 * view.width, 3 * view.height)
            )
        else:
            low = np.asarray((-1.75, -0.9, -0.8, -2.0, -2.0, math.log(0.08)))
            high = np.asarray((1.75, 0.9, 0.8, 2.0, 2.0, math.log(20.0)))
        x0 = np.clip(x0, low + 1e-9, high - 1e-9)
        weights = np.sqrt(np.maximum(view.base_weights * robust_weights, 0.0))
        selected = weights > 1e-8

        def residual(values: np.ndarray) -> np.ndarray:
            camera = self._decode_camera(values, view)
            try:
                predicted = project_points(shape, camera)
            except ValueError:
                return np.full(2 * int(np.count_nonzero(selected)), 1e3)
            return (
                weights[selected, None]
                * (predicted[selected] - view.landmarks[selected])
                / view.normalizer
            ).ravel()

        fitted = least_squares(
            residual,
            x0,
            bounds=(low, high),
            loss="soft_l1",
            f_scale=0.012,
            max_nfev=120,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        return self._decode_camera(fitted.x, view)

    def _refine_nuisance(
        self,
        view: _PreparedView,
        beta: np.ndarray,
        camera: CameraEstimate,
        initial: np.ndarray,
        robust_weights: np.ndarray,
    ) -> np.ndarray:
        """Refine bounded per-view expression while shared identity stays fixed."""

        weights = np.sqrt(np.maximum(view.base_weights * robust_weights, 0.0))
        selected = weights > 1e-8

        def residual(values: np.ndarray) -> np.ndarray:
            predicted = project_points(self._shape(beta, values), camera)
            reprojection = (
                weights[selected, None]
                * (predicted[selected] - view.landmarks[selected])
                / view.normalizer
            ).ravel()
            if self.nuisance_prior:
                return np.concatenate(
                    (reprojection, math.sqrt(self.nuisance_prior) * values)
                )
            return reprojection

        fitted = least_squares(
            residual,
            np.clip(np.asarray(initial, dtype=np.float64), -0.35, 0.35),
            bounds=(-0.35, 0.35),
            loss="soft_l1",
            f_scale=0.012,
            max_nfev=80,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        return fitted.x

    def _projection_jacobian(
        self, points: np.ndarray, camera: CameraEstimate
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = rotation_matrix(camera.yaw, camera.pitch, camera.roll)
        if isinstance(camera, WeakPerspectiveCamera):
            projection = np.asarray(((camera.scale, 0.0, 0.0), (0.0, -camera.scale, 0.0)))
            jacobian = np.broadcast_to(projection @ rotation, (len(points), 2, 3)).copy()
            return project_points(points, camera), jacobian

        rotated = points @ rotation.T
        x = rotated[:, 0] + camera.tx
        y = rotated[:, 1] + camera.ty
        z = camera.tz - rotated[:, 2]
        if np.any(z <= 1e-6):
            raise ValueError("Perspective linearization contains non-positive depth")
        intrinsics = camera.intrinsics
        local = np.zeros((len(points), 2, 3), dtype=np.float64)
        local[:, 0, 0] = intrinsics.fx / z
        local[:, 0, 2] = intrinsics.fx * x / (z * z)
        local[:, 1, 1] = -intrinsics.fy / z
        local[:, 1, 2] = -intrinsics.fy * y / (z * z)
        return project_points(points, camera), np.einsum("nab,bc->nac", local, rotation)

    def _camera_parameter_jacobian(
        self, points: np.ndarray, camera: CameraEstimate, view: _PreparedView
    ) -> np.ndarray:
        """Finite-difference the six encoded camera nuisance directions."""

        if view.lock_camera:
            return np.zeros((len(points), 2, 0), dtype=np.float64)

        values = self._encode_camera(camera)
        jacobian = np.empty((len(points), 2, len(values)), dtype=np.float64)
        for column in range(len(values)):
            step = 1.0e-6 * max(1.0, abs(float(values[column])))
            positive = values.copy()
            negative = values.copy()
            positive[column] += step
            negative[column] -= step
            projected_positive = project_points(
                points, self._decode_camera(positive, view)
            )
            projected_negative = project_points(
                points, self._decode_camera(negative, view)
            )
            jacobian[:, :, column] = (
                projected_positive - projected_negative
            ) / (2.0 * step)
        return jacobian

    def _solve_shape(
        self,
        views: list[_PreparedView],
        cameras: list[CameraEstimate],
        beta: np.ndarray,
        nuisance: list[np.ndarray],
        robust: list[np.ndarray],
        view_weights: np.ndarray,
        stage: int,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, int]:
        # Two Gauss-Newton passes are enough for the mild perspective nonlinearity.
        singular_values = np.empty(0, dtype=np.float64)
        rank = 0
        for _ in range(2):
            identity_rows: list[np.ndarray] = []
            observable_identity_rows: list[np.ndarray] = []
            nuisance_rows: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            row_counts: list[int] = []
            for index, (view, camera) in enumerate(zip(views, cameras, strict=True)):
                points = self._shape(beta, nuisance[index])
                predicted, projection_jacobian = self._projection_jacobian(points, camera)
                jb = np.einsum(
                    "nac,inc->nai", projection_jacobian, self.identity_basis[:stage]
                )
                jn = np.einsum("nac,inc->nai", projection_jacobian, self.nuisance_basis)
                selected = view.base_weights > 1e-8
                weights = np.sqrt(
                    np.maximum(
                        view.base_weights[selected]
                        * robust[index][selected]
                        * view_weights[index],
                        0.0,
                    )
                )
                weighted_jb = (
                    jb[selected] * weights[:, None, None] / view.normalizer
                ).reshape(-1, stage)
                weighted_jn = (
                    jn[selected] * weights[:, None, None] / view.normalizer
                ).reshape(-1, self.nuisance_dim)
                jc = self._camera_parameter_jacobian(points, camera, view)
                weighted_jc = (
                    (
                        jc[selected] * weights[:, None, None] / view.normalizer
                    ).reshape(-1, jc.shape[2])
                    if jc.shape[2]
                    else np.zeros((2 * int(np.count_nonzero(selected)), 0))
                )
                confounds = np.column_stack((weighted_jn, weighted_jc))
                if confounds.size:
                    u, confound_singular, _ = np.linalg.svd(
                        confounds, full_matrices=False
                    )
                    confound_rank = (
                        int(
                            np.count_nonzero(
                                confound_singular
                                > confound_singular[0] * 1.0e-8
                            )
                        )
                        if confound_singular.size and confound_singular[0] > 1.0e-12
                        else 0
                    )
                    if confound_rank:
                        confound_basis = u[:, :confound_rank]
                        observable_jb = weighted_jb - confound_basis @ (
                            confound_basis.T @ weighted_jb
                        )
                    else:
                        observable_jb = weighted_jb
                else:  # pragma: no cover - six camera columns are always present
                    observable_jb = weighted_jb
                linear_target = (
                    view.landmarks[selected]
                    - predicted[selected]
                    + np.einsum("nai,i->na", jb[selected], beta[:stage])
                    + np.einsum("nai,i->na", jn[selected], nuisance[index])
                )
                weighted_target = (
                    linear_target * weights[:, None] / view.normalizer
                ).ravel()
                identity_rows.append(weighted_jb)
                observable_identity_rows.append(observable_jb)
                nuisance_rows.append(weighted_jn)
                targets.append(weighted_target)
                row_counts.append(len(weighted_target))

            identity_matrix = np.vstack(identity_rows)
            observable_identity_matrix = np.vstack(observable_identity_rows)
            _, singular_values, vt = np.linalg.svd(
                observable_identity_matrix, full_matrices=False
            )
            if singular_values.size == 0 or singular_values[0] <= 1e-12:
                rank = 0
                observable_vectors = np.zeros((0, stage), dtype=np.float64)
            else:
                rank = int(
                    np.count_nonzero(
                        singular_values
                        > singular_values[0] * self.observability_rtol
                    )
                )
                observable_vectors = vt[:rank]

            identity_observable = identity_matrix @ observable_vectors.T
            total_rows = sum(row_counts)
            columns = rank + len(views) * self.nuisance_dim
            design = np.zeros((total_rows, columns), dtype=np.float64)
            target = np.concatenate(targets)
            cursor = 0
            for index, count in enumerate(row_counts):
                design[cursor : cursor + count, :rank] = identity_observable[cursor : cursor + count]
                start = rank + index * self.nuisance_dim
                design[cursor : cursor + count, start : start + self.nuisance_dim] = nuisance_rows[index]
                cursor += count

            prior_rows: list[np.ndarray] = []
            prior_targets: list[np.ndarray] = []
            if rank and self.identity_prior:
                block = np.zeros((rank, columns), dtype=np.float64)
                block[:, :rank] = math.sqrt(self.identity_prior) * np.eye(rank)
                prior_rows.append(block)
                prior_targets.append(np.zeros(rank))
            if self.nuisance_dim and self.nuisance_prior:
                count = len(views) * self.nuisance_dim
                block = np.zeros((count, columns), dtype=np.float64)
                block[:, rank:] = math.sqrt(self.nuisance_prior) * np.eye(count)
                prior_rows.append(block)
                prior_targets.append(np.zeros(count))
            if prior_rows:
                design = np.vstack((design, *prior_rows))
                target = np.concatenate((target, *prior_targets))
            solution = _bounded_observable_least_squares(
                design,
                target,
                observable_vectors,
                len(views) * self.nuisance_dim,
            )
            next_beta = np.zeros_like(beta)
            if rank:
                next_beta[:stage] = observable_vectors.T @ solution[:rank]
            # The constrained solution is already feasible in coefficient
            # space.  Do not clip here: clipping would add an unobservable
            # component whenever an observable vector is not axis-aligned.
            beta = next_beta
            nuisance = [
                solution[
                    rank + index * self.nuisance_dim : rank + (index + 1) * self.nuisance_dim
                ]
                for index in range(len(views))
            ]
        return beta, nuisance, singular_values, rank

    @staticmethod
    def _irls_weights(
        views: list[_PreparedView], fitted: list[np.ndarray]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        robust: list[np.ndarray] = []
        normalized_medians: list[float] = []
        for view, predicted in zip(views, fitted, strict=True):
            distances = np.linalg.norm(predicted - view.landmarks, axis=1)
            selected = view.base_weights > 1e-8
            selected_distances = distances[selected]
            median = float(np.median(selected_distances))
            mad = float(np.median(np.abs(selected_distances - median)))
            sigma = max(1.4826 * mad, 0.003 * view.normalizer, 0.25)
            cutoff = 1.345 * sigma
            weights = np.ones(LANDMARK_COUNT, dtype=np.float64)
            large = distances > cutoff
            weights[large] = cutoff / np.maximum(distances[large], 1e-12)
            weights[~selected] = 0.0
            robust.append(weights)
            normalized_medians.append(median / view.normalizer)
        medians = np.asarray(normalized_medians, dtype=np.float64)
        center = float(np.median(medians))
        scale = max(1.4826 * float(np.median(np.abs(medians - center))), 0.004)
        view_weights = np.ones(len(views), dtype=np.float64)
        high = medians > center + 1.5 * scale
        view_weights[high] = np.maximum(0.15, (center + 1.5 * scale) / np.maximum(medians[high], 1e-12))
        return robust, view_weights

    def _identity_consistency(self, views: list[_PreparedView], core: _CoreFit) -> np.ndarray:
        # A conservative local 20-mode signature detects gross mixed identities
        # without comparing unstable high-frequency coefficients.
        signature_modes = min(20, self.stages[-1])
        signatures: list[np.ndarray] = []
        for index, view in enumerate(views):
            beta = np.zeros(GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM, dtype=np.float64)
            nuisance = [np.zeros(self.nuisance_dim, dtype=np.float64)]
            camera = core.cameras[index]
            robust = [np.ones(LANDMARK_COUNT, dtype=np.float64)]
            for _ in range(3):
                beta, nuisance, _, _ = self._solve_shape(
                    [view], [camera], beta, nuisance, robust, np.ones(1), signature_modes
                )
                shape = self._shape(beta, nuisance[0])
                camera = self._refine_camera(view, shape, camera, robust[0])
                fitted = project_points(shape, camera)
                robust, _ = self._irls_weights([view], [fitted])
            signatures.append(self.template + np.einsum("i,ilc->lc", beta, self.identity_basis))
        reference_scale = max(float(np.linalg.norm(self.template[36] - self.template[45])), 1e-8)
        matrix = np.zeros((len(views), len(views)), dtype=np.float64)
        for first in range(len(views)):
            for second in range(first + 1, len(views)):
                difference = signatures[first] - signatures[second]
                distance = float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))) / reference_scale)
                matrix[first, second] = matrix[second, first] = distance
        return matrix

    def _leave_one_out_nme(self, views: list[_PreparedView]) -> np.ndarray:
        """Cross-view error when each view is predicted from all the others.

        Unlike an all-view reprojection residual, this cannot hide a mixed
        identity by bending the shared solution toward the conflicting view.
        Medians make the diagnostic insensitive to a handful of bad landmarks.
        """

        scores = np.zeros(len(views), dtype=np.float64)
        for held_index, held_view in enumerate(views):
            training_indices = [index for index in range(len(views)) if index != held_index]
            training = self._fit_core([views[index] for index in training_indices])
            camera = self._initial_camera(held_view)
            nuisance = np.zeros(self.nuisance_dim, dtype=np.float64)
            robust = np.ones(LANDMARK_COUNT, dtype=np.float64)
            for _ in range(3):
                shape = self._shape(training.beta, nuisance)
                camera = self._refine_camera(held_view, shape, camera, robust)
                nuisance = self._refine_nuisance(
                    held_view, training.beta, camera, nuisance, robust
                )
                shape = self._shape(training.beta, nuisance)
                fitted = project_points(shape, camera)
                distances = np.linalg.norm(fitted - held_view.landmarks, axis=1)
                selected = held_view.base_weights > 1e-8
                median = float(np.median(distances[selected]))
                mad = float(np.median(np.abs(distances[selected] - median)))
                cutoff = max(median + 3.0 * 1.4826 * mad, 1.0)
                robust = np.minimum(1.0, cutoff / np.maximum(distances, 1e-12))
            selected = held_view.base_weights > 1e-8
            scores[held_index] = float(
                np.median(np.linalg.norm(fitted - held_view.landmarks, axis=1)[selected])
                / held_view.normalizer
            )
        return scores

    @staticmethod
    def _detect_inconsistent_views(
        views: list[_PreparedView],
        core: _CoreFit,
        consistency: np.ndarray,
        leave_one_out_nme: np.ndarray,
    ) -> set[int]:
        if len(views) < 3:
            return set()
        worst = int(np.argmax(leave_one_out_nme))
        other_scores = np.delete(leave_one_out_nme, worst)
        baseline = float(np.median(other_scores))
        if (
            leave_one_out_nme[worst] > 0.035
            and leave_one_out_nme[worst] > max(1.8 * baseline, baseline + 0.015)
        ):
            return {worst}
        residual_nmes = np.asarray(
            [
                np.median(np.linalg.norm(core.fitted[index] - view.landmarks, axis=1)[view.visibility])
                / view.normalizer
                for index, view in enumerate(views)
            ],
            dtype=np.float64,
        )
        threshold = 0.045
        adjacency = consistency <= threshold
        counts = np.sum(adjacency, axis=1)
        max_count = int(np.max(counts))
        if max_count < 2 or max_count == len(views):
            # A gross reprojection outlier can still be rejected even if local
            # signatures are inconclusive.
            center = float(np.median(residual_nmes))
            mad = float(np.median(np.abs(residual_nmes - center)))
            cutoff = max(0.055, center + max(3.5 * 1.4826 * mad, 0.02))
            candidates = set(np.flatnonzero(residual_nmes > cutoff).tolist())
            return candidates if len(views) - len(candidates) >= 2 else set()
        candidate_exemplars = np.flatnonzero(counts == max_count)
        exemplar = min(
            candidate_exemplars.tolist(),
            key=lambda index: (float(np.sum(consistency[index])), index),
        )
        consensus = set(np.flatnonzero(adjacency[exemplar]).tolist())
        rejected = set(range(len(views))) - consensus
        # Do not reject a plausible view on a marginal local-signature split.
        if len(consensus) < 2 or not rejected:
            return set()
        if max(float(consistency[exemplar, index]) for index in rejected) < 0.060:
            return set()
        return rejected

    def _expand_core_to_all_views(
        self, views: list[_PreparedView], accepted: list[int], reduced: _CoreFit
    ) -> _CoreFit:
        camera_by_index: list[CameraEstimate | None] = [None] * len(views)
        nuisance_by_index: list[np.ndarray | None] = [None] * len(views)
        fitted_by_index: list[np.ndarray | None] = [None] * len(views)
        robust_by_index: list[np.ndarray | None] = [None] * len(views)
        view_weights = np.full(len(views), 0.0, dtype=np.float64)
        for reduced_index, original_index in enumerate(accepted):
            camera_by_index[original_index] = reduced.cameras[reduced_index]
            nuisance_by_index[original_index] = reduced.nuisance[reduced_index]
            fitted_by_index[original_index] = reduced.fitted[reduced_index]
            robust_by_index[original_index] = reduced.robust_weights[reduced_index]
            view_weights[original_index] = reduced.view_weights[reduced_index]
        for index, view in enumerate(views):
            if camera_by_index[index] is not None:
                continue
            nuisance = np.zeros(self.nuisance_dim, dtype=np.float64)
            camera = self._initial_camera(view)
            robust = np.ones(LANDMARK_COUNT, dtype=np.float64)
            for _ in range(3):
                shape = self._shape(reduced.beta, nuisance)
                camera = self._refine_camera(view, shape, camera, robust)
                nuisance = self._refine_nuisance(
                    view, reduced.beta, camera, nuisance, robust
                )
                shape = self._shape(reduced.beta, nuisance)
                fitted = project_points(shape, camera)
                distances = np.linalg.norm(fitted - view.landmarks, axis=1)
                cutoff = max(float(np.median(distances[view.visibility])) * 2.5, 1.0)
                robust = np.minimum(1.0, cutoff / np.maximum(distances, 1e-12))
            camera_by_index[index] = camera
            nuisance_by_index[index] = nuisance
            fitted_by_index[index] = project_points(self._shape(reduced.beta, nuisance), camera)
            robust_by_index[index] = robust
        return _CoreFit(
            beta=reduced.beta,
            nuisance=[value for value in nuisance_by_index if value is not None],
            cameras=[value for value in camera_by_index if value is not None],
            fitted=[value for value in fitted_by_index if value is not None],
            robust_weights=[value for value in robust_by_index if value is not None],
            view_weights=view_weights,
            stages=reduced.stages,
            singular_values=reduced.singular_values,
            observable_rank=reduced.observable_rank,
        )

    def _build_result(
        self,
        views: list[_PreparedView],
        core: _CoreFit,
        accepted: list[int],
        rejected: set[int],
        consistency: np.ndarray,
        leave_one_out_nme: np.ndarray,
    ) -> MultiViewFitResult:
        reports: list[ViewFitReport] = []
        all_distances: list[np.ndarray] = []
        weighted_nme_numerator = 0.0
        weighted_nme_denominator = 0.0
        for index, view in enumerate(views):
            distances = np.linalg.norm(core.fitted[index] - view.landmarks, axis=1)
            selected = view.base_weights > 1e-8
            selected_distances = distances[selected]
            nme = float(np.mean(selected_distances) / view.normalizer)
            is_accepted = index in accepted and index not in rejected
            if is_accepted:
                all_distances.append(selected_distances)
                weighted_nme_numerator += float(np.sum(selected_distances / view.normalizer))
                weighted_nme_denominator += float(len(selected_distances))
            reports.append(
                ViewFitReport(
                    index=index,
                    role=view.role,
                    camera_kind=core.cameras[index].kind,
                    accepted=is_accepted,
                    rejection_reason=None if is_accepted else "MIXED_IDENTITY_OR_OUTLIER_VIEW",
                    visible_landmarks=int(np.count_nonzero(selected)),
                    nme=nme,
                    mean_pixel_error=float(np.mean(selected_distances)),
                    median_pixel_error=float(np.median(selected_distances)),
                    robust_view_weight=float(core.view_weights[index]),
                )
            )
        nme = weighted_nme_numerator / max(weighted_nme_denominator, 1e-12)
        concatenated = np.concatenate(all_distances) if all_distances else np.asarray((math.inf,))
        singular = core.singular_values
        condition = (
            float(singular[0] / singular[core.observable_rank - 1])
            if core.observable_rank > 0 and singular.size >= core.observable_rank
            else math.inf
        )
        active = core.stages[-1]
        saturation = float(
            np.mean(np.abs(core.beta[:active]) >= IDENTITY_COEFFICIENT_BOUND - 1.0e-3)
        )
        accepted_flag = (
            len(accepted) >= 2
            and np.isfinite(nme)
            and nme <= 0.08
            and saturation <= 0.35
            and core.observable_rank >= min(20, active)
        )
        identity = np.zeros(self.adapter.identity_dim, dtype=np.float32)
        identity[:GNM_LANDMARK_OBSERVABLE_IDENTITY_DIM] = core.beta.astype(np.float32)
        return MultiViewFitResult(
            identity=identity,
            nuisance=tuple(value.astype(np.float32) for value in core.nuisance),
            cameras=tuple(core.cameras),
            fitted_landmarks=tuple(value.astype(np.float32) for value in core.fitted),
            report=MultiViewFitReport(
                accepted=accepted_flag,
                nme=float(nme),
                mean_pixel_error=float(np.mean(concatenated)),
                per_view=tuple(reports),
                accepted_view_indices=tuple(accepted),
                rejected_view_indices=tuple(sorted(rejected)),
                unlocked_stages=core.stages,
                observable_rank=core.observable_rank,
                active_identity_modes=active,
                weakly_observable_directions=max(0, active - core.observable_rank),
                condition_number=condition,
                observability_ratio=float(core.observable_rank / active),
                saturation_fraction=saturation,
                identity_solver=IDENTITY_SOLVER,
                identity_coefficient_bound=IDENTITY_COEFFICIENT_BOUND,
                nuisance_coefficient_bound=NUISANCE_COEFFICIENT_BOUND,
                identity_consistency_matrix=consistency.astype(np.float32),
                leave_one_out_nme=leave_one_out_nme.astype(np.float32),
                metric_scale_caveat=METRIC_SCALE_CAVEAT,
            ),
        )
