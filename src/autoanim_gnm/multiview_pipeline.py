"""Guided multi-photo identity fitting and provenance-aware texture export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .artifacts import sha256
from .camera_bundle import (
    CALIBRATION_POSE_GATE_DEGREES,
    CALIBRATION_RMS_GATE_PX,
    CALIBRATION_SCALE_GATE_FRACTION,
    CalibratedCameraBundle,
    CameraRegistration,
    estimate_camera_registration,
    load_camera_bundle,
    perspective_camera_from_calibration,
    project_calibrated_points,
)
from .errors import AutoAnimError
from .gltf_export import export_gnm_glb
from .gnm_adapter import GNMAdapter
from .gnm_texture import build_gnm_texture_atlas
from .image import DetectedFace, FaceExtractor, MAPPING_NAME
from .image_pipeline import draw_overlay
from .multiview import (
    CameraIntrinsics,
    METRIC_SCALE_CAVEAT,
    MultiViewIdentityFitter,
    MultiViewObservation,
    PERSPECTIVE_CAMERA_CONVENTION,
    PerspectiveCamera as FitPerspectiveCamera,
    rotation_matrix,
)
from .render import MeshRenderer
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json, write_npz
from .texture_baker import (
    PerspectiveCamera as TexturePerspectiveCamera,
    bake_multiview_texture,
)


PSEUDO_INTRINSICS_CAVEAT = (
    "Camera intrinsics are estimated from image dimensions unless calibrated capture "
    "metadata is supplied; metric depth and fine profile geometry are therefore approximate."
)
SPARSE_IDENTITY_CAVEAT = (
    "GNM identity modes 170-252 have no support at the sparse 68 landmarks and remain neutral."
)
TEXTURE_PROVENANCE_CAVEAT = (
    "Only texels labeled observed come from a source photo. Inpainted and generic texels are "
    "explicitly marked and must not be presented as measured appearance."
)
REAR_VIEW_CAVEAT = (
    "Rear-head views have no facial landmarks and require calibrated turntable, SfM, or dense "
    "registration; this guided landmark pipeline supports front, three-quarter, and profile views."
)
NEUTRAL_TEXTURE_CAVEAT = (
    "Person-specific texture is baked onto one neutral GNM mesh. Captures with material "
    "expression are rejected until per-view expression geometry is supported."
)
CALIBRATED_CAPTURE_CAVEAT = (
    "Measured camera calibration and held-out views make the geometric audit stronger, but "
    "they do not establish performer consent, metric scan accuracy, or perceptual likeness."
)

HELDOUT_AGGREGATE_NME_GATE = 0.035
HELDOUT_PER_VIEW_NME_GATE = 0.050
CALIBRATED_FIT_NME_GATE = 0.025
CALIBRATED_PER_FIT_VIEW_NME_GATE = 0.040
CALIBRATED_OBSERVABLE_RANK_GATE = 145
CALIBRATED_OBSERVABILITY_RATIO_GATE = 0.85
CALIBRATED_SATURATION_FRACTION_GATE = 0.10
CALIBRATED_YAW_SPAN_GATE_DEGREES = 120.0

MAX_TEXTURE_BLENDSHAPE_SCORE = 0.55
MAX_TEXTURE_NUISANCE_COEFFICIENT = 0.25

SUPPORTED_ROLE_HINTS = (
    "front",
    "left_3q",
    "right_3q",
    "left_profile",
    "right_profile",
)


@dataclass(slots=True)
class _CalibratedFitOutcome:
    fitted: object
    registration: CameraRegistration
    observations: tuple[MultiViewObservation, ...]
    cameras: tuple[FitPerspectiveCamera, ...]
    fitted_landmarks: tuple[np.ndarray, ...]
    nuisance: tuple[np.ndarray, ...]
    fit_indices: tuple[int, ...]
    held_out_indices: tuple[int, ...]
    accepted_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    held_out_report: dict
    observability: dict


def _undistort_detection(
    detection: DetectedFace,
    view,
) -> DetectedFace:
    if np.max(np.abs(view.distortion), initial=0.0) <= 1e-15:
        # Validate the decoded size but otherwise preserve detector confidence
        # and exact coordinates in the zero-distortion identity case.
        view.undistort_image(detection.image_bgr)
        return detection
    image = view.undistort_image(detection.image_bgr)
    landmarks = view.undistort_points(detection.landmarks)
    all_landmarks = view.undistort_points(detection.all_landmarks)
    height, width = image.shape[:2]
    margin_x, margin_y = 0.05 * width, 0.05 * height
    inside = (
        (landmarks[:, 0] >= -margin_x)
        & (landmarks[:, 0] <= width + margin_x)
        & (landmarks[:, 1] >= -margin_y)
        & (landmarks[:, 1] <= height + margin_y)
    )
    return DetectedFace(
        image_bgr=image,
        landmarks=landmarks,
        all_landmarks=all_landmarks,
        blendshapes=dict(detection.blendshapes),
        face_width=float(np.ptp(landmarks[:, 0])),
        mapped_in_bounds_fraction=float(np.mean(inside)),
        strong_expression_score=detection.strong_expression_score,
    )


def _view_normalizer(points: np.ndarray, visibility: np.ndarray) -> float:
    selected = np.asarray(visibility, dtype=np.float64) > 1e-8
    available = np.asarray(points, dtype=np.float64)[selected]
    extent = max(float(np.ptp(available[:, 0])), float(np.ptp(available[:, 1])))
    eye = (
        float(np.linalg.norm(points[36] - points[45]))
        if selected[36] and selected[45]
        else 0.0
    )
    return max(eye, 0.35 * extent, 1.0)


def _held_out_report(
    detections: Sequence[DetectedFace],
    predicted: Sequence[np.ndarray],
    bundle: CalibratedCameraBundle,
) -> dict:
    per_view: list[dict] = []
    aggregate_numerator = 0.0
    aggregate_denominator = 0.0
    for index in bundle.held_out_indices:
        view = bundle.views[index]
        selected = view.visibility > 1e-8
        distances = np.linalg.norm(
            np.asarray(predicted[index], dtype=np.float64)[selected]
            - np.asarray(detections[index].landmarks, dtype=np.float64)[selected],
            axis=1,
        )
        normalizer = _view_normalizer(detections[index].landmarks, view.visibility)
        normalized = distances / normalizer
        weights = np.asarray(view.visibility[selected], dtype=np.float64)
        nme = float(np.sum(weights * normalized) / np.sum(weights))
        aggregate_numerator += float(np.sum(weights * normalized))
        aggregate_denominator += float(np.sum(weights))
        per_view.append(
            {
                "view_index": index,
                "role": view.role,
                "visible_landmarks": int(np.count_nonzero(selected)),
                "nme": nme,
                "mean_pixel_error": float(np.mean(distances)),
                "p95_pixel_error": float(np.percentile(distances, 95)),
                "passed": nme <= HELDOUT_PER_VIEW_NME_GATE,
            }
        )
    aggregate = (
        float(aggregate_numerator / aggregate_denominator)
        if aggregate_denominator > 0.0
        else None
    )
    maximum = max((float(value["nme"]) for value in per_view), default=None)
    passed = bool(
        aggregate is not None
        and maximum is not None
        and aggregate <= HELDOUT_AGGREGATE_NME_GATE
        and maximum <= HELDOUT_PER_VIEW_NME_GATE
    )
    return {
        "evaluated": bool(per_view),
        "fit_leakage": False,
        "aggregate_nme": aggregate,
        "maximum_view_nme": maximum,
        "aggregate_nme_gate": HELDOUT_AGGREGATE_NME_GATE,
        "per_view_nme_gate": HELDOUT_PER_VIEW_NME_GATE,
        "passed": passed,
        "per_view": per_view,
    }


def _registration_json(
    registration: CameraRegistration, bundle: CalibratedCameraBundle
) -> dict:
    return {
        **registration.as_dict(),
        "meters_per_world_unit": bundle.meters_per_world_unit,
        "meters_per_gnm_model_unit": (
            registration.scale * bundle.meters_per_world_unit
        ),
    }


def _calibrated_observability_report(
    adapter: GNMAdapter,
    fitter: MultiViewIdentityFitter,
    fitted,
    bundle: CalibratedCameraBundle,
    registration: CameraRegistration,
    detections: Sequence[DetectedFace],
) -> dict:
    """Measure identity rank after removing shared pose/scale and view expression.

    Unlike the legacy rank, the seven registration columns are global across all
    fit views.  Per-view nuisance columns remain block diagonal.  Reporting the
    rank conditional on a pre-solved registration would overstate evidence.
    """

    accepted_local = tuple(fitted.report.accepted_view_indices)
    fit_indices = bundle.fit_indices
    identity_modes = min(170, adapter.identity_dim)
    identity_basis = np.asarray(
        adapter.compact_identity_basis[:identity_modes], dtype=np.float64
    )
    nuisance_basis = np.asarray(fitter.nuisance_basis, dtype=np.float64)
    neutral = np.asarray(adapter.compact_template, dtype=np.float64)
    identity_shape = neutral + np.einsum(
        "i,ilc->lc", fitted.identity, adapter.compact_identity_basis, optimize=True
    )
    shapes = [
        identity_shape
        + np.einsum(
            "i,ilc->lc", fitted.nuisance[local_index], nuisance_basis, optimize=True
        )
        for local_index in accepted_local
    ]
    accepted_global = [fit_indices[local_index] for local_index in accepted_local]
    accepted_views = [bundle.views[index] for index in accepted_global]
    base_weights: list[np.ndarray] = []
    robust_weights: list[np.ndarray] = []
    normalizers: list[float] = []
    normalized_medians: list[float] = []
    for local_index, global_index, view in zip(
        accepted_local, accepted_global, accepted_views, strict=True
    ):
        detection = detections[global_index]
        base = (
            detection.mapped_in_bounds_fraction
            * np.asarray(view.visibility, dtype=np.float64)
            * fitter.landmark_weights
        )
        selected = base > 1e-8
        normalizer = _view_normalizer(detection.landmarks, base)
        distances = np.linalg.norm(
            fitted.fitted_landmarks[local_index] - detection.landmarks, axis=1
        )
        selected_distances = distances[selected]
        median = float(np.median(selected_distances))
        mad = float(np.median(np.abs(selected_distances - median)))
        sigma = max(1.4826 * mad, 0.003 * normalizer, 0.25)
        cutoff = 1.345 * sigma
        robust = np.ones(68, dtype=np.float64)
        large = distances > cutoff
        robust[large] = cutoff / np.maximum(distances[large], 1e-12)
        robust[~selected] = 0.0
        base_weights.append(base)
        robust_weights.append(robust)
        normalizers.append(normalizer)
        normalized_medians.append(median / normalizer)
    medians = np.asarray(normalized_medians, dtype=np.float64)
    median_center = float(np.median(medians))
    median_scale = max(
        1.4826 * float(np.median(np.abs(medians - median_center))), 0.004
    )
    view_weights = np.ones(len(accepted_views), dtype=np.float64)
    high = medians > median_center + 1.5 * median_scale
    view_weights[high] = np.maximum(
        0.15,
        (median_center + 1.5 * median_scale) / np.maximum(medians[high], 1e-12),
    )
    evidence_weights = [
        np.sqrt(np.maximum(base * robust * view_weight, 0.0))
        for base, robust, view_weight in zip(
            base_weights, robust_weights, view_weights, strict=True
        )
    ]
    row_counts = [
        2 * int(np.count_nonzero(base > 1e-8)) for base in base_weights
    ]
    total_rows = sum(row_counts)
    identity_jacobian = np.zeros((total_rows, identity_modes), dtype=np.float64)
    nuisance_jacobian = np.zeros(
        (total_rows, len(accepted_views) * fitter.nuisance_dim), dtype=np.float64
    )

    def point_jacobian(shape: np.ndarray, view) -> np.ndarray:
        world = registration.scale * (shape @ registration.rotation.T) + registration.translation
        camera_rotation = view.world_to_camera[:3, :3]
        camera = world @ camera_rotation.T + view.world_to_camera[:3, 3]
        x, y, depth = camera[:, 0], camera[:, 1], camera[:, 2]
        local = np.zeros((len(shape), 2, 3), dtype=np.float64)
        local[:, 0, 0] = view.intrinsics_matrix[0, 0] / depth
        local[:, 0, 2] = -view.intrinsics_matrix[0, 0] * x / (depth * depth)
        local[:, 1, 1] = view.intrinsics_matrix[1, 1] / depth
        local[:, 1, 2] = -view.intrinsics_matrix[1, 1] * y / (depth * depth)
        model_to_camera = registration.scale * camera_rotation @ registration.rotation
        return np.einsum("nab,bc->nac", local, model_to_camera)

    cursor = 0
    for accepted_index, (shape, view) in enumerate(zip(shapes, accepted_views, strict=True)):
        selected = base_weights[accepted_index] > 1e-8
        weights = evidence_weights[accepted_index][selected]
        normalizer = normalizers[accepted_index]
        derivative = point_jacobian(shape, view)
        jb = np.einsum("nac,inc->nai", derivative, identity_basis)[selected]
        jn = np.einsum("nac,inc->nai", derivative, nuisance_basis)[selected]
        count = row_counts[accepted_index]
        identity_jacobian[cursor : cursor + count] = (
            jb * weights[:, None, None] / normalizer
        ).reshape(count, identity_modes)
        start = accepted_index * fitter.nuisance_dim
        nuisance_jacobian[cursor : cursor + count, start : start + fitter.nuisance_dim] = (
            jn * weights[:, None, None] / normalizer
        ).reshape(count, fitter.nuisance_dim)
        cursor += count

    rotation_vector = Rotation.from_matrix(registration.rotation).as_rotvec()
    encoded = np.concatenate(
        (rotation_vector, registration.translation, np.asarray((np.log(registration.scale),)))
    )

    def decode(values: np.ndarray) -> CameraRegistration:
        return CameraRegistration(
            float(np.exp(values[6])),
            Rotation.from_rotvec(values[:3]).as_matrix(),
            values[3:6],
            registration.mean_reprojection_error_px,
            registration.p95_reprojection_error_px,
        )

    registration_jacobian = np.zeros((total_rows, 7), dtype=np.float64)
    for column in range(7):
        step = 1e-6 * max(1.0, abs(float(encoded[column])))
        positive, negative = encoded.copy(), encoded.copy()
        positive[column] += step
        negative[column] -= step
        positive_registration = decode(positive)
        negative_registration = decode(negative)
        cursor = 0
        for accepted_index, (shape, view, count) in enumerate(
            zip(shapes, accepted_views, row_counts, strict=True)
        ):
            selected = base_weights[accepted_index] > 1e-8
            weights = evidence_weights[accepted_index][selected]
            normalizer = normalizers[accepted_index]
            derivative = (
                project_calibrated_points(shape, view, positive_registration)[selected]
                - project_calibrated_points(shape, view, negative_registration)[selected]
            ) / (2.0 * step)
            registration_jacobian[cursor : cursor + count, column] = (
                derivative * weights[:, None] / normalizer
            ).reshape(count)
            cursor += count

    confounds = np.column_stack((nuisance_jacobian, registration_jacobian))
    confound_u, confound_singular, _ = np.linalg.svd(confounds, full_matrices=False)
    confound_rank = (
        int(np.count_nonzero(confound_singular > confound_singular[0] * 1e-8))
        if confound_singular.size and confound_singular[0] > 1e-12
        else 0
    )
    if confound_rank:
        confound_basis = confound_u[:, :confound_rank]
        observable = identity_jacobian - confound_basis @ (
            confound_basis.T @ identity_jacobian
        )
    else:
        observable = identity_jacobian
    singular = np.linalg.svd(observable, compute_uv=False)
    rank = (
        int(np.count_nonzero(singular > singular[0] * fitter.observability_rtol))
        if singular.size and singular[0] > 1e-12
        else 0
    )
    condition = (
        float(singular[0] / singular[rank - 1]) if rank > 0 else float("inf")
    )
    return {
        "method": "effective_evidence_identity_rank_marginalized_shared_similarity_and_per_view_nuisance_v2",
        "active_identity_modes": identity_modes,
        "observable_rank": rank,
        "observability_ratio": float(rank / identity_modes),
        "weakly_observable_directions": identity_modes - rank,
        "condition_number": condition,
        "shared_similarity_confounds": 7,
        "per_view_nuisance_confounds": len(accepted_views) * fitter.nuisance_dim,
    }


def _fit_with_calibrated_bundle(
    adapter: GNMAdapter,
    rig: ControlRig,
    detections: Sequence[DetectedFace],
    bundle: CalibratedCameraBundle,
) -> _CalibratedFitOutcome:
    fitter = MultiViewIdentityFitter(adapter, rig)
    fit_indices = bundle.fit_indices
    held_out_indices = bundle.held_out_indices
    fit_views = tuple(bundle.views[index] for index in fit_indices)
    fit_landmarks = tuple(detections[index].landmarks for index in fit_indices)
    neutral = np.asarray(adapter.compact_template, dtype=np.float64)
    registration = estimate_camera_registration(
        tuple(neutral for _ in fit_indices),
        fit_landmarks,
        fit_views,
        meters_per_world_unit=bundle.meters_per_world_unit,
    )
    fitted = None
    observations = ()
    previous_accepted: tuple[int, ...] | None = None

    def observations_for(current: CameraRegistration) -> tuple[MultiViewObservation, ...]:
        fit_cameras = tuple(
            perspective_camera_from_calibration(view, current) for view in fit_views
        )
        return tuple(
            MultiViewObservation(
                detections[index].landmarks,
                detections[index].image_bgr.shape[:2],
                intrinsics=bundle.views[index].intrinsics,
                role=bundle.views[index].role,
                confidence=detections[index].mapped_in_bounds_fraction,
                visibility=bundle.views[index].visibility,
                initial_camera=camera,
                lock_camera=True,
            )
            for index, camera in zip(fit_indices, fit_cameras, strict=True)
        )

    def shapes_for(result, accepted_local: Sequence[int]) -> tuple[np.ndarray, ...]:
        return tuple(
            neutral
            + np.einsum(
                "i,ilc->lc",
                result.identity,
                adapter.compact_identity_basis,
                optimize=True,
            )
            + np.einsum(
                "i,ilc->lc",
                result.nuisance[local_index],
                fitter.nuisance_basis,
                optimize=True,
            )
            for local_index in accepted_local
        )

    for _ in range(15):
        observations = observations_for(registration)
        current_fit = fitter.fit(observations)
        accepted_local = tuple(current_fit.report.accepted_view_indices)
        if len(accepted_local) < 3:
            raise ValueError("Calibrated fitting retained fewer than three fit views")
        registration_shapes = shapes_for(current_fit, accepted_local)
        accepted_landmarks = tuple(
            fit_landmarks[local_index] for local_index in accepted_local
        )
        accepted_views = tuple(fit_views[local_index] for local_index in accepted_local)
        candidate = estimate_camera_registration(
            registration_shapes,
            accepted_landmarks,
            accepted_views,
            meters_per_world_unit=bundle.meters_per_world_unit,
        )
        projection_change = np.concatenate(
            [
                np.linalg.norm(
                    project_calibrated_points(shape, view, candidate)
                    - project_calibrated_points(shape, view, registration),
                    axis=1,
                )
                for shape, view in zip(registration_shapes, accepted_views, strict=True)
            ]
        )
        registration = candidate
        if (
            previous_accepted == accepted_local
            and float(np.max(projection_change, initial=0.0)) <= 2e-2
        ):
            final_observations = observations_for(registration)
            final_fit = fitter.fit(final_observations)
            final_accepted = tuple(final_fit.report.accepted_view_indices)
            if final_accepted == accepted_local:
                final_shapes = shapes_for(final_fit, final_accepted)
                final_errors = np.concatenate(
                    [
                        np.linalg.norm(
                            project_calibrated_points(shape, view, registration)[
                                view.visibility > 1e-8
                            ]
                            - observed[view.visibility > 1e-8],
                            axis=1,
                        )
                        for shape, observed, view in zip(
                            final_shapes, accepted_landmarks, accepted_views, strict=True
                        )
                    ]
                )
                registration = CameraRegistration(
                    registration.scale,
                    registration.rotation,
                    registration.translation,
                    float(np.mean(final_errors)),
                    float(np.percentile(final_errors, 95)),
                )
                fitted = final_fit
                observations = final_observations
                break
            previous_accepted = final_accepted
            continue
        previous_accepted = accepted_local
    else:
        raise ValueError(
            "Calibrated identity/registration solve did not reach a stable accepted-view set"
        )

    assert fitted is not None
    all_cameras = tuple(
        perspective_camera_from_calibration(view, registration) for view in bundle.views
    )
    identity_shape = neutral + np.einsum(
        "i,ilc->lc",
        fitted.identity,
        adapter.compact_identity_basis,
        optimize=True,
    )
    predictions: list[np.ndarray] = [np.zeros((68, 2), dtype=np.float32) for _ in detections]
    nuisance: list[np.ndarray] = [
        np.zeros(fitter.nuisance_dim, dtype=np.float32) for _ in detections
    ]
    for local_index, global_index in enumerate(fit_indices):
        predictions[global_index] = np.asarray(
            fitted.fitted_landmarks[local_index], dtype=np.float32
        )
        nuisance[global_index] = np.asarray(fitted.nuisance[local_index], dtype=np.float32)
    for index in held_out_indices:
        predictions[index] = project_calibrated_points(
            identity_shape, bundle.views[index], registration
        ).astype(np.float32)
    accepted = tuple(fit_indices[index] for index in fitted.report.accepted_view_indices)
    rejected = tuple(fit_indices[index] for index in fitted.report.rejected_view_indices)
    observability = _calibrated_observability_report(
        adapter, fitter, fitted, bundle, registration, detections
    )
    return _CalibratedFitOutcome(
        fitted=fitted,
        registration=registration,
        observations=observations,
        cameras=all_cameras,
        fitted_landmarks=tuple(predictions),
        nuisance=tuple(nuisance),
        fit_indices=fit_indices,
        held_out_indices=held_out_indices,
        accepted_indices=accepted,
        rejected_indices=rejected,
        held_out_report=_held_out_report(detections, predictions, bundle),
        observability=observability,
    )


def _normalise_roles(roles: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if roles is None or len(roles) == 0:
        defaults = ("front", "left_3q", "right_3q", "left_profile", "right_profile")
        return tuple(defaults[index] if index < len(defaults) else f"view_{index + 1}" for index in range(count))
    values = tuple(str(role).strip().lower().replace("-", "_").replace(" ", "_") for role in roles)
    if len(values) != count or any(not value for value in values):
        raise AutoAnimError(
            "INPUT_INVALID",
            "Provide exactly one non-empty capture role per image.",
            {"images": count, "roles": len(values), "supported_role_hints": SUPPORTED_ROLE_HINTS},
        )
    if any("back" in value or "rear" in value for value in values):
        raise AutoAnimError(
            "INPUT_INVALID",
            REAR_VIEW_CAVEAT,
            {"supported_role_hints": SUPPORTED_ROLE_HINTS},
        )
    return values


def _circular_yaw_span_degrees(yaws: Sequence[float]) -> float:
    angles = np.mod(np.asarray(tuple(yaws), dtype=np.float64), 2.0 * np.pi)
    if angles.size < 2 or not np.isfinite(angles).all():
        return 0.0
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * np.pi)))
    return float(np.degrees(2.0 * np.pi - np.max(gaps)))


def _assumed_intrinsics(image_shape: tuple[int, int], focal_scale: float) -> CameraIntrinsics:
    height, width = image_shape
    focal = float(focal_scale * max(height, width))
    return CameraIntrinsics(focal, focal, 0.5 * (width - 1), 0.5 * (height - 1))


def texture_camera_from_fit(camera: FitPerspectiveCamera) -> TexturePerspectiveCamera:
    """Convert the physical front-facing fit camera to OpenCV baker coordinates."""

    rotation = rotation_matrix(camera.yaw, camera.pitch, camera.roll)
    world_to_camera = np.eye(4, dtype=np.float64)
    # GNM is +X right, +Y up, +Z toward the viewer.  OpenCV camera space is
    # +X right, +Y down, +Z into the scene.  The two axis flips form a proper
    # 180-degree X rotation (determinant +1), not a reflection/back camera.
    world_to_camera[0, :3] = rotation[0]
    world_to_camera[1, :3] = -rotation[1]
    world_to_camera[2, :3] = -rotation[2]
    world_to_camera[:3, 3] = (camera.tx, -camera.ty, camera.tz)
    intrinsics = np.asarray(
        (
            (camera.intrinsics.fx, 0.0, camera.intrinsics.cx),
            (0.0, camera.intrinsics.fy, camera.intrinsics.cy),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return TexturePerspectiveCamera(intrinsics, world_to_camera)


def _face_mask(all_landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    points = np.rint(np.asarray(all_landmarks, dtype=np.float64)).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    hull = cv2.convexHull(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
    return mask > 0


def _component_texture_metrics(baked, atlas) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    """Audit direct/fill coverage separately for every GNM anatomy tile."""

    triangle_index = np.asarray(baked.triangle_index, dtype=np.int64)
    if triangle_index.shape != baked.atlas_mask.shape:
        raise RuntimeError("Texture triangle-index map does not match the atlas")
    component_map = np.full(triangle_index.shape, -1, dtype=np.int16)
    occupied = np.asarray(baked.atlas_mask, dtype=bool)
    if np.any(occupied):
        indices = triangle_index[occupied]
        if np.min(indices) < 0 or np.max(indices) >= len(atlas.triangle_components):
            raise RuntimeError("Texture triangle-index map references invalid topology")
        component_map[occupied] = atlas.triangle_components[indices]

    result: dict[str, dict[str, float | int]] = {}
    provenance = {
        "observed": np.asarray(baked.observed, dtype=bool),
        "mirrored": np.asarray(baked.mirrored, dtype=bool),
        "inpainted": np.asarray(baked.inpainted, dtype=bool),
        "generic": np.asarray(baked.generic, dtype=bool),
    }
    for component_index, name in enumerate(atlas.component_names):
        selected = occupied & (component_map == component_index)
        count = int(np.count_nonzero(selected))
        metrics: dict[str, float | int] = {"atlas_texels": count}
        for provenance_name, values in provenance.items():
            provenance_count = int(np.count_nonzero(selected & values))
            metrics[f"{provenance_name}_texels"] = provenance_count
            metrics[f"{provenance_name}_fraction"] = (
                float(provenance_count / count) if count else 0.0
            )
        if count and sum(
            int(metrics[f"{name}_texels"]) for name in provenance
        ) != count:
            raise RuntimeError(f"Texture provenance is incomplete for {name!r}")
        result[name] = metrics
    return component_map, result


def _save_texture_artifacts(
    output: Path,
    baked,
    atlas,
    component_map: np.ndarray,
    texture_view_local_to_global: Sequence[int],
) -> None:
    Image.fromarray(baked.rgba, mode="RGBA").save(output / "texture.png")
    Image.fromarray(
        np.rint(np.clip(baked.confidence, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).save(output / "texture-confidence.png")
    provenance = np.zeros((*baked.atlas_mask.shape, 4), dtype=np.uint8)
    provenance[baked.observed] = (84, 255, 150, 255)
    provenance[baked.mirrored] = (255, 210, 74, 255)
    provenance[baked.inpainted] = (75, 210, 255, 255)
    provenance[baked.generic] = (255, 86, 166, 255)
    Image.fromarray(provenance, mode="RGBA").save(output / "texture-provenance.png")
    local_to_global = np.asarray(texture_view_local_to_global, dtype=np.int32)
    if local_to_global.shape != (len(baked.color_gain),):
        raise RuntimeError("Texture source-index mapping does not match baked view count")
    source_view_global = np.full(baked.source_view.shape, -1, dtype=np.int32)
    sourced = baked.source_view >= 0
    if np.any(sourced):
        local_indices = np.asarray(baked.source_view[sourced], dtype=np.int64)
        if np.max(local_indices) >= len(local_to_global):
            raise RuntimeError("Texture source map contains an invalid local view index")
        source_view_global[sourced] = local_to_global[local_indices]
    write_npz(
        output / "texture-maps.npz",
        rgba=baked.rgba,
        confidence=baked.confidence,
        source_view=baked.source_view,
        source_view_global=source_view_global,
        texture_view_local_to_global=local_to_global,
        observed=baked.observed,
        mirrored=baked.mirrored,
        inpainted=baked.inpainted,
        generic=baked.generic,
        atlas_mask=baked.atlas_mask,
        triangle_index=baked.triangle_index,
        component_index=component_map,
        overlap_count=baked.overlap_count,
        color_gain=baked.color_gain,
        color_bias=baked.color_bias,
        triangle_uvs=atlas.triangle_uvs,
        triangle_components=atlas.triangle_components,
        component_names=np.asarray(atlas.component_names),
    )


def _report_json(result) -> dict:
    report = result.report
    return {
        "accepted": report.accepted,
        "nme": report.nme,
        "mean_pixel_error": report.mean_pixel_error,
        "accepted_view_indices": list(report.accepted_view_indices),
        "rejected_view_indices": list(report.rejected_view_indices),
        "unlocked_stages": list(report.unlocked_stages),
        "observable_rank": report.observable_rank,
        "active_identity_modes": report.active_identity_modes,
        "weakly_observable_directions": report.weakly_observable_directions,
        "condition_number": report.condition_number,
        "observability_ratio": report.observability_ratio,
        "saturation_fraction": report.saturation_fraction,
        "identity_solver": report.identity_solver,
        "identity_coefficient_bound": report.identity_coefficient_bound,
        "nuisance_coefficient_bound": report.nuisance_coefficient_bound,
        "identity_consistency_matrix": report.identity_consistency_matrix.tolist(),
        "leave_one_out_nme": report.leave_one_out_nme.tolist(),
        "metric_scale_caveat": report.metric_scale_caveat,
        "per_view": [asdict(value) for value in report.per_view],
    }


def _validate_neutral_texture_captures(
    detections, fitted, accepted: Sequence[int], roles: Sequence[str]
) -> None:
    """Fail closed when neutral bake geometry cannot represent a source view."""

    offenders: list[dict[str, float | int | str]] = []
    for index in accepted:
        expression_score = float(detections[index].strong_expression_score)
        nuisance_peak = float(np.max(np.abs(fitted.nuisance[index])))
        if (
            expression_score > MAX_TEXTURE_BLENDSHAPE_SCORE
            or nuisance_peak > MAX_TEXTURE_NUISANCE_COEFFICIENT
        ):
            offenders.append(
                {
                    "view_index": int(index),
                    "role": roles[index],
                    "strong_expression_score": expression_score,
                    "fitted_nuisance_peak": nuisance_peak,
                }
            )
    if offenders:
        raise AutoAnimError(
            "FIT_REJECTED",
            "Texture captures must be neutral because the current baker uses one shared "
            "neutral mesh and cannot align per-view smiles, blinks, or open mouths.",
            {
                "views": offenders,
                "maximum_strong_expression_score": MAX_TEXTURE_BLENDSHAPE_SCORE,
                "maximum_fitted_nuisance_coefficient": MAX_TEXTURE_NUISANCE_COEFFICIENT,
                "alternative": (
                    "Retake front/three-quarter/profile photos with relaxed lips, open eyes, "
                    "and a neutral brow, or use a future per-view expression-aware bake."
                ),
            },
        )


def run_multiview_pipeline(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    model_path: str | Path,
    roles: Sequence[str] | None = None,
    texture_size: int = 256,
    focal_scale: float = 1.25,
    mirror_fill: bool = False,
    camera_bundle_path: str | Path | None = None,
    input_names: Sequence[str] | None = None,
) -> dict:
    """Fit one GNM identity and bake an auditable texture from 2-12 photos."""

    paths = tuple(Path(path) for path in input_paths)
    if not 2 <= len(paths) <= 12:
        raise AutoAnimError("INPUT_INVALID", "Multi-view reconstruction requires 2-12 images")
    if texture_size not in {128, 256, 512, 1024}:
        raise AutoAnimError("INPUT_INVALID", "Texture size must be 128, 256, 512, or 1024")
    if not np.isfinite(focal_scale) or not 0.7 <= focal_scale <= 2.5:
        raise AutoAnimError("INPUT_INVALID", "Assumed focal scale must be in [0.7,2.5]")
    if mirror_fill:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Atlas mirror fill is disabled for GNM because its anatomical UV tiles are not "
            "horizontal mirror pairs. Capture the missing side or use a topology-aware "
            "vertex-symmetry transfer with explicit provenance instead.",
        )
    missing_paths = [index for index, path in enumerate(paths) if not path.is_file()]
    if missing_paths:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Every multi-view input must be an existing file",
            {"missing_view_indices": missing_paths},
        )
    if input_names is not None and len(input_names) != len(paths):
        raise AutoAnimError(
            "INPUT_INVALID", "input_names must contain exactly one name per image"
        )
    source_names = tuple(input_names or tuple(path.name for path in paths))
    capture_roles = _normalise_roles(roles, len(paths))
    digests = tuple(sha256(path) for path in paths)
    duplicate_groups = [
        [index for index, value in enumerate(digests) if value == digest]
        for digest in dict.fromkeys(digests)
    ]
    duplicate_groups = [group for group in duplicate_groups if len(group) > 1]
    if duplicate_groups:
        raise AutoAnimError(
            "FIT_REJECTED",
            "Accepted photos do not provide enough viewpoint diversity for a multi-view fit: "
            "the same image was supplied more than once.",
            {
                "yaw_span_degrees": 0.0,
                "minimum_yaw_span_degrees": 20.0,
                "duplicate_view_indices": duplicate_groups,
                "alternative": "Remove duplicate files and capture distinct front, three-quarter, or profile views.",
            },
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    extractor = FaceExtractor(model_path)
    detections = []
    for index, path in enumerate(paths):
        try:
            detections.append(extractor.detect(path))
        except AutoAnimError as exc:
            raise AutoAnimError(
                exc.code,
                f"View {index + 1} ({capture_roles[index]}) failed: {exc.message}",
                {**exc.details, "view_index": index, "role": capture_roles[index]},
            ) from exc

    camera_bundle: CalibratedCameraBundle | None = None
    if camera_bundle_path is not None:
        camera_bundle = load_camera_bundle(
            camera_bundle_path,
            input_names=source_names,
            image_sizes=tuple(detection.image_bgr.shape[:2] for detection in detections),
        )
        bundle_roles = tuple(view.role for view in camera_bundle.views)
        if roles is not None and len(roles) and capture_roles != bundle_roles:
            raise AutoAnimError(
                "INPUT_INVALID",
                "Form/CLI roles conflict with calibrated camera-bundle roles",
                {"roles": list(capture_roles), "bundle_roles": list(bundle_roles)},
            )
        capture_roles = bundle_roles
        if not camera_bundle.declared_calibration_metadata_gate_passed:
            raise AutoAnimError(
                "FIT_REJECTED",
                "Capture calibration does not pass the requester-declared metadata gates.",
                {
                    "calibration_rms_px": camera_bundle.calibration_rms_px,
                    "maximum_calibration_rms_px": CALIBRATION_RMS_GATE_PX,
                    "pose_error_degrees": camera_bundle.pose_error_degrees,
                    "maximum_pose_error_degrees": CALIBRATION_POSE_GATE_DEGREES,
                    "scale_error_fraction": camera_bundle.scale_error_fraction,
                    "maximum_scale_error_fraction": CALIBRATION_SCALE_GATE_FRACTION,
                },
            )
        detections = [
            _undistort_detection(detection, view)
            for detection, view in zip(detections, camera_bundle.views, strict=True)
        ]

    adapter = GNMAdapter()
    decoder = ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5")
    rig = ControlRig(adapter, decoder)
    calibrated_outcome: _CalibratedFitOutcome | None = None
    try:
        if camera_bundle is not None:
            calibrated_outcome = _fit_with_calibrated_bundle(
                adapter, rig, detections, camera_bundle
            )
            fitted = calibrated_outcome.fitted
            observations = tuple(
                MultiViewObservation(
                    detection.landmarks,
                    detection.image_bgr.shape[:2],
                    intrinsics=view.intrinsics,
                    role=view.role,
                    confidence=detection.mapped_in_bounds_fraction,
                    visibility=view.visibility,
                    initial_camera=calibrated_outcome.cameras[index],
                    lock_camera=True,
                )
                for index, (detection, view) in enumerate(
                    zip(detections, camera_bundle.views, strict=True)
                )
            )
            accepted = calibrated_outcome.accepted_indices
            rejected = calibrated_outcome.rejected_indices
            all_cameras = calibrated_outcome.cameras
            all_predictions = calibrated_outcome.fitted_landmarks
            all_nuisance = calibrated_outcome.nuisance
            fit_indices = calibrated_outcome.fit_indices
            held_out_indices = calibrated_outcome.held_out_indices
            held_out_report = calibrated_outcome.held_out_report
        else:
            observations = tuple(
                MultiViewObservation(
                    detection.landmarks,
                    detection.image_bgr.shape[:2],
                    intrinsics=_assumed_intrinsics(
                        detection.image_bgr.shape[:2], focal_scale
                    ),
                    role=role,
                    confidence=detection.mapped_in_bounds_fraction,
                )
                for detection, role in zip(detections, capture_roles, strict=True)
            )
            fitted = MultiViewIdentityFitter(adapter, rig).fit(observations)
            accepted = fitted.report.accepted_view_indices
            rejected = fitted.report.rejected_view_indices
            all_cameras = tuple(fitted.cameras)
            all_predictions = tuple(fitted.fitted_landmarks)
            all_nuisance = tuple(fitted.nuisance)
            fit_indices = tuple(range(len(paths)))
            held_out_indices = ()
            held_out_report = {
                "evaluated": False,
                "fit_leakage": None,
                "aggregate_nme": None,
                "maximum_view_nme": None,
                "aggregate_nme_gate": HELDOUT_AGGREGATE_NME_GATE,
                "per_view_nme_gate": HELDOUT_PER_VIEW_NME_GATE,
                "passed": False,
                "per_view": [],
            }
    except ValueError as exc:
        raise AutoAnimError("FIT_REJECTED", f"Multi-view fit failed: {exc}") from exc
    if not fitted.report.accepted:
        raise AutoAnimError(
            "FIT_REJECTED",
            "Multi-view identity fit did not pass its observability and residual gates.",
            _report_json(fitted),
        )

    if len(accepted) < 2:
        raise AutoAnimError("FIT_REJECTED", "Fewer than two mutually consistent views remain")
    accepted_yaws = tuple(all_cameras[index].yaw for index in accepted)
    yaw_span_degrees = _circular_yaw_span_degrees(accepted_yaws)
    minimum_yaw_span = (
        CALIBRATED_YAW_SPAN_GATE_DEGREES if calibrated_outcome is not None else 20.0
    )
    if yaw_span_degrees < minimum_yaw_span:
        raise AutoAnimError(
            "FIT_REJECTED",
            "Accepted photos do not provide enough viewpoint diversity for a multi-view fit.",
            {
                "yaw_span_degrees": yaw_span_degrees,
                "minimum_yaw_span_degrees": minimum_yaw_span,
                "accepted_view_indices": list(accepted),
            },
        )
    calibrated_geometry_gate_passed = False
    if calibrated_outcome is not None:
        per_fit_nme = [
            fitted.report.per_view[index].nme
            for index in fitted.report.accepted_view_indices
        ]
        observability = calibrated_outcome.observability
        failures: list[str] = []
        if fitted.report.nme > CALIBRATED_FIT_NME_GATE:
            failures.append("FIT_AGGREGATE_NME")
        if max(per_fit_nme, default=float("inf")) > CALIBRATED_PER_FIT_VIEW_NME_GATE:
            failures.append("FIT_PER_VIEW_NME")
        if observability["observable_rank"] < CALIBRATED_OBSERVABLE_RANK_GATE:
            failures.append("OBSERVABLE_RANK")
        if observability["observability_ratio"] < CALIBRATED_OBSERVABILITY_RATIO_GATE:
            failures.append("OBSERVABILITY_RATIO")
        if fitted.report.saturation_fraction > CALIBRATED_SATURATION_FRACTION_GATE:
            failures.append("COEFFICIENT_SATURATION")
        if not held_out_report["passed"]:
            failures.append("HELD_OUT_REPROJECTION")
        non_neutral_held_out = [
            {
                "view_index": index,
                "role": capture_roles[index],
                "strong_expression_score": detections[index].strong_expression_score,
            }
            for index in held_out_indices
            if detections[index].strong_expression_score > MAX_TEXTURE_BLENDSHAPE_SCORE
        ]
        if non_neutral_held_out:
            failures.append("HELD_OUT_NON_NEUTRAL")
        if failures:
            raise AutoAnimError(
                "FIT_REJECTED",
                "Calibrated reconstruction failed independent geometric validation.",
                {
                    "failures": failures,
                    "fit_nme": fitted.report.nme,
                    "maximum_fit_nme": CALIBRATED_FIT_NME_GATE,
                    "maximum_per_fit_view_nme": max(per_fit_nme, default=None),
                    "per_fit_view_nme_gate": CALIBRATED_PER_FIT_VIEW_NME_GATE,
                    "observability": observability,
                    "minimum_observable_rank": CALIBRATED_OBSERVABLE_RANK_GATE,
                    "minimum_observability_ratio": CALIBRATED_OBSERVABILITY_RATIO_GATE,
                    "coefficient_bound_fraction": fitted.report.saturation_fraction,
                    "maximum_coefficient_bound_fraction": CALIBRATED_SATURATION_FRACTION_GATE,
                    "held_out": held_out_report,
                    "non_neutral_held_out_views": non_neutral_held_out,
                },
            )
        calibrated_geometry_gate_passed = True
    if calibrated_outcome is not None:
        _validate_neutral_texture_captures(
            tuple(detections[index] for index in fit_indices),
            fitted,
            fitted.report.accepted_view_indices,
            tuple(capture_roles[index] for index in fit_indices),
        )
    else:
        _validate_neutral_texture_captures(detections, fitted, accepted, capture_roles)
    neutral_expression = np.zeros(adapter.expression_dim, dtype=np.float32)
    mesh = adapter.mesh(identity=fitted.identity, expression=neutral_expression)
    atlas = build_gnm_texture_atlas(adapter, texture_size)
    images_rgb = [
        cv2.cvtColor(detections[index].image_bgr, cv2.COLOR_BGR2RGB)
        for index in accepted
    ]
    masks = [
        _face_mask(detections[index].all_landmarks, detections[index].image_bgr.shape[:2])
        for index in accepted
    ]
    cameras = []
    for index in accepted:
        camera = all_cameras[index]
        if not isinstance(camera, FitPerspectiveCamera):
            raise AutoAnimError(
                "INTERNAL_ERROR", "Texture baking requires perspective camera estimates"
            )
        cameras.append(texture_camera_from_fit(camera))
    baked = bake_multiview_texture(
        mesh,
        adapter.triangles,
        atlas.triangle_uvs,
        images_rgb,
        cameras,
        texture_size=texture_size,
        masks=masks,
        confidences=[detections[index].mapped_in_bounds_fraction for index in accepted],
        generic_vertex_colors=atlas.generic_vertex_colors,
        mirror_fill=False,
        inpaint=True,
    )
    component_map, component_metrics = _component_texture_metrics(baked, atlas)
    _save_texture_artifacts(output, baked, atlas, component_map, accepted)

    adapter.export_obj(output / "fitted.obj", mesh)
    glb = export_gnm_glb(
        output / "fitted-textured.glb",
        adapter,
        mesh,
        texture_path=output / "texture.png",
        triangle_uvs=atlas.triangle_uvs,
        mapping_path=output / "fitted-glb-mapping.npz",
    )
    MeshRenderer(adapter).save_png(
        output / "mesh-preview.png",
        mesh,
        adapter.landmarks(identity=fitted.identity, expression=neutral_expression),
    )
    overlay_artifacts: dict[str, str] = {}
    for index, (detection, predicted) in enumerate(
        zip(detections, all_predictions, strict=True)
    ):
        name = f"overlay-{index + 1:02d}.png"
        if not cv2.imwrite(
            str(output / name),
            draw_overlay(detection.image_bgr, detection.landmarks, predicted),
        ):
            raise AutoAnimError("INTERNAL_ERROR", f"Could not write {name}")
        overlay_artifacts[f"overlay_{index + 1}"] = name

    camera_values = np.asarray(
        [
            (camera.yaw, camera.pitch, camera.roll, camera.tx, camera.ty, camera.tz)
            for camera in all_cameras
        ],
        dtype=np.float32,
    )
    intrinsics = np.asarray(
        [
            (observation.intrinsics.fx, observation.intrinsics.fy, observation.intrinsics.cx, observation.intrinsics.cy)
            for observation in observations
            if observation.intrinsics is not None
        ],
        dtype=np.float32,
    )
    fit_arrays: dict[str, np.ndarray] = {
        "identity": fitted.identity,
        "nuisance": (
            np.stack(fitted.nuisance)
            if camera_bundle is None
            else np.stack(all_nuisance).astype(np.float32)
        ),
        "intrinsics": intrinsics,
        "observed_landmarks": np.stack(
            [detection.landmarks for detection in detections]
        ).astype(np.float32),
        "fitted_landmarks": (
            np.stack(fitted.fitted_landmarks)
            if camera_bundle is None
            else np.stack(all_predictions).astype(np.float32)
        ),
        "accepted_view_indices": np.asarray(accepted, dtype=np.int32),
    }
    if camera_bundle is None:
        fit_arrays["cameras"] = camera_values
        fit_arrays["camera_convention"] = np.asarray(PERSPECTIVE_CAMERA_CONVENTION)
    else:
        assert calibrated_outcome is not None
        registration_matrix = calibrated_outcome.registration.as_matrix()
        fit_arrays.update(
            {
                "source_K": np.stack(
                    [view.intrinsics_matrix for view in camera_bundle.views]
                ).astype(np.float64),
                "source_D": np.stack(
                    [view.distortion for view in camera_bundle.views]
                ).astype(np.float64),
                "source_world_to_camera": np.stack(
                    [view.world_to_camera for view in camera_bundle.views]
                ).astype(np.float64),
                "gnm_to_world": registration_matrix,
                "effective_gnm_to_camera": np.stack(
                    [view.world_to_camera @ registration_matrix for view in camera_bundle.views]
                ).astype(np.float64),
                "view_usage": np.asarray(
                    [view.usage for view in camera_bundle.views], dtype="<U8"
                ),
                "source_camera_convention": np.asarray(camera_bundle.coordinate_convention),
                "meters_per_world_unit": np.asarray(camera_bundle.meters_per_world_unit),
                "rejected_view_indices": np.asarray(rejected, dtype=np.int32),
                "fit_view_indices": np.asarray(fit_indices, dtype=np.int32),
                "held_out_view_indices": np.asarray(held_out_indices, dtype=np.int32),
            }
        )
        write_json(output / "capture-calibration.json", camera_bundle.as_dict())
        write_json(
            output / "gnm-camera-registration.json",
            _registration_json(calibrated_outcome.registration, camera_bundle),
        )
    write_npz(output / "fit.npz", **fit_arrays)
    report = _report_json(fitted)
    if calibrated_outcome is not None and camera_bundle is not None:
        report.update({
            "fit_view_indices": list(fit_indices),
            "held_out_view_indices": list(held_out_indices),
            "accepted_global_view_indices": list(accepted),
            "rejected_global_view_indices": list(rejected),
            "held_out": held_out_report,
            "calibrated_observability": calibrated_outcome.observability,
            "camera_registration": _registration_json(
                calibrated_outcome.registration, camera_bundle
            ),
        })
    write_json(output / "fit-report.json", report)

    if camera_bundle is None:
        warnings = [
            METRIC_SCALE_CAVEAT,
            PSEUDO_INTRINSICS_CAVEAT,
            SPARSE_IDENTITY_CAVEAT,
            TEXTURE_PROVENANCE_CAVEAT,
            REAR_VIEW_CAVEAT,
            NEUTRAL_TEXTURE_CAVEAT,
        ]
    else:
        warnings = [
            METRIC_SCALE_CAVEAT,
            SPARSE_IDENTITY_CAVEAT,
            TEXTURE_PROVENANCE_CAVEAT,
            REAR_VIEW_CAVEAT,
            NEUTRAL_TEXTURE_CAVEAT,
            CALIBRATED_CAPTURE_CAVEAT,
        ]
    for index in rejected:
        warnings.append(f"REJECTED_VIEW_{index + 1}:{capture_roles[index]}")
    if float(baked.metrics["observed_fraction"]) < 0.50:
        warnings.append("LOW_DIRECT_TEXTURE_COVERAGE")
    for component_name, metrics in component_metrics.items():
        if int(metrics["observed_texels"]) == 0:
            warnings.append(f"NO_DIRECT_TEXTURE:{component_name}")
    if any(detection.strong_expression_score > 0.70 for detection in detections):
        warnings.append("STRONG_EXPRESSION_IN_CAPTURE")

    result = {
        "kind": "multiview_reconstruction",
        "status": "succeeded",
        "model": {
            "gnm_version": "3.0",
            "identity_dim": adapter.identity_dim,
            "landmark_observable_identity_dim": 170,
        },
        "capture": {
            "view_count": len(paths),
            "roles": list(capture_roles),
            "accepted_view_indices": list(accepted),
            "rejected_view_indices": list(rejected),
            "mapping": MAPPING_NAME,
            "accepted_yaw_span_degrees": yaw_span_degrees,
            **(
                {
                    "intrinsics_source": "dimension_assumption",
                    "focal_scale": focal_scale,
                    "camera_convention": PERSPECTIVE_CAMERA_CONVENTION,
                }
                if camera_bundle is None
                else {
                    "intrinsics_source": "measured_calibration",
                    "calibration_source": "uploaded_camera_bundle",
                    "calibration_schema_version": camera_bundle.schema_version,
                    "calibration_sha256": camera_bundle.source_sha256,
                    "distortion_handling": "opencv_undistort_same_K",
                    "fit_view_indices": list(fit_indices),
                    "held_out_view_indices": list(held_out_indices),
                    "held_out": held_out_report,
                    "camera_convention": camera_bundle.coordinate_convention,
                    "meters_per_world_unit": camera_bundle.meters_per_world_unit,
                    "meters_per_gnm_model_unit": (
                        calibrated_outcome.registration.scale
                        * camera_bundle.meters_per_world_unit
                    ),
                }
            ),
        },
        "fit": {
            "nme": fitted.report.nme,
            "mean_pixel_error": fitted.report.mean_pixel_error,
            "observable_rank": (
                calibrated_outcome.observability["observable_rank"]
                if calibrated_outcome is not None
                else fitted.report.observable_rank
            ),
            "active_identity_modes": fitted.report.active_identity_modes,
            "observability_ratio": (
                calibrated_outcome.observability["observability_ratio"]
                if calibrated_outcome is not None
                else fitted.report.observability_ratio
            ),
            "coefficient_bound_fraction": fitted.report.saturation_fraction,
            "identity_solver": fitted.report.identity_solver,
            "identity_coefficient_bound": fitted.report.identity_coefficient_bound,
            "nuisance_coefficient_bound": fitted.report.nuisance_coefficient_bound,
            "glb_vertices": glb.vertex_count,
            "glb_seam_duplicates": glb.seam_duplicates,
            "production_validated": False,
            **(
                {
                    "calibrated_geometry_gate_passed": calibrated_geometry_gate_passed,
                    "validation_scope": (
                        "synthetic_and_geometric_only; real calibrated performer fixture pending"
                    ),
                }
                if camera_bundle is not None
                else {}
            ),
        },
        "texture": {
            **dict(baked.metrics),
            "atlas_layout": atlas.layout_id,
            "component_names": list(atlas.component_names),
            "component_bounds": {
                name: list(bounds) for name, bounds in atlas.component_bounds.items()
            },
            "components": component_metrics,
            "padding_texels": atlas.padding_texels,
            "uv_origin_internal": "lower_left",
            "uv_origin_glb": "top_left",
            "source_view_index_space": "global_capture_index",
            "texture_view_local_to_global": list(accepted),
        },
        "viewer": {
            "schema_version": "1.0",
            "status": "ready",
            "mode": "static_textured",
            "model_artifact": "textured_glb",
            "clock_artifact": None,
            "coordinate_system": "+Y_up_+Z_forward_meters",
        },
        "artifacts": {
            "textured_glb": "fitted-textured.glb",
            "glb_mapping": "fitted-glb-mapping.npz",
            "mesh": "fitted.obj",
            "mesh_preview": "mesh-preview.png",
            "texture": "texture.png",
            "texture_confidence": "texture-confidence.png",
            "texture_provenance": "texture-provenance.png",
            "texture_maps": "texture-maps.npz",
            "parameters": "fit.npz",
            "fit_report": "fit-report.json",
            **(
                {
                    "capture_calibration": "capture-calibration.json",
                    "camera_registration": "gnm-camera-registration.json",
                }
                if camera_bundle is not None
                else {}
            ),
            **overlay_artifacts,
        },
        "warnings": warnings,
    }
    write_json(output / "result.json", result)
    return result


__all__ = ["run_multiview_pipeline", "texture_camera_from_fit"]
