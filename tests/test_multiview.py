from __future__ import annotations

import hashlib

import numpy as np
import pytest

from autoanim_gnm.fitting import IdentityFitter
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.multiview import (
    CameraIntrinsics,
    IDENTITY_COEFFICIENT_BOUND,
    IDENTITY_SOLVER,
    METRIC_SCALE_CAVEAT,
    MultiViewIdentityFitter,
    MultiViewObservation,
    NUISANCE_COEFFICIENT_BOUND,
    PerspectiveCamera,
    WeakPerspectiveCamera,
    _bounded_observable_least_squares,
    project_points,
)
from autoanim_gnm.rig import ControlRig


CALIBRATED_SYNTHETIC_BASELINE_SHA256 = {
    "identity": "8415a7ebb4a1856b81fb754f54101f2fcc545dcd012996ac49a3c301d07a5aff",
    "identity_consistency": "8e418fb12345f1b488ebb839ab33f8931cae6d3b85aa82327d0cf7114b21c1c2",
    "leave_one_out_nme": "dac6490bb1f63a832349179e52e10ae287196373cb3dcbae0492a9a36abcb33b",
    "fitted_landmarks": "e00e66728250364246d4ce06f21747b97c4d21dc70f6bf7b72055712a43ffc7c",
    "nuisance": "c0bb4343a82ba862151e7e7b4e6ba26c3e11d0624360601c90c916adaae64529",
}


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def multiview_fitter(adapter: GNMAdapter, rig: ControlRig) -> MultiViewIdentityFitter:
    return MultiViewIdentityFitter(adapter, rig, max_outer_iterations=2)


def _identity_and_shape(
    adapter: GNMAdapter,
    *,
    seed: int = 7,
    modes: int = 60,
    sigma: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    identity[:modes] = np.random.default_rng(seed).normal(0.0, sigma, modes)
    shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", identity, adapter.compact_identity_basis
    )
    return identity, shape


def _perspective_views(shape: np.ndarray) -> list[MultiViewObservation]:
    intrinsics = CameraIntrinsics(1000.0, 980.0, 320.0, 300.0)
    cameras = (
        PerspectiveCamera(0.0, -0.03, 0.01, 0.0, 0.0, 0.72, intrinsics),
        PerspectiveCamera(0.67, 0.02, -0.015, 0.01, 0.0, 0.74, intrinsics),
        PerspectiveCamera(1.24, -0.01, 0.005, -0.01, 0.005, 0.77, intrinsics),
    )
    roles = ("front", "left_3q", "left_profile")
    return [
        MultiViewObservation(project_points(shape, camera), (640, 640), intrinsics, role)
        for camera, role in zip(cameras, roles, strict=True)
    ]


def _shape_error_mm(adapter: GNMAdapter, identity: np.ndarray, truth: np.ndarray) -> float:
    fitted = adapter.compact_template + np.einsum(
        "i,ilc->lc", identity, adapter.compact_identity_basis
    )
    # Sparse monocular observations cannot fix absolute scale.  Compare the
    # recovered intrinsic shape after the same similarity gauge used by 3DMM
    # benchmarks; the result report separately preserves the metric caveat.
    centered_fitted = fitted - np.mean(fitted, axis=0)
    centered_truth = truth - np.mean(truth, axis=0)
    left, singular, right = np.linalg.svd(centered_fitted.T @ centered_truth)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    scale = float(np.sum(singular) / np.sum(centered_fitted * centered_fitted))
    aligned = scale * centered_fitted @ rotation + np.mean(truth, axis=0)
    return float(np.sqrt(np.mean(np.sum((aligned - truth) ** 2, axis=1))) * 1000.0)


def test_near_exact_perspective_multiview_recovery_and_report(
    adapter: GNMAdapter, multiview_fitter: MultiViewIdentityFitter
) -> None:
    _, shape = _identity_and_shape(adapter, seed=4, modes=60)
    result = multiview_fitter.fit(_perspective_views(shape))

    assert result.report.accepted
    assert result.report.nme < 0.0015
    assert _shape_error_mm(adapter, result.identity, shape) < 1.0
    assert result.report.unlocked_stages == (20, 40, 80, 120, 170)
    assert result.report.observable_rank >= 155
    assert result.report.observability_ratio > 0.90
    assert result.report.identity_solver == IDENTITY_SOLVER
    assert result.report.identity_coefficient_bound == IDENTITY_COEFFICIENT_BOUND
    assert result.report.nuisance_coefficient_bound == NUISANCE_COEFFICIENT_BOUND
    assert np.isfinite(result.report.condition_number)
    assert result.report.saturation_fraction == 0.0
    assert result.report.rejected_view_indices == ()
    assert tuple(view.camera_kind for view in result.report.per_view) == (
        "perspective",
        "perspective",
        "perspective",
    )
    assert result.report.metric_scale_caveat == METRIC_SCALE_CAVEAT


def test_bounded_solver_stays_in_observable_subspace_at_coefficient_limit() -> None:
    observable = np.asarray(((2.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)),))
    design = np.asarray(((1.0, 0.0), (0.0, 1.0)))
    target = np.asarray((10.0, 2.0))

    solution = _bounded_observable_least_squares(
        design, target, observable, nuisance_columns=1
    )
    identity = observable.T @ solution[:1]

    assert identity[0] == pytest.approx(IDENTITY_COEFFICIENT_BOUND, abs=2.0e-7)
    assert identity[1] == pytest.approx(IDENTITY_COEFFICIENT_BOUND / 2.0, abs=2.0e-7)
    assert solution[1] == pytest.approx(NUISANCE_COEFFICIENT_BOUND, abs=2.0e-7)
    assert float(np.dot(np.asarray((-1.0, 2.0)), identity)) == pytest.approx(
        0.0, abs=2.0e-7
    )
    # The previous solve-then-clip behavior produced [3, 3] here, which has a
    # non-zero component in the unsupported [-1, 2] null direction.
    assert not np.allclose(identity, (3.0, 3.0))


def test_multiview_recovers_more_shape_than_single_view_twenty_modes(
    adapter: GNMAdapter,
    rig: ControlRig,
    multiview_fitter: MultiViewIdentityFitter,
) -> None:
    _, shape = _identity_and_shape(adapter, seed=1, modes=60, sigma=0.4)
    views = _perspective_views(shape)
    multi = multiview_fitter.fit(views)
    single = IdentityFitter(adapter, rig).fit(
        views[0].landmarks,
        views[0].image_size,
        modes=20,
        compute_stability=False,
    )

    multi_error = _shape_error_mm(adapter, multi.identity, shape)
    single_error = _shape_error_mm(adapter, single.identity, shape)
    assert multi_error < 1.0
    assert multi_error < 0.5 * single_error
    assert np.linalg.norm(multi.identity[20:60]) > 0.1


def test_duplicate_front_views_do_not_overstate_camera_marginalized_rank(
    adapter: GNMAdapter,
) -> None:
    _, shape = _identity_and_shape(adapter, seed=8, modes=60, sigma=0.3)
    intrinsics = CameraIntrinsics(1000.0, 980.0, 320.0, 300.0)
    camera = PerspectiveCamera(0.0, 0.0, 0.0, 0.0, 0.0, 0.72, intrinsics)
    observation = MultiViewObservation(
        project_points(shape, camera), (640, 640), intrinsics, "front"
    )

    result = MultiViewIdentityFitter(adapter, max_outer_iterations=1).fit(
        (observation, observation)
    )

    # One sparse-68 view supplies at most 136 scalar equations.  Six camera
    # and four expression-nuisance directions must be marginalized rather than
    # counted as identity evidence.  Duplicating the same view adds no new
    # independent identity directions.
    assert result.report.observable_rank <= 126
    assert result.report.observability_ratio <= 126 / 170


def test_noisy_occluded_views_are_robust(
    adapter: GNMAdapter, multiview_fitter: MultiViewIdentityFitter
) -> None:
    _, shape = _identity_and_shape(adapter, seed=12, modes=80)
    rng = np.random.default_rng(91)
    observations = _perspective_views(shape)
    corrupted: list[MultiViewObservation] = []
    for observation in observations:
        landmarks = observation.landmarks + rng.normal(0.0, 0.65, (68, 2))
        visibility = np.ones(68, dtype=np.float64)
        visibility[rng.choice(68, 12, replace=False)] = 0.0
        available = np.flatnonzero(visibility)
        bad = rng.choice(available, 4, replace=False)
        landmarks[bad] += rng.normal(0.0, 18.0, (4, 2))
        confidence = np.ones(68, dtype=np.float64)
        confidence[bad] = 0.5
        corrupted.append(
            MultiViewObservation(
                landmarks,
                observation.image_size,
                observation.intrinsics,
                observation.role,
                confidence,
                visibility,
            )
        )

    result = multiview_fitter.fit(corrupted)
    assert result.report.accepted
    assert result.report.nme < 0.035
    assert _shape_error_mm(adapter, result.identity, shape) < 2.0
    assert result.report.saturation_fraction < 0.05
    assert all(view.visible_landmarks == 56 for view in result.report.per_view)
    assert np.max(result.report.leave_one_out_nme) < 0.025


def test_invisible_landmark_coordinates_cannot_change_fit(
    adapter: GNMAdapter, rig: ControlRig
) -> None:
    _, shape = _identity_and_shape(adapter, seed=5, modes=40, sigma=0.3)
    original = _perspective_views(shape)
    visibility = np.ones(68, dtype=np.float64)
    visibility[[36, 45]] = 0.0
    masked: list[MultiViewObservation] = []
    corrupted: list[MultiViewObservation] = []
    for observation in original:
        masked.append(
            MultiViewObservation(
                observation.landmarks.copy(),
                observation.image_size,
                observation.intrinsics,
                observation.role,
                visibility=visibility,
            )
        )
        points = observation.landmarks.copy()
        points[36] = (1.0e6, -1.0e6)
        points[45] = (-2.0e6, 2.0e6)
        corrupted.append(
            MultiViewObservation(
                points,
                observation.image_size,
                observation.intrinsics,
                observation.role,
                visibility=visibility,
            )
        )

    fitter = MultiViewIdentityFitter(
        adapter, rig, stages=(20, 40), max_outer_iterations=1
    )
    expected = fitter.fit(masked)
    actual = fitter.fit(corrupted)

    np.testing.assert_array_equal(actual.identity, expected.identity)
    assert actual.cameras == expected.cameras
    assert actual.report.nme == expected.report.nme
    np.testing.assert_array_equal(
        actual.report.leave_one_out_nme, expected.report.leave_one_out_nme
    )


def test_mixed_identity_view_is_rejected_and_inliers_are_refit(
    adapter: GNMAdapter, multiview_fitter: MultiViewIdentityFitter
) -> None:
    inlier_identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    inlier_identity[:20] = np.linspace(-0.8, 0.8, 20)
    outlier_identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    outlier_identity[:20] = -3.0 * inlier_identity[:20]
    inlier_shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", inlier_identity, adapter.compact_identity_basis
    )
    outlier_shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", outlier_identity, adapter.compact_identity_basis
    )
    views = _perspective_views(inlier_shape)
    third = views[2]
    outlier_camera = PerspectiveCamera(
        1.24,
        -0.01,
        0.005,
        -0.01,
        0.005,
        0.77,
        third.intrinsics,
    )
    views[2] = MultiViewObservation(
        project_points(outlier_shape, outlier_camera),
        third.image_size,
        third.intrinsics,
        third.role,
    )

    result = multiview_fitter.fit(views)
    assert result.report.accepted
    assert result.report.accepted_view_indices == (0, 1)
    assert result.report.rejected_view_indices == (2,)
    assert not result.report.per_view[2].accepted
    assert result.report.per_view[2].rejection_reason == "MIXED_IDENTITY_OR_OUTLIER_VIEW"
    assert result.report.leave_one_out_nme[2] > 0.035
    assert result.report.nme < 0.002
    assert _shape_error_mm(adapter, result.identity, inlier_shape) < 1.0


def test_mixed_camera_models_and_unobservable_modes_stay_neutral(
    adapter: GNMAdapter, multiview_fitter: MultiViewIdentityFitter
) -> None:
    _, shape = _identity_and_shape(adapter, seed=22, modes=40)
    intrinsics = CameraIntrinsics(900.0, 900.0, 320.0, 320.0)
    cameras = (
        PerspectiveCamera(0.0, 0.0, 0.0, 0.0, 0.0, 0.68, intrinsics),
        WeakPerspectiveCamera(0.65, 0.01, 0.0, 1250.0, 320.0, 320.0),
        WeakPerspectiveCamera(-0.65, -0.01, 0.0, 1200.0, 320.0, 320.0),
    )
    observations = [
        MultiViewObservation(
            project_points(shape, camera),
            (640, 640),
            intrinsics if isinstance(camera, PerspectiveCamera) else None,
            role,
        )
        for camera, role in zip(cameras, ("front", "left_3q", "right_3q"), strict=True)
    ]
    result = multiview_fitter.fit(observations)

    assert result.report.accepted
    assert tuple(camera.kind for camera in result.cameras) == (
        "perspective",
        "weak_perspective",
        "weak_perspective",
    )
    np.testing.assert_array_equal(result.identity[170:], np.zeros(83, dtype=np.float32))
    assert result.identity.shape == (253,)
    assert result.report.active_identity_modes == 170
    assert result.report.observable_rank >= 155


def test_small_per_view_nuisance_expressions_do_not_fragment_identity(
    adapter: GNMAdapter, multiview_fitter: MultiViewIdentityFitter
) -> None:
    _, neutral_shape = _identity_and_shape(adapter, seed=44, modes=40, sigma=0.3)
    intrinsics = CameraIntrinsics(1000.0, 980.0, 320.0, 300.0)
    cameras = (
        PerspectiveCamera(0.0, 0.0, 0.0, 0.0, 0.0, 0.72, intrinsics),
        PerspectiveCamera(0.67, 0.0, 0.0, 0.01, 0.0, 0.74, intrinsics),
        PerspectiveCamera(-0.67, 0.0, 0.0, -0.01, 0.0, 0.74, intrinsics),
    )
    nuisance = (
        np.asarray((0.18, 0.0, 0.0, 0.0)),
        np.asarray((0.0, -0.15, 0.08, 0.0)),
        np.asarray((0.0, 0.0, 0.0, 0.20)),
    )
    observations = []
    for camera, coefficients, role in zip(
        cameras, nuisance, ("front", "left_3q", "right_3q"), strict=True
    ):
        expressed = neutral_shape + np.einsum(
            "i,ilc->lc", coefficients, multiview_fitter.nuisance_basis
        )
        observations.append(
            MultiViewObservation(
                project_points(expressed, camera), (640, 640), intrinsics, role
            )
        )

    result = multiview_fitter.fit(observations)
    assert result.report.accepted
    assert result.report.rejected_view_indices == ()
    assert result.report.nme < 0.0015
    assert max(float(np.max(np.abs(value))) for value in result.nuisance) <= 0.35
    assert all(np.linalg.norm(value) > 0.04 for value in result.nuisance)


def test_deterministic_output(
    adapter: GNMAdapter, multiview_fitter: MultiViewIdentityFitter
) -> None:
    _, shape = _identity_and_shape(adapter, seed=31, modes=50)
    observations = _perspective_views(shape)
    first = multiview_fitter.fit(observations)
    second = multiview_fitter.fit(observations)

    np.testing.assert_array_equal(first.identity, second.identity)
    np.testing.assert_array_equal(
        first.report.identity_consistency_matrix,
        second.report.identity_consistency_matrix,
    )
    np.testing.assert_array_equal(first.report.leave_one_out_nme, second.report.leave_one_out_nme)
    for first_points, second_points in zip(
        first.fitted_landmarks, second.fitted_landmarks, strict=True
    ):
        np.testing.assert_array_equal(first_points, second_points)
    assert _array_sha256(first.identity) == CALIBRATED_SYNTHETIC_BASELINE_SHA256[
        "identity"
    ]
    assert _array_sha256(first.report.identity_consistency_matrix) == (
        CALIBRATED_SYNTHETIC_BASELINE_SHA256["identity_consistency"]
    )
    assert _array_sha256(first.report.leave_one_out_nme) == (
        CALIBRATED_SYNTHETIC_BASELINE_SHA256["leave_one_out_nme"]
    )
    assert _array_sha256(np.stack(first.fitted_landmarks)) == (
        CALIBRATED_SYNTHETIC_BASELINE_SHA256["fitted_landmarks"]
    )
    assert _array_sha256(np.stack(first.nuisance)) == (
        CALIBRATED_SYNTHETIC_BASELINE_SHA256["nuisance"]
    )


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda points: points[:67], r"finite \[68,2\]"),
        (lambda points: np.where(np.indices(points.shape) == (0, 0), np.nan, points), "finite"),
    ],
)
def test_invalid_landmark_arrays_are_rejected(
    multiview_fitter: MultiViewIdentityFitter,
    mutation,
    match: str,
) -> None:
    points = np.zeros((68, 2), dtype=np.float64)
    bad = MultiViewObservation(mutation(points), (640, 640))
    with pytest.raises(ValueError, match=match):
        multiview_fitter.fit((bad, bad))


def test_invalid_metadata_and_insufficient_views_are_rejected(
    multiview_fitter: MultiViewIdentityFitter,
) -> None:
    points = np.column_stack((np.linspace(100, 500, 68), np.linspace(120, 520, 68)))
    valid = MultiViewObservation(points, (640, 640))
    with pytest.raises(ValueError, match="at least two views"):
        multiview_fitter.fit((valid,))
    with pytest.raises(ValueError, match="positive integers"):
        multiview_fitter.fit((MultiViewObservation(points, (0, 640)), valid))
    with pytest.raises(ValueError, match=r"confidence must lie in \[0,1\]"):
        multiview_fitter.fit((MultiViewObservation(points, (640, 640), confidence=1.1), valid))
    with pytest.raises(ValueError, match="at least 24"):
        multiview_fitter.fit(
            (MultiViewObservation(points, (640, 640), visibility=np.zeros(68)), valid)
        )
    with pytest.raises(ValueError, match="intrinsics must be"):
        multiview_fitter.fit(
            (MultiViewObservation(points, (640, 640), intrinsics=object()), valid)  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive fx and fy"):
        CameraIntrinsics(0.0, 900.0, 320.0, 320.0)


def test_projection_rejects_points_behind_perspective_camera() -> None:
    intrinsics = CameraIntrinsics(900.0, 900.0, 320.0, 320.0)
    camera = PerspectiveCamera(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, intrinsics)
    with pytest.raises(ValueError, match="behind the camera"):
        project_points(np.zeros((3, 3)), camera)
