from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import cv2
from scipy.spatial.transform import Rotation
from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.camera_bundle import (
    CAMERA_BUNDLE_CONVENTION,
    CAMERA_BUNDLE_MODEL,
    CAMERA_BUNDLE_SCHEMA,
    CalibratedCameraBundle,
    CalibratedCameraView,
    CameraRegistration,
    estimate_camera_registration,
    load_camera_bundle,
    perspective_camera_from_calibration,
    project_calibrated_points,
)
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.gnm_adapter import GNMAdapter
from autoanim_gnm.image import DetectedFace
from autoanim_gnm.multiview import (
    CameraIntrinsics,
    MultiViewIdentityFitter,
    MultiViewObservation,
    PerspectiveCamera,
    project_points,
)
from autoanim_gnm.multiview_pipeline import (
    _circular_yaw_span_degrees,
    _fit_with_calibrated_bundle,
    _undistort_detection,
    run_multiview_pipeline,
    texture_camera_from_fit,
)
import autoanim_gnm.multiview_pipeline as multiview_pipeline_module
from autoanim_gnm.rig import ControlRig
from autoanim_gnm.service import ApplicationService
import autoanim_gnm.service as service_module


def _fit_camera(yaw: float, *, width: int = 640, height: int = 640) -> PerspectiveCamera:
    intrinsics = CameraIntrinsics(820.0, 810.0, 0.5 * (width - 1), 0.5 * (height - 1))
    return PerspectiveCamera(yaw, 0.015, -0.01, 0.0, 0.0, 0.78, intrinsics)


def _view(index: int, yaw: float, usage: str) -> CalibratedCameraView:
    camera = _fit_camera(yaw)
    texture_camera = texture_camera_from_fit(camera)
    return CalibratedCameraView(
        index=index,
        image_name=f"view-{index}.png",
        role=f"view_{index}",
        usage=usage,
        image_size=(640, 640),
        intrinsics_matrix=texture_camera.intrinsics,
        distortion=np.zeros(5, dtype=np.float64),
        world_to_camera=texture_camera.world_to_camera,
        visibility=np.ones(68, dtype=np.float64),
    )


def _bundle(views: tuple[CalibratedCameraView, ...]) -> CalibratedCameraBundle:
    return CalibratedCameraBundle(
        calibration_rms_px=0.18,
        pose_error_degrees=0.4,
        scale_error_fraction=0.003,
        views=views,
        source_sha256="1" * 64,
        meters_per_world_unit=1.0,
    )


def _payload() -> dict:
    views = tuple(
        _view(index, yaw, "fit" if index < 3 else "held_out")
        for index, yaw in enumerate((-0.8, 0.0, 0.8, 0.45))
    )
    bundle = _bundle(views).as_dict()
    bundle.pop("source_sha256")
    bundle.pop("fit_view_indices")
    bundle.pop("held_out_view_indices")
    bundle.pop("declared_calibration_metadata_gate_passed")
    return bundle


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_camera_bundle_parser_is_strict_and_ordered(tmp_path: Path) -> None:
    source = tmp_path / "calibration.json"
    payload = _payload()
    _write_payload(source, payload)

    bundle = load_camera_bundle(
        source,
        input_names=tuple(f"view-{index}.png" for index in range(4)),
        image_sizes=((640, 640),) * 4,
    )

    assert bundle.schema_version == CAMERA_BUNDLE_SCHEMA
    assert bundle.camera_model == CAMERA_BUNDLE_MODEL
    assert bundle.coordinate_convention == CAMERA_BUNDLE_CONVENTION
    assert bundle.fit_indices == (0, 1, 2)
    assert bundle.held_out_indices == (3,)
    assert bundle.declared_calibration_metadata_gate_passed
    assert bundle.views[0].distortion.shape == (5,)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda value: value["views"][1].__setitem__("index", 0), "ordered and contiguous"),
        (lambda value: value["views"][0].__setitem__("D", [0.0] * 4), "distortion"),
        (lambda value: value["views"][0]["K"][2].__setitem__(2, 2.0), "intrinsics"),
        (
            lambda value: value["views"][0]["world_to_camera"][0].__setitem__(0, -1.0),
            "not rigid",
        ),
        (lambda value: value["views"][0].__setitem__("image_size", [480, 640]), "image_size"),
        (
            lambda value: value["views"][0]["K"][0].__setitem__(0, 10**1000),
            "shape",
        ),
    ),
)
def test_camera_bundle_parser_rejects_malformed_calibration(
    tmp_path: Path, mutator, message: str
) -> None:
    source = tmp_path / "calibration.json"
    payload = deepcopy(_payload())
    mutator(payload)
    _write_payload(source, payload)
    with pytest.raises(AutoAnimError, match=message):
        load_camera_bundle(
            source,
            input_names=tuple(f"view-{index}.png" for index in range(4)),
            image_sizes=((640, 640),) * 4,
        )


def test_camera_bundle_parser_rejects_nonfinite_json(tmp_path: Path) -> None:
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps(_payload()).replace("0.18", "NaN", 1), encoding="utf-8")
    with pytest.raises(AutoAnimError, match="non-finite JSON"):
        load_camera_bundle(
            source,
            input_names=tuple(f"view-{index}.png" for index in range(4)),
            image_sizes=((640, 640),) * 4,
        )


def test_circular_yaw_span_cannot_pass_across_euler_branch_cut() -> None:
    assert _circular_yaw_span_degrees(np.radians((179.0, -179.0))) == pytest.approx(2.0)
    assert _circular_yaw_span_degrees(np.radians((-62.0, 0.0, 62.0))) == pytest.approx(
        124.0
    )


def test_nonzero_distortion_undistorts_image_and_both_landmark_sets() -> None:
    intrinsics = np.asarray(
        ((700.0, 0.0, 319.5), (0.0, 690.0, 319.5), (0.0, 0.0, 1.0))
    )
    distortion = np.asarray((0.16, -0.08, 0.002, -0.001, 0.015))
    view = CalibratedCameraView(
        index=0,
        image_name="distorted.png",
        role="front",
        usage="fit",
        image_size=(640, 640),
        intrinsics_matrix=intrinsics,
        distortion=distortion,
        world_to_camera=np.eye(4),
        visibility=np.ones(68),
    )
    normalized_68 = np.column_stack(
        (
            np.linspace(-0.22, 0.22, 68),
            0.16 * np.sin(np.linspace(0.0, 3.0 * np.pi, 68)),
            np.ones(68),
        )
    )
    normalized_478 = np.tile(normalized_68, (8, 1))[:478]

    def distort(points: np.ndarray) -> np.ndarray:
        projected, _ = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            intrinsics,
            distortion,
        )
        return projected[:, 0].astype(np.float32)

    distorted_68 = distort(normalized_68)
    distorted_478 = distort(normalized_478)
    expected_68 = np.column_stack(
        (
            intrinsics[0, 0] * normalized_68[:, 0] + intrinsics[0, 2],
            intrinsics[1, 1] * normalized_68[:, 1] + intrinsics[1, 2],
        )
    )
    expected_478 = np.column_stack(
        (
            intrinsics[0, 0] * normalized_478[:, 0] + intrinsics[0, 2],
            intrinsics[1, 1] * normalized_478[:, 1] + intrinsics[1, 2],
        )
    )
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    for point in distorted_68:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (255, 255, 255), -1)
    detection = DetectedFace(
        image_bgr=image,
        landmarks=distorted_68,
        all_landmarks=distorted_478,
        blendshapes={},
        face_width=float(np.ptp(distorted_68[:, 0])),
        mapped_in_bounds_fraction=1.0,
        strong_expression_score=0.0,
    )

    corrected = _undistort_detection(detection, view)

    np.testing.assert_allclose(corrected.landmarks, expected_68, atol=2e-3)
    np.testing.assert_allclose(corrected.all_landmarks, expected_478, atol=2e-3)
    assert corrected.image_bgr.shape == image.shape
    assert not np.array_equal(corrected.image_bgr, image)
    assert int(np.count_nonzero(corrected.image_bgr)) > 0


def test_shared_similarity_registration_and_effective_camera_agree(
    adapter: GNMAdapter,
) -> None:
    identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    identity[:40] = np.random.default_rng(14).normal(0.0, 0.25, 40)
    shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", identity, adapter.compact_identity_basis, optimize=True
    )
    views = tuple(_view(index, yaw, "fit") for index, yaw in enumerate((-0.8, 0.0, 0.8)))
    truth = CameraRegistration(
        1.08,
        Rotation.from_euler("xyz", (0.03, -0.04, 0.02)).as_matrix(),
        np.asarray((0.015, -0.01, 0.03)),
        0.0,
        0.0,
    )
    observations = tuple(project_calibrated_points(shape, view, truth) for view in views)

    solved = estimate_camera_registration((shape,) * 3, observations, views)

    assert solved.mean_reprojection_error_px < 1e-5
    assert solved.p95_reprojection_error_px < 2e-5
    for view, observed in zip(views, observations, strict=True):
        np.testing.assert_allclose(
            project_calibrated_points(shape, view, solved), observed, atol=3e-5
        )
        effective = perspective_camera_from_calibration(view, solved)
        np.testing.assert_allclose(project_points(shape, effective), observed, atol=3e-5)

    millimetre_views = tuple(
        CalibratedCameraView(
            index=view.index,
            image_name=view.image_name,
            role=view.role,
            usage=view.usage,
            image_size=view.image_size,
            intrinsics_matrix=view.intrinsics_matrix,
            distortion=view.distortion,
            world_to_camera=np.block(
                [
                    [view.world_to_camera[:3, :3], 1000.0 * view.world_to_camera[:3, 3:4]],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            ),
            visibility=view.visibility,
        )
        for view in views
    )
    truth_mm = CameraRegistration(
        1000.0 * truth.scale,
        truth.rotation,
        1000.0 * truth.translation,
        0.0,
        0.0,
    )
    observations_mm = tuple(
        project_calibrated_points(shape, view, truth_mm) for view in millimetre_views
    )
    solved_mm = estimate_camera_registration(
        (shape,) * 3,
        observations_mm,
        millimetre_views,
        meters_per_world_unit=0.001,
    )
    assert solved_mm.scale * 0.001 == pytest.approx(solved.scale, rel=2e-6)
    np.testing.assert_allclose(
        solved_mm.translation * 0.001, solved.translation, atol=3e-6
    )


def test_locked_calibrated_cameras_are_not_refined(
    adapter: GNMAdapter, rig: ControlRig
) -> None:
    identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    identity[:20] = np.linspace(-0.3, 0.3, 20)
    shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", identity, adapter.compact_identity_basis, optimize=True
    )
    cameras = tuple(_fit_camera(yaw) for yaw in (-0.7, 0.0, 0.7))
    observations = tuple(
        MultiViewObservation(
            project_points(shape, camera),
            (640, 640),
            camera.intrinsics,
            f"view_{index}",
            initial_camera=camera,
            lock_camera=True,
        )
        for index, camera in enumerate(cameras)
    )

    result = MultiViewIdentityFitter(adapter, rig, max_outer_iterations=1).fit(observations)

    assert result.report.accepted
    assert result.cameras == cameras
    assert result.report.nme < 0.002


def _detection(points: np.ndarray) -> DetectedFace:
    tiled = np.resize(np.asarray(points, dtype=np.float32), (478, 2))
    return DetectedFace(
        image_bgr=np.zeros((640, 640, 3), dtype=np.uint8),
        landmarks=np.asarray(points, dtype=np.float32),
        all_landmarks=tiled,
        blendshapes={},
        face_width=float(np.ptp(points[:, 0])),
        mapped_in_bounds_fraction=1.0,
        strong_expression_score=0.0,
    )


def test_held_out_views_cannot_leak_into_identity_or_registration(
    adapter: GNMAdapter, rig: ControlRig
) -> None:
    identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    identity[:35] = np.random.default_rng(3).normal(0.0, 0.18, 35)
    shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", identity, adapter.compact_identity_basis, optimize=True
    )
    yaws = (-1.05, -0.52, 0.0, 0.52, 1.05, -0.78, 0.78)
    views = tuple(
        _view(index, yaw, "fit" if index < 5 else "held_out")
        for index, yaw in enumerate(yaws)
    )
    bundle = _bundle(views)
    registration = CameraRegistration(
        1.0, np.eye(3), np.zeros(3), 0.0, 0.0
    )
    detections = tuple(
        _detection(project_calibrated_points(shape, view, registration)) for view in views
    )

    clean = _fit_with_calibrated_bundle(adapter, rig, detections, bundle)
    corrupted_detections = list(detections)
    for index in bundle.held_out_indices:
        corrupted_detections[index] = _detection(
            detections[index].landmarks + np.asarray((35.0, -22.0))
        )
    corrupted = _fit_with_calibrated_bundle(
        adapter, rig, tuple(corrupted_detections), bundle
    )

    np.testing.assert_array_equal(clean.fitted.identity, corrupted.fitted.identity)
    np.testing.assert_array_equal(
        clean.registration.as_matrix(), corrupted.registration.as_matrix()
    )
    assert clean.accepted_indices == corrupted.accepted_indices
    assert clean.rejected_indices == corrupted.rejected_indices
    assert clean.held_out_report["passed"]
    assert not corrupted.held_out_report["passed"]
    assert clean.held_out_report["fit_leakage"] is False
    assert set(clean.accepted_indices).isdisjoint(bundle.held_out_indices)
    assert clean.observability["observable_rank"] >= 145
    assert clean.observability["observability_ratio"] >= 0.85

    bad_fit_index = 2
    bad_fit_detections = list(detections)
    conflicting_identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    conflicting_identity[:35] = -3.0 * identity[:35]
    conflicting_shape = adapter.compact_template + np.einsum(
        "i,ilc->lc",
        conflicting_identity,
        adapter.compact_identity_basis,
        optimize=True,
    )
    bad_fit_detections[bad_fit_index] = _detection(
        project_calibrated_points(
            conflicting_shape, views[bad_fit_index], registration
        )
    )
    with pytest.raises(ValueError, match="stable accepted-view set"):
        _fit_with_calibrated_bundle(
            adapter, rig, tuple(bad_fit_detections), bundle
        )


def test_calibrated_pipeline_writes_matrix_provenance_and_held_out_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter: GNMAdapter,
) -> None:
    identity = np.zeros(adapter.identity_dim, dtype=np.float64)
    identity[:30] = np.random.default_rng(8).normal(0.0, 0.15, 30)
    shape = adapter.compact_template + np.einsum(
        "i,ilc->lc", identity, adapter.compact_identity_basis, optimize=True
    )
    yaws = (-0.80, -1.08, -0.54, 0.0, 0.80, 0.54, 1.08)
    held_out = {0, 4}
    views = tuple(
        _view(index, yaw, "held_out" if index in held_out else "fit")
        for index, yaw in enumerate(yaws)
    )
    bundle = _bundle(views)
    registration = CameraRegistration(1.0, np.eye(3), np.zeros(3), 0.0, 0.0)
    detections = tuple(
        _detection(project_calibrated_points(shape, view, registration)) for view in views
    )
    inputs = []
    for index in range(len(views)):
        path = tmp_path / f"view-{index}.png"
        path.write_bytes(f"synthetic calibrated image {index}".encode())
        inputs.append(path)
    payload = bundle.as_dict()
    for derived in (
        "source_sha256",
        "fit_view_indices",
        "held_out_view_indices",
        "declared_calibration_metadata_gate_passed",
    ):
        payload.pop(derived)
    sidecar = tmp_path / "rig.json"
    _write_payload(sidecar, payload)

    class FakeExtractor:
        def __init__(self, _model_path):
            pass

        def detect(self, path):
            index = int(Path(path).stem.rsplit("-", 1)[1])
            return detections[index]

    monkeypatch.setattr(multiview_pipeline_module, "FaceExtractor", FakeExtractor)
    output = tmp_path / "output"
    result = run_multiview_pipeline(
        inputs,
        output,
        model_path=tmp_path / "unused.task",
        texture_size=128,
        camera_bundle_path=sidecar,
        input_names=tuple(path.name for path in inputs),
    )

    assert result["capture"]["intrinsics_source"] == "measured_calibration"
    assert result["capture"]["fit_view_indices"] == [1, 2, 3, 5, 6]
    assert result["capture"]["held_out_view_indices"] == [0, 4]
    assert result["capture"]["held_out"]["passed"]
    assert result["fit"]["calibrated_geometry_gate_passed"]
    assert result["fit"]["production_validated"] is False
    assert (output / "capture-calibration.json").is_file()
    assert (output / "gnm-camera-registration.json").is_file()
    with np.load(output / "fit.npz", allow_pickle=False) as values:
        assert "cameras" not in values.files
        assert values["source_K"].shape == (7, 3, 3)
        assert values["source_D"].shape == (7, 5)
        assert values["source_world_to_camera"].shape == (7, 4, 4)
        assert values["effective_gnm_to_camera"].shape == (7, 4, 4)
        np.testing.assert_array_equal(values["fit_view_indices"], (1, 2, 3, 5, 6))
        np.testing.assert_array_equal(values["held_out_view_indices"], (0, 4))
        assert set(values["accepted_view_indices"]).isdisjoint((0, 4))
    with np.load(output / "texture-maps.npz", allow_pickle=False) as texture:
        np.testing.assert_array_equal(
            texture["texture_view_local_to_global"], (1, 2, 3, 5, 6)
        )
        sourced = texture["source_view_global"] >= 0
        assert set(np.unique(texture["source_view_global"][sourced])).issubset(
            {1, 2, 3, 5, 6}
        )
        assert set(np.unique(texture["source_view_global"][sourced])).isdisjoint(
            held_out
        )

    corrupted = list(detections)
    for index in bundle.held_out_indices:
        corrupted[index] = _detection(
            detections[index].landmarks + np.asarray((40.0, -25.0))
        )
    detections = tuple(corrupted)
    with pytest.raises(AutoAnimError, match="independent geometric validation") as caught:
        run_multiview_pipeline(
            inputs,
            tmp_path / "corrupted-output",
            model_path=tmp_path / "unused.task",
            texture_size=128,
            camera_bundle_path=sidecar,
            input_names=tuple(path.name for path in inputs),
        )
    assert caught.value.code == "FIT_REJECTED"
    assert "HELD_OUT_REPROJECTION" in caught.value.details["failures"]
    assert caught.value.details["held_out"]["fit_leakage"] is False


def test_multiview_api_forwards_and_caps_camera_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "unused.task")
    captured: dict = {}

    def fake_multiview(paths, **kwargs):
        captured["image_count"] = len(paths)
        captured.update(kwargs)
        assert Path(kwargs["camera_bundle_path"]).is_file()
        return {"status": "succeeded", "job_id": "test"}

    monkeypatch.setattr(app.state.service, "multiview", fake_multiview)
    client = TestClient(app)
    files = [
        ("files", (f"view-{index}.png", f"image-{index}".encode(), "image/png"))
        for index in range(4)
    ]
    response = client.post(
        "/api/multiview",
        files=files
        + [("calibration", ("rig.json", b'{"schema_version":"test"}', "application/json"))],
    )

    assert response.status_code == 201
    assert captured["image_count"] == 4
    assert captured["input_names"] == tuple(f"view-{index}.png" for index in range(4))
    assert captured["camera_bundle_path"] is not None

    oversized = client.post(
        "/api/multiview",
        files=files
        + [("calibration", ("rig.json", b"x" * (1024 * 1024 + 1), "application/json"))],
    )
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "LIMIT_EXCEEDED"


def test_service_uses_retained_sidecar_hash_and_enforces_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = []
    for index in range(3):
        image = tmp_path / f"view-{index}.png"
        image.write_bytes(f"image-{index}".encode())
        images.append(image)
    sidecar = tmp_path / "rig.json"
    sidecar.write_bytes(b'{"schema_version":"test"}')
    captured: dict = {}

    def fake_pipeline(paths, output, **kwargs):
        captured["bundle_path"] = Path(kwargs["camera_bundle_path"])
        captured["bundle_sha256"] = hashlib.sha256(
            captured["bundle_path"].read_bytes()
        ).hexdigest()
        return {
            "kind": "multiview_reconstruction",
            "status": "succeeded",
            "artifacts": {},
            "warnings": [],
        }

    monkeypatch.setattr(service_module, "run_multiview_pipeline", fake_pipeline)
    service = ApplicationService(tmp_path / "jobs", model_path=tmp_path / "unused.task")
    result = service.multiview(
        images,
        input_names=tuple(path.name for path in images),
        camera_bundle_path=sidecar,
    )

    assert captured["bundle_path"].is_file()
    assert result["configuration"]["calibration_sha256"] == captured["bundle_sha256"]
    assert result["attachments"][0]["sha256"] == captured["bundle_sha256"]
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * 1_000_001)
    with pytest.raises(AutoAnimError) as caught:
        service.multiview(images, camera_bundle_path=oversized)
    assert caught.value.code == "LIMIT_EXCEEDED"
