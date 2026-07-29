from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from fastapi.testclient import TestClient

from autoanim_gnm.api import create_app
from autoanim_gnm.calibration_evidence import (
    CALIBRATION_GRAPH,
    CAMERA_MODEL,
    COORDINATE_CONVENTION,
    OBSERVATIONS_SCHEMA,
    build_calibration_observations,
    load_calibration_evidence_report,
    load_calibration_observations,
    recompute_calibration_evidence,
)
from autoanim_gnm.camera_bundle import CalibratedCameraBundle, CalibratedCameraView
from autoanim_gnm.errors import AutoAnimError
from autoanim_gnm.service import ApplicationService
import autoanim_gnm.service as service_module
from autoanim_gnm.multiview_pipeline import run_multiview_pipeline
import autoanim_gnm.multiview_pipeline as multiview_pipeline_module


def _rig_transform(center_x: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = -center_x
    return transform


def _synthetic_fixture(
    *, noise_px: float = 0.0
) -> tuple[CalibratedCameraBundle, dict, list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(2217)
    matrices = [
        np.asarray(
            (
                (900.0 + 4.0 * index, 0.0, 639.5 + 0.15 * index),
                (895.0 + 3.0 * index, 479.5 - 0.1 * index, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        for index in range(7)
    ]
    # Correct the intentionally readable row construction above into OpenCV K.
    for index, matrix in enumerate(matrices):
        matrix[1] = (0.0, 895.0 + 3.0 * index, 479.5 - 0.1 * index)
    distortions = [np.zeros(5, dtype=np.float64) for _ in range(7)]
    centers = np.linspace(-0.18, 0.18, 7)
    views = tuple(
        CalibratedCameraView(
            index=index,
            image_name=f"face-{index}.png",
            role=(
                "left_profile",
                "left_3q",
                "left_soft_3q",
                "front",
                "right_soft_3q",
                "right_3q",
                "right_profile",
            )[index],
            usage="fit" if index < 5 else "held_out",
            image_size=(960, 1280),
            intrinsics_matrix=matrices[index],
            distortion=distortions[index],
            world_to_camera=_rig_transform(float(centers[index])),
            visibility=np.ones(68, dtype=np.float64),
        )
        for index in range(7)
    )
    bundle = CalibratedCameraBundle(
        calibration_rms_px=0.1,
        pose_error_degrees=0.1,
        scale_error_fraction=0.001,
        views=views,
        source_sha256="b" * 64,
        meters_per_world_unit=1.0,
    )
    object_points = np.asarray(
        [(column * 0.03, row * 0.03, 0.0) for row in range(6) for column in range(9)],
        dtype=np.float64,
    )
    frames = []
    for frame_index in range(15):
        rvec = np.asarray(
            (
                0.35 * np.sin(0.43 * frame_index),
                -0.38 + 0.058 * frame_index,
                0.22 * np.cos(0.31 * frame_index),
            ),
            dtype=np.float64,
        )
        target_to_world = np.eye(4, dtype=np.float64)
        target_to_world[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        target_to_world[:3, 3] = (
            -0.12 + 0.14 * np.sin(frame_index),
            -0.075 + 0.09 * np.cos(0.7 * frame_index),
            0.82 + 0.035 * frame_index,
        )
        detections = []
        for camera_index in range(7):
            target_to_camera = views[camera_index].world_to_camera @ target_to_world
            projected, _ = cv2.projectPoints(
                object_points,
                Rotation.from_matrix(target_to_camera[:3, :3]).as_rotvec(),
                target_to_camera[:3, 3],
                matrices[camera_index],
                distortions[camera_index],
            )
            image_points = projected[:, 0]
            if noise_px:
                image_points = image_points + rng.normal(0.0, noise_px, image_points.shape)
            detections.append(
                {
                    "camera_id": f"camera-{camera_index}",
                    "point_ids": list(range(len(object_points))),
                    "image_points": image_points.tolist(),
                }
            )
        frames.append(
            {
                "frame_id": f"target-{frame_index:03d}",
                "use": "fit" if frame_index < 12 else "held_out",
                "source_artifact_sha256": hashlib.sha256(
                    f"synchronized-target-{frame_index}".encode()
                ).hexdigest(),
                "detections": detections,
            }
        )
    payload = build_calibration_observations(
        {
            "schema_version": OBSERVATIONS_SCHEMA,
            "camera_bundle_sha256": bundle.source_sha256,
            "capture_session_id": "capture-session-001",
            "solver": {
                "opencv_version": cv2.__version__,
                "camera_model": CAMERA_MODEL,
                "flags": 0,
                "max_iterations": 100,
                "epsilon": 1e-12,
                "opencv_threads": 1,
                "calibration_graph": CALIBRATION_GRAPH,
                "reference_camera_id": "camera-0",
            },
            "target": {
                "type": "checkerboard",
                "columns": 9,
                "rows": 6,
                "square_length_m": 0.03,
                "marker_length_m": None,
                "dictionary": None,
                "legacy_pattern": False,
                "coordinate_convention": COORDINATE_CONVENTION,
            },
            "cameras": [
                {
                    "camera_id": f"camera-{index}",
                    "bundle_view_index": index,
                    "image_size": [960, 1280],
                }
                for index in range(7)
            ],
            "frames": frames,
        }
    )
    return bundle, payload, matrices, distortions


@pytest.fixture(scope="module")
def solved_fixture() -> tuple[CalibratedCameraBundle, dict, dict]:
    bundle, payload, _, _ = _synthetic_fixture(noise_px=0.005)
    report = recompute_calibration_evidence(payload, camera_bundle=bundle)
    return bundle, payload, report


def test_synthetic_seven_camera_recomputation_passes_and_stays_non_production(
    solved_fixture,
) -> None:
    bundle, payload, report = solved_fixture

    assert report["raw_calibration_recomputed"] is True
    assert report["calibration_ready_for_identity_fit"] is True
    assert report["asset_identity_validated"] is False
    assert report["production_validated"] is False
    assert report["failures"] == []
    assert report["bindings"]["camera_bundle_sha256"] == bundle.source_sha256
    assert report["bindings"]["observations_sha256"] == payload["observations_sha256"]
    assert len(report["intrinsic_fit"]) == 7
    assert len(report["bindings"]["fit_frame_ids"]) == 12
    assert len(report["bindings"]["held_out_frame_ids"]) == 3
    assert report["held_out_evidence"]["aggregate"]["rms_px"] < 1.0
    assert load_calibration_evidence_report(report) == report
    observations = load_calibration_observations(payload, camera_bundle=bundle)
    assert load_calibration_evidence_report(
        report, observations=observations, camera_bundle=bundle
    ) == report


def test_held_out_corruption_cannot_leak_into_intrinsics_or_rig(
    solved_fixture,
) -> None:
    bundle, payload, baseline = solved_fixture
    corrupt = deepcopy(payload)
    for point in corrupt["frames"][-1]["detections"][-1]["image_points"]:
        point[0] += 35.0
        point[1] -= 20.0
    corrupt = build_calibration_observations(corrupt)

    adversarial = recompute_calibration_evidence(corrupt, camera_bundle=bundle)

    for first, second in zip(
        adversarial["rig_comparison"], baseline["rig_comparison"], strict=True
    ):
        # OpenCV's native reduction can vary at the last few floating-point
        # ulps across repeated calls; the held-out payload still cannot cause a
        # material change in the fit-only solution.
        np.testing.assert_allclose(first["recomputed_K"], second["recomputed_K"], atol=1e-5)
        np.testing.assert_allclose(first["recomputed_D"], second["recomputed_D"], atol=1e-5)
        np.testing.assert_allclose(
            first["recomputed_reference_to_camera"],
            second["recomputed_reference_to_camera"],
            atol=1e-5,
        )
    assert adversarial["fit_evidence"]["aggregate"]["rms_px"] == pytest.approx(
        baseline["fit_evidence"]["aggregate"]["rms_px"], abs=1e-3
    )
    assert adversarial["calibration_ready_for_identity_fit"] is False
    assert "HELD_OUT_REPROJECTION_RMS_FAILED" in adversarial["failures"]
    forged = deepcopy(adversarial)
    forged["failures"] = []
    forged["calibration_ready_for_identity_fit"] = True
    forged.pop("report_sha256")
    forged["report_sha256"] = hashlib.sha256(
        json.dumps(
            forged,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(AutoAnimError, match="failures contradict evidence"):
        load_calibration_evidence_report(forged)


def test_declared_camera_adversary_fails_closed() -> None:
    bundle, payload, _, _ = _synthetic_fixture(noise_px=0.01)
    views = list(bundle.views)
    bad_transform = views[-1].world_to_camera.copy()
    bad_transform[:3, :3] = Rotation.from_euler("y", 4.0, degrees=True).as_matrix()
    views[-1] = CalibratedCameraView(
        index=views[-1].index,
        image_name=views[-1].image_name,
        role=views[-1].role,
        usage=views[-1].usage,
        image_size=views[-1].image_size,
        intrinsics_matrix=views[-1].intrinsics_matrix,
        distortion=views[-1].distortion,
        world_to_camera=bad_transform,
        visibility=views[-1].visibility,
    )
    adversarial_bundle = CalibratedCameraBundle(
        calibration_rms_px=bundle.calibration_rms_px,
        pose_error_degrees=bundle.pose_error_degrees,
        scale_error_fraction=bundle.scale_error_fraction,
        views=tuple(views),
        source_sha256=bundle.source_sha256,
    )

    report = recompute_calibration_evidence(payload, camera_bundle=adversarial_bundle)

    assert report["calibration_ready_for_identity_fit"] is False
    assert "camera-6:RIG_ROTATION_MISMATCH" in report["failures"]


def test_observation_schema_rejects_unknowns_hash_tampering_and_runtime_drift() -> None:
    bundle, payload, _, _ = _synthetic_fixture()
    unknown = deepcopy(payload)
    unknown["surprise"] = True
    with pytest.raises(AutoAnimError, match="fields do not match schema"):
        load_calibration_observations(unknown, camera_bundle=bundle)

    tampered = deepcopy(payload)
    tampered["frames"][0]["detections"][0]["image_points"][0][0] += 1.0
    with pytest.raises(AutoAnimError, match="payload hash mismatch"):
        load_calibration_observations(tampered, camera_bundle=bundle)

    drift = deepcopy(payload)
    drift["solver"]["opencv_version"] = "999.0.0"
    drift.pop("observations_sha256")
    drift["observations_sha256"] = hashlib.sha256(
        json.dumps(
            drift,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(AutoAnimError, match="frozen runtime"):
        load_calibration_observations(drift, camera_bundle=bundle)


def test_report_cannot_escalate_production_claim(solved_fixture) -> None:
    _, _, report = solved_fixture
    forged = deepcopy(report)
    forged["production_validated"] = True
    forged.pop("report_sha256")
    forged["report_sha256"] = hashlib.sha256(
        json.dumps(
            forged,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(AutoAnimError, match="invalid claim"):
        load_calibration_evidence_report(forged)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda report: report["held_out_evidence"]["per_camera"][3].__setitem__(
            "rms_px",
            report["held_out_evidence"]["per_camera"][3]["rms_px"] + 0.125,
        ),
        lambda report: report["rig_comparison"][2]["recomputed_K"][0].__setitem__(
            0,
            report["rig_comparison"][2]["recomputed_K"][0][0] + 2.0,
        ),
    ),
)
def test_source_replay_rejects_self_hashed_derived_evidence_forgery(
    solved_fixture, mutator
) -> None:
    bundle, payload, report = solved_fixture
    forged = deepcopy(report)
    mutator(forged)
    forged.pop("report_sha256")
    forged["report_sha256"] = hashlib.sha256(
        json.dumps(
            forged,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    observations = load_calibration_observations(payload, camera_bundle=bundle)
    with pytest.raises(AutoAnimError, match="does not match deterministic source replay"):
        load_calibration_evidence_report(
            forged,
            observations=observations,
            camera_bundle=bundle,
        )


def test_numeric_types_reject_strings_bools_and_report_scalar_type_forgery(
    solved_fixture,
) -> None:
    bundle, payload, report = solved_fixture
    for bad_value in ("640.0", True):
        forged_observations = deepcopy(payload)
        forged_observations["frames"][0]["detections"][0]["image_points"][0][0] = bad_value
        forged_observations.pop("observations_sha256")
        forged_observations["observations_sha256"] = hashlib.sha256(
            json.dumps(
                forged_observations,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with pytest.raises(AutoAnimError, match="JSON-number"):
            load_calibration_observations(forged_observations, camera_bundle=bundle)

    string_square = deepcopy(payload)
    string_square["target"]["square_length_m"] = "0.03"
    string_square.pop("observations_sha256")
    string_square["observations_sha256"] = hashlib.sha256(
        json.dumps(
            string_square,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(AutoAnimError, match="positive number"):
        load_calibration_observations(string_square, camera_bundle=bundle)

    scalar_type = deepcopy(report)
    scalar_type["held_out_evidence"]["aggregate"]["rms_px"] = str(
        scalar_type["held_out_evidence"]["aggregate"]["rms_px"]
    )
    scalar_type.pop("report_sha256")
    scalar_type["report_sha256"] = hashlib.sha256(
        json.dumps(
            scalar_type,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    observations = load_calibration_observations(payload, camera_bundle=bundle)
    with pytest.raises(AutoAnimError, match="nonnegative number"):
        load_calibration_evidence_report(
            scalar_type,
            observations=observations,
            camera_bundle=bundle,
        )


def test_charuco_object_point_convention_is_explicit_when_aruco_available() -> None:
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV aruco module is unavailable")
    bundle, payload, _, _ = _synthetic_fixture()
    payload["target"] = {
        "type": "charuco",
        "columns": 7,
        "rows": 5,
        "square_length_m": 0.04,
        "marker_length_m": 0.02,
        "dictionary": "DICT_4X4_50",
        "legacy_pattern": False,
        "coordinate_convention": COORDINATE_CONVENTION,
    }
    # We are validating the target convention here, not solving the checkerboard
    # fixture with a different target, so retain only in-range corner IDs.
    for frame in payload["frames"]:
        for detection in frame["detections"]:
            detection["point_ids"] = detection["point_ids"][:24]
            detection["image_points"] = detection["image_points"][:24]
    payload = build_calibration_observations(payload)
    loaded = load_calibration_observations(payload, camera_bundle=bundle)
    points = loaded.target.object_points((0, 1, 6))
    assert points.shape == (3, 3)
    assert np.isfinite(points).all()


def test_api_forwards_calibration_observations_and_enforces_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(tmp_path / "jobs", model_path=tmp_path / "unused.task")
    captured: dict = {}

    def fake_multiview(paths, **kwargs):
        captured.update(kwargs)
        assert Path(kwargs["camera_bundle_path"]).is_file()
        assert Path(kwargs["calibration_observations_path"]).is_file()
        return {"status": "succeeded", "job_id": "test"}

    monkeypatch.setattr(app.state.service, "multiview", fake_multiview)
    client = TestClient(app)
    images = [
        ("files", (f"view-{index}.png", f"image-{index}".encode(), "image/png"))
        for index in range(3)
    ]
    response = client.post(
        "/api/multiview",
        files=images
        + [
            ("calibration", ("rig.json", b"{}", "application/json")),
            (
                "calibration_observations",
                ("targets.json", b"{}", "application/json"),
            ),
        ],
    )
    assert response.status_code == 201
    assert captured["calibration_observations_path"] is not None

    oversized = client.post(
        "/api/multiview",
        files=images
        + [
            ("calibration", ("rig.json", b"{}", "application/json")),
            (
                "calibration_observations",
                ("targets.json", b"x" * (16 * 1024 * 1024 + 1), "application/json"),
            ),
        ],
    )
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "LIMIT_EXCEEDED"


def test_service_retains_and_hashes_observations_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = []
    for index in range(3):
        path = tmp_path / f"view-{index}.png"
        path.write_bytes(f"image-{index}".encode())
        images.append(path)
    bundle = tmp_path / "rig.json"
    bundle.write_bytes(b"bundle")
    observations = tmp_path / "targets.json"
    observations.write_bytes(b"target-observations")
    captured: dict = {}

    def fake_pipeline(paths, output, **kwargs):
        captured["observations"] = Path(kwargs["calibration_observations_path"])
        captured["bundle"] = Path(kwargs["camera_bundle_path"])
        return {
            "kind": "multiview_reconstruction",
            "status": "succeeded",
            "artifacts": {},
            "warnings": [],
        }

    monkeypatch.setattr(service_module, "run_multiview_pipeline", fake_pipeline)
    service = ApplicationService(tmp_path / "service-jobs", model_path=tmp_path / "unused.task")
    result = service.multiview(
        images,
        input_names=tuple(path.name for path in images),
        camera_bundle_path=bundle,
        calibration_observations_path=observations,
    )
    assert captured["observations"].is_file()
    assert captured["bundle"].is_file()
    expected = hashlib.sha256(observations.read_bytes()).hexdigest()
    assert result["configuration"]["calibration_observations_sha256"] == expected
    attachment = next(
        value
        for value in result["attachments"]
        if value["logical_name"] == "calibration_observations"
    )
    assert attachment["sha256"] == expected


def test_multiview_fails_before_identity_solver_when_recomputed_evidence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, _, _ = _synthetic_fixture()
    bundle_payload = bundle.as_dict()
    for derived in (
        "source_sha256",
        "fit_view_indices",
        "held_out_view_indices",
        "declared_calibration_metadata_gate_passed",
    ):
        bundle_payload.pop(derived)
    bundle_path = tmp_path / "rig.json"
    bundle_path.write_text(json.dumps(bundle_payload), encoding="utf-8")
    observations_path = tmp_path / "targets.json"
    observations_path.write_text("{}", encoding="utf-8")
    images = []
    for index in range(7):
        path = tmp_path / f"face-{index}.png"
        path.write_bytes(f"image-{index}".encode())
        images.append(path)

    class FakeExtractor:
        def __init__(self, _model_path):
            pass

        def detect(self, _path):
            return SimpleNamespace(image_bgr=np.zeros((960, 1280, 3), dtype=np.uint8))

    monkeypatch.setattr(multiview_pipeline_module, "FaceExtractor", FakeExtractor)
    monkeypatch.setattr(
        multiview_pipeline_module,
        "recompute_calibration_evidence",
        lambda *args, **kwargs: {
            "calibration_ready_for_identity_fit": False,
            "failures": ["HELD_OUT_REPROJECTION_RMS_FAILED"],
            "report_sha256": "f" * 64,
            "production_validated": False,
        },
    )

    def identity_solver_must_not_start():
        raise AssertionError("identity solver started before calibration readiness")

    monkeypatch.setattr(multiview_pipeline_module, "GNMAdapter", identity_solver_must_not_start)
    with pytest.raises(AutoAnimError, match="failed closed before identity fitting") as caught:
        run_multiview_pipeline(
            images,
            tmp_path / "output",
            model_path=tmp_path / "unused.task",
            camera_bundle_path=bundle_path,
            calibration_observations_path=observations_path,
            input_names=tuple(path.name for path in images),
        )
    assert caught.value.code == "FIT_REJECTED"
    assert caught.value.details["production_validated"] is False


@pytest.mark.skipif(
    not os.environ.get("AUTOANIM_OPENCV_CALIBRATION_SAMPLE_DIR"),
    reason="Set AUTOANIM_OPENCV_CALIBRATION_SAMPLE_DIR to the Apache-2.0 OpenCV left*.jpg sample directory",
)
def test_real_opencv_checkerboard_fixture_runs_canonical_recompute_path() -> None:
    # OpenCV publishes this monocular optical fixture under Apache-2.0.  Schema
    # v1 needs a rig, so the same measured corner observation is replicated as
    # three co-located virtual cameras.  This validates real lens/corner data
    # through the canonical solver, not physical multicamera extrinsics.
    sample_dir = Path(os.environ["AUTOANIM_OPENCV_CALIBRATION_SAMPLE_DIR"])
    images = sorted(sample_dir.glob("left*.jpg"))
    object_points = np.zeros((9 * 6, 3), np.float32)
    object_points[:, :2] = 0.03 * np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    detections: list[tuple[Path, np.ndarray]] = []
    image_size = None
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        success, corners = cv2.findChessboardCorners(image, (9, 6))
        if success:
            detections.append((path, np.asarray(corners).reshape(-1, 2)))
            image_size = (image.shape[1], image.shape[0])
    assert len(detections) >= 10
    fit_detections = detections[:-2]
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        calibration = cv2.calibrateCameraExtended(
            [object_points for _ in fit_detections],
            [
                np.ascontiguousarray(corners, dtype=np.float32).reshape(-1, 1, 2)
                for _, corners in fit_detections
            ],
            image_size,
            None,
            None,
            flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-12),
        )
    finally:
        cv2.setNumThreads(previous_threads)
    rms = float(calibration[0])
    matrix = np.asarray(calibration[1], dtype=np.float64)
    distortion = np.asarray(calibration[2], dtype=np.float64).reshape(5)
    assert rms < 0.5
    width, height = image_size
    source_digest = hashlib.sha256()
    for path, _ in detections:
        source_digest.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    bundle = CalibratedCameraBundle(
        calibration_rms_px=rms,
        pose_error_degrees=0.0,
        scale_error_fraction=0.0,
        views=tuple(
            CalibratedCameraView(
                index=index,
                image_name=f"replicated-optical-camera-{index}.png",
                role=("left", "front", "right")[index],
                usage="fit" if index < 2 else "held_out",
                image_size=(height, width),
                intrinsics_matrix=matrix,
                distortion=distortion,
                world_to_camera=np.eye(4, dtype=np.float64),
                visibility=np.ones(68, dtype=np.float64),
            )
            for index in range(3)
        ),
        source_sha256=source_digest.hexdigest(),
        meters_per_world_unit=1.0,
    )
    observations_payload = build_calibration_observations(
        {
            "schema_version": OBSERVATIONS_SCHEMA,
            "camera_bundle_sha256": bundle.source_sha256,
            "capture_session_id": "opencv-replicated-optical-fixture",
            "solver": {
                "opencv_version": cv2.__version__,
                "camera_model": CAMERA_MODEL,
                "flags": 0,
                "max_iterations": 100,
                "epsilon": 1e-12,
                "opencv_threads": 1,
                "calibration_graph": CALIBRATION_GRAPH,
                "reference_camera_id": "replicated-camera-0",
            },
            "target": {
                "type": "checkerboard",
                "columns": 9,
                "rows": 6,
                "square_length_m": 0.03,
                "marker_length_m": None,
                "dictionary": None,
                "legacy_pattern": False,
                "coordinate_convention": COORDINATE_CONVENTION,
            },
            "cameras": [
                {
                    "camera_id": f"replicated-camera-{index}",
                    "bundle_view_index": index,
                    "image_size": [height, width],
                }
                for index in range(3)
            ],
            "frames": [
                {
                    "frame_id": f"opencv-{path.stem}",
                    "use": "fit" if index < len(detections) - 2 else "held_out",
                    "source_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "detections": [
                        {
                            "camera_id": f"replicated-camera-{camera_index}",
                            "point_ids": list(range(54)),
                            "image_points": corners.tolist(),
                        }
                        for camera_index in range(3)
                    ],
                }
                for index, (path, corners) in enumerate(detections)
            ],
        }
    )
    report = recompute_calibration_evidence(
        observations_payload, camera_bundle=bundle
    )
    assert report["raw_calibration_recomputed"] is True
    assert report["calibration_ready_for_identity_fit"] is True
    assert report["production_validated"] is False
    assert report["held_out_evidence"]["aggregate"]["rms_px"] < 1.0
