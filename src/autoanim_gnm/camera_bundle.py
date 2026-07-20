"""Versioned calibrated-camera ingress and GNM-to-capture registration.

The bundle keeps three coordinate systems explicit:

* images and OpenCV intrinsics are measured in pixels;
* ``world_to_camera`` is the calibrated OpenCV transform supplied by capture;
* one shared similarity registers GNM model coordinates into capture world.

Fit images estimate that shared similarity and identity. Held-out images never
enter either solve; they are projected only after the fit for independent
cross-view evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .artifacts import safe_input_name, sha256
from .errors import AutoAnimError
from .multiview import CameraIntrinsics, PerspectiveCamera, rotation_matrix


CAMERA_BUNDLE_SCHEMA = "autoanim.calibrated_multiview.v1"
CAMERA_BUNDLE_MODEL = "opencv_radtan"
CAMERA_BUNDLE_CONVENTION = "opencv_world_to_camera_+x_right_+y_down_+z_forward"
CALIBRATION_RMS_GATE_PX = 0.40
CALIBRATION_POSE_GATE_DEGREES = 2.0
CALIBRATION_SCALE_GATE_FRACTION = 0.01
ViewUsage = Literal["fit", "held_out"]


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CalibratedCameraView:
    index: int
    image_name: str
    role: str
    usage: ViewUsage
    image_size: tuple[int, int]
    intrinsics_matrix: np.ndarray
    distortion: np.ndarray
    world_to_camera: np.ndarray
    visibility: np.ndarray

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Calibrated camera index must be nonnegative")
        object.__setattr__(
            self, "intrinsics_matrix", _readonly(self.intrinsics_matrix, np.float64)
        )
        object.__setattr__(self, "distortion", _readonly(self.distortion, np.float64))
        object.__setattr__(
            self, "world_to_camera", _readonly(self.world_to_camera, np.float64)
        )
        object.__setattr__(self, "visibility", _readonly(self.visibility, np.float64))
        if not self.image_name or not self.role or self.usage not in {"fit", "held_out"}:
            raise ValueError("Calibrated camera name, role, and usage must be valid")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("Calibrated camera image_size must be positive [height,width]")
        if (
            self.intrinsics_matrix.shape != (3, 3)
            or not np.isfinite(self.intrinsics_matrix).all()
            or self.intrinsics_matrix[0, 0] <= 0.0
            or self.intrinsics_matrix[1, 1] <= 0.0
        ):
            raise ValueError("Calibrated camera intrinsics must be finite [3,3]")
        if self.distortion.shape != (5,) or not np.isfinite(self.distortion).all():
            raise ValueError("Calibrated camera distortion must be finite [5]")
        if self.world_to_camera.shape != (4, 4) or not np.isfinite(
            self.world_to_camera
        ).all():
            raise ValueError("Calibrated world_to_camera must be finite [4,4]")
        camera_rotation = self.world_to_camera[:3, :3]
        if (
            not np.allclose(self.world_to_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(camera_rotation.T @ camera_rotation, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(camera_rotation), 1.0, atol=2e-5)
        ):
            raise ValueError("Calibrated world_to_camera must be a proper rigid transform")
        if (
            self.visibility.shape != (68,)
            or not np.isfinite(self.visibility).all()
            or np.any((self.visibility < 0.0) | (self.visibility > 1.0))
        ):
            raise ValueError("Calibrated visibility must be finite [68] in [0,1]")

    @property
    def intrinsics(self) -> CameraIntrinsics:
        matrix = self.intrinsics_matrix
        return CameraIntrinsics(
            float(matrix[0, 0]),
            float(matrix[1, 1]),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        )

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise AutoAnimError(
                "INPUT_INVALID", "Camera-bundle landmark points must be finite [N,2]"
            )
        if not len(self.distortion) or np.max(np.abs(self.distortion), initial=0.0) <= 1e-15:
            return values.astype(np.float32)
        undistorted = cv2.undistortPoints(
            values.reshape(-1, 1, 2),
            self.intrinsics_matrix,
            self.distortion,
            P=self.intrinsics_matrix,
        )
        return undistorted.reshape(-1, 2).astype(np.float32)

    def undistort_image(self, image_bgr: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr)
        if image.shape[:2] != self.image_size:
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Camera bundle image size does not match {self.image_name}",
                {
                    "bundle_image_size": list(self.image_size),
                    "decoded_image_size": list(image.shape[:2]),
                },
            )
        if not len(self.distortion) or np.max(np.abs(self.distortion), initial=0.0) <= 1e-15:
            return image.copy()
        return cv2.undistort(
            image,
            self.intrinsics_matrix,
            self.distortion,
            None,
            self.intrinsics_matrix,
        )

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "filename": self.image_name,
            "role": self.role,
            "use": self.usage,
            "image_size": list(self.image_size),
            "K": self.intrinsics_matrix.tolist(),
            "D": self.distortion.tolist(),
            "world_to_camera": self.world_to_camera.tolist(),
            "visibility": self.visibility.tolist(),
        }


@dataclass(frozen=True, slots=True)
class CalibratedCameraBundle:
    calibration_rms_px: float
    pose_error_degrees: float
    scale_error_fraction: float
    views: tuple[CalibratedCameraView, ...]
    source_sha256: str
    meters_per_world_unit: float = 1.0
    schema_version: str = CAMERA_BUNDLE_SCHEMA
    camera_model: str = CAMERA_BUNDLE_MODEL
    coordinate_convention: str = CAMERA_BUNDLE_CONVENTION

    def __post_init__(self) -> None:
        metrics = np.asarray(
            (
                self.calibration_rms_px,
                self.pose_error_degrees,
                self.scale_error_fraction,
                self.meters_per_world_unit,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(metrics).all() or np.any(metrics[:3] < 0.0) or metrics[3] <= 0.0:
            raise ValueError("Calibrated bundle metrics and world scale must be finite")
        if tuple(view.index for view in self.views) != tuple(range(len(self.views))):
            raise ValueError("Calibrated bundle view indices must be contiguous")
        if self.schema_version != CAMERA_BUNDLE_SCHEMA:
            raise ValueError("Unsupported calibrated bundle schema")
        if self.camera_model != CAMERA_BUNDLE_MODEL:
            raise ValueError("Unsupported calibrated camera model")
        if self.coordinate_convention != CAMERA_BUNDLE_CONVENTION:
            raise ValueError("Unsupported calibrated camera convention")

    @property
    def fit_indices(self) -> tuple[int, ...]:
        return tuple(index for index, view in enumerate(self.views) if view.usage == "fit")

    @property
    def held_out_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, view in enumerate(self.views) if view.usage == "held_out"
        )

    @property
    def declared_calibration_metadata_gate_passed(self) -> bool:
        return bool(
            self.calibration_rms_px <= CALIBRATION_RMS_GATE_PX
            and self.pose_error_degrees <= CALIBRATION_POSE_GATE_DEGREES
            and self.scale_error_fraction <= CALIBRATION_SCALE_GATE_FRACTION
        )

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "camera_model": self.camera_model,
            "coordinate_convention": self.coordinate_convention,
            "meters_per_world_unit": self.meters_per_world_unit,
            "calibration_rms_px": self.calibration_rms_px,
            "pose_error_degrees": self.pose_error_degrees,
            "scale_error_fraction": self.scale_error_fraction,
            "source_sha256": self.source_sha256,
            "fit_view_indices": list(self.fit_indices),
            "held_out_view_indices": list(self.held_out_indices),
            "declared_calibration_metadata_gate_passed": (
                self.declared_calibration_metadata_gate_passed
            ),
            "views": [view.as_dict() for view in self.views],
        }


@dataclass(frozen=True, slots=True)
class CameraRegistration:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    mean_reprojection_error_px: float
    p95_reprojection_error_px: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Camera registration scale must be positive and finite")
        object.__setattr__(self, "rotation", _readonly(self.rotation, np.float64))
        object.__setattr__(self, "translation", _readonly(self.translation, np.float64))
        if self.rotation.shape != (3, 3) or not np.isfinite(self.rotation).all():
            raise ValueError("Camera registration rotation must be finite [3,3]")
        if (
            not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=2e-5)
        ):
            raise ValueError("Camera registration rotation must be proper and orthonormal")
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError("Camera registration translation must be finite [3]")
        if (
            not np.isfinite(self.mean_reprojection_error_px)
            or self.mean_reprojection_error_px < 0.0
            or not np.isfinite(self.p95_reprojection_error_px)
            or self.p95_reprojection_error_px < 0.0
        ):
            raise ValueError("Camera registration reprojection metrics must be finite and nonnegative")

    def as_matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.scale * self.rotation
        result[:3, 3] = self.translation
        return result

    def as_dict(self) -> dict:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "gnm_to_world": self.as_matrix().tolist(),
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "p95_reprojection_error_px": self.p95_reprojection_error_px,
        }


def _finite_number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise AutoAnimError("INPUT_INVALID", f"Camera bundle {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AutoAnimError("INPUT_INVALID", f"Camera bundle {field} must be numeric") from exc
    if not np.isfinite(number) or number < minimum:
        raise AutoAnimError(
            "INPUT_INVALID", f"Camera bundle {field} must be finite and >= {minimum}"
        )
    return number


def _matrix(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AutoAnimError(
            "INPUT_INVALID", f"Camera bundle {field} must have shape {shape}"
        ) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise AutoAnimError(
            "INPUT_INVALID", f"Camera bundle {field} must be finite with shape {shape}"
        )
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def load_camera_bundle(
    path: str | Path,
    *,
    input_names: Sequence[str],
    image_sizes: Sequence[tuple[int, int]],
) -> CalibratedCameraBundle:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > 1_000_000:
        raise AutoAnimError(
            "INPUT_INVALID", "Camera bundle must be an existing JSON file <= 1 MB"
        )
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AutoAnimError("INPUT_INVALID", f"Could not parse camera bundle JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AutoAnimError("INPUT_INVALID", "Camera bundle root must be an object")
    allowed_root = {
        "schema_version",
        "camera_model",
        "coordinate_convention",
        "meters_per_world_unit",
        "calibration_rms_px",
        "pose_error_degrees",
        "scale_error_fraction",
        "views",
    }
    unknown_root = sorted(set(payload) - allowed_root)
    if unknown_root:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Camera bundle contains unknown root fields",
            {"unknown_fields": unknown_root},
        )
    if payload.get("schema_version") != CAMERA_BUNDLE_SCHEMA:
        raise AutoAnimError(
            "INPUT_INVALID",
            f"Camera bundle schema_version must be {CAMERA_BUNDLE_SCHEMA}",
        )
    if payload.get("camera_model") != CAMERA_BUNDLE_MODEL:
        raise AutoAnimError(
            "INPUT_INVALID",
            f"Camera bundle camera_model must be {CAMERA_BUNDLE_MODEL}",
        )
    if payload.get("coordinate_convention") != CAMERA_BUNDLE_CONVENTION:
        raise AutoAnimError(
            "INPUT_INVALID",
            f"Camera bundle coordinate_convention must be {CAMERA_BUNDLE_CONVENTION}",
        )
    names = tuple(safe_input_name(name) for name in input_names)
    sizes = tuple((int(size[0]), int(size[1])) for size in image_sizes)
    if len(sizes) != len(names):
        raise AutoAnimError(
            "INPUT_INVALID", "Camera bundle image_sizes must match input_names"
        )
    raw_views = payload.get("views")
    if not isinstance(raw_views, list) or len(raw_views) != len(names):
        raise AutoAnimError(
            "INPUT_INVALID",
            "Camera bundle must contain exactly one ordered view per input image",
            {"images": len(names), "bundle_views": len(raw_views) if isinstance(raw_views, list) else None},
        )
    allowed_view = {
        "index",
        "filename",
        "role",
        "use",
        "image_size",
        "K",
        "D",
        "world_to_camera",
        "visibility",
    }
    required_root = allowed_root
    missing_root = sorted(required_root - set(payload))
    if missing_root:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Camera bundle is missing required root fields",
            {"missing_fields": missing_root},
        )
    parsed: list[CalibratedCameraView] = []
    for index, (raw, expected_name, expected_size) in enumerate(
        zip(raw_views, names, sizes, strict=True)
    ):
        if not isinstance(raw, dict):
            raise AutoAnimError("INPUT_INVALID", f"Camera bundle view {index} must be an object")
        unknown = sorted(set(raw) - allowed_view)
        if unknown:
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Camera bundle view {index} contains unknown fields",
                {"unknown_fields": unknown},
            )
        missing = sorted(allowed_view - set(raw))
        if missing:
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Camera bundle view {index} is missing required fields",
                {"missing_fields": missing},
            )
        if raw.get("index") != index:
            raise AutoAnimError(
                "INPUT_INVALID",
                "Camera bundle view indices must be ordered and contiguous from zero",
                {"expected_index": index, "bundle_index": raw.get("index")},
            )
        image_name = safe_input_name(str(raw.get("filename", "")))
        if image_name != expected_name:
            raise AutoAnimError(
                "INPUT_INVALID",
                "Camera bundle image order/name does not match uploaded inputs",
                {
                    "view_index": index,
                    "expected_image": expected_name,
                    "bundle_image": image_name,
                },
            )
        role = str(raw.get("role", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if not role or "rear" in role or "back" in role:
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} has an unsupported role"
            )
        usage = raw.get("use")
        if usage not in {"fit", "held_out"}:
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} usage must be fit or held_out"
            )
        raw_size = raw.get("image_size")
        if (
            not isinstance(raw_size, list)
            or len(raw_size) != 2
            or any(type(value) is not int or value <= 0 for value in raw_size)
        ):
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} image_size must be [height,width]"
            )
        image_size = (int(raw_size[0]), int(raw_size[1]))
        if image_size != expected_size:
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Camera bundle view {index} image_size does not match decoded input",
                {"bundle": list(image_size), "decoded": list(expected_size)},
            )
        intrinsics = _matrix(raw.get("K"), (3, 3), f"views[{index}].K")
        height, width = image_size
        if (
            intrinsics[0, 0] <= 0
            or intrinsics[1, 1] <= 0
            or intrinsics[0, 0] < 0.1 * max(height, width)
            or intrinsics[1, 1] < 0.1 * max(height, width)
            or intrinsics[0, 0] > 20.0 * max(height, width)
            or intrinsics[1, 1] > 20.0 * max(height, width)
            or abs(float(intrinsics[0, 1])) > 1e-8
            or abs(float(intrinsics[1, 0])) > 1e-8
            or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=1e-8)
            or not -0.5 * width <= intrinsics[0, 2] <= 1.5 * width
            or not -0.5 * height <= intrinsics[1, 2] <= 1.5 * height
        ):
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} has invalid pinhole intrinsics"
            )
        try:
            distortion = np.asarray(raw.get("D"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} distortion must be numeric"
            ) from exc
        if (
            distortion.shape != (5,)
            or not np.isfinite(distortion).all()
            or np.max(np.abs(distortion), initial=0.0) > 5.0
        ):
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Camera bundle view {index} has invalid OpenCV distortion coefficients",
            )
        world_to_camera = _matrix(
            raw.get("world_to_camera"), (4, 4), f"views[{index}].world_to_camera"
        )
        rotation = world_to_camera[:3, :3]
        if (
            not np.allclose(world_to_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5)
        ):
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} world_to_camera is not rigid"
            )
        try:
            visibility = np.asarray(raw.get("visibility", np.ones(68)), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AutoAnimError(
                "INPUT_INVALID", f"Camera bundle view {index} visibility must be numeric [68]"
            ) from exc
        if (
            visibility.shape != (68,)
            or not np.isfinite(visibility).all()
            or np.any((visibility < 0.0) | (visibility > 1.0))
            or np.count_nonzero(visibility > 1e-8) < 24
        ):
            raise AutoAnimError(
                "INPUT_INVALID",
                f"Camera bundle view {index} visibility must contain at least 24 finite [0,1] landmarks",
            )
        parsed.append(
            CalibratedCameraView(
                index=index,
                image_name=image_name,
                role=role,
                usage=usage,
                image_size=image_size,
                intrinsics_matrix=intrinsics,
                distortion=distortion,
                world_to_camera=world_to_camera,
                visibility=visibility,
            )
        )
    bundle = CalibratedCameraBundle(
        calibration_rms_px=_finite_number(payload.get("calibration_rms_px"), "calibration_rms_px"),
        pose_error_degrees=_finite_number(payload.get("pose_error_degrees"), "pose_error_degrees"),
        scale_error_fraction=_finite_number(payload.get("scale_error_fraction"), "scale_error_fraction"),
        views=tuple(parsed),
        source_sha256=sha256(source),
        meters_per_world_unit=_finite_number(
            payload.get("meters_per_world_unit"),
            "meters_per_world_unit",
            minimum=np.finfo(np.float64).tiny,
        ),
    )
    if len(bundle.fit_indices) < 3:
        raise AutoAnimError("INPUT_INVALID", "Camera bundle requires at least three fit views")
    if not bundle.held_out_indices:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Calibrated evaluation requires at least one held_out view that is excluded from fitting",
        )
    fit_centers = np.asarray(
        [
            -view.world_to_camera[:3, :3].T @ view.world_to_camera[:3, 3]
            for view in (bundle.views[index] for index in bundle.fit_indices)
        ],
        dtype=np.float64,
    )
    if float(np.max(np.linalg.norm(fit_centers[:, None] - fit_centers[None, :], axis=2))) <= 1e-8:
        raise AutoAnimError(
            "INPUT_INVALID", "Calibrated fit cameras must have a nonzero capture baseline"
        )
    return bundle


def project_calibrated_points(
    points: np.ndarray,
    view: CalibratedCameraView,
    registration: CameraRegistration,
) -> np.ndarray:
    shape = np.asarray(points, dtype=np.float64)
    world = registration.scale * (shape @ registration.rotation.T) + registration.translation
    camera = world @ view.world_to_camera[:3, :3].T + view.world_to_camera[:3, 3]
    depth = camera[:, 2]
    if not np.isfinite(camera).all() or np.any(depth <= 1e-6):
        raise ValueError("Registered GNM points fall behind a calibrated camera")
    matrix = view.intrinsics_matrix
    return np.column_stack(
        (
            matrix[0, 0] * camera[:, 0] / depth + matrix[0, 2],
            matrix[1, 1] * camera[:, 1] / depth + matrix[1, 2],
        )
    )


def estimate_camera_registration(
    shapes: Sequence[np.ndarray],
    observed_landmarks: Sequence[np.ndarray],
    views: Sequence[CalibratedCameraView],
    *,
    meters_per_world_unit: float = 1.0,
) -> CameraRegistration:
    if not len(shapes) or len(shapes) != len(observed_landmarks) or len(shapes) != len(views):
        raise ValueError("Registration shapes, observations, and views must be nonempty and aligned")
    first = max(range(len(views)), key=lambda index: int(np.count_nonzero(views[index].visibility)))
    object_points = np.asarray(shapes[first], dtype=np.float64)
    image_points = np.asarray(observed_landmarks[first], dtype=np.float64)
    visible = views[first].visibility > 1e-8
    success, rvec, tvec = cv2.solvePnP(
        object_points[visible],
        image_points[visible],
        views[first].intrinsics_matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        raise ValueError("Could not initialize GNM-to-world camera registration")
    success, rvec, tvec = cv2.solvePnP(
        object_points[visible],
        image_points[visible],
        views[first].intrinsics_matrix,
        None,
        rvec,
        tvec,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("Could not refine GNM-to-world camera registration")
    if not np.isfinite(meters_per_world_unit) or meters_per_world_unit <= 0.0:
        raise ValueError("meters_per_world_unit must be positive and finite")
    # GNM Head 3 model units are approximately metres. OpenCV solvePnP returns
    # translation in object/model units, whereas calibrated extrinsics may use
    # metres, millimetres, or another declared world unit. Scale the complete
    # PnP similarity before composing it with the capture-world transform.
    initial_scale = 1.0 / meters_per_world_unit
    model_to_camera = np.eye(4, dtype=np.float64)
    model_to_camera[:3, :3] = initial_scale * cv2.Rodrigues(rvec)[0]
    model_to_camera[:3, 3] = initial_scale * tvec[:, 0]
    model_to_world = np.linalg.inv(views[first].world_to_camera) @ model_to_camera
    initial_scaled_rotation = model_to_world[:3, :3]
    initial_scale = float(np.cbrt(np.linalg.det(initial_scaled_rotation)))
    initial_rotation = initial_scaled_rotation / initial_scale
    left, _, right = np.linalg.svd(initial_rotation)
    initial_rotation = left @ right
    if np.linalg.det(initial_rotation) < 0:
        left[:, -1] *= -1
        initial_rotation = left @ right
    x0 = np.concatenate(
        (
            Rotation.from_matrix(initial_rotation).as_rotvec(),
            model_to_world[:3, 3],
            np.asarray((math.log(initial_scale),), dtype=np.float64),
        )
    )
    normalizers: list[float] = []
    for points, view in zip(observed_landmarks, views, strict=True):
        values = np.asarray(points, dtype=np.float64)
        selected = view.visibility > 1e-8
        available = values[selected]
        extent = max(float(np.ptp(available[:, 0])), float(np.ptp(available[:, 1])), 1.0)
        eye = (
            float(np.linalg.norm(values[36] - values[45]))
            if selected[36] and selected[45]
            else 0.0
        )
        normalizers.append(max(eye, 0.35 * extent, 1.0))

    def decode(values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return (
            float(math.exp(values[6])),
            Rotation.from_rotvec(values[:3]).as_matrix(),
            values[3:6],
        )

    def residual(values: np.ndarray) -> np.ndarray:
        scale, rotation, translation = decode(values)
        registration = CameraRegistration(scale, rotation, translation, 0.0, 0.0)
        rows: list[np.ndarray] = []
        for shape, observed, view, normalizer in zip(
            shapes, observed_landmarks, views, normalizers, strict=True
        ):
            selected = view.visibility > 1e-8
            try:
                projected = project_calibrated_points(shape, view, registration)
            except ValueError:
                return np.full(sum(2 * int(np.count_nonzero(v.visibility)) for v in views), 1e3)
            weights = np.sqrt(view.visibility[selected])[:, None]
            rows.append(weights * (projected[selected] - observed[selected]) / normalizer)
        return np.concatenate([row.ravel() for row in rows])

    lower = np.asarray(
        (-np.inf,) * 6 + (math.log(0.25 / meters_per_world_unit),),
        dtype=np.float64,
    )
    upper = np.asarray(
        (np.inf,) * 6 + (math.log(4.0 / meters_per_world_unit),),
        dtype=np.float64,
    )
    solved = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.012,
        max_nfev=500,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    if not solved.success or not np.isfinite(solved.x).all():
        raise ValueError(f"GNM-to-world registration failed: {solved.message}")
    scale, rotation, translation = decode(solved.x)
    provisional = CameraRegistration(scale, rotation, translation, 0.0, 0.0)
    errors: list[np.ndarray] = []
    for shape, observed, view in zip(shapes, observed_landmarks, views, strict=True):
        selected = view.visibility > 1e-8
        projected = project_calibrated_points(shape, view, provisional)
        errors.append(np.linalg.norm(projected[selected] - observed[selected], axis=1))
    all_errors = np.concatenate(errors)
    return CameraRegistration(
        scale=scale,
        rotation=rotation,
        translation=translation,
        mean_reprojection_error_px=float(np.mean(all_errors)),
        p95_reprojection_error_px=float(np.percentile(all_errors, 95)),
    )


def perspective_camera_from_calibration(
    view: CalibratedCameraView,
    registration: CameraRegistration,
) -> PerspectiveCamera:
    combined = view.world_to_camera @ registration.as_matrix()
    scaled_rotation = combined[:3, :3]
    scale = float(np.cbrt(np.linalg.det(scaled_rotation)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Calibrated model-to-camera similarity has invalid scale")
    opencv_rotation = scaled_rotation / scale
    if not np.allclose(opencv_rotation.T @ opencv_rotation, np.eye(3), atol=3e-5):
        raise ValueError("Calibrated model-to-camera rotation is not orthonormal")
    translation = combined[:3, 3] / scale
    fit_rotation = np.stack(
        (opencv_rotation[0], -opencv_rotation[1], -opencv_rotation[2]), axis=0
    )
    yaw, pitch, roll = Rotation.from_matrix(fit_rotation).as_euler("yxz")
    camera = PerspectiveCamera(
        float(yaw),
        float(pitch),
        float(roll),
        float(translation[0]),
        float(-translation[1]),
        float(translation[2]),
        view.intrinsics,
    )
    # Round-trip the convention at ingestion time; a sign error here would
    # silently mirror or back-project every held-out and texture view.
    recovered_fit_rotation = rotation_matrix(camera.yaw, camera.pitch, camera.roll)
    recovered_opencv = np.stack(
        (recovered_fit_rotation[0], -recovered_fit_rotation[1], -recovered_fit_rotation[2]),
        axis=0,
    )
    if not np.allclose(recovered_opencv, opencv_rotation, atol=3e-6):
        raise ValueError("Calibrated camera convention round-trip failed")
    return camera


__all__ = [
    "CALIBRATION_POSE_GATE_DEGREES",
    "CALIBRATION_RMS_GATE_PX",
    "CALIBRATION_SCALE_GATE_FRACTION",
    "CAMERA_BUNDLE_CONVENTION",
    "CAMERA_BUNDLE_MODEL",
    "CAMERA_BUNDLE_SCHEMA",
    "CalibratedCameraBundle",
    "CalibratedCameraView",
    "CameraRegistration",
    "estimate_camera_registration",
    "load_camera_bundle",
    "perspective_camera_from_calibration",
    "project_calibrated_points",
]
