"""Source-backed, independently recomputed multicamera calibration evidence.

The camera bundle used by the identity fitter contains convenient calibration
claims.  This module does not trust those claims.  It consumes retained target
corner observations, recalibrates every camera from fit frames, estimates the
rig transforms from synchronized fit frames, and evaluates frozen calibration
on held-out frames.  Held-out observations never enter an intrinsic, pose, or
rig solve.

The observations document is retained and hash-verified.  Its per-frame
``source_artifact_sha256`` values are exact declarations/bindings, but this
module does not receive the raw target images and therefore cannot verify those
image bytes.  Reports authenticate recomputation from retained corner
coordinates, not the provenance of the photographs that allegedly produced
those coordinates.

Schema v1 deliberately supports a strict reference-camera-star capture.  Every
frame must contain every camera, which makes the independence boundary and the
failure modes auditable.  More general partially connected calibration graphs
can be added under a new schema version.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_bundle import CalibratedCameraBundle
from .errors import AutoAnimError


OBSERVATIONS_SCHEMA = "autoanim.multicamera-calibration-observations/1.0"
REPORT_SCHEMA = "autoanim.multicamera-calibration-report/1.0"
COORDINATE_CONVENTION = "opencv_target_+x_right_+y_down;camera_+x_right_+y_down_+z_forward"
CALIBRATION_GRAPH = "reference_camera_star_v1"
CAMERA_MODEL = "opencv_radtan5"
MAX_OBSERVATIONS_BYTES = 16 * 1024 * 1024
MIN_CAMERAS = 3
MIN_FIT_FRAMES = 6
MIN_HELD_OUT_FRAMES = 2
MIN_POINTS = 6

# Frozen I0.1 evidence gates.  These qualify calibration evidence for a later
# identity fit; they do not qualify a person match or a production asset.
FIT_RMS_GATE_PX = 0.75
HELD_OUT_RMS_GATE_PX = 1.00
FOCAL_RELATIVE_GATE = 0.03
PRINCIPAL_POINT_GATE_PX = 5.0
DISTORTION_OBSERVED_DOMAIN_P95_GATE_PX = 0.50
ROTATION_GATE_DEGREES = 1.0
TRANSLATION_RELATIVE_GATE = 0.02
TRANSLATION_ABSOLUTE_GATE_M = 0.002

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CHARUCO_DICTIONARIES = frozenset(
    name for name in dir(cv2.aruco) if name.startswith("DICT_")
) if hasattr(cv2, "aruco") else frozenset()
_OPENCV_SOLVER_LOCK = threading.Lock()


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise AutoAnimError(
            "INPUT_INVALID", f"Calibration evidence is not canonical JSON: {exc}"
        ) from exc


def _payload_sha256(value: Mapping[str, Any], hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _read_json(source: str | Path | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        try:
            document = json.loads(_canonical(source))
        except json.JSONDecodeError as exc:  # pragma: no cover - canonical output
            raise AutoAnimError("INPUT_INVALID", str(exc)) from exc
    else:
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            path = Path(source)
            if not path.is_file() or path.stat().st_size > MAX_OBSERVATIONS_BYTES:
                raise AutoAnimError(
                    "INPUT_INVALID",
                    "Calibration observations must be an existing JSON file <= 16 MiB",
                )
            raw = path.read_bytes()
        if len(raw) > MAX_OBSERVATIONS_BYTES:
            raise AutoAnimError("INPUT_INVALID", "Calibration JSON exceeds 16 MiB")
        try:
            document = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise AutoAnimError(
                "INPUT_INVALID", f"Could not parse calibration evidence JSON: {exc}"
            ) from exc
    if not isinstance(document, dict):
        raise AutoAnimError("INPUT_INVALID", "Calibration evidence root must be an object")
    return document


def _exact_object(value: object, field: str, keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoAnimError("INPUT_INVALID", f"{field} must be an object")
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise AutoAnimError(
            "INPUT_INVALID",
            f"{field} fields do not match schema",
            {
                "field": field,
                "missing_fields": sorted(expected - actual),
                "unknown_fields": sorted(actual - expected),
            },
        )
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise AutoAnimError("INPUT_INVALID", f"{field} is not a valid identifier")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise AutoAnimError("INPUT_INVALID", f"{field} must be a lowercase SHA-256")
    return value


def _positive_float(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise AutoAnimError("INPUT_INVALID", f"{field} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AutoAnimError("INPUT_INVALID", f"{field} must be a positive finite number")
    return result


def _nonnegative_float(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise AutoAnimError("INPUT_INVALID", f"{field} must be a nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AutoAnimError("INPUT_INVALID", f"{field} must be finite and nonnegative")
    return result


def _numeric_array(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    """Load a JSON numeric array without accepting strings or booleans."""

    def validate(item: object, path: str) -> None:
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{path}.{index}")
            return
        if type(item) not in {int, float}:
            raise AutoAnimError("INPUT_INVALID", f"{path} must be a JSON number")

    validate(value, field)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AutoAnimError("INPUT_INVALID", f"{field} must be numeric {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise AutoAnimError("INPUT_INVALID", f"{field} must be finite {shape}")
    return result


@dataclass(frozen=True, slots=True)
class TargetDefinition:
    target_type: str
    columns: int
    rows: int
    square_length_m: float
    marker_length_m: float | None
    dictionary: str | None
    legacy_pattern: bool

    def object_points(self, point_ids: Sequence[int]) -> np.ndarray:
        if self.target_type == "checkerboard":
            all_points = np.asarray(
                [
                    (column * self.square_length_m, row * self.square_length_m, 0.0)
                    for row in range(self.rows)
                    for column in range(self.columns)
                ],
                dtype=np.float32,
            )
        else:
            dictionary_id = getattr(cv2.aruco, str(self.dictionary))
            dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            board = cv2.aruco.CharucoBoard(
                (self.columns, self.rows),
                self.square_length_m,
                float(self.marker_length_m),
                dictionary,
            )
            board.setLegacyPattern(self.legacy_pattern)
            all_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
        return np.asarray(all_points[np.asarray(point_ids, dtype=np.int64)], dtype=np.float32)

    @property
    def point_count(self) -> int:
        if self.target_type == "checkerboard":
            return self.columns * self.rows
        return (self.columns - 1) * (self.rows - 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.target_type,
            "columns": self.columns,
            "rows": self.rows,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "dictionary": self.dictionary,
            "legacy_pattern": self.legacy_pattern,
            "coordinate_convention": COORDINATE_CONVENTION,
        }


@dataclass(frozen=True, slots=True)
class TargetDetection:
    camera_id: str
    point_ids: tuple[int, ...]
    image_points: np.ndarray


@dataclass(frozen=True, slots=True)
class CalibrationFrame:
    frame_id: str
    usage: str
    source_artifact_sha256: str
    detections: tuple[TargetDetection, ...]


@dataclass(frozen=True, slots=True)
class CalibrationObservations:
    capture_session_id: str
    camera_bundle_sha256: str
    observations_sha256: str
    solver: dict[str, Any]
    target: TargetDefinition
    camera_ids: tuple[str, ...]
    image_sizes: tuple[tuple[int, int], ...]
    frames: tuple[CalibrationFrame, ...]

    @property
    def fit_frames(self) -> tuple[CalibrationFrame, ...]:
        return tuple(frame for frame in self.frames if frame.usage == "fit")

    @property
    def held_out_frames(self) -> tuple[CalibrationFrame, ...]:
        return tuple(frame for frame in self.frames if frame.usage == "held_out")


def build_calibration_observations(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the canonical self-hash and strictly validate an observations document."""

    document = json.loads(_canonical(payload))
    document.pop("observations_sha256", None)
    document["observations_sha256"] = _payload_sha256(
        document, "observations_sha256"
    )
    load_calibration_observations(document)
    return document


def load_calibration_observations(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    camera_bundle: CalibratedCameraBundle | None = None,
) -> CalibrationObservations:
    document = _read_json(source)
    root = _exact_object(
        document,
        "observations",
        (
            "schema_version",
            "camera_bundle_sha256",
            "capture_session_id",
            "solver",
            "target",
            "cameras",
            "frames",
            "observations_sha256",
        ),
    )
    if root["schema_version"] != OBSERVATIONS_SCHEMA:
        raise AutoAnimError("INPUT_INVALID", "Unsupported calibration observations schema")
    bundle_sha = _sha(root["camera_bundle_sha256"], "camera_bundle_sha256")
    if camera_bundle is not None and bundle_sha != camera_bundle.source_sha256:
        raise AutoAnimError(
            "CALIBRATION_EVIDENCE_FAILED",
            "Calibration observations are not bound to the supplied camera bundle bytes",
            {"expected": camera_bundle.source_sha256, "observed": bundle_sha},
        )
    session_id = _identifier(root["capture_session_id"], "capture_session_id")
    solver = _exact_object(
        root["solver"],
        "solver",
        (
            "opencv_version",
            "camera_model",
            "flags",
            "max_iterations",
            "epsilon",
            "opencv_threads",
            "calibration_graph",
            "reference_camera_id",
        ),
    )
    if (
        solver["opencv_version"] != cv2.__version__
        or solver["camera_model"] != CAMERA_MODEL
        or type(solver["flags"]) is not int
        or solver["flags"] != 0
        or type(solver["max_iterations"]) is not int
        or solver["max_iterations"] != 100
        or type(solver["epsilon"]) is not float
        or solver["epsilon"] != 1e-12
        or type(solver["opencv_threads"]) is not int
        or solver["opencv_threads"] != 1
        or solver["calibration_graph"] != CALIBRATION_GRAPH
    ):
        raise AutoAnimError(
            "INPUT_INVALID",
            "Calibration solver configuration does not match schema-v1's frozen runtime",
            {"runtime_opencv_version": cv2.__version__},
        )
    reference_camera_id = _identifier(
        solver["reference_camera_id"], "solver.reference_camera_id"
    )

    raw_target = _exact_object(
        root["target"],
        "target",
        (
            "type",
            "columns",
            "rows",
            "square_length_m",
            "marker_length_m",
            "dictionary",
            "legacy_pattern",
            "coordinate_convention",
        ),
    )
    target_type = raw_target["type"]
    if not isinstance(target_type, str) or target_type not in {"checkerboard", "charuco"}:
        raise AutoAnimError("INPUT_INVALID", "target.type must be checkerboard or charuco")
    columns, rows = raw_target["columns"], raw_target["rows"]
    if (
        type(columns) is not int
        or type(rows) is not int
        or not 3 <= columns <= 40
        or not 3 <= rows <= 40
    ):
        raise AutoAnimError("INPUT_INVALID", "target rows/columns must be integers in [3,40]")
    square_length = _positive_float(raw_target["square_length_m"], "square_length_m")
    if raw_target["coordinate_convention"] != COORDINATE_CONVENTION:
        raise AutoAnimError("INPUT_INVALID", "Target coordinate convention is unsupported")
    if type(raw_target["legacy_pattern"]) is not bool:
        raise AutoAnimError("INPUT_INVALID", "target.legacy_pattern must be boolean")
    if target_type == "checkerboard":
        if (
            raw_target["marker_length_m"] is not None
            or raw_target["dictionary"] is not None
            or raw_target["legacy_pattern"] is not False
        ):
            raise AutoAnimError(
                "INPUT_INVALID", "Checkerboard marker fields must be null/null/false"
            )
        marker_length = None
        dictionary = None
    else:
        marker_length = _positive_float(
            raw_target["marker_length_m"], "target.marker_length_m"
        )
        dictionary = raw_target["dictionary"]
        if (
            marker_length >= square_length
            or not isinstance(dictionary, str)
            or dictionary not in _CHARUCO_DICTIONARIES
        ):
            raise AutoAnimError("INPUT_INVALID", "Charuco marker geometry/dictionary is invalid")
    target = TargetDefinition(
        target_type,
        columns,
        rows,
        square_length,
        marker_length,
        dictionary,
        raw_target["legacy_pattern"],
    )

    raw_cameras = root["cameras"]
    if not isinstance(raw_cameras, list) or not MIN_CAMERAS <= len(raw_cameras) <= 12:
        raise AutoAnimError("INPUT_INVALID", "Calibration requires 3-12 ordered cameras")
    camera_ids: list[str] = []
    image_sizes: list[tuple[int, int]] = []
    for index, value in enumerate(raw_cameras):
        camera = _exact_object(
            value, f"cameras.{index}", ("camera_id", "bundle_view_index", "image_size")
        )
        camera_id = _identifier(camera["camera_id"], f"cameras.{index}.camera_id")
        size = camera["image_size"]
        if (
            camera["bundle_view_index"] != index
            or not isinstance(size, list)
            or len(size) != 2
            or any(type(item) is not int or item <= 0 for item in size)
        ):
            raise AutoAnimError(
                "INPUT_INVALID", "Calibration cameras must be ordered and have [height,width]"
            )
        if camera_bundle is not None and tuple(size) != camera_bundle.views[index].image_size:
            raise AutoAnimError(
                "CALIBRATION_EVIDENCE_FAILED",
                "Calibration observation image size differs from the camera bundle",
                {"camera_id": camera_id},
            )
        camera_ids.append(camera_id)
        image_sizes.append((size[0], size[1]))
    if len(set(camera_ids)) != len(camera_ids) or reference_camera_id != camera_ids[0]:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Camera IDs must be unique and the first camera must be the reference",
        )
    if camera_bundle is not None and len(camera_ids) != len(camera_bundle.views):
        raise AutoAnimError(
            "CALIBRATION_EVIDENCE_FAILED",
            "Calibration observations do not contain every camera-bundle view",
        )

    raw_frames = root["frames"]
    if not isinstance(raw_frames, list) or not 1 <= len(raw_frames) <= 256:
        raise AutoAnimError("INPUT_INVALID", "Calibration frames must contain 1-256 entries")
    frames: list[CalibrationFrame] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _exact_object(
            raw_frame,
            f"frames.{frame_index}",
            ("frame_id", "use", "source_artifact_sha256", "detections"),
        )
        frame_id = _identifier(frame["frame_id"], f"frames.{frame_index}.frame_id")
        if not isinstance(frame["use"], str) or frame["use"] not in {"fit", "held_out"}:
            raise AutoAnimError("INPUT_INVALID", "Frame use must be fit or held_out")
        source_sha = _sha(
            frame["source_artifact_sha256"],
            f"frames.{frame_index}.source_artifact_sha256",
        )
        raw_detections = frame["detections"]
        if not isinstance(raw_detections, list) or len(raw_detections) != len(camera_ids):
            raise AutoAnimError(
                "INPUT_INVALID", "Every calibration frame must contain every ordered camera"
            )
        detections: list[TargetDetection] = []
        for camera_index, raw_detection in enumerate(raw_detections):
            detection = _exact_object(
                raw_detection,
                f"frames.{frame_index}.detections.{camera_index}",
                ("camera_id", "point_ids", "image_points"),
            )
            if detection["camera_id"] != camera_ids[camera_index]:
                raise AutoAnimError("INPUT_INVALID", "Frame camera order does not match cameras")
            point_ids = detection["point_ids"]
            image_points = detection["image_points"]
            if (
                not isinstance(point_ids, list)
                or not MIN_POINTS <= len(point_ids) <= target.point_count
                or any(type(item) is not int for item in point_ids)
                or point_ids != sorted(set(point_ids))
                or point_ids[0] < 0
                or point_ids[-1] >= target.point_count
                or not isinstance(image_points, list)
                or len(image_points) != len(point_ids)
            ):
                raise AutoAnimError("INPUT_INVALID", "Target point IDs must be sorted and valid")
            if any(
                not isinstance(point, list)
                or len(point) != 2
                or any(type(coordinate) not in {int, float} for coordinate in point)
                for point in image_points
            ):
                raise AutoAnimError(
                    "INPUT_INVALID",
                    "Target image points must contain JSON-number [x,y] pairs",
                )
            points = _numeric_array(
                image_points,
                (len(point_ids), 2),
                f"frames.{frame_index}.detections.{camera_index}.image_points",
            )
            height, width = image_sizes[camera_index]
            if (
                points.shape != (len(point_ids), 2)
                or not np.isfinite(points).all()
                or np.any(points[:, 0] < -0.5 * width)
                or np.any(points[:, 0] > 1.5 * width)
                or np.any(points[:, 1] < -0.5 * height)
                or np.any(points[:, 1] > 1.5 * height)
            ):
                raise AutoAnimError("INPUT_INVALID", "Target image points are invalid or implausible")
            points.setflags(write=False)
            detections.append(TargetDetection(camera_ids[camera_index], tuple(point_ids), points))
        frames.append(CalibrationFrame(frame_id, frame["use"], source_sha, tuple(detections)))
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise AutoAnimError("INPUT_INVALID", "Calibration frame IDs must be unique")
    fit_count = sum(frame.usage == "fit" for frame in frames)
    held_count = sum(frame.usage == "held_out" for frame in frames)
    if fit_count < MIN_FIT_FRAMES or held_count < MIN_HELD_OUT_FRAMES:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Calibration evidence requires at least 6 fit and 2 held-out synchronized frames",
        )
    observed_hash = _sha(root["observations_sha256"], "observations_sha256")
    expected_hash = _payload_sha256(root, "observations_sha256")
    if observed_hash != expected_hash:
        raise AutoAnimError(
            "CALIBRATION_EVIDENCE_FAILED", "Calibration observations payload hash mismatch"
        )
    return CalibrationObservations(
        session_id,
        bundle_sha,
        observed_hash,
        dict(solver),
        target,
        tuple(camera_ids),
        tuple(image_sizes),
        tuple(frames),
    )


def _transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(np.asarray(rvec).reshape(3)).as_matrix()
    result[:3, 3] = np.asarray(tvec).reshape(3)
    return result


def _average_transforms(values: Sequence[np.ndarray]) -> np.ndarray:
    rotations = Rotation.from_matrix(np.asarray([value[:3, :3] for value in values]))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotations.mean().as_matrix()
    result[:3, 3] = np.median(
        np.asarray([value[:3, 3] for value in values]), axis=0
    )
    return result


def _project(
    object_points: np.ndarray,
    target_to_camera: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        object_points,
        Rotation.from_matrix(target_to_camera[:3, :3]).as_rotvec(),
        target_to_camera[:3, 3],
        matrix,
        distortion,
    )
    return projected[:, 0].astype(np.float64)


def _solve_target_pose(
    detection: TargetDetection,
    target: TargetDefinition,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    success, rvec, tvec = cv2.solvePnP(
        target.object_points(detection.point_ids),
        detection.image_points,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        raise AutoAnimError(
            "CALIBRATION_EVIDENCE_FAILED", "OpenCV could not solve a target pose"
        )
    return _transform(rvec, tvec)


def _metrics(errors: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "No finite reprojection evidence")
    return {
        "sample_count": int(len(values)),
        "rms_px": float(np.sqrt(np.mean(values * values))),
        "mean_px": float(np.mean(values)),
        "p95_px": float(np.percentile(values, 95.0)),
        "max_px": float(np.max(values)),
    }


def _distortion_observed_domain_error(
    recovered: np.ndarray,
    declared: np.ndarray,
    matrix: np.ndarray,
    declared_matrix: np.ndarray,
    observed_image_points: np.ndarray,
) -> float:
    # Distortion coefficients are not identifiable outside the part of the lens
    # sampled by the calibration target.  Comparing an arbitrary full-sensor
    # grid would turn harmless high-order extrapolation into false evidence.
    # Evaluate both models on the exact declared-undistorted target rays that
    # were retained in the fit set; the report names this bounded domain.
    normalized = cv2.undistortPoints(
        observed_image_points.reshape(-1, 1, 2), declared_matrix, declared
    ).reshape(-1, 2)
    rays = np.column_stack(
        (normalized, np.ones(len(normalized), dtype=np.float64))
    )
    zero = np.zeros(3, dtype=np.float64)
    first, _ = cv2.projectPoints(rays, zero, zero, matrix, recovered)
    second, _ = cv2.projectPoints(rays, zero, zero, matrix, declared)
    errors = np.linalg.norm(first[:, 0] - second[:, 0], axis=1)
    return float(np.percentile(errors, 95.0))


def _solve_calibration_evidence(
    observations_source: (
        str | Path | bytes | bytearray | Mapping[str, Any] | CalibrationObservations
    ),
    *,
    camera_bundle: CalibratedCameraBundle,
) -> dict[str, Any]:
    """Recompute calibration from fit observations and score frozen held-out frames."""

    observations = (
        observations_source
        if isinstance(observations_source, CalibrationObservations)
        else load_calibration_observations(
            observations_source, camera_bundle=camera_bundle
        )
    )
    if observations.camera_bundle_sha256 != camera_bundle.source_sha256:
        raise AutoAnimError(
            "CALIBRATION_EVIDENCE_FAILED", "Camera bundle hash binding does not match"
        )

    matrices: list[np.ndarray] = []
    distortions: list[np.ndarray] = []
    intrinsic_reports: list[dict[str, Any]] = []
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(observations.solver["max_iterations"]),
        float(observations.solver["epsilon"]),
    )
    for camera_index, camera_id in enumerate(observations.camera_ids):
        object_points = [
            observations.target.object_points(frame.detections[camera_index].point_ids)
            for frame in observations.fit_frames
        ]
        image_points = [
            np.asarray(frame.detections[camera_index].image_points, dtype=np.float32)
            for frame in observations.fit_frames
        ]
        height, width = observations.image_sizes[camera_index]
        try:
            result = cv2.calibrateCameraExtended(
                object_points,
                image_points,
                (width, height),
                None,
                None,
                flags=int(observations.solver["flags"]),
                criteria=criteria,
            )
        except cv2.error as exc:
            raise AutoAnimError(
                "CALIBRATION_EVIDENCE_FAILED",
                f"OpenCV intrinsic calibration failed for {camera_id}: {exc}",
            ) from exc
        rms, matrix, distortion = result[0], result[1], np.asarray(result[2]).reshape(-1)
        per_view = np.asarray(result[7], dtype=np.float64).reshape(-1)
        if (
            not math.isfinite(float(rms))
            or matrix.shape != (3, 3)
            or distortion.shape != (5,)
            or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all()
        ):
            raise AutoAnimError(
                "CALIBRATION_EVIDENCE_FAILED", f"Non-finite calibration for {camera_id}"
            )
        matrices.append(np.asarray(matrix, dtype=np.float64))
        distortions.append(np.asarray(distortion, dtype=np.float64))
        intrinsic_reports.append(
            {
                "camera_id": camera_id,
                "fit_frame_count": len(observations.fit_frames),
                "intrinsic_rms_px": float(rms),
                "per_fit_frame_rms_px": [float(value) for value in per_view],
            }
        )

    relative_transforms: list[list[np.ndarray]] = [[] for _ in observations.camera_ids]
    relative_transforms[0].append(np.eye(4, dtype=np.float64))
    for frame in observations.fit_frames:
        target_to_cameras = [
            _solve_target_pose(
                frame.detections[index],
                observations.target,
                matrices[index],
                distortions[index],
            )
            for index in range(len(observations.camera_ids))
        ]
        target_to_reference_inverse = np.linalg.inv(target_to_cameras[0])
        for index in range(1, len(observations.camera_ids)):
            relative_transforms[index].append(
                target_to_cameras[index] @ target_to_reference_inverse
            )
    rig = [np.eye(4, dtype=np.float64)] + [
        _average_transforms(values) for values in relative_transforms[1:]
    ]

    def score_frames(frames: Sequence[CalibrationFrame]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        all_errors: list[float] = []
        by_camera: list[list[float]] = [[] for _ in observations.camera_ids]
        for frame in frames:
            target_to_reference = _solve_target_pose(
                frame.detections[0], observations.target, matrices[0], distortions[0]
            )
            # Camera zero is the pose anchor and is intentionally not scored.
            for camera_index in range(1, len(observations.camera_ids)):
                detection = frame.detections[camera_index]
                predicted = _project(
                    observations.target.object_points(detection.point_ids),
                    rig[camera_index] @ target_to_reference,
                    matrices[camera_index],
                    distortions[camera_index],
                )
                errors = np.linalg.norm(predicted - detection.image_points, axis=1)
                all_errors.extend(float(value) for value in errors)
                by_camera[camera_index].extend(float(value) for value in errors)
        reports = [
            {
                "camera_id": observations.camera_ids[index],
                **(
                    _metrics(values)
                    if values
                    else {"sample_count": 0, "rms_px": 0.0, "mean_px": 0.0, "p95_px": 0.0, "max_px": 0.0}
                ),
            }
            for index, values in enumerate(by_camera)
        ]
        return _metrics(all_errors), reports

    fit_metrics, fit_by_camera = score_frames(observations.fit_frames)
    held_metrics, held_by_camera = score_frames(observations.held_out_frames)

    declared_reference = camera_bundle.views[0].world_to_camera
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, camera_id in enumerate(observations.camera_ids):
        declared_matrix = camera_bundle.views[index].intrinsics_matrix
        focal_relative = float(
            max(
                abs(matrices[index][0, 0] - declared_matrix[0, 0]) / declared_matrix[0, 0],
                abs(matrices[index][1, 1] - declared_matrix[1, 1]) / declared_matrix[1, 1],
            )
        )
        principal_error = float(
            np.linalg.norm(matrices[index][:2, 2] - declared_matrix[:2, 2])
        )
        distortion_error = _distortion_observed_domain_error(
            distortions[index],
            camera_bundle.views[index].distortion,
            matrices[index],
            declared_matrix,
            np.concatenate(
                [frame.detections[index].image_points for frame in observations.fit_frames]
            ),
        )
        declared_relative = (
            camera_bundle.views[index].world_to_camera @ np.linalg.inv(declared_reference)
        )
        declared_relative = declared_relative.copy()
        declared_relative[:3, 3] *= camera_bundle.meters_per_world_unit
        rotation_error = float(
            np.degrees(
                Rotation.from_matrix(
                    rig[index][:3, :3] @ declared_relative[:3, :3].T
                ).magnitude()
            )
        )
        translation_error = float(
            np.linalg.norm(rig[index][:3, 3] - declared_relative[:3, 3])
        )
        translation_relative = float(
            translation_error / max(np.linalg.norm(declared_relative[:3, 3]), 0.01)
        )
        comparisons.append(
            {
                "camera_id": camera_id,
                "recomputed_K": matrices[index].tolist(),
                "recomputed_D": distortions[index].tolist(),
                "recomputed_reference_to_camera": rig[index].tolist(),
                "focal_relative_error": focal_relative,
                "principal_point_error_px": principal_error,
                "distortion_observed_domain_p95_error_px": distortion_error,
                "rotation_error_degrees": rotation_error,
                "translation_error_m": translation_error,
                "translation_relative_error": translation_relative,
            }
        )
        if focal_relative > FOCAL_RELATIVE_GATE:
            failures.append(f"{camera_id}:FOCAL_LENGTH_MISMATCH")
        if principal_error > PRINCIPAL_POINT_GATE_PX:
            failures.append(f"{camera_id}:PRINCIPAL_POINT_MISMATCH")
        if distortion_error > DISTORTION_OBSERVED_DOMAIN_P95_GATE_PX:
            failures.append(f"{camera_id}:DISTORTION_MISMATCH")
        if rotation_error > ROTATION_GATE_DEGREES:
            failures.append(f"{camera_id}:RIG_ROTATION_MISMATCH")
        if (
            translation_error > TRANSLATION_ABSOLUTE_GATE_M
            and translation_relative > TRANSLATION_RELATIVE_GATE
        ):
            failures.append(f"{camera_id}:RIG_TRANSLATION_MISMATCH")
    if fit_metrics["rms_px"] > FIT_RMS_GATE_PX:
        failures.append("FIT_REPROJECTION_RMS_FAILED")
    if held_metrics["rms_px"] > HELD_OUT_RMS_GATE_PX:
        failures.append("HELD_OUT_REPROJECTION_RMS_FAILED")

    thresholds = {
        "minimum_cameras": MIN_CAMERAS,
        "minimum_fit_frames": MIN_FIT_FRAMES,
        "minimum_held_out_frames": MIN_HELD_OUT_FRAMES,
        "maximum_fit_rms_px": FIT_RMS_GATE_PX,
        "maximum_held_out_rms_px": HELD_OUT_RMS_GATE_PX,
        "maximum_focal_relative_error": FOCAL_RELATIVE_GATE,
        "maximum_principal_point_error_px": PRINCIPAL_POINT_GATE_PX,
        "maximum_distortion_observed_domain_p95_error_px": (
            DISTORTION_OBSERVED_DOMAIN_P95_GATE_PX
        ),
        "maximum_rotation_error_degrees": ROTATION_GATE_DEGREES,
        "maximum_translation_relative_error": TRANSLATION_RELATIVE_GATE,
        "maximum_translation_absolute_error_m": TRANSLATION_ABSOLUTE_GATE_M,
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "capture_session_id": observations.capture_session_id,
        "bindings": {
            "camera_bundle_sha256": camera_bundle.source_sha256,
            "observations_sha256": observations.observations_sha256,
            "fit_frame_ids": [frame.frame_id for frame in observations.fit_frames],
            "held_out_frame_ids": [frame.frame_id for frame in observations.held_out_frames],
            "fit_source_artifact_sha256s": [
                frame.source_artifact_sha256 for frame in observations.fit_frames
            ],
            "held_out_source_artifact_sha256s": [
                frame.source_artifact_sha256 for frame in observations.held_out_frames
            ],
        },
        "solver": {
            **observations.solver,
            "solver_configuration_sha256": hashlib.sha256(
                _canonical(
                    {
                        "solver": observations.solver,
                        "target": observations.target.as_dict(),
                        "camera_ids": list(observations.camera_ids),
                        "image_sizes": [list(value) for value in observations.image_sizes],
                    }
                )
            ).hexdigest(),
        },
        "thresholds": thresholds,
        "intrinsic_fit": intrinsic_reports,
        "rig_comparison": comparisons,
        "fit_evidence": {"aggregate": fit_metrics, "per_camera": fit_by_camera},
        "held_out_evidence": {
            "pose_source": "reference_camera_held_out_observation_only",
            "scored_camera_ids": list(observations.camera_ids[1:]),
            "aggregate": held_metrics,
            "per_camera": held_by_camera,
        },
        "raw_calibration_recomputed": True,
        "calibration_ready_for_identity_fit": not failures,
        "asset_identity_validated": False,
        "production_validated": False,
        "failures": sorted(set(failures)),
        "qualification_scope": "i0.1_multicamera_calibration_only_no_identity_or_production_claim",
    }
    report["report_sha256"] = _payload_sha256(report, "report_sha256")
    return json.loads(_canonical(report))


def _run_solver_deterministically(
    observations: CalibrationObservations,
    camera_bundle: CalibratedCameraBundle,
) -> dict[str, Any]:
    # OpenCV can change the last few ulps when its native reductions use a
    # different worker schedule.  Serialize this evidence solve and bind v1 to
    # one OpenCV thread so a later verifier can replay it byte-for-byte.
    with _OPENCV_SOLVER_LOCK:
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            return _solve_calibration_evidence(
                observations, camera_bundle=camera_bundle
            )
        finally:
            cv2.setNumThreads(previous_threads)


def recompute_calibration_evidence(
    observations_source: (
        str | Path | bytes | bytearray | Mapping[str, Any] | CalibrationObservations
    ),
    *,
    camera_bundle: CalibratedCameraBundle,
) -> dict[str, Any]:
    """Recompute and internally validate calibration evidence."""

    observations = (
        observations_source
        if isinstance(observations_source, CalibrationObservations)
        else load_calibration_observations(
            observations_source, camera_bundle=camera_bundle
        )
    )
    report = _run_solver_deterministically(observations, camera_bundle)
    return _load_calibration_evidence_report(
        report,
        observations=observations,
        camera_bundle=camera_bundle,
        verify_replay=False,
    )


def _load_calibration_evidence_report(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    observations: CalibrationObservations | None = None,
    camera_bundle: CalibratedCameraBundle | None = None,
    verify_replay: bool,
) -> dict[str, Any]:
    """Strict report loader enforcing bindings and non-production claims."""

    root = _exact_object(
        _read_json(source),
        "report",
        (
            "schema_version",
            "capture_session_id",
            "bindings",
            "solver",
            "thresholds",
            "intrinsic_fit",
            "rig_comparison",
            "fit_evidence",
            "held_out_evidence",
            "raw_calibration_recomputed",
            "calibration_ready_for_identity_fit",
            "asset_identity_validated",
            "production_validated",
            "failures",
            "qualification_scope",
            "report_sha256",
        ),
    )
    if root["schema_version"] != REPORT_SCHEMA:
        raise AutoAnimError("INPUT_INVALID", "Unsupported calibration report schema")
    _identifier(root["capture_session_id"], "capture_session_id")
    bindings = _exact_object(
        root["bindings"],
        "bindings",
        (
            "camera_bundle_sha256",
            "observations_sha256",
            "fit_frame_ids",
            "held_out_frame_ids",
            "fit_source_artifact_sha256s",
            "held_out_source_artifact_sha256s",
        ),
    )
    bundle_sha = _sha(bindings["camera_bundle_sha256"], "bindings.camera_bundle_sha256")
    observation_sha = _sha(bindings["observations_sha256"], "bindings.observations_sha256")
    for field in ("fit_frame_ids", "held_out_frame_ids"):
        values = bindings[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not _ID_RE.fullmatch(value) for value in values)
            or len(set(values)) != len(values)
        ):
            raise AutoAnimError("INPUT_INVALID", f"bindings.{field} is invalid")
    for field in (
        "fit_source_artifact_sha256s",
        "held_out_source_artifact_sha256s",
    ):
        values = bindings[field]
        frame_field = "fit_frame_ids" if field.startswith("fit_") else "held_out_frame_ids"
        if not isinstance(values, list) or len(values) != len(bindings[frame_field]):
            raise AutoAnimError("INPUT_INVALID", f"bindings.{field} count is invalid")
        for index, value in enumerate(values):
            _sha(value, f"bindings.{field}.{index}")
    if camera_bundle is not None and bundle_sha != camera_bundle.source_sha256:
        raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Report camera-bundle binding mismatch")
    if observations is not None:
        if observation_sha != observations.observations_sha256:
            raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Report observations binding mismatch")
        expected_fit = [frame.frame_id for frame in observations.fit_frames]
        expected_held = [frame.frame_id for frame in observations.held_out_frames]
        if bindings["fit_frame_ids"] != expected_fit or bindings["held_out_frame_ids"] != expected_held:
            raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Report frame split binding mismatch")
        if bindings["fit_source_artifact_sha256s"] != [frame.source_artifact_sha256 for frame in observations.fit_frames] or bindings["held_out_source_artifact_sha256s"] != [frame.source_artifact_sha256 for frame in observations.held_out_frames]:
            raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Report source-artifact binding mismatch")
        if root["capture_session_id"] != observations.capture_session_id:
            raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Report capture-session binding mismatch")

    solver = _exact_object(
        root["solver"],
        "solver",
        (
            "opencv_version",
            "camera_model",
            "flags",
            "max_iterations",
            "epsilon",
            "opencv_threads",
            "calibration_graph",
            "reference_camera_id",
            "solver_configuration_sha256",
        ),
    )
    if (
        solver["opencv_version"] != cv2.__version__
        or solver["camera_model"] != CAMERA_MODEL
        or type(solver["flags"]) is not int
        or solver["flags"] != 0
        or type(solver["max_iterations"]) is not int
        or solver["max_iterations"] != 100
        or type(solver["epsilon"]) is not float
        or solver["epsilon"] != 1e-12
        or type(solver["opencv_threads"]) is not int
        or solver["opencv_threads"] != 1
        or solver["calibration_graph"] != CALIBRATION_GRAPH
    ):
        raise AutoAnimError("INPUT_INVALID", "Calibration report solver is not schema-v1")
    _identifier(solver["reference_camera_id"], "solver.reference_camera_id")
    _sha(solver["solver_configuration_sha256"], "solver.solver_configuration_sha256")
    if observations is not None:
        expected_configuration_hash = hashlib.sha256(
            _canonical(
                {
                    "solver": observations.solver,
                    "target": observations.target.as_dict(),
                    "camera_ids": list(observations.camera_ids),
                    "image_sizes": [list(value) for value in observations.image_sizes],
                }
            )
        ).hexdigest()
        if solver["solver_configuration_sha256"] != expected_configuration_hash:
            raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Solver configuration binding mismatch")

    expected_thresholds = {
        "minimum_cameras": MIN_CAMERAS,
        "minimum_fit_frames": MIN_FIT_FRAMES,
        "minimum_held_out_frames": MIN_HELD_OUT_FRAMES,
        "maximum_fit_rms_px": FIT_RMS_GATE_PX,
        "maximum_held_out_rms_px": HELD_OUT_RMS_GATE_PX,
        "maximum_focal_relative_error": FOCAL_RELATIVE_GATE,
        "maximum_principal_point_error_px": PRINCIPAL_POINT_GATE_PX,
        "maximum_distortion_observed_domain_p95_error_px": (
            DISTORTION_OBSERVED_DOMAIN_P95_GATE_PX
        ),
        "maximum_rotation_error_degrees": ROTATION_GATE_DEGREES,
        "maximum_translation_relative_error": TRANSLATION_RELATIVE_GATE,
        "maximum_translation_absolute_error_m": TRANSLATION_ABSOLUTE_GATE_M,
    }
    if root["thresholds"] != expected_thresholds:
        raise AutoAnimError("INPUT_INVALID", "Calibration report thresholds differ from schema-v1")

    intrinsic_fit = root["intrinsic_fit"]
    rig_comparison = root["rig_comparison"]
    if (
        not isinstance(intrinsic_fit, list)
        or not isinstance(rig_comparison, list)
        or len(intrinsic_fit) < MIN_CAMERAS
        or len(intrinsic_fit) != len(rig_comparison)
    ):
        raise AutoAnimError("INPUT_INVALID", "Calibration report camera evidence is incomplete")
    camera_ids: list[str] = []
    for index, value in enumerate(intrinsic_fit):
        item = _exact_object(
            value,
            f"intrinsic_fit.{index}",
            ("camera_id", "fit_frame_count", "intrinsic_rms_px", "per_fit_frame_rms_px"),
        )
        camera_id = _identifier(item["camera_id"], f"intrinsic_fit.{index}.camera_id")
        if item["fit_frame_count"] != len(bindings["fit_frame_ids"]):
            raise AutoAnimError("INPUT_INVALID", "Intrinsic fit-frame count is inconsistent")
        _nonnegative_float(item["intrinsic_rms_px"], f"intrinsic_fit.{index}.intrinsic_rms_px")
        per_frame = item["per_fit_frame_rms_px"]
        if not isinstance(per_frame, list) or len(per_frame) != item["fit_frame_count"]:
            raise AutoAnimError("INPUT_INVALID", "Per-frame intrinsic evidence is incomplete")
        for frame_index, value in enumerate(per_frame):
            _nonnegative_float(value, f"intrinsic_fit.{index}.per_fit_frame_rms_px.{frame_index}")
        camera_ids.append(camera_id)
    if len(set(camera_ids)) != len(camera_ids):
        raise AutoAnimError("INPUT_INVALID", "Calibration report camera IDs are duplicated")

    derived_failures: list[str] = []
    for index, value in enumerate(rig_comparison):
        item = _exact_object(
            value,
            f"rig_comparison.{index}",
            (
                "camera_id",
                "recomputed_K",
                "recomputed_D",
                "recomputed_reference_to_camera",
                "focal_relative_error",
                "principal_point_error_px",
                "distortion_observed_domain_p95_error_px",
                "rotation_error_degrees",
                "translation_error_m",
                "translation_relative_error",
            ),
        )
        if item["camera_id"] != camera_ids[index]:
            raise AutoAnimError("INPUT_INVALID", "Rig and intrinsic camera order differs")
        for field, shape in (
            ("recomputed_K", (3, 3)),
            ("recomputed_D", (5,)),
            ("recomputed_reference_to_camera", (4, 4)),
        ):
            _numeric_array(item[field], shape, field)
        focal = _nonnegative_float(item["focal_relative_error"], "focal_relative_error")
        principal = _nonnegative_float(item["principal_point_error_px"], "principal_point_error_px")
        distortion = _nonnegative_float(
            item["distortion_observed_domain_p95_error_px"],
            "distortion_observed_domain_p95_error_px",
        )
        rotation = _nonnegative_float(item["rotation_error_degrees"], "rotation_error_degrees")
        translation = _nonnegative_float(item["translation_error_m"], "translation_error_m")
        translation_relative = _nonnegative_float(
            item["translation_relative_error"], "translation_relative_error"
        )
        camera_id = camera_ids[index]
        if focal > FOCAL_RELATIVE_GATE:
            derived_failures.append(f"{camera_id}:FOCAL_LENGTH_MISMATCH")
        if principal > PRINCIPAL_POINT_GATE_PX:
            derived_failures.append(f"{camera_id}:PRINCIPAL_POINT_MISMATCH")
        if distortion > DISTORTION_OBSERVED_DOMAIN_P95_GATE_PX:
            derived_failures.append(f"{camera_id}:DISTORTION_MISMATCH")
        if rotation > ROTATION_GATE_DEGREES:
            derived_failures.append(f"{camera_id}:RIG_ROTATION_MISMATCH")
        if translation > TRANSLATION_ABSOLUTE_GATE_M and translation_relative > TRANSLATION_RELATIVE_GATE:
            derived_failures.append(f"{camera_id}:RIG_TRANSLATION_MISMATCH")

    def validate_metrics(value: object, field: str) -> dict[str, Any]:
        item = _exact_object(
            value, field, ("sample_count", "rms_px", "mean_px", "p95_px", "max_px")
        )
        if type(item["sample_count"]) is not int or item["sample_count"] < 0:
            raise AutoAnimError("INPUT_INVALID", f"{field}.sample_count is invalid")
        for name in ("rms_px", "mean_px", "p95_px", "max_px"):
            _nonnegative_float(item[name], f"{field}.{name}")
        return item

    fit_evidence = _exact_object(root["fit_evidence"], "fit_evidence", ("aggregate", "per_camera"))
    held_evidence = _exact_object(
        root["held_out_evidence"],
        "held_out_evidence",
        ("pose_source", "scored_camera_ids", "aggregate", "per_camera"),
    )
    if held_evidence["pose_source"] != "reference_camera_held_out_observation_only" or held_evidence["scored_camera_ids"] != camera_ids[1:]:
        raise AutoAnimError("INPUT_INVALID", "Held-out anchor/scored-camera declaration is invalid")
    fit_aggregate = validate_metrics(fit_evidence["aggregate"], "fit_evidence.aggregate")
    held_aggregate = validate_metrics(held_evidence["aggregate"], "held_out_evidence.aggregate")
    for field, evidence in (("fit_evidence", fit_evidence), ("held_out_evidence", held_evidence)):
        per_camera = evidence["per_camera"]
        if not isinstance(per_camera, list) or len(per_camera) != len(camera_ids):
            raise AutoAnimError("INPUT_INVALID", f"{field}.per_camera is incomplete")
        for index, value in enumerate(per_camera):
            item = _exact_object(
                value,
                f"{field}.per_camera.{index}",
                ("camera_id", "sample_count", "rms_px", "mean_px", "p95_px", "max_px"),
            )
            if item["camera_id"] != camera_ids[index]:
                raise AutoAnimError("INPUT_INVALID", f"{field} camera order differs")
            validate_metrics(
                {key: item[key] for key in ("sample_count", "rms_px", "mean_px", "p95_px", "max_px")},
                f"{field}.per_camera.{index}.metrics",
            )
    if fit_aggregate["rms_px"] > FIT_RMS_GATE_PX:
        derived_failures.append("FIT_REPROJECTION_RMS_FAILED")
    if held_aggregate["rms_px"] > HELD_OUT_RMS_GATE_PX:
        derived_failures.append("HELD_OUT_REPROJECTION_RMS_FAILED")
    failures = root["failures"]
    if (
        not isinstance(failures, list)
        or any(not isinstance(value, str) for value in failures)
        or failures != sorted(set(failures))
    ):
        raise AutoAnimError("INPUT_INVALID", "Calibration report failures must be sorted strings")
    if failures != sorted(set(derived_failures)):
        raise AutoAnimError("UNSUPPORTED_CLAIM", "Calibration report failures contradict evidence")
    if (
        root["raw_calibration_recomputed"] is not True
        or type(root["calibration_ready_for_identity_fit"]) is not bool
        or root["calibration_ready_for_identity_fit"] != (not failures)
        or root["asset_identity_validated"] is not False
        or root["production_validated"] is not False
        or root["qualification_scope"]
        != "i0.1_multicamera_calibration_only_no_identity_or_production_claim"
    ):
        raise AutoAnimError("UNSUPPORTED_CLAIM", "Calibration report contains an invalid claim")
    observed_hash = _sha(root["report_sha256"], "report_sha256")
    if observed_hash != _payload_sha256(root, "report_sha256"):
        raise AutoAnimError("CALIBRATION_EVIDENCE_FAILED", "Calibration report hash mismatch")
    # A canonical round-trip rejects nested non-finite values and returns a detached value.
    validated = json.loads(_canonical(root))
    if verify_replay:
        if (observations is None) != (camera_bundle is None):
            raise AutoAnimError(
                "INPUT_INVALID",
                "Authenticated calibration report replay requires both observations and camera bundle",
            )
        if observations is not None and camera_bundle is not None:
            replayed = _run_solver_deterministically(observations, camera_bundle)
            if _canonical(validated) != _canonical(replayed):
                raise AutoAnimError(
                    "CALIBRATION_EVIDENCE_FAILED",
                    "Calibration report derived evidence does not match deterministic source replay",
                )
    return validated


def load_calibration_evidence_report(
    source: str | Path | bytes | bytearray | Mapping[str, Any],
    *,
    observations: CalibrationObservations | None = None,
    camera_bundle: CalibratedCameraBundle | None = None,
) -> dict[str, Any]:
    """Strictly validate a report, replaying its sources when both are supplied.

    Loading without sources performs schema and internal-consistency validation
    only.  It must not be treated as authenticity evidence; callers making a
    readiness decision supply both retained observations and their bound bundle.
    """

    return _load_calibration_evidence_report(
        source,
        observations=observations,
        camera_bundle=camera_bundle,
        verify_replay=True,
    )


__all__ = [
    "CALIBRATION_GRAPH",
    "CAMERA_MODEL",
    "COORDINATE_CONVENTION",
    "OBSERVATIONS_SCHEMA",
    "REPORT_SCHEMA",
    "CalibrationObservations",
    "TargetDefinition",
    "build_calibration_observations",
    "load_calibration_evidence_report",
    "load_calibration_observations",
    "recompute_calibration_evidence",
]
