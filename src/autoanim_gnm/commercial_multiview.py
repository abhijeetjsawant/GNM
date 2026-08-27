"""Clean-room calibrated multiview body reconstruction.

The module owns geometry and temporal fitting only.  It consumes ordinary 2D
body observations from a replaceable detector and produces AutoAnim's detailed
55-joint :class:`BodyTrack`.  It deliberately does not import MAMMA, MammaNet,
or research-only body-model assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, product, permutations
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .acting import TICKS_PER_SECOND
from .body import (
    DETAILED_HUMANOID,
    BodyTrack,
    _quaternion_multiply,
    _rotate_vector,
)
from .body_projection import project_generated_foot_contacts


SCHEMA_VERSION = "autoanim.commercial-multiview/1.0"
OBSERVATION_SCHEMA_VERSION = "autoanim.apple-vision-body-observations/1.0"
PROVIDER_ID = "autoanim_cleanroom_multiview"

# Every pixel-denominated constant below is expressed at this detector width.
# A fixed pixel threshold is a *shrinking physical* threshold as the detector
# runs wider: 14 px is ~78 mm at the subject at 1280 but ~26 mm at 3840, which
# silently changes which observations survive. Callers pass a `pixel_scale`
# (detector width / this reference) so the gates stay invariant in millimetres.
# Thresholds already expressed in metres are deliberately left alone.
REFERENCE_DETECTOR_WIDTH_PX = 1280
# How many surplus person-components the graph associator will consider before
# giving up and deferring to the exhaustive search. Bounds the combinatorics.
MAXIMUM_SURPLUS_COMPONENTS = 3
# Ceiling on the exhaustive fallback's search. (subjects!)^cameras reaches
# 331,776 at four subjects on four cameras, which measures in hours for a single
# frame, so past this the frame is refused with its size named rather than
# silently stalling a capture.
MAXIMUM_EXHAUSTIVE_ASSOCIATION_CANDIDATES = 20_000
# Fewest directly-triangulated frames before a limb length is trusted.
MINIMUM_LIMB_SAMPLES = 10
JOINT_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "root",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}
# Limbs whose length is constant within a take, used to resolve joints that only
# one camera can see: a calibrated ray meets a sphere of known radius about the
# parent in at most two points -- limb toward the camera or away from it -- and
# temporal continuity picks between them.
# Head landmarks (nose, eyes, ears) are deliberately absent -- they have no rigid
# bone to a parent, and GNM owns the head regardless, so they stay on the
# existing neck-fallback path.
RIGID_LIMBS = (
    ("root", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("root", "left_hip"),
    ("root", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
)
CORE_ASSOCIATION_JOINTS = (
    "root",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
)


class CommercialMultiviewError(ValueError):
    """The commercial multiview input or reconstruction failed validation."""


@dataclass(frozen=True, slots=True)
class CalibratedCamera:
    name: str
    width: int
    height: int
    intrinsics: np.ndarray
    camera_center_world_m: np.ndarray
    camera_to_world_xyzw: np.ndarray

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        center = np.asarray(self.camera_center_world_m, dtype=np.float64)
        quaternion = np.asarray(self.camera_to_world_xyzw, dtype=np.float64)
        if (
            not self.name
            or type(self.width) is not int
            or type(self.height) is not int
            or self.width <= 0
            or self.height <= 0
            or intrinsics.shape != (3, 3)
            or center.shape != (3,)
            or quaternion.shape != (4,)
            or not np.isfinite(intrinsics).all()
            or not np.isfinite(center).all()
            or not np.isfinite(quaternion).all()
            or intrinsics[0, 0] <= 0.0
            or intrinsics[1, 1] <= 0.0
            or abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-5
        ):
            raise CommercialMultiviewError(f"Invalid calibrated camera: {self.name!r}")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_center_world_m", center)
        object.__setattr__(self, "camera_to_world_xyzw", quaternion)

    @property
    def projection_matrix(self) -> np.ndarray:
        camera_to_world = Rotation.from_quat(self.camera_to_world_xyzw).as_matrix()
        world_to_camera = camera_to_world.T
        extrinsics = np.column_stack(
            (world_to_camera, -world_to_camera @ self.camera_center_world_m)
        )
        return self.intrinsics @ extrinsics

    def scaled(self, width: int, height: int) -> CalibratedCamera:
        if width <= 0 or height <= 0:
            raise CommercialMultiviewError("Scaled camera dimensions must be positive")
        scale_x = width / self.width
        scale_y = height / self.height
        intrinsics = self.intrinsics.copy()
        intrinsics[0] *= scale_x
        intrinsics[1] *= scale_y
        return CalibratedCamera(
            name=self.name,
            width=width,
            height=height,
            intrinsics=intrinsics,
            camera_center_world_m=self.camera_center_world_m,
            camera_to_world_xyzw=self.camera_to_world_xyzw,
        )

    def project(self, points_world_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world_m, dtype=np.float64)
        single = points.ndim == 1
        if single:
            points = points[None]
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise CommercialMultiviewError("Projection points must be finite [N,3]")
        homogeneous = np.column_stack((points, np.ones(len(points))))
        projected = homogeneous @ self.projection_matrix.T
        depth = projected[:, 2]
        uv = projected[:, :2] / depth[:, None]
        return (uv[0], depth[0]) if single else (uv, depth)


@dataclass(frozen=True, slots=True)
class TriangulatedPoint:
    position_world_m: np.ndarray
    used_camera_indices: tuple[int, ...]
    reprojection_errors_px: np.ndarray
    confidence: float


@dataclass(frozen=True, slots=True)
class ReconstructionDiagnostics:
    frame_count: int
    subject_count: int
    camera_count: int
    valid_joint_fraction: float
    constraint_recovered_joint_fraction: float
    frames_deferred_to_exhaustive_association: int
    median_reprojection_error_px: float
    p95_reprojection_error_px: float
    maximum_reprojection_error_px: float
    association_objective_median: float
    interpolated_joint_fraction: float
    temporally_rejected_subject_frames: int
    contact_frames: tuple[tuple[int, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "frame_count": self.frame_count,
            "subject_count": self.subject_count,
            "camera_count": self.camera_count,
            "valid_joint_fraction": self.valid_joint_fraction,
            # A slot recovered from a single ray plus limb-length and temporal
            # constraints is evidence-based, unlike an interpolated one. It is
            # reported separately so `valid_joint_fraction` keeps meaning
            # exactly what it has always meant: per-frame triangulation
            # succeeded from two or more views.
            "constraint_recovered_joint_fraction": self.constraint_recovered_joint_fraction,
            "frames_deferred_to_exhaustive_association": self.frames_deferred_to_exhaustive_association,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "p95_reprojection_error_px": self.p95_reprojection_error_px,
            "maximum_reprojection_error_px": self.maximum_reprojection_error_px,
            "association_objective_median": self.association_objective_median,
            "interpolated_joint_fraction": self.interpolated_joint_fraction,
            "temporally_rejected_subject_frames": self.temporally_rejected_subject_frames,
            "contact_frames": [list(value) for value in self.contact_frames],
        }


def camera_from_dict(name: str, value: Any) -> CalibratedCamera:
    if not isinstance(value, dict):
        raise CommercialMultiviewError("Camera calibration must be an object")
    expected = {
        "resolution",
        "intrinsics",
        "camera_center_world_m",
        "camera_to_world_quaternion_wxyz",
    }
    if set(value) != expected:
        raise CommercialMultiviewError(f"Camera {name} calibration fields differ")
    resolution = value["resolution"]
    intrinsics = value["intrinsics"]
    quaternion_wxyz = np.asarray(value["camera_to_world_quaternion_wxyz"], dtype=np.float64)
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(item) is not int for item in resolution)
        or not isinstance(intrinsics, list)
        or len(intrinsics) != 4
        or quaternion_wxyz.shape != (4,)
    ):
        raise CommercialMultiviewError(f"Camera {name} calibration values differ")
    fx, fy, cx, cy = (float(item) for item in intrinsics)
    return CalibratedCamera(
        name=name,
        width=resolution[0],
        height=resolution[1],
        intrinsics=np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0))),
        camera_center_world_m=np.asarray(value["camera_center_world_m"], dtype=np.float64),
        camera_to_world_xyzw=quaternion_wxyz[[1, 2, 3, 0]],
    )


def load_camera_rig(path: str | Path) -> tuple[CalibratedCamera, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "cameras"}
        or value["schema_version"] != "autoanim.calibrated-camera-rig/1.0"
        or not isinstance(value["cameras"], dict)
        or len(value["cameras"]) < 2
    ):
        raise CommercialMultiviewError("Calibrated camera rig schema is invalid")
    return tuple(camera_from_dict(name, camera) for name, camera in value["cameras"].items())


def triangulate_point(
    cameras: Sequence[CalibratedCamera],
    points_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    minimum_confidence: float = 0.25,
    inlier_threshold_px: float = 14.0,
    pixel_scale: float = 1.0,
) -> TriangulatedPoint | None:
    # `inlier_threshold_px` is quoted at REFERENCE_DETECTOR_WIDTH_PX; scaling it
    # keeps the gate a constant physical size at the subject across widths.
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise CommercialMultiviewError("Pixel scale must be finite and positive")
    scaled_inlier_threshold_px = inlier_threshold_px * pixel_scale
    points = np.asarray(points_xy, dtype=np.float64)
    weights = np.asarray(confidence, dtype=np.float64)
    if points.shape != (len(cameras), 2) or weights.shape != (len(cameras),):
        raise CommercialMultiviewError("Triangulation observations differ from camera count")
    eligible = np.flatnonzero(
        np.isfinite(points).all(axis=1)
        & np.isfinite(weights)
        & (weights >= minimum_confidence)
    )
    if len(eligible) < 2:
        return None

    def solve(indices: Sequence[int]) -> np.ndarray | None:
        rows: list[np.ndarray] = []
        for index in indices:
            projection = cameras[index].projection_matrix
            u, v = points[index]
            weight = math.sqrt(max(float(weights[index]), 1e-6))
            rows.extend((weight * (u * projection[2] - projection[0]), weight * (v * projection[2] - projection[1])))
        _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
        homogeneous = vh[-1]
        if abs(float(homogeneous[3])) <= 1e-10:
            return None
        output = homogeneous[:3] / homogeneous[3]
        return output if np.isfinite(output).all() else None

    best: tuple[int, float, np.ndarray, np.ndarray, tuple[int, ...]] | None = None
    candidates: list[tuple[int, ...]] = []
    for size in range(2, len(eligible) + 1):
        candidates.extend(combinations(eligible.tolist(), size))
    for subset in candidates:
        position = solve(subset)
        if position is None:
            continue
        errors = np.full(len(cameras), np.nan, dtype=np.float64)
        depths = np.full(len(cameras), np.nan, dtype=np.float64)
        for index in eligible:
            projected, depth = cameras[index].project(position)
            errors[index] = float(np.linalg.norm(projected - points[index]))
            depths[index] = float(depth)
        inliers = tuple(
            int(index)
            for index in eligible
            if depths[index] > 0.0 and errors[index] <= scaled_inlier_threshold_px
        )
        if len(inliers) < 2:
            continue
        refined = solve(inliers)
        if refined is None:
            continue
        refined_errors = np.full(len(cameras), np.nan, dtype=np.float64)
        for index in eligible:
            projected, depth = cameras[index].project(refined)
            if depth > 0.0:
                refined_errors[index] = float(np.linalg.norm(projected - points[index]))
        median = float(np.nanmedian(refined_errors[list(inliers)]))
        score = (len(inliers), -median)
        if best is None or score > (best[0], -best[1]):
            best = (len(inliers), median, refined, refined_errors, inliers)
    if best is None:
        return None
    _, _, position, errors, inliers = best
    return TriangulatedPoint(
        position_world_m=position,
        used_camera_indices=inliers,
        reprojection_errors_px=errors,
        confidence=float(np.mean(weights[list(inliers)])),
    )


def _person_array(person: Any) -> np.ndarray:
    output = np.full((len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
    if not isinstance(person, dict) or not isinstance(person.get("joints"), dict):
        return output
    for name, sample in person["joints"].items():
        if name not in JOINT_INDEX or not isinstance(sample, dict):
            continue
        try:
            value = np.asarray(
                (sample["x"], sample["y"], sample["confidence"]), dtype=np.float64
            )
        except (KeyError, TypeError, ValueError):
            continue
        if value.shape == (3,) and np.isfinite(value).all():
            output[JOINT_INDEX[name]] = value
    return output


def load_observation_jsonl(path: str | Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != OBSERVATION_SCHEMA_VERSION
            or not isinstance(value.get("frame_index"), int)
            or not isinstance(value.get("width"), int)
            or not isinstance(value.get("height"), int)
            or not isinstance(value.get("people"), list)
        ):
            raise CommercialMultiviewError("Apple Vision observation schema is invalid")
        frames.append(value)
    if len(frames) < 2 or any(
        frames[index]["frame_index"] >= frames[index + 1]["frame_index"]
        for index in range(len(frames) - 1)
    ):
        raise CommercialMultiviewError("Observation frames must increase strictly")
    return frames


def _assignment_options(detection_count: int, subject_count: int) -> list[tuple[int | None, ...]]:
    if detection_count >= subject_count:
        return [tuple(value) for value in permutations(range(detection_count), subject_count)]
    values: list[int | None] = list(range(detection_count)) + [None] * (subject_count - detection_count)
    return sorted(set(permutations(values, subject_count)), key=lambda row: tuple(-1 if item is None else item for item in row))


def _score_assignment(
    cameras: Sequence[CalibratedCamera],
    associated: np.ndarray,
    *,
    subject_count: int,
    previous_roots_world_m: np.ndarray | None,
    previous_positions_world_m: np.ndarray | None,
    previous_observations_xyc: np.ndarray | None,
    pixel_scale: float,
) -> float | None:
    """Score one cross-view assignment.

    Shared by every association strategy so that the objective, and the
    reported ``association_objective_median``, mean the same thing however
    the candidate assignment was arrived at. Returns ``None`` when the
    assignment cannot be scored, e.g. a subject with no triangulated root.
    """

    errors: list[float] = []
    support_deficits: list[int] = []
    roots: list[np.ndarray] = []
    candidate_positions = np.full(
        (subject_count, len(JOINT_NAMES), 3), np.nan, dtype=np.float64
    )
    valid = True
    for subject_index in range(subject_count):
        subject_root: np.ndarray | None = None
        for name in CORE_ASSOCIATION_JOINTS:
            joint = JOINT_INDEX[name]
            sample = triangulate_point(
                cameras,
                associated[subject_index, :, joint, :2],
                associated[subject_index, :, joint, 2],
                # Association deliberately tolerates a wider residual
                # than final reconstruction. It decides identity in the
                # presence of detector jitter/partial occlusion; the final
                # 14 px robust solve still rejects the bad view.
                inlier_threshold_px=40.0,
                pixel_scale=pixel_scale,
            )
            if sample is None:
                continue
            errors.extend(sample.reprojection_errors_px[list(sample.used_camera_indices)])
            support_deficits.append(len(cameras) - len(sample.used_camera_indices))
            candidate_positions[subject_index, joint] = sample.position_world_m
            if name == "root":
                subject_root = sample.position_world_m
        if subject_root is None:
            valid = False
            break
        roots.append(subject_root)
    if not valid or not errors:
        return None
    cost = float(np.median(errors))
    # A two-camera accidental intersection can have a smaller residual
    # than the correct four-camera person. Reward calibrated consensus,
    # not residual alone, or robust triangulation itself creates ghosts.
    # support_deficits is a camera count (unitless); the weight carries the
    # pixel unit, so it scales with the detector width.
    cost += 12.0 * pixel_scale * float(np.mean(support_deficits))
    if previous_roots_world_m is not None:
        roots_array = np.asarray(roots)
        previous_roots = np.asarray(previous_roots_world_m, dtype=np.float64)
        if roots_array.shape == previous_roots.shape:
            # `reconstruct_multiview` seeds the prior roots all-NaN and fills a
            # row only once that subject is accepted, so a subject missed on the
            # first frame leaves a permanent NaN row. Without this mask its NaN
            # poisons every candidate's cost, `best` stays None, and the take
            # aborts as "no valid assignment". Score against the rows we
            # actually have. Root displacement is in metres (capped at 2.0 m);
            # the weight carries the pixel unit and therefore scales.
            usable = (
                np.isfinite(roots_array).all(axis=1)
                & np.isfinite(previous_roots).all(axis=1)
            )
            if usable.any():
                displacement = np.linalg.norm(
                    roots_array[usable] - previous_roots[usable], axis=1
                )
                cost += 50.0 * pixel_scale * float(np.mean(np.minimum(displacement, 2.0)))
    if previous_positions_world_m is not None:
        previous = np.asarray(previous_positions_world_m, dtype=np.float64)
        if previous.shape == candidate_positions.shape:
            continuity: list[float] = []
            for subject_index in range(subject_count):
                for name in CORE_ASSOCIATION_JOINTS:
                    joint = JOINT_INDEX[name]
                    if np.isfinite(candidate_positions[subject_index, joint]).all() and np.isfinite(previous[subject_index, joint]).all():
                        continuity.append(
                            min(
                                float(
                                    np.linalg.norm(
                                        candidate_positions[subject_index, joint]
                                        - previous[subject_index, joint]
                                    )
                                ),
                                1.0,
                            )
                        )
            if continuity:
                # continuity is in metres (capped at 1.0 m); weight scales.
                cost += 60.0 * pixel_scale * float(np.mean(continuity))
    if previous_observations_xyc is not None:
        previous_2d = np.asarray(previous_observations_xyc, dtype=np.float64)
        if previous_2d.shape == associated.shape:
            screen_steps: list[float] = []
            for subject_index in range(subject_count):
                for camera_index in range(len(cameras)):
                    for name in CORE_ASSOCIATION_JOINTS:
                        joint = JOINT_INDEX[name]
                        current = associated[subject_index, camera_index, joint]
                        prior = previous_2d[subject_index, camera_index, joint]
                        if (
                            np.isfinite(current).all()
                            and np.isfinite(prior).all()
                            and current[2] >= 0.25
                            and prior[2] >= 0.25
                        ):
                            screen_steps.append(
                                min(
                                    float(np.linalg.norm(current[:2] - prior[:2])),
                                    250.0 * pixel_scale,
                                )
                            )
            if screen_steps:
                # Reprojection measures cross-camera consistency within a
                # frame. This term supplies the missing across-frame
                # identity evidence and rejects a low-reprojection ghost.
                # screen_steps are pixels and already scale with width, so
                # this weight is intentionally left unscaled.
                cost += 0.25 * float(np.mean(screen_steps))
    return cost


def frame_is_ambiguous(detections: Sequence[Sequence[np.ndarray]], subject_count: int) -> bool:
    """A view reporting more people than the shot contains means identity is
    genuinely ambiguous, and the graph associator defers such frames to the
    exhaustive search. One definition, so the diagnostic counter cannot drift
    away from the behaviour it counts."""

    return any(len(people) > subject_count for people in detections)


def _fundamental_matrix(source: CalibratedCamera, target: CalibratedCamera) -> np.ndarray:
    """Fundamental matrix F with ``x_target^T F x_source == 0`` on a true match."""

    source_projection = source.projection_matrix
    target_projection = target.projection_matrix
    epipole = target_projection @ np.append(source.camera_center_world_m, 1.0)
    skew = np.asarray(
        (
            (0.0, -epipole[2], epipole[1]),
            (epipole[2], 0.0, -epipole[0]),
            (-epipole[1], epipole[0], 0.0),
        )
    )
    return skew @ target_projection @ np.linalg.pinv(source_projection)


def _epipolar_distance_px(
    fundamental: np.ndarray,
    source_person: np.ndarray,
    target_person: np.ndarray,
    *,
    minimum_confidence: float,
    minimum_shared_joints: int,
) -> float:
    """Median symmetric epipolar distance over joints both cameras can see.

    Median rather than mean: a single mislabelled limb should not decide
    identity. Returns ``inf`` when the two detections share too little evidence.
    """

    source = np.asarray(source_person, dtype=np.float64)
    target = np.asarray(target_person, dtype=np.float64)
    shared = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & (source[:, 2] >= minimum_confidence)
        & (target[:, 2] >= minimum_confidence)
    )
    if int(np.count_nonzero(shared)) < minimum_shared_joints:
        return math.inf
    ones = np.ones((int(np.count_nonzero(shared)), 1))
    source_h = np.hstack((source[shared, :2], ones))
    target_h = np.hstack((target[shared, :2], ones))
    target_lines = source_h @ fundamental.T
    source_lines = target_h @ fundamental
    numerator = np.abs(np.sum(target_h * target_lines, axis=1))
    target_norm = np.hypot(target_lines[:, 0], target_lines[:, 1])
    source_norm = np.hypot(source_lines[:, 0], source_lines[:, 1])
    if not (np.all(target_norm > 1e-9) and np.all(source_norm > 1e-9)):
        return math.inf
    return float(np.median(numerator / target_norm + numerator / source_norm))


def associate_frame_graph(
    cameras: Sequence[CalibratedCamera],
    detections: Sequence[Sequence[np.ndarray]],
    *,
    subject_count: int,
    previous_roots_world_m: np.ndarray | None = None,
    previous_positions_world_m: np.ndarray | None = None,
    previous_observations_xyc: np.ndarray | None = None,
    pixel_scale: float = 1.0,
    maximum_epipolar_px: float = 60.0,
    minimum_shared_joints: int = 4,
    minimum_confidence: float = 0.25,
    ambiguity_ratio: float = 0.7,
    ambiguity_margin_px: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Associate detections across views by cycle-consistent graph matching.

    Replaces an exhaustive search over every per-camera assignment, whose cost
    is the product of per-camera options and which triangulates every core joint
    of every candidate. This instead scores each cross-view detection pair once
    by symmetric epipolar distance, matches each camera pair with the Hungarian
    algorithm, and grows people as connected components under the constraint
    that a person may hold at most one detection per camera -- which is what
    enforces cycle consistency, since a triangle violation would otherwise
    require two detections from the same camera in one component.

    Falls back to :func:`associate_frame` when the frame is ambiguous -- any view
    reporting more people than the shot contains, which is 21% of frames on the
    reference fixture -- or when the resulting assignment cannot be scored. It is a heuristic, not an optimiser:
    it reproduces the exhaustive search's answer on every frame of the reference
    fixture, but it does not *guarantee* the global optimum of the objective.
    """

    if len(cameras) != len(detections) or subject_count < 1:
        raise CommercialMultiviewError("Association inputs are invalid")
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise CommercialMultiviewError("Pixel scale must be finite and positive")

    def exhaustive() -> tuple[np.ndarray, float]:
        # Deferring hands the frame to a (subjects!)^cameras search that
        # triangulates every core joint of every candidate. That is fine at two
        # subjects and four cameras (16 candidates) and catastrophic at four
        # subjects (331,776, measured at hours per frame). Fail loudly with the
        # frame's size rather than stall silently inside a long capture.
        candidates = 1
        for people in detections:
            candidates *= max(len(_assignment_options(len(people), subject_count)), 1)
        if candidates > MAXIMUM_EXHAUSTIVE_ASSOCIATION_CANDIDATES:
            raise CommercialMultiviewError(
                f"Cross-view association would need {candidates} candidates for "
                f"{subject_count} subjects across {len(cameras)} cameras, past the "
                f"{MAXIMUM_EXHAUSTIVE_ASSOCIATION_CANDIDATES} budget"
            )
        return associate_frame(
            cameras,
            detections,
            subject_count=subject_count,
            previous_roots_world_m=previous_roots_world_m,
            previous_positions_world_m=previous_positions_world_m,
            previous_observations_xyc=previous_observations_xyc,
            pixel_scale=pixel_scale,
        )

    if frame_is_ambiguous(detections, subject_count):
        # A view reporting more people than the shot contains is where identity
        # is genuinely ambiguous, and measurement showed the graph heuristic
        # picks a different, higher-cost assignment there. Pay for the exact
        # search on those frames rather than guess.
        return exhaustive()

    nodes = [
        (camera_index, person_index)
        for camera_index, people in enumerate(detections)
        for person_index in range(len(people))
    ]
    if len(nodes) < 2:
        return exhaustive()
    node_index = {node: index for index, node in enumerate(nodes)}
    threshold = maximum_epipolar_px * pixel_scale

    edges: list[tuple[float, int, int]] = []
    for left in range(len(cameras)):
        for right in range(left + 1, len(cameras)):
            if not detections[left] or not detections[right]:
                continue
            fundamental = _fundamental_matrix(cameras[left], cameras[right])
            distances = np.full((len(detections[left]), len(detections[right])), math.inf)
            for a, source_person in enumerate(detections[left]):
                for b, target_person in enumerate(detections[right]):
                    distance = _epipolar_distance_px(
                        fundamental,
                        source_person,
                        target_person,
                        minimum_confidence=minimum_confidence,
                        minimum_shared_joints=minimum_shared_joints,
                    )
                    distances[a, b] = distance
            # Non-assignment must stay available: an occluded person has no
            # counterpart, and forcing one is how ghost identities appear.
            feasible = np.isfinite(distances) & (distances <= threshold)
            if not feasible.any():
                continue
            cost = np.where(feasible, distances, threshold * 1e3)
            rows, columns = linear_sum_assignment(cost)
            for a, b in zip(rows.tolist(), columns.tolist(), strict=True):
                if not feasible[a, b]:
                    continue
                # A match is only evidence if something actually decided it.
                # Where a camera pair's baseline is near-collinear with the
                # people it sees, every entry collapses into one narrow band,
                # the Hungarian pick is arbitrary, and single linkage then
                # freezes that arbitrary pick into a component no later stage
                # can re-partition -- producing a person assembled from two
                # performers. Require the pick to beat its best alternative in
                # both its row and its column, by a ratio and by an absolute
                # margin, so a pair carrying no identity information
                # contributes nothing rather than a confident mistake.
                best = float(distances[a, b])
                alternative = min(
                    float(np.min(np.delete(distances[a], b))) if distances.shape[1] > 1 else math.inf,
                    float(np.min(np.delete(distances[:, b], a))) if distances.shape[0] > 1 else math.inf,
                )
                if alternative < max(
                    best / ambiguity_ratio, best + ambiguity_margin_px * pixel_scale
                ):
                    continue
                edges.append((best, node_index[(left, a)], node_index[(right, b)]))

    parent = list(range(len(nodes)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    cameras_used: list[set[int]] = [{node[0]} for node in nodes]
    # Strongest evidence first, and never merge two detections from one camera:
    # that single rule is what makes the components cycle-consistent, since a
    # triangle violation would otherwise need two detections from one camera in
    # one component. Complete linkage over `pair_distance` was measured as an
    # alternative and was strictly worse -- it fragments components and changes
    # the chosen assignment on more frames, not fewer.
    for _, left_node, right_node in sorted(edges, key=lambda edge: edge[0]):
        left_root, right_root = find(left_node), find(right_node)
        if left_root == right_root or cameras_used[left_root] & cameras_used[right_root]:
            continue
        parent[right_root] = left_root
        cameras_used[left_root] |= cameras_used[right_root]

    components: dict[int, list[int]] = {}
    for index in range(len(nodes)):
        components.setdefault(find(index), []).append(index)
    # A spurious extra detection in one view produces an extra component, and
    # picking the largest ones then quietly picks the wrong person. Score the
    # plausible choices instead -- still a handful of evaluations against the
    # exhaustive path's product over per-camera assignments.
    candidates = sorted(components.values(), key=len, reverse=True)[: subject_count + MAXIMUM_SURPLUS_COMPONENTS]
    if len(candidates) < subject_count:
        return exhaustive()

    best_cost = math.inf
    best: np.ndarray | None = None
    for chosen in combinations(range(len(candidates)), subject_count):
        associated = np.full(
            (subject_count, len(cameras), len(JOINT_NAMES), 3), np.nan, dtype=np.float64
        )
        roots: list[np.ndarray] = []
        complete = True
        for slot, candidate_index in enumerate(chosen):
            for index in candidates[candidate_index]:
                camera_index, person_index = nodes[index]
                associated[slot, camera_index] = detections[camera_index][person_index]
            sample = triangulate_point(
                cameras,
                associated[slot, :, JOINT_INDEX["root"], :2],
                associated[slot, :, JOINT_INDEX["root"], 2],
                inlier_threshold_px=40.0,
                pixel_scale=pixel_scale,
            )
            if sample is None:
                complete = False
                break
            roots.append(sample.position_world_m)
        if not complete:
            continue
        associated = associated[_identity_order(np.asarray(roots), previous_roots_world_m)]
        cost = _score_assignment(
            cameras,
            associated,
            subject_count=subject_count,
            previous_roots_world_m=previous_roots_world_m,
            previous_positions_world_m=previous_positions_world_m,
            previous_observations_xyc=previous_observations_xyc,
            pixel_scale=pixel_scale,
        )
        if cost is not None and cost < best_cost:
            best_cost = cost
            best = associated
    if best is None:
        return exhaustive()
    return best, best_cost


def _identity_order(roots: np.ndarray, previous_roots_world_m: np.ndarray | None) -> np.ndarray:
    """Map components onto stable subject slots.

    With no history, order by capture world X -- the same deterministic contract
    the exhaustive path uses, which everything downstream relies on. With
    history, match to the previous frame's roots. This is a soft prior applied
    once per frame, not a hard constraint: the bounded-acceleration gate in
    :func:`reconstruct_multiview` remains the safety net, so a single bad frame
    cannot pin identities forever.
    """

    if previous_roots_world_m is None or np.asarray(previous_roots_world_m).shape != roots.shape:
        return np.argsort(roots[:, 0])
    previous = np.asarray(previous_roots_world_m, dtype=np.float64)
    distances = np.linalg.norm(roots[:, None, :] - previous[None, :, :], axis=2)
    if not np.isfinite(distances).all():
        return np.argsort(roots[:, 0])
    rows, columns = linear_sum_assignment(distances)
    order = np.empty(len(roots), dtype=np.int64)
    order[columns] = rows
    return order


def associate_frame(
    cameras: Sequence[CalibratedCamera],
    detections: Sequence[Sequence[np.ndarray]],
    *,
    subject_count: int,
    previous_roots_world_m: np.ndarray | None = None,
    previous_positions_world_m: np.ndarray | None = None,
    previous_observations_xyc: np.ndarray | None = None,
    pixel_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    if len(cameras) != len(detections) or subject_count < 1:
        raise CommercialMultiviewError("Association inputs are invalid")
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise CommercialMultiviewError("Pixel scale must be finite and positive")
    anchor = max(range(len(cameras)), key=lambda index: len(detections[index]))
    if len(detections[anchor]) < subject_count:
        raise CommercialMultiviewError("No camera observes every requested subject")
    options: list[list[tuple[int | None, ...]]] = []
    for people in detections:
        # Detector result ordering is not an identity signal. Include anchor
        # permutations as well; otherwise an Apple Vision list-order change
        # creates a geometrically valid but semantically swapped character.
        options.append(_assignment_options(len(people), subject_count))
    best_cost = math.inf
    best: np.ndarray | None = None
    for assignment_rows in product(*options):
        associated = np.full(
            (subject_count, len(cameras), len(JOINT_NAMES), 3), np.nan, dtype=np.float64
        )
        for camera_index, assignment in enumerate(assignment_rows):
            for subject_index, person_index in enumerate(assignment):
                if person_index is not None:
                    associated[subject_index, camera_index] = detections[camera_index][person_index]
        cost = _score_assignment(
            cameras,
            associated,
            subject_count=subject_count,
            previous_roots_world_m=previous_roots_world_m,
            previous_positions_world_m=previous_positions_world_m,
            previous_observations_xyc=previous_observations_xyc,
            pixel_scale=pixel_scale,
        )
        if cost is None:
            continue
        if cost < best_cost:
            best_cost = cost
            best = associated
    if best is None:
        raise CommercialMultiviewError("Cross-view subject association found no valid solution")
    if previous_roots_world_m is None:
        # Stable initial identity independent of detector enumeration. Capture
        # world X is a deterministic ordering; temporal matching owns every
        # following frame.
        initial_roots: list[np.ndarray] = []
        for subject_index in range(subject_count):
            joint = JOINT_INDEX["root"]
            sample = triangulate_point(
                cameras,
                best[subject_index, :, joint, :2],
                best[subject_index, :, joint, 2],
                inlier_threshold_px=40.0,
                pixel_scale=pixel_scale,
            )
            if sample is None:
                raise CommercialMultiviewError("Initial subject root is not triangulated")
            initial_roots.append(sample.position_world_m)
        order = np.argsort(np.asarray(initial_roots)[:, 0])
        best = best[order]
    return best, best_cost


def estimate_limb_lengths_m(world: np.ndarray) -> dict[tuple[str, str], float]:
    """Median limb length over the frames where both endpoints triangulated.

    Median rather than mean, and only over directly-triangulated frames: lengths
    co-estimated against a joint that is 19% missing would drift toward whatever
    makes the residual small. One set per subject per take, which is what
    "per-shot bone lengths" means here.
    """

    positions = np.asarray(world, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise CommercialMultiviewError("Limb estimation needs [frame,joint,3] positions")
    lengths: dict[tuple[str, str], float] = {}
    for parent, child in RIGID_LIMBS:
        a = positions[:, JOINT_INDEX[parent]]
        b = positions[:, JOINT_INDEX[child]]
        usable = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if int(np.count_nonzero(usable)) < MINIMUM_LIMB_SAMPLES:
            continue
        measured = np.linalg.norm(a[usable] - b[usable], axis=1)
        lengths[(parent, child)] = float(np.median(measured))
    return lengths


def _pixels_per_metre(cameras: Sequence[CalibratedCamera], world: np.ndarray) -> float:
    """Scale that makes metre-denominated residuals commensurate with pixels."""

    points = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise CommercialMultiviewError("Cannot scale residuals without any position")
    values: list[float] = []
    for camera in cameras:
        rotation = Rotation.from_quat(camera.camera_to_world_xyzw).as_matrix()
        depth = (points - camera.camera_center_world_m) @ rotation[:, 2]
        depth = depth[depth > 1e-6]
        if len(depth):
            focal = 0.5 * float(camera.intrinsics[0, 0] + camera.intrinsics[1, 1])
            values.append(focal / float(np.median(depth)))
    if not values:
        raise CommercialMultiviewError("Cannot scale residuals without a valid depth")
    return float(np.mean(values))


def solve_sequence_positions(
    cameras: Sequence[CalibratedCamera],
    world: np.ndarray,
    observations_xyc: np.ndarray,
    *,
    pixel_scale: float = 1.0,
    smooth_weight: float = 2.0,
    length_weight: float = 2.0,
    minimum_confidence: float = 0.25,
    robust_scale_px: float = 14.0,
    maximum_evaluations: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a whole take at once, and report which slots that recovered.

    Per-frame triangulation needs two views. On the reference fixture 83% of the
    joints it cannot resolve were seen by exactly one camera -- a single ray,
    underdetermined alone. A ray plus a known distance from an already-solved
    parent leaves a circle, and temporal continuity picks the point on it, so a
    sequence-level solve recovers them from evidence rather than filling them by
    interpolation.

    Returns ``(positions, recovered)``: positions for every slot, and a boolean
    mask marking slots that had no two-view triangulation but were resolved from
    at least one ray. Slots with no observation at all stay NaN for the caller's
    existing fill to handle.
    """

    positions = np.asarray(world, dtype=np.float64)
    observations = np.asarray(observations_xyc, dtype=np.float64)
    frames, joints = positions.shape[0], positions.shape[1]
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise CommercialMultiviewError("Sequence solve needs [frame,joint,3] positions")
    if observations.shape != (frames, len(cameras), joints, 3):
        raise CommercialMultiviewError("Sequence solve needs [frame,camera,joint,3] observations")
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise CommercialMultiviewError("Pixel scale must be finite and positive")

    direct = np.isfinite(positions).all(axis=2)
    seen = (
        np.isfinite(observations[..., :2]).all(axis=3)
        & (observations[..., 2] >= minimum_confidence)
    ).transpose(0, 2, 1)
    support = seen.sum(axis=2)
    # Only slots with an actual ray are candidates. No ray means no evidence,
    # and inventing one here would be interpolation wearing a solver's clothes.
    candidate = (~direct) & (support >= 1)
    if not candidate.any():
        return positions.copy(), np.zeros_like(direct)

    start = _fill_and_smooth_positions(positions)[0]
    step = np.linalg.norm(np.diff(start, axis=0), axis=2)
    motion_scale = float(np.mean(step[np.isfinite(step)])) if step.size else 1.0
    if not math.isfinite(motion_scale) or motion_scale <= 1e-9:
        motion_scale = 1e-3
    lengths = estimate_limb_lengths_m(positions)
    projections = np.stack([camera.projection_matrix for camera in cameras])

    frame_index, joint_index, camera_index = np.nonzero(seen)
    observed_uv = observations[frame_index, camera_index, joint_index, :2]
    observed_weight = np.sqrt(
        np.clip(observations[frame_index, camera_index, joint_index, 2], 0.0, 1.0)
    )
    limb_pairs = [
        (JOINT_INDEX[parent], JOINT_INDEX[child], value) for (parent, child), value in lengths.items()
    ]
    robust = robust_scale_px * pixel_scale

    def unpack(vector: np.ndarray) -> np.ndarray:
        return vector.reshape(frames, joints, 3)

    def residuals(vector: np.ndarray) -> np.ndarray:
        current = unpack(vector)
        selected = current[frame_index, joint_index]
        homogeneous = np.hstack((selected, np.ones((len(selected), 1))))
        projected = np.einsum("nij,nj->ni", projections[camera_index], homogeneous)
        depth = np.where(np.abs(projected[:, 2]) < 1e-9, 1e-9, projected[:, 2])
        error = np.linalg.norm(projected[:, :2] / depth[:, None] - observed_uv, axis=1)
        # Robustify the reprojection block only, in the residual itself, and
        # leave the regularisers quadratic. Handing scipy a global `loss` would
        # down-weight the limb and temporal terms too, which are priors rather
        # than measurements and have no outliers to suppress.
        compressed = 2.0 * robust * (np.sqrt(1.0 + error / robust) - 1.0)
        safe = np.where(error < 1e-9, 1.0, error)
        parts = [((projected[:, :2] / depth[:, None] - observed_uv)
                  * (compressed / safe * observed_weight)[:, None]).ravel()]
        for parent, child, value in limb_pairs:
            measured = np.linalg.norm(current[:, parent] - current[:, child], axis=1)
            # Percentage length error, so one weight covers a forearm and a
            # thigh alike rather than favouring whichever limb is longer.
            parts.append(length_weight * 100.0 * (measured - value) / value)
        if frames > 2:
            acceleration = current[:-2] - 2.0 * current[1:-1] + current[2:]
            # Normalised by the take's own mean per-frame displacement, so the
            # weight means the same thing for a still shot and a fast one.
            parts.append(smooth_weight * acceleration.ravel() / motion_scale)
        return np.concatenate(parts)

    sparsity = _sequence_jacobian_sparsity(
        frames, joints, frame_index, joint_index, limb_pairs
    )
    solution = least_squares(
        residuals,
        start.ravel(),
        jac_sparsity=sparsity,
        method="trf",
        loss="linear",
        ftol=1e-3,
        max_nfev=maximum_evaluations,
        verbose=0,
    )
    solved = unpack(solution.x)

    # A recovery has to survive the same gate a triangulation would. If the
    # solved point does not lie near its own ray it is not evidence, and it is
    # demoted to the caller's interpolation rather than reported as recovered.
    recovered = np.zeros_like(direct)
    rows = np.nonzero(candidate)
    for frame, joint in zip(rows[0].tolist(), rows[1].tolist()):
        errors = []
        for camera in np.nonzero(seen[frame, joint])[0].tolist():
            projected, depth = cameras[camera].project(solved[frame, joint])
            if depth <= 0.0:
                errors.append(math.inf)
                continue
            errors.append(float(np.linalg.norm(projected - observations[frame, camera, joint, :2])))
        recovered[frame, joint] = bool(errors) and max(errors) <= robust

    # Deliberately do NOT overwrite slots that already triangulated from two or
    # more views. Measured, the solve moved them by a median of 11-14 mm and up
    # to 700 mm -- the temporal and limb terms outvoting good geometry. This is
    # a recovery pass for missing evidence, not a re-estimator of present
    # evidence, and the existing Savitzky-Golay stage still smooths the seam.
    output = positions.copy()
    output[recovered] = solved[recovered]
    return output, recovered


def _sequence_jacobian_sparsity(
    frames: int,
    joints: int,
    frame_index: np.ndarray,
    joint_index: np.ndarray,
    limb_pairs: Sequence[tuple[int, int, float]],
) -> Any:
    """Which variables each residual touches. Without this the solve is dense."""

    def variable(frame: int, joint: int) -> int:
        return (frame * joints + joint) * 3

    rows: list[int] = []
    columns: list[int] = []
    row = 0
    for frame, joint in zip(frame_index.tolist(), joint_index.tolist()):
        base = variable(frame, joint)
        for _ in range(2):
            rows.extend([row] * 3)
            columns.extend([base, base + 1, base + 2])
            row += 1
    for parent, child, _ in limb_pairs:
        for frame in range(frames):
            parent_base, child_base = variable(frame, parent), variable(frame, child)
            rows.extend([row] * 6)
            columns.extend(
                [parent_base, parent_base + 1, parent_base + 2, child_base, child_base + 1, child_base + 2]
            )
            row += 1
    if frames > 2:
        for frame in range(frames - 2):
            for joint in range(joints):
                for axis in range(3):
                    for offset in range(3):
                        rows.append(row)
                        columns.append(variable(frame + offset, joint) + axis)
                    row += 1
    return coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(row, frames * joints * 3),
    )


def _fill_and_smooth_positions(values: np.ndarray) -> tuple[np.ndarray, float]:
    output = np.asarray(values, dtype=np.float64).copy()
    if output.ndim != 3 or output.shape[2] != 3:
        raise CommercialMultiviewError("World positions must be [frame,joint,3]")
    missing_before = ~np.isfinite(output).all(axis=2)
    frame_axis = np.arange(len(output), dtype=np.float64)
    optional_head_joints = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
    for joint in range(output.shape[1]):
        valid = np.isfinite(output[:, joint]).all(axis=1)
        if np.count_nonzero(valid) < 2:
            if JOINT_NAMES[joint] in optional_head_joints:
                # Body IK leaves GNM in control of the head and face. Sparse or
                # profile-view facial body points therefore fall back to the
                # reconstructed neck instead of making body capture fail.
                neck = output[:, JOINT_INDEX["neck"]]
                if np.isfinite(neck).all():
                    output[:, joint] = neck
                    continue
            raise CommercialMultiviewError(
                f"Joint {JOINT_NAMES[joint]} has fewer than two multiview samples"
            )
        for axis in range(3):
            output[:, joint, axis] = np.interp(
                frame_axis, frame_axis[valid], output[valid, joint, axis]
            )
        if len(output) >= 7:
            window = min(9, len(output) if len(output) % 2 else len(output) - 1)
            if window >= 5:
                output[:, joint] = savgol_filter(
                    output[:, joint], window_length=window, polyorder=2, axis=0, mode="interp"
                )
    return output, float(np.mean(missing_before))


def _quat_inverse(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    output = quaternion.copy()
    output[..., :3] *= -1.0
    return output / np.sum(quaternion * quaternion, axis=-1, keepdims=True)


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise CommercialMultiviewError("Pose contains a degenerate direction")
    return vector / norm


def _from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    start = _unit(source)
    finish = _unit(target)
    dot = float(np.clip(np.dot(start, finish), -1.0, 1.0))
    if dot >= 1.0 - 1e-10:
        return np.asarray((0.0, 0.0, 0.0, 1.0))
    if dot <= -1.0 + 1e-10:
        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(start)))] = 1.0
        axis = _unit(np.cross(start, basis))
        return np.asarray((*axis, 0.0))
    result = np.asarray((*np.cross(start, finish), 1.0 + dot))
    return result / np.linalg.norm(result)


def _frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    first = _unit(primary)
    second = np.asarray(secondary, dtype=np.float64)
    second -= first * float(np.dot(first, second))
    second = _unit(second)
    third = _unit(np.cross(first, second))
    second = _unit(np.cross(third, first))
    return np.column_stack((first, second, third))


def _frame_alignment(
    source_primary: np.ndarray,
    source_secondary: np.ndarray,
    target_primary: np.ndarray,
    target_secondary: np.ndarray,
) -> np.ndarray:
    source = _frame(source_primary, source_secondary)
    target = _frame(target_primary, target_secondary)
    return Rotation.from_matrix(target @ source.T).as_quat()


def _world_for_bone(parent_world: np.ndarray, rest_child_offset: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = _rotate_vector(parent_world, rest_child_offset)
    correction = _from_to(current, target)
    output = _quaternion_multiply(correction, parent_world)
    return output / np.linalg.norm(output)


def _set_world(
    local: np.ndarray,
    world: np.ndarray,
    frame: int,
    joint_name: str,
    target_world: np.ndarray,
) -> None:
    index = DETAILED_HUMANOID.index(joint_name)
    parent = DETAILED_HUMANOID.joints[index].parent
    normalized = np.asarray(target_world, dtype=np.float64)
    normalized /= np.linalg.norm(normalized)
    world[frame, index] = normalized
    local[frame, index] = (
        normalized
        if parent == -1
        else _quaternion_multiply(_quat_inverse(world[frame, parent]), normalized)
    )
    local[frame, index] /= np.linalg.norm(local[frame, index])


def positions_to_body_track(
    positions_world_z_up_m: np.ndarray,
    *,
    sample_rate_hz: int,
    provenance_sha256: str,
) -> BodyTrack:
    source = np.asarray(positions_world_z_up_m, dtype=np.float64)
    if (
        source.ndim != 3
        or source.shape[1:] != (len(JOINT_NAMES), 3)
        or len(source) < 2
        or not np.isfinite(source).all()
        or sample_rate_hz <= 0
        or TICKS_PER_SECOND % sample_rate_hz
        or len(provenance_sha256) != 64
    ):
        raise CommercialMultiviewError("Body-track reconstruction inputs are invalid")
    # Capture is metric Z-up. AutoAnim is metric Y-up with camera-world +Y
    # becoming canonical -Z.
    points = source[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    frames = len(points)
    joints = len(DETAILED_HUMANOID.joints)
    local = np.zeros((frames, joints, 4), dtype=np.float64)
    world = np.zeros_like(local)
    local[..., 3] = 1.0
    world[..., 3] = 1.0
    root_translation = np.zeros((frames, 3), dtype=np.float64)
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))

    rest = {joint.name: np.asarray(joint.rest_translation_m, dtype=np.float64) for joint in DETAILED_HUMANOID.joints}
    for frame in range(frames):
        p = points[frame]
        at = lambda name: p[JOINT_INDEX[name]]
        pelvis = 0.5 * (at("left_hip") + at("right_hip"))
        neck = at("neck")
        shoulder_across = at("left_shoulder") - at("right_shoulder")
        hip_across = at("left_hip") - at("right_hip")
        torso_up = neck - pelvis
        hips_world = _frame_alignment((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), torso_up, hip_across)
        torso_world = _frame_alignment((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), torso_up, shoulder_across)
        root_translation[frame] = pelvis - np.asarray((0.0, 0.98, 0.0))
        _set_world(local, world, frame, "Root", identity)
        _set_world(local, world, frame, "Hips", hips_world)
        for name in ("Spine", "Chest", "UpperChest", "Neck", "Head"):
            _set_world(local, world, frame, name, torso_world)
        for name in ("LeftEye", "RightEye"):
            _set_world(local, world, frame, name, torso_world)

        chains = (
            ("LeftShoulder", "LeftUpperArm", at("left_shoulder") - (pelvis + 0.72 * torso_up)),
            ("LeftUpperArm", "LeftLowerArm", at("left_elbow") - at("left_shoulder")),
            ("LeftLowerArm", "LeftHand", at("left_wrist") - at("left_elbow")),
            ("RightShoulder", "RightUpperArm", at("right_shoulder") - (pelvis + 0.72 * torso_up)),
            ("RightUpperArm", "RightLowerArm", at("right_elbow") - at("right_shoulder")),
            ("RightLowerArm", "RightHand", at("right_wrist") - at("right_elbow")),
            ("LeftUpperLeg", "LeftLowerLeg", at("left_knee") - at("left_hip")),
            ("LeftLowerLeg", "LeftFoot", at("left_ankle") - at("left_knee")),
            ("RightUpperLeg", "RightLowerLeg", at("right_knee") - at("right_hip")),
            ("RightLowerLeg", "RightFoot", at("right_ankle") - at("right_knee")),
        )
        for joint_name, child_name, direction in chains:
            joint_index = DETAILED_HUMANOID.index(joint_name)
            parent_index = DETAILED_HUMANOID.joints[joint_index].parent
            target_world = _world_for_bone(
                world[frame, parent_index], rest[child_name], direction
            )
            _set_world(local, world, frame, joint_name, target_world)

        for hand in ("LeftHand", "RightHand"):
            hand_index = DETAILED_HUMANOID.index(hand)
            parent = DETAILED_HUMANOID.joints[hand_index].parent
            _set_world(local, world, frame, hand, world[frame, parent])
        for foot in ("LeftFoot", "RightFoot"):
            _set_world(local, world, frame, foot, torso_world)
        for toe in ("LeftToes", "RightToes"):
            toe_index = DETAILED_HUMANOID.index(toe)
            parent = DETAILED_HUMANOID.joints[toe_index].parent
            _set_world(local, world, frame, toe, world[frame, parent])
        for index, joint in enumerate(DETAILED_HUMANOID.joints):
            if joint.name.startswith(("LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing", "LeftLittle", "RightThumb", "RightIndex", "RightMiddle", "RightRing", "RightLittle")):
                parent = joint.parent
                _set_world(local, world, frame, joint.name, world[frame, parent])

    # Ensure quaternion paths do not flip signs across frames.
    for frame in range(1, frames):
        dots = np.sum(local[frame] * local[frame - 1], axis=1)
        local[frame, dots < 0.0] *= -1.0
    step = TICKS_PER_SECOND // sample_rate_hz
    ticks = np.arange(frames, dtype=np.int64) * step
    eyes = np.zeros((frames, 2, 4), dtype=np.float32)
    eyes[..., 3] = 1.0
    track = BodyTrack(
        duration_ticks=int(ticks[-1]),
        ticks_per_second=TICKS_PER_SECOND,
        sample_rate_hz=sample_rate_hz,
        joint_names=DETAILED_HUMANOID.names,
        ticks=ticks,
        root_translation_m=root_translation.astype(np.float32),
        local_rotations_xyzw=local.astype(np.float32),
        foot_contacts=np.zeros((frames, 2), dtype=np.bool_),
        gaze_direction_body=np.broadcast_to(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32), (frames, 3)
        ),
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=eyes,
        source_plan_sha256=provenance_sha256,
    )
    # Calibrated capture contains noisier frame-to-frame ankle estimates than
    # a generated track, so use an observation-appropriate contact envelope.
    # Projection still enforces the canonical 5 cm validation lock per run.
    projected, _ = project_generated_foot_contacts(
        track,
        velocity_threshold_m_per_s=0.30,
        ground_band_m=0.08,
        minimum_contact_frames=2,
        maximum_root_correction_m=0.08,
    )
    return projected


def reconstruct_multiview(
    cameras: Sequence[CalibratedCamera],
    observations_by_camera: Sequence[Sequence[dict[str, Any]]],
    *,
    subject_count: int = 2,
    sample_rate_hz: int = 30,
    # Cycle-consistent graph matching by default: measured identical to the
    # exhaustive search on every frame of the reference fixture, ~2x faster
    # there, and the only tractable option beyond two subjects -- the exhaustive
    # search is (subjects!)^cameras, which is 1,296 assignments at three
    # subjects and four cameras and 110 billion at four subjects and eight.
    associator: Callable[..., tuple[np.ndarray, float]] = associate_frame_graph,
) -> tuple[tuple[BodyTrack, ...], ReconstructionDiagnostics, np.ndarray, np.ndarray]:
    """Reconstruct every subject from calibrated multiview observations.

    Returns ``(tracks, diagnostics, smoothed_positions, raw_positions)``. The
    third array is interpolated and Savitzky-Golay filtered and is what the
    exporters consume; the fourth is the pre-fill triangulation with NaNs
    intact, for honest noise measurement. They share shape and dtype, so keep
    the order straight at call sites.
    """
    if (
        len(cameras) < 2
        or len(cameras) != len(observations_by_camera)
        or any(len(values) != len(observations_by_camera[0]) for values in observations_by_camera)
        or len(observations_by_camera[0]) < 2
    ):
        raise CommercialMultiviewError("Multiview frame streams are not aligned")
    frames = len(observations_by_camera[0])
    frame_ids = [value["frame_index"] for value in observations_by_camera[0]]
    for values in observations_by_camera[1:]:
        if [value["frame_index"] for value in values] != frame_ids:
            raise CommercialMultiviewError("Camera observation frame numbers differ")
    scaled_cameras = tuple(
        camera.scaled(values[0]["width"], values[0]["height"])
        for camera, values in zip(cameras, observations_by_camera, strict=True)
    )
    # Pixel-denominated gates are quoted at REFERENCE_DETECTOR_WIDTH_PX. Without
    # this the detector width silently changes which observations are accepted.
    reference_width = observations_by_camera[0][0]["width"]
    reference_height = observations_by_camera[0][0]["height"]
    for values in observations_by_camera[1:]:
        if values[0]["width"] != reference_width or values[0]["height"] != reference_height:
            # A single pixel_scale cannot describe a rig whose cameras ran at
            # different detector sizes, even though intrinsics are scaled per
            # camera above.
            raise CommercialMultiviewError("Camera observation frame sizes differ")
    pixel_scale = float(reference_width) / REFERENCE_DETECTOR_WIDTH_PX
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise CommercialMultiviewError("Observation width does not yield a valid pixel scale")
    world = np.full((subject_count, frames, len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
    retained_observations = np.full(
        (subject_count, frames, len(cameras), len(JOINT_NAMES), 3), np.nan, dtype=np.float64
    )
    deferred_frames = 0
    reprojection: list[float] = []
    association_costs: list[float] = []
    previous_roots: np.ndarray | None = None
    previous_positions: np.ndarray | None = None
    previous_observations: np.ndarray | None = None
    previous_velocity = np.zeros((subject_count, 3), dtype=np.float64)
    last_good_frame = np.full(subject_count, -1, dtype=np.int64)
    temporally_rejected_subject_frames = 0
    for frame in range(frames):
        detections = [
            [_person_array(person) for person in values[frame]["people"]]
            for values in observations_by_camera
        ]
        if frame_is_ambiguous(detections, subject_count):
            deferred_frames += 1
        associated, cost = associator(
            scaled_cameras,
            detections,
            subject_count=subject_count,
            previous_roots_world_m=previous_roots,
            previous_positions_world_m=previous_positions,
            previous_observations_xyc=previous_observations,
            pixel_scale=pixel_scale,
        )
        association_costs.append(cost)
        candidate_world = np.full(
            (subject_count, len(JOINT_NAMES), 3), np.nan, dtype=np.float64
        )
        candidate_errors: list[list[float]] = [[] for _ in range(subject_count)]
        for subject in range(subject_count):
            for joint in range(len(JOINT_NAMES)):
                result = triangulate_point(
                    scaled_cameras,
                    associated[subject, :, joint, :2],
                    associated[subject, :, joint, 2],
                    pixel_scale=pixel_scale,
                )
                if result is None:
                    continue
                candidate_world[subject, joint] = result.position_world_m
                candidate_errors[subject].extend(
                    result.reprojection_errors_px[list(result.used_camera_indices)]
                )
        candidate_roots = candidate_world[:, JOINT_INDEX["root"]]
        accepted = np.isfinite(candidate_roots).all(axis=1)
        prediction_errors = np.zeros(subject_count, dtype=np.float64)
        for subject in range(subject_count):
            if not accepted[subject] or last_good_frame[subject] < 0 or previous_roots is None:
                continue
            elapsed = (frame - int(last_good_frame[subject])) / sample_rate_hz
            predicted = previous_roots[subject] + previous_velocity[subject] * elapsed
            prediction_errors[subject] = float(
                np.linalg.norm(candidate_roots[subject] - predicted)
            )
            # A bounded acceleration envelope rejects a multi-view ghost while
            # still allowing fast motion when an established root velocity
            # predicts it. Missing frames widen the envelope quadratically.
            tolerance = 0.12 + 1.5 * elapsed * elapsed
            if prediction_errors[subject] > tolerance:
                accepted[subject] = False
        if subject_count == 2 and np.all(np.isfinite(candidate_roots)):
            if float(np.linalg.norm(candidate_roots[0] - candidate_roots[1])) < 0.28:
                # Two pelvis tracks cannot occupy the same body volume. Keep
                # the one better supported by its temporal prediction.
                reject = int(np.argmax(prediction_errors))
                accepted[reject] = False

        temporally_rejected_subject_frames += int(
            np.count_nonzero(np.isfinite(candidate_roots).all(axis=1) & ~accepted)
        )

        if previous_roots is None:
            previous_roots = np.full((subject_count, 3), np.nan, dtype=np.float64)
            previous_positions = np.full(
                (subject_count, len(JOINT_NAMES), 3), np.nan, dtype=np.float64
            )
            previous_observations = np.full(
                (subject_count, len(cameras), len(JOINT_NAMES), 3),
                np.nan,
                dtype=np.float64,
            )
        for subject in range(subject_count):
            if not accepted[subject]:
                continue
            world[subject, frame] = candidate_world[subject]
            retained_observations[subject, frame] = associated[subject]
            reprojection.extend(candidate_errors[subject])
            if last_good_frame[subject] >= 0:
                elapsed = (frame - int(last_good_frame[subject])) / sample_rate_hz
                measured_velocity = (
                    candidate_roots[subject] - previous_roots[subject]
                ) / elapsed
                previous_velocity[subject] = (
                    0.75 * previous_velocity[subject] + 0.25 * measured_velocity
                )
            previous_roots[subject] = candidate_roots[subject]
            previous_positions[subject] = candidate_world[subject]
            previous_observations[subject] = associated[subject]
            last_good_frame[subject] = frame
    valid_fraction = float(np.mean(np.isfinite(world).all(axis=3)))
    smoothed: list[np.ndarray] = []
    interpolated: list[float] = []
    tracks: list[BodyTrack] = []
    provenance = sha256()
    provenance.update(SCHEMA_VERSION.encode("ascii"))
    provenance.update(np.asarray(world, dtype="<f8").tobytes())
    provenance.update(json.dumps(frame_ids, separators=(",", ":")).encode("ascii"))
    source_hash = provenance.hexdigest()
    contacts: list[tuple[int, int]] = []
    recovered_counts: list[float] = []
    for subject in range(subject_count):
        # Recover single-ray joints from limb-length and temporal constraints
        # before falling back to interpolation, so an evidence-based estimate is
        # preferred to a drawn line wherever one is available.
        resolved, recovered = solve_sequence_positions(
            scaled_cameras,
            world[subject],
            retained_observations[subject],
            pixel_scale=pixel_scale,
        )
        recovered_counts.append(float(np.mean(recovered)))
        positions, fraction = _fill_and_smooth_positions(resolved)
        smoothed.append(positions)
        interpolated.append(fraction)
        track = positions_to_body_track(
            positions,
            sample_rate_hz=sample_rate_hz,
            provenance_sha256=sha256(
                f"{source_hash}:{subject}".encode("ascii")
            ).hexdigest(),
        )
        tracks.append(track)
        contacts.append(
            (int(np.sum(track.foot_contacts[:, 0])), int(np.sum(track.foot_contacts[:, 1])))
        )
    errors = np.asarray(reprojection, dtype=np.float64)
    diagnostics = ReconstructionDiagnostics(
        frame_count=frames,
        subject_count=subject_count,
        camera_count=len(cameras),
        valid_joint_fraction=valid_fraction,
        constraint_recovered_joint_fraction=float(np.mean(recovered_counts)) if recovered_counts else 0.0,
        frames_deferred_to_exhaustive_association=deferred_frames,
        median_reprojection_error_px=float(np.median(errors)),
        p95_reprojection_error_px=float(np.percentile(errors, 95)),
        maximum_reprojection_error_px=float(np.max(errors)),
        association_objective_median=float(np.median(association_costs)),
        interpolated_joint_fraction=float(np.mean(interpolated)),
        temporally_rejected_subject_frames=temporally_rejected_subject_frames,
        contact_frames=tuple(contacts),
    )
    # `smoothed` is post-interpolation and Savitzky-Golay filtered, so it cannot
    # measure raw triangulation noise. `world` is what triangulation actually
    # produced, NaNs intact, before any fill or head-joint fallback.
    return tuple(tracks), diagnostics, np.asarray(smoothed), world.copy()


__all__ = [
    "CalibratedCamera",
    "associate_frame_graph",
    "CommercialMultiviewError",
    "JOINT_NAMES",
    "PROVIDER_ID",
    "ReconstructionDiagnostics",
    "TriangulatedPoint",
    "associate_frame",
    "camera_from_dict",
    "load_camera_rig",
    "load_observation_jsonl",
    "positions_to_body_track",
    "reconstruct_multiview",
    "triangulate_point",
]
