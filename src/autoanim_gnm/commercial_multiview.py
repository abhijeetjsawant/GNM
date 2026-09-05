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
    HumanoidSkeleton,
    _quaternion_multiply,
    _rotate_vector,
)
from .body_projection import project_generated_foot_contacts
from .performer_skeleton import performer_skeleton


SCHEMA_VERSION = "autoanim.commercial-multiview/1.0"
# The original observation contract named its detector, which stopped being true
# the moment a second one existed. 1.1 is detector-neutral and carries a
# `detector` field; 1.0 is still accepted and reads as Apple Vision.
OBSERVATION_SCHEMA_VERSION = "autoanim.apple-vision-body-observations/1.0"
BODY_OBSERVATION_SCHEMA_VERSION = "autoanim.body-observations/1.1"
LEGACY_OBSERVATION_DETECTOR = "apple_vision"
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
    # Frames the associator could not solve at all. Previously these aborted the
    # run; they now fall through to interpolation and are counted here, because a
    # silently interpolated frame is worse than a reported one.
    frames_without_association: int
    median_reprojection_error_px: float
    p95_reprojection_error_px: float
    maximum_reprojection_error_px: float
    association_objective_median: float
    interpolated_joint_fraction: float
    temporally_rejected_subject_frames: int
    contact_frames: tuple[tuple[int, int], ...]
    # One entry per subject describing whether head orientation was SOLVED or fell back
    # to the torso frame, and why. A fallback is the pre-existing behaviour -- a head
    # welded to the chest -- and it must never be silent, because that constant scores at
    # parity with a research reference on frame-to-frame jitter while carrying no head
    # information at all. See docs/HEAD_ORIENTATION_MEASURED.md.
    head_orientation: tuple[dict[str, Any], ...] = ()
    # Per-subject ball-of-foot triangulation, the evidence behind the foot
    # orientation. Absent means the feet fell back to the torso frame.
    toe_triangulation: tuple[dict[str, Any], ...] = ()
    # D7. Per-subject `Spine1` triangulation -- the evidence behind the pelvis frame --
    # and per-subject whether `Hips` took that frame or fell back to the trunk line.
    # A fallback is the pre-D7 behaviour: the whole trunk as one rigid block, with the
    # root offset riding the lean. It must never be silent.
    spine_triangulation: tuple[dict[str, Any], ...] = ()
    pelvis_frame: tuple[dict[str, Any], ...] = ()
    # D8. The share of slots whose value was HELD on the parent across a gap too long to
    # interpolate through, quoted on the same denominator as
    # `interpolated_joint_fraction` and reported beside it. Interpolation draws a line
    # through missing evidence; a hold says so. Neither is a measurement and the two must
    # never be summed into one "valid" number.
    held_joint_fraction: float = 0.0
    # D8. Per subject: the conditioning gate's and the reachability reject's own counts,
    # the ceilings they ran at, and the ray-pair geometry they saw. A demotion is not an
    # error and a rejection is not a failure -- but an unreported one is, because the
    # delivered smoothed array carries the consequence of both.
    occlusion_repair: tuple[dict[str, Any], ...] = ()

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
            "frames_without_association": self.frames_without_association,
            "median_reprojection_error_px": self.median_reprojection_error_px,
            "p95_reprojection_error_px": self.p95_reprojection_error_px,
            "maximum_reprojection_error_px": self.maximum_reprojection_error_px,
            "association_objective_median": self.association_objective_median,
            "interpolated_joint_fraction": self.interpolated_joint_fraction,
            "temporally_rejected_subject_frames": self.temporally_rejected_subject_frames,
            "contact_frames": [list(value) for value in self.contact_frames],
            "head_orientation": [dict(value) for value in self.head_orientation],
            "toe_triangulation": [dict(value) for value in self.toe_triangulation],
            "spine_triangulation": [dict(value) for value in self.spine_triangulation],
            "pelvis_frame": [dict(value) for value in self.pelvis_frame],
            "held_joint_fraction": self.held_joint_fraction,
            "occlusion_repair": [dict(value) for value in self.occlusion_repair],
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
    """Load 2D body observations from any detector that honours the contract.

    Accepts the detector-neutral 1.1 schema, which names its detector, and the
    original 1.0 schema, which was Apple-Vision-only by construction and is read
    as such. Every frame comes back carrying a `detector` key either way, so
    downstream code never has to know which version it came from.
    """

    frames: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        version = value.get("schema_version") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or version not in (OBSERVATION_SCHEMA_VERSION, BODY_OBSERVATION_SCHEMA_VERSION)
            or not isinstance(value.get("frame_index"), int)
            or not isinstance(value.get("width"), int)
            or not isinstance(value.get("height"), int)
            or not isinstance(value.get("people"), list)
        ):
            raise CommercialMultiviewError("Body observation schema is invalid")
        if version == BODY_OBSERVATION_SCHEMA_VERSION and not isinstance(value.get("detector"), str):
            raise CommercialMultiviewError("Body observations must name their detector")
        value.setdefault("detector", LEGACY_OBSERVATION_DETECTOR)
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
    weight_before_loss: bool = False,
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
        # Where the weight enters matters and was got wrong here. Applying it
        # *after* the compression scales an already-robustified residual, so a
        # confidence can shrink an outlier's contribution but cannot change what
        # counts as an outlier -- which is most of what a per-observation sigma is
        # for. `fit_hand_sequence` multiplies before the loss, so its loss sees
        # r/sigma, and that is the statistically correct ordering.
        #
        # UNMEASURED, and not measurable on anything we currently have. Flipping the
        # flag gives bit-identical output on every available fixture, because this
        # function returns triangulation on any slot that triangulated and
        # early-returns when nothing failed -- so the branch is only reachable on
        # single-ray slots, of which the synthetic fixture has none and the reference
        # take has ~20 in 5,700. The default stays False until a fixture exists that
        # exercises it. See docs/CONFIDENCE_GATE_NOT_WEIGHT.md.
        if weight_before_loss:
            scaled = error * observed_weight
            compressed = 2.0 * robust * (np.sqrt(1.0 + scaled / robust) - 1.0)
            safe = np.where(scaled < 1e-9, 1.0, scaled)
            parts = [((projected[:, :2] / depth[:, None] - observed_uv)
                      * (compressed / safe * observed_weight)[:, None]).ravel()]
        else:
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


# A fixed 300 ms window at 30 fps. Measured cost: with *noiseless* 2D the smoother
# alone contributes 0.65 mm of median error at the reference fixture's joint speeds
# and 4.42 mm at six times those speeds, where its p95 reaches 34.9 mm. Named here
# rather than inlined so the trade can be measured instead of assumed.
SMOOTHING_WINDOW_FRAMES = 9


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
            window = min(SMOOTHING_WINDOW_FRAMES,
                         len(output) if len(output) % 2 else len(output) - 1)
            if window >= 5:
                output[:, joint] = savgol_filter(
                    output[:, joint], window_length=window, polyorder=2, axis=0, mode="interp"
                )
    return output, float(np.mean(missing_before))


# How much of the head's rotation relative to the chest the NECK carries, with the head
# joint taking the remainder. The delivered rig gave `Neck` the torso frame verbatim, so
# every degree of head turn was applied at the skull and the neck was a constant -- the
# same defect the head itself had, one joint down.
#
# This is an ANATOMICAL distribution, not a measurement, and it is labelled as such:
# SOMA-77's own neck landmarks cannot support one. `Neck1->Neck2` varies 64-115 % in length
# and `Neck2->Head` 36-167 %, against body controls at 2.5-4.1 %
# (tools/head/region_landmark_quality.py) -- the cervical landmarks are noise at this
# framing, exactly as the eye baseline is.
#
# What this changes and what it does not: the composed HEAD world orientation is preserved
# exactly, because the head keeps the remainder of whatever the neck takes. So no measured
# quantity moves and no gate result changes -- the head gate scores head-relative-to-thorax,
# which is untouched. What changes is that the bend is distributed down the chain instead
# of hinging entirely at the skull, which is how a neck works and how the character reads.
#
# 0.5 is the standard rigging split and is deliberately not tuned: there is nothing on this
# fixture to tune it against, and a value fitted to a reference would be a shipped constant
# calibrated on MAMMA.
# ======================================================================================
# D8 -- the captured limbs under occlusion. Three rules, all of them sitting AFTER the raw
# triangulation is captured and BEFORE the sequence solve and the fill, so
# `raw_triangulated_world_positions_z_up_m` is bit-identical across the D7b and D8 builds
# and is the one unchanged reference every D8 band shares.
#
# THE DEFECT. In the reference take's push-and-fall window (frame ids 85-125) cameras B001
# and D001 lose one performer entirely -- B001 on 20 of the 41 frames, D001 on 32, at
# sub-floor confidences whose p10-p90 is 0.05-0.13 -- leaving A001 and C001, which sit
# 171-172 degrees apart at that subject. Two near-collinear rays fix a point across their
# common axis and not along it, so the shoulder line collapses and stretches (68 mm at
# frame 108, 554 at frame 100) while every epipolar and reprojection gate stays satisfied.
# `maximum_epipolar_px` and `inlier_threshold_px` measure agreement, and the two views
# AGREE, to 3-5 px, on the wrong point. It is a CONDITIONING failure, not a noise one, and
# no residual threshold can see it. Measured in `tools/compare/captured_limb_stability.py`
# and recorded in `docs/reviews/occlusion-repair-2026-09-05.md`.
# ======================================================================================

# PROVENANCE: ENGINEERING-LIMIT, derived in closed form, and NOT selected on any take --
# including the synthetic one. The derivation, in one line: two rays meeting at an angle
# theta determine a point ACROSS their common axis and only weakly ALONG it, and the
# along-axis error is amplified by 1/|sin theta| relative to a right-angled pair. 90 deg is
# 1.0x, 150 deg is 2.0x, 172 deg (which is what A001 and C001 make at the falling performer
# in the reference window) is 7.2x. The ceiling below is the angle at which that
# amplification reaches **2x**, and 2x is the declared choice. The complement clause falls
# out of the same expression: 180 - 150 = 30 deg is where |sin theta| passes 0.5 from the
# other side, so the rule is exactly "demote when |sin theta| < 0.5".
#
# WHY IT IS NOT SELECTED ON SYNTHETIC TRUTH, WHICH IS WHAT THE D8 CARD SAID IT WOULD BE.
# `tools/compare/d8_occlusion_synthetic.py` tried and could not, and the reason is a
# property of the fixture rather than of this rule: the fixture's bodies have exactly rigid
# bones and exactly smooth motion, which are precisely the sequence solve's own priors. A
# recovery whose priors are true by construction beats a noisy two-view triangulation at
# EVERY angle -- including a well-conditioned right-angled pair -- so the score curve has no
# crossover and its argmin is always the lowest candidate swept, which is "never trust two
# views": a capacity change rather than evidence, and the thing the card explicitly forbids.
# Shipping that argmin would be the standing error this lane keeps a rule about, a band the
# candidate can optimise. What the fixture CAN do, and does, is confirm the closed form --
# the measured two-view triangulation error is CONSISTENT WITH rising as 1/|sin theta|
# across its angle bins; the bin table and the coefficient are in the report and in the
# review, and neither is quoted as a fit.
#
# THE EVIDENCE THAT THIS IS A FIXTURE ARTEFACT AND NOT A FINDING is in this file, six
# hundred lines up: `solve_sequence_positions` deliberately does NOT overwrite slots that
# already triangulated, because measured on real data the solve moved them 'by a median of
# 11-14 mm and up to 700 mm -- the temporal and limb terms outvoting good geometry'. Those
# priors are true on the fixture and false on the footage, which is exactly why the fixture
# prefers demoting everything and the footage would not.
#
# HONESTY ABOUT THE ORDER THIS WAS FOUND IN: 150.0 was in this file before the closed form
# was written down, as a first guess. k = 2 was RECOGNISED as the factor that yields it,
# not chosen ahead of it. The value did not move; the justification replaced a worse one.
# Recorded in docs/reviews/occlusion-repair-2026-09-05.md section 7.
#
# A two-view slot whose supporting rays meet at more than this angle -- or at less than its
# complement, the near-co-located case -- is depth-unconstrained along their common axis
# and is DEMOTED: its triangulated position is withheld from the sequence solve, whose
# existing single-ray recovery (ray + the performer's own parent distance + temporal
# continuity) then resolves it from the same rays plus evidence the triangulation never
# used. Nothing is invented and no prior is added; the rays are the ones already there.
# A slot with THREE or more supporting views is never demoted: the third view fixes the
# depth the pair cannot.
RAY_PAIR_CONDITIONING_CEILING_DEG = 150.0

# PROVENANCE: ANATOMY. Peak LINEAR speed of each landmark, metres per second, as a
# physical impossibility bound and deliberately not a plausibility one: the rule exists to
# refuse what a body cannot do, and a ceiling tight enough to refuse what a body merely
# rarely does would be a smoother wearing a reject's clothes. Sources, in the units they
# are published in: elite baseball pitching reaches ~34 m/s at the hand and ~9 m/s at the
# shoulder (Fleisig et al., kinematic chain studies); competitive boxing punches reach
# 6-9 m/s at the fist; sprinting reaches ~2x running speed at the foot during swing, so
# ~12 m/s at 6 m/s ground speed. Each ceiling below sits at or above the published peak
# for that landmark, so the rule fires only on motion no measured human has produced.
# The ENVELOPE these are turned into (below) is what synthetic truth selects; these
# numbers are anatomy and are not selected on any take.
REACHABILITY_SPEED_CEILING_M_S = {
    "nose": 8.0, "left_eye": 8.0, "right_eye": 8.0, "left_ear": 8.0, "right_ear": 8.0,
    "neck": 6.0, "root": 6.0,
    "left_shoulder": 9.0, "right_shoulder": 9.0,
    "left_elbow": 18.0, "right_elbow": 18.0,
    "left_wrist": 34.0, "right_wrist": 34.0,
    "left_hip": 6.0, "right_hip": 6.0,
    "left_knee": 10.0, "right_knee": 10.0,
    "left_ankle": 14.0, "right_ankle": 14.0,
}

# PROVENANCE: SYNTHETIC-TRUTH, selected with the ceiling above by
# `tools/compare/d8_occlusion_synthetic.py`. The constant term of the reachability
# envelope, in metres: what the landmark may move by beyond its speed budget before the
# frame is refused. It absorbs the triangulation's own noise, so a stationary joint's
# jitter cannot trip a rule about motion. The envelope is
# ``slack + ceiling * elapsed_seconds`` and it is measured from the LAST ACCEPTED point
# with `elapsed` counting the frames since that point, exactly as
# `_reachable_clavicle_sequence` does -- never a step test against the immediately
# preceding frame, which accepts the wrong plateau between two big steps (CLAUDE.md).
REACHABILITY_SLACK_M = 0.09

# PROVENANCE: SYNTHETIC-TRUTH, selected by `tools/compare/d8_occlusion_synthetic.py`.
# `_fill_and_smooth_positions` interpolates linearly across a gap of any length: on the
# reference take the real per-camera miss runs have median 2, p90 28 and max 40 frames, so
# a straight line can be drawn through more than a second of missing evidence and then
# Savitzky-Golay filtered into something that looks measured. Gaps LONGER than this are
# not interpolated through; the landmark is held on its parent (its world offset from the
# parent at the last accepted frame is carried) and the fraction is REPORTED as
# `held_joint_fraction` beside `interpolated_joint_fraction`. A whole-take hold is the
# must-fail control.
#
# PROVENANCE, AND A SECOND REFUTED PREDICTION. The card said this would be selected on
# synthetic truth. It cannot be, and for the SAME reason the ray-angle ceiling cannot: the
# fixture's motion is smooth by construction, so a straight line through a gap is nearly
# exact there and the score prefers interpolating at every candidate length -- the
# monotone curve again, whose argmin is "never hold". `tools/compare/d8_occlusion_
# synthetic.py` reports that sweep in full and selects nothing from it.
#
# The value is instead a closed-form bound, with its sensitivity stated rather than hidden.
# A straight chord across a gap of duration T departs from a constant-acceleration
# trajectory by at most a*T^2/8. Setting that equal to REACHABILITY_SLACK_M -- the measured
# jitter of our own triangulation, so the bound says "the line stops being defensible when
# it can be wrong by more than the noise" -- gives T = sqrt(8*slack/a). At a limb
# acceleration of 20 m/s^2 (brisk everyday reaching) that is 5.7 frames at 30 fps; at
# 50 m/s^2 (fast sport) it is 3.6. THE VALUE BELOW IS THE 20 m/s^2 END AND THAT CHOICE IS
# DECLARED, NOT DERIVED. It is the conservative end -- a longer permitted gap holds fewer
# slots and departs less from the pre-D8 behaviour.
#
# The clause's value is the DIAGNOSTIC, not an error reduction: it makes the pipeline stop
# claiming a measurement it does not have, and `held_joint_fraction` says how often. On any
# fixture whose motion is smooth it will look like a cost, because there it IS one.
MAXIMUM_INTERPOLATED_GAP_FRAMES = 6

# Which landmark is carried by which when a long gap is held. Parent in the kinematic
# sense, not the detector's: a wrist follows its elbow, an elbow its shoulder, a shoulder
# the neck. `root` has no parent and is never held (it is also never missing on this take).
LANDMARK_PARENT = {
    "left_wrist": "left_elbow", "right_wrist": "right_elbow",
    "left_elbow": "left_shoulder", "right_elbow": "right_shoulder",
    "left_shoulder": "neck", "right_shoulder": "neck",
    "left_ankle": "left_knee", "right_ankle": "right_knee",
    "left_knee": "left_hip", "right_knee": "right_hip",
    "left_hip": "root", "right_hip": "root",
    "neck": "root",
    "nose": "neck", "left_eye": "nose", "right_eye": "nose",
    "left_ear": "nose", "right_ear": "nose",
}


def _ray_pair_angles_deg(
    cameras: Sequence[CalibratedCamera],
    point: np.ndarray,
    view_indices: Sequence[int],
) -> float:
    """Largest angle, in degrees, between any two supporting views' rays to ``point``.

    The conditioning number of a two-view triangulation, expressed as the one geometric
    quantity that says whether depth along the pair's common axis is determined. Near 90
    degrees the pair is well conditioned; near 0 or 180 it is not, and the point slides
    freely along the axis while both reprojections stay perfect.
    """

    directions: list[np.ndarray] = []
    for index in view_indices:
        vector = np.asarray(point, dtype=np.float64) - cameras[index].camera_center_world_m
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            directions.append(vector / norm)
    worst = 0.0
    for position, first in enumerate(directions):
        for second in directions[position + 1:]:
            cosine = float(np.dot(first, second))
            worst = max(worst, math.degrees(math.acos(min(1.0, max(-1.0, cosine)))))
    return worst


def _reachable_landmark_sequence(
    points: np.ndarray,
    ceiling_m_per_s: float,
    slack_m: float,
    sample_rate_hz: int,
    rule: str = "reachability",
) -> np.ndarray:
    """Which frames of ONE landmark's position series a body could have reached.

    ``points`` is ``[frame, 3]``, NaN where the landmark did not triangulate. Returns a
    boolean ``accepted`` of the same length; a non-finite frame is never accepted and never
    becomes the anchor the next frame is measured from.

    The rule is `_reachable_clavicle_sequence`'s, one dimension down: the budget is
    measured from the LAST ACCEPTED point and widened by the frames elapsed since it, so a
    landmark that has been missing for a while may legitimately have travelled further.

    ``rule`` is in the signature on purpose and is the reason this function is module level
    and called by bare name, like :func:`_reachable_clavicle_sequence`, :func:`_joint_origin`
    and :func:`_leg_root_offset`. ``"step"`` is the MUST-FAIL control and it must run
    through this identical call site rather than a re-implementation of it. The two rules
    differ in exactly two things, and each difference is a defect:

    * the step test measures from the last PRESENT sample, accepted or not, so a run of
      rejected frames becomes its own reference and **the wrong plateau between two big
      steps is accepted** -- each frame inside the plateau is a small step from the last
      one (CLAUDE.md);
    * the step test's budget is one frame's worth whatever the gap, so a landmark that
      legitimately travelled while unobserved is **rejected for having moved**, where the
      reachability envelope widens with the frames elapsed and accepts it.

    Both are demonstrated in `tests/test_occlusion_repair.py`.
    """

    values = np.asarray(points, dtype=np.float64)
    frames = len(values)
    accepted = np.zeros(frames, dtype=bool)
    finite = np.isfinite(values).all(axis=1)
    if not finite.any():
        return accepted
    first = int(np.flatnonzero(finite)[0])
    accepted[first] = True
    anchor = first
    previous_present = first
    for frame in range(first + 1, frames):
        if not finite[frame]:
            continue
        if rule == "reachability":
            reference = anchor
            elapsed = max(1, frame - reference) / float(sample_rate_hz)
        else:
            reference = previous_present
            elapsed = 1.0 / float(sample_rate_hz)
        previous_present = frame
        envelope = slack_m + ceiling_m_per_s * elapsed
        if float(np.linalg.norm(values[frame] - values[reference])) <= envelope:
            accepted[frame] = True
            anchor = frame
    return accepted


def _hold_long_gaps_on_parent(
    values: np.ndarray,
    maximum_gap_frames: int,
) -> tuple[np.ndarray, float]:
    """Carry a landmark on its parent across a gap too long to interpolate through.

    ``values`` is ``[frame, joint, 3]`` with NaN where nothing is known. A run of missing
    frames LONGER than ``maximum_gap_frames`` is not left for `_fill_and_smooth_positions`
    to draw a straight line through: the landmark's world offset from its parent at the
    last known frame before the run is held, and translated by the parent's own motion. A
    run that trails off either end is held on the nearest known frame's offset.

    Returns ``(filled, held_fraction)``. Anything this does not fill stays NaN and the
    caller's existing fill handles it, so a short gap behaves exactly as it always has.

    WHY BETWEEN THE SOLVE AND THE FILL, and not inside the fill. `solve_sequence_positions`
    calls `_fill_and_smooth_positions` itself to build its starting point; changing that
    function's behaviour would change the solver's initialisation and the clean oracle
    would stop being bit-identical. This is a separate pass applied after the solve has had
    its chance to recover a slot from evidence -- a hold is what happens when there is no
    evidence at all, and it must never pre-empt a recovery.

    WHAT IT DOES NOT DO. It has no frame to rotate in: at this stage the pipeline holds
    positions, not orientations, so the offset is carried by TRANSLATION only. A limb that
    rotates about its parent while unobserved is held straight, and the diagnostics say how
    many slots that is rather than the result looking measured.
    """

    output = np.asarray(values, dtype=np.float64).copy()
    if output.ndim != 3 or output.shape[2] != 3:
        raise CommercialMultiviewError("World positions must be [frame,joint,3]")
    if maximum_gap_frames < 0:
        raise CommercialMultiviewError("The interpolated-gap ceiling must not be negative")
    held = np.zeros(output.shape[:2], dtype=bool)
    known = np.isfinite(output).all(axis=2)
    for name, parent_name in LANDMARK_PARENT.items():
        if name not in JOINT_INDEX or parent_name not in JOINT_INDEX:
            continue
        joint, parent = JOINT_INDEX[name], JOINT_INDEX[parent_name]
        missing = ~known[:, joint]
        if not missing.any():
            continue
        frame = 0
        while frame < len(output):
            if not missing[frame]:
                frame += 1
                continue
            stop = frame
            while stop < len(output) and missing[stop]:
                stop += 1
            if stop - frame > maximum_gap_frames:
                before = frame - 1 if frame > 0 and known[frame - 1, joint] else -1
                after = stop if stop < len(output) and known[stop, joint] else -1
                anchor = before if before >= 0 else after
                if anchor >= 0 and known[anchor, parent]:
                    offset = output[anchor, joint] - output[anchor, parent]
                    for index in range(frame, stop):
                        if not known[index, parent]:
                            continue
                        output[index, joint] = output[index, parent] + offset
                        held[index, joint] = True
            frame = stop
    return output, float(np.mean(held))


def _repair_occluded_slots(
    cameras: Sequence[CalibratedCamera],
    world: np.ndarray,
    supporting_views: np.ndarray,
    *,
    sample_rate_hz: int,
    ray_pair_ceiling_deg: float | None,
    reachability: bool,
    reachability_slack_m: float = REACHABILITY_SLACK_M,
    reachability_rule: str = "reachability",
) -> tuple[np.ndarray, dict[str, Any]]:
    """One subject's triangulated positions, with the unusable slots withheld.

    ``world`` is ``[frame, joint, 3]`` and is NEVER modified: a copy is returned, and the
    array the caller passes in is the one it returns to its own caller as
    ``raw_triangulated_world_positions_z_up_m``. That is the whole reason this function
    takes a copy rather than repairing in place -- the raw array is D8's fixed reference
    and every band in the step is scored against it.

    ``supporting_views`` is ``[frame, joint, camera]`` booleans: which cameras the
    triangulator actually USED for that slot (its inliers), not which cameras could see it.

    Two rules, applied in this order, each of which only ever turns a position into NaN:

    * **conditioning** -- a slot whose supporting views number exactly two and whose rays
      meet outside the well-conditioned band is DEMOTED. The point is withheld; the rays
      stay in `retained_observations`, so `solve_sequence_positions` sees the slot as one
      it must recover and resolves it from those same rays plus the performer's own limb
      lengths and temporal continuity.
    * **reachability** -- a landmark that cannot have reached its position from its last
      accepted one inside the anatomical envelope is REJECTED. Applied after the
      demotions, so "last accepted" means last surviving, and the two rules cannot
      disagree about what the anchor is.

    Returns ``(withheld, report)``.
    """

    values = np.asarray(world, dtype=np.float64).copy()
    frames, joints = values.shape[0], values.shape[1]
    finite = np.isfinite(values).all(axis=2)
    demoted = np.zeros((frames, joints), dtype=bool)
    rejected = np.zeros((frames, joints), dtype=bool)
    angles = np.full((frames, joints), np.nan, dtype=np.float64)

    if ray_pair_ceiling_deg is not None:
        complement = 180.0 - float(ray_pair_ceiling_deg)
        for frame in range(frames):
            for joint in range(joints):
                if not finite[frame, joint]:
                    continue
                views = np.flatnonzero(supporting_views[frame, joint]).tolist()
                if len(views) != 2:
                    # Three or more supporting views fix the depth a pair cannot; one or
                    # none never produced a triangulation in the first place.
                    continue
                angle = _ray_pair_angles_deg(cameras, values[frame, joint], views)
                angles[frame, joint] = angle
                if angle > float(ray_pair_ceiling_deg) or angle < complement:
                    demoted[frame, joint] = True
        values[demoted] = np.nan

    if reachability:
        for joint in range(joints):
            name = JOINT_NAMES[joint]
            ceiling = REACHABILITY_SPEED_CEILING_M_S.get(name)
            if ceiling is None:
                continue
            accepted = _reachable_landmark_sequence(
                values[:, joint], ceiling, reachability_slack_m, sample_rate_hz,
                rule=reachability_rule)
            present = np.isfinite(values[:, joint]).all(axis=1)
            rejected[:, joint] = present & ~accepted
        values[rejected] = np.nan

    two_view = np.zeros((frames, joints), dtype=bool)
    for frame in range(frames):
        for joint in range(joints):
            two_view[frame, joint] = (
                finite[frame, joint]
                and int(supporting_views[frame, joint].sum()) == 2)

    report = {
        "ray_pair_conditioning_ceiling_deg": (None if ray_pair_ceiling_deg is None
                                              else float(ray_pair_ceiling_deg)),
        "reachability_reject": bool(reachability),
        "reachability_slack_m": float(reachability_slack_m),
        "reachability_rule": reachability_rule,
        "triangulated_slots": int(finite.sum()),
        "two_supporting_view_slots": int(two_view.sum()),
        "demoted_slots": int(demoted.sum()),
        "rejected_slots": int(rejected.sum()),
        "demoted_by_joint": {JOINT_NAMES[j]: int(demoted[:, j].sum())
                             for j in range(joints) if demoted[:, j].any()},
        "rejected_by_joint": {JOINT_NAMES[j]: int(rejected[:, j].sum())
                              for j in range(joints) if rejected[:, j].any()},
        "two_view_ray_angle_deg_median": (
            round(float(np.nanmedian(angles)), 3) if np.isfinite(angles).any() else None),
        "note": ("a demoted slot keeps its rays and is handed to the sequence solve as a "
                 "slot to recover; a rejected slot keeps nothing and falls to the solve "
                 "and then the fill. The raw array is untouched by both."),
    }
    return values, report


NECK_ROTATION_SHARE = 0.5


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc quaternion interpolation, xyzw."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    return (math.sin((1.0 - t) * theta) * a + math.sin(t * theta) * b) / sin_theta


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> xyzw quaternion. Rows are normalised first.

    CLAUDE.md records why: an unnormalised, scaled matrix drives `acos` past its clamp and
    every joint then reads 0.00 degrees apart. The same hazard applies to any trace-based
    quaternion extraction, so normalise rather than assume.
    """
    rotation = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(rotation, axis=1, keepdims=True)
    if not np.all(norms > 1e-9):
        raise CommercialMultiviewError("Rotation matrix has a degenerate row")
    rotation = rotation / norms
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return _unit(np.asarray((x, y, z, w), dtype=np.float64))


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


# A relaxed hand. NOT CAPTURE -- read this before using it as one.
#
# SOMA-77 emits 15 finger joints per hand and they cannot drive this rig. Measured on this
# footage against body controls at 2.5-4.1 % length variation, the finger segments run
# 15.8-111.4 %, and -- the decisive part -- every one of them is roughly HALF its anatomical
# length: the palm reads 39-58 mm where a hand is 85-100 mm, the index-to-pinky span 35-44 mm
# where it should be 65-85 mm. The detector is collapsing the hand toward the wrist. A
# segment that is stable in the wrong place is a prediction, not a measurement, which is
# exactly the "wrist-conditioned prior" docs/FINGER_TRIANGULATION_GATE.md identified.
# tools/head/region_landmark_quality.py.
#
# So the fingers get a POSE, not a solve. The point is that the alternative was never
# neutral: leaving every finger joint at the identity is also an invented pose, and a worse
# one -- it splays the hand into a rigid claw no resting human makes. Both are uncaptured;
# this one is anatomically right and reads as a hand.
#
# Per-joint flexion in degrees about the rig's curl axis, from a relaxed open hand. The
# thumb is shallower and the distal joints curl more than the proximal, which is what a
# hand at rest does.
FINGER_REST_CURL_DEG = {"Proximal": 12.0, "Intermediate": 22.0, "Distal": 16.0,
                        "Metacarpal": 4.0}
FINGER_REST_CURL_THUMB_SCALE = 0.45


def _finger_rest_local(joint_name: str) -> np.ndarray:
    """Relaxed local rotation for one finger joint, xyzw. Identity for non-fingers.

    A pose, never a measurement -- see FINGER_REST_CURL_DEG. Curl is about the rig's local
    X axis, the flexion axis for every finger joint in this skeleton.
    """

    segment = next(
        (name for name in ("Metacarpal", "Proximal", "Intermediate", "Distal")
         if joint_name.endswith(name)),
        None,
    )
    if segment is None:
        return np.asarray((0.0, 0.0, 0.0, 1.0))
    degrees = FINGER_REST_CURL_DEG[segment]
    if "Thumb" in joint_name:
        degrees *= FINGER_REST_CURL_THUMB_SCALE
    # Left and right hands curl toward opposite sides of the rig's X axis. The sign
    # follows the rest skeleton: +X is the anatomical left (see body.CANONICAL_HUMANOID),
    # so it flipped with it on 2026-09-02 -- otherwise the fingers would curl backwards on
    # both hands, which no joint gate and no forward-dot can see.
    sign = 1.0 if joint_name.startswith("Left") else -1.0
    half = math.radians(sign * degrees) * 0.5
    return np.asarray((math.sin(half), 0.0, 0.0, math.cos(half)))


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


def _leg_root_offset(
    hips_world: np.ndarray,
    rest: dict[str, np.ndarray],
) -> np.ndarray:
    """The world vector from the rig's ``Hips`` joint to the midpoint of its two leg roots.

    D2b. Forward kinematics puts the leg roots at
    ``UpperLegMid = root_translation + rest["Root"] + rest["Hips"] + R_hips . mid`` with
    ``mid = 0.5 * (rest["LeftUpperLeg"] + rest["RightUpperLeg"])`` and ``rest["Root"] = 0``.
    The captured ``left_hip`` / ``right_hip`` landmarks ARE the femoral joint centres, so
    the rig's **leg roots** -- not ``Hips`` -- are what belongs on their midpoint. Setting
    ``UpperLegMid = pelvis`` gives
    ``root_translation = pelvis - rest["Hips"] - R_hips . mid``, and this returns the
    ``R_hips . mid`` term.

    NO CONSTANT ENTERS. ``mid`` is read from the caller's own ``rest`` dict, so a patched
    or per-performer-sized skeleton is honoured and the canonical (0, -0.08, 0) is never
    written down. The offset is taken in the HIPS' frame, not the world's: as the pelvis
    tilts its vertical component shrinks and a horizontal component appears, which a world
    vertical of the same length cannot reproduce -- that is the control that discriminates
    this derivation from a number (`tools/compare/d2_clavicle_gate.py`, `d2b_root_placement`).

    Deliberately module level and called by bare name, exactly as :func:`_joint_origin` is,
    so an instrument can substitute it and run every control through the identical code
    path rather than a re-implementation of it.
    """

    mid = 0.5 * (
        np.asarray(rest["LeftUpperLeg"], dtype=np.float64)
        + np.asarray(rest["RightUpperLeg"], dtype=np.float64)
    )
    return _rotate_vector(hips_world, mid)


# ------------------------------------------------------------------- D7: the pelvis frame
#
# THE REST PELVIS, AS A CONVENTION AND NOT AS A FACT -- read this before changing it.
#
# `Hips` used to take the TRUNK line (neck - hip midpoint) for its up axis, exactly as
# `Spine`, `Chest` and `UpperChest` do, so the whole trunk was one rigid block and
# `_leg_root_offset`'s `R_hips . mid` rode the lean. On the fixture's own clips a rigid
# pelvis departs from that trunk line by 26.0 / 32.8 deg median / p95 on the squat clip's
# bent frames (correlation +0.93 with tilt) and 3.9-8.7 deg on the upright ones.
#
# SOMA-77's `Spine1` (index 1) is a DIRECT CHILD of `Hips` (index 0), as are `LeftLeg`
# (67) and `RightLeg` (72). So `Spine1 - root` and `Spine1 - mid(hips)` rotate rigidly
# with the pelvis and a Kabsch fit of the three rest offsets recovers the pelvis rotation
# EXACTLY on noiseless input (residual 4.2e-8 m, `tools/compare/d7_pelvis_synthetic.py`).
# `Spine2` is a child of `Spine1` and therefore carries lumbar flexion (14.5 / 20.9 deg
# median / p95 on the squat clip); it is a LUMBAR direction, reported and never shipped.
#
# THE VECTORS BELOW ARE A CONVENTION. `src/autoanim_gnm/data/somaskel77-v1.json` carries
# names, parents and a coordinate system and NO rest geometry -- the plan card is wrong
# about that. The rest lives per clip in `.cache/autoanim_gnm/gem-x/outputs/*/
# soma_motion.npz` and is PER PERFORMER: five full-body clips' pelvis source frames differ
# by up to 10.03 deg. The delivered performer's own SOMA rest is unobservable, so these
# are the component-wise MEDIAN over those five rests, re-normalised. Third-party schema
# (GEM-X / Kimodo somaskel77). MAMMA-FREE: no MAMMA file, `ma_cap` output, SMPL-X fit or
# report computed from one enters them. Derivation and spread:
# `artifacts/compare/d7-pelvis-frame/rest-pelvis-constants.json`,
# `docs/reviews/pelvis-frame-2026-09-04.md`.
#
# WHAT A RESIDUAL COSTS. A leftover pitch `delta` between this convention and the
# performer's own pelvis moves the root fore/aft by `|mid| . sin(delta)` with `|mid|` about
# 80 mm -- on EVERY frame, upright ones included. At the measured spread that is 0.6-2.6 mm
# for the mid-hips lever. It is a constant, and a rigid fit cannot see a constant.
#
# THE AXES. SOMA-77 publishes right-handed, +Y up, +Z forward; so does this rig, and +X is
# the anatomical left in both. The capture-to-rig change of basis in
# `positions_to_body_track` is exactly the inverse of the SOMA-to-capture rotation the
# fixture applies, so SOMA REST AXES ARE RIG AXES and these vectors need no conversion.
SOMA77_REST_PELVIS_UP = (-0.0036018134417667037, 0.9928183854915898, 0.11957708965267391)
SOMA77_REST_HIPMID_TO_SPINE1 = (-0.003113853958207582, 0.9922427141575809, -0.12427670785277636)
SOMA77_REST_HIP_ACROSS = (0.9999986844845044, -0.0015715732215503752, 0.00040148084638751965)
SOMA77_REST_PELVIS_TEMPLATE_M = (
    (-0.00014224140613805503, 0.03922509402036667, 0.004722291603684425),   # root -> Spine1
    (0.09644327312707901, -0.05622021108865738, 0.017420830205082893),      # root -> LeftLeg
    (-0.0960559993982315, -0.055917683988809586, 0.017343545332551003),     # root -> RightLeg
)

# Which construction ships. SELECTED ON SYNTHETIC TRUTH ONLY, on the NOISY arm -- the clean
# arm cannot select, because every rigid candidate is exact on it by construction. The
# MAMMA arm reported beside it and selected nothing.
# PROVENANCE: `tools/compare/d7_pelvis_synthetic.py` ->
# `artifacts/compare/d7-pelvis-frame/synthetic.json`.
PELVIS_FRAME_SOURCE = "C_kabsch_pelvis"

# Temporal window on the pelvis FRAME, smoothed as a rotation in the tangent space about
# the take's mean -- the same treatment `_thorax_frames` applies, and for the same reason:
# a frame is a NONLINEAR function of its landmarks, so smoothing the points does not smooth
# the frame. 0 means none.
# PROVENANCE: synthetic truth, the same sweep. A window optimises exactly the quantity the
# sweep measures, so it is reported with its LAG and its peak ATTENUATION and it proves
# nothing about the pelvis frame itself.
PELVIS_SMOOTHING_FRAMES = 0

# Below this fraction of frames with a resolved spine point the WHOLE subject falls back to
# the torso frame with a diagnostics status. Deliberately a whole-subject decision: a
# per-frame flip between a spine-derived and a trunk-derived pelvis moves `R_hips . mid` by
# tens of millimetres in one frame and spikes the root. Engineering limit; nothing is
# fitted to it.
PELVIS_MINIMUM_RESOLVED_FRACTION = 0.5


def _pelvis_world_frames(
    points_rig_y_up_m: np.ndarray,
    spine1_rig_y_up_m: np.ndarray,
    *,
    mode: str | None = None,
    smoothing_frames: int | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """`Hips`' own world rotation per frame from the PELVIS's landmarks, or ``None``.

    Returns ``([frame, 4] xyzw, report)``. ``None`` restores the pre-existing behaviour --
    `Hips` takes the trunk line -- which is not a neutral default but the very defect this
    closes, so every path out of here carries a status the diagnostics publish.

    Both inputs are in the RIG's Y-up world, i.e. after `positions_to_body_track`'s change
    of basis. Gaps in the spine point are filled by linear interpolation over frames,
    exactly as `_fill_and_smooth_positions` fills the body joints; a subject with too few
    resolved frames falls back WHOLE, never frame by frame.

    Deliberately module level and called by bare name, exactly as :func:`_joint_origin` and
    :func:`_leg_root_offset` are, so an instrument can substitute it and run every control
    -- thorax-as-pelvis, a world vertical, the lumbar directions -- through the identical
    code path rather than a re-implementation of it.
    """

    mode = PELVIS_FRAME_SOURCE if mode is None else mode
    smoothing_frames = (
        PELVIS_SMOOTHING_FRAMES if smoothing_frames is None else smoothing_frames
    )
    points = np.asarray(points_rig_y_up_m, dtype=np.float64)
    spine = np.asarray(spine1_rig_y_up_m, dtype=np.float64)
    frames = len(points)
    if spine.shape != (frames, 3):
        raise CommercialMultiviewError("Spine positions must be [frame, 3]")
    valid = np.isfinite(spine).all(axis=1)
    resolved = float(valid.mean())
    if resolved < PELVIS_MINIMUM_RESOLVED_FRACTION or int(valid.sum()) < 2:
        return None, {
            "status": "fell_back_to_torso_frame",
            "reason": (
                f"only {resolved:.3f} of frames carry a resolved Spine1 point, below "
                f"PELVIS_MINIMUM_RESOLVED_FRACTION={PELVIS_MINIMUM_RESOLVED_FRACTION}; "
                "the whole subject falls back rather than switching definition per frame"
            ),
            "resolved_fraction": resolved,
            "mode": mode,
        }
    filled = spine.copy()
    axis = np.arange(frames, dtype=np.float64)
    for component in range(3):
        filled[:, component] = np.interp(axis, axis[valid], spine[valid, component])

    root = points[:, JOINT_INDEX["root"]]
    left_hip = points[:, JOINT_INDEX["left_hip"]]
    right_hip = points[:, JOINT_INDEX["right_hip"]]
    hip_mid = 0.5 * (left_hip + right_hip)
    hip_across = left_hip - right_hip

    quaternions = np.zeros((frames, 4), dtype=np.float64)
    if mode == "C_kabsch_pelvis":
        template = np.asarray(SOMA77_REST_PELVIS_TEMPLATE_M, dtype=np.float64)
        for frame in range(frames):
            observed = np.stack(
                (filled[frame] - root[frame],
                 left_hip[frame] - root[frame],
                 right_hip[frame] - root[frame])
            )
            u, _, vt = np.linalg.svd(template.T @ observed)
            sign = float(np.sign(np.linalg.det(vt.T @ u.T)))
            rotation = vt.T @ np.diag((1.0, 1.0, sign)) @ u.T
            quaternions[frame] = Rotation.from_matrix(rotation).as_quat()
    else:
        if mode == "A_root_to_spine1":
            source, target = SOMA77_REST_PELVIS_UP, filled - root
        elif mode == "B_hipmid_to_spine1":
            source, target = SOMA77_REST_HIPMID_TO_SPINE1, filled - hip_mid
        else:
            raise CommercialMultiviewError(f"unknown pelvis frame source {mode!r}")
        for frame in range(frames):
            quaternions[frame] = _frame_alignment(
                source, SOMA77_REST_HIP_ACROSS, target[frame], hip_across[frame]
            )

    if smoothing_frames and smoothing_frames > 1 and frames >= smoothing_frames:
        from .head_orientation import log_so3, orthonormalise, rodrigues

        matrices = Rotation.from_quat(quaternions).as_matrix()
        mean = orthonormalise(matrices.mean(axis=0)[None])[0]
        tangent = log_so3(np.einsum("nij,kj->nik", matrices, mean))
        window = smoothing_frames if smoothing_frames % 2 else smoothing_frames - 1
        tangent = savgol_filter(
            tangent, window_length=window, polyorder=2, axis=0, mode="interp"
        )
        smoothed = orthonormalise(np.einsum("nij,jk->nik", rodrigues(tangent), mean))
        quaternions = Rotation.from_matrix(smoothed).as_quat()

    # Keep the quaternion path continuous; `_set_world` reads one frame at a time and a
    # sign flip between frames would show up as a 360 deg step in every consumer.
    for frame in range(1, frames):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions, {
        "status": "solved",
        "mode": mode,
        "smoothing_frames": int(smoothing_frames),
        "resolved_fraction": resolved,
        "interpolated_frames": int(frames - int(valid.sum())),
    }


def _joint_origin(
    world: np.ndarray,
    frame: int,
    root_translation: np.ndarray,
    rest: dict[str, np.ndarray],
    joint_name: str,
) -> np.ndarray:
    """Where ``joint_name``'s origin actually is on ``frame``, from the rotations set so far.

    This is :func:`autoanim_gnm.body.forward_kinematics_positions`'s recursion, walked for
    one joint instead of all of them, and it must stay EXACTLY that:
    ``positions[root] = roots + rest[root]`` and
    ``positions[child] = positions[parent] + _rotate_vector(world[parent], rest[child])``.
    A converter that measures a direction from a different origin than the one the rotation
    turns about is the D2 defect itself, in miniature.

    `rest` is the caller's own dict, so a patched or per-performer-sized skeleton is
    honoured; `DETAILED_HUMANOID` is looked up as a module global at call time, because the
    swap harness rebinds `commercial_multiview.DETAILED_HUMANOID` to run the converter on a
    sized rig. This function is deliberately module level and called by bare name so an
    instrument can substitute it and run the controls through the identical code path.
    """

    skeleton = DETAILED_HUMANOID
    chain: list[int] = []
    index = skeleton.index(joint_name)
    while index != -1:
        chain.append(index)
        index = skeleton.joints[index].parent
    chain.reverse()
    position = np.asarray(root_translation[frame], dtype=np.float64) + rest[
        skeleton.joints[chain[0]].name
    ]
    for parent, child in zip(chain, chain[1:]):
        position = position + _rotate_vector(
            world[frame, parent], rest[skeleton.joints[child].name]
        )
    return position


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


# A HARD PHYSICAL REJECT ON THE CLAVICLE, NOT A TUNING KNOB, AND NOT A SMOOTHER.
#
# A human joint peaks near 500-800 deg/s (the same physiology `head_orientation.
# MAXIMUM_FRAME_TRAVEL_DEG` is drawn from), so at 30 fps 26.67 deg between consecutive
# frames is already the extreme end. A sternoclavicular joint has roughly 45 deg of
# elevation and 30 deg of protraction in total, so this ceiling is generous by a wide
# margin: it is here to reject the impossible, never to shape the possible.
#
# WHY IT WAS ADDED. D2 aims the clavicle from the rig's own Shoulder origin and D2b puts
# the root on the captured hips. Both are right, and both SHORTEN the lever from the
# pivot to the shoulder landmark -- ~400 mm under the old torso-axis anchor, 100-160 mm
# median after, 13-39 mm at its worst. Landmark wander of 21-35 mm across that lever is
# angle, and when an excursion carries the landmark near the pivot the direction flips
# outright. On the delivered take that produced single steps of 139 and 164 deg -- 4200
# and 4900 deg/s -- and the picture is unambiguous: at frame 48 of camera A001 the near
# performer's arm is drawn in and angled down, and one frame later it has snapped out
# straight and horizontal. A positional score prices that at zero, because the arm is in
# roughly the right place on both frames.
#
# THE RULE IS REACHABILITY, NOT A STEP TEST. A per-step test catches the transition into
# a wrong plateau and then accepts the plateau. A frame is accepted only if its clavicle
# LOCAL rotation lies within `ceiling * (t - t_last_accepted)` of the last accepted
# frame's -- an envelope that GROWS with elapsed frames, so a rejected run terminates on
# its own with no run-length constant anywhere, and an outlier first frame recovers (the
# envelope passes 180 deg after ceil(180 / 26.67) = 7 frames, which bounds both).
#
# LOCAL, NEVER WORLD, for the head lane's stated reason: a bone on a turning body travels
# in world with its own joint perfectly still, so a world-measured ceiling rejects honest
# motion. Measured at the joint.
#
# docs/reviews/clavicle-origin-2026-09-02.md section 15 (D2c).
CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S = 800.0


def _quaternion_travel_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two xyzw quaternions as rotations, degrees. Sign-insensitive."""

    dot = abs(float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, dot)))))


def _reachable_clavicle_sequence(
    local_rotations: np.ndarray,
    parent_world_rotations: np.ndarray,
    ceiling_deg_per_frame: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject clavicle frames a human clavicle could not have reached, and slerp across.

    ``local_rotations`` is ``[frame, 4]`` xyzw for ONE clavicle, in its parent's frame.
    Returns ``(replaced_local, accepted)``: the accepted frames are returned BYTE FOR BYTE
    as they arrived -- nothing here smooths a frame it accepts -- and each rejected run is
    replaced by shortest-arc slerp between the accepted frames that bracket it, or held at
    the last accepted value if the run trails off the end.

    ``parent_world_rotations`` is unused by this rule and is in the signature on purpose:
    the world-measured variant is the control that demonstrates why the rule is local
    (`tools/compare/d2c_clavicle_temporal_gate.py`), and it must run through the identical
    call site rather than a re-implementation of it. This function is deliberately module
    level and called by bare name for the same reason :func:`_joint_origin` and
    :func:`_leg_root_offset` are.
    """

    rotations = np.asarray(local_rotations, dtype=np.float64)
    frames = len(rotations)
    accepted = np.zeros(frames, dtype=bool)
    if frames == 0:
        return rotations.copy(), accepted
    output = rotations.copy()
    accepted[0] = True
    last = 0
    for frame in range(1, frames):
        envelope = ceiling_deg_per_frame * (frame - last)
        if _quaternion_travel_deg(output[last], rotations[frame]) <= envelope:
            accepted[frame] = True
            last = frame
    indices = np.flatnonzero(accepted)
    for start, stop in zip(indices, indices[1:]):
        for frame in range(start + 1, stop):
            output[frame] = _slerp(
                output[start], output[stop], (frame - start) / (stop - start)
            )
    for frame in range(int(indices[-1]) + 1, frames):
        output[frame] = output[int(indices[-1])]
    return output, accepted


def positions_to_body_track(
    positions_world_z_up_m: np.ndarray,
    *,
    sample_rate_hz: int,
    provenance_sha256: str,
    head_world_rotations: np.ndarray | None = None,
    toe_world_z_up_m: np.ndarray | None = None,
    spine_world_z_up_m: np.ndarray | None = None,
    pelvis_report_out: dict[str, Any] | None = None,
    skeleton: HumanoidSkeleton | None = None,
) -> BodyTrack:
    """Convert triangulated positions into a rig track.

    ``skeleton`` is the rest skeleton the track is solved ON, and the track carries it
    out again (``BodyTrack.rest_translations_m``) so that forward kinematics, validation,
    ground projection, the exporter, the sockets and every instrument rebuild the SAME
    body.  D3: before this, the converter read one skeleton and the exporter wrote a
    different one, and the delivered GLB's joints sat 81-195 mm from forward kinematics
    of the same track.

    ``None`` resolves to the module global ``DETAILED_HUMANOID`` **at call time**, not at
    import time, because `tools/swap-harness/retarget_cost.py` rebinds
    ``commercial_multiview.DETAILED_HUMANOID`` to run the converter on a sized rig and
    that must keep working.  A caller with a skeleton in hand should pass it explicitly.

    ``head_world_rotations`` is ``[frame, 3, 3]`` in the SAME world as the positions --
    capture Z-up metres -- from
    :func:`autoanim_gnm.head_orientation.solve_head_orientation`. It is converted to the
    rig's Y-up world here, by the same convention the positions are, so a caller never has
    to hold two frames in its head.
    When it is ``None`` this function behaves exactly as it always has -- the head, the
    neck and both eyes take the torso's frame, which makes head orientation a **constant**.
    That default is retained only so existing callers are unaffected; it is not a good
    head.

    ``spine_world_z_up_m`` is ``[frame, 3]``, SOMA-77's ``Spine1`` triangulated under the
    pipeline's own association, in the SAME world as the positions. Supplying it gives
    ``Hips`` its own frame from the PELVIS's landmarks (D7) instead of the trunk line, so
    the trunk is no longer one rigid block and ``_leg_root_offset`` no longer rides the
    lean. ``None`` restores the previous behaviour **bit for bit** -- the branch is one
    line and nothing else in this function moves -- and that fallback is a defect, not a
    neutral default, so ``pelvis_report_out`` (a dict this function fills in when given)
    carries the status the diagnostics publish. See :func:`_pelvis_world_frames`.

    When rotations are supplied, **the whole head-on-torso rotation is placed on `Head`**
    and `Neck` keeps the torso frame. Distributing it across the neck chain would look
    better on a mesh and is **unmeasured**: the gate in `docs/HEAD_ORIENTATION_MEASURED.md`
    scores the head, and the obvious source for a split -- the reference fitter's own
    neck/head ratio -- is barred, because no shipped constant may be fitted on a
    reference-derived artifact (`docs/BODY_LANE_PLAN.md`). Splitting it is a named next
    step, not something to guess here.
    """
    source = np.asarray(positions_world_z_up_m, dtype=np.float64)
    # The module global is read HERE, at call time -- see the docstring.
    skeleton = DETAILED_HUMANOID if skeleton is None else skeleton
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
    joints = len(skeleton.joints)
    local = np.zeros((frames, joints, 4), dtype=np.float64)
    world = np.zeros_like(local)
    local[..., 3] = 1.0
    world[..., 3] = 1.0
    root_translation = np.zeros((frames, 3), dtype=np.float64)
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    # Pass A computes the torso frame; pass C's feet fall back to it. Kept rather than
    # recomputed so the two passes cannot drift apart by a rounding.
    torso_world_by_frame = np.zeros((frames, 4), dtype=np.float64)

    rest = {joint.name: np.asarray(joint.rest_translation_m, dtype=np.float64) for joint in skeleton.joints}
    head_rotations = None
    # The ball-of-foot positions, converted through the identical change of basis as
    # `points` above. `[frame, 2, 3]` -- left then right -- in the capture's Z-up world.
    toes = None
    if toe_world_z_up_m is not None:
        toes = np.asarray(toe_world_z_up_m, dtype=np.float64)
        if toes.shape != (frames, 2, 3):
            raise CommercialMultiviewError("Toe positions must be [frame, 2, 3]")
        toes = toes[..., (0, 2, 1)].copy()
        toes[..., 2] *= -1.0
    # D7. `Hips`' own frame, or None -- in which case every line below is exactly what it
    # was before D7 and the delivered track is bit-identical.
    pelvis_world = None
    pelvis_report: dict[str, Any] = {
        "status": "not_attempted",
        "reason": "no spine landmarks supplied; Hips takes the trunk line",
    }
    if spine_world_z_up_m is not None:
        spine = np.asarray(spine_world_z_up_m, dtype=np.float64)
        if spine.shape != (frames, 3):
            raise CommercialMultiviewError("Spine positions must be [frame, 3]")
        spine = spine[..., (0, 2, 1)].copy()
        spine[..., 2] *= -1.0
        pelvis_world, pelvis_report = _pelvis_world_frames(points, spine)
    if pelvis_report_out is not None:
        pelvis_report_out.clear()
        pelvis_report_out.update(pelvis_report)
    if head_world_rotations is not None:
        head_rotations = np.asarray(head_world_rotations, dtype=np.float64)
        if head_rotations.shape != (frames, 3, 3):
            raise CommercialMultiviewError("Head rotations must be [frame, 3, 3]")
        if not np.isfinite(head_rotations).all():
            raise CommercialMultiviewError("Head rotations must be finite")
        # Capture is Z-up and the rig is Y-up with camera-world +Y becoming canonical -Z;
        # `points` above applies that to positions, and this applies the same change of
        # basis to rotations. A rotation transforms as C R C^T, not as a point.
        basis = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
        head_rotations = np.einsum("ij,fjk,lk->fil", basis, head_rotations, basis)
    for frame in range(frames):
        p = points[frame]
        at = lambda name: p[JOINT_INDEX[name]]
        pelvis = 0.5 * (at("left_hip") + at("right_hip"))
        neck = at("neck")
        shoulder_across = at("left_shoulder") - at("right_shoulder")
        hip_across = at("left_hip") - at("right_hip")
        torso_up = neck - pelvis
        # The source secondary axis names the rig axis that is the subject's LEFT.
        # It read (-1, 0, 0) until 2026-09-02, which was true of the bone NAMES and
        # false of the geometry those bones carry, so the torso came out yawed 180
        # degrees; +X is the anatomical left in the convention the skeleton publishes.
        # docs/reviews/facing-fix-2026-09-02.md.
        hips_world = _frame_alignment((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), torso_up, hip_across)
        # D7, and it is deliberately THE ONLY line that changes inside this loop. With no
        # spine landmark `pelvis_world` is None and the trunk line above stands, so a
        # legacy caller's track is bit-identical -- which `tests/test_pelvis_frame.py`
        # asserts. `_set_world` normalises its argument IN PLACE, so this hands it a copy
        # rather than a row of the pelvis array.
        if pelvis_world is not None:
            hips_world = pelvis_world[frame].copy()
        torso_world = _frame_alignment((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), torso_up, shoulder_across)
        # Read the hips' rest height from the skeleton rather than repeating it.
        # It was written as a literal (0, 0.98, 0), duplicating body.py's canonical
        # Hips offset -- harmless while every character is canonical, and a latent
        # defect the moment per-performer proportions are stamped: the root would be
        # placed using one hips height while the exporter adds back another, floating
        # or sinking the whole character by the difference, at the feet.
        #
        # D2b, 2026-09-03. The second term is the rest of that same argument. Until now
        # this line put the rig's `Hips` JOINT on the captured hip-landmark midpoint,
        # while the rig's leg roots -- which is what a `left_hip` landmark actually is,
        # the femoral joint centre -- hang `rest["LeftUpperLeg"]` = (0.09, -0.08, 0)
        # below and beside it. The whole skeleton therefore sat 80 mm low on its own
        # hips, on every frame, and `project_generated_foot_contacts` then hoisted it
        # ~140 / ~110 mm to keep it out of the floor. `_leg_root_offset` puts the LEG
        # ROOTS on that midpoint instead, so FK's UpperLeg midpoint lands exactly on the
        # captured one. Every term is the skeleton's own rest geometry; no constant
        # arrives. docs/reviews/clavicle-origin-2026-09-02.md section 12.
        root_translation[frame] = pelvis - rest["Hips"] - _leg_root_offset(
            hips_world, rest)
        _set_world(local, world, frame, "Root", identity)
        _set_world(local, world, frame, "Hips", hips_world)
        # D7b, 2026-09-05, and it is deliberately THE ONLY line added inside this loop.
        #
        # THE STANDING RULE: after replacing a parent, RE-SOLVE the chains below it. D7
        # replaced `Hips`' frame and did not re-solve the spine, and this is that.
        #
        # `torso_world` above aims the trunk from the captured hip MIDPOINT, which is
        # where `Spine`'s origin used to be -- while the whole trunk was one rigid block
        # the two coincided along the aim and the chain landed on the neck. Under a
        # pelvis frame they do not: `Spine` hangs `rest["Spine"]` off `Hips` IN THE
        # PELVIS FRAME, ~197 mm up the pelvis axis from the leg-root midpoint, so a
        # pelvis pitched away from the trunk line carries that origin off the line and a
        # straight rigid chain aimed from a displaced origin misses the captured neck by
        # the displacement -- 56 mm whole take and 116 mm on the bent tercile, measured
        # from the delivered file's own bytes (the delivered `Neck` went 14.3 -> 58.9 mm
        # off its landmark on performer 0 and 28.6 -> 44.0 on performer 1).
        #
        # Aim it from the origin the rotation actually turns about, exactly as D2 did for
        # the clavicle. `Spine`, `Chest` and `UpperChest` share this one world rotation
        # and their rests are collinear +Y, so `Neck`'s origin lands ON the ray to the
        # captured neck and the residual is the trunk's LENGTH error alone
        # (20.9 / 41.7 mm on performer 0, 18.3 / 11.9 on performer 1) -- which belongs to
        # a flexible spine (D5) and cannot be removed by any aim.
        #
        # `_joint_origin` reads `world[frame, Hips]`, so this MUST come after the two
        # `_set_world` calls above; and it is gated on `pelvis_world` so a legacy caller's
        # track stays bit-identical -- `tests/test_pelvis_frame.py` and
        # `tests/test_trunk_resolve.py` assert both halves.
        # docs/reviews/trunk-resolve-2026-09-05.md
        if pelvis_world is not None:
            torso_world = _frame_alignment(
                (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
                neck - _joint_origin(world, frame, root_translation, rest, "Spine"),
                shoulder_across)
        for name in ("Spine", "Chest", "UpperChest"):
            _set_world(local, world, frame, name, torso_world)
        # The head is the one joint here with its own evidence. With no solve it inherits
        # the torso and becomes a constant; with one it carries the measured orientation,
        # and the eyes follow the head rather than the chest.
        head_world = torso_world
        if head_rotations is not None:
            head_world = _quat_from_matrix(head_rotations[frame])
        # The neck takes a share of the chest-to-head rotation and the head keeps the rest,
        # so the composed head orientation is unchanged and the bend is distributed down the
        # chain rather than hinging at the skull. Without a head solve this is the torso
        # frame either way and the behaviour is exactly as before.
        neck_world = (
            torso_world if head_rotations is None
            else _slerp(torso_world, head_world, NECK_ROTATION_SHARE)
        )
        _set_world(local, world, frame, "Neck", neck_world)
        _set_world(local, world, frame, "Head", head_world)
        for name in ("LeftEye", "RightEye"):
            _set_world(local, world, frame, name, head_world)
        # Recorded HERE and not a line earlier, deliberately. `_set_world` normalises its
        # argument IN PLACE, and without a head solve `neck_world`, `head_world` and the
        # eyes' argument are all the same array as `torso_world` -- so by this point it has
        # been through seven normalisations, and that is the state the feet were handed
        # when the whole solve was one loop. Freezing it any earlier hands them a
        # differently-rounded quaternion; the gap is 1e-17 and it is large enough to show
        # up in the toes' near-identity local rotation after the float32 cast, which would
        # make this restructure visible in a bit-identity test that must see nothing.
        torso_world_by_frame[frame] = torso_world

        # NOTE, and it has two halves that came apart on 2026-09-02.
        #
        # THE LEGS. Measuring their directions from the joint's own forward-kinematic
        # origin instead of from the captured landmarks was tried on 2026-08-30 and
        # REVERTED -- it is correct in principle and fails in practice, because no
        # rig LEG joint origin coincided with its captured landmark. `root_translation`
        # placed Hips *at* the captured pelvis while the upper legs hung 80 mm below
        # and 90 mm to each side of it, so a leg direction measured from the rig's
        # own hip origin started 80 mm off. The canonical round trip went 0.00 mm ->
        # 46-67 mm on the legs. THAT PREMISE IS NOW GONE: D2b (below) places the leg
        # ROOTS on the captured hip midpoint, so an FK-origin leg direction would
        # round-trip too. It is deliberately NOT done here -- the legs below are still
        # measured landmark-to-landmark (knee - hip, ankle - knee), and re-measuring
        # them is D3's territory, with its own gate. See docs/BODY_LANE_PLAN.md.
        #
        # THE CLAVICLES. Done, D2, 2026-09-02, and NOT because they escape the root
        # placement above -- they do not. The clavicle's origin hangs off the same
        # misplaced Hips as the legs do. What is different is that measuring the
        # direction from that origin is right ANYWAY, because it is the point
        # `_world_for_bone` turns the bone about: wherever the root puts the rig, the
        # bone pivots there, and a direction measured from anywhere else asks the
        # rotation to do something it cannot. Get the origin right and the root fix,
        # when it comes, carries the clavicle with it and needs no further change here.
        # What was here instead was a synthetic anchor on the torso axis,
        # `pelvis + 0.72 * torso_up` -- 418 mm up the canonical rig -- while
        # `_world_for_bone` turned the bone about the rig's OWN Shoulder origin at
        # Hips + 0.12 + 0.16 + 0.15 + (+-0.11, 0.10, 0): 530 mm up and 110 mm out.
        # Two different origins for one direction. The delivered clavicle came out
        # 14.9 / 19.1 degrees off and the arm root landed 36-47 mm from the requested
        # shoulder EVEN ON A BODY WITH CANONICAL PROPORTIONS, which the canonical
        # round-trip oracle measures directly. The 0.72 was a fitted-looking number
        # nothing fitted (tools/compare/provenance.py) and it LEAVES with this change;
        # nothing replaces it. `_joint_origin` is the rig's own geometry and no
        # reference, MAMMA's least of all, enters it.
        #
        # ONE CONSEQUENCE, MEASURED AND OPEN. The old anchor was expressed in the
        # LANDMARK frame (`pelvis + ...`), so it was immune to the root-placement
        # offset above; this one is in the RIG frame, so it inherits it. The canonical
        # round-trip oracle therefore READS WORSE after this change, 41.57/47.05 ->
        # 67.25/79.32 mm, and every millimetre of that is root placement, not the
        # clavicle: `landmarks_from_fk` feeds the rig's UpperLeg origins back as hip
        # landmarks (anatomically right) and the converter puts `Hips` on them (80 mm
        # wrong), so the re-solve sees the rig 80 mm lower against the same landmarks.
        # Take the rig's own hip drop out of this origin and the round trip reads
        # 0.51/0.08 mm. On the delivery path, where the solve runs once, the arms
        # improve 181.33 -> 124.39 and 217.50 -> 89.56 mm.
        # docs/reviews/clavicle-origin-2026-09-02.md.
        left_shoulder_origin = _joint_origin(
            world, frame, root_translation, rest, "LeftShoulder")
        right_shoulder_origin = _joint_origin(
            world, frame, root_translation, rest, "RightShoulder")
        # PASS A ends with the two clavicles. Everything below them waits for pass C,
        # because the temporal reject in pass B can replace a clavicle and `_world_for_bone`
        # builds every arm bone from its PARENT's world: leave the arm locals as solved and
        # a replaced clavicle swings the whole arm rigidly, taking the elbow and the wrist
        # off the landmarks they were solved onto. Replace, then RE-SOLVE.
        for joint_name, child_name, direction in (
            ("LeftShoulder", "LeftUpperArm", at("left_shoulder") - left_shoulder_origin),
            ("RightShoulder", "RightUpperArm", at("right_shoulder") - right_shoulder_origin),
        ):
            joint_index = skeleton.index(joint_name)
            parent_index = skeleton.joints[joint_index].parent
            target_world = _world_for_bone(
                world[frame, parent_index], rest[child_name], direction
            )
            _set_world(local, world, frame, joint_name, target_world)

    # ---------------------------------------------------------------- PASS B: the reject
    # A sequence rule, so it cannot live inside a per-frame loop. Both clavicles
    # independently; accepted frames are left exactly as pass A wrote them, and only a
    # rejected frame's local rotation and the world derived from it are rewritten.
    ceiling_deg_per_frame = CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S / float(sample_rate_hz)
    for clavicle in ("LeftShoulder", "RightShoulder"):
        clavicle_index = skeleton.index(clavicle)
        parent_index = skeleton.joints[clavicle_index].parent
        replaced, accepted = _reachable_clavicle_sequence(
            local[:, clavicle_index], world[:, parent_index], ceiling_deg_per_frame
        )
        for frame in np.flatnonzero(~accepted):
            rotation = np.asarray(replaced[frame], dtype=np.float64)
            rotation = rotation / np.linalg.norm(rotation)
            local[frame, clavicle_index] = rotation
            composed = _quaternion_multiply(world[frame, parent_index], rotation)
            world[frame, clavicle_index] = composed / np.linalg.norm(composed)

    # ------------------------------------------- PASS C: everything below the clavicles
    for frame in range(frames):
        p = points[frame]
        at = lambda name: p[JOINT_INDEX[name]]
        torso_world = torso_world_by_frame[frame]
        for joint_name, child_name, direction in (
            ("LeftUpperArm", "LeftLowerArm", at("left_elbow") - at("left_shoulder")),
            ("LeftLowerArm", "LeftHand", at("left_wrist") - at("left_elbow")),
            ("RightUpperArm", "RightLowerArm", at("right_elbow") - at("right_shoulder")),
            ("RightLowerArm", "RightHand", at("right_wrist") - at("right_elbow")),
            ("LeftUpperLeg", "LeftLowerLeg", at("left_knee") - at("left_hip")),
            ("LeftLowerLeg", "LeftFoot", at("left_ankle") - at("left_knee")),
            ("RightUpperLeg", "RightLowerLeg", at("right_knee") - at("right_hip")),
            ("RightLowerLeg", "RightFoot", at("right_ankle") - at("right_knee")),
        ):
            joint_index = skeleton.index(joint_name)
            parent_index = skeleton.joints[joint_index].parent
            target_world = _world_for_bone(
                world[frame, parent_index], rest[child_name], direction
            )
            _set_world(local, world, frame, joint_name, target_world)

        for hand in ("LeftHand", "RightHand"):
            hand_index = skeleton.index(hand)
            parent = skeleton.joints[hand_index].parent
            _set_world(local, world, frame, hand, world[frame, parent])
        # FEET. This line used to read `_set_world(..., foot, torso_world)`: the foot was
        # given the TORSO's orientation, so it turned whenever the chest turned and carried
        # no foot information at all. That is exactly the defect the head had -- and worse
        # to catch, because a constant is visibly degenerate while a foot that swings with
        # the chest passes any "does it move?" check.
        #
        # Nothing observed could have constrained it: the 19 landmark targets stop at the
        # ankle, so rotation ABOUT the ankle had no evidence. SOMA-77's `ToeBase` supplies
        # it, and it is the one landmark in this pass that survived measurement -- 5.0-9.6 %
        # length variation against body controls at 2.5-4.1 %, better than the arm controls
        # on both performers. `ToeEnd` fails at 12.8-63.3 % and is deliberately NOT used, so
        # the toes keep riding the foot rather than getting a joint they cannot support.
        # tools/feet/toe_gate.py, docs/FEET_MEASURED.md.
        #
        # Two axes fix the frame: the foot's long axis from ankle to ball, and the shin from
        # knee to ankle. Without toe landmarks the previous behaviour is kept exactly.
        for side, ankle_name, knee_name, foot in (
            (0, "left_ankle", "left_knee", "LeftFoot"),
            (1, "right_ankle", "right_knee", "RightFoot"),
        ):
            foot_world = torso_world
            if toes is not None:
                ball = toes[frame, side]
                forward = ball - at(ankle_name)
                down = at(ankle_name) - at(knee_name)
                if (
                    np.isfinite(ball).all()
                    and np.linalg.norm(forward) > 1e-6
                    and np.linalg.norm(down) > 1e-6
                ):
                    foot_world = _frame_alignment(
                        rest["LeftToes"], (0.0, -1.0, 0.0), forward, down
                    )
            _set_world(local, world, frame, foot, foot_world)
        for toe in ("LeftToes", "RightToes"):
            toe_index = skeleton.index(toe)
            parent = skeleton.joints[toe_index].parent
            _set_world(local, world, frame, toe, world[frame, parent])
        for index, joint in enumerate(skeleton.joints):
            if joint.name.startswith(("LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing", "LeftLittle", "RightThumb", "RightIndex", "RightMiddle", "RightRing", "RightLittle")):
                parent = joint.parent
                # A relaxed rest curl instead of the identity. Both are uncaptured poses;
                # this one is anatomically right. See FINGER_REST_CURL_DEG for why the
                # fingers cannot be solved from this detector.
                local[frame, index] = _finger_rest_local(joint.name)
                world[frame, index] = _quaternion_multiply(
                    world[frame, parent], local[frame, index]
                )

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
        joint_names=skeleton.names,
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
        # D3. The track carries the rest it was solved on. Everything downstream --
        # forward kinematics, validation, the ground projection below, the exporter, the
        # sockets and every instrument -- rebuilds THIS body and not the canonical one.
        rest_translations_m=skeleton.rest_translations_m,
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



# Nine frames at 30 fps. The torso frame is a NONLINEAR function of its landmarks, so
# smoothing the positions does not smooth the frame -- a per-frame frame stays jittery on
# smoothed points. A reference that jitters charges its own wobble to whatever is measured
# against it.
#
# PROVENANCE: synthetic truth, selected without MAMMA (I8, 2026-09-02,
# `tools/head/thorax_window_sweep.py` -> `artifacts/compare/thorax-window-sweep.json`).
# Exact thorax frames from the tracked FK fixture, played at 0.58-0.77x of our own capture's
# thorax speed, with noise at our own detector's self-agreement amplitude injected in pixels
# and recovered through the real triangulator; the p95 angular error has an INTERIOR optimum
# at 9 (bracket 5-9) at every stride that reaches real speed. Two fixture biases (under-speed,
# white rather than correlated noise) both push the answer wider, so 9 is an upper bound.
# p95 alone does not separate 9 from 15 below 0.77x real speed; the case for 9 over the
# previous 15 rests on lag and attenuation: 15 lags the true frame by 1.8 frames and loses
# 32 % of peak yaw rate. The value was 15 from 2026-09-01 (commit 08a6c89), chosen on the
# MAMMA oracle arm -- that sweep is kept in `_thorax_frames`'s docstring as a REPORT, and it
# no longer selects anything. Changed to 9 on the user's decision, 2026-09-02.
THORAX_SMOOTHING_FRAMES = 9


def _thorax_frames(
    positions_world_z_up_m: np.ndarray, *, smoothing_frames: int = THORAX_SMOOTHING_FRAMES,
    across_from: str = "shoulders",
) -> np.ndarray:
    """Per-frame torso frame, [frame, 3, 3], columns (across, back, up).

    The same construction `positions_to_body_track` uses for `torso_world`, built from the
    SMOOTHED positions -- and then temporally smoothed AS A ROTATION, which is a separate
    thing. A frame is a nonlinear function of its landmarks, so smoothed points still give
    a jittery frame, and a jittery reference charges its own wobble to the head measured
    against it. An earlier version built this from raw triangulation and did exactly that;
    an independent audit later showed the reference's OWN best-case arm could not pass
    under it, which is what forced the first repair.

    **History, reports only.** The window was FIRST chosen (2026-09-01) on the oracle arm --
    the reference's own head expressed in this frame, which contains none of our head
    estimate -- which made it a shipped constant selected on MAMMA and therefore a leak
    (I8). It was re-selected on 2026-09-02 from synthetic truth with our own detector's
    noise (see `THORAX_SMOOTHING_FRAMES` above): 9, not 15. The oracle sweep below is kept
    because it shows the same SHAPE (an interior optimum, a median that prefers narrower
    than the p95) and it never selects again. Read it as one performer's p95 minimum at
    15 and the other's falling monotonically to 31 -- no window was the argmin on both:

    | window | oracle P1 median | oracle P1 p95 |
    |---|---|---|
    | none | 5.46 / 4.87 | 17.22 / 17.59 |
    | 5 | 5.42 / 4.87 | 17.00 / 17.21 |
    | **15** | **5.46 / 4.89** | **14.14 / 16.30** |
    | 21 | 5.87 / 4.96 | 14.12 / 17.19 |
    | 31 | 6.52 / 4.88 | 13.78 / 16.39 |

    Past 15 the p95 stops improving while the median degrades -- the frame is starting to
    lose real torso motion. 15 is the p95 minimum before that turn.
    """
    up = positions_world_z_up_m[:, JOINT_INDEX["neck"]] - positions_world_z_up_m[:, JOINT_INDEX["root"]]
    # `across_from` exists so the alternatives can be measured through THIS function rather
    # than a copy of it -- a reimplementation of this construction disagreed with it by
    # 1.4 deg of oracle p95, which is the size of the effect being tested. Default is
    # unchanged and the shoulders path is byte-identical to before.
    def _pair(left: str, right: str) -> np.ndarray:
        return (
            positions_world_z_up_m[:, JOINT_INDEX[left]]
            - positions_world_z_up_m[:, JOINT_INDEX[right]]
        )

    def _unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    if across_from == "shoulders":
        across = _pair("left_shoulder", "right_shoulder")
    elif across_from == "hips":
        across = _pair("left_hip", "right_hip")
    elif across_from == "mean":
        across = _unit(_pair("left_shoulder", "right_shoulder")) + _unit(_pair("left_hip", "right_hip"))
    else:
        raise ValueError(f"unknown across_from {across_from!r}")
    z = up / np.linalg.norm(up, axis=1, keepdims=True)
    x = across - z * np.einsum("ni,ni->n", across, z)[:, None]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    frames = np.stack([x, np.cross(z, x), z], axis=2)
    if smoothing_frames <= 1 or len(frames) < smoothing_frames:
        return frames
    from .head_orientation import log_so3, orthonormalise, rodrigues

    # Smooth in the tangent space about the take's mean, which is well-conditioned for the
    # modest excursions a torso makes, then re-orthonormalise.
    mean = orthonormalise(frames.mean(axis=0)[None])[0]
    tangent = log_so3(np.einsum("nij,kj->nik", frames, mean))
    window = smoothing_frames if smoothing_frames % 2 else smoothing_frames - 1
    tangent = savgol_filter(tangent, window_length=window, polyorder=2, axis=0, mode="interp")
    return orthonormalise(np.einsum("nij,jk->nik", rodrigues(tangent), mean))


def _toe_world_for_subject(
    cameras: Sequence[CalibratedCamera],
    toe_landmarks_by_camera: Sequence[Sequence[Sequence[np.ndarray]]] | None,
    assignment: np.ndarray,
    frames: int,
    *,
    minimum_confidence: float,
    pixel_scale: float,
) -> tuple[np.ndarray | None, dict]:
    """Triangulate the two ball-of-foot points under the pipeline's own association.

    `[frame, 2, 3]`, left then right, NaN where a frame does not resolve -- the caller
    falls back to the previous behaviour on those frames rather than inventing a foot.

    Uses `triangulate_point` at production settings, so the balls of the feet enjoy no
    gate the mapped body joints do not.
    """

    if toe_landmarks_by_camera is None:
        return None, {"status": "not_attempted", "reason": "no toe landmarks supplied"}
    out = np.full((frames, 2, 3), np.nan, dtype=np.float64)
    support_total = 0
    for frame in range(frames):
        for side in range(2):
            points = np.full((len(cameras), 2), np.nan)
            weights = np.zeros(len(cameras))
            for camera in range(len(cameras)):
                person = int(assignment[frame, camera])
                if person < 0:
                    continue
                try:
                    sample = np.asarray(
                        toe_landmarks_by_camera[camera][frame][person], dtype=np.float64
                    )
                except (IndexError, TypeError, ValueError):
                    continue
                if sample.shape != (2, 3):
                    continue
                x, y, confidence = sample[side]
                points[camera], weights[camera] = (x, y), confidence
            result = triangulate_point(
                cameras, points, weights,
                minimum_confidence=minimum_confidence, pixel_scale=pixel_scale,
            )
            if result is not None:
                out[frame, side] = result.position_world_m
                support_total += len(result.used_camera_indices)
    resolved = float(np.isfinite(out).all(axis=2).mean())
    return out, {
        "status": "solved" if resolved > 0.0 else "unresolved",
        "resolved_fraction": resolved,
        "mean_camera_support": support_total / max(1.0, 2.0 * frames),
    }


def _spine_world_for_subject(
    cameras: Sequence[CalibratedCamera],
    spine_landmarks_by_camera: Sequence[Sequence[Sequence[np.ndarray]]] | None,
    assignment: np.ndarray,
    frames: int,
    *,
    minimum_confidence: float,
    pixel_scale: float,
) -> tuple[np.ndarray | None, dict]:
    """Triangulate SOMA-77's ``Spine1`` under the pipeline's own association.

    ``[frame, 3]``, NaN where a frame does not resolve -- `_pelvis_world_frames` fills
    those by interpolation, or falls the WHOLE subject back if there are too few.

    Uses `triangulate_point` at production settings, exactly as
    :func:`_toe_world_for_subject` does, so the lower spine enjoys no gate the mapped body
    joints do not. `Spine2` is deliberately NOT consumed here: it is a child of `Spine1`
    and carries lumbar flexion, which makes it a LUMBAR direction and not a pelvis one
    (14.5 / 20.9 deg median / p95 of flexion on the fixture's squat clip). It is measured
    by the gate as a reported arm and never enters the delivery.
    """

    if spine_landmarks_by_camera is None:
        return None, {"status": "not_attempted", "reason": "no spine landmarks supplied"}
    out = np.full((frames, 3), np.nan, dtype=np.float64)
    support_total = 0
    for frame in range(frames):
        points = np.full((len(cameras), 2), np.nan)
        weights = np.zeros(len(cameras))
        for camera in range(len(cameras)):
            person = int(assignment[frame, camera])
            if person < 0:
                continue
            try:
                sample = np.asarray(
                    spine_landmarks_by_camera[camera][frame][person], dtype=np.float64
                )
            except (IndexError, TypeError, ValueError):
                continue
            if sample.shape != (3,):
                continue
            x, y, confidence = sample
            points[camera], weights[camera] = (x, y), confidence
        result = triangulate_point(
            cameras, points, weights,
            minimum_confidence=minimum_confidence, pixel_scale=pixel_scale,
        )
        if result is not None:
            out[frame] = result.position_world_m
            support_total += len(result.used_camera_indices)
    resolved = float(np.isfinite(out).all(axis=1).mean())
    return out, {
        "status": "solved" if resolved > 0.0 else "unresolved",
        "resolved_fraction": resolved,
        "mean_camera_support": support_total / max(1.0, float(frames)),
    }


def _solve_head_for_subject(
    cameras: Sequence[CalibratedCamera],
    observations_by_camera: Sequence[Sequence[dict[str, Any]]],
    head_landmarks_by_camera: Sequence[Sequence[Sequence[np.ndarray]]] | None,
    head_landmark_names: Sequence[str],
    assignment: np.ndarray,
    positions_world_z_up_m: np.ndarray,
    *,
    minimum_confidence: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Solve this subject's head, or say plainly why it was not solved.

    Returning ``None`` restores the pre-existing behaviour -- head, neck and eyes welded to
    the torso -- which is a **constant**, not a neutral default. Every path out of here
    therefore carries a status the diagnostics publish, because that constant scores at
    parity with a research reference on the obvious jitter metric while carrying no head
    information at all (docs/HEAD_ORIENTATION_MEASURED.md).
    """
    if head_landmarks_by_camera is None:
        return None, {
            "status": "not_attempted",
            "reason": "no head landmarks supplied; head is welded to the torso",
        }
    if not head_landmark_names:
        return None, {
            "status": "not_attempted",
            "reason": "head landmarks supplied without names",
        }
    from .head_orientation import HeadOrientationError, solve_head_orientation

    frames = len(observations_by_camera[0])
    marks = len(head_landmark_names)
    observations = np.full((frames, len(cameras), marks, 3), np.nan, dtype=np.float64)
    for frame in range(frames):
        for camera in range(len(cameras)):
            person = int(assignment[frame, camera])
            if person < 0:
                continue
            try:
                sample = np.asarray(
                    head_landmarks_by_camera[camera][frame][person], dtype=np.float64
                )
            except (IndexError, TypeError, ValueError):
                continue
            if sample.shape == (marks, 3):
                observations[frame, camera] = sample

    try:
        thorax = _thorax_frames(positions_world_z_up_m)
        solved = solve_head_orientation(
            cameras,
            observations,
            head_landmark_names,
            thorax_world=thorax,
            neck_origin_world_m=positions_world_z_up_m[:, JOINT_INDEX["neck"]],
            minimum_confidence=minimum_confidence,
            pixel_scale=float(observations_by_camera[0][0]["width"]) / REFERENCE_DETECTOR_WIDTH_PX,
        )
    except HeadOrientationError as error:
        return None, {"status": "fell_back_to_torso_frame", "reason": str(error)}
    except Exception as error:  # noqa: BLE001 -- deliberate, see below
        # The head is ONE joint of fifty-five. Catching only HeadOrientationError let any
        # other exception escape reconstruct_multiview, which has no handler -- so a
        # numeric degeneracy in the head solve destroyed the whole capture: root, spine,
        # arms, legs and feet along with it. Demonstrated: positions whose `root` and
        # `neck` coincide raise LinAlgError from the thorax orthonormalisation, and the
        # body build died with them. A head that cannot be solved must cost the head only.
        return None, {
            "status": "fell_back_to_torso_frame",
            "reason": f"{type(error).__name__}: {error}",
            "unexpected": True,
        }
    return solved.rotations_world, {"status": "solved", **solved.as_dict()}


def reconstruct_multiview(
    cameras: Sequence[CalibratedCamera],
    observations_by_camera: Sequence[Sequence[dict[str, Any]]],
    *,
    subject_count: int = 2,
    sample_rate_hz: int = 30,
    # The confidence floor is both a gate and, through sqrt(confidence), the only
    # channel an observation has to declare how much it should be trusted. Fixed
    # at 0.25 it confines usable confidences to [0.26, 1.0] and so caps the weight
    # ratio at 2x -- while the detector's own measured disagreement is a mixture
    # with an 11.8x ratio between its components. A caller that has a real
    # per-observation sigma cannot express it through a channel that narrow, which
    # is why this is now reachable. Default unchanged.
    minimum_confidence: float = 0.25,
    weight_before_loss: bool = False,
    # D3. Which rest skeleton the tracks are built on. "performer" (the delivery) sizes
    # the rig to THIS performer's own triangulated limb lengths; "canonical" is the
    # instrument arm that proves the plumbing is byte-stable under the canonical rest
    # (the D3 gate's bit-identity check against the pre-D3 delivery). No other value.
    rest_skeleton: str = "performer",
    # Cycle-consistent graph matching by default: measured identical to the
    # exhaustive search on every frame of the reference fixture, ~2x faster
    # there, and the only tractable option beyond two subjects -- the exhaustive
    # search is (subjects!)^cameras, which is 1,296 assignments at three
    # subjects and four cameras and 110 billion at four subjects and eight.
    associator: Callable[..., tuple[np.ndarray, float]] = associate_frame_graph,
    # Per-camera head landmarks, indexed [camera][frame][person] -> (landmark, 3) of
    # (x, y, confidence) in the same pixel space as the observations. Supplying them turns
    # the head from a CONSTANT welded to the torso into a solved orientation; omitting them
    # keeps the previous behaviour exactly, and the diagnostics say which happened. The
    # detector must emit head landmarks beyond the 19-joint contract for this to be
    # available -- SOMA-77's `landmarks_soma77` is the case it was built for.
    head_landmarks_by_camera: Sequence[Sequence[Sequence[np.ndarray]]] | None = None,
    head_landmark_names: Sequence[str] = (),
    toe_landmarks_by_camera: Sequence[Sequence[Sequence[np.ndarray]]] | None = None,
    # D7. Per-camera SOMA-77 `Spine1` landmarks, indexed [camera][frame][person] -> (3,)
    # of (x, y, confidence) in the same pixel space as the observations. Supplying them
    # gives `Hips` its own frame from the pelvis instead of the trunk line; omitting them
    # keeps the previous behaviour bit for bit, and the diagnostics say which happened.
    spine_landmarks_by_camera: Sequence[Sequence[Sequence[np.ndarray]]] | None = None,
    # D8. The occlusion repair, reachable so an instrument can run every arm through this
    # one function instead of re-implementing it. `None`/`False` turns a rule off, and with
    # all three off the smoothed output is bit-identical to the pre-D8 build -- which is
    # the "today's code" arm of the synthetic selector and a unit test.
    # The RAW array is unaffected by any of them: it is captured before they run.
    ray_pair_conditioning_ceiling_deg: float | None = RAY_PAIR_CONDITIONING_CEILING_DEG,
    reachability_reject: bool = True,
    reachability_slack_m: float = REACHABILITY_SLACK_M,
    reachability_rule: str = "reachability",
    maximum_interpolated_gap_frames: int | None = MAXIMUM_INTERPOLATED_GAP_FRAMES,
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
    # D8. Which cameras the triangulator actually USED for each retained slot -- its
    # inliers, not the views that could see it. Recorded alongside `world` and never fed
    # back into it, so the raw array is exactly what it was before this step.
    supporting_views = np.zeros(
        (subject_count, frames, len(JOINT_NAMES), len(cameras)), dtype=bool
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
    frames_without_association = 0
    # Which detection each subject was matched to, so head landmarks travel with the
    # association instead of being re-derived. Recovered by identity against the arrays the
    # associator returned -- it hands back the very rows it was given.
    head_assignment = np.full((frames, subject_count, len(cameras)), -1, dtype=np.int64)
    # Per subject: a run-global counter stamped onto every subject's report makes a
    # whole-run total look like that subject's, which is the wrong denominator on a
    # per-subject diagnostic.
    head_rows_unmatched = np.zeros(subject_count, dtype=np.int64)
    for frame in range(frames):
        detections = [
            [_person_array(person) for person in values[frame]["people"]]
            for values in observations_by_camera
        ]
        if frame_is_ambiguous(detections, subject_count):
            deferred_frames += 1
        try:
            associated, cost = associator(
                scaled_cameras,
                detections,
                subject_count=subject_count,
                previous_roots_world_m=previous_roots,
                previous_positions_world_m=previous_positions,
                previous_observations_xyc=previous_observations,
                pixel_scale=pixel_scale,
            )
        except CommercialMultiviewError:
            # A frame with too few usable detections used to abort the whole take.
            # Found by the synthetic fixture: a visibility channel that gates 5% of
            # good observations leaves some frames unmatchable, and the run raised
            # rather than degrading -- so the pipeline failed hardest exactly where
            # a detector improvement was being evaluated. One frame's worth of
            # missing association is what interpolation is for.
            frames_without_association += 1
            associated = np.full(
                (subject_count, len(scaled_cameras), len(JOINT_NAMES), 3), np.nan
            )
            cost = float("nan")
        association_costs.append(cost)
        if head_landmarks_by_camera is not None:
            for subject in range(subject_count):
                for camera in range(len(cameras)):
                    row = associated[subject, camera]
                    if not np.isfinite(row).any():
                        continue
                    hits = [
                        index
                        for index, candidate in enumerate(detections[camera])
                        if np.array_equal(row, candidate, equal_nan=True)
                    ]
                    if len(hits) == 1:
                        head_assignment[frame, subject, camera] = hits[0]
                    else:
                        head_rows_unmatched[subject] += 1
        candidate_world = np.full(
            (subject_count, len(JOINT_NAMES), 3), np.nan, dtype=np.float64
        )
        candidate_views = np.zeros(
            (subject_count, len(JOINT_NAMES), len(cameras)), dtype=bool
        )
        candidate_errors: list[list[float]] = [[] for _ in range(subject_count)]
        for subject in range(subject_count):
            for joint in range(len(JOINT_NAMES)):
                result = triangulate_point(
                    scaled_cameras,
                    associated[subject, :, joint, :2],
                    associated[subject, :, joint, 2],
                    pixel_scale=pixel_scale,
                    minimum_confidence=minimum_confidence,
                )
                if result is None:
                    continue
                candidate_world[subject, joint] = result.position_world_m
                candidate_views[subject, joint, list(result.used_camera_indices)] = True
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
            supporting_views[subject, frame] = candidate_views[subject]
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
    held_fractions: list[float] = []
    repair_reports: list[dict[str, Any]] = []
    head_reports: list[dict[str, Any]] = []
    toe_reports: list[dict[str, Any]] = []
    spine_reports: list[dict[str, Any]] = []
    pelvis_reports: list[dict[str, Any]] = []
    for subject in range(subject_count):
        # Recover single-ray joints from limb-length and temporal constraints
        # before falling back to interpolation, so an evidence-based estimate is
        # preferred to a drawn line wherever one is available.
        # D8. Withhold the slots two near-collinear rays cannot fix and the ones no body
        # could have reached, BEFORE the sequence solve, so the solve recovers them from
        # the rays it already has plus limb length and continuity. `world` is not touched:
        # what goes in is a copy, and `world.copy()` is still what this function returns as
        # the raw array.
        withheld, repair = _repair_occluded_slots(
            scaled_cameras,
            world[subject],
            supporting_views[subject],
            sample_rate_hz=sample_rate_hz,
            ray_pair_ceiling_deg=ray_pair_conditioning_ceiling_deg,
            reachability=reachability_reject,
            reachability_slack_m=reachability_slack_m,
            reachability_rule=reachability_rule,
        )
        resolved, recovered = solve_sequence_positions(
            scaled_cameras,
            withheld,
            retained_observations[subject],
            pixel_scale=pixel_scale,
            minimum_confidence=minimum_confidence,
            weight_before_loss=weight_before_loss,
        )
        recovered_counts.append(float(np.mean(recovered)))
        # D8. A gap longer than the ceiling is held on the parent instead of having a
        # straight line drawn through it. Reported, never silent.
        if maximum_interpolated_gap_frames is None:
            held_fraction = 0.0
        else:
            resolved, held_fraction = _hold_long_gaps_on_parent(
                resolved, maximum_interpolated_gap_frames)
        repair["held_joint_fraction"] = held_fraction
        repair["maximum_interpolated_gap_frames"] = maximum_interpolated_gap_frames
        repair["recovered_after_repair_joint_fraction"] = float(np.mean(recovered))
        held_fractions.append(held_fraction)
        repair_reports.append(repair)
        positions, fraction = _fill_and_smooth_positions(resolved)
        smoothed.append(positions)
        interpolated.append(fraction)
        head_rotations, head_report = _solve_head_for_subject(
            scaled_cameras,
            observations_by_camera,
            head_landmarks_by_camera,
            head_landmark_names,
            head_assignment[:, subject],
            positions,
            minimum_confidence=minimum_confidence,
        )
        head_report["unmatched_association_rows"] = int(head_rows_unmatched[subject])
        head_reports.append(head_report)
        toe_world, toe_report = _toe_world_for_subject(
            scaled_cameras,
            toe_landmarks_by_camera,
            head_assignment[:, subject],
            frames,
            minimum_confidence=minimum_confidence,
            pixel_scale=pixel_scale,
        )
        toe_reports.append(toe_report)
        spine_world, spine_report = _spine_world_for_subject(
            scaled_cameras,
            spine_landmarks_by_camera,
            head_assignment[:, subject],
            frames,
            minimum_confidence=minimum_confidence,
            pixel_scale=pixel_scale,
        )
        spine_reports.append(spine_report)
        pelvis_report: dict[str, Any] = {}
        # D3. ONE rest skeleton per performer, built from THIS performer's own
        # triangulated capture -- the same `positions` array the converter is handed and
        # the same array the build writes to disk, so an instrument that re-derives the
        # rest from the npz gets bit-identical bone lengths. The skeleton is passed in,
        # stamped on the track, and carried from here through forward kinematics,
        # validation, the ground projection, the exporter and every socket.
        if rest_skeleton == "performer":
            performer, _sizing = performer_skeleton(DETAILED_HUMANOID, positions)
        elif rest_skeleton == "canonical":
            performer = DETAILED_HUMANOID
        else:
            raise CommercialMultiviewError("rest_skeleton must be 'performer' or 'canonical'")
        track = positions_to_body_track(
            positions,
            sample_rate_hz=sample_rate_hz,
            provenance_sha256=sha256(
                f"{source_hash}:{subject}".encode("ascii")
            ).hexdigest(),
            head_world_rotations=head_rotations,
            toe_world_z_up_m=toe_world,
            spine_world_z_up_m=spine_world,
            pelvis_report_out=pelvis_report,
            skeleton=performer,
        )
        pelvis_reports.append(pelvis_report)
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
        association_objective_median=float(np.nanmedian(association_costs))
        if np.isfinite(association_costs).any()
        else float("nan"),
        frames_without_association=frames_without_association,
        interpolated_joint_fraction=float(np.mean(interpolated)),
        temporally_rejected_subject_frames=temporally_rejected_subject_frames,
        contact_frames=tuple(contacts),
        head_orientation=tuple(head_reports),
        toe_triangulation=tuple(toe_reports),
        spine_triangulation=tuple(spine_reports),
        pelvis_frame=tuple(pelvis_reports),
        held_joint_fraction=float(np.mean(held_fractions)) if held_fractions else 0.0,
        occlusion_repair=tuple(repair_reports),
    )
    # `smoothed` is post-interpolation and Savitzky-Golay filtered, so it cannot
    # measure raw triangulation noise. `world` is what triangulation actually
    # produced, NaNs intact, before any fill or head-joint fallback.
    return tuple(tracks), diagnostics, np.asarray(smoothed), world.copy()


__all__ = [
    "BODY_OBSERVATION_SCHEMA_VERSION",
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
