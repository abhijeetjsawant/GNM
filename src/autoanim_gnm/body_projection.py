"""Conservative contact derivation and bounded foot stabilization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math

import numpy as np

from .body import (
    DETAILED_HUMANOID,
    BodyTrack,
    _quaternion_multiply,
    forward_kinematics_positions,
)


def _quaternion_inverse(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    inverse = quaternion.copy()
    inverse[..., :3] *= -1.0
    return inverse / np.sum(quaternion * quaternion, axis=-1, keepdims=True)


@dataclass(frozen=True, slots=True)
class BodyProjectionDiagnostics:
    contact_frames: tuple[int, int]
    maximum_root_correction_m: float
    maximum_root_step_correction_m: float
    ground_penetration_before_m: float
    ground_penetration_after_m: float
    joint_limit_projection_applied: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "contact_frames": list(self.contact_frames),
            "maximum_root_correction_m": self.maximum_root_correction_m,
            "maximum_root_step_correction_m": self.maximum_root_step_correction_m,
            "ground_penetration_before_m": self.ground_penetration_before_m,
            "ground_penetration_after_m": self.ground_penetration_after_m,
            "joint_limit_projection_applied": self.joint_limit_projection_applied,
        }


@dataclass(frozen=True, slots=True)
class RootStabilizationDiagnostics:
    applied: bool
    horizontal_scale: float
    travel_before_m: float
    travel_after_m: float
    target_travel_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "horizontal_scale": self.horizontal_scale,
            "travel_before_m": self.travel_before_m,
            "travel_after_m": self.travel_after_m,
            "target_travel_m": self.target_travel_m,
        }


@dataclass(frozen=True, slots=True)
class VisualArmConstraintDiagnostics:
    """Evidence and error summary for video-conditioned arm correction."""

    passed: bool
    confidence_threshold: float
    sides: dict[str, dict[str, object]]
    maximum_local_correction_degrees: float
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "autoanim.visual-arm-constraint/1.0",
            "passed": self.passed,
            "confidence_threshold": self.confidence_threshold,
            "sides": self.sides,
            "maximum_local_correction_degrees": (
                self.maximum_local_correction_degrees
            ),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class VisualHandConstraintDiagnostics:
    """Rendered-camera palm and finger fidelity after video constraints."""

    passed: bool
    confidence_threshold: float
    sides: dict[str, dict[str, object]]
    maximum_local_correction_degrees: float
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "autoanim.visual-hand-constraint/1.0",
            "passed": self.passed,
            "confidence_threshold": self.confidence_threshold,
            "sides": self.sides,
            "maximum_local_correction_degrees": self.maximum_local_correction_degrees,
            "failure_reasons": list(self.failure_reasons),
        }


def constrain_restrained_root_travel(
    track: BodyTrack,
    *,
    target_travel_m: float = 0.23,
) -> tuple[BodyTrack, RootStabilizationDiagnostics]:
    """Bound standing-shot X/Z drift while preserving vertical acting motion.

    This operates before contact projection. Existing contact assertions would
    belong to a different root trajectory and therefore fail closed.
    """

    if (
        not math.isfinite(target_travel_m)
        or target_travel_m <= 0.0
        or target_travel_m > 0.25
    ):
        raise ValueError("Restrained root target must be in (0, 0.25] meters")
    if np.any(track.foot_contacts):
        raise ValueError("Restrained root stabilization must precede contact projection")
    roots = np.array(track.root_translation_m, dtype=np.float64, copy=True)
    ranges = np.ptp(roots, axis=0)
    before = float(np.linalg.norm(ranges))
    horizontal_range = float(np.hypot(ranges[0], ranges[2]))
    vertical_range = float(ranges[1])
    if before <= target_travel_m + 1e-9:
        return track, RootStabilizationDiagnostics(
            applied=False,
            horizontal_scale=1.0,
            travel_before_m=before,
            travel_after_m=before,
            target_travel_m=target_travel_m,
        )
    if vertical_range >= target_travel_m or horizontal_range <= 1e-12:
        raise ValueError("Vertical root motion alone exceeds the restrained target")
    allowed_horizontal = math.sqrt(
        max(0.0, target_travel_m**2 - vertical_range**2)
    )
    scale = min(1.0, allowed_horizontal / horizontal_range)
    anchor = roots[0, (0, 2)].copy()
    roots[:, (0, 2)] = anchor + (roots[:, (0, 2)] - anchor) * scale
    after = float(np.linalg.norm(np.ptp(roots, axis=0)))
    if after > target_travel_m + 1e-6:
        raise RuntimeError("Restrained root stabilization did not meet its target")
    provenance = sha256()
    provenance.update(b"autoanim.restrained-root-envelope/1.0\0")
    provenance.update(track.source_plan_sha256.encode("ascii"))
    provenance.update(np.float64(target_travel_m).tobytes())
    stabilized = replace(
        track,
        root_translation_m=roots.astype(np.float32),
        source_plan_sha256=provenance.hexdigest(),
    )
    return stabilized, RootStabilizationDiagnostics(
        applied=True,
        horizontal_scale=scale,
        travel_before_m=before,
        travel_after_m=after,
        target_travel_m=target_travel_m,
    )


def _world_rotations(local: np.ndarray) -> np.ndarray:
    output = np.zeros_like(local, dtype=np.float64)
    output[..., 3] = 1.0
    for index, joint in enumerate(DETAILED_HUMANOID.joints):
        if joint.parent == -1:
            output[:, index] = local[:, index]
        else:
            output[:, index] = _quaternion_multiply(
                output[:, joint.parent], local[:, index]
            )
    output /= np.linalg.norm(output, axis=2, keepdims=True)
    return output


def _unit_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{label} must be a finite non-zero vector")
    return vector / norm


def _quaternion_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest [x,y,z,w] rotation from source to target."""

    start = _unit_vector(source, label="Source direction")
    finish = _unit_vector(target, label="Target direction")
    dot = float(np.clip(np.dot(start, finish), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    if dot < -1.0 + 1e-10:
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(start)))] = 1.0
        axis = _unit_vector(np.cross(start, basis), label="Opposite-vector axis")
        return np.asarray((axis[0], axis[1], axis[2], 0.0), dtype=np.float64)
    output = np.concatenate((np.cross(start, finish), (1.0 + dot,)))
    return output / np.linalg.norm(output)


def _smooth_screen_directions(
    start_xyc: np.ndarray,
    end_xyc: np.ndarray,
    *,
    confidence_threshold: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return confidence-weighted image directions in body X/Y coordinates."""

    confidence = np.minimum(start_xyc[:, 2], end_xyc[:, 2])
    # VitPose values are heatmap amplitudes, not calibrated probabilities, and
    # valid detections can be slightly greater than one. Preserve the raw value
    # for visibility checks but cap its influence on temporal weighting.
    weight_confidence = np.clip(confidence, 0.0, 1.0)
    delta = end_xyc[:, :2] - start_xyc[:, :2]
    # Image Y grows down while canonical Y grows up. The front-facing Blender
    # review camera sees provider-world -X as screen-right.
    directions = np.stack((-delta[:, 0], -delta[:, 1]), axis=1)
    lengths = np.linalg.norm(directions, axis=1)
    valid = (confidence >= confidence_threshold) & (lengths > 1e-6)
    normalized = np.zeros_like(directions)
    normalized[valid] = directions[valid] / lengths[valid, None]
    smoothed = np.zeros_like(directions)
    kernel = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate((kernel, kernel[-2::-1]))
    for frame in range(len(confidence)):
        begin = max(0, frame - radius)
        end = min(len(confidence), frame + radius + 1)
        local_kernel = kernel[
            begin - (frame - radius) : len(kernel) - ((frame + radius + 1) - end)
        ]
        local_valid = valid[begin:end]
        weights = local_kernel * weight_confidence[begin:end] ** 2 * local_valid
        if float(np.sum(weights)) <= 1e-12:
            continue
        direction = np.sum(normalized[begin:end] * weights[:, None], axis=0)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            smoothed[frame] = direction / norm
    return smoothed, confidence


def _screen_angle_errors_degrees(
    positions: np.ndarray,
    observed: np.ndarray,
    *,
    joint_pairs: tuple[tuple[int, int], tuple[int, int]],
    observed_pairs: tuple[tuple[int, int], tuple[int, int]],
    confidence_threshold: float,
) -> np.ndarray:
    errors: list[np.ndarray] = []
    for (joint, child), (start, end) in zip(
        joint_pairs, observed_pairs, strict=True
    ):
        observed_delta = observed[:, end, :2] - observed[:, start, :2]
        observed_xy = np.stack(
            (observed_delta[:, 0], -observed_delta[:, 1]), axis=1
        )
        predicted_delta = positions[:, child] - positions[:, joint]
        predicted_xy = np.stack(
            (-predicted_delta[:, 0], predicted_delta[:, 1]), axis=1
        )
        visible = (
            (observed[:, start, 2] >= confidence_threshold)
            & (observed[:, end, 2] >= confidence_threshold)
            & (np.linalg.norm(observed_xy, axis=1) > 1e-6)
            & (np.linalg.norm(predicted_xy, axis=1) > 1e-8)
        )
        if not np.any(visible):
            continue
        observed_unit = observed_xy[visible] / np.linalg.norm(
            observed_xy[visible], axis=1, keepdims=True
        )
        predicted_unit = predicted_xy[visible] / np.linalg.norm(
            predicted_xy[visible], axis=1, keepdims=True
        )
        cosine = np.clip(np.sum(observed_unit * predicted_unit, axis=1), -1.0, 1.0)
        errors.append(np.rad2deg(np.arccos(cosine)))
    return np.concatenate(errors) if errors else np.zeros(0, dtype=np.float64)


def _provider_arm_state(
    rotations: np.ndarray,
    local_rest_matrices: np.ndarray,
    parents: np.ndarray,
    alignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body-delta world rotations, provider world rotations and origins."""

    from scipy.spatial.transform import Rotation

    frame_count, joint_count = rotations.shape[:2]
    rest = np.asarray(local_rest_matrices, dtype=np.float64)
    hierarchy = np.asarray(parents, dtype=np.int64)
    rest_world = np.zeros_like(rest)
    for joint, parent in enumerate(hierarchy.tolist()):
        rest_world[joint] = rest[joint] if parent == -1 else rest_world[parent] @ rest[joint]
    rest_world_quaternions = Rotation.from_matrix(rest_world[:, :3, :3]).as_quat()
    base = _quaternion_multiply(alignment, rest_world_quaternions)
    target_world = _world_rotations(rotations)
    animated_world = _quaternion_multiply(target_world, base[None, :, :])
    positions = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    translations = rest[:, :3, 3]
    for joint, parent in enumerate(hierarchy.tolist()):
        if parent == -1:
            positions[:, joint] = translations[joint]
        else:
            positions[:, joint] = positions[:, parent] + _rotate_provider_vector(
                animated_world[:, parent],
                np.broadcast_to(translations[joint], (frame_count, 3)),
            )
    return target_world, animated_world, positions


def _rotate_provider_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = np.asarray(quaternion, dtype=np.float64)[..., :3]
    values = np.asarray(vector, dtype=np.float64)
    uv = np.cross(xyz, values)
    uuv = np.cross(xyz, uv)
    return values + 2.0 * (
        quaternion[..., 3, None] * uv + uuv
    )


def constrain_arms_to_video_keypoints(
    track: BodyTrack,
    observed_keypoints_xyc: np.ndarray,
    *,
    provider_local_rest_matrices: np.ndarray | None = None,
    provider_parents: np.ndarray | None = None,
    confidence_threshold: float = 0.65,
    smoothing_radius_frames: int = 2,
    correction_strength: float = 1.0,
    minimum_visible_forward_component: float = 0.10,
    mean_angle_limit_degrees: float = 12.0,
    p95_angle_limit_degrees: float = 20.0,
) -> tuple[BodyTrack, VisualArmConstraintDiagnostics]:
    """Correct arm swing from trusted 2D video evidence while preserving depth.

    GEM-X supplies torso pose, twist and temporal motion. VitPose supplies
    the image-plane shoulder/elbow/wrist directions that GEM-X can lose when its
    learned pose prior contradicts a gesture. Only upper-arm and forearm swing
    are changed; bone lengths, root motion, torso motion, twist, face and fingers
    remain owned by their existing tracks. For a declared front-facing capture,
    a high-confidence visible arm is kept on the camera-facing side of the
    torso; this prevents a monocular depth prior from hiding it behind the body.
    """

    if track.joint_names != DETAILED_HUMANOID.names:
        raise ValueError("Visual arm constraints require AutoAnim-55")
    observed = np.asarray(observed_keypoints_xyc, dtype=np.float64)
    frame_count = len(track.ticks)
    if observed.shape != (frame_count, 77, 3):
        raise ValueError("Observed SOMA keypoints must have shape [frames, 77, 3]")
    if not np.isfinite(observed).all():
        raise ValueError("Observed SOMA keypoints must be finite")
    if np.any(observed[..., 2] < 0.0):
        raise ValueError("Observed SOMA confidence must be nonnegative")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("Confidence threshold must be in (0, 1]")
    if smoothing_radius_frames < 0 or smoothing_radius_frames > 12:
        raise ValueError("Smoothing radius must be between 0 and 12 frames")
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("Correction strength must be in [0, 1]")
    if not 0.0 <= minimum_visible_forward_component < 0.75:
        raise ValueError("Visible forward component must be in [0, 0.75)")
    if mean_angle_limit_degrees <= 0.0 or p95_angle_limit_degrees <= 0.0:
        raise ValueError("Angle limits must be positive")

    joint_count = len(DETAILED_HUMANOID.joints)
    if (provider_local_rest_matrices is None) != (provider_parents is None):
        raise ValueError("Provider rest matrices and parents must be supplied together")
    if provider_local_rest_matrices is None:
        provider_rest = np.broadcast_to(
            np.eye(4, dtype=np.float64), (joint_count, 4, 4)
        ).copy()
        for joint, spec in enumerate(DETAILED_HUMANOID.joints):
            provider_rest[joint, :3, 3] = spec.rest_translation_m
        provider_hierarchy = np.asarray(
            [joint.parent for joint in DETAILED_HUMANOID.joints], dtype=np.int64
        )
        provider_alignment = np.zeros((joint_count, 4), dtype=np.float64)
        provider_alignment[:, 3] = 1.0
    else:
        provider_rest = np.asarray(provider_local_rest_matrices, dtype=np.float64)
        provider_hierarchy = np.asarray(provider_parents, dtype=np.int64)
        if provider_rest.shape != (joint_count, 4, 4):
            raise ValueError("Provider rest matrices must have shape [55,4,4]")
        if provider_hierarchy.shape != (joint_count,):
            raise ValueError("Provider parents must have shape [55]")
        if not np.isfinite(provider_rest).all():
            raise ValueError("Provider rest matrices must be finite")
        from .body_export import _canonical_arm_bind_alignment

        provider_alignment = _canonical_arm_bind_alignment(
            provider_rest,
            provider_hierarchy,
            skeleton=DETAILED_HUMANOID,
        )

    # SOMA and the MPFB provider use opposite anatomical labels at the retarget
    # boundary. Bind by visible screen side: source-left/image-right maps to the
    # provider arm rendered on image-right, and vice versa.
    side_specs = {
        "source_left": {
            "observed": ((12, 13), (13, 14)),
            "target": (
                (
                    DETAILED_HUMANOID.index("RightUpperArm"),
                    DETAILED_HUMANOID.index("RightLowerArm"),
                ),
                (
                    DETAILED_HUMANOID.index("RightLowerArm"),
                    DETAILED_HUMANOID.index("RightHand"),
                ),
            ),
        },
        "source_right": {
            "observed": ((40, 41), (41, 42)),
            "target": (
                (
                    DETAILED_HUMANOID.index("LeftUpperArm"),
                    DETAILED_HUMANOID.index("LeftLowerArm"),
                ),
                (
                    DETAILED_HUMANOID.index("LeftLowerArm"),
                    DETAILED_HUMANOID.index("LeftHand"),
                ),
            ),
        },
    }
    rotations = np.array(track.local_rotations_xyzw, dtype=np.float64, copy=True)
    original = rotations.copy()
    _, _, before_positions = _provider_arm_state(
        rotations, provider_rest, provider_hierarchy, provider_alignment
    )
    side_metrics: dict[str, dict[str, object]] = {}

    for side, spec in side_specs.items():
        observed_pairs = spec["observed"]
        target_pairs = spec["target"]
        before_error = _screen_angle_errors_degrees(
            before_positions,
            observed,
            joint_pairs=target_pairs,
            observed_pairs=observed_pairs,
            confidence_threshold=confidence_threshold,
        )
        applied_frames = 0
        for (observed_start, observed_end), (joint, child) in zip(
            observed_pairs, target_pairs, strict=True
        ):
            desired_screen, confidence = _smooth_screen_directions(
                observed[:, observed_start],
                observed[:, observed_end],
                confidence_threshold=confidence_threshold,
                radius=smoothing_radius_frames,
            )
            target_world, animated_world, positions = _provider_arm_state(
                rotations, provider_rest, provider_hierarchy, provider_alignment
            )
            parent = int(provider_hierarchy[joint])
            from scipy.spatial.transform import Rotation

            rest_world = np.zeros_like(provider_rest)
            for rest_joint, rest_parent in enumerate(provider_hierarchy.tolist()):
                rest_world[rest_joint] = (
                    provider_rest[rest_joint]
                    if rest_parent == -1
                    else rest_world[rest_parent] @ provider_rest[rest_joint]
                )
            rest_world_quaternion = Rotation.from_matrix(
                rest_world[:, :3, :3]
            ).as_quat()
            base_world = _quaternion_multiply(
                provider_alignment, rest_world_quaternion
            )
            prior_bones = before_positions[:, child] - before_positions[:, joint]
            prior_bones /= np.linalg.norm(prior_bones, axis=1, keepdims=True)
            for frame in range(frame_count):
                if (
                    confidence[frame] < confidence_threshold
                    or np.linalg.norm(desired_screen[frame]) <= 1e-12
                ):
                    continue
                current_bone = positions[frame, child] - positions[frame, joint]
                current_direction = _unit_vector(
                    current_bone, label="Current arm bone"
                )
                # Monocular GEM-X often gets the sign of arm depth wrong while
                # its magnitude still contains useful foreshortening. Preserve
                # that magnitude and flip only toward the declared camera side.
                depth = -float(
                    np.clip(
                        max(
                            abs(prior_bones[frame, 2]),
                            minimum_visible_forward_component,
                        ),
                        0.0,
                        0.98,
                    )
                )
                planar = math.sqrt(max(1e-12, 1.0 - depth * depth))
                weight = float(np.clip(confidence[frame], 0.0, 1.0))
                weight *= correction_strength
                current_screen = current_direction[:2]
                current_screen_norm = float(np.linalg.norm(current_screen))
                if current_screen_norm <= 1e-12:
                    current_screen = desired_screen[frame]
                else:
                    current_screen = current_screen / current_screen_norm
                blended_screen = _unit_vector(
                    (1.0 - weight) * current_screen
                    + weight * desired_screen[frame],
                    label="Blended screen arm direction",
                )
                blended = np.asarray(
                    (
                        blended_screen[0] * planar,
                        blended_screen[1] * planar,
                        depth,
                    ),
                    dtype=np.float64,
                )
                swing = _quaternion_from_to(current_direction, blended)
                desired_animated_world = _quaternion_multiply(
                    swing, animated_world[frame, joint]
                )
                desired_target_world = _quaternion_multiply(
                    desired_animated_world,
                    _quaternion_inverse(base_world[joint]),
                )
                local = _quaternion_multiply(
                    _quaternion_inverse(target_world[frame, parent]),
                    desired_target_world,
                )
                rotations[frame, joint] = local / np.linalg.norm(local)
                applied_frames += 1

        _, _, after_positions = _provider_arm_state(
            rotations, provider_rest, provider_hierarchy, provider_alignment
        )
        after_error = _screen_angle_errors_degrees(
            after_positions,
            observed,
            joint_pairs=target_pairs,
            observed_pairs=observed_pairs,
            confidence_threshold=confidence_threshold,
        )
        side_metrics[side] = {
            "applied_segment_frames": applied_frames,
            "visible_segment_frames": int(after_error.size),
            "mean_angle_error_before_degrees": (
                float(np.mean(before_error)) if before_error.size else None
            ),
            "p95_angle_error_before_degrees": (
                float(np.percentile(before_error, 95.0)) if before_error.size else None
            ),
            "mean_angle_error_after_degrees": (
                float(np.mean(after_error)) if after_error.size else None
            ),
            "p95_angle_error_after_degrees": (
                float(np.percentile(after_error, 95.0)) if after_error.size else None
            ),
        }

    # Quaternion signs are representationally equivalent but discontinuous
    # signs make downstream interpolation take the long path.
    for joint in range(rotations.shape[1]):
        for frame in range(1, frame_count):
            if np.dot(rotations[frame, joint], rotations[frame - 1, joint]) < 0.0:
                rotations[frame, joint] *= -1.0

    delta = _quaternion_multiply(rotations, _quaternion_inverse(original))
    delta_w = np.clip(np.abs(delta[..., 3]), 0.0, 1.0)
    maximum_correction = float(np.rad2deg(2.0 * np.arccos(delta_w)).max())
    failures: list[str] = []
    for side, metrics in side_metrics.items():
        if metrics["visible_segment_frames"] == 0:
            failures.append(f"{side}_no_visible_segments")
            continue
        if metrics["mean_angle_error_after_degrees"] > mean_angle_limit_degrees:
            failures.append(f"{side}_mean_angle_error")
        if metrics["p95_angle_error_after_degrees"] > p95_angle_limit_degrees:
            failures.append(f"{side}_p95_angle_error")

    provenance = sha256()
    provenance.update(b"autoanim.visual-arm-constraint/1.0\0")
    provenance.update(track.source_plan_sha256.encode("ascii"))
    provenance.update(np.ascontiguousarray(observed).tobytes())
    provenance.update(np.float64(confidence_threshold).tobytes())
    provenance.update(np.int64(smoothing_radius_frames).tobytes())
    provenance.update(np.float64(correction_strength).tobytes())
    provenance.update(np.float64(minimum_visible_forward_component).tobytes())
    constrained = replace(
        track,
        local_rotations_xyzw=rotations.astype(np.float32),
        source_plan_sha256=provenance.hexdigest(),
    )
    diagnostics = VisualArmConstraintDiagnostics(
        passed=not failures,
        confidence_threshold=confidence_threshold,
        sides=side_metrics,
        maximum_local_correction_degrees=maximum_correction,
        failure_reasons=tuple(failures),
    )
    return constrained, diagnostics


def _hand_side_specs() -> dict[str, dict[str, object]]:
    finger_suffixes = {
        "Thumb": ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal"),
        "Index": ("IndexProximal", "IndexIntermediate", "IndexDistal"),
        "Middle": ("MiddleProximal", "MiddleIntermediate", "MiddleDistal"),
        "Ring": ("RingProximal", "RingIntermediate", "RingDistal"),
        "Little": ("LittleProximal", "LittleIntermediate", "LittleDistal"),
    }
    source = {
        "source_left": {
            "hand": 14,
            "palm_roots": (15, 20, 25, 30, 35),
            "fingers": {
                "Thumb": (15, 16, 17),
                "Index": (20, 21, 22),
                "Middle": (25, 26, 27),
                "Ring": (30, 31, 32),
                "Little": (35, 36, 37),
            },
            "target_side": "Right",
        },
        "source_right": {
            "hand": 42,
            "palm_roots": (43, 48, 53, 58, 63),
            "fingers": {
                "Thumb": (43, 44, 45),
                "Index": (48, 49, 50),
                "Middle": (53, 54, 55),
                "Ring": (58, 59, 60),
                "Little": (63, 64, 65),
            },
            "target_side": "Left",
        },
    }
    output: dict[str, dict[str, object]] = {}
    for label, value in source.items():
        target_side = str(value["target_side"])
        target_hand = DETAILED_HUMANOID.index(f"{target_side}Hand")
        target_roots = tuple(
            DETAILED_HUMANOID.index(f"{target_side}{name}")
            for name in (
                "ThumbMetacarpal",
                "IndexProximal",
                "MiddleProximal",
                "RingProximal",
                "LittleProximal",
            )
        )
        finger_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for finger, observed_chain in value["fingers"].items():
            target_chain = tuple(
                DETAILED_HUMANOID.index(f"{target_side}{suffix}")
                for suffix in finger_suffixes[finger]
            )
            finger_pairs.extend(
                zip(
                    zip(observed_chain[:-1], observed_chain[1:]),
                    zip(target_chain[:-1], target_chain[1:]),
                    strict=True,
                )
            )
        output[label] = {
            "hand": int(value["hand"]),
            "palm_observed_pairs": tuple(
                (int(value["hand"]), int(root)) for root in value["palm_roots"]
            ),
            "palm_target_pairs": tuple(
                (target_hand, root) for root in target_roots
            ),
            "finger_pairs": tuple(finger_pairs),
        }
    return output


def _provider_base_world(
    provider_rest: np.ndarray,
    provider_hierarchy: np.ndarray,
    provider_alignment: np.ndarray,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rest_world = np.zeros_like(provider_rest)
    for joint, parent in enumerate(provider_hierarchy.tolist()):
        rest_world[joint] = (
            provider_rest[joint]
            if parent == -1
            else rest_world[parent] @ provider_rest[joint]
        )
    rest_world_quaternion = Rotation.from_matrix(rest_world[:, :3, :3]).as_quat()
    return _quaternion_multiply(provider_alignment, rest_world_quaternion)


def _solve_provider_joint_world(
    rotations: np.ndarray,
    *,
    frame: int,
    joint: int,
    desired_animated_world: np.ndarray,
    target_world: np.ndarray,
    base_world: np.ndarray,
    parents: np.ndarray,
) -> None:
    parent = int(parents[joint])
    desired_target_world = _quaternion_multiply(
        desired_animated_world, _quaternion_inverse(base_world[joint])
    )
    local = (
        desired_target_world
        if parent == -1
        else _quaternion_multiply(
            _quaternion_inverse(target_world[frame, parent]),
            desired_target_world,
        )
    )
    rotations[frame, joint] = local / np.linalg.norm(local)


def constrain_hands_to_video_keypoints(
    track: BodyTrack,
    observed_keypoints_xyc: np.ndarray,
    *,
    provider_local_rest_matrices: np.ndarray,
    provider_parents: np.ndarray,
    confidence_threshold: float = 0.65,
    confidence_full: float = 0.82,
    smoothing_radius_frames: int = 1,
    palm_mean_angle_limit_degrees: float = 15.0,
    palm_p95_angle_limit_degrees: float = 30.0,
    finger_mean_angle_limit_degrees: float = 18.0,
    finger_p95_angle_limit_degrees: float = 30.0,
) -> tuple[BodyTrack, VisualHandConstraintDiagnostics]:
    """Fit visible palm and finger directions in the actual provider bind space."""

    from scipy.spatial.transform import Rotation

    if track.joint_names != DETAILED_HUMANOID.names:
        raise ValueError("Visual hand constraints require AutoAnim-55")
    observed = np.asarray(observed_keypoints_xyc, dtype=np.float64)
    frame_count = len(track.ticks)
    if observed.shape != (frame_count, 77, 3) or not np.isfinite(observed).all():
        raise ValueError("Observed SOMA keypoints must be finite [frames,77,3]")
    if np.any(observed[..., 2] < 0.0):
        raise ValueError("Observed SOMA confidence must be nonnegative")
    if not 0.0 < confidence_threshold < confidence_full <= 1.0:
        raise ValueError("Hand confidence thresholds are invalid")
    if smoothing_radius_frames < 0 or smoothing_radius_frames > 12:
        raise ValueError("Smoothing radius must be between 0 and 12 frames")

    joint_count = len(DETAILED_HUMANOID.joints)
    provider_rest = np.asarray(provider_local_rest_matrices, dtype=np.float64)
    hierarchy = np.asarray(provider_parents, dtype=np.int64)
    if provider_rest.shape != (joint_count, 4, 4) or hierarchy.shape != (joint_count,):
        raise ValueError("Provider hand skeleton must contain 55 joints")
    if not np.isfinite(provider_rest).all():
        raise ValueError("Provider rest matrices must be finite")
    from .body_export import _canonical_arm_bind_alignment

    alignment = _canonical_arm_bind_alignment(
        provider_rest, hierarchy, skeleton=DETAILED_HUMANOID
    )
    base_world = _provider_base_world(provider_rest, hierarchy, alignment)
    rotations = np.array(track.local_rotations_xyzw, dtype=np.float64, copy=True)
    original = rotations.copy()
    _, _, initial_positions = _provider_arm_state(
        rotations, provider_rest, hierarchy, alignment
    )
    side_metrics: dict[str, dict[str, object]] = {}
    failures: list[str] = []

    for side, spec in _hand_side_specs().items():
        palm_observed = spec["palm_observed_pairs"]
        palm_target = spec["palm_target_pairs"]
        finger_pairs = spec["finger_pairs"]
        # The target rig has one rigid Hand control and no independent
        # non-thumb metacarpal splay. Gate the palm frame it can represent:
        # wrist→middle-root longitudinal plus little→index transverse axes.
        # All five wrist rays still participate in the fit below.
        palm_metric_observed = (
            palm_observed[2],
            (palm_observed[4][1], palm_observed[1][1]),
        )
        palm_metric_target = (
            palm_target[2],
            (palm_target[4][1], palm_target[1][1]),
        )
        palm_before = _screen_angle_errors_degrees(
            initial_positions,
            observed,
            joint_pairs=palm_metric_target,
            observed_pairs=palm_metric_observed,
            confidence_threshold=confidence_threshold,
        )
        finger_before = _screen_angle_errors_degrees(
            initial_positions,
            observed,
            joint_pairs=tuple(target for _, target in finger_pairs),
            observed_pairs=tuple(source for source, _ in finger_pairs),
            confidence_threshold=confidence_threshold,
        )

        # Palm is a rigid frame. Align five visible rays at once so wrist roll
        # and finger-root placement agree rather than fighting per-finger IK.
        target_hand = palm_target[0][0]
        target_world, animated_world, positions = _provider_arm_state(
            rotations, provider_rest, hierarchy, alignment
        )
        for frame in range(frame_count):
            current_vectors: list[np.ndarray] = []
            desired_vectors: list[np.ndarray] = []
            weights: list[float] = []
            for (source_hand, source_root), (_, target_root) in zip(
                palm_observed, palm_target, strict=True
            ):
                confidence = min(
                    observed[frame, source_hand, 2],
                    observed[frame, source_root, 2],
                )
                source_delta = (
                    observed[frame, source_root, :2]
                    - observed[frame, source_hand, :2]
                )
                source_screen = np.asarray((-source_delta[0], -source_delta[1]))
                if confidence < confidence_threshold or np.linalg.norm(source_screen) <= 1e-6:
                    continue
                current = _unit_vector(
                    positions[frame, target_root] - positions[frame, target_hand],
                    label="Provider palm ray",
                )
                screen = _unit_vector(source_screen, label="Observed palm ray")
                depth = float(np.clip(current[2], -0.98, 0.98))
                planar = math.sqrt(max(1e-12, 1.0 - depth * depth))
                desired = np.asarray((screen[0] * planar, screen[1] * planar, depth))
                current_vectors.append(current)
                desired_vectors.append(desired)
                bounded_confidence = float(np.clip(confidence, 0.0, 1.0))
                weights.append(bounded_confidence * bounded_confidence)
            if len(current_vectors) < 3:
                continue
            delta_rotation, _ = Rotation.align_vectors(
                np.asarray(desired_vectors),
                np.asarray(current_vectors),
                weights=np.asarray(weights),
            )
            mean_confidence = float(math.sqrt(np.mean(weights)))
            gain = float(
                np.clip(
                    (mean_confidence - confidence_threshold)
                    / (confidence_full - confidence_threshold),
                    0.0,
                    1.0,
                )
            )
            gain = gain * gain * (3.0 - 2.0 * gain)
            delta_rotation = Rotation.from_rotvec(delta_rotation.as_rotvec() * gain)
            desired_world = _quaternion_multiply(
                delta_rotation.as_quat(), animated_world[frame, target_hand]
            )
            _solve_provider_joint_world(
                rotations,
                frame=frame,
                joint=target_hand,
                desired_animated_world=desired_world,
                target_world=target_world,
                base_world=base_world,
                parents=hierarchy,
            )

        # The regularized 3D fit above preserves the provider's monocular
        # depth, but that prior can leave a small common in-plane residual on
        # all palm rays. Remove only that shared roll around camera/world Z;
        # this cannot change depth or invent a palm-normal solution from 2D.
        target_world, animated_world, positions = _provider_arm_state(
            rotations, provider_rest, hierarchy, alignment
        )
        for frame in range(frame_count):
            residual_angles: list[float] = []
            residual_weights: list[float] = []
            for (source_hand, source_root), (_, target_root) in zip(
                palm_observed, palm_target, strict=True
            ):
                confidence = min(
                    observed[frame, source_hand, 2],
                    observed[frame, source_root, 2],
                )
                observed_delta = (
                    observed[frame, source_root, :2]
                    - observed[frame, source_hand, :2]
                )
                desired_world_xy = np.asarray(
                    (-observed_delta[0], -observed_delta[1]), dtype=np.float64
                )
                current_world_xy = (
                    positions[frame, target_root, :2]
                    - positions[frame, target_hand, :2]
                )
                if (
                    confidence < confidence_threshold
                    or np.linalg.norm(desired_world_xy) <= 1e-6
                    or np.linalg.norm(current_world_xy) <= 1e-8
                ):
                    continue
                desired_world_xy /= np.linalg.norm(desired_world_xy)
                current_world_xy /= np.linalg.norm(current_world_xy)
                residual_angles.append(
                    math.atan2(
                        current_world_xy[0] * desired_world_xy[1]
                        - current_world_xy[1] * desired_world_xy[0],
                        float(np.dot(current_world_xy, desired_world_xy)),
                    )
                )
                bounded = float(np.clip(confidence, 0.0, 1.0))
                residual_weights.append(bounded * bounded)
            if len(residual_angles) < 3:
                continue
            angles = np.asarray(residual_angles)
            weights_array = np.asarray(residual_weights)
            roll = math.atan2(
                float(np.sum(weights_array * np.sin(angles))),
                float(np.sum(weights_array * np.cos(angles))),
            )
            roll = float(np.clip(roll, -math.radians(25.0), math.radians(25.0)))
            desired_world = _quaternion_multiply(
                Rotation.from_rotvec((0.0, 0.0, roll)).as_quat(),
                animated_world[frame, target_hand],
            )
            _solve_provider_joint_world(
                rotations,
                frame=frame,
                joint=target_hand,
                desired_animated_world=desired_world,
                target_world=target_world,
                base_world=base_world,
                parents=hierarchy,
            )

        # Finger phalanges are solved parent-before-child after the palm frame.
        for (source_start, source_end), (joint, child) in finger_pairs:
            desired_screen, confidence = _smooth_screen_directions(
                observed[:, source_start],
                observed[:, source_end],
                confidence_threshold=confidence_threshold,
                radius=smoothing_radius_frames,
            )
            target_world, animated_world, positions = _provider_arm_state(
                rotations, provider_rest, hierarchy, alignment
            )
            prior = positions[:, child] - positions[:, joint]
            prior /= np.linalg.norm(prior, axis=1, keepdims=True)
            for frame in range(frame_count):
                if confidence[frame] < confidence_threshold or np.linalg.norm(desired_screen[frame]) <= 1e-12:
                    continue
                current = _unit_vector(
                    positions[frame, child] - positions[frame, joint],
                    label="Provider finger segment",
                )
                depth = float(np.clip(prior[frame, 2], -0.98, 0.98))
                planar = math.sqrt(max(1e-12, 1.0 - depth * depth))
                desired = np.asarray(
                    (
                        desired_screen[frame, 0] * planar,
                        desired_screen[frame, 1] * planar,
                        depth,
                    )
                )
                raw_gain = float(
                    np.clip(
                        (confidence[frame] - confidence_threshold)
                        / (confidence_full - confidence_threshold),
                        0.0,
                        1.0,
                    )
                )
                gain = raw_gain * raw_gain * (3.0 - 2.0 * raw_gain)
                blended = _unit_vector(
                    (1.0 - gain) * current + gain * desired,
                    label="Blended finger segment",
                )
                swing = _quaternion_from_to(current, blended)
                desired_world = _quaternion_multiply(
                    swing, animated_world[frame, joint]
                )
                _solve_provider_joint_world(
                    rotations,
                    frame=frame,
                    joint=joint,
                    desired_animated_world=desired_world,
                    target_world=target_world,
                    base_world=base_world,
                    parents=hierarchy,
                )

        _, _, final_positions = _provider_arm_state(
            rotations, provider_rest, hierarchy, alignment
        )
        palm_after = _screen_angle_errors_degrees(
            final_positions,
            observed,
            joint_pairs=palm_metric_target,
            observed_pairs=palm_metric_observed,
            confidence_threshold=confidence_threshold,
        )
        finger_after = _screen_angle_errors_degrees(
            final_positions,
            observed,
            joint_pairs=tuple(target for _, target in finger_pairs),
            observed_pairs=tuple(source for source, _ in finger_pairs),
            confidence_threshold=confidence_threshold,
        )
        def angle_stat(values: np.ndarray, percentile: bool = False) -> float | None:
            if values.size == 0:
                return None
            if percentile:
                return float(np.percentile(values, 95.0))
            return float(np.mean(values))

        metrics = {
            "observation_status": (
                "constrained"
                if palm_after.size or finger_after.size
                else "unobserved_passthrough"
            ),
            "palm_visible_segment_frames": int(palm_after.size),
            "finger_visible_segment_frames": int(finger_after.size),
            "palm_mean_angle_before_degrees": angle_stat(palm_before),
            "palm_p95_angle_before_degrees": angle_stat(palm_before, True),
            "palm_mean_angle_after_degrees": angle_stat(palm_after),
            "palm_p95_angle_after_degrees": angle_stat(palm_after, True),
            "finger_mean_angle_before_degrees": angle_stat(finger_before),
            "finger_p95_angle_before_degrees": angle_stat(finger_before, True),
            "finger_mean_angle_after_degrees": angle_stat(finger_after),
            "finger_p95_angle_after_degrees": angle_stat(finger_after, True),
        }
        side_metrics[side] = metrics
        if (
            metrics["palm_mean_angle_after_degrees"] is not None
            and metrics["palm_mean_angle_after_degrees"]
            > palm_mean_angle_limit_degrees
        ):
            failures.append(f"{side}_palm_mean_angle_error")
        if (
            metrics["palm_p95_angle_after_degrees"] is not None
            and metrics["palm_p95_angle_after_degrees"]
            > palm_p95_angle_limit_degrees
        ):
            failures.append(f"{side}_palm_p95_angle_error")
        if (
            metrics["finger_mean_angle_after_degrees"] is not None
            and metrics["finger_mean_angle_after_degrees"]
            > finger_mean_angle_limit_degrees
        ):
            failures.append(f"{side}_finger_mean_angle_error")
        if (
            metrics["finger_p95_angle_after_degrees"] is not None
            and metrics["finger_p95_angle_after_degrees"]
            > finger_p95_angle_limit_degrees
        ):
            failures.append(f"{side}_finger_p95_angle_error")

    for joint in range(rotations.shape[1]):
        for frame in range(1, frame_count):
            if np.dot(rotations[frame, joint], rotations[frame - 1, joint]) < 0.0:
                rotations[frame, joint] *= -1.0
    delta = _quaternion_multiply(rotations, _quaternion_inverse(original))
    maximum_correction = float(
        np.rad2deg(2.0 * np.arccos(np.clip(np.abs(delta[..., 3]), 0.0, 1.0))).max()
    )
    provenance = sha256()
    provenance.update(b"autoanim.visual-hand-constraint/1.0\0")
    provenance.update(track.source_plan_sha256.encode("ascii"))
    provenance.update(np.ascontiguousarray(observed).tobytes())
    provenance.update(np.float64(confidence_threshold).tobytes())
    provenance.update(np.float64(confidence_full).tobytes())
    constrained = replace(
        track,
        local_rotations_xyzw=rotations.astype(np.float32),
        source_plan_sha256=provenance.hexdigest(),
    )
    return constrained, VisualHandConstraintDiagnostics(
        passed=not failures,
        confidence_threshold=confidence_threshold,
        sides=side_metrics,
        maximum_local_correction_degrees=maximum_correction,
        failure_reasons=tuple(failures),
    )


def _runs(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    for frame in range(len(mask) + 1):
        active = frame < len(mask) and bool(mask[frame])
        if active and start is None:
            start = frame
        elif not active and start is not None:
            if frame - start >= minimum_frames:
                output.append((start, frame))
            start = None
    return output


def project_generated_foot_contacts(
    track: BodyTrack,
    *,
    velocity_threshold_m_per_s: float = 0.10,
    ground_band_m: float = 0.035,
    minimum_contact_frames: int = 3,
    maximum_root_correction_m: float = 0.05,
) -> tuple[BodyTrack, BodyProjectionDiagnostics]:
    """Derive one-foot-at-a-time contacts and enforce them without touching GNM."""

    if track.joint_names != DETAILED_HUMANOID.names:
        raise ValueError("Generated contact projection requires AutoAnim-55")
    rotations = np.array(track.local_rotations_xyzw, dtype=np.float64, copy=True)
    roots = np.array(track.root_translation_m, dtype=np.float64, copy=True)
    positions = forward_kinematics_positions(
        roots, rotations, skeleton=DETAILED_HUMANOID
    ).astype(np.float64)
    foot_indices = (
        (
            DETAILED_HUMANOID.index("LeftFoot"),
            DETAILED_HUMANOID.index("LeftToes"),
        ),
        (
            DETAILED_HUMANOID.index("RightFoot"),
            DETAILED_HUMANOID.index("RightToes"),
        ),
    )
    dt = np.diff(track.ticks).astype(np.float64) / track.ticks_per_second
    speeds = np.full((len(track.ticks), 2), np.inf, dtype=np.float64)
    heights = np.zeros((len(track.ticks), 2), dtype=np.float64)
    for side, indices in enumerate(foot_indices):
        points = positions[:, indices]
        segment_speed = np.max(
            np.linalg.norm(np.diff(points, axis=0), axis=2) / dt[:, None], axis=1
        )
        speeds[1:, side] = segment_speed
        speeds[0, side] = segment_speed[0]
        heights[:, side] = np.min(points[:, :, 1], axis=1)
    floor = float(np.percentile(heights, 2.0))
    eligible = (speeds <= velocity_threshold_m_per_s) & (
        heights <= floor + ground_band_m
    )
    # A root-only position lock cannot satisfy two independently moving feet.
    # Keep only the slower eligible side per frame and use short contiguous runs.
    chosen = np.zeros_like(eligible)
    for frame in range(len(track.ticks)):
        sides = np.flatnonzero(eligible[frame])
        if len(sides):
            side = int(sides[np.argmin(speeds[frame, sides])])
            chosen[frame, side] = True

    contacts = np.zeros_like(chosen)
    correction = np.zeros_like(roots)
    world = _world_rotations(rotations)
    for side, (foot_index, _) in enumerate(foot_indices):
        parent = DETAILED_HUMANOID.joints[foot_index].parent
        for start, end in _runs(chosen[:, side], minimum_contact_frames):
            target_world = world[start, foot_index].copy()
            candidate_rotations = rotations[start:end].copy()
            candidate_local = _quaternion_multiply(
                _quaternion_inverse(world[start:end, parent]),
                np.broadcast_to(target_world, (end - start, 4)),
            )
            candidate_local /= np.linalg.norm(candidate_local, axis=1, keepdims=True)
            for offset in range(1, len(candidate_local)):
                if np.dot(candidate_local[offset], candidate_local[offset - 1]) < 0.0:
                    candidate_local[offset] *= -1.0
            candidate_rotations[:, foot_index] = candidate_local
            candidate_positions = forward_kinematics_positions(
                roots[start:end], candidate_rotations, skeleton=DETAILED_HUMANOID
            ).astype(np.float64)
            anchor = candidate_positions[0, foot_index]
            run_correction = anchor - candidate_positions[:, foot_index]
            correction_norm = np.linalg.norm(run_correction, axis=1)
            correction_step = np.linalg.norm(np.diff(run_correction, axis=0), axis=1)
            if (
                np.max(correction_norm, initial=0.0) > maximum_root_correction_m
                or np.max(correction_step, initial=0.0) > maximum_root_correction_m
            ):
                continue
            rotations[start:end] = candidate_rotations
            correction[start:end] = run_correction
            contacts[start:end, side] = True

    roots += correction
    projected_positions = forward_kinematics_positions(
        roots, rotations, skeleton=DETAILED_HUMANOID
    ).astype(np.float64)
    all_feet = np.concatenate(
        [projected_positions[:, indices, 1] for indices in foot_indices], axis=1
    )
    penetration_before = max(0.0, -float(np.min(all_feet)))
    if penetration_before:
        roots[:, 1] += penetration_before
    final_positions = forward_kinematics_positions(
        roots, rotations, skeleton=DETAILED_HUMANOID
    ).astype(np.float64)
    final_feet = np.concatenate(
        [final_positions[:, indices, 1] for indices in foot_indices], axis=1
    )
    projected = replace(
        track,
        root_translation_m=roots.astype(np.float32),
        local_rotations_xyzw=rotations.astype(np.float32),
        foot_contacts=contacts,
    )
    diagnostics = BodyProjectionDiagnostics(
        contact_frames=(int(np.sum(contacts[:, 0])), int(np.sum(contacts[:, 1]))),
        maximum_root_correction_m=float(
            np.max(np.linalg.norm(correction, axis=1), initial=0.0)
        ),
        maximum_root_step_correction_m=float(
            np.max(np.linalg.norm(np.diff(correction, axis=0), axis=1), initial=0.0)
        ),
        ground_penetration_before_m=penetration_before,
        ground_penetration_after_m=max(0.0, -float(np.min(final_feet))),
    )
    return projected, diagnostics


__all__ = [
    "BodyProjectionDiagnostics",
    "RootStabilizationDiagnostics",
    "VisualArmConstraintDiagnostics",
    "VisualHandConstraintDiagnostics",
    "constrain_arms_to_video_keypoints",
    "constrain_hands_to_video_keypoints",
    "constrain_restrained_root_travel",
    "project_generated_foot_contacts",
]
