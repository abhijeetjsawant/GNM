"""Auditable, contact-preserving mouth-aperture correction for GNM tracks.

The correction is intentionally narrower than a general expression edit.  It
measures inner-lip geometry on the exact character identity, solves a spatially
constrained lower-face displacement, and passes all joints and non-mouth
coefficient regions through byte-for-byte.  Contact evidence is a hard veto.

GNM's expression basis is a PCA space, so a lower-face coefficient can have a
small geometric tail outside the lips.  The solver explicitly suppresses that
tail and rejects/reduces candidates against measured mesh-space limits.  The
result reports those residuals instead of claiming blendshape-style locality.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math

import numpy as np

from .errors import AutoAnimError
from .rig import ControlRig


SCHEMA_VERSION = "autoanim.gnm.mouth-aperture-correction.v2"
LOWER_FACE_SLICE = slice(200, 350)
TONGUE_SLICE = slice(350, 382)
_INNER_LIP_PAIRS = ((61, 67), (62, 66), (63, 65))
_INNER_LIP_INDICES = np.asarray((61, 62, 63, 65, 66, 67), dtype=np.int64)
_CONTACT_LABELS = frozenset(("p", "b", "m", "contact", "bilabial"))
_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR = 0.0005


@dataclass(frozen=True, slots=True)
class MouthApertureConfig:
    """Artist intent and hard production bounds in identity-normalized units.

    ``gain`` and ``bias_interocular`` apply to aperture above the character's
    identity-specific neutral gap.  Values are opening-only: gain cannot be
    below one and bias cannot be negative.  The default is an exact no-op.
    """

    gain: float = 1.0
    bias_interocular: float = 0.0
    minimum_open_delta_interocular: float = 0.002
    maximum_target_gap_interocular: float = 0.36
    maximum_added_aperture_interocular: float = 0.06
    maximum_correction_velocity_interocular_per_second: float = 0.36
    maximum_final_mouth_step_interocular: float = 0.03995
    maximum_coefficient_delta: float = 0.90
    maximum_lower_face_coefficient: float = 3.0
    maximum_nonmouth_displacement_interocular: float = 0.0010
    maximum_upper_face_displacement_interocular: float = 0.0005
    maximum_tongue_displacement_interocular: float = 0.00002
    contact_confidence_threshold: float = 0.12
    target_tolerance_interocular: float = 0.0002


@dataclass(frozen=True, slots=True)
class MouthContactEvidence:
    """Per-frame evidence that must veto opening correction.

    ``anchor`` is the caller's hard contact decision.  A P/B/M/contact label
    or confidence at/above the configured threshold also becomes a hard veto,
    so an omitted boolean cannot silently weaken supplied bilabial evidence.
    """

    anchor: np.ndarray
    confidence: np.ndarray
    label: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MouthApertureFrameReport:
    frame_index: int
    status: str
    original_gap_interocular: float
    requested_target_gap_interocular: float
    bounded_target_gap_interocular: float
    final_gap_interocular: float
    correction_scale: float
    maximum_coefficient_delta: float
    nonmouth_displacement_interocular: float
    upper_face_displacement_interocular: float
    tongue_displacement_interocular: float
    original_lip_order_minimum_interocular: float
    lip_order_minimum_interocular: float
    lip_order_inversion_risk: bool
    lip_order_inversion_introduced: bool
    target_attained: bool
    bounds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MouthApertureCorrectionResult:
    schema_version: str
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray
    timestamps_seconds: np.ndarray
    neutral_gap_interocular: float
    protected_contact: np.ndarray
    eligible_open: np.ndarray
    correction_applied: np.ndarray
    target_attained: np.ndarray
    final_continuity_scale: np.ndarray
    final_continuity_limit_interocular: float
    reports: tuple[MouthApertureFrameReport, ...]
    identity_sha256: str
    input_sha256: str
    output_sha256: str
    mesh_validation_passed: bool


@dataclass(frozen=True, slots=True)
class _ApertureCalibration:
    response: np.ndarray
    interocular: float
    neutral_gap: float
    nonmouth_indices: np.ndarray
    upper_face_indices: np.ndarray
    tongue_indices: np.ndarray


def mouth_aperture_target_attainment(
    result: MouthApertureCorrectionResult,
) -> float | None:
    """Return target attainment over every eligible open frame.

    Frames whose solve is bounded to no coefficient change remain in the
    denominator. A track with no eligible open frame has no measurable target
    attainment and returns ``None`` rather than a vacuous perfect score.
    """

    eligible = np.asarray(result.eligible_open, dtype=bool)
    if not np.any(eligible):
        return None
    return float(np.mean(np.asarray(result.target_attained, dtype=bool)[eligible]))


def _invalid(message: str, **details: object) -> AutoAnimError:
    return AutoAnimError("MOUTH_APERTURE_INVALID", message, details=details)


def validate_mouth_aperture_authorship(
    *,
    gain: float,
    author: str | None,
    reason: str | None,
) -> tuple[str | None, str | None]:
    """Require accountable provenance for every non-default artist edit."""

    if not np.isfinite(gain) or gain < 1.0:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Mouth-aperture gain must be finite and at least 1",
        )
    normalized_author = author.strip() if author is not None else ""
    normalized_reason = reason.strip() if reason is not None else ""
    if len(normalized_author) > 160 or len(normalized_reason) > 500:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Mouth-aperture author and reason are limited to 160 and 500 characters",
        )
    if gain != 1.0 and (not normalized_author or not normalized_reason):
        raise AutoAnimError(
            "INPUT_INVALID",
            "A non-default mouth-aperture edit requires both an author and a reason",
        )
    return normalized_author or None, normalized_reason or None


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value).copy()
    output.setflags(write=False)
    return output


def _array_digest(*values: np.ndarray) -> str:
    digest = sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _input_digest(
    arrays: tuple[np.ndarray, ...],
    labels: tuple[str, ...],
    config: MouthApertureConfig,
) -> str:
    """Hash every value that can change correction eligibility or geometry."""

    digest = sha256()
    digest.update(SCHEMA_VERSION.encode("utf-8"))
    for value in arrays:
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    for label in labels:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    config_values = np.asarray(
        [
            float(getattr(config, field))
            for field in MouthApertureConfig.__dataclass_fields__
        ],
        dtype="<f8",
    )
    digest.update(config_values.tobytes())
    return digest.hexdigest()


def _validate_config(config: MouthApertureConfig) -> None:
    values = {
        field: float(getattr(config, field))
        for field in MouthApertureConfig.__dataclass_fields__
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise _invalid("Mouth-aperture configuration must be finite")
    if config.gain < 1.0:
        raise _invalid("Mouth-aperture gain cannot close the source performance")
    if config.bias_interocular < 0.0:
        raise _invalid("Mouth-aperture bias cannot be negative")
    positive = (
        "minimum_open_delta_interocular",
        "maximum_target_gap_interocular",
        "maximum_added_aperture_interocular",
        "maximum_correction_velocity_interocular_per_second",
        "maximum_final_mouth_step_interocular",
        "maximum_coefficient_delta",
        "maximum_lower_face_coefficient",
        "maximum_nonmouth_displacement_interocular",
        "maximum_upper_face_displacement_interocular",
        "maximum_tongue_displacement_interocular",
        "target_tolerance_interocular",
    )
    if any(values[name] <= 0.0 for name in positive):
        raise _invalid("Mouth-aperture limits must be positive")
    if not 0.0 <= config.contact_confidence_threshold <= 1.0:
        raise _invalid("Contact-confidence threshold must be between zero and one")
    if config.maximum_lower_face_coefficient > 3.0:
        raise _invalid("GNM lower-face coefficients cannot exceed the rig limit of 3")


def _require_array(
    name: str,
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise _invalid(f"{name} has an invalid shape", expected=shape, actual=array.shape)
    if array.dtype != dtype:
        raise _invalid(
            f"{name} must use {np.dtype(dtype).name}",
            expected=np.dtype(dtype).name,
            actual=array.dtype.name,
        )
    if finite and not np.isfinite(array).all():
        raise _invalid(f"{name} contains nonfinite values")
    return array


def _gap_interocular(rig: ControlRig, expression: np.ndarray, interocular: float) -> float:
    landmarks = rig.compact_landmarks(expression)
    return float(
        np.mean(
            [
                np.linalg.norm(landmarks[upper] - landmarks[lower]) / interocular
                for upper, lower in _INNER_LIP_PAIRS
            ]
        )
    )


def _lip_order_minimum_interocular(
    rig: ControlRig,
    expression: np.ndarray,
    interocular: float,
) -> float:
    landmarks = rig.compact_landmarks(expression)
    face_up = landmarks[27] - landmarks[8]
    norm = float(np.linalg.norm(face_up))
    if not np.isfinite(norm) or norm <= 1.0e-10:
        raise _invalid("GNM face-local up axis is degenerate")
    face_up = face_up / np.float32(norm)
    return float(
        min(
            np.dot(landmarks[upper] - landmarks[lower], face_up) / interocular
            for upper, lower in _INNER_LIP_PAIRS
        )
    )


def _maximum_group(*groups: np.ndarray) -> np.ndarray:
    if not groups:
        raise _invalid("GNM preservation group list is empty")
    return np.maximum.reduce(tuple(np.asarray(group, dtype=np.float32) for group in groups))


def _calibrate(rig: ControlRig) -> _ApertureCalibration:
    """Build an identity-normalized, mouth-local lower-face inverse."""

    adapter = rig.adapter
    if adapter.expression_dim != 383:
        raise _invalid("Mouth correction requires the GNM Head v3 expression layout")
    expected_neutral = adapter.compact_template + np.einsum(
        "i,ijk->jk",
        rig.identity,
        adapter.compact_identity_basis,
        optimize=True,
    )
    if not np.array_equal(expected_neutral, rig.neutral_landmarks):
        raise _invalid("rig identity and cached neutral geometry do not match")
    neutral = np.asarray(rig.neutral_landmarks, dtype=np.float64)
    interocular = float(np.linalg.norm(neutral[36] - neutral[45]))
    if not np.isfinite(interocular) or interocular <= 0.0:
        raise _invalid("GNM identity has invalid interocular geometry")

    compact_basis = np.asarray(adapter.compact_expression_basis[LOWER_FACE_SLICE], dtype=np.float64)
    vertex_basis = np.asarray(adapter.model.expression_basis[LOWER_FACE_SLICE], dtype=np.float64)
    inner_system = compact_basis[:, _INNER_LIP_INDICES].transpose(1, 2, 0).reshape(-1, 150)
    compact_upper_system = compact_basis[:, :48].transpose(1, 2, 0).reshape(-1, 150)

    lip_support = _maximum_group(
        adapter.vertex_group("upper_lip_region"),
        adapter.vertex_group("lower_lip_region"),
        adapter.vertex_group("upper_lip"),
        adapter.vertex_group("lower_lip"),
    )
    exterior = adapter.vertex_group("skin_exterior")
    nonmouth_indices = np.flatnonzero((exterior > 0.20) & (lip_support < 0.05))
    if len(nonmouth_indices) < 128:
        raise _invalid("GNM has too few non-mouth vertices for a bounded solve")
    stride = max(1, int(math.ceil(len(nonmouth_indices) / 3_000)))
    solve_nonmouth = nonmouth_indices[::stride]

    upper_support = _maximum_group(
        adapter.vertex_group("eyes"),
        adapter.vertex_group("eye_sockets"),
        adapter.vertex_group("forehead_region"),
        adapter.vertex_group("left_brow_region"),
        adapter.vertex_group("middle_brow_region"),
        adapter.vertex_group("right_brow_region"),
        adapter.vertex_group("left_orbital_region"),
        adapter.vertex_group("right_orbital_region"),
    )
    upper_face_indices = np.flatnonzero(upper_support > 0.20)
    tongue_indices = np.flatnonzero(adapter.vertex_group("tongue") > 0.20)
    if not len(upper_face_indices) or not len(tongue_indices):
        raise _invalid("GNM upper-face or tongue preservation groups are missing")

    def vertex_system(indices: np.ndarray) -> np.ndarray:
        return vertex_basis[:, indices].transpose(1, 2, 0).reshape(-1, 150)

    def normalized(system: np.ndarray, weight: float) -> np.ndarray:
        return np.float64(weight) * system / math.sqrt(max(len(system), 1))

    # Tongue preservation is intentionally much stronger than the generic
    # exterior constraint.  GNM's lower-face PCA modes otherwise carry a
    # visible tail into the tongue even though its coefficient block is fixed.
    system = np.vstack(
        (
            normalized(inner_system, 1.0),
            normalized(vertex_system(solve_nonmouth), 20.0),
            normalized(compact_upper_system, 20.0),
            normalized(vertex_system(upper_face_indices), 100.0),
            normalized(vertex_system(tongue_indices), 100.0),
            np.float64(1.0e-4) * np.eye(150, dtype=np.float64),
        )
    )
    response = np.linalg.pinv(system, rcond=1.0e-7)[:, :18] / math.sqrt(18)
    if response.shape != (150, 18) or not np.isfinite(response).all():
        raise _invalid("GNM aperture inverse is invalid")
    neutral_gap = _gap_interocular(
        rig,
        np.zeros(adapter.expression_dim, dtype=np.float32),
        interocular,
    )
    response = np.asarray(response, dtype=np.float32)
    response.setflags(write=False)
    return _ApertureCalibration(
        response=response,
        interocular=interocular,
        neutral_gap=neutral_gap,
        nonmouth_indices=np.asarray(nonmouth_indices, dtype=np.int64),
        upper_face_indices=np.asarray(upper_face_indices, dtype=np.int64),
        tongue_indices=np.asarray(tongue_indices, dtype=np.int64),
    )


def _mesh_displacement_metrics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    calibration: _ApertureCalibration,
) -> tuple[float, float, float]:
    displacement = np.linalg.norm(candidate - baseline, axis=1) / calibration.interocular
    return (
        float(np.max(displacement[calibration.nonmouth_indices], initial=0.0)),
        float(np.max(displacement[calibration.upper_face_indices], initial=0.0)),
        float(np.max(displacement[calibration.tongue_indices], initial=0.0)),
    )


def _coefficient_alpha_limit(
    original_lower: np.ndarray,
    direction: np.ndarray,
    config: MouthApertureConfig,
) -> tuple[float, tuple[str, ...]]:
    candidates: list[tuple[float, str]] = [(2.0, "solver_scale")]
    maximum_delta = float(np.max(np.abs(direction), initial=0.0))
    if maximum_delta > 0.0:
        candidates.append((config.maximum_coefficient_delta / maximum_delta, "coefficient_delta"))
    coefficient_limit = config.maximum_lower_face_coefficient
    for current, delta in zip(original_lower, direction, strict=True):
        if delta > 1.0e-12:
            candidates.append(((coefficient_limit - float(current)) / float(delta), "coefficient_range"))
        elif delta < -1.0e-12:
            candidates.append(((-coefficient_limit - float(current)) / float(delta), "coefficient_range"))
    nonnegative = [(max(0.0, value), reason) for value, reason in candidates]
    limit = min(value for value, _ in nonnegative)
    reasons = tuple(
        sorted({reason for value, reason in nonnegative if abs(value - limit) <= 1.0e-8})
    )
    return float(limit), reasons


def _solve_frame(
    rig: ControlRig,
    calibration: _ApertureCalibration,
    original: np.ndarray,
    baseline_mesh: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
    target_gap: float,
    config: MouthApertureConfig,
) -> tuple[np.ndarray, np.ndarray, float, float, tuple[float, float, float], tuple[str, ...]]:
    landmarks = rig.compact_landmarks(original)
    current_gap = _gap_interocular(rig, original, calibration.interocular)
    lift = max(0.0, target_gap - current_gap)
    desired = np.zeros((68, 3), dtype=np.float32)
    for upper, lower in _INNER_LIP_PAIRS:
        separation = landmarks[lower] - landmarks[upper]
        length = float(np.linalg.norm(separation))
        if length <= 1.0e-10:
            continue
        wanted = length + lift * calibration.interocular
        expansion = separation * np.float32(wanted / length - 1.0)
        desired[upper] -= np.float32(0.5) * expansion
        desired[lower] += np.float32(0.5) * expansion
    direction = calibration.response @ desired[_INNER_LIP_INDICES].reshape(-1)
    direction = np.asarray(direction, dtype=np.float32)
    if not np.isfinite(direction).all() or float(np.max(np.abs(direction), initial=0.0)) <= 1.0e-10:
        return original.copy(), baseline_mesh, current_gap, 0.0, (0.0, 0.0, 0.0), ("degenerate_solve",)

    alpha_limit, coefficient_reasons = _coefficient_alpha_limit(
        original[LOWER_FACE_SLICE], direction, config
    )
    vertex_direction = np.einsum(
        "i,ijk->jk",
        direction.astype(np.float64),
        np.asarray(rig.adapter.model.expression_basis[LOWER_FACE_SLICE], dtype=np.float64),
        optimize=True,
    )
    normalized_displacement = np.linalg.norm(vertex_direction, axis=1) / calibration.interocular
    geometric_limits = (
        (
            config.maximum_nonmouth_displacement_interocular,
            float(np.max(normalized_displacement[calibration.nonmouth_indices], initial=0.0)),
            "nonmouth_displacement",
        ),
        (
            config.maximum_upper_face_displacement_interocular,
            float(np.max(normalized_displacement[calibration.upper_face_indices], initial=0.0)),
            "upper_face_displacement",
        ),
        (
            config.maximum_tongue_displacement_interocular,
            float(np.max(normalized_displacement[calibration.tongue_indices], initial=0.0)),
            "tongue_displacement",
        ),
    )
    limit_reasons = list(coefficient_reasons)
    for allowed, unit_displacement, reason in geometric_limits:
        if unit_displacement > 1.0e-12:
            candidate_limit = allowed / unit_displacement
            if candidate_limit < alpha_limit:
                alpha_limit = max(0.0, candidate_limit)
                limit_reasons = [reason]
            elif abs(candidate_limit - alpha_limit) <= 1.0e-8:
                limit_reasons.append(reason)

    def expression_at(alpha: float) -> np.ndarray:
        output = original.copy()
        output[LOWER_FACE_SLICE] = original[LOWER_FACE_SLICE] + np.float32(alpha) * direction
        return output

    gap_at_limit = _gap_interocular(rig, expression_at(alpha_limit), calibration.interocular)
    target_reachable = gap_at_limit >= target_gap
    if target_reachable:
        lower = 0.0
        upper = alpha_limit
        for _ in range(24):
            middle = 0.5 * (lower + upper)
            if _gap_interocular(rig, expression_at(middle), calibration.interocular) < target_gap:
                lower = middle
            else:
                upper = middle
        alpha = upper
        active_limits: list[str] = []
    else:
        alpha = alpha_limit
        active_limits = sorted(set(limit_reasons))

    candidate = expression_at(alpha)
    candidate_mesh = rig.adapter.mesh(
        identity=rig.identity,
        expression=candidate,
        rotations=rotations,
        translation=translation,
    )
    metrics = _mesh_displacement_metrics(baseline_mesh, candidate_mesh, calibration)
    actual_limits = (
        config.maximum_nonmouth_displacement_interocular,
        config.maximum_upper_face_displacement_interocular,
        config.maximum_tongue_displacement_interocular,
    )
    actual_reasons = ("nonmouth_displacement", "upper_face_displacement", "tongue_displacement")
    # Joint skinning is linear but may slightly change a pre-skin norm.  Reduce
    # from the measured mesh if that makes any audited bound tighter.
    for _ in range(3):
        ratios = [
            allowed / measured
            for allowed, measured in zip(actual_limits, metrics, strict=True)
            if measured > allowed and measured > 0.0
        ]
        if not ratios:
            break
        alpha *= max(0.0, min(ratios)) * 0.999
        candidate = expression_at(alpha)
        candidate_mesh = rig.adapter.mesh(
            identity=rig.identity,
            expression=candidate,
            rotations=rotations,
            translation=translation,
        )
        metrics = _mesh_displacement_metrics(baseline_mesh, candidate_mesh, calibration)
        for allowed, measured, reason in zip(actual_limits, metrics, actual_reasons, strict=True):
            if measured >= allowed * 0.999:
                active_limits.append(reason)

    baseline_order = _lip_order_minimum_interocular(
        rig,
        original,
        calibration.interocular,
    )
    minimum_allowed_order = min(
        baseline_order,
        -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR,
    )
    candidate_order = _lip_order_minimum_interocular(
        rig,
        candidate,
        calibration.interocular,
    )
    if candidate_order < minimum_allowed_order:
        lower = 0.0
        upper = alpha
        for _ in range(28):
            middle = 0.5 * (lower + upper)
            if (
                _lip_order_minimum_interocular(
                    rig,
                    expression_at(middle),
                    calibration.interocular,
                )
                >= minimum_allowed_order
            ):
                lower = middle
            else:
                upper = middle
        alpha = max(0.0, lower * 0.999)
        candidate = expression_at(alpha)
        candidate_mesh = rig.adapter.mesh(
            identity=rig.identity,
            expression=candidate,
            rotations=rotations,
            translation=translation,
        )
        metrics = _mesh_displacement_metrics(baseline_mesh, candidate_mesh, calibration)
        active_limits.append("lip_order_inversion")

    final_gap = _gap_interocular(rig, candidate, calibration.interocular)
    if final_gap <= current_gap + 1.0e-7:
        return original.copy(), baseline_mesh, current_gap, 0.0, (0.0, 0.0, 0.0), ("no_improvement",)
    return (
        candidate,
        candidate_mesh,
        final_gap,
        float(alpha),
        metrics,
        tuple(sorted(set(active_limits))),
    )


def _continuity_bound_lifts(
    lifts: np.ndarray,
    timestamps: np.ndarray,
    maximum_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(lifts, dtype=np.float64).copy()
    limited = np.zeros(len(output), dtype=bool)
    for frame in range(1, len(output)):
        allowed = output[frame - 1] + maximum_velocity * float(timestamps[frame] - timestamps[frame - 1])
        if output[frame] > allowed:
            output[frame] = allowed
            limited[frame] = True
    for frame in range(len(output) - 2, -1, -1):
        allowed = output[frame + 1] + maximum_velocity * float(timestamps[frame + 1] - timestamps[frame])
        if output[frame] > allowed:
            output[frame] = allowed
            limited[frame] = True
    return output.astype(np.float32), limited


def _face_local_mouth(rig: ControlRig, expression: np.ndarray) -> np.ndarray:
    """Measure the mouth in the exact normalized frame used by quality gates."""

    landmarks = rig.compact_landmarks(expression)
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    eye_axis = right_eye - left_eye
    interocular = float(np.linalg.norm(eye_axis))
    if interocular <= 1.0e-8:
        raise _invalid("GNM interocular distance is invalid")
    x_axis = eye_axis / interocular
    eye_midpoint = np.float32(0.5) * (left_eye + right_eye)
    nose_direction = landmarks[30] - eye_midpoint
    y_axis = nose_direction - np.dot(nose_direction, x_axis) * x_axis
    y_length = float(np.linalg.norm(y_axis))
    if y_length <= 1.0e-8:
        raise _invalid("GNM face-local frame is invalid")
    y_axis /= y_length
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-8)
    axes = np.stack((x_axis, y_axis, z_axis), axis=1)
    return ((landmarks[48:68] - eye_midpoint) @ axes) / interocular


def _mouth_step_quality_ratio(
    rig: ControlRig,
    previous: np.ndarray,
    target: np.ndarray,
) -> float:
    previous_mouth = _face_local_mouth(rig, previous)
    return float(
        np.max(
            np.linalg.norm(_face_local_mouth(rig, target) - previous_mouth, axis=1),
            initial=0.0,
        )
    )


def _limit_final_mouth_steps(
    rig: ControlRig,
    baseline: np.ndarray,
    corrected: np.ndarray,
    *,
    maximum_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Locally attenuate only the authored delta until every edge is safe.

    The incoming animation has already passed the compiler's continuity gate.
    An aperture edit can nevertheless push a transition over that exact gate.
    Each violating edge scales the correction at its two endpoints toward the
    byte-identical source frames.  Contact and ineligible frames have zero
    deltas, so they remain exact.  The alternating passes account for adjacent
    edges without globally weakening the performance.
    """

    frame_count = len(corrected)
    scale = np.asarray(
        np.max(np.abs(corrected - baseline), axis=1) > 0.0,
        dtype=np.float64,
    )
    baseline_maximum = max(
        (
            _mouth_step_quality_ratio(rig, baseline[index - 1], baseline[index])
            for index in range(1, frame_count)
        ),
        default=0.0,
    )
    # Audio compiler output is already below the configured production cap.
    # Source-video motion can legitimately exceed it because its cadence is
    # owned by exact capture PTS. In that case the edit is constrained not to
    # worsen the retained source maximum instead of silently smoothing capture.
    effective_maximum = max(maximum_ratio, baseline_maximum)
    if frame_count < 2 or not np.any(scale):
        return corrected.copy(), scale.astype(np.float32), effective_maximum

    def frame(index: int, alpha: float) -> np.ndarray:
        candidate = baseline[index].copy()
        candidate[LOWER_FACE_SLICE] = (
            baseline[index, LOWER_FACE_SLICE]
            + np.float32(alpha)
            * (corrected[index, LOWER_FACE_SLICE] - baseline[index, LOWER_FACE_SLICE])
        )
        return candidate

    def edge(index: int, left_scale: float, right_scale: float) -> float:
        return _mouth_step_quality_ratio(
            rig,
            frame(index - 1, left_scale),
            frame(index, right_scale),
        )

    for _ in range(12):
        changed = False
        for order in (range(1, frame_count), range(frame_count - 1, 0, -1)):
            for index in order:
                if edge(index, scale[index - 1], scale[index]) <= effective_maximum + 1.0e-8:
                    continue
                lower = 0.0
                upper = 1.0
                for _ in range(24):
                    middle = 0.5 * (lower + upper)
                    if (
                        edge(
                            index,
                            scale[index - 1] * middle,
                            scale[index] * middle,
                        )
                        <= effective_maximum
                    ):
                        lower = middle
                    else:
                        upper = middle
                reduction = lower * 0.999999
                scale[index - 1] *= reduction
                scale[index] *= reduction
                changed = True
        if not changed:
            break

    output = baseline.copy()
    output[:, LOWER_FACE_SLICE] = (
        baseline[:, LOWER_FACE_SLICE]
        + scale[:, None].astype(np.float32)
        * (corrected[:, LOWER_FACE_SLICE] - baseline[:, LOWER_FACE_SLICE])
    )
    final_maximum = max(
        (
            _mouth_step_quality_ratio(rig, output[index - 1], output[index])
            for index in range(1, frame_count)
        ),
        default=0.0,
    )
    if final_maximum > effective_maximum + 1.0e-7:
        raise _invalid(
            "Mouth correction could not satisfy final continuity",
            actual=final_maximum,
            maximum=effective_maximum,
        )
    return output.astype(np.float32), scale.astype(np.float32), effective_maximum


def correct_mouth_aperture(
    rig: ControlRig,
    *,
    identity: np.ndarray,
    expression: np.ndarray,
    rotations: np.ndarray,
    translation: np.ndarray,
    timestamps_seconds: np.ndarray,
    eligible_frames: np.ndarray,
    contact_evidence: MouthContactEvidence,
    config: MouthApertureConfig = MouthApertureConfig(),
) -> MouthApertureCorrectionResult:
    """Apply a bounded neutral-relative opening correction to a GNM track.

    The exact character identity is required separately and compared with the
    rig to prevent a correction calibrated for one face from being applied to
    another.  The returned expression only differs in modes 200:350.  Joints,
    translation, tongue modes, eyes, and upper-face controls are exact copies.
    """

    if not isinstance(rig, ControlRig):
        raise _invalid("rig must be a ControlRig")
    if not isinstance(config, MouthApertureConfig):
        raise _invalid("config must be a MouthApertureConfig")
    if not isinstance(contact_evidence, MouthContactEvidence):
        raise _invalid("contact_evidence must be MouthContactEvidence")
    _validate_config(config)
    adapter = rig.adapter
    identity_array = _require_array(
        "identity", identity, shape=(adapter.identity_dim,), dtype=np.dtype(np.float32)
    )
    if not np.array_equal(identity_array, rig.identity):
        raise _invalid("identity does not exactly match the supplied rig")

    expression_array = np.asarray(expression)
    if expression_array.ndim != 2:
        raise _invalid("expression must be a frame-by-coefficient matrix")
    frame_count = expression_array.shape[0]
    expression_array = _require_array(
        "expression",
        expression_array,
        shape=(frame_count, adapter.expression_dim),
        dtype=np.dtype(np.float32),
    )
    if np.max(np.abs(expression_array), initial=0.0) > 3.0 + 1.0e-6:
        raise _invalid("expression contains coefficients outside the GNM rig limit")
    if (
        np.max(np.abs(expression_array[:, LOWER_FACE_SLICE]), initial=0.0)
        > config.maximum_lower_face_coefficient + 1.0e-6
    ):
        raise _invalid("expression lower-face coefficients exceed the configured limit")
    rotations_array = _require_array(
        "rotations",
        rotations,
        shape=(frame_count, adapter.model.num_joints, 3),
        dtype=np.dtype(np.float32),
    )
    translation_array = _require_array(
        "translation",
        translation,
        shape=(frame_count, 3),
        dtype=np.dtype(np.float32),
    )
    timestamps_array = _require_array(
        "timestamps_seconds",
        timestamps_seconds,
        shape=(frame_count,),
        dtype=np.dtype(np.float64),
    )
    if frame_count and (timestamps_array[0] < 0.0 or np.any(np.diff(timestamps_array) <= 0.0)):
        raise _invalid("timestamps_seconds must be nonnegative and strictly increasing")
    eligible_array = _require_array(
        "eligible_frames",
        eligible_frames,
        shape=(frame_count,),
        dtype=np.dtype(np.bool_),
        finite=False,
    )
    anchor = _require_array(
        "contact_evidence.anchor",
        contact_evidence.anchor,
        shape=(frame_count,),
        dtype=np.dtype(np.bool_),
        finite=False,
    )
    confidence = _require_array(
        "contact_evidence.confidence",
        contact_evidence.confidence,
        shape=(frame_count,),
        dtype=np.dtype(np.float32),
    )
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise _invalid("contact evidence confidence must be between zero and one")
    if len(contact_evidence.label) != frame_count or any(
        not isinstance(label, str) for label in contact_evidence.label
    ):
        raise _invalid("contact evidence labels must contain one string per frame")

    calibration = _calibrate(rig)
    labels_protected = np.asarray(
        [label.strip().casefold() in _CONTACT_LABELS for label in contact_evidence.label],
        dtype=bool,
    )
    protected = anchor | labels_protected | (confidence >= config.contact_confidence_threshold)
    original_gaps = np.asarray(
        [
            _gap_interocular(rig, frame, calibration.interocular)
            for frame in expression_array
        ],
        dtype=np.float64,
    )
    opening = np.maximum(original_gaps - calibration.neutral_gap, 0.0)
    requested_targets = calibration.neutral_gap + config.gain * opening + config.bias_interocular
    requested_targets = np.maximum(requested_targets, original_gaps)
    open_frames = opening > config.minimum_open_delta_interocular
    eligible_open = eligible_array & open_frames & ~protected

    bounded_targets = np.minimum(requested_targets, config.maximum_target_gap_interocular)
    bounded_targets = np.minimum(
        bounded_targets,
        original_gaps + config.maximum_added_aperture_interocular,
    )
    bounded_targets = np.maximum(bounded_targets, original_gaps)
    lifts = np.where(eligible_open, bounded_targets - original_gaps, 0.0)
    lifts, continuity_limited = _continuity_bound_lifts(
        lifts,
        timestamps_array,
        config.maximum_correction_velocity_interocular_per_second,
    )
    bounded_targets = original_gaps + lifts

    output = expression_array.copy()
    output_meshes: list[np.ndarray] = []
    applied = np.zeros(frame_count, dtype=bool)
    attained = np.ones(frame_count, dtype=bool)
    reports: list[MouthApertureFrameReport] = []
    exact_noop = config.gain == 1.0 and config.bias_interocular == 0.0
    for frame in range(frame_count):
        baseline_mesh = adapter.mesh(
            identity=rig.identity,
            expression=expression_array[frame],
            rotations=rotations_array[frame],
            translation=translation_array[frame],
        )
        bounds: list[str] = []
        if protected[frame]:
            status = "protected_contact"
            bounds.append("contact_anchor")
        elif not eligible_array[frame]:
            status = "not_eligible"
        elif not open_frames[frame]:
            status = "closed_or_neutral"
        elif exact_noop or bounded_targets[frame] <= original_gaps[frame] + 1.0e-9:
            status = "no_requested_change"
        else:
            if requested_targets[frame] > config.maximum_target_gap_interocular:
                bounds.append("maximum_target_gap")
            if requested_targets[frame] > original_gaps[frame] + config.maximum_added_aperture_interocular:
                bounds.append("maximum_added_aperture")
            if continuity_limited[frame]:
                bounds.append("continuity_velocity")
            candidate, candidate_mesh, final_gap, scale, metrics, solve_bounds = _solve_frame(
                rig,
                calibration,
                expression_array[frame],
                baseline_mesh,
                rotations_array[frame],
                translation_array[frame],
                float(bounded_targets[frame]),
                config,
            )
            output[frame] = candidate
            output_meshes.append(candidate_mesh)
            bounds.extend(solve_bounds)
            applied[frame] = not np.array_equal(candidate, expression_array[frame])
            attained[frame] = bool(
                final_gap + config.target_tolerance_interocular >= bounded_targets[frame]
            )
            if not attained[frame]:
                bounds.append("target_unattained")
            status = "corrected" if attained[frame] else ("limited" if applied[frame] else "unchanged")
            lip_order_minimum = _lip_order_minimum_interocular(
                rig,
                candidate,
                calibration.interocular,
            )
            original_lip_order_minimum = _lip_order_minimum_interocular(
                rig,
                expression_array[frame],
                calibration.interocular,
            )
            reports.append(
                MouthApertureFrameReport(
                    frame_index=frame,
                    status=status,
                    original_gap_interocular=float(original_gaps[frame]),
                    requested_target_gap_interocular=float(requested_targets[frame]),
                    bounded_target_gap_interocular=float(bounded_targets[frame]),
                    final_gap_interocular=float(final_gap),
                    correction_scale=float(scale),
                    maximum_coefficient_delta=float(
                        np.max(np.abs(candidate - expression_array[frame]), initial=0.0)
                    ),
                    nonmouth_displacement_interocular=float(metrics[0]),
                    upper_face_displacement_interocular=float(metrics[1]),
                    tongue_displacement_interocular=float(metrics[2]),
                    original_lip_order_minimum_interocular=(
                        original_lip_order_minimum
                    ),
                    lip_order_minimum_interocular=lip_order_minimum,
                    lip_order_inversion_risk=bool(
                        lip_order_minimum
                        < -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                    ),
                    lip_order_inversion_introduced=bool(
                        original_lip_order_minimum
                        >= -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                        and lip_order_minimum
                        < -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                    ),
                    target_attained=bool(attained[frame]),
                    bounds=tuple(sorted(set(bounds))),
                )
            )
            continue

        output_meshes.append(baseline_mesh)
        lip_order_minimum = _lip_order_minimum_interocular(
            rig,
            expression_array[frame],
            calibration.interocular,
        )
        reports.append(
            MouthApertureFrameReport(
                frame_index=frame,
                status=status,
                original_gap_interocular=float(original_gaps[frame]),
                requested_target_gap_interocular=float(requested_targets[frame]),
                bounded_target_gap_interocular=float(original_gaps[frame]),
                final_gap_interocular=float(original_gaps[frame]),
                correction_scale=0.0,
                maximum_coefficient_delta=0.0,
                nonmouth_displacement_interocular=0.0,
                upper_face_displacement_interocular=0.0,
                tongue_displacement_interocular=0.0,
                original_lip_order_minimum_interocular=lip_order_minimum,
                lip_order_minimum_interocular=lip_order_minimum,
                lip_order_inversion_risk=bool(
                    lip_order_minimum
                    < -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                ),
                lip_order_inversion_introduced=False,
                target_attained=True,
                bounds=tuple(bounds),
            )
        )

    preliminary_output = output.copy()
    output, final_continuity_scale, final_continuity_limit = _limit_final_mouth_steps(
        rig,
        expression_array,
        preliminary_output,
        maximum_ratio=config.maximum_final_mouth_step_interocular,
    )
    output_meshes = []
    revised_reports: list[MouthApertureFrameReport] = []
    for frame, report in enumerate(reports):
        baseline_mesh = adapter.mesh(
            identity=rig.identity,
            expression=expression_array[frame],
            rotations=rotations_array[frame],
            translation=translation_array[frame],
        )
        candidate_mesh = adapter.mesh(
            identity=rig.identity,
            expression=output[frame],
            rotations=rotations_array[frame],
            translation=translation_array[frame],
        )
        output_meshes.append(candidate_mesh)
        changed = not np.array_equal(output[frame], expression_array[frame])
        applied[frame] = changed
        if not changed:
            attained[frame] = bool(
                not eligible_open[frame]
                or original_gaps[frame] + config.target_tolerance_interocular
                >= report.bounded_target_gap_interocular
            )
        elif eligible_open[frame]:
            final_gap = _gap_interocular(rig, output[frame], calibration.interocular)
            attained[frame] = bool(
                final_gap + config.target_tolerance_interocular
                >= report.bounded_target_gap_interocular
            )
        else:
            attained[frame] = True

        was_continuity_limited = bool(
            not np.array_equal(preliminary_output[frame], expression_array[frame])
            and final_continuity_scale[frame] < 1.0 - 1.0e-6
        )
        if not was_continuity_limited:
            revised_reports.append(report)
            continue

        final_gap = _gap_interocular(rig, output[frame], calibration.interocular)
        metrics = _mesh_displacement_metrics(baseline_mesh, candidate_mesh, calibration)
        lip_order_minimum = _lip_order_minimum_interocular(
            rig,
            output[frame],
            calibration.interocular,
        )
        original_lip_order_minimum = report.original_lip_order_minimum_interocular
        bounds = set(report.bounds)
        bounds.add("final_mouth_step_continuity")
        if attained[frame]:
            bounds.discard("target_unattained")
        else:
            bounds.add("target_unattained")
        revised_reports.append(
            replace(
                report,
                status=("corrected" if attained[frame] else "limited"),
                final_gap_interocular=float(final_gap),
                correction_scale=float(
                    report.correction_scale * final_continuity_scale[frame]
                ),
                maximum_coefficient_delta=float(
                    np.max(
                        np.abs(output[frame] - expression_array[frame]),
                        initial=0.0,
                    )
                ),
                nonmouth_displacement_interocular=float(metrics[0]),
                upper_face_displacement_interocular=float(metrics[1]),
                tongue_displacement_interocular=float(metrics[2]),
                lip_order_minimum_interocular=float(lip_order_minimum),
                lip_order_inversion_risk=bool(
                    lip_order_minimum
                    < -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                ),
                lip_order_inversion_introduced=bool(
                    original_lip_order_minimum
                    >= -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                    and lip_order_minimum
                    < -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR
                ),
                target_attained=bool(attained[frame]),
                bounds=tuple(sorted(bounds)),
            )
        )
    reports = revised_reports

    # Parameter-locality is an exact contract, unlike mesh-space PCA locality.
    if not np.array_equal(output[:, :200], expression_array[:, :200]):
        raise _invalid("Mouth correction altered upper-face coefficients")
    if not np.array_equal(output[:, TONGUE_SLICE], expression_array[:, TONGUE_SLICE]):
        raise _invalid("Mouth correction altered tongue coefficients")
    if not np.array_equal(output[:, 382:], expression_array[:, 382:]):
        raise _invalid("Mouth correction altered reserved coefficients")
    original_lip_order = np.asarray(
        [
            _lip_order_minimum_interocular(rig, frame, calibration.interocular)
            for frame in expression_array
        ],
        dtype=np.float64,
    )
    revised_lip_order = np.asarray(
        [
            _lip_order_minimum_interocular(rig, frame, calibration.interocular)
            for frame in output
        ],
        dtype=np.float64,
    )
    minimum_allowed_order = np.minimum(
        original_lip_order,
        -_LIP_ORDER_INVERSION_TOLERANCE_INTEROCULAR,
    )
    if np.any(revised_lip_order < minimum_allowed_order - 1.0e-7):
        raise _invalid("Mouth correction introduced or worsened lip-order inversion")
    if not np.isfinite(output).all() or not all(np.isfinite(mesh).all() for mesh in output_meshes):
        raise _invalid("Mouth correction produced nonfinite GNM geometry")

    input_digest = _input_digest(
        (
            identity_array,
            expression_array,
            rotations_array,
            translation_array,
            timestamps_array,
            eligible_array,
            anchor,
            confidence,
        ),
        contact_evidence.label,
        config,
    )
    output_digest = _array_digest(output, rotations_array, translation_array, timestamps_array)
    return MouthApertureCorrectionResult(
        schema_version=SCHEMA_VERSION,
        expression=_readonly(output),
        rotations=_readonly(rotations_array),
        translation=_readonly(translation_array),
        timestamps_seconds=_readonly(timestamps_array),
        neutral_gap_interocular=float(calibration.neutral_gap),
        protected_contact=_readonly(protected),
        eligible_open=_readonly(eligible_open),
        correction_applied=_readonly(applied),
        target_attained=_readonly(attained),
        final_continuity_scale=_readonly(final_continuity_scale),
        final_continuity_limit_interocular=float(final_continuity_limit),
        reports=tuple(reports),
        identity_sha256=_array_digest(identity_array),
        input_sha256=input_digest,
        output_sha256=output_digest,
        mesh_validation_passed=True,
    )
