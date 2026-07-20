"""Authority-preserving temporal projection for GNM oral controls.

The production mouth-continuity metric observes visible lip landmarks only.
GNM's dedicated tongue and pupil controls therefore cannot help satisfy that
constraint and must never be attenuated by it.  This module makes that channel
ownership explicit, applies the continuity bound only to lower-face modes, and
emits an exact source-loss ledger for every projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable

import numpy as np

from .errors import AutoAnimError


ARTICULATION_PROJECTION_SCHEMA_VERSION = "autoanim.articulation-projection/1.0"
ARTICULATION_PROJECTION_ALGORITHM = "bidirectional-lower-face-edge-projection-v1"
EXPECTED_EXPRESSION_DIM = 383

LEFT_EYE_SLICE = slice(0, 100)
RIGHT_EYE_SLICE = slice(100, 200)
LOWER_FACE_SLICE = slice(200, 350)
TONGUE_SLICE = slice(350, 382)
PUPIL_SLICE = slice(382, 383)
PROTECTED_SLICES = (slice(0, 200), slice(350, 383))

_REGIONS: tuple[tuple[str, slice], ...] = (
    ("left_eye", LEFT_EYE_SLICE),
    ("right_eye", RIGHT_EYE_SLICE),
    ("lower_face", LOWER_FACE_SLICE),
    ("tongue", TONGUE_SLICE),
    ("pupils", PUPIL_SLICE),
)

MouthStepMetric = Callable[[np.ndarray, np.ndarray], float]
FrameMetric = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class ArticulationProjectionResult:
    """Projected controls and the evidence needed to audit source loss."""

    expression: np.ndarray
    desired_expression: np.ndarray
    limited_frames: np.ndarray
    contact_attained: np.ndarray
    contact_continuity_restored: np.ndarray
    report: dict[str, Any]


def articulation_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(
        json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0"
    )
    if array.size:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def articulation_arrays_byte_exact(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare array contracts including dtype, shape, and floating sign bits."""

    before = np.asarray(left)
    after = np.asarray(right)
    return bool(
        before.dtype == after.dtype
        and before.shape == after.shape
        and before.tobytes(order="C") == after.tobytes(order="C")
    )


def articulation_report_canonical_bytes(report: Mapping[str, Any]) -> bytes:
    """Serialize report evidence with JSON type information preserved."""

    try:
        return json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_EVIDENCE_INVALID_JSON",
            "Articulation projection report must be finite canonical JSON",
        ) from error


def _finite_metric(value: float, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_METRIC",
            f"{label} must return a finite non-negative value",
        )
    return number


def _validate_controls(
    desired_expression: np.ndarray,
    timestamps: np.ndarray,
    *,
    maximum_step: float,
    maximum_speed: float,
    contact_horizon_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    desired = np.asarray(desired_expression)
    clock = np.asarray(timestamps)
    if desired.ndim != 2 or desired.shape[1:] != (EXPECTED_EXPRESSION_DIM,):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Desired controls must have shape [frames,383]",
        )
    if desired.dtype != np.float32:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Desired controls must use float32 GNM coefficients",
        )
    if len(desired) < 1 or not np.isfinite(desired).all():
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Desired controls must contain at least one finite frame",
        )
    if float(np.max(np.abs(desired), initial=0.0)) > 3.00001:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Desired GNM coefficients exceed the bounded control range",
        )
    if clock.ndim != 1 or clock.shape != (len(desired),):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Projection timestamps must have one value per control frame",
        )
    clock = clock.astype(np.float64, copy=False)
    if not np.isfinite(clock).all():
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Projection timestamps must be finite",
        )
    edge_seconds = np.diff(clock)
    if np.any(edge_seconds <= 0.0):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Projection timestamps must increase strictly",
        )
    for value, label in (
        (maximum_step, "maximum_step"),
        (maximum_speed, "maximum_speed"),
        (contact_horizon_seconds, "contact_horizon_seconds"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise AutoAnimError(
                "ARTICULATION_PROJECTION_INVALID_INPUT",
                f"{label} must be finite and positive",
            )
    edge_limits = np.minimum(
        np.float64(maximum_step),
        np.float64(maximum_speed) * edge_seconds,
    )
    return desired.copy(), clock.copy(), edge_limits


def limit_articulation_edge(
    previous: np.ndarray,
    target: np.ndarray,
    *,
    mouth_step_metric: MouthStepMetric,
    maximum_ratio: float,
) -> tuple[np.ndarray, bool]:
    """Project one edge while preserving every non-lower-face target channel."""

    before = np.asarray(previous)
    desired = np.asarray(target)
    if (
        before.shape != (EXPECTED_EXPRESSION_DIM,)
        or desired.shape != (EXPECTED_EXPRESSION_DIM,)
        or before.dtype != np.float32
        or desired.dtype != np.float32
        or not np.isfinite(before).all()
        or not np.isfinite(desired).all()
        or not np.isfinite(maximum_ratio)
        or maximum_ratio <= 0.0
    ):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Articulation edge inputs must be finite float32 GNM controls",
        )

    def measured(candidate: np.ndarray) -> float:
        return _finite_metric(
            mouth_step_metric(before, candidate),
            "mouth_step_metric",
        )

    if measured(desired) <= maximum_ratio:
        return desired.copy(), False

    def candidate(alpha: float) -> np.ndarray:
        output = desired.copy()
        output[LOWER_FACE_SLICE] = before[LOWER_FACE_SLICE] + np.float32(alpha) * (
            desired[LOWER_FACE_SLICE] - before[LOWER_FACE_SLICE]
        )
        return output

    protected_only = candidate(0.0)
    protected_step = measured(protected_only)
    if protected_step > maximum_ratio:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INFEASIBLE",
            "Protected-channel motion alone exceeds the visible-mouth continuity bound",
            {
                "protected_only_step": protected_step,
                "maximum_ratio": float(maximum_ratio),
            },
        )

    lower = 0.0
    upper = 1.0
    # Sixteen iterations preserve the byte contract of the previous bounded
    # search for lower-face channels while removing its tongue/upper-face loss.
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        if measured(candidate(middle)) <= maximum_ratio:
            lower = middle
        else:
            upper = middle
    output = candidate(lower).astype(np.float32, copy=False)
    final_step = measured(output)
    if final_step > maximum_ratio + 1.0e-6:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INFEASIBLE",
            "Lower-face projection could not satisfy the visible-mouth continuity bound",
            {"final_step": final_step, "maximum_ratio": float(maximum_ratio)},
        )
    for protected in PROTECTED_SLICES:
        if not articulation_arrays_byte_exact(output[protected], desired[protected]):
            raise AutoAnimError(
                "ARTICULATION_PROJECTION_AUTHORITY_VIOLATION",
                "Articulation projection changed a protected control channel",
            )
    return output, True


def _contact_status(
    expression: np.ndarray,
    target_gap: np.ndarray,
    *,
    mouth_gap_metric: FrameMetric | None,
    contact_tolerance: float,
) -> np.ndarray:
    candidates = target_gap > 0.0
    attained = np.zeros(len(expression), dtype=bool)
    if not np.any(candidates):
        return attained
    if mouth_gap_metric is None:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "A mouth-gap metric is required when contact targets are present",
        )
    gaps = np.asarray(
        [
            _finite_metric(mouth_gap_metric(expression[index]), "mouth_gap_metric")
            for index in np.flatnonzero(candidates)
        ],
        dtype=np.float64,
    )
    attained[candidates] = gaps <= target_gap[candidates] + contact_tolerance
    return attained


def _restore_contact_anchors(
    desired: np.ndarray,
    projected: np.ndarray,
    timestamps: np.ndarray,
    edge_limits: np.ndarray,
    *,
    hard_contact_anchors: np.ndarray,
    restore_needed: np.ndarray,
    mouth_step_metric: MouthStepMetric,
    contact_horizon_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    output = projected.copy()
    anchors = np.asarray(hard_contact_anchors, dtype=bool)
    needed = np.asarray(restore_needed, dtype=bool)
    restored = np.zeros(len(output), dtype=bool)
    if not np.any(needed):
        return output, restored

    for anchor_value in np.flatnonzero(needed):
        anchor = int(anchor_value)
        left_limit = int(
            np.searchsorted(
                timestamps,
                timestamps[anchor] - contact_horizon_seconds,
                side="left",
            )
        )
        right_limit = int(
            np.searchsorted(
                timestamps,
                timestamps[anchor] + contact_horizon_seconds,
                side="right",
            )
            - 1
        )
        left_limit = max(0, min(anchor, left_limit))
        right_limit = min(len(output) - 1, max(anchor, right_limit))
        maximum_radius = max(anchor - left_limit, right_limit - anchor)
        accepted: np.ndarray | None = None
        for radius in range(1, maximum_radius + 1):
            left = max(left_limit, anchor - radius)
            right = min(right_limit, anchor + radius)
            trial = output.copy()
            trial[anchor] = desired[anchor]
            for frame in range(anchor - 1, left - 1, -1):
                if anchors[frame]:
                    trial[frame] = desired[frame]
                else:
                    trial[frame], _ = limit_articulation_edge(
                        trial[frame + 1],
                        output[frame],
                        mouth_step_metric=mouth_step_metric,
                        maximum_ratio=float(edge_limits[frame]),
                    )
            for frame in range(anchor + 1, right + 1):
                if anchors[frame]:
                    trial[frame] = desired[frame]
                else:
                    trial[frame], _ = limit_articulation_edge(
                        trial[frame - 1],
                        output[frame],
                        mouth_step_metric=mouth_step_metric,
                        maximum_ratio=float(edge_limits[frame - 1]),
                    )

            check_left = max(0, left - 1)
            check_right = min(len(output) - 2, right)
            if all(
                _finite_metric(
                    mouth_step_metric(trial[frame], trial[frame + 1]),
                    "mouth_step_metric",
                )
                <= float(edge_limits[frame]) + 1.0e-6
                for frame in range(check_left, check_right + 1)
            ):
                accepted = trial
                break
        if accepted is not None:
            output = accepted
            restored[anchor] = articulation_arrays_byte_exact(
                output[anchor], desired[anchor]
            )
    return output, restored


def _region_delta_report(desired: np.ndarray, final: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, region in _REGIONS:
        delta = final[:, region] - desired[:, region]
        frame_delta = np.max(np.abs(delta), axis=1, initial=0.0)
        report[name] = {
            "start": int(region.start or 0),
            "stop": int(region.stop or EXPECTED_EXPRESSION_DIM),
            "changed_frames": int(np.count_nonzero(frame_delta > 0.0)),
            "max_abs_control_delta": float(np.max(np.abs(delta), initial=0.0)),
            "rms_control_delta": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
            "exactly_preserved": articulation_arrays_byte_exact(
                final[:, region], desired[:, region]
            ),
        }
    return report


def project_articulation_trajectory(
    desired_expression: np.ndarray,
    timestamps: np.ndarray,
    *,
    mouth_step_metric: MouthStepMetric,
    lip_order_metric: FrameMetric | None = None,
    contact_target_gap: np.ndarray | None = None,
    mouth_gap_metric: FrameMetric | None = None,
    maximum_step: float = 0.039,
    maximum_speed: float = 1.17,
    contact_tolerance: float = 0.001,
    lip_order_floor: float = -0.0005,
    contact_horizon_seconds: float = 4.0 / 30.0,
    metric_contract: Mapping[str, str] | None = None,
    rig_binding: Mapping[str, str] | None = None,
) -> ArticulationProjectionResult:
    """Project a whole oral trajectory and prove protected-channel authority.

    Each edge uses ``min(maximum_step, maximum_speed * exact_dt)``.  The
    forward/reverse passes and contact restoration may modify only GNM modes
    ``200:350``.  Any visible-mouth violation caused solely by protected
    channels fails closed instead of silently smoothing eyes, tongue, or
    pupils.
    """

    desired, clock, edge_limits = _validate_controls(
        desired_expression,
        timestamps,
        maximum_step=maximum_step,
        maximum_speed=maximum_speed,
        contact_horizon_seconds=contact_horizon_seconds,
    )
    if not np.isfinite(contact_tolerance) or contact_tolerance < 0.0:
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "contact_tolerance must be finite and non-negative",
        )
    if not np.isfinite(lip_order_floor):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "lip_order_floor must be finite",
        )
    metric_identity = dict(metric_contract or {})
    rig_identity = dict(rig_binding or {})
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for bindings in (metric_identity, rig_identity)
        for key, value in bindings.items()
    ):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Metric and rig evidence bindings must contain non-empty strings",
        )
    required_metrics = {"mouth_step", "mouth_gap", "lip_order"}
    required_rig_bindings = {
        "gnm_head_sha256",
        "landmark_regressor_sha256",
        "identity_array_sha256",
    }
    metric_contract_bound = bool(
        required_metrics.issubset(metric_identity)
        and required_rig_bindings.issubset(rig_identity)
    )
    targets = (
        np.zeros(len(desired), dtype=np.float32)
        if contact_target_gap is None
        else np.asarray(contact_target_gap)
    )
    if (
        targets.shape != (len(desired),)
        or targets.dtype != np.float32
        or not np.isfinite(targets).all()
        or np.any(targets < 0.0)
    ):
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INVALID_INPUT",
            "Contact targets must be finite non-negative float32 values per frame",
        )

    preprojection_contact = _contact_status(
        desired,
        targets,
        mouth_gap_metric=mouth_gap_metric,
        contact_tolerance=contact_tolerance,
    )
    expression = desired.copy()
    for frame in range(1, len(expression)):
        expression[frame], _ = limit_articulation_edge(
            expression[frame - 1],
            expression[frame],
            mouth_step_metric=mouth_step_metric,
            maximum_ratio=float(edge_limits[frame - 1]),
        )
    for frame in range(len(expression) - 2, -1, -1):
        expression[frame], _ = limit_articulation_edge(
            expression[frame + 1],
            expression[frame],
            mouth_step_metric=mouth_step_metric,
            maximum_ratio=float(edge_limits[frame]),
        )

    baseline_contact = _contact_status(
        expression,
        targets,
        mouth_gap_metric=mouth_gap_metric,
        contact_tolerance=contact_tolerance,
    )
    restore_needed = preprojection_contact & ~baseline_contact
    expression, restored = _restore_contact_anchors(
        desired,
        expression,
        clock,
        edge_limits,
        hard_contact_anchors=preprojection_contact,
        restore_needed=restore_needed,
        mouth_step_metric=mouth_step_metric,
        contact_horizon_seconds=contact_horizon_seconds,
    )
    final_contact = _contact_status(
        expression,
        targets,
        mouth_gap_metric=mouth_gap_metric,
        contact_tolerance=contact_tolerance,
    )

    for protected in PROTECTED_SLICES:
        if not articulation_arrays_byte_exact(
            expression[:, protected], desired[:, protected]
        ):
            raise AutoAnimError(
                "ARTICULATION_PROJECTION_AUTHORITY_VIOLATION",
                "Trajectory projection changed a protected control channel",
            )

    observed_steps = np.asarray(
        [
            _finite_metric(
                mouth_step_metric(expression[index], expression[index + 1]),
                "mouth_step_metric",
            )
            for index in range(len(expression) - 1)
        ],
        dtype=np.float64,
    )
    if len(observed_steps) and np.any(observed_steps > edge_limits + 1.0e-6):
        first = int(np.flatnonzero(observed_steps > edge_limits + 1.0e-6)[0])
        raise AutoAnimError(
            "ARTICULATION_PROJECTION_INFEASIBLE",
            "Projected trajectory violates an exact-time mouth continuity edge",
            {
                "edge": first,
                "observed": float(observed_steps[first]),
                "allowed": float(edge_limits[first]),
            },
        )

    introduced_lip_order = np.zeros(len(expression), dtype=bool)
    if lip_order_metric is not None:
        desired_order = np.asarray(
            [float(lip_order_metric(frame)) for frame in desired], dtype=np.float64
        )
        final_order = np.asarray(
            [float(lip_order_metric(frame)) for frame in expression], dtype=np.float64
        )
        if not np.isfinite(desired_order).all() or not np.isfinite(final_order).all():
            raise AutoAnimError(
                "ARTICULATION_PROJECTION_INVALID_METRIC",
                "lip_order_metric must return finite values",
            )
        introduced_lip_order = (final_order < lip_order_floor) & ~(
            desired_order < lip_order_floor
        )
        if np.any(introduced_lip_order):
            raise AutoAnimError(
                "ARTICULATION_PROJECTION_LIP_ORDER_VIOLATION",
                "Trajectory projection introduced an inner-lip inversion",
                {"frames": np.flatnonzero(introduced_lip_order).astype(int).tolist()},
            )

    limited = np.max(np.abs(expression - desired), axis=1) > 1.0e-7
    regions = _region_delta_report(desired, expression)
    report: dict[str, Any] = {
        "schema_version": ARTICULATION_PROJECTION_SCHEMA_VERSION,
        "algorithm": ARTICULATION_PROJECTION_ALGORITHM,
        "frame_count": int(len(expression)),
        "expression_dim": EXPECTED_EXPRESSION_DIM,
        "desired_expression_sha256": articulation_array_sha256(desired),
        "projected_expression_sha256": articulation_array_sha256(expression),
        "timestamps_sha256": articulation_array_sha256(clock),
        "maximum_step_interocular": float(maximum_step),
        "maximum_speed_interocular_per_second": float(maximum_speed),
        "contact_tolerance_interocular": float(contact_tolerance),
        "lip_order_floor_interocular": float(lip_order_floor),
        "contact_horizon_seconds": float(contact_horizon_seconds),
        "metric_contract": metric_identity,
        "rig_binding": rig_identity,
        "metric_contract_bound": metric_contract_bound,
        "maximum_observed_step_interocular": float(
            np.max(observed_steps, initial=0.0)
        ),
        "maximum_observed_speed_interocular_per_second": float(
            np.max(observed_steps / np.diff(clock), initial=0.0)
            if len(observed_steps)
            else 0.0
        ),
        "limited_frames": int(np.count_nonzero(limited)),
        "preprojection_contact_attained_frames": int(
            np.count_nonzero(preprojection_contact)
        ),
        "contact_restore_needed_frames": int(np.count_nonzero(restore_needed)),
        "contact_continuity_restored_frames": int(np.count_nonzero(restored)),
        "contact_anchor_loss_frames": int(
            np.count_nonzero(preprojection_contact & ~final_contact)
        ),
        "introduced_lip_order_risk_frames": int(
            np.count_nonzero(introduced_lip_order)
        ),
        "projectable_range": [LOWER_FACE_SLICE.start, LOWER_FACE_SLICE.stop],
        "protected_ranges": [[0, 200], [350, 383]],
        "protected_channels_exact": bool(
            articulation_arrays_byte_exact(expression[:, :200], desired[:, :200])
            and articulation_arrays_byte_exact(
                expression[:, 350:383], desired[:, 350:383]
            )
        ),
        "projection_induced_tongue_max_abs_control_delta": regions["tongue"][
            "max_abs_control_delta"
        ],
        "regions": regions,
        "production_validated": False,
    }
    return ArticulationProjectionResult(
        expression=expression.astype(np.float32, copy=True),
        desired_expression=desired.copy(),
        limited_frames=limited,
        contact_attained=final_contact,
        contact_continuity_restored=restored,
        report=report,
    )
