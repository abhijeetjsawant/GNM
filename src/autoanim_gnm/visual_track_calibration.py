"""Independent calibration evidence for VisualTrack V2a regional scores.

Calibration is an evaluation artifact, not a motion-control switch.  Even a
fully populated record remains shadow-only until a separate reviewed authority
policy exists.  The retained monocular pipeline currently emits the explicit
``unavailable`` form because it has no independent frame labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import AutoAnimError
from .serialization import write_json
from .visual_track import MOTION_AUTHORITY, REGION_NAMES
from .visual_track_provider import (
    VisualTrackProviderResult,
    array_sha256,
    file_sha256,
    valid_sha256,
)


VISUAL_TRACK_CALIBRATION_SCHEMA_VERSION = "autoanim.visual-track-calibration/1.0"
VISUAL_TRACK_CALIBRATION_POLICY = "independent_labels_shadow_evaluation_v2a"
MAX_CALIBRATION_BYTES = 4 * 1024 * 1024
CALIBRATION_BIN_COUNT = 10
MINIMUM_SAMPLES = 20
MINIMUM_POSITIVE = 5
MINIMUM_NEGATIVE = 5
MAXIMUM_ECE = 0.05
MAXIMUM_BRIER = 0.10


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _require_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Calibration {label} members do not match the schema")
    return value


def _metric_slice(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    count = len(probability)
    if count == 0:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "ece": None,
            "brier": None,
            "accuracy_at_0_5": None,
            "mean_probability": None,
            "empirical_frequency": None,
        }
    probability64 = np.asarray(probability, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, CALIBRATION_BIN_COUNT + 1)
    for index in range(CALIBRATION_BIN_COUNT):
        if index == CALIBRATION_BIN_COUNT - 1:
            selected = (probability64 >= edges[index]) & (
                probability64 <= edges[index + 1]
            )
        else:
            selected = (probability64 >= edges[index]) & (
                probability64 < edges[index + 1]
            )
        selected_count = int(np.count_nonzero(selected))
        if selected_count:
            ece += (selected_count / count) * abs(
                float(np.mean(probability64[selected]))
                - float(np.mean(target64[selected]))
            )
    positive = int(np.count_nonzero(target))
    return {
        "sample_count": count,
        "positive_count": positive,
        "negative_count": count - positive,
        "ece": float(ece),
        "brier": float(np.mean(np.square(probability64 - target64))),
        "accuracy_at_0_5": float(np.mean((probability64 >= 0.5) == target)),
        "mean_probability": float(np.mean(probability64)),
        "empirical_frequency": float(np.mean(target64)),
    }


def _slice_passes(metrics: dict[str, Any], *, require_classes: bool) -> bool:
    return bool(
        metrics["sample_count"] >= MINIMUM_SAMPLES
        and (
            not require_classes
            or (
                metrics["positive_count"] >= MINIMUM_POSITIVE
                and metrics["negative_count"] >= MINIMUM_NEGATIVE
            )
        )
        and metrics["ece"] is not None
        and metrics["ece"] <= MAXIMUM_ECE
        and metrics["brier"] is not None
        and metrics["brier"] <= MAXIMUM_BRIER
    )


def _validate_metric_slice(value: object, label: str) -> dict[str, Any]:
    metrics = _require_keys(
        value,
        {
            "sample_count",
            "positive_count",
            "negative_count",
            "ece",
            "brier",
            "accuracy_at_0_5",
            "mean_probability",
            "empirical_frequency",
        },
        label,
    )
    count = metrics["sample_count"]
    positive = metrics["positive_count"]
    negative = metrics["negative_count"]
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not isinstance(positive, int)
        or isinstance(positive, bool)
        or not isinstance(negative, int)
        or isinstance(negative, bool)
        or min(count, positive, negative) < 0
        or positive + negative != count
    ):
        raise ValueError(f"Calibration {label} sample counts are invalid")
    numeric = (
        "ece",
        "brier",
        "accuracy_at_0_5",
        "mean_probability",
        "empirical_frequency",
    )
    if count == 0:
        if any(metrics[name] is not None for name in numeric):
            raise ValueError(f"Calibration empty {label} metrics must be unknown")
    else:
        for name in numeric:
            item = metrics[name]
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not np.isfinite(item)
                or not 0.0 <= float(item) <= 1.0
            ):
                raise ValueError(f"Calibration {label} {name} is invalid")
    return metrics


@dataclass(frozen=True, slots=True)
class VisualTrackCalibrationEvidence:
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        # Round-trip removes non-JSON values and protects the immutable view.
        clean = json.loads(json.dumps(self.payload, allow_nan=False))
        object.__setattr__(self, "payload", clean)
        self.validate()

    def validate(self) -> None:
        payload = _require_keys(
            self.payload,
            {
                "schema_version",
                "kind",
                "status",
                "policy",
                "motion_authority",
                "consumed_by_retargeting",
                "provider_binding",
                "fixture_binding",
                "method",
                "thresholds",
                "region_order",
                "regions",
                "summary",
                "claims",
            },
            "root",
        )
        provider = _require_keys(
            payload["provider_binding"],
            {
                "provider_result_sha256",
                "source_pts_sha256",
                "region_support_score_sha256",
                "model_sha256",
                "profile_sha256",
            },
            "provider binding",
        )
        fixture = _require_keys(
            payload["fixture_binding"],
            {"state", "fixture_sha256", "labels_sha256", "independent_of_provider"},
            "fixture binding",
        )
        method = _require_keys(
            payload["method"],
            {"target", "binning", "bin_count", "ece", "brier"},
            "method",
        )
        thresholds = _require_keys(
            payload["thresholds"],
            {
                "minimum_samples",
                "minimum_positive",
                "minimum_negative",
                "maximum_ece",
                "maximum_brier",
                "occlusion_slices_required",
            },
            "thresholds",
        )
        summary = _require_keys(
            payload["summary"],
            {"qualified_region_count", "all_regions_qualified", "reason"},
            "summary",
        )
        claims = _require_keys(
            payload["claims"],
            {
                "changes_final_gnm_motion",
                "grants_motion_authority",
                "confidence_calibrated_for_runtime",
                "production_validated",
            },
            "claims",
        )
        if (
            payload["schema_version"] != VISUAL_TRACK_CALIBRATION_SCHEMA_VERSION
            or payload["kind"] != "visual_track_calibration"
            or payload["status"] not in {"unavailable", "calibration_evidence_only"}
            or payload["policy"] != VISUAL_TRACK_CALIBRATION_POLICY
            or payload["motion_authority"] != MOTION_AUTHORITY
            or payload["consumed_by_retargeting"] is not False
            or payload["region_order"] != list(REGION_NAMES)
            or set(payload["regions"]) != set(REGION_NAMES)
            or method
            != {
                "target": "probability_measurement_error_is_within_named_bound",
                "binning": "equal_width_closed_final_bin",
                "bin_count": CALIBRATION_BIN_COUNT,
                "ece": "sum_bin_fraction_times_abs_mean_probability_minus_frequency",
                "brier": "mean_squared_probability_error",
            }
            or thresholds
            != {
                "minimum_samples": MINIMUM_SAMPLES,
                "minimum_positive": MINIMUM_POSITIVE,
                "minimum_negative": MINIMUM_NEGATIVE,
                "maximum_ece": MAXIMUM_ECE,
                "maximum_brier": MAXIMUM_BRIER,
                "occlusion_slices_required": True,
            }
            or any(value is not False for value in claims.values())
        ):
            raise ValueError("Calibration root is inconsistent or not fail-closed")
        for key in ("provider_result_sha256", "source_pts_sha256", "model_sha256"):
            if not valid_sha256(provider[key]):
                raise ValueError(f"Calibration provider binding {key} is invalid")
        if provider["profile_sha256"] is not None:
            raise ValueError("Unqualified V2a provider profile must remain absent")
        if not valid_sha256(provider["region_support_score_sha256"]):
            raise ValueError("Calibration provider score binding is invalid")
        available = payload["status"] == "calibration_evidence_only"
        if available:
            if (
                fixture["state"] != "independent_labels_bound"
                or fixture["independent_of_provider"] is not True
                or not valid_sha256(fixture["fixture_sha256"])
                or not valid_sha256(fixture["labels_sha256"])
            ):
                raise ValueError("Available calibration fixture binding is invalid")
        elif fixture != {
            "state": "unavailable",
            "fixture_sha256": None,
            "labels_sha256": None,
            "independent_of_provider": False,
        }:
            raise ValueError("Unavailable calibration must not imply label evidence")
        qualified = 0
        for name in REGION_NAMES:
            region = _require_keys(
                payload["regions"][name],
                {"error_bound", "overall", "occluded", "unoccluded", "gate_pass", "reasons"},
                f"region {name}",
            )
            error_bound = _require_keys(
                region["error_bound"],
                {"name", "threshold", "units", "state"},
                f"region {name} error bound",
            )
            if not isinstance(region["reasons"], list) or any(
                not isinstance(reason, str) or not reason for reason in region["reasons"]
            ):
                raise ValueError(f"Calibration region {name} reasons are invalid")
            if available:
                if (
                    error_bound["state"] != "named"
                    or not isinstance(error_bound["name"], str)
                    or not error_bound["name"]
                    or not isinstance(error_bound["units"], str)
                    or not error_bound["units"]
                    or not isinstance(error_bound["threshold"], (int, float))
                    or isinstance(error_bound["threshold"], bool)
                    or not np.isfinite(error_bound["threshold"])
                    or error_bound["threshold"] <= 0
                ):
                    raise ValueError(f"Calibration region {name} error bound is invalid")
                overall = _validate_metric_slice(region["overall"], f"{name} overall")
                occluded = _validate_metric_slice(region["occluded"], f"{name} occluded")
                unoccluded = _validate_metric_slice(region["unoccluded"], f"{name} unoccluded")
                expected_pass = bool(
                    _slice_passes(overall, require_classes=True)
                    and _slice_passes(occluded, require_classes=False)
                    and _slice_passes(unoccluded, require_classes=False)
                )
                if region["gate_pass"] is not expected_pass:
                    raise ValueError(f"Calibration region {name} gate is inconsistent")
                qualified += int(expected_pass)
            else:
                if error_bound != {
                    "name": None,
                    "threshold": None,
                    "units": None,
                    "state": "unknown",
                } or region["gate_pass"] is not False:
                    raise ValueError("Unavailable regional calibration must remain unknown")
                for slice_name in ("overall", "occluded", "unoccluded"):
                    metrics = _validate_metric_slice(
                        region[slice_name], f"{name} {slice_name}"
                    )
                    if metrics["sample_count"] != 0:
                        raise ValueError(
                            "Unavailable regional calibration cannot contain samples"
                        )
        if (
            summary["qualified_region_count"] != qualified
            or summary["all_regions_qualified"] is not (qualified == len(REGION_NAMES))
            or not isinstance(summary["reason"], str)
            or not summary["reason"]
        ):
            raise ValueError("Calibration summary is inconsistent")

    def validate_provider(
        self, provider: VisualTrackProviderResult, *, provider_result_sha256: str
    ) -> None:
        binding = self.payload["provider_binding"]
        if (
            binding["provider_result_sha256"] != provider_result_sha256
            or binding["source_pts_sha256"] != array_sha256(provider.source_pts)
            or binding["region_support_score_sha256"]
            != array_sha256(provider.region_support_score)
            or binding["model_sha256"] != provider.metadata["provider"]["model_sha256"]
        ):
            raise ValueError("Calibration evidence does not bind the provider result")


def _provider_binding(
    provider: VisualTrackProviderResult, provider_result_sha256: str
) -> dict[str, Any]:
    if not valid_sha256(provider_result_sha256):
        raise ValueError("Provider result artifact hash must be SHA-256")
    return {
        "provider_result_sha256": provider_result_sha256,
        "source_pts_sha256": array_sha256(provider.source_pts),
        "region_support_score_sha256": array_sha256(
            provider.region_support_score
        ),
        "model_sha256": provider.metadata["provider"]["model_sha256"],
        "profile_sha256": None,
    }


def _base_payload(
    provider: VisualTrackProviderResult, provider_result_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": VISUAL_TRACK_CALIBRATION_SCHEMA_VERSION,
        "kind": "visual_track_calibration",
        "policy": VISUAL_TRACK_CALIBRATION_POLICY,
        "motion_authority": MOTION_AUTHORITY,
        "consumed_by_retargeting": False,
        "provider_binding": _provider_binding(provider, provider_result_sha256),
        "method": {
            "target": "probability_measurement_error_is_within_named_bound",
            "binning": "equal_width_closed_final_bin",
            "bin_count": CALIBRATION_BIN_COUNT,
            "ece": "sum_bin_fraction_times_abs_mean_probability_minus_frequency",
            "brier": "mean_squared_probability_error",
        },
        "thresholds": {
            "minimum_samples": MINIMUM_SAMPLES,
            "minimum_positive": MINIMUM_POSITIVE,
            "minimum_negative": MINIMUM_NEGATIVE,
            "maximum_ece": MAXIMUM_ECE,
            "maximum_brier": MAXIMUM_BRIER,
            "occlusion_slices_required": True,
        },
        "region_order": list(REGION_NAMES),
        "claims": {
            "changes_final_gnm_motion": False,
            "grants_motion_authority": False,
            "confidence_calibrated_for_runtime": False,
            "production_validated": False,
        },
    }


def build_unavailable_calibration_evidence(
    provider: VisualTrackProviderResult,
    *,
    provider_result_sha256: str,
    reason: str = "no_independent_frame_labels",
) -> VisualTrackCalibrationEvidence:
    if not isinstance(reason, str) or not reason:
        raise ValueError("Calibration unavailability reason is required")
    payload = _base_payload(provider, provider_result_sha256)
    payload.update(
        {
            "status": "unavailable",
            "fixture_binding": {
                "state": "unavailable",
                "fixture_sha256": None,
                "labels_sha256": None,
                "independent_of_provider": False,
            },
            "regions": {
                name: {
                    "error_bound": {
                        "name": None,
                        "threshold": None,
                        "units": None,
                        "state": "unknown",
                    },
                    "overall": _metric_slice(
                        np.asarray([], dtype=np.float64),
                        np.asarray([], dtype=np.bool_),
                    ),
                    "occluded": _metric_slice(
                        np.asarray([], dtype=np.float64),
                        np.asarray([], dtype=np.bool_),
                    ),
                    "unoccluded": _metric_slice(
                        np.asarray([], dtype=np.float64),
                        np.asarray([], dtype=np.bool_),
                    ),
                    "gate_pass": False,
                    "reasons": [reason],
                }
                for name in REGION_NAMES
            },
            "summary": {
                "qualified_region_count": 0,
                "all_regions_qualified": False,
                "reason": reason,
            },
        }
    )
    return VisualTrackCalibrationEvidence(payload)


def build_calibration_evidence(
    provider: VisualTrackProviderResult,
    *,
    provider_result_sha256: str,
    fixture_sha256: str,
    labels_sha256: str,
    within_error_bound: np.ndarray,
    label_available: np.ndarray,
    occluded: np.ndarray,
    error_bounds: Mapping[str, tuple[str, float, str]],
) -> VisualTrackCalibrationEvidence:
    """Measure per-region calibration against independently supplied labels."""

    if not valid_sha256(fixture_sha256) or not valid_sha256(labels_sha256):
        raise ValueError("Calibration fixture and label hashes must be SHA-256")
    expected_shape = (provider.frame_count, len(REGION_NAMES))
    probability = np.asarray(provider.region_support_score, dtype=np.float64)
    target = np.asarray(within_error_bound, dtype=np.bool_)
    available = np.asarray(label_available, dtype=np.bool_)
    occlusion = np.asarray(occluded, dtype=np.bool_)
    if any(value.shape != expected_shape for value in (target, available, occlusion)):
        raise ValueError("Calibration arrays must match provider frames and regions")
    if set(error_bounds) != set(REGION_NAMES):
        raise ValueError("Calibration needs one named error bound per region")
    if np.any(~np.isfinite(probability[available])) or np.any(
        (probability[available] < 0) | (probability[available] > 1)
    ):
        raise ValueError(
            "Labeled provider support scores must be finite and lie in [0,1]"
        )
    payload = _base_payload(provider, provider_result_sha256)
    regions: dict[str, Any] = {}
    qualified = 0
    for index, name in enumerate(REGION_NAMES):
        bound_name, threshold, units = error_bounds[name]
        if (
            not isinstance(bound_name, str)
            or not bound_name
            or not isinstance(units, str)
            or not units
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not np.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError(f"Calibration error bound for {name} is invalid")
        selected = available[:, index]
        overall = _metric_slice(probability[selected, index], target[selected, index])
        selected_occluded = selected & occlusion[:, index]
        selected_unoccluded = selected & ~occlusion[:, index]
        occluded_metrics = _metric_slice(
            probability[selected_occluded, index], target[selected_occluded, index]
        )
        unoccluded_metrics = _metric_slice(
            probability[selected_unoccluded, index], target[selected_unoccluded, index]
        )
        passed = bool(
            _slice_passes(overall, require_classes=True)
            and _slice_passes(occluded_metrics, require_classes=False)
            and _slice_passes(unoccluded_metrics, require_classes=False)
        )
        qualified += int(passed)
        reasons: list[str] = []
        if overall["sample_count"] < MINIMUM_SAMPLES:
            reasons.append("insufficient_overall_samples")
        if overall["positive_count"] < MINIMUM_POSITIVE:
            reasons.append("insufficient_positive_samples")
        if overall["negative_count"] < MINIMUM_NEGATIVE:
            reasons.append("insufficient_negative_samples")
        if occluded_metrics["sample_count"] < MINIMUM_SAMPLES:
            reasons.append("insufficient_occluded_samples")
        if unoccluded_metrics["sample_count"] < MINIMUM_SAMPLES:
            reasons.append("insufficient_unoccluded_samples")
        if overall["ece"] is not None and overall["ece"] > MAXIMUM_ECE:
            reasons.append("ece_above_limit")
        if overall["brier"] is not None and overall["brier"] > MAXIMUM_BRIER:
            reasons.append("brier_above_limit")
        for slice_name, slice_metrics in (
            ("occluded", occluded_metrics),
            ("unoccluded", unoccluded_metrics),
        ):
            if (
                slice_metrics["ece"] is not None
                and slice_metrics["ece"] > MAXIMUM_ECE
            ):
                reasons.append(f"{slice_name}_ece_above_limit")
            if (
                slice_metrics["brier"] is not None
                and slice_metrics["brier"] > MAXIMUM_BRIER
            ):
                reasons.append(f"{slice_name}_brier_above_limit")
        if not reasons:
            reasons.append("shadow_calibration_gate_passed_no_authority_granted")
        regions[name] = {
            "error_bound": {
                "name": bound_name,
                "threshold": float(threshold),
                "units": units,
                "state": "named",
            },
            "overall": overall,
            "occluded": occluded_metrics,
            "unoccluded": unoccluded_metrics,
            "gate_pass": passed,
            "reasons": reasons,
        }
    payload.update(
        {
            "status": "calibration_evidence_only",
            "fixture_binding": {
                "state": "independent_labels_bound",
                "fixture_sha256": fixture_sha256,
                "labels_sha256": labels_sha256,
                "independent_of_provider": True,
            },
            "regions": regions,
            "summary": {
                "qualified_region_count": qualified,
                "all_regions_qualified": qualified == len(REGION_NAMES),
                "reason": "shadow_metrics_only_no_motion_authority",
            },
        }
    )
    return VisualTrackCalibrationEvidence(payload)


def write_visual_track_calibration(
    path: str | Path, evidence: VisualTrackCalibrationEvidence
) -> Path:
    return write_json(path, evidence.payload)


def load_visual_track_calibration(
    path: str | Path,
    *,
    expected_provider: VisualTrackProviderResult | None = None,
    expected_provider_path: str | Path | None = None,
) -> VisualTrackCalibrationEvidence:
    artifact = Path(path)
    try:
        if artifact.stat().st_size <= 0 or artifact.stat().st_size > MAX_CALIBRATION_BYTES:
            raise ValueError("Calibration artifact size is outside its bounds")
        payload = json.loads(
            artifact.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {item}")
            ),
        )
        evidence = VisualTrackCalibrationEvidence(payload)
        if expected_provider is not None:
            if expected_provider_path is None:
                raise ValueError("Expected provider path is required for binding validation")
            evidence.validate_provider(
                expected_provider,
                provider_result_sha256=file_sha256(expected_provider_path),
            )
        return evidence
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid VisualTrack calibration: {exc}") from exc


__all__ = [
    "CALIBRATION_BIN_COUNT",
    "MAXIMUM_BRIER",
    "MAXIMUM_ECE",
    "VISUAL_TRACK_CALIBRATION_POLICY",
    "VISUAL_TRACK_CALIBRATION_SCHEMA_VERSION",
    "VisualTrackCalibrationEvidence",
    "build_calibration_evidence",
    "build_unavailable_calibration_evidence",
    "load_visual_track_calibration",
    "write_visual_track_calibration",
]
