"""Pixel-derived, regional diagnostics for retained video performance captures.

Observation v3 is deliberately evidence-only.  It re-decodes the exact source
frames bound to a :class:`CaptureTrack`, measures regional image quality and
temporal consistency, and never mutates or authors animation controls.  The
scores are provisional and capped below the strong-trust tier until they are
calibrated on a locked, labeled stress set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
import zipfile

import cv2
import numpy as np

from .errors import AutoAnimError
from .serialization import write_json, write_npz
from .video_capture import (
    CaptureTrack,
    VideoDecodeLimits,
    decoded_video_frames,
    probe_video,
)
from .video_evidence import REGION_LANDMARKS


PIXEL_OBSERVATION_SCHEMA_VERSION = "autoanim.pixel-observation/1.0"
OBSERVATION_V3_SCHEMA_VERSION = "autoanim.performance-evidence.v3"
OBSERVATION_V3_VIEW_SCHEMA_VERSION = "autoanim.observation-v3-view/1.0"
OBSERVATION_V3_POLICY = "observation_only_pixel_diagnostics_no_motion_effect_v1"
PIXEL_ANALYZER_VERSION = "regional-pixels-v1"
PIXEL_DIAGNOSTIC_CONFIDENCE_CAP = 0.74
PATCH_SIZE = 48
THUMBNAIL_SIZE = 64
MAX_OBSERVATION_V3_BYTES = 4 * 1024 * 1024
MAX_OBSERVATION_V3_VIEW_FRAMES = 1_800
MAX_PIXEL_OBSERVATION_BYTES = 64 * 1024 * 1024
MAX_PIXEL_OBSERVATION_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

REASON_CODES = (
    "DETECTION_MISSING",
    "REGION_ROI_UNAVAILABLE",
    "REGION_OFFSCREEN",
    "REGION_TOO_SMALL",
    "BLUR_OR_LOW_DETAIL",
    "SEVERE_UNDEREXPOSURE",
    "SEVERE_OVEREXPOSURE",
    "LOW_DYNAMIC_RANGE",
    "TEMPORAL_PIXEL_DISCONTINUITY",
    "FLOW_UNAVAILABLE",
    "FLOW_INCONSISTENT",
    "PHOTOMETRIC_DISCONTINUITY_CANDIDATE",
    "SHOT_DISCONTINUITY_CANDIDATE",
    "OBSERVATION_EPOCH_START",
)
_REASON_BITS = {name: 1 << index for index, name in enumerate(REASON_CODES)}

_REGION_PADDING = {
    "mouth": (0.30, 0.65),
    "eyes": (0.18, 0.35),
    "upperFace": (0.18, 0.25),
    "head": (0.06, 0.06),
}
_NON_DEGRADING_REASONS = frozenset(
    {"FLOW_UNAVAILABLE", "OBSERVATION_EPOCH_START"}
)


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_unit_interval(array: np.ndarray) -> bool:
    finite = array[np.isfinite(array)]
    return bool(np.all((finite >= 0.0) & (finite <= 1.0)))


def _quantized(value: float) -> np.float32:
    return np.float32(round(float(value), 6))


def _focus_score_value(metric: float, reference: float) -> np.float32:
    absolute_low = math.log10(1e-4)
    absolute_high = math.log10(0.25)
    absolute = np.clip(
        (math.log10(max(metric, 1e-8)) - absolute_low)
        / (absolute_high - absolute_low),
        0.0,
        1.0,
    )
    relative = np.clip(metric / max(reference, 1e-8), 0.0, 1.0)
    return _quantized(min(float(absolute), float(relative)))


def _confidence_value(
    *,
    clipped_fraction: float,
    focus_score: float,
    shadow_fraction: float,
    highlight_fraction: float,
    dynamic_range: float,
    reason_mask: int,
) -> np.float32:
    shadow_score = 1.0 - float(
        np.clip((shadow_fraction - 0.60) / 0.35, 0.0, 1.0)
    )
    highlight_score = 1.0 - float(
        np.clip((highlight_fraction - 0.40) / 0.50, 0.0, 1.0)
    )
    contrast_score = float(np.clip((dynamic_range - 0.03) / 0.17, 0.0, 1.0))
    quality = min(focus_score, shadow_score, highlight_score, contrast_score)
    confidence = PIXEL_DIAGNOSTIC_CONFIDENCE_CAP * (1.0 - clipped_fraction) * (
        0.25 + 0.75 * quality
    )
    if reason_mask & (
        _REASON_BITS["TEMPORAL_PIXEL_DISCONTINUITY"]
        | _REASON_BITS["FLOW_INCONSISTENT"]
    ):
        confidence = min(confidence, 0.44)
    if reason_mask & (
        _REASON_BITS["SHOT_DISCONTINUITY_CANDIDATE"]
        | _REASON_BITS["PHOTOMETRIC_DISCONTINUITY_CANDIDATE"]
        | _REASON_BITS["REGION_TOO_SMALL"]
    ):
        confidence = min(confidence, 0.20)
    return _quantized(np.clip(confidence, 0.0, PIXEL_DIAGNOSTIC_CONFIDENCE_CAP))


@dataclass(frozen=True, slots=True)
class PixelObservationTrack:
    """Immutable regional evidence on the exact capture frame clock."""

    source_sha256: str
    source_pts: np.ndarray
    decoded_pixel_sha256: tuple[str, ...]
    region_names: tuple[str, ...]
    roi_boxes_xyxy: np.ndarray
    roi_pixel_count: np.ndarray
    clipped_fraction: np.ndarray
    focus_metric: np.ndarray
    focus_reference: np.ndarray
    focus_score: np.ndarray
    luma_mean: np.ndarray
    shadow_fraction: np.ndarray
    highlight_fraction: np.ndarray
    dynamic_range: np.ndarray
    temporal_innovation: np.ndarray
    flow_consistency: np.ndarray
    confidence: np.ndarray
    reason_mask: np.ndarray
    cut_histogram_distance: np.ndarray
    cut_thumbnail_mad: np.ndarray
    cut_thumbnail_zncc: np.ndarray
    photometric_discontinuity_candidate: np.ndarray
    cut_candidate: np.ndarray
    observation_epoch_start: np.ndarray
    schema_version: str = PIXEL_OBSERVATION_SCHEMA_VERSION
    analyzer_version: str = PIXEL_ANALYZER_VERSION

    def __post_init__(self) -> None:
        arrays = {
            "source_pts": (self.source_pts, np.int64),
            "roi_boxes_xyxy": (self.roi_boxes_xyxy, np.int32),
            "roi_pixel_count": (self.roi_pixel_count, np.int32),
            "clipped_fraction": (self.clipped_fraction, np.float32),
            "focus_metric": (self.focus_metric, np.float32),
            "focus_reference": (self.focus_reference, np.float32),
            "focus_score": (self.focus_score, np.float32),
            "luma_mean": (self.luma_mean, np.float32),
            "shadow_fraction": (self.shadow_fraction, np.float32),
            "highlight_fraction": (self.highlight_fraction, np.float32),
            "dynamic_range": (self.dynamic_range, np.float32),
            "temporal_innovation": (self.temporal_innovation, np.float32),
            "flow_consistency": (self.flow_consistency, np.float32),
            "confidence": (self.confidence, np.float32),
            "reason_mask": (self.reason_mask, np.uint32),
            "cut_histogram_distance": (self.cut_histogram_distance, np.float32),
            "cut_thumbnail_mad": (self.cut_thumbnail_mad, np.float32),
            "cut_thumbnail_zncc": (self.cut_thumbnail_zncc, np.float32),
            "photometric_discontinuity_candidate": (
                self.photometric_discontinuity_candidate,
                np.bool_,
            ),
            "cut_candidate": (self.cut_candidate, np.bool_),
            "observation_epoch_start": (self.observation_epoch_start, np.bool_),
        }
        for name, (value, dtype) in arrays.items():
            object.__setattr__(self, name, _readonly(value, dtype))
        names = tuple(self.region_names)
        object.__setattr__(self, "region_names", names)
        pixel_hashes = tuple(self.decoded_pixel_sha256)
        object.__setattr__(self, "decoded_pixel_sha256", pixel_hashes)
        count = len(self.source_pts)
        region_count = len(names)
        expected = {
            "roi_boxes_xyxy": (count, region_count, 4),
            "roi_pixel_count": (count, region_count),
            "clipped_fraction": (count, region_count),
            "focus_metric": (count, region_count),
            "focus_reference": (region_count,),
            "focus_score": (count, region_count),
            "luma_mean": (count, region_count),
            "shadow_fraction": (count, region_count),
            "highlight_fraction": (count, region_count),
            "dynamic_range": (count, region_count),
            "temporal_innovation": (count, region_count),
            "flow_consistency": (count, region_count),
            "confidence": (count, region_count),
            "reason_mask": (count, region_count),
            "cut_histogram_distance": (count,),
            "cut_thumbnail_mad": (count,),
            "cut_thumbnail_zncc": (count,),
            "photometric_discontinuity_candidate": (count,),
            "cut_candidate": (count,),
            "observation_epoch_start": (count,),
        }
        if (
            self.schema_version != PIXEL_OBSERVATION_SCHEMA_VERSION
            or self.analyzer_version != PIXEL_ANALYZER_VERSION
            or count <= 0
            or len(pixel_hashes) != count
            or names != tuple(REGION_LANDMARKS)
            or len(set(names)) != region_count
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ValueError("Pixel observation metadata is invalid")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in pixel_hashes
        ):
            raise ValueError("Decoded pixel hashes are invalid")
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"Pixel observation {name} must have shape {shape}")
        if count > 1 and np.any(np.diff(self.source_pts) <= 0):
            raise ValueError("Pixel observation PTS must be strictly increasing")
        for name in (
            "clipped_fraction",
            "focus_metric",
            "focus_reference",
            "focus_score",
            "luma_mean",
            "shadow_fraction",
            "highlight_fraction",
            "dynamic_range",
            "temporal_innovation",
            "flow_consistency",
            "confidence",
        ):
            if not _finite_unit_interval(getattr(self, name)):
                raise ValueError(f"Pixel observation {name} must lie in [0,1]")
        if np.any(self.confidence[np.isfinite(self.confidence)] > PIXEL_DIAGNOSTIC_CONFIDENCE_CAP):
            raise ValueError("Uncalibrated pixel evidence reached the strong tier")
        for name in ("cut_histogram_distance", "cut_thumbnail_mad"):
            values = getattr(self, name)
            if not _finite_unit_interval(values):
                raise ValueError(f"Pixel observation {name} must lie in [0,1]")
            if not math.isnan(float(values[0])) or (
                count > 1 and not np.all(np.isfinite(values[1:]))
            ):
                raise ValueError(f"Pixel observation {name} availability is invalid")
        zncc = self.cut_thumbnail_zncc
        finite_zncc = zncc[np.isfinite(zncc)]
        if (
            np.any((finite_zncc < -1.0) | (finite_zncc > 1.0))
            or not math.isnan(float(zncc[0]))
        ):
            raise ValueError("Pixel observation cut_thumbnail_zncc is invalid")
        if (
            bool(self.photometric_discontinuity_candidate[0])
            or bool(self.cut_candidate[0])
            or not bool(self.observation_epoch_start[0])
        ):
            raise ValueError("Pixel observation epoch must start on frame zero")
        unknown_bits = int(np.bitwise_or.reduce(self.reason_mask.reshape(-1), initial=0))
        known_bits = sum(_REASON_BITS.values())
        if unknown_bits & ~known_bits:
            raise ValueError("Pixel observation has an unknown reason bit")
        missing = (
            self.reason_mask & _REASON_BITS["DETECTION_MISSING"]
        ) != 0
        roi_unavailable = (
            self.reason_mask & _REASON_BITS["REGION_ROI_UNAVAILABLE"]
        ) != 0
        if np.any(missing & roi_unavailable) or np.any(
            missing != missing[:, :1]
        ):
            raise ValueError("Pixel observation detection availability is inconsistent")
        unavailable = missing | roi_unavailable
        available = ~unavailable
        if (
            np.any(self.roi_boxes_xyxy[unavailable] != -1)
            or np.any(self.roi_pixel_count[unavailable] != 0)
            or np.any(self.roi_boxes_xyxy[available] < 0)
            or np.any(self.roi_pixel_count[available] <= 0)
        ):
            raise ValueError("Pixel observation ROI availability is invalid")
        for name in (
            "clipped_fraction",
            "focus_metric",
            "focus_score",
            "luma_mean",
            "shadow_fraction",
            "highlight_fraction",
            "dynamic_range",
            "confidence",
        ):
            values = getattr(self, name)
            if np.any(np.isfinite(values[unavailable])) or np.any(
                ~np.isfinite(values[available])
            ):
                raise ValueError(f"Pixel observation {name} availability is invalid")
        for name in ("temporal_innovation", "flow_consistency"):
            if np.any(np.isfinite(getattr(self, name)[unavailable])):
                raise ValueError(f"Pixel observation {name} availability is invalid")
        for region_index in range(region_count):
            region_available = available[:, region_index]
            reference = float(self.focus_reference[region_index])
            if np.any(region_available):
                if not math.isfinite(reference) or reference <= 0.0:
                    raise ValueError("Pixel observation focus reference is unavailable")
                expected_scores = np.asarray(
                    [
                        _focus_score_value(float(metric), reference)
                        for metric in self.focus_metric[region_available, region_index]
                    ],
                    dtype=np.float32,
                )
                if not np.array_equal(
                    self.focus_score[region_available, region_index], expected_scores
                ):
                    raise ValueError("Pixel observation focus score is inconsistent")
            elif not math.isnan(reference):
                raise ValueError("Pixel observation focus reference must be unavailable")
        detected = ~missing[:, 0]
        expected_epoch_start = np.zeros(count, dtype=bool)
        expected_epoch_start[0] = True
        if count > 1:
            expected_epoch_start[1:] = (
                self.photometric_discontinuity_candidate[1:]
                | self.cut_candidate[1:]
                | (detected[1:] & ~detected[:-1])
            )
        if not np.array_equal(self.observation_epoch_start, expected_epoch_start):
            raise ValueError("Pixel observation epoch boundaries are inconsistent")
        expected_photometric = (
            (self.cut_histogram_distance >= 0.55)
            & (self.cut_thumbnail_mad >= 0.18)
        )
        expected_photometric[0] = False
        expected_cuts = expected_photometric & np.isfinite(
            self.cut_thumbnail_zncc
        ) & (self.cut_thumbnail_zncc <= 0.45)
        expected_cuts[0] = False
        if not np.array_equal(
            self.photometric_discontinuity_candidate, expected_photometric
        ) or not np.array_equal(self.cut_candidate, expected_cuts):
            raise ValueError("Pixel observation cut decisions are inconsistent")
        temporal_expected = np.zeros((count, region_count), dtype=bool)
        if count > 1:
            temporal_expected[1:] = (
                available[1:]
                & available[:-1]
                & ~self.observation_epoch_start[1:, None]
            )
        if (
            np.any(~np.isfinite(self.temporal_innovation[temporal_expected]))
            or np.any(np.isfinite(self.temporal_innovation[~temporal_expected]))
            or np.any(np.isfinite(self.flow_consistency[~temporal_expected]))
        ):
            raise ValueError("Pixel observation temporal availability is inconsistent")
        cut_reason = (
            self.reason_mask & _REASON_BITS["SHOT_DISCONTINUITY_CANDIDATE"]
        ) != 0
        photometric_reason = (
            self.reason_mask
            & _REASON_BITS["PHOTOMETRIC_DISCONTINUITY_CANDIDATE"]
        ) != 0
        epoch_reason = (
            self.reason_mask & _REASON_BITS["OBSERVATION_EPOCH_START"]
        ) != 0
        expected_cut_reason = np.broadcast_to(
            self.cut_candidate[:, None], cut_reason.shape
        )
        expected_photometric_reason = np.broadcast_to(
            self.photometric_discontinuity_candidate[:, None],
            photometric_reason.shape,
        )
        expected_epoch_reason = np.broadcast_to(
            self.observation_epoch_start[:, None], epoch_reason.shape
        )
        if (
            np.any(cut_reason[available] != expected_cut_reason[available])
            or np.any(
                photometric_reason[available]
                != expected_photometric_reason[available]
            )
            or np.any(epoch_reason[available] != expected_epoch_reason[available])
        ):
            raise ValueError("Pixel observation event reasons are inconsistent")
        expected_masks = np.zeros_like(self.reason_mask)
        for frame_index in range(count):
            for region_index in range(region_count):
                if missing[frame_index, region_index]:
                    expected_masks[frame_index, region_index] = _REASON_BITS[
                        "DETECTION_MISSING"
                    ]
                    continue
                if roi_unavailable[frame_index, region_index]:
                    expected_masks[frame_index, region_index] = _REASON_BITS[
                        "REGION_ROI_UNAVAILABLE"
                    ]
                    continue
                mask = 0
                x0, y0, x1, y1 = self.roi_boxes_xyxy[
                    frame_index, region_index
                ]
                if self.clipped_fraction[frame_index, region_index] > 0.15:
                    mask |= _REASON_BITS["REGION_OFFSCREEN"]
                if (
                    min(int(x1 - x0), int(y1 - y0)) < 12
                    or int(self.roi_pixel_count[frame_index, region_index]) < 256
                ):
                    mask |= _REASON_BITS["REGION_TOO_SMALL"]
                if self.focus_score[frame_index, region_index] < 0.30:
                    mask |= _REASON_BITS["BLUR_OR_LOW_DETAIL"]
                if self.shadow_fraction[frame_index, region_index] > 0.85:
                    mask |= _REASON_BITS["SEVERE_UNDEREXPOSURE"]
                if self.highlight_fraction[frame_index, region_index] > 0.60:
                    mask |= _REASON_BITS["SEVERE_OVEREXPOSURE"]
                if self.dynamic_range[frame_index, region_index] < 0.08:
                    mask |= _REASON_BITS["LOW_DYNAMIC_RANGE"]
                if temporal_expected[frame_index, region_index]:
                    if self.temporal_innovation[frame_index, region_index] > 0.30:
                        mask |= _REASON_BITS["TEMPORAL_PIXEL_DISCONTINUITY"]
                    flow = self.flow_consistency[frame_index, region_index]
                    if math.isnan(float(flow)):
                        mask |= _REASON_BITS["FLOW_UNAVAILABLE"]
                    elif flow < 0.45:
                        mask |= _REASON_BITS["FLOW_INCONSISTENT"]
                else:
                    mask |= _REASON_BITS["FLOW_UNAVAILABLE"]
                if self.photometric_discontinuity_candidate[frame_index]:
                    mask |= _REASON_BITS["PHOTOMETRIC_DISCONTINUITY_CANDIDATE"]
                if self.cut_candidate[frame_index]:
                    mask |= _REASON_BITS["SHOT_DISCONTINUITY_CANDIDATE"]
                if self.observation_epoch_start[frame_index]:
                    mask |= _REASON_BITS["OBSERVATION_EPOCH_START"]
                expected_masks[frame_index, region_index] = mask
                expected_confidence = _confidence_value(
                    clipped_fraction=float(
                        self.clipped_fraction[frame_index, region_index]
                    ),
                    focus_score=float(self.focus_score[frame_index, region_index]),
                    shadow_fraction=float(
                        self.shadow_fraction[frame_index, region_index]
                    ),
                    highlight_fraction=float(
                        self.highlight_fraction[frame_index, region_index]
                    ),
                    dynamic_range=float(
                        self.dynamic_range[frame_index, region_index]
                    ),
                    reason_mask=mask,
                )
                if self.confidence[frame_index, region_index] != expected_confidence:
                    raise ValueError("Pixel observation confidence is inconsistent")
        if not np.array_equal(self.reason_mask, expected_masks):
            raise ValueError("Pixel observation reason mask is inconsistent")

    @property
    def frame_count(self) -> int:
        return len(self.source_pts)

    def validate_capture(self, capture: CaptureTrack) -> None:
        if (
            self.source_sha256 != capture.provenance.source_sha256
            or self.frame_count != capture.frame_count
            or not np.array_equal(self.source_pts, capture.source_pts)
        ):
            raise ValueError("Pixel observations are not bound to this capture")
        missing = (
            self.reason_mask & _REASON_BITS["DETECTION_MISSING"]
        ) != 0
        if not np.array_equal(~missing[:, 0], capture.detected):
            raise ValueError("Pixel observation detection state differs from capture")
        for frame_index in range(self.frame_count):
            if not capture.detected[frame_index]:
                continue
            for region_index, region_name in enumerate(self.region_names):
                expected_box, expected_clipped = _region_box(
                    capture, frame_index, region_name
                )
                roi_unavailable = bool(
                    self.reason_mask[frame_index, region_index]
                    & _REASON_BITS["REGION_ROI_UNAVAILABLE"]
                )
                if expected_box is None:
                    if not roi_unavailable:
                        raise ValueError(
                            "Pixel observation omitted an unavailable capture ROI"
                        )
                    continue
                if roi_unavailable:
                    raise ValueError("Pixel observation discarded an available capture ROI")
                box = tuple(
                    int(value)
                    for value in self.roi_boxes_xyxy[frame_index, region_index]
                )
                x0, y0, x1, y1 = box
                expected_count = (x1 - x0) * (y1 - y0)
                if (
                    box != expected_box
                    or int(self.roi_pixel_count[frame_index, region_index])
                    != expected_count
                    or float(self.clipped_fraction[frame_index, region_index])
                    != float(np.float32(round(expected_clipped, 6)))
                ):
                    raise ValueError("Pixel observation ROI differs from capture geometry")

    def region_record(self, frame_index: int, region_name: str) -> dict[str, Any]:
        try:
            region_index = self.region_names.index(region_name)
        except ValueError as exc:
            raise KeyError(region_name) from exc
        mask = int(self.reason_mask[frame_index, region_index])
        reasons = [name for name in REASON_CODES if mask & _REASON_BITS[name]]
        confidence = float(self.confidence[frame_index, region_index])
        box = self.roi_boxes_xyxy[frame_index, region_index]

        def optional(array: np.ndarray) -> float | None:
            value = float(array[frame_index, region_index])
            return value if math.isfinite(value) else None

        if "DETECTION_MISSING" in reasons:
            state = "missing"
        elif "REGION_ROI_UNAVAILABLE" in reasons:
            state = "unknown"
        elif any(reason not in _NON_DEGRADING_REASONS for reason in reasons):
            state = "degraded"
        else:
            state = "diagnostic_clear"
        return {
            "schemaVersion": self.schema_version,
            "analyzerVersion": self.analyzer_version,
            "qualityState": state,
            "reasonCodes": reasons,
            "confidence": confidence if math.isfinite(confidence) else None,
            "confidenceCalibrated": False,
            "roiBoxXYXY": box.tolist() if np.all(box >= 0) else None,
            "roiPixelCount": int(self.roi_pixel_count[frame_index, region_index]),
            "clippedFraction": optional(self.clipped_fraction),
            "focusMetric": optional(self.focus_metric),
            "focusScore": optional(self.focus_score),
            "lumaMean": optional(self.luma_mean),
            "shadowFraction": optional(self.shadow_fraction),
            "highlightFraction": optional(self.highlight_fraction),
            "dynamicRange": optional(self.dynamic_range),
            "temporalInnovation": optional(self.temporal_innovation),
            "flowConsistency": optional(self.flow_consistency),
            "occlusionState": "unknown",
            "identityContinuityState": "unknown",
        }


def _region_box(
    capture: CaptureTrack,
    frame_index: int,
    region_name: str,
) -> tuple[tuple[int, int, int, int] | None, float]:
    indices = np.asarray(REGION_LANDMARKS[region_name], dtype=np.int64)
    points = np.asarray(capture.landmarks_xyz[frame_index, indices, :2], dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None, 1.0
    pixels = points * np.asarray((capture.width, capture.height), dtype=np.float64)
    low = np.min(pixels, axis=0)
    high = np.max(pixels, axis=0)
    span = np.maximum(high - low, 4.0)
    pad_x, pad_y = _REGION_PADDING[region_name]
    low -= np.asarray((span[0] * pad_x, span[1] * pad_y))
    high += np.asarray((span[0] * pad_x, span[1] * pad_y))
    center = 0.5 * (low + high)
    half = np.maximum(0.5 * (high - low), 4.0)
    low = center - half
    high = center + half
    raw_area = max(float((high[0] - low[0]) * (high[1] - low[1])), 1.0)
    clipped_low = np.maximum(low, 0.0)
    clipped_high = np.minimum(high, (capture.width, capture.height))
    if np.any(clipped_high <= clipped_low):
        return None, 1.0
    clipped_area = float(np.prod(clipped_high - clipped_low))
    clipped_fraction = float(np.clip(1.0 - clipped_area / raw_area, 0.0, 1.0))
    x0, y0 = np.floor(clipped_low).astype(np.int64)
    x1, y1 = np.ceil(clipped_high).astype(np.int64)
    x0 = int(np.clip(x0, 0, capture.width - 1))
    y0 = int(np.clip(y0, 0, capture.height - 1))
    x1 = int(np.clip(x1, x0 + 1, capture.width))
    y1 = int(np.clip(y1, y0 + 1, capture.height))
    return (x0, y0, x1, y1), clipped_fraction


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _patch(gray: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    crop = np.ascontiguousarray(gray[y0:y1, x0:x1])
    resized = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return crop, np.asarray(resized, dtype=np.float32)


def _focus_metric(crop: np.ndarray) -> float:
    laplacian = cv2.Laplacian(crop, cv2.CV_32F, ksize=3)
    return float(np.clip(np.mean(np.square(laplacian)), 0.0, 1.0))


def _flow_consistency(previous: np.ndarray, current: np.ndarray) -> float:
    if float(np.var(previous)) <= 1e-5 or float(np.var(current)) <= 1e-5:
        return math.nan
    before = np.rint(previous * 255.0).astype(np.uint8)
    after = np.rint(current * 255.0).astype(np.uint8)
    forward = cv2.calcOpticalFlowFarneback(
        before, after, None, 0.5, 2, 11, 2, 5, 1.1, 0
    )
    backward = cv2.calcOpticalFlowFarneback(
        after, before, None, 0.5, 2, 11, 2, 5, 1.1, 0
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(PATCH_SIZE, dtype=np.float32),
        np.arange(PATCH_SIZE, dtype=np.float32),
    )
    map_x = grid_x + forward[..., 0]
    map_y = grid_y + forward[..., 1]
    valid = (
        (map_x >= 0)
        & (map_x <= PATCH_SIZE - 1)
        & (map_y >= 0)
        & (map_y <= PATCH_SIZE - 1)
    )
    if np.count_nonzero(valid) < PATCH_SIZE * PATCH_SIZE // 2:
        return math.nan
    sampled_x = cv2.remap(
        backward[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    sampled_y = cv2.remap(
        backward[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    error = np.sqrt(np.square(forward[..., 0] + sampled_x) + np.square(forward[..., 1] + sampled_y))
    median_error = float(np.median(error[valid]))
    return float(np.clip(math.exp(-median_error / 1.5), 0.0, 1.0))


def _shot_cut(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> tuple[float, float, float, bool, bool]:
    if previous is None:
        return math.nan, math.nan, math.nan, False, False
    before_u8 = np.rint(previous * 255.0).astype(np.uint8)
    after_u8 = np.rint(current * 255.0).astype(np.uint8)
    histogram_before = np.bincount((before_u8 >> 3).reshape(-1), minlength=32)
    histogram_after = np.bincount((after_u8 >> 3).reshape(-1), minlength=32)
    histogram_before = histogram_before.astype(np.float64) / before_u8.size
    histogram_after = histogram_after.astype(np.float64) / after_u8.size
    distance = float(_quantized(0.5 * np.sum(np.abs(histogram_before - histogram_after))))
    mad = float(_quantized(np.mean(np.abs(previous - current))))
    before_centered = previous.astype(np.float64) - float(np.mean(previous))
    after_centered = current.astype(np.float64) - float(np.mean(current))
    before_energy = float(np.sum(np.square(before_centered)))
    after_energy = float(np.sum(np.square(after_centered)))
    if (
        before_energy / previous.size <= 1e-12
        or after_energy / current.size <= 1e-12
    ):
        zncc = math.nan
    else:
        denominator = float(np.sqrt(before_energy * after_energy))
        zncc = float(
            _quantized(
                np.clip(
                    np.sum(before_centered * after_centered) / denominator,
                    -1.0,
                    1.0,
                )
            )
        )
    photometric = distance >= 0.55 and mad >= 0.18
    candidate = photometric and math.isfinite(zncc) and zncc <= 0.45
    return distance, mad, zncc, bool(photometric), bool(candidate)


def analyze_rgb_frames(
    capture: CaptureTrack,
    frames: Iterable[np.ndarray],
) -> PixelObservationTrack:
    """Analyze exactly one RGB frame per capture PTS without retaining pixels."""

    count = capture.frame_count
    names = tuple(REGION_LANDMARKS)
    region_count = len(names)
    boxes = np.full((count, region_count, 4), -1, dtype=np.int32)
    pixel_count = np.zeros((count, region_count), dtype=np.int32)
    float_arrays = {
        name: np.full((count, region_count), np.nan, dtype=np.float32)
        for name in (
            "clipped_fraction",
            "focus_metric",
            "focus_score",
            "luma_mean",
            "shadow_fraction",
            "highlight_fraction",
            "dynamic_range",
            "temporal_innovation",
            "flow_consistency",
            "confidence",
        )
    }
    reasons = np.zeros((count, region_count), dtype=np.uint32)
    focus_reference = np.full(region_count, np.nan, dtype=np.float32)
    cut_histogram_distance = np.full(count, np.nan, dtype=np.float32)
    cut_thumbnail_mad = np.full(count, np.nan, dtype=np.float32)
    cut_thumbnail_zncc = np.full(count, np.nan, dtype=np.float32)
    photometric_discontinuities = np.zeros(count, dtype=bool)
    cuts = np.zeros(count, dtype=bool)
    epoch_starts = np.zeros(count, dtype=bool)
    epoch_starts[0] = True
    previous_thumbnail: np.ndarray | None = None
    previous_patches: dict[str, np.ndarray] = {}
    previous_detected = False
    observed_frames = 0
    decoded_hashes: list[str] = []

    for frame_index, rgb in enumerate(frames):
        if frame_index >= count:
            raise ValueError("Pixel decoder produced more frames than the capture")
        image = np.asarray(rgb)
        if image.shape != (capture.height, capture.width, 3) or image.dtype != np.uint8:
            raise ValueError("Pixel frame shape or dtype does not match the capture")
        observed_frames += 1
        decoded_hashes.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
        gray = _gray(image)
        thumbnail = cv2.resize(
            gray, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA
        )
        (
            histogram_distance,
            thumbnail_mad,
            thumbnail_zncc,
            photometric_discontinuity,
            cut_candidate,
        ) = _shot_cut(previous_thumbnail, thumbnail)
        cut_histogram_distance[frame_index] = histogram_distance
        cut_thumbnail_mad[frame_index] = thumbnail_mad
        cut_thumbnail_zncc[frame_index] = thumbnail_zncc
        photometric_discontinuities[frame_index] = photometric_discontinuity
        cuts[frame_index] = cut_candidate
        detected = bool(capture.detected[frame_index])
        if frame_index > 0:
            epoch_starts[frame_index] = bool(
                photometric_discontinuities[frame_index]
                or cuts[frame_index]
                or (detected and not previous_detected)
            )
        for region_index, region_name in enumerate(names):
            if not detected:
                reasons[frame_index, region_index] |= _REASON_BITS["DETECTION_MISSING"]
                continue
            box, clipped = _region_box(capture, frame_index, region_name)
            if box is None:
                reasons[frame_index, region_index] |= _REASON_BITS[
                    "REGION_ROI_UNAVAILABLE"
                ]
                previous_patches.pop(region_name, None)
                continue
            boxes[frame_index, region_index] = box
            x0, y0, x1, y1 = box
            area = (x1 - x0) * (y1 - y0)
            pixel_count[frame_index, region_index] = area
            float_arrays["clipped_fraction"][frame_index, region_index] = clipped
            crop, patch = _patch(gray, box)
            metric = _focus_metric(crop)
            luma = float(np.mean(crop))
            shadow = float(np.mean(crop <= 0.02))
            highlight = float(np.mean(crop >= 0.98))
            p05, p95 = np.percentile(crop, (5, 95))
            span = float(np.clip(p95 - p05, 0.0, 1.0))
            float_arrays["focus_metric"][frame_index, region_index] = metric
            float_arrays["luma_mean"][frame_index, region_index] = luma
            float_arrays["shadow_fraction"][frame_index, region_index] = shadow
            float_arrays["highlight_fraction"][frame_index, region_index] = highlight
            float_arrays["dynamic_range"][frame_index, region_index] = span

            previous = previous_patches.get(region_name)
            if previous is not None and not epoch_starts[frame_index]:
                innovation = float(np.mean(np.abs(previous - patch)))
                consistency = _flow_consistency(previous, patch)
                float_arrays["temporal_innovation"][frame_index, region_index] = innovation
                if math.isfinite(consistency):
                    float_arrays["flow_consistency"][
                        frame_index, region_index
                    ] = consistency
            previous_patches[region_name] = patch

        if not detected:
            previous_patches.clear()
        previous_thumbnail = thumbnail
        previous_detected = detected

    if observed_frames != count:
        raise ValueError(
            f"Pixel decoder produced {observed_frames} frames for a {count}-frame capture"
        )
    for values in (
        *float_arrays.values(),
        cut_histogram_distance,
        cut_thumbnail_mad,
        cut_thumbnail_zncc,
    ):
        finite = np.isfinite(values)
        values[finite] = np.round(values[finite], 6)
    for region_index in range(region_count):
        valid = np.isfinite(float_arrays["focus_metric"][:, region_index])
        if not np.any(valid):
            continue
        eligible = (
            valid
            & (float_arrays["shadow_fraction"][:, region_index] <= 0.85)
            & (float_arrays["highlight_fraction"][:, region_index] <= 0.60)
            & (float_arrays["dynamic_range"][:, region_index] >= 0.08)
        )
        reference_values = float_arrays["focus_metric"][
            eligible if np.any(eligible) else valid, region_index
        ]
        reference = _quantized(max(float(np.percentile(reference_values, 90)), 1e-6))
        focus_reference[region_index] = reference
        for frame_index in np.flatnonzero(valid):
            float_arrays["focus_score"][frame_index, region_index] = (
                _focus_score_value(
                    float(float_arrays["focus_metric"][frame_index, region_index]),
                    float(reference),
                )
            )
    for frame_index in range(count):
        for region_index in range(region_count):
            if reasons[frame_index, region_index] & (
                _REASON_BITS["DETECTION_MISSING"]
                | _REASON_BITS["REGION_ROI_UNAVAILABLE"]
            ):
                continue
            mask = 0
            x0, y0, x1, y1 = boxes[frame_index, region_index]
            if float_arrays["clipped_fraction"][frame_index, region_index] > 0.15:
                mask |= _REASON_BITS["REGION_OFFSCREEN"]
            if (
                min(int(x1 - x0), int(y1 - y0)) < 12
                or int(pixel_count[frame_index, region_index]) < 256
            ):
                mask |= _REASON_BITS["REGION_TOO_SMALL"]
            if float_arrays["focus_score"][frame_index, region_index] < 0.30:
                mask |= _REASON_BITS["BLUR_OR_LOW_DETAIL"]
            if float_arrays["shadow_fraction"][frame_index, region_index] > 0.85:
                mask |= _REASON_BITS["SEVERE_UNDEREXPOSURE"]
            if float_arrays["highlight_fraction"][frame_index, region_index] > 0.60:
                mask |= _REASON_BITS["SEVERE_OVEREXPOSURE"]
            if float_arrays["dynamic_range"][frame_index, region_index] < 0.08:
                mask |= _REASON_BITS["LOW_DYNAMIC_RANGE"]
            innovation = float_arrays["temporal_innovation"][
                frame_index, region_index
            ]
            flow = float_arrays["flow_consistency"][frame_index, region_index]
            if math.isfinite(float(innovation)):
                if innovation > 0.30:
                    mask |= _REASON_BITS["TEMPORAL_PIXEL_DISCONTINUITY"]
                if math.isnan(float(flow)):
                    mask |= _REASON_BITS["FLOW_UNAVAILABLE"]
                elif flow < 0.45:
                    mask |= _REASON_BITS["FLOW_INCONSISTENT"]
            else:
                mask |= _REASON_BITS["FLOW_UNAVAILABLE"]
            if photometric_discontinuities[frame_index]:
                mask |= _REASON_BITS["PHOTOMETRIC_DISCONTINUITY_CANDIDATE"]
            if cuts[frame_index]:
                mask |= _REASON_BITS["SHOT_DISCONTINUITY_CANDIDATE"]
            if epoch_starts[frame_index]:
                mask |= _REASON_BITS["OBSERVATION_EPOCH_START"]
            reasons[frame_index, region_index] = mask
            float_arrays["confidence"][frame_index, region_index] = (
                _confidence_value(
                    clipped_fraction=float(
                        float_arrays["clipped_fraction"][frame_index, region_index]
                    ),
                    focus_score=float(
                        float_arrays["focus_score"][frame_index, region_index]
                    ),
                    shadow_fraction=float(
                        float_arrays["shadow_fraction"][frame_index, region_index]
                    ),
                    highlight_fraction=float(
                        float_arrays["highlight_fraction"][frame_index, region_index]
                    ),
                    dynamic_range=float(
                        float_arrays["dynamic_range"][frame_index, region_index]
                    ),
                    reason_mask=mask,
                )
            )
    return PixelObservationTrack(
        source_sha256=capture.provenance.source_sha256,
        source_pts=capture.source_pts,
        decoded_pixel_sha256=tuple(decoded_hashes),
        region_names=names,
        roi_boxes_xyxy=boxes,
        roi_pixel_count=pixel_count,
        focus_reference=focus_reference,
        reason_mask=reasons,
        cut_histogram_distance=cut_histogram_distance,
        cut_thumbnail_mad=cut_thumbnail_mad,
        cut_thumbnail_zncc=cut_thumbnail_zncc,
        photometric_discontinuity_candidate=photometric_discontinuities,
        cut_candidate=cuts,
        observation_epoch_start=epoch_starts,
        **float_arrays,
    )


def analyze_video_pixels(
    video_path: str | Path,
    capture: CaptureTrack,
    *,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    limits: VideoDecodeLimits = VideoDecodeLimits(),
) -> PixelObservationTrack:
    """Re-read and bind exact source pixels to an existing capture track."""

    source = Path(video_path).expanduser().resolve()
    if (
        not source.is_file()
        or source.stat().st_size != capture.provenance.source_bytes
        or _sha256(source) != capture.provenance.source_sha256
    ):
        raise AutoAnimError(
            "INPUT_CHANGED", "Video source changed before pixel evidence analysis"
        )
    probe = probe_video(source, ffprobe_bin=ffprobe_bin, limits=limits)
    if (
        probe.width != capture.width
        or probe.height != capture.height
        or probe.frame_count != capture.frame_count
        or probe.time_base_numerator != capture.provenance.time_base_numerator
        or probe.time_base_denominator != capture.provenance.time_base_denominator
        or not np.array_equal(probe.source_pts, capture.source_pts)
    ):
        raise AutoAnimError(
            "INPUT_CHANGED", "Video decode clock changed after face capture"
        )
    with decoded_video_frames(probe, ffmpeg_bin=ffmpeg_bin) as decoded:
        observations = analyze_rgb_frames(capture, (frame.rgb for frame in decoded))
    observations.validate_capture(capture)
    return observations


def _pixel_configuration(analyzer_version: str) -> dict[str, Any]:
    return {
        "analyzerVersion": analyzer_version,
        "confidenceCap": PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
        "patchSize": PATCH_SIZE,
        "thumbnailSize": THUMBNAIL_SIZE,
        "focusScoring": {
            "absoluteMetricFloor": 0.0001,
            "absoluteMetricCeiling": 0.25,
            "takeRelativeReference": "eligible_region_p90",
            "blurOrLowDetailMaximum": 0.30,
        },
        "cutThresholds": {
            "histogramDistanceMinimum": 0.55,
            "thumbnailMADMinimum": 0.18,
            "thumbnailZNCCMaximum": 0.45,
        },
        "reasonCodes": list(REASON_CODES),
        "claims": {
            "changesFinalGNMMotion": False,
            "confidenceCalibrated": False,
            "occlusionValidated": False,
            "identityContinuityValidated": False,
            "productionValidated": False,
        },
    }


def write_pixel_observations(
    path: str | Path,
    observations: PixelObservationTrack,
) -> Path:
    configuration = json.dumps(
        _pixel_configuration(observations.analyzer_version),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return write_npz(
        path,
        schema_version=np.asarray(observations.schema_version),
        analyzer_version=np.asarray(observations.analyzer_version),
        source_sha256=np.asarray(observations.source_sha256),
        source_pts=observations.source_pts,
        decoded_pixel_sha256=np.asarray(observations.decoded_pixel_sha256),
        region_names=np.asarray(observations.region_names),
        roi_boxes_xyxy=observations.roi_boxes_xyxy,
        roi_pixel_count=observations.roi_pixel_count,
        clipped_fraction=observations.clipped_fraction,
        focus_metric=observations.focus_metric,
        focus_reference=observations.focus_reference,
        focus_score=observations.focus_score,
        luma_mean=observations.luma_mean,
        shadow_fraction=observations.shadow_fraction,
        highlight_fraction=observations.highlight_fraction,
        dynamic_range=observations.dynamic_range,
        temporal_innovation=observations.temporal_innovation,
        flow_consistency=observations.flow_consistency,
        confidence=observations.confidence,
        reason_mask=observations.reason_mask,
        cut_histogram_distance=observations.cut_histogram_distance,
        cut_thumbnail_mad=observations.cut_thumbnail_mad,
        cut_thumbnail_zncc=observations.cut_thumbnail_zncc,
        photometric_discontinuity_candidate=(
            observations.photometric_discontinuity_candidate
        ),
        cut_candidate=observations.cut_candidate,
        observation_epoch_start=observations.observation_epoch_start,
        configuration_json=np.asarray(configuration),
    )


def load_pixel_observations(path: str | Path) -> PixelObservationTrack:
    source = Path(path)
    if source.stat().st_size <= 0 or source.stat().st_size > MAX_PIXEL_OBSERVATION_BYTES:
        raise AutoAnimError(
            "MEDIA_INVALID", "Pixel observation artifact exceeds its byte limit"
        )
    try:
        with zipfile.ZipFile(source) as archive:
            member_names = [item.filename for item in archive.infolist()]
            if (
                len(member_names) != len(set(member_names))
                or len(member_names) > 64
                or sum(item.file_size for item in archive.infolist())
                > MAX_PIXEL_OBSERVATION_UNCOMPRESSED_BYTES
            ):
                raise ValueError(
                    "Pixel observation archive has duplicate members or exceeds its limit"
                )
        with np.load(source, allow_pickle=False) as values:
            expected_keys = {
                "schema_version",
                "analyzer_version",
                "source_sha256",
                "source_pts",
                "decoded_pixel_sha256",
                "region_names",
                "roi_boxes_xyxy",
                "roi_pixel_count",
                "clipped_fraction",
                "focus_metric",
                "focus_reference",
                "focus_score",
                "luma_mean",
                "shadow_fraction",
                "highlight_fraction",
                "dynamic_range",
                "temporal_innovation",
                "flow_consistency",
                "confidence",
                "reason_mask",
                "cut_histogram_distance",
                "cut_thumbnail_mad",
                "cut_thumbnail_zncc",
                "photometric_discontinuity_candidate",
                "cut_candidate",
                "observation_epoch_start",
                "configuration_json",
            }
            if set(values.files) != expected_keys:
                raise ValueError("Pixel observation arrays do not match the schema")
            exact_dtypes = {
                "source_pts": np.dtype(np.int64),
                "roi_boxes_xyxy": np.dtype(np.int32),
                "roi_pixel_count": np.dtype(np.int32),
                "clipped_fraction": np.dtype(np.float32),
                "focus_metric": np.dtype(np.float32),
                "focus_reference": np.dtype(np.float32),
                "focus_score": np.dtype(np.float32),
                "luma_mean": np.dtype(np.float32),
                "shadow_fraction": np.dtype(np.float32),
                "highlight_fraction": np.dtype(np.float32),
                "dynamic_range": np.dtype(np.float32),
                "temporal_innovation": np.dtype(np.float32),
                "flow_consistency": np.dtype(np.float32),
                "confidence": np.dtype(np.float32),
                "reason_mask": np.dtype(np.uint32),
                "cut_histogram_distance": np.dtype(np.float32),
                "cut_thumbnail_mad": np.dtype(np.float32),
                "cut_thumbnail_zncc": np.dtype(np.float32),
                "photometric_discontinuity_candidate": np.dtype(np.bool_),
                "cut_candidate": np.dtype(np.bool_),
                "observation_epoch_start": np.dtype(np.bool_),
            }
            if any(values[name].dtype != dtype for name, dtype in exact_dtypes.items()):
                raise ValueError("Pixel observation array dtype does not match the schema")
            if any(
                values[name].dtype.kind != "U"
                for name in (
                    "schema_version",
                    "analyzer_version",
                    "source_sha256",
                    "decoded_pixel_sha256",
                    "region_names",
                    "configuration_json",
                )
            ):
                raise ValueError("Pixel observation text arrays must be Unicode")
            configuration = json.loads(
                str(values["configuration_json"].item()),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-finite JSON number: {value}")
                ),
            )
            if configuration != _pixel_configuration(PIXEL_ANALYZER_VERSION):
                raise ValueError("Pixel observation configuration is not fail-closed")
            return PixelObservationTrack(
                schema_version=str(values["schema_version"].item()),
                analyzer_version=str(values["analyzer_version"].item()),
                source_sha256=str(values["source_sha256"].item()),
                source_pts=values["source_pts"],
                decoded_pixel_sha256=tuple(
                    str(item) for item in values["decoded_pixel_sha256"].tolist()
                ),
                region_names=tuple(str(item) for item in values["region_names"].tolist()),
                roi_boxes_xyxy=values["roi_boxes_xyxy"],
                roi_pixel_count=values["roi_pixel_count"],
                clipped_fraction=values["clipped_fraction"],
                focus_metric=values["focus_metric"],
                focus_reference=values["focus_reference"],
                focus_score=values["focus_score"],
                luma_mean=values["luma_mean"],
                shadow_fraction=values["shadow_fraction"],
                highlight_fraction=values["highlight_fraction"],
                dynamic_range=values["dynamic_range"],
                temporal_innovation=values["temporal_innovation"],
                flow_consistency=values["flow_consistency"],
                confidence=values["confidence"],
                reason_mask=values["reason_mask"],
                cut_histogram_distance=values["cut_histogram_distance"],
                cut_thumbnail_mad=values["cut_thumbnail_mad"],
                cut_thumbnail_zncc=values["cut_thumbnail_zncc"],
                photometric_discontinuity_candidate=values[
                    "photometric_discontinuity_candidate"
                ],
                cut_candidate=values["cut_candidate"],
                observation_epoch_start=values["observation_epoch_start"],
            )
    except (
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise AutoAnimError("MEDIA_INVALID", f"Invalid pixel observations: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(array: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(np.asarray(array).astype(dtype, copy=False))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _decoded_sequence_sha256(observations: PixelObservationTrack) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(observations.decoded_pixel_sha256):
        digest.update(index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _compact_summary(
    capture: CaptureTrack,
    observations: PixelObservationTrack,
) -> dict[str, Any]:
    observations.validate_capture(capture)
    regions: dict[str, Any] = {}
    for region_index, region_name in enumerate(observations.region_names):
        masks = observations.reason_mask[:, region_index]
        confidences = observations.confidence[:, region_index]
        finite = confidences[np.isfinite(confidences)]
        focus_scores = observations.focus_score[:, region_index]
        finite_focus = focus_scores[np.isfinite(focus_scores)]
        reason_counts = {
            reason: int(np.count_nonzero(masks & _REASON_BITS[reason]))
            for reason in REASON_CODES
        }
        regions[region_name] = {
            "observedFrames": int(
                capture.frame_count
                - np.count_nonzero(masks & _REASON_BITS["DETECTION_MISSING"])
            ),
            "missingFrames": int(
                np.count_nonzero(masks & _REASON_BITS["DETECTION_MISSING"])
            ),
            "pixelMetricFrames": int(len(finite)),
            "confidenceCoverageFraction": float(len(finite) / capture.frame_count),
            "confidenceMedian": float(np.median(finite)) if len(finite) else None,
            "confidenceP05": float(np.percentile(finite, 5)) if len(finite) else None,
            "confidenceAtCapFrames": int(
                np.count_nonzero(finite == PIXEL_DIAGNOSTIC_CONFIDENCE_CAP)
            ),
            "focusScoreAtCeilingFrames": int(
                np.count_nonzero(finite_focus == 1.0)
            ),
            "strongFrames": int(np.count_nonzero(finite >= 0.75)),
            "reasonCounts": {
                reason: count for reason, count in reason_counts.items() if count
            },
        }
    return {
        "observedFrames": int(np.count_nonzero(capture.detected)),
        "missingFrames": int(capture.frame_count - np.count_nonzero(capture.detected)),
        "photometricDiscontinuityCandidateFrames": int(
            np.count_nonzero(observations.photometric_discontinuity_candidate)
        ),
        "cutCandidateFrames": int(np.count_nonzero(observations.cut_candidate)),
        "observationEpochStarts": int(
            np.count_nonzero(observations.observation_epoch_start)
        ),
        "regions": regions,
    }


def build_observation_v3_summary(
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    *,
    capture_artifact_sha256: str,
    capture_artifact_bytes: int,
    pixel_observations_sha256: str,
    pixel_observations_bytes: int,
) -> dict[str, Any]:
    """Build the bounded v3 JSON envelope; dense frame values remain in NPZ."""

    observations.validate_capture(capture)
    for value in (capture_artifact_sha256, pixel_observations_sha256):
        if not _valid_sha256(value):
            raise ValueError("Observation-v3 artifact SHA-256 is invalid")
    if capture_artifact_bytes <= 0 or pixel_observations_bytes <= 0:
        raise ValueError("Observation-v3 artifact byte counts must be positive")
    cut_frames = np.flatnonzero(observations.cut_candidate).astype(int).tolist()
    photometric_frames = np.flatnonzero(
        observations.photometric_discontinuity_candidate
    ).astype(int).tolist()
    epoch_starts = np.flatnonzero(observations.observation_epoch_start).astype(int).tolist()
    return {
        "schemaVersion": OBSERVATION_V3_SCHEMA_VERSION,
        "kind": "video_performance_evidence",
        "policy": OBSERVATION_V3_POLICY,
        "sourceMode": "video_follow",
        "consumedByRetargeting": False,
        "source": {
            "name": capture.provenance.source_name,
            "sha256": capture.provenance.source_sha256,
            "bytes": capture.provenance.source_bytes,
            "captureSchemaVersion": capture.schema_version,
            "captureModelSha256": capture.provenance.model_sha256,
            "frameCount": capture.frame_count,
            "frameSize": [capture.width, capture.height],
            "sourceStartPTS": int(capture.source_pts[0]),
            "sourceTimeBase": [
                capture.provenance.time_base_numerator,
                capture.provenance.time_base_denominator,
            ],
            "sourcePTSSha256": _array_sha256(capture.source_pts, "<i8"),
            "captureArtifact": {
                "logicalName": "capture",
                "name": "capture.npz",
                "sha256": capture_artifact_sha256,
                "bytes": capture_artifact_bytes,
            },
        },
        "decodedPixels": {
            "format": "rgb24",
            "relationshipToDetectorInput": "redecoded_for_evidence",
            "detectorIngressPixelsRetained": False,
            "perFrameHashesStoredInArrays": True,
            "hashDomain": "rgb8_hwc_contiguous_after_declared_orientation",
            "sequenceSha256": _decoded_sequence_sha256(observations),
        },
        "algorithm": {
            "id": observations.analyzer_version,
            "arraysSchemaVersion": observations.schema_version,
            "regionOrder": list(observations.region_names),
            "patchSize": [PATCH_SIZE, PATCH_SIZE],
            "thumbnailSize": [THUMBNAIL_SIZE, THUMBNAIL_SIZE],
            "reasonCodes": list(REASON_CODES),
            "thresholdProfile": "provisional-diagnostic-v1",
            "confidenceCap": PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
            "confidenceCalibrated": False,
            "flowBackend": "farneback_forward_backward_fixed_patch_v1",
            "cutSignals": [
                "histogram_total_variation_32_bins",
                "thumbnail_mean_absolute_difference",
                "zero_mean_normalized_cross_correlation",
            ],
            "cutThresholds": {
                "histogramDistanceMinimum": 0.55,
                "thumbnailMADMinimum": 0.18,
                "thumbnailZNCCMaximum": 0.45,
            },
            "flatThumbnailCorrelationState": "unavailable_not_zero_or_one",
            "opencvVersion": cv2.__version__,
        },
        "arraysArtifact": {
            "logicalName": "pixel_observations",
            "name": "pixel-observations.npz",
            "sha256": pixel_observations_sha256,
            "bytes": pixel_observations_bytes,
        },
        "summary": _compact_summary(capture, observations),
        "events": {
            "photometricDiscontinuityCandidateFrames": photometric_frames,
            "cutCandidateFrames": cut_frames,
            "observationEpochStarts": epoch_starts,
            "identityContinuityState": "unknown",
            "identityOrTrackingJumpCandidateFrames": None,
        },
        "claims": {
            "sourcePixelsAnalyzed": True,
            "changesFinalGNMMotion": False,
            "confidenceCalibrated": False,
            "occlusionValidated": False,
            "identityContinuityValidated": False,
            "neutralityInferred": False,
            "productionValidated": False,
        },
        "caveats": [
            "The exact source was re-decoded for diagnostics; capture.v1 did not retain detector-ingress pixel hashes.",
            "Pixel scores are provisional diagnostics capped below the strong tier and are not consumed by retargeting.",
            "Occlusion, identity continuity, and neutrality remain unknown rather than being inferred from these proxies.",
        ],
    }


def write_observation_v3_summary(
    path: str | Path,
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    *,
    capture_artifact_sha256: str,
    capture_artifact_bytes: int,
    pixel_observations_sha256: str,
    pixel_observations_bytes: int,
) -> Path:
    return write_json(
        path,
        build_observation_v3_summary(
            capture,
            observations,
            capture_artifact_sha256=capture_artifact_sha256,
            capture_artifact_bytes=capture_artifact_bytes,
            pixel_observations_sha256=pixel_observations_sha256,
            pixel_observations_bytes=pixel_observations_bytes,
        ),
    )


def build_observation_v3_view(
    capture: CaptureTrack,
    observations: PixelObservationTrack,
    verified_summary: dict[str, Any],
    *,
    evidence_binding: dict[str, Any],
    display_binding: dict[str, Any],
) -> dict[str, Any]:
    """Build a path-free per-frame document for exact-time evidence review.

    The sealed capture, dense pixel arrays, and reconstructable v3 summary stay
    canonical. This derived document exposes only review diagnostics; it has no
    animation controls and cannot make provisional pixel evidence authoritative.
    """

    observations.validate_capture(capture)
    if capture.frame_count > MAX_OBSERVATION_V3_VIEW_FRAMES:
        raise ValueError(
            "Observation-v3 interactive review exceeds the frame limit"
        )
    source = verified_summary.get("source", {}) if isinstance(verified_summary, dict) else {}
    algorithm = (
        verified_summary.get("algorithm", {})
        if isinstance(verified_summary, dict)
        else {}
    )
    if (
        not isinstance(verified_summary, dict)
        or verified_summary.get("schemaVersion") != OBSERVATION_V3_SCHEMA_VERSION
        or verified_summary.get("policy") != OBSERVATION_V3_POLICY
        or verified_summary.get("consumedByRetargeting") is not False
        or source.get("sha256") != capture.provenance.source_sha256
        or source.get("frameCount") != capture.frame_count
        or algorithm.get("arraysSchemaVersion") != observations.schema_version
    ):
        raise ValueError("Observation-v3 view source is not a verified evidence contract")
    required_artifacts = {
        "capture",
        "capture_jsonl",
        "performance_evidence",
        "pixel_observations",
        "observation_v3",
        "capture_session",
    }
    bound_artifacts = (
        evidence_binding.get("artifacts", {})
        if isinstance(evidence_binding, dict)
        else {}
    )
    retained_source = (
        evidence_binding.get("retainedSource", {})
        if isinstance(evidence_binding, dict)
        else {}
    )
    if (
        not isinstance(evidence_binding, dict)
        or evidence_binding.get("chainVerified") is not True
        or not _valid_sha256(evidence_binding.get("manifestSha256"))
        or not isinstance(evidence_binding.get("sealSchema"), str)
        or not isinstance(evidence_binding.get("sealKeyId"), str)
        or not isinstance(bound_artifacts, dict)
        or set(bound_artifacts) != required_artifacts
        or any(
            not isinstance(record, dict)
            or not isinstance(record.get("name"), str)
            or Path(record["name"]).name != record["name"]
            or not _valid_sha256(record.get("sha256"))
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] <= 0
            for record in bound_artifacts.values()
        )
        or not isinstance(retained_source, dict)
        or retained_source.get("sha256") != capture.provenance.source_sha256
        or retained_source.get("bytes") != capture.provenance.source_bytes
    ):
        raise ValueError("Observation-v3 view requires a verified sealed evidence chain")
    display_artifact = (
        display_binding.get("artifact", {})
        if isinstance(display_binding, dict)
        else {}
    )
    expected_generation_contract = {
        "schema_version": "autoanim.viewer-display-binding/1.0",
        "artifact": "viewer_media",
        "source_frame_size": [capture.width, capture.height],
        "proxy_frame_size": [capture.width, capture.height],
        "display_rotation_degrees": 0,
        "sample_aspect_ratio": [1, 1],
        "clean_aperture_crop_ltrb": [0, 0, 0, 0],
        "source_to_display_pixel_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "transcode_policy": (
            "ffmpeg_h264_pts_passthrough_no_geometry_filters_v1"
        ),
    }
    if (
        not isinstance(display_binding, dict)
        or display_binding.get("clockVerified") is not True
        or display_binding.get("frameCount") != capture.frame_count
        or display_binding.get("frameSize") != [capture.width, capture.height]
        or display_binding.get("displayRotationDegrees") != 0
        or display_binding.get("sampleAspectRatio") != [1, 1]
        or display_binding.get("cleanApertureCropLTRB") != [0, 0, 0, 0]
        or not isinstance(display_artifact, dict)
        or display_artifact.get("logicalName") != "viewer_media"
        or not isinstance(display_artifact.get("name"), str)
        or Path(display_artifact["name"]).name != display_artifact["name"]
        or not _valid_sha256(display_artifact.get("sha256"))
        or not isinstance(display_artifact.get("bytes"), int)
        or isinstance(display_artifact.get("bytes"), bool)
        or display_artifact["bytes"] <= 0
        or display_binding.get("generationContract")
        != expected_generation_contract
        or display_binding.get("sourceToDisplayPixelTransform")
        != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ):
        raise ValueError("Observation-v3 display proxy is not bound to capture pixels")
    display_timestamps = display_binding.get("frameTimestampsSeconds")
    if (
        not isinstance(display_timestamps, list)
        or len(display_timestamps) != capture.frame_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in display_timestamps
        )
        or any(
            float(display_timestamps[index])
            <= float(display_timestamps[index - 1])
            for index in range(1, len(display_timestamps))
        )
        or max(
            (
                abs(float(display_timestamps[index]) - float(timestamp))
                for index, timestamp in enumerate(capture.timestamps_seconds)
            ),
            default=0.0,
        )
        > 0.002
        or not isinstance(display_binding.get("timestampMaxErrorSeconds"), (int, float))
        or isinstance(display_binding.get("timestampMaxErrorSeconds"), bool)
        or not math.isfinite(float(display_binding["timestampMaxErrorSeconds"]))
        or not 0.0 <= float(display_binding["timestampMaxErrorSeconds"]) <= 0.002
    ):
        raise ValueError("Observation-v3 display proxy frame clock is invalid")

    frames: list[dict[str, Any]] = []
    for frame_index in range(capture.frame_count):
        regions: dict[str, Any] = {}
        for region_name in observations.region_names:
            record = observations.region_record(frame_index, region_name)
            confidence = record["confidence"]
            regions[region_name] = {
                "qualityState": record["qualityState"],
                "reasonCodes": record["reasonCodes"],
                "confidence": (
                    None
                    if confidence is None
                    else min(float(confidence), PIXEL_DIAGNOSTIC_CONFIDENCE_CAP)
                ),
                "confidenceCalibrated": False,
                "roiBoxXYXY": record["roiBoxXYXY"],
                "roiPixelCount": record["roiPixelCount"],
                "clippedFraction": record["clippedFraction"],
                "focusMetric": record["focusMetric"],
                "focusScore": record["focusScore"],
                "lumaMean": record["lumaMean"],
                "shadowFraction": record["shadowFraction"],
                "highlightFraction": record["highlightFraction"],
                "dynamicRange": record["dynamicRange"],
                "temporalInnovation": record["temporalInnovation"],
                "flowConsistency": record["flowConsistency"],
                "occlusionState": "unknown",
                "identityContinuityState": "unknown",
            }
        frames.append(
            {
                "frameIndex": frame_index,
                "sourcePTS": int(capture.source_pts[frame_index]),
                "timestampSeconds": float(capture.timestamps_seconds[frame_index]),
                "detected": bool(capture.detected[frame_index]),
                "photometricDiscontinuityCandidate": bool(
                    observations.photometric_discontinuity_candidate[frame_index]
                ),
                "cutCandidate": bool(observations.cut_candidate[frame_index]),
                "observationEpochStart": bool(
                    observations.observation_epoch_start[frame_index]
                ),
                "regions": regions,
            }
        )

    return {
        "schemaVersion": OBSERVATION_V3_VIEW_SCHEMA_VERSION,
        "kind": "video_performance_evidence_view",
        "sourceMode": "video_follow",
        "consumedByRetargeting": False,
        "source": {
            "sha256": capture.provenance.source_sha256,
            "frameCount": capture.frame_count,
            "frameSize": [capture.width, capture.height],
            "sourceTimeBase": [
                capture.provenance.time_base_numerator,
                capture.provenance.time_base_denominator,
            ],
        },
        "evidenceBinding": evidence_binding,
        "display": display_binding,
        "observation": {
            "schemaVersion": OBSERVATION_V3_SCHEMA_VERSION,
            "arraysSchemaVersion": observations.schema_version,
            "policy": OBSERVATION_V3_POLICY,
            "analyzerVersion": observations.analyzer_version,
            "regionOrder": list(observations.region_names),
            "reasonCodes": list(REASON_CODES),
            "confidenceCap": PIXEL_DIAGNOSTIC_CONFIDENCE_CAP,
            "confidenceCalibrated": False,
        },
        "frames": frames,
        "claims": {
            "derivedFromVerifiedSealedEvidence": True,
            "changesFinalGNMMotion": False,
            "confidenceCalibrated": False,
            "occlusionValidated": False,
            "identityContinuityValidated": False,
            "productionValidated": False,
        },
    }


def load_verified_observation_v3_summary(
    path: str | Path,
    *,
    pixel_observations_path: str | Path,
    capture_artifact_path: str | Path,
    expected_capture: CaptureTrack,
    expected_observations: PixelObservationTrack | None = None,
) -> dict[str, Any]:
    """Verify compact JSON and reconstruct all summaries from the dense arrays."""

    source_path = Path(path)
    if source_path.stat().st_size <= 0 or source_path.stat().st_size > MAX_OBSERVATION_V3_BYTES:
        raise ValueError("Observation-v3 summary size is outside the accepted bounds")
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Observation-v3 summary must be canonical UTF-8 JSON") from exc
    pixel_path = Path(pixel_observations_path)
    capture_path = Path(capture_artifact_path)
    observations = (
        expected_observations
        if expected_observations is not None
        else load_pixel_observations(pixel_path)
    )
    observations.validate_capture(expected_capture)
    expected = build_observation_v3_summary(
        expected_capture,
        observations,
        capture_artifact_sha256=_sha256(capture_path),
        capture_artifact_bytes=capture_path.stat().st_size,
        pixel_observations_sha256=_sha256(pixel_path),
        pixel_observations_bytes=pixel_path.stat().st_size,
    )
    if payload != expected:
        raise ValueError("Observation-v3 summary does not reconstruct from sealed arrays")
    return payload


__all__ = [
    "OBSERVATION_V3_SCHEMA_VERSION",
    "OBSERVATION_V3_VIEW_SCHEMA_VERSION",
    "OBSERVATION_V3_POLICY",
    "MAX_OBSERVATION_V3_BYTES",
    "MAX_OBSERVATION_V3_VIEW_FRAMES",
    "PIXEL_ANALYZER_VERSION",
    "PIXEL_DIAGNOSTIC_CONFIDENCE_CAP",
    "PIXEL_OBSERVATION_SCHEMA_VERSION",
    "PixelObservationTrack",
    "REASON_CODES",
    "analyze_rgb_frames",
    "analyze_video_pixels",
    "build_observation_v3_view",
    "build_observation_v3_summary",
    "load_pixel_observations",
    "load_verified_observation_v3_summary",
    "write_observation_v3_summary",
    "write_pixel_observations",
]
