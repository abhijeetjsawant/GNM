"""Conservative, observation-only evidence emitted by the video lane.

This module intentionally does not retarget, filter, fill, or otherwise modify
animation controls.  It packages what the pinned tracker returned, where it
returned it, and how weakly that evidence can be trusted from the information
already present in :class:`CaptureTrack`.
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .serialization import write_json
from .video_capture import CaptureTrack


PERFORMANCE_EVIDENCE_SCHEMA_VERSION = "autoanim.performance-evidence.v2"
PROJECT_TICKS_PER_SECOND = 48_000
GEOMETRY_ONLY_CONFIDENCE_CAP = 0.50
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024

# MediaPipe Face Mesh landmark regions.  These are diagnostic supports, not a
# claim that every listed point is independently visible or anatomically exact.
MOUTH_LANDMARKS = (
    0,
    13,
    14,
    17,
    37,
    39,
    40,
    61,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    146,
    178,
    181,
    185,
    191,
    267,
    269,
    270,
    291,
    308,
    310,
    311,
    312,
    314,
    317,
    318,
    321,
    324,
    375,
    402,
    405,
    409,
    415,
)
EYE_LANDMARKS = (
    7,
    33,
    133,
    144,
    145,
    153,
    154,
    155,
    157,
    158,
    159,
    160,
    161,
    163,
    173,
    246,
    249,
    263,
    362,
    373,
    374,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    390,
    398,
    466,
    468,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
)
UPPER_FACE_LANDMARKS = (
    6,
    9,
    10,
    46,
    52,
    53,
    54,
    55,
    63,
    65,
    66,
    67,
    70,
    103,
    105,
    107,
    109,
    151,
    156,
    168,
    195,
    197,
    276,
    282,
    283,
    284,
    285,
    293,
    295,
    296,
    297,
    300,
    332,
    334,
    336,
    338,
)
HEAD_LANDMARKS = tuple(range(478))

REGION_LANDMARKS = {
    "mouth": MOUTH_LANDMARKS,
    "eyes": EYE_LANDMARKS,
    "upperFace": UPPER_FACE_LANDMARKS,
    "head": HEAD_LANDMARKS,
}


def _control_regions(names: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "mouth": tuple(
            name
            for name in names
            if name.startswith("mouth") or name.startswith("jaw") or name == "cheekPuff"
        ),
        "eyes": tuple(name for name in names if name.startswith("eye")),
        "upperFace": tuple(
            name
            for name in names
            if name.startswith("brow")
            or name.startswith("cheekSquint")
            or name.startswith("noseSneer")
        ),
        "head": (),
    }


def _optional_stat(values: np.ndarray) -> tuple[float | None, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    coverage = len(finite) / max(np.asarray(values).size, 1)
    return (float(np.median(finite)) if len(finite) else None, float(coverage))


def _confidence_tier(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.45:
        return "review"
    return "weak"


def _region_record(
    track: CaptureTrack,
    frame_index: int,
    region_name: str,
    landmark_indices: tuple[int, ...],
    control_names: tuple[str, ...],
    control_columns: dict[str, int],
) -> dict[str, Any]:
    if not bool(track.detected[frame_index]):
        return {
            "observationState": "missing",
            "semanticState": "unknown",
            "neutralityState": "unknown",
            "confidence": None,
            "confidenceTier": "unknown",
            "confidenceSource": "none_missing_detection",
            "landmarkEvidence": None,
            "trackerControls": None,
            "facialTransform": None,
        }

    indices = np.asarray(landmark_indices, dtype=np.int64)
    points = np.asarray(track.landmarks_xyz[frame_index, indices], dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    xy = points[:, :2]
    in_frame = (
        finite
        & (xy[:, 0] >= -0.05)
        & (xy[:, 0] <= 1.05)
        & (xy[:, 1] >= -0.05)
        & (xy[:, 1] <= 1.05)
    )
    finite_fraction = float(np.mean(finite))
    in_frame_fraction = float(np.mean(in_frame))
    visibility_median, visibility_coverage = _optional_stat(
        track.landmark_visibility[frame_index, indices]
    )
    presence_median, presence_coverage = _optional_stat(
        track.landmark_presence[frame_index, indices]
    )
    support = min(finite_fraction, in_frame_fraction)
    optional_terms: list[float] = []
    optional_sources: list[str] = []
    if visibility_median is not None:
        optional_terms.append(visibility_median * visibility_coverage)
        optional_sources.append("visibility")
    if presence_median is not None:
        optional_terms.append(presence_median * presence_coverage)
        optional_sources.append("presence")
    if optional_terms:
        confidence = float(np.clip(min(support, *optional_terms), 0.0, 1.0))
        confidence_source = "landmark_" + "_and_".join(optional_sources)
    else:
        # Being finite and inside the image is necessary but cannot establish
        # focus, occlusion, landmark accuracy, or semantic correctness.
        confidence = float(
            np.clip(GEOMETRY_ONLY_CONFIDENCE_CAP * support, 0.0, GEOMETRY_ONLY_CONFIDENCE_CAP)
        )
        confidence_source = "geometry_support_only_capped_at_0.5"

    tracker_controls = {
        name: float(track.blendshape_scores[frame_index, control_columns[name]])
        for name in control_names
    }
    transform = (
        track.facial_transforms[frame_index].tolist() if region_name == "head" else None
    )
    return {
        "observationState": "observed",
        # A returned tracker coefficient of zero is an observation, but is not
        # sufficient evidence that the performer was in a labeled neutral pose.
        "semanticState": "unknown",
        "neutralityState": "unknown",
        "confidence": confidence,
        "confidenceTier": _confidence_tier(confidence),
        "confidenceSource": confidence_source,
        "landmarkEvidence": {
            "finiteFraction": finite_fraction,
            "inFrameFraction": in_frame_fraction,
            "visibilityMedian": visibility_median,
            "visibilityCoverageFraction": visibility_coverage,
            "presenceMedian": presence_median,
            "presenceCoverageFraction": presence_coverage,
        },
        "trackerControls": tracker_controls,
        "facialTransform": transform,
    }


def _project_tick(delta_pts: int, time_base: Fraction) -> tuple[int, Fraction]:
    exact = Fraction(delta_pts) * time_base * PROJECT_TICKS_PER_SECOND
    return int(round(exact)), exact


def build_performance_evidence(track: CaptureTrack) -> dict[str, Any]:
    """Build an observation-only video evidence envelope.

    No missing frame is represented by zero-valued controls, and no tracker
    observation is classified as neutral.  Exact rational project time is kept
    alongside the nearest integer project tick for non-integral source clocks.
    """

    source_start_pts = int(track.source_pts[0])
    if track.provenance.source_start_pts != source_start_pts:
        raise ValueError("Capture provenance source_start_pts does not match the first frame")
    time_base = Fraction(
        track.provenance.time_base_numerator,
        track.provenance.time_base_denominator,
    )
    expected_seconds = np.asarray(
        [float(Fraction(int(value) - source_start_pts) * time_base) for value in track.source_pts],
        dtype=np.float64,
    )
    if not np.allclose(
        expected_seconds,
        track.timestamps_seconds,
        rtol=0.0,
        atol=np.finfo(np.float64).eps * 8,
    ):
        raise ValueError("Capture timestamps do not match source PTS and time base")

    control_regions = _control_regions(track.blendshape_names)
    control_columns = {name: index for index, name in enumerate(track.blendshape_names)}
    frames: list[dict[str, Any]] = []
    integer_ticks: list[int] = []
    rounded_tick_frames = 0
    for frame_index, source_pts in enumerate(track.source_pts):
        project_tick, exact_tick = _project_tick(
            int(source_pts) - source_start_pts,
            time_base,
        )
        integer_ticks.append(project_tick)
        rounded_tick_frames += int(exact_tick.denominator != 1)
        detected = bool(track.detected[frame_index])
        frames.append(
            {
                "frameIndex": frame_index,
                "sourcePTS": int(source_pts),
                "normalizedSourcePTS": int(source_pts) - source_start_pts,
                "timestampSeconds": float(track.timestamps_seconds[frame_index]),
                "projectTick": project_tick,
                "projectTickExactRational": [exact_tick.numerator, exact_tick.denominator],
                "projectTickWasRounded": exact_tick.denominator != 1,
                "observationState": "observed" if detected else "missing",
                "semanticState": "unknown",
                "neutralityState": "unknown",
                "trackerFaceConfidence": (
                    float(track.face_confidence[frame_index])
                    if detected and np.isfinite(track.face_confidence[frame_index])
                    else None
                ),
                "trackerInFrameFraction": (
                    float(track.tracking_quality[frame_index]) if detected else None
                ),
                "regions": {
                    region_name: _region_record(
                        track,
                        frame_index,
                        region_name,
                        landmark_indices,
                        control_regions[region_name],
                        control_columns,
                    )
                    for region_name, landmark_indices in REGION_LANDMARKS.items()
                },
            }
        )
    if len(integer_ticks) > 1 and np.any(np.diff(integer_ticks) <= 0):
        raise ValueError("Source PTS collapse or reverse on the 48 kHz project clock")

    region_summary: dict[str, Any] = {}
    for region_name in REGION_LANDMARKS:
        records = [frame["regions"][region_name] for frame in frames]
        confidences = np.asarray(
            [record["confidence"] for record in records if record["confidence"] is not None],
            dtype=np.float64,
        )
        region_summary[region_name] = {
            "observedFrames": sum(
                record["observationState"] == "observed" for record in records
            ),
            "missingFrames": sum(
                record["observationState"] == "missing" for record in records
            ),
            "confidenceMedian": (
                float(np.median(confidences)) if len(confidences) else None
            ),
            "confidenceP05": (
                float(np.percentile(confidences, 5)) if len(confidences) else None
            ),
            "geometryOnlyFrames": sum(
                record["confidenceSource"] == "geometry_support_only_capped_at_0.5"
                for record in records
            ),
        }

    return {
        "schemaVersion": PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
        "kind": "video_performance_evidence",
        "policy": "observation_only_no_motion_effect",
        "sourceMode": "video_follow",
        "consumedByRetargeting": False,
        "source": {
            "name": track.provenance.source_name,
            "sha256": track.provenance.source_sha256,
            "captureSchemaVersion": track.schema_version,
            "frameCount": track.frame_count,
            "frameSize": [track.width, track.height],
            "sourceStartPTS": source_start_pts,
            "sourceTimeBase": [time_base.numerator, time_base.denominator],
        },
        "projectClock": {
            "ticksPerSecond": PROJECT_TICKS_PER_SECOND,
            "integerRounding": "nearest_half_even",
            "exactRationalIncludedPerFrame": True,
            "roundedFrameCount": rounded_tick_frames,
        },
        "confidenceContract": {
            "geometryOnlyMaximum": GEOMETRY_ONLY_CONFIDENCE_CAP,
            "geometryOnlyDoesNotMeasure": [
                "focus",
                "motion_blur",
                "occlusion",
                "identity_continuity",
                "landmark_reprojection_error",
                "semantic_correctness",
            ],
            "strongThreshold": 0.75,
            "reviewThreshold": 0.45,
            "neutralityInferred": False,
            "unknownIsNotNeutral": True,
        },
        "regionDefinitions": {
            region_name: {
                "landmarkIndices": list(landmark_indices),
                "trackerControlNames": list(control_regions[region_name]),
            }
            for region_name, landmark_indices in REGION_LANDMARKS.items()
        },
        "summary": {
            "observedFrames": int(np.count_nonzero(track.detected)),
            "missingFrames": int(track.frame_count - np.count_nonzero(track.detected)),
            "firstProjectTick": integer_ticks[0],
            "lastProjectTick": integer_ticks[-1],
            "regions": region_summary,
        },
        "frames": frames,
        "claims": {
            "changesFinalGNMMotion": False,
            "neutralityInferred": False,
            "occlusionValidated": False,
            "gazeCalibrated": False,
            "tongueObserved": False,
            "productionValidated": False,
        },
        "caveats": [
            "Confidence uses only stored tracker observations; source pixels are not re-read.",
            "Geometry-only evidence is capped because in-frame landmarks do not prove accuracy.",
            "Observed zero-valued controls are not labeled neutral, and missing controls are null.",
            "This artifact is diagnostic and is not consumed by retargeting in video_follow mode.",
        ],
    }


def write_performance_evidence(path: str | Path, track: CaptureTrack) -> Path:
    """Atomically write a deterministic performance-evidence artifact."""

    return write_json(path, build_performance_evidence(track))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def load_verified_performance_evidence(
    path: str | Path,
    *,
    expected_source_sha256: str,
    expected_frame_count: int,
) -> dict[str, Any]:
    """Load and verify the timing/state contract of an Observation-v2 artifact."""

    source_path = Path(path)
    size = source_path.stat().st_size
    if size <= 0 or size > MAX_EVIDENCE_BYTES:
        raise ValueError("Performance evidence size is outside the accepted bounds")
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Performance evidence must be canonical UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Performance evidence root must be an object")
    if (
        payload.get("schemaVersion") != PERFORMANCE_EVIDENCE_SCHEMA_VERSION
        or payload.get("kind") != "video_performance_evidence"
        or payload.get("policy") != "observation_only_no_motion_effect"
        or payload.get("sourceMode") != "video_follow"
        or payload.get("consumedByRetargeting") is not False
    ):
        raise ValueError("Unsupported performance-evidence contract")

    source = payload.get("source")
    frames = payload.get("frames")
    project_clock = payload.get("projectClock")
    claims = payload.get("claims")
    if (
        not isinstance(source, dict)
        or source.get("sha256") != expected_source_sha256
        or source.get("frameCount") != expected_frame_count
        or not isinstance(frames, list)
        or len(frames) != expected_frame_count
        or expected_frame_count <= 0
        or not isinstance(project_clock, dict)
        or project_clock.get("ticksPerSecond") != PROJECT_TICKS_PER_SECOND
        or project_clock.get("exactRationalIncludedPerFrame") is not True
        or not isinstance(claims, dict)
        or claims.get("changesFinalGNMMotion") is not False
        or claims.get("neutralityInferred") is not False
        or claims.get("productionValidated") is not False
    ):
        raise ValueError("Performance evidence is not bound to the source take")

    time_base_value = source.get("sourceTimeBase")
    start_pts = source.get("sourceStartPTS")
    if (
        not isinstance(time_base_value, list)
        or len(time_base_value) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in time_base_value)
        or time_base_value[1] <= 0
        or not isinstance(start_pts, int)
        or isinstance(start_pts, bool)
    ):
        raise ValueError("Performance evidence has an invalid source time base")
    time_base = Fraction(time_base_value[0], time_base_value[1])
    previous_pts: int | None = None
    previous_tick: int | None = None
    observed_count = 0
    missing_count = 0
    region_names = ("mouth", "eyes", "upperFace", "head")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Performance evidence frame {index} is not an object")
        source_pts = frame.get("sourcePTS")
        project_tick = frame.get("projectTick")
        timestamp = frame.get("timestampSeconds")
        exact_value = frame.get("projectTickExactRational")
        delta_pts = source_pts - start_pts if isinstance(source_pts, int) else None
        exact_tick = (
            Fraction(delta_pts) * time_base * PROJECT_TICKS_PER_SECOND
            if delta_pts is not None
            else None
        )
        if (
            frame.get("frameIndex") != index
            or not isinstance(source_pts, int)
            or isinstance(source_pts, bool)
            or frame.get("normalizedSourcePTS") != delta_pts
            or not isinstance(project_tick, int)
            or isinstance(project_tick, bool)
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
            or not isinstance(exact_value, list)
            or exact_tick is None
            or exact_value != [exact_tick.numerator, exact_tick.denominator]
            or project_tick != round(exact_tick)
            or abs(float(timestamp) - float(Fraction(delta_pts) * time_base)) > 1e-9
            or (previous_pts is not None and source_pts <= previous_pts)
            or (previous_tick is not None and project_tick <= previous_tick)
        ):
            raise ValueError(f"Performance evidence frame {index} has invalid timing")
        observation_state = frame.get("observationState")
        if (
            observation_state not in {"observed", "missing"}
            or frame.get("semanticState") != "unknown"
            or frame.get("neutralityState") != "unknown"
        ):
            raise ValueError(f"Performance evidence frame {index} has invalid state")
        observed_count += observation_state == "observed"
        missing_count += observation_state == "missing"
        regions = frame.get("regions")
        if not isinstance(regions, dict):
            raise ValueError(f"Performance evidence frame {index} has no regions")
        for region_name in region_names:
            region = regions.get(region_name)
            if not isinstance(region, dict):
                raise ValueError(
                    f"Performance evidence frame {index} has no {region_name} region"
                )
            confidence = region.get("confidence")
            if (
                region.get("observationState") not in {"observed", "missing"}
                or region.get("semanticState") != "unknown"
                or region.get("neutralityState") != "unknown"
                or (
                    confidence is not None
                    and (
                        not isinstance(confidence, (int, float))
                        or isinstance(confidence, bool)
                        or not math.isfinite(float(confidence))
                        or not 0.0 <= float(confidence) <= 1.0
                    )
                )
                or (
                    region.get("observationState") == "missing"
                    and (
                        confidence is not None
                        or region.get("trackerControls") is not None
                    )
                )
            ):
                raise ValueError(
                    f"Performance evidence frame {index} has invalid {region_name} state"
                )
        previous_pts = source_pts
        previous_tick = project_tick

    summary = payload.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("observedFrames") != observed_count
        or summary.get("missingFrames") != missing_count
        or summary.get("firstProjectTick") != frames[0].get("projectTick")
        or summary.get("lastProjectTick") != frames[-1].get("projectTick")
    ):
        raise ValueError("Performance evidence summary does not match its frames")
    return payload


__all__ = [
    "GEOMETRY_ONLY_CONFIDENCE_CAP",
    "MAX_EVIDENCE_BYTES",
    "PERFORMANCE_EVIDENCE_SCHEMA_VERSION",
    "PROJECT_TICKS_PER_SECOND",
    "build_performance_evidence",
    "load_verified_performance_evidence",
    "write_performance_evidence",
]
