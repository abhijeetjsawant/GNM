"""Conservative, evidence-bound audio repair for a captured GNM performance.

The visual performance is the acting source.  Learned audio controls are only
allowed to replace lower-face controls where the retained video has no trusted
mouth observation, plus dedicated GNM tongue controls which monocular
MediaPipe RGB capture does not expose.  The exact native-sample/video-PTS join
is consumed directly; this module never estimates or silently warps sync.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .animation import _face_local_mouth
from .audio_video_timing import (
    AUDIO_VIDEO_TIMING_POLICY,
    AUDIO_VIDEO_TIMING_SCHEMA_VERSION,
)
from .errors import AutoAnimError
from .rig import ControlRig
from .serialization import write_json, write_npz
from .video_retarget import GNMPerformanceTrack
from .video_retarget import _rig_lip_gap_interocular


AUDIO_VISUAL_REPAIR_SCHEMA_VERSION = "autoanim.audio-visual-repair.v1"
AUDIO_VISUAL_REPAIR_POLICY = "video_authoritative_conservative_audio_repair_v1"
LOWER_FACE_SLICE = slice(200, 350)
TONGUE_SLICE = slice(350, 382)
PUPIL_SLICE = slice(382, 383)


@dataclass(frozen=True, slots=True)
class AudioVisualRepairConfig:
    minimum_trusted_visual_quality: float = 0.65
    minimum_speech_activity: float = 0.05
    visual_contact_protection_confidence: float = 0.50
    audio_contact_conflict_confidence: float = 0.35
    visually_open_gap_interocular: float = 0.055
    maximum_introduced_mouth_step_interocular: float = 0.060
    maximum_introduced_tongue_coefficient_step: float = 0.80
    taper_frames: int = 2

    def __post_init__(self) -> None:
        bounded = (
            self.minimum_trusted_visual_quality,
            self.minimum_speech_activity,
            self.visual_contact_protection_confidence,
            self.audio_contact_conflict_confidence,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("Audio-visual repair confidence values must lie in [0,1]")
        if (
            not np.isfinite(self.visually_open_gap_interocular)
            or self.visually_open_gap_interocular <= 0.0
        ):
            raise ValueError("Audio-visual repair open-mouth threshold must be positive")
        if (
            not np.isfinite(self.maximum_introduced_mouth_step_interocular)
            or self.maximum_introduced_mouth_step_interocular <= 0.0
        ):
            raise ValueError("Audio-visual repair mouth-step threshold must be positive")
        if (
            not np.isfinite(self.maximum_introduced_tongue_coefficient_step)
            or self.maximum_introduced_tongue_coefficient_step <= 0.0
        ):
            raise ValueError("Audio-visual repair tongue-step threshold must be positive")
        if self.taper_frames < 0 or self.taper_frames > 12:
            raise ValueError("Audio-visual repair taper must lie in [0,12] frames")


@dataclass(frozen=True, slots=True)
class AudioVisualRepairResult:
    performance: GNMPerformanceTrack
    report: dict[str, Any]
    frame_audio_times_seconds: np.ndarray
    lower_face_audio_weight: np.ndarray
    tongue_audio_weight: np.ndarray
    speech_activity: np.ndarray
    audio_lip_contact_confidence: np.ndarray
    audio_visual_contact_conflict: np.ndarray
    resampled_audio_expression: np.ndarray
    final_lip_gap_interocular: np.ndarray
    final_audio_contact_attained: np.ndarray


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _required_array(
    values: Mapping[str, np.ndarray],
    name: str,
    *,
    first_dimension: int,
    boolean: bool = False,
) -> np.ndarray:
    if name not in values:
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            f"Learned audio controls are missing required array {name}",
        )
    value = np.asarray(values[name])
    expected_kind = "b" if boolean else "f"
    if (
        value.shape != (first_dimension,)
        or value.dtype.kind != expected_kind
        or not np.isfinite(value).all()
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            f"Learned audio control array {name} must be finite "
            f"[{first_dimension}] {'boolean' if boolean else 'floating-point'} data",
        )
    return value


def _load_audio_controls(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as source:
            values = {name: np.asarray(source[name]).copy() for name in source.files}
    except (OSError, ValueError) as exc:
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            f"Learned audio controls cannot be read: {exc}",
        ) from exc
    expression = np.asarray(values.get("expression"), dtype=np.float32)
    timestamps = np.asarray(values.get("timestamps"), dtype=np.float64)
    if (
        expression.ndim != 2
        or expression.shape[1:] != (383,)
        or len(expression) == 0
        or timestamps.shape != (len(expression),)
        or not np.isfinite(expression).all()
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0.0)
        or float(timestamps[0]) < -1.0e-9
        or np.max(np.abs(expression), initial=0.0) > 3.0 + 1.0e-5
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Learned audio controls do not satisfy the finite GNM timeline contract",
        )
    count = len(expression)
    speech = _required_array(values, "speech_activity", first_dimension=count)
    contact = _required_array(
        values, "lip_contact_confidence", first_dimension=count
    )
    contact_target = _required_array(
        values, "lip_contact_target_gap", first_dimension=count
    )
    _required_array(
        values, "contact_correction_applied", first_dimension=count, boolean=True
    )
    _required_array(values, "lip_contact_attained", first_dimension=count, boolean=True)
    fps_value = np.asarray(values.get("fps"))
    if (
        fps_value.shape != ()
        or fps_value.dtype.kind not in "iu"
        or not 1 <= int(fps_value) <= 240
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Learned audio controls require one integer FPS in [1,240]",
        )
    if (
        np.min(speech, initial=0.0) < 0.0
        or np.max(speech, initial=0.0) > 1.0
        or np.min(contact, initial=0.0) < 0.0
        or np.max(contact, initial=0.0) > 1.0
        or np.min(contact_target, initial=0.0) < 0.0
        or np.max(contact_target, initial=0.0) > 0.5
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Learned speech/contact evidence lies outside its frozen range contract",
        )
    if count > 1 and not np.allclose(
        np.diff(timestamps),
        1.0 / float(int(fps_value)),
        rtol=0.0,
        atol=2.0e-5,
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Learned audio timestamps do not match their declared fixed FPS",
        )
    values["expression"] = expression
    values["timestamps"] = timestamps
    values["fps"] = fps_value
    return values


def _linear_resample(
    timestamps: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values)
    if source.ndim == 1:
        return np.interp(query, timestamps, source.astype(np.float64)).astype(np.float32)
    flattened = source.reshape(len(source), -1)
    output = np.stack(
        [
            np.interp(query, timestamps, flattened[:, column].astype(np.float64))
            for column in range(flattened.shape[1])
        ],
        axis=1,
    )
    return output.reshape((len(query),) + source.shape[1:]).astype(np.float32)


def _nearest_resample(
    timestamps: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    right = np.searchsorted(timestamps, query, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    use_left = np.abs(query - timestamps[left]) <= np.abs(timestamps[right] - query)
    indices = np.where(use_left, left, right)
    return np.asarray(values)[indices].copy()


def _taper_runs(weight: np.ndarray, mask: np.ndarray, taper_frames: int) -> np.ndarray:
    output = np.asarray(weight, dtype=np.float32).copy()
    selected = np.asarray(mask, dtype=bool)
    if taper_frames == 0 or not np.any(selected):
        return output
    padded = np.pad(selected.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    for start, stop in zip(starts, stops, strict=True):
        length = stop - start
        positions = np.arange(length)
        edge_distance = np.minimum(positions + 1, length - positions)
        taper = np.minimum(1.0, edge_distance / float(taper_frames + 1))
        output[start:stop] *= taper.astype(np.float32)
    return output


def _mouth_edge_steps(rig: ControlRig, expression: np.ndarray) -> np.ndarray:
    if len(expression) <= 1:
        return np.zeros(0, dtype=np.float32)
    mouth = np.stack([_face_local_mouth(rig, frame) for frame in expression])
    return np.max(np.linalg.norm(np.diff(mouth, axis=0), axis=2), axis=1).astype(
        np.float32
    )


def _limit_tongue_coefficient_steps(
    *,
    base_expression: np.ndarray,
    audio_expression: np.ndarray,
    output_expression: np.ndarray,
    tongue_weight: np.ndarray,
    tongue_eligible: np.ndarray,
    maximum_step: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    def steps(value: np.ndarray) -> np.ndarray:
        if len(value) <= 1:
            return np.zeros(0, dtype=np.float32)
        return np.linalg.norm(
            np.diff(value[:, TONGUE_SLICE], axis=0), axis=1
        ).astype(np.float32)

    base_steps = steps(base_expression)
    allowed = np.maximum(base_steps, np.float32(maximum_step)) + np.float32(1.0e-5)
    output = np.asarray(output_expression, dtype=np.float32).copy()
    weights = np.asarray(tongue_weight, dtype=np.float32).copy()
    selected = np.asarray(tongue_eligible, dtype=bool)
    padded = np.pad(selected.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    limited_runs = 0
    for start, stop in zip(starts, stops, strict=True):
        if not np.any(weights[start:stop] > 0.0):
            continue
        edge_start = max(0, start - 1)
        edge_stop = min(len(allowed), stop)

        def candidate(scale: float) -> np.ndarray:
            value = output.copy()
            scaled = weights[start:stop, None] * np.float32(scale)
            value[start:stop, TONGUE_SLICE] = (
                (1.0 - scaled) * base_expression[start:stop, TONGUE_SLICE]
                + scaled * audio_expression[start:stop, TONGUE_SLICE]
            )
            return value

        if np.all(steps(output)[edge_start:edge_stop] <= allowed[edge_start:edge_stop]):
            continue
        limited_runs += 1
        lower = 0.0
        upper = 1.0
        for _ in range(18):
            middle = 0.5 * (lower + upper)
            if np.all(
                steps(candidate(middle))[edge_start:edge_stop]
                <= allowed[edge_start:edge_stop]
            ):
                lower = middle
            else:
                upper = middle
        output = candidate(lower)
        weights[start:stop] *= np.float32(lower)
    final_steps = steps(output)
    if np.any(final_steps > allowed):
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Audio-visual repair introduced a tongue step beyond its coefficient bound",
        )
    return output, weights, limited_runs, base_steps, final_steps


def _limit_introduced_mouth_steps(
    *,
    rig: ControlRig,
    base_expression: np.ndarray,
    audio_expression: np.ndarray,
    output_expression: np.ndarray,
    lower_weight: np.ndarray,
    lower_eligible: np.ndarray,
    maximum_step: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Reduce repair weights until no new boundary jump is introduced.

    Existing fast video motion remains allowed. Trusted video frames are never
    modified; if a repair cannot meet the edge contract its run converges to a
    no-op instead of smoothing the source performance.
    """

    base_steps = _mouth_edge_steps(rig, base_expression)
    allowed = np.maximum(base_steps, np.float32(maximum_step)) + np.float32(1.0e-5)
    output = np.asarray(output_expression, dtype=np.float32).copy()
    weights = np.asarray(lower_weight, dtype=np.float32).copy()
    selected = np.asarray(lower_eligible, dtype=bool)
    padded = np.pad(selected.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    limited_runs = 0
    for start, stop in zip(starts, stops, strict=True):
        if not np.any(weights[start:stop] > 0.0):
            continue
        edge_start = max(0, start - 1)
        edge_stop = min(len(allowed), stop)

        def candidate(scale: float) -> np.ndarray:
            value = output.copy()
            scaled = weights[start:stop, None] * np.float32(scale)
            value[start:stop, LOWER_FACE_SLICE] = (
                (1.0 - scaled) * base_expression[start:stop, LOWER_FACE_SLICE]
                + scaled * audio_expression[start:stop, LOWER_FACE_SLICE]
            )
            return value

        current = _mouth_edge_steps(rig, output)
        if np.all(current[edge_start:edge_stop] <= allowed[edge_start:edge_stop]):
            continue
        limited_runs += 1
        lower = 0.0
        upper = 1.0
        for _ in range(18):
            middle = 0.5 * (lower + upper)
            trial = candidate(middle)
            steps = _mouth_edge_steps(rig, trial)
            if np.all(steps[edge_start:edge_stop] <= allowed[edge_start:edge_stop]):
                lower = middle
            else:
                upper = middle
        output = candidate(lower)
        weights[start:stop] *= np.float32(lower)
    final_steps = _mouth_edge_steps(rig, output)
    if np.any(final_steps > allowed):
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Audio-visual repair introduced a mouth step beyond the video/quality bound",
        )
    return output, weights, limited_runs, base_steps, final_steps


def _verified_join(
    timing: Mapping[str, Any],
    performance: GNMPerformanceTrack,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    join = timing.get("audioVideoJoin")
    gate = timing.get("fusionGate")
    source = timing.get("source")
    if (
        timing.get("schemaVersion") != AUDIO_VIDEO_TIMING_SCHEMA_VERSION
        or timing.get("policy") != AUDIO_VIDEO_TIMING_POLICY
        or timing.get("status") != "available_observation"
        or not isinstance(join, Mapping)
        or not isinstance(gate, Mapping)
        or not isinstance(source, Mapping)
        or source.get("sha256") != performance.provenance.capture_source_sha256
        or gate.get("exactClockJoinAvailable") is not True
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Audio-visual repair requires an exact retained-source clock join",
            {"timing_status": timing.get("status")},
        )
    video = join.get("video")
    audio = join.get("audio")
    if not isinstance(video, Mapping) or not isinstance(audio, Mapping):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED", "Audio/video clock join is incomplete"
        )
    frames = video.get("frames")
    mono_pcm = audio.get("monoPcm")
    if not isinstance(frames, list) or not isinstance(mono_pcm, Mapping):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED", "Audio/video sample coverage is missing"
        )
    def rational(value: Any, *, field: str) -> Fraction:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
        ):
            raise ValueError(f"{field} is not an exact rational")
        numerator = int(value[0])
        denominator = int(value[1])
        if denominator <= 0:
            raise ValueError(f"{field} denominator must be positive")
        return Fraction(numerator, denominator)

    try:
        sample_rate = int(audio["sampleRate"])
        pcm_sha256 = str(mono_pcm["sha256"])
        source_pts = np.asarray([int(frame["sourcePTS"]) for frame in frames], dtype=np.int64)
        spans = np.asarray(
            [frame["coveringAudioSampleSpan"] for frame in frames], dtype=np.int64
        )
        coverage = np.asarray(
            [bool(frame["hasAudioSampleCoverage"]) for frame in frames], dtype=bool
        )
        audio_start = rational(
            audio["decodedStartExactRational"], field="decoded audio start"
        )
        audio_end = rational(
            audio["decodedEndExactRational"], field="decoded audio end"
        )
        display_starts = [
            rational(frame["displayStartExactRational"], field=f"video frame {index} start")
            for index, frame in enumerate(frames)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED", "Audio/video sample coverage is malformed"
        ) from exc
    if (
        sample_rate <= 0
        or len(pcm_sha256) != 64
        or spans.shape != (performance.frame_count, 2)
        or coverage.shape != (performance.frame_count,)
        or not np.array_equal(source_pts, performance.source_pts)
        or np.any(spans < 0)
        or np.any(spans[:, 1] < spans[:, 0])
        or audio_end <= audio_start
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Audio/video sample coverage does not bind to the captured source PTS",
        )
    # Controls are normalized to decoded-audio time zero.  A frame is sampled
    # at its exact display start relative to the decoded audio start.  The
    # covering sample span remains independent evidence that the query is
    # backed by retained PCM; its midpoint is deliberately not used because
    # doing so advances audio by roughly half a video frame.
    query_fraction = [value - audio_start for value in display_starts]
    audio_duration = audio_end - audio_start
    if any(
        covered and (value < 0 or value >= audio_duration)
        for covered, value in zip(coverage, query_fraction, strict=True)
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "A covered video frame maps outside the retained decoded-audio interval",
        )
    query = np.asarray([float(value) for value in query_fraction], dtype=np.float64)
    query[~coverage] = 0.0
    return query, coverage, sample_rate, pcm_sha256


def apply_audio_visual_repair(
    performance: GNMPerformanceTrack,
    *,
    audio_controls_path: str | Path,
    audio_result: Mapping[str, Any],
    timing_evidence: Mapping[str, Any],
    output_dir: str | Path | None = None,
    rig: ControlRig | None = None,
    config: AudioVisualRepairConfig = AudioVisualRepairConfig(),
) -> AudioVisualRepairResult:
    """Return a PTS-preserving performance revision and its complete evidence."""

    analysis = audio_result.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("motion_backend") != "learned_a2f":
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Conservative repair requires the learned Audio2Face backend; fallback cues are not accepted",
        )
    controls = _load_audio_controls(audio_controls_path)
    audio_timestamps = np.asarray(controls["timestamps"], dtype=np.float64)
    query, coverage, sample_rate, pcm_sha256 = _verified_join(
        timing_evidence, performance
    )
    control_cadence = float(np.median(np.diff(audio_timestamps)))
    covered_query = query[coverage]
    if (
        len(covered_query)
        and (
            float(np.min(covered_query)) < float(audio_timestamps[0]) - 1.0e-9
            or float(np.max(covered_query))
            >= float(audio_timestamps[-1]) + control_cadence + 1.0e-9
        )
    ):
        raise AutoAnimError(
            "AUDIO_VISUAL_REPAIR_BLOCKED",
            "Exact video-frame audio queries fall outside the learned control-track support",
        )
    expression = _linear_resample(
        audio_timestamps,
        np.asarray(controls["expression"], dtype=np.float32),
        query,
    )
    speech = np.clip(
        _linear_resample(audio_timestamps, controls["speech_activity"], query),
        0.0,
        1.0,
    )
    audio_contact = np.clip(
        _linear_resample(audio_timestamps, controls["lip_contact_confidence"], query),
        0.0,
        1.0,
    )
    audio_contact_target = np.maximum(
        _linear_resample(audio_timestamps, controls["lip_contact_target_gap"], query),
        0.0,
    )
    audio_contact_applied = _nearest_resample(
        audio_timestamps, controls["contact_correction_applied"], query
    ).astype(bool)
    audio_contact_attained = _nearest_resample(
        audio_timestamps, controls["lip_contact_attained"], query
    ).astype(bool)

    trusted_visual = np.asarray(
        performance.detected
        & performance.source_lip_geometry_valid
        & (performance.effective_quality >= config.minimum_trusted_visual_quality),
        dtype=bool,
    )
    protected_visual_contact = np.asarray(
        performance.source_lip_geometry_valid
        & (
            performance.source_lip_contact_confidence
            >= config.visual_contact_protection_confidence
        ),
        dtype=bool,
    )
    speech_available = coverage & (speech >= config.minimum_speech_activity)
    weak_visual = ~trusted_visual
    lower_eligible = speech_available & weak_visual & ~protected_visual_contact
    lower_weight = np.where(
        lower_eligible,
        np.clip(
            (speech - config.minimum_speech_activity)
            / max(1.0 - config.minimum_speech_activity, 1.0e-6),
            0.0,
            1.0,
        ),
        0.0,
    ).astype(np.float32)
    lower_weight = _taper_runs(lower_weight, lower_eligible, config.taper_frames)

    tongue_active = np.max(np.abs(expression[:, TONGUE_SLICE]), axis=1) > 1.0e-6
    tongue_eligible = speech_available & tongue_active
    tongue_weight = np.where(tongue_eligible, speech, 0.0).astype(np.float32)
    tongue_weight = _taper_runs(tongue_weight, tongue_eligible, config.taper_frames)

    visual_closed = (
        trusted_visual
        & (
            performance.source_lip_contact_confidence
            >= config.visual_contact_protection_confidence
        )
    )
    visual_open = (
        trusted_visual
        & (
            performance.source_lip_gap_interocular
            >= config.visually_open_gap_interocular
        )
    )
    audio_closed = audio_contact >= config.audio_contact_conflict_confidence
    conflict = np.asarray(
        speech_available & ((visual_closed & ~audio_closed) | (visual_open & audio_closed)),
        dtype=bool,
    )

    base_expression = np.asarray(performance.expression, dtype=np.float32)
    output_expression = base_expression.copy()
    output_expression[:, LOWER_FACE_SLICE] = (
        (1.0 - lower_weight[:, None]) * base_expression[:, LOWER_FACE_SLICE]
        + lower_weight[:, None] * expression[:, LOWER_FACE_SLICE]
    )
    output_expression[:, TONGUE_SLICE] = (
        (1.0 - tongue_weight[:, None]) * base_expression[:, TONGUE_SLICE]
        + tongue_weight[:, None] * expression[:, TONGUE_SLICE]
    )
    output_expression = np.clip(output_expression, -3.0, 3.0).astype(np.float32)
    # These exact locks are contractual, not merely report fields.
    output_expression[:, : LOWER_FACE_SLICE.start] = base_expression[
        :, : LOWER_FACE_SLICE.start
    ]
    output_expression[:, PUPIL_SLICE] = base_expression[:, PUPIL_SLICE]
    (
        output_expression,
        tongue_weight,
        tongue_continuity_limited_runs,
        baseline_tongue_step,
        final_tongue_step,
    ) = _limit_tongue_coefficient_steps(
        base_expression=base_expression,
        audio_expression=expression,
        output_expression=output_expression,
        tongue_weight=tongue_weight,
        tongue_eligible=tongue_eligible,
        maximum_step=config.maximum_introduced_tongue_coefficient_step,
    )
    continuity_limited_runs = 0
    base_mouth_steps = np.zeros(max(performance.frame_count - 1, 0), dtype=np.float32)
    final_mouth_steps = base_mouth_steps.copy()
    if rig is not None:
        (
            output_expression,
            lower_weight,
            continuity_limited_runs,
            base_mouth_steps,
            final_mouth_steps,
        ) = _limit_introduced_mouth_steps(
            rig=rig,
            base_expression=base_expression,
            audio_expression=expression,
            output_expression=output_expression,
            lower_weight=lower_weight,
            lower_eligible=lower_eligible,
            maximum_step=config.maximum_introduced_mouth_step_interocular,
        )

    lower_changed = np.any(
        np.abs(
            output_expression[:, LOWER_FACE_SLICE]
            - base_expression[:, LOWER_FACE_SLICE]
        )
        > 1.0e-7,
        axis=1,
    )
    tongue_changed = np.any(
        np.abs(output_expression[:, TONGUE_SLICE] - base_expression[:, TONGUE_SLICE])
        > 1.0e-7,
        axis=1,
    )
    contact_target = np.asarray(
        performance.lip_contact_target_gap_interocular, dtype=np.float32
    ).copy()
    contact_applied = np.asarray(performance.contact_correction_applied, dtype=bool).copy()
    contact_attained = np.asarray(performance.lip_contact_attained, dtype=bool).copy()
    aperture_target = np.asarray(
        performance.lip_aperture_target_gap_interocular, dtype=np.float32
    ).copy()
    aperture_applied = np.asarray(
        performance.lip_aperture_correction_applied, dtype=bool
    ).copy()
    aperture_attained = np.asarray(
        performance.lip_aperture_target_attained, dtype=bool
    ).copy()
    contact_target[lower_changed] = 0.0
    contact_applied[lower_changed] = False
    contact_attained[lower_changed] = False
    aperture_target[lower_changed] = 0.0
    aperture_applied[lower_changed] = False
    aperture_attained[lower_changed] = False

    final_lip_gap = np.zeros(performance.frame_count, dtype=np.float32)
    final_audio_contact_attained = np.zeros(performance.frame_count, dtype=bool)
    audio_contact_candidate = np.asarray(
        lower_changed
        & (audio_contact >= config.audio_contact_conflict_confidence)
        & (audio_contact_target > 0.0),
        dtype=bool,
    )
    if rig is not None:
        final_lip_gap = np.asarray(
            [_rig_lip_gap_interocular(rig, frame) for frame in output_expression],
            dtype=np.float32,
        )
        final_audio_contact_attained[audio_contact_candidate] = (
            final_lip_gap[audio_contact_candidate]
            <= audio_contact_target[audio_contact_candidate] + np.float32(0.003)
        )
    # These fields describe the final fused track, not the upstream audio
    # solver.  Retaining the fused target makes the later authored aperture
    # revision treat an audio-derived bilabial as a hard contact anchor, while
    # oral validation independently measures whether final geometry attained it.
    contact_target[audio_contact_candidate] = audio_contact_target[
        audio_contact_candidate
    ]
    contact_applied[audio_contact_candidate] = audio_contact_applied[
        audio_contact_candidate
    ]
    contact_attained[audio_contact_candidate] = final_audio_contact_attained[
        audio_contact_candidate
    ]

    revised = replace(
        performance,
        expression=output_expression,
        lip_contact_target_gap_interocular=contact_target,
        contact_correction_applied=contact_applied,
        lip_contact_attained=contact_attained,
        lip_aperture_target_gap_interocular=aperture_target,
        lip_aperture_correction_applied=aperture_applied,
        lip_aperture_target_attained=aperture_attained,
    )
    upper_locked = np.array_equal(
        revised.expression[:, :200], performance.expression[:, :200]
    )
    pupil_locked = np.array_equal(
        revised.expression[:, PUPIL_SLICE], performance.expression[:, PUPIL_SLICE]
    )
    pose_locked = np.array_equal(revised.rotations, performance.rotations) and np.array_equal(
        revised.translation, performance.translation
    )
    timing_locked = np.array_equal(revised.source_pts, performance.source_pts) and np.array_equal(
        revised.timestamps_seconds, performance.timestamps_seconds
    )
    if not all((upper_locked, pupil_locked, pose_locked, timing_locked)):
        raise AutoAnimError(
            "INTERNAL_ERROR", "Audio-visual repair violated a video-authoritative lock"
        )

    lower_weighted_frames = int(np.count_nonzero(lower_weight > 0.0))
    tongue_weighted_frames = int(np.count_nonzero(tongue_weight > 0.0))
    lower_frames = int(np.count_nonzero(lower_changed))
    tongue_frames = int(np.count_nonzero(tongue_changed))
    conflict_frames = int(np.count_nonzero(conflict))
    contact_candidate_frames = int(np.count_nonzero(audio_contact_candidate))
    contact_attainment = (
        float(np.mean(final_audio_contact_attained[audio_contact_candidate]))
        if contact_candidate_frames and rig is not None
        else None
    )
    report: dict[str, Any] = {
        "schemaVersion": AUDIO_VISUAL_REPAIR_SCHEMA_VERSION,
        "policy": AUDIO_VISUAL_REPAIR_POLICY,
        "status": "repaired" if lower_frames or tongue_frames else "exact_noop",
        "config": {
            "minimumTrustedVisualQuality": config.minimum_trusted_visual_quality,
            "minimumSpeechActivity": config.minimum_speech_activity,
            "visualContactProtectionConfidence": (
                config.visual_contact_protection_confidence
            ),
            "audioContactConflictConfidence": config.audio_contact_conflict_confidence,
            "visuallyOpenGapInterocular": config.visually_open_gap_interocular,
            "maximumIntroducedMouthStepInterocular": (
                config.maximum_introduced_mouth_step_interocular
            ),
            "maximumIntroducedTongueCoefficientStep": (
                config.maximum_introduced_tongue_coefficient_step
            ),
            "taperFrames": config.taper_frames,
        },
        "bindings": {
            "captureSourceSha256": performance.provenance.capture_source_sha256,
            "nativeMonoPcmSha256": pcm_sha256,
            "identitySha256": _array_sha256(performance.identity),
            "sourcePtsSha256": _array_sha256(performance.source_pts),
            "inputExpressionSha256": _array_sha256(performance.expression),
            "audioExpressionSha256": _array_sha256(controls["expression"]),
            "outputExpressionSha256": _array_sha256(revised.expression),
            "learnedBackendName": analysis.get("backend"),
            "learnedRetargeter": analysis.get("retargeter"),
            "learnedRetargetCalibrationSha256": analysis.get(
                "retarget_calibration_hash"
            ),
        },
        "clockJoin": {
            "sampleRate": sample_rate,
            "coveredVideoFrames": int(np.count_nonzero(coverage)),
            "videoFrames": performance.frame_count,
            "mapping": "exact_video_display_start_minus_decoded_audio_start_no_time_warp",
            "controlSupport": "frame_samples_with_final_sample_valid_for_one_control_cadence",
        },
        "sourceAuthority": {
            "headPose": "video_locked",
            "translation": "video_locked",
            "gazeAndPupils": "video_locked",
            "upperFace": "video_locked",
            "trustedVisibleMouth": "video_locked",
            "globallyLowQualityOrMissingLowerFace": "learned_audio_repair",
            "dedicatedTongue": "learned_audio_when_active",
        },
        "metrics": {
            "trustedVisualMouthFrames": int(np.count_nonzero(trusted_visual)),
            "globallyLowQualityOrMissingLowerFaceFrames": int(
                np.count_nonzero(weak_visual)
            ),
            "speechCoveredFrames": int(np.count_nonzero(speech_available)),
            "lowerFaceAudioWeightedFrames": lower_weighted_frames,
            "lowerFaceRepairedFrames": lower_frames,
            "dedicatedTongueAudioWeightedFrames": tongue_weighted_frames,
            "dedicatedTongueDrivenFrames": tongue_frames,
            "audioVisualContactConflictFrames": conflict_frames,
            "audioContactCandidateFrames": contact_candidate_frames,
            "finalAudioContactAttainmentFraction": contact_attainment,
            "mouthContinuityLimitedRuns": continuity_limited_runs,
            "tongueContinuityLimitedRuns": tongue_continuity_limited_runs,
            "baselineMouthStepMaxInterocular": (
                float(np.max(base_mouth_steps, initial=0.0)) if rig is not None else None
            ),
            "finalMouthStepMaxInterocular": (
                float(np.max(final_mouth_steps, initial=0.0)) if rig is not None else None
            ),
            "maximumLowerFaceAudioWeight": float(np.max(lower_weight, initial=0.0)),
            "baselineTongueCoefficientStepMax": float(
                np.max(baseline_tongue_step, initial=0.0)
            ),
            "finalTongueCoefficientStepMax": float(
                np.max(final_tongue_step, initial=0.0)
            ),
        },
        "locks": {
            "upperFaceExact": upper_locked,
            "pupilExact": pupil_locked,
            "headPoseAndTranslationExact": pose_locked,
            "sourcePtsAndTimestampsExact": timing_locked,
            "trustedVisualContactDisagreementIsDiagnostic": bool(
                not np.any(lower_eligible[conflict])
            ),
            "visibleContactProtectedByVisualOwnership": bool(
                not np.any(lower_weight[protected_visual_contact] > 0.0)
            ),
            "mouthContinuityGeometryValidated": rig is not None,
            "tongueCoefficientContinuityValidated": True,
        },
        "claims": {
            "audioContentClassified": False,
            "speechActivityClassified": True,
            "rhubarbMouthCuesUsed": True,
            "lexicalTranscriptGenerated": False,
            "speechMotionInferred": True,
            "changesFinalGNMMotion": bool(lower_frames or tongue_frames),
            "lowerFaceChangedOnlyWhereGlobalVisualQualityWeakOrMissing": bool(
                np.all(~lower_changed | weak_visual)
            ),
            "emotionInferredForFinalMotion": False,
            "tongueVisibleValidated": False,
            "contradictoryMediaValidated": False,
            "artistPreferenceValidated": False,
            "qualificationProfileSha256": None,
            "productionValidated": False,
        },
        "outputRole": "intermediate_pre_artist_mouth_aperture_revision",
        "caveats": [
            "This deterministic fusion policy is not a trained audiovisual performance model.",
            "MediaPipe RGB may infer plausible landmarks through an occluder without exposing mouth-specific uncertainty.",
            "Dedicated tongue motion is inferred from audio; the source video does not independently validate tongue visibility or collision.",
            "A retained labeled audiovisual benchmark and artist approval remain required for production validation.",
        ],
    }
    result = AudioVisualRepairResult(
        performance=revised,
        report=report,
        frame_audio_times_seconds=query.astype(np.float64),
        lower_face_audio_weight=lower_weight,
        tongue_audio_weight=tongue_weight,
        speech_activity=speech,
        audio_lip_contact_confidence=audio_contact,
        audio_visual_contact_conflict=conflict,
        resampled_audio_expression=expression,
        final_lip_gap_interocular=final_lip_gap,
        final_audio_contact_attained=final_audio_contact_attained,
    )
    if output_dir is not None:
        destination = Path(output_dir)
        write_json(destination / "audio-visual-repair.json", report)
        write_npz(
            destination / "audio-visual-repair.npz",
            source_pts=performance.source_pts,
            timestamps_seconds=performance.timestamps_seconds,
            frame_audio_times_seconds=result.frame_audio_times_seconds,
            lower_face_audio_weight=lower_weight,
            tongue_audio_weight=tongue_weight,
            speech_activity=speech,
            audio_lip_contact_confidence=audio_contact,
            audio_visual_contact_conflict=conflict,
            visual_source_lip_geometry_valid=performance.source_lip_geometry_valid,
            visual_source_lip_gap_interocular=performance.source_lip_gap_interocular,
            visual_source_lip_contact_confidence=(
                performance.source_lip_contact_confidence
            ),
            visual_source_contact_correction_applied=(
                performance.contact_correction_applied
            ),
            visual_source_contact_attained=performance.lip_contact_attained,
            audio_contact_target_gap_interocular=audio_contact_target,
            audio_contact_correction_applied=audio_contact_applied,
            audio_contact_attained_upstream=audio_contact_attained,
            final_lip_gap_interocular=final_lip_gap,
            final_audio_contact_candidate=audio_contact_candidate,
            final_audio_contact_attained=final_audio_contact_attained,
            input_expression=performance.expression,
            resampled_audio_expression=expression,
            output_expression=revised.expression,
            baseline_mouth_step_interocular=base_mouth_steps,
            final_mouth_step_interocular=final_mouth_steps,
            baseline_tongue_coefficient_step=baseline_tongue_step,
            final_tongue_coefficient_step=final_tongue_step,
        )
    return result


__all__ = [
    "AUDIO_VISUAL_REPAIR_POLICY",
    "AUDIO_VISUAL_REPAIR_SCHEMA_VERSION",
    "AudioVisualRepairConfig",
    "AudioVisualRepairResult",
    "apply_audio_visual_repair",
]
