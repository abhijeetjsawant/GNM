"""End-to-end monocular video performance capture for GNM."""

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np

from .a2f import ARKitGNMRetargeter
from .animation import _face_local_mouth, calibrate_lip_contact
from .animated_gltf import AnimationCompressionError, export_animated_gnm_glb
from .calibrated_retarget import CalibratedRetargetError, CalibratedRetargeter
from .errors import AutoAnimError
from .gltf_export import export_gnm_glb
from .gnm_adapter import GNMAdapter
from .mouth_aperture_correction import (
    MouthApertureConfig,
    MouthApertureCorrectionResult,
    MouthContactEvidence,
    correct_mouth_aperture,
    mouth_aperture_target_attainment,
    validate_mouth_aperture_authorship,
)
from .oral_validation import (
    OralValidationError,
    require_glb_oral_semantic_preservation,
    validate_controls_npz,
    validate_glb_oral_geometry,
)
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json, write_npz
from .video_capture import (
    MONOCULAR_SCALE_CAVEAT,
    capture_video,
    probe_video,
    serialize_capture,
)
from .video_evidence import (
    AUDIO_VIDEO_TIMING_POLICY,
    AUDIO_VIDEO_TIMING_SCHEMA_VERSION,
    PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
    build_audio_video_timing_evidence,
    write_performance_evidence,
)
from .video_retarget import (
    FAST_CONTACT_CONTROLS,
    NEUTRAL_CALIBRATION_CAVEAT,
    QUARANTINED_EXPRESSION_CONTROLS,
    SOURCE_APERTURE_MAX_TARGET_INTEROCULAR,
    GNMPerformanceTrack,
    filter_blendshapes,
    retarget_capture,
    serialize_performance,
)


VIDEO_SEMANTIC_RETARGET_CAVEAT = (
    "MediaPipe performance timing is captured from video, but its ARKit-like controls are "
    "retargeted through an uncalibrated GNM approximation and are not production-approved."
)
VIDEO_CALIBRATED_RETARGET_CAVEAT = (
    "The Claire-to-GNM geometry mapping is calibrated, but MediaPipe's monocular coefficients "
    "are tracker estimates rather than subject-calibrated FACS measurements; artist review and "
    "a labeled performance benchmark are still required for production approval."
)
VIDEO_TRACKING_CAVEAT = (
    "Monocular RGB capture can lose depth, tongue, gaze, and occluded microexpressions; "
    "TrueDepth or synchronized multiview capture is the higher-accuracy tier."
)
MAX_INTERACTIVE_FRAMES = 1_800
MAX_PROXY_PTS_ERROR_SECONDS = 0.002
MAX_AUTHORED_APERTURE_SOURCE_STEP_INTEROCULAR = 0.08


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _apply_video_mouth_aperture_edit(
    *,
    output_dir: Path,
    rig: ControlRig,
    performance: GNMPerformanceTrack,
    gain: float,
    author: str | None,
    reason: str | None,
    source_sha256: str,
    model_sha256: str,
    retarget_calibration_hash: str | None,
) -> tuple[GNMPerformanceTrack, MouthApertureCorrectionResult]:
    """Create a PTS-bound, contact-vetoed revision of a video performance."""

    author, reason = validate_mouth_aperture_authorship(
        gain=gain,
        author=author,
        reason=reason,
    )
    config = MouthApertureConfig(gain=float(gain))
    contact_anchor = np.asarray(
        (performance.source_lip_contact_confidence >= config.contact_confidence_threshold)
        | (performance.lip_contact_target_gap_interocular > 0.0)
        | performance.contact_correction_applied
        | performance.lip_contact_attained,
        dtype=bool,
    )
    eligible = np.asarray(
        performance.detected
        & performance.source_lip_geometry_valid
        & (performance.source_lip_contact_confidence < config.contact_confidence_threshold)
        & (performance.source_lip_gap_interocular >= 0.055),
        dtype=bool,
    )
    local_mouth = np.stack(
        [_face_local_mouth(rig, frame) for frame in performance.expression]
    )
    source_step = np.zeros(performance.frame_count, dtype=np.float32)
    if performance.frame_count > 1:
        edge_step = np.max(
            np.linalg.norm(np.diff(local_mouth, axis=0), axis=2),
            axis=1,
        ).astype(np.float32)
        source_step[1:] = np.maximum(source_step[1:], edge_step)
        source_step[:-1] = np.maximum(source_step[:-1], edge_step)
    rapid_source_motion = np.asarray(
        source_step > MAX_AUTHORED_APERTURE_SOURCE_STEP_INTEROCULAR,
        dtype=bool,
    )
    eligible &= ~rapid_source_motion
    labels = tuple("contact" if value else "none" for value in contact_anchor)
    base_expression = np.asarray(performance.expression, dtype=np.float32).copy()
    correction = correct_mouth_aperture(
        rig,
        identity=np.asarray(performance.identity, dtype=np.float32),
        expression=base_expression,
        rotations=np.asarray(performance.rotations, dtype=np.float32),
        translation=np.asarray(performance.translation, dtype=np.float32),
        timestamps_seconds=np.asarray(performance.timestamps_seconds, dtype=np.float64),
        eligible_frames=eligible,
        contact_evidence=MouthContactEvidence(
            anchor=contact_anchor,
            confidence=np.asarray(
                performance.source_lip_contact_confidence,
                dtype=np.float32,
            ),
            label=labels,
        ),
        config=config,
    )
    corrected = replace(performance, expression=correction.expression)
    frame_reports = [asdict(report) for report in correction.reports]
    target_attainment = mouth_aperture_target_attainment(correction)
    source_pts_sha256 = _array_sha256(performance.source_pts)
    write_npz(
        output_dir / "mouth-aperture-edit.npz",
        base_expression=base_expression,
        corrected_expression=correction.expression,
        source_pts=performance.source_pts,
        timestamps_seconds=performance.timestamps_seconds,
        eligible_frames=eligible,
        rapid_source_motion=rapid_source_motion,
        source_mouth_step_interocular=source_step,
        protected_contact=correction.protected_contact,
        correction_applied=correction.correction_applied,
        target_attained=correction.target_attained,
        final_continuity_scale=correction.final_continuity_scale,
        requested_target_gap_interocular=np.asarray(
            [report.requested_target_gap_interocular for report in correction.reports],
            dtype=np.float32,
        ),
        bounded_target_gap_interocular=np.asarray(
            [report.bounded_target_gap_interocular for report in correction.reports],
            dtype=np.float32,
        ),
        final_gap_interocular=np.asarray(
            [report.final_gap_interocular for report in correction.reports],
            dtype=np.float32,
        ),
    )
    corrected_count = int(np.count_nonzero(correction.correction_applied))
    write_json(
        output_dir / "mouth-aperture-edit.json",
        {
            "schema_version": correction.schema_version,
            "source_mode": "video_follow",
            "status": (
                "corrected"
                if corrected_count
                else "exact_noop"
                if gain == 1.0
                else "bounded_no_change"
            ),
            "authored_edit": gain != 1.0,
            "author": author,
            "reason": reason,
            "config": asdict(config),
            "bindings": {
                "source_sha256": source_sha256,
                "model_sha256": model_sha256,
                "identity_sha256": correction.identity_sha256,
                "source_pts_sha256": source_pts_sha256,
                "retarget_calibration_sha256": retarget_calibration_hash,
                "base_performance_input_sha256": correction.input_sha256,
                "revised_performance_output_sha256": correction.output_sha256,
            },
            "timeline": {
                "frame_count": performance.frame_count,
                "source_pts": performance.source_pts.tolist(),
                "timestamps_seconds": performance.timestamps_seconds.tolist(),
            },
            "summary": {
                "eligible_open_frames": int(np.count_nonzero(correction.eligible_open)),
                "protected_contact_frames": int(np.count_nonzero(correction.protected_contact)),
                "rapid_source_motion_veto_frames": int(
                    np.count_nonzero(rapid_source_motion)
                ),
                "rapid_source_motion_threshold_interocular": (
                    MAX_AUTHORED_APERTURE_SOURCE_STEP_INTEROCULAR
                ),
                "corrected_frames": corrected_count,
                "final_continuity_limited_frames": int(
                    np.count_nonzero(
                        correction.correction_applied
                        & (correction.final_continuity_scale < 1.0 - 1.0e-6)
                    )
                ),
                "final_continuity_limit_interocular": (
                    correction.final_continuity_limit_interocular
                ),
                "target_attained_fraction": target_attainment,
                "maximum_tongue_mesh_tail_interocular": float(
                    max(
                        (report.tongue_displacement_interocular for report in correction.reports),
                        default=0.0,
                    )
                ),
                "baseline_lip_order_risk_frames": int(
                    sum(
                        report.original_lip_order_minimum_interocular < -0.0005
                        for report in correction.reports
                    )
                ),
                "revised_lip_order_risk_frames": int(
                    sum(report.lip_order_inversion_risk for report in correction.reports)
                ),
                "introduced_lip_order_risk_frames": int(
                    sum(
                        report.lip_order_inversion_introduced
                        for report in correction.reports
                    )
                ),
            },
            "claims": {
                "video_pts_byte_identical": True,
                "pose_and_translation_byte_identical": True,
                "upper_face_coefficients_byte_identical": True,
                "tongue_coefficients_byte_identical": True,
                "tongue_mesh_vertices_exactly_unchanged": False,
                "contact_is_a_hard_veto": True,
                "rapid_source_motion_is_a_hard_veto": True,
                "new_lip_order_inversion_rejected": True,
                "production_validated": False,
            },
            "frame_reports": frame_reports,
        },
    )
    return corrected, correction


def _proxy_video(source: Path, output: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise AutoAnimError("DEPENDENCY_MISSING", "ffmpeg is required for video capture")
    source_probe = probe_video(source)
    # Capture time zero is the first displayable video frame, not necessarily
    # the container's earliest audio/data timestamp.  Output-side accurate
    # seeking discards any leading audio and makes HTMLMediaElement.currentTime
    # share that same zero without changing display-order frame timing.
    first_video_timestamp = float(
        Fraction(int(source_probe.source_pts[0])) * source_probe.time_base
    )
    trim_start = max(first_video_timestamp, 0.0)
    command = (
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{trim_start:.9f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "passthrough",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-metadata",
        "creation_time=",
        str(output),
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise AutoAnimError("LIMIT_EXCEEDED", "Video proxy generation timed out") from exc
    except subprocess.CalledProcessError as exc:
        raise AutoAnimError(
            "MEDIA_INVALID", f"Could not create browser video proxy: {exc.stderr}"
        ) from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise AutoAnimError("INTERNAL_ERROR", "Video proxy was not created")
    return output


def _longest_missing(detected: np.ndarray) -> int:
    longest = current = 0
    for value in detected:
        current = 0 if bool(value) else current + 1
        longest = max(longest, current)
    return longest


def _proxy_timing_error(source: Path, proxy: Path) -> tuple[int, float, float]:
    """Prove that the browser proxy retains the source display-frame clock."""

    source_probe = probe_video(source)
    proxy_probe = probe_video(proxy)
    if source_probe.frame_count != proxy_probe.frame_count:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Browser proxy changed the video frame count "
            f"({source_probe.frame_count} source, {proxy_probe.frame_count} proxy)",
        )
    error = float(
        np.max(
            np.abs(source_probe.timestamps_seconds - proxy_probe.timestamps_seconds),
            initial=0.0,
        )
    )
    if error > MAX_PROXY_PTS_ERROR_SECONDS:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Browser proxy changed presentation timing by "
            f"{error * 1_000:.3f} ms (limit {MAX_PROXY_PTS_ERROR_SECONDS * 1_000:.1f} ms)",
        )
    proxy_video_start = float(
        Fraction(int(proxy_probe.source_pts[0])) * proxy_probe.time_base
    )
    if abs(proxy_video_start) > MAX_PROXY_PTS_ERROR_SECONDS:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            "Browser proxy's first video frame does not start at media time zero "
            f"({proxy_video_start * 1_000:.3f} ms)",
        )
    return proxy_probe.frame_count, error, proxy_video_start


def _source_motion_metrics(capture) -> dict[str, float | bool]:
    """Measure preservation, not accuracy, of tracker-observed temporal detail."""

    filtered = filter_blendshapes(capture)
    names = tuple(capture.blendshape_names)
    non_contact = np.asarray(
        [index for index, name in enumerate(names) if name not in FAST_CONTACT_CONTROLS],
        dtype=np.int64,
    )
    if capture.frame_count <= 1 or not len(non_contact):
        retention = 1.0
    else:
        raw_delta = np.diff(capture.blendshape_scores[:, non_contact], axis=0)
        filtered_delta = np.diff(filtered.scores[:, non_contact], axis=0)
        raw_energy = float(np.linalg.norm(raw_delta))
        retention = (
            float(np.linalg.norm(filtered_delta)) / raw_energy
            if raw_energy > 1e-12
            else 1.0
        )
    contact_indices = np.asarray(
        [index for index, name in enumerate(names) if name in FAST_CONTACT_CONTROLS],
        dtype=np.int64,
    )
    contact_exact = bool(
        not len(contact_indices)
        or np.array_equal(
            filtered.scores[:, contact_indices],
            capture.blendshape_scores[:, contact_indices],
        )
    )
    return {
        "source_noncontact_filter_variation_retention": retention,
        "source_fast_contact_filter_passthrough_exact": contact_exact,
    }


def _unit_curve(values: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values, False
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    span = high - low
    if span <= 1e-8:
        return np.zeros_like(values), False
    return np.clip((values - low) / span, 0.0, 1.0), True


def _curve_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or len(left) < 2:
        return None
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-10:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _event_peaks(values: np.ndarray, threshold: float) -> tuple[int, ...]:
    values = np.asarray(values, dtype=np.float64)
    active = values >= threshold
    peaks: list[int] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            stop = index + 1 if value and index == len(active) - 1 else index
            peaks.append(start + int(np.argmax(values[start:stop])))
            start = None
    return tuple(peaks)


def _event_retention(
    source: np.ndarray,
    output: np.ndarray,
    *,
    source_threshold: float = 0.35,
    output_threshold: float = 0.20,
    radius: int = 1,
) -> tuple[int, float | None]:
    peaks = _event_peaks(source, source_threshold)
    if not peaks:
        return 0, None
    retained = 0
    for peak in peaks:
        start = max(0, peak - radius)
        stop = min(len(output), peak + radius + 1)
        retained += int(float(np.max(output[start:stop], initial=0.0)) >= output_threshold)
    return len(peaks), float(retained / len(peaks))


def _contact_event_curve(values: np.ndarray) -> np.ndarray:
    """Exclude a clip-leading closed rest from dynamic contact timing.

    A subject can begin with closed lips. That is a valid pose for spatial
    correction, but it is not a closure event until the mouth has first
    released. Keeping the distinction prevents neutral lead-in frames from
    receiving a misleading timing score.
    """

    output = np.asarray(values, dtype=np.float64).copy()
    if len(output) and output[0] >= 0.35:
        released = np.flatnonzero(output < 0.12)
        if not len(released):
            output[:] = 0.0
        else:
            output[: int(released[0]) + 1] = 0.0
    return output


def _final_output_retention_metrics(capture, performance, adapter: GNMAdapter) -> dict:
    """Measure whether tracker events survive in final GNM geometry.

    This is transport evidence, not event accuracy: MediaPipe supplies no phone,
    FACS, or eyelid-contact annotation. A missing source event is reported as
    unmeasurable instead of receiving a misleading perfect score.
    """

    filtered = filter_blendshapes(capture)
    columns = {name: index for index, name in enumerate(filtered.names)}
    baseline = np.asarray(
        [dict(performance.provenance.neutral_blendshape_baseline)[name] for name in filtered.names],
        dtype=np.float64,
    )
    calibrated = np.clip(
        (filtered.scores.astype(np.float64) - baseline)
        / np.maximum(1.0 - baseline, 1e-4),
        0.0,
        1.0,
    )

    def control(name: str) -> np.ndarray:
        index = columns.get(name)
        return (
            calibrated[:, index]
            if index is not None
            else np.zeros(performance.frame_count, dtype=np.float64)
        )

    source_blink = np.maximum(control("eyeBlinkLeft"), control("eyeBlinkRight"))
    source_contact = _contact_event_curve(
        performance.source_lip_contact_confidence
    )

    neutral_landmarks = adapter.compact_template + np.einsum(
        "i,ijk->jk",
        performance.identity,
        adapter.compact_identity_basis,
        optimize=True,
    )
    landmarks = neutral_landmarks[None] + np.einsum(
        "ti,ijk->tjk",
        performance.expression,
        adapter.compact_expression_basis,
        optimize=True,
    )
    iod = float(np.linalg.norm(neutral_landmarks[36] - neutral_landmarks[45]))
    iod = max(iod, 1e-8)
    eye_aperture = np.mean(
        np.stack(
            [
                np.linalg.norm(landmarks[:, upper] - landmarks[:, lower], axis=1)
                for upper, lower in ((37, 41), (38, 40), (43, 47), (44, 46))
            ],
            axis=1,
        ),
        axis=1,
    ) / iod
    lip_gap = np.mean(
        np.stack(
            [
                np.linalg.norm(landmarks[:, upper] - landmarks[:, lower], axis=1)
                for upper, lower in ((61, 67), (62, 66), (63, 65))
            ],
            axis=1,
        ),
        axis=1,
    ) / iod
    blink_closure, blink_geometry_varies = _unit_curve(-eye_aperture)
    lip_closure, lip_geometry_varies = _unit_curve(-lip_gap)

    source_expression_columns = np.asarray(
        [
            index
            for index, name in enumerate(filtered.names)
            if name != "_neutral"
            and not name.startswith("eyeLook")
            and name not in QUARANTINED_EXPRESSION_CONTROLS
        ],
        dtype=np.int64,
    )
    source_expression_motion = (
        np.linalg.norm(
            np.diff(calibrated[:, source_expression_columns], axis=0), axis=1
        )
        if performance.frame_count > 1 and len(source_expression_columns)
        else np.zeros(0, dtype=np.float64)
    )
    final_expression_motion = (
        np.max(np.linalg.norm(np.diff(landmarks, axis=0), axis=2), axis=1) / iod
        if performance.frame_count > 1
        else np.zeros(0, dtype=np.float64)
    )
    source_expression_curve, source_expression_varies = _unit_curve(
        source_expression_motion
    )
    final_expression_curve, final_expression_varies = _unit_curve(
        final_expression_motion
    )

    blink_count, blink_retention = _event_retention(source_blink, blink_closure)
    contact_count, contact_retention = _event_retention(
        source_contact,
        performance.lip_contact_attained.astype(np.float64),
        source_threshold=0.65,
        output_threshold=0.5,
    )
    expression_count, expression_retention = _event_retention(
        source_expression_curve,
        final_expression_curve,
        source_threshold=0.55,
        output_threshold=0.20,
    )
    high_confidence_contact = source_contact >= 0.65
    open_geometry = (
        performance.source_lip_geometry_valid
        & (performance.source_lip_contact_confidence < 0.12)
        & (performance.source_lip_gap_interocular >= 0.055)
    )
    if np.count_nonzero(open_geometry) >= 3:
        source_open = performance.source_lip_gap_interocular[open_geometry].astype(
            np.float64
        )
        final_open = lip_gap[open_geometry]
        aperture_correlation = _curve_correlation(source_open, final_open)
        source_variance = float(np.var(source_open))
        aperture_slope = (
            float(np.cov(source_open, final_open, ddof=0)[0, 1] / source_variance)
            if source_variance > 1.0e-12
            else None
        )
        aperture_p95_ratio = float(
            np.percentile(final_open, 95)
            / max(float(np.percentile(source_open, 95)), 1.0e-8)
        )
    else:
        aperture_correlation = None
        aperture_slope = None
        aperture_p95_ratio = None
    aperture_candidates = performance.lip_aperture_target_gap_interocular > 0.0
    return {
        "final_blink_source_event_count": blink_count,
        "final_blink_event_retained_fraction": blink_retention,
        "final_blink_motion_correlation": (
            _curve_correlation(source_blink, blink_closure)
            if blink_geometry_varies
            else None
        ),
        "final_blink_retention_measurable": bool(blink_count and blink_geometry_varies),
        "final_contact_source_event_count": contact_count,
        "final_contact_event_retained_fraction": contact_retention,
        "final_contact_motion_correlation": (
            _curve_correlation(source_contact, lip_closure)
            if lip_geometry_varies
            else None
        ),
        "final_contact_retention_measurable": bool(contact_count and lip_geometry_varies),
        "final_contact_geometry_attained_fraction": (
            float(np.mean(performance.lip_contact_attained[high_confidence_contact]))
            if np.any(high_confidence_contact)
            else None
        ),
        "final_contact_correction_applied_frames": int(
            np.count_nonzero(performance.contact_correction_applied)
        ),
        "source_lip_contact_gap_min_interocular": (
            float(
                np.min(
                    performance.source_lip_gap_interocular[
                        performance.source_lip_geometry_valid
                    ]
                )
            )
            if np.any(performance.source_lip_geometry_valid)
            else None
        ),
        "final_lip_aperture_open_frame_count": int(np.count_nonzero(open_geometry)),
        "final_lip_aperture_source_output_correlation": aperture_correlation,
        "final_lip_aperture_affine_slope": aperture_slope,
        "final_lip_aperture_open_p95_ratio": aperture_p95_ratio,
        "final_lip_aperture_correction_applied_frames": int(
            np.count_nonzero(performance.lip_aperture_correction_applied)
        ),
        "final_lip_aperture_target_attainment_fraction": (
            float(
                np.mean(
                    performance.lip_aperture_target_attained[aperture_candidates]
                )
            )
            if np.any(aperture_candidates)
            else None
        ),
        "final_expression_source_event_count": expression_count,
        "final_expression_motion_retained_fraction": expression_retention,
        "final_expression_motion_correlation": (
            _curve_correlation(source_expression_curve, final_expression_curve)
            if source_expression_varies and final_expression_varies
            else None
        ),
        "final_expression_retention_measurable": bool(
            expression_count and source_expression_varies and final_expression_varies
        ),
        "final_blink_geometry_range_interocular": float(np.ptp(eye_aperture)),
        "final_lip_gap_range_interocular": float(np.ptp(lip_gap)),
        "final_expression_landmark_step_p95_interocular": (
            float(np.percentile(final_expression_motion, 95))
            if len(final_expression_motion)
            else 0.0
        ),
    }


def _export_static_performance_glb(
    output: Path,
    adapter: GNMAdapter,
    frame: np.ndarray,
    *,
    texture_path: str | Path | None,
    runtime_material_paths: Mapping[str, str | Path] | None = None,
    texture_triangle_uvs: np.ndarray | None = None,
):
    """Export the long-track/compression fallback with the character's exact atlas."""

    if runtime_material_paths:
        return export_animated_gnm_glb(
            output / "performance.glb",
            adapter,
            frame[None, ...],
            np.asarray([0.0], dtype=np.float32),
            mapping_path=output / "performance-glb-mapping.npz",
            triangle_uvs=texture_triangle_uvs,
            runtime_material_paths=runtime_material_paths,
        )
    return export_gnm_glb(
        output / "performance.glb",
        adapter,
        frame,
        mapping_path=output / "performance-glb-mapping.npz",
        texture_path=texture_path,
        triangle_uvs=texture_triangle_uvs,
    )


def run_video_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path,
    identity: np.ndarray | None = None,
    a2f_asset_dir: str | Path | None = None,
    texture_path: str | Path | None = None,
    runtime_material_paths: Mapping[str, str | Path] | None = None,
    texture_triangle_uvs: np.ndarray | None = None,
    character_ref: dict[str, Any] | None = None,
    audio_video_timing_evidence: bool = True,
    require_audio_visual_repair: bool = False,
    mouth_aperture_gain: float = 1.0,
    mouth_aperture_author: str | None = None,
    mouth_aperture_reason: str | None = None,
) -> dict:
    """Track a real video, retarget it, and package synchronized 3D playback."""

    source = Path(input_path).resolve()
    mouth_aperture_author, mouth_aperture_reason = validate_mouth_aperture_authorship(
        gain=mouth_aperture_gain,
        author=mouth_aperture_author,
        reason=mouth_aperture_reason,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = capture_video(source, model_path)
    detected_count = int(np.count_nonzero(capture.detected))
    if detected_count == 0:
        raise AutoAnimError("FACE_NOT_FOUND", "No face was detected in the video")
    serialize_capture(output, capture)
    write_performance_evidence(output / "performance-evidence.json", capture)
    if require_audio_visual_repair and not audio_video_timing_evidence:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Required audio-visual timing evidence cannot be disabled",
        )
    audio_video_timing: dict[str, Any] | None = None
    if audio_video_timing_evidence:
        audio_video_timing = build_audio_video_timing_evidence(
            source,
            expected_source_sha256=capture.provenance.source_sha256,
            expected_video_source_pts=capture.source_pts,
            expected_video_time_base=Fraction(
                capture.provenance.time_base_numerator,
                capture.provenance.time_base_denominator,
            ),
            require_available=require_audio_visual_repair,
        )
        write_json(output / "audio-video-timing.json", audio_video_timing)
        if (
            require_audio_visual_repair
            and audio_video_timing["fusionGate"]["status"] != "ready"
        ):
            raise AutoAnimError(
                "AUDIO_VISUAL_REPAIR_BLOCKED",
                "Audio-visual repair was requested, but its fusion evidence is blocked",
                {"reasons": audio_video_timing["fusionGate"]["reasons"]},
            )

    adapter = GNMAdapter()
    identity_value = (
        np.zeros(adapter.identity_dim, dtype=np.float32)
        if identity is None
        else np.asarray(identity, dtype=np.float32).copy()
    )
    if identity_value.shape != (adapter.identity_dim,) or not np.isfinite(identity_value).all():
        raise AutoAnimError("INPUT_INVALID", "Character identity must be one finite (253,) vector")
    decoder = ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5")
    contact_rig = ControlRig(adapter, decoder, identity=identity_value)
    lip_contact_calibration = calibrate_lip_contact(contact_rig)
    calibration_hash: str | None = None
    retarget_artifacts: dict[str, str] = {}
    source_names = set(capture.blendshape_names)
    if a2f_asset_dir is not None:
        try:
            retargeter = CalibratedRetargeter.from_directory(
                a2f_asset_dir,
                adapter=adapter,
            )
            retargeter.calibration.save(output / "retarget_calibration.npz")
        except (CalibratedRetargetError, OSError, ValueError) as exc:
            raise AutoAnimError(
                "DEPENDENCY_MISSING",
                f"Configured geometry-calibrated video retarget assets are unusable: {exc}",
            ) from exc
        calibration_hash = retargeter.calibration.calibration_hash
        calibrated_names = set(retargeter.calibration.skin_pose_names)
        matched_controls = len(source_names & calibrated_names)
        retarget_backend = "geometry_calibrated_dense_contact_aperture_v3"
        retarget_caveat = VIDEO_CALIBRATED_RETARGET_CAVEAT
        retarget_artifacts["retarget_calibration"] = "retarget_calibration.npz"
    else:
        retargeter = ARKitGNMRetargeter(contact_rig)
        matched_controls = len(
            source_names
            & {source for rule in retargeter.rules for source in rule.sources}
        )
        retarget_backend = "semantic_prototype_contact_aperture_v3_fallback"
        retarget_caveat = VIDEO_SEMANTIC_RETARGET_CAVEAT
    performance = retarget_capture(
        capture,
        retargeter,
        identity=identity_value,
        retarget_caveats=(retarget_caveat,),
        contact_rig=contact_rig,
        lip_contact_calibration=lip_contact_calibration,
    )
    performance, mouth_aperture_edit = _apply_video_mouth_aperture_edit(
        output_dir=output,
        rig=contact_rig,
        performance=performance,
        gain=mouth_aperture_gain,
        author=mouth_aperture_author,
        reason=mouth_aperture_reason,
        source_sha256=capture.provenance.source_sha256,
        model_sha256=capture.provenance.model_sha256,
        retarget_calibration_hash=calibration_hash,
    )
    retarget_artifacts.update(
        {
            "mouth_aperture_edit": "mouth-aperture-edit.json",
            "mouth_aperture_edit_arrays": "mouth-aperture-edit.npz",
        }
    )
    serialize_performance(output, performance)
    proxy = _proxy_video(source, output / "source-proxy.mp4")
    proxy_frames, proxy_pts_error, proxy_video_start = _proxy_timing_error(source, proxy)

    warnings = [
        MONOCULAR_SCALE_CAVEAT,
        NEUTRAL_CALIBRATION_CAVEAT,
        retarget_caveat,
        VIDEO_TRACKING_CAVEAT,
    ]
    if audio_video_timing is not None:
        timing_status = str(audio_video_timing["status"])
        if timing_status != "available_observation":
            warnings.append(
                "AUDIO_VIDEO_TIMING_FUSION_BLOCKED: retained-source audio timing is "
                f"{timing_status}; visual-only motion remains unchanged."
            )
        else:
            timing_reasons = audio_video_timing["fusionGate"]["reasons"]
            if "nonzero_av_start_offset" in timing_reasons:
                warnings.append(
                    "AUDIO_VIDEO_START_OFFSET_OBSERVED: the exact retained-source A/V "
                    "start offset is recorded but is not applied to motion."
                )
            if "nonzero_av_duration_drift" in timing_reasons:
                warnings.append(
                    "AUDIO_VIDEO_DURATION_DRIFT_OBSERVED: the exact retained-source A/V "
                    "duration drift is recorded but is not applied to motion."
                )
    corrected_mouth_frames = int(np.count_nonzero(mouth_aperture_edit.correction_applied))
    if mouth_aperture_gain != 1.0:
        warnings.append(
            "ARTIST_MOUTH_APERTURE_EDIT: an authored neutral-relative geometry correction "
            f"changed {corrected_mouth_frames}/{performance.frame_count} frames; visually "
            "closed/contact frames were vetoed and exact source PTS were preserved."
        )
        warnings.append(
            "MOUTH_APERTURE_PCA_TONGUE_TAIL: GNM tongue coefficients are byte-identical, "
            "but lower-face PCA modes can produce a small bounded displacement on tongue vertices."
        )
    if performance.provenance.neutral_baseline_method == "none_expressive_video":
        warnings.append(
            "NEUTRAL_BASELINE_NOT_FOUND: no neutral-compatible window was found and "
            "expression baseline subtraction was disabled; this output is not approvable "
            "without a known neutral reference."
        )
    elif performance.provenance.neutral_baseline_method == "auto_low_activity_window":
        warnings.append(
            "NEUTRAL_BASELINE_RELOCATED: the expressive/moving lead-in was rejected and a "
            "later low-activity window was used; confirm it is neutral before approval."
        )
    if performance.provenance.neutral_baseline_ambiguity_controls:
        warnings.append(
            "NEUTRAL_BASELINE_SEMANTIC_AMBIGUITY: high reference coefficients in "
            + ", ".join(performance.provenance.neutral_baseline_ambiguity_controls)
            + " may be person/tracker bias or a held expression; a labeled neutral reference "
            "is required for production approval."
        )
    if performance.provenance.negative_baseline_residual_clipped_fraction > 0.20:
        warnings.append(
            "NEUTRAL_BASELINE_ONE_SIDED_LOSS: "
            f"{100.0 * performance.provenance.negative_baseline_residual_clipped_fraction:.1f}% "
            "of non-gaze source residual samples fell below the selected baseline and were "
            "clipped by the one-sided source rig; use a labeled neutral and bidirectional "
            "subject calibration before production approval."
        )
    region_bound_active = np.any(
        np.stack(
            [
                np.max(np.abs(performance.expression[:, start:stop]), axis=1)
                >= 3.0 - 1e-5
                for start, stop in ((0, 100), (100, 200), (200, 350), (350, 382))
            ],
            axis=1,
        ),
        axis=1,
    )
    region_bound_active_frames = int(np.count_nonzero(region_bound_active))
    if region_bound_active_frames:
        warnings.append(
            "The GNM expression-region safety bound was active on "
            f"{region_bound_active_frames}/{performance.frame_count} frames; inspect the "
            "performance for flattened peaks before approval."
        )
    viewer_status = "ready"
    viewer_mode = "animation"
    glb_covers_full_track = False
    evaluated_full_track_frames: np.ndarray | None = None
    viewer_reconstruction: dict[str, float | int | str] = {}
    if performance.frame_count <= MAX_INTERACTIVE_FRAMES:
        frames = np.stack(
            [
                adapter.mesh(
                    identity=performance.identity,
                    expression=expression,
                    rotations=rotations,
                    translation=translation,
                )
                for expression, rotations, translation in zip(
                    performance.expression,
                    performance.rotations,
                    performance.translation,
                    strict=True,
                )
            ]
        )
        evaluated_full_track_frames = frames
        try:
            exported = export_animated_gnm_glb(
                output / "performance.glb",
                adapter,
                frames,
                performance.timestamps_seconds,
                mapping_path=output / "performance-glb-mapping.npz",
                texture_path=texture_path,
                triangle_uvs=texture_triangle_uvs,
                runtime_material_paths=runtime_material_paths,
            )
            glb_covers_full_track = True
            if not exported.rank:
                viewer_status = "static_only"
                viewer_mode = "static"
            viewer_reconstruction = {
                "expression_pose_rank": exported.rank,
                "validation_scope": "all_frames",
                "mesh_p95_mm": exported.mesh_p95_mm,
                "mesh_max_mm": exported.mesh_max_mm,
                "landmark_p95_mm": exported.landmark_p95_mm,
                "landmark_max_mm": exported.landmark_max_mm,
            }
        except AnimationCompressionError as exc:
            _export_static_performance_glb(
                output,
                adapter,
                frames[0],
                texture_path=texture_path,
                runtime_material_paths=runtime_material_paths,
                texture_triangle_uvs=texture_triangle_uvs,
            )
            viewer_status = "static_only"
            viewer_mode = "static"
            warnings.append("VIEWER_RECONSTRUCTION_LIMIT")
            viewer_reconstruction = {
                "validation_scope": "all_frames",
                "reason": str(exc),
                **{key: value for key, value in exc.metrics.items()},
            }
    else:
        first = adapter.mesh(
            identity=performance.identity,
            expression=performance.expression[0],
            rotations=performance.rotations[0],
            translation=performance.translation[0],
        )
        _export_static_performance_glb(
            output,
            adapter,
            first,
            texture_path=texture_path,
            runtime_material_paths=runtime_material_paths,
            texture_triangle_uvs=texture_triangle_uvs,
        )
        viewer_status = "static_only"
        viewer_mode = "static"
        warnings.append("VIEWER_TRACK_TOO_LONG")
        viewer_reconstruction = {
            "validation_scope": "not_run",
            "reason": f"Interactive morph export is limited to {MAX_INTERACTIVE_FRAMES} frames",
        }

    try:
        oral_controls = validate_controls_npz(
            output / "performance.npz",
            adapter=adapter,
            identity=identity_value,
            evaluated_frames=evaluated_full_track_frames,
        )
        oral_glb = validate_glb_oral_geometry(
            output / "performance.glb",
            output / "performance-glb-mapping.npz",
            adapter=adapter,
            reference_controls_path=(
                output / "performance.npz" if glb_covers_full_track else None
            ),
            reference_frames=(
                evaluated_full_track_frames if glb_covers_full_track else None
            ),
            identity=identity_value,
        )
        if glb_covers_full_track:
            require_glb_oral_semantic_preservation(oral_controls, oral_glb)
    except OralValidationError as exc:
        raise AutoAnimError(
            "INTERNAL_ERROR",
            f"Required oral geometry validation failed ({exc.code}): {exc}",
        ) from exc
    write_json(output / "oral-validation.json", oral_controls.as_dict())
    write_json(output / "oral-glb-validation.json", oral_glb.as_dict())

    quality = performance.effective_quality
    duration = (
        float(performance.timestamps_seconds[-1])
        + (
            float(np.median(np.diff(performance.timestamps_seconds)))
            if performance.frame_count > 1
            else 0.0
        )
    )
    lower_delta = (
        np.linalg.norm(np.diff(performance.expression[:, 200:382], axis=0), axis=1)
        if performance.frame_count > 1
        else np.zeros(0, dtype=np.float32)
    )
    source_motion = _source_motion_metrics(capture)
    final_output_retention = _final_output_retention_metrics(
        capture,
        performance,
        adapter,
    )
    for label in ("blink", "contact", "expression"):
        retained = final_output_retention.get(
            f"final_{label}_event_retained_fraction"
            if label != "expression"
            else "final_expression_motion_retained_fraction"
        )
        if retained is not None and float(retained) < 0.60:
            warnings.append(
                f"FINAL_{label.upper()}_RETENTION_LOW: fewer than 60% of tracker-observed "
                "events survive in final GNM geometry; inspect the retarget before approval."
            )
        correlation = final_output_retention.get(
            f"final_{label}_motion_correlation"
        )
        measurable = final_output_retention.get(
            f"final_{label}_retention_measurable"
        )
        if measurable and correlation is not None and float(correlation) < 0.20:
            warnings.append(
                f"FINAL_{label.upper()}_TIMING_MISMATCH: final GNM geometry does not track "
                "the source event envelope (correlation below 0.20); artist correction or a "
                "better calibrated mapping is required."
            )
    aperture_ratio = final_output_retention.get(
        "final_lip_aperture_open_p95_ratio"
    )
    if aperture_ratio is not None and not 0.85 <= float(aperture_ratio) <= 1.15:
        warnings.append(
            "FINAL_LIP_APERTURE_AMPLITUDE_MISMATCH: open-mouth p95 is outside "
            "the 0.85-1.15 source/output review band; artist correction or a "
            "subject-calibrated jaw solve is required."
        )
    if mouth_aperture_gain != 1.0:
        aperture_correlation = final_output_retention.get(
            "final_lip_aperture_source_output_correlation"
        )
        aperture_slope = final_output_retention.get(
            "final_lip_aperture_affine_slope"
        )
        edit_attainment = mouth_aperture_target_attainment(
            mouth_aperture_edit
        )
        if (
            aperture_ratio is None
            or not 0.90 <= float(aperture_ratio) <= 1.10
            or aperture_correlation is None
            or float(aperture_correlation) < 0.95
            or aperture_slope is None
            or not 0.90 <= float(aperture_slope) <= 1.10
            or edit_attainment is None
            or edit_attainment < 0.95
        ):
            warnings.append(
                "MOUTH_APERTURE_EDIT_PRODUCTION_GATE_FAILED: the authored revision did not "
                "meet the frozen source-correlation, amplitude, slope, and target-attainment "
                "acceptance band; keep it as review-only."
            )
    oral_control_report = oral_controls.report
    oral_glb_report = oral_glb.report
    control_lip_order_risk_frames = int(
        np.count_nonzero(oral_controls.lip_order_inversion_risk_frames)
    )
    viewer_lip_order_risk_frames = int(
        np.count_nonzero(oral_glb.lip_order_inversion_risk_frames)
    )
    if (
        oral_controls.lip_order_inversion_risk_frames.shape
        == oral_glb.lip_order_inversion_risk_frames.shape
    ):
        lip_order_risk_frames = int(
            np.count_nonzero(
                oral_controls.lip_order_inversion_risk_frames
                | oral_glb.lip_order_inversion_risk_frames
            )
        )
    else:
        lip_order_risk_frames = max(
            control_lip_order_risk_frames,
            viewer_lip_order_risk_frames,
        )
    control_tongue_teeth_risk_frames = int(
        np.count_nonzero(oral_controls.tongue_teeth_collision_risk_frames)
    )
    viewer_tongue_teeth_risk_frames = int(
        np.count_nonzero(oral_glb.tongue_teeth_collision_risk_frames)
    )
    if (
        oral_controls.tongue_teeth_collision_risk_frames.shape
        == oral_glb.tongue_teeth_collision_risk_frames.shape
    ):
        tongue_teeth_risk_frames = int(
            np.count_nonzero(
                oral_controls.tongue_teeth_collision_risk_frames
                | oral_glb.tongue_teeth_collision_risk_frames
            )
        )
    else:
        tongue_teeth_risk_frames = max(
            control_tongue_teeth_risk_frames,
            viewer_tongue_teeth_risk_frames,
        )
    if lip_order_risk_frames:
        warnings.append(
            "ORAL_LIP_ORDER_RISK: structurally inverted inner-lip landmark ordering was "
            "measured in the control track or reconstructed viewer; inspect those frames "
            "before approval."
        )
    oral_contact_attainment = oral_control_report["lip_contact"]["target_evidence"][
        "geometry_attainment_fraction"
    ]
    if oral_contact_attainment is not None and float(oral_contact_attainment) < 0.95:
        warnings.append(
            "ORAL_CONTACT_TARGET_UNATTAINED: fewer than 95% of the unvalidated contact "
            "targets were reached without violating geometry bounds; unresolved frames remain "
            "artist-review blockers."
        )
    if tongue_teeth_risk_frames:
        warnings.append(
            "ORAL_TONGUE_TEETH_PROXIMITY_RISK: tongue vertices entered the conservative "
            "teeth-proximity risk band in the control track or reconstructed viewer; exact "
            "surface penetration is not established."
        )
    warnings.append(
        "ORAL_TONGUE_SOURCE_UNAVAILABLE: monocular RGB does not provide a dedicated tongue "
        "motion signal, so this pipeline does not infer or validate tongue articulation."
    )
    tongue_geometry_motion_frames = int(
        oral_control_report["tongue_motion"]["moving_frames_over_0_1mm"]
    )
    if tongue_geometry_motion_frames:
        warnings.append(
            "ORAL_UNSOURCED_TONGUE_BASIS_COUPLING: GNM lower-face expression modes moved "
            "tongue vertices even though the video supplied no dedicated tongue signal; "
            "treat the tongue performance as unsourced and require an artist or dedicated "
            "tongue-capture pass."
        )
    if not oral_glb_report["claims"]["structural_reconstruction_validated"]:
        warnings.append(
            "ORAL_GLB_NOT_STRUCTURALLY_VALIDATED: the viewer fallback is static or its "
            "oral reconstruction has not passed the source-control comparison."
        )
    result = {
        "kind": "video_performance",
        "status": "succeeded",
        "model": {
            "gnm_version": "3.0",
            "identity_dim": adapter.identity_dim,
            "expression_dim": adapter.expression_dim,
            "character": character_ref,
            "character_texture_applied_to_glb": (
                texture_path is not None or bool(runtime_material_paths)
            ),
            "character_pbr_runtime_applied_to_glb": bool(runtime_material_paths),
            "source_proxy_is_character_render": False,
        },
        "capture": {
            "backend": "mediapipe-face-landmarker-video",
            "frames": performance.frame_count,
            "duration_s": duration,
            "width": capture.width,
            "height": capture.height,
            "detected_frames": detected_count,
            "identity_fixed_for_all_frames": True,
            "capture_quality_source": (
                "landmark_visibility_when_available_otherwise_in_frame_fraction"
            ),
            "performance_evidence_schema_version": (
                PERFORMANCE_EVIDENCE_SCHEMA_VERSION
            ),
            "performance_evidence_policy": "observation_only_no_motion_effect",
            "production_validated": False,
            **(
                {
                    "audio_video_timing_schema_version": (
                        AUDIO_VIDEO_TIMING_SCHEMA_VERSION
                    ),
                    "audio_video_timing_policy": AUDIO_VIDEO_TIMING_POLICY,
                    "audio_video_timing_status": audio_video_timing["status"],
                    "audio_video_timing_consumed_by_retargeting": False,
                }
                if audio_video_timing is not None
                else {}
            ),
        },
        "retargeting": {
            "backend": retarget_backend,
            "geometry_calibrated": calibration_hash is not None,
            "calibration_hash": calibration_hash,
            "source_controls": len(source_names),
            "matched_source_controls": matched_controls,
            "matched_source_fraction": matched_controls / max(len(source_names), 1),
            "effective_matched_source_controls": (
                matched_controls
                - len(source_names & QUARANTINED_EXPRESSION_CONTROLS)
            ),
            "quarantined_expression_controls": sorted(
                source_names & QUARANTINED_EXPRESSION_CONTROLS
            ),
            "contact_source": performance.provenance.contact_source_method,
            "contact_calibration_hash": (
                performance.provenance.contact_calibration_hash
            ),
            "aperture_source": performance.provenance.aperture_source_method,
            "aperture_target_max_interocular": (
                SOURCE_APERTURE_MAX_TARGET_INTEROCULAR
            ),
            "aperture_subject_calibrated": False,
            "mouth_aperture_artist_edit": {
                "gain": mouth_aperture_gain,
                "authored": mouth_aperture_gain != 1.0,
                "author": mouth_aperture_author,
                "reason": mouth_aperture_reason,
                "corrected_frames": corrected_mouth_frames,
                "input_sha256": mouth_aperture_edit.input_sha256,
                "output_sha256": mouth_aperture_edit.output_sha256,
                "production_validated": False,
            },
            "subject_calibrated": False,
            "neutral_baseline_frame_indices": list(
                performance.provenance.baseline_frame_indices
            ),
            "neutral_baseline_method": performance.provenance.neutral_baseline_method,
            "neutral_baseline_validated": (
                performance.provenance.neutral_baseline_validated
            ),
            "neutral_baseline_correction_applied": (
                performance.provenance.neutral_baseline_correction_applied
            ),
        },
        "viewer": {
            "schema_version": "1.0",
            "status": viewer_status,
            "mode": viewer_mode,
            "model_artifact": "glb",
            "animation_clip": "autoanim" if viewer_status == "ready" else None,
            "clock_artifact": "viewer_media",
            "duration_s": duration,
            "coordinate_system": "+Y_up_+Z_forward_meters",
            "glb_covers_full_track": glb_covers_full_track,
            "reconstruction": viewer_reconstruction,
        },
        "oral_validation": {
            "schema_version": oral_control_report["schema_version"],
            "status": oral_control_report["status"],
            "all_control_frames_evaluated": oral_control_report["source"][
                "all_frames_evaluated"
            ],
            "tongue_control_active_frames": oral_control_report["control_evidence"][
                "tongue_control_active_frames"
            ],
            "isolated_tongue_geometry_active_frames": oral_control_report[
                "control_evidence"
            ]["isolated_tongue_geometry_active_frames"],
            "tongue_geometry_motion_frames": tongue_geometry_motion_frames,
            "tongue_motion_source": (
                "gnm_lower_face_basis_coupling_no_dedicated_source"
            ),
            "tongue_teeth_collision_risk_frames": tongue_teeth_risk_frames,
            "control_tongue_teeth_collision_risk_frames": (
                control_tongue_teeth_risk_frames
            ),
            "viewer_tongue_teeth_collision_risk_frames": (
                viewer_tongue_teeth_risk_frames
            ),
            "lip_order_inversion_risk_frames": lip_order_risk_frames,
            "control_lip_order_inversion_risk_frames": (
                control_lip_order_risk_frames
            ),
            "viewer_lip_order_inversion_risk_frames": (
                viewer_lip_order_risk_frames
            ),
            "lip_contact_target_attainment_fraction": oral_contact_attainment,
            "viewer_structural_reconstruction_validated": oral_glb_report["claims"][
                "structural_reconstruction_validated"
            ],
            "production_validated": False,
        },
        "metrics": {
            "face_presence_fraction": detected_count / performance.frame_count,
            "longest_missing_frames": _longest_missing(capture.detected),
            "effective_capture_quality_median": float(np.median(quality)),
            "effective_capture_quality_p05": float(np.percentile(quality, 5)),
            "landmark_in_frame_fraction_median": float(
                np.median(capture.tracking_quality)
            ),
            "landmark_in_frame_fraction_p05": float(
                np.percentile(capture.tracking_quality, 5)
            ),
            "lower_face_stationary_fraction": (
                float(np.mean(lower_delta <= 1e-7)) if len(lower_delta) else 0.0
            ),
            "head_rotation_max_degrees": float(
                np.rad2deg(np.max(np.linalg.norm(performance.rotations[:, 1], axis=1)))
            ),
            "translation_max": float(
                np.max(np.linalg.norm(performance.translation, axis=1))
            ),
            "gaze_rotation_max_degrees": float(
                np.rad2deg(
                    np.max(np.linalg.norm(performance.rotations[:, 2:4], axis=2))
                )
            ),
            "proxy_frames": proxy_frames,
            "proxy_video_start_ms": proxy_video_start * 1_000.0,
            "proxy_pts_max_error_ms": proxy_pts_error * 1_000.0,
            "retarget_bound_active_frames": region_bound_active_frames,
            "retarget_bound_active_fraction": (
                region_bound_active_frames / performance.frame_count
            ),
            "neutral_baseline_score": performance.provenance.neutral_baseline_score,
            "neutral_baseline_score_limit": (
                performance.provenance.neutral_baseline_score_limit
            ),
            "neutral_baseline_semantic_peak": (
                performance.provenance.neutral_baseline_semantic_peak
            ),
            "neutral_baseline_ambiguity_control_count": len(
                performance.provenance.neutral_baseline_ambiguity_controls
            ),
            "negative_baseline_residual_clipped_fraction": (
                performance.provenance.negative_baseline_residual_clipped_fraction
            ),
            "mouth_aperture_edit_corrected_frames": corrected_mouth_frames,
            "mouth_aperture_edit_protected_contact_frames": int(
                np.count_nonzero(mouth_aperture_edit.protected_contact)
            ),
            "mouth_aperture_edit_target_attained_fraction": (
                mouth_aperture_target_attainment(mouth_aperture_edit)
            ),
            "mouth_aperture_edit_introduced_lip_order_risk_frames": int(
                sum(
                    report.lip_order_inversion_introduced
                    for report in mouth_aperture_edit.reports
                )
            ),
            **source_motion,
            **final_output_retention,
            "mesh_finite": True,
        },
        "artifacts": {
            "capture": "capture.npz",
            "capture_jsonl": "capture.jsonl",
            "performance_evidence": "performance-evidence.json",
            "controls": "performance.npz",
            "controls_jsonl": "performance.jsonl",
            "glb": "performance.glb",
            "glb_mapping": "performance-glb-mapping.npz",
            "oral_validation": "oral-validation.json",
            "oral_glb_validation": "oral-glb-validation.json",
            "viewer_media": proxy.name,
            **(
                {"audio_video_timing": "audio-video-timing.json"}
                if audio_video_timing is not None
                else {}
            ),
            **retarget_artifacts,
        },
        "warnings": warnings,
    }
    write_json(output / "result.json", result)
    return result
