"""End-to-end monocular video performance capture for GNM."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .a2f import ARKitGNMRetargeter
from .animation import calibrate_lip_contact
from .animated_gltf import AnimationCompressionError, export_animated_gnm_glb
from .calibrated_retarget import CalibratedRetargetError, CalibratedRetargeter
from .errors import AutoAnimError
from .gltf_export import export_gnm_glb
from .gnm_adapter import GNMAdapter
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json
from .video_capture import (
    MONOCULAR_SCALE_CAVEAT,
    capture_video,
    probe_video,
    serialize_capture,
)
from .video_retarget import (
    FAST_CONTACT_CONTROLS,
    NEUTRAL_CALIBRATION_CAVEAT,
    QUARANTINED_EXPRESSION_CONTROLS,
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


def run_video_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path,
    identity: np.ndarray | None = None,
    a2f_asset_dir: str | Path | None = None,
) -> dict:
    """Track a real video, retarget it, and package synchronized 3D playback."""

    source = Path(input_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = capture_video(source, model_path)
    detected_count = int(np.count_nonzero(capture.detected))
    if detected_count == 0:
        raise AutoAnimError("FACE_NOT_FOUND", "No face was detected in the video")
    serialize_capture(output, capture)

    adapter = GNMAdapter()
    decoder = ExpressionDecoder("gnm/shape/data/semantic_sampler/expression_decoder_model.h5")
    contact_rig = ControlRig(adapter, decoder)
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
        retarget_backend = "geometry_calibrated_dense_contact_v2"
        retarget_caveat = VIDEO_CALIBRATED_RETARGET_CAVEAT
        retarget_artifacts["retarget_calibration"] = "retarget_calibration.npz"
    else:
        retargeter = ARKitGNMRetargeter(contact_rig)
        matched_controls = len(
            source_names
            & {source for rule in retargeter.rules for source in rule.sources}
        )
        retarget_backend = "semantic_prototype_contact_v2_fallback"
        retarget_caveat = VIDEO_SEMANTIC_RETARGET_CAVEAT
    performance = retarget_capture(
        capture,
        retargeter,
        identity=identity,
        retarget_caveats=(retarget_caveat,),
        contact_rig=contact_rig,
        lip_contact_calibration=lip_contact_calibration,
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
        try:
            exported = export_animated_gnm_glb(
                output / "performance.glb",
                adapter,
                frames,
                performance.timestamps_seconds,
                mapping_path=output / "performance-glb-mapping.npz",
            )
            viewer_reconstruction = {
                "expression_pose_rank": exported.rank,
                "validation_scope": "all_frames",
                "mesh_p95_mm": exported.mesh_p95_mm,
                "mesh_max_mm": exported.mesh_max_mm,
                "landmark_p95_mm": exported.landmark_p95_mm,
                "landmark_max_mm": exported.landmark_max_mm,
            }
        except AnimationCompressionError as exc:
            export_gnm_glb(
                output / "performance.glb",
                adapter,
                frames[0],
                mapping_path=output / "performance-glb-mapping.npz",
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
        export_gnm_glb(
            output / "performance.glb",
            adapter,
            first,
            mapping_path=output / "performance-glb-mapping.npz",
        )
        viewer_status = "static_only"
        viewer_mode = "static"
        warnings.append("VIEWER_TRACK_TOO_LONG")
        viewer_reconstruction = {
            "validation_scope": "not_run",
            "reason": f"Interactive morph export is limited to {MAX_INTERACTIVE_FRAMES} frames",
        }

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
    result = {
        "kind": "video_performance",
        "status": "succeeded",
        "model": {
            "gnm_version": "3.0",
            "identity_dim": adapter.identity_dim,
            "expression_dim": adapter.expression_dim,
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
            "production_validated": False,
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
            "reconstruction": viewer_reconstruction,
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
            **source_motion,
            **final_output_retention,
            "mesh_finite": True,
        },
        "artifacts": {
            "capture": "capture.npz",
            "capture_jsonl": "capture.jsonl",
            "controls": "performance.npz",
            "controls_jsonl": "performance.jsonl",
            "glb": "performance.glb",
            "glb_mapping": "performance-glb-mapping.npz",
            "viewer_media": proxy.name,
            **retarget_artifacts,
        },
        "warnings": warnings,
    }
    write_json(output / "result.json", result)
    return result
