"""Deliberate native-GNM facial range-of-motion showcase.

The sequence follows Google's checked-in demos: semantic expressions are sampled
from the released CVAE, while the wide-mouth and tongue poses start from the
controls shown in ``gnm_head_demo.gif``. A volumetric Rust XPBD pass gives the
extended tongue and lower lip two-way soft contact before export.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import trimesh

from .animated_gltf import AnimatedGLBExport, export_animated_gnm_glb
from .gnm_adapter import GNMAdapter
from .mesh_contact import count_triangle_intersection_pairs
from .mouth_geometry import (
    discover_mouth_boundary,
    measure_mouth_boundary,
)
from .oral_validation import OralValidationResult, validate_oral_frames
from .render import MeshRenderer
from .rig import ControlRig
from .semantic_decoder import ExpressionDecoder
from .serialization import write_json
from .soft_contact import (
    SoftContactConfig,
    simulate_gnm_tongue_lip_soft_contact,
)


SCHEMA_VERSION = "autoanim.expression-showcase/1.4"
_GOOGLE_WIDE_MOUTH = np.asarray((-2.6, 1.0, -2.1), dtype=np.float32)
_GOOGLE_TONGUE_FORWARD = np.asarray((0.7, -1.7, 0.0, 0.0), dtype=np.float32)
_EXTENDED_TONGUE = np.asarray((0.739, 2.18, 0.0, 0.0), dtype=np.float32)
_PRESSING_TONGUE = _EXTENDED_TONGUE.copy()
_TONGUE_CONTACT_CORRECTIVE_MODE = 203
_TONGUE_CONTACT_CORRECTIVE_AMOUNT = np.float32(-0.82)
_TONGUE_PRESS_CORRECTIVE_AMOUNT = _TONGUE_CONTACT_CORRECTIVE_AMOUNT


@dataclass(frozen=True, slots=True)
class ShowcaseKeyframe:
    time_seconds: float
    label: str
    expression: np.ndarray
    rotations: np.ndarray
    mouth_opening: float


@dataclass(frozen=True, slots=True)
class ShowcaseTrack:
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray
    timestamps: np.ndarray
    mouth_opening: np.ndarray
    labels: tuple[str, ...]
    keyframes: tuple[ShowcaseKeyframe, ...]
    fps: int


@dataclass(frozen=True, slots=True)
class ShowcaseBuild:
    track: ShowcaseTrack
    target_frames: np.ndarray
    frames: np.ndarray
    oral_validation: OralValidationResult
    glb_export: AnimatedGLBExport
    soft_contact_report: dict[str, Any]
    timeline_document: dict[str, Any]


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value).copy()
    output.setflags(write=False)
    return output


def _head_rotation(
    *,
    pitch_degrees: float = 0.0,
    yaw_degrees: float = 0.0,
    roll_degrees: float = 0.0,
) -> np.ndarray:
    rotations = np.zeros((4, 3), dtype=np.float32)
    rotations[1] = np.deg2rad(
        np.asarray((pitch_degrees, yaw_degrees, roll_degrees), dtype=np.float32)
    )
    return rotations


def _pose(
    rig: ControlRig,
    *,
    semantic_expression: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    value = np.zeros(rig.adapter.expression_dim, dtype=np.float32)
    if semantic_expression is not None:
        value[:] = rig.decoder.sample(
            semantic_expression,
            rng=np.random.default_rng(seed),
        )
    return np.clip(value, -3.0, 3.0).astype(np.float32)


def _keyframes(rig: ControlRig) -> tuple[ShowcaseKeyframe, ...]:
    neutral = _pose(rig)
    smile = _pose(rig, semantic_expression="happy", seed=0)
    wink = _pose(rig, semantic_expression="wink_left", seed=2)
    frown = _pose(rig, semantic_expression="corners_down", seed=1)
    disgust = _pose(rig, semantic_expression="disgust", seed=0)
    surprise = _pose(rig, semantic_expression="surprise", seed=2)

    # These are the exact native controls visible in Google's checked-in
    # gnm_head_demo.gif. Keeping them in parameter space is the important bit:
    # GNM evaluates skin, lips, mouth sock, teeth, and tongue coherently.
    wide = neutral.copy()
    wide[200:203] = _GOOGLE_WIDE_MOUTH
    tongue = wide.copy()
    tongue[350:354] = _GOOGLE_TONGUE_FORWARD
    extended_tongue = wide.copy()
    # This target enters the solver's 1.0 mm contact shell without crossing the
    # lip. The physics pass, rather than a large coefficient-space gap, supplies
    # the final supported separation and two-way tissue deformation.
    extended_tongue[_TONGUE_CONTACT_CORRECTIVE_MODE] = (
        _TONGUE_CONTACT_CORRECTIVE_AMOUNT
    )
    extended_tongue[350:354] = _EXTENDED_TONGUE
    pressing_tongue = wide.copy()
    pressing_tongue[_TONGUE_CONTACT_CORRECTIVE_MODE] = (
        _TONGUE_PRESS_CORRECTIVE_AMOUNT
    )
    pressing_tongue[350:354] = _PRESSING_TONGUE

    entries = (
        (0.0, "Neutral", neutral, 0.0, {}),
        (1.0, "Neutral hold", neutral, 0.0, {}),
        (2.2, "CVAE happy sample", smile, 0.30, {"yaw_degrees": -4.0, "roll_degrees": -2.0}),
        (3.2, "CVAE happy hold", smile, 0.30, {"yaw_degrees": -4.0, "roll_degrees": -2.0}),
        (4.0, "CVAE left wink sample", wink, 0.25, {"yaw_degrees": -7.0, "roll_degrees": -3.0}),
        (4.8, "Reset", neutral, 0.0, {}),
        (5.8, "CVAE corners down sample", frown, 0.0, {"pitch_degrees": -3.0, "yaw_degrees": 3.0}),
        (6.8, "CVAE disgust sample", disgust, 0.25, {"yaw_degrees": 5.0, "roll_degrees": 2.0}),
        (7.6, "Reset", neutral, 0.0, {}),
        (8.8, "CVAE surprise sample", surprise, 0.72, {"pitch_degrees": 2.0}),
        (9.8, "Google native wide mouth", wide, 1.0, {"pitch_degrees": 3.0}),
        (10.7, "Google native wide mouth hold", wide, 1.0, {"pitch_degrees": 3.0}),
        (11.6, "Google native tongue forward", tongue, 1.0, {"pitch_degrees": 2.0}),
        (12.4, "Tongue settles on lower lip", extended_tongue, 1.0, {"pitch_degrees": 2.0}),
        (13.2, "Tongue supported by soft tissue", pressing_tongue, 1.0, {"pitch_degrees": 2.0}),
        (13.8, "Tongue pressure releases", extended_tongue, 1.0, {"pitch_degrees": 2.0}),
        (14.2, "Collision-safe tongue recovery", tongue, 1.0, {"pitch_degrees": 2.0}),
        (14.6, "Open-mouth recovery", wide, 1.0, {"pitch_degrees": 2.0}),
        (15.0, "Neutral", neutral, 0.0, {}),
    )
    return tuple(
        ShowcaseKeyframe(
            time_seconds=float(time_seconds),
            label=label,
            expression=_readonly(expression),
            rotations=_readonly(_head_rotation(**rotation)),
            mouth_opening=float(mouth_opening),
        )
        for time_seconds, label, expression, mouth_opening, rotation in entries
    )


def create_showcase_track(
    adapter: GNMAdapter,
    decoder: ExpressionDecoder,
    *,
    fps: int = 30,
) -> ShowcaseTrack:
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 24 or fps > 120:
        raise ValueError("showcase fps must be an integer between 24 and 120")
    rig = ControlRig(adapter, decoder)
    keyframes = _keyframes(rig)
    frame_count = int(round(keyframes[-1].time_seconds * fps)) + 1
    timestamps = np.arange(frame_count, dtype=np.float64) / float(fps)
    expression = np.empty((frame_count, adapter.expression_dim), dtype=np.float32)
    rotations = np.empty((frame_count, adapter.model.num_joints, 3), dtype=np.float32)
    mouth_opening = np.empty(frame_count, dtype=np.float32)
    labels: list[str] = []

    segment = 0
    for index, timestamp in enumerate(timestamps):
        while (
            segment + 1 < len(keyframes) - 1
            and timestamp > keyframes[segment + 1].time_seconds
        ):
            segment += 1
        left = keyframes[segment]
        right = keyframes[min(segment + 1, len(keyframes) - 1)]
        duration = right.time_seconds - left.time_seconds
        phase = 1.0 if duration <= 0.0 else (timestamp - left.time_seconds) / duration
        phase = min(1.0, max(0.0, float(phase)))
        eased = 0.5 - 0.5 * math.cos(math.pi * phase)
        expression[index] = (
            np.float32(1.0 - eased) * left.expression
            + np.float32(eased) * right.expression
        )
        np.clip(expression[index], -3.0, 3.0, out=expression[index])
        rotations[index] = (
            np.float32(1.0 - eased) * left.rotations
            + np.float32(eased) * right.rotations
        )
        mouth_opening[index] = (
            np.float32(1.0 - eased) * left.mouth_opening
            + np.float32(eased) * right.mouth_opening
        )
        labels.append(left.label if phase < 0.5 else right.label)

    return ShowcaseTrack(
        expression=_readonly(expression),
        rotations=_readonly(rotations),
        translation=_readonly(np.zeros((frame_count, 3), dtype=np.float32)),
        timestamps=_readonly(timestamps),
        mouth_opening=_readonly(mouth_opening),
        labels=tuple(labels),
        keyframes=keyframes,
        fps=fps,
    )


def evaluate_showcase_frames(
    adapter: GNMAdapter,
    track: ShowcaseTrack,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    output = np.empty(
        (len(track.timestamps), adapter.model.num_vertices, 3),
        dtype=np.float32,
    )
    for start in range(0, len(output), batch_size):
        stop = min(start + batch_size, len(output))
        output[start:stop] = adapter.mesh(
            expression=track.expression[start:stop],
            rotations=track.rotations[start:stop],
            translation=track.translation[start:stop],
        )
    return output


def _bake_soft_contact_corrective(
    target_frames: np.ndarray,
    raw_soft_frames: np.ndarray,
    timestamps: np.ndarray,
    *,
    anchor_time_seconds: float = 13.2,
    engage_time_seconds: float = 11.6,
    settled_time_seconds: float = 12.4,
    release_time_seconds: float = 13.8,
    disengaged_time_seconds: float = 14.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bake a verified native solve into one smooth oral corrective basis."""
    anchor = int(np.argmin(np.abs(timestamps - anchor_time_seconds)))
    residual = np.asarray(
        raw_soft_frames[anchor] - target_frames[anchor],
        dtype=np.float32,
    )
    weights = np.zeros(len(timestamps), dtype=np.float32)

    def smoothstep(value: np.ndarray) -> np.ndarray:
        phase = np.clip(value, 0.0, 1.0)
        return np.asarray(
            phase * phase * (3.0 - 2.0 * phase),
            dtype=np.float32,
        )

    engaging = (timestamps > engage_time_seconds) & (
        timestamps < settled_time_seconds
    )
    weights[engaging] = smoothstep(
        (timestamps[engaging] - engage_time_seconds)
        / (settled_time_seconds - engage_time_seconds)
    )
    weights[
        (timestamps >= settled_time_seconds) & (timestamps <= release_time_seconds)
    ] = 1.0
    releasing = (timestamps > release_time_seconds) & (
        timestamps < disengaged_time_seconds
    )
    weights[releasing] = 1.0 - smoothstep(
        (timestamps[releasing] - release_time_seconds)
        / (disengaged_time_seconds - release_time_seconds)
    )
    frames = np.ascontiguousarray(
        target_frames + weights[:, None, None] * residual[None, :, :],
        dtype=np.float32,
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(residual).tobytes())
    digest.update(np.ascontiguousarray(weights).tobytes())
    return frames, {
        "schema_version": "autoanim.soft-contact-bake/1.0",
        "source": "native_rust_xpbd_anchor_residual",
        "corrective_rank": 1,
        "anchor_time_seconds": anchor_time_seconds,
        "engage_time_seconds": engage_time_seconds,
        "settled_time_seconds": settled_time_seconds,
        "release_time_seconds": release_time_seconds,
        "disengaged_time_seconds": disengaged_time_seconds,
        "active_frame_count": int(np.count_nonzero(weights > 0.0)),
        "maximum_weight": float(np.max(weights, initial=0.0)),
        "maximum_corrective_displacement_mm": float(
            np.max(np.linalg.norm(residual, axis=1), initial=0.0) * 1_000.0
        ),
        "sha256": digest.hexdigest(),
    }


def _render_evaluated_video(
    frames: np.ndarray,
    adapter: GNMAdapter,
    output_path: Path,
    *,
    fps: int,
) -> None:
    renderer = MeshRenderer(adapter, include_internal_anatomy=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        "640x640",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-metadata",
        "creation_time=",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(renderer.render(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(
            f"Showcase preview render failed: {stderr.decode(errors='replace')}"
        )


def build_expression_showcase(
    output_dir: str | Path,
    *,
    texture_path: str | Path | None = None,
    fps: int = 30,
) -> ShowcaseBuild:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = GNMAdapter()
    decoder = ExpressionDecoder(
        "gnm/shape/data/semantic_sampler/expression_decoder_model.h5"
    )
    track = create_showcase_track(adapter, decoder, fps=fps)
    target_frames = evaluate_showcase_frames(adapter, track)
    raw_soft_frames, soft_topology, soft_contact_report = (
        simulate_gnm_tongue_lip_soft_contact(
            adapter,
            target_frames,
            config=SoftContactConfig(frames_per_second=float(track.fps)),
        )
    )
    frames, soft_contact_bake = _bake_soft_contact_corrective(
        target_frames,
        raw_soft_frames,
        track.timestamps,
    )
    soft_contact_report = {
        **soft_contact_report,
        "viewer_bake": soft_contact_bake,
    }
    oral = validate_oral_frames(
        frames,
        adapter=adapter,
        timestamps=track.timestamps,
        source_kind="deliberate_expression_range_showcase",
    )
    export = export_animated_gnm_glb(
        output / "animation.glb",
        adapter,
        frames,
        track.timestamps,
        # The physics cache is baked into one smooth oral corrective basis.
        max_targets=48,
        mapping_path=output / "animation-glb-mapping.npz",
        texture_path=texture_path,
    )
    np.savez_compressed(
        output / "controls.npz",
        identity=np.zeros(adapter.identity_dim, dtype=np.float32),
        expression=track.expression,
        rotations=track.rotations,
        translation=track.translation,
        timestamps=track.timestamps,
        mouth_opening=track.mouth_opening,
        labels=np.asarray(track.labels),
    )
    _render_evaluated_video(
        frames,
        adapter,
        output / "preview.mp4",
        fps=track.fps,
    )

    topology = discover_mouth_boundary(adapter)
    keyframe_metrics: list[dict[str, Any]] = []
    tongue_indices = np.flatnonzero(adapter.vertex_group("tongue") > 0.5)
    tongue_triangles = np.asarray(
        adapter.model.triangles_group("tongue"),
        dtype=np.int32,
    )
    upper_lip_triangles = np.asarray(
        adapter.model.triangles_group("upper_lip"),
        dtype=np.int32,
    )
    lower_lip_triangles = np.asarray(
        adapter.model.triangles_group("lower_lip"),
        dtype=np.int32,
    )
    neutral_mesh = adapter.mesh()
    triangles = adapter.triangles
    neutral_normals = np.cross(
        neutral_mesh[triangles[:, 1]] - neutral_mesh[triangles[:, 0]],
        neutral_mesh[triangles[:, 2]] - neutral_mesh[triangles[:, 0]],
    )
    for keyframe in track.keyframes:
        frame_index = int(round(keyframe.time_seconds * track.fps))
        mesh = frames[frame_index]
        target_mesh = target_frames[frame_index]
        local_mesh = mesh[soft_topology.global_vertex_indices]
        local_target_mesh = target_mesh[soft_topology.global_vertex_indices]
        contact_points = soft_topology.contact_pairs[:, 0]
        contact_triangles = soft_topology.contact_pairs[:, 1:4]

        def candidate_distance(value: np.ndarray) -> float:
            points = value[contact_points]
            closest = trimesh.triangles.closest_point(
                value[contact_triangles],
                points,
            )
            return float(
                np.min(np.linalg.norm(points - closest, axis=1), initial=np.inf)
            )
        landmarks = np.sum(
            mesh[adapter.landmark_indices]
            * adapter.landmark_weights[..., None],
            axis=-2,
        )
        mouth = measure_mouth_boundary(mesh, landmarks, topology)
        normals = np.cross(
            mesh[triangles[:, 1]] - mesh[triangles[:, 0]],
            mesh[triangles[:, 2]] - mesh[triangles[:, 0]],
        )
        reversed_triangles = int(
            np.count_nonzero(np.einsum("ij,ij->i", neutral_normals, normals) <= 0.0)
        )
        keyframe_metrics.append(
            {
                "time_seconds": keyframe.time_seconds,
                "label": keyframe.label,
                "mouth_opening_reference": keyframe.mouth_opening,
                "mouth_central_gap_signed_mm": (
                    mouth.signed_central_gap_m * 1_000.0
                ),
                "mouth_opening_area_mm2": mouth.opening_area_m2 * 1_000_000.0,
                "maximum_tongue_vertex_displacement_from_neutral_mm": float(
                    np.max(
                        np.linalg.norm(
                            mesh[tongue_indices] - neutral_mesh[tongue_indices],
                            axis=1,
                        ),
                        initial=0.0,
                    )
                    * 1_000.0
                ),
                "reversed_triangles_against_neutral": reversed_triangles,
                "maximum_expression_coefficient": float(
                    np.max(np.abs(keyframe.expression), initial=0.0)
                ),
                "tongue_lip_triangle_intersection_pairs": (
                    count_triangle_intersection_pairs(
                        mesh,
                        tongue_triangles,
                        upper_lip_triangles,
                    )
                    + count_triangle_intersection_pairs(
                        mesh,
                        tongue_triangles,
                        lower_lip_triangles,
                    )
                ),
                "target_tongue_lip_triangle_intersection_pairs": (
                    count_triangle_intersection_pairs(
                        target_mesh,
                        tongue_triangles,
                        upper_lip_triangles,
                    )
                    + count_triangle_intersection_pairs(
                        target_mesh,
                        tongue_triangles,
                        lower_lip_triangles,
                    )
                ),
                "target_minimum_candidate_contact_distance_mm": (
                    candidate_distance(local_target_mesh) * 1_000.0
                ),
                "solved_minimum_candidate_contact_distance_mm": (
                    candidate_distance(local_mesh) * 1_000.0
                ),
            }
        )
    local_target = target_frames[:, soft_topology.global_vertex_indices]
    local_solved = frames[:, soft_topology.global_vertex_indices]
    soft_displacement = np.linalg.norm(local_solved - local_target, axis=2)
    tongue_soft_displacement = soft_displacement[
        :, : soft_topology.tongue_vertex_count
    ]
    lip_soft_displacement = soft_displacement[
        :,
        soft_topology.tongue_vertex_count : (
            soft_topology.tongue_vertex_count
            + soft_topology.lower_lip_vertex_count
        ),
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Native GNM range-of-motion diagnostic reproducing Google's "
            "semantic sampler and checked-in manual mouth/tongue controls."
        ),
        "fps": track.fps,
        "frame_count": len(track.timestamps),
        "duration_seconds": float(track.timestamps[-1]),
        "keyframes": keyframe_metrics,
        "metrics": {
            "maximum_absolute_mouth_central_gap_mm": max(
                abs(entry["mouth_central_gap_signed_mm"])
                for entry in keyframe_metrics
            ),
            "maximum_mouth_opening_area_mm2": max(
                entry["mouth_opening_area_mm2"] for entry in keyframe_metrics
            ),
            "maximum_tongue_vertex_displacement_from_neutral_mm": max(
                entry["maximum_tongue_vertex_displacement_from_neutral_mm"]
                for entry in keyframe_metrics
            ),
            "maximum_reversed_triangles_against_neutral": max(
                entry["reversed_triangles_against_neutral"]
                for entry in keyframe_metrics
            ),
            "legacy_topology_signed_gap_negative_frames": int(
                np.count_nonzero(oral.lip_gap_interocular < 0.0)
            ),
            "legacy_sparse_or_topology_lip_order_risk_frames": int(
                np.count_nonzero(oral.lip_order_inversion_risk_frames)
            ),
            "tongue_teeth_collision_risk_frames": int(
                np.count_nonzero(oral.tongue_teeth_collision_risk_frames)
            ),
            "maximum_keyframe_tongue_lip_triangle_intersection_pairs": max(
                entry["tongue_lip_triangle_intersection_pairs"]
                for entry in keyframe_metrics
            ),
            "maximum_target_keyframe_tongue_lip_triangle_intersection_pairs": max(
                entry["target_tongue_lip_triangle_intersection_pairs"]
                for entry in keyframe_metrics
            ),
            "minimum_target_keyframe_candidate_contact_distance_mm": min(
                entry["target_minimum_candidate_contact_distance_mm"]
                for entry in keyframe_metrics
            ),
            "minimum_solved_keyframe_candidate_contact_distance_mm": min(
                entry["solved_minimum_candidate_contact_distance_mm"]
                for entry in keyframe_metrics
            ),
            "maximum_soft_contact_displacement_mm": float(
                np.max(soft_displacement, initial=0.0) * 1_000.0
            ),
            "maximum_tongue_soft_contact_displacement_mm": float(
                np.max(tongue_soft_displacement, initial=0.0) * 1_000.0
            ),
            "maximum_lower_lip_soft_contact_displacement_mm": float(
                np.max(lip_soft_displacement, initial=0.0) * 1_000.0
            ),
        },
        "soft_contact": soft_contact_report,
        "limitations": [
            (
                "GNM Head v3 has no jaw joint or named production controls; "
                "the native mouth pose is a combination of PCA coefficients."
            ),
            (
                "The existing topology signed-gap validator assigns the two "
                "lip-boundary paths by neutral height, opposite to GNM's "
                "upper_lip/lower_lip vertex groups. Its signed-order warning "
                "is retained as legacy diagnostic evidence, not used to alter "
                "this native showcase."
            ),
            (
                "The GNM tongue is tetrahedralized and solved as a volume with "
                "two-way lower- and upper-lip contact. The tetrahedra and target "
                "attachments are not anatomically calibrated muscle/FEM data."
            ),
            (
                "This controlled sequence uses conservative discrete contact "
                "and collision-safe target paths. It does not claim continuous "
                "collision detection for arbitrary fast input."
            ),
        ],
        "glb": {
            "morph_rank": export.rank,
            "oral_corrective_targets": export.oral_corrective_targets,
            "mesh_p95_error_mm": export.mesh_p95_mm,
            "mesh_max_error_mm": export.mesh_max_mm,
            "landmark_p95_error_mm": export.landmark_p95_mm,
            "landmark_max_error_mm": export.landmark_max_mm,
        },
    }
    write_json(output / "showcase-timeline.json", document)
    write_json(output / "oral-validation.json", oral.report)
    write_json(output / "soft-contact-report.json", soft_contact_report)
    return ShowcaseBuild(
        track=track,
        target_frames=target_frames,
        frames=frames,
        oral_validation=oral,
        glb_export=export,
        soft_contact_report=soft_contact_report,
        timeline_document=document,
    )


__all__ = [
    "SCHEMA_VERSION",
    "ShowcaseBuild",
    "ShowcaseKeyframe",
    "ShowcaseTrack",
    "build_expression_showcase",
    "create_showcase_track",
    "evaluate_showcase_frames",
]
