"""Connected MPFB-body + GNM-head preview export.

This exporter consumes a sealed :mod:`body_binding` artifact. It emits one
mesh, one skin and one animation, with the provider head removed and a
topological zipper strip between the two exact plane-cut loops. The artifact
is deliberately preview-only until its measured blockers are cleared.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .animated_gltf import (
    _LIP_NORMAL_FRAME_P95_LIMIT_DEGREES,
    _LIP_NORMAL_MAX_LIMIT_DEGREES,
    _MOUTH_BOUNDARY_NORMAL_MAX_LIMIT_DEGREES,
    _NORMAL_FRAME_P95_LIMIT_DEGREES,
    AnimationCompressionError,
    _fit_morph_normals,
    factor_vertex_animation,
)
from .body import BodyTrack, skeleton_for_track, validate_body_track
from .body_binding import (
    GNM_TO_AUTOANIM_BASIS,
    array_sha256,
    interpolate_clipped_attribute,
    load_gnm_body_binding,
)
from .body_export import (
    _Builder,
    _atomic_bytes,
    _canonical_arm_bind_alignment,
    _compose_rest_and_delta,
    _glb_bytes,
)
from .body_provider import load_and_validate_body_asset, sha256_file
from .gnm_adapter import GNMAdapter
from .mouth_geometry import discover_mouth_boundary
from .serialization import write_json, write_npz


UNIFIED_GLTF_SCHEMA_VERSION = "autoanim.unified-character-gltf/1.0"
FOUR_WEIGHT_RMS_GATE_M = 0.00010
FOUR_WEIGHT_MAX_GATE_M = 0.00050


class UnifiedGLTFError(ValueError):
    """A connected character export failed a sealed validation gate."""


def _composition_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    dtype = array.dtype.str.encode("ascii")
    shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    digest.update(len(dtype).to_bytes(4, "little"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(4, "little"))
    digest.update(shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class UnifiedCharacterGLBExport:
    path: Path
    report_path: Path
    mapping_path: Path
    frame_count: int
    vertex_count: int
    triangle_count: int
    morph_target_count: int
    four_weight_rms_mm: float
    four_weight_max_mm: float
    publish_ready: bool
    publish_blockers: tuple[str, ...]


def _zipper_loops(
    first_loop: np.ndarray,
    second_loop: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Triangulate two ordered unequal closed loops without adding vertices."""

    first = np.asarray(first_loop, dtype=np.int64)
    second = np.asarray(second_loop, dtype=np.int64)
    points = np.asarray(vertices, dtype=np.float64)
    if (
        first.ndim != 1
        or second.ndim != 1
        or len(first) < 3
        or len(second) < 3
        or len(np.unique(first)) != len(first)
        or len(np.unique(second)) != len(second)
    ):
        raise UnifiedGLTFError("Zipper loops are invalid")
    triangles: list[tuple[int, int, int]] = []
    first_step = 0
    second_step = 0
    while first_step < len(first) or second_step < len(second):
        next_first = (first_step + 1) / len(first)
        next_second = (second_step + 1) / len(second)
        if first_step < len(first) and (
            second_step == len(second) or next_first <= next_second
        ):
            triangles.append(
                (
                    int(first[first_step % len(first)]),
                    int(first[(first_step + 1) % len(first)]),
                    int(second[second_step % len(second)]),
                )
            )
            first_step += 1
        else:
            triangles.append(
                (
                    int(first[first_step % len(first)]),
                    int(second[(second_step + 1) % len(second)]),
                    int(second[second_step % len(second)]),
                )
            )
            second_step += 1
    faces = np.asarray(triangles, dtype=np.int32)
    p0, p1, p2 = (points[faces[:, corner]] for corner in range(3))
    normals = np.cross(p1 - p0, p2 - p0)
    centroids = (p0 + p1 + p2) / 3.0
    center = np.mean(points[np.concatenate((first, second))], axis=0)
    radial = centroids - center
    radial[:, 1] = 0.0
    if float(np.median(np.sum(normals * radial, axis=1))) < 0.0:
        faces = faces[:, [0, 2, 1]]
        p0, p1, p2 = (points[faces[:, corner]] for corner in range(3))
    double_area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    if np.any(double_area <= 1.0e-10):
        raise UnifiedGLTFError("Zipper generated a degenerate neck triangle")
    return faces


def _dense_skin_weights(
    joint_indices: np.ndarray, joint_weights: np.ndarray, joint_count: int
) -> np.ndarray:
    indices = np.asarray(joint_indices, dtype=np.int64)
    weights = np.asarray(joint_weights, dtype=np.float64)
    dense = np.zeros((len(indices), joint_count), dtype=np.float64)
    rows = np.arange(len(indices))
    for slot in range(indices.shape[1]):
        np.add.at(dense, (rows, indices[:, slot]), weights[:, slot])
    sums = np.sum(dense, axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        raise UnifiedGLTFError("Body skin contains an unweighted vertex")
    return dense / sums


def _skin_matrices(
    local_rest: np.ndarray,
    inverse_bind: np.ndarray,
    parents: np.ndarray,
    track: BodyTrack,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # D3: the track's OWN skeleton and its OWN rest, so the unified character's joints
    # land where `forward_kinematics_positions` puts them, exactly as `body_export` does.
    target_skeleton = skeleton_for_track(track)
    translations, rotations = _compose_rest_and_delta(
        local_rest,
        track.local_rotations_xyzw,
        parents,
        world_bind_alignment_xyzw=_canonical_arm_bind_alignment(
            local_rest, parents, skeleton=target_skeleton
        ),
        rest_translations_m=track.rest_translations_m,
    )
    frame_count, joint_count = rotations.shape[:2]
    local = np.repeat(
        np.eye(4, dtype=np.float64)[None, None, :, :],
        frame_count * joint_count,
        axis=0,
    ).reshape(frame_count, joint_count, 4, 4)
    local[:, :, :3, :3] = Rotation.from_quat(rotations.reshape(-1, 4)).as_matrix().reshape(
        frame_count, joint_count, 3, 3
    )
    local[:, :, :3, 3] = translations[None, :, :]
    local[:, 0, :3, 3] = translations[0] + track.root_translation_m
    world = np.empty_like(local)
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        world[:, index] = (
            local[:, index] if parent == -1 else world[:, parent] @ local[:, index]
        )
    return (
        world @ np.asarray(inverse_bind, dtype=np.float64)[None, :, :, :],
        translations,
        rotations,
    )


def _trajectory_project_four_weights(
    vertices: np.ndarray,
    dense_weights: np.ndarray,
    skin_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit four influences against the exact motion, preserving a measured error."""

    points = np.column_stack(
        (np.asarray(vertices, dtype=np.float64), np.ones(len(vertices)))
    )
    dense = np.asarray(dense_weights, dtype=np.float64)
    matrices = np.asarray(skin_matrices, dtype=np.float64)
    frame_count = len(matrices)
    output_indices = np.zeros((len(points), 4), dtype=np.uint16)
    output_weights = np.zeros((len(points), 4), dtype=np.float32)
    errors = np.zeros((frame_count, len(points)), dtype=np.float64)
    for vertex_index, point in enumerate(points):
        candidates = np.flatnonzero(dense[vertex_index] > 1.0e-8)
        trajectories = np.stack(
            [
                (matrices[:, joint] @ point)[:, :3].reshape(-1)
                for joint in candidates
            ],
            axis=1,
        )
        target = trajectories @ dense[vertex_index, candidates]
        if len(candidates) <= 4:
            selected = np.arange(len(candidates))
            fitted = dense[vertex_index, candidates]
        else:
            best: tuple[float, np.ndarray, np.ndarray] | None = None
            for combination in itertools.combinations(range(len(candidates)), 4):
                selected_candidate = np.asarray(combination, dtype=np.int64)
                design = trajectories[:, selected_candidate]
                # Equality-constrained least squares through a strongly weighted
                # sum-to-one row; negative candidates are invalid.
                augmented = np.vstack((design, np.full((1, 4), 100.0)))
                expected = np.concatenate((target, np.asarray((100.0,))))
                fitted_candidate = np.linalg.lstsq(
                    augmented, expected, rcond=None
                )[0]
                if float(np.min(fitted_candidate)) < -1.0e-5:
                    continue
                fitted_candidate = np.maximum(fitted_candidate, 0.0)
                fitted_candidate /= np.sum(fitted_candidate)
                residual = (
                    design @ fitted_candidate - target
                ).reshape(frame_count, 3)
                maximum = float(np.max(np.linalg.norm(residual, axis=1)))
                if best is None or maximum < best[0]:
                    best = (maximum, selected_candidate, fitted_candidate)
            if best is None:
                selected = np.argsort(
                    -dense[vertex_index, candidates], kind="stable"
                )[:4]
                fitted = dense[vertex_index, candidates[selected]]
                fitted /= np.sum(fitted)
            else:
                _, selected, fitted = best
        selected_joints = candidates[selected]
        output_indices[vertex_index, : len(selected_joints)] = selected_joints
        output_weights[vertex_index, : len(selected_joints)] = fitted
        predicted = np.stack(
            [(matrices[:, joint] @ point)[:, :3] for joint in selected_joints],
            axis=1,
        )
        exact = target.reshape(frame_count, 3)
        errors[:, vertex_index] = np.linalg.norm(
            np.sum(predicted * fitted[None, :, None], axis=1) - exact,
            axis=1,
        )
    metrics = {
        "rms_m": float(np.sqrt(np.mean(errors * errors))),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }
    return output_indices, output_weights, metrics


def _boundary_edge_count(triangles: np.ndarray, selected_vertices: np.ndarray) -> int:
    faces = np.asarray(triangles, dtype=np.int64)
    edges = np.sort(
        np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    selected = set(np.asarray(selected_vertices, dtype=np.int64).tolist())
    return int(
        np.count_nonzero(
            [
                count == 1 and int(edge[0]) in selected and int(edge[1]) in selected
                for edge, count in zip(unique, counts, strict=True)
            ]
        )
    )


def _load_and_validate_composition(
    manifest_path: str | Path,
    arrays_path: str | Path,
    *,
    body_asset_path: str | Path,
    body_manifest_path: str | Path,
    face_performance_path: str | Path,
    performance: dict[str, np.ndarray],
    track: BodyTrack,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnifiedGLTFError("Sealed composition manifest is unreadable") from exc
    payload = dict(manifest)
    claimed = payload.pop("payload_sha256", None)
    if claimed != sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        raise UnifiedGLTFError("Sealed composition manifest digest is invalid")
    if (
        manifest.get("schema_version") != "autoanim.unified-performance/1.0"
        or manifest.get("artifacts", {}).get("unified_arrays", {}).get("sha256")
        != sha256_file(arrays_path)
        or manifest.get("artifacts", {}).get("body_asset", {}).get("sha256")
        != sha256_file(body_asset_path)
        or manifest.get("artifacts", {}).get("body_manifest", {}).get("sha256")
        != sha256_file(body_manifest_path)
        or manifest.get("artifacts", {}).get("face_performance", {}).get("sha256")
        != sha256_file(face_performance_path)
        or manifest.get("production_validated") is not False
    ):
        raise UnifiedGLTFError("Sealed composition provenance does not match inputs")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise UnifiedGLTFError("Sealed composition arrays are unreadable") from exc
    required = {
        "schema_version",
        "ticks",
        "source_pts",
        "timestamps_seconds",
        "gnm_identity",
        "gnm_expression",
        "gnm_owned_rotations",
        "gnm_owned_translation",
        "body_root_translation_m",
        "body_local_rotations_xyzw",
        "body_foot_contacts",
    }
    if set(arrays) != required or str(arrays["schema_version"]) != (
        "autoanim.unified-performance-arrays/1.0"
    ):
        raise UnifiedGLTFError("Sealed composition arrays have an invalid schema")
    expected_owned_rotations = np.array(performance["rotations"], copy=True)
    expected_owned_rotations[:, :2] = 0.0
    exact_pairs = (
        (arrays["ticks"], track.ticks),
        (arrays["source_pts"], performance["source_pts"]),
        (arrays["timestamps_seconds"], performance["timestamps_seconds"]),
        (arrays["gnm_identity"], performance["identity"]),
        (arrays["gnm_expression"], performance["expression"]),
        (arrays["gnm_owned_rotations"], expected_owned_rotations),
        (arrays["gnm_owned_translation"], np.zeros_like(performance["translation"])),
        (arrays["body_root_translation_m"], track.root_translation_m),
        (arrays["body_local_rotations_xyzw"], track.local_rotations_xyzw),
        (arrays["body_foot_contacts"], track.foot_contacts),
    )
    if any(not np.array_equal(observed, expected) for observed, expected in exact_pairs):
        raise UnifiedGLTFError("Face/body arrays do not match the sealed composition")
    timeline = manifest.get("timeline", {})
    if (
        timeline.get("ticks_sha256")
        != _composition_array_sha256(arrays["ticks"])
        or timeline.get("source_pts_sha256")
        != _composition_array_sha256(arrays["source_pts"])
        or timeline.get("face_body_exact_pts_match") is not True
        or timeline.get("face_body_exact_ticks_match") is not True
    ):
        raise UnifiedGLTFError("Sealed composition timeline hashes are invalid")
    return manifest, arrays


def _deform_four(
    vertices: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    skin_matrices: np.ndarray,
) -> np.ndarray:
    points = np.column_stack(
        (np.asarray(vertices, dtype=np.float64), np.ones(len(vertices)))
    )
    indices = np.asarray(joint_indices, dtype=np.int64)
    weights = np.asarray(joint_weights, dtype=np.float64)
    output = np.zeros((len(skin_matrices), len(points), 3), dtype=np.float64)
    for slot in range(4):
        transformed = np.einsum(
            "fvij,vj->fvi",
            skin_matrices[:, indices[:, slot]],
            points,
            optimize=True,
        )[..., :3]
        output += transformed * weights[None, :, slot, None]
    return output


def _bridge_motion_metrics(
    vertices: np.ndarray,
    triangles: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    skin_matrices: np.ndarray,
) -> dict[str, float]:
    faces = np.asarray(triangles, dtype=np.int64)
    deformed = _deform_four(
        vertices, joint_indices, joint_weights, skin_matrices
    )
    p0, p1, p2 = (deformed[:, faces[:, corner]] for corner in range(3))
    edges = np.stack(
        (
            np.linalg.norm(p1 - p0, axis=2),
            np.linalg.norm(p2 - p1, axis=2),
            np.linalg.norm(p0 - p2, axis=2),
        ),
        axis=2,
    )
    double_area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=2)
    longest = np.max(edges, axis=2)
    aspect = longest * longest / np.maximum(double_area, 1.0e-15)
    neutral_edges = edges[0]
    stretch = edges / np.maximum(neutral_edges[None, :, :], 1.0e-12)
    return {
        "neutral_edge_min_m": float(np.min(neutral_edges)),
        "neutral_edge_p95_m": float(np.percentile(neutral_edges, 95)),
        "neutral_edge_max_m": float(np.max(neutral_edges)),
        "aspect_p95": float(np.percentile(aspect, 95)),
        "aspect_max": float(np.max(aspect)),
        "area_min_m2": float(np.min(double_area) * 0.5),
        "edge_stretch_min_ratio": float(np.min(stretch)),
        "edge_stretch_max_ratio": float(np.max(stretch)),
    }


def export_unified_character_glb(
    path: str | Path,
    *,
    report_path: str | Path,
    mapping_path: str | Path,
    body_manifest_path: str | Path,
    body_asset_path: str | Path,
    binding_manifest_path: str | Path,
    binding_arrays_path: str | Path,
    composition_manifest_path: str | Path,
    composition_arrays_path: str | Path,
    face_performance_path: str | Path,
    track: BodyTrack,
    adapter: GNMAdapter | None = None,
) -> UnifiedCharacterGLBExport:
    """Export one connected, ownership-safe diagnostic character."""

    target_skeleton = skeleton_for_track(track)
    validate_body_track(track, skeleton=target_skeleton)
    load_and_validate_body_asset(
        body_manifest_path, body_asset_path
    )
    with np.load(body_asset_path, allow_pickle=False) as archive:
        body = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(face_performance_path, allow_pickle=False) as archive:
        performance = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    identity = np.asarray(performance["identity"], dtype=np.float32)
    if identity.shape != (253,):
        raise UnifiedGLTFError("Face performance identity is invalid")
    composition_manifest, composition_arrays = _load_and_validate_composition(
        composition_manifest_path,
        composition_arrays_path,
        body_asset_path=body_asset_path,
        body_manifest_path=body_manifest_path,
        face_performance_path=face_performance_path,
        performance=performance,
        track=track,
    )
    binding_manifest, binding = load_gnm_body_binding(
        binding_manifest_path,
        binding_arrays_path,
        expected_body_asset_sha256=sha256_file(body_asset_path),
        expected_identity_sha256=array_sha256(identity),
        expected_body_manifest_sha256=sha256_file(body_manifest_path),
    )
    frame_count = len(track.ticks)
    if (
        performance["expression"].shape != (frame_count, 383)
        or performance["rotations"].shape != (frame_count, 4, 3)
        or performance["translation"].shape != (frame_count, 3)
        or performance["timestamps_seconds"].shape != (frame_count,)
        or not np.array_equal(
            performance["timestamps_seconds"],
            track.ticks.astype(np.float64) / track.ticks_per_second,
        )
    ):
        raise UnifiedGLTFError("Face and body timelines do not match exactly")
    eye_identity = np.zeros_like(track.gnm_eye_rotations_xyzw)
    eye_identity[..., 3] = 1.0
    if not np.array_equal(track.gnm_eye_rotations_xyzw, eye_identity):
        raise UnifiedGLTFError("Body track may not own GNM eye rotation")
    for eye_name in ("LeftEye", "RightEye"):
        eye_delta = track.local_rotations_xyzw[
            :, target_skeleton.index(eye_name)
        ]
        if not np.array_equal(eye_delta, eye_identity[:, 0]):
            raise UnifiedGLTFError("Body joint track may not own eye rotation")

    gnm = adapter or GNMAdapter()
    neutral = gnm.mesh(identity=identity)
    if array_sha256(neutral) != binding_manifest["gnm"]["neutral_vertices_sha256"]:
        raise UnifiedGLTFError("Binding neutral GNM identity does not match")
    expression_hash = array_sha256(performance["expression"])
    eye_rotation_hash = array_sha256(performance["rotations"][:, 2:])
    owned_rotations = np.array(performance["rotations"], copy=True)
    owned_rotations[:, :2] = 0.0
    owned_translation = np.zeros_like(performance["translation"])
    batch_identity = np.broadcast_to(identity, (frame_count, len(identity)))
    full_frames = gnm.mesh(
        identity=batch_identity,
        expression=performance["expression"],
        rotations=owned_rotations,
        translation=owned_translation,
    )
    if (
        array_sha256(performance["expression"]) != expression_hash
        or array_sha256(performance["rotations"][:, 2:]) != eye_rotation_hash
    ):
        raise AssertionError("Ownership preparation mutated face-owned arrays")
    baked_frames = full_frames @ GNM_TO_AUTOANIM_BASIS.T
    source_indices = binding["gnm_source_indices"]
    source_weights = binding["gnm_source_weights"]
    socket = np.asarray(binding["gnm_to_body_world_matrix"], dtype=np.float64)
    world_frames = (
        baked_frames @ socket[:3, :3].T + socket[:3, 3]
    ).astype(np.float32)
    mouth_boundary = discover_mouth_boundary(gnm)
    factor = factor_vertex_animation(
        world_frames,
        max_targets=28,
        landmark_indices=gnm.landmark_indices,
        landmark_weights=gnm.landmark_weights,
        preserve_oral_semantics=True,
        mouth_boundary=mouth_boundary,
        tongue_vertex_indices=np.flatnonzero(gnm.vertex_group("tongue") > 0.5),
        teeth_vertex_indices=np.flatnonzero(gnm.vertex_group("teeth") > 0.5),
    )
    head_vertices = interpolate_clipped_attribute(
        factor.base_vertices, source_indices, source_weights
    )
    # A body-only capture can intentionally request a neutral GNM face.  That
    # is a valid attachment review with zero expression targets, not an empty
    # animation error.  Preserve the zero-width morph contract so the skin and
    # body animation still export and the report makes the facial mode clear.
    if factor.rank:
        head_morph_positions = np.stack(
            [
                interpolate_clipped_attribute(target, source_indices, source_weights)
                for target in factor.morph_positions
            ],
            axis=0,
        )
    else:
        head_morph_positions = np.zeros(
            (0, len(source_indices), 3), dtype=np.float32
        )
    # Expression motion is exactly zero at the shared collar edge.
    head_loop = np.asarray(binding["gnm_neck_loop_indices"], dtype=np.int64)
    head_morph_positions[:, head_loop] = 0.0

    original_to_clipped = np.full(gnm.model.num_vertices, -1, dtype=np.int64)
    retained_original = np.logical_and(
        source_indices[:, 0] == source_indices[:, 1],
        source_weights[:, 0] == 1.0,
    )
    original_to_clipped[source_indices[retained_original, 0]] = np.flatnonzero(
        retained_original
    )
    lip_support = np.maximum.reduce(
        (
            gnm.vertex_group("upper_lip_region"),
            gnm.vertex_group("lower_lip_region"),
            gnm.vertex_group("upper_lip"),
            gnm.vertex_group("lower_lip"),
        )
    )
    clipped_lip_indices = original_to_clipped[np.flatnonzero(lip_support > 0.05)]
    clipped_mouth_boundary = original_to_clipped[
        np.asarray(mouth_boundary.cycle, dtype=np.int64)
    ]
    if np.any(clipped_lip_indices < 0) or np.any(clipped_mouth_boundary < 0):
        raise UnifiedGLTFError("Neck clipping unexpectedly removed oral geometry")

    parents = np.asarray(body["parents"], dtype=np.int64)
    skin_matrices, rest_translations, animated_rotations = _skin_matrices(
        body["local_rest_matrices"],
        body["inverse_bind_matrices"],
        parents,
        track,
    )
    dense = _dense_skin_weights(
        body["joint_indices"], body["joint_weights"], len(target_skeleton.names)
    )
    clipped_dense = interpolate_clipped_attribute(
        dense,
        binding["body_source_indices"],
        binding["body_source_weights"],
    ).astype(np.float64)
    clipped_dense /= np.sum(clipped_dense, axis=1, keepdims=True)
    body_indices, body_weights, four_metrics = _trajectory_project_four_weights(
        binding["body_vertices"], clipped_dense, skin_matrices
    )

    body_vertices = np.asarray(binding["body_vertices"], dtype=np.float32)
    head_offset = len(body_vertices)
    vertices = np.concatenate((body_vertices, head_vertices), axis=0)
    body_triangles = np.asarray(binding["body_triangles"], dtype=np.int32)
    head_triangles = (
        np.asarray(binding["gnm_triangles"], dtype=np.int32) + head_offset
    )
    body_loop = np.asarray(binding["body_neck_loop_indices"], dtype=np.int32)
    combined_head_loop = head_loop.astype(np.int32) + head_offset
    bridge = _zipper_loops(body_loop, combined_head_loop, vertices)
    triangles = np.concatenate((body_triangles, head_triangles, bridge), axis=0)
    if _boundary_edge_count(
        triangles, np.concatenate((body_loop, combined_head_loop))
    ):
        raise UnifiedGLTFError("Connected neck still contains an open seam edge")

    head_joint = target_skeleton.index("Head")
    joint_indices = np.zeros((len(vertices), 4), dtype=np.uint16)
    joint_weights = np.zeros((len(vertices), 4), dtype=np.float32)
    joint_indices[:head_offset] = body_indices
    joint_weights[:head_offset] = body_weights
    neck_joint = target_skeleton.index("Neck")
    gnm_weights = np.asarray(gnm.model.skinning_weights, dtype=np.float64).T
    gnm_canonical_dense = np.zeros(
        (gnm.model.num_vertices, len(target_skeleton.names)),
        dtype=np.float64,
    )
    gnm_canonical_dense[:, neck_joint] = gnm_weights[:, 0]
    gnm_canonical_dense[:, head_joint] = np.sum(gnm_weights[:, 1:], axis=1)
    clipped_gnm_dense = interpolate_clipped_attribute(
        gnm_canonical_dense, source_indices, source_weights
    )
    clipped_gnm_dense /= np.sum(clipped_gnm_dense, axis=1, keepdims=True)
    head_joint_order = np.argsort(
        -clipped_gnm_dense, axis=1, kind="stable"
    )[:, :4]
    head_joint_weights = np.take_along_axis(
        clipped_gnm_dense, head_joint_order, axis=1
    )
    joint_indices[head_offset:] = head_joint_order.astype(np.uint16)
    joint_weights[head_offset:] = head_joint_weights.astype(np.float32)
    joint_indices[joint_weights <= 0.0] = 0
    if not np.allclose(np.sum(joint_weights, axis=1), 1.0, atol=1.0e-6):
        raise UnifiedGLTFError("Unified skin weights do not sum to one")
    bridge_motion = _bridge_motion_metrics(
        vertices,
        bridge,
        joint_indices,
        joint_weights,
        skin_matrices,
    )

    morph_positions = np.zeros(
        (factor.rank, len(vertices), 3), dtype=np.float32
    )
    morph_positions[:, head_offset:] = head_morph_positions
    normal_fit = _fit_morph_normals(
        vertices,
        morph_positions,
        factor.weights,
        triangles,
        priority_vertex_indices=clipped_lip_indices + head_offset,
        boundary_vertex_indices=clipped_mouth_boundary + head_offset,
        # Strong theatrical expressions can need more normal-only basis terms
        # than restrained dialogue even when position reconstruction already
        # passes. Keep the complete glTF morph set within the 64-target viewer
        # budget instead of weakening lip and mouth-boundary angle limits.
        max_corrective_targets=min(8, max(0, 64 - factor.rank)),
    )
    if (
        normal_fit.frame_p95_max_degrees
        > _NORMAL_FRAME_P95_LIMIT_DEGREES
        or normal_fit.priority_frame_p95_max_degrees
        > _LIP_NORMAL_FRAME_P95_LIMIT_DEGREES
        or normal_fit.priority_max_degrees > _LIP_NORMAL_MAX_LIMIT_DEGREES
        or normal_fit.boundary_max_degrees
        > _MOUTH_BOUNDARY_NORMAL_MAX_LIMIT_DEGREES
    ):
        raise AnimationCompressionError(
            "Unified morph-normal reconstruction exceeds quality gates",
            {
                "normal_frame_p95_max_degrees": (
                    normal_fit.frame_p95_max_degrees
                ),
                "lip_normal_frame_p95_max_degrees": (
                    normal_fit.priority_frame_p95_max_degrees
                ),
                "lip_normal_max_degrees": normal_fit.priority_max_degrees,
                "mouth_boundary_normal_max_degrees": normal_fit.boundary_max_degrees,
            },
        )
    if normal_fit.corrective_targets:
        morph_positions = np.concatenate(
            (
                morph_positions,
                np.zeros(
                    (normal_fit.corrective_targets, len(vertices), 3),
                    dtype=np.float32,
                ),
            ),
            axis=0,
        )
    morph_weights = normal_fit.weights

    builder = _Builder()
    attributes = {
        "POSITION": builder.accessor(
            vertices,
            component_type=5126,
            accessor_type="VEC3",
            target=34962,
            bounds=True,
        ),
        "NORMAL": builder.accessor(
            normal_fit.base_normals,
            component_type=5126,
            accessor_type="VEC3",
            target=34962,
        ),
        "JOINTS_0": builder.accessor(
            joint_indices,
            component_type=5123,
            accessor_type="VEC4",
            target=34962,
        ),
        "WEIGHTS_0": builder.accessor(
            joint_weights,
            component_type=5126,
            accessor_type="VEC4",
            target=34962,
        ),
    }
    index_type = np.uint16 if len(vertices) <= 65_535 else np.uint32
    index_accessor = builder.accessor(
        triangles.astype(index_type),
        component_type=5123 if index_type is np.uint16 else 5125,
        accessor_type="SCALAR",
        target=34963,
        bounds=True,
    )
    targets = []
    for position_delta, normal_delta in zip(
        morph_positions, normal_fit.morph_normals, strict=True
    ):
        targets.append(
            {
                "POSITION": builder.accessor(
                    position_delta,
                    component_type=5126,
                    accessor_type="VEC3",
                    target=34962,
                    bounds=True,
                ),
                "NORMAL": builder.accessor(
                    normal_delta,
                    component_type=5126,
                    accessor_type="VEC3",
                    target=34962,
                ),
            }
        )
    inverse_bind_accessor = builder.accessor(
        np.asarray(body["inverse_bind_matrices"], dtype=np.float32).transpose(
            0, 2, 1
        ),
        component_type=5126,
        accessor_type="MAT4",
    )
    timestamps = (
        track.ticks.astype(np.float64) / track.ticks_per_second
    ).astype(np.float32)
    time_accessor = builder.accessor(
        timestamps,
        component_type=5126,
        accessor_type="SCALAR",
        bounds=True,
    )

    root_translation = (
        rest_translations[0][None, :] + track.root_translation_m
    ).astype(np.float32)
    nodes: list[dict[str, Any]] = []
    for index, name in enumerate(target_skeleton.names):
        node: dict[str, Any] = {
            "name": name,
            "translation": (
                root_translation[0]
                if index == 0
                else rest_translations[index]
            ).astype(float).tolist(),
            "rotation": animated_rotations[0, index].astype(float).tolist(),
        }
        children = np.flatnonzero(parents == index).astype(int).tolist()
        if children:
            node["children"] = children
        nodes.append(node)
    mesh_node = len(nodes)
    mesh_node_payload: dict[str, Any] = {
        "name": "AutoAnim_Connected_Character",
        "mesh": 0,
        "skin": 0,
    }
    if len(morph_weights[0]):
        mesh_node_payload["weights"] = morph_weights[0].astype(float).tolist()
    nodes.append(mesh_node_payload)
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []

    def channel(output: int, node: int, path_name: str) -> None:
        sampler = len(samplers)
        samplers.append(
            {
                "input": time_accessor,
                "output": output,
                "interpolation": "LINEAR",
            }
        )
        channels.append(
            {"sampler": sampler, "target": {"node": node, "path": path_name}}
        )

    channel(
        builder.accessor(
            root_translation, component_type=5126, accessor_type="VEC3"
        ),
        0,
        "translation",
    )
    excluded_eyes = {
        target_skeleton.index("LeftEye"),
        target_skeleton.index("RightEye"),
    }
    for joint_index in range(len(target_skeleton.names)):
        if joint_index in excluded_eyes:
            continue
        channel(
            builder.accessor(
                animated_rotations[:, joint_index],
                component_type=5126,
                accessor_type="VEC4",
            ),
            joint_index,
            "rotation",
        )
    if len(morph_weights[0]):
        channel(
            builder.accessor(
                morph_weights.reshape(-1),
                component_type=5126,
                accessor_type="SCALAR",
            ),
            mesh_node,
            "weights",
        )

    blockers = list(binding_manifest["publish_blockers"])
    if four_metrics["rms_m"] > FOUR_WEIGHT_RMS_GATE_M:
        blockers.append("THREEJS_FOUR_WEIGHT_RMS_ABOVE_0_1MM")
    if four_metrics["max_m"] > FOUR_WEIGHT_MAX_GATE_M:
        blockers.append("THREEJS_FOUR_WEIGHT_MAX_ABOVE_0_5MM")
    if bridge_motion["aspect_p95"] > 5.0 or bridge_motion["aspect_max"] > 10.0:
        blockers.append("NECK_BRIDGE_TRIANGLE_QUALITY_FAILED")
    if (
        bridge_motion["edge_stretch_min_ratio"] < 0.8
        or bridge_motion["edge_stretch_max_ratio"] > 1.25
    ):
        blockers.append("NECK_BRIDGE_ANIMATED_STRETCH_FAILED")
    blockers = list(dict.fromkeys(blockers))
    report: dict[str, Any] = {
        "schema_version": UNIFIED_GLTF_SCHEMA_VERSION,
        "status": "connected_character_preview",
        "production_validated": False,
        "publish_ready": False,
        "single_scene": True,
        "single_mesh": True,
        "single_skin": True,
        "single_animation": True,
        "single_connected_exterior_topology": True,
        "provider_head_removed": True,
        "expression_sha256": expression_hash,
        "eye_rotations_sha256": eye_rotation_hash,
        "body_asset_sha256": sha256_file(body_asset_path),
        "binding_arrays_sha256": sha256_file(binding_arrays_path),
        "composition_manifest_sha256": sha256_file(composition_manifest_path),
        "composition_arrays_sha256": sha256_file(composition_arrays_path),
        "composition_payload_sha256": composition_manifest["payload_sha256"],
        "body_track_sha256": sha256(
            track.canonical_json_bytes()
        ).hexdigest(),
        "ticks_sha256": composition_manifest["timeline"]["ticks_sha256"],
        "source_pts_sha256": composition_manifest["timeline"][
            "source_pts_sha256"
        ],
        "ownership_contract_sha256": composition_manifest["ownership"][
            "contract_sha256"
        ],
        "face_performance_sha256": sha256_file(face_performance_path),
        "metrics": {
            "four_weight_runtime": four_metrics,
            "face_factor_rank": factor.rank,
            "face_mesh_p95_m": factor.mesh_p95_m,
            "face_mesh_max_m": factor.mesh_max_m,
            "normal_frame_p95_max_degrees": (
                normal_fit.frame_p95_max_degrees
            ),
            "normal_max_degrees": normal_fit.max_degrees,
            "lip_normal_frame_p95_max_degrees": (
                normal_fit.priority_frame_p95_max_degrees
            ),
            "lip_normal_max_degrees": normal_fit.priority_max_degrees,
            "mouth_boundary_normal_max_degrees": normal_fit.boundary_max_degrees,
            "neck_bridge_triangle_count": int(len(bridge)),
            "neck_open_boundary_edge_count": 0,
            "neck_bridge_motion": bridge_motion,
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(triangles)),
        },
        "publish_blockers": blockers,
    }
    report["payload_sha256"] = sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "AutoAnim unified character exporter 1.0",
            "extras": report,
        },
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "MPFB_GNM_Connected_Character",
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                        "targets": targets,
                    }
                ],
                "extras": {
                    "targetNames": [
                        f"AutoAnimTarget_{index:02d}"
                        for index in range(len(targets))
                    ]
                },
            }
        ],
        "materials": [
            {
                "name": "Connected anatomical preview",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.48, 0.22, 0.15, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.72,
                },
                "doubleSided": False,
            }
        ],
        "skins": [
            {
                "name": "AutoAnim canonical humanoid",
                "inverseBindMatrices": inverse_bind_accessor,
                "skeleton": 0,
                "joints": list(range(len(target_skeleton.names))),
            }
        ],
        "animations": [
            {
                "name": "autoanim_connected_performance",
                "samplers": samplers,
                "channels": channels,
            }
        ],
        "buffers": [{"byteLength": len(builder.data)}],
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
    }
    output = Path(path)
    _atomic_bytes(output, _glb_bytes(document, bytes(builder.data)))
    write_json(report_path, report)
    write_npz(
        mapping_path,
        schema_version=np.asarray(UNIFIED_GLTF_SCHEMA_VERSION),
        ticks=track.ticks,
        source_pts=performance["source_pts"],
        timestamps_seconds=timestamps,
        expression_sha256=np.asarray(expression_hash),
        eye_rotations_sha256=np.asarray(eye_rotation_hash),
        body_joint_indices_4=body_indices,
        body_joint_weights_4=body_weights,
        morph_weights=morph_weights,
        binding_arrays_sha256=np.asarray(sha256_file(binding_arrays_path)),
        composition_manifest_sha256=np.asarray(
            sha256_file(composition_manifest_path)
        ),
        composition_arrays_sha256=np.asarray(sha256_file(composition_arrays_path)),
        body_track_sha256=np.asarray(
            sha256(track.canonical_json_bytes()).hexdigest()
        ),
    )
    return UnifiedCharacterGLBExport(
        path=output,
        report_path=Path(report_path),
        mapping_path=Path(mapping_path),
        frame_count=frame_count,
        vertex_count=len(vertices),
        triangle_count=len(triangles),
        morph_target_count=len(targets),
        four_weight_rms_mm=four_metrics["rms_m"] * 1000.0,
        four_weight_max_mm=four_metrics["max_m"] * 1000.0,
        publish_ready=False,
        publish_blockers=tuple(blockers),
    )


__all__ = [
    "FOUR_WEIGHT_MAX_GATE_M",
    "FOUR_WEIGHT_RMS_GATE_M",
    "UNIFIED_GLTF_SCHEMA_VERSION",
    "UnifiedCharacterGLBExport",
    "UnifiedGLTFError",
    "export_unified_character_glb",
]
