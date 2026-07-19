"""Fail-closed structural validation for GNM oral animation geometry.

The checks in this module deliberately stop at geometry.  They measure GNM's
evaluated lips, tongue, and teeth, and can audit the reconstruction stored in an
AutoAnim GLB.  They do *not* infer phonemes, judge speech intelligibility, or
claim perceptual correctness.

GNM's tongue and teeth components are open triangle surfaces.  Consequently a
watertight inside/outside penetration test would be misleading.  Collision
metrics below are conservative vertex-separation and lip-order risk proxies;
the report always records that exact surface intersection was not validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import struct
from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .gnm_adapter import GNMAdapter


SCHEMA_VERSION = "autoanim.oral-validation/1.0"
_INNER_LIP_PAIRS = ((61, 67), (62, 66), (63, 65))
_EXPECTED_VERTICES = 17_821
_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BINARY_CHUNK = 0x004E4942


class OralValidationError(ValueError):
    """A required geometry or structural-validation input was absent/invalid."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class OralValidationThresholds:
    """Dimensionless gates except for explicit reconstruction tolerances."""

    lip_contact_gap_interocular: float = 0.006
    lip_order_inversion_tolerance_interocular: float = 0.0005
    tongue_teeth_near_contact_interocular: float = 0.010
    tongue_teeth_collision_risk_interocular: float = 0.001
    structural_reconstruction_p95_m: float = 0.00010
    structural_reconstruction_max_m: float = 0.00050

    def __post_init__(self) -> None:
        values = (
            self.lip_contact_gap_interocular,
            self.lip_order_inversion_tolerance_interocular,
            self.tongue_teeth_near_contact_interocular,
            self.tongue_teeth_collision_risk_interocular,
            self.structural_reconstruction_p95_m,
            self.structural_reconstruction_max_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise OralValidationError(
                "INVALID_THRESHOLDS", "Oral validation thresholds must be finite and positive"
            )
        if (
            self.tongue_teeth_collision_risk_interocular
            >= self.tongue_teeth_near_contact_interocular
        ):
            raise OralValidationError(
                "INVALID_THRESHOLDS",
                "Tongue/teeth collision-risk tolerance must be smaller than near-contact tolerance",
            )
        if self.structural_reconstruction_p95_m > self.structural_reconstruction_max_m:
            raise OralValidationError(
                "INVALID_THRESHOLDS",
                "Structural p95 tolerance may not exceed the maximum tolerance",
            )


@dataclass(frozen=True, slots=True)
class OralValidationResult:
    """JSON report plus immutable per-frame geometry evidence."""

    report: dict[str, Any]
    timestamps: np.ndarray
    lip_gap_interocular: np.ndarray
    lip_contact_frames: np.ndarray
    lip_order_inversion_risk_frames: np.ndarray
    tongue_upper_teeth_gap_interocular: np.ndarray
    tongue_lower_teeth_gap_interocular: np.ndarray
    tongue_teeth_near_contact_frames: np.ndarray
    tongue_teeth_collision_risk_frames: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.report)


@dataclass(frozen=True, slots=True)
class _OralTopology:
    required_native: np.ndarray
    native_to_required: np.ndarray
    groups: dict[str, np.ndarray]
    landmark_local_indices: np.ndarray
    landmark_weights: np.ndarray
    inventory: dict[str, Any]


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value).copy()
    output.setflags(write=False)
    return output


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise OralValidationError("INVALID_GEOMETRY", "Metric arrays must be finite and non-empty")
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _boundary_edges(triangles: np.ndarray) -> int:
    faces = np.asarray(triangles, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise OralValidationError("GEOMETRY_ABSENT", "GNM oral component has no triangles")
    edges = np.sort(
        np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def _topology(adapter: GNMAdapter) -> _OralTopology:
    if adapter.model.num_vertices != _EXPECTED_VERTICES:
        raise OralValidationError("GEOMETRY_ABSENT", "Expected complete GNM Head 3.0 geometry")

    try:
        masks = {
            name: adapter.vertex_group(name) > 0.5
            for name in (
                "tongue",
                "teeth",
                "upper_teeth_and_gums",
                "lower_teeth_and_gums",
                "upper_lip",
                "lower_lip",
            )
        }
    except KeyError as exc:
        raise OralValidationError(
            "GEOMETRY_ABSENT", f"GNM oral vertex group is absent: {exc.args[0]}"
        ) from exc

    native_groups = {
        "tongue": np.flatnonzero(masks["tongue"]),
        "upper_teeth": np.flatnonzero(masks["teeth"] & masks["upper_teeth_and_gums"]),
        "lower_teeth": np.flatnonzero(masks["teeth"] & masks["lower_teeth_and_gums"]),
        "upper_lip": np.flatnonzero(masks["upper_lip"]),
        "lower_lip": np.flatnonzero(masks["lower_lip"]),
    }
    minimum_counts = {
        "tongue": 100,
        "upper_teeth": 100,
        "lower_teeth": 100,
        "upper_lip": 20,
        "lower_lip": 20,
    }
    for name, indices in native_groups.items():
        if len(indices) < minimum_counts[name]:
            raise OralValidationError(
                "GEOMETRY_ABSENT", f"GNM {name} geometry is absent or incomplete"
            )

    landmark_native = np.asarray(adapter.landmark_indices, dtype=np.int64)
    landmark_weights = np.asarray(adapter.landmark_weights, dtype=np.float64)
    if landmark_native.shape != (68, 3) or landmark_weights.shape != (68, 3):
        raise OralValidationError("GEOMETRY_ABSENT", "GNM sparse oral landmarks are absent")
    required = np.unique(
        np.concatenate([landmark_native.reshape(-1), *native_groups.values()])
    ).astype(np.int32)
    native_to_required = np.full(adapter.model.num_vertices, -1, dtype=np.int32)
    native_to_required[required] = np.arange(len(required), dtype=np.int32)
    local_groups = {
        name: native_to_required[indices]
        for name, indices in native_groups.items()
    }
    if any(np.any(indices < 0) for indices in local_groups.values()):
        raise OralValidationError("GEOMETRY_ABSENT", "Oral topology indexing is incomplete")

    component_names = tuple(adapter.model.mesh_component_names)
    inventory_components: dict[str, dict[str, Any]] = {}
    for name in ("tongue", "upper_teeth_and_gums", "lower_teeth_and_gums"):
        if name not in component_names:
            raise OralValidationError("GEOMETRY_ABSENT", f"GNM component is absent: {name}")
        triangles = np.asarray(adapter.model.triangles_group(name), dtype=np.int32)
        boundary_count = _boundary_edges(triangles)
        inventory_components[name] = {
            "triangle_count": int(len(triangles)),
            "boundary_edge_count": boundary_count,
            "watertight": boundary_count == 0,
        }

    inventory = {
        "gnm_version": "3.0",
        "source_vertex_count": int(adapter.model.num_vertices),
        "required_oral_and_landmark_vertex_count": int(len(required)),
        "vertex_groups": {name: int(len(value)) for name, value in native_groups.items()},
        "components": inventory_components,
        "all_required_groups_present": True,
    }
    return _OralTopology(
        required_native=_readonly(required),
        native_to_required=_readonly(native_to_required),
        groups={name: _readonly(value) for name, value in local_groups.items()},
        landmark_local_indices=_readonly(native_to_required[landmark_native]),
        landmark_weights=_readonly(landmark_weights),
        inventory=inventory,
    )


def _validate_timestamps(timestamps: np.ndarray | None, frame_count: int) -> np.ndarray:
    if timestamps is None:
        values = np.arange(frame_count, dtype=np.float64)
    else:
        values = np.asarray(timestamps, dtype=np.float64)
    if values.shape != (frame_count,) or not np.isfinite(values).all():
        raise OralValidationError(
            "INVALID_GEOMETRY", "Timestamps must contain one finite value per geometry frame"
        )
    if len(values) > 1 and np.any(np.diff(values) <= 0.0):
        raise OralValidationError("INVALID_GEOMETRY", "Timestamps must be strictly increasing")
    return values


def _landmarks(frames: np.ndarray, topology: _OralTopology) -> np.ndarray:
    selected = frames[:, topology.landmark_local_indices, :]
    return np.sum(selected * topology.landmark_weights[None, :, :, None], axis=2)


def _nearest_ratios(
    frames: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    interocular: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.empty(len(frames), dtype=np.float64)
    p01 = np.empty(len(frames), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        distances, _ = cKDTree(frame[target_indices]).query(
            frame[source_indices], k=1, workers=1
        )
        minimum[frame_index] = float(np.min(distances)) / interocular[frame_index]
        p01[frame_index] = float(np.percentile(distances, 1)) / interocular[frame_index]
    return minimum, p01


def _analyze_required_geometry_chunks(
    frame_chunks: Iterable[np.ndarray],
    topology: _OralTopology,
    *,
    frame_count: int,
    timestamps: np.ndarray | None,
    thresholds: OralValidationThresholds,
    source: dict[str, Any],
    expected_lip_contact_target: np.ndarray | None = None,
    declared_lip_contact_attained: np.ndarray | None = None,
) -> OralValidationResult:
    if frame_count < 1:
        raise OralValidationError("INVALID_GEOMETRY", "Oral geometry is empty")
    times = _validate_timestamps(timestamps, frame_count)
    lip_pair_gaps = np.empty((frame_count, len(_INNER_LIP_PAIRS)), dtype=np.float64)
    lip_gap = np.empty(frame_count, dtype=np.float64)
    lip_order_risk = np.empty(frame_count, dtype=bool)
    tongue_upper_min = np.empty(frame_count, dtype=np.float64)
    tongue_upper_p01 = np.empty(frame_count, dtype=np.float64)
    tongue_lower_min = np.empty(frame_count, dtype=np.float64)
    tongue_lower_p01 = np.empty(frame_count, dtype=np.float64)
    tongue_motion_frame_p95 = np.empty(frame_count, dtype=np.float64)
    tongue_motion_frame_max = np.empty(frame_count, dtype=np.float64)
    geometry_hasher = hashlib.sha256()
    reference_center: np.ndarray | None = None
    reference_centered: np.ndarray | None = None
    reference_local_tongue: np.ndarray | None = None
    offset = 0

    for raw_chunk in frame_chunks:
        values = np.asarray(raw_chunk, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (len(topology.required_native), 3):
            raise OralValidationError(
                "GEOMETRY_ABSENT",
                "Oral validation requires every GNM oral and landmark vertex for each frame",
            )
        if not len(values) or not np.isfinite(values).all():
            raise OralValidationError(
                "INVALID_GEOMETRY", "Oral geometry is empty or non-finite"
            )
        stop = offset + len(values)
        if stop > frame_count:
            raise OralValidationError(
                "INVALID_GEOMETRY", "Oral geometry contains more frames than its timeline"
            )
        geometry_hasher.update(np.ascontiguousarray(values).tobytes())
        landmarks = _landmarks(values, topology)
        interocular = np.linalg.norm(landmarks[:, 36] - landmarks[:, 45], axis=1)
        if not np.isfinite(interocular).all() or np.any(interocular <= 1.0e-6):
            raise OralValidationError(
                "INVALID_GEOMETRY", "Interocular geometry scale is invalid"
            )

        chunk_pair_gaps = np.stack(
            [
                np.linalg.norm(landmarks[:, upper] - landmarks[:, lower], axis=1)
                / interocular
                for upper, lower in _INNER_LIP_PAIRS
            ],
            axis=1,
        )
        lip_pair_gaps[offset:stop] = chunk_pair_gaps
        lip_gap[offset:stop] = np.mean(chunk_pair_gaps, axis=1)
        face_up = landmarks[:, 27] - landmarks[:, 8]
        face_up_norm = np.linalg.norm(face_up, axis=1)
        if np.any(face_up_norm <= 1.0e-6):
            raise OralValidationError("INVALID_GEOMETRY", "Face-local up axis is degenerate")
        face_up /= face_up_norm[:, None]
        signed_pair_order = np.stack(
            [
                np.sum((landmarks[:, upper] - landmarks[:, lower]) * face_up, axis=1)
                / interocular
                for upper, lower in _INNER_LIP_PAIRS
            ],
            axis=1,
        )
        lip_order_risk[offset:stop] = np.min(signed_pair_order, axis=1) < (
            -thresholds.lip_order_inversion_tolerance_interocular
        )

        upper_min, upper_p01 = _nearest_ratios(
            values,
            topology.groups["tongue"],
            topology.groups["upper_teeth"],
            interocular,
        )
        lower_min, lower_p01 = _nearest_ratios(
            values,
            topology.groups["tongue"],
            topology.groups["lower_teeth"],
            interocular,
        )
        tongue_upper_min[offset:stop] = upper_min
        tongue_upper_p01[offset:stop] = upper_p01
        tongue_lower_min[offset:stop] = lower_min
        tongue_lower_p01[offset:stop] = lower_p01

        # Anchor every chunk to the first frame so chunking cannot change the
        # tongue-motion metric or retain a full T x tongue-vertex array.
        upper_teeth = values[:, topology.groups["upper_teeth"]]
        tongue = values[:, topology.groups["tongue"]]
        if reference_center is None or reference_centered is None:
            reference_center = np.mean(upper_teeth[0], axis=0)
            reference_centered = upper_teeth[0] - reference_center
        for local_index in range(len(values)):
            center = np.mean(upper_teeth[local_index], axis=0)
            centered = upper_teeth[local_index] - center
            covariance = centered.T @ reference_centered
            try:
                left, singular_values, right_transpose = np.linalg.svd(covariance)
            except np.linalg.LinAlgError as exc:
                raise OralValidationError(
                    "INVALID_GEOMETRY", "Upper-teeth pose alignment did not converge"
                ) from exc
            if not np.isfinite(singular_values).all() or singular_values[1] <= 1.0e-10:
                raise OralValidationError(
                    "INVALID_GEOMETRY", "Upper-teeth pose anchor is degenerate"
                )
            rotation = left @ right_transpose
            if np.linalg.det(rotation) < 0.0:
                left[:, -1] *= -1.0
                rotation = left @ right_transpose
            local_tongue = (
                (tongue[local_index] - center) @ rotation + reference_center
            )
            if reference_local_tongue is None:
                reference_local_tongue = local_tongue.copy()
            motion = np.linalg.norm(local_tongue - reference_local_tongue, axis=1)
            frame_index = offset + local_index
            tongue_motion_frame_p95[frame_index] = float(np.percentile(motion, 95))
            tongue_motion_frame_max[frame_index] = float(np.max(motion, initial=0.0))
        offset = stop

    if offset != frame_count:
        raise OralValidationError(
            "INVALID_GEOMETRY", "Oral geometry contains fewer frames than its timeline"
        )

    lip_contact = lip_gap <= thresholds.lip_contact_gap_interocular
    tongue_teeth_min = np.minimum(tongue_upper_min, tongue_lower_min)
    near_contact = tongue_teeth_min <= thresholds.tongue_teeth_near_contact_interocular
    collision_risk = (
        tongue_teeth_min <= thresholds.tongue_teeth_collision_risk_interocular
    )

    target_evidence: dict[str, Any]
    if expected_lip_contact_target is None:
        target_evidence = {
            "source": "absent",
            "candidate_frames": 0,
            "geometry_attained_frames": 0,
            "geometry_attainment_fraction": None,
            "declared_geometry_disagreement_frames": None,
            "phoneme_ground_truth": False,
        }
    else:
        target = np.asarray(expected_lip_contact_target, dtype=np.float64)
        if target.shape != (frame_count,) or not np.isfinite(target).all() or np.any(target < 0.0):
            raise OralValidationError(
                "INVALID_GEOMETRY",
                "Lip-contact targets must be finite, non-negative, and frame-aligned",
            )
        candidates = target > 0.0
        attained = candidates & (lip_gap <= target + 1.0e-3)
        candidate_count = int(np.count_nonzero(candidates))
        disagreement: int | None = None
        if declared_lip_contact_attained is not None:
            declared = np.asarray(declared_lip_contact_attained)
            if declared.shape != (frame_count,) or declared.dtype.kind != "b":
                raise OralValidationError(
                    "INVALID_GEOMETRY",
                    "Declared lip-contact attainment must be one boolean per frame",
                )
            disagreement = int(np.count_nonzero(declared != attained))
        target_evidence = {
            "source": "unvalidated_pipeline_control_target",
            "candidate_frames": candidate_count,
            "geometry_attained_frames": int(np.count_nonzero(attained)),
            "geometry_attainment_fraction": (
                float(np.mean(attained[candidates])) if candidate_count else None
            ),
            "declared_geometry_disagreement_frames": disagreement,
            "phoneme_ground_truth": False,
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_not_production_validated",
        "source": {
            **source,
            "frame_count": int(frame_count),
            "all_frames_evaluated": True,
            "geometry_sha256": geometry_hasher.hexdigest(),
        },
        "geometry_inventory": copy.deepcopy(topology.inventory),
        "coordinate_contract": {
            "distance_scale": "per-frame GNM interocular distance",
            "tongue_motion_frame": "rigid first-frame alignment anchored by upper teeth",
            "rigid_transform_invariant_distance_metrics": True,
            "differential_joint_skinning_is_measured_geometry": True,
        },
        "lip_contact": {
            "method": "three_inner_lip_landmark_pair_mean",
            "gap_interocular": _summary(lip_gap),
            "pair_gap_interocular": {
                f"{upper}_{lower}": _summary(lip_pair_gaps[:, index])
                for index, (upper, lower) in enumerate(_INNER_LIP_PAIRS)
            },
            "contact_threshold_interocular": thresholds.lip_contact_gap_interocular,
            "contact_frames": int(np.count_nonzero(lip_contact)),
            "contact_fraction": float(np.mean(lip_contact)),
            "order_inversion_tolerance_interocular": (
                thresholds.lip_order_inversion_tolerance_interocular
            ),
            "order_inversion_risk_frames": int(np.count_nonzero(lip_order_risk)),
            "order_inversion_risk_fraction": float(np.mean(lip_order_risk)),
            "collision_interpretation": (
                "pair-order risk proxy only; exact lip surface intersection was not tested"
            ),
            "target_evidence": target_evidence,
        },
        "tongue_teeth": {
            "method": "nearest GNM tongue-to-teeth vertex separation",
            "upper_minimum_gap_interocular": _summary(tongue_upper_min),
            "lower_minimum_gap_interocular": _summary(tongue_lower_min),
            "upper_p01_gap_interocular": _summary(tongue_upper_p01),
            "lower_p01_gap_interocular": _summary(tongue_lower_p01),
            "near_contact_threshold_interocular": (
                thresholds.tongue_teeth_near_contact_interocular
            ),
            "near_contact_frames": int(np.count_nonzero(near_contact)),
            "near_contact_fraction": float(np.mean(near_contact)),
            "collision_risk_threshold_interocular": (
                thresholds.tongue_teeth_collision_risk_interocular
            ),
            "collision_risk_frames": int(np.count_nonzero(collision_risk)),
            "collision_risk_fraction": float(np.mean(collision_risk)),
            "collision_interpretation": (
                "open-surface proximity risk only; penetration-free geometry was not established"
            ),
        },
        "tongue_motion": {
            "method": "first-frame-relative face-local tongue vertex displacement",
            "frame_p95_m": _summary(tongue_motion_frame_p95),
            "frame_max_m": _summary(tongue_motion_frame_max),
            "moving_frames_over_0_1mm": int(
                np.count_nonzero(tongue_motion_frame_max > 0.0001)
            ),
        },
        "structural_reconstruction": {
            "status": "not_evaluated_no_reference",
            "validated": False,
        },
        "claims": {
            "oral_geometry_present": True,
            "structural_geometry_measured": True,
            "lip_contact_structurally_measured": True,
            "tongue_teeth_proximity_structurally_measured": True,
            "structural_reconstruction_validated": False,
            "exact_surface_intersection_validated": False,
            "penetration_free_validated": False,
            "phoneme_correctness_validated": False,
            "perceptual_correctness_validated": False,
            "tongue_visibility_validated": False,
            "production_validated": False,
        },
        "limitations": [
            "GNM oral components are open surfaces, so proximity is not a signed penetration test.",
            "A control-track contact target is pipeline intent, not phoneme ground truth.",
            (
                "Structural reconstruction does not establish intelligibility or "
                "perceptual speech quality."
            ),
            (
                "Tongue visibility and tongue/teeth collision response require "
                "renderer/rig-specific validation."
            ),
        ],
    }
    return OralValidationResult(
        report=report,
        timestamps=_readonly(times),
        lip_gap_interocular=_readonly(lip_gap),
        lip_contact_frames=_readonly(lip_contact),
        lip_order_inversion_risk_frames=_readonly(lip_order_risk),
        tongue_upper_teeth_gap_interocular=_readonly(tongue_upper_min),
        tongue_lower_teeth_gap_interocular=_readonly(tongue_lower_min),
        tongue_teeth_near_contact_frames=_readonly(near_contact),
        tongue_teeth_collision_risk_frames=_readonly(collision_risk),
    )


def _analyze_required_geometry(
    frames: np.ndarray,
    topology: _OralTopology,
    *,
    timestamps: np.ndarray | None,
    thresholds: OralValidationThresholds,
    source: dict[str, Any],
    expected_lip_contact_target: np.ndarray | None = None,
    declared_lip_contact_attained: np.ndarray | None = None,
) -> OralValidationResult:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3:
        raise OralValidationError(
            "GEOMETRY_ABSENT",
            "Oral validation requires every GNM oral and landmark vertex for each frame",
        )
    return _analyze_required_geometry_chunks(
        (values,),
        topology,
        frame_count=len(values),
        timestamps=timestamps,
        thresholds=thresholds,
        source=source,
        expected_lip_contact_target=expected_lip_contact_target,
        declared_lip_contact_attained=declared_lip_contact_attained,
    )


def validate_oral_frames(
    frames: np.ndarray,
    *,
    adapter: GNMAdapter | None = None,
    timestamps: np.ndarray | None = None,
    expected_lip_contact_target: np.ndarray | None = None,
    declared_lip_contact_attained: np.ndarray | None = None,
    thresholds: OralValidationThresholds = OralValidationThresholds(),
    source_kind: str = "evaluated_gnm_frames",
) -> OralValidationResult:
    """Measure complete evaluated GNM frames; missing oral geometry is an error."""

    selected_adapter = adapter or GNMAdapter()
    topology = _topology(selected_adapter)
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (selected_adapter.model.num_vertices, 3):
        raise OralValidationError(
            "GEOMETRY_ABSENT",
            f"Expected GNM frames [T,{selected_adapter.model.num_vertices},3]",
        )
    if not len(values) or not np.isfinite(values).all():
        raise OralValidationError("INVALID_GEOMETRY", "GNM frames are empty or non-finite")
    return _analyze_required_geometry(
        values[:, topology.required_native],
        topology,
        timestamps=timestamps,
        thresholds=thresholds,
        source={"kind": source_kind},
        expected_lip_contact_target=expected_lip_contact_target,
        declared_lip_contact_attained=declared_lip_contact_attained,
    )


def _load_controls(
    controls_path: str | Path,
    adapter: GNMAdapter,
    identity: np.ndarray | None,
) -> dict[str, np.ndarray]:
    path = Path(controls_path)
    if not path.is_file():
        raise OralValidationError("GEOMETRY_ABSENT", "GNM controls artifact is absent")
    try:
        with np.load(path, allow_pickle=False) as values:
            available = set(values.files)
            timestamp_name = (
                "timestamps" if "timestamps" in available else "timestamps_seconds"
            )
            if "expression" not in available or timestamp_name not in available:
                raise OralValidationError(
                    "GEOMETRY_ABSENT",
                    "Controls artifact has no expression/timestamp geometry source",
                )
            arrays = {
                "expression": np.asarray(values["expression"]).copy(),
                "timestamps": np.asarray(values[timestamp_name]).copy(),
            }
            for name in ("identity", "rotations", "translation", "lip_contact_attained"):
                if name in available:
                    arrays[name] = np.asarray(values[name]).copy()
            target_name = (
                "lip_contact_target_gap"
                if "lip_contact_target_gap" in available
                else "lip_contact_target_gap_interocular"
            )
            if target_name in available:
                arrays["lip_contact_target_gap"] = np.asarray(values[target_name]).copy()
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, OralValidationError):
            raise
        raise OralValidationError("INVALID_CONTROLS", "Controls artifact is unreadable") from exc

    expression = np.asarray(arrays["expression"], dtype=np.float32)
    timestamps = np.asarray(arrays["timestamps"], dtype=np.float64)
    if (
        expression.ndim != 2
        or expression.shape[1] != adapter.expression_dim
        or not len(expression)
        or not np.isfinite(expression).all()
    ):
        raise OralValidationError("INVALID_CONTROLS", "GNM expression controls are invalid")
    _validate_timestamps(timestamps, len(expression))
    if identity is None:
        identity_value = (
            np.asarray(arrays["identity"], dtype=np.float32)
            if "identity" in arrays
            else np.zeros(adapter.identity_dim, dtype=np.float32)
        )
    else:
        identity_value = np.asarray(identity, dtype=np.float32)
    if identity_value.shape != (adapter.identity_dim,) or not np.isfinite(identity_value).all():
        raise OralValidationError("INVALID_CONTROLS", "GNM identity controls are invalid")
    arrays["expression"] = expression
    arrays["timestamps"] = timestamps
    arrays["identity"] = identity_value.copy()
    return arrays


def _control_required_chunks(
    controls: dict[str, np.ndarray],
    adapter: GNMAdapter,
    topology: _OralTopology,
    *,
    posed: bool,
    batch_size: int,
    evaluated_frames: np.ndarray | None = None,
) -> Iterable[np.ndarray]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise OralValidationError("INVALID_CONTROLS", "batch_size must be a positive integer")
    expression = controls["expression"]
    frame_count = len(expression)
    if evaluated_frames is not None:
        provided = np.asarray(evaluated_frames)
        if provided.shape != (frame_count, adapter.model.num_vertices, 3):
            raise OralValidationError(
                "GEOMETRY_ABSENT",
                "Provided evaluated frames do not match the complete controls track",
            )
        for start in range(0, frame_count, batch_size):
            stop = min(start + batch_size, frame_count)
            chunk = np.asarray(provided[start:stop], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise OralValidationError(
                    "INVALID_GEOMETRY", "Provided evaluated frames are non-finite"
                )
            yield chunk[:, topology.required_native]
        return

    rotations = controls.get("rotations")
    translation = controls.get("translation")
    if posed:
        if rotations is None or translation is None:
            raise OralValidationError(
                "GEOMETRY_ABSENT",
                "Posed reconstruction reference requires rotations and translation",
            )
        rotations = np.asarray(rotations, dtype=np.float32)
        translation = np.asarray(translation, dtype=np.float32)
        if (
            rotations.shape != (frame_count, 4, 3)
            or translation.shape != (frame_count, 3)
            or not np.isfinite(rotations).all()
            or not np.isfinite(translation).all()
        ):
            raise OralValidationError("INVALID_CONTROLS", "GNM pose controls are invalid")

    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        identity_batch = np.broadcast_to(
            controls["identity"], (stop - start, adapter.identity_dim)
        )
        try:
            mesh = adapter.mesh(
                identity=identity_batch,
                expression=expression[start:stop],
                rotations=rotations[start:stop] if posed and rotations is not None else None,
                translation=translation[start:stop] if posed and translation is not None else None,
            )
        except (ValueError, TypeError, FloatingPointError) as exc:
            raise OralValidationError(
                "INVALID_CONTROLS", "GNM could not evaluate the supplied controls"
            ) from exc
        if mesh.shape != (stop - start, adapter.model.num_vertices, 3):
            raise OralValidationError(
                "GEOMETRY_ABSENT", "GNM did not evaluate complete head geometry"
            )
        yield mesh[:, topology.required_native]


def _evaluate_controls_required(
    controls: dict[str, np.ndarray],
    adapter: GNMAdapter,
    topology: _OralTopology,
    *,
    posed: bool,
    batch_size: int,
) -> np.ndarray:
    """Materialize a bounded-size caller's reference; control reports stream instead."""

    output = np.empty(
        (len(controls["expression"]), len(topology.required_native), 3),
        dtype=np.float32,
    )
    offset = 0
    for chunk in _control_required_chunks(
        controls,
        adapter,
        topology,
        posed=posed,
        batch_size=batch_size,
    ):
        output[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return output


def _tongue_control_evidence(
    expression: np.ndarray,
    tongue_basis: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Summarize isolated tongue transfer without retaining T x vertex geometry."""

    coefficient_peak = 0.0
    coefficient_active_frames = 0
    geometry_active_frames = 0
    active_without_geometry_frames = 0
    displacement_max = 0.0
    frame_count = len(expression)
    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        coefficients = expression[start:stop, 350:382]
        isolated = np.einsum("tc,cvj->tvj", coefficients, tongue_basis, optimize=True)
        norms = np.linalg.norm(isolated, axis=2)
        coefficient_active = np.max(np.abs(coefficients), axis=1) > 1.0e-5
        geometry_active = np.max(norms, axis=1) > 1.0e-6
        coefficient_peak = max(
            coefficient_peak, float(np.max(np.abs(coefficients), initial=0.0))
        )
        displacement_max = max(
            displacement_max, float(np.max(norms, initial=0.0))
        )
        coefficient_active_frames += int(np.count_nonzero(coefficient_active))
        geometry_active_frames += int(np.count_nonzero(geometry_active))
        active_without_geometry_frames += int(
            np.count_nonzero(coefficient_active & ~geometry_active)
        )

    # A fixed-size second-pass histogram keeps the all-vertex p95 bounded in
    # memory. The upper edge is reported conservatively and its resolution is
    # recorded in the evidence rather than implying an exact order statistic.
    histogram_bins = 8192
    histogram = np.zeros(histogram_bins, dtype=np.int64)
    sample_count = frame_count * tongue_basis.shape[1]
    if displacement_max > 0.0:
        for start in range(0, frame_count, batch_size):
            stop = min(start + batch_size, frame_count)
            isolated = np.einsum(
                "tc,cvj->tvj",
                expression[start:stop, 350:382],
                tongue_basis,
                optimize=True,
            )
            norms = np.linalg.norm(isolated, axis=2)
            counts, _ = np.histogram(
                norms,
                bins=histogram_bins,
                range=(0.0, displacement_max),
            )
            histogram += counts
        rank = max(1, int(np.ceil(0.95 * sample_count)))
        bin_index = int(np.searchsorted(np.cumsum(histogram), rank, side="left"))
        displacement_p95 = min(
            displacement_max,
            (bin_index + 1) * displacement_max / histogram_bins,
        )
        quantile_resolution = displacement_max / histogram_bins
    else:
        displacement_p95 = 0.0
        quantile_resolution = 0.0

    return {
        "tongue_coefficient_peak": coefficient_peak,
        "tongue_control_active_frames": coefficient_active_frames,
        "isolated_tongue_geometry_active_frames": geometry_active_frames,
        "active_control_without_geometry_frames": active_without_geometry_frames,
        "isolated_tongue_displacement_p95_m": float(displacement_p95),
        "isolated_tongue_displacement_max_m": displacement_max,
        "p95_method": "bounded_memory_histogram_upper_edge",
        "p95_resolution_m": float(quantile_resolution),
        "interpretation": (
            "control-to-geometry transfer only; timing and tongue pose correctness "
            "are not established"
        ),
    }


def validate_controls_npz(
    controls_path: str | Path,
    *,
    adapter: GNMAdapter | None = None,
    identity: np.ndarray | None = None,
    evaluated_frames: np.ndarray | None = None,
    thresholds: OralValidationThresholds = OralValidationThresholds(),
    batch_size: int = 64,
) -> OralValidationResult:
    """Evaluate every control frame with bounded memory, or audit supplied exact frames."""

    selected_adapter = adapter or GNMAdapter()
    topology = _topology(selected_adapter)
    controls = _load_controls(controls_path, selected_adapter, identity)
    target = controls.get("lip_contact_target_gap")
    declared = controls.get("lip_contact_attained")
    result = _analyze_required_geometry_chunks(
        _control_required_chunks(
            controls,
            selected_adapter,
            topology,
            posed=False,
            batch_size=batch_size,
            evaluated_frames=evaluated_frames,
        ),
        topology,
        frame_count=len(controls["expression"]),
        timestamps=controls["timestamps"],
        thresholds=thresholds,
        source={
            "kind": "gnm_controls_npz",
            "artifact_sha256": _sha256(controls_path),
            "artifact_name": Path(controls_path).name,
            "evaluation_mode": (
                "provided_complete_gnm_frames"
                if evaluated_frames is not None
                else "streamed_controls"
            ),
            "evaluation_batch_size": batch_size,
        },
        expected_lip_contact_target=target,
        declared_lip_contact_attained=declared,
    )

    expression = controls["expression"]
    tongue_native = topology.required_native[topology.groups["tongue"]]
    tongue_basis = np.asarray(
        selected_adapter.model.expression_basis[350:382, tongue_native],
        dtype=np.float32,
    )
    result.report["control_evidence"] = _tongue_control_evidence(
        expression,
        tongue_basis,
        batch_size=batch_size,
    )
    return result


def _glb_chunks(path: str | Path) -> tuple[dict[str, Any], bytes]:
    file_path = Path(path)
    if not file_path.is_file():
        raise OralValidationError("GEOMETRY_ABSENT", "GLB artifact is absent")
    try:
        payload = file_path.read_bytes()
    except OSError as exc:
        raise OralValidationError("INVALID_GLB", "GLB artifact is unreadable") from exc
    if len(payload) < 20:
        raise OralValidationError("INVALID_GLB", "GLB is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise OralValidationError("INVALID_GLB", "GLB header is invalid")
    offset = 12
    chunks: dict[int, bytes] = {}
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise OralValidationError("INVALID_GLB", "GLB chunk header is truncated")
        length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        if offset + length > len(payload) or chunk_type in chunks:
            raise OralValidationError("INVALID_GLB", "GLB chunk table is invalid")
        chunks[chunk_type] = payload[offset : offset + length]
        offset += length
    if offset != len(payload) or _GLB_JSON_CHUNK not in chunks or _GLB_BINARY_CHUNK not in chunks:
        raise OralValidationError("INVALID_GLB", "GLB JSON or binary geometry chunk is absent")
    try:
        document = json.loads(chunks[_GLB_JSON_CHUNK].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OralValidationError("INVALID_GLB", "GLB JSON is invalid") from exc
    if not isinstance(document, dict):
        raise OralValidationError("INVALID_GLB", "GLB JSON root is invalid")
    return document, chunks[_GLB_BINARY_CHUNK]


def _float_accessor(document: dict[str, Any], binary: bytes, index: Any) -> np.ndarray:
    try:
        accessor = document["accessors"][int(index)]
        view = document["bufferViews"][int(accessor["bufferView"])]
        count = int(accessor["count"])
        accessor_type = str(accessor["type"])
        component_type = int(accessor["componentType"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise OralValidationError("INVALID_GLB", "GLB position accessor is invalid") from exc
    component_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(
        accessor_type
    )
    if (
        component_count is None
        or component_type != 5126
        or count < 1
        or "sparse" in accessor
        or int(view.get("buffer", 0)) != 0
    ):
        raise OralValidationError(
            "INVALID_GLB", "GLB geometry must use bounded, dense float32 accessors"
        )
    item_size = component_count * 4
    try:
        stride = int(view.get("byteStride", item_size))
        view_offset = int(view.get("byteOffset", 0))
        view_length = int(view["byteLength"])
        offset = view_offset + int(accessor.get("byteOffset", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise OralValidationError("INVALID_GLB", "GLB accessor bounds are invalid") from exc
    if stride < item_size or stride % 4 or offset < view_offset:
        raise OralValidationError("INVALID_GLB", "GLB accessor stride/offset is invalid")
    final_byte = offset + (count - 1) * stride + item_size
    if final_byte > view_offset + view_length or final_byte > len(binary):
        raise OralValidationError("INVALID_GLB", "GLB accessor exceeds its binary view")
    try:
        values = np.ndarray(
            shape=(count, component_count),
            dtype="<f4",
            buffer=binary,
            offset=offset,
            strides=(stride, 4),
        ).copy()
    except (TypeError, ValueError) as exc:
        raise OralValidationError("INVALID_GLB", "GLB accessor storage is invalid") from exc
    if not np.isfinite(values).all():
        raise OralValidationError("INVALID_GLB", "GLB geometry contains non-finite values")
    return values


def _load_glb_geometry(
    glb_path: str | Path,
    mapping_path: str | Path,
    adapter: GNMAdapter,
    topology: _OralTopology,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    document, binary = _glb_chunks(glb_path)
    try:
        meshes = document["meshes"]
        if len(meshes) != 1 or len(meshes[0]["primitives"]) != 1:
            raise OralValidationError(
                "INVALID_GLB", "Structural oral validation requires one GNM mesh primitive"
            )
        primitive = meshes[0]["primitives"][0]
        base = _float_accessor(document, binary, primitive["attributes"]["POSITION"])
        targets = primitive.get("targets", [])
        morph = np.stack(
            [_float_accessor(document, binary, target["POSITION"]) for target in targets]
        ) if targets else np.zeros((0, len(base), 3), dtype=np.float32)
    except (KeyError, IndexError, TypeError) as exc:
        if isinstance(exc, OralValidationError):
            raise
        raise OralValidationError(
            "GEOMETRY_ABSENT", "GLB has no complete GNM position geometry"
        ) from exc
    if base.shape[1:] != (3,) or morph.shape[1:] != base.shape:
        raise OralValidationError("INVALID_GLB", "GLB morph geometry is inconsistent")

    sidecar = Path(mapping_path)
    if not sidecar.is_file():
        raise OralValidationError(
            "GEOMETRY_ABSENT", "GLB-to-GNM mapping is required for oral geometry validation"
        )
    try:
        with np.load(sidecar, allow_pickle=False) as values:
            if "glb_vertex_to_gnm_vertex" not in values.files:
                raise OralValidationError(
                    "GEOMETRY_ABSENT", "GLB mapping does not identify native GNM oral vertices"
                )
            mapping = np.asarray(values["glb_vertex_to_gnm_vertex"], dtype=np.int64)
            weights = (
                np.asarray(values["morph_weights"], dtype=np.float32)
                if "morph_weights" in values.files
                else None
            )
            timestamps = (
                np.asarray(values["timestamps"], dtype=np.float64)
                if "timestamps" in values.files
                else None
            )
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, OralValidationError):
            raise
        raise OralValidationError("INVALID_GLB", "GLB mapping artifact is unreadable") from exc
    if (
        mapping.shape != (len(base),)
        or np.any(mapping < 0)
        or np.any(mapping >= adapter.model.num_vertices)
    ):
        raise OralValidationError("INVALID_GLB", "GLB-to-GNM vertex mapping is invalid")
    first = np.full(adapter.model.num_vertices, -1, dtype=np.int64)
    for glb_index, native_index in enumerate(mapping.tolist()):
        if first[native_index] < 0:
            first[native_index] = glb_index
    if np.any(first < 0):
        raise OralValidationError(
            "GEOMETRY_ABSENT", "GLB mapping does not cover every native GNM vertex"
        )
    duplicate_base_error = np.linalg.norm(base - base[first[mapping]], axis=1)
    if float(np.max(duplicate_base_error, initial=0.0)) > 1.0e-7:
        raise OralValidationError("INVALID_GLB", "Seam-duplicate GLB base vertices disagree")
    if len(morph):
        duplicate_morph_error = np.linalg.norm(
            morph - morph[:, first[mapping], :], axis=2
        )
        if float(np.max(duplicate_morph_error, initial=0.0)) > 1.0e-7:
            raise OralValidationError("INVALID_GLB", "Seam-duplicate GLB morph vertices disagree")

    if len(morph):
        if weights is None or timestamps is None:
            raise OralValidationError(
                "GEOMETRY_ABSENT", "Animated GLB mapping has no animation track"
            )
        if (
            weights.ndim != 2
            or weights.shape[1] != len(morph)
            or not len(weights)
            or not np.isfinite(weights).all()
        ):
            raise OralValidationError("INVALID_GLB", "GLB morph-weight track is invalid")
        times = _validate_timestamps(timestamps, len(weights))
        try:
            animations = document["animations"]
            if len(animations) != 1:
                raise OralValidationError("INVALID_GLB", "GLB must contain one morph animation")
            animation = animations[0]
            weight_channels = [
                channel
                for channel in animation["channels"]
                if channel["target"].get("path") == "weights"
            ]
            if len(weight_channels) != 1:
                raise OralValidationError("INVALID_GLB", "GLB has no unique weight animation")
            sampler = animation["samplers"][weight_channels[0]["sampler"]]
            glb_times = _float_accessor(document, binary, sampler["input"]).reshape(-1)
            glb_weights = _float_accessor(document, binary, sampler["output"]).reshape(
                len(times), len(morph)
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if isinstance(exc, OralValidationError):
                raise
            raise OralValidationError("INVALID_GLB", "GLB animation accessors are invalid") from exc
        if not np.allclose(glb_times, times, rtol=0.0, atol=1.0e-7) or not np.allclose(
            glb_weights, weights, rtol=0.0, atol=1.0e-7
        ):
            raise OralValidationError(
                "INVALID_GLB", "GLB animation and mapping sidecar disagree"
            )
    else:
        if weights is not None and (weights.ndim != 2 or weights.shape[1] != 0):
            raise OralValidationError("INVALID_GLB", "Static GLB has incompatible morph weights")
        if timestamps is None:
            times = np.asarray([0.0], dtype=np.float64)
        else:
            times = _validate_timestamps(timestamps, len(timestamps))
        weights = np.zeros((len(times), 0), dtype=np.float32)

    selected = first[topology.required_native]
    required_base = base[selected]
    required_morph = morph[:, selected]
    frames = required_base[None, :, :] + np.einsum(
        "tr,rvj->tvj", weights, required_morph, optimize=True
    )
    evidence = {
        "glb_sha256": _sha256(glb_path),
        "mapping_sha256": _sha256(mapping_path),
        "glb_vertex_count": int(len(base)),
        "native_vertex_coverage": int(np.count_nonzero(first >= 0)),
        "morph_target_count": int(len(morph)),
    }
    return frames.astype(np.float32), times, evidence


def validate_glb_oral_geometry(
    glb_path: str | Path,
    mapping_path: str | Path,
    *,
    adapter: GNMAdapter | None = None,
    reference_controls_path: str | Path | None = None,
    reference_frames: np.ndarray | None = None,
    identity: np.ndarray | None = None,
    thresholds: OralValidationThresholds = OralValidationThresholds(),
    batch_size: int = 64,
) -> OralValidationResult:
    """Measure a mapped GNM GLB and optionally audit it against source controls."""

    selected_adapter = adapter or GNMAdapter()
    topology = _topology(selected_adapter)
    frames, timestamps, evidence = _load_glb_geometry(
        glb_path, mapping_path, selected_adapter, topology
    )
    result = _analyze_required_geometry(
        frames,
        topology,
        timestamps=timestamps,
        thresholds=thresholds,
        source={
            "kind": "mapped_gnm_glb",
            "artifact_name": Path(glb_path).name,
            **evidence,
        },
    )
    report = result.report
    if reference_controls_path is None:
        if reference_frames is not None:
            raise OralValidationError(
                "REFERENCE_MISMATCH",
                "Reference frames require a controls artifact for timeline provenance",
            )
        return result

    controls = _load_controls(reference_controls_path, selected_adapter, identity)
    if reference_frames is None:
        reference = _evaluate_controls_required(
            controls, selected_adapter, topology, posed=True, batch_size=batch_size
        )
        reference_kind = "gnm_controls_npz_evaluated"
    else:
        provided = np.asarray(reference_frames)
        if provided.shape != (
            len(controls["expression"]),
            selected_adapter.model.num_vertices,
            3,
        ) or not np.isfinite(provided).all():
            raise OralValidationError(
                "REFERENCE_MISMATCH",
                "Provided GLB reference frames do not match the complete controls track",
            )
        reference = np.asarray(
            provided[:, topology.required_native], dtype=np.float32
        )
        reference_kind = "provided_complete_gnm_frames"
    reference_times = controls["timestamps"]
    if reference.shape != frames.shape or not np.allclose(
        reference_times, timestamps, rtol=0.0, atol=1.0e-6
    ):
        raise OralValidationError(
            "REFERENCE_MISMATCH", "GLB and reference controls are not frame/timestamp aligned"
        )
    oral_local = np.unique(
        np.concatenate(
            [
                topology.groups["tongue"],
                topology.groups["upper_teeth"],
                topology.groups["lower_teeth"],
                topology.groups["upper_lip"],
                topology.groups["lower_lip"],
            ]
        )
    )
    oral_error = np.linalg.norm(frames[:, oral_local] - reference[:, oral_local], axis=2)
    tongue_error = np.linalg.norm(
        frames[:, topology.groups["tongue"]]
        - reference[:, topology.groups["tongue"]],
        axis=2,
    )
    p95 = float(np.percentile(oral_error, 95))
    maximum = float(np.max(oral_error, initial=0.0))
    validated = (
        p95 <= thresholds.structural_reconstruction_p95_m
        and maximum <= thresholds.structural_reconstruction_max_m
    )
    report["structural_reconstruction"] = {
        "status": "passed" if validated else "failed",
        "validated": validated,
        "reference_kind": "gnm_controls_npz",
        "reference_evaluation_mode": reference_kind,
        "reference_sha256": _sha256(reference_controls_path),
        "frame_count": int(len(frames)),
        "oral_vertex_count": int(len(oral_local)),
        "oral_error_p95_mm": p95 * 1000.0,
        "oral_error_max_mm": maximum * 1000.0,
        "tongue_error_p95_mm": float(np.percentile(tongue_error, 95)) * 1000.0,
        "tongue_error_max_mm": float(np.max(tongue_error, initial=0.0)) * 1000.0,
        "p95_tolerance_mm": thresholds.structural_reconstruction_p95_m * 1000.0,
        "maximum_tolerance_mm": thresholds.structural_reconstruction_max_m * 1000.0,
        "interpretation": (
            "geometry reconstruction only; this does not validate phoneme timing or perception"
        ),
    }
    report["claims"]["structural_reconstruction_validated"] = validated
    return result


__all__ = [
    "SCHEMA_VERSION",
    "OralValidationError",
    "OralValidationResult",
    "OralValidationThresholds",
    "validate_controls_npz",
    "validate_glb_oral_geometry",
    "validate_oral_frames",
]
