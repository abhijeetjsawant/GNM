#!/usr/bin/env python3
"""Export the mean-shape locked-head SMPL-X body as the macap base model FBX.

Unlike ``export_mamma_smplx_neutral_fbx.py`` this export carries no fitted
identity.  Every beta is zero, so the delivered character is the SMPL-X
template with its 55-joint skeleton, normalized skinning weights, and UV map.
It is the reusable base a per-subject shape is later applied to.

Two modes are supported:

``raw``
    Reproduce the released model exactly.  SMPL-X ships a mirror-symmetric
    template but an asymmetric joint regressor and asymmetric skinning
    weights, so the delivered skeleton inherits a left elbow roughly three
    centimetres below the right.

``symmetric``
    Symmetrize the template, the joint regressor, and the skinning weights
    about the YZ plane before building the rig, then mirror bone rolls.  Left
    and right then share rest positions, bone lengths, local axes, and
    deformation.  ``shapedirs`` is deliberately left untouched: this base model
    is fitted at zero betas, so applying a subject shape later would need the
    shape basis symmetrized too.

Run through Blender:

    blender --background --python scripts/export_macap_base_model_fbx.py -- \
        SMPLX_NEUTRAL.npz smplx55-v1.json MODE OUTPUT.fbx REPORT.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_mamma_smplx_neutral_fbx import (  # noqa: E402
    _clear_scene,
    _scene_contract,
    _sha256,
)


CHARACTER_NAME = "macap-Base-model"
FORBIDDEN_BINARY_LABELS = (b"mamma", b"MAMMA")
BETA_COUNT = 16
MAX_INFLUENCES = 8
MODES = ("raw", "symmetric", "symmetric-v2")

# Explicit tails for every joint with more than one child, plus the leaves whose
# direction-from-parent fallback would otherwise pick an anatomically wrong axis.
# Keys are joint names; values are either a child joint name or a local offset.
BRANCH_TAIL_CHILD = {
    "pelvis": "spine1",
    "spine3": "neck",
    "left_wrist": "left_middle1",
    "right_wrist": "right_middle1",
}
STUB_TAIL_OFFSET_M = {
    "head": (0.0, 0.08, 0.0),
    "jaw": (0.0, -0.02, 0.04),
    "left_eye_smplhf": (0.0, 0.0, 0.03),
    "right_eye_smplhf": (0.0, 0.0, 0.03),
}
LEAF_STUB_LENGTH_M = 0.03

# SMPL-X regresses each spine joint independently, so the rest column zigzags
# forward and back by up to 28 degrees between consecutive segments. The chain
# is refitted to a smooth quadratic in Z against joint height, with the root and
# the head pinned: the root keeps world placement stable and the head carries
# the GNM neck seam. Joint heights and the mesh are never touched.
SPINE_CHAIN = ("pelvis", "spine1", "spine2", "spine3", "neck", "head")
SPINE_PINNED = ("pelvis", "head")
SPINE_FIT_DEGREE = 2
# Weight rebalancing blends the learned SMPL-X distribution with distance to the
# relocated bones. A pure geometric solve would discard SMPL-X's localization;
# a pure copy would leave falloff boundaries anchored to the old pivots.
SPINE_REBALANCE_BLEND = 0.5
SPINE_REBALANCE_FALLOFF_M = 0.02

# SMPL-X also regresses the ankle well medial of the leg it drives and the toe
# base close under the sole, leaving the two shallowest pivots in the rig. Both
# are recentred on their own local cross-section: the ankle within the slab at
# its height, the toe base within the slab at its fore-aft position, so the
# ball of the foot keeps its anatomical placement along the foot.
FOOT_SLAB_HALF_WIDTH_M = 0.010
FOOT_LEG_SPLIT_M = 0.02

# The knee is regressed two centimetres lateral of the leg it bends, which reads
# as a kink in the thigh-to-shin line from the front. Only its across-the-leg
# position is recentred: the knee's own fore-aft offset is behind the centre of
# the leg, which is where a flexion axis belongs, so it is left alone.
KNEE_SLAB_HALF_WIDTH_M = 0.012

# ``symmetric-v2`` only. SMPL-X regresses the hip joints 7.07% of stature apart
# and 8.4 cm below the crotch, which reads as short legs pivoting from inside the
# upper thigh. Biomechanics places the hip joint centre at roughly 9.7% of
# stature either side of the midline, at about pubic-symphysis height. Moving
# them there is a deliberate trade, not a defect fix: v1 stays faithful to the
# pose MAMMA solved against SMPL-X's own placement, v2 is anatomical.
#
# The fore-aft position is taken from the pelvis joint rather than left where
# SMPL-X had it. Raising the hip to crotch height leaves only about 1.5 cm of
# vertical run between root and hip, so any fore-aft difference reads from the
# side as a near-horizontal jog rather than a joint.
HIP_SEPARATION_FRACTION = 0.097
HIP_MIDLINE_HALF_WIDTH_M = 0.015

# ``symmetric-v2`` only. The elbow sits about a centimetre below the axis of the
# arm it bends. Cross-sections are taken perpendicular to the arm rather than by
# height, because the arm is horizontal in the rest pose.
ELBOW_SLAB_HALF_WIDTH_M = 0.012
ARM_TORSO_SPLIT_M = 0.20
# Recentring a joint inside an out-of-round limb can cost a little clearance -
# the elbow currently sits in the thicker part of its section. Requiring
# clearance to strictly increase would block a correct move, so the gate is a
# floor: a recentred joint may not end up meaningfully shallower than it started.
CLEARANCE_FLOOR_FRACTION = 0.9


def _arguments() -> tuple[Path, Path, str, Path, Path]:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit(
            "Expected MODEL.npz SKELETON.json MODE OUTPUT.fbx REPORT.json after '--'"
        ) from exc
    values = sys.argv[separator + 1 :]
    if len(values) != 5 or values[2] not in MODES:
        raise SystemExit(
            "Expected MODEL.npz SKELETON.json "
            + "|".join(MODES)
            + " OUTPUT.fbx REPORT.json after '--'"
        )
    model, skeleton = (Path(value).expanduser().resolve() for value in values[:2])
    mode = values[2]
    output, report = (Path(value).expanduser().resolve() for value in values[3:])
    for source in (model, skeleton):
        if not source.is_file():
            raise SystemExit(f"Required source file is missing: {source}")
    if output.suffix.lower() != ".fbx":
        raise SystemExit(f"Output must use the .fbx extension: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return model, skeleton, mode, output, report


def _mirror_joint_map(names: tuple[str, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Map each joint index to its left/right partner, midline joints to itself."""

    index = {name: position for position, name in enumerate(names)}
    mirror = np.arange(len(names), dtype=np.int64)
    midline: list[int] = []
    for name, position in index.items():
        if name.startswith("left_"):
            mirror[position] = index["right_" + name[len("left_") :]]
        elif name.startswith("right_"):
            mirror[position] = index["left_" + name[len("right_") :]]
        else:
            midline.append(position)
    if not (mirror[mirror] == np.arange(len(names))).all():
        raise RuntimeError("Joint mirror map is not an involution")
    return mirror, tuple(sorted(midline))


def _mirror_error(values: np.ndarray, mirror: np.ndarray) -> float:
    """Largest deviation between a point set and its own mirrored partner set."""

    flipped = values[mirror].copy()
    flipped[:, 0] *= -1.0
    return float(np.abs(values - flipped).max())


def _symmetrize_model(
    vertices: np.ndarray,
    regressor: np.ndarray,
    weights: np.ndarray,
    vertex_mirror: np.ndarray,
    joint_mirror: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the template, regressor, and weights against their own mirror.

    Symmetrizing the operators rather than only the baked rest joints keeps the
    regressor and the skin usable: any later evaluation stays symmetric instead
    of only the one skeleton written into this file.
    """

    mirrored_vertices = vertices[vertex_mirror].copy()
    mirrored_vertices[:, 0] *= -1.0
    symmetric_vertices = 0.5 * (vertices + mirrored_vertices)

    # Mirror the regressor across both axes: joint rows and vertex columns.
    symmetric_regressor = 0.5 * (regressor + regressor[joint_mirror][:, vertex_mirror])
    symmetric_weights = 0.5 * (weights + weights[vertex_mirror][:, joint_mirror])
    return symmetric_vertices, symmetric_regressor, symmetric_weights


def _spine_leans(names: tuple[str, ...], joints: np.ndarray) -> list[dict[str, object]]:
    """Per-segment forward/back lean of the spine chain, in degrees from vertical."""

    index = {name: position for position, name in enumerate(names)}
    leans = []
    for upper, lower in zip(SPINE_CHAIN[:-1], SPINE_CHAIN[1:]):
        delta = joints[index[lower]] - joints[index[upper]]
        leans.append(
            {
                "segment": f"{upper}->{lower}",
                "lean_deg": float(np.degrees(np.arctan2(delta[2], delta[1]))),
            }
        )
    return leans


def _correct_spine(
    names: tuple[str, ...], joints: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...], dict[str, float]]:
    """Refit the spine column to a smooth curve without moving joint heights."""

    index = {name: position for position, name in enumerate(names)}
    positions = [index[name] for name in SPINE_CHAIN]
    heights = joints[positions, 1]
    depths = joints[positions, 2]
    coefficients = np.polyfit(heights, depths, SPINE_FIT_DEGREE)
    fitted = np.polyval(coefficients, heights)
    for slot, name in enumerate(SPINE_CHAIN):
        if name in SPINE_PINNED:
            fitted[slot] = depths[slot]

    corrected = np.array(joints, dtype=np.float64, copy=True)
    corrected[positions, 2] = fitted
    moved = tuple(
        position
        for slot, position in enumerate(positions)
        if abs(float(fitted[slot] - depths[slot])) > 0.0
    )
    metrics = {
        "max_joint_move_m": float(np.abs(fitted - depths).max()),
        "worst_lean_before_deg": max(
            abs(entry["lean_deg"]) for entry in _spine_leans(names, joints)
        ),
        "worst_lean_after_deg": max(
            abs(entry["lean_deg"]) for entry in _spine_leans(names, corrected)
        ),
    }
    return corrected, moved, metrics


def _skin_clearance(vertices: np.ndarray, point: np.ndarray) -> float:
    """Distance from a joint to the nearest skin vertex."""

    return float(np.linalg.norm(vertices - point, axis=1).min())


def _correct_feet(
    names: tuple[str, ...],
    joints: np.ndarray,
    vertices: np.ndarray,
    joint_mirror: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, object]]:
    """Recentre the ankle and toe-base joints inside the limb they deform.

    Only the left side is measured; the right is written as its exact mirror, so
    the correction cannot introduce the asymmetry the rig just removed.
    """

    index = {name: position for position, name in enumerate(names)}
    ankle, toe = index["left_ankle"], index["left_foot"]
    corrected = np.array(joints, dtype=np.float64, copy=True)
    report: dict[str, object] = {}

    # Ankle: centre it within the leg cross-section at its own height.
    slab = vertices[
        (np.abs(vertices[:, 1] - joints[ankle, 1]) < FOOT_SLAB_HALF_WIDTH_M)
        & (vertices[:, 0] > FOOT_LEG_SPLIT_M)
    ]
    if len(slab) < 8:
        raise RuntimeError("Ankle cross-section slab is too sparse to recentre")
    new_ankle = np.array(
        [
            0.5 * (slab[:, 0].min() + slab[:, 0].max()),
            joints[ankle, 1],
            0.5 * (slab[:, 2].min() + slab[:, 2].max()),
        ]
    )

    # Toe base: keep its fore-aft position, which is what makes it the ball of
    # the foot, and centre it within the foot cross-section there.
    slab = vertices[
        (np.abs(vertices[:, 2] - joints[toe, 2]) < FOOT_SLAB_HALF_WIDTH_M)
        & (vertices[:, 0] > FOOT_LEG_SPLIT_M)
        & (vertices[:, 1] < joints[ankle, 1])
    ]
    if len(slab) < 8:
        raise RuntimeError("Toe-base cross-section slab is too sparse to recentre")
    new_toe = np.array(
        [
            0.5 * (slab[:, 0].min() + slab[:, 0].max()),
            0.5 * (slab[:, 1].min() + slab[:, 1].max()),
            joints[toe, 2],
        ]
    )

    mirror = np.array([-1.0, 1.0, 1.0])
    for position, target in ((ankle, new_ankle), (toe, new_toe)):
        report[names[position]] = {
            "clearance_before_m": _skin_clearance(vertices, joints[position]),
            "clearance_after_m": _skin_clearance(vertices, target),
            "move_m": float(np.linalg.norm(target - joints[position])),
        }
        corrected[position] = target
        corrected[int(joint_mirror[position])] = target * mirror

    moved = (ankle, toe, int(joint_mirror[ankle]), int(joint_mirror[toe]))
    return corrected, moved, report


def _correct_knees(
    names: tuple[str, ...],
    joints: np.ndarray,
    vertices: np.ndarray,
    joint_mirror: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, object]]:
    """Centre each knee across the leg it bends, left measured and right mirrored."""

    index = {name: position for position, name in enumerate(names)}
    knee = index["left_knee"]
    corrected = np.array(joints, dtype=np.float64, copy=True)

    slab = vertices[
        (np.abs(vertices[:, 1] - joints[knee, 1]) < KNEE_SLAB_HALF_WIDTH_M)
        & (vertices[:, 0] > FOOT_LEG_SPLIT_M)
    ]
    if len(slab) < 8:
        raise RuntimeError("Knee cross-section slab is too sparse to recentre")
    target = np.array(
        [0.5 * (slab[:, 0].min() + slab[:, 0].max()), joints[knee, 1], joints[knee, 2]]
    )

    report = {
        names[knee]: {
            "clearance_before_m": _skin_clearance(vertices, joints[knee]),
            "clearance_after_m": _skin_clearance(vertices, target),
            "move_m": float(np.linalg.norm(target - joints[knee])),
        }
    }
    corrected[knee] = target
    corrected[int(joint_mirror[knee])] = target * np.array([-1.0, 1.0, 1.0])
    return corrected, (knee, int(joint_mirror[knee])), report


def _correct_hips(
    names: tuple[str, ...],
    joints: np.ndarray,
    vertices: np.ndarray,
    joint_mirror: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, object]]:
    """Move the hip joints out and up to an anatomical hip-joint centre."""

    index = {name: position for position, name in enumerate(names)}
    hip, knee, pelvis = index["left_hip"], index["left_knee"], index["pelvis"]
    stature = float(vertices[:, 1].max() - vertices[:, 1].min())
    midline = vertices[
        (np.abs(vertices[:, 0]) < HIP_MIDLINE_HALF_WIDTH_M)
        & (vertices[:, 1] < joints[pelvis, 1])
    ]
    if len(midline) < 8:
        raise RuntimeError("Cannot locate the crotch to place the hip joints")
    crotch = float(midline[:, 1].max())

    corrected = np.array(joints, dtype=np.float64, copy=True)
    target = np.array(
        [0.5 * HIP_SEPARATION_FRACTION * stature, crotch, joints[pelvis, 2]]
    )
    report = {
        "left_hip": {
            "clearance_before_m": _skin_clearance(vertices, joints[hip]),
            "clearance_after_m": _skin_clearance(vertices, target),
            "move_m": float(np.linalg.norm(target - joints[hip])),
        },
        "stature_m": stature,
        "crotch_height_m": crotch,
        "separation_before_fraction_of_stature": float(abs(joints[hip, 0]) * 2 / stature),
        "separation_after_fraction_of_stature": HIP_SEPARATION_FRACTION,
        "thigh_before_m": float(np.linalg.norm(joints[knee] - joints[hip])),
        "thigh_after_m": float(np.linalg.norm(joints[knee] - target)),
        "root_to_hip_lean_before_deg": float(
            np.degrees(
                np.arctan2(
                    abs(joints[hip, 2] - joints[pelvis, 2]),
                    abs(joints[hip, 1] - joints[pelvis, 1]),
                )
            )
        ),
        "root_to_hip_lean_after_deg": float(
            np.degrees(
                np.arctan2(
                    abs(target[2] - joints[pelvis, 2]),
                    abs(target[1] - joints[pelvis, 1]),
                )
            )
        ),
    }
    corrected[hip] = target
    corrected[int(joint_mirror[hip])] = target * np.array([-1.0, 1.0, 1.0])
    return corrected, (hip, int(joint_mirror[hip])), report


def _correct_elbows(
    names: tuple[str, ...],
    joints: np.ndarray,
    vertices: np.ndarray,
    joint_mirror: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, object]]:
    """Centre each elbow on the axis of the arm it bends."""

    index = {name: position for position, name in enumerate(names)}
    elbow = index["left_elbow"]
    corrected = np.array(joints, dtype=np.float64, copy=True)

    slab = vertices[
        (np.abs(vertices[:, 0] - joints[elbow, 0]) < ELBOW_SLAB_HALF_WIDTH_M)
        & (vertices[:, 0] > ARM_TORSO_SPLIT_M)
    ]
    if len(slab) < 8:
        raise RuntimeError("Elbow cross-section slab is too sparse to recentre")
    target = np.array(
        [
            joints[elbow, 0],
            0.5 * (slab[:, 1].min() + slab[:, 1].max()),
            0.5 * (slab[:, 2].min() + slab[:, 2].max()),
        ]
    )
    report = {
        "left_elbow": {
            "clearance_before_m": _skin_clearance(vertices, joints[elbow]),
            "clearance_after_m": _skin_clearance(vertices, target),
            "move_m": float(np.linalg.norm(target - joints[elbow])),
        }
    }
    corrected[elbow] = target
    corrected[int(joint_mirror[elbow])] = target * np.array([-1.0, 1.0, 1.0])
    return corrected, (elbow, int(joint_mirror[elbow])), report


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = end - start
    length_squared = float(axis @ axis)
    if length_squared <= 0.0:
        return np.linalg.norm(points - start, axis=1)
    travel = np.clip(((points - start) @ axis) / length_squared, 0.0, 1.0)
    return np.linalg.norm(points - (start + travel[:, None] * axis), axis=1)


def _rebalance_joint_weights(
    weights: np.ndarray,
    vertices: np.ndarray,
    joints: np.ndarray,
    moved: tuple[int, ...],
    tails: dict[int, np.ndarray],
    vertex_mirror: np.ndarray,
    joint_mirror: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Re-anchor skin weights onto a set of relocated bones.

    Only columns for joints that actually moved are rewritten, and each vertex
    keeps exactly the influence mass it already assigned to them, so every other
    joint's weights and the partition of unity are untouched. The rest mesh is
    unaffected either way: at rest the skin evaluates to the template.
    """

    if not moved:
        return weights, {"max_weight_delta": 0.0, "mean_weight_delta": 0.0, "vertices_changed": 0}

    columns = list(moved)
    if sorted(int(joint_mirror[joint]) for joint in columns) != sorted(columns):
        raise RuntimeError("Rebalanced joint set is not closed under mirroring")
    column_slot = {joint: slot for slot, joint in enumerate(columns)}
    mirrored_slots = [column_slot[int(joint_mirror[joint])] for joint in columns]
    mass = weights[:, columns].sum(axis=1)

    affinity = np.zeros((len(vertices), len(columns)), dtype=np.float64)
    for slot, joint in enumerate(columns):
        distance = _point_segment_distance(vertices, joints[joint], tails[joint])
        affinity[:, slot] = 1.0 / (distance**2 + SPINE_REBALANCE_FALLOFF_M**2)
    # A mirrored vertex is equidistant from the mirrored bone, so averaging the
    # affinity against its own mirror is exact and keeps the result symmetric
    # regardless of floating-point ordering.
    affinity = 0.5 * (affinity + affinity[vertex_mirror][:, mirrored_slots])
    affinity /= affinity.sum(axis=1, keepdims=True)

    original = weights[:, columns].copy()
    active = mass > 0.0
    prior = np.zeros_like(original)
    prior[active] = original[active] / mass[active, None]
    blended = (1.0 - SPINE_REBALANCE_BLEND) * prior + SPINE_REBALANCE_BLEND * affinity
    blended /= blended.sum(axis=1, keepdims=True)

    rebalanced = np.array(weights, dtype=np.float64, copy=True)
    rebalanced[:, columns] = blended * mass[:, None]
    rebalanced[~active][:, columns] = 0.0

    delta = np.abs(rebalanced - weights)
    return rebalanced, {
        "max_weight_delta": float(delta.max()),
        "mean_weight_delta": float(delta[delta > 0.0].mean()) if delta.max() > 0.0 else 0.0,
        "vertices_changed": int((delta.max(axis=1) > 1e-12).sum()),
        "max_untouched_column_delta": float(
            np.abs(
                np.delete(rebalanced, columns, axis=1) - np.delete(weights, columns, axis=1)
            ).max()
        ),
    }


def _limit_influences(
    weights: np.ndarray, vertex_mirror: np.ndarray, joint_mirror: np.ndarray
) -> tuple[np.ndarray, int]:
    """Renormalize skin weights and cap them at the production influence budget.

    Averaging two mirrored weight rows takes the union of their support, so a
    vertex can gain influences it never had. Ranking by weight alone would then
    break symmetry again, because equal weights are ordered by joint index. Each
    mirrored pair therefore shares one decision made on its lower-indexed
    vertex, and a self-mirroring midline vertex keeps only influences that
    survive on both sides.
    """

    limited = np.array(weights, dtype=np.float64, copy=True)
    limited[limited < 0.0] = 0.0
    counts = (limited > 0.0).sum(axis=1)
    masks: dict[int, np.ndarray] = {}
    pruned = 0
    for vertex in np.flatnonzero(counts > MAX_INFLUENCES):
        vertex = int(vertex)
        canonical = min(vertex, int(vertex_mirror[vertex]))
        if canonical not in masks:
            row = limited[canonical]
            mask = np.zeros(row.shape, dtype=bool)
            mask[np.argsort(-row, kind="stable")[:MAX_INFLUENCES]] = True
            if int(vertex_mirror[canonical]) == canonical:
                mask &= mask[joint_mirror]
            masks[canonical] = mask
        mask = masks[canonical] if vertex == canonical else masks[canonical][joint_mirror]
        limited[vertex, ~mask] = 0.0
        pruned += 1
    totals = limited.sum(axis=1, keepdims=True)
    if not np.all(totals > 1e-8):
        raise RuntimeError("A vertex lost all skinning influence during symmetrization")
    limited /= totals
    return limited, pruned


def _load_sources(model_path: Path, skeleton_path: Path, mode: str) -> dict[str, object]:
    """Load the SMPL-X template at its mean shape, optionally symmetrized."""

    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    names = tuple(str(value) for value in skeleton["joint_names"])
    parents = np.asarray(skeleton["parents"], dtype=np.int64)
    if len(names) != 55 or parents.shape != (55,):
        raise RuntimeError("Expected the exact 55-joint SMPL-X contract")

    # allow_pickle is required because the released SMPL-X archive stores
    # ``joint2num``/``part2num`` as object arrays. The input is the licensed MPI
    # locked-head model already on disk, and its SHA-256 is recorded in the
    # report; no untrusted archive reaches this loader.
    with np.load(model_path, allow_pickle=True) as model:
        vertices = np.asarray(model["v_template"], dtype=np.float64)
        regressor = np.asarray(model["J_regressor"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        faces = np.asarray(model["f"], dtype=np.int32)
        texture_vertices = np.asarray(model["vt"], dtype=np.float64)
        texture_faces = np.asarray(model["ft"], dtype=np.int32)
        baked_joints = np.asarray(model["J"], dtype=np.float64)
        kintree = np.asarray(model["kintree_table"], dtype=np.int64)
        vertex_mirror = np.asarray(model["vert_sym_idxs"], dtype=np.int64)

    if vertices.shape != (10475, 3) or faces.shape != (20908, 3):
        raise RuntimeError("Locked-head SMPL-X topology differs from the expected model")
    if weights.shape != (10475, 55) or regressor.shape != (55, 10475):
        raise RuntimeError("Locked-head SMPL-X skeleton or weights differ")
    if texture_faces.shape != faces.shape:
        raise RuntimeError("SMPL-X UV topology differs from the geometry")
    if vertex_mirror.shape != (10475,):
        raise RuntimeError("SMPL-X vertex mirror table is missing or malformed")
    if not (vertex_mirror[vertex_mirror] == np.arange(len(vertices))).all():
        raise RuntimeError("SMPL-X vertex mirror table is not an involution")

    joint_mirror, midline = _mirror_joint_map(names)
    before = {
        "mesh_mirror_error_m": _mirror_error(vertices, vertex_mirror),
        "joint_mirror_error_m": _mirror_error(regressor @ vertices, joint_mirror),
        "weight_mirror_error": float(
            np.abs(weights - weights[vertex_mirror][:, joint_mirror]).max()
        ),
        "midline_joint_offset_m": float(
            np.abs((regressor @ vertices)[list(midline), 0]).max()
        ),
    }

    symmetric = mode != "raw"
    pruned = 0
    joints = regressor @ vertices
    leans_before = _spine_leans(names, joints)
    spine_metrics: dict[str, float] = {}
    rebalance: dict[str, float] = {}
    foot_metrics: dict[str, object] = {}
    foot_rebalance: dict[str, float] = {}
    knee_metrics: dict[str, object] = {}
    knee_rebalance: dict[str, float] = {}
    hip_metrics: dict[str, object] = {}
    hip_rebalance: dict[str, float] = {}
    elbow_metrics: dict[str, object] = {}
    elbow_rebalance: dict[str, float] = {}
    if symmetric:
        vertices, regressor, weights = _symmetrize_model(
            vertices, regressor, weights, vertex_mirror, joint_mirror
        )
        joints = regressor @ vertices
        leans_before = _spine_leans(names, joints)
        index = {name: position for position, name in enumerate(names)}
        joints, spine_moved, spine_metrics = _correct_spine(names, joints)
        successor = {
            index[upper]: index[lower]
            for upper, lower in zip(SPINE_CHAIN[:-1], SPINE_CHAIN[1:])
        }
        weights, rebalance = _rebalance_joint_weights(
            weights,
            vertices,
            joints,
            spine_moved,
            {joint: joints[successor[joint]] for joint in spine_moved},
            vertex_mirror,
            joint_mirror,
        )
        joints, foot_moved, foot_metrics = _correct_feet(
            names, joints, vertices, joint_mirror
        )
        foot_tails: dict[int, np.ndarray] = {}
        for side in ("left", "right"):
            ankle, toe = index[f"{side}_ankle"], index[f"{side}_foot"]
            foot_tails[ankle] = joints[toe]
            direction = joints[toe] - joints[ankle]
            length = float(np.linalg.norm(direction))
            if length <= 1e-6:
                raise RuntimeError(f"{side} toe base coincides with its ankle")
            foot_tails[toe] = joints[toe] + direction / length * LEAF_STUB_LENGTH_M
        weights, foot_rebalance = _rebalance_joint_weights(
            weights, vertices, joints, foot_moved, foot_tails, vertex_mirror, joint_mirror
        )
        joints, knee_moved, knee_metrics = _correct_knees(
            names, joints, vertices, joint_mirror
        )
        weights, knee_rebalance = _rebalance_joint_weights(
            weights,
            vertices,
            joints,
            knee_moved,
            {joint: joints[index["%s_ankle" % names[joint].split("_")[0]]] for joint in knee_moved},
            vertex_mirror,
            joint_mirror,
        )
        if mode == "symmetric-v2":
            joints, hip_moved, hip_metrics = _correct_hips(
                names, joints, vertices, joint_mirror
            )
            weights, hip_rebalance = _rebalance_joint_weights(
                weights,
                vertices,
                joints,
                hip_moved,
                {joint: joints[index["%s_knee" % names[joint].split("_")[0]]] for joint in hip_moved},
                vertex_mirror,
                joint_mirror,
            )
            joints, elbow_moved, elbow_metrics = _correct_elbows(
                names, joints, vertices, joint_mirror
            )
            weights, elbow_rebalance = _rebalance_joint_weights(
                weights,
                vertices,
                joints,
                elbow_moved,
                {joint: joints[index["%s_wrist" % names[joint].split("_")[0]]] for joint in elbow_moved},
                vertex_mirror,
                joint_mirror,
            )
        weights, pruned = _limit_influences(weights, vertex_mirror, joint_mirror)
    leans_after = _spine_leans(names, joints)

    after = {
        "mesh_mirror_error_m": _mirror_error(vertices, vertex_mirror),
        "joint_mirror_error_m": _mirror_error(joints, joint_mirror),
        "weight_mirror_error": float(
            np.abs(weights - weights[vertex_mirror][:, joint_mirror]).max()
        ),
        "midline_joint_offset_m": float(np.abs(joints[list(midline), 0]).max()),
    }

    if not all(
        np.isfinite(value).all()
        for value in (vertices, joints, weights, texture_vertices)
    ):
        raise RuntimeError("SMPL-X base asset contains non-finite values")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("SMPL-X skinning weights are not normalized")

    # The model ships both a baked rest skeleton and the regressor that derives
    # it. They agree for the released mean shape; the residual is recorded
    # rather than enforced so a differing locked-head variant stays visible.
    model_parents = kintree[0].copy()
    model_parents[0] = -1

    return {
        "names": names,
        "parents": parents,
        "joint_mirror": joint_mirror,
        "midline_joints": midline,
        "vertices": vertices.astype(np.float32),
        "faces": faces,
        "joints": joints.astype(np.float32),
        "weights": weights.astype(np.float32),
        "texture_vertices": texture_vertices.astype(np.float32),
        "texture_faces": texture_faces,
        "betas": np.zeros(BETA_COUNT, dtype=np.float32),
        "joint_regressor_residual_m": float(
            np.abs((regressor @ vertices) - baked_joints).max()
        ),
        "model_parents_match_contract": bool(
            (model_parents.astype(np.int64) == parents).all()
        ),
        "max_influences_per_vertex": int((weights > 1e-4).sum(axis=1).max()),
        "vertices_pruned_to_influence_budget": pruned,
        "symmetry_before": before,
        "symmetry_after": after,
        "spine_corrected": symmetric,
        "spine_leans_before": leans_before,
        "spine_leans_after": leans_after,
        "spine_metrics": spine_metrics,
        "spine_reskin": rebalance,
        "feet_corrected": symmetric,
        "feet_metrics": foot_metrics,
        "feet_reskin": foot_rebalance,
        "knees_corrected": symmetric,
        "knees_metrics": knee_metrics,
        "knees_reskin": knee_rebalance,
        "hips_corrected": mode == "symmetric-v2",
        "hips_metrics": hip_metrics,
        "hips_reskin": hip_rebalance,
        "elbows_corrected": mode == "symmetric-v2",
        "elbows_metrics": elbow_metrics,
        "elbows_reskin": elbow_rebalance,
        "template_vertices": vertices.astype(np.float64),
    }


def _bone_tails(
    names: tuple[str, ...], parents: np.ndarray, joints: np.ndarray
) -> dict[str, Vector]:
    """Resolve one explicit tail per bone.

    Branching joints use a fixed anatomical child instead of a most-vertical-
    child heuristic, which would otherwise aim the head bone at the left eye and
    break left/right parity.
    """

    index = {name: position for position, name in enumerate(names)}
    tails: dict[str, Vector] = {}
    for position, name in enumerate(names):
        head = Vector(joints[position].tolist())
        if name in STUB_TAIL_OFFSET_M:
            tails[name] = head + Vector(STUB_TAIL_OFFSET_M[name])
            continue
        if name in BRANCH_TAIL_CHILD:
            tails[name] = Vector(joints[index[BRANCH_TAIL_CHILD[name]]].tolist())
            continue
        children = np.flatnonzero(parents == position).astype(int).tolist()
        if len(children) > 1:
            raise RuntimeError(f"Branching joint {name} has no explicit tail rule")
        if children:
            tails[name] = Vector(joints[children[0]].tolist())
            continue
        parent = int(parents[position])
        direction = head - Vector(joints[parent].tolist()) if parent >= 0 else Vector((0.0, 1.0, 0.0))
        if direction.length < 1e-5:
            direction = Vector((0.0, 1.0, 0.0))
        direction.normalize()
        tails[name] = head + direction * LEAF_STUB_LENGTH_M
    for name, tail in tails.items():
        if (tail - Vector(joints[index[name]].tolist())).length < 1e-5:
            raise RuntimeError(f"Bone {name} resolved to a zero-length tail")
    return tails


def _build_character(data: dict[str, object], mirror_rolls: bool) -> None:
    names = data["names"]
    parents = data["parents"]
    vertices = data["vertices"]
    faces = data["faces"]
    joints = data["joints"]
    weights = data["weights"]

    mesh_data = bpy.data.meshes.new(f"{CHARACTER_NAME}-Body-Mesh")
    mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh_data.update(calc_edges=True)
    mesh_object = bpy.data.objects.new(f"{CHARACTER_NAME}-Body", mesh_data)
    bpy.context.collection.objects.link(mesh_object)

    uv_layer = mesh_data.uv_layers.new(name="SMPLX_UV")
    texture_vertices = data["texture_vertices"]
    texture_faces = data["texture_faces"]
    for polygon in mesh_data.polygons:
        for corner, loop_index in enumerate(polygon.loop_indices):
            uv = texture_vertices[texture_faces[polygon.index, corner]]
            uv_layer.data[loop_index].uv = (float(uv[0]), float(uv[1]))

    material = bpy.data.materials.new(f"{CHARACTER_NAME}-Material-00")
    material.diffuse_color = (0.55, 0.30, 0.23, 1.0)
    material.roughness = 0.82
    mesh_data.materials.append(material)

    armature_data = bpy.data.armatures.new(f"{CHARACTER_NAME}-Skeleton")
    armature = bpy.data.objects.new(CHARACTER_NAME, armature_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    tails = _bone_tails(names, parents, joints)
    edit_bones = []
    for index, name in enumerate(names):
        bone = armature_data.edit_bones.new(name)
        bone.head = Vector(joints[index].tolist())
        bone.tail = tails[name]
        bone.use_connect = False
        edit_bones.append(bone)
    for index, parent in enumerate(parents):
        if parent >= 0:
            edit_bones[index].parent = edit_bones[int(parent)]
    if mirror_rolls:
        # Bone heads and tails are exact mirrors here, so negating the roll of
        # the left bone gives the right bone a mirrored local frame. The export
        # verifies this numerically rather than trusting the convention.
        joint_mirror = data["joint_mirror"]
        for index, name in enumerate(names):
            if name.startswith("right_"):
                edit_bones[index].roll = -edit_bones[int(joint_mirror[index])].roll
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.data.pose_position = "REST"

    for joint_index, name in enumerate(names):
        group = mesh_object.vertex_groups.new(name=name)
        joint_weights = weights[:, joint_index]
        indices = np.flatnonzero(joint_weights > 1e-8)
        for vertex_index in indices:
            group.add([int(vertex_index)], float(joint_weights[vertex_index]), "REPLACE")
    modifier = mesh_object.modifiers.new(name=f"{CHARACTER_NAME}-Skin", type="ARMATURE")
    modifier.object = armature
    mesh_object.parent = armature
    bpy.context.view_layer.update()


def _axis_mirror_errors(names: tuple[str, ...], joint_mirror: np.ndarray) -> dict[str, float]:
    """Largest left/right disagreement in each bone's rest local axes."""

    armature = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"][0]
    bones = {bone.name: bone for bone in armature.data.bones}
    worst: dict[str, float] = {}
    for index, name in enumerate(names):
        if not name.startswith("left_"):
            continue
        partner = names[int(joint_mirror[index])]
        error = 0.0
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            left = bones[name].matrix_local.to_3x3() @ Vector(axis)
            right = bones[partner].matrix_local.to_3x3() @ Vector(axis)
            mirrored = Vector((-right.x, right.y, right.z))
            # The X axis runs along the bone, which mirrors with a sign flip.
            error = max(error, min((left - mirrored).length, (left + mirrored).length))
        worst[name] = error
    return worst


def _forbidden_labels(path: Path) -> list[str]:
    payload = path.read_bytes()
    return [label.decode("ascii") for label in FORBIDDEN_BINARY_LABELS if label in payload]


def main() -> None:
    model_path, skeleton_path, mode, output, report_path = _arguments()
    _clear_scene()
    data = _load_sources(model_path, skeleton_path, mode)
    _build_character(data, mirror_rolls=mode == "symmetric")
    source_contract = _scene_contract()
    source_axis_errors = _axis_mirror_errors(data["names"], data["joint_mirror"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(output),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        axis_forward="-Z",
        axis_up="Y",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        bake_anim=False,
        path_mode="AUTO",
        embed_textures=False,
    )
    if not output.is_file() or output.stat().st_size < 100_000:
        raise RuntimeError("macap base model FBX is missing or unexpectedly small")
    leaked = _forbidden_labels(output)

    _clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(output), automatic_bone_orientation=False)
    roundtrip = _scene_contract()
    roundtrip_axis_errors = _axis_mirror_errors(data["names"], data["joint_mirror"])
    mesh = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    # "The body mesh stays as is" is a delivery constraint, so the rest surface
    # of the round-tripped file is compared against the template it was built
    # from rather than assumed unchanged by the spine edit.
    imported_vertices = np.empty(len(mesh.data.vertices) * 3, dtype=np.float64)
    mesh.data.vertices.foreach_get("co", imported_vertices)
    rest_mesh_error = float(
        np.abs(imported_vertices.reshape(-1, 3) - data["template_vertices"]).max()
    )
    weight_totals = [
        sum(group.weight for group in vertex.groups) for vertex in mesh.data.vertices
    ]
    worst_weight_total = max(abs(total - 1.0) for total in weight_totals)
    worst_influences = max(len(vertex.groups) for vertex in mesh.data.vertices)

    failures = []
    for key in ("bone_count", "bone_names", "bone_parents", "root_bones"):
        if roundtrip[key] != source_contract[key]:
            failures.append(f"{key} changed during FBX round trip")
    if roundtrip["vertices"] != 10475 or roundtrip["triangles"] != 20908:
        failures.append("locked-head SMPL-X topology changed")
    if roundtrip["uv_layers"] < 1:
        failures.append("SMPL-X UV map is absent")
    if roundtrip["vertex_groups"] != 55 or roundtrip["armature_modifiers"] != 1:
        failures.append("SMPL-X skinning contract changed")
    if roundtrip["actions"] != 0 or roundtrip["pose_basis_error"] > 1e-6:
        failures.append("macap base model FBX is not static in rest pose")
    if worst_weight_total > 1e-4:
        failures.append("skin weights do not sum to one after the FBX round trip")
    if worst_influences > MAX_INFLUENCES:
        failures.append(f"a vertex exceeds {MAX_INFLUENCES} influences after the round trip")
    if leaked:
        failures.append(f"forbidden provider labels present: {', '.join(leaked)}")
    if not data["model_parents_match_contract"]:
        failures.append("SMPL-X kintree parents differ from the 55-joint contract")
    if mode != "raw":
        after = data["symmetry_after"]
        if after["mesh_mirror_error_m"] > 1e-9:
            failures.append("symmetrized mesh is not mirror symmetric")
        if after["joint_mirror_error_m"] > 1e-9:
            failures.append("symmetrized skeleton is not mirror symmetric")
        if after["weight_mirror_error"] > 1e-6:
            failures.append("symmetrized skin weights are not mirror symmetric")
        if after["midline_joint_offset_m"] > 1e-9:
            failures.append("midline joints are not centred on the YZ plane")
        worst_axis = max(roundtrip_axis_errors.values())
        if worst_axis > 1e-3:
            failures.append(f"left/right bone axes differ by {worst_axis:.6f}")
        if data["spine_metrics"]["worst_lean_after_deg"] > 10.0:
            failures.append("spine column still kinks after refitting")
        if data["spine_reskin"].get("max_untouched_column_delta", 0.0) > 0.0:
            failures.append("spine reskin altered weights outside the relocated joints")
        if data["feet_reskin"].get("max_untouched_column_delta", 0.0) > 0.0:
            failures.append("foot reskin altered weights outside the relocated joints")
        if data["knees_reskin"].get("max_untouched_column_delta", 0.0) > 0.0:
            failures.append("knee reskin altered weights outside the relocated joints")
        if data["hips_reskin"].get("max_untouched_column_delta", 0.0) > 0.0:
            failures.append("hip reskin altered weights outside the relocated joints")
        if data["elbows_reskin"].get("max_untouched_column_delta", 0.0) > 0.0:
            failures.append("elbow reskin altered weights outside the relocated joints")
        for entry_name in ("feet_metrics", "knees_metrics", "hips_metrics", "elbows_metrics"):
            for joint, entry in data[entry_name].items():
                if not isinstance(entry, dict):
                    continue
                floor = CLEARANCE_FLOOR_FRACTION * entry["clearance_before_m"]
                if entry["clearance_after_m"] < floor:
                    failures.append(f"{joint} recentring lost too much skin clearance")
    if rest_mesh_error > 1e-6:
        failures.append(f"delivered rest mesh moved by {rest_mesh_error:.9f} m")

    report = {
        "schema_version": "autoanim.macap-base-model-fbx/4.2",
        "status": "passed" if not failures else "failed",
        "mode": mode,
        "character_name": CHARACTER_NAME,
        "output_fbx": str(output),
        "identity": "mean-shape neutral locked-head SMPL-X",
        "pose": "native SMPL-X rest pose with flat-hand mean",
        "smplx_betas": [0.0] * BETA_COUNT,
        "symmetrized": mode != "raw",
        "symmetrized_arrays": ["v_template", "J_regressor", "weights"] if mode != "raw" else [],
        "shapedirs_symmetrized": False,
        "symmetry_before": data["symmetry_before"],
        "symmetry_after": data["symmetry_after"],
        "spine_corrected": data["spine_corrected"],
        "spine_leans_before": data["spine_leans_before"],
        "spine_leans_after": data["spine_leans_after"],
        "spine_metrics": data["spine_metrics"],
        "spine_reskin": data["spine_reskin"],
        "feet_corrected": data["feet_corrected"],
        "feet_metrics": data["feet_metrics"],
        "feet_reskin": data["feet_reskin"],
        "knees_corrected": data["knees_corrected"],
        "knees_metrics": data["knees_metrics"],
        "knees_reskin": data["knees_reskin"],
        "hips_corrected": data["hips_corrected"],
        "hips_metrics": data["hips_metrics"],
        "hips_reskin": data["hips_reskin"],
        "elbows_corrected": data["elbows_corrected"],
        "elbows_metrics": data["elbows_metrics"],
        "elbows_reskin": data["elbows_reskin"],
        "rest_mesh_error_m": rest_mesh_error,
        "bone_axis_mirror_error_source": max(source_axis_errors.values()),
        "bone_axis_mirror_error_roundtrip": max(roundtrip_axis_errors.values()),
        "bone_axis_mirror_error_worst_bone": max(
            roundtrip_axis_errors, key=roundtrip_axis_errors.get
        ),
        "joint_regressor_residual_m": data["joint_regressor_residual_m"],
        "model_parents_match_contract": data["model_parents_match_contract"],
        "max_influences_per_vertex": data["max_influences_per_vertex"],
        "max_influences_after_roundtrip": worst_influences,
        "vertices_pruned_to_influence_budget": data["vertices_pruned_to_influence_budget"],
        "worst_weight_sum_error_roundtrip": worst_weight_total,
        "pose_correctives_baked": False,
        "texture_included": False,
        "gnm_included": False,
        "model_sha256": _sha256(model_path),
        "skeleton_contract_sha256": _sha256(skeleton_path),
        "output_sha256": _sha256(output),
        "source": source_contract,
        "roundtrip": roundtrip,
        "failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("macap base model FBX verification failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
