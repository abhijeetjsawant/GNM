"""Constrained hand-chain fit for calibrated multiview capture.

Finger joints cannot be triangulated independently from four wide cameras. At
~4.7 m the rays to a finger are near-parallel relative to the feature size, so
depth is barely constrained and a free 3D point can reproject within a few pixels
of every view while sitting decimetres away. Measured on the reference fixture,
independent per-joint triangulation produced phalanges of 68-196 mm where anatomy
says 25-45 mm, with standard deviations exceeding their means. See
`docs/FINGER_TRIANGULATION_GATE.md`.

The information is nonetheless present: MAMMA recovers articulating, stable
fingers from those same four videos. The difference is the estimator. It never
solves for a finger as a free point; it fits a kinematic chain whose bone lengths
are fixed, jointly across every view and every frame, with temporal coupling.

This module does the same thing on a rig we own. The hand is a chain of rotations
about fixed-length bones taken from MHR (Apache-2.0), anchored at a wrist the
body reconstruction already provides, and the only unknowns are joint angles.
A finger therefore cannot land at an anatomically impossible distance from its
parent, because that distance is not a variable.

What this is not: contact-precise. Expect pose-plausible fingers -- grasp state,
pointing, spread, coarse curl -- and treat inter-finger contact as unmeasured.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .commercial_multiview import CalibratedCamera, CommercialMultiviewError


SCHEMA_VERSION = "autoanim.hand-fit/1.0"
SKELETON_PATH = Path(__file__).resolve().parent / "data" / "mhr-skeleton-v1.json"

# MHR's own per-DoF channels: proximal joints rotate on three axes, middle and
# distal finger joints are single-axis hinges, the thumb carpometacarpal has two.
# 27 DoF per hand rather than the 45 free rotation would give -- 40% of the
# search space removed by anatomy before the optimiser starts.
# `pinky0` is the pinky metacarpal and carries no rotation channels of its own;
# it is kept in the chain as a fixed offset so the pinky starts in the right
# place. The `_null` joints are fingertips -- also fixed, and valuable because
# SOMA-77 predicts tip landmarks we can reproject against.
HAND_CHANNELS: dict[str, tuple[str, ...]] = {
    "thumb0": ("y", "z"), "thumb1": ("x", "y", "z"), "thumb2": ("z",), "thumb3": ("z",),
    "thumb_null": (),
    "index1": ("x", "y", "z"), "index2": ("z",), "index3": ("z",), "index_null": (),
    "middle1": ("x", "y", "z"), "middle2": ("z",), "middle3": ("z",), "middle_null": (),
    "ring1": ("x", "y", "z"), "ring2": ("z",), "ring3": ("z",), "ring_null": (),
    "pinky0": (),
    "pinky1": ("x", "y", "z"), "pinky2": ("z",), "pinky3": ("z",), "pinky_null": (),
}
# Parent-before-child, which forward kinematics relies on.
CHAIN_ORDER: tuple[str, ...] = (
    "thumb0", "thumb1", "thumb2", "thumb3", "thumb_null",
    "index1", "index2", "index3", "index_null",
    "middle1", "middle2", "middle3", "middle_null",
    "ring1", "ring2", "ring3", "ring_null",
    "pinky0", "pinky1", "pinky2", "pinky3", "pinky_null",
)

# MHR's own limits ship unmapped -- the parameter-name to index correspondence
# could not be recovered without pymomentum, and a wrong limit is worse than
# none because it silently constrains the solve to the wrong manifold. These come
# from standard hand biomechanics instead: flexion is generous, abduction and
# hyperextension are tight, and the hinge joints cannot deviate at all.
ANATOMICAL_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "x": (-0.35, 0.35),    # abduction / spread at the proximal joints
    "y": (-0.35, 0.35),    # axial twist, near-rigid in a real finger
    "z": (-0.20, 1.75),    # flexion, with only slight hyperextension
}
THUMB_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "x": (-0.60, 0.60), "y": (-0.90, 0.90), "z": (-0.60, 1.40),
}

# SOMA-77 names the four numbered joints of each finger plus an end effector.
# MHR names three plus a null. The first four SOMA joints of a chain are taken
# to correspond to MHR's three joints and its tip; `resolve_chain_alignment`
# checks that empirically rather than trusting it.
SOMA_FINGER_PREFIX = {
    "thumb": "HandThumb", "index": "HandIndex", "middle": "HandMiddle",
    "ring": "HandRing", "pinky": "HandPinky",
}


def load_mhr_skeleton(path: str | Path = SKELETON_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "autoanim.mhr-skeleton/1.0":
        raise CommercialMultiviewError("MHR skeleton asset schema is invalid")
    if value.get("linear_unit") != "centimetre":
        raise CommercialMultiviewError("MHR skeleton asset is not in centimetres")
    return value


@dataclass(frozen=True, slots=True)
class HandChain:
    """Fixed-length bone chain for one hand, in metres, rooted at the wrist."""

    side: str
    joints: tuple[str, ...]
    parents: tuple[int, ...]
    offsets_m: np.ndarray          # [J,3] translation from parent, rest pose
    prerotations_xyzw: np.ndarray  # [J,4]
    channels: tuple[tuple[str, ...], ...]

    @property
    def degrees_of_freedom(self) -> int:
        return sum(len(axes) for axes in self.channels)


def build_hand_chain(side: str, skeleton: dict[str, Any] | None = None) -> HandChain:
    """Assemble one hand's chain from the MHR asset. `side` is 'l' or 'r'."""

    if side not in ("l", "r"):
        raise CommercialMultiviewError("Hand side must be 'l' or 'r'")
    data = skeleton if skeleton is not None else load_mhr_skeleton()
    names: list[str] = data["joint_names"]
    parents: list[int] = data["joint_parents"]
    offsets = np.asarray(data["joint_translation_offsets_cm"], dtype=np.float64) / 100.0
    prerotations = np.asarray(data["joint_prerotations_xyzw"], dtype=np.float64)

    wanted = [f"{side}_{joint}" for joint in CHAIN_ORDER]
    missing = [name for name in wanted if name not in names]
    if missing:
        raise CommercialMultiviewError(f"MHR skeleton lacks hand joints: {missing}")
    index_of = {name: names.index(name) for name in wanted}
    wrist = names.index(f"{side}_wrist")

    local_parents: list[int] = []
    for name in wanted:
        parent = parents[index_of[name]]
        # The wrist anchors the chain, so it is index -1 in local terms.
        local_parents.append(-1 if parent == wrist else wanted.index(names[parent]))
    order = np.asarray([index_of[name] for name in wanted])
    return HandChain(
        side=side,
        joints=tuple(wanted),
        parents=tuple(local_parents),
        offsets_m=offsets[order],
        # Already xyzw, which is what scipy wants -- no permutation.
        prerotations_xyzw=prerotations[order],
        channels=tuple(HAND_CHANNELS[joint] for joint in CHAIN_ORDER),
    )


def _axis_rotation(axes: Sequence[str], angles: np.ndarray) -> Rotation:
    rotation = Rotation.identity()
    for axis, angle in zip(axes, angles, strict=True):
        rotation = rotation * Rotation.from_euler(axis, float(angle))
    return rotation


def forward_kinematics(
    chain: HandChain,
    wrist_position_m: np.ndarray,
    wrist_rotation_xyzw: np.ndarray,
    angles: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Joint positions for one frame. Bone lengths are fixed by construction."""

    rotations: list[Rotation] = []
    positions = np.zeros((len(chain.joints), 3), dtype=np.float64)
    wrist_rotation = Rotation.from_quat(wrist_rotation_xyzw)
    cursor = 0
    for index, joint in enumerate(chain.joints):
        axes = chain.channels[index]
        local = _axis_rotation(axes, angles[cursor : cursor + len(axes)])
        cursor += len(axes)
        prerotation = Rotation.from_quat(chain.prerotations_xyzw[index])
        parent = chain.parents[index]
        parent_rotation = wrist_rotation if parent < 0 else rotations[parent]
        parent_position = wrist_position_m if parent < 0 else positions[parent]
        positions[index] = parent_position + parent_rotation.apply(chain.offsets_m[index] * scale)
        rotations.append(parent_rotation * prerotation * local)
    return positions


def gate_observations_by_cross_view_agreement(
    cameras: Sequence[CalibratedCamera],
    observations: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Drop 2D observations the other views contradict.

    This is the load-bearing safety mechanism, and the reason a naive port of
    MAMMA's fit would fail. MammaNet is *trained* to emit per-landmark visibility
    and uncertainty, and its optimiser downweights by both. SOMA-77 emits a
    heatmap peak, which stays high when an occluded joint's response is captured
    by the nearest visible structure -- so during a hand clasp, the frames this
    exists for, the solver would receive confidently wrong 2D and the temporal
    term would lock it in: smooth, stable, wrong, and passing a bone-length gate.

    Without a visibility channel the substitute is geometric. A joint seen truly
    by several views agrees with itself epipolarly; a hallucinated one does not.

    `observations` is [F,C,J,3] as (x, y, confidence). Returns a boolean
    [F,C,J] mask of observations to keep.
    """

    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 4 or values.shape[3] != 3 or values.shape[1] != len(cameras):
        raise CommercialMultiviewError("Hand observations must be [frame,camera,joint,3]")
    frames, camera_count, joints, _ = values.shape
    keep = np.isfinite(values[..., :2]).all(axis=3) & (values[..., 2] > 0.0)

    from .commercial_multiview import _fundamental_matrix, _epipolar_distance_px

    for left in range(camera_count):
        for right in range(left + 1, camera_count):
            fundamental = _fundamental_matrix(cameras[left], cameras[right])
            for frame in range(frames):
                for joint in range(joints):
                    if not (keep[frame, left, joint] and keep[frame, right, joint]):
                        continue
                    a = values[frame, left, joint][None, :]
                    b = values[frame, right, joint][None, :]
                    distance = _epipolar_distance_px(
                        fundamental, a, b, minimum_confidence=0.0, minimum_shared_joints=1
                    )
                    if distance > threshold_px:
                        # Neither view is trustworthy on its own evidence; the
                        # pair disagrees, so both lose a vote on this joint.
                        keep[frame, left, joint] = False
                        keep[frame, right, joint] = False
    return keep


def _limits_for(chain: HandChain) -> tuple[np.ndarray, np.ndarray]:
    low: list[float] = []
    high: list[float] = []
    for joint, axes in zip(chain.joints, chain.channels, strict=True):
        table = THUMB_LIMITS_RAD if "thumb" in joint else ANATOMICAL_LIMITS_RAD
        for axis in axes:
            lo, hi = table[axis]
            low.append(lo)
            high.append(hi)
    return np.asarray(low), np.asarray(high)


def _pixels_per_metre(
    cameras: Sequence[CalibratedCamera], wrists: np.ndarray
) -> float:
    """How many pixels a hand-sized displacement spans, averaged over the rig.

    The temporal prior is a distance and the data term is a pixel count. Without
    this conversion the smoothing weight silently means something different on
    every rig -- change the resolution or move the cameras back and the same
    number buys a different amount of regularisation. Measured on this fixture
    the two blocks were 1.9% and 98.1% of the objective, which is how a hand fit
    that reprojects to 6 px ended up thrashing 26 mm between frames.
    """

    finite = wrists[np.isfinite(wrists).all(axis=1)]
    if not len(finite):
        raise CommercialMultiviewError("Wrist track has no finite frames")
    values: list[float] = []
    for camera in cameras:
        depth = float(np.median(np.linalg.norm(finite - camera.camera_center_world_m, axis=1)))
        focal = float(camera.intrinsics[0, 0] + camera.intrinsics[1, 1]) / 2.0
        if depth > 1e-6:
            values.append(focal / depth)
    if not values:
        raise CommercialMultiviewError("Cameras are degenerate for the wrist track")
    return float(np.mean(values))


def fit_hand_sequence(
    cameras: Sequence[CalibratedCamera],
    chain: HandChain,
    wrist_positions_m: np.ndarray,
    observations: np.ndarray,
    *,
    keep: np.ndarray | None = None,
    scale: float = 1.0,
    smooth_weight: float = 2.0,
    pose_smooth_weight: float = 0.25,
    limit_weight: float = 50.0,
    robust_scale_px: float = 8.0,
    maximum_evaluations: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the whole take at once.

    Variables per frame: wrist orientation (3, axis-angle) plus the chain's
    joint angles. Bone lengths are constants, which is the entire point.

    Two temporal terms, and the second is the one that matters. ``smooth_weight``
    penalises acceleration in *angle* space, which leaves the three wrist
    orientation parameters unregularised and is measured in radians against a
    data term measured in pixels. ``pose_smooth_weight`` penalises acceleration
    of the wrist-relative joint *positions*: it is dimensionless, it covers the
    wrist block and the joint angles together because positions depend on both,
    it has no Euler-gauge or axis-angle-wrap freedom to hide in, and it is the
    same quantity the articulation measurement reports as jitter.

    Returns ``(angles, positions, used)`` -- the per-frame parameter vector, the
    fitted joint positions in metres, and the observation mask actually used.
    """

    values = np.asarray(observations, dtype=np.float64)
    wrists = np.asarray(wrist_positions_m, dtype=np.float64)
    frames = values.shape[0]
    joints = len(chain.joints)
    if wrists.shape != (frames, 3):
        raise CommercialMultiviewError("Wrist positions must be [frame,3]")
    if values.shape != (frames, len(cameras), joints, 3):
        raise CommercialMultiviewError("Observations must be [frame,camera,joint,3]")
    used = keep if keep is not None else (
        np.isfinite(values[..., :2]).all(axis=3) & (values[..., 2] > 0.0)
    )
    if not used.any():
        raise CommercialMultiviewError("No hand observations survived gating")

    dof = chain.degrees_of_freedom
    per_frame = 3 + dof
    low, high = _limits_for(chain)
    projections = np.stack([camera.projection_matrix for camera in cameras])

    def unpack(vector: np.ndarray) -> np.ndarray:
        return vector.reshape(frames, per_frame)

    pixels_per_metre = _pixels_per_metre(cameras, wrists)

    def residuals(vector: np.ndarray, pose_weight: float = -1.0) -> np.ndarray:
        if pose_weight < 0.0:
            pose_weight = pose_smooth_weight
        state = unpack(vector)
        parts: list[np.ndarray] = []
        reprojection: list[float] = []
        posed = np.empty((frames, joints, 3), dtype=np.float64)
        for frame in range(frames):
            rotation = Rotation.from_rotvec(state[frame, :3]).as_quat()
            positions = forward_kinematics(
                chain, wrists[frame], rotation, state[frame, 3:], scale
            )
            posed[frame] = positions
            for camera_index in range(len(cameras)):
                mask = used[frame, camera_index]
                if not mask.any():
                    continue
                selected = positions[mask]
                homogeneous = np.hstack((selected, np.ones((len(selected), 1))))
                projected = homogeneous @ projections[camera_index].T
                depth = np.where(np.abs(projected[:, 2]) < 1e-9, 1e-9, projected[:, 2])
                error = projected[:, :2] / depth[:, None] - values[frame, camera_index, mask, :2]
                reprojection.extend(error.ravel().tolist())
        # soft-l1 on the measurement block only; the regularisers are priors and
        # have no outliers to suppress.
        residual = np.asarray(reprojection)
        magnitude = np.abs(residual)
        parts.append(
            np.sign(residual)
            * 2.0
            * robust_scale_px
            * (np.sqrt(1.0 + magnitude / robust_scale_px) - 1.0)
        )
        angles = state[:, 3:]
        parts.append(limit_weight * np.minimum(angles - low, 0.0).ravel())
        parts.append(limit_weight * np.maximum(angles - high, 0.0).ravel())
        if frames > 2:
            parts.append(smooth_weight * (angles[:-2] - 2.0 * angles[1:-1] + angles[2:]).ravel())
            # Wrist translation is an input, not a variable, so subtracting it
            # keeps this from fighting the body track's real acceleration.
            relative = posed - wrists[:, None, :]
            parts.append(
                pose_weight
                * pixels_per_metre
                * (relative[:-2] - 2.0 * relative[1:-1] + relative[2:]).ravel()
            )
        return np.concatenate(parts)

    start = np.zeros(frames * per_frame, dtype=np.float64)
    # Without this the Jacobian is dense: frames * per_frame columns, each
    # costing a full residual evaluation. At 150 frames that is 4,500 forward
    # passes per iteration and the solve never finishes. Every residual touches
    # only its own frame, except the temporal terms, which span three.
    sparsity = _jacobian_sparsity(
        residuals(start, 0.0), frames, per_frame, dof, used, joints
    )

    def solve(initial: np.ndarray, pose_weight: float, evaluations: int) -> np.ndarray:
        return least_squares(
            lambda vector: residuals(vector, pose_weight), initial, jac_sparsity=sparsity,
            method="trf", loss="linear", ftol=1e-3, x_scale="jac",
            max_nfev=evaluations, verbose=0,
        ).x

    # Two stages, and the first one is not optional. A constant pose has zero
    # acceleration, so the rest pose is a *global* minimum of the position prior
    # -- and the solve starts there. Once the prior outweighs the data the first
    # trust-region step is tiny, ftol fires, and the solver returns the rest pose
    # having barely moved. Measured: at weights 1.0 and 4.0 the fit returned in
    # 28 s against 651 s at 0.25, with byte-identical held-out error, which is
    # the signature of a solver that never left its start point. Fitting the data
    # first and warm-starting the regularised pass from it removes the trap
    # without weakening the prior.
    warm = solve(start, 0.0, max(maximum_evaluations // 3, 8))
    state = unpack(solve(warm, pose_smooth_weight, maximum_evaluations))
    positions = np.stack(
        [
            forward_kinematics(
                chain, wrists[frame], Rotation.from_rotvec(state[frame, :3]).as_quat(),
                state[frame, 3:], scale,
            )
            for frame in range(frames)
        ]
    )
    return state[:, 3:], positions, used


def _jacobian_sparsity(
    sample: np.ndarray, frames: int, per_frame: int, dof: int, used: np.ndarray,
    joints: int = 0,
) -> Any:
    """Which variables each residual touches.

    Reprojection residuals for a frame depend on that frame's wrist rotation and
    joint angles and nothing else. Limit residuals touch a single angle. Both
    temporal terms span three consecutive frames -- the angle one over the angle
    columns, the position one over every column of those frames, since a joint
    position depends on the wrist rotation as well as its own chain.
    """

    rows: list[int] = []
    columns: list[int] = []
    row = 0

    def frame_block(frame: int) -> range:
        base = frame * per_frame
        return range(base, base + per_frame)

    for frame in range(frames):
        for camera in range(used.shape[1]):
            count = int(used[frame, camera].sum())
            for _ in range(count * 2):
                for column in frame_block(frame):
                    rows.append(row)
                    columns.append(column)
                row += 1
    for _ in range(2):  # lower then upper limit blocks
        for frame in range(frames):
            base = frame * per_frame + 3
            for k in range(dof):
                rows.append(row)
                columns.append(base + k)
                row += 1
    if frames > 2:
        for frame in range(frames - 2):
            for k in range(dof):
                for offset in range(3):
                    rows.append(row)
                    columns.append((frame + offset) * per_frame + 3 + k)
                row += 1
    if frames > 2 and joints:
        for frame in range(frames - 2):
            for _ in range(joints * 3):
                for offset in range(3):
                    for column in frame_block(frame + offset):
                        rows.append(row)
                        columns.append(column)
                row += 1
    if row != len(sample):
        raise CommercialMultiviewError(
            f"Jacobian sparsity has {row} rows against {len(sample)} residuals"
        )
    return coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(row, frames * per_frame),
    )


__all__ = [
    "HandChain",
    "SCHEMA_VERSION",
    "build_hand_chain",
    "fit_hand_sequence",
    "forward_kinematics",
    "gate_observations_by_cross_view_agreement",
    "load_mhr_skeleton",
]
