"""Head orientation from calibrated multi-view landmarks.

**Why this exists as a stage at all.** Before it, `positions_to_body_track` assigned the
head, the neck and both eyes the torso's frame verbatim, so every delivered track carried
the identity quaternion on those joints and the head could not move relative to the chest.
That is a constant, and it scored at parity with a research reference on frame-to-frame
jitter while carrying no head information — which is exactly why a jitter gate cannot be
the gate. See `docs/HEAD_ORIENTATION_MEASURED.md`.

**Why a fit and not triangulation.** Measured on the reference fixture, the head's 2D are
*more* cross-view consistent than the body's — 0.29–0.78x the body control on symmetric
epipolar distance. The failure is therefore depth, not detection: rays to a ~120 mm
feature at 5 m are near-parallel relative to it, so excellent 2D still triangulates into
decimetres of scatter along the viewing direction. The estimator is what has to change, so
**no head landmark is ever triangulated on its own here.** One rigid head template and a
per-frame rotation are solved against all cameras and all frames at once.

**What is regularised, and where.** The temporal prior acts on the head's rotation
*relative to the thorax* — the neck — not on its world rotation, because the head in world
also carries the torso's motion. Anatomy, and it costs no parameter. The head's position is
tied to the neck joint by a soft prior, which damps the position-vs-rotation trade a small
rigid body at range is prone to; measured, that term is worth little and is retained
because it is right rather than because it earned anything.

**What this stage does NOT claim.** The gate behind it measures *parity with an
instrument on a tracking metric*. Nothing here is an accuracy claim, and per
`docs/BODY_LANE_PLAN.md` §1 none is available until the marker session.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# The weight grid the temporal prior is selected over, and the rule. Selection consults
# only our own reprojection -- never any reference -- so a gate scored against a reference
# cannot be tuned through it. The grid is swept per take rather than hardcoded: a weight
# fitted once on one fixture and then shipped would be a constant calibrated on that
# fixture, which this lane forbids.
DEFAULT_WEIGHTS: tuple[float, ...] = (0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
MINIMUM_LANDMARKS_PER_VIEW = 3
MINIMUM_SOLVED_FRACTION = 0.5
# A hard physical reject, not a tuning knob. A human head peaks around 500-800 deg/s, so
# at 30 fps roughly 27 deg between consecutive frames is already extreme; 60 deg is
# 1800 deg/s and is not a neck. Solutions containing such a step are discarded before the
# reprojection rule chooses, because minimum reprojection alone has no notion of what a
# neck can do -- added after an unsmoothed solve delivered a 140 deg single-frame flip
# that reprojected perfectly well.
MAXIMUM_FRAME_TRAVEL_DEG = 60.0


# The head's anatomical axes, expressed in the CAPTURE convention (Z-up, and the rig's
# canonical facing -Z maps to capture +Y -- see `positions_to_body_track`). Columns are
# (subject's left, up through the skull, forward). Right-handed: left x up = forward.
CANONICAL_HEAD_AXES = np.asarray(((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))).T


class HeadOrientationError(RuntimeError):
    """The head solve could not run. Callers fall back and report; they never guess."""


@dataclass(frozen=True)
class HeadOrientation:
    """Per-frame head rotation in world, plus what the solve is worth knowing about."""

    rotations_world: np.ndarray          # [frame, 3, 3]
    positions_world_m: np.ndarray        # [frame, 3]
    template_m: np.ndarray               # [landmark, 3], head-local, constant over the take
    temporal_weight: float
    reprojection_px: float
    observed_frame_fraction: float
    landmark_names: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "temporal_weight": self.temporal_weight,
            "reprojection_px": self.reprojection_px,
            "observed_frame_fraction": self.observed_frame_fraction,
            "landmarks": list(self.landmark_names),
            "template_extent_mm": {
                name: float(np.linalg.norm(self.template_m[index]) * 1000.0)
                for index, name in enumerate(self.landmark_names)
            },
        }


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.divide(axis_angle, theta, out=np.zeros_like(axis_angle), where=theta > 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    angle = theta[..., None]
    return (
        np.broadcast_to(np.eye(3), skew.shape)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def log_so3(matrices: np.ndarray) -> np.ndarray:
    trace = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)[:, None]
    vee = np.stack(
        [
            matrices[:, 2, 1] - matrices[:, 1, 2],
            matrices[:, 0, 2] - matrices[:, 2, 0],
            matrices[:, 1, 0] - matrices[:, 0, 1],
        ],
        axis=1,
    )
    scale = np.divide(
        angle, 2.0 * np.sin(angle), out=np.full_like(angle, 0.5),
        where=np.abs(np.sin(angle)) > 1e-8,
    )
    return vee * scale


def orthonormalise(matrices: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrices)
    out = u @ vt
    flip = np.linalg.det(out) < 0
    if np.any(flip):
        u = u.copy()
        u[flip, :, -1] *= -1.0
        out = u @ vt
    return out


def _initialise(
    observations: np.ndarray, cameras: Sequence, minimum_confidence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Seed the optimiser from independent per-frame triangulation.

    Deliberately the estimator this module exists to replace: it is a starting point and
    never a result. Frames it cannot resolve are interpolated so the optimiser starts from
    a connected trajectory.
    """
    from .commercial_multiview import triangulate_point

    frames, _, marks, _ = observations.shape
    points = np.full((frames, marks, 3), np.nan)
    for frame in range(frames):
        for mark in range(marks):
            sample = observations[frame, :, mark]
            result = triangulate_point(
                cameras, sample[:, :2], np.nan_to_num(sample[:, 2]),
                minimum_confidence=minimum_confidence,
            )
            if result is not None:
                points[frame, mark] = result.position_world_m

    resolved = np.isfinite(points).all(axis=(1, 2))
    if not resolved.any():
        raise HeadOrientationError("no frame resolved every head landmark for initialisation")
    centred = points[resolved] - points[resolved].mean(axis=1, keepdims=True)
    template = np.median(centred, axis=0)
    template -= template.mean(axis=0)

    rotations = np.zeros((frames, 3))
    translations = np.full((frames, 3), np.nan)
    for frame in np.flatnonzero(resolved):
        target = points[frame] - points[frame].mean(axis=0)
        u, _, vt = np.linalg.svd(target.T @ template)
        rotation = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
        rotations[frame] = log_so3(rotation[None])[0]
        translations[frame] = points[frame].mean(axis=0)
    index = np.arange(frames, dtype=np.float64)
    for axis in range(3):
        column = translations[:, axis]
        valid = np.isfinite(column)
        translations[:, axis] = np.interp(index, index[valid], column[valid])
    return template, rotations, translations, resolved


def _solve_once(
    observations: np.ndarray,
    cameras: Sequence,
    template0: np.ndarray,
    rotations0: np.ndarray,
    translations0: np.ndarray,
    *,
    weight: float,
    thorax_world: np.ndarray | None,
    neck_origin_world_m: np.ndarray | None,
    neck_sigma_m: float,
    template_prior: float,
    minimum_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, camera_count, marks, _ = observations.shape
    projections = np.stack([camera.projection_matrix for camera in cameras])
    visible = np.isfinite(observations[..., :2]).all(axis=3) & (
        observations[..., 2] >= minimum_confidence
    )
    support = (visible.sum(axis=2) >= MINIMUM_LANDMARKS_PER_VIEW).sum(axis=1).astype(np.float64)
    # Inverse-variance in spirit: a frame seen by fewer cameras determines its rotation
    # less well, so the prior carries more of it.
    scale = (
        4.0 / np.maximum(support[1:-1], 1.0) if frames > 2 else np.ones(0)
    )

    frame_index, camera_index, mark_index = np.nonzero(visible)
    observed = observations[frame_index, camera_index, mark_index, :2]
    confidence = np.sqrt(np.clip(observations[frame_index, camera_index, mark_index, 2], 0.0, 1.0))

    def unpack(vector: np.ndarray):
        return (
            vector[: 3 * marks].reshape(marks, 3),
            vector[3 * marks : 3 * marks + 3 * frames].reshape(frames, 3),
            vector[3 * marks + 3 * frames :].reshape(frames, 3),
        )

    def residuals(vector: np.ndarray) -> np.ndarray:
        template, rotations, translations = unpack(vector)
        matrices = rodrigues(rotations)
        world = np.einsum("fij,kj->fki", matrices, template) + translations[:, None, :]
        points = world[frame_index, mark_index]
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        projected = np.einsum("nij,nj->ni", projections[camera_index], homogeneous)
        depth = np.where(np.abs(projected[:, 2]) < 1e-6, 1e-6, projected[:, 2])
        raw = ((projected[:, :2] / depth[:, None] - observed) * confidence[:, None]).ravel()
        # Robust: the detector tail reaches decimetres and an L2 fit inherits it.
        parts = [np.sign(raw) * np.sqrt(2.0 * (np.sqrt(1.0 + np.abs(raw)) - 1.0)) * 3.0]
        if frames > 2 and weight > 0.0:
            if thorax_world is None:
                relative = np.einsum("fji,fjk->fik", matrices[:-1], matrices[1:])
            else:
                neck = np.einsum("fji,fjk->fik", thorax_world, matrices)
                relative = np.einsum("fji,fjk->fik", neck[:-1], neck[1:])
            spin = log_so3(relative)
            parts.append((weight * scale[:, None] * (spin[1:] - spin[:-1])).ravel())
            parts.append(
                (
                    weight * 0.02 * scale[:, None]
                    * (translations[:-2] - 2.0 * translations[1:-1] + translations[2:])
                ).ravel()
            )
        if neck_origin_world_m is not None and thorax_world is not None:
            offset = np.einsum("fji,fj->fi", thorax_world, translations - neck_origin_world_m)
            parts.append(((offset - offset.mean(axis=0)) / neck_sigma_m).ravel())
        parts.append(template_prior * (template - template0).ravel())
        return np.concatenate(parts)

    start = np.concatenate([template0.ravel(), rotations0.ravel(), translations0.ravel()])
    rows = len(residuals(start))
    sparsity = lil_matrix((rows, len(start)), dtype=int)
    sparsity[: 2 * len(frame_index), : 3 * marks] = 1
    for row, (frame, mark) in enumerate(zip(frame_index.tolist(), mark_index.tolist())):
        base_r = 3 * marks + 3 * frame
        base_t = 3 * marks + 3 * frames + 3 * frame
        sparsity[2 * row : 2 * row + 2, base_r : base_r + 3] = 1
        sparsity[2 * row : 2 * row + 2, base_t : base_t + 3] = 1
    sparsity[2 * len(frame_index) :, :] = 1
    solution = least_squares(
        residuals, start, jac_sparsity=sparsity, method="trf",
        ftol=1e-4, xtol=1e-8, max_nfev=120, verbose=0,
    )
    return unpack(solution.x)


def _reprojection_px(
    template: np.ndarray, rotations: np.ndarray, translations: np.ndarray,
    observations: np.ndarray, cameras: Sequence, minimum_confidence: float,
) -> float:
    matrices = rodrigues(rotations)
    world = np.einsum("fij,kj->fki", matrices, template) + translations[:, None, :]
    errors: list[float] = []
    for camera_index, camera in enumerate(cameras):
        projection = camera.projection_matrix
        for frame in range(observations.shape[0]):
            for mark in range(observations.shape[2]):
                sample = observations[frame, camera_index, mark]
                if not np.isfinite(sample).all() or sample[2] < minimum_confidence:
                    continue
                homogeneous = np.append(world[frame, mark], 1.0)
                projected = projection @ homogeneous
                if abs(projected[2]) < 1e-6:
                    continue
                errors.append(float(np.linalg.norm(projected[:2] / projected[2] - sample[:2])))
    return float(np.median(errors)) if errors else float("nan")


def _maximum_frame_travel_deg(
    rotations_world: np.ndarray, thorax_world: np.ndarray | None
) -> float:
    """Largest frame-to-frame head rotation, measured at the neck where possible.

    Taken relative to the thorax when one is supplied, because a head on a turning body
    travels in world without the neck moving at all -- scoring the world rotation would
    reject honest motion and accept a genuine neck flip during a still moment.
    """
    if len(rotations_world) < 2:
        return 0.0
    frames = rotations_world if thorax_world is None else np.einsum(
        "fji,fjk->fik", thorax_world, rotations_world
    )
    relative = np.einsum("fji,fjk->fik", frames[:-1], frames[1:])
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)).max())


def _anatomical_gauge(
    template: np.ndarray, landmark_names: Sequence[str]
) -> np.ndarray:
    """The constant rotation that gives the solved head an absolute, anatomical zero.

    **A rigid fit determines orientation only up to a constant**, because nothing in the
    objective observes where the skull's "zero" is: the template's local frame is whatever
    the optimiser happened to land on. The gate never noticed, because it removes each
    take's mean and scores *tracking*. **Delivery cannot**: a track needs an absolute head
    direction, and shipping the raw solve puts an 80-176 degree constant offset on the
    joint -- a head pointing sideways, smoothly.

    The zero is fixed from the template's own anatomy and nothing else: `HeadEnd` above
    `Head` gives the skull's long axis, the eyes give the lateral axis. **No reference is
    consulted**, which matters because the obvious alternative -- aligning to a research
    fitter's head -- would be a shipped constant fitted on a reference-derived artifact,
    which `docs/BODY_LANE_PLAN.md` forbids outright.

    Requires `Head`, `HeadEnd` and both eyes in the landmark set; raises otherwise, rather
    than shipping an arbitrary frame.
    """
    names = list(landmark_names)
    required = ("Head", "HeadEnd", "LeftEye", "RightEye")
    missing = [name for name in required if name not in names]
    if missing:
        raise HeadOrientationError(
            f"cannot fix the head's anatomical zero without {', '.join(missing)}; "
            "refusing to ship an arbitrary orientation"
        )
    up = template[names.index("HeadEnd")] - template[names.index("Head")]
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        raise HeadOrientationError("Head and HeadEnd coincide; the skull has no long axis")
    up = up / norm
    left = template[names.index("LeftEye")] - template[names.index("RightEye")]
    left = left - up * float(left @ up)
    norm = np.linalg.norm(left)
    if norm < 1e-6:
        raise HeadOrientationError("the eye axis is parallel to the skull axis")
    left = left / norm
    forward = np.cross(left, up)
    anatomical = np.stack([left, up, forward], axis=1)   # template-local -> anatomical
    return anatomical @ CANONICAL_HEAD_AXES.T            # canonical head -> template-local


def solve_head_orientation(
    cameras: Sequence,
    observations_xyc: np.ndarray,
    landmark_names: Sequence[str],
    *,
    thorax_world: np.ndarray | None = None,
    neck_origin_world_m: np.ndarray | None = None,
    neck_sigma_m: float = 0.010,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    template_prior: float = 0.5,
    minimum_confidence: float = 0.25,
) -> HeadOrientation:
    """Solve one rigid head over a whole take against every calibrated view at once.

    ``observations_xyc`` is ``[frame, camera, landmark, 3]`` of ``(x, y, confidence)`` in
    the same pixel space the cameras are scaled to. ``thorax_world`` is ``[frame, 3, 3]``
    and, when given, moves the temporal prior into the neck -- which is where anatomy says
    the smoothness lives.

    The temporal weight is chosen by **minimum in-frame reprojection over ``weights``**.
    That rule consults only our own observations, so nothing downstream that scores this
    against a reference can be tuned through it. It is swept per take rather than fixed,
    because a weight chosen once on one fixture and then shipped is a constant calibrated
    on that fixture.

    Raises :class:`HeadOrientationError` when the evidence cannot support a solve. Callers
    fall back to the previous behaviour and **report having done so**; they never guess.
    """
    observations = np.asarray(observations_xyc, dtype=np.float64)
    if observations.ndim != 4 or observations.shape[1] != len(cameras) or observations.shape[3] != 3:
        raise HeadOrientationError("Head observations must be [frame, camera, landmark, 3]")
    if observations.shape[2] != len(landmark_names):
        raise HeadOrientationError("Head landmark names do not match the observation array")
    frames = observations.shape[0]
    if frames < 3:
        raise HeadOrientationError("Head solve needs at least three frames")
    if thorax_world is not None and np.asarray(thorax_world).shape != (frames, 3, 3):
        raise HeadOrientationError("Thorax frames must be [frame, 3, 3]")

    visible = np.isfinite(observations[..., :2]).all(axis=3) & (
        observations[..., 2] >= minimum_confidence
    )
    observed = (visible.sum(axis=2) >= MINIMUM_LANDMARKS_PER_VIEW).sum(axis=1) >= 2
    fraction = float(observed.mean())
    if fraction < MINIMUM_SOLVED_FRACTION:
        raise HeadOrientationError(
            f"only {fraction:.0%} of frames have two views of the head; "
            f"below the {MINIMUM_SOLVED_FRACTION:.0%} floor"
        )

    template0, rotations0, translations0 = _initialise(
        observations, cameras, minimum_confidence
    )[:3]
    thorax = None if thorax_world is None else orthonormalise(np.asarray(thorax_world, float))
    neck = None if neck_origin_world_m is None else np.asarray(neck_origin_world_m, float)

    best: tuple[float, float, tuple] | None = None
    rejected: list[str] = []
    for weight in weights:
        template, rotations, translations = _solve_once(
            observations, cameras, template0, rotations0, translations0,
            weight=float(weight), thorax_world=thorax, neck_origin_world_m=neck,
            neck_sigma_m=neck_sigma_m, template_prior=template_prior,
            minimum_confidence=minimum_confidence,
        )
        error = _reprojection_px(
            template, rotations, translations, observations, cameras, minimum_confidence
        )
        if not math.isfinite(error):
            continue
        travel = _maximum_frame_travel_deg(rodrigues(rotations), thorax)
        if travel > MAXIMUM_FRAME_TRAVEL_DEG:
            # Reprojection is blind to this: a head can flip between frames and still sit
            # on every observation, because the flip happens along the viewing rays.
            rejected.append(f"weight {weight:g}: {travel:.0f} deg/frame")
            continue
        if best is None or error < best[0]:
            best = (error, float(weight), (template, rotations, translations))
    if best is None:
        detail = "; ".join(rejected) if rejected else "none converged"
        raise HeadOrientationError(
            f"no head solve was both convergent and anatomically possible ({detail})"
        )

    error, weight, (template, rotations, translations) = best
    # Absolute zero, from anatomy. Without it the solve is correct up to a constant and
    # useless for delivery -- see `_anatomical_gauge`.
    gauge = _anatomical_gauge(template, landmark_names)
    return HeadOrientation(
        rotations_world=orthonormalise(rodrigues(rotations) @ gauge),
        positions_world_m=translations,
        template_m=template,
        temporal_weight=weight,
        reprojection_px=error,
        observed_frame_fraction=fraction,
        landmark_names=tuple(landmark_names),
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "HeadOrientation",
    "HeadOrientationError",
    "solve_head_orientation",
]
