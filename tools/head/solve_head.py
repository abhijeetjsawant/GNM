#!/usr/bin/env python3
"""A rigid-head fit across all views and all frames. The estimator the head has lacked.

**Why this shape, and it is not a guess.** `head_epipolar_gate.py` scores every head
landmark's cross-view 2D consistency at **0.29-0.78x the body control** -- the head 2D
are *better* than body 2D on both detectors and both subjects. So the head's 3D scatter
is not a detector failure; it is depth ambiguity, because rays to a small feature at 5 m
are near-parallel relative to that feature. That is exactly the situation
`FINGER_TRIANGULATION_GATE.md` concluded requires a **model-constrained fit** rather than
independent per-joint triangulation, and it is what MAMMA does.

So: never triangulate a head landmark on its own. Solve, per subject and over the whole
take at once, for

  * one rigid head **template** T (head-local positions of the 5 SOMA-77 head landmarks,
    constant over the take -- the skull is rigid), and
  * a per-frame head **rotation** R_t and **position** p_t,

minimising reprojection into all four calibrated cameras with a robust loss, plus a
temporal prior on R_t and p_t.

**The smoothing weight is chosen by held-out-camera cross-validation, never by agreement
with MAMMA.** Fit on three cameras, predict the fourth's 2D, take the weight with the
lowest held-out reprojection. Tuning the prior against the reference the gate scores
against would make the gate circular, which is the overlay-flatterer degenerate solution
one level up.

Blind to: the template's own gauge. A rigid head fit determines orientation only up to a
constant rotation, because nothing observes where the skull's "zero" is. Every downstream
comparison is therefore **mean-removed** -- it scores *tracking*, not absolute head
direction. A constant heading error is invisible here and needs a different instrument.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from associate import CAMERAS, OUT, load  # noqa: E402
from triangulate_soma import triangulate  # noqa: E402

HEAD = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}
NAMES = list(HEAD)
MIN_CONFIDENCE = 0.25
# Pre-registered sweep, and the two held-out cameras. Both were fixed before any fit
# was scored -- the grid was trimmed from six weights to four and from four held-out
# cameras to two purely for runtime (one solve is ~24 s), not after seeing a result.
# A001 and C001 are the opposed pair, so each fold drops a genuinely different view.
SMOOTH_WEIGHTS = (0.0, 10.0, 30.0, 100.0)
HELD_OUT_CAMERAS = (0, 2)
TEMPLATE_PRIOR = 0.5   # weak, and only to fix the gauge


def rodrigues(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(*aa.shape[:-1], 3, 3)
    t = theta[..., None]
    return np.broadcast_to(np.eye(3), skew.shape) + np.sin(t) * skew + (1 - np.cos(t)) * (skew @ skew)


def log_so3(matrices: np.ndarray) -> np.ndarray:
    trace = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)[:, None]
    vee = np.stack([matrices[:, 2, 1] - matrices[:, 1, 2],
                    matrices[:, 0, 2] - matrices[:, 2, 0],
                    matrices[:, 1, 0] - matrices[:, 0, 1]], axis=1)
    scale = np.divide(angle, 2.0 * np.sin(angle), out=np.full_like(angle, 0.5),
                      where=np.abs(np.sin(angle)) > 1e-8)
    return vee * scale


def gather(subject: int) -> tuple[np.ndarray, list]:
    """observations[frame, camera, landmark, 3] of (x, y, confidence)."""
    cameras, observations = load()
    state = np.load(OUT / "association.npz")
    assignment, used = state["assignment"], state["used"]
    frames = len(observations[0])
    marks = [[[np.asarray(p["landmarks_soma77"], dtype=np.float64) for p in v[f]["people"]]
              for f in range(frames)] for v in observations]
    out = np.full((frames, len(CAMERAS), len(NAMES), 3), np.nan)
    for frame in range(frames):
        if not used[frame, subject]:
            continue
        for camera in range(len(CAMERAS)):
            person = int(assignment[frame, subject, camera])
            if person < 0:
                continue
            for slot, name in enumerate(NAMES):
                out[frame, camera, slot] = marks[camera][frame][person][HEAD[name]]
    return out, cameras


def initialise(subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Template, rotations, positions from independent triangulation -- a start, not a result."""
    positions = triangulate([HEAD[name] for name in NAMES])[0][subject]  # [frame, k, 3]
    good = np.isfinite(positions).all(axis=(1, 2))
    centred = positions[good] - positions[good].mean(axis=1, keepdims=True)
    template = np.median(centred, axis=0)
    template -= template.mean(axis=0)
    rotations = np.zeros((len(positions), 3))
    translations = np.full((len(positions), 3), np.nan)
    for frame in np.flatnonzero(good):
        target = positions[frame] - positions[frame].mean(axis=0)
        u, _, vt = np.linalg.svd(target.T @ template)
        r = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
        rotations[frame] = log_so3(r[None])[0]
        translations[frame] = positions[frame].mean(axis=0)
    # Fill gaps so the optimiser starts from a connected trajectory.
    for axis in range(3):
        column = translations[:, axis]
        valid = np.isfinite(column)
        translations[:, axis] = np.interp(np.arange(len(column)), np.flatnonzero(valid), column[valid])
    return template, rotations, translations


# Cameras that must see at least three head landmarks for a frame to count as
# well-observed. Four is the rig; the scale is a ratio so the constant is arbitrary.
FULL_SUPPORT = 4.0


def thorax_frames(subject: int) -> np.ndarray:
    """The pipeline's own smoothed torso frame per frame, [frame, 3, 3].

    The same construction and the same source the gate uses, so the prior below smooths
    the quantity the gate scores rather than a differently-conditioned cousin.
    """
    from autoanim_gnm.commercial_multiview import JOINT_INDEX

    smoothed = np.load(
        f"artifacts/commercial-multiview-soma77/subject-{subject:02d}.body-track.npz"
    )["triangulated_world_positions_z_up_m"]
    up = smoothed[:, JOINT_INDEX["neck"]] - smoothed[:, JOINT_INDEX["root"]]
    across = smoothed[:, JOINT_INDEX["left_shoulder"]] - smoothed[:, JOINT_INDEX["right_shoulder"]]
    z = up / np.linalg.norm(up, axis=1, keepdims=True)
    x = across - z * np.einsum("ni,ni->n", across, z)[:, None]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=2)


def solve(observations: np.ndarray, cameras, template0: np.ndarray, rotations0: np.ndarray,
          translations0: np.ndarray, weight: float, camera_mask: np.ndarray,
          support_conditioned: bool = True, thorax: np.ndarray | None = None) -> dict:
    """Fit the rigid head. `support_conditioned` scales the temporal prior per frame by
    how little evidence that frame carries.

    **`thorax` changes what the temporal prior smooths, and it is the substantive
    choice.** Given a per-frame torso frame, the prior acts on the head's rotation
    *relative to the thorax* -- the neck -- instead of on its world rotation. Anatomy is
    the argument: the neck is what moves smoothly, while the head in world also inherits
    the torso's motion, so a world-space prior fights the body and under-constrains the
    neck at the same time. It adds no parameter; the same L-curve rule picks the same
    weight. Measured on this fixture, the reference's head-relative-to-thorax travel is
    1.06-1.46 deg median against 1.32-1.93 deg in world, which is the same statement.

    This is inverse-variance weighting, and it needs no measurement to justify: a frame
    seen by two cameras determines its rotation less well than one seen by four, so the
    prior should carry more of it. Measured on this fixture afterwards, per-frame
    disagreement with the reference correlates **-0.60** with camera support and the
    worst decile averages 2.8 supporting cameras against 3.7 for the rest -- so the term
    the prior is being asked to cover is the one that is actually there.
    """
    frames, n_cameras, n_marks = observations.shape[:3]
    projections = np.stack([c.projection_matrix for c in cameras])
    visible = (np.isfinite(observations[..., :2]).all(axis=3)
               & (observations[..., 2] >= MIN_CONFIDENCE) & camera_mask[None, :, None])
    # Per-frame evidence: cameras seeing >= 3 of the head landmarks.
    support = (visible.sum(axis=2) >= 3).sum(axis=1).astype(np.float64)
    scale = (FULL_SUPPORT / np.maximum(support[1:-1], 1.0)) if support_conditioned \
        else np.ones(max(frames - 2, 0))

    f_idx, c_idx, k_idx = np.nonzero(visible)
    observed = observations[f_idx, c_idx, k_idx, :2]
    confidence = np.sqrt(np.clip(observations[f_idx, c_idx, k_idx, 2], 0.0, 1.0))

    def unpack(v):
        return (v[: 3 * n_marks].reshape(n_marks, 3),
                v[3 * n_marks : 3 * n_marks + 3 * frames].reshape(frames, 3),
                v[3 * n_marks + 3 * frames :].reshape(frames, 3))

    def residuals(v):
        template, rotations, translations = unpack(v)
        matrices = rodrigues(rotations)
        world = np.einsum("fij,kj->fki", matrices, template) + translations[:, None, :]
        points = world[f_idx, k_idx]
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        projected = np.einsum("nij,nj->ni", projections[c_idx], homogeneous)
        depth = np.where(np.abs(projected[:, 2]) < 1e-6, 1e-6, projected[:, 2])
        uv = projected[:, :2] / depth[:, None]
        # Robust: the detector tail is 0.9-1.35 m in 3D and an L2 fit inherits it.
        raw = ((uv - observed) * confidence[:, None]).ravel()
        reprojection = np.sign(raw) * np.sqrt(2.0 * (np.sqrt(1.0 + np.abs(raw)) - 1.0)) * 3.0
        parts = [reprojection]
        if frames > 2 and weight > 0.0:
            if thorax is None:
                relative = np.einsum("fji,fjk->fik", matrices[:-1], matrices[1:])
            else:
                neck = np.einsum("fji,fjk->fik", thorax, matrices)
                relative = np.einsum("fji,fjk->fik", neck[:-1], neck[1:])
            spin = log_so3(relative)
            parts.append((weight * scale[:, None] * (spin[1:] - spin[:-1])).ravel())
            parts.append((weight * 0.02 * scale[:, None] *
                          (translations[:-2] - 2 * translations[1:-1] + translations[2:])).ravel())
        parts.append(TEMPLATE_PRIOR * (template - template0).ravel())
        return np.concatenate(parts)

    start = np.concatenate([template0.ravel(), rotations0.ravel(), translations0.ravel()])
    n = len(start)
    rows = len(residuals(start))
    sparsity = lil_matrix((rows, n), dtype=int)
    sparsity[: 2 * len(f_idx), : 3 * n_marks] = 1
    for row, (frame, mark) in enumerate(zip(f_idx.tolist(), k_idx.tolist())):
        sparsity[2 * row : 2 * row + 2, 3 * n_marks + 3 * frame : 3 * n_marks + 3 * frame + 3] = 1
        sparsity[2 * row : 2 * row + 2,
                 3 * n_marks + 3 * frames + 3 * frame : 3 * n_marks + 3 * frames + 3 * frame + 3] = 1
    sparsity[2 * len(f_idx) :, :] = 1
    result = least_squares(residuals, start, jac_sparsity=sparsity, method="trf",
                           ftol=1e-4, xtol=1e-8, max_nfev=120, verbose=0)
    template, rotations, translations = unpack(result.x)
    return {"template": template, "rotations": rotations, "translations": translations}


def held_out_px(fit: dict, observations: np.ndarray, cameras, camera: int) -> float:
    matrices = rodrigues(fit["rotations"])
    world = np.einsum("fij,kj->fki", matrices, fit["template"]) + fit["translations"][:, None, :]
    projection = cameras[camera].projection_matrix
    errors = []
    for frame in range(observations.shape[0]):
        for mark in range(observations.shape[2]):
            sample = observations[frame, camera, mark]
            if not np.isfinite(sample).all() or sample[2] < MIN_CONFIDENCE:
                continue
            h = np.append(world[frame, mark], 1.0)
            p = projection @ h
            if abs(p[2]) < 1e-6:
                continue
            errors.append(float(np.linalg.norm(p[:2] / p[2] - sample[:2])))
    return float(np.median(errors)) if errors else float("nan")


def main() -> None:
    report: dict = {}
    output: dict[str, np.ndarray] = {}
    for subject in range(2):
        observations, cameras = gather(subject)
        template0, rotations0, translations0 = initialise(subject)

        # --- choose the temporal weight by held-out camera, never by MAMMA -------
        scores = {}
        for weight in SMOOTH_WEIGHTS:
            per_camera = []
            for held in HELD_OUT_CAMERAS:
                mask = np.ones(len(CAMERAS), dtype=bool)
                mask[held] = False
                fit = solve(observations, cameras, template0, rotations0, translations0, weight, mask)
                per_camera.append(held_out_px(fit, observations, cameras, held))
            scores[weight] = float(np.nanmean(per_camera))
            print(f"  subject {subject}  weight {weight:6.1f} -> held-out {scores[weight]:.3f} px")
        best = min(scores, key=scores.get)

        fit = solve(observations, cameras, template0, rotations0, translations0, best,
                    np.ones(len(CAMERAS), dtype=bool))
        matrices = rodrigues(fit["rotations"])
        output[f"subject_{subject:02d}_head_world"] = matrices
        output[f"subject_{subject:02d}_head_position_m"] = fit["translations"]
        output[f"subject_{subject:02d}_template_m"] = fit["template"]
        report[f"subject_{subject:02d}"] = {
            "held_out_px_by_weight": scores,
            "chosen_weight": best,
            "chosen_held_out_px": scores[best],
            "template_extent_mm": {
                name: float(np.linalg.norm(fit["template"][i]) * 1000.0)
                for i, name in enumerate(NAMES)
            },
            "in_frame_reprojection_px": float(np.nanmean([
                held_out_px(fit, observations, cameras, c) for c in range(len(CAMERAS))])),
        }
        print(f"subject {subject}: chose weight {best} at {scores[best]:.3f} px held out")

    np.savez(OUT / "head-solve.npz", **output)
    (OUT / "head-solve.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
