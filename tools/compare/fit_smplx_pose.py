#!/usr/bin/env python3
"""RUNG 2: pose-fit SMPL-X to OUR triangulated joints. Our capture, their body and fit.

Rung 1 showed a parametric body reproduces both performers' proportions to sub-millimetre
where our rig is 39 mm out. That proves the body CAN represent them; it says nothing about
whether our capture can drive it. This does.

One part changes: the body model and the way pose is recovered. The detector, the
association and the triangulation stay ours. So if this lands near the capture's own 36-41 mm
against MAMMA, the ~100 mm the retarget loses was the RIG. If it stalls short, the loss is in
how we recover pose and the next rung is the fit, not the body.

Under-determined by construction and treated as such: 15 observed joints give 45 residuals
against 69 parameters, so the unobserved degrees of freedom -- spine twist above all, which
no landmark of ours sees -- are held near zero by an explicit prior rather than left to
wander somewhere flattering. Frames are solved in order, each warm-started from the last,
which supplies temporal continuity without a smoothness term that could mask jitter.

INSTRUMENT ONLY. SMPL-X is MPI research-licensed; `smplx_model` stays false in delivered
run reports.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tools" / "head"))
from autoanim_gnm.commercial_multiview import JOINT_INDEX, estimate_limb_lengths_m  # noqa: E402
from fit_smplx_to_capture import fit_shape  # noqa: E402
from smplx_body import SMPLXBody, BODY_JOINTS  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402

MA3D = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
TRACKS = ROOT / "artifacts/commercial-multiview-soma77"

# our 19-joint name -> SMPL-X body joint index
TARGETS = {
    "root": 0, "left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8, "neck": 12, "nose": 15,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}
POSE_PRIOR = 6.0     # mm per radian; holds unobserved DOFs near rest


def forward(body: SMPLXBody, rest: np.ndarray, rotvec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    rot = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
    world = np.zeros((BODY_JOINTS, 3, 3))
    pos = np.zeros((BODY_JOINTS, 3))
    for j in range(BODY_JOINTS):
        p = body.parents[j]
        if p < 0:
            world[j], pos[j] = rot[j], rest[j]
        else:
            world[j] = world[p] @ rot[j]
            pos[j] = pos[p] + world[p] @ (rest[j] - rest[p])
    return pos + trans


def main() -> None:
    body = SMPLXBody()
    ours = np.stack([
        np.load(TRACKS / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        for s in (0, 1)
    ])
    mapping = mamma_index_for(ours)
    slots = [JOINT_INDEX[n] for n in TARGETS]
    smplx_idx = list(TARGETS.values())
    report = {"note": "agreement with an instrument, not accuracy", "subjects": {}}

    for s in (0, 1):
        cap = ours[s]
        betas, _ = fit_shape(body, estimate_limb_lengths_m(cap))
        rest = body.rest_joints(betas)[:BODY_JOINTS]
        frames = len(cap)
        state = np.zeros(BODY_JOINTS * 3 + 3)
        posed = np.full((frames, BODY_JOINTS, 3), np.nan)
        for f in range(frames):
            target = cap[f, slots]
            ok = np.isfinite(target).all(axis=1)
            if ok.sum() < 6:
                continue
            tgt, idx = target[ok], np.asarray(smplx_idx)[ok]

            def residual(x: np.ndarray) -> np.ndarray:
                j = forward(body, rest, x[:-3], x[-3:])
                return np.concatenate([
                    ((j[idx] - tgt) * 1000.0).ravel(),
                    # The prior holds UNOBSERVED joints near rest. It must exclude the
                    # global orientation (joint 0): our capture is Z-up and SMPL-X is
                    # Y-up, so that term carries a ~90 deg correction, and penalising it
                    # made the optimiser split the difference -- limbs compensated with
                    # their own DOF while the head chain, reachable only through four
                    # penalised spine joints, ended up 429-807 mm out. Everything else
                    # fitted to 5-30 mm, which is exactly what a mis-specified prior on
                    # one term looks like.
                    POSE_PRIOR * x[3:-3],
                ])

            if f == 0:   # seed translation at the pelvis so frame 0 does not start 5 m away
                state[-3:] = cap[f, JOINT_INDEX["root"]] - rest[0]
            solution = least_squares(residual, state, method="trf", ftol=1e-6, max_nfev=220)
            state = solution.x
            posed[f] = forward(body, rest, state[:-3], state[-3:])

        # score against MAMMA on the same joints, and against our own capture
        m_joints = np.load(MA3D / f"verts_joints_body_id-{mapping[s]:02d}.npz",
                           allow_pickle=True)["pred_joints"].astype(np.float64)
        n = min(frames, len(m_joints))
        rows, vs_mamma, vs_cap = {}, [], []
        for name, mi in TARGETS.items():
            p = posed[:n, mi]
            ref = m_joints[:n, mi]
            c = cap[:n, JOINT_INDEX[name]]
            ok = np.isfinite(p).all(axis=1) & np.isfinite(ref).all(axis=1) & np.isfinite(c).all(axis=1)
            dm = float(np.median(np.linalg.norm(p[ok] - ref[ok], axis=1)) * 1000)
            dc = float(np.median(np.linalg.norm(p[ok] - c[ok], axis=1)) * 1000)
            rows[name] = {"vs_mamma_mm": round(dm, 1), "vs_our_capture_mm": round(dc, 1)}
            vs_mamma.append(dm); vs_cap.append(dc)
        report["subjects"][f"subject_{s:02d}"] = {
            "per_joint": rows,
            "median_vs_mamma_mm": round(float(np.median(vs_mamma)), 1),
            "median_vs_our_capture_mm": round(float(np.median(vs_cap)), 1),
            "frames_solved": int(np.isfinite(posed).all(axis=(1, 2)).sum()),
        }
        np.savez(ROOT / f"artifacts/compare/smplx-posed-subject-{s:02d}.npz", joints=posed, betas=betas)
        print(f"\n=== our subject {s} (MAMMA body_id-{mapping[s]:02d}) ===")
        print(f"{'joint':16s} {'vs MAMMA':>10s} {'vs our capture':>15s}")
        for name, r in rows.items():
            print(f"{name:16s} {r['vs_mamma_mm']:10.1f} {r['vs_our_capture_mm']:15.1f}")
        print(f"{'MEDIAN':16s} {np.median(vs_mamma):10.1f} {np.median(vs_cap):15.1f}")

    (ROOT / "artifacts/compare/smplx-pose-fit.json").write_text(json.dumps(report, indent=2))
    print("\nwrote artifacts/compare/smplx-pose-fit.json")


if __name__ == "__main__":
    main()
