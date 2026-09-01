#!/usr/bin/env python3
"""Fit SMPL-X SHAPE to our own measured limb lengths. Our capture, their body.

The first rung of the substitution ladder: change ONE part -- the body model -- and keep
our detector, our association, our triangulation. If the ~100 mm the retarget loses is the
body, this recovers it; if it does not, the body was not the problem and the next rung is.

Shape only, no pose. Bone lengths are pose-invariant, so this answers "can a real human body
have these proportions, and how close does the best one get" without a per-frame solve --
which is the cheap decisive question before spending anything on the expensive one.

INSTRUMENT ONLY. SMPL-X is MPI research-licensed and `smplx_model` stays false in every
delivered run report.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoanim_gnm.commercial_multiview import estimate_limb_lengths_m  # noqa: E402
from smplx_body import SMPLXBody  # noqa: E402

# our measured limb -> the SMPL-X joint pair spanning it
LIMB_TO_SMPLX = {
    ("left_shoulder", "left_elbow"): (16, 18),
    ("left_elbow", "left_wrist"): (18, 20),
    ("right_shoulder", "right_elbow"): (17, 19),
    ("right_elbow", "right_wrist"): (19, 21),
    ("left_hip", "left_knee"): (1, 4),
    ("left_knee", "left_ankle"): (4, 7),
    ("right_hip", "right_knee"): (2, 5),
    ("right_knee", "right_ankle"): (5, 8),
    ("left_shoulder", "right_shoulder"): (16, 17),
    ("left_hip", "right_hip"): (1, 2),
    ("root", "neck"): (0, 12),
}
CANONICAL_MM = {   # our delivered rig, for the comparison column
    ("left_shoulder", "left_elbow"): 260.0, ("left_elbow", "left_wrist"): 240.0,
    ("right_shoulder", "right_elbow"): 260.0, ("right_elbow", "right_wrist"): 240.0,
    ("left_hip", "left_knee"): 430.0, ("left_knee", "left_ankle"): 420.5,
    ("right_hip", "right_knee"): 430.0, ("right_knee", "right_ankle"): 420.5,
    ("left_shoulder", "right_shoulder"): 540.0, ("left_hip", "right_hip"): 180.0,
    ("root", "neck"): 580.0,
}


# SMPL-X betas are standardised by construction -- the shape space is built so that a real
# population sits near N(0,1) -- so an L2 pull toward zero is the model's own prior, not a
# knob. It is REQUIRED here, not optional: 10 free betas against 11 limb targets hit every
# target exactly and produced an anatomically impossible body, with the head 624 mm BELOW
# the pelvis. The unconstrained directions of the shape space are free to do anything, and
# without a prior they do. A sub-millimetre fit on the terms you constrained is not evidence
# the body is right; it is evidence you had enough parameters.
BETA_PRIOR_MM = 12.0


def fit_shape(body: SMPLXBody, measured: dict) -> tuple[np.ndarray, dict]:
    pairs = [(LIMB_TO_SMPLX[k], v) for k, v in measured.items() if k in LIMB_TO_SMPLX]

    def residual(betas: np.ndarray) -> np.ndarray:
        j = body.rest_joints(betas)
        return np.concatenate([
            np.asarray([np.linalg.norm(j[a] - j[b]) - target for (a, b), target in pairs]) * 1000.0,
            BETA_PRIOR_MM * np.asarray(betas, dtype=np.float64),
        ])

    solution = least_squares(residual, np.zeros(body.shape_terms), method="trf", max_nfev=400)
    limb_residual = solution.fun[: len(pairs)]
    joints = body.rest_joints(solution.x)
    # A body whose head is not above its pelvis is not a body. Cheap, and it would have
    # caught the overfit immediately.
    upright = float(joints[15, 1] - joints[0, 1])
    return solution.x, {
        "limb_rms_mm": float(np.sqrt(np.mean(limb_residual ** 2))),
        "beta_norm": float(np.linalg.norm(solution.x)),
        "head_above_pelvis_mm": round(1000.0 * upright, 1),
        "anatomically_plausible": bool(upright > 0.3),
    }


def main() -> None:
    body = SMPLXBody()
    tracks = ROOT / "artifacts/commercial-multiview-soma77"
    report = {}
    for s in (0, 1):
        pos = np.load(tracks / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        measured = estimate_limb_lengths_m(pos)
        betas, info = fit_shape(body, measured)
        j = body.rest_joints(betas)
        print(f"\n=== our subject {s} ===   limb RMS {info['limb_rms_mm']:.1f} mm | "
              f"|betas| {info['beta_norm']:.2f} | head above pelvis "
              f"{info['head_above_pelvis_mm']:.0f} mm -> "
              f"{'PLAUSIBLE' if info['anatomically_plausible'] else 'IMPOSSIBLE BODY'}")
        print(f"{'limb':34s} {'measured':>9s} {'SMPL-X fit':>11s} {'our rig':>9s}")
        rows = {}
        errs_smplx, errs_rig = [], []
        for limb, target in measured.items():
            if limb not in LIMB_TO_SMPLX:
                continue
            a, b = LIMB_TO_SMPLX[limb]
            got = float(np.linalg.norm(j[a] - j[b])) * 1000.0
            tgt = target * 1000.0
            rig = CANONICAL_MM[limb]
            rows[f"{limb[0]}->{limb[1]}"] = {"measured_mm": round(tgt, 1),
                                             "smplx_mm": round(got, 1), "rig_mm": rig}
            errs_smplx.append(abs(got - tgt)); errs_rig.append(abs(rig - tgt))
            print(f"{limb[0]+'->'+limb[1]:34s} {tgt:9.1f} {got:11.1f} {rig:9.1f}")
        print(f"{'MEAN ABSOLUTE ERROR':34s} {'':9s} {np.mean(errs_smplx):11.1f} {np.mean(errs_rig):9.1f}")
        report[f"subject_{s:02d}"] = {"betas": betas.round(4).tolist(), "limbs": rows,
                                      "mean_abs_error_mm": {"smplx": round(float(np.mean(errs_smplx)), 1),
                                                            "our_rig": round(float(np.mean(errs_rig)), 1)}}
    out = ROOT / "artifacts/compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smplx-shape-fit.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out/'smplx-shape-fit.json'}")


if __name__ == "__main__":
    main()
