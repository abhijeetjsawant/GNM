#!/usr/bin/env python3
"""Which torso frame is SHARPEST? Decided by the ORACLE arm, pre-registered.

§6h decomposes the gate: our own head error is 9.91/9.80 deg p95, while the ORACLE floor
-- a *perfect* head carried through our torso frame -- is 14.14/16.30. **The floor is the
larger half of the score**, and it is not the head at all: it is our landmark-derived
thorax disagreeing with MAMMA's fitted kinematic chain.

THE RULE, fixed before running:

    Adopt the frame with the LOWEST ORACLE error, whatever it does to the candidate.

The oracle is a perfect head, so every degree it scores is pure frame mismatch and nothing
else. Minimising it makes the gate a sharper instrument. **This is instrument repair, not
candidate tuning**, and it is the same move §6g already made in the UNFLATTERING direction
-- a sharper frame there raised the candidate's score and was adopted anyway, because
"reverting to the jittery frame because it flatters us would be gate-tuning with the sign
flipped". The candidate's number is reported but takes no part in the choice.

Candidates. The incumbent builds the across-axis from the SHOULDERS. Shoulders articulate:
the glenohumeral joints swing with the arms, which is exactly why the multi-point rigid
Procrustes fit failed (§6i). The pelvis does not. So the alternatives replace or blend the
across-axis with a hip-derived one, keeping the up-axis and the orthogonalisation identical.

Blind to: accuracy. A sharper frame is one that agrees better with MAMMA's frame, and
MAMMA is an instrument, not truth. This can make the gate more sensitive; it cannot make
either head correct.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_gate as hg  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX  # noqa: E402

TRACKS = Path("artifacts/commercial-multiview-soma77")
SMOOTH = 15


def smooth_rotation(frames_matrices: np.ndarray, window: int) -> np.ndarray:
    """Smooth a frame sequence AS ROTATIONS (a frame is nonlinear in its landmarks)."""
    out = np.array(frames_matrices, dtype=float, copy=True)
    n = len(out)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        block = frames_matrices[lo:hi]
        block = block[np.isfinite(block).all(axis=(1, 2))]
        if len(block) == 0:
            continue
        u, _, vt = np.linalg.svd(block.mean(axis=0))
        d = np.sign(np.linalg.det(u @ vt))
        out[i] = u @ np.diag([1.0, 1.0, d]) @ vt
    return out


def build(pos: np.ndarray, across: str) -> np.ndarray:
    up = pos[:, JOINT_INDEX["neck"]] - pos[:, JOINT_INDEX["root"]]
    sh = pos[:, JOINT_INDEX["left_shoulder"]] - pos[:, JOINT_INDEX["right_shoulder"]]
    hip = pos[:, JOINT_INDEX["left_hip"]] - pos[:, JOINT_INDEX["right_hip"]]
    if across == "shoulders":
        ax = sh
    elif across == "hips":
        ax = hip
    elif across == "mean":
        ax = hg.unit(sh) + hg.unit(hip)
    else:
        raise ValueError(across)
    raw = hg.frame_from(up, ax)
    return smooth_rotation(raw, SMOOTH)


def main() -> None:
    solved = np.load(hg.OUT / "head-solve-shipped.npz")
    soma = hg.triangulate(
        [hg.SOMA_FRAME[n] for n in hg.SOMA_FRAME] + [hg.HEAD[n] for n in hg.NAMES])[0]
    mapping = hg.mamma_index_for(np.load(hg.OUT / "three-detector-cache.npz")["apple_vision"])
    rows = {}
    for subject in range(2):
        params = np.load(hg.MA3D / f"smplx_params_body_id-{mapping[subject]:02d}.npz", allow_pickle=True)
        joints = np.load(hg.MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz",
                         allow_pickle=True)["pred_joints"].astype(np.float64)
        m_head = hg.chain_world(params["smplx_pose"].astype(np.float64),
                                (hg.PELVIS, hg.SPINE1, hg.SPINE2, hg.SPINE3, hg.NECK, hg.HEAD_J))
        m_thorax = hg.frame_from(joints[:, hg.M_NECK] - joints[:, hg.M_PELVIS],
                                 joints[:, hg.M_L_SH] - joints[:, hg.M_R_SH])
        m_rel = np.einsum("nji,njk->nik", m_thorax, m_head)
        m_travel = float(np.nanpercentile(
            hg.travel(hg.mean_removed(m_rel), np.ones(len(m_rel), bool)), 95))
        pos = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        cand = solved[f"subject_{subject:02d}_head_world"]
        ok = np.isfinite(pos).all(axis=(1, 2))
        for across in ("shoulders", "hips", "mean"):
            th = np.full((len(pos), 3, 3), np.nan)
            th[ok] = build(pos, across)[ok]
            rel_o = np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0), m_head)
            rel_c = np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0), cand)
            oracle = hg.score(hg.mean_removed(rel_o, ok), hg.mean_removed(m_rel, ok), ok, m_travel)
            candid = hg.score(hg.mean_removed(rel_c, ok), hg.mean_removed(m_rel, ok), ok, m_travel)
            rows.setdefault(across, {})[subject] = {
                "oracle_p95": oracle["P1_agreement_with_mamma_deg"]["p95"],
                "oracle_median": oracle["P1_agreement_with_mamma_deg"]["median"],
                "candidate_p95": candid["P1_agreement_with_mamma_deg"]["p95"],
                "candidate_verdict": candid["verdict"],
            }
    print(f"{'across-axis':12s} {'ORACLE p95 s0':>14s} {'s1':>8s} {'| cand p95 s0':>15s} {'s1':>8s}")
    for across, r in rows.items():
        print(f"{across:12s} {r[0]['oracle_p95']:14.2f} {r[1]['oracle_p95']:8.2f} "
              f"{r[0]['candidate_p95']:15.2f} {r[1]['candidate_p95']:8.2f}   "
              f"{r[0]['candidate_verdict']}/{r[1]['candidate_verdict']}")
    best = min(rows, key=lambda a: rows[a][0]["oracle_p95"] + rows[a][1]["oracle_p95"])
    print(f"\nSHARPEST BY ORACLE (the pre-registered rule): {best}")
    (hg.OUT / "thorax-frame-choice.json").write_text(json.dumps(
        {"rule": "lowest summed ORACLE p95; candidate takes no part", "rows":
         {a: {str(k): v for k, v in r.items()} for a, r in rows.items()}, "chosen": best}, indent=2))


if __name__ == "__main__":
    main()
