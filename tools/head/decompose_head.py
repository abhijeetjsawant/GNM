#!/usr/bin/env python3
"""Split the gate's P1 into the part that is OUR head and the part that is the FRAME.

The gate scores our head against MAMMA's head, each through its own torso frame. That
number therefore carries two terms that nothing in the gate separates:

  * **our head's own error** -- would survive a perfect reference, and
  * **the frame mismatch** -- our landmark-derived thorax against MAMMA's fitted chain,
    which a *perfect* head still pays.

The oracle arm bounds the second (a perfect head through our frame). This measures the
first directly: our head against the ORACLE'S head, both carried through the SAME (our)
thorax, mean-removed on the same population. The frame cancels; what is left is ours.

Run after `tools/head/gate_the_shipped_head.py` so the scored rotations are the ones the
pipeline delivers, not the prototype's.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_gate as hg  # noqa: E402


def main() -> None:
    shipped = hg.OUT / "head-solve-shipped.npz"
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    else:
        source = shipped if shipped.is_file() else hg.OUT / "head-solve.npz"
    solved = np.load(source)
    print(f"decomposing {source}"
          + ("  [the head the PIPELINE delivers]\n" if source == shipped
             else "  [WARNING: NOT the delivered head -- prototype estimator]\n"))
    soma = hg.triangulate(
        [hg.SOMA_FRAME[n] for n in hg.SOMA_FRAME] + [hg.HEAD[n] for n in hg.NAMES]
    )[0]
    mapping = hg.mamma_index_for(
        np.load(hg.OUT / "three-detector-cache.npz")["apple_vision"]
    )
    rows = []
    for subject in range(2):
        params = np.load(
            hg.MA3D / f"smplx_params_body_id-{mapping[subject]:02d}.npz", allow_pickle=True
        )
        joints = np.load(
            hg.MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz", allow_pickle=True
        )["pred_joints"].astype(np.float64)
        m_head = hg.chain_world(
            params["smplx_pose"].astype(np.float64),
            (hg.PELVIS, hg.SPINE1, hg.SPINE2, hg.SPINE3, hg.NECK, hg.HEAD_J),
        )
        m_thorax = hg.frame_from(
            joints[:, hg.M_NECK] - joints[:, hg.M_PELVIS],
            joints[:, hg.M_L_SH] - joints[:, hg.M_R_SH],
        )
        m_relative = np.einsum("nji,njk->nik", m_thorax, m_head)
        m_dev = hg.mean_removed(m_relative)
        m_travel_p95 = float(
            np.nanpercentile(hg.travel(m_dev, np.ones(len(m_dev), bool)), 95)
        )

        smoothed = np.load(
            f"artifacts/commercial-multiview-soma77/subject-{subject:02d}.body-track.npz"
        )["triangulated_world_positions_z_up_m"]
        torso_ok = np.isfinite(smoothed).all(axis=(1, 2))
        thorax = np.full((len(smoothed), 3, 3), np.nan)
        thorax[torso_ok] = hg._thorax_frames(smoothed)[torso_ok]

        candidate_world = solved[f"subject_{subject:02d}_head_world"]
        rel_c = np.einsum("nji,njk->nik", np.nan_to_num(thorax, nan=0.0), candidate_world)
        rel_o = np.einsum("nji,njk->nik", np.nan_to_num(thorax, nan=0.0), m_head)

        # what the gate scores: candidate against MAMMA, each through its own frame
        gate = hg.score(
            hg.mean_removed(rel_c, torso_ok), hg.mean_removed(m_relative, torso_ok),
            torso_ok, m_travel_p95,
        )
        # the floor a PERFECT head still pays: oracle through our frame vs MAMMA's
        floor = hg.score(
            hg.mean_removed(rel_o, torso_ok), hg.mean_removed(m_relative, torso_ok),
            torso_ok, m_travel_p95,
        )
        # purely ours: candidate against the oracle, both through OUR frame
        ours = hg.score(
            hg.mean_removed(rel_c, torso_ok), hg.mean_removed(rel_o, torso_ok),
            torso_ok, m_travel_p95,
        )
        rows.append((subject, gate, floor, ours))

    print(f"{'':46s} {'subject 0':>12s} {'subject 1':>12s}")
    for label, key, field in (
        ("what the gate scores (cand vs MAMMA, own frames)", "gate", "p95"),
        ("the oracle floor (a PERFECT head, our frame)", "floor", "p95"),
        ("purely OUR head (cand vs oracle, same frame)", "ours", "p95"),
        ("   ...median", "ours", "median"),
    ):
        vals = []
        for subject, gate, floor, ours in rows:
            arm = {"gate": gate, "floor": floor, "ours": ours}[key]
            vals.append(arm["P1_agreement_with_mamma_deg"][field])
        print(f"{label:46s} {vals[0]:11.2f}° {vals[1]:11.2f}°")
    key = "P1_agreement_with_mamma_deg"
    d_floor = abs(rows[0][2][key]["p95"] - rows[1][2][key]["p95"])
    d_ours = abs(rows[0][3][key]["p95"] - rows[1][3][key]["p95"])
    print(f"\nfloors differ by {d_floor:.2f}°; our own error differs by {d_ours:.2f}°")
    print("=> the split between performers is mostly the FRAME"
          if d_floor > d_ours else
          "=> the split between performers is mostly OUR HEAD")


if __name__ == "__main__":
    main()
