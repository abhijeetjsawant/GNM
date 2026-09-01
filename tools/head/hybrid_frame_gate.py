#!/usr/bin/env python3
"""The 1280 head scored through the NATIVE torso frame. Governed by §6o's rule.

§6o pre-registered: *adopt the frame with the lowest ORACLE error, whatever it does to the
candidate.* The oracle is a perfect head, so its score is pure frame mismatch, and
minimising it sharpens the instrument. §6u produced a new frame candidate that rule had not
seen: the torso frame from the native-width build, whose oracle is 10.46/13.13 against the
incumbent's 14.14/16.30. **By the standing rule that frame is sharper and is selected.**

§6t and §6u together say why the pairing below is the measured optimum rather than a
convenience: the seventeen body joints ARE resolution-limited, the five head landmarks are
NOT. So the body wants native width and the head does not.

Every arm is scored through the SAME frame, so this is a same-denominator change to the
instrument, not a change to the candidate. The controls come with it, because a gate result
without its controls is not a gate result.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_gate as hg  # noqa: E402
from autoanim_gnm.commercial_multiview import _thorax_frames  # noqa: E402

BASE = Path("artifacts/commercial-multiview-soma77")     # 1280 -- the head solve we ship
NATIVE = Path("artifacts/commercial-multiview-native")   # 3840 -- the sharper torso frame


def main() -> None:
    solved = np.load(hg.OUT / "head-solve-shipped.npz")   # the DELIVERED 1280 head
    mapping = hg.mamma_index_for(np.load(hg.OUT / "three-detector-cache.npz")["apple_vision"])
    out = {}
    for s in (0, 1):
        params = np.load(hg.MA3D / f"smplx_params_body_id-{mapping[s]:02d}.npz", allow_pickle=True)
        joints = np.load(hg.MA3D / f"verts_joints_body_id-{mapping[s]:02d}.npz",
                         allow_pickle=True)["pred_joints"].astype(np.float64)
        m_head = hg.chain_world(params["smplx_pose"].astype(np.float64),
                                (hg.PELVIS, hg.SPINE1, hg.SPINE2, hg.SPINE3, hg.NECK, hg.HEAD_J))
        m_thorax = hg.frame_from(joints[:, hg.M_NECK] - joints[:, hg.M_PELVIS],
                                 joints[:, hg.M_L_SH] - joints[:, hg.M_R_SH])
        m_rel = np.einsum("nji,njk->nik", m_thorax, m_head)
        m_travel = float(np.nanpercentile(
            hg.travel(hg.mean_removed(m_rel), np.ones(len(m_rel), bool)), 95))
        cand_world = solved[f"subject_{s:02d}_head_world"]
        row = {}
        for label, tracks in (("frame_1280", BASE), ("frame_3840", NATIVE)):
            pos = np.load(tracks / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
            n = min(len(pos), len(cand_world))
            ok = np.isfinite(pos).all(axis=(1, 2))[:n]
            th = np.full((n, 3, 3), np.nan)
            th[ok] = _thorax_frames(pos)[:n][ok]
            rel = lambda R: np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0), R[:n])
            ref = hg.mean_removed(m_rel[:n], ok)
            cand = hg.score(hg.mean_removed(rel(cand_world), ok), ref, ok, m_travel)
            orac = hg.score(hg.mean_removed(rel(m_head), ok), ref, ok, m_travel)
            locked = np.broadcast_to(np.eye(3), (n, 3, 3))
            c1 = hg.score(locked, ref, ok, m_travel)
            row[label] = {
                "frames": int(ok.sum()),
                "oracle_p95": orac["P1_agreement_with_mamma_deg"]["p95"],
                "candidate_median": cand["P1_agreement_with_mamma_deg"]["median"],
                "candidate_p95": cand["P1_agreement_with_mamma_deg"]["p95"],
                "candidate_P2": cand["P2_spread_about_take_mean_deg_median"],
                "candidate_P4": cand["P4_travel_p95_deg"],
                "candidate_verdict": cand["verdict"],
                "C1_constant_p95": c1["P1_agreement_with_mamma_deg"]["p95"],
                "C1_constant_P2": c1["P2_spread_about_take_mean_deg_median"],
                "C1_constant_verdict": c1["verdict"],
            }
        out[f"subject_{s:02d}"] = row
    print(f"{'':22s} {'oracle':>8s} {'cand med':>9s} {'cand p95':>9s} {'P2':>7s} {'P4':>7s}  verdict   "
          f"| C1 p95 / P2 -> verdict")
    for s, row in out.items():
        for label, r in row.items():
            print(f"{s} {label:12s} {r['oracle_p95']:8.2f} {r['candidate_median']:9.2f} "
                  f"{r['candidate_p95']:9.2f} {r['candidate_P2']:7.2f} {r['candidate_P4']:7.2f}  "
                  f"{r['candidate_verdict']:7s}  | {r['C1_constant_p95']:6.2f} / {r['C1_constant_P2']:.2f} -> "
                  f"{r['C1_constant_verdict']}")
    (hg.OUT / "hybrid-frame-gate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
