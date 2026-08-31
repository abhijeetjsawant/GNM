#!/usr/bin/env python3
"""Is the 1.18 deg miss RESOLVABLE on this take? Moving-block bootstrap, per CLAUDE.md.

CLAUDE.md, standing: *"Whole-take medians on one 150-frame take are not robust passes.
Per-frame agreement here has lag-1 autocorrelation 0.99, so ordinary resampling is invalid;
a moving-block bootstrap put a 7.54 deg median against an 8 deg band at P(fail) = 0.48.
Quote a margin only with a block bootstrap behind it, and pair it against the oracle on
identical draws so the fragility is attributed to the right term."*

**The 21.18 deg against the 20 deg band has never had that bootstrap behind it.** This
supplies it.

This does NOT change the verdict and cannot: the point estimate is 21.18 and the band is
20, so subject 1 fails. What it answers is a different and decision-relevant question --
whether a 1.18 deg gap is *resolvable* on 150 autocorrelated frames, or whether it is
inside the noise of the instrument. That determines whether further head work on this
fixture could ever be evaluated, not whether the head passes.

Paired: candidate and oracle are resampled on IDENTICAL block draws, so the difference
between them is not itself a random variable across draws.

Blind to: everything the gate is blind to. A resampling CI describes sampling variability
of THIS take under THIS instrument. It says nothing about accuracy, and a wide interval is
not permission to call a failure a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_gate as hg  # noqa: E402
from autoanim_gnm.commercial_multiview import _thorax_frames  # noqa: E402

TRACKS = Path("artifacts/commercial-multiview-soma77")
DRAWS = 2000
BLOCKS = (10, 20, 30)
BAND_P95 = 20.0
SEED = 20260901


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def main() -> None:
    solved = np.load(hg.OUT / "head-solve-shipped.npz")
    mapping = hg.mamma_index_for(np.load(hg.OUT / "three-detector-cache.npz")["apple_vision"])
    out = {}
    for subject in range(2):
        params = np.load(hg.MA3D / f"smplx_params_body_id-{mapping[subject]:02d}.npz", allow_pickle=True)
        joints = np.load(hg.MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz",
                         allow_pickle=True)["pred_joints"].astype(np.float64)
        m_head = hg.chain_world(params["smplx_pose"].astype(np.float64),
                                (hg.PELVIS, hg.SPINE1, hg.SPINE2, hg.SPINE3, hg.NECK, hg.HEAD_J))
        m_thorax = hg.frame_from(joints[:, hg.M_NECK] - joints[:, hg.M_PELVIS],
                                 joints[:, hg.M_L_SH] - joints[:, hg.M_R_SH])
        m_rel = np.einsum("nji,njk->nik", m_thorax, m_head)
        pos = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        ok = np.isfinite(pos).all(axis=(1, 2))
        th = np.full((len(pos), 3, 3), np.nan)
        th[ok] = _thorax_frames(pos)[ok]
        ref = hg.mean_removed(m_rel, ok)
        cand = hg.mean_removed(np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0),
                                         solved[f"subject_{subject:02d}_head_world"]), ok)
        orac = hg.mean_removed(np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0), m_head), ok)
        # per-frame agreement series, the quantity P1's p95 is taken over
        a_cand = hg.geodesic_deg(cand[ok], ref[ok])
        a_orac = hg.geodesic_deg(orac[ok], ref[ok])
        n = a_cand.size
        block_out = {}
        for block in BLOCKS:
            rng = np.random.default_rng(SEED + subject)
            pc, po = [], []
            for _ in range(DRAWS):
                idx = moving_block_indices(n, block, rng)   # IDENTICAL draw for both arms
                pc.append(np.percentile(a_cand[idx], 95))
                po.append(np.percentile(a_orac[idx], 95))
            pc, po = np.asarray(pc), np.asarray(po)
            block_out[block] = {
                "candidate_p95_ci": [float(np.percentile(pc, 2.5)), float(np.percentile(pc, 97.5))],
                "candidate_point": float(np.percentile(a_cand, 95)),
                "P_candidate_passes": float((pc <= BAND_P95).mean()),
                "oracle_p95_ci": [float(np.percentile(po, 2.5)), float(np.percentile(po, 97.5))],
                "P_oracle_passes": float((po <= BAND_P95).mean()),
                "paired_gap_median_deg": float(np.median(pc - po)),
            }
            b = block_out[block]
            print(f"subject {subject}  block {block:2d}:  candidate p95 "
                  f"{b['candidate_point']:.2f}  CI [{b['candidate_p95_ci'][0]:.2f}, "
                  f"{b['candidate_p95_ci'][1]:.2f}]  P(pass)={b['P_candidate_passes']:.2f}   "
                  f"| oracle P(pass)={b['P_oracle_passes']:.2f}  gap {b['paired_gap_median_deg']:.2f}")
        out[f"subject_{subject:02d}"] = block_out
        print()
    (hg.OUT / "bootstrap-margin.json").write_text(json.dumps(
        {"draws": DRAWS, "band_p95_deg": BAND_P95, "seed": SEED, "subjects": out}, indent=2))


if __name__ == "__main__":
    main()
