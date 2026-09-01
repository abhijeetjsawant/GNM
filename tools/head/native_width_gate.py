#!/usr/bin/env python3
"""The gate on the NATIVE-WIDTH head. Closing §6s's pre-registration.

§6s promised: *"the gate result will be reported whatever it is, on both performers,
including the case where the head landmarks improve and the gate does not move."* §6t
answered the input question and skipped this with a justification. The justification was
reasonable and the promise was unconditional, so this runs it.

Both arms use the gate's own scoring (`head_gate.score`, `mean_removed`) and the pipeline's
own torso frame -- wrapped, never reimplemented. The only thing repointed is which
artifacts are read.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_gate as hg  # noqa: E402
from associate import CAMERAS  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, _thorax_frames, load_camera_rig, load_observation_jsonl,
)
from autoanim_gnm.head_orientation import solve_head_orientation  # noqa: E402

H = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}
ARMS = {
    "width_1280": dict(rig="artifacts/soma77-full/camera-rig.json",
                       work="artifacts/soma77-full/work", suffix="-observations.jsonl",
                       tracks="artifacts/commercial-multiview-soma77",
                       assoc="artifacts/head-lane/association.npz"),
    "width_3840": dict(rig="artifacts/commercial-multiview-native/camera-rig.json",
                       work="artifacts/commercial-multiview-native/work",
                       suffix="-soma77-observations.jsonl",
                       tracks="artifacts/commercial-multiview-native",
                       assoc="artifacts/head-lane/native-width_3840/association.npz"),
}


def solve(arm: dict) -> dict[int, np.ndarray]:
    rig = {c.name: c for c in load_camera_rig(Path(arm["rig"]))}
    raw = [load_observation_jsonl(Path(arm["work"]) / f"{n}{arm['suffix']}") for n in CAMERAS]
    cams = [rig[n].scaled(raw[0][0]["width"], raw[0][0]["height"]) for n in CAMERAS]
    head = [[[np.asarray([p["landmarks_soma77"][i] for i in H.values()], float)
              for p in r["people"]] for r in c] for c in raw]
    asg = np.load(arm["assoc"])["assignment"]
    frames = len(raw[0])
    out = {}
    for s in (0, 1):
        sm = np.load(f"{arm['tracks']}/subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        obs = np.full((frames, len(CAMERAS), len(H), 3), np.nan)
        for f in range(frames):
            for c in range(len(CAMERAS)):
                p = int(asg[f, s, c])
                if p >= 0:
                    obs[f, c] = head[c][f][p]
        sol = solve_head_orientation(cams, obs, tuple(H), thorax_world=_thorax_frames(sm),
                                     neck_origin_world_m=sm[:, JOINT_INDEX["neck"]])
        out[s] = sol.rotations_world
        print(f"  {s}: weight {sol.temporal_weight:g}, reprojection {sol.reprojection_px:.4f} px")
    return out


def main() -> None:
    mapping = hg.mamma_index_for(np.load(hg.OUT / "three-detector-cache.npz")["apple_vision"])
    report = {}
    for name, arm in ARMS.items():
        print(f"--- solving {name} ---")
        rot = solve(arm)
        report[name] = {}
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
            pos = np.load(f"{arm['tracks']}/subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
            ok = np.isfinite(pos).all(axis=(1, 2))
            th = np.full((len(pos), 3, 3), np.nan)
            th[ok] = _thorax_frames(pos)[ok]
            cand = hg.score(hg.mean_removed(np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0), rot[s]), ok),
                            hg.mean_removed(m_rel, ok), ok, m_travel)
            orac = hg.score(hg.mean_removed(np.einsum("nji,njk->nik", np.nan_to_num(th, nan=0.0), m_head), ok),
                            hg.mean_removed(m_rel, ok), ok, m_travel)
            report[name][f"subject_{s:02d}"] = {
                "candidate_p95": cand["P1_agreement_with_mamma_deg"]["p95"],
                "candidate_median": cand["P1_agreement_with_mamma_deg"]["median"],
                "candidate_verdict": cand["verdict"],
                "oracle_p95": orac["P1_agreement_with_mamma_deg"]["p95"],
            }
    print(f"\n{'arm':12s} {'cand p95 s0':>12s} {'s1':>8s} {'| oracle s0':>12s} {'s1':>8s}   verdicts")
    for name, r in report.items():
        a, b = r["subject_00"], r["subject_01"]
        print(f"{name:12s} {a['candidate_p95']:12.2f} {b['candidate_p95']:8.2f} "
              f"{a['oracle_p95']:12.2f} {b['oracle_p95']:8.2f}   "
              f"{a['candidate_verdict']}/{b['candidate_verdict']}")
    (hg.OUT / "native-width-gate.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
