"""D4 B4 (REPORTED, NEVER SELECTED): the delivered MHR joints against MAMMA's `pred_joints`.

MAMMA is a measuring instrument and is never in the shipping path (CLAUDE.md). Nothing here
selects a constant, sets a band, or gates anything: it is a second opinion with a known
convention gap, and the gap is exactly what it measures.

Reused, not re-implemented: `tools/head/subject_map.py` resolves our subject index to MAMMA's
`body_id` from 3D pelvis agreement (pairing by index silently crosses the performers and is
invisible in every per-subject statistic taken separately); `tools/swap-harness/retarget_cost.py`'s
`PAIRS` is the our-name -> SMPL-X index table, whose owner is `mamma_scoreboard.py`. Composing
PAIRS with the delivered track's own `landmark_to_joint` gives MHR joint <-> SMPL-X index for the
15 names both tables cover. The delivered joints are read back from the GLB's own bytes.

BIAS AND SPREAD ARE REPORTED SEPARATELY, because they mean different things here. A constant
offset between an MHR joint and an SMPL-X joint of the same name is a CONVENTION difference -- two
skeletons put "the elbow" in different places inside the same arm -- and says nothing about
tracking. The spread about that constant is what moves frame to frame. "MAMMA cannot referee a
width" (CLAUDE.md) is the same point for lengths: its joints are conventions.

Both an absolute (world) and a root-relative (hip-midpoint-subtracted) reading are given; the
absolute one includes any world placement difference and the root-relative one does not.

    .venv/bin/python tools/compare/d4_b4_mamma_arm.py --delivery DIR --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
sys.path.insert(0, str(ROOT / "tools/head"))
sys.path.insert(0, str(ROOT / "tools/swap-harness"))
# allow_pickle below: this repository's own retained outputs under the gitignored artifacts/ tree.
import subject_map  # noqa: E402
from d4_glb_closure import glb_joint_positions, to_capture  # noqa: E402

MA3D = ROOT / ("artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/"
               "pushing_and_lifting_from_ground")
subject_map.MA3D = MA3D
# Copied from tools/swap-harness/retarget_cost.py, whose source is mamma_scoreboard.py.
PAIRS = {"root": 0, "neck": 12, "nose": 15, "left_shoulder": 16, "right_shoulder": 17,
         "left_elbow": 18, "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
         "left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
         "left_ankle": 7, "right_ankle": 8}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    delivery = arguments.delivery.resolve()
    capture = [np.asarray(np.load(delivery / f"subject-{s:02d}.body-track.npz",
                                  allow_pickle=True)["triangulated_world_positions_z_up_m"],
                          np.float64) for s in (0, 1)]
    mapping = subject_map.mamma_index_for(capture)
    report = {"delivery": str(delivery), "status": "REPORTED, never selected",
              "subject_to_mamma_body_id": {str(k): int(v) for k, v in mapping.items()},
              "mapped_joints": len(PAIRS), "subjects": {}}
    for subject in (0, 1):
        declared = json.loads((delivery / f"subject-{subject:02d}.body-track.json")
                              .read_text(encoding="utf-8"))["landmark_to_joint"]
        joint_names, positions = glb_joint_positions(delivery / f"subject-{subject:02d}.glb")
        ours = to_capture(positions)
        theirs = np.load(MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz",
                         allow_pickle=True)["pred_joints"].astype(np.float64)
        frames = min(ours.shape[0], theirs.shape[0])
        rows = {}
        for name, smplx in PAIRS.items():
            a = ours[:frames, joint_names.index(declared[name])]
            b = theirs[:frames, smplx]
            offset = a - b
            bias = offset.mean(axis=0)
            rows[name] = {
                "mhr_joint": declared[name], "smplx_index": smplx,
                "absolute_median_mm": round(float(np.median(np.linalg.norm(offset, axis=1))
                                                  * 1000.0), 2),
                "bias_mm": round(float(np.linalg.norm(bias) * 1000.0), 2),
                "spread_about_the_bias_median_mm": round(
                    float(np.median(np.linalg.norm(offset - bias, axis=1)) * 1000.0), 2)}
        # root-relative: the hip midpoint removed from both sides, per frame
        our_mid = 0.5 * (ours[:frames, joint_names.index(declared["left_hip"])]
                         + ours[:frames, joint_names.index(declared["right_hip"])])
        their_mid = 0.5 * (theirs[:frames, PAIRS["left_hip"]] + theirs[:frames, PAIRS["right_hip"]])
        relative = {}
        for name, smplx in PAIRS.items():
            offset = ((ours[:frames, joint_names.index(declared[name])] - our_mid)
                      - (theirs[:frames, smplx] - their_mid))
            bias = offset.mean(axis=0)
            relative[name] = {
                "median_mm": round(float(np.median(np.linalg.norm(offset, axis=1)) * 1000.0), 2),
                "bias_mm": round(float(np.linalg.norm(bias) * 1000.0), 2),
                "spread_about_the_bias_median_mm": round(
                    float(np.median(np.linalg.norm(offset - bias, axis=1)) * 1000.0), 2)}
        report["subjects"][f"subject_{subject:02d}"] = {
            "absolute": rows, "root_relative": relative,
            "absolute_all_joint_median_mm": round(
                float(np.median([r["absolute_median_mm"] for r in rows.values()])), 2),
            "absolute_all_joint_bias_median_mm": round(
                float(np.median([r["bias_mm"] for r in rows.values()])), 2),
            "absolute_all_joint_spread_median_mm": round(
                float(np.median([r["spread_about_the_bias_median_mm"] for r in rows.values()])), 2),
            "root_relative_all_joint_median_mm": round(
                float(np.median([r["median_mm"] for r in relative.values()])), 2),
            "root_relative_all_joint_bias_median_mm": round(
                float(np.median([r["bias_mm"] for r in relative.values()])), 2),
            "root_relative_all_joint_spread_median_mm": round(
                float(np.median([r["spread_about_the_bias_median_mm"]
                                 for r in relative.values()])), 2)}
        row = report["subjects"][f"subject_{subject:02d}"]
        print(f"subject {subject:02d} (MAMMA body_id-{mapping[subject]:02d}): absolute median "
              f"{row['absolute_all_joint_median_mm']} mm (bias "
              f"{row['absolute_all_joint_bias_median_mm']}, spread "
              f"{row['absolute_all_joint_spread_median_mm']}); root-relative median "
              f"{row['root_relative_all_joint_median_mm']} mm (bias "
              f"{row['root_relative_all_joint_bias_median_mm']}, spread "
              f"{row['root_relative_all_joint_spread_median_mm']})")
    Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
