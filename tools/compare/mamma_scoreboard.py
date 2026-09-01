#!/usr/bin/env python3
"""THE SCOREBOARD. Our solve against MAMMA's, same joints, same frames, every time.

The method this exists to serve: MAMMA is a stack of separable parts -- 2D detection
(`ma_2d`), calibration (`ma_cap`), dense 3D points (`triangulated_3d_pts`, 512 of them),
a per-subject body shape (`smplx_betas`), and a model fit (`smplx_pose`). Ours has a
counterpart for each. **Substitute one part at a time and re-run this**, so when a number
moves you know which part moved it. Changing several at once buys a number nobody can
attribute.

WHAT THIS IS NOT. MAMMA is a measuring instrument and never ships (CLAUDE.md). Agreement
with it is not accuracy -- neither side has ground truth, and a change that improves
agreement could be moving toward MAMMA's error. It is the best available reference and it
is a reference, not a target to overfit.

TWO TRAPS THIS FILE HANDLES SO CALLERS DO NOT HAVE TO:
  * MAMMA's subject indices are CROSSED on this fixture -- its body_id-00 is our subject 1.
    Pairing by index silently swaps the performers and is invisible in every per-subject
    statistic taken separately. Resolved from 3D pelvis agreement via subject_map.
  * `gt_joints`/`gt_vertices` in its output are byte-copies of `pred_*`. There is no ground
    truth on disk and nothing here scores against a `gt_` variable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_joint_names  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX  # noqa: E402
from sized_skeleton import sized_skeleton  # noqa: E402
from subject_map import mamma_index_for  # noqa: E402

MA3D = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"

# our 19-joint name -> SMPL-X joint index in MAMMA's `pred_joints`
PAIRS = {
    "root": 0, "neck": 12, "nose": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
    "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8,
}
# our 19-joint name -> the rig joint whose FK position stands for it
RIG = {
    "neck": "Neck", "nose": "Head", "root": "Hips",
    "left_shoulder": "LeftUpperArm", "right_shoulder": "RightUpperArm",
    "left_elbow": "LeftLowerArm", "right_elbow": "RightLowerArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_hip": "LeftUpperLeg", "right_hip": "RightUpperLeg",
    "left_knee": "LeftLowerLeg", "right_knee": "RightLowerLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
}


def to_rig_basis(z_up: np.ndarray) -> np.ndarray:
    out = z_up[..., (0, 2, 1)].copy()
    out[..., 2] *= -1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="artifacts/commercial-multiview-soma77")
    ap.add_argument("--label", default=None, help="name this run in the report")
    args = ap.parse_args()
    tracks = ROOT / args.tracks
    label = args.label or Path(args.tracks).name

    names = json.loads((tracks / "subject-00.body-track.json").read_text())["joint_names"]
    base = skeleton_for_joint_names(names)
    ours = np.stack([
        np.load(tracks / f"subject-{s:02d}.body-track.npz")["triangulated_world_positions_z_up_m"]
        for s in (0, 1)
    ])
    mapping = mamma_index_for(ours)

    report: dict = {"label": label, "tracks": args.tracks,
                    "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}" for k, v in mapping.items()},
                    "note": "agreement with an instrument, not accuracy", "subjects": {}}
    print(f"=== {label} ===   subject map: " +
          ", ".join(f"ours {k} -> body_id-{v:02d}" for k, v in mapping.items()))
    print(f"{'joint':16s} {'capture':>9s} {'canon rig':>10s} {'sized rig':>10s}   (median mm vs MAMMA)")

    for s in (0, 1):
        track = np.load(tracks / f"subject-{s:02d}.body-track.npz")
        cap_z = track["triangulated_world_positions_z_up_m"]
        m_joints = np.load(MA3D / f"verts_joints_body_id-{mapping[s]:02d}.npz",
                           allow_pickle=True)["pred_joints"].astype(np.float64)
        n = min(len(cap_z), len(m_joints))
        fitted, limbs = sized_skeleton(base, cap_z)
        w_canon = to_rig_basis_inv = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=base)
        w_sized = forward_kinematics_positions(
            track["root_translation_m"], track["local_rotations_xyzw"], skeleton=fitted)
        # everything compared in the capture's Z-up metres
        rig_to_z = lambda w: np.stack([w[..., 0], -w[..., 2], w[..., 1]], axis=-1)
        arms = {"capture": cap_z[:n], "canon": rig_to_z(w_canon)[:n], "sized": rig_to_z(w_sized)[:n]}
        rows: dict = {}
        print(f"--- our subject {s}  (MAMMA body_id-{mapping[s]:02d}) ---")
        for name, mi in PAIRS.items():
            ref = m_joints[:n, mi]
            vals = {}
            for arm, data in arms.items():
                idx = JOINT_INDEX[name] if arm == "capture" else names.index(RIG[name])
                p = data[:, idx]
                ok = np.isfinite(p).all(axis=1) & np.isfinite(ref).all(axis=1)
                vals[arm] = float(np.median(np.linalg.norm(p[ok] - ref[ok], axis=1)) * 1000.0)
            rows[name] = vals
            print(f"{name:16s} {vals['capture']:9.1f} {vals['canon']:10.1f} {vals['sized']:10.1f}")
        med = {a: float(np.median([r[a] for r in rows.values()])) for a in arms}
        print(f"{'MEDIAN':16s} {med['capture']:9.1f} {med['canon']:10.1f} {med['sized']:10.1f}")
        report["subjects"][f"subject_{s:02d}"] = {"per_joint_mm": rows, "median_mm": med,
                                                  "measured_limbs_mm": limbs}
    out = ROOT / "artifacts/compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scoreboard-{label}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / f'scoreboard-{label}.json'}")


if __name__ == "__main__":
    main()
