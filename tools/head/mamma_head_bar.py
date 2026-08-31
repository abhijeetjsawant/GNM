#!/usr/bin/env python3
"""Score MAMMA's own head and neck on THIS footage. The bar, measured not assumed.

**Rows are keyed by OUR subject index, not MAMMA's.** They differ: MAMMA's
`body_id-00` is our subject 1. Pairing by index silently crosses the two performers in
every parity comparison. See `subject_map.py`.

`docs/HEAD_FEET_HANDS_PLAN.md` §7 requires the bar be measured before any parity
claim. MAMMA's retained fit carries `smplx_pose` (150, 165) = 55 joints x 3
axis-angle, so its head *rigid* orientation is directly readable: compose the
parent chain pelvis(0) -> spine1(3) -> spine2(6) -> spine3(9) -> neck(12) ->
head(15), verified against `kintree_table` in the local SMPLX_NEUTRAL.npz.

Reported statistics deliberately mirror HEAD_FEET_HANDS_PLAN §1b/§1c so the bar
and our numbers share a denominator.

Blind to:
  * ACCURACY. Nothing here is ground truth. These are tracking statistics of an
    estimate, and MAMMA's smoothness is partly its own temporal prior -- a smooth
    head can be smoothly wrong.
  * The jaw. `run_args.json` names `smplx_locked_head`; joint 22 is identically
    zero on every frame of both subjects. MAMMA estimates no jaw here at all.
Never scored against anything named `gt_` in this tree: CLAUDE.md records those
as byte-copies of `pred_`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

MA3D = Path(
    "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
)
SMPLX = Path(".cache/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz")

PELVIS, SPINE1, SPINE2, SPINE3 = 0, 3, 6, 9
NECK, HEAD, JAW, L_EYE, R_EYE = 12, 15, 22, 23, 24
L_SHOULDER, R_SHOULDER = 16, 17


def rodrigues(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle -> (N, 3, 3)."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    skew = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(*aa.shape[:-1], 3, 3)
    t = theta[..., None]
    eye = np.broadcast_to(np.eye(3), skew.shape)
    return eye + np.sin(t) * skew + (1.0 - np.cos(t)) * (skew @ skew)


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of the relative rotation, in degrees, row-normalised per CLAUDE.md."""
    rel = np.einsum("nij,nkj->nik", a, b)
    trace = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def chain_world(pose: np.ndarray, chain: tuple[int, ...]) -> np.ndarray:
    """World rotation of the last joint in `chain`, composing parent-relative rotations."""
    out = rodrigues(pose[:, 3 * chain[0] : 3 * chain[0] + 3])
    for joint in chain[1:]:
        out = out @ rodrigues(pose[:, 3 * joint : 3 * joint + 3])
    return out


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def verify_kintree() -> None:
    kin = np.load(SMPLX, allow_pickle=True)["kintree_table"]
    parent = {int(kin[1, i]): int(kin[0, i]) for i in range(kin.shape[1])}
    expected = {HEAD: NECK, NECK: SPINE3, SPINE3: SPINE2, SPINE2: SPINE1, SPINE1: PELVIS}
    for child, want in expected.items():
        got = parent[child]
        assert got == want, f"kintree says parent[{child}] = {got}, chain assumes {want}"
    assert parent[L_EYE] == HEAD and parent[R_EYE] == HEAD and parent[JAW] == HEAD
    print("kintree_table confirms chain 0 -> 3 -> 6 -> 9 -> 12 -> 15, eyes/jaw on 15")


def main() -> None:
    verify_kintree()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from subject_map import mamma_index_for

    mapping = mamma_index_for(
        np.load("artifacts/head-lane/three-detector-cache.npz")["apple_vision"]
    )
    report: dict[str, dict] = {
        "_subject_correspondence": {
            f"our_subject_{ours:02d}": f"mamma_body_id-{theirs:02d}"
            for ours, theirs in sorted(mapping.items())
        }
    }
    for ours in (0, 1):
        sid = f"{mapping[ours]:02d}"
        params = np.load(MA3D / f"smplx_params_body_id-{sid}.npz", allow_pickle=True)
        joints = np.load(MA3D / f"verts_joints_body_id-{sid}.npz", allow_pickle=True)["pred_joints"]
        pose = params["smplx_pose"].astype(np.float64)

        head_w = chain_world(pose, (PELVIS, SPINE1, SPINE2, SPINE3, NECK, HEAD))
        neck_w = chain_world(pose, (PELVIS, SPINE1, SPINE2, SPINE3, NECK))
        thorax_w = chain_world(pose, (PELVIS, SPINE1, SPINE2, SPINE3))

        # --- instrument self-check -------------------------------------------------
        # Eye joint POSITIONS depend only on the head's world rotation and a fixed
        # rest offset, so head_w^T @ (l_eye - r_eye) must be constant over the take.
        # If the chain composition were wrong this collapses immediately.
        eye_world = joints[:, L_EYE] - joints[:, R_EYE]
        eye_local = np.einsum("nji,nj->ni", head_w, eye_world)
        residual_mm = np.linalg.norm(eye_local - eye_local.mean(axis=0), axis=1) * 1000.0
        baseline_mm = float(np.linalg.norm(eye_world, axis=1).mean() * 1000.0)

        # --- the bar ---------------------------------------------------------------
        d_head = geodesic_deg(head_w[1:], head_w[:-1])
        d_neck = geodesic_deg(neck_w[1:], neck_w[:-1])
        d_thorax = geodesic_deg(thorax_w[1:], thorax_w[:-1])

        # Head relative to thorax: THE tracking signal. A head locked to the chest
        # is a constant here, so its spread is exactly zero.
        rel = np.einsum("nji,njk->nik", thorax_w, head_w)
        rel_mean_from = geodesic_deg(rel, np.broadcast_to(np.eye(3), rel.shape))
        rel_travel = geodesic_deg(rel[1:], rel[:-1])
        # Spread about the take's own mean relative pose, which is what a locked
        # head cannot produce however it is oriented.
        mean_rel = rel.mean(axis=0)
        u, _, vt = np.linalg.svd(mean_rel)
        mean_rel = u @ vt
        rel_spread = geodesic_deg(rel, np.broadcast_to(mean_rel, rel.shape))

        # --- eye axis vs shoulder axis, the §1b statistics -------------------------
        shoulder = joints[:, L_SHOULDER] - joints[:, R_SHOULDER]
        eye_u = eye_world / np.linalg.norm(eye_world, axis=1, keepdims=True)
        sh_u = shoulder / np.linalg.norm(shoulder, axis=1, keepdims=True)
        dot = np.einsum("ni,ni->n", eye_u, sh_u)
        axis_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        opposing = int((dot < 0).sum())

        report[f"our_subject_{ours:02d}"] = {
            "mamma_body_id": mapping[ours],
            "instrument_selfcheck": {
                "eye_baseline_mm": baseline_mm,
                "head_local_eye_vector_residual_mm": stats(residual_mm),
            },
            "frame_to_frame_deg": {
                "head": stats(d_head),
                "neck": stats(d_neck),
                "thorax_spine3": stats(d_thorax),
            },
            "head_relative_to_thorax_deg": {
                "offset_from_identity": stats(rel_mean_from),
                "spread_about_take_mean": stats(rel_spread),
                "frame_to_frame_travel": stats(rel_travel),
                "range": float(rel_spread.max() - rel_spread.min()),
            },
            "eye_axis_vs_shoulder_axis": {
                "median_deg": float(np.median(axis_angle)),
                "p95_deg": float(np.percentile(axis_angle, 95)),
                "opposing_frames": opposing,
                "frames": int(len(dot)),
            },
            "jaw_axis_angle_max_deg": float(
                np.degrees(np.linalg.norm(pose[:, 3 * JAW : 3 * JAW + 3], axis=1).max())
            ),
        }

    out = Path("artifacts/head-lane")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mamma-head-bar.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
