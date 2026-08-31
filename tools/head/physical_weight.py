#!/usr/bin/env python3
"""Select the temporal weight by NECK PHYSIOLOGY. Pre-registered in §6l before running.

Rule: the SMALLEST weight whose head motion is physiologically possible.
Bound: p99 of per-frame head-relative-to-thorax travel <= 26.67 deg/frame (800 deg/s at
30 fps), the permissive end of the 500-800 deg/s figure already recorded in
`head_orientation.py`.

Consults our own solve only -- never MAMMA, never the gate, never reprojection. Reprojection
is printed for comparison with the incumbent rule and takes no part in the selection.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, _thorax_frames, load_camera_rig, load_observation_jsonl,
)
from autoanim_gnm.head_orientation import (  # noqa: E402
    DEFAULT_WEIGHTS, _initialise, _reprojection_px, _residuals_px, _solve_once,
    orthonormalise, rodrigues,
)
from associate import CAMERAS, OUT, RIG, WORK  # noqa: E402

HEAD_INDICES = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}
FPS = 30.0
BOUND_DEG_PER_S = 800.0
BOUND_DEG_PER_FRAME = BOUND_DEG_PER_S / FPS


def relative_travel_deg(rotations_world: np.ndarray, thorax: np.ndarray) -> np.ndarray:
    """Per-frame head rotation RELATIVE TO THE THORAX -- the neck's own motion."""
    rel = np.einsum("nji,njk->nik", np.nan_to_num(thorax, nan=0.0), rotations_world)
    ok = np.isfinite(rel).all(axis=(1, 2)) & np.isfinite(thorax).all(axis=(1, 2))
    idx = np.flatnonzero(ok)
    pairs = idx[1:][np.diff(idx) == 1]
    delta = np.einsum("nji,njk->nik", rel[pairs - 1], rel[pairs])
    trace = np.clip((np.trace(delta, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def main() -> None:
    rig = {c.name: c for c in load_camera_rig(RIG)}
    raw = [load_observation_jsonl(WORK / f"{n}-observations.jsonl") for n in CAMERAS]
    cams = [rig[n].scaled(raw[0][0]["width"], raw[0][0]["height"]) for n in CAMERAS]
    head = [[[np.asarray([p["landmarks_soma77"][i] for i in HEAD_INDICES.values()], float)
              for p in r["people"]] for r in c] for c in raw]
    asg = np.load(OUT / "association.npz")["assignment"]
    frames = len(raw[0])
    report = {"rule": "smallest weight with p99 relative travel <= %.2f deg/frame (%.0f deg/s)"
                      % (BOUND_DEG_PER_FRAME, BOUND_DEG_PER_S), "subjects": {}}
    for subject in (0, 1):
        sm = np.load(
            f"artifacts/commercial-multiview-soma77/subject-{subject:02d}.body-track.npz"
        )["triangulated_world_positions_z_up_m"]
        obs = np.full((frames, len(CAMERAS), len(HEAD_INDICES), 3), np.nan)
        for f in range(frames):
            for c in range(len(CAMERAS)):
                person = int(asg[f, subject, c])
                if person >= 0:
                    obs[f, c] = head[c][f][person]
        thorax = orthonormalise(_thorax_frames(sm))
        neck = sm[:, JOINT_INDEX["neck"]]
        t0, r0, x0 = _initialise(obs, cams, 0.25, 1.0)[:3]
        seed = _residuals_px(t0, r0, x0, obs, cams, 0.25)
        scale = max(1.4826 * float(np.median(np.abs(seed - np.median(seed)))), 1.0) if seed.size else 4.0
        rows = []
        print(f"\n=== subject {subject} ===")
        print(f"{'weight':>10s} {'p99 deg/fr':>11s} {'max deg/fr':>11s} {'deg/s p99':>10s} "
              f"{'physical?':>10s} {'reproj px':>10s}")
        for w in DEFAULT_WEIGHTS:
            sol = _solve_once(obs, cams, t0, r0, x0, weight=float(w), thorax_world=thorax,
                              neck_origin_world_m=neck, neck_sigma_m=0.010,
                              template_prior=0.5, minimum_confidence=0.25, robust_scale=scale)
            travel = relative_travel_deg(rodrigues(sol[1]), thorax)
            p99 = float(np.percentile(travel, 99)); mx = float(travel.max())
            px = float(_reprojection_px(*sol, obs, cams, 0.25))
            ok = p99 <= BOUND_DEG_PER_FRAME
            rows.append({"weight": float(w), "p99_deg_per_frame": p99,
                         "max_deg_per_frame": mx, "physical": bool(ok), "reprojection_px": px})
            print(f"{w:10.1f} {p99:11.2f} {mx:11.2f} {p99*FPS:10.0f} {str(ok):>10s} {px:10.4f}")
        chosen = next((r for r in rows if r["physical"]), None)
        report["subjects"][f"subject_{subject:02d}"] = {
            "curve": rows,
            "selected_weight": None if chosen is None else chosen["weight"],
            "selected_reprojection_px": None if chosen is None else chosen["reprojection_px"],
            "argmin_reprojection_weight": min(rows, key=lambda r: r["reprojection_px"])["weight"],
        }
        if chosen is None:
            print("  -> NO weight is physiologically possible; the solve fails, per §6l")
        else:
            print(f"  -> PHYSICAL rule selects weight {chosen['weight']:g} "
                  f"({chosen['reprojection_px']:.4f} px); argmin reprojection would select "
                  f"{min(rows, key=lambda r: r['reprojection_px'])['weight']:g}")
    (OUT / "physical-weight.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
