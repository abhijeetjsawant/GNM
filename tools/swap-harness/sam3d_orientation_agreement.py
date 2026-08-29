#!/usr/bin/env python3
"""Do SAM 3D's per-joint ORIENTATIONS agree across views, even though its positions do not?

The one integration shape still standing. Our reconstruction produces good positions
(~10 mm against MAMMA's) and **no orientations at all** — which is what blocked the
local-frame calibration in the arm (i) work, and what any skeletal animation needs
for bone twist. SAM 3D produces the reverse: orientations and hand pose, on our own
127-joint rig, but positions that disagree 118 mm across views.

So the question is not whether to replace our positions with its — rung 6 settled
that — but whether its *rotational* degrees of freedom are usable. Depth ambiguity
displaces a monocular body along the ray without necessarily rotating it, so
orientation could be far better conditioned than position. Or not; that is the
measurement.

Reference-free, and blind the same way rung 6 was: four monocular fits can agree and
be wrong together. Necessary condition, never sufficient.
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from scipy.spatial.transform import Rotation
from autoanim_gnm import commercial_multiview as cm

SP = Path("/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim"
          "/1272444c-d20a-4bd1-894f-9af1152c694a/scratchpad/sam3d")
CAMS = ("A001", "B001", "C001", "D001")
cams = tuple(c.scaled(1280, 720) for c in cm.load_camera_rig("artifacts/soma77-full/camera-rig.json"))
body = np.load("artifacts/handfit-arrays/body-track.npz", allow_pickle=True)["positions"]
ROOT, OFFSET = cm.JOINT_INDEX["root"], 60
names = json.load(open("src/autoanim_gnm/data/mhr-skeleton-v1.json"))["joint_names"]

recs = {}
for cam in CAMS:
    for line in (SP / f"{cam}.jsonl").read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            recs.setdefault(r["frame_index"], {})[cam] = r["people"]
frames = sorted(recs)

pick = {}
for f in frames:
    for ci, cam in enumerate(CAMS):
        for s in (0, 1):
            r3 = body[s, f - OFFSET, ROOT]
            if not np.isfinite(r3).all():
                continue
            uv, _ = cams[ci].project(r3)
            best = None
            for pi, p in enumerate(recs[f][cam]):
                b = np.asarray(p["bbox"]); c = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
                d = float(np.linalg.norm(c - uv))
                if best is None or d < best[0]:
                    best = (d, pi)
            if best and best[0] < 160:
                pick[(f, ci, s)] = best[1]

per_joint = {}
overall = []
for f in frames:
    for s in (0, 1):
        views = []
        for ci in range(4):
            if (f, ci, s) not in pick:
                continue
            p = recs[f][CAMS[ci]][pick[(f, ci, s)]]
            local = np.asarray(p["pred_global_rots"])           # (127,3,3), camera frame
            c2w = Rotation.from_quat(cams[ci].camera_to_world_xyzw).as_matrix()
            views.append(c2w @ local)                            # into world
        if len(views) < 3:
            continue
        stack = np.stack(views)                                  # (V,127,3,3)
        for j in range(stack.shape[1]):
            spread = []
            for a in range(len(views)):
                for b in range(a + 1, len(views)):
                    rel = stack[a, j].T @ stack[b, j]
                    ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
                    spread.append(ang)
            if spread:
                per_joint.setdefault(j, []).extend(spread)
                overall.extend(spread)

overall = np.asarray(overall)
print(f"{len(overall)} pairwise cross-view orientation disagreements, "
      f"{len(frames)} frames, both subjects\n")
print(f"  median {np.median(overall):5.1f} deg    p90 {np.percentile(overall,90):5.1f} deg")
med = {j: float(np.median(v)) for j, v in per_joint.items() if v}
order = sorted(med, key=med.get)
print(f"\n{'best 8 joints':>28}   {'worst 8 joints':>30}")
for k in range(8):
    a, b = order[k], order[-(k+1)]
    print(f"{names[a]:>22} {med[a]:5.1f}    {names[b]:>26} {med[b]:5.1f}")
core = [j for j in med if names[j] in ("root","b_spine1","b_spine2","b_neck","l_upleg",
        "r_upleg","l_lowleg","r_lowleg","l_uparm","r_uparm","l_lowarm","r_lowarm")]
if core:
    v = np.concatenate([per_joint[j] for j in core])
    print(f"\ncore body joints only: median {np.median(v):.1f} deg, p90 {np.percentile(v,90):.1f}")
print(f"\nfor scale: 10 deg of forearm twist moves a wrist ~{np.sin(np.radians(10))*0.25*1000:.0f} mm")
