#!/usr/bin/env python3
"""Is fusing four monocular MHR fits well-conditioned, or averaging four bad fits?

Fable's (e)(i), and it gates whether a fusion solver is worth building at all.

SAM 3D Body emits joints in a **body-local** frame (root-relative, Y-down) plus
`pred_cam_t`; `pred + cam_t` is the camera-frame position, verified by reprojecting
to its own `pred_keypoints_2d` at 7.5 px median. Our calibrated extrinsics carry that
into world, so four independent monocular fits can be compared directly.

The decomposition is the point. Monocular depth is the ambiguous axis: if the four
views disagree **along their own viewing rays**, the rays still intersect and fusion
is well-conditioned — each camera constrains what it is good at. If they disagree
**transversely**, fusion is averaging four differently-wrong bodies and buys little.

Reference-free: no MAMMA, no truth. **And blind in a way that must be stated before
the number is read** — four monocular fits can agree and be wrong the same way
through shared appearance bias. Agreement here is a *necessary condition* for fusion,
never an accuracy claim.
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

recs = {}
for cam in CAMS:
    for line in (SP / f"{cam}.jsonl").read_text().splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            recs.setdefault(r["frame_index"], {})[cam] = r["people"]
frames = sorted(recs)

def to_world(person, camera):
    """body-local + cam_t -> camera frame -> world, via our own extrinsics."""
    local = np.asarray(person["pred_joint_coords"])
    cam_point = local + np.asarray(person["pred_cam_t"])
    rotation = Rotation.from_quat(camera.camera_to_world_xyzw).as_matrix()
    return cam_point @ rotation.T + camera.camera_center_world_m

# associate each camera's people to a subject, as elsewhere in this harness
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

along, across, total, centroid_spread = [], [], [], []
for f in frames:
    for s in (0, 1):
        views = [(ci, to_world(recs[f][CAMS[ci]][pick[(f, ci, s)]], cams[ci]))
                 for ci in range(4) if (f, ci, s) in pick]
        if len(views) < 3:
            continue
        stack = np.stack([v for _, v in views])          # (V,127,3)
        mean = stack.mean(axis=0)
        centroid_spread.append(np.linalg.norm(stack.mean(axis=1) - mean.mean(axis=0), axis=1).mean())
        for ci, world in views:
            ray = mean - cams[ci].camera_center_world_m
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            delta = world - mean
            a = np.einsum("ij,ij->i", delta, ray)         # along the viewing ray
            t = delta - a[:, None] * ray                  # transverse to it
            along.extend(np.abs(a) * 1000)
            across.extend(np.linalg.norm(t, axis=1) * 1000)
            total.extend(np.linalg.norm(delta, axis=1) * 1000)

along, across, total = map(np.asarray, (along, across, total))
print(f"{len(total)} view-joint deviations from the four-view mean, over "
      f"{len(frames)} frames and both subjects\n")
print(f"{'':>22} {'median':>9} {'p90':>9}")
print("-" * 44)
print(f"{'total disagreement':>22} {np.median(total):>7.1f} mm {np.percentile(total,90):>7.1f} mm")
print(f"{'along the viewing ray':>22} {np.median(along):>7.1f} mm {np.percentile(along,90):>7.1f} mm")
print(f"{'transverse to it':>22} {np.median(across):>7.1f} mm {np.percentile(across,90):>7.1f} mm")
share = np.median(along) / (np.median(along) + np.median(across))
print("-" * 44)
print(f"\nalong-ray share of the disagreement: {share:.0%}")
print(f"body-centre spread across views: {np.median(centroid_spread)*1000:.0f} mm")
print()
if share > 0.7:
    print("VERDICT: disagreement is dominated by DEPTH. Four rays still intersect --")
    print("  fusion is well-conditioned and each camera constrains what it is good at.")
elif share < 0.5:
    print("VERDICT: disagreement is largely TRANSVERSE. Fusion would average four")
    print("  differently-wrong bodies; the depth-ambiguity story does not hold.")
else:
    print("VERDICT: mixed. Fusion helps on the depth component only; size the gain")
    print("  before building a solver.")
