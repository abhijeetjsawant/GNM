#!/usr/bin/env python3
"""What would more cameras buy? The lever nobody in this lane has priced.

Four threads in the plan converge on camera count, and no document treats it as a
design parameter: the single-frame CRLB is quoted at 4 views; 35.3% of slots have
only 2 ELIGIBLE views, where `triangulate_point`'s inlier gate is provably helpless;
association margins halve under contact; and hands die at ~20 px crops. We are about
to specify an owned rig, so this is decided once and lived with.

Two questions, two methods, and the second is weaker -- say so.

(a) INFORMATION. Extend the Cramer-Rao bound to N synthetic views placed on the same
    ring as the real rig. This is exact for the geometry, given the noise model.
(b) REDUNDANCY. How often would a joint be starved of the >=3 eligible views the
    inlier gate needs? Extrapolated from the measured per-camera eligibility rate
    under an INDEPENDENCE assumption, which real occlusion violates -- occlusion is
    correlated across neighbouring views, so this is the OPTIMISTIC bound, not an
    estimate. Labelled as such wherever it is quoted.

CONTROL, and it must pass or (a) is void: placing exactly 4 synthetic cameras by the
same construction must reproduce the real rig's bound. If the synthetic ring is not
representative of the real one, nothing extrapolated from it means anything.

Corrected noise model throughout: clean 2.75 px, 7% bad at 36 px (one-sided).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "workers/commercial_multiview"))
from autoanim_gnm import commercial_multiview as cm
from soma77_pose import SOMA77_TO_AUTOANIM

CLEAN, BAD, FRAC = 2.75, 36.0, 0.07
F = ROOT / "artifacts/synthetic-truth"
real = tuple(c.scaled(1280, 720) for c in cm.load_camera_rig(F / "camera-rig.json"))
subj = np.load(F / "truth.npz", allow_pickle=True)["subjects_soma77_m"]
names = [n for n in cm.JOINT_INDEX if n in SOMA77_TO_AUTOANIM]
pts = np.concatenate([np.stack([subj[s, :, SOMA77_TO_AUTOANIM[n]] for n in names], axis=1).reshape(-1, 3)
                      for s in range(2)])
centroid = pts.mean(axis=0)

# --- confirm the camera convention before building any synthetic camera
c0 = real[0]
R0 = Rotation.from_quat(c0.camera_to_world_xyzw).as_matrix()
to_centre = centroid - c0.camera_center_world_m
cos_fwd = float(R0[:, 2] @ to_centre / np.linalg.norm(to_centre))
print(f"convention check: camera +Z vs direction-to-subject, cos = {cos_fwd:+.3f} "
      f"({'+Z is forward' if cos_fwd > 0.8 else 'UNEXPECTED -- stop'})")

centres = np.stack([c.camera_center_world_m for c in real])
ring_centre = centres.mean(axis=0)
radius = float(np.mean(np.linalg.norm(centres[:, :2] - ring_centre[:2], axis=1)))
height = float(np.mean(centres[:, 2]))
base_angle = float(np.arctan2(centres[0, 1] - ring_centre[1], centres[0, 0] - ring_centre[0]))
print(f"fitted ring: radius {radius:.2f} m, height {height:.2f} m, "
      f"subject centroid at {np.round(centroid, 2)}")


def synth(n):
    out = []
    for i in range(n):
        a = base_angle + 2.0 * np.pi * i / n
        pos = np.array([ring_centre[0] + radius * np.cos(a), ring_centre[1] + radius * np.sin(a), height])
        f = centroid - pos; f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up); r /= np.linalg.norm(r)
        d = np.cross(f, r)
        R = np.column_stack((r, d, f))
        out.append(cm.CalibratedCamera(
            name=f"S{i:02d}", width=c0.width, height=c0.height,
            intrinsics=c0.intrinsics.copy(), camera_center_world_m=pos,
            camera_to_world_xyzw=Rotation.from_matrix(R).as_quat()))
    return tuple(out)


def jac(cam, X, eps=1e-4):
    J = np.zeros((2, 3))
    for k in range(3):
        a = X.copy(); a[k] += eps; b = X.copy(); b[k] -= eps
        ua, _ = cam.project(a); ub, _ = cam.project(b)
        J[:, k] = (np.asarray(ua) - np.asarray(ub)) / (2 * eps)
    return J


def bound(cams, X, sigmas):
    info = np.zeros((3, 3))
    for cam, s in zip(cams, sigmas):
        if not np.isfinite(s):
            continue
        uv, depth = cam.project(X)
        if depth <= 0:
            continue
        J = jac(cam, X)
        info += J.T @ J / (s * s)
    if np.linalg.matrix_rank(info) < 3:
        return np.nan
    return float(np.sqrt(np.trace(np.linalg.inv(info)))) * 1000.0


rng = np.random.default_rng(20260830)
sample = pts[rng.choice(len(pts), 500, replace=False)]

def run(cams, label):
    n = len(cams)
    clean = [bound(cams, X, [CLEAN] * n) for X in sample]
    mix = []
    for X in sample:
        bad = rng.random(n) < FRAC
        mix.append(bound(cams, X, np.where(bad, BAD, CLEAN)))
    c = np.asarray(clean, float); m = np.asarray(mix, float)
    c = c[np.isfinite(c)]; m = m[np.isfinite(m)]
    print(f"{label:>34} {n:>3}  {np.mean(c):>8.2f} mm {np.mean(m):>10.2f} mm")
    return float(np.mean(c))

print(f"\nsingle-frame Cramer-Rao bound, clean 2.75 px / {FRAC*100:.0f}% bad at {BAD} px")
print(f"{'rig':>34} {'views':>5} {'all clean':>11} {'with outliers':>13}")
print("-" * 68)
b_real = run(real, "REAL rig (4 cameras)")
b_s4 = run(synth(4), "synthetic ring, 4  <-- CONTROL")
ok = abs(b_s4 - b_real) / b_real < 0.25
print(f"{'':>34}      CONTROL {'PASS' if ok else 'FAIL'}: synthetic-4 within "
      f"{100*abs(b_s4-b_real)/b_real:.0f}% of the real rig")
if not ok:
    print("  Synthetic placement is not representative; the rows below are void.")
else:
    for n in (5, 6, 8, 10, 12):
        run(synth(n), f"synthetic ring, {n}")

# --- (b) redundancy, from the MEASURED per-camera eligibility rate
WORK = ROOT / "artifacts/commercial-multiview-soma77/work"
CAMS = ("A001", "B001", "C001", "D001")
seen = tot = 0
for cname in CAMS:
    for line in (WORK / f"{cname}-soma77-observations.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        for person in json.loads(line)["people"][:2]:
            for n in names:
                v = person["joints"].get(n)
                tot += 1
                if v is not None and v.get("confidence", 0.0) >= 0.25:
                    seen += 1
p = seen / max(tot, 1)
print(f"\nmeasured per-camera eligibility rate p = {p:.3f}  (n={tot} joint-camera slots)")
print(f"{'views':>7} {'P(<=2 eligible)':>17} {'P(0 or 1) -- slot dies':>24}")
print("-" * 52)
from math import comb
for n in (4, 6, 8, 10):
    p_le2 = sum(comb(n, k) * p**k * (1-p)**(n-k) for k in range(0, 3))
    p_dead = sum(comb(n, k) * p**k * (1-p)**(n-k) for k in range(0, 2))
    print(f"{n:>7} {100*p_le2:>16.1f}% {100*p_dead:>23.1f}%")
print("\nINDEPENDENCE ASSUMED, which real occlusion violates -- neighbouring views")
print("occlude together, so these are OPTIMISTIC bounds, not estimates. The measured")
print("4-camera figure is 35.3% at exactly 2 eligible; compare that to the row above.")
