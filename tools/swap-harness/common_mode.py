#!/usr/bin/env python3
"""Is our coherent 2D bias a per-view COMMON-MODE shift from the crop, not per-joint error?

Hypothesis (from the external research pass, C2): bbox centre/scale jitter, crop
resize conventions, half-pixel choices or pose-conditioned box movement translate
MOST joints in one image coherently. Triangulation then reads that coherent image
displacement as a coherent 3D displacement. It would masquerade as dozens of
independent joint errors, and unlike independent keypoint noise **it does not average
down with more views.**

This matters because ~18.6 px of coherent bias is now HALF our remaining detector gap
(the tail is handled by geometry; see the veto-ceiling entry in the plan), and a crop
convention is a deterministic coordinate fix that costs nothing to apply -- far
cheaper than anything that touches the detector.

Our detector is TOP-DOWN: SOMA-77 runs on a person box supplied by an earlier pass,
at CROP=256 with the model seeing only the centre 192 columns. That is exactly the
architecture where this failure mode lives.

WHAT THIS CAN AND CANNOT SHOW. The common mode is estimated against MAMMA's fit
projections, i.e. with oracle knowledge. So this measures the **ceiling** of what a
crop fix could buy, exactly as the oracle-veto pass measured the ceiling of a veto.
It does not deliver the fix.

CONTROLS, because subtracting an oracle-fitted offset flatters itself by construction:
  * SHUFFLED -- subtract the common mode fitted to a DIFFERENT frame. Same
    distribution of shifts, no correspondence. Must NOT help.
  * The residual after removal must also shrink in 2D, not only in 3D.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from autoanim_gnm import commercial_multiview as cm

CAMS = ("A001", "B001", "C001", "D001")
W, H, FIRST = 1280, 720, 60
SMPLX = {"root": 0, "left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
         "left_ankle": 7, "right_ankle": 8, "neck": 12, "left_shoulder": 16,
         "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
         "left_wrist": 20, "right_wrist": 21}
NAMES = list(SMPLX)
M3 = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
WORK = ROOT / "artifacts/commercial-multiview-soma77/work"
ASSIGN_MARGIN_PX = 30.0

cameras = tuple(c.scaled(W, H) for c in cm.load_camera_rig(ROOT / "artifacts/soma77-full/camera-rig.json"))
truth3d = np.stack([np.load(M3 / f"verts_joints_body_id-{s:02d}.npz")["pred_joints"] for s in (0, 1)])
truth3d = truth3d[:, :, [SMPLX[n] for n in NAMES], :].astype(np.float64)
S, F, J = truth3d.shape[:3]

ref = np.full((S, F, len(CAMS), J, 2), np.nan)
for s in range(S):
    for f in range(F):
        for c in range(len(CAMS)):
            for j in range(J):
                uv, d = cameras[c].project(truth3d[s, f, j])
                if d > 0:
                    ref[s, f, c, j] = uv

det = np.full((len(CAMS), F, 2, J, 3), np.nan)
for c, cname in enumerate(CAMS):
    for line in (WORK / f"{cname}-soma77-observations.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line); f = row["frame_index"] - FIRST
        if not (0 <= f < F):
            continue
        for person in row["people"][:2]:
            for j, n in enumerate(NAMES):
                v = person["joints"].get(n)
                if v is not None:
                    det[c, f, person["index"], j] = (v["x"], v["y"], v["confidence"])

obs = np.full((S, F, len(CAMS), J, 3), np.nan)
for c in range(len(CAMS)):
    for f in range(F):
        cost = np.full((2, 2), np.inf)
        for p in range(2):
            for s in range(S):
                d = np.linalg.norm(det[c, f, p, :, :2] - ref[s, f, c], axis=1)
                if np.isfinite(d).sum() >= 4:
                    cost[p, s] = np.nanmedian(d)
        if not np.isfinite(cost).all():
            continue
        direct, swapped = cost[0, 0] + cost[1, 1], cost[0, 1] + cost[1, 0]
        if abs(swapped - direct) < ASSIGN_MARGIN_PX:
            continue
        order = (0, 1) if direct <= swapped else (1, 0)
        for p in range(2):
            obs[order[p], f, c] = det[c, f, p]

# ---- decompose the per-(subject, frame, camera) residual
resid = obs[..., :2] - ref
common = np.full((S, F, len(CAMS), 2), np.nan)
scale_fit = np.full((S, F, len(CAMS)), np.nan)
extent = np.full((S, F, len(CAMS)), np.nan)
for s in range(S):
    for f in range(F):
        for c in range(len(CAMS)):
            r = resid[s, f, c]
            m = np.isfinite(r).all(axis=1)
            if m.sum() < 6:
                continue
            common[s, f, c] = np.median(r[m], axis=0)           # robust translation
            p = obs[s, f, c, m, :2]
            extent[s, f, c] = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
            q = ref[s, f, c, m]
            pc, qc = p - p.mean(0), q - q.mean(0)
            denom = float((qc * qc).sum())
            if denom > 1e-9:
                scale_fit[s, f, c] = float((pc * qc).sum() / denom)

valid = np.isfinite(resid).all(axis=-1)
mag = np.linalg.norm(resid, axis=-1)[valid]
cm_mag = np.linalg.norm(common, axis=-1)
cm_ok = np.isfinite(cm_mag)
after = np.linalg.norm(resid - common[..., None, :], axis=-1)[valid]

print(f"per-(subject,frame,camera) common-mode translation, {int(cm_ok.sum())} view-frames")
print(f"  magnitude  median {np.nanmedian(cm_mag[cm_ok]):6.2f} px   p90 {np.nanpercentile(cm_mag[cm_ok],90):6.2f}   "
      f"max {np.nanmax(cm_mag[cm_ok]):6.2f}")
print(f"  implied scale factor  median {np.nanmedian(scale_fit[cm_ok]):.4f}  "
      f"p05 {np.nanpercentile(scale_fit[cm_ok],5):.4f}  p95 {np.nanpercentile(scale_fit[cm_ok],95):.4f}")
print(f"\n2D residual magnitude (px at {W}):")
print(f"  as measured        median {np.median(mag):6.2f}  p90 {np.percentile(mag,90):6.2f}  p95 {np.percentile(mag,95):6.2f}")
print(f"  common mode removed median {np.median(after):6.2f}  p90 {np.percentile(after,90):6.2f}  p95 {np.percentile(after,95):6.2f}")
print(f"  --> the common-mode component is {100*(1-np.median(after)/np.median(mag)):.0f}% of the median residual")
ok = np.isfinite(extent) & cm_ok
print(f"\ncorrelation of common-mode magnitude with the person's 2D extent (a crop proxy): "
      f"r = {np.corrcoef(extent[ok], cm_mag[ok])[0,1]:+.3f}")


def triangulate(o):
    out = np.full((S, F, J, 3), np.nan)
    for s in range(S):
        for f in range(F):
            for j in range(J):
                pts = o[s, f, :, j, :2]
                if np.isfinite(pts).all(axis=1).sum() < 2:
                    continue
                r = cm.triangulate_point(cameras, pts, np.nan_to_num(o[s, f, :, j, 2], nan=0.0), pixel_scale=1.0)
                if r is not None:
                    out[s, f, j] = r.position_world_m
    return out


# Split the common mode into a STATIC per-camera part and a TIME-VARYING part.
# This decides whether the effect is fixable at all: a static offset is a
# convention/calibration constant you solve once, while a per-frame shift needs the
# crop itself fixed. Only the static arm is computable in production.
static = np.full_like(common, np.nan)
for c in range(len(CAMS)):
    v = common[:, :, c]
    static[:, :, c] = np.nanmedian(v.reshape(-1, 2), axis=0)
timevar = common - static
sm = np.linalg.norm(static, axis=-1); tm = np.linalg.norm(timevar, axis=-1)
good = np.isfinite(sm) & np.isfinite(tm)
print(f"\ncommon-mode split:")
for c, cname in enumerate(CAMS):
    print(f"  {cname}: static {np.linalg.norm(np.nanmedian(common[:, :, c].reshape(-1,2), axis=0)):5.2f} px   "
          f"time-varying median {np.nanmedian(np.linalg.norm(timevar[:, :, c], axis=-1)):5.2f} px")
print(f"  overall: static median {np.nanmedian(sm[good]):5.2f} px, "
      f"time-varying median {np.nanmedian(tm[good]):5.2f} px "
      f"-> {'STATIC dominates (fixable once)' if np.nanmedian(sm[good]) > np.nanmedian(tm[good]) else 'TIME-VARYING dominates (needs the crop fixed)'}")

base = triangulate(obs)
keep = np.isfinite(base).all(axis=-1)
e_base = np.linalg.norm(base - truth3d, axis=-1) * 1000.0

fixed = obs.copy(); fixed[..., :2] -= common[..., None, :]
e_fix = np.linalg.norm(triangulate(fixed) - truth3d, axis=-1) * 1000.0

rng = np.random.default_rng(20260830)
perm = rng.permutation(F)
shuf = obs.copy(); shuf[..., :2] -= common[:, perm][..., None, :]
e_shuf = np.linalg.norm(triangulate(shuf) - truth3d, axis=-1) * 1000.0

st = obs.copy(); st[..., :2] -= static[..., None, :]
e_static = np.linalg.norm(triangulate(st) - truth3d, axis=-1) * 1000.0

m = keep & np.isfinite(e_fix) & np.isfinite(e_shuf) & np.isfinite(e_static)
print(f"\n3D error vs MAMMA's fit, on the {int(m.sum())} slots the baseline resolved:")
for label, e in (("as measured", e_base),
                 ("STATIC per-camera part only", e_static),
                 ("full common mode removed (ORACLE)", e_fix),
                 ("shuffled common mode (CONTROL)", e_shuf)):
    print(f"  {label:>32}  median {np.nanmedian(e[m]):7.2f} mm   p95 {np.nanpercentile(e[m],95):7.2f} mm")
print("\nThe oracle row is a CEILING -- the shift was fitted against the reference and")
print("cannot be computed in production. The control must not reproduce it.")
