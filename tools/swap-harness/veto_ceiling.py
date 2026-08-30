#!/usr/bin/env python3
"""Step 0a -- what would a PERFECT veto of the detector's tail be worth, in mm?

The lane's one durable finding is that our detector's tail is ~7.5x MAMMA's at p95.
That is a ratio, not a design spec. This converts it into the only number that
decides whether anything gets built: **if an oracle that deletes every bad
observation buys almost nothing, then no veto of any provenance is worth building
-- SAM 3D's occlusion prior included.**

Pre-registered band: **< 2 mm closes the veto question entirely. >= 5 mm makes
veto/visibility work a first-class Battle 4 requirement.**

The four redesigns both advisors required, all implemented here:

1. DEBIAS BEFORE THRESHOLDING. SOMA-77 carries a large coherent offset against
   SMPL-X joint definitions, and a raw threshold censors by direction: an event
   aligned with the bias enters the census early, an opposed one never does. The
   per-(camera, joint) median offset is removed first. This is first-order only --
   convention offsets project pose-dependently -- and it is why the threshold is
   SWEPT rather than picked.
2. CORRESPONDENCE. 14 joints map semantically onto the SMPL-X kinematic tree
   (verified: MAMMA's projected root lands within ~2 px of ours). Ears, nose and
   eyes have no SMPL-X counterpart and are excluded rather than approximated.
3. SUBJECT ASSIGNMENT FROM MAMMA'S LABELS, never root proximity -- the 160 px
   root-proximity rule is in the plan's WITHDRAWN table as the contaminant of the
   epipolar tail. Detections are assigned by agreement over all 14 joints against
   the labelled reference, and the assignment margin is published.
4. FIXED DENOMINATOR. Both arms are scored on the slots the BASELINE could
   triangulate, so a veto cannot win by attrition, and the newly-untriangulable
   count is printed beside the millimetres.

Control that can fail: a SHUFFLED veto dropping the same NUMBER of observations at
random. If it reproduces the oracle's gain, the effect is dropping, not knowing
what to drop, and nothing here is reportable.

On the reference's own error: MAMMA's fit sits ~10 mm from its own triangulation,
so no ABSOLUTE number below that is readable. The arms share the reference exactly,
so the PAIRED DIFFERENCE is readable well below the floor -- that is what is quoted.

    python3 tools/swap-harness/veto_ceiling.py
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from autoanim_gnm import commercial_multiview as cm

CAMS = ("A001", "B001", "C001", "D001")
W, H = 1280, 720
FIRST_FRAME = 60
SMPLX = {"root": 0, "left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
         "left_ankle": 7, "right_ankle": 8, "neck": 12, "left_shoulder": 16,
         "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
         "left_wrist": 20, "right_wrist": 21}
NAMES = list(SMPLX)
M3 = ROOT / "artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
WORK = ROOT / "artifacts/commercial-multiview-soma77/work"

cameras = tuple(c.scaled(W, H) for c in cm.load_camera_rig(ROOT / "artifacts/soma77-full/camera-rig.json"))
truth3d = np.stack([np.load(M3 / f"verts_joints_body_id-{s:02d}.npz")["pred_joints"] for s in (0, 1)])
truth3d = truth3d[:, :, [SMPLX[n] for n in NAMES], :].astype(np.float64)      # [2,F,14,3]
S, F, J = truth3d.shape[:3]

# Reference 2D: MAMMA's fitted joints projected into each camera.
ref = np.full((S, F, len(CAMS), J, 2), np.nan)
for s in range(S):
    for f in range(F):
        for c in range(len(CAMS)):
            for j in range(J):
                uv, depth = cameras[c].project(truth3d[s, f, j])
                if depth > 0:
                    ref[s, f, c, j] = uv

# Our detections.
raw = [[json.loads(l) for l in (WORK / f"{c}-soma77-observations.jsonl").read_text().splitlines() if l.strip()]
       for c in CAMS]
det = np.full((len(CAMS), F, 2, J, 3), np.nan)     # [cam, frame, person, joint, (x,y,conf)]
for c, rows in enumerate(raw):
    for row in rows:
        f = row["frame_index"] - FIRST_FRAME
        if not (0 <= f < F):
            continue
        for person in row["people"][:2]:
            p = person["index"]
            for j, n in enumerate(NAMES):
                v = person["joints"].get(n)
                if v is not None:
                    det[c, f, p, j] = (v["x"], v["y"], v["confidence"])

# --- redesign 3: assign our detections to MAMMA's labelled subjects, all 14 joints
ASSIGNMENT_MARGIN_PX = 30.0
obs = np.full((S, F, len(CAMS), J, 3), np.nan)
margins, excluded = [], []
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
        margin = abs(swapped - direct)
        margins.append(margin)
        # A near-tie is a coin flip, and a coin flip injects a whole body of fake
        # "tail" into BOTH the flagging and the scoring at once. Exclude the
        # camera-frame outright rather than assign it, and publish the count.
        if margin < ASSIGNMENT_MARGIN_PX:
            excluded.append((c, f)); continue
        order = (0, 1) if direct <= swapped else (1, 0)
        for p in range(2):
            obs[order[p], f, c] = det[c, f, p]

# --- redesign 1: remove the per-(camera, joint) median offset before thresholding
delta = obs[..., :2] - ref
bias = np.nanmedian(delta, axis=(0, 1))                     # [cam, joint, 2]
resid = np.linalg.norm(delta - bias[None, None], axis=-1)   # [S,F,cam,joint]
valid = np.isfinite(resid) & np.isfinite(obs[..., 2])

print(f"fixture: 150 frames x 4 cameras x 2 subjects x {J} joints at {W}x{H}")
print(f"subject assignment margin (median joint distance, px): "
      f"min {np.min(margins):.1f}  p05 {np.percentile(margins,5):.1f}  median {np.median(margins):.1f}"
      f"   [n={len(margins)} of {len(CAMS)*F} camera-frames]")
print(f"camera-frames excluded as a near-tie (margin < {ASSIGNMENT_MARGIN_PX:.0f} px): "
      f"{len(excluded)} of {len(CAMS)*F} -- a coin flip there would inject a whole body of "
      f"fake tail into the flagging AND the scoring together")
print(f"per-(camera,joint) bias removed: median magnitude {np.median(np.linalg.norm(bias,axis=-1)):.1f} px, "
      f"max {np.max(np.linalg.norm(bias,axis=-1)):.1f} px")
r = resid[valid]
print(f"debiased residual vs MAMMA's fit projections (px at {W}): "
      f"p50 {np.percentile(r,50):.1f}  p90 {np.percentile(r,90):.1f}  p95 {np.percentile(r,95):.1f}  "
      f"p99 {np.percentile(r,99):.1f}  max {r.max():.1f}   n={r.size}\n")


def triangulate_all(conf, flag=None):
    """Positions [S,F,J,3]; and if `flag` is given, how many flagged observations
    the inlier subset ACTUALLY consumed -- the question that explains a null."""
    out = np.full((S, F, J, 3), np.nan)
    flagged_seen = flagged_used = 0
    for s in range(S):
        for f in range(F):
            for j in range(J):
                pts = obs[s, f, :, j, :2]
                c = conf[s, f, :, j]
                if np.isfinite(pts).all(axis=1).sum() < 2:
                    continue
                res = cm.triangulate_point(cameras, pts, np.nan_to_num(c, nan=0.0), pixel_scale=1.0)
                if res is None:
                    continue
                out[s, f, j] = res.position_world_m
                if flag is not None:
                    for ci in range(len(CAMS)):
                        if flag[s, f, ci, j]:
                            flagged_seen += 1
                            if ci in res.used_camera_indices:
                                flagged_used += 1
    return out, (flagged_used, flagged_seen)


base_conf = np.nan_to_num(obs[..., 2], nan=0.0)
baseline, _ = triangulate_all(base_conf)
ok = np.isfinite(baseline).all(axis=-1)                     # THE fixed denominator
err_base = np.linalg.norm(baseline - truth3d, axis=-1) * 1000.0
# How much redundancy does the inlier gate actually have? With exactly two eligible
# views `triangulate_point` cannot reject anything -- the pair either both pass or
# the slot dies -- so any "the gate already handles it" conclusion is conditional on
# view count. Bound that here rather than assert it.
_vc = np.zeros(5, dtype=int)
for _s in range(S):
    for _f in range(F):
        for _j in range(J):
            _pts = obs[_s, _f, :, _j, :2]
            if np.isfinite(_pts).all(axis=1).sum() < 2:
                continue
            _r = cm.triangulate_point(cameras, _pts, np.nan_to_num(base_conf[_s, _f, :, _j], nan=0.0),
                                      pixel_scale=1.0)
            if _r is not None:
                _vc[len(_r.used_camera_indices)] += 1
# ELIGIBLE views (finite + conf >= 0.25) is what bounds the gate's power: at two
# ELIGIBLE views it cannot reject anything. "Used" can be 2 because the gate trimmed
# a bad third -- that is the mechanism working, not absence of exposure.
_el = np.zeros(5, dtype=int)
for _s in range(S):
    for _f in range(F):
        for _j in range(J):
            _n = int((np.isfinite(obs[_s, _f, :, _j, :2]).all(axis=1)
                      & (np.nan_to_num(base_conf[_s, _f, :, _j], nan=0.0) >= 0.25)).sum())
            if _n >= 2:
                _el[min(_n, 4)] += 1
print("views ELIGIBLE (the bound on the gate): " +
      "  ".join(f"{k}-view {100.0*_el[k]/max(_el.sum(),1):.1f}%" for k in (2, 3, 4)))
_rr = resid[valid]
print(f"where the tail sits, real: <14px {100.0*(_rr<14).mean():.1f}%   "
      f"14-45px (the gate's ambiguous band) {100.0*((_rr>=14)&(_rr<45)).mean():.1f}%   "
      f">=45px (trivially gated) {100.0*(_rr>=45).mean():.1f}%")
print(f"views actually used by the inlier subset: " +
      "  ".join(f"{k}-view {100.0*_vc[k]/max(_vc.sum(),1):.1f}%" for k in (2, 3, 4)) +
      f"   (at 2 views the gate CANNOT reject anything)")
print(f"baseline triangulated {ok.sum()} of {ok.size} slots; error vs MAMMA's fit "
      f"median {np.nanmedian(err_base[ok]):.1f} mm  p95 {np.nanpercentile(err_base[ok],95):.1f} mm")
print("  (absolute value is floored by the reference's own ~10 mm; the PAIRED delta below is not)\n")

rng = np.random.default_rng(20260830)
print(f"{'threshold':>10} {'flagged':>8} {'%obs':>6} | {'oracle Δp50':>12} {'Δp95':>9} | "
      f"{'shuffled Δp50':>14} {'Δp95':>9} | {'lost':>5} {'gate used':>10}")
print("-" * 100)
for tau in (10.0, 15.0, 20.0, 30.0, 45.0, 60.0):
    flag = valid & (resid > tau)
    n = int(flag.sum())
    if n == 0:
        continue
    _, (fu, fs) = triangulate_all(base_conf, flag=flag)
    conf_v = base_conf.copy(); conf_v[flag] = 0.0
    veto, _ = triangulate_all(conf_v)
    shuf = np.zeros_like(flag)
    idx = np.flatnonzero(valid.ravel())
    shuf.ravel()[rng.choice(idx, size=n, replace=False)] = True
    conf_s = base_conf.copy(); conf_s[shuf] = 0.0
    ctrl, _ = triangulate_all(conf_s)

    e_v = np.linalg.norm(veto - truth3d, axis=-1) * 1000.0
    e_s = np.linalg.norm(ctrl - truth3d, axis=-1) * 1000.0
    # Fixed denominator: slots the BASELINE resolved, scored wherever the veto put them.
    keep = ok & np.isfinite(e_v) & np.isfinite(e_s)
    # The median is dominated by systematic error against a differently-defined
    # reference; a TAIL veto would show up at p95. Report both or the band is read
    # on the wrong statistic.
    d_v = np.nanmedian(err_base[keep]) - np.nanmedian(e_v[keep])
    d_s = np.nanmedian(err_base[keep]) - np.nanmedian(e_s[keep])
    q_v = np.nanpercentile(err_base[keep], 95) - np.nanpercentile(e_v[keep], 95)
    q_s = np.nanpercentile(err_base[keep], 95) - np.nanpercentile(e_s[keep], 95)
    lost = int((ok & ~np.isfinite(veto).all(axis=-1)).sum())
    print(f"{tau:>8.0f}px {n:>8d} {100.0*n/valid.sum():>5.1f}% | {d_v:>+9.2f} mm {q_v:>+8.2f} | "
          f"{d_s:>+11.2f} mm {q_s:>+8.2f} | {lost:>5d} {100.0*fu/max(fs,1):>9.1f}%")
print(f"\nfixed denominator: slots the baseline resolved. 'lost slots' are baseline-resolved")
print(f"slots the veto could no longer triangulate -- attrition, printed beside the millimetres.")
