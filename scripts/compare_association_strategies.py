"""Compare cross-view association strategies on an existing artifact.

The regression instrument for any change to association: runs each strategy over
the same cached detections and reports the gates from
docs/OWNED_BODY_CAPTURE_PLAN.md -- identity switches, valid coverage, temporal
rejections, bone-length stability -- plus wall clock and a frame-by-frame count
of where the assignments disagree.

    python scripts/compare_association_strategies.py [ARTIFACT_DIR]

ARTIFACT_DIR defaults to artifacts/commercial-multiview-q2 and must contain
camera-rig.json and work/<camera>-observations.jsonl.
"""
import json, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from autoanim_gnm.commercial_multiview import (
    JOINT_NAMES, associate_frame, associate_frame_graph, load_camera_rig, reconstruct_multiview)

R = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/commercial-multiview-q2")
CAMS = ("A001", "B001", "C001", "D001")
LIMBS = [("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_shoulder","left_elbow"),
         ("left_elbow","left_wrist"),("right_hip","right_knee"),("right_knee","right_ankle"),
         ("left_hip","left_knee"),("left_knee","left_ankle")]
cams = load_camera_rig(R/"camera-rig.json")
obs = [[json.loads(x) for x in (R/"work"/f"{c}-observations.jsonl").read_text().splitlines() if x.strip()]
       for c in CAMS]

def bone_sd(P):
    out=[]
    for a,b in LIMBS:
        L=np.linalg.norm(P[:,JOINT_NAMES.index(a)]-P[:,JOINT_NAMES.index(b)],axis=-1)*1000
        L=L[np.isfinite(L)]
        if len(L)>5: out.append(np.std(L))
    return float(np.median(out))

def switches(raw):
    """Identity switches: a frame where a subject's root jumps further than the
    distance between the two subjects, i.e. the labels most likely swapped."""
    root = raw[:, :, JOINT_NAMES.index("root"), :]
    n = 0
    for f in range(1, root.shape[1]):
        a, b = root[:, f - 1], root[:, f]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        keep = np.linalg.norm(a - b, axis=1).sum()
        swap = np.linalg.norm(a - b[::-1], axis=1).sum()
        if swap < keep:
            n += 1
    return n

rows = {}
for name, fn in (("exhaustive", associate_frame), ("graph", associate_frame_graph)):
    t0 = time.perf_counter()
    tracks, dg, sm, raw = reconstruct_multiview(cams, obs, subject_count=2, sample_rate_hz=30, associator=fn)
    dt = time.perf_counter() - t0
    d = dg.as_dict()
    rows[name] = dict(d=d, raw=raw, sm=sm, seconds=dt,
                      bone=float(np.median([bone_sd(sm[s]) for s in range(sm.shape[0])])),
                      switches=switches(raw))

print(f"{'associator':>12} | {'valid':>7} {'interp':>7} {'temporal rej':>12} {'bone sd':>8} "
      f"{'switches':>8} {'assoc obj':>9} {'seconds':>8}")
print("-"*90)
for name, r in rows.items():
    d = r["d"]
    print(f"{name:>12} | {d['valid_joint_fraction']*100:6.1f}% {d['interpolated_joint_fraction']*100:6.1f}% "
          f"{d['temporally_rejected_subject_frames']:>12} {r['bone']:8.1f} {r['switches']:>8} "
          f"{d['association_objective_median']:9.2f} {r['seconds']:8.1f}")

a, b = rows["exhaustive"]["raw"], rows["graph"]["raw"]
differ = ~np.isclose(a, b, equal_nan=True)
frames = np.unique(np.where(differ.any(axis=(0,2,3)))[0]) if differ.any() else np.array([])
print(f"\nframes where the two assignments differ: {len(frames)}/{a.shape[1]}")
if len(frames):
    print("  first 15:", frames[:15].tolist())
print(f"\nreprojection px  exhaustive med {rows['exhaustive']['d']['median_reprojection_error_px']:.3f} "
      f"p95 {rows['exhaustive']['d']['p95_reprojection_error_px']:.3f}")
print(f"                 graph      med {rows['graph']['d']['median_reprojection_error_px']:.3f} "
      f"p95 {rows['graph']['d']['p95_reprojection_error_px']:.3f}")
