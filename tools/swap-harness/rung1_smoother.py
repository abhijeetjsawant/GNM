"""Rung 1: isolate the temporal smoother, on real footage at real acting speed.

Gate G10 measured the smoother imposing 4.4 mm median and 34.9 mm p95 of lag on
SYNTHETIC motion played back at 6x, because the owned clips were too slow to
exercise it. This is the same question on real data: MAMMA's 2D through our
triangulator gives per-landmark tracks on the one genuinely fast clip (p95 body-joint
speed 1.76 m/s), and MAMMA's own landmarks are the reference. Score before and after
smoothing, stratified by speed.

The smoothing is replicated rather than called: `_fill_and_smooth_positions` indexes
JOINT_NAMES in its failure branch and would not accept 512 landmarks. Same operations
-- linear interpolation of gaps, then Savitzky-Golay, window 9, order 2.
"""
import pickle, sys, numpy as np
sys.path.insert(0,"src")
from scipy.signal import savgol_filter
from autoanim_gnm import commercial_multiview as cm
B="artifacts/mamma/mamma-4cam-five-second-v2/output/%s/pushing_and_lifting_from_ground"
CAMS=("A001","B001","C001","D001")
cams=cm.load_camera_rig("artifacts/soma77-full/camera-rig.json")
R=np.asarray(pickle.load(open(".cache/mamma/data/body_models/downsampled_verts/verts_512.pkl","rb")))
L=np.stack([np.load(f"{B%'ma_2d'}/{c}.npz")["landmarks"] for c in CAMS])
FRAMES=150
raw_all,sm_all,spd_all=[],[],[]
for subj in (0,1):
    verts=np.load(f"{B%'ma_3d'}/verts_joints_body_id-{subj:02d}.npz")["pred_vertices"]
    target=np.einsum("kv,fvc->fkc", R, verts)              # (150,512,3) exact
    raw=np.full((FRAMES,512,3),np.nan)
    for f in range(FRAMES):
        pts=np.full((512,4,2),np.nan)
        for ci in range(4): pts[:,ci,:]=L[ci,f,subj,:,:2]
        for k in range(512):
            r=cm.triangulate_point(cams,pts[k],np.full(4,0.9),minimum_confidence=0.3,
                                   inlier_threshold_px=40.0)
            if r is not None: raw[f,k]=r.position_world_m
    out=raw.copy(); axis=np.arange(FRAMES,dtype=float)
    for k in range(512):
        good=np.isfinite(out[:,k]).all(axis=1)
        if good.sum()<2: continue
        for c in range(3):
            out[:,k,c]=np.interp(axis,axis[good],out[good,k,c])
        out[:,k]=savgol_filter(out[:,k],window_length=9,polyorder=2,axis=0,mode="interp")
    speed=np.zeros((FRAMES,512))
    speed[1:-1]=np.linalg.norm(target[2:]-target[:-2],axis=2)/2.0*30.0
    m=np.isfinite(raw).all(axis=2)
    raw_all.append(np.linalg.norm(raw[m]-target[m],axis=1)*1000)
    sm_all.append(np.linalg.norm(out[m]-target[m],axis=1)*1000)
    spd_all.append(speed[m])
raw=np.concatenate(raw_all); sm=np.concatenate(sm_all); spd=np.concatenate(spd_all)
print(f"n={len(raw)} landmark-frames, speed p50 {np.median(spd):.2f} p95 {np.percentile(spd,95):.2f} m/s\n")
print(f"{'speed band':>20} {'n':>7} {'raw':>9} {'smoothed':>10} {'change':>9}")
print("-"*60)
edges=np.percentile(spd,[0,20,40,60,80,95,100])
for i in range(6):
    s=(spd>=edges[i])&(spd<edges[i+1] if i<5 else spd<=edges[i+1])
    if s.sum()<50: continue
    print(f"{edges[i]:>8.2f}-{edges[i+1]:<7.2f} m/s {int(s.sum()):>7} {np.median(raw[s]):>6.1f} mm "
          f"{np.median(sm[s]):>7.1f} mm {np.median(sm[s])-np.median(raw[s]):>+8.1f}")
print("-"*60)
print(f"{'ALL':>20} {len(raw):>7} {np.median(raw):>6.1f} mm {np.median(sm):>7.1f} mm "
      f"{np.median(sm)-np.median(raw):>+8.1f}")
print(f"{'p95':>20} {'':>7} {np.percentile(raw,95):>6.1f} mm {np.percentile(sm,95):>7.1f} mm")
