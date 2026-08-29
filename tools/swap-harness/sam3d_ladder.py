"""SAM 3D Body through the substitution slot: the same cross-view ladder as SOMA-77.

One variable. Same footage, same frames, same cameras, same statistic -- the
one-sided epipolar disagreement between camera pairs on the same physical person.
It needs no reference and no joint-convention mapping, because it only asks whether
a detector agrees with *itself* across views, which is exactly the property that
separated MAMMA from SOMA-77 in the tail.

SOMA-77 on this footage, one-sided, 1280 px: 0.53 / 1.37 / 3.11 / 6.04 / 11.49 / 32.79
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"src")
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.commercial_multiview import _fundamental_matrix
SP=Path("/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/1272444c-d20a-4bd1-894f-9af1152c694a/scratchpad/sam3d")
CAMS=("A001","B001","C001","D001")
rig=cm.load_camera_rig("artifacts/soma77-full/camera-rig.json")
cams=tuple(c.scaled(1280,720) for c in rig)
body=np.load("artifacts/handfit-arrays/body-track.npz",allow_pickle=True)["positions"]
ROOT=cm.JOINT_INDEX["root"]
# frame_index in the files is the absolute source index; the body track is 0-based
OFFSET=60
recs={}
for cam in CAMS:
    p=SP/f"{cam}.jsonl"
    if not p.exists(): continue
    for line in p.read_text().splitlines():
        # the upstream prints progress to stdout alongside our JSON
        if line.startswith("{"):
            r=json.loads(line); recs.setdefault(r["frame_index"],{})[cam]=r["people"]
frames=sorted(recs)
print(f"{len(frames)} frames with detections: {frames}")
# associate each camera's people to a subject by projected-root proximity to its box
pick={}
for f in frames:
    for ci,cam in enumerate(CAMS):
        people=recs[f].get(cam,[])
        for s in (0,1):
            r3=body[s,f-OFFSET,ROOT]
            if not np.isfinite(r3).all(): continue
            uv,_=cams[ci].project(r3)
            best=None
            for pi,p in enumerate(people):
                b=np.asarray(p["bbox"]); c=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
                d=float(np.linalg.norm(c-uv))
                if best is None or d<best[0]: best=(d,pi)
            if best and best[0]<160: pick[(f,ci,s)]=best[1]
one=[]
for a in range(4):
    for b in range(a+1,4):
        F=_fundamental_matrix(cams[a],cams[b])
        for f in frames:
            for s in (0,1):
                ia,ib=pick.get((f,a,s)),pick.get((f,b,s))
                if ia is None or ib is None: continue
                ka=np.asarray(recs[f][CAMS[a]][ia]["pred_keypoints_2d"])
                kb=np.asarray(recs[f][CAMS[b]][ib]["pred_keypoints_2d"])
                for k in range(len(ka)):
                    sh=np.array([ka[k,0],ka[k,1],1.0]); th=np.array([kb[k,0],kb[k,1],1.0])
                    tl=F@sh; tn=np.hypot(tl[0],tl[1])
                    if tn>1e-9: one.append(abs(th@tl)/tn)
one=np.asarray(one)
Q=[10,25,50,75,90,95]
print(f"\nSAM 3D Body, one-sided cross-view epipolar, n={len(one)} at 1280 px")
print("  ", np.round(np.percentile(one,Q),2))
print("SOMA-77, identical footage and statistic:")
print("   [ 0.53  1.37  3.11  6.04 11.49 32.79]")
soma=np.asarray([0.53,1.37,3.11,6.04,11.49,32.79])
r=np.percentile(one,Q)/soma
print(f"\nratio SAM3D/SOMA-77: {np.round(r,2)}")
print(f"  median {r[2]:.2f}x   p95 {r[5]:.2f}x   <- the tail is where SOMA-77 lost to MAMMA")
