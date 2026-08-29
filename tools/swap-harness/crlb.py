"""How much estimator headroom is left, before any detector improves?

Everything measured so far prices a better *detector*. The unasked question is
whether our *estimator* is near the information limit for the 2D it already
receives. If it is, estimator work is done and the campaign is the only lever; if
it is not, there is free accuracy on the table.

The Cramer-Rao bound gives the smallest covariance any unbiased estimator can
achieve for one point from one frame: inv(sum_i J_i^T Sigma_i^-1 J_i), with J_i the
2x3 projection Jacobian at the true point. Expected position error is
sqrt(trace(cov)).

Two caveats, both load-bearing. It is a **single-frame** bound, and our pipeline
also smooths temporally and constrains limb lengths, so it can legitimately beat
this. And it assumes an unbiased estimator with known sigma, which is the ideal
case. So the bound is a reference point, not a target.
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"src"); sys.path.insert(0,"workers/commercial_multiview")
from autoanim_gnm import commercial_multiview as cm
from soma77_pose import SOMA77_TO_AUTOANIM
F=Path("artifacts/synthetic-truth")
cams=tuple(c.scaled(1280,720) for c in cm.load_camera_rig(F/"camera-rig.json"))
subj=np.load(F/"truth.npz",allow_pickle=True)["subjects_soma77_m"]
names=[n for n in cm.JOINT_INDEX if n in SOMA77_TO_AUTOANIM]
pts=np.concatenate([np.stack([subj[s,:,SOMA77_TO_AUTOANIM[n]] for n in names],axis=1).reshape(-1,3)
                    for s in range(2)])
def jacobian(cam,X,eps=1e-4):
    J=np.zeros((2,3))
    for k in range(3):
        a=X.copy(); a[k]+=eps; b=X.copy(); b[k]-=eps
        ua,_=cam.project(a); ub,_=cam.project(b)
        J[:,k]=(np.asarray(ua)-np.asarray(ub))/(2*eps)
    return J
def bound(X,sigmas):
    info=np.zeros((3,3))
    for cam,s in zip(cams,sigmas):
        if not np.isfinite(s): continue
        J=jacobian(cam,X)
        info+=J.T@J/(s*s)
    if np.linalg.matrix_rank(info)<3: return np.nan
    return float(np.sqrt(np.trace(np.linalg.inv(info))))*1000.0
CLEAN,BAD,FRAC=5.5,65.0,0.075
rng=np.random.default_rng(20260829)
sample=pts[rng.choice(len(pts),600,replace=False)]
allclean=[bound(X,[CLEAN]*4) for X in sample]
mix,oracle=[],[]
for X in sample:
    bad=rng.random(4)<FRAC
    mix.append(bound(X,np.where(bad,BAD,CLEAN)))
    s=np.where(bad,np.nan,CLEAN)
    oracle.append(bound(X,s))
def rep(name,v,measured):
    v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
    print(f"{name:>46} {np.mean(v):>7.2f} mm   measured {measured}")
print("single-frame Cramer-Rao bound, 4 cameras, this rig and these joints\n")
rep("all four views clean at 5.5 px",allclean,"16.4 mm (our clean run)")
rep("mixture, estimator uses all four views",mix,"23.5 mm (our contaminated run)")
rep("mixture, oracle drops the bad views",oracle,"19.0 mm (our perfect-visibility run)")
