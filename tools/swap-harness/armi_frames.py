"""Can the calibration frame be built from positions alone?

The local-frame offset removes 85% of the detector's bias, but applying it needs the
joint's orientation and our pipeline estimates positions only. So build the frame
from positions we DO have -- neighbouring joints -- and see whether it works as well
as the true orientation did.

Two schemes:
* **body frame** -- one frame for the whole subject, from the pelvis-to-neck axis
  and the shoulder line. Robust; blind to a limb rotating on its own.
* **limb frame** -- per joint, primary axis along the bone to its reference
  neighbour, secondary from the body axis. Follows limbs, degenerate when a bone
  lines up with the body axis.
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"src"); sys.path.insert(0,"workers/commercial_multiview")
from scipy.spatial.transform import Rotation
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.soma_body_mesh import SomaBodyMesh
from autoanim_gnm.soma_motion import _soma_world_rotations
from soma77_pose import SOMA77_TO_AUTOANIM
CAMS=("A001","B001","C001","D001"); SOMA_TO_WORLD=np.asarray([[1.,0,0],[0,0,-1.],[0,1.,0]])
cams=tuple(c.scaled(1280,720) for c in cm.load_camera_rig("artifacts/soma77-full/camera-rig.json"))
body=SomaBodyMesh(); names=list(SOMA77_TO_AUTOANIM)
# reference neighbour for each joint's primary axis, from anatomy
NEIGHBOUR={"left_elbow":"left_shoulder","right_elbow":"right_shoulder",
 "left_wrist":"left_elbow","right_wrist":"right_elbow","left_knee":"left_hip",
 "right_knee":"right_hip","left_ankle":"left_knee","right_ankle":"right_knee",
 "left_shoulder":"neck","right_shoulder":"neck","left_hip":"root","right_hip":"root",
 "neck":"root","nose":"neck","left_eye":"neck","right_eye":"neck","root":"neck"}

def orthonormal(primary, secondary):
    e1=primary/np.linalg.norm(primary)
    v=secondary-e1*(e1@secondary)
    if np.linalg.norm(v)<1e-6: v=np.cross(e1,[0.,0.,1.])
    e2=v/np.linalg.norm(v)
    return np.column_stack([e1,e2,np.cross(e1,e2)])

def gather(out: Path):
    t2=json.loads((out/"truth-2d.json").read_text())
    det=json.loads((out/"soma77-detections.json").read_text())
    t3=np.load(out/"truth-3d.npz")["joints_world_m"]
    frames=t2["frames"]; yaw=np.radians(t2.get("yaw_degrees",0.0))
    m=np.load(Path(".cache/autoanim_gnm/gem-x/outputs")/t2["clip"]/"soma_motion.npz",allow_pickle=True)
    wq=_soma_world_rotations(m["local_rotations_xyzw"][:frames], body.rest_world_xyzw)
    spin=np.asarray([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    rec=[]
    for f in range(frames):
        est={}
        for n in names:
            soma=SOMA77_TO_AUTOANIM[n]
            pts=np.full((4,2),np.nan); conf=np.zeros(4)
            for ci in range(4):
                d=det["detections"][CAMS[ci]][f]
                if d is None: continue
                pts[ci]=np.asarray(d)[soma,:2]; conf[ci]=0.9
            r=cm.triangulate_point(cams,pts,conf,inlier_threshold_px=60.0)
            if r is not None: est[n]=r.position_world_m
        if len(est)<len(names): continue
        # frames built from the ESTIMATED positions -- what we would have in production
        up=est["neck"]-est["root"]; across=est["left_shoulder"]-est["right_shoulder"]
        Rbody=orthonormal(up, across)
        for n in names:
            delta=est[n]-t3[f,SOMA77_TO_AUTOANIM[n]]
            ref=NEIGHBOUR[n]
            prim=est[n]-est[ref]
            Rlimb=orthonormal(prim, up) if np.linalg.norm(prim)>1e-4 else Rbody
            Rtrue=spin@SOMA_TO_WORLD@Rotation.from_quat(wq[f,SOMA77_TO_AUTOANIM[n]]).as_matrix()
            rec.append((n,delta,Rbody,Rlimb,Rtrue))
    return rec

train=gather(Path("artifacts/arm-i"))
def fit(rec,which):
    out={}
    for n in names:
        v=[R.T@d for (m,d,Rb,Rl,Rt) in rec if m==n for R in [ {"body":Rb,"limb":Rl,"true":Rt}[which] ]]
        if v: out[n]=np.mean(v,axis=0)
    return out
F={w:fit(train,w) for w in ("body","limb","true")}
print(f"{'held out':>18} {'raw':>9} {'body frame':>12} {'limb frame':>12} {'true orient':>13}")
print("-"*70)
for tag in ("artifacts/arm-i-yaw90","artifacts/arm-i-yaw180"):
    rec=gather(Path(tag)); res={"raw":[]}
    for w in ("body","limb","true"): res[w]=[]
    for (n,d,Rb,Rl,Rt) in rec:
        res["raw"].append(np.linalg.norm(d))
        for w,R in (("body",Rb),("limb",Rl),("true",Rt)):
            res[w].append(np.linalg.norm(d-R@F[w][n]))
    print(f"{Path(tag).name:>18} {np.mean(res['raw'])*1000:>8.1f} {np.mean(res['body'])*1000:>11.1f} "
          f"{np.mean(res['limb'])*1000:>11.1f} {np.mean(res['true'])*1000:>12.1f}")
