"""Does a per-joint offset calibration transfer to a body the detector sees anew?

Fit the mean 3D offset per joint on the character at yaw 0, then apply it to the
same motion rendered at yaw 90 and 180 -- a body the detector is looking at from
completely different sides. Three conditions:

* **none** -- the raw error, for reference;
* **world-frame** -- subtract a constant world vector per joint. If the offset is a
  registration artefact this works and tells us nothing about the detector;
* **local-frame** -- subtract a constant expressed in the joint's own frame, so it
  rotates with the body. If this transfers and the world one does not, the offset is
  **body-fixed**: a landmark convention, learned, and correctable in the rig.

Held-out by construction: the calibration never sees the yawed renders.
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"src"); sys.path.insert(0,"workers/commercial_multiview")
from scipy.spatial.transform import Rotation
from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.soma_body_mesh import SomaBodyMesh
from autoanim_gnm.soma_motion import _soma_world_rotations
from soma77_pose import SOMA77_TO_AUTOANIM
CAMS=("A001","B001","C001","D001")
SOMA_TO_WORLD=np.asarray([[1.,0,0],[0,0,-1.],[0,1.,0]])
cams=tuple(c.scaled(1280,720) for c in cm.load_camera_rig("artifacts/soma77-full/camera-rig.json"))
body=SomaBodyMesh(); names=list(SOMA77_TO_AUTOANIM)

def gather(out: Path):
    truth2=json.loads((out/"truth-2d.json").read_text())
    det=json.loads((out/"soma77-detections.json").read_text())
    truth3=np.load(out/"truth-3d.npz")["joints_world_m"]
    frames=truth2["frames"]; yaw=np.radians(truth2.get("yaw_degrees",0.0))
    m=np.load(Path(".cache/autoanim_gnm/gem-x/outputs")/truth2["clip"]/"soma_motion.npz",allow_pickle=True)
    wq=_soma_world_rotations(m["local_rotations_xyzw"][:frames], body.rest_world_xyzw)
    spin=np.asarray([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    world, local, ref = {}, {}, {}
    for n in names:
        soma=SOMA77_TO_AUTOANIM[n]; W,L,E=[],[],[]
        for f in range(frames):
            pts=np.full((4,2),np.nan); conf=np.zeros(4)
            for ci in range(4):
                d=det["detections"][CAMS[ci]][f]
                if d is None: continue
                pts[ci]=np.asarray(d)[soma,:2]; conf[ci]=0.9
            r=cm.triangulate_point(cams,pts,conf,inlier_threshold_px=60.0)
            if r is None: continue
            delta=r.position_world_m-truth3[f,soma]
            R=spin@SOMA_TO_WORLD@Rotation.from_quat(wq[f,soma]).as_matrix()
            W.append(delta); L.append(R.T@delta); E.append(R)
        world[n]=np.asarray(W); local[n]=np.asarray(L); ref[n]=np.asarray(E)
    return world, local, ref

train_w, train_l, _ = gather(Path("artifacts/arm-i"))
fit_world={n:v.mean(0) for n,v in train_w.items() if len(v)}
fit_local={n:v.mean(0) for n,v in train_l.items() if len(v)}
print(f"{'held-out':>18} {'no calibration':>16} {'world-frame':>14} {'local-frame':>14}")
print("-"*66)
for tag in ("artifacts/arm-i-yaw90","artifacts/arm-i-yaw180","artifacts/arm-i"):
    path=Path(tag)
    if not (path/"soma77-detections.json").exists():
        print(f"{path.name:>18}  (not rendered yet)"); continue
    W,L,R=gather(path)
    raw,wc,lc=[],[],[]
    for n in names:
        if n not in fit_world or not len(W[n]): continue
        raw.extend(np.linalg.norm(W[n],axis=1))
        wc.extend(np.linalg.norm(W[n]-fit_world[n],axis=1))
        for k in range(len(W[n])):
            lc.append(np.linalg.norm(W[n][k]-R[n][k]@fit_local[n]))
    note=" (in-sample)" if path.name=="arm-i" else ""
    print(f"{path.name:>18} {np.mean(raw)*1000:>13.1f} mm {np.mean(wc)*1000:>11.1f} mm "
          f"{np.mean(lc)*1000:>11.1f} mm{note}")
