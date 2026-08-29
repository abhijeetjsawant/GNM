"""Does feeding SOMA-77 native-resolution frames close part of the gap, for free?

Battle 0 concluded resolution is not a lever -- but that was Apple Vision, a
whole-image model. SOMA-77 is **top-down**: it crops the person and resizes into a
256x192 input. At 1280 wide our subject is ~300 px tall and gets *upsampled* into
that crop; at 3840 there are ~900 px of real detail to downsample from. Completely
different situation, and it is a live confound in the swap comparison, where our
detector ran at 1280 and MAMMA's at 3840.

One variable: input resolution. Same source video, same ffmpeg quality, same person
boxes in normalised coordinates, same model, same reference.

The reference is MAMMA's fitted SMPL-X joints projected into this camera. Those are
not SOMA's joint definitions, so the **bias** term is meaningless here -- but it is
identical between the two arms, so the **spread** is a fair comparison and it is the
term the ablation is about.
"""
import json, sys, numpy as np, cv2, onnxruntime
from pathlib import Path
sys.path.insert(0,"workers/commercial_multiview"); sys.path.insert(0,"src")
import soma77_pose as S
from autoanim_gnm import commercial_multiview as cm
SP=Path("/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/1272444c-d20a-4bd1-894f-9af1152c694a/scratchpad/res")
sess=onnxruntime.InferenceSession(".cache/autoanim_gnm/gem-x/inputs/onnx/vitpose.onnx",providers=["CPUExecutionProvider"])
inp=sess.get_inputs()[0].name
cams=cm.load_camera_rig("artifacts/soma77-full/camera-rig.json")   # native 3840
A=cams[0]
MB="artifacts/mamma/mamma-4cam-five-second-v2/output/ma_3d/pushing_and_lifting_from_ground"
# reference: MAMMA's fitted SMPL-X joints, projected into A001 at native resolution
SM={"neck":12,"left_shoulder":16,"right_shoulder":17,"left_elbow":18,"right_elbow":19,
    "left_wrist":20,"right_wrist":21,"root":0,"left_hip":1,"right_hip":2,"left_knee":4,
    "right_knee":5,"left_ankle":7,"right_ankle":8}
J={s:np.load(f"{MB}/verts_joints_body_id-{s:02d}.npz")["pred_joints"].astype(float) for s in (0,1)}
FRAMES=range(0,150,3)
res={}
for width in (1280,3840):
    scale=width/3840.0
    err={n:[] for n in SM}
    for f in FRAMES:
        img=cv2.imread(str(SP/str(width)/f"{f:06d}.jpg"))
        for subj in (0,1):
            ref={}
            for n,ji in SM.items():
                uv,z=A.project(J[subj][f,ji])
                if z>0: ref[n]=np.asarray(uv)*scale
            if len(ref)<len(SM): continue
            joints={k:{"x":float(v[0]),"y":float(v[1]),"confidence":0.9} for k,v in ref.items()}
            box=S._square_box(joints)
            if box is None: continue
            hm=sess.run(None,{inp:S._crop(img,box)[None].astype(np.float32)})[0][0]
            d=S._decode(hm,box)
            for n in SM:
                if n in S.SOMA77_TO_AUTOANIM:
                    err[n].append(d[S.SOMA77_TO_AUTOANIM[n],:2]-ref[n])
    # report in NATIVE 3840 pixels so the two arms are commensurable
    rows=[]
    for n,v in err.items():
        if len(v)<10: continue
        v=np.asarray(v)/scale
        rows.append((n,len(v),np.linalg.norm(v.mean(0)),np.sqrt(((v-v.mean(0))**2).sum(1).mean())))
    res[width]=rows
    print(f"{width}: {len(rows)} joints, mean bias {np.mean([r[2] for r in rows]):.1f} px, "
          f"mean spread {np.mean([r[3] for r in rows]):.1f} px  (native-3840 pixels)")
print(f"\n{'joint':>16} {'1280 spread':>12} {'3840 spread':>12} {'change':>9}")
print("-"*54)
d1={r[0]:r for r in res[1280]}; d3={r[0]:r for r in res[3840]}
c=[]
for n in d1:
    if n in d3:
        c.append(d3[n][3]-d1[n][3])
        print(f"{n:>16} {d1[n][3]:>11.1f} {d3[n][3]:>11.1f} {d3[n][3]-d1[n][3]:>+8.1f}")
print("-"*54)
print(f"{'MEAN':>16} {np.mean([d1[n][3] for n in d1 if n in d3]):>11.1f} "
      f"{np.mean([d3[n][3] for n in d1 if n in d3]):>11.1f} {np.mean(c):>+8.1f}")
