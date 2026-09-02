"""The swap, redone against the TRUE landmark correspondence.

The first pass compared each triangulated point to the nearest vertex of a
10,475-vertex mesh, which cannot see tangential error: a point sliding along the
surface reads at most half the local vertex spacing. Measured spacing is 5.4 mm
median, so the ceiling was ~2.7 mm -- our 7.3 mm was above it, so the metric was not
saturated, but tangential error was still invisible.

verts_512.pkl turns out to be on disk: a (512, 10475) regressor, so MAMMA's own 512
landmark positions can be computed exactly from its fitted mesh. No approximation.

Also un-confounds the two channels the first pass credited together: it fed MAMMA's
`visibilities` as the confidence, so 7.3 mm was its positions PLUS its visibility
gating through our solver.
"""
import json, pickle, sys, numpy as np
from pathlib import Path
sys.path.insert(0,"src")
from autoanim_gnm import commercial_multiview as cm
B="artifacts/mamma/mamma-4cam-five-second-v2/output/%s/pushing_and_lifting_from_ground"
CAMS=("A001","B001","C001","D001")
cams=cm.load_camera_rig("artifacts/soma77-full/camera-rig.json")
R=np.asarray(pickle.load(open(".cache/mamma/data/body_models/downsampled_verts/verts_512.pkl","rb")))
print(f"regressor {R.shape}, rows sum to {R.sum(1).min():.3f}-{R.sum(1).max():.3f}")
L=np.stack([np.load(f"{B%'ma_2d'}/{c}.npz")["landmarks"] for c in CAMS])
V=np.stack([np.load(f"{B%'ma_2d'}/{c}.npz")["visibilities"] for c in CAMS])
FRAMES=range(0,150,5)
print(f"\n{'confidence source':>26} {'subject':>8} {'n/512':>8} {'true-correspondence err':>25}")
print("-"*72)
REPORT={"note":"agreement with an instrument, not accuracy",
        "rung":"MAMMA 2D + MAMMA calibration -> OUR triangulate_point, scored against MAMMA's exact 512 landmarks (verts_512 @ pred_vertices)",
        "frames":list(FRAMES),"arms":{}}
for label, use_vis in (("MAMMA visibility", True), ("uniform (positions only)", False)):
    meds=[]
    for subj in (0,1):
        verts=np.load(f"{B%'ma_3d'}/verts_joints_body_id-{subj:02d}.npz")["pred_vertices"]
        got, err = [], []
        for frame in FRAMES:
            target = R @ verts[frame]                      # exact 512 landmarks
            pts=np.full((512,4,2),np.nan); conf=np.zeros((512,4))
            for ci in range(4):
                pts[:,ci,:]=L[ci,frame,subj,:,:2]
                conf[:,ci]=V[ci,frame,subj] if use_vis else 0.9
            n=0
            for k in range(512):
                r=cm.triangulate_point(cams,pts[k],conf[k],minimum_confidence=0.3,
                                       inlier_threshold_px=40.0)
                if r is None: continue
                n+=1; err.append(np.linalg.norm(r.position_world_m-target[k]))
            got.append(n)
        e=np.asarray(err)*1000; meds.append(np.median(e))
        print(f"{label:>26} {subj:>8} {np.mean(got):>6.0f} "
              f"{np.median(e):>17.1f} mm  p90 {np.percentile(e,90):.1f}")
        REPORT["arms"].setdefault(label,{})[f"subject_{subj:02d}"]={
            "landmarks_triangulated_of_512":float(np.mean(got)),
            "median_mm":float(np.median(e)),"p90_mm":float(np.percentile(e,90))}
    print(f"{'':>26} {'MEAN':>8} {'':>8} {np.mean(meds):>17.1f} mm")
    REPORT["arms"][label]["mean_of_subject_medians_mm"]=float(np.mean(meds))
out=Path("artifacts/compare"); out.mkdir(parents=True,exist_ok=True)
(out/"swap-2d-into-our-triangulation.json").write_text(json.dumps(REPORT,indent=2))
print(f"wrote {out/'swap-2d-into-our-triangulation.json'}")
