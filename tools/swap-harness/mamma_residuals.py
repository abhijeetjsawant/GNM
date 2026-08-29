"""What does a 13.5 mm-class detector's 2D residual distribution actually look like?

Reproject MAMMA's fitted landmarks through the rig and subtract its own 2D. That is
not a clean measure of MammaNet's error -- the fit CONSUMED these landmarks, so the
residual is regularised toward them and is a lower bound. What it is good for is the
SHAPE, and as a target spec: when a candidate detector goes into the swap slot, "does
its residual distribution look like this" is the acceptance question.
"""
import pickle, sys, numpy as np
sys.path.insert(0,"src")
from autoanim_gnm import commercial_multiview as cm
B="artifacts/mamma/mamma-4cam-five-second-v2/output/%s/pushing_and_lifting_from_ground"
CAMS=("A001","B001","C001","D001")
cams=cm.load_camera_rig("artifacts/soma77-full/camera-rig.json")
R=np.asarray(pickle.load(open(".cache/mamma/data/body_models/downsampled_verts/verts_512.pkl","rb")))
L=np.stack([np.load(f"{B%'ma_2d'}/{c}.npz")["landmarks"] for c in CAMS])
V=np.stack([np.load(f"{B%'ma_2d'}/{c}.npz")["visibilities"] for c in CAMS])
res_vis, res_all, logsig = [], [], []
for subj in (0,1):
    verts=np.load(f"{B%'ma_3d'}/verts_joints_body_id-{subj:02d}.npz")["pred_vertices"]
    lm=np.einsum("kv,fvc->fkc", R, verts)
    for ci in range(4):
        for f in range(0,150,2):
            for k in range(512):
                uv,z=cams[ci].project(lm[f,k])
                if z<=0: continue
                d=float(np.linalg.norm(np.asarray(uv)-L[ci,f,subj,k,:2]))
                res_all.append(d); logsig.append(L[ci,f,subj,k,2])
                if V[ci,f,subj,k]>=0.5: res_vis.append(d)
res_all=np.asarray(res_all); res_vis=np.asarray(res_vis); logsig=np.asarray(logsig)
Q=[10,25,50,75,90,95,99]
print(f"MAMMA fit-vs-own-2D residual, native 3840 px  (n={len(res_all)})")
print(f"  all landmarks     : {np.round(np.percentile(res_all,Q),2)}")
print(f"  visibility >= 0.5 : {np.round(np.percentile(res_vis,Q),2)}   (n={len(res_vis)})")
print(f"  deciles are        {Q}")
print(f"\nfor comparison, OUR detector's cross-view one-sided ladder (a different")
print(f"statistic, but the same tail question): [0.53 1.37 3.11 6.04 11.49 32.79] at 1280")
print(f"                                      = [1.59 4.11 9.33 18.1 34.5  98.4 ] at 3840")
# --- which shape fits: two-point mixture or lognormal continuum? -------------
meas=np.percentile(res_vis,[10,25,50,75,90,95]); QQ=[10,25,50,75,90,95]
rng=np.random.default_rng(5); N=200000
best_m=best_l=None
for clean in np.arange(0.5,6.01,0.25):
    for frac in np.arange(0.02,0.35,0.02):
        for bad in np.arange(4.0,60.0,2.0):
            s=np.where(rng.random(N)<frac,bad,clean)
            d=np.abs(rng.normal(0,s))*np.sqrt(2)
            got=np.percentile(d,QQ); loss=float(np.mean(np.abs(np.log(got/meas))))
            if best_m is None or loss<best_m[0]: best_m=(loss,clean,frac,bad,got)
for med in np.arange(0.5,6.01,0.25):
    for sl in np.arange(0.3,2.01,0.05):
        s=med*np.exp(rng.normal(0,sl,N)); d=np.abs(rng.normal(0,s))*np.sqrt(2)
        got=np.percentile(d,QQ); loss=float(np.mean(np.abs(np.log(got/meas))))
        if best_l is None or loss<best_l[0]: best_l=(loss,med,sl,got)
print(f"\nSHAPE FIT to the visibility>=0.5 residual:")
print(f"  two-point mixture : clean {best_m[1]:.2f} px, {best_m[2]*100:.0f}% bad at {best_m[3]:.0f} px"
      f"   |log ratio| {best_m[0]:.3f}")
print(f"  lognormal         : median {best_l[1]:.2f} px, sigma_log {best_l[2]:.2f}"
      f"                 |log ratio| {best_l[0]:.3f}")
print(f"  -> {'LOGNORMAL fits better' if best_l[0]<best_m[0] else 'MIXTURE fits better'}")
print(f"\nlog-sigma channel: median {np.median(logsig):.3f}; exp() = {np.exp(np.median(logsig)):.5f}")
for unit,scale in (("normalised by image width",3840),("normalised by 256-crop",256),("raw px",1)):
    print(f"   if it is {unit:26}: sigma = {np.exp(np.median(logsig))*scale:8.3f} px  "
          f"(residual median is {np.median(res_vis):.2f})")
