import pymomentum.geometry as g, pymomentum.marker_tracking as mt, numpy as np, json, sys
S='/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/694467ef-7cde-45ab-b64a-1eed22e17800/scratchpad'
A=S+'/mhr/assets'
SUBJ=int(sys.argv[1]) if len(sys.argv)>1 else 0
MAP = {'root':'root','neck':'c_neck','nose':'c_head',
 'left_shoulder':'l_uparm','right_shoulder':'r_uparm','left_elbow':'l_lowarm','right_elbow':'r_lowarm',
 'left_wrist':'l_wrist','right_wrist':'r_wrist','left_hip':'l_upleg','right_hip':'r_upleg',
 'left_knee':'l_lowleg','right_knee':'r_lowleg','left_ankle':'l_foot','right_ankle':'r_foot',
 'left_eye':'l_eye','right_eye':'r_eye'}
def build():
    c = g.Character.load_fbx(A+'/lod6.fbx').load_model_definition(A+'/compact_v6_1.model')
    jn=list(c.skeleton.joint_names)
    import os
    lw=float(os.environ.get('LOCATOR_LIMIT_WEIGHT','0'))
    # Regularise each offset toward the joint. Without this, bone length and locator
    # offset trade off freely and are not separately identifiable from 19 surface
    # landmarks -- the locators absorb the performer's proportions and the skeleton
    # stays at the mean, which is useless for a mesh that follows the skeleton.
    locs=[g.Locator(name=k, parent=jn.index(v), offset=np.zeros(3,dtype=np.float32),
                    limit_origin=np.zeros(3,dtype=np.float32),
                    limit_weight=np.full(3, lw, dtype=np.float32))
          for k,v in MAP.items()]
    return c.with_locators(locs)
data=json.load(open(S+'/markers_mhr_cm.json')); names=data['names']
arr=np.asarray(data['subjects'][SUBJ],dtype=np.float64)
md=[[g.Marker(name=lm,pos=(arr[f,names.index(lm)] if np.isfinite(arr[f,names.index(lm)]).all() else np.zeros(3)),
              occluded=not np.isfinite(arr[f,names.index(lm)]).all()) for lm in MAP]
    for f in range(arr.shape[0])]

def locator_positions(c, params):
    """World positions of the LOCATORS -- the things that correspond to our landmarks."""
    st=np.asarray(g.model_parameters_to_skeleton_state(c, params.astype(np.float32)))
    out={}
    for l in c.locators:
        t=st[l.parent]; p=t[:3]; q=t[3:7]; s=t[7]
        o=np.asarray(l.offset,dtype=np.float64)*s
        x,y,z,w=q
        # rotate offset by quaternion
        u=np.array([x,y,z]); rot=o+2*np.cross(u, np.cross(u,o)+w*o)
        out[l.name]=p+rot
    return out

SEG={'shoulder width':('left_shoulder','right_shoulder'),'pelvis -> neck':('root','neck'),
     'hip width':('left_hip','right_hip'),'upper arm':('left_shoulder','left_elbow'),
     'forearm':('left_elbow','left_wrist'),'thigh':('left_hip','left_knee'),'shin':('left_knee','left_ankle')}
MEAS={0:{'shoulder width':346.4,'pelvis -> neck':576.6,'hip width':207.1,'upper arm':287.2,
         'forearm':268.9,'thigh':399.8,'shin':396.5},
      1:{'shoulder width':363.1,'pelvis -> neck':513.3,'hip width':214.5,'upper arm':277.2,
         'forearm':258.0,'thigh':402.0,'shin':405.3}}[SUBJ]
def score(c, params, label):
    P=locator_positions(c, params)
    errs={}
    for k,(a,b) in SEG.items():
        errs[k]=(float(np.linalg.norm(P[a]-P[b]))*10.0, MEAS[k])
    m=np.mean([abs(v-t) for v,t in errs.values()])
    return errs, m

c0=build(); zero=np.zeros(len(c0.parameter_transform.names),dtype=np.float32)
_, base = score(c0, zero, 'mean')

c=build()
cfgA=mt.CalibrationConfig(); cfgA.calib_frames=100; cfgA.locators_only=True
cfgA.global_scale_only=False; cfgA.loss_alpha=2.0; cfgA.max_iter=30
mt.calibrate_markers(c, zero.copy(), md, cfgA)
_, afterA = score(c, zero, 'stage A')

cfgB=mt.CalibrationConfig(); cfgB.calib_frames=100; cfgB.locators_only=False
cfgB.global_scale_only=False; cfgB.loss_alpha=2.0; cfgB.max_iter=30
identB,_,_ = mt.calibrate_markers(c, zero.copy(), md, cfgB)
identB=np.asarray(identB,dtype=np.float32)
errsB, afterB = score(c, identB, 'stage B')

print(f"\n=== performer {SUBJ}: locator-to-locator distances vs measured landmarks (mm) ===")
print(f"{'segment':<16}{'measured':>10}{'stage B':>10}{'err':>8}")
print('-'*46)
for k,(v,t) in errsB.items(): print(f"{k:<16}{t:>10.1f}{v:>10.1f}{abs(v-t):>8.1f}")
print('-'*46)
print(f"{'MEAN ABS ERROR':<16}{'':>10}{'':>10}{afterB:>8.1f}")
print()
print(f"  MHR mean body        {base:6.1f} mm   <-- the number to beat")
print(f"  after stage A only   {afterA:6.1f} mm")
import os
print(f"  after stage A + B    {afterB:6.1f} mm   {'PASS' if afterB<base else 'FAIL -- worse than doing nothing'}")
print(f"  (locator limit_weight = {os.environ.get('LOCATOR_LIMIT_WEIGHT','0')})")
off=np.array([np.linalg.norm(np.asarray(l.offset)) for l in c.locators])*10.0
print(f"  locator offsets: median {np.median(off):.1f} mm, max {off.max():.1f} mm")
st=np.asarray(g.model_parameters_to_skeleton_state(c, identB))[...,:3]
jn2=list(c.skeleton.joint_names); P=lambda n: st[jn2.index(n)]
print(f"  JOINT shoulder width {np.linalg.norm(P('l_uparm')-P('r_uparm'))*10:.1f} mm, "
      f"upper arm {np.linalg.norm(P('l_uparm')-P('l_lowarm'))*10:.1f} mm")
np.save(S+f'/twostage_ident_{SUBJ}.npy', identB)
