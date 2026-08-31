import pymomentum.geometry as g, pymomentum.marker_tracking as mt, numpy as np, json
import os
# Working directory for markers, MHR assets and outputs. Override with FITTER_WORK.
S = os.environ.get('FITTER_WORK', os.path.expanduser('~/.cache/autoanim-fitter'))
A=S+'/mhr/assets'
MAP={'root':'root','neck':'c_neck','nose':'c_head','left_shoulder':'l_uparm','right_shoulder':'r_uparm',
 'left_elbow':'l_lowarm','right_elbow':'r_lowarm','left_wrist':'l_wrist','right_wrist':'r_wrist',
 'left_hip':'l_upleg','right_hip':'r_upleg','left_knee':'l_lowleg','right_knee':'r_lowleg',
 'left_ankle':'l_foot','right_ankle':'r_foot','left_eye':'l_eye','right_eye':'r_eye'}
c=g.Character.load_fbx(A+'/lod6.fbx').load_model_definition(A+'/compact_v6_1.model')
jn=list(c.skeleton.joint_names)
c=c.with_locators([g.Locator(name=k,parent=jn.index(v),offset=np.zeros(3,dtype=np.float32),
   limit_origin=np.zeros(3,dtype=np.float32),limit_weight=np.full(3,10.0,dtype=np.float32)) for k,v in MAP.items()])
d=json.load(open(S+'/markers_mhr_cm.json')); names=d['names']
arr=np.asarray(d['subjects'][1],dtype=np.float64)
md=[[g.Marker(name=lm,pos=(arr[f,names.index(lm)] if np.isfinite(arr[f,names.index(lm)]).all() else np.zeros(3)),
              occluded=not np.isfinite(arr[f,names.index(lm)]).all()) for lm in MAP] for f in range(arr.shape[0])]
pt=c.parameter_transform; zero=np.zeros(len(pt.names),dtype=np.float32)
cA=mt.CalibrationConfig(); cA.calib_frames=100; cA.locators_only=True; cA.loss_alpha=2.0; cA.max_iter=30
mt.calibrate_markers(c, zero.copy(), md, cA)
cB=mt.CalibrationConfig(); cB.calib_frames=100; cB.locators_only=False; cB.global_scale_only=False
cB.loss_alpha=2.0; cB.max_iter=30
ident,_,_=mt.calibrate_markers(c, zero.copy(), md, cB); ident=np.asarray(ident,dtype=np.float32)
tc=mt.TrackingConfig(); tc.smoothing=0.0; tc.max_iter=30; tc.loss_alpha=2.0
motion=np.asarray(mt.process_markers(c, ident, md, tc, cB, calibrate=False),dtype=np.float32)
F=70
st=np.asarray(g.model_parameters_to_skeleton_state(c, motion[F]))[...,:3]
fitroot=st[jn.index('root')]
mk=arr[F, names.index('root')]
tocap=lambda p: np.array([p[0]/100.0, -p[2]/100.0, p[1]/100.0])
print('frame', F)
print('  marker root  MHR cm', np.round(mk,1), '-> capture m', np.round(tocap(mk),3))
print('  FITTED root  MHR cm', np.round(fitroot,1), '-> capture m', np.round(tocap(fitroot),3))
print('  displacement:', round(float(np.linalg.norm(fitroot-mk))*10,1), 'mm')
