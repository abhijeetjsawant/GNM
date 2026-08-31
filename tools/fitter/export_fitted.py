import pymomentum.geometry as g, pymomentum.marker_tracking as mt, numpy as np, json, sys
import os
# Working directory for markers, MHR assets and outputs. Override with FITTER_WORK.
S = os.environ.get('FITTER_WORK', os.path.expanduser('~/.cache/autoanim-fitter'))
A=S+'/mhr/assets'; SUBJ=int(sys.argv[1])
MAP={'root':'root','neck':'c_neck','nose':'c_head','left_shoulder':'l_uparm','right_shoulder':'r_uparm',
 'left_elbow':'l_lowarm','right_elbow':'r_lowarm','left_wrist':'l_wrist','right_wrist':'r_wrist',
 'left_hip':'l_upleg','right_hip':'r_upleg','left_knee':'l_lowleg','right_knee':'r_lowleg',
 'left_ankle':'l_foot','right_ankle':'r_foot','left_eye':'l_eye','right_eye':'r_eye'}
c=g.Character.load_fbx(A+'/lod6.fbx').load_model_definition(A+'/compact_v6_1.model')
jn=list(c.skeleton.joint_names)
c=c.with_locators([g.Locator(name=k,parent=jn.index(v),offset=np.zeros(3,dtype=np.float32),
        limit_origin=np.zeros(3,dtype=np.float32),limit_weight=np.full(3,10.0,dtype=np.float32))
        for k,v in MAP.items()])
data=json.load(open(S+'/markers_mhr_cm.json')); names=data['names']
arr=np.asarray(data['subjects'][SUBJ],dtype=np.float64)
md=[[g.Marker(name=lm,pos=(arr[f,names.index(lm)] if np.isfinite(arr[f,names.index(lm)]).all() else np.zeros(3)),
              occluded=not np.isfinite(arr[f,names.index(lm)]).all()) for lm in MAP] for f in range(arr.shape[0])]
pt=c.parameter_transform; zero=np.zeros(len(pt.names),dtype=np.float32)
cfgA=mt.CalibrationConfig(); cfgA.calib_frames=100; cfgA.locators_only=True; cfgA.loss_alpha=2.0; cfgA.max_iter=30
mt.calibrate_markers(c, zero.copy(), md, cfgA)
cfgB=mt.CalibrationConfig(); cfgB.calib_frames=100; cfgB.locators_only=False
cfgB.global_scale_only=False; cfgB.loss_alpha=2.0; cfgB.max_iter=30
ident,_,_=mt.calibrate_markers(c, zero.copy(), md, cfgB)
ident=np.asarray(ident,dtype=np.float32)
tcfg=mt.TrackingConfig(); tcfg.smoothing=0.0; tcfg.max_iter=30; tcfg.loss_alpha=2.0
motion=np.asarray(mt.process_markers(c, ident, md, tcfg, cfgB, calibrate=False),dtype=np.float32)
print('motion:', motion.shape)
b=g.GltfBuilder(); b.add_motion(c, fps=30.0, motion=(list(pt.names), motion))
b.save(S+f'/fitted_{SUBJ}.gltf')
print('wrote', S+f'/fitted_{SUBJ}.gltf')
