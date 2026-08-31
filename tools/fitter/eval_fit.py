import pymomentum.geometry as g, numpy as np, sys
S='/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/694467ef-7cde-45ab-b64a-1eed22e17800/scratchpad'
A=S+'/mhr/assets'
c = g.Character.load_fbx(A+'/lod6.fbx').load_model_definition(A+'/compact_v6_1.model')
pt=c.parameter_transform; jn=list(c.skeleton.joint_names)
SUBJ=int(sys.argv[1]); d=np.load(S+f'/fit_subject_{SUBJ}.npz')
ident=d['identity'].astype(np.float32)
SEG={'shoulder width':('l_uparm','r_uparm'),'pelvis -> neck':('root','c_neck'),
     'hip width':('l_upleg','r_upleg'),'upper arm':('l_uparm','l_lowarm'),
     'forearm':('l_lowarm','l_wrist'),'thigh':('l_upleg','l_lowleg'),'shin':('l_lowleg','l_foot')}
def segs(params):
    st=np.asarray(g.model_parameters_to_skeleton_state(c, params.astype(np.float32)))[...,:3]
    return {k: float(np.linalg.norm(st[jn.index(a)]-st[jn.index(b)]))*10.0 for k,(a,b) in SEG.items()}
mean=segs(np.zeros(len(pt.names),dtype=np.float32)); fit=segs(ident)
MEAS={0:{'shoulder width':346.4,'pelvis -> neck':576.6,'hip width':207.1,'upper arm':287.2,
         'forearm':268.9,'thigh':399.8,'shin':396.5},
      1:{'shoulder width':363.1,'pelvis -> neck':513.3,'hip width':214.5,'upper arm':277.2,
         'forearm':258.0,'thigh':402.0,'shin':405.3}}[SUBJ]
print(f"=== performer {SUBJ}: did the fitter move toward THIS body? (mm) ===")
print(f"{'segment':<16}{'MHR mean':>9}{'FITTED':>9}{'measured':>10}{'mean err':>10}{'fit err':>9}")
print('-'*64)
em,ef=[],[]
for k in SEG:
    e0=abs(mean[k]-MEAS[k]); e1=abs(fit[k]-MEAS[k]); em.append(e0); ef.append(e1)
    print(f"{k:<16}{mean[k]:>9.1f}{fit[k]:>9.1f}{MEAS[k]:>10.1f}{e0:>10.1f}{e1:>9.1f}")
print('-'*64)
print(f"{'MEAN ABS ERROR':<16}{'':>28}{np.mean(em):>10.1f}{np.mean(ef):>9.1f}")
sc=np.flatnonzero(np.asarray(pt.scaling_parameters))
moved=[(pt.names[i], float(ident[i])) for i in sc if abs(float(ident[i]))>1e-4]
print(f"\nscale parameters the fit moved off zero: {len(moved)} of {len(sc)}")
for n,v in sorted(moved,key=lambda x:-abs(x[1]))[:8]: print(f"   {n:<24}{v:+.4f}")
