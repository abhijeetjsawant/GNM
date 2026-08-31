import pymomentum.geometry as g, pymomentum.marker_tracking as mt, numpy as np, json, sys
S='/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/694467ef-7cde-45ab-b64a-1eed22e17800/scratchpad'
A=S+'/mhr/assets'
c = g.Character.load_fbx(A+'/lod6.fbx').load_model_definition(A+'/compact_v6_1.model')
jn = list(c.skeleton.joint_names); pt = c.parameter_transform

# our 19 landmarks -> the MHR joint each one sits on. Ears omitted: SOMA-77 emits none.
MAP = {'root':'root','neck':'c_neck','nose':'c_head',
 'left_shoulder':'l_uparm','right_shoulder':'r_uparm','left_elbow':'l_lowarm','right_elbow':'r_lowarm',
 'left_wrist':'l_wrist','right_wrist':'r_wrist','left_hip':'l_upleg','right_hip':'r_upleg',
 'left_knee':'l_lowleg','right_knee':'r_lowleg','left_ankle':'l_foot','right_ankle':'r_foot',
 'left_eye':'l_eye','right_eye':'r_eye'}

data = json.load(open(S+'/markers_mhr_cm.json'))
names = data['names']
locs = [g.Locator(name=lm, parent=jn.index(j), offset=np.zeros(3, dtype=np.float32))
        for lm, j in MAP.items()]
c = c.with_locators(locs)
print(f'locators attached: {len(c.locators)}  mesh retained: {c.has_mesh}')

SUBJ = int(sys.argv[1]) if len(sys.argv) > 1 else 0
arr = np.asarray(data['subjects'][SUBJ], dtype=np.float64)
marker_data = []
for f in range(arr.shape[0]):
    row = []
    for lm in MAP:
        p = arr[f, names.index(lm)]
        ok = np.isfinite(p).all()
        row.append(g.Marker(name=lm, pos=(p if ok else np.zeros(3)).astype(np.float64), occluded=not ok))
    marker_data.append(row)
print(f'marker frames: {len(marker_data)}, markers/frame: {len(marker_data[0])}')

cfg = mt.CalibrationConfig()
cfg.calib_frames = 100
cfg.global_scale_only = False     # per-bone scale is the whole point
cfg.locators_only = False
cfg.loss_alpha = 2.0              # robust loss -- degenerate solution 5
cfg.max_iter = 30
cfg.debug = False
identity = np.zeros(len(pt.names), dtype=np.float32)
print('calibrating...')
ident_out, param_idx, motion = mt.calibrate_markers(c, identity, marker_data, cfg)
print('DONE. identity out:', np.asarray(ident_out).shape, ' motion:', np.asarray(motion).shape)
np.savez(S+f'/fit_subject_{SUBJ}.npz', identity=np.asarray(ident_out),
         param_idx=np.asarray(param_idx), motion=np.asarray(motion))
