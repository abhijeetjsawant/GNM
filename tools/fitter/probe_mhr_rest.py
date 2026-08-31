import pymomentum.geometry as g, numpy as np
A='/private/tmp/claude-501/-Users-abhi-macbook-Projects-apps-AutoAnim/694467ef-7cde-45ab-b64a-1eed22e17800/scratchpad/mhr/assets'
c = g.Character.load_fbx(A+'/lod6.fbx').load_model_definition(A+'/compact_v6_1.model')
pt = c.parameter_transform
zero = np.zeros(len(pt.names), dtype=np.float32)
st = np.asarray(g.model_parameters_to_skeleton_state(c, zero))
print("skeleton_state shape:", st.shape)
pos = st[..., :3].reshape(-1, 3)
ext = pos.max(0) - pos.min(0)
print("REST extent per axis:", np.round(ext, 2), "->", "CENTIMETRES" if ext.max() > 50 else "METRES")
jn = list(c.skeleton.joint_names)
def P(n): return pos[jn.index(n)]
print("\nMHR rest segments, on our landmark definition (same units):")
for lbl, (a, b) in {"shoulder width": ("l_uparm","r_uparm"), "hip width": ("l_upleg","r_upleg"),
                    "upper arm": ("l_uparm","l_lowarm"), "forearm": ("l_lowarm","l_wrist"),
                    "thigh": ("l_upleg","l_lowleg"), "shin": ("l_lowleg","l_foot"),
                    "pelvis->neck": ("root","c_neck")}.items():
    print(f"  {lbl:<14} {np.linalg.norm(P(a)-P(b)):7.1f}")
