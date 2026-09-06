"""Pre-card measurement: the foot-contact hoist and the per-bone miss it causes, from the
delivered D8c files' own bytes. Two independent recoveries of the hoist (arm-fit, converter line)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path("/Users/abhi_macbook/Projects/apps/AutoAnim")
for rel in ("src", "tools/compare", "tools/head", "scripts", "tools/swap-harness"):
    sys.path.insert(0, str(ROOT / rel))
import delivered_vs_capture as dvc
import d8c_hip_placement as hp
import autoanim_gnm.commercial_multiview as cm

DELIV = ROOT / "artifacts/commercial-multiview-soma77"
RAYS = [  # (parent joint whose rotation is aimed, child joint on the ray, landmark target, family)
    ("LeftUpperArm", "LeftLowerArm", "left_elbow", "arm"),
    ("LeftLowerArm", "LeftHand", "left_wrist", "arm"),
    ("RightUpperArm", "RightLowerArm", "right_elbow", "arm"),
    ("RightLowerArm", "RightHand", "right_wrist", "arm"),
    ("LeftShoulder", "LeftUpperArm", "left_shoulder", "clavicle"),
    ("RightShoulder", "RightUpperArm", "right_shoulder", "clavicle"),
    ("Spine", "Neck", "neck", "trunk"),
    ("LeftUpperLeg", "LeftLowerLeg", "left_knee", "leg"),
    ("LeftLowerLeg", "LeftFoot", "left_ankle", "leg"),
    ("RightUpperLeg", "RightLowerLeg", "right_knee", "leg"),
    ("RightLowerLeg", "RightFoot", "right_ankle", "leg"),
]
def s(v):
    v = np.asarray(v, float)
    if v.size == 0: return {"n": 0}
    return {"n": int(v.size), "median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}

out = {}
for subject in (0, 1):
    index, world = dvc.delivered_positions(DELIV, subject)
    cap = dvc.capture_positions(DELIV, subject)
    F = len(cap)
    # (a) hoist from the converter line (rig y-up) -> capture z-up
    b = hp.load(DELIV, subject)
    h_rig = b["hoist"][:F]
    h_line = np.stack([h_rig[:, 0], -h_rig[:, 2], h_rig[:, 1]], -1)
    # (b) hoist from the four arm bones (D9 gate method)
    h_arm = np.zeros((F, 3)); resid = np.zeros(F)
    for f in range(F):
        rows, t = [], []
        for parent, child, lm, fam in RAYS:
            if fam != "arm": continue
            o = world[f, index[parent]]; u = world[f, index[child]] - o; u /= np.linalg.norm(u)
            P = np.eye(3) - np.outer(u, u); rows.append(P); t.append(-P @ (cap[f, cm.JOINT_INDEX[lm]] - o))
        sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(t), rcond=None)
        h_arm[f] = sol; resid[f] = np.linalg.norm(np.vstack(rows) @ sol - np.concatenate(t))
    mag = 1e3 * np.linalg.norm(h_line, axis=1)
    agree = 1e3 * np.linalg.norm(h_line - h_arm, axis=1)
    hoisted = mag > 0.5
    zero = mag < 1e-3
    with np.load(DELIV / f"subject-{subject:02d}.body-track.npz") as a:
        contacts = np.asarray(a["foot_contacts"])
    runs = []
    for side in (0, 1):
        on = np.flatnonzero(contacts[:, side])
        if on.size:
            splits = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
            runs.append([(int(r[0]), int(r[-1])) for r in splits])
        else: runs.append([])
    feet_y = np.min(world[:, [index[n] for n in ("LeftFoot", "LeftToes", "RightFoot", "RightToes")], 2], axis=1)
    rec = {"frames": F, "hoist_mm": s(mag), "frames_over_0_5mm": int(hoisted.sum()), "frames_exactly_zero(<1e-3mm)": int(zero.sum()),
           "hoisted_frames": [int(i) for i in np.flatnonzero(hoisted)],
           "arm_fit_vs_converter_line_disagreement_mm": s(agree), "arm_fit_residual_mm": s(1e3 * resid),
           "hoist_component_mm(x,y,z capture)": {"median_abs": [round(float(v), 3) for v in 1e3 * np.median(np.abs(h_line[hoisted]), axis=0)] if hoisted.any() else None},
           "whole_take_constant_lift_mm": round(float(1e3 * np.min(mag)), 4),
           "contacts": {"count": [int(v) for v in contacts.sum(0)], "runs": runs},
           "delivered_lowest_foot_or_toe_y_mm": {"min": round(float(1e3 * feet_y.min()), 2), "on_hoisted": round(float(1e3 * feet_y[hoisted].min()), 2) if hoisted.any() else None},
           "rays": {}}
    for parent, child, lm, fam in RAYS:
        o = world[:, index[parent]]; c = world[:, index[child]]; u = c - o; L = np.linalg.norm(u, axis=1); u /= L[:, None]
        tgt = cap[:, cm.JOINT_INDEX[lm]]
        def miss(origin):
            d = tgt - origin; return 1e3 * np.linalg.norm(d - np.sum(d * u, 1)[:, None] * u, axis=1)
        m_after = miss(o); m_before = miss(o - h_line)
        # the re-aim: same origin, the bone turned onto the ray to the target
        u2 = tgt - o; u2 /= np.linalg.norm(u2, axis=1)[:, None]
        move = 1e3 * L * np.linalg.norm(u2 - u, axis=1)
        rec["rays"][f"{parent}->{lm}"] = {"family": fam,
            "miss_from_delivered_origin_mm": {"whole": s(m_after), "hoisted": s(m_after[hoisted]), "unhoisted": s(m_after[~hoisted])},
            "miss_from_pre_hoist_origin_mm": {"whole": s(m_before), "hoisted": s(m_before[hoisted])},
            "child_move_if_re_aimed_mm": {"hoisted": s(move[hoisted]), "unhoisted": s(move[~hoisted])}}
    out[f"subject_{subject:02d}"] = rec
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, "hoist", v["hoist_mm"], "over0.5", v["frames_over_0_5mm"], "zero", v["frames_exactly_zero(<1e-3mm)"], "const", v["whole_take_constant_lift_mm"])
    print("  agree", v["arm_fit_vs_converter_line_disagreement_mm"], "resid", v["arm_fit_residual_mm"], "contacts", v["contacts"]["count"], "runs", v["contacts"]["runs"], "feet_y", v["delivered_lowest_foot_or_toe_y_mm"], "comp", v["hoist_component_mm(x,y,z capture)"])
    for r, d in v["rays"].items():
        print(f"  {r:32s} after w/h/u {d['miss_from_delivered_origin_mm']['whole']['median']}/{d['miss_from_delivered_origin_mm']['hoisted'].get('median')}/{d['miss_from_delivered_origin_mm']['unhoisted']['median']}  p95h {d['miss_from_delivered_origin_mm']['hoisted'].get('p95')} | before w/h {d['miss_from_pre_hoist_origin_mm']['whole']['median']}/{d['miss_from_pre_hoist_origin_mm']['hoisted'].get('median')} max_h {d['miss_from_pre_hoist_origin_mm']['hoisted'].get('max')} | move h med/p95/max {d['child_move_if_re_aimed_mm']['hoisted'].get('median')}/{d['child_move_if_re_aimed_mm']['hoisted'].get('p95')}/{d['child_move_if_re_aimed_mm']['hoisted'].get('max')} unh max {d['child_move_if_re_aimed_mm']['unhoisted'].get('max')}")
