"""Pre-card: do contacts fire on the D3 gate's exact-skeleton oracle, how big is its hoist, and how
much of the oracle's arm/torso error is the hoist. Also the clavicle reject's accepted set."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path("/Users/abhi_macbook/Projects/apps/AutoAnim")
for rel in ("src", "tools/compare", "tools/head", "scripts", "tools/swap-harness"):
    sys.path.insert(0, str(ROOT / rel))
import d3_skeleton_gate as d3
import autoanim_gnm.commercial_multiview as cm
from autoanim_gnm.body import forward_kinematics_positions, DETAILED_HUMANOID
rc = d3.rc
RAYS = [("LeftUpperArm","LeftLowerArm","left_elbow","arm"),("LeftLowerArm","LeftHand","left_wrist","arm"),
        ("RightUpperArm","RightLowerArm","right_elbow","arm"),("RightLowerArm","RightHand","right_wrist","arm"),
        ("LeftShoulder","LeftUpperArm","left_shoulder","clav"),("RightShoulder","RightUpperArm","right_shoulder","clav"),
        ("Spine","Neck","neck","trunk"),("LeftUpperLeg","LeftLowerLeg","left_knee","leg"),("LeftLowerLeg","LeftFoot","left_ankle","leg")]
def s(v):
    v=np.asarray(v,float)
    return {"n":int(v.size),"median":round(float(np.median(v)),3),"p95":round(float(np.percentile(v,95)),3),"max":round(float(v.max()),3)} if v.size else {"n":0}
donor = d3.load_track(d3.DELIVERED, 0)
rotations = np.asarray(donor.local_rotations_xyzw, np.float64); roots = np.asarray(donor.root_translation_m, np.float64)
out = {}
for seed in d3.SEEDS:
    rng = np.random.default_rng(seed); rest, factors = d3.perturbed_rest(rng)
    skeleton = DETAILED_HUMANOID.with_rest_translations(rest)
    truth = forward_kinematics_positions(roots, rotations, skeleton=skeleton).astype(np.float64)
    toe_low = float(min(truth[:, skeleton.index("LeftToes"), 1].min(), truth[:, skeleton.index("RightToes"), 1].min()))
    truth = truth.copy(); truth[..., 1] -= toe_low
    landmarks = rc.landmarks_from_fk(truth, skeleton)
    spine_truth = truth[:, skeleton.index("Spine")]
    toes = np.stack([truth[:, skeleton.index("LeftToes")], truth[:, skeleton.index("RightToes")]], axis=1)
    captured = []; diag = []
    saved = cm.project_generated_foot_contacts
    def watcher(track, **kw):
        captured.append(track); p, d = saved(track, **kw); diag.append(d); return p, d
    saved_reach = cm._reachable_clavicle_sequence; accepted_log = []
    def reach(local, parent, ceiling):
        r, a = saved_reach(local, parent, ceiling); accepted_log.append(int((~a).sum())); return r, a
    cm.project_generated_foot_contacts = watcher; cm._reachable_clavicle_sequence = reach
    try:
        track = cm.positions_to_body_track(rc.Z_UP_FROM_Y_UP(landmarks), sample_rate_hz=30, provenance_sha256="0"*64,
            toe_world_z_up_m=rc.Z_UP_FROM_Y_UP(toes), spine_world_z_up_m=rc.Z_UP_FROM_Y_UP(spine_truth), skeleton=skeleton)
    finally:
        cm.project_generated_foot_contacts = saved; cm._reachable_clavicle_sequence = saved_reach
    pre = captured[-1]; d = diag[-1]
    hoist = np.asarray(track.root_translation_m, np.float64) - np.asarray(pre.root_translation_m, np.float64)
    mag = 1e3*np.linalg.norm(hoist, axis=1); hoisted = mag > 0.5
    fk_after = forward_kinematics_positions(np.asarray(track.root_translation_m,np.float64), np.asarray(track.local_rotations_xyzw,np.float64), skeleton=skeleton)
    fk_pre = forward_kinematics_positions(np.asarray(pre.root_translation_m,np.float64), np.asarray(pre.local_rotations_xyzw,np.float64), skeleton=skeleton)
    g_after = d3.groups_mm(rc.score(fk_after, landmarks, skeleton)); g_pre = d3.groups_mm(rc.score(fk_pre, landmarks, skeleton))
    # the oracle's arms/torso with the hoist subtracted from every joint (what a re-aim from the hoisted origin would leave, to first order: exactly, since targets are exact)
    fk_unh = fk_after - hoist[:, None, :]
    g_unh = d3.groups_mm(rc.score(fk_unh, landmarks, skeleton))
    rays = {}
    for parent, child, lm, fam in RAYS:
        o = fk_after[:, skeleton.index(parent)]; c = fk_after[:, skeleton.index(child)]; u = c-o; u /= np.linalg.norm(u,axis=1)[:,None]
        t = landmarks[:, cm.JOINT_INDEX[lm]]
        def miss(org):
            dd = t-org; return 1e3*np.linalg.norm(dd-np.sum(dd*u,1)[:,None]*u,axis=1)
        rays[f"{parent}->{lm}"] = {"after_hoisted": s(miss(o)[hoisted]), "after_unhoisted": s(miss(o)[~hoisted]), "pre_hoist_origin_hoisted": s(miss(o-hoist)[hoisted])}
    out[str(seed)] = {"contacts": d.contact_frames, "max_correction_mm": round(1e3*d.maximum_root_correction_m,3), "penetration_before_mm": round(1e3*d.ground_penetration_before_m,4),
        "penetration_after_mm": round(1e3*d.ground_penetration_after_m,4), "hoist_mm": s(mag), "frames_over_0_5": int(hoisted.sum()), "frames_zero": int((mag<1e-3).sum()),
        "groups_delivered": g_after, "groups_pre_projection": g_pre, "groups_hoist_removed_everywhere": g_unh, "clavicle_rejected_frames(L,R)": accepted_log, "rays": rays}
    print(seed, "contacts", d.contact_frames, "hoist", s(mag), "over0.5", int(hoisted.sum()), "pen", out[str(seed)]["penetration_before_mm"], "groups after", g_after, "pre", g_pre, "unh", g_unh, "clav rej", accepted_log)
    for k, v in rays.items(): print("   ", k, "after h/u", v["after_hoisted"].get("median"), v["after_unhoisted"].get("median"), "pre-origin h", v["pre_hoist_origin_hoisted"].get("median"), v["pre_hoist_origin_hoisted"].get("max"))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
