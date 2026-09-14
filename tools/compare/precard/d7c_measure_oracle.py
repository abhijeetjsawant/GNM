"""Pre-card D7c: on the D3 gate's six exact-skeleton bodies, how far is the delivered PELVIS
from the truth's, what does it cost above the leg roots, and does a template built from the
rig's OWN rest offsets (no constant) recover it exactly.

Three arms per seed, through the identical converter code path (the template is a module
global read at call time inside `_pelvis_world_frames`):
  shipped          -- `SOMA77_REST_PELVIS_TEMPLATE_M` as merged at D7 (SOMA-77's rest, a convention)
  candidate_hipmid -- {rest[Spine], rest[LeftUpperLeg], rest[RightUpperLeg]} minus the leg-root
                      midpoint, i.e. the rig's own rest offsets about the point the observed
                      `root` landmark actually is (`landmarks_from_fk` puts `root` on the hip
                      midpoint; the converter's own root formula puts the LEG ROOTS there too)
  wrong_origin     -- the same rig offsets taken from `Hips` instead of the leg-root midpoint,
                      pre-registered as WRONG on the oracle: the observed origin is the hip
                      midpoint, 80 mm below `Hips`, so this is a translation the rotation-only
                      fit must absorb as a tilt.
Every figure is on the ABSOLUTE frame (truth and delivery share one world); `rc.score`'s
leg-root-aligned groups are printed beside them with the gauge named.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/Users/abhi_macbook/Projects/apps/AutoAnim")
for rel in ("src", "tools/compare", "tools/head", "scripts", "tools/swap-harness"):
    sys.path.insert(0, str(ROOT / rel))
import d3_skeleton_gate as d3                       # noqa: E402
import d9b_hoist_gate as d9b                        # noqa: E402
import autoanim_gnm.commercial_multiview as cm      # noqa: E402
from autoanim_gnm.body import (                     # noqa: E402
    forward_kinematics_positions, DETAILED_HUMANOID, _quaternion_multiply)
rc = d3.rc
OUT = ROOT / "artifacts/compare/d7c-pelvis-rest/precard-oracle.json"


def s(v):
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}


def world_rotations(rotations, skeleton):
    out = np.zeros_like(rotations)
    for index, joint in enumerate(skeleton.joints):
        if joint.parent == -1:
            out[:, index] = rotations[:, index]
        else:
            out[:, index] = _quaternion_multiply(out[:, joint.parent], rotations[:, index])
    return out


def tilt(delivered_q, truth_q, hip_axis_world):
    """Relative rotation delivered * truth^-1 as an angle, and its component about the truth's
    hip line (pitch), the truth's forward (roll) and up (yaw)."""
    rd = Rotation.from_quat(delivered_q); rt = Rotation.from_quat(truth_q)
    rel = rd * rt.inv()
    vec = rel.as_rotvec()
    angle = np.degrees(np.linalg.norm(vec, axis=1))
    frames = rt.as_matrix()                      # columns: truth's X (hip line), Y (up), Z (forward)
    pitch = np.degrees(np.einsum("ni,ni->n", vec, frames[:, :, 0]))
    yaw = np.degrees(np.einsum("ni,ni->n", vec, frames[:, :, 1]))
    roll = np.degrees(np.einsum("ni,ni->n", vec, frames[:, :, 2]))
    return angle, pitch, yaw, roll


def hipline_primary_frames(points, spine, rest_arr, skeleton):
    """Candidate (b): the hip line is the EXACT primary axis (the rig's rest across, L - R),
    the pelvis up is the secondary (rest[Spine] - mid, orthogonalised against the hip line),
    targets the captured hip line and Spine1 - hip midpoint. No constant; `_frame_alignment`
    is the converter's own, so the hip line is held exactly and Spine1 sets only the pitch."""
    i = skeleton.index
    mid = 0.5 * (rest_arr[i("LeftUpperLeg")] + rest_arr[i("RightUpperLeg")])
    src_across = rest_arr[i("LeftUpperLeg")] - rest_arr[i("RightUpperLeg")]
    src_up = rest_arr[i("Spine")] - mid
    points = np.asarray(points, float); spine = np.asarray(spine, float)
    left = points[:, cm.JOINT_INDEX["left_hip"]]; right = points[:, cm.JOINT_INDEX["right_hip"]]
    hip_mid = 0.5 * (left + right)
    q = np.zeros((len(points), 4))
    for f in range(len(points)):
        q[f] = cm._frame_alignment(src_across, src_up, left[f] - right[f], spine[f] - hip_mid[f])
    for f in range(1, len(points)):
        if float(np.dot(q[f], q[f - 1])) < 0.0:
            q[f] *= -1.0
    return q, {"status": "solved", "mode": "hipline_primary_rig_rest", "resolved_fraction": 1.0}


def run(seed, mode, rest, skeleton, truth, landmarks, spine_truth, toes, truth_world):
    template_saved = cm.SOMA77_REST_PELVIS_TEMPLATE_M
    i = skeleton.index
    if mode == "candidate_hipmid":
        mid = 0.5 * (rest[i("LeftUpperLeg")] + rest[i("RightUpperLeg")])
        cm.SOMA77_REST_PELVIS_TEMPLATE_M = tuple(map(tuple, (
            rest[i("Spine")] - mid, rest[i("LeftUpperLeg")] - mid, rest[i("RightUpperLeg")] - mid)))
    elif mode == "wrong_origin":
        cm.SOMA77_REST_PELVIS_TEMPLATE_M = tuple(map(tuple, (
            rest[i("Spine")], rest[i("LeftUpperLeg")], rest[i("RightUpperLeg")])))
    pelvis_saved = cm._pelvis_world_frames
    if mode == "hipline_primary":
        cm._pelvis_world_frames = lambda points, spine, **kw: hipline_primary_frames(points, spine, rest, skeleton)
    captured, diag = [], []
    saved = cm.project_generated_foot_contacts

    def watcher(track, **kw):
        captured.append(track); p, d = saved(track, **kw); diag.append(d); return p, d
    cm.project_generated_foot_contacts = watcher
    try:
        track = cm.positions_to_body_track(
            rc.Z_UP_FROM_Y_UP(landmarks), sample_rate_hz=30, provenance_sha256="0" * 64,
            toe_world_z_up_m=rc.Z_UP_FROM_Y_UP(toes),
            spine_world_z_up_m=rc.Z_UP_FROM_Y_UP(spine_truth), skeleton=skeleton)
    finally:
        cm.project_generated_foot_contacts = saved
        cm.SOMA77_REST_PELVIS_TEMPLATE_M = template_saved
        cm._pelvis_world_frames = pelvis_saved
    pre = captured[-1]
    roots = np.asarray(track.root_translation_m, np.float64)
    rots = np.asarray(track.local_rotations_xyzw, np.float64)
    hoist = roots - np.asarray(pre.root_translation_m, np.float64)
    hmag = 1e3 * np.linalg.norm(hoist, axis=1); hoisted = hmag > 0.5
    fk = forward_kinematics_positions(roots, rots, skeleton=skeleton).astype(np.float64)
    fk_unh = fk - hoist[:, None, :]
    wr = world_rotations(rots, skeleton)
    hip_axis = Rotation.from_quat(truth_world[:, i("Hips")]).apply(np.array([1.0, 0, 0]))
    angle, pitch, yaw, roll = tilt(wr[:, i("Hips")], truth_world[:, i("Hips")], hip_axis)
    # the Kabsch residual of the fit the converter actually took, recomputed from its output
    template = np.asarray(cm.SOMA77_REST_PELVIS_TEMPLATE_M if mode == "shipped" else (
        (rest[i("Spine")] - 0.5 * (rest[i("LeftUpperLeg")] + rest[i("RightUpperLeg")]),
         rest[i("LeftUpperLeg")] - 0.5 * (rest[i("LeftUpperLeg")] + rest[i("RightUpperLeg")]),
         rest[i("RightUpperLeg")] - 0.5 * (rest[i("LeftUpperLeg")] + rest[i("RightUpperLeg")]))
        if mode in ("candidate_hipmid", "hipline_primary") else
        (rest[i("Spine")], rest[i("LeftUpperLeg")], rest[i("RightUpperLeg")])), float)
    root_lm = landmarks[:, cm.JOINT_INDEX["root"]]
    observed = np.stack([spine_truth - root_lm,
                         landmarks[:, cm.JOINT_INDEX["left_hip"]] - root_lm,
                         landmarks[:, cm.JOINT_INDEX["right_hip"]] - root_lm], axis=1)
    R = Rotation.from_quat(wr[:, i("Hips")]).as_matrix()          # [frame, 3, 3]
    rotated = np.einsum("nij,kj->nki", R, template)                # [frame, 3 points, 3]
    kabsch_resid = 1e3 * np.linalg.norm(rotated - observed, axis=2).mean(axis=1)
    absolute = d9b.absolute_score(fk, landmarks, skeleton)
    absolute_unh = d9b.absolute_score(fk_unh, landmarks, skeleton)
    aligned = rc.score(fk, landmarks, skeleton)

    def joint_miss(name, arr):
        return 1e3 * np.linalg.norm(arr[:, i(name)] - truth[:, i(name)], axis=1)
    return {
        "hoist_mm": s(hmag), "hoisted_frames": int(hoisted.sum()),
        "contacts": [int(c) for c in np.asarray(track.foot_contacts).sum(axis=0)],
        "pelvis_vs_truth_deg": {"angle": s(angle), "pitch_about_hip_line": s(pitch),
                                "pitch_signed_median": round(float(np.median(pitch)), 3),
                                "yaw": s(np.abs(yaw)), "roll": s(np.abs(roll))},
        "kabsch_residual_mm_mean_of_3": s(kabsch_resid),
        "spine_origin_miss_mm": {"raw": s(joint_miss("Spine", fk)),
                                 "hoist_subtracted": s(joint_miss("Spine", fk_unh)),
                                 "unhoisted_frames": s(joint_miss("Spine", fk)[~hoisted])},
        "hips_joint_miss_mm": {"hoist_subtracted": s(joint_miss("Hips", fk_unh))},
        "neck_miss_mm": {"hoist_subtracted": s(joint_miss("Neck", fk_unh))},
        "root_vs_truth_root_mm": {"raw": s(1e3 * np.linalg.norm(roots - (truth[:, i("Hips")] - rest[i("Hips")]), axis=1)),
                                  "hoist_subtracted": s(1e3 * np.linalg.norm(roots - hoist - (truth[:, i("Hips")] - rest[i("Hips")]), axis=1))},
        "ABSOLUTE_groups_mm": {"whole": d3.groups_mm(absolute),
                               "hoisted": d9b.groups_masked(absolute, hoisted) if hoisted.any() else None,
                               "unhoisted": d9b.groups_masked(absolute, ~hoisted),
                               "hoist_subtracted_whole": d3.groups_mm(absolute_unh),
                               "p95": d9b.groups_p95(absolute)},
        "ALIGNED_rc_score_groups_mm(leg-root-aligned gauge, the D3 gate's)": d3.groups_mm(aligned),
        "template_used": np.asarray(template).round(6).tolist(),
    }


def main():
    donor = d3.load_track(d3.DELIVERED, 0)
    rotations = np.asarray(donor.local_rotations_xyzw, np.float64)
    roots = np.asarray(donor.root_translation_m, np.float64)
    out = {"reference": d3.REF_SYNTH, "arms": {}, "seeds": {}}
    for seed in d3.SEEDS:
        rng = np.random.default_rng(seed); rest, factors = d3.perturbed_rest(rng)
        skeleton = DETAILED_HUMANOID.with_rest_translations(rest)
        truth = forward_kinematics_positions(roots, rotations, skeleton=skeleton).astype(np.float64)
        toe_low = float(min(truth[:, skeleton.index("LeftToes"), 1].min(),
                            truth[:, skeleton.index("RightToes"), 1].min()))
        truth = truth.copy(); truth[..., 1] -= toe_low
        truth_world = world_rotations(rotations, skeleton)
        landmarks = rc.landmarks_from_fk(truth, skeleton)
        spine_truth = truth[:, skeleton.index("Spine")]
        toes = np.stack([truth[:, skeleton.index("LeftToes")], truth[:, skeleton.index("RightToes")]], axis=1)
        i = skeleton.index
        rec = {"factors": factors,
               "rig_rest_mm": {n: (1e3 * rest[i(n)]).round(2).tolist() for n in ("Hips", "Spine", "LeftUpperLeg", "RightUpperLeg")},
               "root_landmark_is_hip_midpoint_mm": round(float(1e3 * np.abs(
                   landmarks[:, cm.JOINT_INDEX["root"]] - 0.5 * (landmarks[:, cm.JOINT_INDEX["left_hip"]] + landmarks[:, cm.JOINT_INDEX["right_hip"]])).max()), 6)}
        for mode in ("shipped", "candidate_hipmid", "wrong_origin", "hipline_primary"):
            rec[mode] = run(seed, mode, rest, skeleton, truth, landmarks, spine_truth, toes, truth_world)
            r = rec[mode]
            print(f"seed {seed} {mode:17s} tilt med/p95 {r['pelvis_vs_truth_deg']['angle']['median']:6.3f}/{r['pelvis_vs_truth_deg']['angle']['p95']:6.3f} deg"
                  f"  pitch {r['pelvis_vs_truth_deg']['pitch_signed_median']:+6.3f}  kabsch {r['kabsch_residual_mm_mean_of_3']['median']:7.4f} mm"
                  f"  Spine miss(h-sub) {r['spine_origin_miss_mm']['hoist_subtracted']['median']:6.2f}  Hips {r['hips_joint_miss_mm']['hoist_subtracted']['median']:6.2f}"
                  f"  ABS torso/arms/legs {r['ABSOLUTE_groups_mm']['hoist_subtracted_whole']}  ALIGNED {r['ALIGNED_rc_score_groups_mm(leg-root-aligned gauge, the D3 gate\'s)']}"
                  f"  hoist p95 {r['hoist_mm']['p95']} n={r['hoisted_frames']} contacts {r['contacts']}")
        out["seeds"][str(seed)] = rec
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
