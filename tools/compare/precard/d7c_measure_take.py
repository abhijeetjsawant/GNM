"""Pre-card D7c, the real take, from the rebuilt deliveries' own bytes and the converter inputs
`d7c_take_build.py` dumped: what the rig-rest pelvis fit changes on the two performers.

Every figure is shipped (D9b, today's SOMA template) vs candidate (the rig's own rest offsets
about the hip midpoint), on identical landmarks (checked byte for byte).
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/Users/abhi_macbook/Projects/apps/AutoAnim")
for rel in ("src", "tools/compare", "tools/swap-harness"):
    sys.path.insert(0, str(ROOT / rel))
import autoanim_gnm.commercial_multiview as cm      # noqa: E402
from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_track_dict  # noqa: E402
import retarget_cost as rc                          # noqa: E402
import d3_skeleton_gate as d3                       # noqa: E402

BASE = ROOT / "artifacts/compare/d7c-pelvis-rest"
SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
SHIP = BASE / "precard-take-shipped"
CAND = BASE / (sys.argv[1] if len(sys.argv) > 1 else "precard-take-candidate")
D8C = json.loads((ROOT / "artifacts/compare/d8c-hip/gate.json").read_text())
OUT = BASE / ("precard-take.json" if len(sys.argv) < 2 else f"precard-take-{sys.argv[1].removeprefix('precard-take-')}.json")


def s(v):
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": round(float(np.median(v)), 3),
            "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(d, subject):
    z = np.load(d / f"subject-{subject:02d}.body-track.npz")
    meta = json.loads((d / f"subject-{subject:02d}.body-track.json").read_text())
    skel = skeleton_for_track_dict(meta)
    return {k: z[k] for k in z.files}, skel


def world_hips(track):
    # Root is identity in the converter, so Hips' local IS its world rotation.
    return np.asarray(track["local_rotations_xyzw"][:, 1], np.float64)


def hoist_by_converter_line(track, skel, points_y_up):
    i = skel.index; rest = np.asarray(skel.rest_translations_m, np.float64)
    pelvis = 0.5 * (points_y_up[:, cm.JOINT_INDEX["left_hip"]] + points_y_up[:, cm.JOINT_INDEX["right_hip"]])
    mid = 0.5 * (rest[i("LeftUpperLeg")] + rest[i("RightUpperLeg")])
    R = Rotation.from_quat(world_hips(track))
    pre = pelvis - rest[i("Hips")] - R.apply(np.broadcast_to(mid, (len(pelvis), 3)))
    return np.asarray(track["root_translation_m"], np.float64) - pre


def y_up(z_up):
    p = np.asarray(z_up, np.float64)[..., (0, 2, 1)].copy(); p[..., 2] *= -1.0; return p


def main():
    out = {"hygiene": {}, "subjects": {}}
    for f in ("subject-00.glb", "subject-01.glb", "subject-00.body-track.npz", "subject-01.body-track.npz",
              "subject-00.mapping.npz", "subject-01.mapping.npz", "subject-00.body-track.json", "subject-01.body-track.json"):
        out["hygiene"][f] = {"shipped_mode_rebuild_byte_identical_to_shipped": sha(SHIP / f) == sha(SHIPPED / f),
                             "candidate_differs": sha(CAND / f) != sha(SHIPPED / f)}
    print("hygiene", {k: v["shipped_mode_rebuild_byte_identical_to_shipped"] for k, v in out["hygiene"].items()})
    calls = sorted((CAND / "converter-inputs").glob("call-*.npz"))
    calls_ship = sorted((SHIP / "converter-inputs").glob("call-*.npz"))
    for subject in (0, 1):
        ship, skel = load(SHIP, subject); cand, skel_c = load(CAND, subject)
        assert np.array_equal(np.asarray(skel.rest_translations_m), np.asarray(skel_c.rest_translations_m)), "rest moved"
        # the converter is called once per subject in order; verify the dumped positions ARE this subject's
        inp = np.load(calls[subject]); inp_s = np.load(calls_ship[subject])
        assert np.array_equal(inp["positions_world_z_up_m"], inp_s["positions_world_z_up_m"]), "inputs differ between arms"
        assert np.array_equal(inp["positions_world_z_up_m"], ship["triangulated_world_positions_z_up_m"]), "dump is not this subject"
        points = y_up(inp["positions_world_z_up_m"]); spine = y_up(inp["spine_world_z_up_m"])
        i = skel.index; rest = np.asarray(skel.rest_translations_m, np.float64)
        names = list(skel.names)
        left = points[:, cm.JOINT_INDEX["left_hip"]]; right = points[:, cm.JOINT_INDEX["right_hip"]]
        root_lm = points[:, cm.JOINT_INDEX["root"]]; hipmid = 0.5 * (left + right)
        valid = np.isfinite(spine).all(axis=1)
        # honest mask: D8c's hip-line fires removed (smoothed array, this subject)
        off = set()
        for arr in ("smoothed", "raw"):
            node = D8C["B4_the_frames_off_on_a_fixed_denominator"]["every_segment_both_arrays"][arr].get(f"subject_{subject:02d}", {}).get("hip_line", {})
            off |= set(int(x) for x in node.get("off_before_ids", []))
        honest = np.ones(len(points), bool); honest[list(off)] = False
        honest &= valid
        Rs = Rotation.from_quat(world_hips(ship)); Rc = Rotation.from_quat(world_hips(cand))
        rel = Rc * Rs.inv(); vec = rel.as_rotvec()
        angle = np.degrees(np.linalg.norm(vec, axis=1))
        hip_axis_s = Rs.as_matrix()[:, :, 0]
        pitch = np.degrees(np.einsum("ni,ni->n", vec, hip_axis_s))
        # the delivered pelvis +Y against the captured pelvis landmark directions
        up_s = Rs.as_matrix()[:, :, 1]; up_c = Rc.as_matrix()[:, :, 1]
        def ang(a, b):
            a = a / np.linalg.norm(a, axis=1)[:, None]; b = b / np.linalg.norm(b, axis=1)[:, None]
            return np.degrees(np.arccos(np.clip(np.sum(a * b, 1), -1, 1)))
        d_mid = spine - hipmid; d_root = spine - root_lm
        # Kabsch residual of each fit, per point, on the honest mask
        mid = 0.5 * (rest[i("LeftUpperLeg")] + rest[i("RightUpperLeg")])
        tmpl_c = np.stack((rest[i("Spine")] - mid, rest[i("LeftUpperLeg")] - mid, rest[i("RightUpperLeg")] - mid))
        obs_c = np.stack((spine - hipmid, left - hipmid, right - hipmid), axis=1)
        res_c = 1e3 * np.linalg.norm(obs_c - np.einsum("nij,kj->nki", Rc.as_matrix(), tmpl_c), axis=2)
        tmpl_s = np.asarray(cm.SOMA77_REST_PELVIS_TEMPLATE_M)
        obs_s = np.stack((spine - root_lm, left - root_lm, right - root_lm), axis=1)
        res_s = 1e3 * np.linalg.norm(obs_s - np.einsum("nij,kj->nki", Rs.as_matrix(), tmpl_s), axis=2)
        # the captured hips in the candidate pelvis frame vs the rig's own leg-root offsets
        hips_in_frame = np.stack((Rc.inv().apply(left - hipmid), Rc.inv().apply(right - hipmid)), axis=1)
        rig_half = np.linalg.norm(rest[i("LeftUpperLeg")] - mid)
        half_len = np.linalg.norm(hips_in_frame, axis=2)
        # FK and the moves
        hs = hoist_by_converter_line(ship, skel, points); hc = hoist_by_converter_line(cand, skel, points)
        fk_s = forward_kinematics_positions(ship["root_translation_m"], ship["local_rotations_xyzw"], skeleton=skel).astype(np.float64)
        fk_c = forward_kinematics_positions(cand["root_translation_m"], cand["local_rotations_xyzw"], skeleton=skel).astype(np.float64)
        move = 1e3 * np.linalg.norm(fk_c - fk_s, axis=2)                       # [frame, joint]
        move_unh = 1e3 * np.linalg.norm((fk_c - hc[:, None]) - (fk_s - hs[:, None]), axis=2)
        root_move = 1e3 * np.linalg.norm(cand["root_translation_m"] - ship["root_translation_m"], axis=1)
        root_move_unh = 1e3 * np.linalg.norm((cand["root_translation_m"] - hc) - (ship["root_translation_m"] - hs), axis=1)
        # fore/aft component of the root move in the shipped pelvis frame (Z is forward)
        root_fa = 1e3 * np.einsum("ni,ni->n", (cand["root_translation_m"] - hc) - (ship["root_translation_m"] - hs), Rs.as_matrix()[:, :, 2])
        ali_s = d3.groups_mm(rc.score(fk_s, points, skel)); ali_c = d3.groups_mm(rc.score(fk_c, points, skel))
        groups = {g: s(np.concatenate([move[:, i(rc.RIG_FOR[n])] for n in ns])) for g, ns in rc.GROUPS.items()}
        rec = {
            "frames": int(len(points)), "spine_resolved": int(valid.sum()), "honest_frames": int(honest.sum()),
            "hip_line_off_ids_removed": sorted(off),
            "rig_rest_mm": {n: (1e3 * rest[i(n)]).round(2).tolist() for n in ("Hips", "Spine", "LeftUpperLeg", "RightUpperLeg")},
            "pelvis_change_candidate_vs_shipped_deg": {"angle": s(angle), "pitch_about_hip_line_signed_median": round(float(np.median(pitch)), 3),
                                                       "pitch_abs": s(np.abs(pitch)), "angle_honest": s(angle[honest])},
            "delivered_up_vs_captured_directions_deg": {
                "shipped_up_vs_spine1_minus_hipmid": s(ang(up_s[valid], d_mid[valid])),
                "shipped_up_vs_spine1_minus_root": s(ang(up_s[valid], d_root[valid])),
                "candidate_up_vs_spine1_minus_hipmid": s(ang(up_c[valid], d_mid[valid])),
                "candidate_up_vs_spine1_minus_root": s(ang(up_c[valid], d_root[valid])),
                "spine1_minus_hipmid_length_mm": s(1e3 * np.linalg.norm(d_mid[valid], axis=1)),
                "spine1_minus_root_length_mm": s(1e3 * np.linalg.norm(d_root[valid], axis=1)),
                "root_landmark_above_hipmid_mm": s(1e3 * np.linalg.norm((root_lm - hipmid)[valid], axis=1))},
            "kabsch_residual_mm_honest": {
                "shipped_SOMA_template_about_root": {"spine1": s(res_s[honest, 0]), "left_hip": s(res_s[honest, 1]), "right_hip": s(res_s[honest, 2])},
                "candidate_rig_rest_about_hipmid": {"spine1": s(res_c[honest, 0]), "left_hip": s(res_c[honest, 1]), "right_hip": s(res_c[honest, 2])}},
            "captured_hip_half_span_in_candidate_pelvis_frame_mm_honest": {
                "rig_rest_half_span_mm": round(float(1e3 * rig_half), 2),
                "left": s(1e3 * half_len[honest, 0]), "right": s(1e3 * half_len[honest, 1]),
                "left_p5_p95_pct_of_rig": [round(float(100 * (np.percentile(half_len[honest, 0], q) / rig_half - 1)), 2) for q in (5, 95)],
                "right_p5_p95_pct_of_rig": [round(float(100 * (np.percentile(half_len[honest, 1], q) / rig_half - 1)), 2) for q in (5, 95)],
                "left_p5_p95_pct_of_own_median": [round(float(100 * (np.percentile(half_len[honest, 0], q) / np.median(half_len[honest, 0]) - 1)), 2) for q in (5, 95)],
                "right_p5_p95_pct_of_own_median": [round(float(100 * (np.percentile(half_len[honest, 1], q) / np.median(half_len[honest, 1]) - 1)), 2) for q in (5, 95)]},
            "hoist_mm": {"shipped": s(1e3 * np.linalg.norm(hs, axis=1)), "candidate": s(1e3 * np.linalg.norm(hc, axis=1)),
                         "frames_over_0_5": [int((1e3 * np.linalg.norm(hs, axis=1) > 0.5).sum()), int((1e3 * np.linalg.norm(hc, axis=1) > 0.5).sum())],
                         "hoist_change": s(1e3 * np.linalg.norm(hc - hs, axis=1))},
            "contacts": {"shipped": [int(c) for c in ship["foot_contacts"].sum(0)], "candidate": [int(c) for c in cand["foot_contacts"].sum(0)],
                         "identical": bool(np.array_equal(ship["foot_contacts"], cand["foot_contacts"]))},
            "root_move_mm": {"raw": s(root_move), "hoist_subtracted": s(root_move_unh),
                             "fore_aft_signed_median_in_shipped_pelvis_frame": round(float(np.median(root_fa)), 3)},
            "joint_move_mm_hoist_subtracted": {n: s(move_unh[:, i(n)]) for n in ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftShoulder", "LeftUpperArm", "LeftHand", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes")},
            "landmark_groups_move_mm_raw": groups,
            "leg_roots_stay_on_captured_hips_mm": s(1e3 * np.linalg.norm(0.5 * (fk_c[:, i("LeftUpperLeg")] + fk_c[:, i("RightUpperLeg")]) - hc - hipmid, axis=1)),
            "ALIGNED_rc_score_groups_mm(the D3 gate's gauge, our capture)": {"shipped": ali_s, "candidate": ali_c},
            "rotations_bit_identical_joints": [n for k, n in enumerate(names) if np.array_equal(ship["local_rotations_xyzw"][:, k], cand["local_rotations_xyzw"][:, k])],
        }
        out["subjects"][f"subject_{subject:02d}"] = rec
        print(f"\nsubject {subject}: pelvis change {rec['pelvis_change_candidate_vs_shipped_deg']}")
        print(" up vs captured:", json.dumps(rec["delivered_up_vs_captured_directions_deg"]))
        print(" kabsch:", json.dumps(rec["kabsch_residual_mm_honest"]))
        print(" half span:", json.dumps(rec["captured_hip_half_span_in_candidate_pelvis_frame_mm_honest"]))
        print(" hoist:", json.dumps(rec["hoist_mm"]), "contacts", rec["contacts"])
        print(" root:", json.dumps(rec["root_move_mm"]))
        print(" joints:", json.dumps(rec["joint_move_mm_hoist_subtracted"]))
        print(" groups raw:", json.dumps(rec["landmark_groups_move_mm_raw"]))
        print(" leg roots on hips:", rec["leg_roots_stay_on_captured_hips_mm"], " aligned:", rec["ALIGNED_rc_score_groups_mm(the D3 gate's gauge, our capture)"])
        print(" bit-identical rotations:", rec["rotations_bit_identical_joints"])
    OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT)


if __name__ == "__main__":
    main()
