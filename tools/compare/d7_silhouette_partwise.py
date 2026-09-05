#!/usr/bin/env python3
"""D7's B7: the photographs, part-wise and by trunk tilt. The band the candidate cannot optimise.

INSTRUMENT ONLY. Nothing ships. `silhouette.py`'s and `silhouette_partwise.py`'s own
outputs are untouched; their rasteriser, scorer, mask store, part split, block draws and
paired bootstrap are IMPORTED and reused, so both builds and the oracle go through one
pixel path and one statistic. **Nothing is written under
`artifacts/commercial-multiview-soma77/` or `artifacts/compare/i6/`**: the D3 mesh cache is
COPIED into this step's own work directory (verified byte-identical to
`artifacts/compare/d3-skeleton/silhouette-work/delivered-mesh.npz`, and the committed
delivery's GLBs are byte-identical to the D3 rebuild's), so `delivered_mesh()` can never
re-export into a directory a committed instrument reads.

WHY A WRAPPER AND NOT AN EDIT. `silhouette_partwise.py` and `silhouette_vs_tilt.py` carry
D2's three builds, D2's folded-arm control and D2's exact root/clavicle decomposition in
their `BUILDS` and their `main`, and their committed reports are the record of that pass.
Parameterising them would rewrite a trusted instrument to serve a different question. This
file takes the two builds as module constants and reuses every function that does the
measuring.

THE PRE-REGISTERED CLAUSES, verbatim from `docs/reviews/pelvis-frame-2026-09-04.md`
section 0.5, band B7, written before the rebuild existed:

    torso+legs RISES on the bent tercile, interval clear of zero;
    arms WITHIN their CI of D3 (the thorax frame is untouched);
    the upright tercile WITHIN its CI;
    MAMMA's mesh oracle BIT-IDENTICAL between runs.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_silhouette_partwise.py

Writes `artifacts/compare/d7-pelvis-frame/silhouette-partwise.json` and caches every
per-frame score in `silhouette-partwise-per-frame.npz`, so a re-cut is free.

BLIND TO: depth; a left/right mirror of a fore-aft symmetric pose; and, within each part,
where inside the outline a limb sits. Every figure is read against a mesh bound to an asset
whose proportions are not the performer's -- the oracle reaches 0.71-0.88 where we reach
0.52-0.66 -- so this scores a CHANGE against a large standing shape mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "head"))
sys.path.insert(0, str(ROOT / "tools" / "compare"))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import silhouette as sil  # noqa: E402
import silhouette_partwise as pw  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
REPORT = OUT_DIR / "silhouette-partwise.json"
PER_FRAME = OUT_DIR / "silhouette-partwise-per-frame.npz"

# label -> (delivered directory, the work directory holding ITS posed-mesh cache).
# BEFORE is the COMMITTED delivery, which is D3's: its GLBs are byte-identical to
# `artifacts/compare/d3-skeleton/delivery`'s, asserted below.
BUILDS = (
    ("D3", ROOT / "artifacts/commercial-multiview-soma77", OUT_DIR / "silhouette-work-d3"),
    ("D7", OUT_DIR / "delivery", OUT_DIR / "silhouette-work"),
    # Option (c). A CONTROL that never ships: the pelvis frozen bolt upright, built by
    # `d7_world_vertical_delivery.py` through the converter's identical call site.
    ("WORLD_VERTICAL", OUT_DIR / "delivery-world-vertical",
     OUT_DIR / "silhouette-work-world-vertical"),
)
MASK_WORK = OUT_DIR / "silhouette-work"          # holds the cached SAM2 masks; read only
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260904

PREREGISTERED = {
    "source": "docs/reviews/pelvis-frame-2026-09-04.md section 0.5, band B7, committed "
              "before the delivery was rebuilt",
    "torso_legs_bent_tercile": "RISES against D3, interval clear of zero. The root no "
                               "longer carries 80*sin(tilt) of trunk-axis offset on bent "
                               "frames; D2b isolated the root's own share at -0.022 / "
                               "-0.013 IoU whole-person, tilt-dependent.",
    "arms": "WITHIN their block-bootstrap CI of D3 -- the thorax frame is untouched and "
            "the clavicle chain hangs off it, not off Hips.",
    "upright_tercile": "WITHIN its CI of D3.",
    "mamma_mesh_oracle": "BIT-IDENTICAL between runs -- it reads none of our track, and "
                         "that identity is the proof that reusing the caches changed "
                         "nothing.",
    "the_competing_prediction_named_before_the_rebuild": (
        "the shipped rest-pitch is a convention with a measured residual (1.75-4.42 deg "
        "median, up to 10.03). A residual pitch delta moves the root by |mid|*sin(delta) "
        "~= 1.4-6.9 mm on EVERY frame, upright ones included, where D2b measured a 10 mm "
        "root shift costing 0.045 and 0.093 IoU. SO THE UPRIGHT TERCILE MAY FALL, and if "
        "it does the mechanism is the convention constant and not the pelvis frame."),
}

KEYS = ("torso_iou", "torso_precision", "torso_recall",
        "arm_iou", "arm_precision", "arm_recall",
        "whole_iou", "whole_precision", "whole_recall")


def per_frame_mean(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Mean over the four cameras of a frame's score, on frames every camera scored.

    ONE DENOMINATOR for every arm: trunk tilt is a per-frame property, so a tercile has to
    be a set of frames, not a set of camera-frames.
    """
    out = np.full(values.shape[-1], np.nan)
    out[keep] = values[:, keep].mean(axis=0)
    return out


def control_figures() -> dict:
    """The control delivery's own hoist and `Hips`-joint offset, beside section 7b's offline arm.

    Two routes, deliberately, because they answer different questions.

      * THE HOIST is `projected_root - unprojected_root`, and the unprojected root is not on
        disk -- so it is recomputed with the watcher pattern on each build's own capture
        positions, WITHOUT the head and toe solves, exactly as review section 7b did. It
        therefore reproduces 7b by construction, and that is the point: it is the check that
        7b's offline substitution and this rendered control are the same converter.
      * THE `Hips`-JOINT OFFSET and the lowest joint's height are read from the REBUILT
        tracks, head and toe solves included. Those are new numbers, and any difference from
        7b's offline figures belongs to the head and toe solves and to the projection.
    """
    from autoanim_gnm.body import forward_kinematics_positions, skeleton_for_track  # noqa: E402
    from autoanim_gnm.performer_skeleton import performer_skeleton  # noqa: E402
    import d7_world_vertical_delivery as wv  # noqa: E402
    from autoanim_gnm import commercial_multiview as cmv  # noqa: E402
    import triangulate_soma as ts  # noqa: E402

    ts.OUT = OUT_DIR
    spine, _support, _used = ts.triangulate([1])
    shipped = cmv._pelvis_world_frames
    out: dict = {}
    for subject in (0, 1):
        source = np.load(
            (OUT_DIR / "delivery" / f"subject-{subject:02d}.body-track.npz")
        )["triangulated_world_positions_z_up_m"].astype(np.float64)
        skeleton = performer_skeleton(cmv.DETAILED_HUMANOID, source)[0]
        hip_mid = 0.5 * (source[:, cmv.JOINT_INDEX["left_hip"]]
                         + source[:, cmv.JOINT_INDEX["right_hip"]])
        trunk = source[:, cmv.JOINT_INDEX["neck"]] - hip_mid
        tilt_s = np.degrees(np.arccos(np.clip(
            trunk[:, 2] / np.linalg.norm(trunk, axis=1), -1.0, 1.0)))
        bent = tilt_s >= np.percentile(tilt_s, 66.7)
        row: dict = {}
        for label, patch, spine_input in (
            ("D3_no_pelvis_frame", None, None),
            ("D7", None, spine[subject, :, 0, :]),
            ("WORLD_VERTICAL", wv.world_vertical_frames, spine[subject, :, 0, :]),
        ):
            saved = cmv.project_generated_foot_contacts
            captured: list = []

            def watcher(track, **kwargs):
                captured.append(track)
                return saved(track, **kwargs)

            cmv.project_generated_foot_contacts = watcher
            if patch is not None:
                cmv._pelvis_world_frames = patch
            try:
                solved = cmv.positions_to_body_track(
                    source, sample_rate_hz=30, provenance_sha256="0" * 64,
                    spine_world_z_up_m=spine_input, skeleton=skeleton)
            finally:
                cmv.project_generated_foot_contacts = saved
                cmv._pelvis_world_frames = shipped
            unprojected = np.asarray(captured[-1].root_translation_m, np.float64)
            projected = np.asarray(solved.root_translation_m, np.float64)
            row[label] = {"hoist_median_mm_OFFLINE_no_head_or_toe_solve": round(
                1000.0 * float(np.median(np.linalg.norm(projected - unprojected, axis=1))), 2)}
        # and from the REBUILT tracks, head and toe solves included
        for label, delivery in (("D3", ROOT / "artifacts/commercial-multiview-soma77"),
                                ("D7", OUT_DIR / "delivery"),
                                ("WORLD_VERTICAL", OUT_DIR / "delivery-world-vertical")):
            track = cmv.BodyTrack.from_dict(json.loads(
                (delivery / f"subject-{subject:02d}.body-track.json").read_text()))
            base = skeleton_for_track(track)
            world = forward_kinematics_positions(
                np.asarray(track.root_translation_m, np.float64),
                np.asarray(track.local_rotations_xyzw, np.float64), skeleton=base)
            rig_hips = world[:, track.joint_names.index("Hips")]
            capture = np.stack([rig_hips[:, 0], -rig_hips[:, 2], rig_hips[:, 1]], axis=-1)
            horizontal = np.linalg.norm((capture - hip_mid)[:, :2], axis=1) * 1000.0
            key = "D3_no_pelvis_frame" if label == "D3" else label
            row.setdefault(key, {}).update({
                "hips_offset_median_mm_REBUILT": round(float(np.median(horizontal)), 4),
                "hips_offset_bent_tercile_median_mm_REBUILT": round(
                    float(np.median(horizontal[bent])), 4),
                "correlation_with_trunk_tilt_REBUILT": round(
                    float(np.corrcoef(horizontal, tilt_s)[0, 1]), 4),
                "lowest_joint_y_m_REBUILT": round(float(np.min(world[..., 1])), 5),
                "foot_contacts_REBUILT": [int(v) for v in np.sum(track.foot_contacts, axis=0)],
            })
        out[f"subject_{subject:02d}"] = row
    out["section_7b_offline_figures_for_comparison"] = {
        "hoist_median_mm": {"subject_00": {"before": 72.68, "D7": 25.34, "world_vertical": 20.13},
                            "subject_01": {"before": 35.54, "D7": 34.11, "world_vertical": 60.15}},
        "hips_offset_bent_tercile_mm": {
            "subject_00": {"before": 68.74, "D7": 27.98, "world_vertical": 6.36},
            "subject_01": {"before": 79.20, "D7": 79.75, "world_vertical": 0.0000001}},
        "note": ("the OFFLINE hoist here must reproduce these by construction -- same "
                 "converter, same substitution, same omission of the head and toe solves. "
                 "The REBUILT offsets need not: those carry the head and toe solves."),
    }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    shape = (width, height)
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}
    frames = sil.FRAMES

    # ---- the two builds must share triangulated positions, or tilt is not one variable
    positions = {label: np.stack([
        np.load(delivery / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"] for s in (0, 1)])
        for label, delivery, _ in BUILDS}
    identical = positions["D3"].tobytes() == positions["D7"].tobytes()
    if not identical:
        raise SystemExit("the two builds do not share triangulated positions")
    cap = positions["D3"]
    glbs_identical = all(
        (ROOT / f"artifacts/commercial-multiview-soma77/subject-{s:02d}.glb").read_bytes()
        == (ROOT / f"artifacts/compare/d3-skeleton/delivery/subject-{s:02d}.glb").read_bytes()
        for s in (0, 1))

    from subject_map import mamma_index_for  # noqa: E402

    subject_to_body = mamma_index_for(cap)
    tilt_report = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/silhouette-vs-tilt.json").read_text())
    subject_to_tracklet = {
        cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
        for cam, row in tilt_report["identity"]["our_subject_to_mask_tracklet"].items()}

    # ---- trunk tilt per frame, camera-independent and identical for both builds
    tilt = np.zeros((2, frames))
    for s in (0, 1):
        pelvis = 0.5 * (cap[s][:, JOINT_INDEX["left_hip"]] + cap[s][:, JOINT_INDEX["right_hip"]])
        up = cap[s][:, JOINT_INDEX["neck"]] - pelvis
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        tilt[s] = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))

    # ---- meshes. WORK is rebound to THIS step's own directories; i6 is never written.
    saved_delivery, saved_work = sil.DELIVERY, sil.WORK
    sil.WORK = MASK_WORK
    masks = sil.MaskStore(SCALE, cams)
    meshes = {}
    for label, delivery, work in BUILDS:
        if not delivery.exists():
            raise SystemExit(f"{delivery} is missing; build the {label} arm first")
        # `delivered_mesh()` exports through the REAL Blender path (fps 30 before import)
        # when its cache is absent, which is how the control arm's mesh is made -- the same
        # code path the other two arms' caches came from.
        sil.DELIVERY, sil.WORK = delivery, work
        meshes[label] = sil.delivered_mesh()
    sil.DELIVERY, sil.WORK = saved_delivery, saved_work

    split = pw.split_ours()
    ours_faces = split["triangles"]
    smplx_face_is_arm, _ = pw.split_smplx()
    with np.load(sil.SMPLX, allow_pickle=True) as data:
        smplx_faces = data["f"].astype(np.int32)
    pred_vertices = {b: np.load(sil.MA3D / f"verts_joints_body_id-{b:02d}.npz",
                                allow_pickle=True)["pred_vertices"] for b in (0, 1)}

    arms: dict = {}
    for label in [name for name, _, _ in BUILDS]:
        arms[label] = {s: (meshes[label][f"verts_{s:02d}"].astype(np.float32),
                           ours_faces[split["face_is_arm"]],
                           ours_faces[~split["face_is_arm"]]) for s in (0, 1)}
    arms["ORACLE_mamma_mesh"] = {
        s: (pred_vertices[subject_to_body[s]].astype(np.float32),
            smplx_faces[smplx_face_is_arm], smplx_faces[~smplx_face_is_arm])
        for s in (0, 1)}
    names = list(arms)

    stats = {n: {k: np.full((len(cams), 2, frames), np.nan) for k in KEYS} for n in names}
    population = np.zeros((len(cams), 2, frames), dtype=bool)
    todo = list(names)
    if PER_FRAME.exists():
        cached = np.load(PER_FRAME)
        population = cached["population"].astype(bool)
        todo = []
        for n in names:
            if all(f"{n}|{k}" in cached.files for k in KEYS):
                for k in KEYS:
                    stats[n][k] = cached[f"{n}|{k}"]
            else:
                todo.append(n)
        print(f"cached: {[n for n in names if n not in todo]}; rendering: {todo}")

    for c, cam in enumerate(cams):
        for s in (0, 1):
            if not todo:
                break
            mask = masks.get(cam, subject_to_tracklet[cam][s])
            area = mask.reshape(frames, -1).sum(axis=1)
            population[c, s] = area >= sil.MIN_MASK_PX
            for f in np.nonzero(population[c, s])[0]:
                m = mask[f]
                for n in todo:
                    verts, arm_faces, torso_faces = arms[n][s]
                    v = verts[f if len(verts) > 1 else 0]
                    arm = sil.rasterise(v, arm_faces, scaled[cam], shape)
                    torso = sil.rasterise(v, torso_faces, scaled[cam], shape)
                    whole = arm | torso
                    for tag, raster in (("torso", torso), ("arm", arm), ("whole", whole)):
                        p, r, i = sil.score(raster, m)
                        stats[n][f"{tag}_precision"][c, s, f] = p
                        stats[n][f"{tag}_recall"][c, s, f] = r
                        stats[n][f"{tag}_iou"][c, s, f] = i
            print(f"scored {cam} subject {s:02d}: {int(population[c, s].sum())} frames")
    if todo:
        np.savez_compressed(
            PER_FRAME, population=population,
            **{f"{n}|{k}": stats[n][k] for n in names for k in KEYS})

    # ---------------------------------------------------------------------- the clauses
    rng = np.random.default_rng(SEED)
    draws = pw.block_draws(rng, frames)
    report: dict = {
        "title": "D7 B7 -- the photographs, part-wise and by trunk tilt",
        "instrument_only": True,
        "reference": ("MAMMA's SAM2 person masks -- the pixels of the actual footage. The "
                      "one band the candidate cannot optimise."),
        "preregistered": PREREGISTERED,
        "builds": {label: str(delivery.relative_to(ROOT)) for label, delivery, _ in BUILDS},
        "before_arm_is_the_committed_delivery_and_it_IS_D3": {
            "committed_glbs_byte_identical_to_the_d3_rebuild": glbs_identical,
            "d3_mesh_cache_copied_not_shared": (
                "artifacts/compare/i6/delivered-mesh.npz was COPIED into "
                "artifacts/compare/d7-pelvis-frame/silhouette-work-d3/, byte-identical to "
                "artifacts/compare/d3-skeleton/silhouette-work/delivered-mesh.npz; nothing "
                "under artifacts/compare/i6 or artifacts/commercial-multiview-soma77 is "
                "written by this instrument"),
        },
        "triangulated_positions_byte_identical": identical,
        "part_split": {
            "how": ("dominant skin weight from the body asset the delivery was built from; "
                    "a face goes wholly to the part owning two or three of its corners, so "
                    "the two rasters partition the surface"),
            "arm_faces": int(split["face_is_arm"].sum()),
            "torso_faces": int((~split["face_is_arm"]).sum()),
            "clavicle_weight_bleed": pw.clavicle_weight_bleed(split),
        },
        "statistics": {"moving_block": BLOCK, "draws": DRAWS, "seed": SEED,
                       "both_arms_on_identical_draws": True,
                       "lag1_autocorrelation_on_this_take": 0.99},
        "subjects": {},
    }

    verdicts: dict = {}
    build_names = [name for name, _, _ in BUILDS]
    pairs = [("D7", "D3"), ("WORLD_VERTICAL", "D3"), ("D7", "WORLD_VERTICAL"),
             ("WORLD_VERTICAL", "D7")]
    pairs = [(a, b) for a, b in pairs if a in build_names and b in build_names]
    for s in (0, 1):
        keep = population[:, s, :].all(axis=0)          # frames EVERY camera scored
        edges = [float(np.percentile(tilt[s][keep], 100 / 3.0)),
                 float(np.percentile(tilt[s][keep], 200 / 3.0))]
        terciles = {
            "upright": keep & (tilt[s] <= edges[0]),
            "middle": keep & (tilt[s] > edges[0]) & (tilt[s] <= edges[1]),
            "bent": keep & (tilt[s] > edges[1]),
        }
        row: dict = {
            "frames_every_camera_scored": int(keep.sum()),
            "tercile_edges_deg": [round(e, 2) for e in edges],
            "tilt_range_deg": [round(float(tilt[s][keep].min()), 2),
                               round(float(tilt[s][keep].max()), 2)],
            "terciles": {},
        }
        series: dict = {}
        for part in ("torso", "arm", "whole"):
            series[part] = {n: per_frame_mean(stats[n][f"{part}_iou"][:, s, :], keep)
                            for n in build_names}
            series[part]["ORACLE"] = per_frame_mean(
                stats["ORACLE_mamma_mesh"][f"{part}_iou"][:, s, :], keep)
            for name, mask_frames in terciles.items():
                cell = row["terciles"].setdefault(
                    name, {"n": int(mask_frames.sum()),
                           "tilt_deg": [round(float(tilt[s][mask_frames].min()), 2),
                                        round(float(tilt[s][mask_frames].max()), 2)]})
                for arm_name in build_names + ["ORACLE"]:
                    cell[f"{part}_iou_{arm_name}"] = round(
                        float(np.median(series[part][arm_name][mask_frames])), 5)
                for a, b in pairs:
                    cell[f"{part}_{a}_minus_{b}"] = pw.paired(
                        series[part][a], series[part][b], mask_frames, draws)
            row[f"{part}_iou_all_frames"] = {
                **{arm_name: round(float(np.median(series[part][arm_name][keep])), 5)
                   for arm_name in build_names + ["ORACLE"]},
                **{f"{a}_minus_{b}": pw.paired(series[part][a], series[part][b], keep, draws)
                   for a, b in pairs},
            }
        report["subjects"][f"subject_{s:02d}"] = row

        bent = row["terciles"]["bent"]
        upright = row["terciles"]["upright"]
        d7_d3 = bent["torso_D7_minus_D3"]
        arm_d7 = row["arm_iou_all_frames"]["D7_minus_D3"]
        up_d7 = upright["whole_D7_minus_D3"]
        verdicts[f"subject_{s:02d}"] = {
            "clause_1_torso_legs_rises_on_the_bent_tercile": {
                "difference": d7_d3["median_difference"], "ci95": d7_d3["ci95"],
                "verdict": "PASS" if (d7_d3["ci95"] and d7_d3["ci95"][0] > 0.0) else "FAIL",
                "band": "rises, interval clear of zero",
            },
            "clause_2_arms_within_their_CI_of_D3": {
                "difference": arm_d7["median_difference"], "ci95": arm_d7["ci95"],
                "verdict": "PASS" if (arm_d7["ci95"] and arm_d7["ci95"][0] <= 0.0 <= arm_d7["ci95"][1])
                           else "FAIL",
                "band": "the CI of the difference contains zero",
            },
            "clause_3_upright_tercile_within_its_CI": {
                "difference": up_d7["median_difference"], "ci95": up_d7["ci95"],
                "verdict": "PASS" if (up_d7["ci95"] and up_d7["ci95"][0] <= 0.0 <= up_d7["ci95"][1])
                           else "FAIL",
                "band": "the CI of the difference contains zero",
            },
        }
        if "WORLD_VERTICAL" in build_names:
            control_d3 = bent["torso_WORLD_VERTICAL_minus_D3"]
            arm_control = row["arm_iou_all_frames"]["WORLD_VERTICAL_minus_D3"]
            verdicts[f"subject_{s:02d}"]["option_c_predictions"] = {
                "P1_control_rise_within_D7s_CI_on_performer_0": {
                    "applies_to": "subject_00",
                    "control_minus_D3": control_d3["median_difference"],
                    "control_ci95": control_d3["ci95"],
                    "D7_minus_D3": d7_d3["median_difference"], "D7_ci95": d7_d3["ci95"],
                    "control_point_estimate_inside_D7s_CI": bool(
                        d7_d3["ci95"] and d7_d3["ci95"][0] <= control_d3["median_difference"]
                        <= d7_d3["ci95"][1]),
                },
                "P2_control_FALLS_on_performer_1_while_D7_stays_within_CI": {
                    "applies_to": "subject_01",
                    "control_minus_D3": control_d3["median_difference"],
                    "control_ci95": control_d3["ci95"],
                    "control_falls_with_ci_clear_of_zero": bool(
                        control_d3["ci95"] and control_d3["ci95"][1] < 0.0),
                    "D7_within_its_CI_of_D3": bool(
                        d7_d3["ci95"] and d7_d3["ci95"][0] <= 0.0 <= d7_d3["ci95"][1]),
                },
                "P3_arms_within_CI_on_every_arm": {
                    "D7_minus_D3": arm_d7["median_difference"], "D7_ci95": arm_d7["ci95"],
                    "control_minus_D3": arm_control["median_difference"],
                    "control_ci95": arm_control["ci95"],
                    "both_contain_zero": bool(
                        arm_d7["ci95"] and arm_d7["ci95"][0] <= 0.0 <= arm_d7["ci95"][1]
                        and arm_control["ci95"] and arm_control["ci95"][0] <= 0.0 <= arm_control["ci95"][1]),
                },
            }

    # ---- clause 4: the oracle reads none of our track
    committed = {}
    for label, path in (("silhouette_d3", ROOT / "artifacts/compare/d3-skeleton/silhouette-d3.json"),
                        ("silhouette_d7", OUT_DIR / "silhouette-d7.json")):
        payload = json.loads(path.read_text())
        committed[label] = {
            f"{cam}/subject_{s:02d}": payload["arms"]["ORACLE_mamma_mesh"][cam][
                f"subject_{s:02d}"]["iou"]["median"]
            for cam in sil.CAMERAS for s in (0, 1)}
    ours = {}
    for c, cam in enumerate(sil.CAMERAS):
        for s in (0, 1):
            values = stats["ORACLE_mamma_mesh"]["whole_iou"][c, s, population[c, s]]
            ours[f"{cam}/subject_{s:02d}"] = float(np.median(values))
    same_between_runs = committed["silhouette_d3"] == committed["silhouette_d7"]
    worst_split = max(abs(ours[k] - committed["silhouette_d3"][k]) for k in ours)
    verdicts["clause_4_mamma_mesh_oracle"] = {
        "bit_identical_between_the_two_whole_person_runs": same_between_runs,
        "per_cell_medians_between_runs": {
            k: {"d3_run": committed["silhouette_d3"][k], "d7_run": committed["silhouette_d7"][k]}
            for k in committed["silhouette_d3"]},
        "this_instruments_split_oracle_vs_the_unsplit_one_worst_abs_difference": round(worst_split, 8),
        "why_the_split_check_matters": (
            "the oracle here is rasterised as arm|torso, the committed runs rasterised it "
            "whole. Agreement to float precision says the part split partitions the surface "
            "exactly and adds no pixels of its own."),
        "verdict": "PASS" if same_between_runs and worst_split < 1e-9 else "FAIL",
    }
    # ------------------------------------------------- THE DECISION RULE, fixed before
    if "WORLD_VERTICAL" in [name for name, _, _ in BUILDS]:
        rule: dict = {
            "verbatim": (
                "On the bent tercile's torso+legs figure with the block bootstrap on "
                "identical draws: D7 is MERGED with the B2a failure stated (as D2 was) if "
                "BOTH hold -- (i) on neither performer is the control better than D7 with a "
                "CI clear of zero, and (ii) on at least one performer D7 is better than the "
                "control with a CI clear of zero. Otherwise D7 is HELD."),
            "fixed_by": "the coordinator, before any number in this pass existed",
            "per_subject": {},
        }
        control_better_anywhere = False
        d7_better_somewhere = False
        for s in (0, 1):
            bent = report["subjects"][f"subject_{s:02d}"]["terciles"]["bent"]
            control_minus_d7 = bent["torso_WORLD_VERTICAL_minus_D7"]
            d7_minus_control = bent["torso_D7_minus_WORLD_VERTICAL"]
            control_wins = bool(control_minus_d7["ci95"] and control_minus_d7["ci95"][0] > 0.0)
            d7_wins = bool(d7_minus_control["ci95"] and d7_minus_control["ci95"][0] > 0.0)
            control_better_anywhere = control_better_anywhere or control_wins
            d7_better_somewhere = d7_better_somewhere or d7_wins
            rule["per_subject"][f"subject_{s:02d}"] = {
                "torso_legs_bent_tercile": {
                    "D3": bent["torso_iou_D3"], "D7": bent["torso_iou_D7"],
                    "WORLD_VERTICAL": bent["torso_iou_WORLD_VERTICAL"],
                    "ORACLE": bent["torso_iou_ORACLE"]},
                "control_minus_D7": control_minus_d7,
                "D7_minus_control": d7_minus_control,
                "control_better_than_D7_with_CI_clear_of_zero": control_wins,
                "D7_better_than_control_with_CI_clear_of_zero": d7_wins,
            }
        rule["clause_i_no_performer_where_control_beats_D7"] = not control_better_anywhere
        rule["clause_ii_at_least_one_performer_where_D7_beats_control"] = d7_better_somewhere
        rule["verdict"] = ("MERGE, with the B2a failure stated"
                           if (not control_better_anywhere) and d7_better_somewhere
                           else "HOLD")
        rule["what_this_verdict_is_NOT"] = (
            "it is not a finding that D7 is right. A silhouette cannot score orientation "
            "against truth, and B2a's failure is untouched by it. The rule asks only whether "
            "the photographs DISTINGUISH the measured pelvis from a frozen one.")
        report["DECISION_RULE"] = rule
    report["preregistered_clause_verdicts"] = verdicts
    clause_pass = all(
        v["verdict"] == "PASS"
        for subject in ("subject_00", "subject_01")
        for v in verdicts[subject].values()) and verdicts["clause_4_mamma_mesh_oracle"]["verdict"] == "PASS"
    report["verdict"] = "PASS" if clause_pass else "FAIL"
    if "WORLD_VERTICAL" in [name for name, _, _ in BUILDS]:
        report["control_delivery_hoist_and_offset"] = control_figures()
    report["blind_to"] = (
        "depth; a left/right mirror of a fore-aft symmetric pose; and, WITHIN each part, "
        "where inside the outline a limb sits -- an arm wrongly placed but still inside the "
        "arm region is not separated from a right one. Every figure is read against a mesh "
        "bound to an asset whose proportions are not the performer's, and that mismatch is "
        "the background all of it sits on.")
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    for s in (0, 1):
        row = report["subjects"][f"subject_{s:02d}"]
        print(f"\n=== subject {s:02d}  ({row['frames_every_camera_scored']} frames every "
              f"camera scored; tercile edges {row['tercile_edges_deg']} deg) ===")
        for name, cell in row["terciles"].items():
            print(f"  {name:8s} n={cell['n']:3d} tilt {cell['tilt_deg']}  "
                  f"torso {cell['torso_iou_D3']:.4f} -> {cell['torso_iou_D7']:.4f} "
                  f"({cell['torso_D7_minus_D3']['median_difference']:+.4f} "
                  f"{cell['torso_D7_minus_D3']['ci95']})  "
                  f"whole {cell['whole_iou_D3']:.4f} -> {cell['whole_iou_D7']:.4f} "
                  f"({cell['whole_D7_minus_D3']['median_difference']:+.4f})")
        arm = row["arm_iou_all_frames"]
        print(f"  arms, all frames: {arm['D3']:.4f} -> {arm['D7']:.4f} "
              f"({arm['D7_minus_D3']['median_difference']:+.4f} {arm['D7_minus_D3']['ci95']})")
    print("\nCLAUSE VERDICTS")
    print(json.dumps(verdicts, indent=1)[:2600])
    print(f"\nOVERALL: {report['verdict']}  ->  {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
