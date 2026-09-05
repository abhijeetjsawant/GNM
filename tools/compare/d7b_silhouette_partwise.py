#!/usr/bin/env python3
"""D7b's B4: the photographs, part-wise and by trunk tilt. The band the candidate cannot optimise.

INSTRUMENT ONLY. Nothing ships. It is `d7_silhouette_partwise.py`'s structure with THREE
arms -- the D3 archive, the shipped D7 build, and this step's D7b rebuild -- and it reuses
`silhouette.py`'s rasteriser, scorer and mask store and `silhouette_partwise.py`'s part
split, block draws and paired bootstrap, so every arm goes through one pixel path and one
statistic.

WHY A NEW FILE AND NOT A PARAMETER. `d7_silhouette_partwise.py` carries D7's three builds,
D7's pre-registered clause text and D7's option-(c) decision rule in its constants and its
`main`, and its committed report is the record of that pass. Parameterising it would
rewrite a trusted instrument to serve a different question -- exactly the reason D7 did not
edit `silhouette_partwise.py`.

**NOTHING is written under `artifacts/commercial-multiview-soma77/`, `artifacts/compare/i6/`
or `artifacts/compare/d7-pelvis-frame/`.** The mesh caches and the mask cache are COPIED
into this step's own work directories and asserted byte-identical to their sources, so
`delivered_mesh()` can never re-export into a directory a committed instrument reads. Only
the D7b arm's mesh is new, and it is exported through the REAL Blender path (fps 30 before
import) -- the same code path the other two caches came from.

THE PRE-REGISTERED CLAUSES, verbatim from `docs/LADDER_EXECUTION_PLAN.md` section 2, the
D7b card, band B4, written before this rebuild existed:

    part-wise silhouette on the tilt terciles, torso AND arms not worse than D7 on EITHER
    performer with the CI clear, improvement predicted on performer 0's bent tercile
    torso; MAMMA mesh bit-identical

BLIND TO: depth; a left/right mirror of a fore-aft symmetric pose; and, within each part,
where inside the outline a limb sits. Every figure is read against a mesh bound to an asset
whose proportions are not the performer's, so this scores a CHANGE against a large standing
shape mismatch. It also cannot see the mesh distortion the post-D7 review found (ballooning
hips on lying frames, a tearing thigh in the squat): a dilated blob can raise recall while
precision falls. Precision and recall are reported beside every IoU for that reason.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7b_silhouette_partwise.py

Writes `artifacts/compare/d7b-trunk/silhouette-partwise.json` and caches every per-frame
score in `silhouette-partwise-per-frame.npz`, so a re-cut is free.
"""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
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
import d7_silhouette_partwise as p7  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7b-trunk"
D7_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
REPORT = OUT_DIR / "silhouette-partwise.json"
PER_FRAME = OUT_DIR / "silhouette-partwise-per-frame.npz"

# label -> (delivery, this step's OWN work directory, the cache to seed it from or None)
BUILDS = (
    ("D3", ROOT / "artifacts/compare/delivered-before-d7-2026-09-05",
     OUT_DIR / "silhouette-work-d3", D7_DIR / "silhouette-work-d3/delivered-mesh.npz"),
    ("D7", ROOT / "artifacts/commercial-multiview-soma77",
     OUT_DIR / "silhouette-work-d7", D7_DIR / "silhouette-work/delivered-mesh.npz"),
    ("D7b", OUT_DIR / "delivery", OUT_DIR / "silhouette-work-d7b", None),
)
MASK_WORK = OUT_DIR / "silhouette-work"          # holds the copied SAM2 mask cache
MASK_SOURCE = D7_DIR / "silhouette-work"
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260905

PREREGISTERED = {
    "source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D7b card, band B4, committed "
               "before the rebuild existed"),
    "torso_not_worse_than_D7": "on EITHER performer, with the CI clear.",
    "arms_not_worse_than_D7": "on EITHER performer, with the CI clear.",
    "performer_0_bent_tercile_torso": "IMPROVEMENT predicted -- that is where the delivered "
                                      "neck moved 131 -> 42 mm and the arms with it.",
    "mamma_mesh_oracle": "BIT-IDENTICAL between runs -- it reads none of our track, and that "
                         "identity is the proof that reusing the caches changed nothing.",
    "what_this_cannot_settle": (
        "the silhouette scores the MESH. The skin has been stretched onto a per-performer "
        "skeleton with the asset's old weights since D3, and the post-D7 review saw it "
        "balloon at the hips and tear at the thigh. A rise here is not proof the skin is "
        "right; D6 owns that, with a mesh-distortion instrument first."),
}

KEYS = p7.KEYS
per_frame_mean = p7.per_frame_mean


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def seed_caches() -> dict:
    """COPY the mask cache and the two existing mesh caches in, and prove the copies."""

    proof: dict = {}
    MASK_WORK.mkdir(parents=True, exist_ok=True)
    for cache in sorted(MASK_SOURCE.glob("masks-*.npz")):
        destination = MASK_WORK / cache.name
        if not destination.exists():
            shutil.copy2(cache, destination)
        proof[f"mask/{cache.name}"] = {
            "source": str(cache.relative_to(ROOT)),
            "byte_identical": digest(cache) == digest(destination)}
    for label, _delivery, work, source in BUILDS:
        if source is None:
            continue
        work.mkdir(parents=True, exist_ok=True)
        destination = work / "delivered-mesh.npz"
        if not destination.exists():
            shutil.copy2(source, destination)
        proof[f"mesh/{label}"] = {
            "source": str(source.relative_to(ROOT)),
            "byte_identical": digest(source) == digest(destination)}
    return proof


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copies = seed_caches()
    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    shape = (width, height)
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}
    frames = sil.FRAMES

    # ---- every arm must share triangulated positions, or tilt is not one variable
    positions = {label: np.stack([
        np.load(delivery / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"] for s in (0, 1)])
        for label, delivery, _, _ in BUILDS}
    labels = [label for label, _, _, _ in BUILDS]
    identical = all(positions[label].tobytes() == positions["D3"].tobytes()
                    for label in labels)
    if not identical:
        raise SystemExit("the builds do not share triangulated positions")
    cap = positions["D3"]

    from subject_map import mamma_index_for  # noqa: E402

    subject_to_body = mamma_index_for(cap)
    tilt_report = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/silhouette-vs-tilt.json").read_text())
    subject_to_tracklet = {
        cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
        for cam, row in tilt_report["identity"]["our_subject_to_mask_tracklet"].items()}

    # ---- trunk tilt per frame, camera-independent and identical for every arm
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
    for label, delivery, work, _ in BUILDS:
        if not delivery.exists():
            raise SystemExit(f"{delivery} is missing; build the {label} arm first")
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
    for label in labels:
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
        "title": "D7b B4 -- the photographs, part-wise and by trunk tilt",
        "instrument_only": True,
        "reference": ("MAMMA's SAM2 person masks -- the pixels of the actual footage. The "
                      "one band the candidate cannot optimise."),
        "preregistered": PREREGISTERED,
        "builds": {label: str(delivery.relative_to(ROOT)) for label, delivery, _, _ in BUILDS},
        "caches_copied_never_shared": copies,
        "triangulated_positions_byte_identical": identical,
        "part_split": {
            "how": ("dominant skin weight from the body asset the delivery was built from; "
                    "a face goes wholly to the part owning two or three of its corners, so "
                    "the two rasters partition the surface"),
            "arm_faces": int(split["face_is_arm"].sum()),
            "torso_faces": int((~split["face_is_arm"]).sum()),
        },
        "statistics": {"moving_block": BLOCK, "draws": DRAWS, "seed": SEED,
                       "every_arm_on_identical_draws": True,
                       "lag1_autocorrelation_on_this_take": 0.99},
        "subjects": {},
    }

    verdicts: dict = {}
    pairs = [("D7b", "D7"), ("D7b", "D3"), ("D7", "D3")]
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
                            for n in labels}
            series[part]["ORACLE"] = per_frame_mean(
                stats["ORACLE_mamma_mesh"][f"{part}_iou"][:, s, :], keep)
            for name, mask_frames in terciles.items():
                cell = row["terciles"].setdefault(
                    name, {"n": int(mask_frames.sum()),
                           "tilt_deg": [round(float(tilt[s][mask_frames].min()), 2),
                                        round(float(tilt[s][mask_frames].max()), 2)]})
                for arm_name in labels + ["ORACLE"]:
                    cell[f"{part}_iou_{arm_name}"] = round(
                        float(np.median(series[part][arm_name][mask_frames])), 5)
                for a, b in pairs:
                    cell[f"{part}_{a}_minus_{b}"] = pw.paired(
                        series[part][a], series[part][b], mask_frames, draws)
            row[f"{part}_iou_all_frames"] = {
                **{arm_name: round(float(np.median(series[part][arm_name][keep])), 5)
                   for arm_name in labels + ["ORACLE"]},
                **{f"{a}_minus_{b}": pw.paired(series[part][a], series[part][b], keep, draws)
                   for a, b in pairs},
            }
        # precision and recall beside the IoU, because a dilated blob raises one and
        # lowers the other and the IoU alone cannot tell that story.
        row["precision_recall_all_frames"] = {
            arm_name: {key: round(float(np.nanmedian(
                per_frame_mean(stats[arm_name][key][:, s, :], keep)[keep])), 5)
                for key in ("torso_precision", "torso_recall", "arm_precision", "arm_recall")}
            for arm_name in labels}
        report["subjects"][f"subject_{s:02d}"] = row

        bent = row["terciles"]["bent"]
        torso_all = row["torso_iou_all_frames"]["D7b_minus_D7"]
        arm_all = row["arm_iou_all_frames"]["D7b_minus_D7"]
        torso_bent = bent["torso_D7b_minus_D7"]
        not_worse = lambda d: bool(d["ci95"] and d["ci95"][1] >= 0.0)
        verdicts[f"subject_{s:02d}"] = {
            "clause_1_torso_not_worse_than_D7": {
                "difference": torso_all["median_difference"], "ci95": torso_all["ci95"],
                "band": "not worse: the CI's upper bound is at or above zero",
                "verdict": "PASS" if not_worse(torso_all) else "FAIL"},
            "clause_2_arms_not_worse_than_D7": {
                "difference": arm_all["median_difference"], "ci95": arm_all["ci95"],
                "band": "not worse: the CI's upper bound is at or above zero",
                "verdict": "PASS" if not_worse(arm_all) else "FAIL"},
            "clause_3_bent_tercile_torso_not_worse_than_D7": {
                "difference": torso_bent["median_difference"], "ci95": torso_bent["ci95"],
                "band": "not worse; on subject_00 an IMPROVEMENT was predicted",
                "verdict": "PASS" if not_worse(torso_bent) else "FAIL",
                "improvement_predicted_here": s == 0,
                "improved_with_ci_clear_of_zero": bool(
                    torso_bent["ci95"] and torso_bent["ci95"][0] > 0.0)},
        }

    # ---- the oracle reads none of our track
    committed = {}
    for label, path in (("silhouette_d3", ROOT / "artifacts/compare/d3-skeleton/silhouette-d3.json"),
                        ("silhouette_d7", D7_DIR / "silhouette-d7.json")):
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
        "bit_identical_between_the_committed_whole_person_runs": same_between_runs,
        "this_instruments_split_oracle_vs_the_unsplit_one_worst_abs_difference":
            round(worst_split, 8),
        "why_it_matters": ("the oracle is rasterised here as arm|torso and was rasterised "
                           "whole in the committed runs. Agreement to float precision says "
                           "the part split partitions the surface exactly."),
        "verdict": "PASS" if same_between_runs and worst_split < 1e-9 else "FAIL",
    }
    clauses = [row["verdict"] for key, row in verdicts.items() if key.startswith("subject")
               for row in verdicts[key].values()]
    report["preregistered_clause_verdicts"] = verdicts
    report["verdict"] = ("PASS" if all(v == "PASS" for v in clauses)
                         and verdicts["clause_4_mamma_mesh_oracle"]["verdict"] == "PASS"
                         else "FAIL")
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
