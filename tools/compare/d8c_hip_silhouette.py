#!/usr/bin/env python3
"""D8c's B1: the photographs, part-wise, TORSO+LEGS and ARMS. The band the candidate cannot optimise.

INSTRUMENT ONLY. Nothing ships. It is `d8b_length_silhouette.py`'s structure with its two
arms rebound -- the shipped D8b build and this step's D8c rebuild -- and it reuses
`silhouette.py`'s rasteriser, scorer and mask store and `silhouette_partwise.py`'s part
split, block draws and paired bootstrap, so both arms go through one pixel path and one
statistic. That file is not edited: it carries D8b's builds, D8b's clause text and its
committed report, which is the record of that pass.

**NOTHING is written under `artifacts/commercial-multiview-soma77/` or under any earlier
step's directory.** The SAM2 mask cache is COPIED into this step's own work directory and
asserted byte-identical to its source. BOTH meshes are exported fresh through the REAL
Blender path into this step's own directories.

THE PART. `silhouette_partwise.split_ours()` labels every face by whether its largest
influence is an ARM joint. The complement -- what this file rasterises as "torso" -- is the
TORSO, THE LEGS AND THE HEAD, and it is the part the card names, because the hip line is a
pelvis segment and the legs hang off it. The arm part is scored and banded too, and it is
the CONTROL that says the change stayed where it belongs: the hip row charges only the hips.

THE DENOMINATOR IS THE RAW ARRAY, and the smoothed one is REPORTED. The hip row sits between
the raw triangulation and the smoothed landmarks, exactly where D8's three rules and D8b's
nine sit, so the smoothed array MUST move -- that is the step -- and a smoothed array that
had NOT moved would mean the step did nothing. The raw array is captured before the reject
and is asserted byte-identical; it is the one reference both arms share.

FOUR CUTS AND THE CARD ASKS FOR ALL OF THEM: the whole take, and each of the three runs the
card classifies as its own cut -- 83-87 (the one-hip frames plus a frame either side),
109-119 (the inward collapse in agreeing views) and 158-168 (the outward stretch along the
A-C baseline). Two of those cuts are FIVE and ELEVEN frames long, which is under one moving
block, and the intervals there are correspondingly enormous. THAT IS STATED BEFORE THE
NUMBERS AND NOT AFTER: on a run of five frames this instrument cannot resolve anything, and
the whole take and the pooled three-run cut are what the band is really read on.

THE PRE-REGISTERED CLAUSE, verbatim from `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8c
card, band B1, committed at 85b8113 before this rebuild existed:

    (B1) the photographs: part-wise silhouette, the TORSO+LEGS part, whole take AND the
    three runs (83-87, 109-119, 158-168) as their own cut, not worse on either performer
    with the CI clear on the D7b/D8 predicate (upper bound >= 0), improvement predicted on
    performer 1's runs and reported with its CI

BLIND TO: depth; a left/right mirror of a fore-aft symmetric pose; and, within each part,
where inside the outline a joint sits -- a hip can move centimetres inside its own
silhouette without moving a pixel, and on a body lying on the floor seen by two cameras that
is exactly what a depth stretch does. So this band CANNOT settle class (ii), and it is
paired with B3's placement figures and B2's bit-identity and never quoted alone. Every
figure is read against a mesh bound to an asset whose proportions are not the performer's,
so it scores a CHANGE against a large standing shape mismatch.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_silhouette.py

Writes `artifacts/compare/d8c-hip/silhouette-partwise.json` and caches every per-frame score
in `silhouette-partwise-per-frame.npz`, so a re-cut is free.
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

OUT_DIR = ROOT / "artifacts/compare/d8c-hip"
D7_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
REPORT = OUT_DIR / "silhouette-partwise.json"
PER_FRAME = OUT_DIR / "silhouette-partwise-per-frame.npz"

# label -> (delivery, this step's OWN work directory). Both meshes are exported fresh.
BUILDS = (
    ("D8b", ROOT / "artifacts/commercial-multiview-soma77", OUT_DIR / "silhouette-work-d8b"),
    ("D8c", OUT_DIR / "delivery", OUT_DIR / "silhouette-work-d8c"),
)
MASK_WORK = OUT_DIR / "silhouette-work"
MASK_SOURCE = D7_DIR / "silhouette-work"
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260906

# The card's push-and-fall window, in the observations' own absolute frame ids, and the
# array indices they correspond to. The take's own frame ids run 60..209.
FIRST_FRAME_ID = 60
WINDOW_FIRST_ID, WINDOW_LAST_ID = 85, 125
# The card's three runs, each its own cut, in absolute frame ids, inclusive. 83-87 is the
# one-hip run with a frame either side, as the card writes it.
RUN_CUTS = {"83_87": (83, 87), "109_119": (109, 119), "158_168": (158, 168)}

PREREGISTERED = {
    "source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D8c card, band B1, committed "
               "at 85b8113 before this rebuild existed"),
    "clause": ("the photographs: part-wise silhouette, the TORSO+LEGS part, whole take AND "
               "the three runs (83-87, 109-119, 158-168) as their own cut, not worse on "
               "either performer with the CI clear on the D7b/D8 predicate (upper bound "
               ">= 0), improvement predicted on performer 1's runs and reported with its CI"),
    "torso_not_worse_than_D8b": (
        "whole take AND each run cut, on EITHER performer, with the CI clear on the D7b/D8 "
        "predicate: the upper bound of the 95 % interval on the paired difference is at or "
        "above 0."),
    "arms_are_the_control": (
        "the hip row charges only the hips. The ARM part is scored and banded the same way "
        "and is expected to be unmoved; a fall there would say the change did not stay "
        "where it belongs."),
    "improvement_predicted_on_performer_1s_runs": (
        "his captured hip line reads 125-174 mm on 110-119 and 130-153 on 84-86 against his "
        "own 215, and the delivered root, pelvis frame and legs follow it. Withholding "
        "those frames should put the mesh back over the person."),
    "mamma_mesh_oracle": "BIT-IDENTICAL between runs -- it reads none of our track.",
    "what_this_cannot_settle": (
        "the silhouette scores the MESH, and a joint can move inside its own outline "
        "without moving a pixel. On 158-168 the performer is lying down and seen by two "
        "cameras whose baseline is the very axis the error lives on, so a depth stretch is "
        "close to invisible here BY CONSTRUCTION. It also cannot say whether a WITHHELD "
        "frame was recovered correctly -- only whether the mesh covers more of the person. "
        "Paired with B3's placement figures and B2's bit-identity, never quoted alone."),
    "the_run_cuts_are_tiny": (
        "83-87 is five frames and 109-119 eleven, against a moving block of "
        f"{sil.BLOCK} frames. The intervals on those cuts are enormous and are reported as "
        "such; the whole take and the pooled three-run cut are where the band is read."),
}

KEYS = p7.KEYS
per_frame_mean = p7.per_frame_mean


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def seed_masks() -> dict:
    """COPY the SAM2 mask cache in, and prove the copies. The meshes are NOT copied."""

    proof: dict = {}
    MASK_WORK.mkdir(parents=True, exist_ok=True)
    for cache in sorted(MASK_SOURCE.glob("masks-*.npz")):
        destination = MASK_WORK / cache.name
        if not destination.exists():
            shutil.copy2(cache, destination)
        proof[f"mask/{cache.name}"] = {
            "source": str(cache.relative_to(ROOT)),
            "byte_identical": digest(cache) == digest(destination)}
    return proof


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copies = seed_masks()
    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    shape = (width, height)
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}
    frames = sil.FRAMES

    # ---- SAME DENOMINATOR, on the array D8 does not touch. See the module docstring.
    raw = {label: np.stack([
        np.load(delivery / f"subject-{s:02d}.body-track.npz")[
            "raw_triangulated_world_positions_z_up_m"] for s in (0, 1)])
        for label, delivery, _ in BUILDS}
    smoothed = {label: np.stack([
        np.load(delivery / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"] for s in (0, 1)])
        for label, delivery, _ in BUILDS}
    labels = [label for label, _, _ in BUILDS]
    raw_identical = all(np.array_equal(raw[label], raw["D8b"], equal_nan=True)
                        for label in labels)
    smoothed_identical = all(np.array_equal(smoothed[label], smoothed["D8b"])
                             for label in labels)
    # Only the RAW array is a verdict here. D8b's reject sits between the raw triangulation
    # and the smoothed landmarks, so the smoothed array MUST move; a True there would mean
    # the step did nothing. The raw array is the denominator both arms share and a
    # difference in it would mean something upstream moved and nothing below is comparable.
    if not raw_identical:
        raise SystemExit("the builds do not share their RAW triangulated landmarks -- "
                         "D8b's reject sits after that array is captured and cannot move "
                         "it, so nothing below is comparable")
    cap = raw["D8b"]

    from subject_map import mamma_index_for  # noqa: E402

    subject_to_body = mamma_index_for(smoothed["D8b"])
    tilt_report = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/silhouette-vs-tilt.json").read_text())
    subject_to_tracklet = {
        cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
        for cam, row in tilt_report["identity"]["our_subject_to_mask_tracklet"].items()}

    # ---- trunk tilt from the RAW array, reported beside the cut and never the cut itself
    tilt = np.zeros((2, frames))
    for s in (0, 1):
        pelvis = 0.5 * (cap[s][:, JOINT_INDEX["left_hip"]] + cap[s][:, JOINT_INDEX["right_hip"]])
        up = cap[s][:, JOINT_INDEX["neck"]] - pelvis
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        tilt[s] = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))

    window = np.zeros(frames, dtype=bool)
    window[WINDOW_FIRST_ID - FIRST_FRAME_ID:WINDOW_LAST_ID - FIRST_FRAME_ID + 1] = True

    # ---- meshes. WORK is rebound to THIS step's own directories; i6 is never written.
    saved_delivery, saved_work = sil.DELIVERY, sil.WORK
    sil.WORK = MASK_WORK
    masks = sil.MaskStore(SCALE, cams)
    meshes = {}
    glb_digests = {}
    for label, delivery, work in BUILDS:
        if not delivery.exists():
            raise SystemExit(f"{delivery} is missing; build the {label} arm first")
        work.mkdir(parents=True, exist_ok=True)
        sil.DELIVERY, sil.WORK = delivery, work
        meshes[label] = sil.delivered_mesh()
        glb_digests[label] = {f"subject-{s:02d}.glb":
                              digest(delivery / f"subject-{s:02d}.glb") for s in (0, 1)}
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
        "title": "D8b B1 -- the photographs, part-wise, ARMS and TORSO, "
                 "whole take and window",
        "instrument_only": True,
        "reference": ("MAMMA's SAM2 person masks -- the pixels of the actual footage. The "
                      "one band the candidate cannot optimise."),
        "preregistered": PREREGISTERED,
        "builds": {label: str(delivery.relative_to(ROOT)) for label, delivery, _ in BUILDS},
        "delivered_glb_sha256": glb_digests,
        "masks_copied_never_shared": copies,
        "meshes_exported_fresh": ("both arms, through the real Blender path into this "
                                  "step's own work directories. D9's committed mesh caches "
                                  "are not reused: a cache whose provenance cannot be "
                                  "proved byte for byte is not a saving worth making."),
        "raw_triangulation_byte_identical": raw_identical,
        "smoothed_triangulation_byte_identical": smoothed_identical,
        "denominator_note": (
            "the RAW array is the shared denominator and is ASSERTED; the smoothed array "
            "is REPORTED and is expected to differ, because D8b's reject sits between the "
            "two. A smoothed array that had not moved would mean the step did nothing."),
        "smoothed_expected_to_differ": True,
        "window": {"frame_ids": [WINDOW_FIRST_ID, WINDOW_LAST_ID],
                   "frames": int(window.sum()),
                   "note": ("D8's push-and-fall window, which contains frames "
                            "110-122 where the captured shoulder line collapses. Both cuts "
                            "are BANDED here: a repair that helped the window by hurting "
                            "everything else would pass a window-only band.")},
        "statistics": {"moving_block": BLOCK, "draws": DRAWS, "seed": SEED,
                       "every_arm_on_identical_draws": True,
                       "lag1_autocorrelation_on_this_take": 0.99,
                       "blocks_per_draw_on_the_window": round(int(window.sum()) / BLOCK, 2),
                       "width_warning": "about three blocks per draw on a 41-frame window; "
                                        "the intervals are correspondingly wide and this "
                                        "is stated before the numbers, not after"},
        "subjects": {},
    }

    verdicts: dict = {}
    for s in (0, 1):
        scored = population[:, s, :].all(axis=0)       # frames EVERY camera scored
        # D7b's tilt cut beside D8's window cut. The terciles are computed on the frames
        # every camera scored, from the SHARED landmark array, so they classify the same
        # frames for both arms; they are REPORTED, and the band is the whole take and the
        # window.
        edges = [float(np.percentile(tilt[s][scored], 100 / 3.0)),
                 float(np.percentile(tilt[s][scored], 200 / 3.0))]
        runs = {name: np.zeros(frames, dtype=bool) for name in RUN_CUTS}
        for name, (lo, hi) in RUN_CUTS.items():
            runs[name][lo - FIRST_FRAME_ID:hi - FIRST_FRAME_ID + 1] = True
        pooled = np.zeros(frames, dtype=bool)
        for value in runs.values():
            pooled |= value
        cuts = {"whole_take": scored,
                "three_runs_pooled": scored & pooled,
                **{f"run_{name}": scored & value for name, value in runs.items()},
                "outside_the_three_runs": scored & ~pooled,
                "window": scored & window,
                "upright_tercile": scored & (tilt[s] <= edges[0]),
                "bent_tercile": scored & (tilt[s] > edges[1])}
        row: dict = {
            "frames_every_camera_scored": int(scored.sum()),
            "frames_every_camera_scored_in_window": int((scored & window).sum()),
            "tilt_deg_in_window": [round(float(tilt[s][scored & window].min()), 2),
                                   round(float(tilt[s][scored & window].max()), 2)],
            "tilt_tercile_edges_deg": [round(edges[0], 2), round(edges[1], 2)],
            "cuts": {},
        }
        series: dict = {}
        for part in ("torso", "arm", "whole"):
            series[part] = {n: per_frame_mean(stats[n][f"{part}_iou"][:, s, :], scored)
                            for n in labels}
            series[part]["ORACLE"] = per_frame_mean(
                stats["ORACLE_mamma_mesh"][f"{part}_iou"][:, s, :], scored)
            for name, mask_frames in cuts.items():
                cell = row["cuts"].setdefault(name, {"n": int(mask_frames.sum())})
                for arm_name in labels + ["ORACLE"]:
                    cell[f"{part}_iou_{arm_name}"] = round(
                        float(np.median(series[part][arm_name][mask_frames])), 5)
                cell[f"{part}_D8c_minus_D8b"] = pw.paired(
                    series[part]["D8c"], series[part]["D8b"], mask_frames, draws)
        row["precision_recall_in_window"] = {
            arm_name: {key: round(float(np.nanmedian(
                per_frame_mean(stats[arm_name][key][:, s, :], scored)[cuts["window"]])), 5)
                for key in ("torso_precision", "torso_recall", "arm_precision", "arm_recall")}
            for arm_name in labels}
        report["subjects"][f"subject_{s:02d}"] = row

        cells: dict = {}
        for part in ("arm", "torso"):
            for cut in ("whole_take", "three_runs_pooled") + tuple(
                    f"run_{name}" for name in RUN_CUTS):
                cells[(part, cut)] = row["cuts"][cut][f"{part}_D8c_minus_D8b"]
        not_worse = lambda d: bool(d["ci95"] and d["ci95"][1] >= 0.0)
        rose = lambda d: bool(d["ci95"] and d["ci95"][0] > 0.0)
        subject_verdicts: dict = {}
        cells_n = {key: row["cuts"][key[1]]["n"] for key in cells}
        for (part, cut), cell in cells.items():
            subject_verdicts[f"clause_{part}_{cut}_not_worse_than_D8b"] = {
                "difference": cell["median_difference"], "ci95": cell["ci95"],
                "band": "not worse: the CI's upper bound is at or above zero",
                "verdict": "PASS" if not_worse(cell) else "FAIL",
                "improvement_predicted_here": bool(
                    s == 1 and cut in ("three_runs_pooled", "run_109_119", "run_83_87")),
                "cut_frames": int(cells_n.get((part, cut), 0)),
                "rose_with_ci_clear_of_zero": rose(cell)}
        subject_verdicts["reported_torso_outside_the_three_runs"] = row["cuts"][
            "outside_the_three_runs"]["torso_D8c_minus_D8b"]
        subject_verdicts["reported_arms_outside_the_three_runs"] = row["cuts"][
            "outside_the_three_runs"]["arm_D8c_minus_D8b"]
        subject_verdicts["reported_torso_on_D8s_window"] = row["cuts"][
            "window"]["torso_D8c_minus_D8b"]
        subject_verdicts["reported_whole_person"] = row["cuts"]["whole_take"][
            "whole_D8c_minus_D8b"]
        verdicts[f"subject_{s:02d}"] = subject_verdicts

    # ---- the oracle reads none of our track
    ours = {}
    for c, cam in enumerate(sil.CAMERAS):
        for s in (0, 1):
            values = stats["ORACLE_mamma_mesh"]["whole_iou"][c, s, population[c, s]]
            ours[f"{cam}/subject_{s:02d}"] = float(np.median(values))
    committed = {}
    payload = json.loads(
        (ROOT / "artifacts/compare/d3-skeleton/silhouette-d3.json").read_text())
    for cam in sil.CAMERAS:
        for s in (0, 1):
            committed[f"{cam}/subject_{s:02d}"] = payload["arms"]["ORACLE_mamma_mesh"][cam][
                f"subject_{s:02d}"]["iou"]["median"]
    worst = max(abs(ours[k] - committed[k]) for k in ours)
    verdicts["clause_mamma_mesh_oracle"] = {
        "this_instruments_split_oracle_vs_the_committed_unsplit_one_worst_abs_difference":
            round(worst, 8),
        "why_it_matters": ("the oracle is rasterised here as arm|torso and was rasterised "
                           "whole in the committed D3 run. Agreement to float precision "
                           "says the part split partitions the surface exactly AND that "
                           "the copied mask cache is the same cache."),
        "verdict": "PASS" if worst < 1e-9 else "FAIL",
    }
    clauses = [value["verdict"]
               for s in (0, 1)
               for name, value in verdicts[f"subject_{s:02d}"].items()
               if name.startswith("clause_")]
    report["preregistered_clause_verdicts"] = verdicts
    report["verdict"] = ("PASS" if all(v == "PASS" for v in clauses)
                         and verdicts["clause_mamma_mesh_oracle"]["verdict"] == "PASS"
                         else "FAIL")
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
