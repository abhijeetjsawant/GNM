#!/usr/bin/env python3
"""D8's B1: the photographs, part-wise, on the push-and-fall WINDOW. The band the candidate
cannot optimise.

INSTRUMENT ONLY. Nothing ships. It is `d7b_silhouette_partwise.py`'s structure with TWO
arms -- the shipped D7b build and this step's D8 rebuild -- and it reuses `silhouette.py`'s
rasteriser, scorer and mask store and `silhouette_partwise.py`'s part split, block draws
and paired bootstrap, so both arms go through one pixel path and one statistic.

WHY A NEW FILE AND NOT A PARAMETER. `d7b_silhouette_partwise.py` carries D7b's three
builds, D7b's pre-registered clause text and its committed report, which is the record of
that pass. Parameterising it would rewrite a trusted instrument to serve a different
question -- the reason D7b did not edit D7's file and D7 did not edit `silhouette_partwise`.

**NOTHING is written under `artifacts/commercial-multiview-soma77/`, `artifacts/compare/i6/`,
`artifacts/compare/d7-pelvis-frame/` or `artifacts/compare/d7b-trunk/`.** The SAM2 mask
cache is COPIED into this step's own work directory and asserted byte-identical to its
source. BOTH meshes are exported fresh through the REAL Blender path into this step's own
directories -- D7b's committed mesh cache is NOT reused, because it was built from that
branch's own `delivery/` and the shipped delivery has since been rebuilt in place; a cache
whose provenance cannot be proved byte-for-byte is not a saving worth making.

THE ONE STRUCTURAL DIFFERENCE FROM EVERY EARLIER SILHOUETTE PASS
----------------------------------------------------------------
Those instruments asserted the arms' `triangulated_world_positions_z_up_m` byte-identical,
because a converter change cannot move a landmark. **D8 moves that array by construction**
-- the conditioning gate, the reachability reject and the gap clause all act between the
raw triangulation and it. So the shared denominator here is
`raw_triangulated_world_positions_z_up_m`, which D8 does not touch, and it is asserted
instead. The trunk tilt reported beside the cut is computed from the RAW array for the same
reason: it must mean one thing for both arms.

THE CUT IS THE WINDOW, NOT THE TILT TERCILES. D7b cut by trunk tilt because a pelvis frame
does its damage on bent frames. D8's defect is an OCCLUSION, and it lives in frame ids
85-125 -- 41 frames, of which the ones every camera's mask scored are the population here.
Both the window and the whole take are reported; the window is the band.

THE PRE-REGISTERED CLAUSE, verbatim from `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8
card, band B1, written before this rebuild existed:

    (B1) part-wise silhouette, arms, on the window frames vs D7b, not worse on either
    performer with the CI clear and predicted to rise on performer 1

BLIND TO: depth; a left/right mirror of a fore-aft symmetric pose; and, within each part,
where inside the outline a limb sits -- which matters more here than it did for D7b,
because a limb whose depth is wrong along the A-C axis can stay inside the outline of a
camera that looks along that axis. A folded-arm control does not realise "a limb hidden
inside the outline" under four cameras (CLAUDE.md), so this band is paired with B2's
held-out camera and B4's raw-referenced placement rather than quoted alone. Every figure is
read against a mesh bound to an asset whose proportions are not the performer's, so it
scores a CHANGE against a large standing shape mismatch, and it cannot see the mesh
distortion the post-D7 review found. Precision and recall are reported beside every IoU.

**A 41-frame window with a block of 15 gives about three blocks per draw.** The intervals
are correspondingly wide and that is stated rather than discovered later; D7b's bent
tercile had the same problem on ~50 frames and only 6 of 12 of its cells carried an
interval clear of zero.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8_occlusion_silhouette.py

Writes `artifacts/compare/d8-occlusion/silhouette-partwise.json` and caches every per-frame
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

OUT_DIR = ROOT / "artifacts/compare/d8-occlusion"
D7_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
REPORT = OUT_DIR / "silhouette-partwise.json"
PER_FRAME = OUT_DIR / "silhouette-partwise-per-frame.npz"

# label -> (delivery, this step's OWN work directory). Both meshes are exported fresh.
BUILDS = (
    ("D7b", ROOT / "artifacts/commercial-multiview-soma77", OUT_DIR / "silhouette-work-d7b"),
    ("D8", OUT_DIR / "delivery", OUT_DIR / "silhouette-work-d8"),
)
MASK_WORK = OUT_DIR / "silhouette-work"
MASK_SOURCE = D7_DIR / "silhouette-work"
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260905

# The card's push-and-fall window, in the observations' own absolute frame ids, and the
# array indices they correspond to. The take's own frame ids run 60..209.
FIRST_FRAME_ID = 60
WINDOW_FIRST_ID, WINDOW_LAST_ID = 85, 125

PREREGISTERED = {
    "source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D8 card, band B1, committed "
               "before this rebuild existed"),
    "arms_on_the_window_not_worse_than_D7b": "on EITHER performer, with the CI clear.",
    "predicted_to_rise_on_performer_1": ("that is the performer whose captured forearm is "
                                         "off its own median on 18 frames and whose "
                                         "delivered left hand is 276 mm p95."),
    "mamma_mesh_oracle": "BIT-IDENTICAL between runs -- it reads none of our track.",
    "what_this_cannot_settle": (
        "the silhouette scores the MESH, and a limb whose depth is wrong ALONG the A-C "
        "axis can sit inside the outline of a camera looking down that axis. This band is "
        "paired with B2's held-out camera and B4's raw-referenced placement and is never "
        "quoted alone."),
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
    raw_identical = all(np.array_equal(raw[label], raw["D7b"], equal_nan=True)
                        for label in labels)
    if not raw_identical:
        raise SystemExit("the builds do not share their RAW triangulation -- D8's one "
                         "unchanged reference has moved and nothing below is comparable")
    smoothed_identical = all(np.array_equal(smoothed[label], smoothed["D7b"])
                             for label in labels)
    cap = raw["D7b"]

    from subject_map import mamma_index_for  # noqa: E402

    subject_to_body = mamma_index_for(smoothed["D7b"])
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
        "title": "D8 B1 -- the photographs, part-wise, on the push-and-fall window",
        "instrument_only": True,
        "reference": ("MAMMA's SAM2 person masks -- the pixels of the actual footage. The "
                      "one band the candidate cannot optimise."),
        "preregistered": PREREGISTERED,
        "builds": {label: str(delivery.relative_to(ROOT)) for label, delivery, _ in BUILDS},
        "delivered_glb_sha256": glb_digests,
        "masks_copied_never_shared": copies,
        "meshes_exported_fresh": ("both arms, through the real Blender path into this "
                                  "step's own work directories. D7b's committed mesh cache "
                                  "is not reused: it was built from that branch's own "
                                  "delivery/ and the shipped delivery has been rebuilt in "
                                  "place since."),
        "raw_triangulation_byte_identical": raw_identical,
        "smoothed_triangulation_byte_identical": smoothed_identical,
        "denominator_note": (
            "the shared denominator is the RAW array, not the smoothed one. D8 moves the "
            "smoothed landmarks by construction -- that is the step -- so "
            "`smoothed_triangulation_byte_identical` is EXPECTED to be false and is "
            "reported rather than asserted."),
        "window": {"frame_ids": [WINDOW_FIRST_ID, WINDOW_LAST_ID],
                   "frames": int(window.sum()),
                   "note": "the push-and-fall window, where B001 and D001 lose a performer"},
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
        cuts = {"window": scored & window, "whole_take": scored,
                "outside_window": scored & ~window}
        row: dict = {
            "frames_every_camera_scored": int(scored.sum()),
            "frames_every_camera_scored_in_window": int((scored & window).sum()),
            "tilt_deg_in_window": [round(float(tilt[s][scored & window].min()), 2),
                                   round(float(tilt[s][scored & window].max()), 2)],
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
                cell[f"{part}_D8_minus_D7b"] = pw.paired(
                    series[part]["D8"], series[part]["D7b"], mask_frames, draws)
        row["precision_recall_in_window"] = {
            arm_name: {key: round(float(np.nanmedian(
                per_frame_mean(stats[arm_name][key][:, s, :], scored)[cuts["window"]])), 5)
                for key in ("torso_precision", "torso_recall", "arm_precision", "arm_recall")}
            for arm_name in labels}
        report["subjects"][f"subject_{s:02d}"] = row

        arm_window = row["cuts"]["window"]["arm_D8_minus_D7b"]
        not_worse = lambda d: bool(d["ci95"] and d["ci95"][1] >= 0.0)
        verdicts[f"subject_{s:02d}"] = {
            "clause_1_arms_on_the_window_not_worse_than_D7b": {
                "difference": arm_window["median_difference"], "ci95": arm_window["ci95"],
                "band": "not worse: the CI's upper bound is at or above zero",
                "verdict": "PASS" if not_worse(arm_window) else "FAIL",
                "rise_predicted_here": s == 1,
                "rose_with_ci_clear_of_zero": bool(
                    arm_window["ci95"] and arm_window["ci95"][0] > 0.0)},
            "reported_torso_on_the_window": row["cuts"]["window"]["torso_D8_minus_D7b"],
            "reported_arms_outside_the_window": row["cuts"]["outside_window"][
                "arm_D8_minus_D7b"],
        }

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
    verdicts["clause_2_mamma_mesh_oracle"] = {
        "this_instruments_split_oracle_vs_the_committed_unsplit_one_worst_abs_difference":
            round(worst, 8),
        "why_it_matters": ("the oracle is rasterised here as arm|torso and was rasterised "
                           "whole in the committed D3 run. Agreement to float precision "
                           "says the part split partitions the surface exactly AND that "
                           "the copied mask cache is the same cache."),
        "verdict": "PASS" if worst < 1e-9 else "FAIL",
    }
    clauses = [verdicts[f"subject_{s:02d}"][
        "clause_1_arms_on_the_window_not_worse_than_D7b"]["verdict"] for s in (0, 1)]
    report["preregistered_clause_verdicts"] = verdicts
    report["verdict"] = ("PASS" if all(v == "PASS" for v in clauses)
                         and verdicts["clause_2_mamma_mesh_oracle"]["verdict"] == "PASS"
                         else "FAIL")
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
