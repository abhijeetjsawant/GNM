#!/usr/bin/env python3
"""D7c's B1: the photographs, part-wise, ARMS and TORSO+LEGS. The band the candidate cannot optimise.

INSTRUMENT ONLY. Nothing ships. It is `d9b_hoist_silhouette.py`'s structure with its two arms
rebound -- the shipped D9b build and this step's D7c rebuild -- and it reuses
`silhouette.py`'s rasteriser, scorer and mask store and `silhouette_partwise.py`'s part
split, block draws and paired bootstrap, so both arms go through one pixel path and one
statistic. That file is not edited: it carries D9b's builds, D9b's clause text and its
committed report, which is the record of that pass.

**NOTHING is written under `artifacts/commercial-multiview-soma77/` or under any earlier
step's directory.** The SAM2 mask cache is COPIED into this step's own work directory and
asserted byte-identical to its source. BOTH meshes are exported fresh through the REAL
Blender path into this step's own directories.

THE CUT. D7c's change is WHOLE-TAKE: a pelvis frame moves everything below `Root` on every
frame, and the card says so up front -- bit-identity is not a clause here. So unlike D9b
there is no "frames that moved" cut to take; the hoisted-frame cut is kept only because the
contacts and the hoist MAY move under the new pelvis and the card asks for that to be
reported. The band is the whole take and the bent tercile, both parts, both performers.

THE DENOMINATOR IS BOTH ARRAYS. D7c is a CONVERTER-ONLY change on byte-identical landmarks:
nothing here can move either the raw triangulation or the smoothed one. BOTH are asserted,
and a difference in either would mean the change did more than refit the pelvis.

THE PRE-REGISTERED CLAUSE, verbatim from the D7c card, band B1, registered before this
rebuild existed:

    B1 the photographs: `d7b_silhouette_partwise` D9b vs candidate on identical draws,
    TORSO+LEGS and ARMS, whole take and the bent tercile, both performers, worsening not
    established: the CI's upper bound >= 0 on the D7b/D8 predicate (`ci95[1] >= 0`,
    `d7b_silhouette_partwise.py:327` -- it does NOT establish non-worsening, and the card
    says so); performer 1's lying end (D8c's A-C stretch) is where the silhouettes are
    blind to the depth error and a pass there settles nothing; improvement NOT predicted;
    the MAMMA mesh oracle bit-identical

WHAT THE PREDICATE DOES AND DOES NOT SAY, in the card's own words: `ci95[1] >= 0` establishes
that WORSENING IS NOT ESTABLISHED. It does NOT establish non-worsening. A wide interval
passes it for want of power, and this instrument's power on a quarter-resolution raster is
not large.

BLIND TO: depth; a left/right mirror of a fore-aft symmetric pose; and, within each part,
where inside the outline a joint sits. It is also blind to THE THING THIS STEP CHANGES: a
constant change of pelvis frame is a rotation of the whole trunk about the hip line, and a
mesh can rotate inside its own outline. The card is explicit that no landmark instrument and
no photograph resolves a constant change of frame; this band is here to catch a REGRESSION.
Every figure is read against a mesh bound to an asset whose proportions are not the
performer's, so it scores a CHANGE against a large standing shape mismatch. D7's own
world-vertical control cost -0.218 IoU on this instrument, which is the scale at which it
does have power.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_silhouette.py

Writes `artifacts/compare/d7c-pelvis-rest/silhouette-partwise.json` and caches every per-frame
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
import d9b_hoist_gate as gate  # noqa: E402
from d7c_source_fingerprint import fingerprint_now as source_fingerprint  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7c-pelvis-rest"
D7_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
REPORT = OUT_DIR / "silhouette-partwise.json"
PER_FRAME = OUT_DIR / "silhouette-partwise-per-frame.npz"

BUILDS = (
    ("D9b", ROOT / "artifacts/commercial-multiview-soma77", OUT_DIR / "silhouette-work-d9b"),
    ("D7c", OUT_DIR / "delivery", OUT_DIR / "silhouette-work-d7c"),
)
MASK_WORK = OUT_DIR / "silhouette-work"
MASK_SOURCE = D7_DIR / "silhouette-work"
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260914

PREREGISTERED = {
    "source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D7c card, band B1, "
               "registered before this rebuild existed"),
    "clause": ("the photographs, `d7b_silhouette_partwise`-style: D9b vs candidate on "
               "identical draws, TORSO+LEGS and ARMS, whole take and the bent tercile, both "
               "performers, WORSENING NOT ESTABLISHED (ci95[1] >= 0); improvement is NOT "
               "predicted; the MAMMA mesh oracle bit-identical"),
    "what_the_predicate_does_not_say": (
        "`ci95[1] >= 0` establishes that worsening is NOT ESTABLISHED. It does NOT establish "
        "non-worsening: a wide interval passes it for want of power. The card says so."),
    "performer_1s_lying_end": (
        "D8c's unresolved A-C stretch is where the silhouettes are blind to the depth error, "
        "and a pass there settles nothing."),
    "improvement_is_not_predicted": (
        "the pelvis frame rotates the whole trunk about the hip line by ~9 deg and moves the "
        "root ~12-13 mm. A mesh can rotate inside its own outline, and no photograph "
        "resolves a CONSTANT change of frame -- which is what this step makes. A rise here "
        "would be a surprise and would need its own explanation; a FALL is what this band "
        "exists to catch."),
    "arms_and_torso_both_banded": (
        "a pelvis frame is WHOLE-TAKE: everything below `Root` moves on every frame, arms "
        "included, so neither part is a pure control and bit-identity is not a clause. The "
        "card states that up front."),
    "mamma_mesh_oracle": "BIT-IDENTICAL between runs -- it reads none of our track.",
    "what_this_cannot_settle": (
        "the silhouette scores the MESH, and a joint can move inside its own outline "
        "without moving a pixel. It cannot say whether a re-aimed bone is on its landmark; "
        "that is B1's job, and this is never quoted alone."),
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


def hoisted_masks() -> np.ndarray:
    """[subject, frame] -- the frames the foot-contact hoist moved, from the SHIPPED (D9b) bytes.

    Read through the gate's own reader, so the cut this instrument reports and the frames
    B1 and B2 are stated on are the same frames.
    """
    out = []
    for subject in (0, 1):
        data = gate.read_build(BUILDS[0][1], subject)
        magnitude = 1e3 * np.linalg.norm(data["hoist"], axis=1)
        out.append(magnitude > gate.HOIST_REPORT_CUT_MM)
    return np.stack(out)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-stage", choices=("pre_change", "refactored"), required=True,
                        help="which source stage this run belongs to; stated by the operator "
                             "and checked by the gate against the converter's hash and git")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copies = seed_masks()
    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    shape = (width, height)
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}
    frames = sil.FRAMES

    raw = {label: np.stack([
        np.load(delivery / f"subject-{s:02d}.body-track.npz")[
            "raw_triangulated_world_positions_z_up_m"] for s in (0, 1)])
        for label, delivery, _ in BUILDS}
    smoothed = {label: np.stack([
        np.load(delivery / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"] for s in (0, 1)])
        for label, delivery, _ in BUILDS}
    labels = [label for label, _, _ in BUILDS]
    raw_identical = all(np.array_equal(raw[label], raw["D9b"], equal_nan=True)
                        for label in labels)
    smoothed_identical = all(np.array_equal(smoothed[label], smoothed["D9b"])
                             for label in labels)
    # BOTH are verdicts on this step: a converter-only change cannot move a landmark.
    if not (raw_identical and smoothed_identical):
        raise SystemExit("the builds do not share their triangulated landmarks -- D9b is a "
                         "converter-only change and cannot move either array, so nothing "
                         "below is comparable and the change did more than re-aim")
    cap = raw["D9b"]

    from subject_map import mamma_index_for  # noqa: E402

    subject_to_body = mamma_index_for(smoothed["D9b"])
    tilt_report = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/silhouette-vs-tilt.json").read_text())
    subject_to_tracklet = {
        cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
        for cam, row in tilt_report["identity"]["our_subject_to_mask_tracklet"].items()}

    tilt = np.zeros((2, frames))
    for s in (0, 1):
        pelvis = 0.5 * (cap[s][:, JOINT_INDEX["left_hip"]] + cap[s][:, JOINT_INDEX["right_hip"]])
        up = cap[s][:, JOINT_INDEX["neck"]] - pelvis
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        tilt[s] = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))
    hoisted = hoisted_masks()

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

    # THE CACHE THE READER ACTUALLY LOADED, by name and content. `masks_copied_never_shared`
    # proves every cache in the work directory is a faithful copy; it does not say WHICH one
    # was consumed, and Astra's round 9 deleted the consumed entry from that map with every
    # clause unchanged. `sil.MaskStore` builds its path deterministically from the scale and
    # the camera set (silhouette.py:244), so the same name is computed here and hashed.
    consumed = (MASK_WORK / f"masks-{sil.NATIVE[0] // SCALE}x{sil.NATIVE[1] // SCALE}-"
                f"{'_'.join(cams)}.npz")
    if not consumed.exists():
        raise SystemExit(f"the mask cache the reader loads is missing: {consumed}")

    rng = np.random.default_rng(SEED)
    draws = pw.block_draws(rng, frames)
    # ONE DRAW LIST FOR THE WHOLE TAKE, hashed. Equal `draws_used` counts are consistency,
    # not identity -- round 8's own lesson, applied to itself. This hash is what every cut
    # and every part is then indexed from.
    draw_list_sha256 = sha256(
        np.concatenate([np.asarray(d, dtype=np.int64) for d in draws]).tobytes()).hexdigest()
    report: dict = {
        "title": "D9b B3 -- the photographs, part-wise, ARMS and TORSO+LEGS",
        "instrument_only": True,
        "reference": ("MAMMA's SAM2 person masks -- the pixels of the actual footage. The "
                      "one band the candidate cannot optimise."),
        "preregistered": PREREGISTERED,
        "builds": {label: str(delivery.relative_to(ROOT)) for label, delivery, _ in BUILDS},
        "delivered_glb_sha256": glb_digests,
        "masks_copied_never_shared": copies,
        "mask_cache_consumed": {"name": f"mask/{consumed.name}",
                                "sha256": digest(consumed),
                                "loaded_by": "silhouette.MaskStore at the scale and camera "
                                             "set this run scored"},
        "source_fingerprint": source_fingerprint("E_rig_rest_kabsch", stage=args.src_stage),
        "meshes_exported_fresh": ("both arms, through the real Blender path into this "
                                  "step's own work directories."),
        "raw_triangulation_byte_identical": raw_identical,
        "smoothed_triangulation_byte_identical": smoothed_identical,
        "denominator_note": (
            "BOTH arrays are asserted on this step: a converter-only "
            "change on byte-identical landmarks cannot move either."),
        "statistics": {"moving_block": BLOCK, "draws": DRAWS, "seed": SEED,
                       "every_arm_on_identical_draws": True,
                       "draw_list_sha256": draw_list_sha256,
                       "draw_identity_note": (
                           "ONE list, built once from the seed and indexed by every cut and "
                           "every part. Each cell publishes the sha256 of the draws that "
                           "survived its own cut's five-frame floor, so identity is checkable "
                           "rather than inferred from equal counts."),
                       "lag1_autocorrelation_on_this_take": 0.99},
        "subjects": {},
    }

    verdicts: dict = {}
    for s in (0, 1):
        scored = population[:, s, :].all(axis=0)
        edges = [float(np.percentile(tilt[s][scored], 100 / 3.0)),
                 float(np.percentile(tilt[s][scored], 200 / 3.0))]
        cuts = {"whole_take": scored,
                "bent_tercile": scored & (tilt[s] > edges[1]),
                "upright_tercile": scored & (tilt[s] <= edges[0]),
                "hoisted_frames": scored & hoisted[s],
                "unhoisted_frames": scored & ~hoisted[s]}
        row: dict = {
            "frames_every_camera_scored": int(scored.sum()),
            "frames_hoisted": int(hoisted[s].sum()),
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
                cell[f"{part}_D7c_minus_D9b"] = pw.paired(
                    series[part]["D7c"], series[part]["D9b"], mask_frames, draws)
                # the identity of the draws THIS cell used, not merely how many
                cell[f"{part}_D7c_minus_D9b"]["draws_sha256"] = sha256(
                    np.concatenate([idx[mask_frames[idx]] for idx in draws
                                    if int(mask_frames[idx].sum()) >= 5]
                                   ).astype(np.int64).tobytes()).hexdigest()
        report["subjects"][f"subject_{s:02d}"] = row

        subject_verdicts: dict = {}
        not_worse = lambda d: bool(d["ci95"] and d["ci95"][1] >= 0.0)
        rose = lambda d: bool(d["ci95"] and d["ci95"][0] > 0.0)
        for part in ("arm", "torso"):
            for cut in ("whole_take", "bent_tercile"):
                cell = row["cuts"][cut][f"{part}_D7c_minus_D9b"]
                subject_verdicts[f"clause_{part}_{cut}_worsening_not_established_vs_D9b"] = {
                    "difference": cell["median_difference"], "ci95": cell["ci95"],
                    "band": ("worsening NOT ESTABLISHED: the CI's upper bound is at "
                             "or above zero. This does NOT establish non-worsening."),
                    "verdict": "PASS" if not_worse(cell) else "FAIL",
                    "cut_frames": int(row["cuts"][cut]["n"]),
                    "rose_with_ci_clear_of_zero": rose(cell)}
        for part in ("arm", "torso", "whole"):
            subject_verdicts[f"reported_{part}_on_the_hoisted_frames"] = row["cuts"][
                "hoisted_frames"][f"{part}_D7c_minus_D9b"]
            subject_verdicts[f"reported_{part}_on_the_unhoisted_frames"] = row["cuts"][
                "unhoisted_frames"][f"{part}_D7c_minus_D9b"]
        verdicts[f"subject_{s:02d}"] = subject_verdicts

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
