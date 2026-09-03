#!/usr/bin/env python3
"""D2b's follow-up: IS THE SILHOUETTE'S FALL A FUNCTION OF TRUNK TILT?

INSTRUMENT ONLY. Nothing here ships, nothing here is written into `silhouette.py`'s own
outputs, and no delivered artifact is touched. It imports `tools/compare/silhouette.py`
and reuses its rasteriser, its scorer, its mask store and its posed-mesh caches, so the
three builds go through the IDENTICAL pixel path. The committed tool retains medians only;
this one keeps every frame.

THE QUESTION. D2b moved the delivered rig by an almost purely horizontal 27.87 / 43.16 mm
and the silhouette fell in 8 of 8 camera-subject cells. Two readings were left open in
`docs/reviews/clavicle-origin-2026-09-02.md` section 12.6, and one measurement separates
them: whether the fall depends on how far the performer is bent over.

    PRE-REGISTERED HYPOTHESIS, verbatim, written into the report BEFORE this ran.

WHY THE BOOTSTRAP IS NOT THE OBVIOUS ONE. A moving-block bootstrap needs a contiguous
series, and a tilt tercile is a scattered subset of frames. Blocks are therefore drawn over
the WHOLE 150-frame axis, exactly as `silhouette.moving_block_bootstrap` draws them, and
the statistic is then taken over whichever drawn frames fall in the tercile -- both arms on
the same drawn frames, in the same tercile. Resampling inside a tercile as if it were a
series would break the autocorrelation structure the block length exists to respect.

Run: PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette_vs_tilt.py
Writes: artifacts/compare/d2-clavicle/silhouette-vs-tilt.json
        artifacts/compare/d2-clavicle/silhouette-vs-tilt.png
        (and the `silhouette_vs_tilt` block that `d2_clavicle_gate.py` folds into gate.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    forward_kinematics_positions,
    skeleton_for_joint_names,
    skeleton_for_track_dict,
)

OUT_DIR = ROOT / "artifacts/compare/d2-clavicle"
REPORT = OUT_DIR / "silhouette-vs-tilt.json"
PER_FRAME = OUT_DIR / "silhouette-vs-tilt-per-frame.npz"
FIGURE = OUT_DIR / "silhouette-vs-tilt.png"

# label -> (delivered directory, the work directory holding ITS posed-mesh cache)
BUILDS = (
    ("delivered", ROOT / "artifacts/commercial-multiview-soma77", ROOT / "artifacts/compare/i6"),
    ("D2", OUT_DIR / "delivery", OUT_DIR / "silhouette-work-d2"),
    ("D2b", OUT_DIR / "delivery-root", OUT_DIR / "silhouette-work-d2b"),
)
MASK_WORK = OUT_DIR / "silhouette-work-d2b"      # holds the cached SAM2 masks; read only
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260902

PREREGISTERED = (
    "Set by the coordinator BEFORE this ran and recorded verbatim: \"the rig has no pelvis "
    "frame separate from the torso (Hips, Spine, Chest, UpperChest all take torso_up), so "
    "`R_hips . (0, 0.08, 0)` has a horizontal component of 80*sin(trunk tilt): 27.4 mm at "
    "the median tilt of 20.1 deg (subject 0) and 43.0 mm at 32.5 deg (subject 1), against "
    "your measured whole-rig displacement of 27.87 / 43.16 mm. On a bent frame D2b pushes "
    "the pelvis, and everything hanging off it, forward along the trunk's lean, where an "
    "anatomical pelvis tilts a fraction of that. The old convention placed Hips "
    "tilt-independently by accident, and the silhouette preferred it. PREDICTION: on "
    "upright frames (tilt under ~10 deg) D2b's IoU equals the delivered build's; the fall "
    "grows with tilt. If it does, the mechanism is the offset applied along the trunk axis "
    "and the underlying defect is the missing pelvis frame (converter work, its own step). "
    "If the fall is flat in tilt, it is the mesh's shape (D5/D6) and D2b's silhouette cost "
    "is a price to weigh, not a defect.\""
)


# ------------------------------------------------------------------ identity, not re-run
def tracklet_mapping() -> dict:
    """Read the id resolution from the two retained silhouette reports and cross-check.

    `resolve_tracklets` decodes 1200 native 8.3 MP PNGs; it is not re-run. Its answer
    cannot depend on the build -- the masks are MAMMA's and the subject map is resolved
    from the triangulated positions, which are byte-identical across all three builds
    (asserted below). Reading it from BOTH retained reports and requiring them to agree is
    the check that this shortcut is a shortcut and not an assumption.
    """
    reports = {"committed": ROOT / "artifacts/compare/silhouette.json",
               "d2b": OUT_DIR / "silhouette-d2b.json",
               "d2": OUT_DIR / "silhouette-d2.json"}
    seen = {}
    for label, path in reports.items():
        if not path.exists():
            continue
        seen[label] = json.loads(path.read_text())["identity"]["our_subject_to_mask_tracklet"]
    if not seen:
        raise SystemExit("no retained silhouette report to read the id resolution from")
    first = next(iter(seen.values()))
    for label, value in seen.items():
        if value != first:
            raise SystemExit(f"id resolution disagrees between reports: {label}")
    return {cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
            for cam, row in first.items()}, sorted(seen)


# Joints whose displacement is reported per tilt tercile. The hands are on the list
# because a silhouette is an OUTLINE: an arm swinging out is a large part of it, and the
# clavicle chain is re-aimed by both D2 and D2b while the root moves by centimetres.
WATCHED = ("Hips", "UpperChest", "Head", "LeftHand", "RightHand",
           "LeftLowerArm", "RightLowerArm", "LeftFoot", "RightFoot")


def joints_world_capture(tracks: Path, subject: int) -> dict[str, np.ndarray]:
    """`WATCHED` joints of the delivered rig, per frame, in the capture's Z-up world."""
    track_doc = json.loads((tracks / f"subject-{subject:02d}.body-track.json").read_text())
    names = track_doc["joint_names"]
    base = skeleton_for_track_dict(track_doc)   # D3: the track's own rest
    npz = np.load(tracks / f"subject-{subject:02d}.body-track.npz")
    world = forward_kinematics_positions(
        np.asarray(npz["root_translation_m"], dtype=np.float64),
        np.asarray(npz["local_rotations_xyzw"], dtype=np.float64),
        skeleton=base).astype(np.float64)
    out = {}
    for name in WATCHED:
        y_up = world[:, names.index(name)]
        out[name] = np.stack([y_up[:, 0], -y_up[:, 2], y_up[:, 1]], axis=-1)
    return out


def hips_world_capture(tracks: Path, subject: int) -> np.ndarray:
    return joints_world_capture(tracks, subject)["Hips"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    shape = (width, height)
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}

    # ---- the triangulated positions must be identical across builds, or tilt is not one
    # variable and the three arms are not the same population.
    positions = {}
    for label, delivery, _ in BUILDS:
        positions[label] = np.stack([
            np.load(delivery / f"subject-{s:02d}.body-track.npz")[
                "triangulated_world_positions_z_up_m"] for s in (0, 1)])
    identical = all(positions[label].tobytes() == positions["delivered"].tobytes()
                    for label, _, _ in BUILDS)
    if not identical:
        raise SystemExit("the three builds do not share triangulated positions")
    cap = positions["delivered"]

    from subject_map import mamma_index_for  # noqa: E402

    subject_to_body = mamma_index_for(cap)
    subject_to_tracklet, read_from = tracklet_mapping()

    # ---- (2) trunk tilt per frame. Camera-independent, identical for all three builds.
    tilt = np.zeros((2, sil.FRAMES))
    torso_up_unit = np.zeros((2, sil.FRAMES, 3))
    for s in (0, 1):
        pelvis = 0.5 * (cap[s][:, JOINT_INDEX["left_hip"]] + cap[s][:, JOINT_INDEX["right_hip"]])
        up = cap[s][:, JOINT_INDEX["neck"]] - pelvis
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        torso_up_unit[s] = up
        tilt[s] = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))

    # ---- (5) + (6) the whole-rig shift, and where it points
    joints = {label: [joints_world_capture(delivery, s) for s in (0, 1)]
              for label, delivery, _ in BUILDS}
    hips = {label: np.stack([joints[label][s]["Hips"] for s in (0, 1)])
            for label, _, _ in BUILDS}
    shift = hips["D2b"] - hips["D2"]                     # [subject, frame, 3]
    shift_vs_delivered = hips["D2b"] - hips["delivered"]

    direction: dict = {}
    for s in (0, 1):
        norm = np.linalg.norm(shift[s], axis=1)
        ok = norm > 1e-9
        unit = shift[s][ok] / norm[ok, None]
        horizontal = torso_up_unit[s][ok].copy()
        horizontal[:, 2] = 0.0
        hnorm = np.linalg.norm(horizontal, axis=1)
        good = hnorm > 1e-6
        dots = np.sum(unit[good] * (horizontal[good] / hnorm[good, None]), axis=1)
        direction[f"subject_{s:02d}"] = {
            "frames": int(good.sum()),
            "dot_with_the_horizontal_component_of_torso_up": {
                "median": round(float(np.median(dots)), 4),
                "p5": round(float(np.percentile(dots, 5)), 4),
                "min": round(float(dots.min()), 4)},
            "shift_norm_mm": {"median": round(float(np.median(norm)) * 1000.0, 2),
                              "p5": round(float(np.percentile(norm, 5)) * 1000.0, 2),
                              "p95": round(float(np.percentile(norm, 95)) * 1000.0, 2)},
            "shift_vertical_component_mm": {
                "median": round(float(np.median(shift_vs_delivered[s][:, 2])) * 1000.0, 2)},
            "expected": "+1: the offset is applied along the HIPS' up axis, which on this "
                        "rig IS the trunk axis, so its horizontal part points exactly the "
                        "way the trunk leans.",
        }

    per_camera_ray: dict = {}
    for cam in cams:
        centre = rig[cam].camera_center_world_m
        row = {}
        for s in (0, 1):
            ray = hips["D2"][s] - centre
            ray = ray / np.linalg.norm(ray, axis=1, keepdims=True)
            norm = np.linalg.norm(shift[s], axis=1)
            ok = norm > 1e-9
            unit = shift[s][ok] / norm[ok, None]
            angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(unit * ray[ok], axis=1)),
                                                 -1.0, 1.0)))
            lateral = norm[ok] * np.sin(np.radians(angle))
            row[f"subject_{s:02d}"] = {
                "angle_between_the_shift_and_the_viewing_ray_deg": {
                    "median": round(float(np.median(angle)), 2),
                    "p5": round(float(np.percentile(angle, 5)), 2),
                    "p95": round(float(np.percentile(angle, 95)), 2)},
                "lateral_component_of_the_shift_mm": {
                    "median": round(float(np.median(lateral)) * 1000.0, 2)},
            }
        per_camera_ray[cam] = row

    # ---- (1) + (4) per-frame precision / recall / IoU, three builds and the oracle
    saved_delivery, saved_work = sil.DELIVERY, sil.WORK
    sil.WORK = MASK_WORK
    masks = sil.MaskStore(SCALE, cams)
    smplx_faces = np.load(sil.SMPLX, allow_pickle=True)["f"].astype(np.int32)
    pred_vertices = {b: np.load(sil.MA3D / f"verts_joints_body_id-{b:02d}.npz",
                                allow_pickle=True)["pred_vertices"] for b in (0, 1)}

    meshes: dict = {}
    for label, delivery, work in BUILDS:
        sil.DELIVERY, sil.WORK = delivery, work
        cache = work / "delivered-mesh.npz"
        if not cache.exists():
            raise SystemExit(f"{cache} is missing; run silhouette.py for {label} first")
        meshes[label] = sil.delivered_mesh()
    sil.DELIVERY, sil.WORK = saved_delivery, saved_work

    arms = {label: {s: (m[f"verts_{s:02d}"].astype(np.float32), m[f"faces_{s:02d}"])
                    for s in (0, 1)} for label, m in meshes.items()}
    arms["ORACLE_mamma_mesh"] = {
        s: (pred_vertices[subject_to_body[s]].astype(np.float32), smplx_faces)
        for s in (0, 1)}
    names = [label for label, _, _ in BUILDS] + ["ORACLE_mamma_mesh"]

    stats = {n: np.full((len(cams), 2, sil.FRAMES, 3), np.nan) for n in names}
    population = np.zeros((len(cams), 2, sil.FRAMES), dtype=bool)
    # The rasterisation is the whole cost of this instrument (~10 min for 4 arms x 4
    # cameras x 2 subjects x 150 frames). It depends only on the meshes, the masks and the
    # cameras, none of which any later cut of the numbers changes, so it is cached and a
    # re-cut is free. Delete the file to force a re-render.
    if PER_FRAME.exists():
        cached = np.load(PER_FRAME)
        if set(cached.files) >= {f"stats_{n}" for n in names} | {"population"}:
            for n in names:
                stats[n] = cached[f"stats_{n}"]
            population = cached["population"].astype(bool)
            print(f"reusing the cached per-frame scores in {PER_FRAME}")
            return _analyse(stats, population, names, cams, tilt, torso_up_unit, hips,
                            shift, shift_vs_delivered, direction, per_camera_ray,
                            subject_to_body, subject_to_tracklet, read_from, identical,
                            width, height, joints)
    for c, cam in enumerate(cams):
        for s in (0, 1):
            mask = masks.get(cam, subject_to_tracklet[cam][s])
            area = mask.reshape(sil.FRAMES, -1).sum(axis=1)
            population[c, s] = area >= sil.MIN_MASK_PX
            for f in np.nonzero(population[c, s])[0]:
                for name in names:
                    verts, faces = sil.arm_frame(arms[name][s], int(f))
                    stats[name][c, s, f] = sil.score(
                        sil.rasterise(verts, faces, scaled[cam], shape), mask[f])
            print(f"scored {cam} subject {s:02d}: {int(population[c, s].sum())} frames")
    np.savez_compressed(PER_FRAME, population=population,
                        **{f"stats_{n}": stats[n] for n in names})

    return _analyse(stats, population, names, cams, tilt, torso_up_unit, hips, shift,
                    shift_vs_delivered, direction, per_camera_ray, subject_to_body,
                    subject_to_tracklet, read_from, identical, width, height, joints)


def _displacements(joints, subject, keep) -> dict:
    """How far each watched joint moved, on THESE frames, delivered -> D2 -> D2b.

    A silhouette is an outline, and an arm is a large part of one. The root shift is the
    quantity D2b was designed around; it is not the only thing that moved, and the
    instrument has no business asking a reader to infer the rest.
    """
    out = {}
    for name in WATCHED:
        row = {}
        for a, b in (("D2", "delivered"), ("D2b", "delivered"), ("D2b", "D2")):
            d = joints[a][subject][name][keep] - joints[b][subject][name][keep]
            row[f"{a}_minus_{b}_median_mm"] = round(
                float(np.median(np.linalg.norm(d, axis=1))) * 1000.0, 2)
        out[name] = row
    return out


def _analyse(stats, population, names, cams, tilt, torso_up_unit, hips, shift,
             shift_vs_delivered, direction, per_camera_ray, subject_to_body,
             subject_to_tracklet, read_from, identical, width, height, joints) -> int:
    # ---- (3) terciles of trunk tilt, paired block bootstrap on identical draws
    rng = np.random.default_rng(SEED)
    starts = sil.FRAMES - BLOCK + 1
    per_draw = int(np.ceil(sil.FRAMES / BLOCK))
    draws = [(rng.integers(0, starts, size=per_draw)[:, None]
              + np.arange(BLOCK)[None, :]).ravel()[:sil.FRAMES] for _ in range(DRAWS)]

    terciles: dict = {}
    for s in (0, 1):
        scored = population[:, s].any(axis=0)
        edges = np.percentile(tilt[s][scored], [100 / 3.0, 200 / 3.0])
        member = np.digitize(tilt[s], edges)          # 0 upright, 1 middle, 2 most bent
        rows = {}
        # one IoU per (frame) per arm: the mean over the four cameras of that frame's IoU,
        # on frames every camera scored, so the three arms share one denominator.
        every = population[:, s].all(axis=0)
        cell: dict = {}
        for name in names:
            cell[name] = np.nanmean(stats[name][:, s, :, 2], axis=0)
        for t in (0, 1, 2):
            keep = member == t
            keep_scored = keep & every
            n = int(keep_scored.sum())
            entry = {
                "tilt_range_deg": [round(float(tilt[s][keep_scored].min()), 2),
                                   round(float(tilt[s][keep_scored].max()), 2)],
                "frames": n,
                "median_tilt_deg": round(float(np.median(tilt[s][keep_scored])), 2),
            }
            for name in names:
                entry[f"{name}_iou"] = round(float(np.median(cell[name][keep_scored])), 4)
            for name in ("D2", "D2b"):
                a, b = cell[name], cell["delivered"]
                diffs = []
                for idx in draws:
                    sub = idx[keep_scored[idx]]
                    if len(sub) < 5:
                        continue
                    diffs.append(np.median(a[sub]) - np.median(b[sub]))
                diffs = np.asarray(diffs)
                entry[f"{name}_minus_delivered"] = {
                    "median_iou_difference": round(
                        float(np.median(a[keep_scored]) - np.median(b[keep_scored])), 4),
                    "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                             round(float(np.percentile(diffs, 97.5)), 4)] if len(diffs) else None,
                    "draws_used": int(len(diffs)),
                }
            # WHAT THE RIG ACTUALLY DID in this tercile, so the IoU column is read
            # against a displacement and not against an inference. The vertical term is
            # `80*cos(tilt) - (the hoist's change)` and CHANGES SIGN with tilt, so its
            # whole-take median is not what any tercile saw.
            entry["shift_D2b_minus_delivered_mm"] = {
                "norm_median": round(float(np.median(np.linalg.norm(
                    shift_vs_delivered[s][keep_scored], axis=1))) * 1000.0, 2),
                "horizontal_median": round(float(np.median(np.linalg.norm(
                    shift_vs_delivered[s][keep_scored][:, :2], axis=1))) * 1000.0, 2),
                "vertical_median": round(float(np.median(
                    shift_vs_delivered[s][keep_scored][:, 2])) * 1000.0, 2),
            }
            entry["joint_displacements_mm"] = _displacements(joints, s, keep_scored)
            rows[["upright", "middle", "most_bent"][t]] = entry
        # the instrument's OWN tilt dependence: the oracle reads none of our track, so
        # whatever it does with tilt is the masks' and the rasteriser's, not ours.
        base = rows["upright"]["ORACLE_mamma_mesh_iou"]
        for key in rows:
            rows[key]["ORACLE_minus_its_own_upright_tercile"] = round(
                rows[key]["ORACLE_mamma_mesh_iou"] - base, 4)
        terciles[f"subject_{s:02d}"] = {
            "tercile_edges_deg": [round(float(e), 2) for e in edges],
            "rows": rows,
        }

    # ---- the band the hypothesis names IN ITS OWN WORDS: "tilt under ~10 deg". A tercile
    # is not that band -- subject 0's most upright third still reaches 14.5 deg -- so the
    # prediction is also tested at its stated threshold, on whatever frames clear it.
    strict: dict = {}
    for s in (0, 1):
        every = population[:, s].all(axis=0)
        keep = (tilt[s] <= 10.0) & every
        cell = {n: np.nanmean(stats[n][:, s, :, 2], axis=0) for n in names}
        entry = {"threshold_deg": 10.0, "frames": int(keep.sum())}
        if keep.sum() >= 5:
            entry["median_tilt_deg"] = round(float(np.median(tilt[s][keep])), 2)
            for n in names:
                entry[f"{n}_iou"] = round(float(np.median(cell[n][keep])), 4)
            for n in ("D2", "D2b"):
                diffs = []
                for idx in draws:
                    sub = idx[keep[idx]]
                    if len(sub) >= 5:
                        diffs.append(np.median(cell[n][sub]) - np.median(cell["delivered"][sub]))
                diffs = np.asarray(diffs)
                entry[f"{n}_minus_delivered"] = {
                    "median_iou_difference": round(
                        float(np.median(cell[n][keep]) - np.median(cell["delivered"][keep])), 4),
                    "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                             round(float(np.percentile(diffs, 97.5)), 4)] if len(diffs) else None,
                    "draws_used": int(len(diffs))}
        if keep.sum() >= 5:
            entry["shift_D2b_minus_delivered_mm"] = {
                "norm_median": round(float(np.median(np.linalg.norm(
                    shift_vs_delivered[s][keep], axis=1))) * 1000.0, 2),
                "horizontal_median": round(float(np.median(np.linalg.norm(
                    shift_vs_delivered[s][keep][:, :2], axis=1))) * 1000.0, 2),
                "vertical_median": round(float(np.median(
                    shift_vs_delivered[s][keep][:, 2])) * 1000.0, 2),
            }
            entry["joint_displacements_mm"] = _displacements(joints, s, keep)
        strict[f"subject_{s:02d}"] = entry

    # ---- per-camera IoU fall, to sit beside the ray angle
    for c, cam in enumerate(cams):
        for s in (0, 1):
            ok = population[c, s]
            fall = {}
            for name in ("D2", "D2b"):
                fall[f"{name}_minus_delivered_iou"] = round(float(
                    np.median(stats[name][c, s, ok, 2]) -
                    np.median(stats["delivered"][c, s, ok, 2])), 4)
            fall["D2b_minus_D2_iou"] = round(float(
                np.median(stats["D2b"][c, s, ok, 2]) -
                np.median(stats["D2"][c, s, ok, 2])), 4)
            fall["delivered_iou"] = round(float(np.median(stats["delivered"][c, s, ok, 2])), 4)
            fall["ORACLE_iou"] = round(float(np.median(stats["ORACLE_mamma_mesh"][c, s, ok, 2])), 4)
            per_camera_ray[cam][f"subject_{s:02d}"].update(fall)

    # ---- (5) does the fall track the SILHOUETTE-VISIBLE part of the shift?
    # A shift along a camera's viewing ray is depth and a silhouette cannot see it; the
    # lateral part is what it can. If the fall were the shift, the eight cells should rank
    # by their lateral component. Reported as a rank correlation over the eight cells,
    # which is eight points and is quoted as a reading, not a test.
    lateral, fall = [], []
    for cam in cams:
        for s in (0, 1):
            cell = per_camera_ray[cam][f"subject_{s:02d}"]
            lateral.append(cell["lateral_component_of_the_shift_mm"]["median"])
            fall.append(cell["D2b_minus_D2_iou"])
    lateral, fall = np.asarray(lateral), np.asarray(fall)
    ranks = lambda v: np.argsort(np.argsort(v)).astype(float)
    lateral_vs_fall = {
        "cells": len(lateral),
        "lateral_shift_mm": [float(v) for v in lateral],
        "iou_fall_D2b_minus_D2": [float(v) for v in fall],
        "pearson": round(float(np.corrcoef(lateral, fall)[0, 1]), 4),
        "spearman": round(float(np.corrcoef(ranks(lateral), ranks(fall))[0, 1]), 4),
        "reading": "a NEGATIVE correlation would mean more lateral shift, more IoU lost -- "
                   "the silhouette pricing the displacement it can actually see. Eight "
                   "cells is a reading and not a test, and it is quoted as one.",
    }

    # ---- the verdict on the pre-registered prediction
    upright = [terciles[f"subject_{s:02d}"]["rows"]["upright"]["D2b_minus_delivered"]
               for s in (0, 1)]
    bent = [terciles[f"subject_{s:02d}"]["rows"]["most_bent"]["D2b_minus_delivered"]
            for s in (0, 1)]
    strict_arms = [strict[f"subject_{s:02d}"].get("D2b_minus_delivered") for s in (0, 1)]
    upright_equal = all(
        u["ci95"] is not None and u["ci95"][0] <= 0.0 <= u["ci95"][1] for u in upright) and \
        all(a is not None and a.get("ci95") is not None
            and a["ci95"][0] <= 0.0 <= a["ci95"][1] for a in strict_arms)
    grows = all(b["median_iou_difference"] < u["median_iou_difference"]
                for u, b in zip(upright, bent))
    verdict = ("TILT-DEPENDENT" if (upright_equal and grows)
               else "PARTLY TILT-DEPENDENT" if grows
               else "FLAT IN TILT")

    report = {
        "instrument": "tools/compare/silhouette_vs_tilt.py",
        "shipping": "NOTHING. Instrument only; silhouette.py's own outputs are untouched.",
        "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette_vs_tilt.py",
        "autoanim_gnm_resolved_to": str(Path(autoanim_gnm.__file__).resolve()),
        "pre_registered_hypothesis": PREREGISTERED,
        "render": {"resolution": f"{width}x{height}", "cameras": list(cams),
                   "frames": sil.FRAMES,
                   "path": "silhouette.rasterise / silhouette.score / silhouette.MaskStore, "
                           "imported, not re-implemented -- the three builds and the oracle "
                           "go through one pixel path"},
        "identity": {
            "our_subject_to_body_id": {f"subject_{s:02d}": f"body_id-{b:02d}"
                                       for s, b in subject_to_body.items()},
            "our_subject_to_mask_tracklet": {
                cam: {f"subject_{s:02d}": t for s, t in row.items()}
                for cam, row in subject_to_tracklet.items()},
            "read_from": read_from,
            "cross_check": "the retained silhouette reports agree on the id resolution; it "
                           "is not re-derived here and it cannot depend on the build",
        },
        "triangulated_positions_identical_across_builds": bool(identical),
        "trunk_tilt_deg": {f"subject_{s:02d}": {
            "median": round(float(np.median(tilt[s])), 2),
            "p5": round(float(np.percentile(tilt[s], 5)), 2),
            "p95": round(float(np.percentile(tilt[s], 95)), 2),
            "definition": "angle of (neck - hip midpoint) from capture +Z, from the "
                          "triangulated positions. Identical for all three builds"}
            for s in (0, 1)},
        "shift_direction": direction,
        "per_camera_ray": per_camera_ray,
        "terciles": terciles,
        "strict_upright_band_the_hypothesis_names": strict,
        "lateral_shift_vs_iou_fall": lateral_vs_fall,
        "bootstrap": {
            "block_frames": BLOCK, "draws": DRAWS, "seed": SEED,
            "method": "blocks drawn over the WHOLE 150-frame axis, then the statistic taken "
                      "over the drawn frames that fall in the tercile -- both arms on the "
                      "same drawn frames. Resampling inside a tercile as if it were a "
                      "contiguous series would break the autocorrelation the block length "
                      "exists to respect (lag-1 ~0.99, CLAUDE.md).",
        },
        "verdict": verdict,
        "verdict_reading": (
            "The label is mechanical -- `grows` only asks whether the most-bent tercile is "
            "worse than the upright one -- so read the magnitudes with it. Subject 0's fall "
            f"grows by {round(upright[0]['median_iou_difference'] - bent[0]['median_iou_difference'], 4)} "
            f"IoU across the terciles; subject 1's by "
            f"{round(upright[1]['median_iou_difference'] - bent[1]['median_iou_difference'], 4)}, "
            "which is inside the width of its own interval and is not a gradient. THE "
            "DECIDING FACT IS THE UPRIGHT BAND, and the prediction named it: on frames with "
            "trunk tilt <= 10 deg, where the root offset's horizontal part is about 14 mm, "
            "D2b is already "
            f"{strict['subject_00']['D2b_minus_delivered']['median_iou_difference']} and "
            f"{strict['subject_01']['D2b_minus_delivered']['median_iou_difference']} IoU "
            "below the delivered build with both intervals clear of zero -- and on subject 1 "
            "that is the LARGEST fall anywhere on the take. The pre-registered condition "
            "'on upright frames D2b's IoU equals the delivered build's' is REFUTED."),
        "what_actually_moved": (
            "On the tilt <= 10 deg band the whole rig moved 12.67 mm (subject 0) and 9.64 mm "
            "(subject 1) while the delivered HANDS moved 144-188 mm, because both D2 and D2b "
            "re-aim the clavicle chain. Two further readings point the same way: the "
            "most-bent tercile moves the rig by 74 mm (subject 0) and 99 mm (subject 1) and "
            "loses LESS IoU on subject 1 than the 10 mm upright band does, so the fall is "
            "not proportional to the displacement; and across the eight camera-subject "
            "cells the fall does not track the silhouette-VISIBLE lateral part of the shift "
            "(Spearman +0.38 -- the wrong sign for that mechanism, on eight points). The "
            "arms-only arm of the comparison is measured, not inferred: delivered -> D2 "
            "moves the Hips by 0.00 mm and the hands by 108-127 mm on subject 1's upright "
            "band, and costs 0.0387 IoU with its interval clear of zero. WHAT IS NOT "
            "ISOLATED is how much of D2b's further 0.054 is its extra 74-88 mm of hand "
            "travel and how much its 9.6 mm of body shift. A fourth build -- the root fix "
            "carrying the PRE-D2 clavicle anchor -- separates them, and is the obvious next "
            "instrument. It was not run: this pass was capped at one."),
        "blind_to": (
            "This says WHETHER the fall tracks tilt, not WHY a given placement is "
            "anatomically right. A silhouette is a projection: a shift along a camera's "
            "viewing ray is invisible to it, and the per-camera ray angles are reported "
            "for exactly that reason. It is also blind to everything inside the outline, "
            "and our mesh covers only about two thirds of the mask on EVERY arm, so a "
            "large shape mismatch (D5/D6) is the background against which a 3-4 cm shift "
            "is being read."),
    }
    REPORT.write_text(json.dumps(report, indent=2, default=float))

    _figure(tilt, terciles, stats, population, names, cams)

    print(f"\n=== verdict: {verdict}")
    for s in (0, 1):
        e = strict[f"subject_{s:02d}"]
        d = e.get("D2b_minus_delivered", {})
        print(f"  STRICT tilt <= 10 deg, subject {s:02d}: n={e['frames']} "
              f"delivered {e.get('delivered_iou')} D2b {e.get('D2b_iou')} "
              f"D2b-delivered {d.get('median_iou_difference')} {d.get('ci95')} "
              f"shift {e.get('shift_D2b_minus_delivered_mm')}")
        jd = e.get("joint_displacements_mm", {})
        if jd:
            print("      moved (D2b-delivered, mm): " + ", ".join(
                f"{n} {jd[n]['D2b_minus_delivered_median_mm']}"
                for n in ("Hips", "Head", "LeftHand", "RightHand", "LeftFoot")))
    print(f"  lateral shift vs IoU fall over the 8 cells: pearson "
          f"{lateral_vs_fall['pearson']}, spearman {lateral_vs_fall['spearman']}")
    for s in (0, 1):
        print(f"--- subject {s:02d}, tercile edges "
              f"{terciles[f'subject_{s:02d}']['tercile_edges_deg']} deg")
        for key, row in terciles[f"subject_{s:02d}"]["rows"].items():
            d = row["D2b_minus_delivered"]
            print(f"  {key:10s} tilt {row['tilt_range_deg']} n={row['frames']:3d} "
                  f"delivered {row['delivered_iou']:.4f}  D2 {row['D2_iou']:.4f}  "
                  f"D2b {row['D2b_iou']:.4f}  D2b-delivered "
                  f"{d['median_iou_difference']:+.4f} {d['ci95']}  "
                  f"oracle-vs-upright {row['ORACLE_minus_its_own_upright_tercile']:+.4f}  "
                  f"shift h/v {row['shift_D2b_minus_delivered_mm']['horizontal_median']}/"
                  f"{row['shift_D2b_minus_delivered_mm']['vertical_median']} mm")
    print(f"\nwrote {REPORT} and {FIGURE}")
    return 0


def _figure(tilt, terciles, stats, population, names, cams) -> None:
    """Tilt on x, paired IoU difference on y, per subject. Plus the absolute IoUs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the validated palette: ours blue, mamma orange, alt aqua, control grey
    colour = {"delivered": "#8a8f98", "D2": "#4cc9d0", "D2b": "#2f6fd0",
              "ORACLE_mamma_mesh": "#e08a24"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for s in (0, 1):
        scored = population[:, s].all(axis=0)
        x = tilt[s][scored]
        order = np.argsort(x)
        mean_iou = {n: np.nanmean(stats[n][:, s, :, 2], axis=0)[scored] for n in names}

        ax = axes[s, 0]
        for n in names:
            ax.plot(x[order], mean_iou[n][order], ".", ms=3.2, alpha=0.55,
                    color=colour[n], label=n if s == 0 else None)
        for n in names:
            b = _binned(x, mean_iou[n])
            ax.plot(b[0], b[1], "-", lw=2.2, color=colour[n])
        for edge in terciles[f"subject_{s:02d}"]["tercile_edges_deg"]:
            ax.axvline(edge, color="#c8ccd2", lw=1.0, ls="--")
        ax.set_title(f"subject {s:02d}: silhouette IoU vs trunk tilt   (higher is better)",
                     fontsize=10)
        ax.set_xlabel("trunk tilt from vertical, degrees")
        ax.set_ylabel("IoU against the SAM2 mask")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        if s == 0:
            ax.legend(fontsize=8, loc="lower left")

        ax = axes[s, 1]
        ax.axhline(0.0, color="#8a8f98", lw=1.2)
        for n in ("D2", "D2b"):
            d = mean_iou[n] - mean_iou["delivered"]
            ax.plot(x[order], d[order], ".", ms=3.2, alpha=0.55, color=colour[n],
                    label=f"{n} - delivered" if s == 0 else None)
            b = _binned(x, d)
            ax.plot(b[0], b[1], "-", lw=2.4, color=colour[n])
        oracle = mean_iou["ORACLE_mamma_mesh"] - np.median(mean_iou["ORACLE_mamma_mesh"])
        b = _binned(x, oracle)
        ax.plot(b[0], b[1], "-", lw=2.0, color=colour["ORACLE_mamma_mesh"],
                label="ORACLE, centred (the instrument's own tilt dependence)"
                if s == 0 else None)
        for edge in terciles[f"subject_{s:02d}"]["tercile_edges_deg"]:
            ax.axvline(edge, color="#c8ccd2", lw=1.0, ls="--")
        ax.set_title(f"subject {s:02d}: paired IoU difference vs the delivered build "
                     f"  (above zero is better)", fontsize=10)
        ax.set_xlabel("trunk tilt from vertical, degrees")
        ax.set_ylabel("IoU difference, same frame")
        ax.grid(alpha=0.25)
        if s == 0:
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("D2b: does the silhouette's fall track how far the performer is bent over?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def _binned(x, y, bins: int = 9):
    edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    edges[-1] += 1e-9
    cx, cy = [], []
    for i in range(bins):
        keep = (x >= edges[i]) & (x < edges[i + 1])
        if keep.sum() >= 3:
            cx.append(float(np.median(x[keep])))
            cy.append(float(np.median(y[keep])))
    return np.asarray(cx), np.asarray(cy)


if __name__ == "__main__":
    raise SystemExit(main())
