#!/usr/bin/env python3
"""D7c: what made the three torso cells RISE? A DIAGNOSTIC, and it decides nothing.

B1 passed its clause -- worsening not established -- on all eight cells, but three of them
rose with the interval clear of zero, and IMPROVEMENT WAS NOT PREDICTED. An unexplained rise
is not a result to bank; it is a question. This splits it.

THE SPLIT. A pelvis frame changes two things at once in the delivered file: the ROOT moves
(12.4 / 13.1 mm hoist-subtracted, which translates every skinned vertex rigidly) and the
ARTICULATION changes (the trunk rotates about the hip line, which moves the skin relative to
itself). A silhouette sees both. So a third build is rendered that has ONE of them and not the
other:

    ABLATION = the candidate's LOCAL rotations and the candidate's REST, with D9b's
               PER-FRAME ROOT TRANSLATION

against the shipped D9b build, through the identical pixel path, the identical masks and the
identical frozen draws. Then:

    candidate - D9b   = the translation's share + the articulation's share  (what B1 measured)
    ablation  - D9b   = the articulation's share alone
    candidate - ablation = the translation's share alone

DIAGNOSTIC ONLY. Nothing here is banded, nothing here can change B1's verdict, and an
attribution is not a justification: if the rise turns out to be the root's translation, that
says the masks moved with a rigid shift, not that the pelvis is more right.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_b1_attribution.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts"):
    if str(ROOT / _relative) not in sys.path:
        sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(f"PYTHONPATH trap: {autoanim_gnm.__file__}")

from dataclasses import replace  # noqa: E402

import silhouette as sil  # noqa: E402
import silhouette_partwise as pw  # noqa: E402
import d7_silhouette_partwise as p7  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402
from autoanim_gnm.commercial_multiview import JOINT_INDEX, load_camera_rig  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d7c-pelvis-rest"
D9B = ROOT / "artifacts/commercial-multiview-soma77"
D7C = OUT_DIR / "delivery"
ABLATION = OUT_DIR / "ablation-d9b-root"
REPORT = OUT_DIR / "b1-attribution.json"
SCALE = 4
SEED = 20260914          # the same seed B1 used, so the draws are the same draws


def build_ablation() -> dict:
    """The candidate's locals and rest, with D9b's per-frame root. Exported for real."""
    ABLATION.mkdir(parents=True, exist_ok=True)
    proof = {}
    for subject in (0, 1):
        candidate = d3.load_track(D7C, subject)
        baseline = d3.load_track(D9B, subject)
        # THE CONTACT MASK IS CLEARED, and it has to be. `BodyTrack.__post_init__` runs
        # `validate_body_track`, which refuses a track whose asserted contacts do not hold
        # once the root is swapped ("left foot contact moved 0.00459275 m"). That refusal is
        # the shipping path working -- the same one that made P1's first control unbuildable
        # -- and it says the ablation is NOT a delivery and must never be read as one. The
        # silhouette does not read the contact mask; it rasterises the mesh.
        hybrid = replace(candidate,
                         root_translation_m=np.array(baseline.root_translation_m, copy=True),
                         foot_contacts=np.zeros_like(np.asarray(candidate.foot_contacts)))
        d3.export(hybrid, ABLATION / f"subject-{subject:02d}.glb")
        shutil.copy2(D7C / f"subject-{subject:02d}.body-track.npz",
                     ABLATION / f"subject-{subject:02d}.body-track.npz")
        proof[f"subject_{subject:02d}"] = {
            "locals_are_the_candidates": bool(np.array_equal(
                np.asarray(hybrid.local_rotations_xyzw),
                np.asarray(candidate.local_rotations_xyzw))),
            "rest_is_the_candidates": bool(np.array_equal(
                np.asarray(hybrid.rest_translations_m),
                np.asarray(candidate.rest_translations_m))),
            "root_is_D9bs": bool(np.array_equal(
                np.asarray(hybrid.root_translation_m),
                np.asarray(baseline.root_translation_m))),
            "root_differs_from_the_candidates": bool(not np.array_equal(
                np.asarray(hybrid.root_translation_m),
                np.asarray(candidate.root_translation_m))),
            "contacts_cleared_because_the_validator_refuses_a_swapped_root": True,
        }
    return proof


def main() -> int:
    proof = build_ablation()
    if not all(all(row.values()) for row in proof.values()):
        raise SystemExit(f"the ablation is not what it claims to be: {proof}")

    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}
    frames = sil.FRAMES
    builds = (("D9b", D9B, OUT_DIR / "silhouette-work-d9b"),
              ("D7c", D7C, OUT_DIR / "silhouette-work-d7c"),
              ("ABLATION", ABLATION, OUT_DIR / "silhouette-work-ablation"))

    capture = np.stack([np.load(D9B / f"subject-{s:02d}.body-track.npz")[
        "raw_triangulated_world_positions_z_up_m"] for s in (0, 1)])
    tilt = np.zeros((2, frames))
    for s in (0, 1):
        pelvis = 0.5 * (capture[s][:, JOINT_INDEX["left_hip"]]
                        + capture[s][:, JOINT_INDEX["right_hip"]])
        up = capture[s][:, JOINT_INDEX["neck"]] - pelvis
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        tilt[s] = np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0)))

    saved_delivery, saved_work = sil.DELIVERY, sil.WORK
    sil.WORK = OUT_DIR / "silhouette-work"
    masks = sil.MaskStore(SCALE, cams)
    meshes = {}
    for label, delivery, work in builds:
        work.mkdir(parents=True, exist_ok=True)
        sil.DELIVERY, sil.WORK = delivery, work
        meshes[label] = sil.delivered_mesh()
    sil.DELIVERY, sil.WORK = saved_delivery, saved_work

    split = pw.split_ours()
    faces = split["triangles"]
    torso_faces = faces[~split["face_is_arm"]]
    arm_faces = faces[split["face_is_arm"]]
    tilt_report = json.loads(
        (ROOT / "artifacts/compare/d2-clavicle/silhouette-vs-tilt.json").read_text())
    tracklet = {cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
                for cam, row in tilt_report["identity"]["our_subject_to_mask_tracklet"].items()}

    labels = [b[0] for b in builds]
    iou = {n: {part: np.full((len(cams), 2, frames), np.nan) for part in ("torso", "arm")}
           for n in labels}
    population = np.zeros((len(cams), 2, frames), dtype=bool)
    for c, cam in enumerate(cams):
        for s in (0, 1):
            mask = masks.get(cam, tracklet[cam][s])
            population[c, s] = mask.reshape(frames, -1).sum(axis=1) >= sil.MIN_MASK_PX
            for f in np.nonzero(population[c, s])[0]:
                m = mask[f]
                for n in labels:
                    verts = meshes[n][f"verts_{s:02d}"].astype(np.float32)[f]
                    for part, face_set in (("torso", torso_faces), ("arm", arm_faces)):
                        _, _, value = sil.score(
                            sil.rasterise(verts, face_set, scaled[cam], (width, height)), m)
                        iou[n][part][c, s, f] = value
            print(f"scored {cam} subject {s:02d}")

    rng = np.random.default_rng(SEED)
    draws = pw.block_draws(rng, frames)
    report = {
        "title": "D7c: attribution of B1's three rising torso cells. DIAGNOSTIC ONLY.",
        "what_it_cannot_do": ("change B1's verdict, or turn a rise into a justification. An "
                              "attribution says WHICH of two simultaneous changes moved the "
                              "pixels, not that the pelvis is more right."),
        "ablation": {"definition": ("the candidate's LOCAL rotations and REST, with D9b's "
                                    "PER-FRAME ROOT TRANSLATION"),
                     "verified": proof},
        "identical_draws_and_masks_as_B1": {"seed": SEED, "block": pw.BLOCK
                                            if hasattr(pw, "BLOCK") else sil.BLOCK},
        "subjects": {},
    }
    for s in (0, 1):
        scored = population[:, s, :].all(axis=0)
        edges = float(np.percentile(tilt[s][scored], 200 / 3.0))
        cuts = {"whole_take": scored, "bent_tercile": scored & (tilt[s] > edges)}
        row = {}
        for part in ("torso", "arm"):
            series = {n: p7.per_frame_mean(iou[n][part][:, s, :], scored) for n in labels}
            row[part] = {}
            for cut, mask_frames in cuts.items():
                row[part][cut] = {
                    "candidate_minus_D9b__both_effects": pw.paired(
                        series["D7c"], series["D9b"], mask_frames, draws),
                    "ablation_minus_D9b__the_ARTICULATION_alone": pw.paired(
                        series["ABLATION"], series["D9b"], mask_frames, draws),
                    "candidate_minus_ablation__the_ROOT_TRANSLATION_alone": pw.paired(
                        series["D7c"], series["ABLATION"], mask_frames, draws),
                }
        report["subjects"][f"subject_{s:02d}"] = row
        for part in ("torso", "arm"):
            for cut in cuts:
                cell = row[part][cut]
                print(f"subject {s} {part:5s} {cut:13s} "
                      f"both {cell['candidate_minus_D9b__both_effects']['median_difference']:+8.5f} "
                      f"= articulation {cell['ablation_minus_D9b__the_ARTICULATION_alone']['median_difference']:+8.5f} "
                      f"+ root {cell['candidate_minus_ablation__the_ROOT_TRANSLATION_alone']['median_difference']:+8.5f}")
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
