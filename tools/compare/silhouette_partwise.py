#!/usr/bin/env python3
"""D2b, third pass: WHICH PART OF THE BODY is the silhouette's fall in?

INSTRUMENT ONLY. Nothing ships. `silhouette.py`'s own outputs are untouched; its
rasteriser, scorer, mask store and posed-mesh caches are imported and reused so every arm
goes through one pixel path.

THE READING BEING TESTED, pre-registered verbatim in the report before this ran. Section
13 established that the silhouette's fall is present at full size on frames where the whole
rig has moved 10 mm and the delivered hands have moved 144-188 mm. This pass asks the
question that separates: split the mesh and score the parts.

TWO THINGS THIS CAN FIND THAT NOTHING ELSE HERE CAN.

  * The TORSO+LEGS-only IoU isolates D2b's own cost. Those vertices carry no clavicle-chain
    weight, so between `delivered` and `D2` they can only be identical -- D2 moves no root
    -- and between `D2` and `D2b` they move by exactly the root shift and nothing else.
    Whatever that costs IS D2b's silhouette cost, with the arms taken out of the question.
  * The ARM-HIDDEN-INSIDE-THE-BODY fraction is the degenerate a person-mask gate cannot
    see. A mask is one blob: an arm folded across the chest is inside the outline and
    scores as intersection whatever it is doing. If the delivered build was hiding its arms
    inside the torso and D2 pulled them out onto a rig 190 mm too wide at the shoulders,
    the mask would report that as a loss.

Run:  PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette_partwise.py
Writes: artifacts/compare/d2-clavicle/silhouette-partwise.json
        artifacts/compare/d2-clavicle/silhouette-partwise.png
        artifacts/compare/d2-clavicle/folded-arms/    (the control build, instrument only)
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
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
from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    forward_kinematics_positions,
    skeleton_for_joint_names,
)
from autoanim_gnm.body_export import export_animated_body_glb  # noqa: E402
from autoanim_gnm.commercial_multiview import load_camera_rig  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d2-clavicle"
REPORT = OUT_DIR / "silhouette-partwise.json"
FIGURE = OUT_DIR / "silhouette-partwise.png"
PER_FRAME = OUT_DIR / "silhouette-partwise-per-frame.npz"
FOLDED = OUT_DIR / "folded-arms"
FOLDED_MESH = OUT_DIR / "folded-arms-mesh.npz"
BODY_RUN = ROOT / ".cache/autoanim_gnm/body-provider/run/detailed-hands-fbd9784b"
TRACKS = ROOT / "artifacts/commercial-multiview-soma77"
MASK_WORK = OUT_DIR / "silhouette-work-d2b"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
SCALE = 4
BLOCK, DRAWS, SEED = sil.BLOCK, sil.RESAMPLES, 20260902

BUILDS = (
    ("delivered", TRACKS, ROOT / "artifacts/compare/i6"),
    ("D2", OUT_DIR / "delivery", OUT_DIR / "silhouette-work-d2"),
    ("D2b", OUT_DIR / "delivery-root", OUT_DIR / "silhouette-work-d2b"),
)

# Our rig: the clavicle chain and everything below it.
ARM_PREFIXES = ("LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
                "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
                "LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing", "LeftLittle",
                "RightThumb", "RightIndex", "RightMiddle", "RightRing", "RightLittle")
# SMPL-X: collars 13/14, shoulders 16/17, elbows 18/19, wrists 20/21, fingers 25..54.
# The collars are included so the two meshes are cut at the SAME anatomical place; the
# choice is recorded because it moves the shoulder cap between the two parts.
SMPLX_ARM_JOINTS = frozenset({13, 14, 16, 17, 18, 19, 20, 21} | set(range(25, 55)))

PREREGISTERED = {
    "the_reading": (
        "Set by the coordinator BEFORE this ran, verbatim: \"a person mask is blind to "
        "where INSIDE the outline a limb sits. An arm placed inside the body's silhouette "
        "counts as intersection whatever it is doing; a correctly aimed arm on a rig "
        "190 mm too wide at the shoulders (540 vs ~350 measured) lands outside the real "
        "arm and loses precision and recall. That is the only mechanism that makes all "
        "three facts true at once: joints closer to our landmarks (181 -> 51), joints "
        "closer to MAMMA's (152 -> 71), and mask overlap falling.\""),
    "1a_torso_legs_only_iou": (
        "PRE-REGISTERED: unchanged delivered -> D2 (the root is identical and only the "
        "clavicle chain moved), and moving D2 -> D2b by whatever the 10-43 mm body shift "
        "costs. THAT number is D2b's own silhouette cost, isolated. If it falls by more "
        "than a few thousandths with the interval clear of zero, the root has a cost of "
        "its own that no joint instrument sees, and this report will say so plainly."),
    "1b_arms_only_precision": (
        "PRE-REGISTERED: falls delivered -> D2 -> D2b."),
    "1c_arm_pixels_hidden_inside_the_bodys_own_torso_raster": (
        "PRE-REGISTERED: falls delivered -> D2. A mask is one blob, so an arm inside the "
        "torso's own outline is free; pulling it out onto a rig that is too wide is not."),
    "2_folded_arm_control": (
        "PRE-REGISTERED: with both arms folded across the torso, whole-person IoU and "
        "precision go UP against the delivered build. A limb-collapsed body passing a "
        "person-mask gate is the 'no gate a constant can pass' finding for I6."),
}


# --------------------------------------------------------------------- the part split
def split_ours() -> dict:
    """Face partition of the delivered mesh, by DOMINANT SKIN WEIGHT.

    Read from the body asset the delivery was built from, not guessed from geometry. The
    asset carries 13380 vertices and 26756 triangles and the Blender export returns exactly
    those counts, which is what licenses using the asset's per-vertex weights to label the
    exported vertices; `nearest_joint_agreement` below is the independent check that the
    order really is preserved, and it needs no correspondence at all.
    """
    with np.load(BODY_RUN / "neutral-body.npz", allow_pickle=False) as archive:
        names = [str(v) for v in archive["joint_names"].tolist()]
        indices = archive["joint_indices"]
        weights = archive["joint_weights"]
        triangles = archive["triangles"]
    dominant = indices[np.arange(len(indices)), np.argmax(weights, axis=1)]
    is_arm_joint = np.array([n.startswith(ARM_PREFIXES) for n in names])
    vertex_is_arm = is_arm_joint[dominant]
    # a face goes wholly to the part that owns two or three of its corners, so the two
    # rasters partition the surface and no triangle is scored twice or dropped
    face_is_arm = vertex_is_arm[triangles].sum(axis=1) >= 2
    return {"names": names, "vertex_is_arm": vertex_is_arm, "face_is_arm": face_is_arm,
            "triangles": triangles}


def split_smplx() -> np.ndarray:
    with np.load(sil.SMPLX, allow_pickle=True) as data:
        weights = data["weights"].astype(np.float64)
        faces = data["f"].astype(np.int32)
    vertex_is_arm = np.array([int(j) in SMPLX_ARM_JOINTS
                              for j in np.argmax(weights, axis=1)])
    return vertex_is_arm[faces].sum(axis=1) >= 2, vertex_is_arm


def hips_capture(tracks: Path, subject: int) -> np.ndarray:
    """The delivered rig's Hips joint per frame, capture Z-up. IT IS THE ROOT.

    `positions[Hips] = root + rest[Root] + rotate(world[Root], rest[Hips])` and the Root's
    world rotation is the identity in every track this converter writes, so the Hips
    position is the root plus a CONSTANT and its frame-to-frame difference between two
    builds is exactly the root's.
    """
    names = json.loads((tracks / f"subject-{subject:02d}.body-track.json").read_text())[
        "joint_names"]
    base = skeleton_for_joint_names(names)
    npz = np.load(tracks / f"subject-{subject:02d}.body-track.npz")
    world = forward_kinematics_positions(
        np.asarray(npz["root_translation_m"], dtype=np.float64),
        np.asarray(npz["local_rotations_xyzw"], dtype=np.float64),
        skeleton=base).astype(np.float64)
    y = world[:, names.index("Hips")]
    return np.stack([y[:, 0], -y[:, 2], y[:, 1]], axis=-1)


def clavicle_weight_bleed(split: dict) -> dict:
    """How much clavicle-chain weight the TORSO+LEGS vertices actually carry.

    Prediction 1a assumed they carry none. Dominant-weight labelling only says no arm joint
    is the LARGEST influence; linear blend skinning is a blend, so a vertex on the upper
    chest can be labelled torso and still be dragged by the clavicle. This measures it, so
    the refutation of 1a is explained rather than merely reported.
    """
    with np.load(BODY_RUN / "neutral-body.npz", allow_pickle=False) as archive:
        names = [str(v) for v in archive["joint_names"].tolist()]
        indices = archive["joint_indices"]
        weights = archive["joint_weights"]
    is_arm = np.array([n.startswith(ARM_PREFIXES) for n in names])
    arm_weight = (weights * is_arm[indices]).sum(axis=1)
    torso = ~split["vertex_is_arm"]
    return {
        "torso_vertices": int(torso.sum()),
        "with_any_arm_weight": int((arm_weight[torso] > 1e-6).sum()),
        "fraction_with_any_arm_weight": round(
            float((arm_weight[torso] > 1e-6).mean()), 4),
        "their_arm_weight_median": round(float(np.median(
            arm_weight[torso][arm_weight[torso] > 1e-6])), 4) if (arm_weight[torso] > 1e-6).any() else 0.0,
        "their_arm_weight_p95": round(float(np.percentile(
            arm_weight[torso][arm_weight[torso] > 1e-6], 95)), 4) if (arm_weight[torso] > 1e-6).any() else 0.0,
        "why": "prediction 1a assumed the torso+legs part is untouched by a clavicle "
               "re-aim. Dominant-weight labelling does not give that: these vertices are "
               "labelled torso because no ARM joint is their largest influence, not "
               "because they have none.",
    }


def displacement_separation(split: dict, meshes: dict, subject: int) -> dict:
    """THE CORRESPONDENCE CHECK THAT NEEDS NO ANATOMY AT ALL.

    Measured on `delivered -> D2`, which is the arm where THE ROOT DOES NOT MOVE AT ALL
    (0.00 mm, asserted in the D2b gate): the only thing that changed is the clavicle chain.
    So the ARM vertices must travel far and the TORSO ones only as far as their blended
    clavicle weight drags them. If the asset's vertex order had not survived the Blender
    round trip, the two labelled sets would be random halves of the same mesh and their
    displacement distributions would be identical. A large ratio is the evidence; anything
    near 1.0 refutes the split. Measuring it on `D2 -> D2b` instead would be confounded by
    the root shift, which moves every vertex equally -- that version was written first and
    read 3.85x / 1.57x for exactly that reason.
    """
    a = meshes["delivered"][f"verts_{subject:02d}"].astype(np.float64)
    b = meshes["D2"][f"verts_{subject:02d}"].astype(np.float64)
    d = np.linalg.norm(b - a, axis=2)                      # [frame, vertex]
    arm = float(np.median(d[:, split["vertex_is_arm"]])) * 1000.0
    torso = float(np.median(d[:, ~split["vertex_is_arm"]])) * 1000.0
    moved = float((d[:, ~split["vertex_is_arm"]] > 1e-6).mean())
    return {"arm_vertex_displacement_median_mm": round(arm, 4),
            "torso_vertex_displacement_median_mm": round(torso, 4),
            "torso_vertex_frames_that_moved_at_all_fraction": round(moved, 4),
            "ratio": round(arm / torso, 2) if torso > 1e-9 else "infinite -- the torso "
                     "set's median displacement is EXACTLY ZERO under a clavicle-only "
                     "change, which is the strongest form this evidence can take",
            "measured_on": "delivered -> D2, where the root does not move at all",
            "expected": "the arm set must travel MUCH further. A ratio near 1.0 would mean "
                        "the two labelled sets are random halves of the mesh, i.e. the "
                        "vertex order did not survive the Blender export.",
            "needs_no_anatomy": True}


def nearest_joint_agreement(split: dict, verts: np.ndarray, tracks: Path,
                            subject: int, frames=(0, 40, 75, 110, 149)) -> dict:
    """Does the weight-derived label agree with the nearest FK JOINT on the same mesh?

    An independent labelling that needs no vertex correspondence: if the asset's vertex
    order did not survive the Blender round trip, the two would disagree wildly. It is a
    check on the correspondence, not a second opinion on anatomy.
    """
    names = json.loads((tracks / f"subject-{subject:02d}.body-track.json").read_text())[
        "joint_names"]
    base = skeleton_for_joint_names(names)
    npz = np.load(tracks / f"subject-{subject:02d}.body-track.npz")
    world = forward_kinematics_positions(
        np.asarray(npz["root_translation_m"], dtype=np.float64),
        np.asarray(npz["local_rotations_xyzw"], dtype=np.float64),
        skeleton=base).astype(np.float64)
    to_z = np.stack([world[..., 0], -world[..., 2], world[..., 1]], axis=-1)
    is_arm_joint = np.array([n.startswith(ARM_PREFIXES) for n in names])
    agree = []
    for f in frames:
        d = np.linalg.norm(verts[f][:, None, :] - to_z[f][None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        agree.append(float(np.mean(is_arm_joint[nearest] == split["vertex_is_arm"])))
    # CHANCE LEVEL for that agreement: the same two labellings with one of them shuffled.
    # Without it, "0.83" is a number with no scale -- and a scrambled vertex order would
    # land near this value, which is exactly what the check has to be able to say.
    rng = np.random.default_rng(20260902)
    shuffled = split["vertex_is_arm"].copy()
    chance = []
    for f in frames:
        d = np.linalg.norm(verts[f][:, None, :] - to_z[f][None, :, :], axis=2)
        nearest = is_arm_joint[np.argmin(d, axis=1)]
        for _ in range(5):
            rng.shuffle(shuffled)
            chance.append(float(np.mean(nearest == shuffled)))
    return {"frames_checked": list(frames),
            "agreement_with_the_nearest_fk_joint": round(float(np.mean(agree)), 4),
            "chance_level_same_labels_shuffled": round(float(np.mean(chance)), 4),
            "per_frame": [round(a, 4) for a in agree],
            "why": "an INDEPENDENT labelling of the SAME exported vertices, needing no "
                   "correspondence with the asset. It does NOT reach 1.0 and should not: "
                   "nearest-JOINT puts the shoulder cap and the armpit on the other side "
                   "of the cut from nearest-WEIGHT, which is a difference between two "
                   "coarse criteria and not a correspondence failure. What matters is the "
                   "distance from the shuffled-label chance level beside it."}


# ------------------------------------------------------------- the folded-arm control
def folded_arm_track(subject: int):
    """The delivered track with both arms swung across the chest. Instrument only.

    The rotation is built with the converter's OWN `_world_for_bone`, aiming each upper
    arm at a direction expressed in the rig's published contract (+X is the subject's
    LEFT, +Z forward), carried into world by the UpperChest's own frame. The forearm and
    hand keep their local rotations, so the whole arm swings rigidly and the elbow keeps
    whatever bend it had -- a folded arm, not a straightened one.
    """
    names = json.loads((TRACKS / f"subject-{subject:02d}.body-track.json").read_text())[
        "joint_names"]
    skeleton = skeleton_for_joint_names(names)
    npz = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")
    local = np.asarray(npz["local_rotations_xyzw"], dtype=np.float64).copy()
    rest = {j.name: np.asarray(j.rest_translation_m, dtype=np.float64)
            for j in skeleton.joints}
    frames = len(local)
    world = np.zeros_like(local)
    for index, joint in enumerate(skeleton.joints):
        world[:, index] = (local[:, index] if joint.parent == -1
                           else cm._quaternion_multiply(world[:, joint.parent],
                                                        local[:, index]))
    chest = skeleton.index("UpperChest")
    for side, child, across in (("LeftUpperArm", "LeftLowerArm", -1.0),
                                ("RightUpperArm", "RightLowerArm", +1.0)):
        arm = skeleton.index(side)
        parent = skeleton.joints[arm].parent
        target_local = np.asarray((across, 0.0, 0.35))
        target_local /= np.linalg.norm(target_local)
        for f in range(frames):
            target = cm._rotate_vector(world[f, chest], target_local)
            new_world = cm._world_for_bone(world[f, parent], rest[child], target)
            world[f, arm] = new_world
            local[f, arm] = cm._quaternion_multiply(
                cm._quat_inverse(world[f, parent]), new_world)
        # every descendant's world rotation moves with it; their LOCALS are untouched
        for index, joint in enumerate(skeleton.joints):
            if joint.parent != -1:
                world[:, index] = cm._quaternion_multiply(world[:, joint.parent],
                                                          local[:, index])
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    track = cm.BodyTrack(
        duration_ticks=int(npz["ticks"][-1]),
        ticks_per_second=int(cm.TICKS_PER_SECOND),
        sample_rate_hz=30,
        joint_names=tuple(names),
        ticks=np.asarray(npz["ticks"], dtype=np.int64),
        root_translation_m=np.asarray(npz["root_translation_m"], dtype=np.float32),
        local_rotations_xyzw=local.astype(np.float32),
        foot_contacts=np.asarray(npz["foot_contacts"], dtype=np.bool_),
        gaze_direction_body=np.broadcast_to(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32), (frames, 3)),
        gaze_strength=np.zeros(frames, dtype=np.float32),
        gnm_eye_rotations_xyzw=np.tile(
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32), (frames, 2, 1)),
        source_plan_sha256="0" * 64,
    )
    return track


def verify_the_fold() -> dict:
    """Did the arms actually fold? Measured, before the control is called one.

    A control named for what it was meant to be rather than for what it is is the whole
    failure mode this lane keeps guarding against. This is the median distance from each
    hand to the UpperChest joint, delivered against folded, straight from the tracks.
    """
    out = {}
    for subject in (0, 1):
        names = json.loads((TRACKS / f"subject-{subject:02d}.body-track.json").read_text())[
            "joint_names"]
        base = skeleton_for_joint_names(names)
        npz = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")
        root = np.asarray(npz["root_translation_m"], dtype=np.float64)
        row = {}
        for label, local in (
            ("delivered", np.asarray(npz["local_rotations_xyzw"], dtype=np.float64)),
            ("folded", np.asarray(folded_arm_track(subject).local_rotations_xyzw,
                                  dtype=np.float64)),
        ):
            w = forward_kinematics_positions(root, local, skeleton=base).astype(np.float64)
            chest = w[:, names.index("UpperChest")]
            row[label] = {h: round(float(np.median(np.linalg.norm(
                w[:, names.index(h)] - chest, axis=1))) * 1000.0, 1)
                for h in ("LeftHand", "RightHand")}
        out[f"subject_{subject:02d}"] = row
    out["reading"] = (
        "the hands come roughly 200 mm closer to the chest, so the arms really are folded. "
        "That is NOT the same as 'the arm is inside the body's outline': the "
        "arm_hidden_fraction barely moves (see the figures), because with four cameras "
        "around the subject no arm pose projects inside the torso in all of them. What "
        "this control actually tests is a WRONG, FOLDED arm pose -- not the "
        "inside-the-outline degenerate the hypothesis named, which a four-camera average "
        "cannot realise. The distinction is stated wherever this control is cited.")
    return out


def build_folded_mesh() -> dict:
    """Export the folded-arm GLBs through the REAL exporter, then the real Blender path."""
    if FOLDED_MESH.exists():
        print(f"reusing {FOLDED_MESH}")
        return dict(np.load(FOLDED_MESH))
    FOLDED.mkdir(parents=True, exist_ok=True)
    for subject in (0, 1):
        export_animated_body_glb(
            FOLDED / f"subject-{subject:02d}.glb",
            body_manifest_path=BODY_RUN / "neutral-body.json",
            body_asset_path=BODY_RUN / "neutral-body.npz",
            track=folded_arm_track(subject))
        print(f"exported {FOLDED / f'subject-{subject:02d}.glb'}")
    subprocess.run([BLENDER, "--background", "--python",
                    str(ROOT / "tools/compare/blender_export_mesh.py"), "--",
                    str(FOLDED_MESH), str(FOLDED), "30"], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    return dict(np.load(FOLDED_MESH))


# ------------------------------------------------------------------------- statistics
def block_draws(rng, n):
    starts = n - BLOCK + 1
    per = int(np.ceil(n / BLOCK))
    return [(rng.integers(0, starts, size=per)[:, None]
             + np.arange(BLOCK)[None, :]).ravel()[:n] for _ in range(DRAWS)]


def paired(a, b, keep, draws) -> dict:
    """median(a) - median(b) on identical drawn frames, moving block."""
    diffs = []
    for idx in draws:
        sub = idx[keep[idx]]
        if len(sub) >= 5:
            diffs.append(np.median(a[sub]) - np.median(b[sub]))
    diffs = np.asarray(diffs)
    return {"median_difference": round(float(np.median(a[keep]) - np.median(b[keep])), 5),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 5),
                     round(float(np.percentile(diffs, 97.5)), 5)] if len(diffs) else None,
            "draws_used": int(len(diffs)),
            "note": "negative means the FIRST arm is worse"}


# ------------------------------------------------------------------------------- main
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = sil.NATIVE[0] // SCALE, sil.NATIVE[1] // SCALE
    shape = (width, height)
    cams = sil.CAMERAS
    rig = {c.name: c for c in load_camera_rig(sil.RIG_PATH)}
    scaled = {name: cam.scaled(width, height) for name, cam in rig.items()}

    from subject_map import mamma_index_for  # noqa: E402

    cap = np.stack([np.load(TRACKS / f"subject-{s:02d}.body-track.npz")[
        "triangulated_world_positions_z_up_m"] for s in (0, 1)])
    subject_to_body = mamma_index_for(cap)
    tilt_report = json.loads((OUT_DIR / "silhouette-vs-tilt.json").read_text())
    subject_to_tracklet = {
        cam: {int(k.split("_")[1]): int(v) for k, v in row.items()}
        for cam, row in tilt_report["identity"]["our_subject_to_mask_tracklet"].items()}

    saved_delivery, saved_work = sil.DELIVERY, sil.WORK
    sil.WORK = MASK_WORK
    masks = sil.MaskStore(SCALE, cams)
    meshes = {}
    for label, delivery, work in BUILDS:
        sil.DELIVERY, sil.WORK = delivery, work
        meshes[label] = sil.delivered_mesh()
    sil.DELIVERY, sil.WORK = saved_delivery, saved_work
    meshes["folded_arms_CONTROL"] = build_folded_mesh()

    pred_vertices = {b: np.load(sil.MA3D / f"verts_joints_body_id-{b:02d}.npz",
                                allow_pickle=True)["pred_vertices"] for b in (0, 1)}
    smplx_face_is_arm, _ = split_smplx()
    with np.load(sil.SMPLX, allow_pickle=True) as data:
        smplx_faces = data["f"].astype(np.int32)

    split = split_ours()
    ours_faces = split["triangles"]
    correspondence = {}
    for s in (0, 1):
        correspondence[f"subject_{s:02d}"] = {
            "displacement_separation": displacement_separation(split, meshes, s),
            "nearest_fk_joint": nearest_joint_agreement(
                split, meshes["delivered"][f"verts_{s:02d}"].astype(np.float64), TRACKS, s),
        }
    print("correspondence: " + ", ".join(
        f"{k} arm {v['displacement_separation']['arm_vertex_displacement_median_mm']} mm "
        f"vs torso {v['displacement_separation']['torso_vertex_displacement_median_mm']} mm "
        f"({v['displacement_separation']['torso_vertex_frames_that_moved_at_all_fraction']} moved), nearest-joint "
        f"{v['nearest_fk_joint']['agreement_with_the_nearest_fk_joint']} "
        f"(chance {v['nearest_fk_joint']['chance_level_same_labels_shuffled']})"
        for k, v in correspondence.items()))

    # arm -> {subject: (verts, faces_arm, faces_torso)}
    arms: dict = {}
    for label in ("delivered", "D2", "D2b", "folded_arms_CONTROL"):
        arms[label] = {s: (meshes[label][f"verts_{s:02d}"].astype(np.float32),
                           ours_faces[split["face_is_arm"]],
                           ours_faces[~split["face_is_arm"]]) for s in (0, 1)}
    # THE EXACT DECOMPOSITION, and it costs no render of its own to construct.
    # `root_translation` enters forward kinematics only as `positions[Root] = roots +
    # rest[Root]`; every other joint is placed relative to it. So changing ONLY the root
    # translates every joint, and therefore every skinned vertex, by exactly that vector.
    # D2's mesh moved by (root_D2b - root_D2) IS "D2's rotations at D2b's root", and D2b's
    # mesh moved back IS "D2b's rotations at D2's root". Together they split D2 -> D2b into
    # a pure translation and everything else, with no re-solve and no second Blender pass.
    # The Hips FK position IS the root plus a constant rest offset, so its displacement is
    # the root's; `hips_capture` below reads it from the delivered tracks.
    delta = {s: (hips_capture(OUT_DIR / "delivery-root", s)
                 - hips_capture(OUT_DIR / "delivery", s)) for s in (0, 1)}
    arms["ROOT_ONLY_D2_pose_at_D2b_root"] = {
        s: (meshes["D2"][f"verts_{s:02d}"].astype(np.float64)
            + delta[s][:, None, :], ours_faces[split["face_is_arm"]],
            ours_faces[~split["face_is_arm"]]) for s in (0, 1)}
    # NOT "clavicle only", and the name was corrected after the fact: D2b's pose minus
    # the root carries everything that is not the translation, which on the torso+legs part
    # includes the FOOT locals `project_generated_foot_contacts` rewrites inside contact
    # runs (they moved when the root did: [47, 42] -> [37, 60]). The clavicle re-aim
    # dominates it, but it is not alone in it.
    arms["NON_ROOT_D2b_pose_at_D2_root"] = {
        s: (meshes["D2b"][f"verts_{s:02d}"].astype(np.float64)
            - delta[s][:, None, :], ours_faces[split["face_is_arm"]],
            ours_faces[~split["face_is_arm"]]) for s in (0, 1)}
    arms["ORACLE_mamma_mesh"] = {
        s: (pred_vertices[subject_to_body[s]].astype(np.float32),
            smplx_faces[smplx_face_is_arm], smplx_faces[~smplx_face_is_arm])
        for s in (0, 1)}
    names = list(arms)

    F = sil.FRAMES
    # [camera, subject, frame]: torso IoU/prec/recall, arm precision, hidden fraction,
    # whole-person IoU/prec/recall
    keys = ("torso_iou", "torso_precision", "torso_recall", "arm_precision",
            "arm_recall_of_mask", "arm_hidden_fraction", "arm_pixels",
            "whole_iou", "whole_precision", "whole_recall")
    stats = {n: {k: np.full((len(cams), 2, F), np.nan) for k in keys} for n in names}
    population = np.zeros((len(cams), 2, F), dtype=bool)

    todo = list(names)
    if PER_FRAME.exists():
        cached = np.load(PER_FRAME)
        population = cached["population"].astype(bool)
        todo = []
        for n in names:
            if all(f"{n}|{k}" in cached.files for k in keys):
                for k in keys:
                    stats[n][k] = cached[f"{n}|{k}"]
            else:
                todo.append(n)
        print(f"cached: {[n for n in names if n not in todo]}; rendering: {todo}")
    if not todo:
        return _analyse(stats, population, names, cams, keys, correspondence,
                        subject_to_body, subject_to_tracklet, split, width, height,
                        clavicle_weight_bleed(split))

    for c, cam in enumerate(cams):
        for s in (0, 1):
            mask = masks.get(cam, subject_to_tracklet[cam][s])
            area = mask.reshape(F, -1).sum(axis=1)
            population[c, s] = area >= sil.MIN_MASK_PX
            for f in np.nonzero(population[c, s])[0]:
                m = mask[f]
                m_px = float(np.count_nonzero(m))
                for n in todo:
                    verts, arm_faces, torso_faces = arms[n][s]
                    v = verts[f if len(verts) > 1 else 0]
                    arm = sil.rasterise(v, arm_faces, scaled[cam], shape)
                    torso = sil.rasterise(v, torso_faces, scaled[cam], shape)
                    whole = arm | torso
                    p, r, i = sil.score(torso, m)
                    stats[n]["torso_precision"][c, s, f] = p
                    stats[n]["torso_recall"][c, s, f] = r
                    stats[n]["torso_iou"][c, s, f] = i
                    p, r, i = sil.score(whole, m)
                    stats[n]["whole_precision"][c, s, f] = p
                    stats[n]["whole_recall"][c, s, f] = r
                    stats[n]["whole_iou"][c, s, f] = i
                    a_px = float(np.count_nonzero(arm))
                    stats[n]["arm_pixels"][c, s, f] = a_px
                    stats[n]["arm_precision"][c, s, f] = (
                        float(np.count_nonzero(arm & m)) / a_px if a_px else np.nan)
                    stats[n]["arm_recall_of_mask"][c, s, f] = (
                        float(np.count_nonzero(arm & m)) / m_px if m_px else np.nan)
                    stats[n]["arm_hidden_fraction"][c, s, f] = (
                        float(np.count_nonzero(arm & torso)) / a_px if a_px else np.nan)
            print(f"scored {cam} subject {s:02d}: {int(population[c, s].sum())} frames")
    np.savez_compressed(PER_FRAME, population=population,
                        **{f"{n}|{k}": stats[n][k] for n in names for k in keys})
    return _analyse(stats, population, names, cams, keys, correspondence,
                    subject_to_body, subject_to_tracklet, split, width, height,
                    clavicle_weight_bleed(split))


def _analyse(stats, population, names, cams, keys, correspondence, subject_to_body,
             subject_to_tracklet, split, width, height, bleed) -> int:
    cap = np.stack([np.load(TRACKS / f"subject-{s:02d}.body-track.npz")[
        "triangulated_world_positions_z_up_m"] for s in (0, 1)])
    tilt_by_subject = []
    for s in (0, 1):
        pelvis = 0.5 * (cap[s][:, cm.JOINT_INDEX["left_hip"]]
                        + cap[s][:, cm.JOINT_INDEX["right_hip"]])
        up = cap[s][:, cm.JOINT_INDEX["neck"]] - pelvis
        up = up / np.linalg.norm(up, axis=1, keepdims=True)
        tilt_by_subject.append(np.degrees(np.arccos(np.clip(up[:, 2], -1.0, 1.0))))
    rng = np.random.default_rng(SEED)
    draws = block_draws(rng, sil.FRAMES)
    out: dict = {"subjects": {}}
    for s in (0, 1):
        every = population[:, s].all(axis=0)
        # one value per frame per arm: the mean over the four cameras, on frames every
        # camera scored, so every arm shares one denominator
        cell = {n: {k: np.nanmean(stats[n][k][:, s, :], axis=0) for k in keys}
                for n in names}
        entry: dict = {"frames": int(every.sum())}
        for k in keys:
            entry[k] = {n: round(float(np.median(cell[n][k][every])), 5) for n in names}
        # THE SAME CUT ON THE UPRIGHT FRAMES ONLY. Section 13 found the whole-person fall
        # at full size where the body has barely moved; if the root has a cost of its own
        # it must be small HERE and large on the bent frames, and if it is large here it is
        # not the root. Same tilt series and same band as section 13.
        tilt = np.asarray(tilt_by_subject[s])
        upright = every & (tilt <= 10.0)
        entry["upright_band_tilt_le_10deg"] = {
            "frames": int(upright.sum()),
            "torso_iou": {n: round(float(np.median(cell[n]["torso_iou"][upright])), 5)
                          for n in names},
            "arm_precision": {n: round(float(np.median(cell[n]["arm_precision"][upright])), 5)
                              for n in names},
            "whole_iou": {n: round(float(np.median(cell[n]["whole_iou"][upright])), 5)
                          for n in names},
        }
        for k in ("torso_iou", "whole_iou"):
            for a, b in (("D2", "delivered"), ("D2b", "D2"),
                         ("ROOT_ONLY_D2_pose_at_D2b_root", "D2"),
                         ("NON_ROOT_D2b_pose_at_D2_root", "D2")):
                entry["upright_band_tilt_le_10deg"][f"{k}__{a}_minus_{b}"] = paired(
                    cell[a][k], cell[b][k], upright, draws)
        entry["margins"] = {}
        for k in ("torso_iou", "torso_precision", "torso_recall", "arm_precision",
                  "arm_hidden_fraction", "whole_iou", "whole_precision", "whole_recall"):
            entry["margins"][f"{k}__D2_minus_delivered"] = paired(
                cell["D2"][k], cell["delivered"][k], every, draws)
            entry["margins"][f"{k}__D2b_minus_D2"] = paired(
                cell["D2b"][k], cell["D2"][k], every, draws)
            entry["margins"][f"{k}__D2b_minus_delivered"] = paired(
                cell["D2b"][k], cell["delivered"][k], every, draws)
            entry["margins"][f"{k}__folded_minus_delivered"] = paired(
                cell["folded_arms_CONTROL"][k], cell["delivered"][k], every, draws)
            entry["margins"][f"{k}__ROOT_ONLY_minus_D2"] = paired(
                cell["ROOT_ONLY_D2_pose_at_D2b_root"][k], cell["D2"][k], every, draws)
            entry["margins"][f"{k}__NON_ROOT_minus_D2"] = paired(
                cell["NON_ROOT_D2b_pose_at_D2_root"][k], cell["D2"][k], every, draws)
        out["subjects"][f"subject_{s:02d}"] = entry

    # per camera, for the record
    per_camera: dict = {}
    for c, cam in enumerate(cams):
        per_camera[cam] = {}
        for s in (0, 1):
            ok = population[c, s]
            per_camera[cam][f"subject_{s:02d}"] = {
                k: {n: round(float(np.nanmedian(stats[n][k][c, s, ok])), 5) for n in names}
                for k in ("torso_iou", "arm_precision", "arm_hidden_fraction")}
    out["per_camera"] = per_camera

    # ---- were the pre-registered predictions met?
    met: dict = {}
    for s in ("subject_00", "subject_01"):
        m = out["subjects"][s]["margins"]
        d2 = m["torso_iou__D2_minus_delivered"]
        d2b = m["torso_iou__D2b_minus_D2"]
        met[s] = {
            "1a_torso_unchanged_delivered_to_D2": {
                "difference": d2["median_difference"], "ci95": d2["ci95"],
                "met": bool(d2["ci95"] is not None and d2["ci95"][0] <= 0.0 <= d2["ci95"][1]),
                "why": "the torso+legs vertices carry no clavicle-chain weight and D2 moves "
                       "no root, so this arm can only be identical. Anything else means the "
                       "split is wrong, not that the pipeline moved."},
            "1a_D2b_own_cost_on_torso_and_legs": {
                "difference": d2b["median_difference"], "ci95": d2b["ci95"],
                "material": bool(d2b["ci95"] is not None and d2b["ci95"][1] < -0.003),
                "why": "THE ISOLATED FIGURE. Between D2 and D2b these vertices move by "
                       "exactly the root shift and nothing else, so this IS D2b's own "
                       "silhouette cost with the arms taken out of the question."},
            "1b_arm_precision_falls": {
                "delivered": out["subjects"][s]["arm_precision"]["delivered"],
                "D2": out["subjects"][s]["arm_precision"]["D2"],
                "D2b": out["subjects"][s]["arm_precision"]["D2b"],
                "ORACLE_ceiling": out["subjects"][s]["arm_precision"]["ORACLE_mamma_mesh"],
                "met": bool(out["subjects"][s]["arm_precision"]["D2b"]
                            < out["subjects"][s]["arm_precision"]["D2"]
                            < out["subjects"][s]["arm_precision"]["delivered"])},
            "1c_hidden_fraction_falls_delivered_to_D2": {
                "delivered": out["subjects"][s]["arm_hidden_fraction"]["delivered"],
                "D2": out["subjects"][s]["arm_hidden_fraction"]["D2"],
                "D2b": out["subjects"][s]["arm_hidden_fraction"]["D2b"],
                "ORACLE": out["subjects"][s]["arm_hidden_fraction"]["ORACLE_mamma_mesh"],
                "met": bool(out["subjects"][s]["arm_hidden_fraction"]["D2"]
                            < out["subjects"][s]["arm_hidden_fraction"]["delivered"])},
            "2_folded_arms_score_higher": {
                "whole_iou": m["whole_iou__folded_minus_delivered"],
                "whole_precision": m["whole_precision__folded_minus_delivered"],
                "whole_recall": m["whole_recall__folded_minus_delivered"],
                "met": bool(m["whole_iou__folded_minus_delivered"]["median_difference"] > 0
                            and m["whole_precision__folded_minus_delivered"]["median_difference"] > 0)},
        }
    out["pre_registered_outcomes"] = met

    report = {
        "instrument": "tools/compare/silhouette_partwise.py",
        "shipping": "NOTHING. Instrument only; silhouette.py's own outputs are untouched, "
                    "and the folded-arm build lives under artifacts/compare/d2-clavicle/.",
        "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/silhouette_partwise.py",
        "autoanim_gnm_resolved_to": str(Path(autoanim_gnm.__file__).resolve()),
        "pre_registered": PREREGISTERED,
        "render": {"resolution": f"{width}x{height}", "cameras": list(cams),
                   "frames": sil.FRAMES,
                   "path": "silhouette.rasterise / silhouette.score / silhouette.MaskStore, "
                           "imported, not re-implemented"},
        "clavicle_weight_bleed_into_the_torso_part": bleed,
        "exact_decomposition": (
            "ROOT_ONLY_D2_pose_at_D2b_root and NON_ROOT_D2b_pose_at_D2_root split "
            "D2 -> D2b into a pure translation and everything else. A root change "
            "translates every skinned vertex by exactly that vector -- `root_translation` "
            "enters FK only at the Root joint -- so both arms are constructed from the "
            "already-rendered meshes by adding or subtracting the per-frame root delta, "
            "with no re-solve and no second Blender pass. ROOT_ONLY minus D2 is the root's "
            "OWN silhouette cost, isolated exactly."),
        "part_split": {
            "ours": "dominant skin weight from the body asset the delivery was built from "
                    f"({BODY_RUN.name}/neutral-body.npz). ARMS = the clavicle chain and "
                    "below: Shoulder, UpperArm, LowerArm, Hand and every finger. "
                    "TORSO+LEGS = everything else, head and eyes included. A face goes "
                    "wholly to the part owning two or three of its corners, so the two "
                    "rasters partition the surface.",
            "oracle": "SMPL-X dominant skin weight; ARMS = collars 13/14, shoulders 16/17, "
                      "elbows 18/19, wrists 20/21 and fingers 25-54. The collars are "
                      "INCLUDED so both meshes are cut at the same anatomical place; the "
                      "choice moves the shoulder cap between the parts and is recorded.",
            "arm_faces": int(split["face_is_arm"].sum()),
            "torso_faces": int((~split["face_is_arm"]).sum()),
            "arm_vertices": int(split["vertex_is_arm"].sum()),
            "correspondence_check": correspondence,
        },
        "folded_arm_control_geometry": verify_the_fold(),
        "folded_arm_control": {
            "what": "the DELIVERED track with both upper arms re-aimed across the chest by "
                    "the converter's own `_world_for_bone`, at a direction expressed in the "
                    "skeleton's published contract (+X is the subject's LEFT, +Z forward) "
                    "carried into world by the UpperChest's frame. The forearm and hand "
                    "keep their local rotations, so the arm folds rigidly and the elbow "
                    "keeps its bend. Exported through the REAL `export_animated_body_glb` "
                    "and the REAL blender_export_mesh.py -- wrapped, never re-implemented.",
            "root_and_every_other_joint": "untouched",
            "why": "the control I6 never had. If a limb-collapsed body scores HIGHER than "
                   "the delivery, the mask cannot be read as a limb-placement gate.",
            "what_it_is_NOT": "it is not the 'limb hidden inside the outline' degenerate "
                              "the hypothesis named. The fold is real -- the hands come "
                              "~200 mm closer to the chest, see folded_arm_control_geometry "
                              "-- but the arm_hidden_fraction barely moves (0.5076 -> "
                              "0.5394 and 0.5481 -> 0.5514), because with four cameras "
                              "around the subject no arm pose projects inside the torso in "
                              "all of them. What is tested is a WRONG FOLDED arm pose. The "
                              "inside-the-outline degenerate needs a SINGLE-camera test and "
                              "is not attempted here.",
        },
        "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}"
                                   for k, v in subject_to_body.items()},
        "blind_to": (
            "A silhouette is still a projection: depth is invisible, a left/right mirror of "
            "a fore-aft symmetric pose is invisible, and everything inside the outline is "
            "invisible -- which is the whole point of the folded-arm control. The part "
            "split cures the LAST of those only for the two parts it cuts: an arm wrongly "
            "placed but still inside the arm region is not separated from a right one. "
            "This pass says WHERE the fall is, never which placement is anatomically "
            "right; no instrument in this lane does that yet."),
        **out,
    }
    REPORT.write_text(json.dumps(report, indent=2, default=float))
    _figure(stats, population, names, cams)

    for s in ("subject_00", "subject_01"):
        e = out["subjects"][s]
        print(f"\n--- {s}, {e['frames']} frames")
        print("  torso+legs IoU:   " + "  ".join(f"{n} {e['torso_iou'][n]:.4f}" for n in names))
        print(f"    D2-delivered {met[s]['1a_torso_unchanged_delivered_to_D2']['difference']} "
              f"{met[s]['1a_torso_unchanged_delivered_to_D2']['ci95']}  |  "
              f"D2b-D2 {met[s]['1a_D2b_own_cost_on_torso_and_legs']['difference']} "
              f"{met[s]['1a_D2b_own_cost_on_torso_and_legs']['ci95']}")
        print("  arm precision:    " + "  ".join(f"{n} {e['arm_precision'][n]:.4f}" for n in names))
        print("  arm recall(mask): " + "  ".join(f"{n} {e['arm_recall_of_mask'][n]:.4f}" for n in names))
        print("  arm hidden frac:  " + "  ".join(f"{n} {e['arm_hidden_fraction'][n]:.4f}" for n in names))
        print("  whole IoU:        " + "  ".join(f"{n} {e['whole_iou'][n]:.4f}" for n in names))
        m = e["margins"]
        print(f"    ROOT ONLY - D2:      torso {m['torso_iou__ROOT_ONLY_minus_D2']['median_difference']} "
              f"{m['torso_iou__ROOT_ONLY_minus_D2']['ci95']}   whole "
              f"{m['whole_iou__ROOT_ONLY_minus_D2']['median_difference']} "
              f"{m['whole_iou__ROOT_ONLY_minus_D2']['ci95']}")
        print(f"    NON ROOT   - D2:     torso {m['torso_iou__NON_ROOT_minus_D2']['median_difference']} "
              f"{m['torso_iou__NON_ROOT_minus_D2']['ci95']}   whole "
              f"{m['whole_iou__NON_ROOT_minus_D2']['median_difference']} "
              f"{m['whole_iou__NON_ROOT_minus_D2']['ci95']}")
        u = e["upright_band_tilt_le_10deg"]
        print(f"    UPRIGHT <=10 deg, n={u['frames']}: torso IoU " + "  ".join(
            f"{n} {u['torso_iou'][n]:.4f}" for n in ("delivered", "D2", "D2b",
                                                     "ROOT_ONLY_D2_pose_at_D2b_root")))
        print(f"      torso D2b-D2 {u['torso_iou__D2b_minus_D2']['median_difference']} "
              f"{u['torso_iou__D2b_minus_D2']['ci95']}  |  ROOT_ONLY-D2 "
              f"{u['torso_iou__ROOT_ONLY_D2_pose_at_D2b_root_minus_D2']['median_difference']} "
              f"{u['torso_iou__ROOT_ONLY_D2_pose_at_D2b_root_minus_D2']['ci95']}")
        f = met[s]["2_folded_arms_score_higher"]
        print(f"    FOLDED - delivered: IoU {f['whole_iou']['median_difference']} "
              f"{f['whole_iou']['ci95']}  precision {f['whole_precision']['median_difference']} "
              f"{f['whole_precision']['ci95']}  recall {f['whole_recall']['median_difference']} "
              f"{f['whole_recall']['ci95']}")
    print(f"\nwrote {REPORT} and {FIGURE}")
    return 0


def _figure(stats, population, names, cams) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {"delivered": "#8a8f98", "D2": "#4cc9d0", "D2b": "#2f6fd0",
              "ORACLE_mamma_mesh": "#e08a24", "folded_arms_CONTROL": "#b4508f",
              "ROOT_ONLY_D2_pose_at_D2b_root": "#7ba7e8",
              "NON_ROOT_D2b_pose_at_D2_root": "#9fd8dc"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for s in (0, 1):
        every = population[:, s].all(axis=0)
        x = np.nonzero(every)[0]
        for col, key, title, lo in (
            (0, "torso_iou", "torso + legs only: IoU against the SAM2 mask", 0.0),
            (1, "arm_precision", "arms only: fraction of arm pixels inside the mask", 0.0),
        ):
            ax = axes[s, col]
            for n in names:
                y = np.nanmean(stats[n][key][:, s, :], axis=0)[every]
                ax.plot(x, y, "-", lw=1.3, alpha=0.85, color=colour[n],
                        label=n if (s == 0 and col == 0) else None)
            ax.set_title(f"subject {s:02d}: {title}   (higher is better)", fontsize=10)
            ax.set_xlabel("frame")
            ax.set_ylabel("IoU" if col == 0 else "precision")
            ax.set_ylim(lo, 1.0)
            ax.grid(alpha=0.25)
            if s == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.suptitle("D2b: which part of the body is the silhouette's fall in?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
