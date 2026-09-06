#!/usr/bin/env python3
"""D9b: the foot-contact hoist re-aim. The gate.

THE DEFECT. `positions_to_body_track` aims every root-dependent bone -- the D7b trunk
frame, the two clavicles, the four arm bones -- from origins computed with the PRE-hoist
root, and then ends by calling `project_generated_foot_contacts`, which TRANSLATES the
root (a per-run correction that plants a slow, low foot, plus a whole-take penetration
lift). A translation leaves every rotation alone, so each of those bones still points
along `target - origin_BEFORE_the_hoist` while its origin has moved. It is D7b's defect
on the neck and D9's on the arms, one operation later, and present since D2b.

THE INSTRUMENT, and every figure it prints comes from a delivered file's own bytes:

  (i)   THE HOIST, two independent recoveries that share no line of code -- D8c's
        converter-line subtraction (`root - (pelvis - rest[Hips] - _leg_root_offset)`,
        through `d8c_hip_placement.load`) and D9's over-determined least-squares fit of
        one translation to the four arm bones' rays. They must agree; a disagreement means
        one of them is wrong and no number below can be trusted.
  (ii)  THE RAY MISS of every aimed bone, from the DELIVERED origin and from the PRE-HOIST
        origin. This is the defect's signature: a bone aimed before the hoist sits EXACTLY
        on its ray from the old origin (0.0 mm) and misses from the new one by the
        perpendicular part of the hoist. The four LEG bones are reported and are NOT part
        of any band -- they are aimed landmark-to-landmark, which is D9-legs' territory.
  (iii) THE CONTACTS, the runs, `ground_penetration_before_m` and the lowest delivered
        foot, so the reader can see that on THIS take the whole-take lift is zero and every
        millimetre of hoist is the per-run plant.
  (iv)  THE CLOSED-FORM RE-AIM: what turning each bone about its DELIVERED origin onto the
        ray to its own target would move the child joint by. Predicted before the build.
  (v)   THE ORACLE: the D3 gate's own six exact-skeleton bodies, rebuilt THROUGH ITS OWN
        `oracle_block` with watchers on `positions_to_body_track`,
        `project_generated_foot_contacts` and `_reachable_clavicle_sequence`. Nothing here
        re-implements the construction; the wrappers record what the gate's own code did.

WHAT IT IS BLIND TO, stated with the numbers:

  * It scores PLACEMENT. A bone on its ray can still be rolled about its own axis, and a
    ray miss of 0.0 says nothing about that roll.
  * The ray-miss reference is THE DELIVERY'S OWN captured landmarks. It cannot say the
    landmarks are right; `delivered_vs_capture.py` carries the same blindness and says so.
  * It cannot see the MESH. Four to six millimetres on 45 % / 15 % of frames is below what
    the SAM2 masks resolve; the photographs (B3) are the arm that renders the file.
  * On the oracle, `retarget_cost.score` ALIGNS each frame on the leg-root midpoint -- it
    "removes the ground-projection shift" by design -- so it is STRUCTURALLY BLIND to a
    root move and reads the correct fix as WORSE. That is measured here, attributed, and
    no band is moved for it; the absolute-frame companion row is the answer.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_gate.py --oracle \
        --out artifacts/compare/d9b-hoist/instrument.json
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "tools/swap-harness", "scripts"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import forward_kinematics_positions  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402
import d8c_hip_placement as hp  # noqa: E402
import delivered_vs_capture as dvc  # noqa: E402
import retarget_cost as rc  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d9b-hoist"
SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
HOIST_REPORT_CUT_MM = 0.5      # a REPORT cut. Never control flow -- the re-solve runs on
ZERO_CUT_MM = 1.0e-3           # every frame, and these two only split the tables.

# (the joint whose ROTATION aims the bone, the child joint on the ray, the captured
#  landmark that child is aimed at, the family). The seven root-dependent rows are the
#  ones the bands read; the four leg rows are REPORTED and handed to D9-legs.
RAYS = (
    ("LeftUpperArm", "LeftLowerArm", "left_elbow", "arm"),
    ("LeftLowerArm", "LeftHand", "left_wrist", "arm"),
    ("RightUpperArm", "RightLowerArm", "right_elbow", "arm"),
    ("RightLowerArm", "RightHand", "right_wrist", "arm"),
    ("LeftShoulder", "LeftUpperArm", "left_shoulder", "clavicle"),
    ("RightShoulder", "RightUpperArm", "right_shoulder", "clavicle"),
    ("Spine", "Neck", "neck", "trunk"),
    ("LeftUpperLeg", "LeftLowerLeg", "left_knee", "leg"),
    ("LeftLowerLeg", "LeftFoot", "left_ankle", "leg"),
    ("RightUpperLeg", "RightLowerLeg", "right_knee", "leg"),
    ("RightLowerLeg", "RightFoot", "right_ankle", "leg"),
)
ROOT_DEPENDENT = tuple(row for row in RAYS if row[3] != "leg")
FOOT_JOINTS = ("LeftFoot", "LeftToes", "RightFoot", "RightToes")
# The joints the re-solve is ALLOWED to rewrite. Everything else must be bit-identical.
RESOLVED_JOINTS = (
    "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftEye", "RightEye",
    "LeftShoulder", "RightShoulder", "LeftUpperArm", "LeftLowerArm",
    "RightUpperArm", "RightLowerArm", "LeftHand", "RightHand",
)
FROZEN_JOINTS = (
    "Root", "Hips", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
)
DELIVERED_FILES = tuple(
    f"subject-{s:02d}{suffix}" for s in (0, 1)
    for suffix in (".glb", ".body-track.json", ".body-track.npz", ".mapping.npz"))


# ------------------------------------------------------------------------ small helpers
def summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(values.max()), 4)}


def runs_of(flags: np.ndarray) -> list[list[int]]:
    on = np.flatnonzero(flags)
    if not on.size:
        return []
    return [[int(r[0]), int(r[-1])]
            for r in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def rig_to_capture(vector: np.ndarray) -> np.ndarray:
    """The converter's basis, inverted: rig (x, y, z) -> capture (x, -z, y)."""
    vector = np.asarray(vector, dtype=np.float64)
    return np.stack([vector[..., 0], -vector[..., 2], vector[..., 1]], axis=-1)


def absolute_score(fk, reference_y_up, skeleton) -> dict[str, np.ndarray]:
    """`retarget_cost.score` WITHOUT its leg-root alignment, in the same units.

    The companion row the D3 gate does not have. `rc.score` subtracts each frame's
    leg-root midpoint from the rig and each frame's hip midpoint from the reference, which
    is exactly a per-frame translation -- so it cannot see a root move at all, and after a
    re-aim from a hoisted origin it reads the CORRECT answer as worse by the perpendicular
    part of the hoist. This one does not align, and on exact truth (where the reference IS
    the truth, in the same world) it is the quantity that matters.
    """
    out: dict[str, np.ndarray] = {}
    for name, rig in rc.RIG_FOR.items():
        out[name] = 1000.0 * np.linalg.norm(
            fk[:, skeleton.index(rig)] - reference_y_up[:, cm.JOINT_INDEX[name]], axis=1)
    return out


def groups_p95(error: dict[str, np.ndarray]) -> dict[str, float]:
    return {group: round(float(np.percentile(
        np.concatenate([error[n] for n in names]), 95)), 2)
        for group, names in rc.GROUPS.items()}


def groups_masked(error: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    return {group: round(float(np.median(
        np.concatenate([error[n][mask] for n in names]))), 2)
        for group, names in rc.GROUPS.items()}


# --------------------------------------------------------- (i)-(iv), one delivered build
def read_build(directory: Path, subject: int) -> dict:
    """Everything the bands read for one performer, from that build's own bytes."""

    loaded = hp.load(directory, subject)
    index, world = dvc.delivered_positions(directory, subject)
    capture = dvc.capture_positions(directory, subject)
    frames = len(capture)
    hoist = rig_to_capture(loaded["hoist"])[:frames]
    with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
        contacts = np.asarray(archive["foot_contacts"])
        rotations = np.asarray(archive["local_rotations_xyzw"])
        root = np.asarray(archive["root_translation_m"])
    return {"index": index, "world": world, "capture": capture, "frames": frames,
            "hoist": hoist, "contacts": contacts, "rotations": rotations, "root": root,
            "names": loaded["track_names"], "rest": loaded["rest"]}


def arm_fit_hoist(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """D9's recovery: the one translation that best explains the four arm bones' misses.

    Over-determined -- four bones, twelve equations, three unknowns -- so the residual is
    itself the check that a single translation is the whole story.
    """
    index, world, capture = data["index"], data["world"], data["capture"]
    fitted = np.zeros((data["frames"], 3))
    residual = np.zeros(data["frames"])
    for frame in range(data["frames"]):
        rows, targets = [], []
        for parent, child, landmark, family in RAYS:
            if family != "arm":
                continue
            origin = world[frame, index[parent]]
            axis = world[frame, index[child]] - origin
            axis = axis / np.linalg.norm(axis)
            projector = np.eye(3) - np.outer(axis, axis)
            rows.append(projector)
            targets.append(-projector @ (capture[frame, cm.JOINT_INDEX[landmark]] - origin))
        matrix, target = np.vstack(rows), np.concatenate(targets)
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        fitted[frame] = solution
        residual[frame] = np.linalg.norm(matrix @ solution - target)
    return fitted, residual


def build_block(directory: Path, label: str) -> dict:
    block: dict = {"label": label, "directory": str(directory), "subjects": {}}
    for subject in (0, 1):
        data = read_build(directory, subject)
        index, world, capture = data["index"], data["world"], data["capture"]
        hoist = data["hoist"]
        magnitude = 1e3 * np.linalg.norm(hoist, axis=1)
        fitted, residual = arm_fit_hoist(data)
        agreement = 1e3 * np.linalg.norm(hoist - fitted, axis=1)
        hoisted = magnitude > HOIST_REPORT_CUT_MM
        feet = np.min(world[:, [index[n] for n in FOOT_JOINTS], 2], axis=1)
        rows: dict = {}
        for parent, child, landmark, family in RAYS:
            origin = world[:, index[parent]]
            axis = world[:, index[child]] - origin
            length = np.linalg.norm(axis, axis=1)
            axis = axis / length[:, None]
            target = capture[:, cm.JOINT_INDEX[landmark]]

            def miss(from_origin):
                delta = target - from_origin
                return 1e3 * np.linalg.norm(
                    delta - np.sum(delta * axis, axis=1)[:, None] * axis, axis=1)

            delivered_miss = miss(origin)
            pre_hoist_miss = miss(origin - hoist)
            wanted = target - origin
            wanted = wanted / np.linalg.norm(wanted, axis=1)[:, None]
            re_aim_move = 1e3 * length * np.linalg.norm(wanted - axis, axis=1)
            rows[f"{parent}->{landmark}"] = {
                "family": family,
                "miss_from_delivered_origin_mm": {
                    "whole": summary(delivered_miss),
                    "hoisted": summary(delivered_miss[hoisted]),
                    "unhoisted": summary(delivered_miss[~hoisted])},
                "miss_from_pre_hoist_origin_mm": {
                    "whole": summary(pre_hoist_miss),
                    "hoisted": summary(pre_hoist_miss[hoisted])},
                "closed_form_re_aim_child_move_mm": {
                    "hoisted": summary(re_aim_move[hoisted]),
                    "unhoisted": summary(re_aim_move[~hoisted])},
            }
        block["subjects"][f"subject_{subject:02d}"] = {
            "frames": data["frames"],
            "hoist_mm": summary(magnitude),
            "frames_over_report_cut": int(hoisted.sum()),
            "frames_under_zero_cut": int((magnitude < ZERO_CUT_MM).sum()),
            "hoisted_frames": [int(i) for i in np.flatnonzero(hoisted)],
            "two_recoveries_disagreement_mm": summary(agreement),
            "arm_fit_residual_mm": summary(1e3 * residual),
            "whole_take_constant_lift_mm": round(float(magnitude.min()), 4),
            "contacts": {"count": [int(v) for v in data["contacts"].sum(0)],
                         "runs": [runs_of(data["contacts"][:, s]) for s in (0, 1)]},
            "lowest_delivered_foot_or_toe_mm": round(float(1e3 * feet.min()), 3),
            "rays": rows,
        }
    return block


# --------------------------------------------------------------------- B2: bit-identity
def bit_identity(baseline: Path, candidate: Path) -> dict:
    block: dict = {"baseline": str(baseline), "candidate": str(candidate), "subjects": {}}
    for subject in (0, 1):
        with np.load(baseline / f"subject-{subject:02d}.body-track.npz") as a, \
             np.load(candidate / f"subject-{subject:02d}.body-track.npz") as b:
            names = json.loads((candidate / f"subject-{subject:02d}.body-track.json"
                                ).read_text())["joint_names"]
            base_rot = np.asarray(a["local_rotations_xyzw"])
            cand_rot = np.asarray(b["local_rotations_xyzw"])
            base_root = np.asarray(a["root_translation_m"])
            cand_root = np.asarray(b["root_translation_m"])
            base_contacts = np.asarray(a["foot_contacts"])
            cand_contacts = np.asarray(b["foot_contacts"])
            landmarks_identical = bool(np.array_equal(
                a["triangulated_world_positions_z_up_m"],
                b["triangulated_world_positions_z_up_m"]))
            raw_identical = bool(np.array_equal(
                a["raw_triangulated_world_positions_z_up_m"],
                b["raw_triangulated_world_positions_z_up_m"], equal_nan=True))
        hoist = rig_to_capture(
            np.asarray(cand_root, np.float64) - np.asarray(base_root, np.float64))
        # The candidate's own hoist, recovered from the BASELINE build (the frames the
        # baseline's projection moved) -- the split the clauses are stated on.
        base_data = read_build(baseline, subject)
        magnitude = 1e3 * np.linalg.norm(base_data["hoist"], axis=1)
        hoisted = magnitude > HOIST_REPORT_CUT_MM
        moved = ~np.all(base_rot == cand_rot, axis=2)          # [frame, joint]
        per_joint = {}
        for position, name in enumerate(names):
            frames_moved = np.flatnonzero(moved[:, position])
            per_joint[name] = {
                "frames_moved": int(frames_moved.size),
                "moved_on_unhoisted_frames": int((~hoisted[frames_moved]).sum()),
                "max_abs_delta": float(np.max(np.abs(
                    base_rot[:, position] - cand_rot[:, position]))) if frames_moved.size
                else 0.0,
            }
        block["subjects"][f"subject_{subject:02d}"] = {
            "root_identical": bool(np.array_equal(base_root, cand_root)),
            "root_max_delta_m": float(np.max(np.abs(base_root - cand_root))),
            "contacts_identical": bool(np.array_equal(base_contacts, cand_contacts)),
            "smoothed_landmarks_identical": landmarks_identical,
            "raw_landmarks_identical": raw_identical,
            "frames_hoisted_in_baseline": int(hoisted.sum()),
            "everything_identical_on_unhoisted_frames": bool(
                not moved[~hoisted].any()),
            "joints_that_moved": sorted(n for n in per_joint
                                        if per_joint[n]["frames_moved"]),
            "frozen_joints_identical": bool(all(
                per_joint[n]["frames_moved"] == 0 for n in FROZEN_JOINTS)),
            "moved_set_is_within_the_resolved_set": bool(all(
                n in RESOLVED_JOINTS for n in per_joint
                if per_joint[n]["frames_moved"])),
            "per_joint": per_joint,
        }
    return block


# ------------------------------------------------------- B4: contacts and the plant
def plant_block(baseline: Path, candidate: Path | None) -> dict:
    """The contact runs, and how far the planted foot travels over each of them.

    The clause that matters on a take whose penetration is zero: a degenerate that keeps
    the lock and drops `roots += correction` zeroes B1 and leaves the foot sliding.
    """
    block: dict = {"subjects": {}}
    labels = {"baseline": baseline}
    if candidate is not None:
        labels["candidate"] = candidate
    for subject in (0, 1):
        row: dict = {}
        for label, directory in labels.items():
            data = read_build(directory, subject)
            index, world = data["index"], data["world"]
            travel: dict = {}
            for side, foot in ((0, "LeftFoot"), (1, "RightFoot")):
                per_run = []
                for start, end in runs_of(data["contacts"][:, side]):
                    points = world[start:end + 1, index[foot]]
                    per_run.append(round(float(1e3 * np.max(np.linalg.norm(
                        points - points[0], axis=1))), 4))
                travel[foot] = per_run
            row[label] = {
                "contacts": [int(v) for v in data["contacts"].sum(0)],
                "runs": [runs_of(data["contacts"][:, s]) for s in (0, 1)],
                "planted_foot_travel_per_run_mm": travel,
                "max_planted_foot_travel_mm": round(float(max(
                    [max(v) for v in travel.values() if v] or [0.0])), 4),
                "lowest_foot_or_toe_mm": round(float(1e3 * np.min(
                    world[:, [index[n] for n in FOOT_JOINTS], 2])), 3),
            }
        if candidate is not None:
            row["identical"] = bool(
                row["baseline"]["contacts"] == row["candidate"]["contacts"]
                and row["baseline"]["runs"] == row["candidate"]["runs"]
                and row["baseline"]["planted_foot_travel_per_run_mm"]
                == row["candidate"]["planted_foot_travel_per_run_mm"])
        block["subjects"][f"subject_{subject:02d}"] = row
    return block


# ------------------------------------------------ B1: the excess over the aim's own floor
def excess_block(baseline: Path, candidate: Path, labels: tuple[str, str]) -> dict:
    """`delivered_vs_capture` on identical draws, plus the PER-FRAME excess over the floor.

    The floor is what an aim cannot remove -- a bone LENGTH error -- and the excess over it
    is exactly the ray miss. It is the CLOSURE quantity: the candidate sets it directly,
    which is why the merge rule reads it only in conjunction with B2, B4 and the tripwire.
    """
    deliveries = {labels[0]: baseline, labels[1]: candidate}
    payload = dvc.report(deliveries, "smoothed")
    block: dict = {
        "same_denominator": payload["same_denominator"],
        "triangulated_landmarks_byte_identical_across_arms":
            payload["triangulated_landmarks_byte_identical_across_arms"],
        "bootstrap": payload["bootstrap"],
        "subjects": {},
    }
    pairs = (("LeftLowerArm", "left_elbow"), ("LeftHand", "left_wrist"),
             ("RightLowerArm", "right_elbow"), ("RightHand", "right_wrist"))
    for subject in (0, 1):
        key = f"subject_{subject:02d}"
        joints = payload["subjects"][key]["joints"]
        excess: dict = {}
        base_data = read_build(baseline, subject)
        magnitude = 1e3 * np.linalg.norm(base_data["hoist"], axis=1)
        hoisted = magnitude > HOIST_REPORT_CUT_MM
        for label, directory in deliveries.items():
            measured = dvc.measure(directory, "smoothed")[subject]
            floor = dvc.arm_aim_floor_per_frame(directory, subject)
            trunk_residual, trunk_length = dvc.trunk_length_floor(directory, subject)
            rows = {}
            for joint, floor_key in pairs:
                delta = measured[joint] - floor[floor_key]
                rows[joint] = {"whole": summary(np.abs(delta)),
                               "hoisted": summary(np.abs(delta[hoisted])),
                               "unhoisted": summary(np.abs(delta[~hoisted]))}
            neck = np.abs(measured["Neck"] - trunk_residual)
            rows["Neck"] = {"whole": summary(neck), "hoisted": summary(neck[hoisted]),
                            "unhoisted": summary(neck[~hoisted]),
                            "trunk_rest_length_mm": round(trunk_length, 2)}
            excess[label] = rows
        # THE WHOLE-TAKE MEDIAN DILUTES THIS CHANGE. 83 of 150 frames on performer 0 and
        # 128 of 150 on performer 1 are byte-identical between the arms, so the paired
        # median difference over the take is 0.0 by construction on more than half the
        # draws. The hoisted-frame cut is the one that carries the change, and the
        # placement difference there is reported beside the excess.
        hoisted_placement = {}
        measured = {label: dvc.measure(directory, "smoothed")[subject]
                    for label, directory in deliveries.items()}
        for joint in ("Neck", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm",
                      "RightLowerArm", "RightHand", "Head", "LeftUpperLeg", "LeftLowerLeg",
                      "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot"):
            delta = (measured[labels[1]][joint] - measured[labels[0]][joint])[hoisted]
            hoisted_placement[joint] = {
                "median_mm": round(float(np.median(delta)), 4) if delta.size else None,
                "p05_mm": round(float(np.percentile(delta, 5)), 4) if delta.size else None,
                "p95_mm": round(float(np.percentile(delta, 95)), 4) if delta.size else None,
                "frames": int(delta.size),
                "note": "negative means the candidate sits CLOSER to the landmark"}
        block["subjects"][key] = {
            "frames_hoisted": int(hoisted.sum()),
            "placement_difference_on_the_hoisted_frames_mm": hoisted_placement,
            "per_frame_excess_over_the_aim_floor_mm": excess,
            "joint_medians_and_paired_differences": {
                joint: joints[joint] for joint in
                ("Neck", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm",
                 "RightLowerArm", "RightHand", "LeftUpperLeg", "RightUpperLeg",
                 "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot", "Head")
                if joint in joints},
        }
    return block


# ------------------------------------------------------------------- the refactor tripwire
def tripwire_block(old: Path, new: Path) -> dict:
    """Two builds whose projection returned the PRE-hoist root, one per src state.

    Why an end-to-end pair and not an in-process assertion: pass A and the re-solve share
    one helper, so a refactor that changed the helper's arithmetic would move BOTH and an
    in-process "the re-solve changed nothing" check would still pass. The reference has to
    come from the OLD src, and it does. `zero-hoist` keeps the contacts and the foot LOCAL
    lock and replaces only the returned root: the re-solve still RUNS, on the pre-projection
    root, which is what the clause is for.
    """
    files = {name: {"old": digest(old / name), "new": digest(new / name)}
             for name in DELIVERED_FILES}
    for row in files.values():
        row["identical"] = row["old"] == row["new"]
    arrays = {}
    for subject in (0, 1):
        with np.load(old / f"subject-{subject:02d}.body-track.npz") as a, \
             np.load(new / f"subject-{subject:02d}.body-track.npz") as b:
            arrays[f"subject_{subject:02d}"] = {
                "local_rotations_bit_identical": bool(np.array_equal(
                    a["local_rotations_xyzw"], b["local_rotations_xyzw"])),
                "root_bit_identical": bool(np.array_equal(
                    a["root_translation_m"], b["root_translation_m"])),
                "contacts_identical": bool(np.array_equal(
                    a["foot_contacts"], b["foot_contacts"])),
            }
    return {"old_src_build": str(old), "new_src_build": str(new),
            "delivered_files": files,
            "all_eight_byte_identical": all(r["identical"] for r in files.values()),
            "arrays": arrays,
            "passes": bool(all(r["identical"] for r in files.values())
                           and all(all(v.values()) for v in arrays.values()))}


# ----------------------------------------------------------------------------- the oracle
def oracle_block(tag: str, save: Path, reference: Path | None,
                 zero_hoist: bool = False) -> dict:
    """The D3 gate's own six bodies, through the D3 gate's own code, under watchers.

    `zero_hoist` is O4, the refactor tripwire on the oracle: the projection runs in full
    and the root it returns is replaced by the root that went in, with `foot_contacts`
    cleared so `validate_body_track`'s 1 e-5 m contact assertion does not refuse a locked
    foot that was not moved. Under the PREVIOUS src the delivered track of such a run was,
    definitionally, `replace(projected, root=pre_root, contacts=0)` -- there was no code
    after the projection -- so the reference is the `post_rotations` and `pre_root` this
    same instrument recorded on the D8c run, and no second old-src run is needed.
    """

    save.mkdir(parents=True, exist_ok=True)
    calls: list[dict] = []
    projections: list[dict] = []
    rejects: list[dict] = []
    real_converter = cm.positions_to_body_track
    real_projection = cm.project_generated_foot_contacts
    real_reject = cm._reachable_clavicle_sequence

    def converter(positions, **kwargs):
        # The bookkeeping is by call WINDOW, not by a marker comparison: the re-solve runs
        # pass B AFTER the projection, so a reject recorded by this converter call can sit
        # either side of it, and only "everything appended while this call ran" is right.
        first_projection, first_reject = len(projections), len(rejects)
        track = real_converter(positions, **kwargs)
        calls.append({"positions": np.asarray(positions, dtype=np.float64),
                      "skeleton": kwargs.get("skeleton"),
                      "track": track,
                      "projection_index": first_projection,
                      "reject_indices": list(range(first_reject, len(rejects)))})
        return track

    def projection(track, **kwargs):
        projected, diagnostics = real_projection(track, **kwargs)
        projections.append({"pre": track, "post": projected,
                            "diagnostics": diagnostics.as_dict()})
        if zero_hoist:
            from dataclasses import replace as _replace
            projected = _replace(
                projected,
                root_translation_m=np.array(track.root_translation_m, copy=True),
                foot_contacts=np.zeros_like(np.asarray(projected.foot_contacts)))
        return projected, diagnostics

    def reject(local_rotations, parent_world_rotations, ceiling):
        replaced, accepted = real_reject(
            local_rotations, parent_world_rotations, ceiling)
        rejects.append({"after": len(projections),
                        "rejected": [int(i) for i in np.flatnonzero(~accepted)],
                        "sha256": sha256(
                            np.ascontiguousarray(accepted).tobytes()).hexdigest()})
        return replaced, accepted

    scratch = d3.SCRATCH
    d3.SCRATCH = save / "scratch"
    d3.SCRATCH.mkdir(parents=True, exist_ok=True)
    cm.positions_to_body_track = converter
    cm.project_generated_foot_contacts = projection
    cm._reachable_clavicle_sequence = reject
    d3_report: dict = {}
    try:
        d3.oracle_block(d3_report)
    finally:
        cm.positions_to_body_track = real_converter
        cm.project_generated_foot_contacts = real_projection
        cm._reachable_clavicle_sequence = real_reject
        d3.SCRATCH = scratch

    # `oracle_block` runs the converter twice per seed: the exact skeleton, then the
    # global-scale ALTERNATIVE. Only the first is the oracle.
    per_seed = d3_report["exact_skeleton_oracle"]["seeds"]
    assert len(calls) == 2 * len(per_seed), (len(calls), len(per_seed))
    block: dict = {
        "d3_gate_bands": d3_report["exact_skeleton_oracle"]["bands"],
        "d3_gate_worst_legs_mm": d3_report["exact_skeleton_oracle"]["worst_legs_mm"],
        "d3_gate_worst_arms_mm": d3_report["exact_skeleton_oracle"]["worst_arms_mm"],
        "d3_gate_passes": d3_report["exact_skeleton_oracle"]["passes"],
        "gauge": (
            "`retarget_cost.score` subtracts each frame's leg-root midpoint from the rig "
            "and each frame's hip midpoint from the reference. That is a per-frame "
            "TRANSLATION, so the aligned score cannot see a root move at all, and after a "
            "correct re-aim from the hoisted origin the re-aimed bones miss the ALIGNED "
            "target by the perpendicular part of the hoist. `absolute_mm` beside it is the "
            "companion row; neither band moves in this step."),
        "seeds": [],
    }
    for position, seed_row in enumerate(per_seed):
        call = calls[2 * position]
        skeleton = call["skeleton"]
        # The gate hands the converter `rc.Z_UP_FROM_Y_UP(landmarks)`; `rc.Y_UP_FROM_Z_UP`
        # is that map's own inverse, so the truth comes back exactly, not re-derived.
        landmarks = rc.Y_UP_FROM_Z_UP(call["positions"])
        record = projections[call["projection_index"]]
        pre, post = record["pre"], record["post"]
        hoist = (np.asarray(post.root_translation_m, np.float64)
                 - np.asarray(pre.root_translation_m, np.float64))
        magnitude = 1e3 * np.linalg.norm(hoist, axis=1)
        hoisted = magnitude > HOIST_REPORT_CUT_MM
        delivered = call["track"]
        fk_delivered = forward_kinematics_positions(
            np.asarray(delivered.root_translation_m, np.float64),
            np.asarray(delivered.local_rotations_xyzw, np.float64),
            skeleton=skeleton).astype(np.float64)
        fk_pre = forward_kinematics_positions(
            np.asarray(pre.root_translation_m, np.float64),
            np.asarray(pre.local_rotations_xyzw, np.float64),
            skeleton=skeleton).astype(np.float64)
        aligned = rc.score(fk_delivered, landmarks, skeleton)
        absolute = absolute_score(fk_delivered, landmarks, skeleton)
        rays = {}
        for parent, child, landmark, family in ROOT_DEPENDENT:
            origin = fk_delivered[:, skeleton.index(parent)]
            axis = fk_delivered[:, skeleton.index(child)] - origin
            axis = axis / np.linalg.norm(axis, axis=1)[:, None]
            target = landmarks[:, cm.JOINT_INDEX[landmark]]

            def miss(from_origin):
                delta = target - from_origin
                return 1e3 * np.linalg.norm(
                    delta - np.sum(delta * axis, axis=1)[:, None] * axis, axis=1)

            rays[f"{parent}->{landmark}"] = {
                "family": family,
                "miss_from_delivered_origin_mm": {
                    "hoisted": summary(miss(origin)[hoisted]),
                    "unhoisted": summary(miss(origin)[~hoisted])},
                "miss_from_pre_hoist_origin_mm": {
                    "hoisted": summary(miss(origin - hoist)[hoisted])},
            }
        np.savez_compressed(
            save / f"oracle-{seed_row['seed']}.npz",
            root=np.asarray(delivered.root_translation_m),
            rotations=np.asarray(delivered.local_rotations_xyzw),
            contacts=np.asarray(delivered.foot_contacts),
            pre_root=np.asarray(pre.root_translation_m),
            pre_rotations=np.asarray(pre.local_rotations_xyzw),
            post_root=np.asarray(post.root_translation_m),
            post_rotations=np.asarray(post.local_rotations_xyzw),
            landmarks=landmarks,
            rest=np.asarray(skeleton.rest_translations_m))
        row = {
            "seed": seed_row["seed"],
            "contacts": record["diagnostics"]["contact_frames"],
            "penetration_before_mm": round(
                1e3 * record["diagnostics"]["ground_penetration_before_m"], 4),
            "max_correction_mm": round(
                1e3 * record["diagnostics"]["maximum_root_correction_m"], 4),
            "hoist_mm": summary(magnitude),
            "frames_over_report_cut": int(hoisted.sum()),
            "frames_under_zero_cut": int((magnitude < ZERO_CUT_MM).sum()),
            "frames_below_the_report_cut_but_not_zero": int(
                ((magnitude <= HOIST_REPORT_CUT_MM) & (magnitude >= ZERO_CUT_MM)).sum()),
            "clavicle_rejected": [rejects[i]["rejected"] for i in call["reject_indices"]],
            "clavicle_accepted_sha256": [rejects[i]["sha256"][:16]
                                         for i in call["reject_indices"]],
            "d3_gate_aligned_glb_vs_truth_mm": seed_row["glb_vs_truth_mm"],
            "aligned_fk_groups_mm": d3.groups_mm(aligned),
            "aligned_fk_groups_hoisted_mm": groups_masked(aligned, hoisted),
            "aligned_fk_groups_p95_mm": groups_p95(aligned),
            "absolute_fk_groups_mm": d3.groups_mm(absolute),
            "absolute_fk_groups_hoisted_mm": groups_masked(absolute, hoisted),
            "absolute_fk_groups_p95_mm": groups_p95(absolute),
            "aligned_groups_pre_projection_mm": d3.groups_mm(
                rc.score(fk_pre, landmarks, skeleton)),
            "aligned_groups_hoist_removed_mm": d3.groups_mm(
                rc.score(fk_delivered - hoist[:, None, :], landmarks, skeleton)),
            "rays": rays,
        }
        if reference is not None:
            with np.load(reference / f"oracle-{seed_row['seed']}.npz") as before:
                base_root = before["pre_root"] if zero_hoist else before["root"]
                base_contacts = (np.zeros_like(before["contacts"]) if zero_hoist
                                 else before["contacts"])
                frozen = {
                    "root_bit_identical": bool(np.array_equal(
                        base_root, np.asarray(delivered.root_translation_m))),
                    "contacts_identical": bool(np.array_equal(
                        base_contacts, np.asarray(delivered.foot_contacts))),
                    "landmarks_identical": bool(np.array_equal(
                        before["landmarks"], landmarks)),
                }
                names = list(skeleton.names)
                base_rot = (before["post_rotations"] if zero_hoist
                            else before["rotations"])
                cand_rot = np.asarray(delivered.local_rotations_xyzw)
                moved = ~np.all(base_rot == cand_rot, axis=2)
                frozen["frozen_joints_identical"] = bool(all(
                    not moved[:, names.index(n)].any() for n in FROZEN_JOINTS))
                frozen["joints_that_moved"] = [
                    n for n in names if moved[:, names.index(n)].any()]
                frozen["identical_on_unhoisted_frames"] = bool(
                    not moved[~hoisted].any())
                frozen["all_rotations_bit_identical"] = bool(not moved.any())
                row["vs_reference"] = frozen
        block["seeds"].append(row)
    block["tag"] = tag
    return block


# ------------------------------------------------------------- the mechanical merge rule
CLOSURE_BAND_MM = 0.01     # the float32 forward-kinematics floor, measured at 0.0005 mm


def merge_rule(report: dict) -> dict:
    """Every conjunct of the card's merge rule, evaluated from this report and nothing else.

    `hygiene AND the tripwire AND B1 AND B2 AND B3 on both performers AND B4 AND O1 AND O2`.
    O3 and B5 REPORT. B3 lives in `d9b_hoist_silhouette.py` and is folded in by the caller;
    hygiene lives in `d9b_hoist_delivery.py`. Both are named here with their file so the
    verdict can never be read without them.
    """
    conjuncts: dict = {}
    tripwire = report.get("refactor_tripwire")
    if tripwire is not None:
        conjuncts["tripwire"] = {
            "clause": "with the hoist forced to zero, the eight delivered files are "
                      "byte-identical between the previous src and this one",
            "measured": tripwire["all_eight_byte_identical"],
            "verdict": "PASS" if tripwire["passes"] else "FAIL"}
    b1 = report.get("B1_excess_over_the_aim_floor")
    if b1 is not None:
        worst = 0.0
        for row in b1["subjects"].values():
            for joint in row["per_frame_excess_over_the_aim_floor_mm"].get("D9b", {}).values():
                worst = max(worst, float(joint["whole"]["max"]))
        conjuncts["B1_excess_over_the_aim_floor"] = {
            "clause": "the per-frame excess over the arm-aim and trunk-length floors is "
                      f"0 on EVERY frame (max, not median), banded at {CLOSURE_BAND_MM} mm "
                      "-- the float32 forward-kinematics floor",
            "worst_max_mm": round(worst, 4),
            "verdict": "PASS" if worst <= CLOSURE_BAND_MM else "FAIL"}
        conjuncts["B1_same_denominator"] = {
            "clause": "the triangulated landmarks are byte-identical across the arms; "
                      "pre-registered as an expected PASS, CHANGED means the change did "
                      "more than re-aim",
            "measured": b1["same_denominator"],
            "verdict": "PASS" if b1["same_denominator"] else "FAIL"}
    b2 = report.get("B2_bit_identity")
    if b2 is not None:
        rows = b2["subjects"].values()
        checks = {
            "root_identical": all(r["root_identical"] for r in rows),
            "contacts_identical": all(r["contacts_identical"] for r in rows),
            "landmarks_identical": all(r["smoothed_landmarks_identical"]
                                       and r["raw_landmarks_identical"] for r in rows),
            "frozen_joints_identical": all(r["frozen_joints_identical"] for r in rows),
            "everything_identical_on_unhoisted_frames": all(
                r["everything_identical_on_unhoisted_frames"] for r in rows),
            "moved_set_within_the_resolved_set": all(
                r["moved_set_is_within_the_resolved_set"] for r in rows)}
        conjuncts["B2_bit_identity"] = {
            "clause": "root, contacts, Hips, legs, feet and toes identical on every frame; "
                      "every joint identical on every unhoisted frame; the moved joints "
                      "only from Spine up and out the arms",
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL"}
    b4 = report.get("B4_contacts_and_the_plant")
    if b4 is not None and all("identical" in r for r in b4["subjects"].values()):
        identical = all(r["identical"] for r in b4["subjects"].values())
        conjuncts["B4_contacts_and_the_plant"] = {
            "clause": "the contact count, the runs, and the foot travel over each planted "
                      "run are IDENTICAL by construction",
            "measured": identical,
            "verdict": "PASS" if identical else "FAIL"}
    oracle = report.get("oracle")
    if oracle is not None:
        worst_ray = max(
            float(row["miss_from_delivered_origin_mm"]["hoisted"].get("max", 0.0) or 0.0)
            for seed in oracle["seeds"] for row in seed["rays"].values())
        conjuncts["O1_absolute_ray_miss"] = {
            "clause": "on all six exact-skeleton bodies the absolute ray miss from the "
                      "DELIVERED origin, on the arms, the clavicles and the trunk, is at "
                      f"the float32 floor on every hoisted frame (banded {CLOSURE_BAND_MM} "
                      "mm; the card's weaker form, 'at or below the unhoisted-frame "
                      "floor', is satisfied a fortiori)",
            "worst_max_mm": round(worst_ray, 4),
            "verdict": "PASS" if worst_ray <= CLOSURE_BAND_MM else "FAIL"}
        frozen = [seed.get("vs_reference") for seed in oracle["seeds"]]
        if all(frozen):
            checks = {
                "root_bit_identical": all(f["root_bit_identical"] for f in frozen),
                "contacts_identical": all(f["contacts_identical"] for f in frozen),
                "landmarks_identical": all(f["landmarks_identical"] for f in frozen),
                "frozen_joints_identical": all(f["frozen_joints_identical"] for f in frozen)}
            conjuncts["O2_frozen_on_the_oracle"] = {
                "clause": "the legs' rotations, the root, the contacts and the hoist are "
                          "bit-identical per seed",
                "checks": checks,
                "note": ("`identical_on_unhoisted_frames` is REPORTED and is not this "
                         "clause: on the oracle the 0.5 mm cut is a report cut and frames "
                         "carrying a real sub-cut hoist legitimately move, seed 20260906's "
                         "0.0262 mm whole-take penetration lift among them"),
                "verdict": "PASS" if all(checks.values()) else "FAIL"}
    conjuncts["hygiene"] = {
        "clause": "today's code rebuilds the shipped delivery byte-identically, 8 of 8",
        "where": "artifacts/compare/d9b-hoist/delivery-hygiene-build.json",
        "verdict": "SEE THAT FILE"}
    conjuncts["B3_photographs"] = {
        "clause": "part-wise silhouette, ARMS and TORSO+LEGS, whole take and the bent "
                  "tercile, both performers, not worse with the CI's upper bound >= 0",
        "where": "artifacts/compare/d9b-hoist/silhouette-partwise.json",
        "verdict": "SEE THAT FILE"}
    decidable = [row["verdict"] for row in conjuncts.values()
                 if row["verdict"] in ("PASS", "FAIL")]
    return {
        "rule": ("hygiene AND the tripwire AND B1 (excess -> 0 on every frame, same "
                 "denominator PASS) AND B2 AND B3 on both performers AND B4 AND O1 AND "
                 "O2. O3 and B5 report."),
        "conjuncts": conjuncts,
        "conjuncts_decided_here": "FAIL" if "FAIL" in decidable else "PASS",
        "outstanding": [name for name, row in conjuncts.items()
                        if row["verdict"] not in ("PASS", "FAIL")],
    }


# ------------------------------------------------------------------------------- the main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=SHIPPED)
    parser.add_argument("--baseline-label", default="D8c")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-label", default="D9b")
    parser.add_argument("--tripwire", nargs=2, type=Path, metavar=("OLD", "NEW"))
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--oracle-tag", default="")
    parser.add_argument("--oracle-save", type=Path)
    parser.add_argument("--oracle-reference", type=Path)
    parser.add_argument("--oracle-zero-hoist", action="store_true",
                        help="O4: the refactor tripwire on the six oracle bodies")
    parser.add_argument("--degenerate", action="append", default=[],
                        metavar="NAME=DIR")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report: dict = {
        "title": "D9b -- the foot-contact hoist re-aim",
        "report_cut_mm": HOIST_REPORT_CUT_MM,
        "report_cut_note": (
            "a REPORT cut and never control flow: the re-solve runs on every frame, and "
            "the tripwire is what says the zero-hoist frames do not move."),
    }
    report[f"build_{args.baseline_label}"] = build_block(args.baseline, args.baseline_label)
    if args.candidate is not None:
        report[f"build_{args.candidate_label}"] = build_block(
            args.candidate, args.candidate_label)
        report["B1_excess_over_the_aim_floor"] = excess_block(
            args.baseline, args.candidate, (args.baseline_label, args.candidate_label))
        report["B2_bit_identity"] = bit_identity(args.baseline, args.candidate)
        report["B4_contacts_and_the_plant"] = plant_block(args.baseline, args.candidate)
    else:
        report["B4_contacts_and_the_plant"] = plant_block(args.baseline, None)
    if args.tripwire is not None:
        report["refactor_tripwire"] = tripwire_block(*args.tripwire)
    for entry in args.degenerate:
        name, _, path = entry.partition("=")
        directory = Path(path)
        block = {"build": build_block(directory, name)}
        block["B2_bit_identity"] = bit_identity(args.baseline, directory)
        block["B4_contacts_and_the_plant"] = plant_block(args.baseline, directory)
        report.setdefault("degenerates", {})[name] = block
    if args.oracle:
        report["oracle"] = oracle_block(
            args.oracle_tag or args.candidate_label,
            args.oracle_save or (OUT_DIR / "oracle"),
            args.oracle_reference,
            args.oracle_zero_hoist)
    report["merge_rule"] = merge_rule(report)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
