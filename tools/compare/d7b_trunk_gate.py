#!/usr/bin/env python3
"""D7b's gate: the trunk chain re-solved onto the captured neck after the pelvis frame.

Every band from the card, each as its own keyed block, with the degenerate that fails it
beside it. The card in `docs/LADDER_EXECUTION_PLAN.md` section 2 is the pre-registration
and it is frozen: where a number refutes a prediction the prediction is RECORDED as
refuted and the band is not moved.

WHAT THIS GATE READS, AND FROM WHERE
  * the DELIVERED files' own bytes, through `delivered_vs_capture.py` -- every placement
    figure. `retarget_cost.py` cannot see this class of change and is labelled, not fixed.
  * the two rebuilt tracks, for the bit-identity bands. Same detections, same
    triangulation, asserted byte-identical.
  * this step's own part-wise silhouette wrapper, for the photographs.
  * the committed instruments for the reported arms: `mamma_scoreboard.py` (rung 11),
    `facing_location.py` (facing and the 16 handedness signs), `head_gate.py`.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7b_trunk_gate.py

Writes `artifacts/compare/d7b-trunk/gate.json`. NOTHING is written under
`artifacts/commercial-multiview-soma77/`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import d3_skeleton_gate as d3  # noqa: E402
import delivered_vs_capture as dvc  # noqa: E402
from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import (  # noqa: E402
    BodyTrack, DETAILED_HUMANOID, forward_kinematics_positions, skeleton_for_track)

OUT_DIR = ROOT / "artifacts/compare/d7b-trunk"
REPORT = OUT_DIR / "gate.json"
MEASURED = OUT_DIR / "delivered-vs-capture.json"

D3_DIR = ROOT / "artifacts/compare/delivered-before-d7-2026-09-05"
D7_DIR = ROOT / "artifacts/commercial-multiview-soma77"
D7B_DIR = OUT_DIR / "delivery"

FLOOR_BAND_MM = 5.0            # B1: the neck within this of the length floor
CLOSURE_BAND_M = 1.0e-6        # B5: the D3 closure on the rebuilt GLB, from its own bytes
HEAD_BAND = 1.0e-9             # B5: Head WORLD orientation unchanged, per frame
TURN_CEILING_DEG_PER_FRAME = 60.0   # B6, reported: the torso frame's own travel

# The card, verbatim. Frozen before any number below existed.
PRE_REGISTRATION = {
 "source": "docs/LADDER_EXECUTION_PLAN.md section 2, the D7b card, written 2026-09-05",
 "floor": ("the length residual |L_rest - ||neck - Spine_origin|| | under the D7 pelvis is "
           "20.9 / 41.7 mm median whole-take / bent tercile on performer 0 and 18.3 / 11.9 "
           "on performer 1 (L_rest 448 / 401 mm) -- the trunk chord shortens when the spine "
           "flexes and a rigid straight chain cannot follow; that share is handed to D5"),
 "B1": ("the placement claim; the candidate optimises it, so it is paired with the floor "
        "and the photographs: Neck-from-file median within 5 mm of the length floor on both "
        "performers, whole take and bent tercile, and below D7 with the CI clear (predict "
        "21 / 42 and 18 / 12; against D3 whole-take performer 0 is predicted slightly WORSE, "
        "14 -> 21, because D3's chain lay on the trunk line by construction, and better on "
        "every bent cut, 105 -> 42)"),
 "B1_must_fail": ("the shipped D7 (59 / 44) and a root-translation degenerate that zeroes "
                  "the neck by moving the hips"),
 "B2": ("untouchable: root translation, Hips, UpperLeg, LowerLeg locals bit-identical to "
        "D7; hips / knees / ankles from the file unchanged; feet locals reported"),
 "B3": ("shoulders / elbows / wrists from the file not worse than D7 with CI, predicted to "
        "improve on bent frames"),
 "B4": ("the photographs: part-wise silhouette on the tilt terciles, torso AND arms not "
        "worse than D7 on EITHER performer with the CI clear, improvement predicted on "
        "performer 0's bent tercile torso; MAMMA mesh bit-identical"),
 "B5": ("Head WORLD orientation unchanged to 1e-9 on every frame (the locals of Neck/Head "
        "change with UpperChest, so the head gate is RERUN and its figures reported, "
        "byte-equality not claimed); D3 closure on the rebuilt GLB from bytes <= 1e-6 m; "
        "canonical round trip legs 0.00, torso and arms reported before / after, not banded"),
 "B6": ("reported: rung 11 vs MAMMA per joint; facing dots; all 16 handedness signs; frames "
        "where the torso frame turns > 60 deg/frame (predict 0)"),
 "B7": ("synthetic (SOMASKEL77 posed clips, I7 noise): neck placement error of "
        "aim-from-Spine-origin vs aim-from-hip-mid under the Kabsch pelvis; clean arm must "
        "reach the length floor to 1e-6"),
 "merge_rule": ("fixed before numbers: B1 on both performers AND B2 exact AND B4 on both "
                "performers AND B5; B3, B6, B7 report; any failed clause stated in the "
                "review"),
}


# ------------------------------------------------------------------------------- helpers
def load_track(directory: Path, subject: int) -> BodyTrack:
    return BodyTrack.from_dict(json.loads(
        (directory / f"subject-{subject:02d}.body-track.json").read_text()))


def world_rotations(track: BodyTrack) -> np.ndarray:
    """Per-joint WORLD quaternions, composed down the skeleton's own parent chain."""

    local = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    skeleton = skeleton_for_track(track)
    world = np.zeros_like(local)
    for index, joint in enumerate(skeleton.joints):
        world[:, index] = (local[:, index] if joint.parent == -1
                           else cm._quaternion_multiply(world[:, joint.parent],
                                                        local[:, index]))
    return world


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("nij,nkj->nik", a, b)
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def measured() -> dict:
    """The delivered-vs-capture report, regenerated here so it can never go stale."""

    payload = dvc.report({"D3": D3_DIR, "D7": D7_DIR, "D7b": D7B_DIR})
    MEASURED.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def joint_row(payload: dict, subject: int, joint: str) -> dict:
    return payload["subjects"][f"subject_{subject:02d}"]["joints"][joint]


# ---------------------------------------------------------------------------- THE FLOOR
def floor_block(report: dict, payload: dict) -> None:
    block: dict = {
        "what_it_is": ("| L_rest - ||neck_landmark - Spine_origin|| |, per frame, from each "
                       "delivery's own Spine origin. Spine, Chest and UpperChest share one "
                       "world rotation and their rests are collinear +Y, so the best any "
                       "aim can do is put the Neck ON the ray to the landmark; the LENGTH "
                       "difference remains and no aim can remove it."),
        "belongs_to": "D5 -- spine ratios and a flexible chain",
        "pre_registered_values_mm": {"subject_00": [20.9, 41.7], "subject_01": [18.3, 11.9],
                                     "as": "median whole take, median bent tercile, under D7"},
        "subjects": {},
    }
    ok = True
    for subject in (0, 1):
        row = payload["subjects"][f"subject_{subject:02d}"]["trunk_length_floor"]
        expected = block["pre_registered_values_mm"][f"subject_{subject:02d}"]
        got = [row["D7"]["median_mm"], row["D7"]["bent_tercile_median_mm"]]
        matches = all(abs(a - b) <= 0.1 for a, b in zip(expected, got))
        ok = ok and matches
        block["subjects"][f"subject_{subject:02d}"] = {
            "per_delivery": row,
            "reproduces_the_pre_registered_floor_to_0_1_mm": bool(matches),
            "d7b_floor_equals_d7_floor": bool(
                abs(row["D7"]["median_mm"] - row["D7b"]["median_mm"]) < 1e-9
                and abs(row["D7"]["bent_tercile_median_mm"]
                        - row["D7b"]["bent_tercile_median_mm"]) < 1e-9),
            "why_they_are_equal": ("the Spine ORIGIN does not move -- `Hips` and the root "
                                   "formula are untouched -- so the floor is the same "
                                   "quantity on both arms and B1 is a fair comparison."),
            "distributed_flexion_would_recover":
                payload["subjects"][f"subject_{subject:02d}"][
                    "distributed_flexion_would_recover"]["D7b"],
        }
    block["verdict"] = "PASS" if ok else "FAIL"
    report["floor"] = block


# --------------------------------------------------------- B1: the placement claim
def root_translation_degenerate() -> dict:
    """THE MUST-FAIL B2 IS FOR. A root shift that puts the neck exactly on its landmark.

    It costs nothing to build and it scores a PERFECT neck: translate the root, per frame,
    by `neck_landmark - Neck_origin`. B1 alone would wave it through -- which is the whole
    reason B1 is paired with B2 and with the photographs. Computed by forward kinematics of
    the D7b track with a modified root; nothing is exported and nothing is written.
    """

    out: dict = {}
    for subject in (0, 1):
        track = load_track(D7B_DIR, subject)
        skeleton = skeleton_for_track(track)
        roots = np.asarray(track.root_translation_m, np.float64)
        rotations = np.asarray(track.local_rotations_xyzw, np.float64)
        world = forward_kinematics_positions(roots, rotations, skeleton=skeleton)
        capture = dvc.capture_positions(D7B_DIR, subject)
        to_capture = lambda v: np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)
        neck = capture[:, cm.JOINT_INDEX["neck"]]
        delta_capture = neck - to_capture(world[:, track.joint_names.index("Neck")])
        # back into the rig's Y-up world: capture (x, y, z) -> rig (x, z, -y)
        delta_rig = np.stack([delta_capture[:, 0], delta_capture[:, 2],
                              -delta_capture[:, 1]], axis=-1)
        moved = forward_kinematics_positions(roots + delta_rig, rotations, skeleton=skeleton)
        moved_capture = to_capture(moved)
        error = lambda name, landmark: 1000.0 * np.linalg.norm(
            moved_capture[:, track.joint_names.index(name)]
            - capture[:, cm.JOINT_INDEX[landmark]], axis=1)
        out[f"subject_{subject:02d}"] = {
            "neck_median_mm": round(float(np.median(error("Neck", "neck"))), 4),
            "left_hip_median_mm": round(float(np.median(error("LeftUpperLeg", "left_hip"))), 3),
            "right_hip_median_mm": round(float(np.median(error("RightUpperLeg", "right_hip"))), 3),
            "left_knee_median_mm": round(float(np.median(error("LeftLowerLeg", "left_knee"))), 3),
            "left_ankle_median_mm": round(float(np.median(error("LeftFoot", "left_ankle"))), 3),
            "root_translation_bit_identical_to_D7": False,
            "root_moved_median_mm": round(
                1000.0 * float(np.median(np.linalg.norm(delta_rig, axis=1))), 3),
        }
    return out


def b1_block(report: dict, payload: dict) -> None:
    block: dict = {
        "band": PRE_REGISTRATION["B1"],
        "reference": dvc.REFERENCE,
        "why_it_is_paired": ("the candidate optimises this directly -- the aim IS at this "
                             "landmark -- so on its own it proves nothing. It is paired "
                             "with the length floor (which no aim can move), with B2 (the "
                             "lower body must not have been traded away) and with the "
                             "photographs (which the candidate cannot optimise)."),
        "subjects": {},
    }
    passes = {}
    for subject in (0, 1):
        row = joint_row(payload, subject, "Neck")
        floor = payload["subjects"][f"subject_{subject:02d}"]["trunk_length_floor"]["D7b"]
        difference = row["paired_differences"]["D7b_minus_D7"]
        whole_gap = row["D7b"]["median_mm"] - floor["median_mm"]
        bent_gap = (row["D7b"]["bent_tercile_median_mm"]
                    - floor["bent_tercile_median_mm"])
        within = abs(whole_gap) <= FLOOR_BAND_MM and abs(bent_gap) <= FLOOR_BAND_MM
        below = bool(difference["ci95_mm"][1] < 0.0)
        passes[subject] = within and below
        block["subjects"][f"subject_{subject:02d}"] = {
            "neck_median_mm": {label: row[label]["median_mm"] for label in ("D3", "D7", "D7b")},
            "neck_bent_tercile_median_mm": {
                label: row[label]["bent_tercile_median_mm"] for label in ("D3", "D7", "D7b")},
            "neck_p95_mm": {label: row[label]["p95_mm"] for label in ("D3", "D7", "D7b")},
            "length_floor_mm": [floor["median_mm"], floor["bent_tercile_median_mm"]],
            "gap_to_the_floor_mm": [round(whole_gap, 3), round(bent_gap, 3)],
            "within_5_mm_of_the_floor": bool(within),
            "D7b_minus_D7": difference,
            "below_D7_with_the_CI_clear": below,
            "D7b_minus_D3": row["paired_differences"]["D7b_minus_D3"],
            "verdict": "PASS" if passes[subject] else "FAIL",
        }
    block["must_fail"] = {
        "the_shipped_D7": {
            "neck_median_mm": {f"subject_{s:02d}": joint_row(payload, s, "Neck")["D7"]["median_mm"]
                               for s in (0, 1)},
            "gap_to_its_own_floor_mm": {
                f"subject_{s:02d}": round(
                    joint_row(payload, s, "Neck")["D7"]["median_mm"]
                    - payload["subjects"][f"subject_{s:02d}"]["trunk_length_floor"]["D7"]["median_mm"], 3)
                for s in (0, 1)},
            "fails_the_5_mm_band": all(
                abs(joint_row(payload, s, "Neck")["D7"]["median_mm"]
                    - payload["subjects"][f"subject_{s:02d}"]["trunk_length_floor"]["D7"]["median_mm"])
                > FLOOR_BAND_MM for s in (0, 1)),
        },
        "the_root_translation_degenerate": {
            "what": ("the root translated per frame by `neck_landmark - Neck_origin`. It "
                     "scores a PERFECT neck and B1 alone cannot reject it."),
            "figures": root_translation_degenerate(),
            "B1_would_pass_it": True,
            "B2_catches_it": ("its root translation is not bit-identical to D7 by "
                              "construction, and its hips / knees / ankles move by tens of "
                              "millimetres -- see B2's `degenerate_scored_through_B2`."),
        },
    }
    block["verdict"] = "PASS" if all(passes.values()) else "FAIL"
    report["B1_placement"] = block


# ------------------------------------------------------------------- B2: untouchable
def b2_block(report: dict, payload: dict) -> None:
    exact: dict = {}
    for subject in (0, 1):
        d7, d7b = load_track(D7_DIR, subject), load_track(D7B_DIR, subject)
        names = d7.joint_names
        row = {
            "root_translation_m": bool(np.array_equal(
                d7.root_translation_m, d7b.root_translation_m)),
            "foot_contacts": bool(np.array_equal(d7.foot_contacts, d7b.foot_contacts)),
            "rest_translations_m": bool(np.array_equal(
                d7.rest_translations_m, d7b.rest_translations_m)),
        }
        for name in ("Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg",
                     "RightLowerLeg"):
            index = names.index(name)
            row[f"local_{name}"] = bool(np.array_equal(
                d7.local_rotations_xyzw[:, index], d7b.local_rotations_xyzw[:, index]))
        reported = {}
        for name in ("LeftFoot", "RightFoot", "LeftToes", "RightToes"):
            index = names.index(name)
            a = np.asarray(d7.local_rotations_xyzw[:, index], np.float64)
            b = np.asarray(d7b.local_rotations_xyzw[:, index], np.float64)
            reported[name] = {
                "bit_identical": bool(np.array_equal(
                    d7.local_rotations_xyzw[:, index], d7b.local_rotations_xyzw[:, index])),
                "worst_deg": round(float(geodesic_deg(
                    Rotation.from_quat(a).as_matrix(),
                    Rotation.from_quat(b).as_matrix()).max()), 6),
            }
        # ... and from the delivered FILE, which is the figure that matters
        from_file = {}
        for joint in ("LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
                      "LeftFoot", "RightFoot", "leg_root_midpoint_vs_hip_midpoint",
                      "hips_joint_vs_hip_midpoint"):
            cell = joint_row(payload, subject, joint)
            from_file[joint] = {
                "D7_median_mm": cell["D7"]["median_mm"],
                "D7b_median_mm": cell["D7b"]["median_mm"],
                "unchanged": bool(abs(cell["D7"]["median_mm"] - cell["D7b"]["median_mm"]) < 1e-9),
            }
        exact[f"subject_{subject:02d}"] = {
            "track_bit_identity": row,
            "feet_reported_never_banded": reported,
            "from_the_delivered_file": from_file,
            "all_banded_arrays_bit_identical": all(row.values()),
            "hips_knees_ankles_unchanged_from_the_file": all(
                v["unchanged"] for v in from_file.values()),
        }
    degenerate = report["B1_placement"]["must_fail"]["the_root_translation_degenerate"]["figures"]
    caught = {}
    for subject in (0, 1):
        d7 = joint_row(payload, subject, "LeftUpperLeg")["D7"]["median_mm"]
        caught[f"subject_{subject:02d}"] = {
            "left_hip_D7_mm": d7,
            "left_hip_degenerate_mm": degenerate[f"subject_{subject:02d}"]["left_hip_median_mm"],
            "root_translation_bit_identical": False,
            "B2_rejects_it": bool(
                abs(degenerate[f"subject_{subject:02d}"]["left_hip_median_mm"] - d7) > 1e-9),
        }
    ok = all(v["all_banded_arrays_bit_identical"]
             and v["hips_knees_ankles_unchanged_from_the_file"] for v in exact.values())
    report["B2_untouchable"] = {
        "band": PRE_REGISTRATION["B2"],
        "how": ("the two rebuilt tracks compared array by array. The D7 arm is the shipped "
                "delivery; the D7b arm is this branch's rebuild from the same cached "
                "detections, with the triangulated landmarks asserted byte-identical."),
        "subjects": exact,
        "degenerate_scored_through_B2": caught,
        "verdict": "PASS" if ok else "FAIL",
    }


# ------------------------------------------------------------------------ B3: the arms
def b3_block(report: dict, payload: dict) -> None:
    joints = ("LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm",
              "LeftHand", "RightHand")
    subjects: dict = {}
    ok = True
    for subject in (0, 1):
        rows = {}
        for joint in joints:
            cell = joint_row(payload, subject, joint)
            difference = cell["paired_differences"]["D7b_minus_D7"]
            not_worse = bool(difference["ci95_mm"][0] <= 0.0)
            ok = ok and not_worse
            rows[joint] = {
                "median_mm": {label: cell[label]["median_mm"] for label in ("D3", "D7", "D7b")},
                "bent_tercile_median_mm": {
                    label: cell[label]["bent_tercile_median_mm"] for label in ("D3", "D7", "D7b")},
                "D7b_minus_D7": difference,
                "not_worse_than_D7": not_worse,
                "improved_on_the_bent_tercile": bool(
                    difference["bent_tercile_ci95_mm"][1] < 0.0),
            }
        subjects[f"subject_{subject:02d}"] = rows
    report["B3_arms"] = {
        "band": PRE_REGISTRATION["B3"],
        "reference": dvc.REFERENCE,
        "why_they_move_at_all": ("D2 aims the clavicle from the FK'd Shoulder origin, and "
                                 "that origin hangs off UpperChest. Move the trunk chain "
                                 "and the whole arm follows; the elbow and wrist are then "
                                 "RE-SOLVED in pass C from the new parent."),
        "subjects": subjects,
        "verdict": "PASS" if ok else "FAIL",
    }


# ------------------------------------------------------------ B4: the photographs
def b4_block(report: dict) -> None:
    path = OUT_DIR / "silhouette-partwise.json"
    if not path.exists():
        report["B4_silhouette"] = {
            "band": PRE_REGISTRATION["B4"],
            "verdict": "NOT RUN",
            "reason": f"{path.relative_to(ROOT)} is missing; run "
                      "tools/compare/d7b_silhouette_partwise.py",
        }
        return
    payload = json.loads(path.read_text())
    report["B4_silhouette"] = {
        "band": PRE_REGISTRATION["B4"],
        "instrument": "tools/compare/d7b_silhouette_partwise.py",
        "reference": payload["reference"],
        "preregistered": payload["preregistered"],
        "clause_verdicts": payload["preregistered_clause_verdicts"],
        "partwise_and_tercile": payload["subjects"],
        "statistics": payload["statistics"],
        "caches_copied_never_shared": payload["caches_copied_never_shared"],
        "blind_to": ("the silhouette scores the MESH, not limb placement, and it cannot "
                     "see the skin distortion the post-D7 review found. Precision and "
                     "recall are reported beside every IoU for that reason."),
        "verdict": payload["verdict"],
    }


# --------------------------------------------------------------- B5: nothing else moved
def b5_block(report: dict) -> None:
    head: dict = {}
    for subject in (0, 1):
        d7, d7b = load_track(D7_DIR, subject), load_track(D7B_DIR, subject)
        index = d7.joint_names.index("Head")
        a = world_rotations(d7)[:, index]
        b = world_rotations(d7b)[:, index]
        worst = float(geodesic_deg(Rotation.from_quat(a).as_matrix(),
                                   Rotation.from_quat(b).as_matrix()).max())
        neck_index = d7.joint_names.index("Neck")
        neck_local_worst = float(geodesic_deg(
            Rotation.from_quat(np.asarray(d7.local_rotations_xyzw[:, neck_index], np.float64)).as_matrix(),
            Rotation.from_quat(np.asarray(d7b.local_rotations_xyzw[:, neck_index], np.float64)).as_matrix()).max())
        head[f"subject_{subject:02d}"] = {
            "head_world_worst_deg": round(worst, 12),
            "within_band": bool(np.radians(worst) <= HEAD_BAND),
            "neck_LOCAL_worst_deg": round(neck_local_worst, 6),
            "why_the_local_moves": ("the head's WORLD rotation comes from the head solve "
                                    "and not from the torso; its LOCAL is that world "
                                    "relative to UpperChest, which this step turns. The "
                                    "band is on the world, which is what is delivered."),
        }
    closure: dict = {}
    for subject in (0, 1):
        track = load_track(D7B_DIR, subject)
        skeleton = skeleton_for_track(track)
        expected = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64), skeleton=skeleton)
        names, got, rest = d3.glb_joint_positions(D7B_DIR / f"subject-{subject:02d}.glb")
        order = [track.joint_names.index(name) for name in names]
        worst = float(np.abs(got - expected[:, order]).max())
        closure[f"subject_{subject:02d}"] = {
            "max_m": worst,
            "within_band": bool(worst <= CLOSURE_BAND_M),
            "glb_rest_equals_track_rest_max_m": float(np.abs(
                rest - np.asarray(track.rest_translations_m, np.float64)[order]).max()),
            "glb_rest_note": ("expected to be large and it reads 1.9585 on the D7 build "
                              "too: a glTF node translation is expressed in its PARENT's "
                              "aligned asset rest frame, so bone LENGTHS are preserved and "
                              "directions are not. The closure band above is the figure; "
                              "this is not banded (d3_skeleton_gate.closure_block)."),
            "joint_names_match": names == list(skeleton.names),
        }
    # THE FLOOR THIS BAND RUNS INTO, demonstrated rather than asserted. The delivered
    # track stores every local rotation as float32. `Head`'s parent is `Neck`, whose world
    # is a slerp of `torso_world` -- which this step turns -- so BOTH locals change even
    # though the head's world orientation does not. Recomposing an unchanged world through
    # a different float32 decomposition cannot be exact. This control performs exactly that
    # arithmetic with a head that is unchanged BY CONSTRUCTION: D7's own Neck and Head
    # worlds, re-expressed under D7b's `UpperChest`, cast to float32 as the file casts
    # them, and recomposed. Whatever it reads is the floor.
    floor_control: dict = {}
    for subject in (0, 1):
        d7, d7b = load_track(D7_DIR, subject), load_track(D7B_DIR, subject)
        w7, w7b = world_rotations(d7), world_rotations(d7b)
        neck_i = d7.joint_names.index("Neck")
        head_i = d7.joint_names.index("Head")
        f32 = lambda q: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(
            np.asarray(q, np.float32).astype(np.float64))
        # Exactly what `_set_world` does for `Head`: the local is
        # `inv(world[Neck]) . head_world` in float64, and the track then casts it to
        # float32. Here `world[Neck]` is D7b's OWN delivered neck world and `head_world` is
        # D7's -- a head that did not move, written through D7b's chain.
        head_local = f32(cm._quaternion_multiply(
            cm._quat_inverse(w7b[:, neck_i]), w7[:, head_i]))
        recomposed = cm._quaternion_multiply(w7b[:, neck_i], head_local)
        worst = float(geodesic_deg(
            Rotation.from_quat(recomposed).as_matrix(),
            Rotation.from_quat(w7[:, head_i]).as_matrix()).max())
        floor_control[f"subject_{subject:02d}"] = {
            "worst_deg": round(worst, 12), "worst_rad": round(np.radians(worst), 14)}

    report["B5_nothing_else_moved"] = {
        "band": PRE_REGISTRATION["B5"],
        "head_world_orientation": head,
        "float32_storage_floor_control": {
            "what": ("D7's OWN Neck and Head worlds -- unchanged by construction -- "
                     "re-expressed under D7b's UpperChest, cast to float32 exactly as the "
                     "delivered track stores them, and recomposed. This is the arithmetic "
                     "the pipeline performs on a head that did not move."),
            "figures": floor_control,
            "one_float32_ulp_deg": round(float(np.degrees(2.0 * 2.0 ** -24)), 12),
            "measured_departure_in_ulps": {
                f"subject_{s:02d}": round(
                    head[f"subject_{s:02d}"]["head_world_worst_deg"]
                    / float(np.degrees(2.0 * 2.0 ** -24)), 3) for s in (0, 1)},
            "reading": ("the single-cast floor is one float32 ULP. The measured departure "
                        "is ~1.5 ULP over a six-link chain whose every local this step "
                        "re-derives, so the head's WORLD orientation did not move -- it is "
                        "the same `head_rotations` matrix through the same `_set_world`, "
                        "written through a different decomposition. The band as literally "
                        "written (1e-9 per frame) is one no float32 delivery can pass. It "
                        "is NOT moved; the failure is recorded. See the review, section 7."),
        },
        "d3_closure_on_the_rebuilt_glb_from_its_own_bytes": closure,
        "closure_band_m": CLOSURE_BAND_M,
        "verdict": "PASS" if all(v["within_band"] for v in head.values())
                   and all(v["within_band"] for v in closure.values()) else "FAIL",
    }


def canonical_block(report: dict) -> None:
    """D3's pattern: the converter scored against its OWN output on a canonical body.

    REPORTED, not banded, on the torso and the arms: the round trip rebuilds its torso
    frame from the upper-arm origins this step moves, so its second pass is not scoring
    the same thing its first pass did (CLAUDE.md: 'the round trip cannot score a temporal
    step', and the same argument applies to any change to the trunk chain). The LEGS are
    the part it can still score, and they must stay at 0.00.
    """

    sys.path.insert(0, str(ROOT / "tools/swap-harness"))
    import retarget_cost as rc  # noqa: E402

    rows = {}
    for subject in (0, 1):
        source = dvc.capture_positions(D7B_DIR, subject)
        fk1 = rc.retarget_then_fk(source, DETAILED_HUMANOID)
        synthetic = rc.landmarks_from_fk(fk1, DETAILED_HUMANOID)
        fk2 = rc.retarget_then_fk(rc.Z_UP_FROM_Y_UP(synthetic), DETAILED_HUMANOID)
        rows[f"subject_{subject:02d}"] = d3.groups_mm(
            rc.score(fk2, synthetic, DETAILED_HUMANOID))
    committed = json.loads(
        (ROOT / "artifacts/compare/d7-pelvis-frame/gate.json").read_text()
    ).get("B7_canonical_round_trip", {})
    report["canonical_round_trip"] = {
        "reference": "the converter scored against its OWN output, on a canonical body",
        "note": ("the retarget harness has NO spine landmark, so this arm runs the legacy "
                 "trunk-line path -- it is blind to D7 and to D7b alike, exactly as "
                 "`retarget_cost.py` is. It is here to show the legacy path still closes."),
        "subjects": rows,
        "legs_at_zero": all(row["legs"] == 0.0 for row in rows.values()),
        "committed_d7_figures": committed.get("subjects", committed) or None,
    }


# ---------------------------------------------------------------------- B6: reported
def b6_block(report: dict, payload: dict) -> None:
    block: dict = {"band": PRE_REGISTRATION["B6"], "never_banded": True}

    # ---- the torso frame's own travel, per frame
    turn = {}
    for subject in (0, 1):
        row = {}
        for label, directory in (("D7", D7_DIR), ("D7b", D7B_DIR)):
            track = load_track(directory, subject)
            index = track.joint_names.index("Spine")
            world = world_rotations(track)[:, index]
            matrices = Rotation.from_quat(world).as_matrix()
            step = geodesic_deg(matrices[1:], matrices[:-1])
            row[label] = {
                "median_deg_per_frame": round(float(np.median(step)), 4),
                "p95_deg_per_frame": round(float(np.percentile(step, 95)), 4),
                "max_deg_per_frame": round(float(step.max()), 4),
                "frames_over_60_deg_per_frame": int((step > TURN_CEILING_DEG_PER_FRAME).sum()),
            }
        turn[f"subject_{subject:02d}"] = row
    block["torso_frame_turn_per_frame"] = turn
    block["prediction_was_zero_frames_over_the_ceiling"] = all(
        row[label]["frames_over_60_deg_per_frame"] == 0
        for row in turn.values() for label in ("D7", "D7b"))

    # ---- rung 11 vs MAMMA, per joint. AGREEMENT ONLY; it selects nothing.
    scoreboards = {
        "D7": ROOT / "artifacts/compare/scoreboard-d7-pelvis-frame.json",
        "D7b": ROOT / "artifacts/compare/scoreboard-d7b-trunk.json",
    }
    mamma: dict = {"reference": "MAMMA pred_joints -- AGREEMENT ONLY. It reports and "
                                "selects nothing, and nothing from it enters src.",
                   "subject_map": "tools/head/subject_map.py -- MAMMA's body_id-00 is our "
                                  "subject 1; pairing by index crosses the performers."}
    for label, path in scoreboards.items():
        mamma[label] = json.loads(path.read_text()) if path.exists() else "NOT RUN"
    block["rung_11_vs_mamma"] = mamma

    # ---- facing and the 16 handedness signs
    facing_before = ROOT / "artifacts/compare/d7-pelvis-frame/facing-d7.json"
    facing_after = OUT_DIR / "facing-d7b.json"
    if facing_before.exists() and facing_after.exists():
        before, after = (json.loads(p.read_text()) for p in (facing_before, facing_after))

        def signs(doc: dict) -> dict:
            found: dict = {}

            def walk(node, path):
                if isinstance(node, dict):
                    if "sign_median" in node:
                        found[path] = node["sign_median"]
                    else:
                        for key, value in node.items():
                            walk(value, f"{path}/{key}")

            walk(doc["triple_product"], "")
            return found

        sa, sb = signs(before), signs(after)
        changed = [k for k in sa if sa[k] != sb.get(k)]
        moved, dots = [], {}
        for subject in ("subject_00", "subject_01"):
            for joint, cell in after["forward_dot"][subject].items():
                if not isinstance(cell, dict):
                    continue
                for reference, value in cell.items():
                    if not isinstance(value, dict) or "median" not in value:
                        continue
                    old = before["forward_dot"][subject][joint][reference]["median"]
                    if abs(value["median"] - old) > 0.02:
                        moved.append({"subject": subject, "joint": joint,
                                      "reference": reference, "D7": round(old, 4),
                                      "D7b": round(value["median"], 4)})
                    dots[f"{subject}/{joint}/{reference}"] = {
                        "D7": round(old, 4), "D7b": round(value["median"], 4),
                        "above_0_9": bool(value["median"] > 0.9)}
        block["facing"] = {
            "handedness_signs_total": len(sa),
            "handedness_signs_changed": changed or "NONE",
            "forward_dot_medians_that_moved_more_than_0_02": moved,
            "forward_dots": dots,
        }
    else:
        block["facing"] = "NOT RUN"

    # ---- the head gate's own figures, rerun
    head_gate = ROOT / "artifacts/head-lane/head-gate-shipped.json"
    block["head_gate_rerun"] = (
        {"path": str(head_gate.relative_to(ROOT)),
         "note": ("the head gate reads the head solve and the triangulated landmarks, both "
                  "byte-identical across these builds, so its figures cannot move. Rerun "
                  "and quoted rather than assumed."),
         "bands": json.loads(head_gate.read_text()).get("bands"),
         "verdicts": {k: v for k, v in json.loads(head_gate.read_text()).items()
                      if k.startswith("subject")}}
        if head_gate.exists() else "NOT RUN")

    # ---- the head, from the delivered files, against the head landmark
    block["head_joint_from_the_file"] = {
        f"subject_{s:02d}": {label: joint_row(payload, s, "Head")[label]["median_mm"]
                             for label in ("D3", "D7", "D7b")} for s in (0, 1)}
    block["head_joint_note"] = (
        "scored against `nose`, which in the SOMA-77 adapter is index 6, the Head SKELETAL "
        "joint -- a joint-vs-joint pairing, not a surface nose (CLAUDE.md). Reported only.")
    report["B6_reported"] = block


# ------------------------------------------------------------------- B7: synthetic truth
def b7_block(report: dict) -> None:
    """The aim, on posed SOMASKEL77 clips, clean and under I7's own measured noise.

    Both arms run the REAL converter. The D7 arm is reached by substituting
    `_joint_origin` so that `Spine` reports the hip midpoint -- the shipped D7 line,
    computed at the identical call site -- and every other joint is untouched.
    """

    import d7_pelvis_synthetic as syn  # noqa: E402
    from soma77_pose import SOMA77_TO_AUTOANIM  # noqa: E402

    rest_names = ("Chest", "UpperChest", "Neck")
    length = float(sum(np.linalg.norm(np.asarray(
        DETAILED_HUMANOID.joints[DETAILED_HUMANOID.index(name)].rest_translation_m,
        np.float64)) for name in rest_names))

    def landmarks(take: np.ndarray) -> np.ndarray:
        out = np.zeros((len(take), len(cm.JOINT_NAMES), 3), dtype=np.float64)
        for name, soma in SOMA77_TO_AUTOANIM.items():
            out[:, cm.JOINT_INDEX[name]] = take[:, soma]
        for name in ("left_ear", "right_ear"):
            out[:, cm.JOINT_INDEX[name]] = take[:, SOMA77_TO_AUTOANIM["neck"]]
        return out

    def neck_error_mm(positions: np.ndarray, spine: np.ndarray, d7_aim: bool) -> tuple:
        shipped = cm._joint_origin
        if d7_aim:
            points = positions[..., (0, 2, 1)].copy()
            points[..., 2] *= -1.0
            hip_mid = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                             + points[:, cm.JOINT_INDEX["right_hip"]])

            def origin(world, frame, root_translation, rest, joint_name):
                if joint_name == "Spine":
                    return hip_mid[frame]
                return shipped(world, frame, root_translation, rest, joint_name)

            cm._joint_origin = origin
        # THE GROUND PROJECTION RUNS AFTER THE AIM, and it translates the root -- so on a
        # delivered track the `Spine` origin has moved since the aim was taken and the
        # identity `neck error == length floor` no longer holds exactly. The claim is
        # about the CONVERTER, so the unprojected track is captured with D3's watcher
        # pattern and BOTH are reported: the converter's own identity, and what a delivery
        # would carry once the feet are put on the floor.
        saved = cm.project_generated_foot_contacts
        captured: list = []

        def watcher(track, **kwargs):
            captured.append(track)
            return saved(track, **kwargs)

        cm.project_generated_foot_contacts = watcher
        try:
            track = cm.positions_to_body_track(
                positions, sample_rate_hz=30, provenance_sha256="0" * 64,
                spine_world_z_up_m=spine, skeleton=DETAILED_HUMANOID)
        finally:
            cm._joint_origin = shipped
            cm.project_generated_foot_contacts = saved
        points = positions[..., (0, 2, 1)].copy()
        points[..., 2] *= -1.0
        neck = points[:, cm.JOINT_INDEX["neck"]]
        out = []
        for arm in (captured[-1], track):
            world = forward_kinematics_positions(
                np.asarray(arm.root_translation_m, np.float64),
                np.asarray(arm.local_rotations_xyzw, np.float64), skeleton=DETAILED_HUMANOID)
            error = np.linalg.norm(world[:, DETAILED_HUMANOID.index("Neck")] - neck, axis=1)
            floor = np.abs(length - np.linalg.norm(
                neck - world[:, DETAILED_HUMANOID.index("Spine")], axis=1))
            out.append((1000.0 * error, 1000.0 * floor))
        hoist = 1000.0 * float(np.median(np.linalg.norm(
            np.asarray(track.root_translation_m, np.float64)
            - np.asarray(captured[-1].root_translation_m, np.float64), axis=1)))
        return out[0], out[1], hoist

    rig = cm.load_camera_rig(syn.RIG)
    cameras = tuple(c.scaled(syn.WORKING_WIDTH, syn.WORKING_HEIGHT) for c in rig)
    clips: dict = {}
    worst_clean_gap_m = 0.0
    for clip, (take, _rest) in syn.load(syn.FAST_STRIDE).items():
        positions = landmarks(take)
        spine = take[:, 1]
        tilt = syn.tilt_deg(take)
        bent = tilt > syn.BENT_TILT_DEG
        (clean_d7b, clean_floor), (proj_d7b, proj_floor), hoist = neck_error_mm(
            positions, spine, False)
        (clean_d7, _), (proj_d7, _), _ = neck_error_mm(positions, spine, True)
        worst_clean_gap_m = max(worst_clean_gap_m,
                                float(np.abs(clean_d7b - clean_floor).max()) / 1000.0)
        noisy_rows = []
        for seed in syn.SEEDS[:3]:
            rng = np.random.default_rng(seed)
            noised = syn.observe(cameras, take, rng, smooth_spine=False)
            if not np.isfinite(noised).all():
                continue
            n_positions = landmarks(noised)
            n_spine = noised[:, 1]
            (d7b, floor), _projected, _hoist = neck_error_mm(n_positions, n_spine, False)
            (d7, _), _projected, _hoist = neck_error_mm(n_positions, n_spine, True)
            noisy_rows.append((d7b, d7, floor))
        pooled = lambda k: np.concatenate([row[k] for row in noisy_rows]) if noisy_rows else None
        clips[clip] = {
            "frames": int(len(take)),
            "bent_frames": int(bent.sum()),
            "tilt_median_deg": round(float(np.median(tilt)), 2),
            "clean": {
                "aim_from_spine_origin_median_mm": round(float(np.median(clean_d7b)), 4),
                "aim_from_hip_midpoint_median_mm": round(float(np.median(clean_d7)), 4),
                "length_floor_median_mm": round(float(np.median(clean_floor)), 4),
                "worst_gap_to_the_floor_mm": round(float(np.abs(clean_d7b - clean_floor).max()), 6),
                "AFTER_the_ground_projection": {
                    "aim_from_spine_origin_median_mm": round(float(np.median(proj_d7b)), 4),
                    "aim_from_hip_midpoint_median_mm": round(float(np.median(proj_d7)), 4),
                    "length_floor_median_mm": round(float(np.median(proj_floor)), 4),
                    "worst_gap_to_the_floor_mm": round(
                        float(np.abs(proj_d7b - proj_floor).max()), 4),
                    "hoist_median_mm": round(hoist, 3)},
                "bent_aim_from_spine_origin_median_mm": (
                    round(float(np.median(clean_d7b[bent])), 4) if bent.any() else None),
                "bent_aim_from_hip_midpoint_median_mm": (
                    round(float(np.median(clean_d7[bent])), 4) if bent.any() else None),
            },
            "noisy_I7": ({
                "seeds": list(syn.SEEDS[:3]),
                "aim_from_spine_origin_median_mm": round(float(np.median(pooled(0))), 4),
                "aim_from_hip_midpoint_median_mm": round(float(np.median(pooled(1))), 4),
                "length_floor_median_mm": round(float(np.median(pooled(2))), 4),
            } if noisy_rows else "no seed produced finite triangulation"),
        }
    report["B7_synthetic"] = {
        "band": PRE_REGISTRATION["B7"],
        "reference": ("exact synthetic truth: SOMASKEL77 posed with real GEM-X rotations, "
                      "the Kabsch pelvis frame, and I7's own measured heavy-tail "
                      "frame-correlated pixel noise recovered through the real "
                      "triangulator. MAMMA-FREE."),
        "how_the_D7_arm_is_reached": ("`_joint_origin` substituted to report the hip "
                                      "midpoint for `Spine` -- the shipped D7 line at the "
                                      "identical call site, never a re-implementation"),
        "clean_arm_band_m": 1.0e-6,
        "clean_arm_is_measured_BEFORE_the_ground_projection": (
            "the projection translates the root AFTER the aim is taken, so on a projected "
            "track the identity `neck error == length floor` is displaced by the hoist. "
            "The claim is about the CONVERTER; both arms are reported per clip."),
        "clean_arm_worst_gap_to_the_floor_m": round(worst_clean_gap_m, 12),
        "clean_arm_reaches_the_floor": bool(worst_clean_gap_m <= 1.0e-6),
        "clips": clips,
        "verdict": "REPORTED",
    }


# --------------------------------------------------------------------------- hygiene
def hygiene_block(report: dict) -> None:
    rows = {}
    for label, path in (("hygiene_arm_src_UNCHANGED", OUT_DIR / "delivery-hygiene-build.json"),
                        ("d7b_arm", OUT_DIR / "delivery-build.json")):
        rows[label] = json.loads(path.read_text()) if path.exists() else "NOT RUN"
    identical = (rows["hygiene_arm_src_UNCHANGED"]["hygiene"]["all_delivered_files_identical"]
                 if isinstance(rows["hygiene_arm_src_UNCHANGED"], dict) else False)
    report["hygiene"] = {
        "what_it_proves": ("the rebuild harness reproduces the SHIPPED delivery byte for "
                           "byte when src is unchanged, so every difference in the D7b arm "
                           "belongs to the one-line src change and to nothing else."),
        "work_source": ("artifacts/commercial-multiview-soma77/work, copied never "
                        "symlinked. The brief named artifacts/commercial-multiview-b2/work; "
                        "that is an older non-soma77 build and the clause cannot be "
                        "satisfied from it -- recorded in the review, section 7."),
        "builds": rows,
        "shipped_delivery_never_written": True,
        "verdict": "PASS" if identical else "FAIL",
    }


# ------------------------------------------------------------------------------- main
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = measured()
    report: dict = {
        "title": "D7b -- re-solve the trunk chain onto the captured neck after the pelvis frame",
        "preregistration": PRE_REGISTRATION,
        "same_denominator": payload["same_denominator"],
        "deliveries": payload["deliveries"],
        "measured_report": str(MEASURED.relative_to(ROOT)),
    }
    floor_block(report, payload)
    b1_block(report, payload)
    b2_block(report, payload)
    b3_block(report, payload)
    b4_block(report)
    b5_block(report)
    canonical_block(report)
    b6_block(report, payload)
    b7_block(report)
    hygiene_block(report)

    banded = {
        "B1_placement": report["B1_placement"]["verdict"],
        "B2_untouchable": report["B2_untouchable"]["verdict"],
        "B4_silhouette": report["B4_silhouette"]["verdict"],
        "B5_nothing_else_moved": report["B5_nothing_else_moved"]["verdict"],
    }
    reported = {"B3_arms": report["B3_arms"]["verdict"],
                "B6_reported": "REPORTED",
                "B7_synthetic": report["B7_synthetic"]["verdict"],
                "floor": report["floor"]["verdict"],
                "hygiene": report["hygiene"]["verdict"]}
    report["verdicts"] = {"banded_by_the_merge_rule": banded, "reported": reported}
    report["merge_rule"] = {
        "text": PRE_REGISTRATION["merge_rule"],
        "outcome": "MERGE" if all(v == "PASS" for v in banded.values()) else "DO NOT MERGE",
        "failed_clauses": [k for k, v in banded.items() if v != "PASS"] or "NONE",
    }
    report["overall"] = report["merge_rule"]["outcome"]
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"verdicts": report["verdicts"], "merge_rule": report["merge_rule"]},
                     indent=1))
    print(f"\nwrote {REPORT}")
    return 0 if report["overall"] == "MERGE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
