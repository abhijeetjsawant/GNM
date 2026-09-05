#!/usr/bin/env python3
"""D9's gate: the arms aimed from their own origins, so the delivered arm sits on the points.

Every band from the card, each as its own keyed block, with the degenerate that fails it
beside it. The card in `docs/LADDER_EXECUTION_PLAN.md` section 2 is the pre-registration
and it is FROZEN: where a number refutes a prediction the prediction is RECORDED as refuted
and the band is not moved.

WHAT THIS GATE READS, AND FROM WHERE
  * the DELIVERED files' own bytes, through `delivered_vs_capture.py` -- every placement
    figure, and the arm-aim FLOOR beside it. `retarget_cost.py` re-solves the track through
    the converter and is structurally blind to what the exporter wrote; it is labelled, not
    fixed (D7b's finding, unchanged here).
  * the two rebuilt tracks, for the bit-identity bands. Same cached detections, same
    triangulation, with BOTH landmark arrays asserted byte-identical -- which a
    converter-only step must satisfy.
  * this step's own part-wise silhouette wrapper, for the photographs.
  * the committed instruments for the reported arms: `mamma_scoreboard.py` (rung 11),
    `head_gate.py`, and the canonical round trip in `tools/swap-harness/retarget_cost.py`.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9_arm_gate.py

Writes `artifacts/compare/d9-arms/gate.json`. NOTHING is written under
`artifacts/commercial-multiview-soma77/`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

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

OUT_DIR = ROOT / "artifacts/compare/d9-arms"
REPORT = OUT_DIR / "gate.json"
MEASURED = OUT_DIR / "delivered-vs-capture.json"

D8_DIR = ROOT / "artifacts/commercial-multiview-soma77"
D9_DIR = OUT_DIR / "delivery"

FLOOR_BAND_MM = 3.0            # B1: elbow and wrist within this of the real floor
CLOSURE_BAND_M = 1.0e-6        # B4: the D3 closure on the rebuilt GLB, from its own bytes
ORACLE_BAND = 1.0e-9           # B5: the two aims agree where the clavicle is exactly right
CLEAN_FLOOR_BAND_M = 1.0e-6    # B5: the clean synthetic arm reaches its length floor
CLAVICLE_MISMATCH_M = 0.030    # B5: the injected clavicle-length mismatch, both signs

# rig joint -> the landmark it is aimed at, and the bone whose rest length it travels.
ARMS = (
    ("left", "LeftUpperArm", "LeftLowerArm", "LeftHand",
     "left_shoulder", "left_elbow", "left_wrist"),
    ("right", "RightUpperArm", "RightLowerArm", "RightHand",
     "right_shoulder", "right_elbow", "right_wrist"),
)
LEGS = (
    ("left", "LeftUpperLeg", "LeftLowerLeg", "left_hip", "left_knee"),
    ("right", "RightUpperLeg", "RightLowerLeg", "right_hip", "right_knee"),
)
UNTOUCHABLE = ("Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
               "LeftEye", "RightEye", "LeftShoulder", "RightShoulder",
               "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg",
               "LeftFoot", "RightFoot", "LeftToes", "RightToes")

# The card, verbatim. Frozen before any number below existed.
PRE_REGISTRATION = {
 "source": "docs/LADDER_EXECUTION_PLAN.md section 2, the D9 card, written 2026-09-05",
 "defect": ("the delivered `UpperArm` joint sits 10-13 mm median off the captured shoulder "
            "(69 mm on frame 118, performer 1 left) because the clavicle has a fixed rest "
            "length; pass C then aimed every arm bone by a LANDMARK-TO-LANDMARK direction "
            "from that displaced origin, so the elbow and wrist carried the same "
            "displacement (14 / 15-20 mm median)"),
 "floor": ("measured 2026-09-05 from the files for the ACTUAL operation (elbow on the ray "
           "from the real UpperArm origin at rest length, wrist on the ray from that "
           "placed elbow): elbow 5.5 / 5.7 (performer 0 L / R) and 8.8 / 6.7 (performer 1) "
           "mm median, p95 20-32; wrist chained 7.3 / 9.2 and 10.4 / 11.9, p95 22-36; the "
           "window (85-125) 4-15. Frame 118 is LENGTH-limited, not aim-limited: the "
           "captured upper arm there is 333 mm against a rest of 277, so the floor is "
           "36 / 41 mm and D9's visible gain on that frame is partial"),
 "B1": ("placement; the candidate optimises it, so it is paired with the floor and the "
        "photographs: elbow and wrist medians within 3 mm of the real floor on both "
        "performers and both sides, whole take; bent-tercile and window cuts reported; "
        "predicted elbow 14 -> 6-9, wrist 15-20 -> 7-12"),
 "B1_must_fail": ("the shipped D8 build and a root-translation degenerate that zeroes a "
                  "wrist (B2 catches it)"),
 "B2": ("untouchable: root, Hips, Spine/Chest/UpperChest, Neck, Head, Shoulder (clavicle) "
        "and every LEG local bit-identical to D8; the UpperArm-origin miss UNCHANGED "
        "(10-13 mm; it is D5's); the leg dry run REPORTED from the closed form (aiming the "
        "thigh from its own origin would move the knee 3.1-4.2 mm median, p95 9-26) and "
        "NOT shipped"),
 "B3": ("the photographs: part-wise silhouette, ARMS, whole take and the window, not worse "
        "on either performer with the CI clear on the D7b/D8 predicate (upper bound >= 0), "
        "improvement predicted on both"),
 "B4": ("D3 closure <= 1e-6 m and same-denominator PASS; head gate rerun (predicted "
        "byte-equal figures: it reads landmarks and the head solve, neither touched); "
        "canonical round trip arms reported before / after (0.55 / 0.08 may close toward "
        "zero: the re-solve is exact by construction)"),
 "B5": ("synthetic (SOMASKEL77 posed clips, I7 noise): elbow / wrist placement of "
        "aim-from-own-origin vs landmark-direction under an injected clavicle-length "
        "mismatch of +/-30 mm; clean arm reaches the length floor to 1e-6; oracle: with "
        "the clavicle exactly right the two aims agree to 1e-9 (must hold, or the change "
        "did more than aiming)"),
 "MAMMA": "per joint, window and whole take: report only",
 "merge_rule": ("fixed before numbers: B1 on both performers and both sides AND B2 exact "
                "AND B3 on both performers AND B4's closure and denominator clauses; the "
                "rest reports"),
}


# ------------------------------------------------------------------------------- helpers
def load_track(directory: Path, subject: int) -> BodyTrack:
    return BodyTrack.from_dict(json.loads(
        (directory / f"subject-{subject:02d}.body-track.json").read_text()))


def measured() -> dict:
    if not MEASURED.exists():
        raise SystemExit(
            f"{MEASURED.relative_to(ROOT)} is missing; run tools/compare/"
            "delivered_vs_capture.py with --delivery D8=... --delivery D9=... first")
    return json.loads(MEASURED.read_text())


def joint_row(payload: dict, subject: int, joint: str) -> dict:
    return payload["subjects"][f"subject_{subject:02d}"]["joints"][joint]


def floor_row(payload: dict, subject: int, label: str, side: str) -> dict:
    return payload["subjects"][f"subject_{subject:02d}"]["arm_aim_floor"][label]["sides"][side]


def rig_points(capture: np.ndarray) -> np.ndarray:
    """The converter's own change of basis, applied exactly as `positions_to_body_track` does."""

    points = np.asarray(capture, np.float64)[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    return points


def to_capture(vector: np.ndarray) -> np.ndarray:
    return np.stack([vector[..., 0], -vector[..., 2], vector[..., 1]], axis=-1)


# ---------------------------------------------------------------------------- THE FLOOR
def floor_block(report: dict, payload: dict) -> None:
    block: dict = {
        "what_it_is": (
            "elbow: | ||captured_elbow - UpperArm_origin|| - L_upper |, per frame, from "
            "each delivery's OWN UpperArm origin -- the residual an aim cannot remove, "
            "because a rigid bone can only be pointed, not stretched. wrist: the same "
            "quantity chained from the PLACED elbow, which is where the delivered elbow "
            "will be. The companion of D7b's trunk-length floor, for D9's operation."),
        "belongs_to": ("the UpperArm-origin miss belongs to D5 (the trunk chord 8.4 / 7.9 mm "
                       "and a missing shoulder translation 6.2 / 7.3 mm); the residual "
                       "LENGTH error belongs to a fitted arm, also D5"),
        "pre_registered_values_mm": {
            "subject_00": {"elbow": [5.5, 5.7], "wrist": [7.3, 9.2]},
            "subject_01": {"elbow": [8.8, 6.7], "wrist": [10.4, 11.9]},
            "as": "median whole take, left then right, measured from the shipped D8 files"},
        "subjects": {},
    }
    ok = True
    for subject in (0, 1):
        expected = block["pre_registered_values_mm"][f"subject_{subject:02d}"]
        rows: dict = {}
        for part in ("elbow", "wrist"):
            got = [floor_row(payload, subject, "D8", side)[part]["median_mm"]
                   for side in ("left", "right")]
            matches = all(abs(a - b) <= 0.1 for a, b in zip(expected[part], got))
            ok = ok and matches
            rows[part] = {
                "pre_registered_mm": expected[part],
                "measured_mm": got,
                "reproduces_to_0_1_mm": bool(matches)}
        unchanged = all(
            abs(floor_row(payload, subject, "D8", side)[part]["median_mm"]
                - floor_row(payload, subject, "D9", side)[part]["median_mm"]) < 1e-9
            for side in ("left", "right") for part in ("elbow", "wrist"))
        ok = ok and unchanged
        rows["d9_floor_equals_d8_floor"] = bool(unchanged)
        rows["why_they_must_be_equal"] = (
            "the floor is read from the UpperArm ORIGIN, and D9 does not move it -- the "
            "clavicle, the trunk chain and the root are untouched. Equal floors are a "
            "second proof of B2 and they are what makes B1 a comparison of like with like.")
        rows["per_delivery"] = {
            label: payload["subjects"][f"subject_{subject:02d}"]["arm_aim_floor"][label]
            for label in ("D8", "D9")}
        block["subjects"][f"subject_{subject:02d}"] = rows
    block["frame_118_is_length_limited"] = {
        f"subject_{s:02d}": {side: floor_row(payload, s, "D9", side)["frame_id_118"]
                             for side in ("left", "right")} for s in (0, 1)}
    block["verdict"] = "PASS" if ok else "FAIL"
    report["floor"] = block


# ------------------------------------------------------- B1: the placement claim
def wrist_zeroing_degenerate() -> dict:
    """THE MUST-FAIL B2 IS FOR. A root shift that puts a wrist exactly on its landmark.

    It costs nothing to build and it scores a PERFECT left wrist: translate the root, per
    frame, by `left_wrist_landmark - LeftHand_origin`. B1 alone would wave it through --
    which is the whole reason B1 is paired with B2 and with the photographs. Computed by
    forward kinematics of the D9 track with a modified root; nothing is exported and
    nothing is written. The D2 precedent: translate the same rendered body.
    """

    out: dict = {}
    for subject in (0, 1):
        track = load_track(D9_DIR, subject)
        skeleton = skeleton_for_track(track)
        roots = np.asarray(track.root_translation_m, np.float64)
        rotations = np.asarray(track.local_rotations_xyzw, np.float64)
        world = forward_kinematics_positions(roots, rotations, skeleton=skeleton)
        capture = dvc.capture_positions(D9_DIR, subject)
        delta_capture = (capture[:, cm.JOINT_INDEX["left_wrist"]]
                         - to_capture(world[:, track.joint_names.index("LeftHand")]))
        # back into the rig's Y-up world: capture (x, y, z) -> rig (x, z, -y)
        delta_rig = np.stack([delta_capture[:, 0], delta_capture[:, 2],
                              -delta_capture[:, 1]], axis=-1)
        moved = to_capture(forward_kinematics_positions(
            roots + delta_rig, rotations, skeleton=skeleton))
        error = lambda name, landmark: 1000.0 * np.linalg.norm(
            moved[:, track.joint_names.index(name)]
            - capture[:, cm.JOINT_INDEX[landmark]], axis=1)
        out[f"subject_{subject:02d}"] = {
            "left_wrist_median_mm": round(float(np.median(error("LeftHand", "left_wrist"))), 6),
            "left_elbow_median_mm": round(float(np.median(error("LeftLowerArm", "left_elbow"))), 3),
            "right_wrist_median_mm": round(float(np.median(error("RightHand", "right_wrist"))), 3),
            "left_hip_median_mm": round(float(np.median(error("LeftUpperLeg", "left_hip"))), 3),
            "left_knee_median_mm": round(float(np.median(error("LeftLowerLeg", "left_knee"))), 3),
            "left_ankle_median_mm": round(float(np.median(error("LeftFoot", "left_ankle"))), 3),
            "root_translation_bit_identical_to_D8": False,
            "root_moved_median_mm": round(
                1000.0 * float(np.median(np.linalg.norm(delta_rig, axis=1))), 3),
        }
    return out


def ground_projection_block() -> dict:
    """WHY THE PER-FRAME EXCESS OVER THE FLOOR IS NOT ZERO, measured rather than assumed.

    `positions_to_body_track` ends with `project_generated_foot_contacts`, which TRANSLATES
    the root after every aim has been taken (D7b's B7 records the same fact for the trunk:
    "the ground projection runs after the aim"). A translation leaves every rotation alone,
    so the delivered bone still points along `elbow - origin_BEFORE_the_hoist`, while this
    gate measures the ray from the origin AFTER it. The two differ by the hoist.

    The hoist is RECOVERED here from the delivered file alone, and it is over-determined:
    each of the four arm bones gives two linear equations in the same three unknowns
    (`(I - u u^T)(target - origin + delta) = 0` for the delivered unit direction `u`), so
    if a single rigid translation explains all four the residual is zero. It does, to
    1e-4 mm. That is the second, independent measurement the lane's rule asks for before a
    number is explained.
    """

    out: dict = {}
    for subject in (0, 1):
        index, world = dvc.delivered_positions(D9_DIR, subject)
        capture = dvc.capture_positions(D9_DIR, subject)
        deltas = np.zeros((len(capture), 3))
        residual = np.zeros(len(capture))
        for frame in range(len(capture)):
            rows, targets = [], []
            for _side, upper, lower, hand, _sh, elbow, wrist in ARMS:
                for parent, child, landmark in ((upper, lower, elbow), (lower, hand, wrist)):
                    origin = world[frame, index[parent]]
                    direction = world[frame, index[child]] - origin
                    direction = direction / np.linalg.norm(direction)
                    projector = np.eye(3) - np.outer(direction, direction)
                    rows.append(projector)
                    targets.append(-projector @ (capture[frame, cm.JOINT_INDEX[landmark]]
                                                 - origin))
            solution, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets),
                                           rcond=None)
            deltas[frame] = solution
            residual[frame] = np.linalg.norm(np.vstack(rows) @ solution
                                             - np.concatenate(targets))
        magnitude = 1000.0 * np.linalg.norm(deltas, axis=1)
        out[f"subject_{subject:02d}"] = {
            "recovered_root_shift_mm": {
                "median": round(float(np.median(magnitude)), 4),
                "p95": round(float(np.percentile(magnitude, 95)), 3),
                "max": round(float(magnitude.max()), 3),
                "frames_over_0_5_mm": int((magnitude > 0.5).sum()),
                "of_frames": int(len(magnitude))},
            "residual_of_the_single_translation_mm": {
                "median": round(1000.0 * float(np.median(residual)), 6),
                "max": round(1000.0 * float(residual.max()), 6)},
            "reading": ("one rigid translation explains all four arm bones to 1e-4 mm, "
                        "which is what a post-solve root hoist looks like and what a "
                        "mis-aimed arm does not."),
        }
    return {
        "what": ("the residual above the floor is the ground projection's root hoist, "
                 "applied AFTER the aim. It is bounded by `maximum_root_correction_m=0.08` "
                 "in `positions_to_body_track` and is present on the D8 build identically."),
        "why_it_is_here": ("a prediction made before the numbers -- that the delivered "
                           "elbow would sit ON the floor to storage precision on every "
                           "frame -- is REFUTED at the frame level and holds at the "
                           "median. The mechanism is measured, not asserted."),
        "figures": out,
    }


def b1_block(report: dict, payload: dict) -> None:
    block: dict = {
        "band": PRE_REGISTRATION["B1"],
        "band_mm": FLOOR_BAND_MM,
        "reference": dvc.REFERENCE,
        "why_it_is_paired": ("the candidate optimises this directly -- the aim IS at these "
                             "landmarks -- so on its own it proves nothing. It is paired "
                             "with the arm-aim floor (which no aim can move and which is "
                             "bit-equal between the arms), with B2 (nothing else may have "
                             "been traded away) and with the photographs (which the "
                             "candidate cannot optimise)."),
        "predictions": {"elbow": "14 -> 6-9 mm median", "wrist": "15-20 -> 7-12 mm median"},
        "subjects": {},
    }
    passes: dict = {}
    for subject in (0, 1):
        rows: dict = {}
        subject_ok = True
        for side, _upper, lower, hand, _sh, _el, _wr in ARMS:
            for part, joint in (("elbow", lower), ("wrist", hand)):
                cell = joint_row(payload, subject, joint)
                floor = floor_row(payload, subject, "D9", side)[part]
                gap = cell["D9"]["median_mm"] - floor["median_mm"]
                within = abs(gap) <= FLOOR_BAND_MM
                subject_ok = subject_ok and within
                rows[f"{side}_{part}"] = {
                    "rig_joint": joint,
                    "median_mm": {label: cell[label]["median_mm"] for label in ("D8", "D9")},
                    "floor_median_mm": floor["median_mm"],
                    "gap_to_the_floor_mm": round(gap, 3),
                    f"within_{FLOOR_BAND_MM:g}_mm_of_the_floor": bool(within),
                    "p95_mm": {label: cell[label]["p95_mm"] for label in ("D8", "D9")},
                    "bent_tercile_median_mm": {
                        label: cell[label]["bent_tercile_median_mm"] for label in ("D8", "D9")},
                    "bent_tercile_floor_mm": floor["bent_tercile_median_mm"],
                    "window_floor_mm": floor["window_median_mm"],
                    "D9_minus_D8": cell["paired_differences"]["D9_minus_D8"],
                    "improved_with_the_CI_clear": bool(
                        cell["paired_differences"]["D9_minus_D8"]["ci95_mm"][1] < 0.0),
                    "per_frame_excess_over_the_floor_mm":
                        floor.get("delivered_minus_floor_mm"),
                }
        passes[subject] = subject_ok
        rows["verdict"] = "PASS" if subject_ok else "FAIL"
        block["subjects"][f"subject_{subject:02d}"] = rows
    block["prediction_check"] = {
        "elbow_medians_mm": {f"subject_{s:02d}": [
            joint_row(payload, s, name)["D9"]["median_mm"]
            for name in ("LeftLowerArm", "RightLowerArm")] for s in (0, 1)},
        "wrist_medians_mm": {f"subject_{s:02d}": [
            joint_row(payload, s, name)["D9"]["median_mm"]
            for name in ("LeftHand", "RightHand")] for s in (0, 1)},
        "elbow_prediction_6_to_9_mm_holds": all(
            6.0 <= joint_row(payload, s, name)["D9"]["median_mm"] <= 9.0
            for s in (0, 1) for name in ("LeftLowerArm", "RightLowerArm")),
        "wrist_prediction_7_to_12_mm_holds": all(
            7.0 <= joint_row(payload, s, name)["D9"]["median_mm"] <= 12.0
            for s in (0, 1) for name in ("LeftHand", "RightHand")),
    }
    block["must_fail"] = {
        "the_shipped_D8": {
            "elbow_and_wrist_median_mm": {f"subject_{s:02d}": {
                name: joint_row(payload, s, name)["D8"]["median_mm"]
                for name in ("LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand")}
                for s in (0, 1)},
            "gap_to_its_own_floor_mm": {f"subject_{s:02d}": {
                f"{side}_{part}": round(
                    joint_row(payload, s, joint)["D8"]["median_mm"]
                    - floor_row(payload, s, "D8", side)[part]["median_mm"], 3)
                for side, _u, lower, hand, _sh, _e, _w in ARMS
                for part, joint in (("elbow", lower), ("wrist", hand))} for s in (0, 1)},
            "fails_the_band": all(
                abs(joint_row(payload, s, joint)["D8"]["median_mm"]
                    - floor_row(payload, s, "D8", side)[part]["median_mm"]) > FLOOR_BAND_MM
                for s in (0, 1)
                for side, _u, lower, hand, _sh, _e, _w in ARMS
                for part, joint in (("elbow", lower), ("wrist", hand))),
        },
        "the_root_translation_degenerate": {
            "what": ("the root translated per frame by `left_wrist - LeftHand_origin`. It "
                     "scores a PERFECT left wrist and B1 alone cannot reject it."),
            "figures": wrist_zeroing_degenerate(),
            "B1_would_pass_it_on_that_joint": True,
            "B2_catches_it": ("its root translation is not bit-identical to D8 by "
                              "construction, and its hips, knees and ankles move by tens "
                              "of millimetres -- see B2's `degenerate_scored_through_B2`."),
        },
    }
    block["the_ground_projection"] = ground_projection_block()
    block["verdict"] = "PASS" if all(passes.values()) else "FAIL"
    report["B1_placement"] = block


# ------------------------------------------------------------------- B2: untouchable
def leg_dry_run(payload: dict) -> dict:
    """THE LEG SHIFT, CLOSED FORM, REPORTED AND NOT SHIPPED.

    What aiming the thigh from its own FK origin would do to the delivered knee: the knee
    would move from where the landmark direction put it to
    `UpperLeg_origin + L_thigh . unit(knee_landmark - UpperLeg_origin)`. Read from the D9
    delivery's own bytes, so it is a statement about the file that ships today.

    D9 is arms only, one part at a time. The legs are spared the arm's defect because D2b
    puts the leg-root MIDPOINT on the captured hip midpoint (0.0 / 1.8 mm from the file),
    so their origins are not displaced the way the arm root is -- and the number below says
    how much is nevertheless left. Moving it is its own step with its own gate.
    """

    out: dict = {}
    for subject in (0, 1):
        rest = dvc.track_rest(D9_DIR, subject)
        index, world = dvc.delivered_positions(D9_DIR, subject)
        capture = dvc.capture_positions(D9_DIR, subject)
        row: dict = {}
        for side, thigh, shin, hip, knee in LEGS:
            origin = world[:, index[thigh]]
            captured = capture[:, cm.JOINT_INDEX[knee]]
            length = float(np.linalg.norm(rest[shin]))
            reach = captured - origin
            placed = origin + (length / np.linalg.norm(reach, axis=1))[:, None] * reach
            shift = 1000.0 * np.linalg.norm(placed - world[:, index[shin]], axis=1)
            floor = 1000.0 * np.abs(np.linalg.norm(reach, axis=1) - length)
            delivered = 1000.0 * np.linalg.norm(world[:, index[shin]] - captured, axis=1)
            row[side] = {
                "knee_would_move_mm": {"median": round(float(np.median(shift)), 3),
                                       "p95": round(float(np.percentile(shift, 95)), 3),
                                       "max": round(float(shift.max()), 3)},
                "delivered_knee_error_mm": round(float(np.median(delivered)), 3),
                "would_become_mm": round(float(np.median(floor)), 3),
                "hip_origin_vs_hip_landmark_mm": round(1000.0 * float(np.median(
                    np.linalg.norm(origin - capture[:, cm.JOINT_INDEX[hip]], axis=1))), 3),
            }
        out[f"subject_{subject:02d}"] = row
    return {
        "pre_registered": "knee shift 3.1-4.2 mm median, p95 9-26",
        "shipped": False,
        "why_not": ("one part at a time. The legs have their own origin story (D2b) and "
                    "their own gate; changing them in the same pass would make the "
                    "photographs and the placement figures uninterpretable."),
        "figures": out,
        "reproduces_the_pre_registered_range": bool(all(
            3.0 <= out[f"subject_{s:02d}"][side]["knee_would_move_mm"]["median"] <= 4.3
            for s in (0, 1) for side in ("left", "right"))),
    }


def b2_block(report: dict, payload: dict) -> None:
    exact: dict = {}
    for subject in (0, 1):
        d8, d9 = load_track(D8_DIR, subject), load_track(D9_DIR, subject)
        names = d8.joint_names
        row = {
            "root_translation_m": bool(np.array_equal(
                d8.root_translation_m, d9.root_translation_m)),
            "foot_contacts": bool(np.array_equal(d8.foot_contacts, d9.foot_contacts)),
            "rest_translations_m": bool(np.array_equal(
                d8.rest_translations_m, d9.rest_translations_m)),
            "joint_names": bool(d8.joint_names == d9.joint_names),
            "ticks": bool(np.array_equal(d8.ticks, d9.ticks)),
        }
        for name in UNTOUCHABLE:
            index = names.index(name)
            row[f"local_{name}"] = bool(np.array_equal(
                d8.local_rotations_xyzw[:, index], d9.local_rotations_xyzw[:, index]))
        fingers = [i for i, name in enumerate(names)
                   if name.startswith(("LeftThumb", "LeftIndex", "LeftMiddle", "LeftRing",
                                       "LeftLittle", "RightThumb", "RightIndex",
                                       "RightMiddle", "RightRing", "RightLittle"))]
        row["local_every_finger"] = bool(all(
            np.array_equal(d8.local_rotations_xyzw[:, i], d9.local_rotations_xyzw[:, i])
            for i in fingers))
        # The hands are NOT bit-identical and cannot be: `_set_world` gives the hand its
        # PARENT's world, so its local is a quaternion composed with its own inverse and
        # the rounding follows the parent. Reported with the departure from the identity.
        hands = {}
        for name in ("LeftHand", "RightHand"):
            index = names.index(name)
            a = np.asarray(d8.local_rotations_xyzw[:, index], np.float64)
            b = np.asarray(d9.local_rotations_xyzw[:, index], np.float64)
            hands[name] = {
                "bit_identical": bool(np.array_equal(
                    d8.local_rotations_xyzw[:, index], d9.local_rotations_xyzw[:, index])),
                "worst_component_gap": float(np.abs(a - b).max()),
                "worst_departure_from_identity": float(
                    np.abs(b - np.asarray((0.0, 0.0, 0.0, 1.0))).max()),
                "why": ("the hand takes its parent's world, so its LOCAL is the identity "
                        "up to float32 rounding on both builds; it is a rounding, not a "
                        "rotation."),
            }
        # ... and the four arm bones DID move, or the list above proves nothing.
        moved = {}
        for _side, upper, lower, _hand, _sh, _el, _wr in ARMS:
            for name in (upper, lower):
                index = names.index(name)
                moved[name] = not bool(np.array_equal(
                    d8.local_rotations_xyzw[:, index], d9.local_rotations_xyzw[:, index]))
        from_file = {}
        for joint in ("LeftUpperArm", "RightUpperArm", "Neck", "LeftUpperLeg",
                      "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot",
                      "RightFoot", "Head", "leg_root_midpoint_vs_hip_midpoint",
                      "hips_joint_vs_hip_midpoint"):
            cell = joint_row(payload, subject, joint)
            from_file[joint] = {
                "D8_median_mm": cell["D8"]["median_mm"],
                "D9_median_mm": cell["D9"]["median_mm"],
                "unchanged": bool(abs(cell["D8"]["median_mm"]
                                      - cell["D9"]["median_mm"]) < 1e-9)}
        exact[f"subject_{subject:02d}"] = {
            "track_bit_identity": row,
            "hands_reported_never_banded": hands,
            "the_four_arm_bones_moved": moved,
            "from_the_delivered_file": from_file,
            "all_banded_arrays_bit_identical": all(row.values()),
            "everything_outside_the_arms_unchanged_from_the_file": all(
                v["unchanged"] for v in from_file.values()),
            "upper_arm_origin_miss_unchanged_mm": {
                name: joint_row(payload, subject, name)["D9"]["median_mm"]
                for name in ("LeftUpperArm", "RightUpperArm")},
        }
    degenerate = report["B1_placement"]["must_fail"][
        "the_root_translation_degenerate"]["figures"]
    caught = {}
    for subject in (0, 1):
        hip = joint_row(payload, subject, "LeftUpperLeg")["D9"]["median_mm"]
        caught[f"subject_{subject:02d}"] = {
            "left_hip_D9_mm": hip,
            "left_hip_degenerate_mm": degenerate[f"subject_{subject:02d}"]["left_hip_median_mm"],
            "root_translation_bit_identical": False,
            "B2_rejects_it": bool(
                abs(degenerate[f"subject_{subject:02d}"]["left_hip_median_mm"] - hip) > 1e-9),
        }
    ok = all(v["all_banded_arrays_bit_identical"]
             and v["everything_outside_the_arms_unchanged_from_the_file"]
             and all(v["the_four_arm_bones_moved"].values())
             for v in exact.values())
    report["B2_untouchable"] = {
        "band": PRE_REGISTRATION["B2"],
        "how": ("the two rebuilt tracks compared array by array. The D8 arm is the shipped "
                "delivery; the D9 arm is this branch's rebuild from the same cached "
                "detections, with BOTH triangulated landmark arrays asserted "
                "byte-identical -- which a converter-only step must satisfy."),
        "subjects": exact,
        "degenerate_scored_through_B2": caught,
        "leg_dry_run": leg_dry_run(payload),
        "verdict": "PASS" if ok else "FAIL",
    }


# ------------------------------------------------------------ B3: the photographs
def b3_block(report: dict) -> None:
    path = OUT_DIR / "silhouette-partwise.json"
    if not path.exists():
        report["B3_silhouette"] = {
            "band": PRE_REGISTRATION["B3"],
            "verdict": "NOT RUN",
            "reason": f"{path.relative_to(ROOT)} is missing; run "
                      "tools/compare/d9_arm_silhouette.py",
        }
        return
    payload = json.loads(path.read_text())
    report["B3_silhouette"] = {
        "band": PRE_REGISTRATION["B3"],
        "instrument": "tools/compare/d9_arm_silhouette.py",
        "reference": payload["reference"],
        "preregistered": payload["preregistered"],
        "clause_verdicts": payload["preregistered_clause_verdicts"],
        "cuts": payload["subjects"],
        "statistics": payload["statistics"],
        "landmarks_byte_identical": {
            "raw": payload["raw_triangulation_byte_identical"],
            "smoothed": payload["smoothed_triangulation_byte_identical"]},
        "blind_to": ("the silhouette scores the MESH. The elbow and wrist move by "
                     "millimetres and an arm can move that far inside its own outline "
                     "without moving a pixel, so a LEVEL reading here is this "
                     "instrument's resolution and not evidence against B1."),
        "verdict": payload["verdict"],
    }


# ------------------------------------------------ B4: closure, denominator, head, round trip
def canonical_round_trip() -> dict:
    """D3's pattern: the converter scored against its OWN output on a canonical body.

    REPORTED, not banded. The round trip rebuilds its torso frame from the upper-arm
    ORIGINS, which this step does not move -- but its second pass re-solves the arms from
    the synthetic landmarks its first pass emitted, and those landmarks are the FK'd joint
    positions, so on the arms the round trip is now closed by construction and the figure
    is expected to fall toward zero. That is a property of the instrument, not evidence.
    """

    sys.path.insert(0, str(ROOT / "tools/swap-harness"))
    import retarget_cost as rc  # noqa: E402

    rows = {}
    for subject in (0, 1):
        source = dvc.capture_positions(D9_DIR, subject)
        fk1 = rc.retarget_then_fk(source, DETAILED_HUMANOID)
        synthetic = rc.landmarks_from_fk(fk1, DETAILED_HUMANOID)
        fk2 = rc.retarget_then_fk(rc.Z_UP_FROM_Y_UP(synthetic), DETAILED_HUMANOID)
        rows[f"subject_{subject:02d}"] = d3.groups_mm(
            rc.score(fk2, synthetic, DETAILED_HUMANOID))
    committed = json.loads(
        (ROOT / "artifacts/compare/d7b-trunk/gate.json").read_text()
    ).get("canonical_round_trip", {}).get("subjects")
    return {
        "reference": "the converter scored against its OWN output, on a canonical body",
        "note": ("the retarget harness has NO spine landmark, so this arm runs the legacy "
                 "trunk-line path. It is here to show the legacy path still closes and to "
                 "report the arms before and after."),
        "before_D9_committed_at_D7b": committed,
        "after_D9": rows,
        "legs_at_zero": all(row["legs"] == 0.0 for row in rows.values()),
    }


def b4_block(report: dict, payload: dict) -> None:
    closure: dict = {}
    for subject in (0, 1):
        track = load_track(D9_DIR, subject)
        skeleton = skeleton_for_track(track)
        expected = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64), skeleton=skeleton)
        names, got, rest = d3.glb_joint_positions(D9_DIR / f"subject-{subject:02d}.glb")
        order = [track.joint_names.index(name) for name in names]
        worst = float(np.abs(got - expected[:, order]).max())
        closure[f"subject_{subject:02d}"] = {
            "max_m": worst,
            "within_band": bool(worst <= CLOSURE_BAND_M),
            "joint_names_match": names == list(skeleton.names),
        }
    before = OUT_DIR / "head-gate-shipped-before-d9.json"
    after = OUT_DIR / "head-gate-shipped-after-d9.json"
    head_block = "NOT RUN"
    if before.exists() and after.exists():
        old, new = json.loads(before.read_text()), json.loads(after.read_text())
        # The ONE key that legitimately differs is an absolute path: the D8 run was made
        # from the main checkout and this rerun from the worktree. It is dropped by name
        # and the drop is stated, never a silent normalisation.
        strip = lambda doc: json.dumps(
            {k: v for k, v in doc.items() if k != "absolute_facing_not_a_band"},
            sort_keys=True)
        head_block = {
            "before": str(before.relative_to(ROOT)),
            "after": str(after.relative_to(ROOT)),
            "note": ("the head gate reads the head SOLVE and the triangulated landmarks, "
                     "both byte-identical across these builds, so its figures cannot "
                     "move. Rerun and compared rather than assumed."),
            "every_figure_byte_equal_before_and_after": strip(old) == strip(new),
            "the_only_difference": ("`absolute_facing_not_a_band.source`, an ABSOLUTE "
                                    "path: the committed run was made from the main "
                                    "checkout and the rerun from this worktree. Excluded "
                                    "by name; no figure is normalised."),
            "bands": new.get("bands"),
            "verdicts": {k: v for k, v in new.items() if k.startswith("subject")},
        }
    denominator = {
        "same_denominator": payload["same_denominator"],
        "byte_identical_across_arms": payload["triangulated_landmarks_byte_identical_across_arms"],
        "expected": ("PASS. D8 had to fall back on the raw array because its repair moves "
                     "the smoothed one by construction; D9 is converter-only and the "
                     "smoothed array returns as the shared denominator. Pre-registered as "
                     "an expected pass: if it read CHANGED the step did more than aiming."),
    }
    ok = all(v["within_band"] for v in closure.values()) and payload["same_denominator"]
    report["B4_closure_and_denominator"] = {
        "band": PRE_REGISTRATION["B4"],
        "closure_band_m": CLOSURE_BAND_M,
        "d3_closure_on_the_rebuilt_glb_from_its_own_bytes": closure,
        "same_denominator": denominator,
        "head_gate_rerun": head_block,
        "canonical_round_trip": canonical_round_trip(),
        "verdict": "PASS" if ok else "FAIL",
    }


# ------------------------------------------------------------------- B5: synthetic truth
def b5_block(report: dict) -> None:
    """The aim, on posed SOMASKEL77 clips, clean and under I7's own measured noise.

    BOTH arms run the REAL converter. The D8 arm is reached by substituting `_joint_origin`
    so that `*UpperArm` reports the captured shoulder and `*LowerArm` the captured elbow --
    the shipped D8 directions, computed at the identical call site, never a
    re-implementation. `commercial_multiview` documents that substitution point in
    `_joint_origin`'s own docstring, and `tests/test_trunk_resolve.py` used the same door
    for D7.

    THE CLAVICLE MISMATCH is injected on the SKELETON, by shortening and lengthening the
    `UpperArm` rest offset -- which is the clavicle bone's own length -- by 30 mm. That is
    the defect's mechanism: an origin the clavicle cannot put on the captured shoulder.
    """

    import d7_pelvis_synthetic as syn  # noqa: E402
    from soma77_pose import SOMA77_TO_AUTOANIM  # noqa: E402

    def landmarks(take: np.ndarray) -> np.ndarray:
        out = np.zeros((len(take), len(cm.JOINT_NAMES), 3), dtype=np.float64)
        for name, soma in SOMA77_TO_AUTOANIM.items():
            out[:, cm.JOINT_INDEX[name]] = take[:, soma]
        for name in ("left_ear", "right_ear"):
            out[:, cm.JOINT_INDEX[name]] = take[:, SOMA77_TO_AUTOANIM["neck"]]
        return out

    def skeleton_with(mismatch_m: float):
        rest = np.array(DETAILED_HUMANOID.rest_translations_m, np.float64)
        for name in ("LeftUpperArm", "RightUpperArm"):
            index = DETAILED_HUMANOID.index(name)
            length = float(np.linalg.norm(rest[index]))
            rest[index] = rest[index] * ((length - mismatch_m) / length)
        return DETAILED_HUMANOID.with_rest_translations(rest)

    def d8_substitute(points: np.ndarray):
        shipped = cm._joint_origin
        landmark_for = {"LeftUpperArm": "left_shoulder", "LeftLowerArm": "left_elbow",
                        "RightUpperArm": "right_shoulder", "RightLowerArm": "right_elbow"}

        def origin(world, frame, root_translation, rest, joint_name):
            if joint_name in landmark_for:
                return points[frame, cm.JOINT_INDEX[landmark_for[joint_name]]]
            return shipped(world, frame, root_translation, rest, joint_name)

        return origin

    def solve(positions: np.ndarray, spine: np.ndarray, skeleton, d8_aim: bool):
        """Returns (elbow_mm, wrist_mm, elbow_floor_mm, wrist_floor_mm) per side, BEFORE
        the ground projection -- the claim is about the CONVERTER, and the projection
        translates the root after every aim is taken (D7b's B7 records the same)."""

        shipped = cm._joint_origin
        saved = cm.project_generated_foot_contacts
        captured: list = []

        def watcher(track, **kwargs):
            captured.append(track)
            return saved(track, **kwargs)

        points = rig_points(positions)
        if d8_aim:
            cm._joint_origin = d8_substitute(points)
        cm.project_generated_foot_contacts = watcher
        try:
            cm.positions_to_body_track(
                positions, sample_rate_hz=30, provenance_sha256="0" * 64,
                spine_world_z_up_m=spine, skeleton=skeleton)
        finally:
            cm._joint_origin = shipped
            cm.project_generated_foot_contacts = saved
        track = captured[-1]
        world = forward_kinematics_positions(
            np.asarray(track.root_translation_m, np.float64),
            np.asarray(track.local_rotations_xyzw, np.float64), skeleton=skeleton)
        rest = {joint.name: np.asarray(joint.rest_translation_m, np.float64)
                for joint in skeleton.joints}
        out = {}
        for side, upper, lower, hand, _sh, elbow, wrist in ARMS:
            captured_elbow = points[:, cm.JOINT_INDEX[elbow]]
            captured_wrist = points[:, cm.JOINT_INDEX[wrist]]
            origin = world[:, skeleton.index(upper)]
            delivered_elbow = world[:, skeleton.index(lower)]
            reach = np.linalg.norm(captured_elbow - origin, axis=1)
            placed = origin + (float(np.linalg.norm(rest[lower])) / reach)[:, None] * (
                captured_elbow - origin)
            out[side] = {
                "elbow_mm": 1000.0 * np.linalg.norm(delivered_elbow - captured_elbow, axis=1),
                "wrist_mm": 1000.0 * np.linalg.norm(
                    world[:, skeleton.index(hand)] - captured_wrist, axis=1),
                "elbow_floor_mm": 1000.0 * np.abs(
                    reach - float(np.linalg.norm(rest[lower]))),
                "wrist_floor_mm": 1000.0 * np.abs(
                    np.linalg.norm(captured_wrist - placed, axis=1)
                    - float(np.linalg.norm(rest[hand]))),
            }
        return out, track

    rig = cm.load_camera_rig(syn.RIG)
    cameras = tuple(c.scaled(syn.WORKING_WIDTH, syn.WORKING_HEIGHT) for c in rig)
    clips: dict = {}
    worst_clean_gap_m = 0.0
    for clip, (take, _rest) in syn.load(syn.FAST_STRIDE).items():
        positions = landmarks(take)
        spine = take[:, 1]
        row: dict = {"frames": int(len(take))}
        for mismatch in (CLAVICLE_MISMATCH_M, -CLAVICLE_MISMATCH_M):
            skeleton = skeleton_with(mismatch)
            d9, _ = solve(positions, spine, skeleton, d8_aim=False)
            d8, _ = solve(positions, spine, skeleton, d8_aim=True)
            cell = {}
            for side in ("left", "right"):
                worst_clean_gap_m = max(
                    worst_clean_gap_m,
                    float(np.abs(d9[side]["elbow_mm"] - d9[side]["elbow_floor_mm"]).max()) / 1000.0,
                    float(np.abs(d9[side]["wrist_mm"] - d9[side]["wrist_floor_mm"]).max()) / 1000.0)
                cell[side] = {
                    "aim_from_own_origin": {
                        "elbow_median_mm": round(float(np.median(d9[side]["elbow_mm"])), 4),
                        "wrist_median_mm": round(float(np.median(d9[side]["wrist_mm"])), 4)},
                    "landmark_direction_D8": {
                        "elbow_median_mm": round(float(np.median(d8[side]["elbow_mm"])), 4),
                        "wrist_median_mm": round(float(np.median(d8[side]["wrist_mm"])), 4)},
                    "length_floor_median_mm": {
                        "elbow": round(float(np.median(d9[side]["elbow_floor_mm"])), 4),
                        "wrist": round(float(np.median(d9[side]["wrist_floor_mm"])), 4)},
                    "worst_gap_to_the_floor_mm": {
                        "elbow": round(float(np.abs(
                            d9[side]["elbow_mm"] - d9[side]["elbow_floor_mm"]).max()), 8),
                        "wrist": round(float(np.abs(
                            d9[side]["wrist_mm"] - d9[side]["wrist_floor_mm"]).max()), 8)},
                }
            row[f"clavicle_{int(1000 * mismatch):+d}mm_clean"] = cell
        # ---- I7's own measured noise, three seeds, pooled
        skeleton = skeleton_with(CLAVICLE_MISMATCH_M)
        pooled: dict = {side: {"d9_elbow": [], "d8_elbow": [], "d9_wrist": [],
                               "d8_wrist": [], "floor_elbow": []} for side in ("left", "right")}
        seeds_used = []
        for seed in syn.SEEDS[:3]:
            rng = np.random.default_rng(seed)
            noised = syn.observe(cameras, take, rng, smooth_spine=False)
            if not np.isfinite(noised).all():
                continue
            seeds_used.append(int(seed))
            n_positions, n_spine = landmarks(noised), noised[:, 1]
            d9, _ = solve(n_positions, n_spine, skeleton, d8_aim=False)
            d8, _ = solve(n_positions, n_spine, skeleton, d8_aim=True)
            for side in ("left", "right"):
                pooled[side]["d9_elbow"].append(d9[side]["elbow_mm"])
                pooled[side]["d8_elbow"].append(d8[side]["elbow_mm"])
                pooled[side]["d9_wrist"].append(d9[side]["wrist_mm"])
                pooled[side]["d8_wrist"].append(d8[side]["wrist_mm"])
                pooled[side]["floor_elbow"].append(d9[side]["elbow_floor_mm"])
        row["noisy_I7"] = ({
            "seeds": seeds_used,
            "clavicle_mismatch_mm": int(1000 * CLAVICLE_MISMATCH_M),
            **{side: {key: round(float(np.median(np.concatenate(values))), 4)
                      for key, values in pooled[side].items()}
               for side in ("left", "right")},
        } if seeds_used else "no seed produced finite triangulation")
        clips[clip] = row

    # ---- THE ORACLE. With the clavicle exactly right the two aims are one direction.
    # The fixture is a FIXED POINT: the captured shoulder is replaced by the origin the
    # converter's own pass C reports for `UpperArm`, and the elbow and wrist are put at
    # the rig's own bone lengths, until nothing moves. `tests/test_arm_origin.py` carries
    # the same construction as a unit test.
    oracle: dict = {}
    take = next(iter(syn.load(syn.FAST_STRIDE).values()))[0]
    positions = landmarks(take)
    spine = take[:, 1]
    skeleton = DETAILED_HUMANOID
    rest = {joint.name: np.asarray(joint.rest_translation_m, np.float64)
            for joint in skeleton.joints}
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    shipped = cm._joint_origin
    worst_step = float("inf")
    for iteration in range(1, 12):
        for _side, _upper, lower, hand, shoulder, elbow, wrist in ARMS:
            origin = positions[:, cm.JOINT_INDEX[shoulder]]
            new_elbow = origin + float(np.linalg.norm(rest[lower])) * unit(
                positions[:, cm.JOINT_INDEX[elbow]] - origin)
            new_wrist = new_elbow + float(np.linalg.norm(rest[hand])) * unit(
                positions[:, cm.JOINT_INDEX[wrist]] - positions[:, cm.JOINT_INDEX[elbow]])
            positions[:, cm.JOINT_INDEX[elbow]] = new_elbow
            positions[:, cm.JOINT_INDEX[wrist]] = new_wrist
        seen: dict = {}

        def recorder(world, frame, root_translation, rest_dict, joint_name):
            value = shipped(world, frame, root_translation, rest_dict, joint_name)
            seen[(frame, joint_name)] = np.asarray(value, np.float64).copy()
            return value

        cm._joint_origin = recorder
        try:
            cm.positions_to_body_track(
                positions, sample_rate_hz=30, provenance_sha256="0" * 64,
                spine_world_z_up_m=spine, skeleton=skeleton)
        finally:
            cm._joint_origin = shipped
        worst_step = 0.0
        for _side, upper, _lower, _hand, shoulder, _elbow, _wrist in ARMS:
            for frame in range(len(positions)):
                landmark = to_capture(seen[(frame, upper)])
                worst_step = max(worst_step, float(np.linalg.norm(
                    landmark - positions[frame, cm.JOINT_INDEX[shoulder]])))
                positions[frame, cm.JOINT_INDEX[shoulder]] = landmark
        if worst_step < 1.0e-15:
            break
    points = rig_points(positions)
    cm._joint_origin = d8_substitute(points)
    try:
        d8_track = cm.positions_to_body_track(
            positions, sample_rate_hz=30, provenance_sha256="0" * 64,
            spine_world_z_up_m=spine, skeleton=skeleton)
    finally:
        cm._joint_origin = shipped
    d9_track = cm.positions_to_body_track(
        positions, sample_rate_hz=30, provenance_sha256="0" * 64,
        spine_world_z_up_m=spine, skeleton=skeleton)
    gap = float(np.abs(np.asarray(d8_track.local_rotations_xyzw, np.float64)
                       - np.asarray(d9_track.local_rotations_xyzw, np.float64)).max())
    oracle = {
        "how": ("a fixed point on the fixture: the captured shoulder IS the UpperArm "
                "origin and the arm bones are at the rig's own lengths, so the two aims "
                "are handed the same directions"),
        "fixed_point_iterations": iteration,
        "fixed_point_worst_step_m": worst_step,
        "worst_quaternion_component_gap": gap,
        "band": ORACLE_BAND,
        "within_band": bool(gap <= ORACLE_BAND),
        "bit_identical": bool(np.array_equal(d8_track.local_rotations_xyzw,
                                             d9_track.local_rotations_xyzw)),
        "why_bit_identity_is_not_claimed": (
            "the fixed point converges to ~1e-18 m rather than exactly, and the hands' "
            "locals are a quaternion composed with their own inverse, which rounds with "
            "the parent. The measured gap is seven orders below the float32 storage "
            "floor, so the card's 1e-9 is met on its own terms and is not a band written "
            "below what the file can carry (the D7b precedent)."),
    }
    report["B5_synthetic"] = {
        "band": PRE_REGISTRATION["B5"],
        "reference": ("exact synthetic truth: SOMASKEL77 posed with real GEM-X rotations, "
                      "and I7's own measured heavy-tail frame-correlated pixel noise "
                      "recovered through the real triangulator. MAMMA-FREE."),
        "how_the_D8_arm_is_reached": ("`_joint_origin` substituted to report the captured "
                                      "shoulder for `*UpperArm` and the captured elbow for "
                                      "`*LowerArm` -- the shipped D8 directions at the "
                                      "identical call site, never a re-implementation"),
        "measured_BEFORE_the_ground_projection": (
            "the projection translates the root AFTER the aim is taken, so on a projected "
            "track the identity `error == length floor` is displaced by the hoist. The "
            "claim is about the CONVERTER; B1's `the_ground_projection` block measures "
            "the hoist on the delivered file."),
        "clean_arm_band_m": CLEAN_FLOOR_BAND_M,
        "clean_arm_worst_gap_to_the_floor_m": round(worst_clean_gap_m, 12),
        "clean_arm_reaches_the_floor": bool(worst_clean_gap_m <= CLEAN_FLOOR_BAND_M),
        "oracle": oracle,
        "clips": clips,
        "verdict": ("PASS" if worst_clean_gap_m <= CLEAN_FLOOR_BAND_M
                    and oracle["within_band"] else "FAIL"),
    }


# --------------------------------------------------------------- MAMMA: report only
def mamma_block(report: dict) -> None:
    scoreboards = {
        "D8": ROOT / "artifacts/compare/scoreboard-d8-occlusion.json",
        "D9": ROOT / "artifacts/compare/scoreboard-d9-arms.json",
    }
    block: dict = {
        "band": PRE_REGISTRATION["MAMMA"],
        "never_banded": True,
        "reference": ("MAMMA pred_joints -- AGREEMENT ONLY. It reports and selects "
                      "nothing, and nothing from it enters src or a constant."),
        "subject_map": ("tools/head/subject_map.py -- MAMMA's body_id-00 is our subject 1; "
                        "pairing by index silently crosses the performers."),
    }
    for label, path in scoreboards.items():
        block[label] = json.loads(path.read_text()) if path.exists() else "NOT RUN"
    report["MAMMA_reported"] = block


# --------------------------------------------------------------------------- hygiene
def hygiene_block(report: dict) -> None:
    rows = {}
    for label, path in (("hygiene_arm_src_UNCHANGED", OUT_DIR / "delivery-hygiene-build.json"),
                        ("d9_arm", OUT_DIR / "delivery-build.json")):
        rows[label] = json.loads(path.read_text()) if path.exists() else "NOT RUN"
    identical = (rows["hygiene_arm_src_UNCHANGED"]["hygiene"]["all_delivered_files_identical"]
                 if isinstance(rows["hygiene_arm_src_UNCHANGED"], dict) else False)
    report["hygiene"] = {
        "what_it_proves": ("the rebuild harness reproduces the SHIPPED delivery byte for "
                           "byte when src is unchanged, so every difference in the D9 arm "
                           "belongs to the src change and to nothing else."),
        "work_source": ("artifacts/commercial-multiview-soma77/work, copied never "
                        "symlinked, hashes asserted before and after the build."),
        "builds": rows,
        "shipped_delivery_never_written": True,
        "verdict": "PASS" if identical else "FAIL",
    }


# ------------------------------------------------------------------------------- main
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = measured()
    report: dict = {
        "title": "D9 -- aim the arms from their own origins, so the delivered arm sits on "
                 "the captured points",
        "preregistration": PRE_REGISTRATION,
        "same_denominator": payload["same_denominator"],
        "deliveries": payload["deliveries"],
        "measured_report": str(MEASURED.relative_to(ROOT)),
    }
    floor_block(report, payload)
    b1_block(report, payload)
    b2_block(report, payload)
    b3_block(report)
    b4_block(report, payload)
    b5_block(report)
    mamma_block(report)
    hygiene_block(report)

    banded = {
        "B1_placement": report["B1_placement"]["verdict"],
        "B2_untouchable": report["B2_untouchable"]["verdict"],
        "B3_silhouette": report["B3_silhouette"]["verdict"],
        "B4_closure_and_denominator": report["B4_closure_and_denominator"]["verdict"],
    }
    reported = {"B5_synthetic": report["B5_synthetic"]["verdict"],
                "floor": report["floor"]["verdict"],
                "MAMMA_reported": "REPORTED",
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
