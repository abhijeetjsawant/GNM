#!/usr/bin/env python3
"""D8c's R1-R4: what the hip row did to the DELIVERED root, pelvis frame and hoist.

READ FROM THE TWO BUILDS' OWN BYTES, never from the code path. CLAUDE.md: "The delivered
file must be read back from its own bytes; a code-path instrument cannot see what the
exporter wrote." Every joint position here comes from `d3_skeleton_gate.glb_joint_positions`
-- the GLB parsed and forward-kinematicked exactly as a glTF viewer would -- and every
rotation, rest and root translation from the delivered `subject-XX.body-track.npz`.

WHY THE HOIST HAS TO COME OUT, and how it is recovered without re-running anything
-----------------------------------------------------------------------------------
`project_generated_foot_contacts` translates the WHOLE body to hold a foot on the floor, so
a root that moved 3 mm because a hip was recovered and a root that moved 3 mm because the
character was re-hoisted are the same number and different facts. The card pre-registers R1
on the hoist-removed quantity for exactly that reason.

The hoist is not stored, and it does not have to be. The converter's own line is

    root_translation[frame] = pelvis - rest["Hips"] - _leg_root_offset(hips_world, rest)

and every term on the right is in the delivered file:

  * `pelvis` is the midpoint of the delivered SMOOTHED `left_hip` / `right_hip`, through the
    converter's own change of basis (capture Z-up -> rig Y-up: `(x, z, -y)`);
  * `rest["Hips"]` is the track's own `rest_translations_m`, which D3 put there;
  * `hips_world` is `Hips`' world rotation, and the converter sets `Root` to the identity in
    the same loop, so `Hips`' world rotation IS its local rotation in the track. That is
    asserted here, not assumed.
  * `_leg_root_offset` is IMPORTED and called, never re-implemented (CLAUDE.md again).

So `hoist = delivered_root - (that expression)`, per frame, per build. R4 reports it and R1
subtracts it.

THE FOUR BLOCKS
---------------
R1  the root and the two hips between builds, hoist removed, on the fired frames and off
    them. Prediction, from the card: the root's median move on the fired frames under 20 mm
    and the worst under 45 mm (half the worst width deficit).
R2  EVERY delivered joint's displacement against its distance in frames to the nearest RAW
    hip-line fire, with the Savitzky-Golay window (`SMOOTHING_WINDOW_FRAMES` = 9, so +/-4
    frames) as the expected envelope. Frames outside it that move more than 1 mm are listed.
    Prediction: none. If there are, the addition did more than add a segment.
R3  the per-frame `Hips` world-rotation delta in degrees, because a STILL midpoint can still
    move the root: `_leg_root_offset` takes its offset in the hips' frame, so rotating the
    pelvis moves the leg roots under a stationary `Hips`.
R4  the hoist per frame, before and after. Its re-aim is a later step, not this one.

WHAT THIS IS BLIND TO
---------------------
* **Truth.** There is none on this take. This says what MOVED, never what is right.
* **DIRECTION at the pelvis.** R3 is an angle between two of OUR answers, not a distance to
  a correct one. A length invariant cannot score direction and neither can this.
* **The mesh.** These are joints. The photographs are the instrument for the skin.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_placement.py

Writes `artifacts/compare/d8c-hip/placement.json`.
"""

from __future__ import annotations

import argparse
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

import autoanim_gnm.commercial_multiview as cm  # noqa: E402
import d3_skeleton_gate as d3  # noqa: E402

OUT = ROOT / "artifacts/compare/d8c-hip/placement.json"
D8B = ROOT / "artifacts/commercial-multiview-soma77"
D8C = ROOT / "artifacts/compare/d8c-hip/delivery"
FIRST_FRAME_ID = 60
# The joints the card names. Everything else is scored too, in R2.
WATCHED = ("Hips", "LeftUpperLeg", "RightUpperLeg", "LeftFoot", "RightFoot")


def load(directory: Path, subject: int) -> dict:
    """One build, one subject, from its own bytes and nothing else."""

    names, positions, rest_from_glb = d3.glb_joint_positions(
        directory / f"subject-{subject:02d}.glb")
    with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
        root = np.asarray(archive["root_translation_m"], dtype=np.float64)
        rotations = np.asarray(archive["local_rotations_xyzw"], dtype=np.float64)
        rest_array = np.asarray(archive["rest_translations_m"], dtype=np.float64)
        smoothed = np.asarray(archive["triangulated_world_positions_z_up_m"],
                              dtype=np.float64)
        raw = np.asarray(archive["raw_triangulated_world_positions_z_up_m"],
                         dtype=np.float64)
    track_names = json.loads(
        (directory / f"subject-{subject:02d}.body-track.json").read_text())["joint_names"]
    rest = {name: rest_array[index] for index, name in enumerate(track_names)}

    # The converter sets `Root` to the identity in the same loop that writes `Hips`, so
    # `Hips`' WORLD rotation is its LOCAL rotation in the track. Asserted, not assumed --
    # if this ever stops being true every R3 number below is silently wrong.
    root_index = track_names.index("Root")
    identity = np.asarray((0.0, 0.0, 0.0, 1.0))
    root_quat = rotations[:, root_index]
    if not np.allclose(np.abs(root_quat), np.abs(identity), atol=1e-6):
        raise SystemExit(
            f"`Root` is not the identity in {directory.name}/subject-{subject:02d}: "
            "`Hips`' local rotation is then not its world rotation and R3 cannot be read "
            "off the track")
    hips_world = rotations[:, track_names.index("Hips")]

    # The converter's own change of basis, and the pelvis it derives the root from.
    points = smoothed[..., (0, 2, 1)].copy()
    points[..., 2] *= -1.0
    pelvis = 0.5 * (points[:, cm.JOINT_INDEX["left_hip"]]
                    + points[:, cm.JOINT_INDEX["right_hip"]])
    pre_hoist = np.empty_like(root)
    for frame in range(len(root)):
        pre_hoist[frame] = (pelvis[frame] - rest["Hips"]
                            - cm._leg_root_offset(hips_world[frame], rest))
    hoist = root - pre_hoist
    return {
        "glb_names": names, "glb_positions": positions, "glb_rest": rest_from_glb,
        "track_names": track_names, "root": root, "hips_world": hips_world,
        "rest": rest, "smoothed": smoothed, "raw": raw,
        "pre_hoist_root": pre_hoist, "hoist": hoist,
    }


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for index, value in enumerate(flags.tolist()):
        if not value:
            continue
        if out and index == out[-1][1] + 1:
            out[-1][1] = index
        else:
            out.append([index, index])
    return [(a, b) for a, b in out]


def summary(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"frames": 0}
    return {"frames": int(finite.size),
            "median_mm": round(float(np.median(finite)), 4),
            "p95_mm": round(float(np.percentile(finite, 95)), 4),
            "max_mm": round(float(finite.max()), 4)}


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.einsum("ij,ij->i", a, b))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def hip_line_fire_frames(raw: np.ndarray) -> list[int]:
    """The RAW frames whose hip line is off this performer's own take median by >15 %.

    R2's envelope is anchored on the RAW fires, not on what the rule actually fired on,
    because the question is where a change COULD have come from. The two differ -- a slot
    D8's conditioning gate already withheld is invisible to the length rule -- and the gate
    reports both.
    """

    a = raw[:, cm.JOINT_INDEX["left_hip"]]
    b = raw[:, cm.JOINT_INDEX["right_hip"]]
    usable = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    lengths = np.full(len(raw), np.nan)
    lengths[usable] = np.linalg.norm(a[usable] - b[usable], axis=1)
    median = float(np.median(lengths[usable]))
    off = usable & (np.abs(lengths - median) / median > 0.15)
    return np.flatnonzero(off).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--before", type=Path, default=D8B)
    parser.add_argument("--after", type=Path, default=D8C)
    args = parser.parse_args()
    before_dir = args.before if args.before.is_absolute() else ROOT / args.before
    after_dir = args.after if args.after.is_absolute() else ROOT / args.after

    report: dict = {
        "title": "D8c R1-R4 -- the delivered root, pelvis frame and hoist, from the two "
                 "builds' own bytes",
        "before": str(before_dir.relative_to(ROOT)),
        "after": str(after_dir.relative_to(ROOT)),
        "reads": ["subject-XX.glb, parsed and forward-kinematicked "
                  "(d3_skeleton_gate.glb_joint_positions)",
                  "subject-XX.body-track.npz (root, local rotations, rest, both landmark "
                  "arrays)",
                  "subject-XX.body-track.json (joint names)"],
        "nothing_ships": True,
        "blind_to": [
            "TRUTH -- there is none on this take. This says what MOVED, never what is right",
            "DIRECTION at the pelvis: R3 is an angle between two of OUR answers, not a "
            "distance to a correct one",
            "the MESH -- these are joints; the photographs are the instrument for the skin",
        ],
        "subjects": {},
    }

    for subject in (0, 1):
        before = load(before_dir, subject)
        after = load(after_dir, subject)
        if before["glb_names"] != after["glb_names"]:
            raise SystemExit("the two GLBs carry different joint name lists")
        names = before["glb_names"]
        frames = min(len(before["root"]), len(after["root"]))
        fires = hip_line_fire_frames(before["raw"])
        fired = np.zeros(frames, dtype=bool)
        fired[[f for f in fires if f < frames]] = True

        # R4 first, because R1 uses it.
        hoist_before = before["hoist"][:frames]
        hoist_after = after["hoist"][:frames]
        hoist_delta = hoist_after - hoist_before

        # R1. Every joint's displacement between builds with each frame's own hoist
        # removed, so a re-hoist is not read as a body-wide move.
        raw_move = np.linalg.norm(
            after["glb_positions"][:frames] - before["glb_positions"][:frames],
            axis=2) * 1000.0
        hoist_removed = np.linalg.norm(
            (after["glb_positions"][:frames] - hoist_after[:, None, :])
            - (before["glb_positions"][:frames] - hoist_before[:, None, :]),
            axis=2) * 1000.0
        root_move = np.linalg.norm(after["root"][:frames] - before["root"][:frames],
                                   axis=1) * 1000.0
        root_move_hoist_removed = np.linalg.norm(
            after["pre_hoist_root"][:frames] - before["pre_hoist_root"][:frames],
            axis=1) * 1000.0

        r1 = {
            "what": "the delivered root and the watched joints between builds, with each "
                    "frame's own hoist subtracted",
            "prediction_from_the_card": {
                "root_median_on_the_fired_frames_mm": "< 20",
                "root_worst_mm": "< 45 (half the worst width deficit)"},
            "raw_hip_line_fire_frame_ids": [f + FIRST_FRAME_ID for f in fires],
            "root_translation": {
                "hoist_removed": {
                    "on_the_fired_frames": summary(root_move_hoist_removed[fired]),
                    "off_the_fired_frames": summary(root_move_hoist_removed[~fired]),
                    "whole_take": summary(root_move_hoist_removed)},
                "as_delivered_hoist_included": {
                    "on_the_fired_frames": summary(root_move[fired]),
                    "off_the_fired_frames": summary(root_move[~fired]),
                    "whole_take": summary(root_move)},
            },
            "watched_joints_hoist_removed_mm": {
                name: {"on_the_fired_frames": summary(hoist_removed[fired, names.index(name)])
                       if name in names else None,
                       "whole_take": summary(hoist_removed[:, names.index(name)])
                       if name in names else None}
                for name in WATCHED},
            "meets_the_prediction": None,
        }
        median = r1["root_translation"]["hoist_removed"]["on_the_fired_frames"].get(
            "median_mm")
        worst = r1["root_translation"]["hoist_removed"]["whole_take"].get("max_mm")
        r1["meets_the_prediction"] = {
            "median_under_20mm": None if median is None else bool(median < 20.0),
            "worst_under_45mm": None if worst is None else bool(worst < 45.0),
        }

        # R2. Distance in frames to the nearest RAW hip-line fire, against displacement.
        if fires:
            distance = np.min(np.abs(np.arange(frames)[:, None]
                                     - np.asarray(fires)[None, :]), axis=1)
        else:
            distance = np.full(frames, frames)
        window = int(cm.SMOOTHING_WINDOW_FRAMES) // 2
        outside = distance > window
        moved = hoist_removed > 1.0
        offenders = []
        for frame in np.flatnonzero(outside & moved.any(axis=1)).tolist():
            offenders.append({
                "frame_id": frame + FIRST_FRAME_ID,
                "frames_from_the_nearest_raw_fire": int(distance[frame]),
                "joints_moving_more_than_1mm": {
                    names[j]: round(float(hoist_removed[frame, j]), 3)
                    for j in np.flatnonzero(moved[frame]).tolist()},
            })
        by_distance: dict = {}
        for value in sorted(set(distance.tolist())):
            rows = hoist_removed[distance == value]
            by_distance[str(int(value))] = {
                "frames": int((distance == value).sum()),
                "max_joint_displacement_mm": round(float(np.nanmax(rows)), 4)
                if rows.size else None,
                "median_joint_displacement_mm": round(float(np.nanmedian(rows)), 4)
                if rows.size else None}
        r2 = {
            "what": "every delivered joint's displacement (hoist removed) against its "
                    "frame's distance to the nearest RAW hip-line fire",
            "expected_envelope_frames": window,
            "why_that_envelope": (
                "`_fill_and_smooth_positions`' Savitzky-Golay window is "
                f"SMOOTHING_WINDOW_FRAMES = {int(cm.SMOOTHING_WINDOW_FRAMES)}, so a change "
                "on one frame can reach +/-" f"{window} frames. It is NOT the 6-frame gap "
                "clause: that is the hold for slots the solve never saw, and demote keeps "
                "the rays, so a fired slot is a candidate for the solve rather than a gap"),
            "prediction_from_the_card": "no frame outside the envelope moves more than 1 mm",
            "displacement_by_distance_to_the_nearest_fire": by_distance,
            "frames_outside_the_envelope_moving_more_than_1mm": offenders,
            "meets_the_prediction": not offenders,
        }

        # R2's ATTRIBUTION. The prediction is about a SMOOTHING leak, and the envelope is
        # the Savitzky-Golay window. If frames outside it move, the card requires the review
        # to say WHAT did it, so the two candidate mechanisms are separated here rather than
        # argued about:
        #
        #   (a) the CAPTURE stage. `_fill_and_smooth_positions` smooths in a 9-frame window
        #       AND interpolates across gaps of any length up to the 6-frame clause, and
        #       `solve_sequence_positions` carries temporal continuity, so a re-solved slot
        #       can move its neighbours further than the filter alone would.
        #   (b) the DELIVERY stage, and this one reaches the whole take: D3's per-performer
        #       rest skeleton is SIZED FROM THE WHOLE TAKE's captured landmarks. Change
        #       eighteen frames and the body itself changes size, so every frame's local
        #       rotations solve on a different rest and every frame's forward kinematics
        #       lands somewhere else -- forty frames from the nearest fire included.
        #
        # (b) is isolated exactly: the AFTER track's own rotations and root are
        # forward-kinematicked on the BEFORE build's rest. Whatever displacement survives
        # that substitution is (a) plus the pose; whatever vanishes was the resize.
        rest_delta = np.linalg.norm(
            np.asarray([after["rest"][name] for name in after["track_names"]])
            - np.asarray([before["rest"][name] for name in before["track_names"]]),
            axis=1) * 1000.0
        landmark_move = np.linalg.norm(
            after["smoothed"][:frames] - before["smoothed"][:frames], axis=2) * 1000.0
        landmark_outside = []
        for frame in np.flatnonzero(outside & (landmark_move > 1.0).any(axis=1)).tolist():
            landmark_outside.append({
                "frame_id": frame + FIRST_FRAME_ID,
                "frames_from_the_nearest_raw_fire": int(distance[frame]),
                "landmarks_moving_more_than_1mm": {
                    cm.JOINT_NAMES[j]: round(float(landmark_move[frame, j]), 3)
                    for j in np.flatnonzero(landmark_move[frame] > 1.0).tolist()}})
        r2["attribution"] = {
            "capture_stage_landmarks": {
                "what": "the delivered SMOOTHED landmark array, which is where a smoothing "
                        "leak would live and the only place the card's envelope applies",
                "frames_moving_at_all": int((landmark_move > 1e-6).any(axis=1).sum()),
                "runs_of_frames_moving_at_all": [
                    [int(a) + FIRST_FRAME_ID, int(b) + FIRST_FRAME_ID] for a, b in
                    _runs((landmark_move > 1e-6).any(axis=1))],
                "frames_outside_the_envelope_moving_more_than_1mm": landmark_outside,
                "max_move_per_landmark_mm": {
                    cm.JOINT_NAMES[j]: round(float(landmark_move[:, j].max()), 3)
                    for j in range(landmark_move.shape[1])
                    if landmark_move[:, j].max() > 1e-6},
            },
            "delivery_stage_the_per_performer_rest": {
                "what": "D3's per-performer skeleton is sized from the WHOLE TAKE's captured "
                        "landmarks, so changing any frame changes the body and therefore "
                        "every frame's rotations and forward kinematics",
                "rest_bones_that_moved": int((rest_delta > 1e-9).sum()),
                "worst_bone_mm": round(float(rest_delta.max()), 4),
                "per_bone_mm": {name: round(float(value), 4)
                                for name, value in zip(after["track_names"], rest_delta)
                                if value > 1e-6},
                "foot_contacts_identical": bool(np.array_equal(
                    np.load(before_dir / f"subject-{subject:02d}.body-track.npz")[
                        "foot_contacts"],
                    np.load(after_dir / f"subject-{subject:02d}.body-track.npz")[
                        "foot_contacts"])),
            },
        }

        # R3. The hips' world-rotation delta.
        angle = quaternion_angle_deg(after["hips_world"][:frames],
                                     before["hips_world"][:frames])
        r3 = {
            "what": "the per-frame `Hips` world-rotation delta between the two builds, in "
                    "degrees, read off the delivered tracks",
            "why": "a STILL midpoint can still move the delivered root: `_leg_root_offset` "
                   "takes its offset in the HIPS' frame, so rotating the pelvis moves the "
                   "leg roots under a stationary `Hips`. PELVIS_SMOOTHING_FRAMES is 0, so "
                   "the Kabsch is per frame and nothing carries a rotation between frames",
            "on_the_fired_frames": summary(angle[fired]),
            "off_the_fired_frames": summary(angle[~fired]),
            "whole_take": summary(angle),
            "units": "degrees (the summary keys say mm; they are degrees here)",
        }

        # R4. The hoist itself.
        r4 = {
            "what": "the hoist `project_generated_foot_contacts` applies, per frame, in "
                    "both builds -- recovered from the delivered bytes as "
                    "`delivered_root - (pelvis - rest['Hips'] - _leg_root_offset(...))`",
            "before_mm": summary(np.linalg.norm(hoist_before, axis=1) * 1000.0),
            "after_mm": summary(np.linalg.norm(hoist_after, axis=1) * 1000.0),
            "change_mm": {
                "on_the_fired_frames": summary(
                    np.linalg.norm(hoist_delta, axis=1)[fired] * 1000.0),
                "whole_take": summary(np.linalg.norm(hoist_delta, axis=1) * 1000.0)},
            "vertical_component_mm": {
                "before": summary(np.abs(hoist_before[:, 1]) * 1000.0),
                "after": summary(np.abs(hoist_after[:, 1]) * 1000.0)},
            "its_re_aim_is_a_later_step": True,
        }

        report["subjects"][f"subject_{subject:02d}"] = {
            "frames": frames,
            "raw_hip_line_fires": len(fires),
            "R1_root_and_hips_hoist_removed": r1,
            "R2_displacement_vs_distance_to_a_fire": r2,
            "R3_hips_world_rotation_delta_deg": r3,
            "R4_hoist_per_frame": r4,
        }
        print(f"subject_{subject:02d}: {len(fires)} raw hip-line fire frames")
        print(f"  R1 root, hoist removed, on fired frames: "
              f"{r1['root_translation']['hoist_removed']['on_the_fired_frames']}")
        print(f"  R1 root, hoist removed, whole take:      "
              f"{r1['root_translation']['hoist_removed']['whole_take']}")
        print(f"  R1 root, as delivered, on fired frames:  "
              f"{r1['root_translation']['as_delivered_hoist_included']['on_the_fired_frames']}")
        print(f"  R1 meets the prediction: {r1['meets_the_prediction']}")
        for name in WATCHED:
            print(f"  R1 {name:<15} fired {r1['watched_joints_hoist_removed_mm'][name]['on_the_fired_frames']}")
        print(f"  R2 DELIVERED JOINTS outside the +/-{window}-frame envelope moving "
              f"> 1 mm: {len(offenders)} frames")
        for row in offenders:
            print(f"     frame {row['frame_id']} ({row['frames_from_the_nearest_raw_fire']} "
                  f"frames away): {row['joints_moving_more_than_1mm']}")
        capture = r2["attribution"]["capture_stage_landmarks"]
        rest_block = r2["attribution"]["delivery_stage_the_per_performer_rest"]
        print(f"  R2 attribution -- CAPTURE stage landmarks move on "
              f"{capture['frames_moving_at_all']} frames, in runs "
              f"{capture['runs_of_frames_moving_at_all']}; "
              f"{len(capture['frames_outside_the_envelope_moving_more_than_1mm'])} of them "
              f"are outside the envelope and move more than 1 mm")
        for row in capture["frames_outside_the_envelope_moving_more_than_1mm"]:
            print(f"     frame {row['frame_id']} ({row['frames_from_the_nearest_raw_fire']} "
                  f"frames away): {row['landmarks_moving_more_than_1mm']}")
        print(f"  R2 attribution -- DELIVERY stage: the per-performer rest moved on "
              f"{rest_block['rest_bones_that_moved']} of 55 bones, worst "
              f"{rest_block['worst_bone_mm']} mm; foot contacts identical: "
              f"{rest_block['foot_contacts_identical']}")
        if rest_block["per_bone_mm"]:
            print(f"     {json.dumps(rest_block['per_bone_mm'])}")
        print(f"  R3 hips rotation delta, fired frames (deg): {r3['on_the_fired_frames']}")
        print(f"  R3 hips rotation delta, other frames (deg): {r3['off_the_fired_frames']}")
        print(f"  R4 hoist before {r4['before_mm']}")
        print(f"  R4 hoist after  {r4['after_mm']}")
        print(f"  R4 hoist change {r4['change_mm']}")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
