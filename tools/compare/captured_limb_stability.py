#!/usr/bin/env python3
"""D8's first deliverable: what the CAPTURED limbs do when two bodies overlap.

INSTRUMENT ONLY. Nothing here ships and nothing here is a band. It is written and
committed BEFORE any change to `src/`, and its job is to reproduce the coordinator's
measured effect from the delivered artifacts alone, so that the src change is scored
against a figure that already existed (the D7b way,
`docs/reviews/trunk-resolve-2026-09-05.md` section 2).

WHAT IT READS, AND WHAT IT NEVER DOES
-------------------------------------
* `artifacts/commercial-multiview-soma77/subject-XX.body-track.npz` -- both landmark
  arrays the delivery carries, `raw_triangulated_world_positions_z_up_m` (the pre-fill
  triangulation, NaNs intact) and `triangulated_world_positions_z_up_m` (after the
  sequence solve, the fill and the Savitzky-Golay window; the array the exporter
  consumes).
* `artifacts/commercial-multiview-soma77/work/*-soma77-observations.jsonl` -- the
  delivered build's own cached per-camera detections.
* `artifacts/commercial-multiview-soma77/camera-rig.json`.

**Nothing is re-detected.** **Nothing is re-triangulated.** **Nothing is written under
`artifacts/commercial-multiview-soma77/`.**

WHICH DETECTION BELONGS TO WHICH PERFORMER
------------------------------------------
Two answers are computed and compared, because they can disagree and the disagreement is
itself a finding.

1. **The pipeline's own association**, obtained by wrapping the real
   `reconstruct_multiview` with a recording associator -- the rows the associator hands
   back are the very rows it was given, so this is the pipeline's own assignment and not a
   second copy of its association loop (CLAUDE.md: a hand replication of that loop drifted
   9-19 mm; `tools/compare/temporal.py::real_run_seen_mask` is the same idiom and this
   file's boolean reduction is asserted equal to it). This is the arm every per-view
   figure below is quoted on, because it is the assignment the delivered numbers came
   from.
2. **The coordinator's proxy**: per camera, the person whose 2D hip midpoint is nearest
   the projection of the smoothed 3D hip midpoint. Reported as a cross-check, with the
   count of window frames where the two disagree.

THE FIGURES THIS MUST REPRODUCE (from the D8 card, measured 2026-09-05 before any band)
---------------------------------------------------------------------------------------
legs +/-5 % p5-p95 with 0 frames off the performer's own median by >15 %; performer 1
forearm L 18 frames off and upper arm L 12; the shoulder line 245-375 mm on performer 0
(22 frames off) and 265-543 on performer 1 (27 frames), reading 68 mm at frame 108 and
554 at frame 100; B001/D001 dropping the falling performer on 20-32 of the window's 41
frames at confidence 0.05-0.13; and A001-C001 171-172 degrees apart at frames 100 and 108.

**Two arrays, and the card quotes both.** Reproduced here and stated so it is never
misread: the RANGES and the frame COUNTS are on the SMOOTHED array; the two point values
(68 mm at frame 108, 554 at frame 100) are on the RAW array, where the same frames read
101.7 and 542.9 smoothed. Every figure below is emitted on both arrays with the array
named in the key, and `reproduction` says which array each card figure came from. The card
did not say, and reading one array for all of them does not reproduce it.

FRAME NUMBERING. The take is 150 frames whose own `frame_index` values run 60..209. The
card's window "85-125" and its "frame 100" / "frame 108" are those absolute ids: window
ids 85..125 are array indices 25..65, which is 41 frames, which is the card's denominator.
Every per-frame row below carries both `frame_id` and `frame_index`.

EXTENDED FOR D8b (2026-09-06), and the default path is unchanged
---------------------------------------------------------------
D8b is D8's leftover: after D8 the same instrument still counts performer 1's shoulder line
off its own width by more than 15 % on frames the card names, and the mechanism there is a
different one -- three cameras SEE the performer and all three agree with a collapsed
shoulder, so no conditioning, epipolar or reprojection gate can fire. Two things were added
and nothing was removed:

* `--reproduce {d8,d8b,none}`. `d8` is the default and is the original behaviour;
  `--skip-reproduction` is kept as an alias for `none`. `d8b` checks the D8b card's own
  figures instead of D8's -- the D8 clauses describe the pre-D8 defect and asserting them on
  a post-D8 build would be asking the repair to leave the defect in place.
* A PER-CAMERA CLASSIFICATION TABLE over a frame range (`--classify-frames FIRST LAST`,
  default 110-122, emitted with `--reproduce d8b`): per performer, per landmark, per frame,
  which cameras supported the slot, each camera's confidence, and each camera's reprojection
  residual of the RAW point. The D8b card's claim is "cameras A, C and D all see him
  (confidence 0.4-0.95) and all three agree with the collapsed point to 1-11 px", and this
  table is that claim in a report rather than in a chat.

WHAT THIS INSTRUMENT IS BLIND TO
--------------------------------
* **Truth.** There is none on this take. A segment holding its own median is not a correct
  segment: a limb frozen at the median scores perfectly here (and that is why the D8 bands
  live on synthetic truth and on the photographs, not on this file).
* **A common-mode error.** If every camera is wrong the same way, the reprojection
  residuals stay small and the ray angles stay wide; nothing here fires.
* **Direction.** A length invariant cannot score direction (CLAUDE.md). The shoulder line
  collapsing along the A-C axis is visible here only because it changes LENGTH; a
  same-length rotation about that axis is invisible.
* **The delivered file.** This scores the capture, not the rig or the mesh bound to it.
  `tools/compare/delivered_vs_capture.py` and the part-wise silhouette are for those.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/captured_limb_stability.py

Writes `artifacts/compare/d8-occlusion/limb-stability.json`. Exit 0 if every reproduction
clause matches the card, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402

DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
WORK = DELIVERY / "work"
RIG = DELIVERY / "camera-rig.json"
OUT = ROOT / "artifacts/compare/d8-occlusion/limb-stability.json"

# The delivery whose LANDMARK ARRAYS are read. The observations, the rig and the
# association always come from the shipped build: they are the same bytes in every build
# (the rebuild harness asserts it) and re-associating per arm would put a second variable
# in a comparison that is supposed to have one. Only the two landmark arrays move.
LANDMARK_SOURCE = DELIVERY

CAMERAS = ("A001", "B001", "C001", "D001")
SAMPLE_RATE_HZ = 30
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720

# The card's push-and-fall window, in the observations' own absolute frame ids, inclusive.
WINDOW_FIRST_ID, WINDOW_LAST_ID = 85, 125

# The support threshold is the pipeline's own, imported and never re-typed: a view counts
# as supporting a slot when its observation is finite and its confidence clears the same
# floor `triangulate_point` uses.
MINIMUM_CONFIDENCE = 0.25

# Off the performer's own take median by more than this is "off". The card's classification
# threshold, and a REPORTING cut here -- nothing selects on it.
OFF_MEDIAN_FRACTION = 0.15

# (parent, child) landmark pairs, grouped. The shoulder line is not a limb; it is the pair
# the card names as the thing that collapses, and it is kept in its own group so it never
# shares a summary row with a bone.
# D8b's own frame range and the landmarks the per-camera classification table covers. The
# card names frames 110-122 on the falling performer; the shoulders are the segment that
# collapses and the elbow and wrist are what the collapse drags after it.
D8B_FIRST_ID, D8B_LAST_ID = 110, 122
D8B_CLASSIFY_LANDMARKS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                          "left_wrist", "right_wrist", "neck")

SEGMENTS = {
    "arms": (
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ),
    "legs": (
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ),
    "shoulder_line": (("left_shoulder", "right_shoulder"),),
    "hip_line": (("left_hip", "right_hip"),),
}
ARM_LANDMARKS = ("left_shoulder", "left_elbow", "left_wrist",
                 "right_shoulder", "right_elbow", "right_wrist")
ARRAYS = ("raw", "smoothed")
ARRAY_KEYS = {"raw": "raw_triangulated_world_positions_z_up_m",
              "smoothed": "triangulated_world_positions_z_up_m"}


# ------------------------------------------------------------------------------- readers
def cameras_scaled() -> tuple:
    return tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT)
                 for camera in cm.load_camera_rig(RIG))


def records() -> list[list[dict]]:
    return [cm.load_observation_jsonl(WORK / f"{name}-soma77-observations.jsonl")
            for name in CAMERAS]


def landmark_arrays(subject: int) -> dict[str, np.ndarray]:
    with np.load(LANDMARK_SOURCE / f"subject-{subject:02d}.body-track.npz") as archive:
        return {name: np.asarray(archive[key], np.float64)
                for name, key in ARRAY_KEYS.items()}


def assigned_observations(cameras, rows) -> np.ndarray:
    """`[subject, frame, camera, joint, 3]` of (x, y, confidence), the PIPELINE's own.

    A recording wrapper around the real `associate_frame_graph`, run inside the real
    `reconstruct_multiview`. `temporal.real_run_seen_mask` is the identical idiom and the
    caller asserts this array's boolean reduction equals what that function returns.
    """

    captured: list[np.ndarray] = []

    def recording_associator(*args, **kwargs):
        result = cm.associate_frame_graph(*args, **kwargs)
        captured.append(np.array(result[0], copy=True))
        return result

    cm.reconstruct_multiview(cameras, rows, subject_count=2,
                             sample_rate_hz=SAMPLE_RATE_HZ,
                             associator=recording_associator)
    return np.stack(captured, axis=1)


def proxy_assignment(cameras, rows, smoothed: dict[int, np.ndarray]) -> np.ndarray:
    """The coordinator's own person match: nearest 2D hip midpoint to the projected 3D one.

    `[subject, frame, camera]` of detection index, -1 where no detection has both hips.
    Reported as a CROSS-CHECK of the pipeline's association and never used for a figure.
    """

    frames = len(rows[0])
    subjects = sorted(smoothed)
    out = np.full((len(subjects), frames, len(cameras)), -1, dtype=np.int64)
    left, right = cm.JOINT_INDEX["left_hip"], cm.JOINT_INDEX["right_hip"]
    for subject_slot, subject in enumerate(subjects):
        hips3d = 0.5 * (smoothed[subject][:, left] + smoothed[subject][:, right])
        for camera_index, camera in enumerate(cameras):
            for frame in range(frames):
                if not np.isfinite(hips3d[frame]).all():
                    continue
                target, depth = camera.project(hips3d[frame])
                if depth <= 0.0:
                    continue
                best, best_distance = -1, math.inf
                for index, person in enumerate(rows[camera_index][frame]["people"]):
                    joints = person.get("joints", {})
                    if "left_hip" not in joints or "right_hip" not in joints:
                        continue
                    midpoint = np.asarray([
                        0.5 * (joints["left_hip"]["x"] + joints["right_hip"]["x"]),
                        0.5 * (joints["left_hip"]["y"] + joints["right_hip"]["y"])])
                    distance = float(np.linalg.norm(midpoint - target))
                    if distance < best_distance:
                        best, best_distance = index, distance
                out[subject_slot, frame, camera_index] = best
    return out


def pipeline_assignment(assigned: np.ndarray, rows) -> np.ndarray:
    """`[subject, frame, camera]` detection index recovered from the associator's own rows."""

    subjects, frames, cameras_n = assigned.shape[:3]
    out = np.full((subjects, frames, cameras_n), -1, dtype=np.int64)
    for camera in range(cameras_n):
        for frame in range(frames):
            people = [cm._person_array(person)
                      for person in rows[camera][frame]["people"]]
            for subject in range(subjects):
                row = assigned[subject, frame, camera]
                if not np.isfinite(row).any():
                    continue
                hits = [index for index, candidate in enumerate(people)
                        if np.array_equal(row, candidate, equal_nan=True)]
                if len(hits) == 1:
                    out[subject, frame, camera] = hits[0]
    return out


# ---------------------------------------------------------------------------- statistics
def run_lengths(flags: np.ndarray) -> list[dict]:
    """Contiguous runs of True in a 1-D boolean, as {start, length} on ARRAY indices."""

    out: list[dict] = []
    start = None
    for index, value in enumerate(flags.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            out.append({"first_index": start, "length": index - start})
            start = None
    if start is not None:
        out.append({"first_index": start, "length": len(flags) - start})
    return out


def segment_summary(values: np.ndarray) -> dict:
    """One segment's length series in millimetres against its own take median."""

    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"frames_measured": 0}
    median = float(np.median(finite))
    ratio = np.abs(values - median) / median
    off = np.isfinite(ratio) & (ratio > OFF_MEDIAN_FRACTION)
    return {
        "frames_measured": int(finite.size),
        "median_mm": round(median, 2),
        "min_mm": round(float(finite.min()), 2),
        "max_mm": round(float(finite.max()), 2),
        "p5_mm": round(float(np.percentile(finite, 5)), 2),
        "p95_mm": round(float(np.percentile(finite, 95)), 2),
        "p5_p95_fraction_of_median": [
            round(float(np.percentile(finite, 5) / median - 1.0), 4),
            round(float(np.percentile(finite, 95) / median - 1.0), 4)],
        "frames_off_median_by_more_than_15pct": int(off.sum()),
        "worst_frame_id": (int(np.nanargmax(np.where(np.isfinite(ratio), ratio, -1.0)))
                           + FIRST_FRAME_ID),
        "frames_off_ids": [index + FIRST_FRAME_ID for index in np.flatnonzero(off).tolist()],
    }


def ray_angles(cameras, point: np.ndarray, support: np.ndarray) -> tuple[float, list]:
    """Max pairwise angle in degrees between the rays of the SUPPORTING views.

    The conditioning figure. Two rays that meet near 180 degrees (or near 0) leave the
    depth along their common axis unconstrained, and every epipolar and reprojection gate
    stays satisfied while it slides -- which is the defect this step exists for. Returned
    as the max over supporting pairs, with every pair listed.
    """

    indices = np.flatnonzero(support).tolist()
    if len(indices) < 2 or not np.isfinite(point).all():
        return float("nan"), []
    directions = {}
    for index in indices:
        vector = point - cameras[index].camera_center_world_m
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            directions[index] = vector / norm
    pairs = []
    for position, first in enumerate(sorted(directions)):
        for second in sorted(directions)[position + 1:]:
            cosine = float(np.dot(directions[first], directions[second]))
            pairs.append({"views": [CAMERAS[first], CAMERAS[second]],
                          "angle_deg": round(float(np.degrees(
                              np.arccos(max(-1.0, min(1.0, cosine))))), 3)})
    if not pairs:
        return float("nan"), []
    return max(pair["angle_deg"] for pair in pairs), pairs


# --------------------------------------------------------------------------------- main
FIRST_FRAME_ID = 60          # set from the observations at run time; this is the fallback


def build_report(check_reproduction: bool = True, mode: str = "d8",
                 classify: tuple[int, int] | None = None) -> dict:
    global FIRST_FRAME_ID
    cameras = cameras_scaled()
    rows = records()
    FIRST_FRAME_ID = int(rows[0][0]["frame_index"])
    frames = len(rows[0])
    frame_ids = [int(row["frame_index"]) for row in rows[0]]
    window = np.asarray([WINDOW_FIRST_ID <= value <= WINDOW_LAST_ID for value in frame_ids])

    landmarks = {subject: landmark_arrays(subject) for subject in (0, 1)}
    assigned = assigned_observations(cameras, rows)
    seen = (np.isfinite(assigned[..., :2]).all(axis=-1)
            & (assigned[..., 2] >= MINIMUM_CONFIDENCE))

    # Cross-check against the committed instrument that already computes this mask.
    import temporal  # noqa: E402  -- I7's own; imported, never copied
    reference_seen = temporal.real_run_seen_mask()
    seen_matches = bool(np.array_equal(seen, reference_seen))

    proxy = proxy_assignment(cameras, rows,
                             {s: landmarks[s]["smoothed"] for s in (0, 1)})
    pipeline = pipeline_assignment(assigned, rows)
    disagreement = (pipeline >= 0) & (proxy >= 0) & (pipeline != proxy)

    report: dict = {
        "title": "D8: the captured limbs under occlusion, from the delivered artifacts alone",
        "instrument": "tools/compare/captured_limb_stability.py",
        "nothing_ships": True,
        "reads": {
            "landmarks": f"{LANDMARK_SOURCE.relative_to(ROOT)}/subject-XX.body-track.npz "
                         "(both arrays: raw and smoothed)",
            "landmarks_are_the_only_thing_that_moves_between_arms": (
                "the observations, the rig and the association always come from the "
                "shipped build; re-associating per arm would put a second variable into a "
                "one-variable comparison"),
            "observations": f"{WORK.relative_to(ROOT)}/*-soma77-observations.jsonl",
            "rig": str(RIG.relative_to(ROOT)),
            "re_detects": False,
            "re_triangulates": False,
        },
        "two_arrays_note": (
            "The card quotes BOTH arrays. The ranges and the frame counts are on the "
            "SMOOTHED array (`triangulated_world_positions_z_up_m`, after the sequence "
            "solve, the fill and the Savitzky-Golay window -- what the exporter consumes); "
            "the two point values, 68 mm at frame 108 and 554 at frame 100, are on the RAW "
            "array (`raw_triangulated_world_positions_z_up_m`, the pre-fill triangulation). "
            "The same two frames read 101.7 and 542.9 on the smoothed array. Every figure "
            "here is emitted on both, with the array in the key."),
        "frames": {
            "count": frames,
            "first_frame_id": frame_ids[0],
            "last_frame_id": frame_ids[-1],
            "window_frame_ids": [WINDOW_FIRST_ID, WINDOW_LAST_ID],
            "window_indices": [int(np.flatnonzero(window)[0]), int(np.flatnonzero(window)[-1])],
            "window_frame_count": int(window.sum()),
            "numbering_note": "the card's frame numbers are the observations' own absolute "
                              "`frame_index`; array index = frame_id - first_frame_id",
        },
        "association": {
            "source": "the pipeline's own, via a recording associator wrapped around the "
                      "real `reconstruct_multiview` (CLAUDE.md: wrap, never re-implement)",
            "seen_mask_equals_temporal_real_run_seen_mask": seen_matches,
            "proxy_rule": "per camera, the person whose 2D hip midpoint is nearest the "
                          "projection of the SMOOTHED 3D hip midpoint -- the coordinator's "
                          "own match, computed as a cross-check only",
            "frames_where_proxy_disagrees_with_the_pipeline": {
                "whole_take": int(disagreement.sum()),
                "in_window": int(disagreement[:, window].sum()),
                "per_subject_in_window": {f"subject_{s:02d}":
                                          int(disagreement[s][window].sum()) for s in (0, 1)},
            },
        },
        "blind_to": [
            "TRUTH -- there is none on this take; a limb frozen at its own median scores "
            "perfectly here, which is why D8's bands live on synthetic truth and on the "
            "photographs and not on this file",
            "a common-mode detector error, which leaves residuals small and angles wide",
            "DIRECTION -- a length invariant cannot score direction (CLAUDE.md); a "
            "same-length rotation of the shoulder line about the A-C axis is invisible",
            "the delivered rig and the mesh bound to it",
        ],
        "subjects": {},
    }

    per_frame_rows: dict = {}
    for subject in (0, 1):
        entry: dict = {"segments": {}, "landmarks": {}, "nan_runs": {}}
        for array in ARRAYS:
            positions = landmarks[subject][array]
            group_block: dict = {}
            for group, pairs in SEGMENTS.items():
                group_block[group] = {}
                for parent, child in pairs:
                    values = 1000.0 * np.linalg.norm(
                        positions[:, cm.JOINT_INDEX[parent]]
                        - positions[:, cm.JOINT_INDEX[child]], axis=1)
                    summary = segment_summary(values)
                    summary["window"] = segment_summary(values[window])
                    summary["series_mm"] = [None if not math.isfinite(v) else round(float(v), 2)
                                            for v in values.tolist()]
                    group_block[group][f"{parent}__{child}"] = summary
            entry["segments"][array] = group_block

        raw = landmarks[subject]["raw"]
        for name in cm.JOINT_NAMES:
            joint = cm.JOINT_INDEX[name]
            missing = ~np.isfinite(raw[:, joint]).all(axis=1)
            runs = run_lengths(missing)
            entry["nan_runs"][name] = {
                "raw_nan_frames": int(missing.sum()),
                "runs": [dict(run, first_frame_id=run["first_index"] + frame_ids[0])
                         for run in runs],
                "longest_run_frames": max((run["length"] for run in runs), default=0),
            }

        # Per landmark per frame: support, per-view confidence, per-view reprojection
        # residual of the RAW point, and the supporting views' ray geometry.
        landmark_block: dict = {}
        rows_out: dict = {}
        for name in ARM_LANDMARKS + ("left_hip", "right_hip", "left_knee", "left_ankle",
                                     "right_knee", "right_ankle", "neck", "root"):
            joint = cm.JOINT_INDEX[name]
            support = seen[subject, :, :, joint]
            confidence = assigned[subject, :, :, joint, 2]
            residual = np.full((frames, len(cameras)), np.nan)
            max_angle = np.full(frames, np.nan)
            pairs_by_frame: dict = {}
            for frame in range(frames):
                point = raw[frame, joint]
                if np.isfinite(point).all():
                    for camera_index, camera in enumerate(cameras):
                        if not support[frame, camera_index]:
                            continue
                        projected, depth = camera.project(point)
                        if depth <= 0.0:
                            continue
                        residual[frame, camera_index] = float(np.linalg.norm(
                            projected - assigned[subject, frame, camera_index, joint, :2]))
                    angle, pairs = ray_angles(cameras, point, support[frame])
                    max_angle[frame] = angle
                    if window[frame]:
                        pairs_by_frame[str(frame_ids[frame])] = pairs
            landmark_block[name] = {
                "camera_support_histogram_0_to_4":
                    np.bincount(support.sum(axis=1), minlength=5).tolist(),
                "window_camera_support_histogram_0_to_4":
                    np.bincount(support[window].sum(axis=1), minlength=5).tolist(),
                "per_camera_supported_frames": {
                    CAMERAS[c]: int(support[:, c].sum()) for c in range(len(cameras))},
                "per_camera_supported_frames_in_window": {
                    CAMERAS[c]: int(support[window, c].sum()) for c in range(len(cameras))},
                "per_camera_dropped_frames_in_window": {
                    CAMERAS[c]: int((~support[window, c]).sum()) for c in range(len(cameras))},
                "per_camera_confidence_on_dropped_window_frames": {
                    CAMERAS[c]: ([round(float(np.nanmin(v)), 3), round(float(np.nanmax(v)), 3)]
                                 if (v := confidence[window][~support[window, c], c]).size
                                 and np.isfinite(v).any() else None)
                    for c in range(len(cameras))},
                "max_pairwise_ray_angle_deg": {
                    "median": (round(float(np.nanmedian(max_angle)), 3)
                               if np.isfinite(max_angle).any() else None),
                    "window_median": (round(float(np.nanmedian(max_angle[window])), 3)
                                      if np.isfinite(max_angle[window]).any() else None),
                    "window_max": (round(float(np.nanmax(max_angle[window])), 3)
                                   if np.isfinite(max_angle[window]).any() else None)},
                "raw_reprojection_residual_px1280": {
                    "median": (round(float(np.nanmedian(residual)), 3)
                               if np.isfinite(residual).any() else None),
                    "window_median": (round(float(np.nanmedian(residual[window])), 3)
                                      if np.isfinite(residual[window]).any() else None),
                    "window_max": (round(float(np.nanmax(residual[window])), 3)
                                   if np.isfinite(residual[window]).any() else None)},
                "two_view_window_frames": int((support[window].sum(axis=1) == 2).sum()),
                "three_or_more_view_window_frames":
                    int((support[window].sum(axis=1) >= 3).sum()),
            }
            rows_out[name] = {
                "frame_ids": frame_ids,
                "support": support.sum(axis=1).tolist(),
                "supporting_views": [[CAMERAS[c] for c in np.flatnonzero(support[f]).tolist()]
                                     for f in range(frames)],
                "confidence": [[None if not math.isfinite(v) else round(float(v), 4)
                                for v in confidence[f].tolist()] for f in range(frames)],
                "raw_reprojection_residual_px1280": [
                    [None if not math.isfinite(v) else round(float(v), 3)
                     for v in residual[f].tolist()] for f in range(frames)],
                "max_pairwise_ray_angle_deg": [
                    None if not math.isfinite(v) else round(float(v), 3)
                    for v in max_angle.tolist()],
                "window_ray_pairs": pairs_by_frame,
            }
        entry["landmarks"] = landmark_block
        per_frame_rows[f"subject_{subject:02d}"] = rows_out
        report["subjects"][f"subject_{subject:02d}"] = entry

    report["per_frame"] = per_frame_rows

    # PERSON-level dropout: the frames on which a camera's detector emits nothing the
    # associator can hand this performer at all. Distinct from a low-confidence landmark,
    # and it is the card's "20-32 of 41" figure.
    report["person_dropout"] = {}
    for subject in (0, 1):
        rows_present = np.isfinite(assigned[subject, :, :, :, :2]).any(axis=(2, 3))
        report["person_dropout"][f"subject_{subject:02d}"] = {
            "definition": "a window frame on which the associator gave this performer no "
                          "row at all in that camera -- the whole person is missing, not "
                          "one landmark",
            "whole_person_absent_window_frames": {
                CAMERAS[c]: int((~rows_present[window, c]).sum()) for c in range(len(cameras))},
            "whole_person_absent_whole_take_frames": {
                CAMERAS[c]: int((~rows_present[:, c]).sum()) for c in range(len(cameras))},
            "window_frames_with_only_A001_and_C001": int(
                (rows_present[window][:, [0, 2]].all(axis=1)
                 & ~rows_present[window][:, [1, 3]].any(axis=1)).sum()),
        }

    # Sub-floor confidence: where the detector DID emit the landmark but below the floor.
    report["sub_floor_confidence"] = {}
    for subject in (0, 1):
        block: dict = {}
        for camera_index, camera in enumerate(CAMERAS):
            values = []
            for name in ARM_LANDMARKS:
                joint = cm.JOINT_INDEX[name]
                confidence = assigned[subject, :, camera_index, joint, 2][window]
                values.extend(confidence[np.isfinite(confidence)
                                         & (confidence < MINIMUM_CONFIDENCE)].tolist())
            block[camera] = ({
                "n": len(values),
                "min": round(float(np.min(values)), 3),
                "p10": round(float(np.percentile(values, 10)), 3),
                "median": round(float(np.median(values)), 3),
                "p90": round(float(np.percentile(values, 90)), 3),
                "max": round(float(np.max(values)), 3)} if values else {"n": 0})
        report["sub_floor_confidence"][f"subject_{subject:02d}"] = block
    report["sub_floor_confidence"]["definition"] = (
        "over the six arm landmarks in the window: the confidences the detector DID emit "
        f"for this performer but below the pipeline's {MINIMUM_CONFIDENCE} floor. A "
        "landmark the detector never emitted has no confidence and is not in here; it is "
        "counted in `person_dropout` instead.")

    report["held_out_denominator"] = {
        "note": "B2's arm needs a THIRD view to hold one out. These are the window frames "
                "that have one, per performer per landmark; the two-view frames have no "
                "held-out arm and are reported as such rather than scored.",
        "subjects": {
            f"subject_{subject:02d}": {
                name: {"three_or_more_view_window_frames":
                       report["subjects"][f"subject_{subject:02d}"]["landmarks"][name]
                       ["three_or_more_view_window_frames"],
                       "two_view_window_frames":
                       report["subjects"][f"subject_{subject:02d}"]["landmarks"][name]
                       ["two_view_window_frames"]}
                for name in ARM_LANDMARKS}
            for subject in (0, 1)},
    }

    if classify is not None:
        report["per_camera_classification"] = per_camera_classification(
            report, classify[0], classify[1])

    if not check_reproduction:
        report["reproduction"] = {
            "skipped": True,
            "why": ("the reproduction clauses describe the DEFECT as it stands in the "
                    "shipped build. Asserting them on a repaired build would be asking "
                    "the repair to leave the defect in place."),
        }
        report["verdict"] = "REPORTED"
    elif mode == "d8b":
        report["reproduction"] = d8b_reproduction(report)
        report["reproduction_card"] = "D8b"
        report["verdict"] = ("PASS" if all(row["matches"] for row in
                                           report["reproduction"]["clauses"]) else "FAIL")
    else:
        report["reproduction"] = reproduction(report)
        report["reproduction_card"] = "D8"
        report["verdict"] = ("PASS" if all(row["matches"] for row in
                                           report["reproduction"]["clauses"]) else "FAIL")
    return report


def reproduction(report: dict) -> dict:
    """Every figure the card states, recomputed here, with the array it came from named."""

    def segment(subject: int, array: str, group: str, pair: str) -> dict:
        return report["subjects"][f"subject_{subject:02d}"]["segments"][array][group][pair]

    clauses: list[dict] = []

    def clause(name: str, expected, measured, matches: bool, array: str, note: str = ""):
        clauses.append({"clause": name, "card_says": expected, "measured": measured,
                        "array": array, "matches": bool(matches), "note": note})

    # legs, both performers, smoothed. TWO separate things live in the card's one
    # sentence and they are kept apart here: the count (exact) and the spread (rounded).
    leg_detail = {}
    leg_off_total = 0
    worst_low, worst_high = 0.0, 0.0
    within_five = 0
    for subject in (0, 1):
        for pair in ("left_hip__left_knee", "left_knee__left_ankle",
                     "right_hip__right_knee", "right_knee__right_ankle"):
            row = segment(subject, "smoothed", "legs", pair)
            spread = row["p5_p95_fraction_of_median"]
            leg_detail[f"subject_{subject:02d}.{pair}"] = {
                "p5_p95_fraction": spread,
                "frames_off": row["frames_off_median_by_more_than_15pct"]}
            leg_off_total += row["frames_off_median_by_more_than_15pct"]
            worst_low = min(worst_low, spread[0])
            worst_high = max(worst_high, spread[1])
            within_five += int(abs(spread[0]) <= 0.05 and abs(spread[1]) <= 0.05)
    clause("the legs are off their own median by more than 15 % on 0 frames, "
           "both performers, all eight segments",
           0, leg_off_total, leg_off_total == 0, "smoothed",
           "this is the card's exact sub-claim and it holds on 8 of 8 segments")

    leg_seed = {
        "card_says": "+/-5 % p5-p95",
        "measured_worst_p5_fraction": round(worst_low, 4),
        "measured_worst_p95_fraction": round(worst_high, 4),
        "segments_inside_+/-5pct": f"{within_five} of 8",
        "per_segment": leg_detail,
        "recorded_difference": (
            "The card's '+/-5 %' is a ROUNDING of this. Measured, six of the eight leg "
            "segments sit inside +/-5 % and two do not: performer 0's left shin reads "
            f"{round(worst_low * 100, 1)} % at p5 and performer 1's right thigh "
            f"+{round(worst_high * 100, 1)} % at p95. The card's counting claim (0 frames "
            "off by more than 15 %) is exact and is the clause above. THE SEED THE CARD "
            "REGISTERS IS THIS MEASURED SPREAD, not the rounded one: it is a REAL-TAKE "
            "figure, registered as own-capture with its source stated, and it selects "
            "nothing -- every D8 ceiling is selected on synthetic truth."),
    }

    p1_forearm = segment(1, "smoothed", "arms", "left_elbow__left_wrist")

    clause("performer 1 forearm L off median on 18 frames", 18,
           p1_forearm["frames_off_median_by_more_than_15pct"],
           p1_forearm["frames_off_median_by_more_than_15pct"] == 18, "smoothed",
           f"max {p1_forearm['max_mm']} mm against a median of {p1_forearm['median_mm']}")

    p1_upper = segment(1, "smoothed", "arms", "left_shoulder__left_elbow")
    clause("performer 1 upper arm L off median on 12 frames", 12,
           p1_upper["frames_off_median_by_more_than_15pct"],
           p1_upper["frames_off_median_by_more_than_15pct"] == 12, "smoothed",
           f"max {p1_upper['max_mm']} mm against a median of {p1_upper['median_mm']}")

    p0_forearms = [segment(0, "smoothed", "arms", pair)["frames_off_median_by_more_than_15pct"]
                   for pair in ("left_elbow__left_wrist", "right_elbow__right_wrist")]
    clause("performer 0 forearms off median on 7 and 8 frames", [7, 8], p0_forearms,
           p0_forearms == [7, 8], "smoothed")

    for subject, low, high, count in ((0, 245, 375, 22), (1, 265, 543, 27)):
        row = segment(subject, "smoothed", "shoulder_line", "left_shoulder__right_shoulder")
        measured = {"p5_mm": row["p5_mm"], "p95_mm": row["p95_mm"], "max_mm": row["max_mm"],
                    "frames_off": row["frames_off_median_by_more_than_15pct"],
                    "median_mm": row["median_mm"]}
        # The card's low figure is the p5 and its high figure is the p95 on performer 0 and
        # the MAX on performer 1; both are stated rather than assumed.
        top = row["p95_mm"] if subject == 0 else row["max_mm"]
        ok = (round(row["p5_mm"]) == low and round(top) == high
              and row["frames_off_median_by_more_than_15pct"] == count)
        clause(f"performer {subject} shoulder line {low}-{high} mm, {count} frames off",
               {"range_mm": [low, high], "frames_off": count}, measured, ok, "smoothed",
               "the low figure is p5; the high is p95 on performer 0 and the max on "
               "performer 1")

    raw_row = segment(1, "raw", "shoulder_line", "left_shoulder__right_shoulder")
    series = raw_row["series_mm"]
    first = report["frames"]["first_frame_id"]
    at_108 = series[108 - first]
    at_100 = series[100 - first]
    clause("performer 1 shoulder line 68 mm at frame 108 and 554 at frame 100",
           {"frame_108_mm": 68, "frame_100_mm": 554},
           {"frame_108_mm": at_108, "frame_100_mm": at_100},
           round(at_108) == 68 and round(at_100) == 554, "raw",
           "these two point values are on the RAW array; the same frames read "
           f"{report['subjects']['subject_01']['segments']['smoothed']['shoulder_line']['left_shoulder__right_shoulder']['series_mm'][108 - first]}"
           " and "
           f"{report['subjects']['subject_01']['segments']['smoothed']['shoulder_line']['left_shoulder__right_shoulder']['series_mm'][100 - first]}"
           " smoothed")

    # B/D dropout. The card's "20-32 of 41" is a PERSON-level count and it reproduces
    # exactly: in the window B001 loses performer 0 entirely on 20 frames and D001 on 32,
    # while A001 and C001 lose neither performer on any frame. Both readings -- the whole
    # person and the per-landmark sub-floor confidence -- are reported, because they are
    # different quantities and only the first is the card's.
    person = report["person_dropout"]
    matched = [subject for subject in ("subject_00", "subject_01")
               if sorted(person[subject]["whole_person_absent_window_frames"][camera]
                         for camera in ("B001", "D001")) == [20, 32]
               and person[subject]["whole_person_absent_window_frames"]["A001"] == 0
               and person[subject]["whole_person_absent_window_frames"]["C001"] == 0]
    clause("B001 and D001 drop the falling performer on 20-32 of the window's 41 frames, "
           "leaving A001 and C001",
           {"range": [20, 32], "of": 41},
           {"reproduces_on": matched, "whole_person_absent_window_frames":
            {subject: person[subject]["whole_person_absent_window_frames"]
             for subject in ("subject_00", "subject_01")}},
           len(matched) == 1, "observations",
           "PERSON level: the frames in the window on which that camera's detector emits "
           "no detection the associator can give this performer at all. A001 and C001 "
           "lose neither performer on any window frame, which is what leaves the two of "
           "them alone")

    sub_floor = report["sub_floor_confidence"]
    band = sub_floor[matched[0]]["D001"] if matched else None
    clause("the dropped views' confidence is 0.05-0.13",
           [0.05, 0.13],
           {"reading": "D001's sub-floor confidences on that performer's six arm "
                       "landmarks in the window, p10-p90",
            "p10_p90": [band["p10"], band["p90"]] if band else None,
            "full_matrix": sub_floor},
           bool(band) and round(band["p10"], 2) == 0.05 and round(band["p90"], 2) == 0.13,
           "observations",
           "the card gives one band for two cameras; measured they differ -- D001 p10-p90 "
           f"{[band['p10'], band['p90']] if band else None} and B001 "
           f"{[sub_floor[matched[0]]['B001']['p10'], sub_floor[matched[0]]['B001']['p90']] if matched else None}. "
           "The card's figure is D001's. Both are here")

    angles = {}
    ok_angle = True
    for frame_id in (100, 108):
        rows = report["per_frame"]["subject_01"]["left_shoulder"]["window_ray_pairs"]
        pairs = rows.get(str(frame_id), [])
        ac = [pair["angle_deg"] for pair in pairs
              if sorted(pair["views"]) == ["A001", "C001"]]
        angles[frame_id] = ac[0] if ac else None
        ok_angle = ok_angle and ac and 170.5 <= ac[0] <= 172.5
    clause("A001 and C001 are 171-172 degrees apart at frames 100 and 108",
           [171, 172], angles, bool(ok_angle), "rig + raw point",
           "the angle between the two supporting rays at the raw point; near-collinear "
           "rays cannot fix depth along their common axis")

    return {
        "source": "docs/LADDER_EXECUTION_PLAN.md section 2, the D8 card, written 2026-09-05",
        "clauses": clauses,
        "legs_length_band_seed": leg_seed,
        "all_match": all(row["matches"] for row in clauses),
    }



# ------------------------------------------------------------------------------ D8b
def per_camera_classification(report: dict, first_id: int, last_id: int) -> dict:
    """Per performer / landmark / frame over a range: who saw it, how sure, how far off.

    Every number here is already computed in `report["per_frame"]`; this block cuts it to
    the frames the D8b card names and lays it out as a table so the card's per-camera claim
    can be read rather than reconstructed. Nothing is recomputed and nothing is re-detected.

    The residual is the RAW triangulated point reprojected into each SUPPORTING camera and
    compared with that camera's own detection, in the pipeline's reference pixel space
    (`px1280`). A slot whose three supporting views all agree to a few pixels and whose
    segment length is impossible is the D8b defect: a WELL-CONDITIONED triangulation of a
    point the detector placed wrongly in every view, which is exactly what no geometric gate
    can see.
    """

    out: dict = {
        "what": ("per performer, per landmark, per frame over the card's own range: the "
                 "cameras that SUPPORTED the slot, each camera's confidence, and each "
                 "supporting camera's reprojection residual of the RAW point"),
        "frame_ids": [first_id, last_id],
        "cameras": list(CAMERAS),
        "residual_definition": ("the raw triangulated point projected into that camera and "
                                "compared with that camera's own detection, px at "
                                "1280 width. It is NOT a distance to truth: if every camera "
                                "is wrong the same way it stays small, which is the whole "
                                "of the D8b defect"),
        "subjects": {},
    }
    for subject_key, rows in report["per_frame"].items():
        block: dict = {}
        for name in D8B_CLASSIFY_LANDMARKS:
            if name not in rows:
                continue
            row = rows[name]
            frames = []
            for index, frame_id in enumerate(row["frame_ids"]):
                if not (first_id <= frame_id <= last_id):
                    continue
                frames.append({
                    "frame_id": frame_id,
                    "supporting_views": row["supporting_views"][index],
                    "support": row["support"][index],
                    "confidence": {CAMERAS[c]: row["confidence"][index][c]
                                   for c in range(len(CAMERAS))},
                    "raw_reprojection_residual_px1280": {
                        CAMERAS[c]: row["raw_reprojection_residual_px1280"][index][c]
                        for c in range(len(CAMERAS))},
                    "max_pairwise_ray_angle_deg": row["max_pairwise_ray_angle_deg"][index],
                })
            counts = {camera: sum(1 for f in frames if camera in f["supporting_views"])
                      for camera in CAMERAS}
            # "the cameras that see him there", counted rather than asserted. A camera that
            # supports EVERY frame and one that supports all but one are different facts and
            # both are here; the majority reading is the card's and the strict one is beside
            # it, so neither has to be guessed later.
            supported = [camera for camera in CAMERAS if counts[camera] > len(frames) // 2]
            every = [camera for camera in CAMERAS if counts[camera] == len(frames)]
            confidences = [value for f in frames for camera, value in f["confidence"].items()
                           if camera in supported and camera in f["supporting_views"]
                           and value is not None]
            residuals = [value for f in frames
                         for camera, value in f["raw_reprojection_residual_px1280"].items()
                         if camera in supported and value is not None]
            block[name] = {
                "frames": frames,
                "frames_in_range": len(frames),
                "per_camera_supported_frames_in_range": counts,
                "cameras_supporting_a_majority_of_the_range": supported,
                "cameras_supporting_every_frame_in_the_range": every,
                "confidence_range_on_those_cameras": (
                    [round(min(confidences), 3), round(max(confidences), 3)]
                    if confidences else None),
                "residual_range_on_those_cameras_px1280": (
                    [round(min(residuals), 3), round(max(residuals), 3)]
                    if residuals else None),
            }
        out["subjects"][subject_key] = block
    return out


def d8b_reproduction(report: dict) -> dict:
    """The D8b card's figures, recomputed here, with the array each came from named.

    The card, `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8b row, written 2026-09-06:
    "on frames 110-122 the captured shoulder line reads 122-274 mm against his own 363 (raw
    122 at frame 110), the captured upper arm 428 -> 280 -> 333 mm against a 277 mm bone";
    "cameras A, C and D all see him there (confidence 0.4-0.95) and all three agree with the
    collapsed point to 1-11 px"; "after D8, performer 1's shoulder line is still off its own
    width by >15 % on 16 frames, the forearm on 4, performer 0's shoulder line on 4; the
    legs on 0".

    LIKE D8's CARD, THIS ONE QUOTES TWO ARRAYS and does not say so. Measured: the shoulder
    line's 122-274 mm range is the RAW array; the upper arm's 428 -> 280 -> 333 sequence is
    the SMOOTHED one; the frame counts are the SMOOTHED array. Every clause below names its
    array.
    """

    clauses: list[dict] = []

    def clause(name, expected, measured, matches, array, note=""):
        clauses.append({"clause": name, "card_says": expected, "measured": measured,
                        "array": array, "matches": bool(matches), "note": note})

    def segment(subject, array, group, pair):
        return report["subjects"][f"subject_{subject:02d}"]["segments"][array][group][pair]

    first = report["frames"]["first_frame_id"]
    lo, hi = D8B_FIRST_ID - first, D8B_LAST_ID - first + 1

    # ---- the collapsed shoulder line, RAW, on the card's own frames
    raw_line = segment(1, "raw", "shoulder_line", "left_shoulder__right_shoulder")
    series = np.asarray([np.nan if v is None else v for v in raw_line["series_mm"]])
    median = raw_line["median_mm"]
    window_values = series[lo:hi]
    off = np.abs(window_values - median) / median > OFF_MEDIAN_FRACTION
    collapsed = window_values[off & np.isfinite(window_values)]
    clause("performer 1's captured shoulder line reads 122-274 mm on frames 110-122 "
           "against his own 363",
           {"range_mm": [122, 274], "own_median_mm": 363},
           {"collapsed_frames_range_mm": [round(float(collapsed.min()), 2),
                                          round(float(collapsed.max()), 2)],
            "collapsed_frame_count": int(off.sum()),
            "all_frames_in_range_mm": [round(float(np.nanmin(window_values)), 2),
                                       round(float(np.nanmax(window_values)), 2)],
            "own_take_median_mm": median},
           collapsed.size and round(float(collapsed.min())) == 122
           and round(float(collapsed.max())) == 274 and round(median) == 364,
           "raw",
           "THE CARD'S RANGE IS THE COLLAPSED FRAMES' RANGE, not the range over all 13 "
           "frames: of the 13, nine are off the performer's own median by more than 15 % "
           "and read 122-274 mm, and the other four read 346-357, near the median. The "
           "card's '363' is this median; measured it is 364.1 on the raw array and 361.4 "
           "on the smoothed one")

    clause("raw 122 at frame 110", 122, series[D8B_FIRST_ID - first],
           round(float(series[D8B_FIRST_ID - first])) == 122, "raw",
           "the card names this one point value as raw, and it is")

    # ---- the upper arm, SMOOTHED
    upper = segment(1, "smoothed", "arms", "left_shoulder__left_elbow")
    smooth_series = [None if v is None else round(float(v), 1) for v in upper["series_mm"]]
    sampled = {fid: smooth_series[fid - first] for fid in (110, 116, 118)}
    clause("the captured upper arm 428 -> 280 -> 333 mm against a 277 mm bone",
           {"110": 428, "116": 280, "118": 333, "own_median_mm": 277},
           {"sampled": sampled, "own_take_median_mm": upper["median_mm"]},
           all(abs(sampled[f] - v) < 1.0 for f, v in ((110, 428.0), (116, 280.2),
                                                      (118, 332.5)))
           and abs(upper["median_mm"] - 277.0) < 1.0,
           "smoothed",
           "the three quoted values are the SMOOTHED array at frames 110, 116 and 118; on "
           "the raw array the same frames read "
           f"{[segment(1, 'raw', 'arms', 'left_shoulder__left_elbow')['series_mm'][f - first] for f in (110, 116, 118)]}")

    # ---- the frames-off counts after D8
    counts = {
        "performer_1_shoulder_line": segment(
            1, "smoothed", "shoulder_line", "left_shoulder__right_shoulder"
        )["frames_off_median_by_more_than_15pct"],
        "performer_1_forearm_L": segment(
            1, "smoothed", "arms", "left_elbow__left_wrist"
        )["frames_off_median_by_more_than_15pct"],
        "performer_0_shoulder_line": segment(
            0, "smoothed", "shoulder_line", "left_shoulder__right_shoulder"
        )["frames_off_median_by_more_than_15pct"],
    }
    clause("performer 1's shoulder line off by >15 % on 16 frames after D8",
           16, counts["performer_1_shoulder_line"],
           counts["performer_1_shoulder_line"] == 16, "smoothed",
           "MEASURED 18 ON THE SHIPPED D9 BUILD. D9 is converter-only and moves no "
           "landmark, and D8's own committed review (docs/reviews/"
           "occlusion-repair-2026-09-05.md section 5) records this figure as 27 -> 18 after "
           "D8, so 18 is what the D8 build left and the card's 16 does not reproduce on "
           "either build. B4's band is on the AFTER value and is not moved by this")
    clause("performer 1's forearm off by >15 % on 4 frames after D8",
           4, counts["performer_1_forearm_L"], counts["performer_1_forearm_L"] == 4,
           "smoothed")
    clause("performer 0's shoulder line off by >15 % on 4 frames after D8",
           4, counts["performer_0_shoulder_line"], counts["performer_0_shoulder_line"] == 4,
           "smoothed")

    leg_off = 0
    leg_detail = {}
    for subject in (0, 1):
        for pair in ("left_hip__left_knee", "left_knee__left_ankle",
                     "right_hip__right_knee", "right_knee__right_ankle"):
            row = segment(subject, "smoothed", "legs", pair)
            leg_off += row["frames_off_median_by_more_than_15pct"]
            leg_detail[f"subject_{subject:02d}.{pair}"] = {
                "frames_off": row["frames_off_median_by_more_than_15pct"],
                "p5_p95_fraction_of_median": row["p5_p95_fraction_of_median"]}
    clause("the legs are off their own median by more than 15 % on 0 frames",
           0, leg_off, leg_off == 0, "smoothed",
           "eight segments, both performers")

    # ---- the per-camera claim, split into the three things the card's one sentence says
    table = report.get("per_camera_classification", {}).get("subjects", {})
    shoulders = {name: table.get("subject_01", {}).get(name, {})
                 for name in ("left_shoulder", "right_shoulder")}
    supported = sorted({camera for row in shoulders.values()
                        for camera in row.get(
                            "cameras_supporting_a_majority_of_the_range", [])})
    per_camera = {name: row.get("per_camera_supported_frames_in_range")
                  for name, row in shoulders.items()}
    clause("cameras A, C and D all see him on frames 110-122",
           ["A001", "C001", "D001"],
           {"cameras_supporting_a_majority_of_the_range": supported,
            "per_camera_supported_frames_of_13": per_camera},
           supported == ["A001", "C001", "D001"],
           "observations",
           "COUNTED, not asserted: over the 13 frames A001 and C001 support both shoulders "
           "on all 13, D001 on 12 of 13 (absent on frame 113, the one two-view frame in the "
           "range and the one D8's conditioning gate acts on), and B001 on 4 of 13 -- it is "
           "sub-floor at 0.13-0.19 on frames 110-118 and only rejoins at 119. That is the "
           "card's 'A, C and D all see him there', with the single-frame exception named")

    # The population the card's residual figure is quoted on: "all three agree with the
    # COLLAPSED point". The collapsed frames are the ones whose shoulder line is off the
    # performer's own median by more than 15 %, and both cuts are reported.
    collapsed_ids = [D8B_FIRST_ID + index for index in np.flatnonzero(off).tolist()]
    ranges: dict = {}
    for cut, wanted in (("collapsed_frames", collapsed_ids),
                        ("all_frames_110_122", list(range(D8B_FIRST_ID, D8B_LAST_ID + 1))),
                        ("frames_all_three_support", [
                            fid for fid in range(D8B_FIRST_ID, D8B_LAST_ID + 1)
                            if fid not in (113,) and fid < 119])):
        values_confidence: list[float] = []
        values_residual: list[float] = []
        for row in shoulders.values():
            for entry in row.get("frames", []):
                if entry["frame_id"] not in wanted:
                    continue
                for camera in ("A001", "C001", "D001"):
                    if camera not in entry["supporting_views"]:
                        continue
                    if entry["confidence"][camera] is not None:
                        values_confidence.append(entry["confidence"][camera])
                    value = entry["raw_reprojection_residual_px1280"][camera]
                    if value is not None:
                        values_residual.append(value)
        ranges[cut] = {
            "confidence": ([round(min(values_confidence), 4),
                            round(max(values_confidence), 4)]
                           if values_confidence else None),
            "residual_px1280": ([round(min(values_residual), 3),
                                 round(max(values_residual), 3)]
                                if values_residual else None),
            "frame_ids": wanted,
        }
    full = ranges["all_frames_110_122"]["residual_px1280"]
    clause("all three agree with the collapsed point to 1-11 px",
           [1, 11], ranges,
           bool(full) and round(full[1]) == 11 and full[0] < 1.5,
           "observations + raw point",
           "the card's 11 is the MAXIMUM over the whole range 110-122 (11.26 px, D001 at "
           "frame 119) and its 1 is a rounding of the minimum, 0.49 px. On the COLLAPSED "
           "frames alone -- the nine whose shoulder line is off the performer's own median "
           "by more than 15 %, which is the population the card's sentence names -- the "
           "three cameras agree to 0.5-6.9 px. Either reading says the same thing: this is "
           "a WELL-CONDITIONED triangulation of a point every camera places wrongly, and no "
           "residual threshold can see one")

    three = ranges["frames_all_three_support"]["confidence"]
    clause("at confidence 0.4-0.95", [0.4, 0.95], ranges,
           bool(three) and round(three[0], 2) == 0.40 and three[1] <= 0.99,
           "observations",
           "the card's LOWER figure reproduces exactly on the frames all three cameras "
           "support (110-112 and 114-118): 0.3975, D001 at frame 110. Its upper figure is a "
           "ROUNDING -- measured 0.9889 (C001). Over the whole range 110-122 the minimum "
           "falls to 0.2650 (D001 at frame 120, a four-view frame), which is above the "
           "pipeline's own 0.25 floor and is why that camera still supports the slot. All "
           "three cuts are in `measured` so none of them has to be guessed again")

    return {
        "source": "docs/LADDER_EXECUTION_PLAN.md section 2, the D8b card, written 2026-09-06",
        "clauses": clauses,
        "frames_off_after_d8": counts,
        "legs_after_d8": leg_detail,
        "all_match": all(row["matches"] for row in clauses),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--landmarks-from", type=Path, default=None,
                        help="a delivery directory whose subject-XX.body-track.npz supply "
                             "the landmark arrays. Defaults to the shipped build. The "
                             "observations, rig and association always come from the "
                             "shipped build whatever this is set to.")
    parser.add_argument("--reproduce", choices=("d8", "d8b", "none"), default="d8",
                        help="which card's figures to check. `d8` is the original "
                             "behaviour and the default; `d8b` checks the D8b card's own "
                             "figures on a POST-D8 build and emits the per-camera "
                             "classification table beside them; `none` checks nothing.")
    parser.add_argument("--classify-frames", type=int, nargs=2, default=None,
                        metavar=("FIRST", "LAST"),
                        help="emit the per-camera classification table over this inclusive "
                             "range of absolute frame ids. Defaults to "
                             f"{D8B_FIRST_ID}-{D8B_LAST_ID} when --reproduce d8b is set.")
    parser.add_argument("--skip-reproduction", action="store_true",
                        help="do not check the card's figures. Set when scoring a build "
                             "OTHER than the shipped one, whose numbers are supposed to "
                             "have moved -- the reproduction clauses describe the DEFECT "
                             "and asserting them on the repaired build would be nonsense.")
    args = parser.parse_args()
    global LANDMARK_SOURCE
    if args.landmarks_from is not None:
        LANDMARK_SOURCE = (args.landmarks_from if args.landmarks_from.is_absolute()
                           else ROOT / args.landmarks_from)
        if not LANDMARK_SOURCE.exists():
            raise SystemExit(f"{LANDMARK_SOURCE} does not exist")
    mode = "none" if args.skip_reproduction else args.reproduce
    classify = args.classify_frames
    if classify is None and mode == "d8b":
        classify = (D8B_FIRST_ID, D8B_LAST_ID)
    report = build_report(check_reproduction=(mode != "none"), mode=mode,
                          classify=None if classify is None else tuple(classify))
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\nassociation: pipeline mask == temporal.real_run_seen_mask: "
          f"{report['association']['seen_mask_equals_temporal_real_run_seen_mask']}")
    print(f"proxy match disagrees with the pipeline on "
          f"{report['association']['frames_where_proxy_disagrees_with_the_pipeline']['whole_take']}"
          " (subject, frame, camera) cells\n")
    for subject in ("subject_00", "subject_01"):
        print(f"{subject}  segment length vs the performer's own take median (SMOOTHED)")
        print(f"  {'segment':<34}{'median':>9}{'p5':>9}{'p95':>9}{'min':>9}{'max':>9}{'off>15%':>9}")
        for group, pairs in report["subjects"][subject]["segments"]["smoothed"].items():
            for name, row in pairs.items():
                print(f"  {group + '/' + name:<34}{row['median_mm']:>9.1f}{row['p5_mm']:>9.1f}"
                      f"{row['p95_mm']:>9.1f}{row['min_mm']:>9.1f}{row['max_mm']:>9.1f}"
                      f"{row['frames_off_median_by_more_than_15pct']:>9d}")
        print()
    if report["reproduction"].get("skipped"):
        print("reproduction: skipped (scoring a build other than the shipped one)")
    else:
        print(f"reproduction of the {report.get('reproduction_card', 'D8')} card's figures")
        for row in report["reproduction"]["clauses"]:
            print(f"  [{'ok ' if row['matches'] else 'NO '}] ({row['array']}) {row['clause']}")
    print(f"\nverdict: {report['verdict']}")
    print(f"wrote {out}")
    return 0 if report["verdict"] in ("PASS", "REPORTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
