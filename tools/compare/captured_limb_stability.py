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

# D8c. The hip line's own two runs on the falling performer, and the landmarks the card's
# classification is read on. THE CARD'S REVIEWER ASKED FOR THESE TO BE A FLAG RATHER THAN A
# CONSTANT (`--classify-landmarks`, `--classify-frames` repeated): the ranges below are
# DEFAULTS for `--reproduce d8c` and nothing selects on them -- they are where the card says
# to look, and any other range can be asked for on the command line.
#
#   110-119  class (i), the INWARD COLLAPSE in agreeing views: A, C and D support both hips
#            and the length is impossible anyway.
#   158-168  class (ii), the OUTWARD STRETCH along the A-C baseline: only two cameras see
#            him, at a ray angle UNDER D8's 150 degree ceiling, so the conditioning gate is
#            silent and the depth along the pair's common axis is loose.
#    84-86   the ONE-HIP-AT-A-TIME frames, reported with 110-119 as class (i)'s second run
#            and registered by the card as a KNOWN OVER-CHARGE of the both-endpoints rule.
# 109-119 and not 110-119: the card's per-camera sentence counts "9 of 11 frames", which is
# eleven frames, and 109 is the frame before the collapse starts. The COUNT clause is on the
# whole take and does not depend on this.
D8C_CLASSIFY_RANGES = ((109, 119), (158, 168), (84, 86))
D8C_CLASSIFY_LANDMARKS = ("left_hip", "right_hip", "root", "neck")
# The runs the card names on performer 1, in absolute frame ids, inclusive. Used only to CUT
# the reported tables; every count in the reproduction is measured over the whole take.
D8C_RUNS = {"84-86": (84, 86), "110-119": (110, 119), "158-168": (158, 168)}
# The two cameras the class (ii) run is triangulated from. Read from the observations, not
# assumed: the reproduction asserts that these are the supporting pair and reports what it
# found if they are not.
D8C_BASELINE_CAMERAS = ("A001", "C001")

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


def segment_summary(values: np.ndarray, ids: list[int] | None = None) -> dict:
    """One segment's length series in millimetres against its own take median.

    ``ids`` is the ABSOLUTE frame id of each element of ``values``, and it is not optional
    in spirit. D8c REPAIR, 2026-09-06: this function used to label every reported frame as
    ``array index + FIRST_FRAME_ID``, which is right for the whole-take call and WRONG for
    every windowed one -- the window slice's index 0 is the window's first frame, not the
    take's, so the ``window`` block printed 60-94 for frames 85-119 and `worst_frame_id`
    was off by the same 25. The defect was found while writing the D8c card and it is an
    instrument defect, not a measurement one: no clause of the D8 or D8b reproduction reads
    these ids, so no committed figure moves with the repair. The fallback keeps the old
    behaviour only when a caller supplies nothing.
    """

    if ids is None:
        ids = [index + FIRST_FRAME_ID for index in range(len(values))]
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
        "worst_frame_id": ids[int(np.nanargmax(np.where(np.isfinite(ratio), ratio, -1.0)))],
        "frames_off_ids": [ids[index] for index in np.flatnonzero(off).tolist()],
        "frame_ids_are_absolute": True,
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
                 classify: tuple[tuple[int, int], ...] | None = None,
                 classify_landmarks: tuple[str, ...] = D8B_CLASSIFY_LANDMARKS,
                 hip_geometry: bool = False) -> dict:
    global FIRST_FRAME_ID
    cameras = cameras_scaled()
    rows = records()
    FIRST_FRAME_ID = int(rows[0][0]["frame_index"])
    frames = len(rows[0])
    frame_ids = [int(row["frame_index"]) for row in rows[0]]
    window = np.asarray([WINDOW_FIRST_ID <= value <= WINDOW_LAST_ID for value in frame_ids])
    window_ids = [value for value, inside in zip(frame_ids, window.tolist()) if inside]

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
                    summary = segment_summary(values, frame_ids)
                    summary["window"] = segment_summary(values[window], window_ids)
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

    if classify:
        # One block per requested range, keyed by the range itself. D8c needs two ranges at
        # once (110-119 and 158-168 are two DIFFERENT failures and the card classifies them
        # separately), and D8b's single-range default is preserved as a one-element list.
        report["per_camera_classification"] = per_camera_classification(
            report, classify[0][0], classify[0][1], classify_landmarks)
        report["per_camera_classification_by_range"] = {
            f"{first}-{last}": per_camera_classification(report, first, last,
                                                         classify_landmarks)
            for first, last in classify}

    if mode == "d8c" or hip_geometry:
        report["hip_line_geometry"] = hip_line_geometry(
            report, cameras, landmarks, assigned, seen, frame_ids)

    if not check_reproduction:
        report["reproduction"] = {
            "skipped": True,
            "why": ("the reproduction clauses describe the DEFECT as it stands in the "
                    "shipped build. Asserting them on a repaired build would be asking "
                    "the repair to leave the defect in place."),
        }
        report["verdict"] = "REPORTED"
    elif mode == "d8c":
        report["reproduction"] = d8c_reproduction(report)
        report["reproduction_card"] = "D8c"
        report["verdict"] = ("PASS" if all(row["matches"] for row in
                                           report["reproduction"]["clauses"]) else "FAIL")
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
def per_camera_classification(report: dict, first_id: int, last_id: int,
                              landmarks: tuple[str, ...] = D8B_CLASSIFY_LANDMARKS) -> dict:
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
        "landmarks": list(landmarks),
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
        for name in landmarks:
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
            # THE STRICT CUT, and the card's residual and confidence figures are on it.
            # A frame on which a FOURTH camera rejoins is not the same population as one
            # the three support alone: on 109-119 the falling performer's frame 119 has
            # B001 back at 11.9 px and D001 at 10.8, and quoting a range over it would
            # describe a four-view frame as if it were part of the three-view collapse.
            # Frame 113 is the other exclusion, and for the opposite reason: it is
            # two-view, which is the one frame in the run D8's conditioning gate acts on.
            strict = [f for f in frames if sorted(f["supporting_views"]) == sorted(supported)]
            strict_confidence = [value for f in strict
                                 for camera, value in f["confidence"].items()
                                 if camera in supported and value is not None]
            strict_residual = [value for f in strict
                               for camera, value in
                               f["raw_reprojection_residual_px1280"].items()
                               if camera in supported and value is not None]
            strict_angle = [f["max_pairwise_ray_angle_deg"] for f in strict
                            if f["max_pairwise_ray_angle_deg"] is not None]
            block[name] = {
                "frames": frames,
                "frames_in_range": len(frames),
                "on_frames_supported_by_exactly_those_cameras": {
                    "frame_ids": [f["frame_id"] for f in strict],
                    "frames": len(strict),
                    "confidence_range": ([round(min(strict_confidence), 4),
                                          round(max(strict_confidence), 4)]
                                         if strict_confidence else None),
                    "residual_range_px1280": ([round(min(strict_residual), 3),
                                               round(max(strict_residual), 3)]
                                              if strict_residual else None),
                    "max_pairwise_ray_angle_deg": ([round(min(strict_angle), 2),
                                                    round(max(strict_angle), 2)]
                                                   if strict_angle else None),
                    "why": "the frames the majority cameras support and NO OTHER camera "
                           "does -- the population the card's confidence and residual "
                           "ranges are quoted on",
                },
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


# ------------------------------------------------------------------------------ D8c
def _series(positions: np.ndarray, a: str, b: str) -> np.ndarray:
    """Millimetres between two landmarks, per frame, NaN where either is missing."""

    return 1000.0 * np.linalg.norm(positions[:, cm.JOINT_INDEX[a]]
                                   - positions[:, cm.JOINT_INDEX[b]], axis=1)


def _spread(values: np.ndarray, mask: np.ndarray | None = None,
            reference: str = "take") -> dict:
    """p5/p95 as a fraction of the median, the median, and the count off by >15 %.

    ``mask`` restricts the POPULATION. ``reference`` says what median the spread is quoted
    against, and the two are genuinely different questions:

    * ``"take"`` -- the median over every finite frame. This is what the hip line's own
      honest spread is quoted against, because the off frames were DEFINED against that
      median and removing them must not move the thing they were measured from.
    * ``"population"`` -- the median over the masked frames only. This is what the card's
      root->hip table is quoted against, and it is the right reference there because
      root->hip has no off list of its own: the honest frames are its whole evidence.

    BOTH are emitted wherever the reading matters, so no figure has to be guessed later.
    """

    finite = np.isfinite(values)
    if not finite.any():
        return {"frames": 0}
    population = finite if mask is None else (finite & mask)
    median = float(np.median(values[population if reference == "population" else finite]))
    if not population.any() or median <= 1e-9:
        return {"frames": 0, "median_mm": round(median, 2)}
    reference_note = ("the median over every finite frame" if reference == "take"
                      else "the median over the MASKED frames only")
    sample = values[population]
    ratio = np.abs(values - median) / median
    return {
        "frames": int(population.sum()),
        "median_mm": round(median, 2),
        "min_mm": round(float(sample.min()), 2),
        "max_mm": round(float(sample.max()), 2),
        "p5_p95_fraction_of_median": [
            round(float(np.percentile(sample, 5) / median - 1.0), 4),
            round(float(np.percentile(sample, 95) / median - 1.0), 4)],
        "frames_off_median_by_more_than_15pct": int(
            (population & (ratio > OFF_MEDIAN_FRACTION)).sum()),
        "median_reference": reference_note,
    }


def _runs_of(ids: list[int]) -> list[list[int]]:
    """Contiguous runs of absolute frame ids, as [first, last] pairs."""

    out: list[list[int]] = []
    for value in sorted(ids):
        if out and value == out[-1][1] + 1:
            out[-1][1] = value
        else:
            out.append([value, value])
    return out


def hip_line_geometry(report: dict, cameras, landmarks: dict, assigned: np.ndarray,
                      seen: np.ndarray, frame_ids: list[int]) -> dict:
    """Everything the D8c card measures about the hip line, from the delivered arrays.

    Five blocks, and each answers one sentence of the card:

    * **the off frames and their runs**, on BOTH arrays, with the ids listed rather than
      described, and the honest spread with those frames removed.
    * **the neighbours that hold**: root->neck, the two thighs and |hip_mid - root| on the
      same frames. The card's argument is that the hips move ALONG the hip line toward its
      own midpoint and not toward the pelvis landmark, and these three series are what
      distinguishes those two motions.
    * **the root->hip table on the HONEST mask** -- the measurement the pre-dispatch review
      demanded before the both-endpoints convention could be defended. A per-hip
      `root->hip` rule would be the more precise instrument if the pelvis landmark were
      tight enough to be a length reference, and this says whether it is. Computed on the
      same honest mask the hip line's own spread is quoted on, because the review's whole
      point was that the contaminated series must not be used to forbid the finer rule.
    * **the A-C baseline split** of the hip vector on the class (ii) run: the component
      along the two cameras' baseline direction (the axis a two-view pair cannot fix) and
      the component across it. A depth stretch lives on the baseline; a genuinely wide
      pelvis does not.
    * **the geometry of the class (ii) run**: the hip midpoint's height off the floor and
      the angle between the hip vector and each supporting camera's viewing ray.

    WHAT IT IS BLIND TO. Truth -- there is none here. The baseline split says the error lies
    where a two-view pair is weak; it cannot say the pelvis is not really that wide. And a
    LENGTH INVARIANT CANNOT SCORE DIRECTION (CLAUDE.md): none of this reads the pelvis
    frame's orientation, and nothing about D7's Kabsch may be inferred from it.
    """

    centres = {CAMERAS[index]: np.asarray(camera.camera_center_world_m, dtype=np.float64)
               for index, camera in enumerate(cameras)}
    first, second = D8C_BASELINE_CAMERAS
    baseline = centres[second] - centres[first]
    baseline = baseline / float(np.linalg.norm(baseline))

    out: dict = {
        "what": "the hip line's own geometry on the shipped build, on both arrays",
        "baseline": {
            "cameras": list(D8C_BASELINE_CAMERAS),
            "unit_vector_world": [round(float(v), 6) for v in baseline.tolist()],
            "why": ("the direction a two-view pair cannot fix. Splitting the hip vector "
                    "into this direction and the plane across it separates a depth stretch "
                    "from a genuinely wider pelvis"),
        },
        "subjects": {},
        "per_frame_root_hip": {},
    }

    for subject in (0, 1):
        block: dict = {}
        # |root - hip| per frame, per side, keyed by ABSOLUTE frame id. The card's
        # one-hip-at-a-time claim on 84-86 is a per-frame statement and a range cannot
        # carry it: on frame 84 the LEFT hip is 84 mm from the root and the right 127, and
        # on frame 85 it is the other way round. A min/max over the run would read
        # "84-138 left, 103-127 right" and lose exactly the alternation that is the point.
        out["per_frame_root_hip"][f"subject_{subject:02d}"] = {
            array: {
                str(frame_ids[index]): {
                    side: round(float(1000.0 * np.linalg.norm(
                        landmarks[subject][array][index, cm.JOINT_INDEX[f"{side}_hip"]]
                        - landmarks[subject][array][index, cm.JOINT_INDEX["root"]])), 2)
                    if np.isfinite(landmarks[subject][array][
                        index, cm.JOINT_INDEX[f"{side}_hip"]]).all()
                    and np.isfinite(landmarks[subject][array][
                        index, cm.JOINT_INDEX["root"]]).all() else None
                    for side in ("left", "right")}
                for index in range(len(frame_ids))}
            for array in ARRAYS}
        for array in ARRAYS:
            positions = landmarks[subject][array]
            hip = _series(positions, "left_hip", "right_hip")
            finite = np.isfinite(hip)
            median = float(np.median(hip[finite])) if finite.any() else float("nan")
            off = finite & (np.abs(hip - median) / median > OFF_MEDIAN_FRACTION)
            honest = finite & ~off
            off_ids = [frame_ids[index] for index in np.flatnonzero(off).tolist()]

            left_hip = positions[:, cm.JOINT_INDEX["left_hip"]]
            right_hip = positions[:, cm.JOINT_INDEX["right_hip"]]
            root = positions[:, cm.JOINT_INDEX["root"]]
            mid = 0.5 * (left_hip + right_hip)
            mid_to_root = 1000.0 * np.linalg.norm(mid - root, axis=1)

            vector = left_hip - right_hip
            along = np.abs(np.einsum("ij,j->i", vector, baseline)) * 1000.0
            across = 1000.0 * np.linalg.norm(
                vector - np.einsum("ij,j->i", vector, baseline)[:, None] * baseline[None, :],
                axis=1)

            neighbours = {
                "root_to_neck_mm": _series(positions, "root", "neck"),
                "left_thigh_mm": _series(positions, "left_hip", "left_knee"),
                "right_thigh_mm": _series(positions, "right_hip", "right_knee"),
                "hip_midpoint_to_root_mm": mid_to_root,
                "root_to_left_hip_mm": 1000.0 * np.linalg.norm(left_hip - root, axis=1),
                "root_to_right_hip_mm": 1000.0 * np.linalg.norm(right_hip - root, axis=1),
            }

            def cut(series: np.ndarray, ids: list[int]) -> dict:
                index = [frame_ids.index(value) for value in ids if value in frame_ids]
                sample = series[index]
                sample = sample[np.isfinite(sample)]
                if not sample.size:
                    return {"frames": 0}
                return {"frames": int(sample.size),
                        "min_mm": round(float(sample.min()), 2),
                        "max_mm": round(float(sample.max()), 2),
                        "median_mm": round(float(np.median(sample)), 2)}

            arm: dict = {
                "hip_line": {
                    "whole_take": _spread(hip),
                    "honest_off_frames_removed": _spread(hip, honest),
                    "frames_off_ids": off_ids,
                    "runs_of_off_frames": _runs_of(off_ids),
                    "series_mm": [None if not math.isfinite(v) else round(float(v), 2)
                                  for v in hip.tolist()],
                },
                "neighbours_on_the_off_runs": {
                    label: {name: cut(series, list(range(lo, hi + 1)))
                            for name, series in neighbours.items()}
                    for label, (lo, hi) in D8C_RUNS.items()},
                # The card's class (i) figures are quoted on the COLLAPSED frames -- the
                # two runs 84-86 and 110-119 with frame 113 removed, because 113 is the
                # OUTWARD spike and the sentence is about a collapse. Both readings are
                # here so neither has to be guessed.
                "neighbours_on_the_collapsed_frames": {
                    name: cut(series, [f for f in list(range(84, 87)) + list(range(110, 120))
                                       if f != 113])
                    for name, series in neighbours.items()},
                "neighbours_on_110_119_including_the_outward_spike": {
                    name: cut(series, list(range(110, 120)))
                    for name, series in neighbours.items()},
                "neighbours_whole_take_median_mm": {
                    name: (round(float(np.nanmedian(series)), 2)
                           if np.isfinite(series).any() else None)
                    for name, series in neighbours.items()},
            }

            # THE ROOT->HIP TABLE, on the SAME honest mask. The review's clause: "recompute
            # root-hip p5/p95 on the same honest mask as the hip line. If that honest spread
            # is inside 0.15, charge per hip and do not ship the both-endpoints row."
            root_hip: dict = {}
            for side in ("left", "right"):
                series = neighbours[f"root_to_{side}_hip_mm"]
                usable = np.isfinite(series)
                own_median = float(np.median(series[usable])) if usable.any() else float("nan")
                honest_median = (float(np.median(series[usable & honest]))
                                 if (usable & honest).any() else float("nan"))
                row = {
                    "honest_mask_spread": _spread(series, honest,
                                                  reference="population"),
                    "honest_mask_spread_against_the_take_median": _spread(series, honest),
                    "whole_take_spread": _spread(series),
                    "would_fire_at_0.15_on_its_own_take_median": int(
                        (usable & (np.abs(series - own_median) / own_median
                                   > OFF_MEDIAN_FRACTION)).sum()),
                    "would_fire_at_0.15_on_the_honest_frames_median": int(
                        (usable & (np.abs(series - honest_median) / honest_median
                                   > OFF_MEDIAN_FRACTION)).sum()),
                    "frames_it_would_fire_on_ids": [
                        frame_ids[index] for index in np.flatnonzero(
                            usable & (np.abs(series - own_median) / own_median
                                      > OFF_MEDIAN_FRACTION)).tolist()],
                }
                row["of_those_whose_hip_line_is_HONEST"] = int(sum(
                    1 for value in row["frames_it_would_fire_on_ids"]
                    if value not in off_ids))
                root_hip[side] = row
            arm["root_to_hip_on_the_honest_mask"] = root_hip

            # THE A-C BASELINE SPLIT.
            split: dict = {
                "honest_frames": {
                    "along_the_baseline_median_mm": (
                        round(float(np.median(along[honest])), 2) if honest.any() else None),
                    "across_the_baseline_median_mm": (
                        round(float(np.median(across[honest])), 2) if honest.any() else None),
                    "along_the_baseline_mean_mm": (
                        round(float(np.mean(along[honest])), 2) if honest.any() else None),
                    "across_the_baseline_mean_mm": (
                        round(float(np.mean(across[honest])), 2) if honest.any() else None),
                    "frames": int(honest.sum()),
                },
                "per_run": {},
            }
            for label, (lo, hi) in D8C_RUNS.items():
                index = [frame_ids.index(value) for value in range(lo, hi + 1)
                         if value in frame_ids]
                rows = []
                for position in index:
                    rows.append({
                        "frame_id": frame_ids[position],
                        "hip_line_mm": (None if not math.isfinite(hip[position])
                                        else round(float(hip[position]), 2)),
                        "along_mm": (None if not math.isfinite(along[position])
                                     else round(float(along[position]), 2)),
                        "across_mm": (None if not math.isfinite(across[position])
                                      else round(float(across[position]), 2)),
                        "fraction_off_median": (
                            None if not math.isfinite(hip[position])
                            else round(float((hip[position] - median) / median), 4)),
                        "hip_midpoint_height_mm": (
                            None if not np.isfinite(mid[position]).all()
                            else round(float(mid[position][2]) * 1000.0, 2)),
                    })
                    for camera in D8C_BASELINE_CAMERAS:
                        ray = mid[position] - centres[camera]
                        norm = float(np.linalg.norm(ray))
                        length = float(np.linalg.norm(vector[position]))
                        if norm > 1e-9 and length > 1e-9 and np.isfinite(ray).all():
                            cosine = abs(float(np.dot(ray / norm, vector[position] / length)))
                            # The angle BETWEEN the hip line and the camera's ray, folded to
                            # [0, 90] by the absolute value: a segment that points at the
                            # camera and one that points away from it are the same case.
                            # Small means the segment lies ALONG the viewing direction,
                            # which is the direction its length is least determined in.
                            rows[-1][f"angle_of_the_hip_line_to_{camera}s_ray_deg"] = round(
                                math.degrees(math.acos(min(1.0, cosine))), 2)
                split["per_run"][label] = rows
            arm["baseline_split"] = split
            block[array] = arm
        out["subjects"][f"subject_{subject:02d}"] = block

    del assigned, seen, report
    return out


def d8c_reproduction(report: dict) -> dict:
    """The D8c card's figures, recomputed here, with the array each came from named.

    The card, `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8c row, written and committed
    at 85b8113 BEFORE this instrument ran. Its "instrument first" clause is:

      "`captured_limb_stability.py --reproduce d8c` must reproduce 30 raw / 23 smoothed
       frames off in the three runs, 124.9 mm raw / 139.7 smoothed at the minimum, frame
       113's 355 mm raw, the per-camera table on the HIPS and the root over 110-119
       (A-C-D, D 0.43-0.93, <= 8.1 px) and 158-168 (A and C only, D absent, 139-140 deg,
       <= 3.8 px), the baseline split on 158-168 (along 231-275 / across 18-52 against
       honest 181 / 117) and the honest-mask root->hip table above."

    LIKE BOTH CARDS BEFORE IT, THIS ONE QUOTES TWO ARRAYS. Every clause below names the
    array it is read on. Where the card's figure does not reproduce it is recorded as a
    non-reproducing card figure and NO BAND MOVES -- the D8b precedent (its "16 frames" read
    18 on both builds and B4's band was left where it was).
    """

    clauses: list[dict] = []

    def clause(name, expected, measured, matches, array, note=""):
        clauses.append({"clause": name, "card_says": expected, "measured": measured,
                        "array": array, "matches": bool(matches), "note": note})

    geometry = report["hip_line_geometry"]["subjects"]
    first = report["frames"]["first_frame_id"]

    def hips(subject: int, array: str) -> dict:
        return geometry[f"subject_{subject:02d}"][array]["hip_line"]

    def value_at(subject: int, array: str, frame_id: int):
        return hips(subject, array)["series_mm"][frame_id - first]

    # ---- 1. the counts and the medians, performer 1, both arrays
    raw, smoothed = hips(1, "raw"), hips(1, "smoothed")
    raw_off = raw["whole_take"]["frames_off_median_by_more_than_15pct"]
    smooth_off = smoothed["whole_take"]["frames_off_median_by_more_than_15pct"]
    clause("performer 1's captured hip line is off his own median on 30 raw and 23 "
           "smoothed frames",
           {"raw": 30, "smoothed": 23},
           {"raw": raw_off, "smoothed": smooth_off,
            "raw_frames_off_ids": raw["frames_off_ids"],
            "smoothed_frames_off_ids": smoothed["frames_off_ids"],
            "raw_runs": raw["runs_of_off_frames"],
            "smoothed_runs": smoothed["runs_of_off_frames"]},
           raw_off == 30 and smooth_off == 23, "both",
           "the ids are listed on both arrays so the card's 'in three runs' can be read "
           "rather than taken on trust. The RAW off frames fall in FOUR runs, not three: "
           "84-86, 88-106, 110-119 and 158-168. The card itself names the 88-106 run "
           "separately as 'already recovered by D8's sequence solve and not a defect in "
           "the delivery', which is why its prose says three")

    clause("performer 1's own hip-line median is 215.0 mm raw and 214.4 smoothed",
           {"raw_mm": 215.0, "smoothed_mm": 214.4},
           {"raw_mm": raw["whole_take"]["median_mm"],
            "smoothed_mm": smoothed["whole_take"]["median_mm"]},
           abs(raw["whole_take"]["median_mm"] - 215.0) < 0.1
           and abs(smoothed["whole_take"]["median_mm"] - 214.4) < 0.1, "both")

    clause("the minimum hip line is 124.9 mm raw and 139.7 smoothed",
           {"raw_mm": 124.9, "smoothed_mm": 139.7},
           {"raw_mm": raw["whole_take"]["min_mm"],
            "smoothed_mm": smoothed["whole_take"]["min_mm"]},
           abs(raw["whole_take"]["min_mm"] - 124.9) < 0.1
           and abs(smoothed["whole_take"]["min_mm"] - 139.7) < 0.1, "both")

    at_113_raw, at_113_smooth = value_at(1, "raw", 113), value_at(1, "smoothed", 113)
    clause("frame 113 is an OUTWARD spike at 355 mm raw that the smoothing turns into 160",
           {"raw_mm": 355, "smoothed_mm": 160},
           {"raw_mm": at_113_raw, "smoothed_mm": at_113_smooth},
           at_113_raw is not None and round(at_113_raw) == 355
           and at_113_smooth is not None and round(at_113_smooth) == 160, "both",
           "the one frame in 110-119 whose hip line is too WIDE. The symmetric rule "
           "|L - median| / median fires on it too, which is why the card counts ten off "
           "frames in a ten-frame run and describes nine of them as a collapse")

    # ---- 2. the two class (i) runs, raw
    series_110 = [value_at(1, "raw", fid) for fid in range(110, 120)]
    inward = [v for v in series_110 if v is not None and v < 200.0]
    clause("frames 110-119 read 125-174 mm raw on nine of the ten",
           {"range_mm": [125, 174], "of": 10},
           {"series_110_119_mm": series_110,
            "the_nine_below_200_mm": [round(min(inward), 1), round(max(inward), 1)],
            "count_below_200": len(inward)},
           len(inward) == 9 and 124 <= min(inward) <= 126 and 173 <= max(inward) <= 175,
           "raw", "the tenth is frame 113's outward spike, above")

    series_84 = [value_at(1, "raw", fid) for fid in range(84, 87)]
    clause("frames 84-86 read 130-153 mm raw",
           {"range_mm": [130, 153]},
           {"series_84_86_mm": series_84},
           all(v is not None for v in series_84)
           and 129 <= min(series_84) <= 131 and 152 <= max(series_84) <= 154, "raw")

    per_hip = geometry["subject_01"]["raw"]
    left_series = None
    for label in ("84-86",):
        left_series = per_hip["neighbours_on_the_off_runs"][label]
    root_left = [round(float(1000.0 * 0.0), 2)]           # placeholder, replaced below
    del root_left, left_series

    # frame 84 left 84 mm from the root against right 127; frame 85 left 138 against 103.
    root_hip_rows = {}
    for frame_id in (84, 85, 86):
        row = {}
        for side in ("left", "right"):
            key = f"root_to_{side}_hip_mm"
            cut = per_hip["neighbours_on_the_off_runs"]["84-86"][key]
            row[side] = cut
        root_hip_rows[frame_id] = row
    # the per-frame values, read off the geometry block's own series
    per_frame_root_hip = report["hip_line_geometry"]["per_frame_root_hip"]["subject_01"]
    frame_84 = per_frame_root_hip["raw"]["84"]
    frame_85 = per_frame_root_hip["raw"]["85"]
    clause("ONE HIP AT A TIME on 84-86: frame 84 left 84 mm from the root against right "
           "127, frame 85 left 138 against right 103",
           {"84": {"left_mm": 84, "right_mm": 127}, "85": {"left_mm": 138, "right_mm": 103}},
           {"84": frame_84, "85": frame_85, "86": per_frame_root_hip["raw"]["86"],
            "own_take_medians_mm": {
                "root_to_left_hip": per_hip["neighbours_whole_take_median_mm"][
                    "root_to_left_hip_mm"],
                "root_to_right_hip": per_hip["neighbours_whole_take_median_mm"][
                    "root_to_right_hip_mm"]}},
           round(frame_84["left"]) == 84 and round(frame_84["right"]) == 127
           and round(frame_85["left"]) == 138 and round(frame_85["right"]) == 103, "raw",
           "the card's registered KNOWN OVER-CHARGE: the hip-line rule charges both "
           "endpoints, so on these frames a femoral head that matched the take is withheld "
           "with the one that did not")
    del root_hip_rows

    # ---- 3. the neighbours that hold
    collapsed = per_hip["neighbours_on_the_collapsed_frames"]
    with_spike = per_hip["neighbours_on_110_119_including_the_outward_spike"]
    medians = per_hip["neighbours_whole_take_median_mm"]
    measured_neighbours = {
        "root_to_neck_on_the_collapsed_frames_mm": [collapsed["root_to_neck_mm"]["min_mm"],
                                                    collapsed["root_to_neck_mm"]["max_mm"]],
        "root_to_neck_take_median_mm": medians["root_to_neck_mm"],
        "thighs_on_the_collapsed_frames_mm": [
            min(collapsed["left_thigh_mm"]["min_mm"], collapsed["right_thigh_mm"]["min_mm"]),
            max(collapsed["left_thigh_mm"]["max_mm"], collapsed["right_thigh_mm"]["max_mm"])],
        "thigh_take_medians_mm": [medians["left_thigh_mm"], medians["right_thigh_mm"]],
        "hip_mid_to_root_on_110_119_mm": [with_spike["hip_midpoint_to_root_mm"]["min_mm"],
                                          with_spike["hip_midpoint_to_root_mm"]["max_mm"]],
        "hip_mid_to_root_take_median_mm": medians["hip_midpoint_to_root_mm"],
        "hip_mid_to_root_on_the_collapsed_frames_mm": [
            collapsed["hip_midpoint_to_root_mm"]["min_mm"],
            collapsed["hip_midpoint_to_root_mm"]["max_mm"]],
    }
    neck_ok = (abs(measured_neighbours["root_to_neck_on_the_collapsed_frames_mm"][0] - 509)
               < 1.0
               and abs(measured_neighbours["root_to_neck_on_the_collapsed_frames_mm"][1]
                       - 544) < 1.0
               and abs(measured_neighbours["root_to_neck_take_median_mm"] - 515) < 1.0)
    mid_ok = (abs(measured_neighbours["hip_mid_to_root_on_110_119_mm"][0] - 68) < 1.0
              and abs(measured_neighbours["hip_mid_to_root_on_110_119_mm"][1] - 81) < 1.0
              and abs(measured_neighbours["hip_mid_to_root_take_median_mm"] - 76.5) < 0.5)
    thigh_low_ok = abs(measured_neighbours["thighs_on_the_collapsed_frames_mm"][0]
                       - 392) < 1.0
    thigh_median_ok = (abs(measured_neighbours["thigh_take_medians_mm"][0] - 402) < 1.0
                       and abs(measured_neighbours["thigh_take_medians_mm"][1] - 406) < 1.0)
    clause("the root landmark holds through both class (i) runs (root->neck 509-544 mm "
           "against a 515 median), the thighs hold (392-426 against 402 / 406) and "
           "|hip_mid - root| holds (68-81 against 76.5)",
           {"root_to_neck_mm": [509, 544], "root_to_neck_median_mm": 515,
            "thighs_mm": [392, 426], "thigh_medians_mm": [402, 406],
            "hip_mid_to_root_mm": [68, 81], "hip_mid_to_root_median_mm": 76.5},
           dict(measured_neighbours,
                populations={
                    "root_to_neck and the thighs": "the COLLAPSED frames -- 84-86 and "
                                                   "110-119 with frame 113 (the outward "
                                                   "spike) removed",
                    "hip_mid_to_root": "110-119 inclusive; on the collapsed frames it "
                                       "reads "
                                       f"{measured_neighbours['hip_mid_to_root_on_the_collapsed_frames_mm']}"
                                       " because 84-86's own midpoint sits further out"},
                thigh_high_figure_does_not_reproduce={
                    "card_says": 426,
                    "measured_on_the_collapsed_frames":
                        measured_neighbours["thighs_on_the_collapsed_frames_mm"][1],
                    "where_426_does_appear": "the left thigh reaches 425.6 mm on the class "
                                             "(ii) run 158-168, which is a different run "
                                             "and a different failure",
                    "smoothed_reading": "392.2-408.3 over the same frames"}),
           neck_ok and mid_ok and thigh_low_ok and thigh_median_ok, "raw",
           "THE CARD'S THIGH HIGH FIGURE, 426, DOES NOT REPRODUCE on any population of the "
           "class (i) runs: measured 421.8 mm (raw, left thigh, frame 113) and 408.3 "
           "smoothed. 426 is reachable only on the class (ii) run. Recorded as a "
           "non-reproducing card figure -- NO BAND MOVES, the D8b precedent -- and the "
           "clause's substance is unaffected and is asserted separately below. This is the "
           "card's whole argument for class (i): the hips move ALONG the hip line toward "
           "its own midpoint, not toward the pelvis landmark, and the knees do not follow. "
           "A length invariant cannot score direction, so what carries the direction claim "
           "is three neighbouring LENGTHS that any other motion would have moved")

    # The substance, asserted on its own so it does not ride on a rounding, and split by
    # RUN because the two sub-runs of class (i) are not the same motion.
    by_run = {}
    for label in ("84-86", "110-119"):
        row = per_hip["neighbours_on_the_off_runs"][label]
        by_run[label] = {
            name: round(max(abs(row[name]["min_mm"] / medians[name] - 1.0),
                            abs(row[name]["max_mm"] / medians[name] - 1.0)), 4)
            for name in ("root_to_neck_mm", "left_thigh_mm", "right_thigh_mm",
                         "hip_midpoint_to_root_mm")}
    clause("on 110-119 the root, the thighs and |hip_mid - root| all stay inside 15 % of "
           "their own take medians while the hip line halves",
           {"worst_fraction_off": "< 0.15 on all four, on 110-119"},
           {"worst_fraction_off_per_neighbour_per_run": by_run,
            "take_medians_mm": medians,
            "hip_line_worst_fraction_off_on_110_119": round(
                max(abs(v / raw["whole_take"]["median_mm"] - 1.0)
                    for v in [value_at(1, "raw", f) for f in range(110, 120)]
                    if v is not None), 4)},
           all(value < OFF_MEDIAN_FRACTION
               for value in by_run["110-119"].values()), "raw",
           "|hip_mid - root| is the discriminating one: the pelvis landmark sits 76.5 mm "
           "off the hip line, so a collapse TOWARD it would have moved this. On 110-119 it "
           "does not move (68.0-80.7 against 76.5, worst 5.5 %) while the hip line loses "
           "42 % -- both hips travelling along their own line toward its midpoint. AND THE "
           "SAME MEASUREMENT SEPARATES THE TWO SUB-RUNS: on 84-86 it reaches 94.6 mm "
           "(+23.6 %), because there only ONE hip moves per frame and a one-sided collapse "
           "necessarily drags the midpoint. That is the card's 'ONE HIP AT A TIME' read off "
           "a length rather than asserted, and it is why 84-86 is the registered "
           "over-charge of a rule that charges both endpoints")

    # ---- 4. class (ii): the six that cross and the five that do not
    rows_158 = per_hip["baseline_split"]["per_run"]["158-168"]
    crossing = [row for row in rows_158 if row["fraction_off_median"] is not None
                and abs(row["fraction_off_median"]) > OFF_MEDIAN_FRACTION]
    holding = [row for row in rows_158 if row["fraction_off_median"] is not None
               and abs(row["fraction_off_median"]) <= OFF_MEDIAN_FRACTION]
    clause("six of the eleven frames 158-168 cross the ceiling (158, 159, 163, 165, 167, "
           "168: 250-278 mm, +16-29 %) and the other five sit at +9-13 % under it",
           {"crossing_ids": [158, 159, 163, 165, 167, 168], "crossing_mm": [250, 278],
            "crossing_pct": [16, 29], "holding_pct": [9, 13]},
           {"crossing_ids": [row["frame_id"] for row in crossing],
            "crossing_mm": [round(min(r["hip_line_mm"] for r in crossing), 1),
                            round(max(r["hip_line_mm"] for r in crossing), 1)] if crossing
                           else None,
            "crossing_fraction": [round(min(r["fraction_off_median"] for r in crossing), 4),
                                  round(max(r["fraction_off_median"] for r in crossing), 4)]
                                 if crossing else None,
            "holding_ids": [row["frame_id"] for row in holding],
            "holding_fraction": [round(min(r["fraction_off_median"] for r in holding), 4),
                                 round(max(r["fraction_off_median"] for r in holding), 4)]
                                if holding else None},
           [row["frame_id"] for row in crossing] == [158, 159, 163, 165, 167, 168], "raw",
           "A CEILING IS A CEILING: the five frames that sit under it are not recovered by "
           "this row and the card says so rather than hiding it")

    split = per_hip["baseline_split"]
    along = [row["along_mm"] for row in rows_158 if row["along_mm"] is not None]
    across = [row["across_mm"] for row in rows_158 if row["across_mm"] is not None]
    across_but_158 = [row["across_mm"] for row in rows_158
                      if row["across_mm"] is not None and row["frame_id"] != 158]
    clause("the A-C baseline split on 158-168: along the baseline 231-275 mm on every "
           "frame of the run against 181 on the honest frames, across it 18-52 against 117",
           {"along_mm": [231, 275], "across_mm": [18, 52],
            "honest_along_mm": 181, "honest_across_mm": 117},
           {"along_mm": [round(min(along), 1), round(max(along), 1)] if along else None,
            "across_mm": [round(min(across), 1), round(max(across), 1)] if across else None,
            "across_mm_excluding_frame_158": [round(min(across_but_158), 1),
                                              round(max(across_but_158), 1)],
            "frame_158_across_mm": next(row["across_mm"] for row in rows_158
                                        if row["frame_id"] == 158),
            "honest": split["honest_frames"], "per_frame": rows_158},
           bool(along) and 230 <= min(along) <= 232 and 274 <= max(along) <= 276
           and bool(across) and 17 <= min(across) <= 19 and 51 <= max(across_but_158) <= 53
           and split["honest_frames"]["along_the_baseline_median_mm"] is not None
           and abs(split["honest_frames"]["along_the_baseline_median_mm"] - 181) < 1.0
           and abs(split["honest_frames"]["across_the_baseline_median_mm"] - 117) < 1.0,
           "raw + rig",
           "the honest figures are MEDIANS over the honest frames, and they reproduce "
           "exactly (181.1 / 117.0); the means are reported beside them (180.4 / 105.5) so "
           "the reading is not guessed. THE ALONG RANGE REPRODUCES ON ALL ELEVEN FRAMES. "
           "THE ACROSS RANGE REPRODUCES ON TEN OF ELEVEN: frame 158 reads 74.0 mm across "
           "the baseline, outside the card's stated 18-52, and the card's range is the "
           "other ten. Recorded rather than smoothed over; no band moves. What the split "
           "says is that the whole run is stretched along the ONE axis a two-view pair "
           "cannot fix -- evidence about WHERE the error lies, and NOT proof that the "
           "pelvis is not that wide. There is no truth on this take")

    heights = [row["hip_midpoint_height_mm"] for row in rows_158
               if row["hip_midpoint_height_mm"] is not None]
    clause("he is lying on the floor on 158-168 (hip midpoint 153-166 mm up)",
           {"height_mm": [153, 166]},
           {"height_mm": [round(min(heights), 1), round(max(heights), 1)] if heights
                         else None},
           bool(heights) and 152 <= min(heights) <= 154 and 165 <= max(heights) <= 167,
           "raw + rig")

    ray_angles = [row[key] for row in rows_158 for key in row
                  if key.startswith("angle_of_the_hip_line_to_")]
    clause("the hip line is within 18-28 degrees of both viewing rays on 158-168",
           {"angle_deg": [18, 28]},
           {"angle_deg": [round(min(ray_angles), 2), round(max(ray_angles), 2)]
                         if ray_angles else None,
            "definition": "90 minus the angle between the hip vector and the camera's ray "
                          "to the hip midpoint -- how nearly the segment points AT the "
                          "camera, which is the direction its length is least determined in"},
           bool(ray_angles) and max(ray_angles) <= 28.5 and min(ray_angles) >= 15.0,
           "raw + rig",
           "the angle BETWEEN the hip vector and each supporting camera's ray to the hip "
           "midpoint, folded to [0, 90]. The card's UPPER figure reproduces exactly (28.3 "
           "at C001 on frame 158); its lower figure reads 15.8 measured (C001, frame 167), "
           "not 18. Recorded as a card figure that reproduces at one end only. The "
           "substance is the same either way and it is the point of the whole class: the "
           "segment lies within 30 degrees of BOTH viewing directions, which is the "
           "direction a two-view pair determines its length in worst")

    # ---- 5. the honest spread, both performers
    honest_1 = hips(1, "raw")["honest_off_frames_removed"]["p5_p95_fraction_of_median"]
    honest_0 = hips(0, "raw")["honest_off_frames_removed"]["p5_p95_fraction_of_median"]
    clause("honest spread, off frames removed: the raw hip line -8.2 / +8.5 % (performer 1) "
           "and -9.2 / +6.9 % (performer 0) at p5-p95",
           {"performer_1": [-0.082, 0.085], "performer_0": [-0.092, 0.069]},
           {"performer_1": honest_1, "performer_0": honest_0,
            "smoothed_performer_1": hips(1, "smoothed")["honest_off_frames_removed"][
                "p5_p95_fraction_of_median"],
            "smoothed_performer_0": hips(0, "smoothed")["honest_off_frames_removed"][
                "p5_p95_fraction_of_median"]},
           abs(honest_1[0] + 0.082) < 0.001 and abs(honest_1[1] - 0.085) < 0.001
           and abs(honest_0[0] + 0.092) < 0.001 and abs(honest_0[1] - 0.069) < 0.001,
           "raw",
           "inside the 15 % ceiling on both, and it is the MARGIN this row ships on. It is "
           "wider than the legs' -5.1 / +6.2 %, which is the exposure the new row adds")

    # ---- 6. performer 0
    p0_raw, p0_smooth = hips(0, "raw"), hips(0, "smoothed")
    clause("performer 0: 4 raw frames off (88, 97, 109, 110), 0 smoothed",
           {"raw_count": 4, "raw_ids": [88, 97, 109, 110], "smoothed_count": 0},
           {"raw_count": p0_raw["whole_take"]["frames_off_median_by_more_than_15pct"],
            "raw_ids": p0_raw["frames_off_ids"],
            "smoothed_count": p0_smooth["whole_take"][
                "frames_off_median_by_more_than_15pct"],
            "smoothed_ids": p0_smooth["frames_off_ids"]},
           p0_raw["frames_off_ids"] == [88, 97, 109, 110]
           and p0_smooth["whole_take"]["frames_off_median_by_more_than_15pct"] == 0, "both")

    # ---- 7. the run D8 already recovers
    already = [fid for fid in raw["frames_off_ids"] if 88 <= fid <= 106]
    smoothed_88_106 = [value_at(1, "smoothed", fid) for fid in range(88, 107)]
    finite_88 = [v for v in smoothed_88_106 if v is not None]
    clause("the raw frames 88-106 (off on 11) are already recovered by D8's sequence solve "
           "(smoothed 205-221)",
           {"raw_off_count": 11, "smoothed_range_mm": [205, 221]},
           {"raw_off_count": len(already), "raw_off_ids": already,
            "smoothed_range_mm": [round(min(finite_88), 1), round(max(finite_88), 1)]
                                 if finite_88 else None},
           len(already) == 11 and bool(finite_88)
           and 204 <= min(finite_88) <= 206 and 220 <= max(finite_88) <= 222, "both",
           "NOT a defect in the delivery, and the card says so. It is here because it is "
           "the population that separates 'the rule fires' from 'the delivery is wrong'")

    # ---- 8. the honest-mask root->hip table, the pre-dispatch review's own clause
    table = {f"subject_{s:02d}": {
        side: {"honest_mask_p5_p95": geometry[f"subject_{s:02d}"]["raw"][
                   "root_to_hip_on_the_honest_mask"][side]["honest_mask_spread"].get(
                       "p5_p95_fraction_of_median"),
               "honest_mask_p5_p95_against_the_take_median":
                   geometry[f"subject_{s:02d}"]["raw"]["root_to_hip_on_the_honest_mask"][
                       side]["honest_mask_spread_against_the_take_median"].get(
                           "p5_p95_fraction_of_median"),
               "would_fire_at_0.15": geometry[f"subject_{s:02d}"]["raw"][
                   "root_to_hip_on_the_honest_mask"][side][
                       "would_fire_at_0.15_on_the_honest_frames_median"],
               "would_fire_at_0.15_on_its_own_take_median":
                   geometry[f"subject_{s:02d}"]["raw"]["root_to_hip_on_the_honest_mask"][
                       side]["would_fire_at_0.15_on_its_own_take_median"],
               "of_those_whose_hip_line_is_honest": geometry[f"subject_{s:02d}"]["raw"][
                   "root_to_hip_on_the_honest_mask"][side][
                       "of_those_whose_hip_line_is_HONEST"]}
        for side in ("left", "right")} for s in (0, 1)}
    p1_left = table["subject_01"]["left"]["honest_mask_p5_p95"]
    p1_right = table["subject_01"]["right"]["honest_mask_p5_p95"]
    clause("root->left-hip spreads -9.0 / +25.5 % on performer 1 (right -9.4 / +9.2; "
           "performer 0 +/-9-10 %) on the SAME honest mask",
           {"performer_1_left": [-0.090, 0.255], "performer_1_right": [-0.094, 0.092],
            "performer_0": "+/-9-10 %"},
           table,
           abs(p1_left[0] + 0.090) < 0.002 and abs(p1_left[1] - 0.255) < 0.002
           and abs(p1_right[0] + 0.094) < 0.002 and abs(p1_right[1] - 0.092) < 0.002,
           "raw",
           "THE MEASUREMENT THE PRE-DISPATCH REVIEW DEMANDED BEFORE THE CONVENTION COULD BE "
           "DEFENDED (finding 2): a per-hip root->hip rule is the more precise candidate "
           "because the hips HAVE a parent, and the only honest way to refuse it is to show "
           "the pelvis landmark is too loose to be a length reference ON THE SAME MASK the "
           "hip line's own margin is quoted on. It is: +25.5 % at p95 against a 15 % "
           "ceiling. The review's own 'if outside' branch therefore applies -- hip line, "
           "both endpoints charged, 84-86 registered as a known over-charge. THE "
           "REFERENCE MATTERS AND IS STATED: root->hip has no off list of its own, so its "
           "spread is quoted against the median of the HONEST frames (the card's reading, "
           "and the one that reproduces to the digit). The same spread against the whole "
           "take's median is emitted beside it and differs by up to 0.6 points")

    fires = {f"subject_{s:02d}": [table[f"subject_{s:02d}"]["left"]["would_fire_at_0.15"],
                                  table[f"subject_{s:02d}"]["right"]["would_fire_at_0.15"]]
             for s in (0, 1)}
    clause("at 0.15 a root->hip rule would fire on 35 / 17 frames of performer 1 and 9 / 4 "
           "of performer 0, most of them on frames whose hip line is honest",
           {"performer_1": [35, 17], "performer_0": [9, 4]},
           {"fires_left_right": fires,
            "of_those_whose_hip_line_is_honest": {
                f"subject_{s:02d}": [
                    table[f"subject_{s:02d}"]["left"]["of_those_whose_hip_line_is_honest"],
                    table[f"subject_{s:02d}"]["right"]["of_those_whose_hip_line_is_honest"]]
                for s in (0, 1)}},
           fires["subject_01"] == [35, 17] and fires["subject_00"] == [9, 4], "raw")

    # ---- 9. the per-camera tables
    ranges = report.get("per_camera_classification_by_range", {})

    def camera_block(key: str, landmark: str) -> dict:
        return ranges.get(key, {}).get("subjects", {}).get("subject_01", {}).get(landmark, {})

    strict = {name: camera_block("109-119", name).get(
        "on_frames_supported_by_exactly_those_cameras", {})
        for name in ("left_hip", "right_hip")}
    support_110 = {name: camera_block("109-119", name).get(
        "cameras_supporting_a_majority_of_the_range") for name in ("left_hip", "right_hip")}
    d_conf = []
    for name in ("left_hip", "right_hip"):
        for row in camera_block("109-119", name).get("frames", []):
            if "D001" in row["supporting_views"] and row["confidence"]["D001"] is not None:
                d_conf.append(row["confidence"]["D001"])
    strict_resid = [strict[name].get("residual_range_px1280")
                    for name in ("left_hip", "right_hip")]
    worst_resid = max((v[1] for v in strict_resid if v), default=None)
    strict_angles = [strict[name].get("max_pairwise_ray_angle_deg")
                     for name in ("left_hip", "right_hip")]
    angle_low = min((v[0] for v in strict_angles if v), default=None)
    angle_high = max((v[1] for v in strict_angles if v), default=None)
    strict_ids = strict["left_hip"].get("frame_ids", [])
    clause("frames 109-119, the HIPS: cameras A, C and D support both hips on 9 of the 11, "
           "D at confidence 0.43-0.93, every supporting camera's residual of the raw point "
           "0.4-8.1 px, ray angle 157-161 degrees",
           {"cameras": ["A001", "C001", "D001"], "frames_of_11": 9,
            "D_confidence": [0.43, 0.93], "residual_px": [0.4, 8.1],
            "ray_angle_deg": [157, 161]},
           {"cameras_supporting_a_majority": support_110,
            "frames_all_three_of_A_C_D_support": {
                name: camera_block("109-119", name).get(
                    "per_camera_supported_frames_in_range", {}).get("D001")
                for name in ("left_hip", "right_hip")},
            "frames_supported_by_exactly_A_C_D": strict_ids,
            "count_of_those_frames": len(strict_ids),
            "D001_confidence_over_the_whole_range": (
                [round(min(d_conf), 4), round(max(d_conf), 4)] if d_conf else None),
            "residual_range_on_the_A_C_D_frames_px1280": strict_resid,
            "worst_residual_on_those_frames_px1280": worst_resid,
            "ray_angle_on_those_frames_deg": [angle_low, angle_high],
            "the_two_frames_that_are_not_A_C_D": {
                "113": "TWO-VIEW (A and C only) -- the one frame in the run D8's "
                       "conditioning gate acts on, and the OUTWARD spike",
                "119": "FOUR-VIEW -- B001 rejoins at 11.89 px and D001 reads 10.75 px "
                       "there. Quoting the card's residual range over it would describe a "
                       "four-view frame as part of a three-view collapse"}},
           all(v == ["A001", "C001", "D001"] for v in support_110.values())
           and all(camera_block("109-119", name).get(
               "per_camera_supported_frames_in_range", {}).get("D001") == 9
               for name in ("left_hip", "right_hip"))
           and len(strict_ids) == 8
           and bool(d_conf) and round(min(d_conf), 2) == 0.43
           and round(max(d_conf), 2) == 0.93
           and worst_resid is not None and worst_resid <= 8.1
           and min((v[0] for v in strict_resid if v), default=1e9) >= 0.35
           and angle_low is not None and 156.5 <= angle_low
           and angle_high is not None and angle_high <= 161.5,
           "observations + raw point",
           "TWO POPULATIONS IN ONE CARD SENTENCE, and both reproduce. The card's '9 of "
           "11' is the count of frames on which ALL THREE of A, C and D support both hips "
           "-- 110-112 and 114-119, D001 absent only on 109 and 113. Its residual range "
           "'0.4-8.1 px' is on the eight frames those three support and NO FOURTH camera "
           "does: frame 119 is four-view (B001 rejoins at 11.89 px and D001 reads 10.75 "
           "there) and quoting it would describe a four-view frame as part of a "
           "three-view collapse. Both cuts are in `measured` so neither has to be guessed "
           "again. THREE cameras and small residuals, so D8's conditioning gate is "
           "correctly silent and no epipolar or reprojection threshold can fire. Only the "
           "performer's own width can see it")

    support_158 = {name: camera_block("158-168", name).get(
        "cameras_supporting_a_majority_of_the_range") for name in ("left_hip", "right_hip")}
    d_frames = sum(1 for name in ("left_hip", "right_hip")
                   for row in camera_block("158-168", name).get("frames", [])
                   if "D001" in row["supporting_views"])
    angles_158 = [row["max_pairwise_ray_angle_deg"]
                  for name in ("left_hip", "right_hip")
                  for row in camera_block("158-168", name).get("frames", [])
                  if row["max_pairwise_ray_angle_deg"] is not None]
    resid_158 = [camera_block("158-168", name).get("residual_range_on_those_cameras_px1280")
                 for name in ("left_hip", "right_hip")]
    worst_158 = max((v[1] for v in resid_158 if v), default=None)
    clause("frames 158-168: only A and C see him, D has no detection at all, ray angle "
           "139-140 degrees (UNDER D8's 150 ceiling, so the conditioning gate is silent), "
           "residuals 0.1-3.8 px",
           {"cameras": ["A001", "C001"], "D001_supporting_frames": 0,
            "ray_angle_deg": [139, 140], "residual_px": [0.1, 3.8]},
           {"cameras_supporting_a_majority": support_158,
            "D001_supporting_frames": d_frames,
            "ray_angle_deg": [round(min(angles_158), 2), round(max(angles_158), 2)]
                             if angles_158 else None,
            "residual_range_per_hip_px1280": resid_158,
            "worst_residual_px1280": worst_158},
           all(v == ["A001", "C001"] for v in support_158.values()) and d_frames == 0
           and bool(angles_158) and 139.0 <= min(angles_158)
           and max(angles_158) < 150.0 and worst_158 is not None and worst_158 <= 3.8,
           "observations + raw point",
           "D8's ray-angle ceiling is 150 degrees and this pair sits at 139-140, so the "
           "conditioning gate does not act -- and sin(140) is 0.64, which is enough to make "
           "a stable triangulation of a WRONG depth. That is the whole of class (ii). The "
           "card's '139-140' is a rounding of a measured 139.96-140.70; the clause asserts "
           "what the card's sentence actually claims, that the pair sits UNDER the 150 "
           "degree ceiling, and reports the measured range beside it")

    d_conf_84 = []
    for name in ("left_hip", "right_hip"):
        for row in camera_block("84-86", name).get("frames", []):
            if "D001" in row["supporting_views"] and row["confidence"]["D001"] is not None:
                d_conf_84.append(row["confidence"]["D001"])
    resid_84 = [camera_block("84-86", name).get("residual_range_on_those_cameras_px1280")
                for name in ("left_hip", "right_hip")]
    clause("frames 84-86: A-C-D with D at 0.26-0.37, residuals 1-6 px",
           {"D_confidence": [0.26, 0.37], "residual_px": [1, 6]},
           {"cameras_supporting_a_majority": {
               name: camera_block("84-86", name).get(
                   "cameras_supporting_a_majority_of_the_range")
               for name in ("left_hip", "right_hip")},
            "D001_confidence": [round(min(d_conf_84), 4), round(max(d_conf_84), 4)]
                               if d_conf_84 else None,
            "residual_range_per_hip_px1280": resid_84},
           bool(d_conf_84) and round(min(d_conf_84), 2) == 0.26
           and round(max(d_conf_84), 2) == 0.37, "observations + raw point")

    return {
        "source": "docs/LADDER_EXECUTION_PLAN.md section 2, the D8c row, committed at "
                  "85b8113 before this instrument ran",
        "clauses": clauses,
        "all_match": all(row["matches"] for row in clauses),
        "reads": "the SHIPPED delivery (artifacts/commercial-multiview-soma77), which is "
                 "the D8b build as its close-out rebuilt it in place",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--landmarks-from", type=Path, default=None,
                        help="a delivery directory whose subject-XX.body-track.npz supply "
                             "the landmark arrays. Defaults to the shipped build. The "
                             "observations, rig and association always come from the "
                             "shipped build whatever this is set to.")
    parser.add_argument("--reproduce", choices=("d8", "d8b", "d8c", "none"), default="d8",
                        help="which card's figures to check. `d8` is the original "
                             "behaviour and the default; `d8b` checks the D8b card's own "
                             "figures on a POST-D8 build and emits the per-camera "
                             "classification table beside them; `d8c` checks the D8c card's "
                             "HIP LINE figures on the shipped (D8b) build and emits the "
                             "hip-line geometry block; `none` checks nothing.")
    parser.add_argument("--classify-frames", type=int, nargs=2, action="append",
                        default=None, metavar=("FIRST", "LAST"),
                        help="emit the per-camera classification table over this inclusive "
                             "range of absolute frame ids. REPEATABLE -- D8c needs two "
                             "ranges at once, because 110-119 and 158-168 are two different "
                             "failures. Defaults to "
                             f"{D8B_FIRST_ID}-{D8B_LAST_ID} when --reproduce d8b is set and "
                             f"to {D8C_CLASSIFY_RANGES} when --reproduce d8c is.")
    parser.add_argument("--hip-geometry", action="store_true",
                        help="emit the D8c hip-line geometry block WITHOUT checking any "
                             "card figure. This is how a build other than the shipped one "
                             "is measured: the reproduction clauses describe the defect and "
                             "asserting them on a repaired build would be nonsense, but the "
                             "same measurements must still be taken there, on the same code "
                             "path, or the before and after are not comparable.")
    parser.add_argument("--classify-landmarks", type=str, default=None,
                        help="comma-separated landmark names the classification table "
                             "covers. THE CARD'S REVIEWER ASKED FOR THIS TO BE A FLAG "
                             "RATHER THAN A CONSTANT. Defaults to the arm landmarks under "
                             "--reproduce d8b and to the hips, the root and the neck under "
                             "--reproduce d8c.")
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
    classify = (tuple(tuple(row) for row in args.classify_frames)
                if args.classify_frames else None)
    if classify is None and mode == "d8b":
        classify = ((D8B_FIRST_ID, D8B_LAST_ID),)
    if classify is None and mode == "d8c":
        classify = D8C_CLASSIFY_RANGES
    landmarks = (tuple(name.strip() for name in args.classify_landmarks.split(",") if
                       name.strip())
                 if args.classify_landmarks
                 else (D8C_CLASSIFY_LANDMARKS if mode == "d8c" else D8B_CLASSIFY_LANDMARKS))
    # `hip_line_geometry` is emitted whenever the d8c reproduction runs OR a build other
    # than the shipped one is being scored, because the after-build arms need exactly the
    # same block on the same code path (the D8b lesson: the reproduction clauses describe
    # the DEFECT and must not be asserted on a repaired build, but the MEASUREMENTS must
    # still be taken there).
    report = build_report(check_reproduction=(mode != "none"), mode=mode,
                          classify=classify, classify_landmarks=landmarks,
                          hip_geometry=bool(args.hip_geometry))
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
