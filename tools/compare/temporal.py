#!/usr/bin/env python3
"""I7 -- the temporal stage, measured. Recovery, and what smoothing costs.

Rung 5 of `docs/SUBSTITUTION_LADDER.md` has been **dark** since it was written:
`solve_sequence_positions` + the fill + the Savitzky-Golay window touch every
joint, and both candidate references are banned (MAMMA's mesh is itself smoothed;
its `triangulated_3d_pts` are correlated with the 2D being smoothed). I2 gave the
rung an oracle on perfect 2D and immediately named its own blind spot: *"the
temporal stage is what recovers joints when a camera loses one, and there are no
such frames here."* This instrument makes such frames.

Two halves, per `LADDER_EXECUTION_PLAN.md` §2, row I7.

**5a -- recovery.** Project an exactly-known trajectory into the four calibrated
cameras, then delete observations until chosen (joint, frame) cells are seen by
exactly one camera. A single ray is underdetermined alone; a ray plus a limb
length from an already-solved parent leaves a circle, and temporal continuity
picks the point on it. Score the recovered cells against exact truth, beside an
arm with the sequence solve switched off so the same cells are filled by
interpolation instead.

**5b -- smoothing.** On full coverage, inject isolated 2D spikes at our own
detector's measured amplitude, gaps of 1/3/5 frames, and let the trajectory's own
wrist and ankle peaks be the fast events. Measure 3D error, phase lag by shift
sweep, peak attenuation and recovery latency, against the window at 3x and at
"no Savitzky-Golay" (window below the filter's own minimum).

**Plus a held-out-camera lag on real footage.** Reconstruct from three cameras,
reproject the smoothed 3D into the fourth, and sweep against that camera's own
detections. Four folds. Blind to depth, by construction.

    .venv/bin/python tools/compare/temporal.py

Two trajectory sources, labelled, and NEVER on one axis
-------------------------------------------------------
1. **`fk_synthetic`** -- forward kinematics of SOMA-77 clips through
   `scripts/build_synthetic_truth_fixture.py`'s own take builder, played at
   `--stride` times speed. **MAMMA-free.** This is the only arm that could ever
   select a shipped constant (`smooth_weight`, `length_weight`,
   `SMOOTHING_WINDOW_FRAMES` are all on I8's unknown-origin list). It selects
   nothing today; it reports.
2. **`mamma_pred_joints`** -- MAMMA's fitted joints as a trajectory. Realistic
   motion at real speed, exact truth once projected, and **MAMMA-derived**: it
   reports and never selects. Separate reference string, separate charts.

Standing rules honoured here
----------------------------
* **Wrap the pipeline; never re-implement it.** Every arm calls the real
  `reconstruct_multiview`. The observation records come from the real record
  builders (`oracle_2d.observations_for`, `build_synthetic_truth_fixture.
  observations_for`); this file masks and displaces entries in what they built and
  never writes a second projection loop. The per-cell `recovered` mask comes from
  a recording wrapper around the real `solve_sequence_positions`, and the
  interpolation-only control is that same wrapper with a pass-through body -- an
  ablation of one stage, not a copy of it.
* **No gate a constant can pass.** Frozen-trajectory controls on both halves.
* **Same denominator.** Every arm and its control are scored on the *same cells*;
  every lag figure is scored on the *same fixed frame set for every shift*.
* MAMMA is an instrument; the report lands under `artifacts/compare/`.
* Subject correspondence never comes from the index.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
from hashlib import sha256
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts", "workers/commercial_multiview"):
    sys.path.insert(0, str(ROOT / _relative))

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
import oracle_2d as o2  # noqa: E402
import build_synthetic_truth_fixture as fx  # noqa: E402
from soma77_pose import SOMA77_TO_AUTOANIM  # noqa: E402

CAMERAS = ("A001", "B001", "C001", "D001")
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720
SAMPLE_RATE_HZ = 30
RIG = ROOT / "artifacts/commercial-multiview-soma77/camera-rig.json"
SOMA77_WORK = ROOT / "artifacts/commercial-multiview-soma77/work"
SELF_AGREEMENT = ROOT / "artifacts/compare/detector-self-agreement.json"
REPORT = ROOT / "artifacts/compare/temporal.json"

# `_fill_and_smooth_positions` skips Savitzky-Golay entirely when the window it
# computes is under 5, so 3 is "fill, then nothing" without touching src/.
WINDOW_DEFAULT = cm.SMOOTHING_WINDOW_FRAMES          # 9 -- 300 ms at 30 fps
WINDOW_IDENTITY = 3
WINDOW_OVERSMOOTHED = 3 * WINDOW_DEFAULT             # 27 -- the plan's "3x the default"

# The recovery gate inside `solve_sequence_positions`: a solved point must lie
# within `robust_scale_px` of every ray that saw it or it is demoted to the
# caller's interpolation. Named here so the demoted fraction can be read.
RECOVERY_RAY_GATE_PX = 14.0

BLOCK_FRAMES = 15
BOOTSTRAP_DRAWS = 1000
LAG_SWEEP = 3                # +/- frames swept for every lag figure
GAP_LENGTHS = (1, 3, 5)
LATENCY_CAP_FRAMES = 15

# Joints starved in the recovery arms. `root` is deliberately absent: a starved
# root fails `triangulate_point`, the subject-frame is rejected wholesale, and
# `retained_observations` keeps NOTHING for that frame -- so a single ray on a
# well-seen elbow is discarded too. That is whole-frame loss, not single-ray
# recovery, and it gets its own diagnostic arm instead of contaminating these.
ROOT_NAME = "root"


# --------------------------------------------------------------------------- utilities

def sha256_of(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarise(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "median_mm": None, "p95_mm": None, "max_mm": None}
    return {"n": int(finite.size),
            "median_mm": round(float(np.median(finite)), 4),
            "p95_mm": round(float(np.percentile(finite, 95)), 4),
            "max_mm": round(float(finite.max()), 4)}


def lag1_autocorrelation(series: np.ndarray) -> float | None:
    """Lag-1 autocorrelation of this instrument's OWN residual series.

    CLAUDE.md's 0.99 belongs to another lane's series and is never assumed here.
    """
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0:
        return None
    return round(float(np.dot(centred[:-1], centred[1:]) / denominator), 4)


def block_bootstrap_pair(a_by_frame: list[np.ndarray], b_by_frame: list[np.ndarray],
                         seed: int = 7) -> dict:
    """Moving-block bootstrap of two arms' medians on IDENTICAL draws.

    Both arms are lists indexed by frame, holding that frame's error values. A
    draw picks the same block of frames for both, so the pairing survives and the
    difference is what is bootstrapped -- never two independent resamplings.
    """
    frames = len(a_by_frame)
    if frames < 2:
        return {"draws": 0, "note": "too few frames for a block bootstrap"}
    block = min(BLOCK_FRAMES, max(2, frames // 2))
    blocks = int(math.ceil(frames / block))
    rng = np.random.default_rng(seed)
    starts_max = max(frames - block, 0)
    a_medians, b_medians, wins = [], [], 0
    for _ in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, starts_max + 1, blocks)
        a_draw = np.concatenate([np.concatenate(a_by_frame[s:s + block] + [np.empty(0)])
                                 for s in starts])
        b_draw = np.concatenate([np.concatenate(b_by_frame[s:s + block] + [np.empty(0)])
                                 for s in starts])
        a_draw, b_draw = a_draw[np.isfinite(a_draw)], b_draw[np.isfinite(b_draw)]
        if a_draw.size == 0 or b_draw.size == 0:
            continue
        a_median, b_median = float(np.median(a_draw)), float(np.median(b_draw))
        a_medians.append(a_median)
        b_medians.append(b_median)
        wins += int(a_median < b_median)
    if not a_medians:
        return {"draws": 0, "note": "no draw held finite values in both arms"}
    return {
        "block_frames": block, "blocks_per_draw": blocks, "draws": len(a_medians),
        "candidate_median_ci95_mm": [round(float(np.percentile(a_medians, 2.5)), 4),
                                     round(float(np.percentile(a_medians, 97.5)), 4)],
        "control_median_ci95_mm": [round(float(np.percentile(b_medians, 2.5)), 4),
                                   round(float(np.percentile(b_medians, 97.5)), 4)],
        "p_candidate_beats_control": round(wins / len(a_medians), 4),
    }


# ------------------------------------------------------------------- pipeline wrappers

class SequenceSolveLog:
    """One entry per subject, in the order `reconstruct_multiview` solves them."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def recovered(self, subject: int) -> np.ndarray:
        return self.entries[subject]["recovered"]

    def support(self, subject: int) -> np.ndarray:
        return self.entries[subject]["support"]

    def direct(self, subject: int) -> np.ndarray:
        return self.entries[subject]["direct"]


@contextlib.contextmanager
def sequence_solve(mode: str = "real"):
    """Wrap the REAL `solve_sequence_positions`, or ablate it. Never copy it.

    `mode="real"` calls the real function and records, per subject, the per-cell
    `recovered` mask, the pre-solve `direct` mask and the per-cell camera support
    -- the three things the pipeline computes and then averages away. `mode="off"`
    substitutes a pass-through that returns its input untouched with an
    all-False mask, which is the interpolation-only control: the caller's
    `_fill_and_smooth_positions` then draws a line through the same cells.
    """
    real = cm.solve_sequence_positions
    log = SequenceSolveLog()

    def record(cameras, world, observations, **kwargs):
        positions = np.asarray(world, dtype=np.float64)
        observed = np.asarray(observations, dtype=np.float64)
        seen = (np.isfinite(observed[..., :2]).all(axis=3)
                & (observed[..., 2] >= kwargs.get("minimum_confidence", 0.25))).transpose(0, 2, 1)
        entry = {"direct": np.isfinite(positions).all(axis=2), "support": seen.sum(axis=2)}
        if mode == "off":
            output, recovered = positions.copy(), np.zeros(positions.shape[:2], dtype=bool)
        else:
            output, recovered = real(cameras, world, observations, **kwargs)
        entry["recovered"] = np.asarray(recovered, dtype=bool)
        log.entries.append(entry)
        return output, recovered

    cm.solve_sequence_positions = record
    try:
        yield log
    finally:
        cm.solve_sequence_positions = real


@contextlib.contextmanager
def smoothing_window(frames: int):
    """Set the shipped Savitzky-Golay window for the duration. A knob, not a copy."""
    previous = cm.SMOOTHING_WINDOW_FRAMES
    cm.SMOOTHING_WINDOW_FRAMES = frames
    try:
        yield
    finally:
        cm.SMOOTHING_WINDOW_FRAMES = previous


def run_pipeline(cameras, records, *, subject_count: int = 2, solve: str = "real",
                 window: int = WINDOW_DEFAULT):
    """One pass of the real pipeline. Returns (positions, raw, diagnostics, log, seconds)."""
    started = time.time()
    with smoothing_window(window), sequence_solve(solve) as log:
        _tracks, diagnostics, positions, raw = cm.reconstruct_multiview(
            cameras, records, subject_count=subject_count, sample_rate_hz=SAMPLE_RATE_HZ
        )
    return positions, raw, diagnostics.as_dict(), log, time.time() - started


# ------------------------------------------------------------------------ trajectories

def working_cameras():
    return tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT)
                 for camera in cm.load_camera_rig(RIG))


def truth19_from(source: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    """[subject, frame, 19, 3] in the pipeline's own joint order, NaN where absent.

    An index gather, not a stage. `build_take` returns [frame, 77, 3] in SOMA-77
    order -- NOT the 19-joint order the original plan brief claimed, a correction
    `tools/head/thorax_window_sweep.py` already recorded and this file inherits.
    """
    frames = source.shape[1]
    out = np.full((source.shape[0], frames, len(cm.JOINT_NAMES), 3), np.nan, dtype=np.float64)
    for name, index in mapping.items():
        out[:, :, cm.JOINT_INDEX[name]] = source[:, :, index]
    return out


def mamma_trajectory():
    """MAMMA's `pred_joints` as a trajectory: 150 frames, real speed, MAMMA-derived."""
    truth127 = o2.load_truth()
    truth19 = truth19_from(truth127, o2.PAIRS)
    return truth127, truth19, tuple(o2.PAIRS)


def fk_trajectory(stride: int, clips: tuple[str, str]):
    """Two FK takes from the synthetic fixture's own builder. MAMMA-FREE."""
    placement = ROOT / "artifacts/handfit-arrays/body-track.npz"
    if placement.exists():
        track = np.load(placement, allow_pickle=True)["positions"]
        roots = track[:, :, cm.JOINT_INDEX["root"], :2]
        ground = np.asarray([np.nanmedian(roots[s], axis=0) for s in range(min(2, len(roots)))])
    else:
        ground = np.asarray([[0.5, 4.9], [-0.9, 4.7]])
    takes = [fx.build_take(clip, None, ground[index % len(ground)], stride)
             for index, clip in enumerate(clips)]
    length = min(len(take) for take in takes)
    soma = np.stack([take[:length] for take in takes])
    return soma, truth19_from(soma, SOMA77_TO_AUTOANIM), tuple(SOMA77_TO_AUTOANIM)


def joint_speed_m_s(truth19: np.ndarray, names: tuple[str, ...]) -> dict:
    columns = [cm.JOINT_INDEX[n] for n in names]
    step = np.linalg.norm(np.diff(truth19[:, :, columns], axis=1), axis=3) * SAMPLE_RATE_HZ
    finite = step[np.isfinite(step)]
    return {"median_m_s": round(float(np.median(finite)), 4),
            "p95_m_s": round(float(np.percentile(finite, 95)), 4),
            "max_m_s": round(float(finite.max()), 4)}


# ------------------------------------------------------------------------- observations

def build_records(cameras, source_name: str, source_array: np.ndarray) -> list[list[dict]]:
    """The REAL record builders, one per source. No projection loop lives here."""
    if source_name == "mamma_pred_joints":
        return o2.observations_for(cameras, source_array)
    built = fx.observations_for(cameras, source_array)
    return [built[name] for name in CAMERAS]


def check_person_order(records: list[list[dict]], subject_count: int) -> None:
    """Both record builders append `people` in subject order; assert it holds.

    If any frame in any camera drops a subject entirely the person index stops
    meaning the subject index, and every mask this file applies would be wrong in
    a way no error message would show.
    """
    for camera, rows in enumerate(records):
        for row in rows:
            if len(row["people"]) != subject_count:
                raise SystemExit(
                    f"camera {CAMERAS[camera]} frame {row['frame_index']} has "
                    f"{len(row['people'])} people, not {subject_count}: the person index no "
                    "longer means the subject index and every mask below would be misapplied")


def apply_keep_mask(records: list[list[dict]], keep: np.ndarray) -> list[list[dict]]:
    """Delete observations. `keep` is [subject, frame, camera, 19] -- True survives."""
    out: list[list[dict]] = []
    for camera, rows in enumerate(records):
        new_rows = []
        for frame, row in enumerate(rows):
            people = []
            for subject, person in enumerate(row["people"]):
                joints = {name: value for name, value in person["joints"].items()
                          if keep[subject, frame, camera, cm.JOINT_INDEX[name]]}
                people.append({"index": person["index"], "joints": joints})
            new_rows.append(dict(row, people=people))
        out.append(new_rows)
    return out


def apply_displacements(records: list[list[dict]], displacement: dict) -> list[list[dict]]:
    """Move observations. `displacement` maps (subject, frame, camera, joint) -> (du, dv)."""
    out: list[list[dict]] = []
    for camera, rows in enumerate(records):
        new_rows = []
        for frame, row in enumerate(rows):
            people = []
            for subject, person in enumerate(row["people"]):
                joints = {}
                for name, value in person["joints"].items():
                    shift = displacement.get((subject, frame, camera, cm.JOINT_INDEX[name]))
                    if shift is None:
                        joints[name] = value
                    else:
                        joints[name] = dict(value, x=float(value["x"] + shift[0]),
                                            y=float(value["y"] + shift[1]))
                people.append({"index": person["index"], "joints": joints})
            new_rows.append(dict(row, people=people))
        out.append(new_rows)
    return out


# ------------------------------------------------------------------------ outage models

def real_run_seen_mask() -> np.ndarray:
    """[subject, frame, camera, 19] booleans from the REAL SOMA-77 run's own association.

    Obtained by wrapping `reconstruct_multiview` with a recording associator --
    the rows the associator hands back are the very rows it was given, so this is
    the pipeline's own assignment and not a second copy of its association loop
    (CLAUDE.md; a hand replication drifted 9-19 mm).
    """
    cameras = working_cameras()
    records = [cm.load_observation_jsonl(SOMA77_WORK / f"{name}-soma77-observations.jsonl")
               for name in CAMERAS]
    captured: list[np.ndarray] = []

    def recording_associator(*args, **kwargs):
        result = cm.associate_frame_graph(*args, **kwargs)
        captured.append(np.array(result[0], copy=True))
        return result

    cm.reconstruct_multiview(cameras, records, subject_count=2,
                             sample_rate_hz=SAMPLE_RATE_HZ, associator=recording_associator)
    assigned = np.stack(captured, axis=1)                     # [subject, frame, camera, 19, 3]
    return (np.isfinite(assigned[..., :2]).all(axis=-1)) & (assigned[..., 2] >= 0.25)


def describe_real_pattern(seen: np.ndarray) -> dict:
    """What the reference footage's occlusion actually is. A plan correction lives here."""
    emitted = [cm.JOINT_INDEX[n] for n in cm.JOINT_NAMES if n not in ("left_ear", "right_ear")]
    restricted = seen[:, :, :, emitted]
    support = restricted.sum(axis=2)
    runs: list[int] = []
    for subject in range(restricted.shape[0]):
        for camera in range(restricted.shape[2]):
            for joint in range(restricted.shape[3]):
                length = 0
                for value in ~restricted[subject, :, camera, joint]:
                    if value:
                        length += 1
                    elif length:
                        runs.append(length)
                        length = 0
                if length:
                    runs.append(length)
    runs_array = np.asarray(runs) if runs else np.zeros(0)
    seen_rate = float(restricted.mean())
    independent = (1 - seen_rate) ** 4 + 4 * seen_rate * (1 - seen_rate) ** 3
    correlations = {}
    misses = ~restricted
    for first in range(4):
        for second in range(first + 1, 4):
            a = misses[:, :, first, :].ravel().astype(float)
            b = misses[:, :, second, :].ravel().astype(float)
            if a.std() > 0 and b.std() > 0:
                correlations[f"{CAMERAS[first]}-{CAMERAS[second]}"] = round(
                    float(np.corrcoef(a, b)[0, 1]), 4)
    histogram = np.bincount(support.ravel(), minlength=5).tolist()
    return {
        "population": "the delivered SOMA-77 run, 2 subjects x 150 frames x 17 emitted "
                      "joints = 5,100 slots; the two ear joints are excluded because "
                      "SOMA-77 never emits them and they carry zero rays on every frame",
        "camera_support_histogram_0_to_4": histogram,
        "single_ray_slots": histogram[1],
        "single_ray_fraction": round(histogram[1] / support.size, 6),
        "no_ray_slots": histogram[0],
        "two_eligible_view_fraction": round(histogram[2] / support.size, 4),
        "per_camera_seen_rate": {name: round(float(restricted[:, :, index].mean()), 4)
                                 for index, name in enumerate(CAMERAS)},
        "cross_camera_miss_correlation": correlations,
        "miss_run_length_frames": {
            "n": int(runs_array.size),
            "median": float(np.median(runs_array)) if runs_array.size else None,
            "p90": float(np.percentile(runs_array, 90)) if runs_array.size else None,
            "max": int(runs_array.max()) if runs_array.size else None},
        "starved_fraction_if_views_were_independent": round(float(independent), 5),
        "starved_fraction_measured": round(float((support <= 1).mean()), 5),
        "plan_correction": (
            "The I7 brief said the board records occlusion as 'strongly correlated across "
            "views: 35% starved vs 6% if independent', and told this instrument to replay "
            "that pattern. Two things about it are wrong for this purpose. (1) The board's "
            "35.3% is the TWO-ELIGIBLE-VIEW fraction -- slots a four-camera rig triangulates "
            "perfectly well -- not a starvation fraction; recounted on the 17 emitted joints "
            f"it is {round(histogram[2] / support.size * 100, 1)}%. The fraction that actually "
            "reaches `solve_sequence_positions`, one ray or none, is "
            f"{histogram[1]} slots in {support.size}. (2) The correlation runs the OTHER way "
            "for starvation: two cameras see almost everything and two drop in correlated "
            "bursts (per-camera seen rate on emitted joints "
            + ", ".join(f"{name} {restricted[:, :, index].mean():.3f}"
                        for index, name in enumerate(CAMERAS))
            + "), so support stays >= 2 nearly always and the independent model OVER-predicts "
            "starvation "
            f"({round(independent * 100, 2)}% predicted against "
            f"{round(float((support <= 1).mean()) * 100, 3)}% measured). Replaying the real "
            "pattern therefore exercises the sequence solve on a handful of cells and no "
            "more, which is why the amplified arm below exists and carries the figure."),
    }


def replayed_keep(seen: np.ndarray, frames: int, names: tuple[str, ...], offset: int) -> np.ndarray:
    """Arm (i): the real run's own camera-count pattern, frame-for-frame."""
    keep = np.ones((seen.shape[0], frames, 4, len(cm.JOINT_NAMES)), dtype=bool)
    window = np.arange(offset, offset + frames) % seen.shape[1]
    for name in names:
        if name == ROOT_NAME:
            continue
        joint = cm.JOINT_INDEX[name]
        keep[:, :, :, joint] = seen[:, window][:, :, :, joint]
    return keep


def amplified_keep(seen: np.ndarray, frames: int, names: tuple[str, ...], offset: int,
                   subjects: int) -> tuple[np.ndarray, dict]:
    """Arm (i'): every camera carries a MEASURED occlusion series, from the worst four.

    The real run's structure is two cameras that see almost everything and two
    that drop in correlated bursts, so the solve is barely exercised. This arm
    keeps every measured property of the occlusion -- the burst length
    distribution, and the 0.59 within-subject correlation between B001 and D001 --
    and changes only WHICH camera carries which series: the four most-occluded of
    the eight measured (subject, camera) series are dealt to the four cameras.
    Nothing about the outage is invented; the amplification is the assignment, and
    it is declared.
    """
    rates = seen.mean(axis=(1, 3))
    order = np.dstack(np.unravel_index(np.argsort(rates, axis=None), rates.shape))[0][:4]
    keep = np.ones((subjects, frames, 4, len(cm.JOINT_NAMES)), dtype=bool)
    window = np.arange(offset, offset + frames) % seen.shape[1]
    for camera, (subject, source_camera) in enumerate(order):
        for name in names:
            if name == ROOT_NAME:
                continue
            joint = cm.JOINT_INDEX[name]
            keep[:, :, camera, joint] = seen[subject, window, source_camera, joint][None, :]
    provenance = {CAMERAS[c]: f"measured series of subject {int(s)} camera {CAMERAS[int(k)]}"
                  for c, (s, k) in enumerate(order)}
    return keep, provenance


def iid_keep(keep_reference: np.ndarray, names: tuple[str, ...], seed: int) -> np.ndarray:
    """Arm (ii): the SAME overall per-view miss rate, independent across everything.

    The contrast arm. Matching the per-view rate rather than the single-ray rate is
    deliberate: the gap between the two arms' single-ray fractions IS the
    correlation finding, and forcing them equal would erase it.
    """
    rng = np.random.default_rng(seed)
    columns = [cm.JOINT_INDEX[n] for n in names if n != ROOT_NAME]
    rate = float(keep_reference[:, :, :, columns].mean())
    keep = np.ones_like(keep_reference)
    draw = rng.random(keep_reference[:, :, :, columns].shape) < rate
    keep[:, :, :, columns] = draw
    return keep


def mask_is_runnable(keep: np.ndarray, names: tuple[str, ...]) -> list[str]:
    """Which joints would make `_fill_and_smooth_positions` raise under this mask.

    A joint needs at least two frames it can triangulate (support >= 2) or the fill
    has fewer than two samples to interpolate between and the whole take raises --
    a live risk on short FK takes, where the real run's 40-frame miss bursts can
    cover the entire clip.
    """
    offenders = []
    for name in names:
        joint = cm.JOINT_INDEX[name]
        support = keep[:, :, :, joint].sum(axis=2)
        if int(support.min(initial=4)) >= 2:
            continue
        if (support >= 2).sum(axis=1).min() < 2:
            offenders.append(name)
    return offenders


# ---------------------------------------------------------------------------- scoring

def pair_subjects(positions: np.ndarray, truth19: np.ndarray) -> dict[int, int]:
    """our output slot -> truth subject, by median root distance. Never by index."""
    root = cm.JOINT_INDEX["root"]
    frames = min(positions.shape[1], truth19.shape[1])
    cost = np.asarray([[np.nanmedian(np.linalg.norm(
        positions[a, :frames, root] - truth19[b, :frames, root], axis=1))
        for b in range(truth19.shape[0])] for a in range(positions.shape[0])])
    if cost.shape == (2, 2) and cost[0, 0] + cost[1, 1] > cost[0, 1] + cost[1, 0]:
        return {0: 1, 1: 0}
    return {index: index for index in range(positions.shape[0])}


def error_mm(positions: np.ndarray, truth19: np.ndarray, mapping: dict[int, int],
             names: tuple[str, ...]) -> np.ndarray:
    """[our subject, frame, len(names)] millimetres against exact truth."""
    columns = [cm.JOINT_INDEX[n] for n in names]
    frames = min(positions.shape[1], truth19.shape[1])
    out = np.full((positions.shape[0], frames, len(columns)), np.nan)
    for ours, theirs in mapping.items():
        out[ours] = np.linalg.norm(positions[ours, :frames, columns].transpose(1, 0, 2)
                                   - truth19[theirs, :frames, columns].transpose(1, 0, 2),
                                   axis=2) * 1000.0
    return out


def cells_from_log(log: SequenceSolveLog, mapping: dict[int, int], names: tuple[str, ...],
                   frames: int) -> dict[str, np.ndarray]:
    """Boolean [our subject, frame, len(names)] cell sets, in the scoring layout."""
    columns = [cm.JOINT_INDEX[n] for n in names]
    shape = (len(log.entries), frames, len(columns))
    support = np.zeros(shape, dtype=np.int64)
    recovered = np.zeros(shape, dtype=bool)
    direct = np.zeros(shape, dtype=bool)
    for subject in range(len(log.entries)):
        support[subject] = log.support(subject)[:frames][:, columns]
        recovered[subject] = log.recovered(subject)[:frames][:, columns]
        direct[subject] = log.direct(subject)[:frames][:, columns]
    del mapping
    single_ray = (~direct) & (support == 1)
    return {"support": support, "direct": direct, "recovered": recovered & single_ray,
            "single_ray": single_ray, "demoted": single_ray & ~recovered,
            "no_ray": (~direct) & (support == 0)}


def by_frame(values: np.ndarray, cells: np.ndarray) -> list[np.ndarray]:
    """Per-frame lists for the block bootstrap: frames are the resampling unit."""
    return [values[:, frame][cells[:, frame]] for frame in range(values.shape[1])]


# ------------------------------------------------------------------------- 5b measures

def lag_sweep(positions: np.ndarray, truth19: np.ndarray, mapping: dict[int, int],
              names: tuple[str, ...], sweep: int = LAG_SWEEP) -> dict:
    """Phase lag by shift sweep, ON ONE FIXED FRAME SET FOR EVERY SHIFT.

    I6 hit the shifting-denominator bug on exactly this check and had to redo it:
    if the scored frames move with the shift, the curve mixes a lag with a change
    of population and the minimum lands wherever the easy frames are. The set here
    is the interior frames [sweep, N-sweep) and it does not move.

    S(k) = RMS over the fixed set of ||ours[t+k] - truth[t]||. A smoother that lags
    by L frames puts its minimum at k = +L, because ours[t+L] is then truth[t].
    The k = +/-1 entries ARE the time-shifted-truth control: both must be worse
    than k = 0 for the stage to be reporting no lag.
    """
    columns = [cm.JOINT_INDEX[n] for n in names]
    frames = min(positions.shape[1], truth19.shape[1])
    scored = np.arange(sweep, frames - sweep)
    if scored.size < 4:
        return {"note": "take too short for a lag sweep", "frames_scored": int(scored.size)}
    curve = {}
    for shift in range(-sweep, sweep + 1):
        pieces = []
        for ours, theirs in mapping.items():
            a = positions[ours][scored + shift][:, columns]
            b = truth19[theirs][scored][:, columns]
            pieces.append(np.linalg.norm(a - b, axis=2) * 1000.0)
        pooled = np.concatenate(pieces).ravel()
        pooled = pooled[np.isfinite(pooled)]
        curve[shift] = round(float(np.sqrt(np.mean(pooled ** 2))), 4)
    shifts = sorted(curve)
    values = [curve[s] for s in shifts]
    best = int(np.argmin(values))
    sub_frame = float(shifts[best])
    if 0 < best < len(values) - 1:
        left, centre, right = values[best - 1], values[best], values[best + 1]
        denominator = left - 2 * centre + right
        if abs(denominator) > 1e-12:
            sub_frame = shifts[best] + 0.5 * (left - right) / denominator
    return {
        "frames_scored": int(scored.size),
        "frame_set": [int(scored[0]), int(scored[-1]) + 1],
        "frame_set_note": "the SAME frames at every shift; the shift moves our own "
                          "trajectory, never the population",
        "rms_mm_by_shift": {str(s): curve[s] for s in shifts},
        "argmin_frames": shifts[best],
        "argmin_subframe": round(sub_frame, 3),
        "control_time_shifted_truth": {
            "minus_one_frame_rms_mm": curve.get(-1), "zero_rms_mm": curve.get(0),
            "plus_one_frame_rms_mm": curve.get(1),
            "both_shifts_worse_than_zero": bool(
                curve.get(-1, math.inf) > curve.get(0, 0.0)
                and curve.get(1, math.inf) > curve.get(0, 0.0))},
    }


def peak_attenuation(positions: np.ndarray, truth19: np.ndarray, mapping: dict[int, int],
                     names: tuple[str, ...], quantile: float = 95.0) -> dict:
    """How much of a fast event survives the stage. Speed ratio on the fastest cells."""
    columns = [cm.JOINT_INDEX[n] for n in names]
    frames = min(positions.shape[1], truth19.shape[1])
    ratios, true_speeds, our_speeds = [], [], []
    for ours, theirs in mapping.items():
        true_step = np.linalg.norm(np.diff(truth19[theirs, :frames][:, columns], axis=0), axis=2)
        our_step = np.linalg.norm(np.diff(positions[ours, :frames][:, columns], axis=0), axis=2)
        finite = np.isfinite(true_step) & np.isfinite(our_step) & (true_step > 1e-9)
        if not finite.any():
            continue
        cut = np.percentile(true_step[finite], quantile)
        fast = finite & (true_step >= cut)
        ratios.append((our_step[fast] / true_step[fast]))
        true_speeds.append(true_step[fast])
        our_speeds.append(our_step[fast])
    if not ratios:
        return {"n": 0}
    pooled = np.concatenate(ratios)
    return {
        "n": int(pooled.size),
        "fast_event_definition": f"per-frame displacement at or above the p{quantile:g} of "
                                 "the TRUE displacement, per subject",
        "true_peak_speed_m_s": round(float(np.median(np.concatenate(true_speeds))
                                           * SAMPLE_RATE_HZ), 4),
        "our_peak_speed_m_s": round(float(np.median(np.concatenate(our_speeds))
                                          * SAMPLE_RATE_HZ), 4),
        "surviving_fraction_median": round(float(np.median(pooled)), 4),
        "attenuation_median": round(float(1.0 - np.median(pooled)), 4),
    }


def recovery_latency(errors: np.ndarray, gap_cells: np.ndarray, baseline_p95: float) -> dict:
    """Frames after a gap closes before the error is back inside the no-gap p95."""
    latencies: list[int] = []
    censored = 0
    subjects, frames, joints = errors.shape
    for subject in range(subjects):
        for joint in range(joints):
            column = gap_cells[subject, :, joint]
            frame = 0
            while frame < frames:
                if not column[frame]:
                    frame += 1
                    continue
                end = frame
                while end < frames and column[end]:
                    end += 1
                latency = None
                for offset in range(0, LATENCY_CAP_FRAMES + 1):
                    probe = end + offset
                    if probe >= frames:
                        break
                    value = errors[subject, probe, joint]
                    if np.isfinite(value) and value <= baseline_p95:
                        latency = offset
                        break
                if latency is None:
                    censored += 1
                else:
                    latencies.append(latency)
                frame = end
    if not latencies:
        return {"gaps": censored, "note": "every gap censored at the cap",
                "cap_frames": LATENCY_CAP_FRAMES}
    array = np.asarray(latencies)
    return {"gaps_measured": int(array.size), "gaps_censored_at_cap": censored,
            "cap_frames": LATENCY_CAP_FRAMES,
            "threshold_mm": round(float(baseline_p95), 4),
            "threshold_note": "the no-injection arm's p95 error on the same joints",
            "median_frames": float(np.median(array)), "p95_frames": float(np.percentile(array, 95)),
            "max_frames": int(array.max())}


# ------------------------------------------------------------------------- injections

def spike_cells(truth19: np.ndarray, names: tuple[str, ...], every: int = 12,
                joints: tuple[str, ...] = ("left_wrist", "right_wrist",
                                           "left_ankle", "right_ankle")) -> list[tuple]:
    """Isolated single-frame outliers, spaced so no smoothing window sees two."""
    frames = truth19.shape[1]
    chosen = [n for n in joints if n in names]
    return [(subject, frame, cm.JOINT_INDEX[name])
            for subject in range(truth19.shape[0])
            for frame in range(every, frames - every, every)
            for name in chosen]


def spike_displacements(cameras, truth19: np.ndarray, cells: list[tuple],
                        amplitude_px: float, camera_count: int,
                        seed: int) -> tuple[dict, dict]:
    """Displace 2D by `amplitude_px`, in one camera or in two consistently.

    `camera_count == 1` is what a detector does: one view's landmark jumps. It is
    reported separately BECAUSE triangulation, not the smoother, is expected to
    absorb it -- `triangulate_point` searches every camera subset and keeps the
    largest inlier set, so a lone outlier is dropped before the temporal stage
    ever sees it. `camera_count == 2` displaces the underlying 3D point and
    projects the moved point into two views, so the outlier is geometrically
    consistent and DOES reach 3D. Both amplitudes are our own detector's, never
    MAMMA's.
    """
    rng = np.random.default_rng(seed)
    displacement: dict = {}
    reached = []
    for subject, frame, joint in cells:
        # `cells` are in the RECORD subject space, which is the trajectory's own
        # order; the pipeline's output order is resolved separately by pelvis
        # distance and never assumed to match.
        point = truth19[subject, frame, joint]
        if not np.isfinite(point).all():
            continue
        order = rng.permutation(len(cameras))[:camera_count]
        primary = cameras[int(order[0])]
        ray = point - primary.camera_center_world_m
        depth = float(np.linalg.norm(ray))
        focal = float(primary.intrinsics[0, 0])
        # a metric step whose image size in the primary camera is `amplitude_px`
        direction = rng.normal(size=3)
        direction -= direction.dot(ray) / max(depth ** 2, 1e-12) * ray
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        step = direction / norm * (amplitude_px * depth / focal)
        moved = point + step
        for index in order:
            camera = cameras[int(index)]
            before, depth_before = camera.project(point)
            after, depth_after = camera.project(moved)
            if depth_before <= 0.0 or depth_after <= 0.0:
                continue
            displacement[(subject, frame, int(index), joint)] = (float(after[0] - before[0]),
                                                                 float(after[1] - before[1]))
            reached.append(float(np.linalg.norm(after - before)))
    return displacement, {
        "cells": len(cells), "observations_displaced": len(displacement),
        "cameras_per_cell": camera_count,
        "requested_amplitude_px1280": round(amplitude_px, 3),
        "realised_median_px1280": round(float(np.median(reached)), 3) if reached else None,
        "realised_note": "the step is a metric displacement sized to `requested` pixels in the "
                         "PRIMARY camera; a second camera sees the same 3D step at its own "
                         "angle, so the realised median over both is not the requested value",
    }


def gap_cells(truth19: np.ndarray, names: tuple[str, ...], length: int) -> list[tuple]:
    """Runs of `length` frames where every camera loses the joint.

    The spacing scales with the take: a fixed 20-frame stride put zero gaps in a
    28-frame stride-3 FK clip and the arm reported nothing while looking as though
    it had run. `length + 8` leaves at least the shipped window's own half-width of
    clean frames after each gap for the latency measurement to land in.
    """
    frames = truth19.shape[1]
    every = max(length + 8, 10)
    chosen = [n for n in ("left_elbow", "right_elbow", "left_knee", "right_knee") if n in names]
    out = []
    for subject in range(truth19.shape[0]):
        for start in range(2, max(3, frames - length - 2), every):
            for name in chosen:
                for offset in range(length):
                    out.append((subject, start + offset, cm.JOINT_INDEX[name]))
    return out


def _compare_outage_models(recovery: dict) -> dict:
    """The finding the two outage models exist to produce, stated in the report.

    The plan expected interpolation-only to lose on the recovered cells in both
    models. It does not, and the split is the point: a sequence solve buys its
    keep where outages are CORRELATED and bursty -- both neighbours of a starved
    frame are starved too, so there is nothing to draw a line between -- and costs
    a little where they are isolated, because two good neighbours one frame apart
    are a better estimate than a limb-length-and-smoothness solve.
    """
    out: dict = {}
    for name in ("correlated_amplified", "iid_same_rate"):
        arm = recovery.get(name)
        if not arm or arm.get("ran") is False or not arm.get("cells", {}).get("recovered"):
            continue
        ours = arm["error_on_recovered_cells"]["median_mm"]
        interp = arm["control_interpolation_only_on_recovered_cells"]["median_mm"]
        margin = arm.get("margin_on_recovered_cells", {})
        out[name] = {
            "single_ray_fraction": arm["cells"]["single_ray_fraction"],
            "no_ray_fraction": arm["cells"]["no_ray_fraction"],
            "sequence_solve_mm": ours,
            "interpolation_only_mm": interp,
            "ratio_interpolation_over_solve": round(interp / ours, 3) if ours else None,
            "p_solve_beats_interpolation": margin.get("p_candidate_beats_control"),
            "blocks_per_draw": margin.get("blocks_per_draw"),
        }
    if len(out) == 2:
        correlated = out["correlated_amplified"]
        independent = out["iid_same_rate"]
        out["reading"] = (
            "At the same per-view miss rate, correlated outages starve "
            f"{correlated['single_ray_fraction'] * 100:.2f} % of cells of all but one ray and "
            f"leave {correlated['no_ray_fraction'] * 100:.2f} % with none, while independent "
            f"outages starve {independent['single_ray_fraction'] * 100:.2f} % and leave "
            f"{independent['no_ray_fraction'] * 100:.2f} % with none -- correlation moves the "
            "damage from isolated single frames into runs. On the correlated arm the sequence "
            f"solve beats interpolation {correlated['ratio_interpolation_over_solve']}x "
            f"(P = {correlated['p_solve_beats_interpolation']}); on the independent arm it "
            f"LOSES to it ({independent['sequence_solve_mm']} vs "
            f"{independent['interpolation_only_mm']} mm, "
            f"P = {independent['p_solve_beats_interpolation']}). Neither is a defect. A drawn "
            "line between two good neighbours one frame apart is an excellent estimate and a "
            "constrained solve cannot beat it; a drawn line across a burst has no good "
            "neighbours to draw between. The stage is worth what it costs precisely because "
            "real occlusion is bursty -- see `real_run_occlusion_pattern."
            "miss_run_length_frames` for what the reference footage's own miss runs look "
            "like.")
    return out


# ------------------------------------------------------------------------------- 5a arm

def score_recovery_arm(name: str, cameras, records, keep: np.ndarray, truth19: np.ndarray,
                       names: tuple[str, ...], reference_mapping: dict[int, int] | None,
                       verbose: bool = True) -> dict:
    """One outage arm plus its interpolation-only control on the SAME cells."""
    masked = apply_keep_mask(records, keep)
    positions, raw, diagnostics, log, seconds = run_pipeline(cameras, masked, solve="real")
    mapping = reference_mapping or pair_subjects(positions, truth19)
    frames = positions.shape[1]
    errors = error_mm(positions, truth19, mapping, names)
    cells = cells_from_log(log, mapping, names, frames)

    control_positions, _raw, control_diagnostics, control_log, control_seconds = run_pipeline(
        cameras, masked, solve="off")
    control_errors = error_mm(control_positions, truth19, mapping, names)
    del control_log

    single = cells["single_ray"]
    entry = {
        "outage_model": name,
        "seconds": round(seconds + control_seconds, 1),
        "cells": {
            "scored_joints": list(names),
            "total": int(single.size),
            "single_ray": int(single.sum()),
            "single_ray_fraction": round(float(single.mean()), 5),
            "no_ray": int(cells["no_ray"].sum()),
            "no_ray_fraction": round(float(cells["no_ray"].mean()), 5),
            "recovered": int(cells["recovered"].sum()),
            "demoted_to_interpolation": int(cells["demoted"].sum()),
            "demoted_note": f"a solved point must lie within {RECOVERY_RAY_GATE_PX} px "
                            "(at 1280) of every ray that saw it or the solve hands it back "
                            "to the fill",
        },
        "diagnostics": {
            "constraint_recovered_joint_fraction":
                diagnostics["constraint_recovered_joint_fraction"],
            "interpolated_joint_fraction": diagnostics["interpolated_joint_fraction"],
            "valid_joint_fraction": diagnostics["valid_joint_fraction"],
            "denominator_note": "these three are quoted over all 19 contract joints, so the "
                                "joints this source never populates are inside them; the "
                                "cell counts above use the scored joints only",
        },
        "control_diagnostics": {
            "constraint_recovered_joint_fraction":
                control_diagnostics["constraint_recovered_joint_fraction"],
            "interpolated_joint_fraction": control_diagnostics["interpolated_joint_fraction"],
        },
        "error_on_all_cells": summarise(errors),
        "control_interpolation_only_on_all_cells": summarise(control_errors),
    }
    for label, mask in (("recovered_cells", cells["recovered"]),
                        ("demoted_cells", cells["demoted"]),
                        ("no_ray_cells", cells["no_ray"]),
                        ("two_or_more_view_cells", cells["direct"])):
        entry[f"error_on_{label}"] = summarise(errors[mask])
        entry[f"control_interpolation_only_on_{label}"] = summarise(control_errors[mask])
    if int(cells["recovered"].sum()) >= 2:
        entry["margin_on_recovered_cells"] = block_bootstrap_pair(
            by_frame(errors, cells["recovered"]), by_frame(control_errors, cells["recovered"]))
        entry["margin_note"] = ("candidate = the sequence solve, control = interpolation only, "
                                "both on the SAME recovered cells and identical bootstrap draws")
    per_frame = np.array([np.nanmedian(errors[:, f][cells["recovered"][:, f]])
                          if cells["recovered"][:, f].any() else np.nan
                          for f in range(frames)])
    entry["lag1_autocorrelation_of_per_frame_recovered_median"] = lag1_autocorrelation(per_frame)
    if verbose:
        print(f"    {name}: single-ray {entry['cells']['single_ray']}, "
              f"recovered {entry['cells']['recovered']}, "
              f"error {entry['error_on_recovered_cells']['median_mm']} mm vs "
              f"interp-only {entry['control_interpolation_only_on_recovered_cells']['median_mm']}"
              f" mm  ({entry['seconds']}s)")
    return entry, mapping


# --------------------------------------------------------------------- held-out camera

def held_out_camera_lag(sweep: int = LAG_SWEEP) -> dict:
    """Real footage. Three cameras in, the fourth used only to score, four folds.

    The 3D under test never saw the held-out camera. The IDENTITY of each subject
    in that camera does come from the full four-camera association -- a labelling,
    not a measurement, and stated as such: without it there is no way to say which
    detection in the held-out view is which performer, and inventing a matcher
    here would be a second association loop.

    BLIND TO DEPTH. A reprojection residual along the viewing ray is free, so this
    figure can see a transverse lag and cannot see a lag along the camera's axis.
    """
    cameras = working_cameras()
    records = [cm.load_observation_jsonl(SOMA77_WORK / f"{name}-soma77-observations.jsonl")
               for name in CAMERAS]
    captured: list[np.ndarray] = []

    def recording_associator(*args, **kwargs):
        result = cm.associate_frame_graph(*args, **kwargs)
        captured.append(np.array(result[0], copy=True))
        return result

    cm.reconstruct_multiview(cameras, records, subject_count=2,
                             sample_rate_hz=SAMPLE_RATE_HZ, associator=recording_associator)
    assigned = np.stack(captured, axis=1)              # [subject, frame, camera, 19, 3]
    frames = assigned.shape[1]

    folds = {}
    for held in range(len(CAMERAS)):
        kept = [index for index in range(len(CAMERAS)) if index != held]
        try:
            positions, _raw, diagnostics, _log, seconds = run_pipeline(
                tuple(cameras[i] for i in kept), [records[i] for i in kept])
        except cm.CommercialMultiviewError as error:
            folds[CAMERAS[held]] = {"ran": False, "reason": str(error)}
            print(f"    fold {CAMERAS[held]}: DID NOT RUN -- {error}")
            continue
        camera = cameras[held]
        observed = assigned[:, :, held]                # [subject, frame, 19, 3]
        detected = (np.isfinite(observed[..., :2]).all(axis=-1)) & (observed[..., 2] >= 0.25)
        # One frame set, fixed before the sweep and identical at every shift: the
        # detection must exist AND our reprojection must be finite at every shift
        # in the range, or the curve would be comparing populations.
        usable = np.ones((positions.shape[0], frames, len(cm.JOINT_NAMES)), dtype=bool)
        for shift in range(-sweep, sweep + 1):
            shifted = np.clip(np.arange(frames) + shift, 0, frames - 1)
            usable &= detected & np.isfinite(positions[:, shifted]).all(axis=3)
        interior = np.zeros(frames, dtype=bool)
        interior[sweep:frames - sweep] = True
        usable &= interior[None, :, None]
        curve = {}
        for shift in range(-sweep, sweep + 1):
            residuals = []
            for subject in range(positions.shape[0]):
                rows, joints = np.nonzero(usable[subject])
                if rows.size == 0:
                    continue
                points = positions[subject, np.clip(rows + shift, 0, frames - 1), joints]
                projected, depth = camera.project(points)
                target = observed[subject, rows, joints, :2]
                good = depth > 0.0
                residuals.append(np.linalg.norm(projected[good] - target[good], axis=1))
            pooled = np.concatenate(residuals) if residuals else np.zeros(0)
            curve[shift] = round(float(np.median(pooled)), 4) if pooled.size else None
        shifts = sorted(curve)
        values = [curve[s] for s in shifts]
        best = int(np.argmin([v if v is not None else math.inf for v in values]))
        sub_frame = float(shifts[best])
        if 0 < best < len(values) - 1 and all(v is not None for v in values[best - 1:best + 2]):
            left, centre, right = values[best - 1], values[best], values[best + 1]
            denominator = left - 2 * centre + right
            if abs(denominator) > 1e-12:
                sub_frame = shifts[best] + 0.5 * (left - right) / denominator
        folds[CAMERAS[held]] = {
            "ran": True,
            "cameras_used": [CAMERAS[i] for i in kept],
            "seconds": round(seconds, 1),
            "scored_slots": int(usable.sum()),
            "median_reprojection_px1280_by_shift": {str(s): curve[s] for s in shifts},
            "argmin_frames": shifts[best],
            "argmin_subframe": round(sub_frame, 3),
            "interpolated_joint_fraction_of_the_3d_under_test":
                diagnostics["interpolated_joint_fraction"],
            "valid_joint_fraction_of_the_3d_under_test": diagnostics["valid_joint_fraction"],
        }
        print(f"    fold hold-out {CAMERAS[held]}: argmin {shifts[best]} frames "
              f"(sub-frame {round(sub_frame, 2)}), "
              f"{curve[shifts[best]]} px at the minimum, {int(usable.sum())} slots "
              f"({round(seconds, 1)}s)")
    return {
        "what_it_is": "our smoothed 3D from three cameras, reprojected into the fourth and "
                      "swept against that camera's own SOMA-77 detections",
        "footage": "the delivered SOMA-77 run, frames [60,210) of "
                   "pushing_and_lifting_from_ground, 2 subjects",
        "frame_set": "fixed before the sweep and identical at every shift: the held-out "
                     "detection must exist and our reprojection be finite at EVERY shift in "
                     "the range, and the frame must be interior to it",
        "identity_in_the_held_out_view": "from the full four-camera association (a labelling, "
                                         "not a measurement); the 3D under test used three "
                                         "cameras only",
        "blind_to": "DEPTH. A residual along the held-out camera's viewing ray costs nothing, "
                    "so this sees transverse lag only. It is also charged with the detector's "
                    "own 2D error, which is 2.77 px one-sided at 1280 and much larger than any "
                    "lag effect at these speeds -- read the SHAPE of the curve and its "
                    "minimum, never the absolute pixels.",
        "folds": folds,
    }


# ------------------------------------------------------------------------------ sources

def run_source(source_key: str, source_array: np.ndarray, truth19: np.ndarray,
               names: tuple[str, ...], seen: np.ndarray, reference: str, may_select: bool,
               spike_px: float, spike_source: str) -> dict:
    cameras = working_cameras()
    frames = truth19.shape[1]
    subjects = truth19.shape[0]
    print(f"\n[{source_key}] {subjects} subjects x {frames} frames, "
          f"{len(names)} joints scored")

    records = build_records(cameras, source_key, source_array)
    check_person_order(records, subjects)

    out: dict = {
        "trajectory_source": source_key,
        "reference": reference,
        "may_select_a_shipped_constant": may_select,
        "frames": int(frames), "subjects": int(subjects), "joints_scored": list(names),
        "true_joint_speed": joint_speed_m_s(truth19, names),
    }

    # ---- oracle: no drops at all. Raw triangulation must return the projections.
    positions, raw, diagnostics, log, seconds = run_pipeline(cameras, records)
    mapping = pair_subjects(positions, truth19)
    raw_errors = error_mm(raw, truth19, mapping, names)
    smoothed_errors = error_mm(positions, truth19, mapping, names)
    out["subject_pairing"] = {f"our_{k}": f"truth_{v}" for k, v in sorted(mapping.items())}
    out["oracle_no_drops"] = {
        "what_it_is": "every camera sees every joint; the trajectory that went in is the "
                      "answer. Raw triangulation must return it at machine precision, and "
                      "whatever the smoothed arm costs above that IS the temporal stage.",
        "raw_triangulation": summarise(raw_errors),
        "raw_max_micrometres": float(np.nanmax(raw_errors) * 1000.0),
        "after_the_temporal_stage": summarise(smoothed_errors),
        "recovered_cells": int(log.recovered(0).sum() + log.recovered(1).sum())
        if len(log.entries) >= 2 else None,
        "seconds": round(seconds, 1),
    }
    print(f"  oracle (no drops): raw {out['oracle_no_drops']['raw_triangulation']['median_mm']}"
          f" mm (max {out['oracle_no_drops']['raw_max_micrometres']:.3g} um), after the stage "
          f"{out['oracle_no_drops']['after_the_temporal_stage']['median_mm']} mm  ({seconds:.1f}s)")
    baseline_p95 = out["oracle_no_drops"]["after_the_temporal_stage"]["p95_mm"] or 1.0

    # ---- 5a: recovery ---------------------------------------------------------
    print("  5a recovery")
    recovery: dict = {}
    replayed = replayed_keep(seen, frames, names, offset=0)
    amplified, provenance = amplified_keep(seen, frames, names, offset=0, subjects=subjects)
    unrunnable = {"replayed_real": mask_is_runnable(replayed, names),
                  "correlated_amplified": mask_is_runnable(amplified, names)}
    iid = iid_keep(amplified, names, seed=17)
    unrunnable["iid_same_rate"] = mask_is_runnable(iid, names)

    arms = [("replayed_real", replayed), ("correlated_amplified", amplified),
            ("iid_same_rate", iid)]
    for arm_name, keep in arms:
        if unrunnable[arm_name]:
            recovery[arm_name] = {
                "ran": False,
                "reason": "under this mask these joints never reach two views on two frames, "
                          "so `_fill_and_smooth_positions` has fewer than two samples to "
                          "interpolate between and the take raises",
                "joints": unrunnable[arm_name]}
            print(f"    {arm_name}: NOT RUNNABLE -- {unrunnable[arm_name]}")
            continue
        entry, _ = score_recovery_arm(arm_name, cameras, records, keep, truth19, names, mapping)
        recovery[arm_name] = entry
    recovery["correlated_amplified"].setdefault("series_provenance", provenance)
    if recovery["replayed_real"].get("cells", {}).get("single_ray") == 0:
        recovery["replayed_real"]["why_zero"] = (
            "the real run's 6 single-ray slots do not fall in this take's frame window. The "
            "mask is replayed frame-for-frame from frame 0 and this take is shorter than 150 "
            "frames; scanning the window for an offset that contains them would be choosing "
            "the population after seeing it. The `mamma_pred_joints` source is 150 frames on "
            "the very take the mask was measured on and catches all six.")
    recovery["what_the_two_outage_models_say"] = _compare_outage_models(recovery)
    recovery["amplification_note"] = (
        "The amplified arm deals the four most-occluded of the real run's eight measured "
        "(subject, camera) visibility series onto the four cameras. Every property of the "
        "outage -- the burst length distribution, the 0.587 within-subject correlation "
        "between B001 and D001 -- is measured. Only the assignment is amplified, because "
        "replaying the pattern as it stands starves 6 slots in 5,100 and no instrument can "
        "be built on six cells.")

    # controls that must fail
    controls: dict = {}
    frozen_truth19 = np.repeat(truth19[:, :1], frames, axis=1)
    frozen_source = np.repeat(source_array[:, :1], frames, axis=1)
    frozen_records = build_records(cameras, source_key, frozen_source)
    frozen_masked = apply_keep_mask(frozen_records, amplified)
    frozen_positions, _raw, _d, frozen_log, frozen_seconds = run_pipeline(cameras, frozen_masked)
    frozen_mapping = pair_subjects(frozen_positions, frozen_truth19)
    frozen_cells = cells_from_log(frozen_log, frozen_mapping, names, frames)
    controls["frozen_trajectory"] = {
        "what_it_is": "frame 0's pose projected on every frame, under the amplified mask. It "
                      "reconstructs and recovers perfectly and agrees with the real "
                      "trajectory not at all -- the figure a degenerate solution scores.",
        "error_against_the_real_trajectory": summarise(
            error_mm(frozen_positions, truth19, frozen_mapping, names)),
        "error_against_its_own_frozen_truth": summarise(
            error_mm(frozen_positions, frozen_truth19, frozen_mapping, names)),
        "recovered_cells": int(frozen_cells["recovered"].sum()),
        "seconds": round(frozen_seconds, 1)}
    print(f"    control frozen trajectory: "
          f"{controls['frozen_trajectory']['error_against_the_real_trajectory']['median_mm']} mm "
          f"against the real trajectory, "
          f"{controls['frozen_trajectory']['error_against_its_own_frozen_truth']['median_mm']}"
          f" mm against its own")

    # root starvation: whole-frame loss, its own diagnostic
    root_keep = amplified.copy()
    root_frames = np.arange(5, frames - 5, max(7, frames // 12))
    for frame in root_frames:
        root_keep[:, frame, 1:, cm.JOINT_INDEX[ROOT_NAME]] = False
    if not mask_is_runnable(root_keep, names):
        root_positions, _raw, root_diagnostics, root_log, _s = run_pipeline(
            cameras, apply_keep_mask(records, root_keep))
        root_map = pair_subjects(root_positions, truth19)
        root_errors = error_mm(root_positions, truth19, root_map, names)
        root_cells = cells_from_log(root_log, root_map, names, frames)
        starved = np.zeros(frames, dtype=bool)
        starved[root_frames] = True
        controls["root_starved_frames"] = {
            "what_it_is": "the amplified mask plus a single-ray ROOT on "
                          f"{len(root_frames)} frames. A starved root fails "
                          "`triangulate_point`, the subject-frame is rejected wholesale, and "
                          "`retained_observations` keeps NOTHING for it -- so the sequence "
                          "solve cannot recover even a well-seen elbow on that frame. This is "
                          "whole-frame loss, not single-ray recovery, and it is why root is "
                          "excluded from every arm above.",
            "frames_starved": len(root_frames),
            "error_on_starved_frames": summarise(root_errors[:, starved]),
            "error_on_other_frames": summarise(root_errors[:, ~starved]),
            "single_ray_cells_on_starved_frames": int(root_cells["single_ray"][:, starved].sum()),
            "no_ray_cells_on_starved_frames": int(root_cells["no_ray"][:, starved].sum()),
            "interpolated_joint_fraction": root_diagnostics["interpolated_joint_fraction"]}
        print(f"    diagnostic root starvation: "
              f"{controls['root_starved_frames']['error_on_starved_frames']['median_mm']} mm on "
              f"starved frames vs "
              f"{controls['root_starved_frames']['error_on_other_frames']['median_mm']} mm "
              "elsewhere")
    out["recovery_5a"] = {"arms": recovery, "controls": controls}

    # ---- 5b: smoothing --------------------------------------------------------
    print("  5b smoothing")
    smoothing: dict = {"noise_source": spike_source,
                       "spike_amplitude_px1280": round(spike_px, 3)}

    windows = {"identity_no_savitzky_golay": WINDOW_IDENTITY,
               "shipped_window_9": WINDOW_DEFAULT,
               "over_smoothed_window_27": WINDOW_OVERSMOOTHED}

    # (c) fast events and lag, on clean full coverage: the stage alone.
    clean: dict = {}
    for label, window in windows.items():
        window_positions, _raw, _d, _log, window_seconds = run_pipeline(
            cameras, records, window=window)
        window_map = pair_subjects(window_positions, truth19)
        clean[label] = {
            "window_frames": window,
            "error": summarise(error_mm(window_positions, truth19, window_map, names)),
            "phase_lag": lag_sweep(window_positions, truth19, window_map, names),
            "peak_attenuation": peak_attenuation(window_positions, truth19, window_map, names),
            "seconds": round(window_seconds, 1)}
        print(f"    clean, {label}: error "
              f"{clean[label]['error']['median_mm']} mm, lag "
              f"{clean[label]['phase_lag'].get('argmin_subframe')} frames, attenuation "
              f"{clean[label]['peak_attenuation'].get('attenuation_median')}")
    smoothing["clean_coverage"] = clean
    smoothing["clean_coverage_note"] = (
        "No drops and no injection: the only thing between the projected trajectory and these "
        "figures is the temporal stage. `identity_no_savitzky_golay` sets the shipped window "
        "below the filter's own minimum of 5, so the fill runs and the filter does not.")

    # (a) spikes, one camera and two
    spikes: dict = {}
    cells = spike_cells(truth19, names)
    # `cells` are in the trajectory's subject order; the error matrices are in the
    # pipeline's output order. Permute once, here, rather than pairing by index.
    index_of = {cm.JOINT_INDEX[n]: i for i, n in enumerate(names)}
    touched_truth = np.zeros((subjects, frames, len(names)), dtype=bool)
    for subject, frame, joint in cells:
        if joint in index_of:
            touched_truth[subject, frame, index_of[joint]] = True
    for count in (1, 2):
        displacement, spike_meta = spike_displacements(
            cameras, truth19, cells, spike_px, count, seed=23 + count)
        spiked_records = apply_displacements(records, displacement)
        arm: dict = {"injection": spike_meta}
        for label, window in windows.items():
            spiked_positions, _raw, _d, _log, _s = run_pipeline(
                cameras, spiked_records, window=window)
            spiked_map = pair_subjects(spiked_positions, truth19)
            spiked_errors = error_mm(spiked_positions, truth19, spiked_map, names)
            touched = np.zeros_like(touched_truth)
            for ours, theirs in spiked_map.items():
                touched[ours] = touched_truth[theirs]
            arm[label] = {"window_frames": window,
                          "error_on_spiked_cells": summarise(spiked_errors[touched]),
                          "error_on_clean_cells": summarise(spiked_errors[~touched])}
        spikes[f"{count}_camera"] = arm
        print(f"    spikes in {count} camera(s): shipped window "
              f"{arm['shipped_window_9']['error_on_spiked_cells']['median_mm']} mm on spiked "
              f"cells, no filter "
              f"{arm['identity_no_savitzky_golay']['error_on_spiked_cells']['median_mm']} mm")
    smoothing["spikes"] = spikes
    smoothing["spikes_note"] = (
        "A one-camera spike is what a detector produces; a two-camera spike is what reaches "
        "3D. `triangulate_point` searches every camera subset and keeps the largest inlier "
        "set within 14 px, so with four cameras a lone outlier is dropped by TRIANGULATION "
        "before the temporal stage sees it. The two arms are reported separately so the "
        "absorbing stage is named rather than assumed.")

    # (b) gaps
    gaps: dict = {}
    for length in GAP_LENGTHS:
        cells_gap = gap_cells(truth19, names, length)
        if not cells_gap:
            gaps[f"{length}_frame"] = {"ran": False, "reason": "take too short to place a gap "
                                                              "with clean frames on both sides"}
            continue
        keep = np.ones((subjects, frames, 4, len(cm.JOINT_NAMES)), dtype=bool)
        for subject, frame, joint in cells_gap:
            if 0 <= frame < frames:
                keep[subject, frame, :, joint] = False
        if mask_is_runnable(keep, names):
            gaps[f"{length}_frame"] = {"ran": False, "reason": "gap covers the whole take"}
            continue
        gap_positions, _raw, gap_diagnostics, gap_log, _s = run_pipeline(
            cameras, apply_keep_mask(records, keep))
        gap_map = pair_subjects(gap_positions, truth19)
        gap_errors = error_mm(gap_positions, truth19, gap_map, names)
        gap_marks = cells_from_log(gap_log, gap_map, names, frames)["no_ray"]
        gaps[f"{length}_frame"] = {
            "ran": True, "cells": int(gap_marks.sum()),
            "error_inside_the_gap": summarise(gap_errors[gap_marks]),
            "error_outside_the_gap": summarise(gap_errors[~gap_marks]),
            "recovery_latency": recovery_latency(gap_errors, gap_marks, baseline_p95),
            "interpolated_joint_fraction": gap_diagnostics["interpolated_joint_fraction"]}
        print(f"    gap {length} frames: {gaps[f'{length}_frame']['cells']} cells, "
              f"{gaps[f'{length}_frame']['error_inside_the_gap']['median_mm']} mm inside, "
              f"latency {gaps[f'{length}_frame']['recovery_latency'].get('median_frames')} frames")
    smoothing["gaps"] = gaps
    smoothing["gaps_note"] = (
        "Every camera loses the joint for the run, so there is no ray and the sequence solve "
        "has nothing to recover from -- the fill draws a line and the filter smooths it. "
        "Latency is frames after the gap closes until the error is back inside the "
        "no-injection arm's p95 on the same joints.")

    # frozen control for 5b
    frozen_clean, _raw, _d, _log, _s = run_pipeline(cameras, frozen_records)
    frozen_clean_map = pair_subjects(frozen_clean, frozen_truth19)
    smoothing["control_frozen_trajectory"] = {
        "what_it_is": "frame 0 projected on every frame, full coverage. Its lag is undefined "
                      "and its peak attenuation meaningless because it has no peaks: a "
                      "constant passes every smoothness band there is.",
        "error_against_the_real_trajectory": summarise(
            error_mm(frozen_clean, truth19, frozen_clean_map, names)),
        "true_joint_speed": joint_speed_m_s(frozen_truth19, names),
        "peak_attenuation": peak_attenuation(frozen_clean, frozen_truth19,
                                             frozen_clean_map, names)}
    out["smoothing_5b"] = smoothing
    return out


# --------------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=3,
                        help="FK playback multiple; 3 is what thorax_window_sweep.py found "
                             "necessary to approach real joint speeds")
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--skip-held-out", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)

    started = time.time()
    print("recovering the real run's own camera-count pattern (recording associator) ...")
    seen = real_run_seen_mask()
    pattern = describe_real_pattern(seen)
    print(f"  support histogram over 17 emitted joints: "
          f"{pattern['camera_support_histogram_0_to_4']}")
    print(f"  single-ray slots: {pattern['single_ray_slots']} "
          f"({pattern['single_ray_fraction'] * 100:.3f} %)")

    # our own detector's amplitude, from our own instrument. Never mamma_residuals.py.
    self_agreement = json.loads(SELF_AGREEMENT.read_text())
    quantiles = self_agreement["arms"]["ours"]["one_sided_quantiles_px1280"]
    spike_px = float(quantiles["99"])
    spike_source = (
        "our own detector's cross-view self-agreement, reference-free: the p99 of the "
        f"one-sided epipolar distance, {spike_px} px at 1280 "
        "(artifacts/compare/detector-self-agreement.json, I3 report 1). Its median is "
        f"{quantiles['50']} px and its p95 {quantiles['95']} px, so an injected spike at the "
        "p99 is an outlier by our detector's own distribution. `mamma_residuals.py` is "
        "deliberately not consulted: a spike amplitude taken from MAMMA would make the "
        "MAMMA-free arm MAMMA-derived.")
    print(f"spike amplitude: {spike_px} px at 1280 (our own detector's p99)")

    sources: dict = {}

    # ---- source 1: FK synthetic. MAMMA-FREE. -----------------------------------
    clips = (fx.FULL_BODY_CLIPS[0], fx.FULL_BODY_CLIPS[2])
    fk_soma, fk_truth19, fk_names = fk_trajectory(args.stride, clips)
    sources["fk_synthetic"] = run_source(
        "fk_synthetic", fk_soma, fk_truth19, fk_names, seen,
        reference=("exact forward-kinematic truth from "
                   "scripts/build_synthetic_truth_fixture.py's own take builder, played at "
                   f"stride {args.stride}. MAMMA-FREE: no MAMMA output enters the trajectory, "
                   "the noise amplitude or any figure on this arm. This is the only arm in "
                   "this report that could ever select a shipped constant, and it selects "
                   "nothing today."),
        may_select=True, spike_px=spike_px, spike_source=spike_source)
    sources["fk_synthetic"]["clips"] = list(clips)
    sources["fk_synthetic"]["stride"] = args.stride

    # ---- source 2: MAMMA pred_joints. REPORTS, NEVER SELECTS. ------------------
    mamma_source, mamma_truth19, mamma_names = mamma_trajectory()
    sources["mamma_pred_joints"] = run_source(
        "mamma_pred_joints", mamma_source, mamma_truth19, mamma_names, seen,
        reference=("MAMMA's pred_joints used as a TRAJECTORY: exact truth once projected, at "
                   "the real take's own speed and on the very 150 frames the real occlusion "
                   "mask was measured on, so the replayed arm aligns frame for frame. "
                   "MAMMA-DERIVED -- this arm reports and never selects, and its figures may "
                   "not share an axis with the fk_synthetic arm's."),
        may_select=False, spike_px=spike_px, spike_source=spike_source)

    held_out = {"ran": False, "reason": "skipped by --skip-held-out"}
    if not args.skip_held_out:
        print("\nheld-out-camera lag on real footage")
        held_out = held_out_camera_lag()

    report = {
        "instrument": "I7 temporal -- recovery and smoothing",
        "generated_by": "tools/compare/temporal.py",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "regenerate": ".venv/bin/python tools/compare/temporal.py",
        "rung": "5 (temporal) of docs/SUBSTITUTION_LADDER.md",
        "seconds": round(time.time() - started, 1),
        "stage_under_test": (
            "`solve_sequence_positions` (single-ray recovery under limb-length and temporal "
            "constraints), then `_fill_and_smooth_positions` (linear fill + Savitzky-Golay, "
            f"window {WINDOW_DEFAULT}, polyorder 2). Both reached through the real "
            "`reconstruct_multiview`; nothing here re-implements either."),
        "two_sources_never_on_one_axis": (
            "`fk_synthetic` is MAMMA-free and could select a constant; `mamma_pred_joints` is "
            "MAMMA-derived and reports only. They measure the same stage on different motion "
            "and their millimetres are not comparable -- different joint sets (17 vs 15), "
            "different speeds, different takes. Two charts, never two bars."),
        "real_run_occlusion_pattern": pattern,
        "noise_source": spike_source,
        "sources": sources,
        "held_out_camera_lag": held_out,
        "statistics": {
            "moving_block_bootstrap": f"block {BLOCK_FRAMES} frames (0.5 s), "
                                      f"{BOOTSTRAP_DRAWS} draws, candidate and control on "
                                      "IDENTICAL draws; the block shrinks to half the take on "
                                      "short FK clips and the block count is reported so a "
                                      "thin interval can be read as thin",
            "lag1_note": "every arm quotes the lag-1 autocorrelation of its OWN per-frame "
                         "residual series; CLAUDE.md's 0.99 belongs to another lane's series "
                         "and is not assumed here",
        },
        "inputs": {
            "camera_rig": {"path": str(RIG.relative_to(ROOT)), "sha256": sha256_of(RIG)},
            "soma77_observations": {
                name: sha256_of(SOMA77_WORK / f"{name}-soma77-observations.jsonl")
                for name in CAMERAS},
            "detector_self_agreement": sha256_of(SELF_AGREEMENT),
            "mamma_pred_joints": {
                f"body_id-{b:02d}": sha256_of(o2.MA3D / f"verts_joints_body_id-{b:02d}.npz")
                for b in (0, 1)},
            "fk_clips": {clip: sha256_of(ROOT / fx.MOTION_ROOT / clip / "soma_motion.npz")
                         for clip in clips},
            "script": sha256_of(Path(__file__)),
            "commercial_multiview": sha256_of(ROOT / "src/autoanim_gnm/commercial_multiview.py"),
        },
        "blind_to": (
            "DEPTH, twice over and in two different ways. The synthetic halves are blind to "
            "it because the trajectory is exact and the cameras are the rig's four: a "
            "recovered point that sits along a shared viewing ray is cheap for the solve to "
            "find, so single-ray recovery is flattered wherever the surviving camera happens "
            "to look down the direction the joint moved. The held-out arm is blind to it "
            "because a reprojection residual along the held-out camera's ray costs nothing -- "
            "it sees transverse lag only, and cannot see a lag along that camera's axis. "
            "Beyond depth: there are no images anywhere in this report, so nothing here "
            "measures detector error, calibration error, lens distortion, sync error or "
            "soft-tissue artefact, and NO joint-definition error at all, which is the term "
            "that dominates every real-footage figure in this lane. The outages are injected "
            "and therefore have the structure this instrument gave them; the amplified arm "
            "borrows the real run's measured burst statistics but deals them to cameras that "
            "did not have them. The injected spikes are ISOLATED and the injected noise is "
            "otherwise absent -- real detector error is correlated across joints, frames and "
            "cameras, and correlated error is exactly what a temporal filter cannot remove, "
            "so every window here is flattered equally and the SHAPE of the window curve is "
            "the claim, not its absolute millimetres. `root` is never starved in the recovery "
            "arms because starving it is whole-frame loss rather than single-ray recovery; "
            "the separate root diagnostic reports that regime and no recovery figure covers "
            "it. And the smoother is a knob acting on precisely the quantity the lag and "
            "attenuation bands measure, so those bands discriminate the CONTROLS -- they say "
            "what a 3x window costs and what no window costs, and they cannot certify the "
            "shipped window against anything the shipped window is not. One more thing this "
            "instrument cannot see: the solve's own 14 px ray gate, which hands a recovered "
            "point that does not lie near its ray back to the fill. The 2D here is exact, so "
            "the gate passes every candidate and `demoted_to_interpolation` is 0 on every arm "
            "-- that path is unexercised by construction and nothing here says what it does "
            "with a real detector's rays."),
        "plan_corrections": [
            pattern["plan_correction"],
            "The brief said to score `constraint_recovered_joint_fraction` per arm as a "
            "recovery figure. It is quoted over all 19 contract joints including the ones the "
            "source never populates, so it is a fraction of the wrong denominator -- the same "
            "defect BODY_LANE_PLAN.md already recorded for the delivered run ('6 of ~20 "
            "eligible slots, not 6 of 5,700'). Both the diagnostic and the per-cell counts "
            "are in every arm, and the per-cell counts are the figures.",
            "The pipeline surfaces only the MEAN of the sequence solve's `recovered` mask, so "
            "'score the recovered cells only' is not reachable through the public return. It "
            "is reached here by a recording wrapper around the real `solve_sequence_positions` "
            "-- the same wrap-do-not-re-implement pattern the associator uses -- and the "
            "interpolation-only control is that wrapper with a pass-through body.",
            "The brief's 'recovery disabled / interpolation only' and 'identity / no "
            "smoothing' controls both name stages inside `reconstruct_multiview` with no "
            "caller-facing switch. Neither is re-implemented: the first is the pass-through "
            "wrapper, the second is `SMOOTHING_WINDOW_FRAMES` set below the filter's own "
            "minimum of 5, which makes `_fill_and_smooth_positions` fill and not filter.",
            "A single-camera spike at our detector's p99 is absorbed by TRIANGULATION, not by "
            "the smoother: `triangulate_point` searches every camera subset and keeps the "
            "largest inlier set within 14 px. The brief expected the identity-smoothing "
            "control to show spikes surviving; on four cameras it shows them never arriving. "
            "The two-camera arm was added so there is an outlier that reaches 3D at all.",
            "A starved ROOT is whole-frame loss, not single-ray recovery: `triangulate_point` "
            "returns None, the subject-frame is rejected, and `retained_observations` keeps "
            "nothing for it -- so a single ray on a well-seen elbow in that frame is discarded "
            "too and the sequence solve cannot recover it. Root is excluded from the recovery "
            "arms and given its own diagnostic.",
            "Real occlusion bursts run to 40 frames (p90 = 28) while stride-3 FK takes are "
            "28-32 frames, so a replayed mask can cover a joint for an entire short take and "
            "`_fill_and_smooth_positions` raises rather than degrading. Every mask is checked "
            "for that before it is run and an arm that cannot run says so with the joint "
            "named, rather than being quietly stubbed or having its stride chosen to avoid it.",
            "The brief expected interpolation-only to be worse than the sequence solve on the "
            "recovered cells in BOTH outage models. It is not, and the split is the "
            "instrument's main result: the solve wins by 6.7x on correlated bursts and loses "
            "to a straight line on isolated single-frame drops, robustly in both directions on "
            "identical block-bootstrap draws. The control is therefore checked on the "
            "correlated arm -- the regime the stage exists for and the regime real occlusion "
            "is in -- and the inversion is reported in "
            "`sources.*.recovery_5a.arms.what_the_two_outage_models_say`, never suppressed.",
            "The brief expected an over-smoothed window to blow up LAG and attenuation. "
            "Savitzky-Golay at polyorder 2 with `mode='interp'` is a symmetric, zero-phase "
            "filter: tripling the window to 27 moved the measured lag by less than a twentieth "
            "of a frame on both sources while peak attenuation went from 0.32 to 0.69 "
            "(fk_synthetic) and 0.01 to 0.16 (mamma_pred_joints), and error rose 3.7x and "
            "11.9x. This stage does not delay the motion; it flattens it. A lag band on this "
            "filter discriminates nothing, and the attenuation and error bands are what the "
            "window is answerable to.",
            "A single-camera spike at our detector's p99 is NOT fully absorbed by "
            "triangulation. `triangulate_point` needs two inliers to reject anything, so on a "
            "cell only two or three cameras see, a lone outlier is inside the minimum subset "
            "and reaches 3D. It is suppressed where four cameras see the joint and survives "
            "where fewer do, which is why both the one- and two-camera arms are reported.",
            "The sequence solve's benefit does not stop at the cells it recovers. On the "
            "correlated arm, the cells with NO ray at all -- which the solve cannot touch and "
            "the fill must interpolate -- are also far better in the solve arm than in the "
            "interpolation-only arm (61.7 vs 159.4 mm on mamma_pred_joints), because the "
            "recovered single-ray cells become the anchors the fill draws between. Recovery "
            "and fill are not separable by cell, and any figure that scores only the recovered "
            "cells understates the stage.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return verdict(report)


def verdict(report: dict) -> int:
    """The no-drop oracle must be exact, and every control must fail as documented."""
    problems: list[str] = []
    for key, source in report["sources"].items():
        oracle = source["oracle_no_drops"]
        if oracle["raw_max_micrometres"] > 1.0:
            problems.append(f"{key}: no-drop raw triangulation is not exact "
                            f"({oracle['raw_max_micrometres']:.3g} um)")
        arms = source["recovery_5a"]["arms"]
        # The plan's control is "interpolation only must be worse on the recovered
        # cells". It is checked on the CORRELATED arm, which is the regime the stage
        # exists for and the regime real occlusion is in. On the independent arm the
        # inequality reverses, robustly, and that is reported as a finding rather
        # than asserted as a control failure -- see `what_the_two_outage_models_say`.
        arm = arms.get("correlated_amplified")
        if arm and arm.get("ran") is not False and arm["cells"]["recovered"]:
            ours = arm["error_on_recovered_cells"]["median_mm"]
            interp = arm["control_interpolation_only_on_recovered_cells"]["median_mm"]
            if ours is not None and interp is not None and interp <= ours:
                problems.append(f"{key}/correlated_amplified: interpolation-only did NOT lose "
                                f"on the recovered cells ({interp} vs {ours} mm)")
        frozen = source["recovery_5a"]["controls"]["frozen_trajectory"]
        if (frozen["error_against_the_real_trajectory"]["median_mm"] or 0) <= 10 * (
                source["oracle_no_drops"]["after_the_temporal_stage"]["median_mm"] or 1):
            problems.append(f"{key}: frozen trajectory did not fail by 10x")
        clean = source["smoothing_5b"]["clean_coverage"]
        over = clean["over_smoothed_window_27"]
        shipped = clean["shipped_window_9"]
        if (over["error"]["median_mm"] or 0) <= (shipped["error"]["median_mm"] or 0):
            problems.append(f"{key}: the 3x window did not cost more error than the shipped one")
        if (over["peak_attenuation"].get("attenuation_median") or 0) <= (
                shipped["peak_attenuation"].get("attenuation_median") or 0):
            problems.append(f"{key}: the 3x window did not attenuate peaks more")
        if not shipped["phase_lag"].get("control_time_shifted_truth", {}).get(
                "both_shifts_worse_than_zero"):
            problems.append(f"{key}: time-shifted truth was not worse than unshifted")
    if problems:
        print("\nCHECKS FAILED:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("\nno-drop oracle exact; every control failed in the documented way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
