#!/usr/bin/env python3
"""D2c's gate: the clavicle's temporal defect, selected on synthetic truth.

THE DEFECT. D2 aims the clavicle from the rig's own Shoulder origin and D2b puts the root
on the captured hips. Both are right, and both SHORTEN the lever from the pivot to the
shoulder landmark -- ~400 mm under the old torso-axis anchor, 100-160 mm median after,
13-39 mm at its worst. Landmark wander of 21-35 mm across that lever is angle, and an
excursion that carries the landmark near the pivot flips the direction outright: single
steps of 139 and 164 deg on the delivered take, four to five thousand degrees a second.

THE CHANGE. `_reachable_clavicle_sequence` -- a per-frame REACHABILITY reject on the
clavicle's LOCAL rotation with an envelope that grows with elapsed frames, a slerp across
each rejected run, and a RE-SOLVE of everything below the clavicle. One constant arrives,
`CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S = 800.0`, which is the head lane's own physiology
and is registered in `tools/compare/provenance.py` as ANATOMY.

WHAT IS BANDED AND WHAT IS NOT. Frames-over-ceiling is exactly what a reject zeroes, so it
is REPORTED and never banded (see P2). The bands are things the mechanism cannot optimise
directly: exactness on clean synthetic input, error against synthetic truth, attenuation of
true clavicle peaks, phase lag, and the delivered arms it does not touch.

THE SELECTOR IS SYNTHETIC TRUTH. I7's `fk_trajectory` through the real record builder, the
real `reconstruct_multiview` and the real converter, with I8's own noise definitions.
Nothing here is selected on MAMMA or on the real take; MAMMA appears once, as an oracle
that reports.

EVERY "BEFORE" ARM IS THE PASS-THROUGH RULE, not a separate code path. That is exact:
with `_reachable_clavicle_sequence` swapped for a pass-through the converter reproduces
the pre-D2c code BIT-FOR-BIT on both performers and both rigs, checked against
`artifacts/compare/d2-clavicle/d2b-baseline-tracks.npz`. To regenerate that baseline:
`git show 0a2979a:src/autoanim_gnm/commercial_multiview.py > /tmp/cm.py`, put it in place
of the current file, and run `rc.solve` on each subject's
`triangulated_world_positions_z_up_m` on the canonical and the sized rig.

THREE REFERENCES AND THEY NEVER SHARE AN AXIS: synthetic truth (the converter's own output
on noise-free landmarks, degrees); our own capture (root-relative millimetres); MAMMA
(`pred_joints`, agreement with an instrument, and a 2-DoF DIRECTION step where ours is a
3-DoF rotation step).

Pre-registration: docs/reviews/clavicle-origin-2026-09-02.md section 15, committed at
3760c2e BEFORE the mechanism existed.

Run:  PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2c_clavicle_temporal_gate.py
Writes: artifacts/compare/d2-clavicle/gate-d2c.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "tools/swap-harness", "scripts",
                  "workers/commercial_multiview"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src."
    )

from autoanim_gnm import commercial_multiview as cm  # noqa: E402
from autoanim_gnm.body import DETAILED_HUMANOID  # noqa: E402
import retarget_cost as rc  # noqa: E402
import subject_map  # noqa: E402
import temporal as T  # noqa: E402
import build_synthetic_truth_fixture as fx  # noqa: E402
from sized_skeleton import sized_skeleton  # noqa: E402
from d2_clavicle_gate import (  # noqa: E402  -- imported, never copied
    PHYSICAL_CEILING_DEG_PER_FRAME,
    block_bootstrap,
    groups,
    rung11_block,
    scoreboard_errors,
    step_angles_deg,
)

subject_map.MA3D = rc.MA3D

OUT_DIR = ROOT / "artifacts/compare/d2-clavicle"
REPORT = OUT_DIR / "gate-d2c.json"
D2B_BUILD = OUT_DIR / "delivery-root"          # the BEFORE arm on the delivery path
D2C_BUILD = OUT_DIR / "delivery-d2c"           # the AFTER arm
TRACKS = ROOT / "artifacts/commercial-multiview-soma77"
D2_GATE = OUT_DIR / "gate.json"                # D2 + D2b's committed figures
BASELINE = OUT_DIR / "d2b-baseline-tracks.npz"

SAMPLE_RATE_HZ = 30
CEILING = cm.CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S / SAMPLE_RATE_HZ
CLAVICLES = ("LeftShoulder", "RightShoulder")
CLAVICLE_CHAIN = ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                  "LeftLowerArm", "RightLowerArm")
DOWNSTREAM = CLAVICLE_CHAIN + ("LeftHand", "RightHand")
SEEDS = 5
STRIDE = 3
LAG_SWEEP = 3
LOW_CEILING_DEG_PER_FRAME = 1.0        # BELOW the fixture's own true clavicle peak (1.38)
OVERSMOOTHER_WINDOW = 3 * cm.SMOOTHING_WINDOW_FRAMES     # 27, I7's own "3x the default"

REF_SYNTHETIC = ("exact synthetic truth: the converter's OWN output on the noise-free "
                 "landmarks of I7's FK fixture. MAMMA-FREE. Degrees of clavicle LOCAL "
                 "rotation; never a millimetre and never on an axis with one")
REF_OURS = ("our own triangulated capture, root-relative (hip midpoint) -- the converter "
            "scored against its OWN input, ONE solve, the delivered configuration")
REF_ROUNDTRIP = ("a canonical-proportioned body BY CONSTRUCTION -- the converter scored "
                 "against its OWN output")
REF_MAMMA = ("MAMMA `pred_joints`, agreement with an instrument and not accuracy. The "
             "oracle's step is a 2-DoF DIRECTION step; ours is a 3-DoF rotation step "
             "including twist. They must never share an axis")
REF_PHYSICAL = ("the rig's own joint angles against human physiology -- a peak joint rate "
                "near 800 deg/s, 26.67 deg per frame at 30 fps. No reference fitter enters")

PREREGISTERED = {
    "committed": "3760c2e, docs/reviews/clavicle-origin-2026-09-02.md section 15",
    "P1": "real take: frames over the ceiling and the max step FALL; the MEDIAN step does "
          "not (it is the short lever, which is D5's)",
    "P2": "after D2c the two Shoulders read ZERO over the ceiling BY CONSTRUCTION. "
          "Arithmetic, not evidence. Residual chain counts are UpperArm/LowerArm",
    "P3": "accepted frames are bit-identical to D2b; only rejected runs and their "
          "downstream chains change",
    "P4": "no rejected run exceeds 6 frames and a bad frame 0 recovers within 7 -- the "
          "envelope reaches 180 deg at ceil(180/26.67) = 7. Theorems, measured beside",
    "P5": "delivered arms and arm B not regressed beyond the paired moving-block bootstrap",
    "P6": "the round trip CANNOT reproduce D2b to 0.000 mm (D2c moves pass 1 on rejected "
          "frames, so pass 2's input moves). Banded instead: pass 2's reject fires on ZERO "
          "frames and pass 2 is bit-identical with the reject on and off",
    "P7": "the D2 and D2b figures still reproduce their committed report through the "
          "pass-through rule",
    "P8": "clean synthetic: zero rejects, bit-identical track, attenuation exactly 0",
    "P9": "noisy synthetic: max better, median not worse (0.05 deg). p95 better is the "
          "MARGINAL band and is banded as the brief asks",
    "P10": "attenuation: exactly 0 on clean input; on the noisy arm no worse than D2b on "
           "IDENTICAL draws. I7's 0.3183 is 3D displacement over 17 joints and is reported "
           "beside ours for shape only, never on one axis",
    "P11": "lag zero, shift sweep on one fixed frame set",
    "P12": "the over-smoother wins on max and LOSES on attenuation. It cannot lose on lag: "
           "zero-phase means lag zero by construction",
    "P13": "the brief's 5 deg/frame low ceiling is INERT here (true peak 1.38 deg/frame). "
           "1.0 deg/frame is substituted and fires on TRUE frames of the clean fixture",
    "P14": "the world-measured variant rejects honest motion on a turning body: neither "
           "fires at 12 deg/frame, the world one fires at 45 deg/frame and the local one "
           "still does not",
    "P15": "the step test accepts the plateau of a ~100 deg multi-frame excursion that "
           "reachability rejects whole",
}


# --------------------------------------------------------------- the swappable rules
@contextlib.contextmanager
def rule(replacement):
    """`with rule(fn):` -- swap the module attribute the converter calls by name.

    Exactly as `d2_clavicle_gate.origin` swaps `_joint_origin`: every control runs through
    the identical call site rather than a re-implementation of the converter.
    """
    saved = cm._reachable_clavicle_sequence
    cm._reachable_clavicle_sequence = replacement
    try:
        yield
    finally:
        cm._reachable_clavicle_sequence = saved


SHIPPED = cm._reachable_clavicle_sequence


def passthrough_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """No reject. Bit-identical to the pre-D2c converter (see the header)."""
    rotations = np.asarray(local_rotations, dtype=np.float64)
    return rotations.copy(), np.ones(len(rotations), dtype=bool)


def low_ceiling_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """The same rule at a ceiling BELOW the fixture's own true clavicle peak. Must fail."""
    return SHIPPED(local_rotations, parent_world_rotations, LOW_CEILING_DEG_PER_FRAME)


def step_test_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """Accept unless the SINGLE step exceeds the ceiling -- and take the wrong plateau."""
    rotations = np.asarray(local_rotations, dtype=np.float64)
    accepted = np.ones(len(rotations), dtype=bool)
    for frame in range(1, len(rotations)):
        accepted[frame] = (cm._quaternion_travel_deg(rotations[frame - 1], rotations[frame])
                           <= ceiling_deg_per_frame)
    output = rotations.copy()
    indices = np.flatnonzero(accepted)
    if len(indices):
        for start, stop in zip(indices, indices[1:]):
            for frame in range(start + 1, stop):
                output[frame] = cm._slerp(output[start], output[stop],
                                          (frame - start) / (stop - start))
        for frame in range(int(indices[-1]) + 1, len(rotations)):
            output[frame] = output[int(indices[-1])]
    return output, accepted


def world_measured_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """The same reject read in WORLD. The head lane's lesson, as a control."""
    local = np.asarray(local_rotations, dtype=np.float64)
    parent = np.asarray(parent_world_rotations, dtype=np.float64)
    world = np.stack([cm._quaternion_multiply(parent[f], local[f]) for f in range(len(local))])
    replaced_world, accepted = SHIPPED(world, parent, ceiling_deg_per_frame)
    output = np.stack([
        cm._quaternion_multiply(cm._quat_inverse(parent[f]), replaced_world[f])
        for f in range(len(local))])
    output /= np.linalg.norm(output, axis=1, keepdims=True)
    return output, accepted


def oversmoother_rule(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
    """A zero-phase low-pass of the clavicle locals at 3x I7's shipped window. Must fail.

    Savitzky-Golay at window 27, polyorder 2 -- the same filter and order
    `_fill_and_smooth_positions` runs on positions -- on SIGN-CONTINUOUS quaternion
    components, then renormalised. It rejects nothing, so its reject columns read zero and
    that is "not applicable", never a pass.
    """
    rotations = np.asarray(local_rotations, dtype=np.float64).copy()
    for frame in range(1, len(rotations)):
        if float(np.dot(rotations[frame], rotations[frame - 1])) < 0.0:
            rotations[frame] *= -1.0
    window = min(OVERSMOOTHER_WINDOW,
                 len(rotations) if len(rotations) % 2 else len(rotations) - 1)
    if window >= 5:
        rotations = savgol_filter(rotations, window_length=window, polyorder=2, axis=0,
                                  mode="interp")
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    # `accepted` means "kept exactly as solved", and a filter keeps nothing: every frame is
    # replaced by its filtered value. Its reject columns therefore read "every frame" and
    # are NOT a reject rate -- reporting them as one would be nonsense.
    return rotations, np.zeros(len(rotations), dtype=bool)


class Recorder:
    """Wrap a rule and keep every accepted mask it produced, in call order."""

    def __init__(self, inner):
        self.inner = inner
        self.masks: list[np.ndarray] = []

    def __call__(self, local_rotations, parent_world_rotations, ceiling_deg_per_frame):
        replaced, accepted = self.inner(local_rotations, parent_world_rotations,
                                        ceiling_deg_per_frame)
        self.masks.append(np.asarray(accepted, dtype=bool).copy())
        return replaced, accepted

    def rejected_runs(self) -> list[int]:
        runs: list[int] = []
        for mask in self.masks:
            length = 0
            for value in mask:
                if value:
                    if length:
                        runs.append(length)
                    length = 0
                else:
                    length += 1
            if length:
                runs.append(length)
        return runs

    def summary(self) -> dict:
        runs = self.rejected_runs()
        rejected = int(sum(int((~m).sum()) for m in self.masks))
        total = int(sum(m.size for m in self.masks))
        return {
            "sequences": len(self.masks),
            "frames": total,
            "rejected_frames": rejected,
            "rejected_fraction": round(rejected / total, 5) if total else 0.0,
            "runs": len(runs),
            "run_length_max": int(max(runs)) if runs else 0,
            "run_length_median": float(np.median(runs)) if runs else 0.0,
            "run_length_histogram": {str(k): int(v) for k, v in
                                     zip(*np.unique(runs, return_counts=True))} if runs else {},
        }


RULES = {
    "d2b_passthrough": passthrough_rule,
    "d2c_shipped": SHIPPED,
    "ctrl_oversmoother": oversmoother_rule,
    "ctrl_low_ceiling": low_ceiling_rule,
    "ctrl_step_test": step_test_rule,
    "ctrl_world_measured": world_measured_rule,
}


# ------------------------------------------------------------------------- small helpers
def clavicle_locals(track, name) -> np.ndarray:
    index = DETAILED_HUMANOID.index(name)
    return np.asarray(track.local_rotations_xyzw, dtype=np.float64)[:, index]


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(np.sum(np.asarray(a) * np.asarray(b), axis=-1)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def stats_deg(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median_deg": round(float(np.median(v)), 4),
            "p95_deg": round(float(np.percentile(v, 95)), 4),
            "max_deg": round(float(v.max()), 4)}


def solve_with(positions_z_up, replacement, skeleton=None):
    recorder = Recorder(replacement)
    with rule(recorder):
        track = rc.solve(positions_z_up, skeleton)
    return track, recorder


# ------------------------------------------------------------- 1. synthetic truth block
def attenuation_and_lag(ours: np.ndarray, truth: np.ndarray) -> dict:
    """Peak attenuation and phase lag of one clavicle's LOCAL rotation, I7's method.

    Copied in METHOD, not in code: I7's `peak_attenuation` and `lag_sweep` act on 3D joint
    displacement and this acts on angular speed, so the two figures are different
    quantities and never share an axis. Fast events are I7's definition -- the frames whose
    TRUE step is at or above the p95 of the true step -- and the lag sweep scores the SAME
    fixed frame set at every shift, so the shift moves our own series and never the
    population.
    """
    true_step = angle_deg(truth[:-1], truth[1:])
    our_step = angle_deg(ours[:-1], ours[1:])
    threshold = float(np.percentile(true_step, 95))
    fast = true_step >= threshold
    surviving = our_step[fast] / np.maximum(true_step[fast], 1e-9)

    frames = np.arange(LAG_SWEEP, len(truth) - LAG_SWEEP)
    rms = {}
    for shift in range(-LAG_SWEEP, LAG_SWEEP + 1):
        errors = angle_deg(ours[frames + shift], truth[frames])
        rms[str(shift)] = round(float(np.sqrt(np.mean(errors ** 2))), 4)
    argmin = int(min(rms, key=lambda k: rms[k]))
    return {
        "fast_events": int(fast.sum()),
        "true_peak_step_deg": round(float(true_step.max()), 4),
        "our_peak_step_deg": round(float(our_step.max()), 4),
        "surviving_fraction_median": round(float(np.median(surviving)), 4),
        "attenuation_median": round(float(1.0 - np.median(surviving)), 4),
        "rms_deg_by_shift": rms,
        "argmin_frames": argmin,
        "frames_scored": int(len(frames)),
    }


def score_arm(positions, mapping, truth_by_trajectory, replacement) -> dict:
    """One rule, one set of positions. Errors, attenuation, lag, rejects -- pooled."""
    errors, attenuations, lags, per_series = [], [], [], []
    series: dict = {}
    over_ceiling, steps_total = 0, 0
    chain = []
    recorder = Recorder(replacement)
    for ours in range(positions.shape[0]):
        with rule(recorder):
            track = rc.solve(positions[ours])
        truth_track = truth_by_trajectory[mapping[ours]]
        for name in CLAVICLES:
            ours_q = clavicle_locals(track, name)
            truth_q = clavicle_locals(truth_track, name)
            error = angle_deg(ours_q, truth_q)
            errors.append(error)
            measures = attenuation_and_lag(ours_q, truth_q)
            attenuations.append(measures["attenuation_median"])
            lags.append(measures["argmin_frames"])
            per_series.append({"subject": ours, "joint": name, **stats_deg(error),
                               **{k: measures[k] for k in
                                  ("attenuation_median", "argmin_frames",
                                   "true_peak_step_deg", "our_peak_step_deg")}})
            series[(ours, name)] = error
            step = angle_deg(ours_q[:-1], ours_q[1:])
            over_ceiling += int((step > PHYSICAL_CEILING_DEG_PER_FRAME).sum())
            steps_total += int(step.size)
        chain.append(chain_over_ceiling(track))
    pooled = np.concatenate(errors) if errors else np.asarray([])
    return {
        "error_vs_truth": stats_deg(pooled),
        "attenuation_median": round(float(np.median(attenuations)), 4),
        "lag_frames_median": float(np.median(lags)),
        "lag_frames_all_zero": bool(all(v == 0 for v in lags)),
        "shoulder_steps_over_ceiling": over_ceiling,
        "shoulder_steps": steps_total,
        "rejects": recorder.summary(),
        "per_series": per_series,
        "chain_over_ceiling": sum(row["over"] for row in chain),
        "chain_steps": sum(row["steps"] for row in chain),
        "chain_max_step_deg": round(max(row["max_deg"] for row in chain), 2) if chain else 0.0,
        "_pooled": pooled,
        "_series": series,
        "_masks": {key: mask for key, mask in zip(series, recorder.masks)}
        if len(recorder.masks) == len(series) else {},
    }


def chain_over_ceiling(track) -> dict:
    q = np.asarray(track.local_rotations_xyzw, dtype=np.float64)
    over, total, worst = 0, 0, 0.0
    per = {}
    for name in CLAVICLE_CHAIN:
        step = step_angles_deg(q[:, DETAILED_HUMANOID.index(name)])
        per[name] = {"over": int((step > PHYSICAL_CEILING_DEG_PER_FRAME).sum()),
                     "n": int(step.size), "max_deg": round(float(step.max()), 2),
                     "median_deg": round(float(np.median(step)), 2)}
        over += per[name]["over"]
        total += per[name]["n"]
        worst = max(worst, float(step.max()))
    return {"over": over, "steps": total, "fraction": round(over / total, 4),
            "max_deg": round(worst, 2), "per_joint": per}


def yawed(positions_z_up: np.ndarray, degrees_per_frame: float) -> np.ndarray:
    """Spin the whole landmark cloud about its own vertical, in place, per frame.

    Capture is Z-up, so the yaw is about +Z. The clavicle's LOCAL rotation is exactly
    invariant to this -- which is the point of the control.
    """
    source = np.asarray(positions_z_up, dtype=np.float64)
    centre = source[:, cm.JOINT_INDEX["root"], :].copy()
    out = source.copy()
    for frame in range(len(source)):
        angle = math.radians(degrees_per_frame * frame)
        cos, sin = math.cos(angle), math.sin(angle)
        matrix = np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        out[frame] = (source[frame] - centre[frame]) @ matrix.T + centre[frame]
    return out


def synthetic_block(seeds: int = SEEDS, stride: int = STRIDE) -> dict:
    cameras = T.working_cameras()
    clips = (fx.FULL_BODY_CLIPS[0], fx.FULL_BODY_CLIPS[2])
    soma, truth19, names = T.fk_trajectory(stride, clips)
    records = T.build_records(cameras, "fk_synthetic", soma)
    T.check_person_order(records, 2)
    seen = T.real_run_seen_mask()
    runs = T.miss_run_lengths(seen)

    out: dict = {
        "reference": REF_SYNTHETIC,
        "trajectory": {"source": "tools/compare/temporal.py fk_trajectory", "stride": stride,
                       "clips": list(clips), "frames": int(truth19.shape[1]),
                       "subjects": int(truth19.shape[0]),
                       "frames_per_seed": int(truth19.shape[1] * truth19.shape[0]),
                       "note": "28 frames x 2 subjects. Short: `build_take` uses ONE clip "
                               "per subject and the longest full-body clip on disk is 96 "
                               "frames. The over-smoother's window 27 on 28 frames is close "
                               "to a global parabola, which is recorded rather than hidden"},
        "may_select_a_shipped_constant": True,
        "noise": {"definitions": "tools/head/thorax_window_sweep.py, imported by "
                                 "temporal._thorax_noise -- OUR OWN detector's "
                                 "self-agreement, never MAMMA's residual",
                  "joints_noised": "root, neck, both shoulders, both hips -- exactly the "
                                   "six landmarks the clavicle solve reads",
                  "model": "heavy_tail_frame_correlated", "seeds": seeds},
    }

    # ---- the clean arm. P8 (exactness) and P13 (the low ceiling that must fail).
    clean_positions, _raw, _diag, _log, _s = T.run_pipeline(cameras, records)
    mapping_clean = T.pair_subjects(clean_positions, truth19)
    truth_by_trajectory: dict[int, object] = {}
    for ours in range(clean_positions.shape[0]):
        track, _r = solve_with(clean_positions[ours], passthrough_rule)
        truth_by_trajectory[mapping_clean[ours]] = track

    clean: dict = {"subject_pairing": {str(k): int(v) for k, v in mapping_clean.items()}}
    for label, replacement in (("d2b_passthrough", passthrough_rule),
                               ("d2c_shipped", SHIPPED),
                               ("ctrl_low_ceiling", low_ceiling_rule),
                               ("ctrl_oversmoother", oversmoother_rule)):
        entry = score_arm(clean_positions, mapping_clean, truth_by_trajectory, replacement)
        for private in ("_pooled", "_series", "_masks"):
            entry.pop(private, None)
        clean[label] = entry
    identical = []
    for ours in range(clean_positions.shape[0]):
        a, _ = solve_with(clean_positions[ours], SHIPPED)
        b, _ = solve_with(clean_positions[ours], passthrough_rule)
        identical.append(bool(np.array_equal(np.asarray(a.local_rotations_xyzw),
                                             np.asarray(b.local_rotations_xyzw))))
    clean["d2c_track_bit_identical_to_no_reject"] = bool(all(identical))
    true_peak = max(row["true_peak_step_deg"] for row in clean["d2b_passthrough"]["per_series"])
    clean["true_clavicle_peak_deg_per_frame"] = true_peak
    clean["true_clavicle_peak_deg_per_s"] = round(true_peak * SAMPLE_RATE_HZ, 2)
    clean["ceiling_deg_per_frame"] = round(CEILING, 4)
    clean["margin_x"] = round(CEILING / true_peak, 2) if true_peak else None
    clean["low_ceiling_deg_per_frame"] = LOW_CEILING_DEG_PER_FRAME
    clean["low_ceiling_note"] = (
        "the brief's 5 deg/frame is INERT on this fixture: the true clavicle peak is "
        f"{true_peak:.2f} deg/frame, so a 5 deg/frame ceiling fires on nothing and the "
        "must-fail control would PASS. 1.0 deg/frame sits below the true peak and is the "
        "degenerate that discriminates. A reject ceiling's 'no gate a constant can pass' "
        "demonstration needs true motion faster than the constant")
    out["clean"] = clean

    # ---- the world-measured control, on a turning body. P14.
    turning: dict = {}
    for degrees_per_frame in (12.0, 45.0):
        spun = yawed(clean_positions[0], degrees_per_frame)
        row = {}
        for label, replacement in (("local_shipped", SHIPPED),
                                   ("world_measured", world_measured_rule)):
            _track, recorder = solve_with(spun, replacement)
            row[label] = recorder.summary()
        turning[f"{degrees_per_frame:.0f}_deg_per_frame"] = {
            "deg_per_s": round(degrees_per_frame * SAMPLE_RATE_HZ, 1),
            "what_it_is": ("a pirouette, ~360 deg/s" if degrees_per_frame < 20
                           else "a skater's scratch spin, ~1350 deg/s"),
            **row}
    out["turning_body"] = turning

    # ---- the noisy arms. P9, P10, P11, P12, P15, and the flip-rate check.
    arms: dict[str, dict] = {label: {"pooled": [], "attenuation": [], "lag": [],
                                     "over": 0, "steps": 0, "rejected": 0, "frames": 0,
                                     "runs": [], "chain_over": 0, "chain_steps": 0,
                                     "chain_max": 0.0}
                             for label in RULES if label != "ctrl_world_measured"}
    on_rejected: dict[str, list] = {}
    chain_rates = []
    per_seed: list = []
    step_vs_reach = {"step_test_accepts_what_reachability_rejects": 0,
                     "reachability_rejects": 0, "step_test_rejects": 0}
    for seed in range(seeds):
        displacement, injection = T._thorax_noise(cameras, truth19, names,
                                                  "heavy_tail_frame_correlated", runs, seed)
        out["noise"].setdefault("injection", injection)
        noisy, _raw, _diag, _log, _s = T.run_pipeline(
            cameras, T.apply_displacements(records, displacement))
        mapping = T.pair_subjects(noisy, truth19)
        entries = {}
        for label, replacement in RULES.items():
            if label == "ctrl_world_measured":
                continue
            entries[label] = score_arm(noisy, mapping, truth_by_trajectory, replacement)
        # THE PAIRED SUBSET, same denominator: the frames the SHIPPED rule rejects, scored
        # on every arm. The pooled statistics above are dominated by the 98 % of frames the
        # reject never touches, so they cannot say what the mechanism did; this can.
        shipped_masks = entries["d2c_shipped"]["_masks"]
        for key, mask in shipped_masks.items():
            rejected = ~np.asarray(mask, dtype=bool)
            if not rejected.any():
                continue
            for label, entry in entries.items():
                on_rejected.setdefault(label, []).append(entry["_series"][key][rejected])
        for label, entry in entries.items():
            bucket = arms[label]
            bucket["pooled"].append(entry.pop("_pooled"))
            entry.pop("_series", None)
            entry.pop("_masks", None)
            bucket["chain_over"] += entry["chain_over_ceiling"]
            bucket["chain_steps"] += entry["chain_steps"]
            bucket["chain_max"] = max(bucket["chain_max"], entry["chain_max_step_deg"])
            bucket["attenuation"].append(entry["attenuation_median"])
            bucket["lag"].extend([row["argmin_frames"] for row in entry["per_series"]])
            bucket["over"] += entry["shoulder_steps_over_ceiling"]
            bucket["steps"] += entry["shoulder_steps"]
            bucket["rejected"] += entry["rejects"].get("rejected_frames", 0)
            bucket["frames"] += entry["rejects"].get("frames", 0)
            bucket["runs"].extend(
                [int(k)] * int(v) for k, v in entry["rejects"].get(
                    "run_length_histogram", {}).items())
        per_seed.append({
            "seed": seed,
            "d2b_clavicle_chain_over_the_ceiling": entries["d2b_passthrough"]["chain_over_ceiling"],
            "d2c_rejected_frames": entries["d2c_shipped"]["rejects"]["rejected_frames"],
            "d2b_error_p95_deg": entries["d2b_passthrough"]["error_vs_truth"]["p95_deg"],
            "d2c_error_p95_deg": entries["d2c_shipped"]["error_vs_truth"]["p95_deg"],
            "d2b_error_max_deg": entries["d2b_passthrough"]["error_vs_truth"]["max_deg"],
            "d2c_error_max_deg": entries["d2c_shipped"]["error_vs_truth"]["max_deg"],
        })
        # the flip-rate check, on the D2b arm: does the synthetic contain the phenomenon?
        for ours in range(noisy.shape[0]):
            track, _r = solve_with(noisy[ours], passthrough_rule)
            chain_rates.append(chain_over_ceiling(track))
            reach = SHIPPED(clavicle_locals(track, "LeftShoulder"), None, CEILING)[1]
            step = step_test_rule(clavicle_locals(track, "LeftShoulder"), None, CEILING)[1]
            step_vs_reach["reachability_rejects"] += int((~reach).sum())
            step_vs_reach["step_test_rejects"] += int((~step).sum())
            step_vs_reach["step_test_accepts_what_reachability_rejects"] += int(
                (step & ~reach).sum())

    noisy_out: dict = {}
    for label, bucket in arms.items():
        pooled = np.concatenate(bucket["pooled"])
        flat = [length for group in bucket["runs"] for length in group]
        noisy_out[label] = {
            "error_vs_truth": stats_deg(pooled),
            "attenuation_median": round(float(np.median(bucket["attenuation"])), 4),
            "lag_frames_median": float(np.median(bucket["lag"])),
            "lag_frames_all_zero": bool(all(v == 0 for v in bucket["lag"])),
            "shoulder_steps_over_ceiling": bucket["over"],
            "shoulder_steps": bucket["steps"],
            "rejected_frames": bucket["rejected"],
            "rejected_fraction": round(bucket["rejected"] / bucket["frames"], 5)
            if bucket["frames"] else 0.0,
            "run_length_max": max(flat) if flat else 0,
            "run_length_histogram": {str(k): flat.count(k) for k in sorted(set(flat))},
            "clavicle_chain_over_ceiling": bucket["chain_over"],
            "clavicle_chain_steps": bucket["chain_steps"],
            "clavicle_chain_max_step_deg": round(bucket["chain_max"], 2),
        }
        if label in on_rejected:
            noisy_out[label]["error_on_the_frames_d2c_rejects"] = stats_deg(
                np.concatenate(on_rejected[label]))
    noisy_out["error_on_rejected_frames_note"] = (
        "the SAME frames on every arm -- the ones the shipped rule rejects -- so the four "
        "rules are scored on one population. Same denominator")
    noisy_out["ctrl_oversmoother"]["rejects_nothing"] = (
        "a filter, not a reject: it REPLACES every frame, so its rejected_frames column is "
        "the frame count and is NOT a reject rate. Never read it as a pass")
    over = sum(row["over"] for row in chain_rates)
    steps = sum(row["steps"] for row in chain_rates)
    noisy_out["flip_rate_check"] = {
        "clavicle_chain_steps_over_the_ceiling": over,
        "clavicle_chain_steps": steps,
        "fraction": round(over / steps, 4),
        "real_take_fraction_d2b": [round(40 / 894, 4), round(33 / 894, 4)],
        "max_step_deg": round(max(row["max_deg"] for row in chain_rates), 2),
        "reproduces_the_real_take": bool(0.02 <= over / steps <= 0.08),
        "why": "the brief's go/no-go, taken BEFORE anything was pre-registered: if injected "
               "noise never brings the landmark near the pivot it is a different "
               "phenomenon and cannot select",
    }
    noisy_out["step_test_vs_reachability"] = step_vs_reach
    noisy_out["per_seed"] = per_seed
    counts = [row["d2b_clavicle_chain_over_the_ceiling"] for row in per_seed]
    noisy_out["per_seed_spread"] = {
        "d2b_over_ceiling_by_seed": counts,
        "seeds_with_no_flip_at_all": int(sum(1 for c in counts if c == 0)),
        "why_it_matters": "the phenomenon is a heavy-tail event: a flip needs a tail draw "
                          "to land on a shoulder while the lever is short, and on a "
                          "28-frame take most seeds contain none. The pooled rate is "
                          "therefore an average over seeds that differ by an order of "
                          "magnitude, and a five-seed pooled figure would be one seed's. "
                          "Recorded rather than smoothed over",
        "seeds": seeds,
    }
    out["noisy"] = noisy_out
    out["i7_shipped_window_figures_for_shape_only"] = {
        "attenuation_median": 0.3183, "window_27_attenuation_median": 0.6889,
        "argmin_frames": 0,
        "reference": "I7, artifacts/compare/temporal.json, fk_synthetic clean coverage. "
                     "3D DISPLACEMENT over 17 joints, in millimetres. Ours is CLAVICLE "
                     "ANGULAR SPEED. Different quantities, different references, never on "
                     "one axis -- printed here only so the shape can be compared",
    }
    return out


# ------------------------------------------------------------------- 2. the delivery path
def delivery_block() -> dict:
    """`delivery-d2c` against `delivery-root` (D2b), on disk. P1, P2, P3, P4.

    Frames over the ceiling is REPORTED and never banded: it is exactly what a reject
    zeroes, so a band on it would be a band the mechanism optimises directly.
    """
    out: dict = {"reference": REF_PHYSICAL,
                 "compares": f"{D2C_BUILD} (D2c) against {D2B_BUILD} (D2b)",
                 "ceiling_deg_per_frame": round(PHYSICAL_CEILING_DEG_PER_FRAME, 2),
                 "banded": False,
                 "why_not_banded": "frames over the ceiling is precisely what a reject "
                                   "zeroes. Banding it would be a band the candidate can "
                                   "optimise directly, which proves nothing",
                 "subjects": {}}
    for subject in (0, 1):
        before = np.load(D2B_BUILD / f"subject-{subject:02d}.body-track.npz")
        after = np.load(D2C_BUILD / f"subject-{subject:02d}.body-track.npz")
        names = json.loads(
            (D2C_BUILD / f"subject-{subject:02d}.body-track.json").read_text())["joint_names"]
        entry: dict = {}
        for key in ("triangulated_world_positions_z_up_m",
                    "raw_triangulated_world_positions_z_up_m", "ticks"):
            entry[f"{key}_byte_identical"] = bool(before[key].tobytes() == after[key].tobytes())
        qb = np.asarray(before["local_rotations_xyzw"], dtype=np.float64)
        qa = np.asarray(after["local_rotations_xyzw"], dtype=np.float64)
        joints: dict = {}
        chain = {"before": 0, "after": 0, "steps": 0}
        for name in CLAVICLE_CHAIN + ("LeftLowerLeg", "RightLowerLeg", "Neck", "Head"):
            index = names.index(name)
            a, b = step_angles_deg(qb[:, index]), step_angles_deg(qa[:, index])
            row = {}
            for label, values in (("d2b", a), ("d2c", b)):
                row[label] = {
                    "median_deg": round(float(np.median(values)), 2),
                    "p95_deg": round(float(np.percentile(values, 95)), 2),
                    "max_deg": round(float(values.max()), 2),
                    "over_the_ceiling": int((values > PHYSICAL_CEILING_DEG_PER_FRAME).sum()),
                }
            row["step_series_identical"] = bool(np.array_equal(a, b))
            joints[name] = row
            if name in CLAVICLE_CHAIN:
                chain["before"] += row["d2b"]["over_the_ceiling"]
                chain["after"] += row["d2c"]["over_the_ceiling"]
                chain["steps"] += int(a.size)
        entry["joints"] = joints
        entry["clavicle_chain_over_the_ceiling"] = chain
        entry["legs_neck_and_head_are_bit_identical"] = bool(all(
            joints[n]["step_series_identical"]
            for n in ("LeftLowerLeg", "RightLowerLeg", "Neck", "Head")))

        differ = {names[i] for i in range(len(names))
                  if not np.array_equal(before["local_rotations_xyzw"][:, i],
                                        after["local_rotations_xyzw"][:, i])}
        entry["differing_local_rotations"] = sorted(differ)
        entry["differing_locals_are_the_clavicle_chain_and_the_hands"] = bool(
            differ <= set(DOWNSTREAM))
        entry["root_translation_identical"] = bool(np.array_equal(
            before["root_translation_m"], after["root_translation_m"]))
        entry["foot_contacts_identical"] = bool(np.array_equal(
            before["foot_contacts"], after["foot_contacts"]))

        # P3 and P4 on the SHIPPED path: re-run the rule on D2b's own clavicle locals --
        # the very input pass B saw -- and check that the accepted frames came through
        # untouched and the rejected runs are short.
        rejects: dict = {}
        for name in CLAVICLES:
            index = names.index(name)
            _replaced, accepted = SHIPPED(qb[:, index], None, CEILING)
            same = np.array_equal(before["local_rotations_xyzw"][accepted, index],
                                  after["local_rotations_xyzw"][accepted, index])
            runs, length = [], 0
            for value in accepted:
                if value:
                    if length:
                        runs.append(length)
                    length = 0
                else:
                    length += 1
            if length:
                runs.append(length)
            rejects[name] = {
                "rejected_frames": int((~accepted).sum()),
                "frames": int(accepted.size),
                "rejected_fraction": round(float((~accepted).mean()), 4),
                "accepted_frames_bit_identical_to_d2b": bool(same),
                "runs": len(runs),
                "run_length_max": int(max(runs)) if runs else 0,
                "run_length_histogram": {str(k): int(v) for k, v in
                                         zip(*np.unique(runs, return_counts=True))} if runs else {},
                "theorem_run_length_bound": math.ceil(180.0 / CEILING) - 1,
            }
        entry["rejects"] = rejects
        out["subjects"][f"subject_{subject:02d}"] = entry
    for name in sorted(p.name for p in (D2C_BUILD / "work").glob("*observations.jsonl")):
        out.setdefault("work_byte_identical", {})[name] = bool(
            (D2C_BUILD / "work" / name).read_bytes()
            == (D2B_BUILD / "work" / name).read_bytes())
    return out


# ------------------------------------------------- 3. the arms this step does not touch
def capture_block(subject: int, mamma_body_id: int, rng) -> dict:
    """Delivered arms, round trips and arm B: D2b (pass-through) against D2c. P5, P6, P7.

    Both arms are one converter with one module attribute swapped, so the denominators are
    identical by construction and the BEFORE series is the real pre-D2c behaviour (the
    header records the bit-identity check that licenses that).
    """
    from d2_clavicle_gate import on_capture, round_trip  # noqa: E402  -- imported, not copied

    src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
        "triangulated_world_positions_z_up_m"]
    src = src[np.isfinite(src).all(axis=(1, 2))]
    points_y = rc.Y_UP_FROM_Z_UP(src)
    skeleton_sized, _limbs = sized_skeleton(DETAILED_HUMANOID, src)
    out: dict = {"frames_scored": int(len(src))}

    errors: dict = {}
    for label, replacement in (("d2b", passthrough_rule), ("d2c", SHIPPED)):
        with rule(replacement):
            e_canon, _t, _fk = on_capture(src)
            e_sized, _t2, _fk2 = on_capture(src, skeleton_sized)
            recorder = Recorder(replacement)
            with rule(recorder):
                e_round, _f1, _f2, _s = round_trip(src)
            # `round_trip` runs the converter TWICE, so the recorder holds four masks in
            # call order: pass 1's two clavicles, then pass 2's. P6 is about pass 2 alone.
            pass1_recorder, pass2_recorder = Recorder(replacement), Recorder(replacement)
            pass1_recorder.masks = recorder.masks[:2]
            pass2_recorder.masks = recorder.masks[2:]
            pass_rejects = {"pass1_on_the_real_capture": pass1_recorder.summary(),
                            "pass2_on_the_synthetic_body": pass2_recorder.summary()}
            e_round_sized, _f1s, _f2s, _ss = round_trip(src, skeleton_sized)
        errors[label] = {"canonical": e_canon, "sized": e_sized,
                         "roundtrip": e_round, "roundtrip_sized": e_round_sized}
        out[f"{label}_on_our_capture_canonical"] = groups(e_canon)
        out[f"{label}_on_our_capture_sized"] = groups(e_sized)
        out[f"{label}_roundtrip_canonical"] = groups(e_round)
        out[f"{label}_roundtrip_sized"] = groups(e_round_sized)
        if label == "d2c":
            out["roundtrip_reject_activity"] = pass_rejects
    # P6's attribution: what does pass 2 look like with the reject switched OFF for pass 2
    # alone? Pass 1 shipped, pass 2 pass-through, so any violation here is pass 2's own.
    with rule(SHIPPED):
        track1 = rc.solve(src)
        synthetic = rc.landmarks_from_fk(rc.fk_of(track1))
    with rule(passthrough_rule):
        track2_raw = rc.solve(rc.Z_UP_FROM_Y_UP(synthetic))
    raw: dict = {}
    for name in CLAVICLES:
        index = DETAILED_HUMANOID.index(name)
        step = step_angles_deg(
            np.asarray(track2_raw.local_rotations_xyzw, dtype=np.float64)[:, index])
        raw[name] = {"max_deg": round(float(step.max()), 2),
                     "over_the_ceiling": int((step > PHYSICAL_CEILING_DEG_PER_FRAME).sum())}
    out["roundtrip_pass2_raw_without_the_reject"] = {
        "joints": raw,
        "why": "P6 predicted pass 2 would reject NOTHING because the synthetic body is "
               "noise-free. REFUTED, and the cause is the round trip's own "
               "self-reference: `landmarks_from_fk` writes the rig's UpperArm ORIGINS into "
               "the `left_shoulder`/`right_shoulder` slots, and the converter builds the "
               "TORSO FRAME from exactly those two landmarks. So a clavicle the reject "
               "replaced moves the arm roots, which moves the torso frame pass 2 measures "
               "the clavicle against, and the local rotation picks up a jumping twist. The "
               "steps are an order of magnitude smaller than the real capture's 139-164 "
               "deg, and the round-trip STATISTIC barely moves",
    }
    out["reference_on_our_capture"] = REF_OURS
    out["reference_roundtrip"] = REF_ROUNDTRIP
    out["margins"] = {}
    for label, key in (("on_capture_canonical", "canonical"), ("on_capture_sized", "sized"),
                       ("roundtrip_canonical", "roundtrip")):
        for group in ("arms", "legs"):
            out["margins"][f"{label}_{group}_d2b_minus_d2c"] = rc.block_bootstrap_margin(
                errors["d2b"][key], errors["d2c"][key], group, rng)
    out["margins"]["note"] = (
        "moving-block bootstrap, block 15, 2000 draws, seed 20260902, IDENTICAL drawn "
        "frames on both arms, via `retarget_cost.block_bootstrap_margin` imported verbatim. "
        "Positive means D2b is worse; lower is better")

    # ---- arm B. MAMMA's own joints in, a separate reference, reports and never selects.
    pred = np.load(rc.MA3D / f"verts_joints_body_id-{mamma_body_id:02d}.npz",
                   allow_pickle=True)["pred_joints"].astype(np.float64)
    pred = pred[np.isfinite(pred).all(axis=(1, 2))]
    mamma_src = rc.adapt_mamma(pred)
    mamma_skeleton, _ = sized_skeleton(DETAILED_HUMANOID, mamma_src)
    arm_b: dict = {"reference": ("a canonical body built from arm B's own output -- MAMMA "
                                 "`pred_joints` in. A different body, different poses, a "
                                 "different joint convention; NEVER the same axis as the "
                                 "own-capture figures"),
                   "mamma_body_id": f"body_id-{mamma_body_id:02d}",
                   "frames_scored": int(len(pred))}
    b_errors: dict = {}
    for label, replacement in (("d2b", passthrough_rule), ("d2c", SHIPPED)):
        with rule(replacement):
            e_canon, _t, _fk = on_capture(mamma_src)
            e_sized, _t2, _fk2 = on_capture(mamma_src, mamma_skeleton)
        b_errors[label] = {"canonical": e_canon, "sized": e_sized}
        arm_b[f"{label}_canonical"] = groups(e_canon)
        arm_b[f"{label}_sized"] = groups(e_sized)
    for key in ("canonical", "sized"):
        arm_b[f"margin_{key}_arms_d2b_minus_d2c"] = rc.block_bootstrap_margin(
            b_errors["d2b"][key], b_errors["d2c"][key], "arms", rng)
    out["arm_B_mamma_joints_in"] = arm_b
    return out


def committed_regression() -> dict:
    """P7: through the pass-through rule the D2 and D2b figures still reproduce `gate.json`.

    D2's and D2b's committed bands were measured against a converter with no temporal
    reject. The pass-through IS that converter, so the check is that installing D2c has
    not moved a single one of them.
    """
    from d2_clavicle_gate import on_capture, round_trip  # noqa: E402

    if not D2_GATE.exists():
        return {"ok": None, "why": f"no {D2_GATE.name} on disk to compare against"}
    prior = json.loads(D2_GATE.read_text())
    rows, worst = [], 0.0
    with rule(passthrough_rule):
        for subject in (0, 1):
            src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
                "triangulated_world_positions_z_up_m"]
            src = src[np.isfinite(src).all(axis=(1, 2))]
            skeleton_sized, _ = sized_skeleton(DETAILED_HUMANOID, src)
            e_canon, _t, _f = on_capture(src)
            e_round, _f1, _f2, _s = round_trip(src)
            e_round_sized, _a, _b, _c = round_trip(src, skeleton_sized)
            key = f"subject_{subject:02d}"
            got = {
                "d2b_on_our_capture_canonical_arms": groups(e_canon)["arms"],
                "d2b_roundtrip_canonical_arms": groups(e_round)["arms"],
                "d2b_roundtrip_sized_arms": groups(e_round_sized)["arms"],
            }
            want = {
                "d2b_on_our_capture_canonical_arms": _dig(
                    prior, "d2b_root_placement", "subjects", key, "rigs", "canonical",
                    "D2b_shipped", "delivered_on_our_capture", "arms"),
                "d2b_roundtrip_canonical_arms": _dig(
                    prior, "d2b_root_placement", "subjects", key, "rigs", "canonical",
                    "D2b_shipped", "roundtrip", "arms"),
                "d2b_roundtrip_sized_arms": _dig(
                    prior, "d2b_root_placement", "subjects", key, "rigs", "sized_resolved",
                    "D2b_shipped", "roundtrip", "arms"),
            }
            for name, value in got.items():
                reference = want[name]
                if reference is None:
                    rows.append({"subject": key, "figure": name, "committed": None,
                                 "now": round(value, 4)})
                    continue
                delta = abs(float(reference) - float(value))
                worst = max(worst, delta)
                if delta > 0.005:
                    rows.append({"subject": key, "figure": name,
                                 "committed": reference, "now": round(value, 4),
                                 "delta_mm": round(delta, 4)})
    return {"ok": not rows, "max_abs_difference_mm": round(worst, 6), "band_mm": 0.005,
            "moved": rows, "compared_against": str(D2_GATE),
            "why": "the pass-through rule is the pre-D2c converter, bit-for-bit. If any "
                   "committed D2 or D2b figure moved, D2c rewrote them and the earlier "
                   "report can no longer be read as D2's"}


def _dig(node, *path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


# ------------------------------------------------------------------------- 4. the oracle
SMPLX_COLLARS = {"left": (13, 16), "right": (14, 17)}     # (collar, shoulder)


def oracle_block(mapping) -> dict:
    """MAMMA's own collar->shoulder DIRECTION through the same ceiling. Reports only.

    This is the floor a real clavicle sequence shows on this take, from an instrument that
    never saw our rig. It is a 2-DoF DIRECTION step and ours is a 3-DoF rotation step
    including twist, so the two numbers describe different quantities and must never share
    an axis or a chart. MAMMA reports; it never selects. `pred_joints` only -- `gt_joints`
    on this fixture is a byte-copy of `pred_joints` and is not ground truth.
    """
    out: dict = {"reference": REF_MAMMA, "selects": "nothing",
                 "quantity": "collar->shoulder unit DIRECTION, frame-to-frame angle in "
                             "degrees. 2 degrees of freedom; ours is a 3-DoF rotation step",
                 "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}"
                                            for k, v in mapping.items()},
                 "subjects": {}}
    for subject in (0, 1):
        pred = np.load(rc.MA3D / f"verts_joints_body_id-{mapping[subject]:02d}.npz",
                       allow_pickle=True)["pred_joints"].astype(np.float64)
        entry: dict = {}
        for side, (collar, shoulder) in SMPLX_COLLARS.items():
            direction = pred[:, shoulder] - pred[:, collar]
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            unit = direction / np.maximum(norm, 1e-12)
            dots = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0)
            step = np.degrees(np.arccos(dots))
            entry[side] = {
                "median_deg": round(float(np.median(step)), 3),
                "p95_deg": round(float(np.percentile(step, 95)), 3),
                "max_deg": round(float(step.max()), 3),
                "over_the_ceiling": int((step > PHYSICAL_CEILING_DEG_PER_FRAME).sum()),
                "steps": int(step.size),
                "collar_to_shoulder_mm_median": round(float(np.median(norm)) * 1000.0, 1),
            }
        out["subjects"][f"subject_{subject:02d}"] = entry
    return out


# --------------------------------------------------------------------- 5. the silhouette
SILHOUETTE_D2B = OUT_DIR / "silhouette-d2b.json"
SILHOUETTE_D2C = OUT_DIR / "silhouette-d2c.json"


def silhouette_block() -> dict:
    """I6 on the D2c rebuild against the D2b one. REPORTED, never banded.

    The reference is MAMMA's SAM2 masks -- pixels of the actual footage, the one figure in
    this lane whose reference is not another model -- and it is blind to depth, to a
    left/right mirror of a fore-aft symmetric pose, and to everything inside the outline.
    D2c moves ~2-9 % of clavicle frames, so the expectation is that it barely moves at all.
    """
    from d2_clavicle_gate import _silhouette_rows  # noqa: E402  -- imported, not copied

    out: dict = {
        "reference": ("MAMMA's SAM2 masks -- pixels of the actual footage. Blind to depth, "
                      "to a fore-aft symmetric mirror and to everything inside the outline"),
        "banded": False,
        "regenerate": ".venv/bin/python tools/compare/silhouette.py --delivery "
                      "artifacts/compare/d2-clavicle/delivery-d2c --work "
                      "artifacts/compare/d2-clavicle/silhouette-work-d2c --out "
                      "artifacts/compare/d2-clavicle/silhouette-d2c.json  (the work "
                      "directory is seeded with silhouette-work-d2b's MAMMA-derived caches, "
                      "which read masks and mean bodies and never our track -- the ORACLE "
                      "bit-identity row below is the proof that reusing them changed nothing)",
        "available": SILHOUETTE_D2C.exists() and SILHOUETTE_D2B.exists(),
    }
    if not out["available"]:
        out["why_absent"] = f"{SILHOUETTE_D2C.name} or {SILHOUETTE_D2B.name} not on disk"
        return out
    before = json.loads(SILHOUETTE_D2B.read_text())
    after = json.loads(SILHOUETTE_D2C.read_text())
    out["d2b_per_camera_subject"] = _silhouette_rows(before)
    out["d2c_per_camera_subject"] = _silhouette_rows(after)
    out["d2b_oracle_mamma_mesh"] = _silhouette_rows(before, "ORACLE_mamma_mesh")
    out["d2c_oracle_mamma_mesh"] = _silhouette_rows(after, "ORACLE_mamma_mesh")
    pairs = [(v.get("iou_median"), out["d2c_per_camera_subject"][k].get("iou_median"))
             for k, v in out["d2b_per_camera_subject"].items()
             if k in out["d2c_per_camera_subject"]]
    pairs = [(a, b) for a, b in pairs
             if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    if pairs:
        out["d2b_iou"] = round(float(np.median([a for a, _ in pairs])), 4)
        out["d2c_iou"] = round(float(np.median([b for _, b in pairs])), 4)
        out["cells"] = len(pairs)
        out["cells_where_iou_fell"] = sum(1 for a, b in pairs if b < a)
    oracle_gap = [abs(v["iou_median"] - out["d2c_oracle_mamma_mesh"][k]["iou_median"])
                  for k, v in out["d2b_oracle_mamma_mesh"].items()
                  if k in out["d2c_oracle_mamma_mesh"]]
    out["oracle_must_be_unchanged"] = {
        "max_abs_iou_difference": round(max(oracle_gap), 6) if oracle_gap else None,
        "why": "the oracle is MAMMA's own mesh through our scoring path and reads none of "
               "our track. Anything but 0 means the two runs are not comparable",
    }
    return out


# ------------------------------------------------------------------------- 6. verdicts
def verdicts(report: dict) -> list:
    rows: list = []

    def row(band, verdict, before=None, after=None, detail="", reference="", banded=True):
        rows.append({"band": band, "verdict": verdict, "before": before, "after": after,
                     "detail": detail, "reference": reference, "banded": banded})

    clean = report["synthetic"]["clean"]
    noisy = report["synthetic"]["noisy"]
    exact = (clean["d2c_shipped"]["rejects"]["rejected_frames"] == 0
             and clean["d2c_track_bit_identical_to_no_reject"]
             and abs(clean["d2c_shipped"]["attenuation_median"]) < 1e-9)
    row("P8 exactness on clean synthetic input: zero rejects, bit-identical, zero attenuation",
        "PASS" if exact else "FAIL",
        clean["d2b_passthrough"]["rejects"]["rejected_frames"],
        clean["d2c_shipped"]["rejects"]["rejected_frames"],
        f"true clavicle peak {clean['true_clavicle_peak_deg_per_frame']} deg/frame "
        f"({clean['true_clavicle_peak_deg_per_s']} deg/s), ceiling {clean['ceiling_deg_per_frame']}, "
        f"margin {clean['margin_x']}x", REF_SYNTHETIC)

    b, a = noisy["d2b_passthrough"], noisy["d2c_shipped"]
    row("P9a synthetic truth: clavicle error median NOT WORSE (0.05 deg)",
        "PASS" if a["error_vs_truth"]["median_deg"] <= b["error_vs_truth"]["median_deg"] + 0.05
        else "FAIL", b["error_vs_truth"]["median_deg"], a["error_vs_truth"]["median_deg"],
        "pooled over every clavicle frame", REF_SYNTHETIC)
    row("P9b synthetic truth: clavicle error p95 BETTER",
        "PASS" if a["error_vs_truth"]["p95_deg"] < b["error_vs_truth"]["p95_deg"] else "FAIL",
        b["error_vs_truth"]["p95_deg"], a["error_vs_truth"]["p95_deg"],
        "pre-registered as the MARGINAL band: the reject touches ~2 % of shoulder frames "
        "and p95 is the top 5 %", REF_SYNTHETIC)
    row("P9c synthetic truth: clavicle error max BETTER",
        "PASS" if a["error_vs_truth"]["max_deg"] < b["error_vs_truth"]["max_deg"] else "FAIL",
        b["error_vs_truth"]["max_deg"], a["error_vs_truth"]["max_deg"],
        "pooled over EVERY frame, so the worst frame need not be one the reject touches",
        REF_SYNTHETIC)
    if "error_on_the_frames_d2c_rejects" in a:
        row("P9d the same error on the frames the reject actually touches (same denominator)",
            "PASS" if (a["error_on_the_frames_d2c_rejects"]["max_deg"]
                       < b["error_on_the_frames_d2c_rejects"]["max_deg"]) else "FAIL",
            b["error_on_the_frames_d2c_rejects"]["max_deg"],
            a["error_on_the_frames_d2c_rejects"]["max_deg"],
            "the population the mechanism acts on, scored identically on both arms",
            REF_SYNTHETIC)
    row("P10 attenuation of true clavicle peaks NO WORSE than D2b on identical draws",
        "PASS" if a["attenuation_median"] <= b["attenuation_median"] + 1e-6 else "FAIL",
        b["attenuation_median"], a["attenuation_median"],
        "negative means the noisy arm moves FASTER than truth, which is what noise does. "
        "I7's 0.3183 is 3D displacement over 17 joints and is NOT on this axis",
        REF_SYNTHETIC)
    row("P11 phase lag zero on the NOISY arm (shift sweep, one fixed frame set)",
        "PASS" if a["lag_frames_all_zero"] else "FAIL",
        b["lag_frames_median"], a["lag_frames_median"],
        "PRE-REGISTERED AS WRITTEN AND LEFT AS WRITTEN. It is not interpretable on this "
        "arm: D2b, D2c and the step test all read the SAME lag, so the figure is a "
        "property of the noisy positions and not of any rule. The two rows below are the "
        "interpretable versions and neither is a substitute for this verdict",
        REF_SYNTHETIC)
    row("P11a REPORTED: phase lag on CLEAN input, where a phase question is answerable",
        "REPORT", clean["d2b_passthrough"]["lag_frames_median"],
        clean["d2c_shipped"]["lag_frames_median"],
        f"the over-smoother reads {clean['ctrl_oversmoother']['lag_frames_median']} on the "
        "same fixture, which is what a filter does and a reject does not", REF_SYNTHETIC,
        banded=False)
    row("P11b REPORTED: on the noisy arm D2c's lag is IDENTICAL to D2b's",
        "REPORT", b["lag_frames_median"], a["lag_frames_median"],
        "identical to the frame; the low-ceiling control reads "
        f"{noisy['ctrl_low_ceiling']['lag_frames_median']}. D2c adds no phase of its own",
        REF_SYNTHETIC, banded=False)

    over = clean["ctrl_oversmoother"]
    row("P12 CONTROL the over-smoother (window 27) must FAIL on attenuation",
        "PASS" if over["attenuation_median"] > 0.1 else "FAIL",
        clean["d2c_shipped"]["attenuation_median"], over["attenuation_median"],
        "zero-phase, so it cannot fail on lag -- which is exactly why attenuation is the "
        "discriminating band and this is a control, not the fix", REF_SYNTHETIC)
    low = clean["ctrl_low_ceiling"]
    row("P13 CONTROL a ceiling below the fixture's own true peak must FIRE on true motion",
        "PASS" if (low["rejects"]["rejected_frames"] > 0
                   and low["attenuation_median"] > 0.0) else "FAIL",
        clean["d2c_shipped"]["rejects"]["rejected_frames"], low["rejects"]["rejected_frames"],
        clean["low_ceiling_note"], REF_SYNTHETIC)
    turning = report["synthetic"]["turning_body"]
    fast = turning["45_deg_per_frame"]
    slow = turning["12_deg_per_frame"]
    row("P14 CONTROL the WORLD-measured reject must reject honest motion on a turning body",
        "PASS" if (fast["world_measured"]["rejected_frames"] > fast["local_shipped"]["frames"] // 2
                   and fast["local_shipped"]["rejected_frames"] == 0
                   and slow["world_measured"]["rejected_frames"] == 0
                   and slow["local_shipped"]["rejected_frames"] == 0) else "FAIL",
        fast["local_shipped"]["rejected_frames"], fast["world_measured"]["rejected_frames"],
        "a scratch spin at 45 deg/frame (1350 deg/s); at a pirouette's 12 deg/frame neither "
        "variant fires", REF_SYNTHETIC)
    step = noisy["step_test_vs_reachability"]
    row("P15 CONTROL a step test must ACCEPT what reachability rejects",
        "PASS" if step["step_test_accepts_what_reachability_rejects"] > 0 else "FAIL",
        step["reachability_rejects"], step["step_test_rejects"],
        f"{step['step_test_accepts_what_reachability_rejects']} frames accepted by the step "
        "test and rejected by reachability -- the wrong plateau, held", REF_SYNTHETIC)
    flip = noisy["flip_rate_check"]
    row("GO/NO-GO the synthetic arm reproduces the real take's flip rate",
        "PASS" if flip["reproduces_the_real_take"] else "FAIL",
        flip["real_take_fraction_d2b"][0], flip["fraction"],
        "taken BEFORE anything was pre-registered, as the brief orders", REF_SYNTHETIC)

    delivery = report.get("delivery")
    if delivery:
        identical = all(
            s["differing_locals_are_the_clavicle_chain_and_the_hands"]
            and s["root_translation_identical"] and s["legs_neck_and_head_are_bit_identical"]
            and all(r["accepted_frames_bit_identical_to_d2b"] for r in s["rejects"].values())
            for s in delivery["subjects"].values())
        row("P3 accepted frames bit-identical to D2b; only the clavicle chain moves",
            "PASS" if identical else "FAIL", None, None,
            "root translation, foot contacts, legs, neck, head and every finger local are "
            "bit-identical on every frame", REF_PHYSICAL)
        bounded = all(r["run_length_max"] <= r["theorem_run_length_bound"]
                      for s in delivery["subjects"].values() for r in s["rejects"].values())
        row("P4 no rejected run exceeds the envelope bound",
            "PASS" if bounded else "FAIL",
            math.ceil(180.0 / CEILING) - 1,
            max(r["run_length_max"] for s in delivery["subjects"].values()
                for r in s["rejects"].values()),
            "the envelope reaches 180 deg at ceil(180/26.67) = 7 frames, so a rejected run "
            "cannot exceed 6 -- a theorem, measured beside", REF_PHYSICAL)
        for name, s in delivery["subjects"].items():
            chain = s["clavicle_chain_over_the_ceiling"]
            row(f"P1 REPORTED (no band) {name}: clavicle-chain frames over the ceiling",
                "REPORT", chain["before"], chain["after"],
                "a reject zeroes this by construction on the two Shoulders; the residual is "
                "UpperArm/LowerArm, which inherit and are not treated", REF_PHYSICAL,
                banded=False)
            row(f"P1 REPORTED (no band) {name}: median clavicle step, which must NOT move",
                "REPORT",
                s["joints"]["LeftShoulder"]["d2b"]["median_deg"],
                s["joints"]["LeftShoulder"]["d2c"]["median_deg"],
                "the median is the short lever D2 and D2b created, and D2c does not touch "
                "it. That is D5's", REF_PHYSICAL, banded=False)

    for name, block in report.get("capture", {}).items():
        if not isinstance(block, dict) or "margins" not in block:
            continue
        for key in ("on_capture_canonical_arms_d2b_minus_d2c",
                    "on_capture_sized_arms_d2b_minus_d2c"):
            margin = block["margins"][key]
            ok = margin["ci95_mm"][1] >= 0.0
            row(f"P5 {name}: delivered arms not regressed -- {key}",
                "PASS" if ok else "FAIL",
                block["d2b_on_our_capture_canonical"]["arms"]
                if "canonical" in key else block["d2b_on_our_capture_sized"]["arms"],
                block["d2c_on_our_capture_canonical"]["arms"]
                if "canonical" in key else block["d2c_on_our_capture_sized"]["arms"],
                f"margin {margin['median_mm']} mm, CI {margin['ci95_mm']}, positive means "
                "D2b is worse", REF_OURS)
        activity = block.get("roundtrip_reject_activity", {}).get(
            "pass2_on_the_synthetic_body", {})
        row(f"P6 {name}: the round trip's second pass rejects NOTHING",
            "PASS" if activity.get("rejected_frames", 1) == 0 else "FAIL",
            0, activity.get("rejected_frames"),
            "REFUTED, and the reason is in `roundtrip_pass2_raw_without_the_reject`: "
            "`landmarks_from_fk` writes the rig's UpperArm origins into the shoulder slots "
            "and the converter builds the TORSO FRAME from those two landmarks, so a "
            "replaced clavicle moves the frame its own local rotation is measured against. "
            "The round trip is self-referential and this is that, in miniature. The raw "
            "pass-2 steps peak near 33 deg against the real capture's 139-164, and the "
            "round-trip STATISTIC moves 0.51 -> 0.55 and 0.08 -> 0.08 mm",
            REF_ROUNDTRIP)

    regression = report.get("committed_regression", {})
    row("P7 the committed D2 and D2b figures reproduce through the pass-through rule",
        "PASS" if regression.get("ok") else ("FAIL" if regression.get("ok") is False else "REPORT"),
        0.0, regression.get("max_abs_difference_mm"),
        regression.get("why", ""), REF_OURS)
    return rows


# --------------------------------------------------------------------------- 7. the picture
PICTURE_FRAMES = range(44, 55)


def picture() -> dict:
    """D2b beside D2c on camera A001, frames 44-54, through `d2_jitter_sheet`'s own renderer.

    The module is IMPORTED and its `render` / `tile` / `sheet` used verbatim -- it wraps
    `tools/swap-harness/camera_overlay.py`, which SETS THE SCENE FPS FROM THE TRACK BEFORE
    THE glTF IMPORT. momentum writes keyframe times in seconds and Blender converts them
    with the scene fps, so a 30 fps motion in the 24 fps factory scene runs 25 % fast and
    freezes; the symptom reads as a placement error and is a timebase error (CLAUDE.md).
    Its own module-level paths are rebound rather than edited: `d2_jitter_sheet.py` is not
    this step's file.
    """
    import d2_jitter_sheet as js  # noqa: E402

    folder = OUT_DIR / "jitter-d2c"
    folder.mkdir(parents=True, exist_ok=True)
    js.WINDOW = PICTURE_FRAMES
    js.OUT = folder
    steps, images = {}, {}
    index = DETAILED_HUMANOID.index(js.JOINT)
    for label, build in (("d2b", D2B_BUILD), ("d2c", D2C_BUILD)):
        js.render(build, folder / label)
        track = np.load(build / f"subject-{js.SUBJECT:02d}.body-track.npz")
        steps[label] = js.step_angles(track["local_rotations_xyzw"][:, index])
    frames = list(PICTURE_FRAMES)
    rows = [[js.tile(folder / label / f"f{f:03d}.jpg", (340, 165, 600, 395), 2,
                     f"f{f}  step {steps[label][f]:5.1f} deg",
                     steps[label][f] > js.CEILING_DEG, header=26, size=16)
             for f in frames] for label in ("d2b", "d2c")]
    sheet_path = folder / f"d2c-contact-sheet-subject{js.SUBJECT:02d}-{js.JOINT}.png"
    js.sheet(rows, len(frames),
             f"Subject {js.SUBJECT}, camera {js.CAMERA}, {js.JOINT}, frames "
             f"{frames[0]}-{frames[-1]}. Red = the step about to be taken exceeds a human "
             f"joint's peak rate.",
             "Top: D2b (the arm before the reject). Bottom: D2c. Same frames, same camera, "
             "scene fps set to 30 before import.", sheet_path)
    near = [47, 48, 49, 50]
    rows = [[js.tile(folder / label / f"f{f:03d}.jpg", (430, 175, 580, 300), 5,
                     f"{label.upper()}  f{f}   next step {steps[label][f]:.1f} deg",
                     steps[label][f] > js.CEILING_DEG) for f in near]
            for label in ("d2b", "d2c")]
    zoom_path = folder / f"d2c-zoom-subject{js.SUBJECT:02d}-f{near[0]:03d}-{near[-1]:03d}.png"
    js.sheet(rows, len(near),
             f"Subject {js.SUBJECT}, {js.JOINT}, the f48 pop. Top: D2b. Bottom: D2c. "
             f"Camera {js.CAMERA}.", "", zoom_path)
    images["contact_sheet"] = str(sheet_path.relative_to(ROOT))
    images["zoom"] = str(zoom_path.relative_to(ROOT))
    images["steps_d2b"] = {str(f): round(float(steps["d2b"][f]), 2) for f in frames}
    images["steps_d2c"] = {str(f): round(float(steps["d2c"][f]), 2) for f in frames}
    return images


# ------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--picture", action="store_true",
                        help="render the D2b/D2c contact sheet (needs Blender)")
    parser.add_argument("--skip-synthetic", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ours = np.stack([
        np.load(TRACKS / f"subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"] for s in (0, 1)])
    mapping = subject_map.mamma_index_for(ours)
    rng = np.random.default_rng(rc.RNG_SEED)

    report: dict = {
        "instrument": "tools/compare/d2c_clavicle_temporal_gate.py",
        "step": "D2c -- the clavicle's temporal defect",
        "regenerate": "PYTHONPATH=$PWD/src .venv/bin/python "
                      "tools/compare/d2c_clavicle_temporal_gate.py",
        "rebuild_regenerate": (
            "mkdir -p artifacts/compare/d2-clavicle/delivery-d2c && cp -R "
            "artifacts/commercial-multiview-soma77/work "
            "artifacts/compare/d2-clavicle/delivery-d2c/work && PYTHONPATH=$PWD/src "
            ".venv/bin/python scripts/build_commercial_multiview_comparison.py --videos "
            ".cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos "
            "--calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml "
            "--detector soma77 --output artifacts/compare/d2-clavicle/delivery-d2c"),
        "autoanim_gnm_resolved_to": str(Path(autoanim_gnm.__file__).resolve()),
        "shipping": ("src/autoanim_gnm/commercial_multiview.py: "
                     "`CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S`, "
                     "`_reachable_clavicle_sequence`, and `positions_to_body_track`'s "
                     "three passes"),
        "constant": {"name": "CLAVICLE_MAXIMUM_FRAME_TRAVEL_DEG_PER_S", "value": 800.0,
                     "deg_per_frame_at_30hz": round(CEILING, 4),
                     "provenance": "ANATOMY -- the head lane's own physiology "
                                   "(head_orientation.MAXIMUM_FRAME_TRAVEL_DEG). Registered "
                                   "in tools/compare/provenance.py; the audit reads CLEAN",
                     "selected_on": "nothing. Its INERTNESS is demonstrated on synthetic "
                                    "truth and the degenerate that must fail it is a "
                                    "ceiling below the fixture's own true peak"},
        "pre_registered": PREREGISTERED,
        "before_arm_is_the_passthrough": (
            "every BEFORE arm is `_reachable_clavicle_sequence` swapped for a pass-through, "
            "not a separate code path. With it installed the converter reproduces the "
            f"pre-D2c code BIT-FOR-BIT on both performers and both rigs ({BASELINE.name})"),
        "passthrough_is_bit_identical_to_pre_d2c": _baseline_check(),
        "subject_correspondence": {f"our_{k}": f"body_id-{v:02d}" for k, v in mapping.items()},
        "subject_correspondence_source": "tools/head/subject_map.py, 3D pelvis agreement",
        "references_never_on_one_axis": {
            "synthetic_truth": REF_SYNTHETIC, "our_own_capture": REF_OURS,
            "roundtrip": REF_ROUNDTRIP, "mamma": REF_MAMMA, "physical": REF_PHYSICAL},
        "blind_to": [
            "the MEDIAN clavicle step, which is the short lever D2 and D2b created and "
            "which D2c does not touch. That is D5's, and it stays",
            "a wrong clavicle plateau that is REACHABLE: a physical reject can only see "
            "physics violations, never errors",
            "everything the synthetic fixture does not contain -- calibration error, camera "
            "sync, soft tissue -- and specifically its clavicle moves SLOWER than the real "
            "take's, so 'the ceiling never fires on true motion' is established above THIS "
            "fixture's clavicle and not above a sprinter's",
            "the arms below the clavicle: their own landmark noise is untouched, which is "
            "why the chain's residual over-ceiling count is UpperArm/LowerArm",
            "the silhouette's arm share, which is D6's",
        ],
    }

    if not args.skip_synthetic:
        report["synthetic"] = synthetic_block(seeds=args.seeds)
    report["delivery"] = delivery_block() if (
        D2C_BUILD / "subject-00.body-track.npz").exists() else None
    report["capture"] = {f"subject_{s:02d}": capture_block(s, mapping[s], rng) for s in (0, 1)}
    report["committed_regression"] = committed_regression()
    if (D2C_BUILD / "subject-00.body-track.npz").exists():
        # RUNG 11: absolute capture world, arms and legs kept apart, against MAMMA's
        # `pred_joints`. REPORTED with intervals. Both arms are the same build path on the
        # same frames, so the denominators are identical; the labels are the ones
        # `rung11_block` recognises, and `D2` here is D2b (the build D2c is measured from).
        report["rung11_vs_mamma"] = rung11_block(
            {"D2": scoreboard_errors(D2B_BUILD, mapping),
             "D2b": scoreboard_errors(D2C_BUILD, mapping)}, rng)
        report["rung11_vs_mamma"]["label_note"] = (
            "`D2` is the D2b build (delivery-root) and `D2b` is the D2c build "
            "(delivery-d2c). The keys are `rung11_block`'s own and are NOT renamed here, "
            "because that helper is D2's file and this step does not edit it. "
            "`margin_D2_minus_D2b` therefore reads D2b minus D2c: positive means D2b is "
            "worse. Reported, not banded")
    report["oracle_mamma_collars"] = oracle_block(mapping)
    report["silhouette"] = silhouette_block()
    if args.picture:
        report["picture"] = picture()

    report["gate"] = verdicts(report)
    banded = [r for r in report["gate"] if r.get("banded", True)]
    report["verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in banded) else "FAIL"
    report["seconds"] = round(time.time() - started, 1)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    console(report)
    print(f"\nwrote {args.out}  ->  {report['verdict']}")
    return 0


def _baseline_check() -> dict:
    """Is the pass-through arm the pre-D2c converter, bit for bit? Checked, not assumed."""
    if not BASELINE.exists():
        return {"checked": False, "why": f"{BASELINE} absent"}
    baseline = np.load(BASELINE)
    rows: dict = {}
    ok = True
    for subject in (0, 1):
        src = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"]
        src = src[np.isfinite(src).all(axis=(1, 2))]
        skeleton_sized, _ = sized_skeleton(DETAILED_HUMANOID, src)
        with rule(passthrough_rule):
            canonical = rc.solve(src)
            sized = rc.solve(src, skeleton_sized)
        for tag, track, key in (("canonical", canonical, "local"), ("sized", sized, "slocal")):
            same = np.array_equal(np.asarray(track.local_rotations_xyzw),
                                  baseline[f"{key}_{subject}"])
            rows[f"subject_{subject:02d}_{tag}"] = bool(same)
            ok = ok and same
    return {"checked": True, "bit_identical": ok, "per_arm": rows,
            "baseline": str(BASELINE.relative_to(ROOT)),
            "regenerate": "git show 0a2979a:src/autoanim_gnm/commercial_multiview.py into "
                          "place, then rc.solve each subject's "
                          "triangulated_world_positions_z_up_m on the canonical and the "
                          "sized rig"}


def console(report: dict) -> None:
    print(f"\nD2c -- {report['step']}")
    if "synthetic" in report:
        clean = report["synthetic"]["clean"]
        print(f"  synthetic clean: true clavicle peak "
              f"{clean['true_clavicle_peak_deg_per_frame']} deg/frame, ceiling "
              f"{clean['ceiling_deg_per_frame']:.2f}, margin {clean['margin_x']}x")
    if report.get("delivery"):
        for name, s in report["delivery"]["subjects"].items():
            chain = s["clavicle_chain_over_the_ceiling"]
            print(f"  {name}: clavicle-chain frames over the ceiling "
                  f"{chain['before']} -> {chain['after']} of {chain['steps']}")
    print()
    for row in report["gate"]:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "REPORT": "----"}[row["verdict"]]
        print(f"  [{mark}] {row['band']}")
        if row["before"] is not None or row["after"] is not None:
            print(f"         before {row['before']}   after {row['after']}")


if __name__ == "__main__":
    raise SystemExit(main())
