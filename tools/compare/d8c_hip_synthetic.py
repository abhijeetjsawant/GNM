#!/usr/bin/env python3
"""D8c's selector: the HIP LINE row scored against SYNTHETIC TRUTH, and nowhere else.

THIS FILE CONFIRMS D8b's MODE AND CEILING ON THE HIPS' OWN GEOMETRY, AND NOTHING ELSE MAY.
`SEGMENT_LENGTH_CEILING_FRACTION` (0.15) and `SEGMENT_LENGTH_MODES`' selected value
(`demote`) are D8b's and are NOT re-selected here. The card is explicit: "the ceiling (0.15)
and the mode (`demote`) are D8b's and are not re-selected -- the synthetic CONFIRMS them on
the hips' own geometry, and if it selects otherwise the step STOPS and is re-carded, because
a per-segment mode or ceiling is a new constant." That stop condition is mechanical in this
file's `verdict` and in `selector.selects_the_shipped_mode`.

Nothing is selected on a MAMMA-referenced arm; MAMMA does not appear in this file at all.

THE FIXTURE, WRAPPED AND NEVER RE-IMPLEMENTED
----------------------------------------------
`tools/compare/d8b_length_synthetic.py`'s **fixture v2** is imported and its builders are
called: `build_fixture_v2` (the squat clip at stride 2, the clips and stride D8b's reviewer's
repair selected), `scaled_heavy_tail` (I7's own heavy-tail model with its magnitude scaled),
`honest_leg_spread` and `collapse_displacements`. That file is not edited and its committed
reports are untouched. Through it, `d8_occlusion_synthetic.py`'s `run`, `score`, `real_seen`
and `window_ground` are the real pipeline's own entry points, so every arm here goes through
`cm.reconstruct_multiview` and never through a second association loop (CLAUDE.md: a hand
replication of that loop drifted 9-19 mm; wrapping reproduces at 0.0 mm).

THE ONE THING THIS FILE DOES THAT D8b's DOES NOT: IT RUNS BEFORE THE SRC CHANGE
-------------------------------------------------------------------------------
`cm.reconstruct_multiview` reads `cm.SEGMENT_LENGTH_RULES` from the module, and there is no
keyword argument for the rule list. So this file holds BOTH tuples explicitly --
`TODAY_RULES` (the nine rows as D8b shipped them) and `CANDIDATE_RULES` (those nine plus the
hip line) -- and rebinds the module attribute around each run inside a context manager that
restores it and ASSERTS the restore. Two consequences, and both are deliberate:

  * the selector can be run, read and committed with `src/` untouched, which is the order
    the card requires;
  * after the src change the same file must produce the SAME numbers, because
    `TODAY_RULES` is defined by REMOVING any hip row rather than by trusting the module's
    current contents. That equality is a free src-hygiene check and the gate reads it.

THE INJECTION, AND IT IS NOT D8b's
-----------------------------------
D8b moved both shoulders toward the NECK. That is the wrong motion for this defect and the
card's pre-dispatch review says why: on the take the hip line halves while `|hip_mid - root|`
holds (68.0-80.7 mm against a 76.5 median on frames 110-119) and both thighs hold
(391.9-421.8 against 401.8 / 406.2). A collapse toward the pelvis landmark -- which sits
76.5 mm OFF the hip line -- would have moved both of those, and an injection that lengthens
the thighs would make the thigh rows fire, which on the real run they do not.

So the injection is: **in every view that sees them, both hip detections are moved toward
the 2D MIDPOINT OF THE TWO HIPS, along the line between them, by a factor** -- the
inter-hip vector scaled in the image, with the knees, the root and everything else
untouched -- before the noise and before the keep mask. The triangulation that follows is
well conditioned and its reprojection residuals stay small, by construction, which is the
point: only the performer's own hip width can tell that anything happened.

FOUR ARMS ON ONE POPULATION, plus THREE INJECTION VARIANTS REPORTED BESIDE THEM
-------------------------------------------------------------------------------
Arms: today (D8b: the nine rows) / demote / reject / best_ray, each with the hip row added,
across the ceiling sweep 0.05-0.30 and at the shipped 0.15. Variants, reported and never
selecting: the factor sensitivity (0.35 / 0.5 / 0.75), a ONE-HIP arm (the 84-86 mode, one
hip moved and the other exact) and a STRETCH arm at 1.25 (class (ii)).

WHAT THIS FIXTURE IS BLIND TO, and the last one is the important one
---------------------------------------------------------------------
* **The real detector's bias.** The injection is a fixed scaling along the hip line; the
  footage's is whatever the detector does as a body twists and falls.
* **Speed.** The clips move slower than the footage (D8b's measured shortfall: 123.7 mm of
  landmark travel over the run against the take's 188.1).
* **Rigidity.** The fixture's bones are exactly rigid, so its honest frames are easier than
  the take's. That is what S0 measures and calibrates against.
* **CLASS (ii) IS NOT REALISED HERE, AND THE STRETCH ARM DOES NOT REALISE IT.** The take's
  class (ii) is a TWO-VIEW DEPTH stretch: A and C at 139.96-140.70 degrees, the hip line
  within 15.8-28.3 degrees of both viewing rays, the error living on the pair's baseline
  where neither reprojection nor epipolar distance can see it. The stretch arm at 1.25 is a
  2D-consistent widening in EVERY view, which is a different fault with the same sign. It
  says whether the rule fires on an outward departure; it says nothing about whether demote
  recovers a two-view depth stretch. That question is answered on the real take by B1 and
  B4 and by nothing in this file.
* **The mesh, the rig and the delivery.** This is the capture stage, plus the converter's
  root and pelvis frame, and nothing below them.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_synthetic.py --calibrate
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_synthetic.py

Writes `artifacts/compare/d8c-hip/synthetic.json`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts",
                  "workers/commercial_multiview"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402
import d8_occlusion_synthetic as d8s  # noqa: E402
import d8b_length_synthetic as d8b  # noqa: E402
import temporal  # noqa: E402

OUT = ROOT / "artifacts/compare/d8c-hip/synthetic.json"

# THE ROW UNDER TEST. It is written here as data so this file is identical before and after
# the src change, and so a reader can see exactly what is being confirmed.
HIP_ROW = ("hip_line", "left_hip", "right_hip", ("left_hip", "right_hip"))
# The nine rows D8b shipped, obtained by REMOVING any hip row from the module's own tuple
# rather than by copying it, so this file cannot drift from `src/` and cannot be fooled by
# the src change it is scored across.
TODAY_RULES = tuple(row for row in cm.SEGMENT_LENGTH_RULES if row[0] != HIP_ROW[0])
CANDIDATE_RULES = TODAY_RULES + (HIP_ROW,)

# The cells every arm is scored on: the hips (injected), the knees and the ankles (what the
# solve carries the fault into, and what a cascade would damage).
SCORED_LANDMARKS = ("left_hip", "right_hip", "left_knee", "right_knee",
                    "left_ankle", "right_ankle")
GROUPS = {"hips": ("left_hip", "right_hip"),
          "knees": ("left_knee", "right_knee"),
          "ankles": ("left_ankle", "right_ankle")}
INJECTED = ("left_hip", "right_hip")
ARM_LANDMARKS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist")

CEILINGS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
MODES = ("demote", "reject", "best_ray")
MODE_LABEL = {"demote": "(a) demote, rays kept",
              "reject": "(b) reject, rays withheld",
              "best_ray": "(c) keep the best ray"}

COLLAPSE_FACTOR = 0.5
COLLAPSE_FRAMES = 10
FACTOR_SENSITIVITY = (0.35, 0.5, 0.75)
STRETCH_FACTOR = 1.25

# The reference take's own HONEST hip-line spread, off frames removed, measured by
# `tools/compare/captured_limb_stability.py --reproduce d8c` on the shipped (D8b) delivery's
# RAW array and reproduced there as a card clause. The S0 calibration target, and a
# real-take measurement that selects no shipped constant.
TAKE_HIP_SPREAD = {"subject_01": (-0.082, 0.085), "subject_00": (-0.092, 0.069)}
# The widest of the two performers is the target: a fixture as honest as the EASIER
# performer would be an easier fixture than the take.
TAKE_HIP_TARGET = (-0.092, 0.085)
NOISE_SCALES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)

# S0's ANSWER, MEASURED BEFORE S1 RAN AND WRITTEN DOWN BEFORE ANY SELECTOR NUMBER EXISTED.
# The sweep is committed at `artifacts/compare/d8c-hip/synthetic-noise-calibration.json`.
#
# D8b calibrated its fixture's noise on the LEGS and got 0.20. The hip line is a ~200 mm
# segment where the legs are ~400, so the same pixel noise buys twice the fractional spread,
# and at 0.20 the fixture's own honest hip line reads -9.55 % / +7.96 % at p5-p95 against
# the reference take's -9.2 % / +8.5 %: OUTSIDE, at the p5 end, by 0.35 of a point. Under
# D8b's own calibration rule -- the LARGEST scale whose honest frames stay inside the take's
# -- the answer for the hip line is 0.15, where the fixture reads -8.42 % / +5.96 %.
#
# THIS IS A FIXTURE PARAMETER AND IT IS NOT IN `src/`. The card's instruction is verbatim:
# "if it is wider the fixture is recalibrated as a fixture parameter with src/
# byte-identical across it and never the ceiling." `src/` is byte-identical across this
# recalibration -- it had not been touched at all when the sweep ran. Both scales are run
# and both are in the report (`--noise-scale 0.20` reproduces D8b's), so nothing is hidden
# by the choice: a quieter fixture is an EASIER fixture, and the 0.20 arm is the harder one.
NOISE_SCALE_D8C = 0.15
D8B_NOISE_SCALE_ON_THE_LEGS = 0.20


@contextlib.contextmanager
def segment_rules(value):
    """Rebind `cm.SEGMENT_LENGTH_RULES` for one run, restore it, and ASSERT the restore.

    The pipeline reads the rule list off the module and takes no keyword for it, so this is
    how a candidate row is exercised before it is in `src/`. Restoring is not enough on its
    own: a later arm reading a leaked tuple would silently score the wrong thing, and the
    assertion is what makes that impossible rather than unlikely.
    """

    before = cm.SEGMENT_LENGTH_RULES
    cm.SEGMENT_LENGTH_RULES = tuple(value)
    try:
        yield
    finally:
        cm.SEGMENT_LENGTH_RULES = before
        if cm.SEGMENT_LENGTH_RULES is not before:
            raise SystemExit("the segment rule list was not restored")


def run(cameras, records, rules, **kwargs):
    """One pass of the REAL pipeline with one rule list. `d8s.run` is wrapped, not copied."""

    with segment_rules(rules):
        return d8s.run(cameras, records, **kwargs)


# ------------------------------------------------------------------------- the injection
def hip_collapse_displacements(records, subject: int, frames: list[int], factor: float,
                               sides: tuple[str, ...] = INJECTED) -> dict:
    """Move the hips toward the 2D MIDPOINT OF THE TWO HIPS, in every camera's own pixels.

    `(subject, frame, camera, joint) -> (du, dv)` in the shape `temporal.apply_displacements`
    consumes, which ADDS. The shift is computed from the CLEAN records, so the heavy-tail
    noise lands on top of the collapse exactly as the detector's noise lands on top of its
    bias.

    `sides` is the one-hip arm's only difference: passing a single landmark moves that hip
    and leaves the other exact, which is the 84-86 mode -- and it moves the midpoint, which
    is what makes it a different fault rather than a weaker one.
    """

    displacement: dict = {}
    for camera, rows in enumerate(records):
        for frame in frames:
            people = rows[frame]["people"]
            if subject >= len(people):
                continue
            joints = people[subject]["joints"]
            if "left_hip" not in joints or "right_hip" not in joints:
                continue
            left = np.array([float(joints["left_hip"]["x"]), float(joints["left_hip"]["y"])])
            right = np.array([float(joints["right_hip"]["x"]),
                              float(joints["right_hip"]["y"])])
            midpoint = 0.5 * (left + right)
            for name in sides:
                point = left if name == "left_hip" else right
                shift = (factor - 1.0) * (point - midpoint)
                displacement[(subject, frame, camera, cm.JOINT_INDEX[name])] = (
                    float(shift[0]), float(shift[1]))
    return displacement


def hip_collapse_run(keep: np.ndarray, truth19: np.ndarray, person: int, length: int):
    """The run with the MOST landmark travel among those the conditioning gate leaves alone.

    D8b's `collapse_run_v2` reads its own module globals for which landmarks must be
    supported and which are scored, so this is the hips' version of the same rule rather
    than a monkeypatch of that function. Eligibility is the same and for the same reason:
    both hips must keep three or more supporting views on every frame of the window, or D8's
    conditioning gate would eat the collapse before the length rule ever saw it.
    """

    columns = [cm.JOINT_INDEX[name] for name in INJECTED]
    support = keep[person][:, :, columns].sum(axis=1)
    good = (support >= 3).all(axis=1)
    scored = [cm.JOINT_INDEX[name] for name in SCORED_LANDMARKS]
    best = (-1.0, None, None)
    for start in range(0, len(good) - length + 1):
        if not good[start:start + length].all():
            continue
        travel = []
        for joint in scored:
            anchor = truth19[person, start, joint]
            for frame in range(start, start + length):
                point = truth19[person, frame, joint]
                if np.isfinite(anchor).all() and np.isfinite(point).all():
                    travel.append(float(np.linalg.norm(point - anchor)) * 1000.0)
        if travel and float(np.median(travel)) > best[0]:
            best = (float(np.median(travel)), start, float(np.max(travel)))
    if best[1] is None:
        raise SystemExit(
            f"no window of {length} frames keeps three or more views on both hips. "
            "Refusing to inject a collapse on cells D8's conditioning gate would eat.")
    return list(range(best[1], best[1] + length)), {
        "travel_from_the_runs_first_frame_median_mm": round(best[0], 2),
        "travel_from_the_runs_first_frame_max_mm": round(best[2], 2),
        "the_takes_own_figure_on_110_119_mm": 188.1,
        "what": "distance from the window's first frame to each frame, over the six scored "
                "landmarks, from TRUTH. It is the bound on a frozen limb's error, so it is "
                "the number that says whether a hold-based control can lose at all",
    }


# ------------------------------------------------------------------------------- scoring
def honest_hip_spread(raw: np.ndarray) -> dict:
    """The fixture's own uncollapsed HIP LINE against its own take median, per subject.

    The same statistic `tools/compare/captured_limb_stability.py` reports on the reference
    take, so S0's comparison is like for like.
    """

    out: dict = {}
    worst_low, worst_high, off_total = 0.0, 0.0, 0
    for slot in range(raw.shape[0]):
        a = raw[slot, :, cm.JOINT_INDEX["left_hip"]]
        b = raw[slot, :, cm.JOINT_INDEX["right_hip"]]
        usable = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if int(usable.sum()) < 10:
            out[f"subject_{slot:02d}"] = {"frames_measured": int(usable.sum())}
            continue
        lengths = np.linalg.norm(a[usable] - b[usable], axis=1) * 1000.0
        median = float(np.median(lengths))
        low = float(np.percentile(lengths, 5) / median - 1.0)
        high = float(np.percentile(lengths, 95) / median - 1.0)
        off = int((np.abs(lengths - median) / median > 0.15).sum())
        out[f"subject_{slot:02d}"] = {
            "frames_measured": int(usable.sum()),
            "median_mm": round(median, 2),
            "p5_p95_fraction_of_median": [round(low, 4), round(high, 4)],
            "frames_off_median_by_more_than_15pct": off,
            "worst_fraction_off": round(
                float(np.max(np.abs(lengths - median) / median)), 4)}
        worst_low, worst_high = min(worst_low, low), max(worst_high, high)
        off_total += off
    return {"per_subject": out,
            "worst_p5_fraction": round(worst_low, 4),
            "worst_p95_fraction": round(worst_high, 4),
            "frames_off_by_more_than_15pct": off_total}


def fire_counts(diagnostics) -> list[dict]:
    """Per output slot: what fired, on which segment, on which frames, and the cascade."""

    rows = []
    for row in diagnostics.as_dict()["occlusion_repair"]:
        rows.append({
            "length_rejected_slots": row.get("length_rejected_slots"),
            "length_rejected_by_segment": row.get("length_rejected_by_segment") or {},
            "length_rejected_frames_by_segment": (
                row.get("length_rejected_frames_by_segment") or {}),
            "length_rejected_by_joint": row.get("length_rejected_by_joint") or {},
            "children_marked_under_a_marked_parent_segment": row.get(
                "children_marked_under_a_marked_parent_segment"),
            "demoted_slots": row.get("demoted_slots"),
            "rejected_slots": row.get("rejected_slots"),
            "held_joint_fraction": row.get("held_joint_fraction"),
        })
    return rows


def cascade_under_the_hips(rows: list[dict]) -> dict:
    """Thigh and shin fires split into those under a newly marked HIP and those not.

    B2 on the real take asserts the thigh and shin counts are identical to D8b's "except the
    counted cascade under a newly marked hip". The same split is measured here so the real
    take's number has a synthetic companion rather than standing alone. A thigh fire whose
    frame carries no hip fire is NOT a cascade: it is the new row changing something it was
    not supposed to change.
    """

    out: dict = {}
    for slot, row in enumerate(rows):
        by_segment = row["length_rejected_frames_by_segment"]
        hip_frames = set(by_segment.get("hip_line", []))
        block = {}
        for name in ("left_thigh", "right_thigh", "left_shin", "right_shin"):
            frames = set(by_segment.get(name, []))
            block[name] = {
                "fires": len(frames),
                "under_a_hip_fire": len(frames & hip_frames),
                "not_under_a_hip_fire": sorted(frames - hip_frames),
            }
        out[f"subject_{slot:02d}"] = block
    return out


def converter_delta(candidate: np.ndarray, truth: np.ndarray, frames: list[int]) -> dict:
    """The CONVERTER's own answer on the recovered positions against the same call on truth.

    `positions_to_body_track` is called on both, with identical arguments, so the difference
    is the positions and nothing else. Two quantities, and the card names both:

      * the ROOT TRANSLATION, in millimetres. `_leg_root_offset` puts the rig's leg-root
        midpoint on the captured hip midpoint, so a hip that is recovered in the wrong place
        moves the delivered root even when nothing else does.
      * the HIPS' WORLD ROTATION, in degrees. `Root` is set to identity in the converter's
        own loop, so `Hips`' world rotation IS its local rotation and it is read straight
        off the track. On a still midpoint this is the term that still moves the root, which
        is the mechanism the card pre-registers.

    WHAT IT IS NOT. `spine_world_z_up_m` is None on BOTH sides -- the fixture's SOMA-77
    mapping carries no `Spine1` -- so this exercises the LEGACY trunk-line hips frame, not
    D7's per-frame Kabsch of root, both hips and the spine. The comparison is honest because
    both arms take the same path, and it is NOT a measurement of D7's pelvis frame. The
    delivered pelvis frame is measured on the real take, from the two builds' own bytes, in
    the gate's R3.
    """

    def fill_ears(positions):
        """`positions_to_body_track` requires every cell finite; SOMA-77 has no ears.

        CLAUDE.md: "SOMA-77 has no ears; `left_ear`/`right_ear` are schema-only and
        populated on zero frames." Truth therefore carries NaN there on every frame and the
        converter refuses it. Both sides get the SAME substitution -- each array's own
        `nose` -- so the two calls differ in exactly the joints the injection moved. The
        ears reach only the head and eye joints; neither `root_translation` nor `Hips`'
        world rotation reads them, so neither scored quantity is touched by this.
        """

        out = np.array(positions, dtype=np.float64, copy=True)
        nose = out[:, cm.JOINT_INDEX["nose"]]
        for name in ("left_ear", "right_ear"):
            column = out[:, cm.JOINT_INDEX[name]]
            column[~np.isfinite(column).all(axis=1)] = nose[
                ~np.isfinite(column).all(axis=1)]
        return out

    def track_of(positions):
        return cm.positions_to_body_track(
            fill_ears(positions), sample_rate_hz=d8s.SAMPLE_RATE_HZ,
            provenance_sha256="0" * 64)

    a, b = track_of(candidate), track_of(truth)
    hips = a.joint_names.index("Hips")
    root_a = np.asarray(a.root_translation_m, dtype=np.float64)
    root_b = np.asarray(b.root_translation_m, dtype=np.float64)
    root_mm = np.linalg.norm(root_a - root_b, axis=1) * 1000.0
    quat_a = np.asarray(a.local_rotations_xyzw, dtype=np.float64)[:, hips]
    quat_b = np.asarray(b.local_rotations_xyzw, dtype=np.float64)[:, hips]
    dot = np.abs(np.einsum("ij,ij->i", quat_a, quat_b))
    angle = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def summary(values, index):
        sample = values[index]
        sample = sample[np.isfinite(sample)]
        if not sample.size:
            return {"frames": 0}
        return {"frames": int(sample.size),
                "median": round(float(np.median(sample)), 4),
                "p95": round(float(np.percentile(sample, 95)), 4),
                "max": round(float(sample.max()), 4)}

    everywhere = np.ones(len(root_mm), dtype=bool)
    inside = np.zeros(len(root_mm), dtype=bool)
    inside[frames] = True
    return {
        "root_translation_mm": {"on_the_injected_frames": summary(root_mm, inside),
                                "whole_take": summary(root_mm, everywhere)},
        "hips_world_rotation_deg": {"on_the_injected_frames": summary(angle, inside),
                                    "whole_take": summary(angle, everywhere)},
        "spine_supplied": False,
        "ears_substituted_on_both_sides": "each array's own `nose`; SOMA-77 has no ears and "
                                          "the converter requires finite input. Neither "
                                          "scored quantity reads them",
        "what_frame_this_is": ("the LEGACY trunk-line hips frame -- the fixture has no "
                               "Spine1, so `positions_to_body_track` takes the pre-D7 path "
                               "on BOTH sides. Not D7's Kabsch, and not a measurement of "
                               "the delivered pelvis frame"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--calibrate", action="store_true",
                        help="S0: sweep the fixture's noise scale, report the honest HIP "
                             "LINE spread at each, and select the largest whose honest hip "
                             "line stays inside the reference take's own. Writes the sweep "
                             "and exits without running the selector.")
    parser.add_argument("--noise-scale", type=float, default=None,
                        help="override the fixture noise scale. Defaults to S0's "
                             f"calibrated {NOISE_SCALE_D8C}; pass "
                             f"{D8B_NOISE_SCALE_ON_THE_LEGS} to reproduce D8b's, which is "
                             "the harder fixture and is run and reported too.")
    args = parser.parse_args()

    cameras, records, truth19, names, ground, frames = d8b.build_fixture_v2()
    seen = d8s.real_seen()
    keep = temporal.replayed_keep(seen, frames, names, offset=0)
    offenders = temporal.mask_is_runnable(keep, names)
    injected_person = 0

    # ------------------------------------------------------------------ S0: calibration
    if args.calibrate:
        rows = []
        for candidate in NOISE_SCALES:
            displacement, _ = d8b.scaled_heavy_tail(cameras, frames, names, d8s.NOISE_SEED,
                                                    candidate)
            clean = temporal.apply_keep_mask(
                temporal.apply_displacements(records, displacement), keep)
            _positions, raw, _diag = run(cameras, clean, CANDIDATE_RULES,
                                         segment_length_ceiling_fraction=None)
            hips = honest_hip_spread(raw)
            legs = d8b.honest_leg_spread(raw)
            rows.append({"scale": candidate,
                         "hip_line_worst_p5_fraction": hips["worst_p5_fraction"],
                         "hip_line_worst_p95_fraction": hips["worst_p95_fraction"],
                         "hip_line_frames_off_by_more_than_15pct": hips[
                             "frames_off_by_more_than_15pct"],
                         "hip_line_per_subject": hips["per_subject"],
                         "legs_worst_p5_fraction": legs["legs_worst_p5_fraction"],
                         "legs_worst_p95_fraction": legs["legs_worst_p95_fraction"],
                         "legs_frames_off_by_more_than_15pct": legs[
                             "legs_frames_off_by_more_than_15pct"]})
            print(f"  scale {candidate:5.2f}: hip line p5/p95 "
                  f"{rows[-1]['hip_line_worst_p5_fraction']:+.4f}/"
                  f"{rows[-1]['hip_line_worst_p95_fraction']:+.4f}  off "
                  f"{rows[-1]['hip_line_frames_off_by_more_than_15pct']}   legs "
                  f"{rows[-1]['legs_worst_p5_fraction']:+.4f}/"
                  f"{rows[-1]['legs_worst_p95_fraction']:+.4f}")
        inside = [row for row in rows
                  if row["hip_line_worst_p5_fraction"] >= TAKE_HIP_TARGET[0]
                  and row["hip_line_worst_p95_fraction"] <= TAKE_HIP_TARGET[1]]
        selected = max((row["scale"] for row in inside), default=None)
        payload = {
            "what": "S0. The fixture's own UNCOLLAPSED HIP LINE against its own take "
                    "median, at each candidate noise scale -- the same statistic "
                    "captured_limb_stability.py reports on the reference take",
            "target": {"reference_take_honest_hip_line_p5_p95": TAKE_HIP_SPREAD,
                       "target_used": list(TAKE_HIP_TARGET),
                       "why_the_wider_of_the_two": "a fixture as honest as the EASIER "
                                                   "performer would be an easier fixture "
                                                   "than the take",
                       "source": "artifacts/compare/d8c-hip/limb-stability-d8b.json, the "
                                 "shipped (D8b) delivery's RAW array, reproduced as a card "
                                 "clause"},
            "rule": "the LARGEST scale whose honest hip line stays inside the take's own "
                    "p5-p95 spread -- D8b's own calibration rule, unchanged. A quieter "
                    "fixture is an easier one",
            "d8b_selected_scale_for_the_LEGS": d8b.NOISE_SCALE_V2,
            "why_the_hip_line_needs_its_own": (
                "the hip line is a ~200 mm segment where the legs are ~400, so the same "
                "pixel noise buys twice the fractional spread. At D8b's 0.20 the fixture's "
                "honest hip line is WIDER than the take's; the ceiling is not moved for "
                "that, the fixture is"),
            "candidates": rows,
            "selected_scale": selected,
            "in_src": False,
            "note": "a FIXTURE parameter. It is not in `src/`, it is not a band, and it "
                    "selects no shipped constant. THE CEILING IS NEVER MOVED TO "
                    "ACCOMMODATE A FIXTURE (the card, verbatim: 'if it is wider the "
                    "fixture is recalibrated as a fixture parameter with src/ "
                    "byte-identical across it and never the ceiling').",
        }
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nselected noise scale: {selected}\nwrote {out}")
        return 0 if selected is not None else 1

    scale = args.noise_scale if args.noise_scale is not None else NOISE_SCALE_D8C
    noise_displacement, noise = d8b.scaled_heavy_tail(cameras, frames, names,
                                                      d8s.NOISE_SEED, scale)
    run_frames, travel = hip_collapse_run(keep, truth19, injected_person, COLLAPSE_FRAMES)
    noisy = temporal.apply_displacements(records, noise_displacement)
    clean_masked = temporal.apply_keep_mask(noisy, keep)
    collapse = hip_collapse_displacements(records, injected_person, run_frames,
                                          COLLAPSE_FACTOR)
    collapsed_masked = temporal.apply_keep_mask(
        temporal.apply_displacements(noisy, collapse), keep)

    shipped = float(cm.SEGMENT_LENGTH_CEILING_FRACTION)
    report: dict = {
        "title": "D8c selector -- the hip-line row against synthetic truth, fixture v2",
        "confirms_but_does_not_select": {
            "SEGMENT_LENGTH_CEILING_FRACTION": f"{shipped}, D8b's. Confirmed here on the "
                                               "hips' own geometry, never re-selected.",
            "segment_length_mode": "demote, D8b's, selected on D8b's own fixture. Confirmed "
                                   "here on the hips' own geometry. IF THIS FILE SELECTS "
                                   "OTHERWISE THE STEP STOPS AND IS RE-CARDED -- a "
                                   "per-segment mode or ceiling is a new constant.",
        },
        "selects": ["nothing. This file confirms."],
        "mamma_free": True,
        "rules": {
            "today": [list(row[:3]) + [list(row[3])] for row in TODAY_RULES],
            "candidate": [list(row[:3]) + [list(row[3])] for row in CANDIDATE_RULES],
            "the_one_row_added": list(HIP_ROW[:3]) + [list(HIP_ROW[3])],
            "how": "cm.SEGMENT_LENGTH_RULES is rebound around each run and restored, with "
                   "the restore asserted. TODAY_RULES is defined by REMOVING any hip row "
                   "from the module's own tuple, so this file gives identical numbers "
                   "before and after the src change -- which the gate checks.",
            "src_already_carries_the_row": bool(
                any(row[0] == HIP_ROW[0] for row in cm.SEGMENT_LENGTH_RULES)),
        },
        "fixture": {
            "wrapped": "tools/compare/d8b_length_synthetic.py's fixture v2 "
                       "(build_fixture_v2, scaled_heavy_tail, honest_leg_spread) over "
                       "tools/compare/d8_occlusion_synthetic.py's run/score/real_seen; "
                       "neither file is edited",
            "clips": list(d8b.CLIPS_V2), "stride": d8b.STRIDE_V2, "frames": frames,
            "noise_scale": scale,
            "placement_xy_m": [[round(float(v), 3) for v in row] for row in ground],
            "keep_mask": "temporal.replayed_keep over temporal.real_run_seen_mask",
            "mask_offenders": offenders,
            "noise": noise,
        },
        "injection": {
            "what": "both hip detections moved toward the 2D MIDPOINT OF THE TWO HIPS, "
                    "along the line between them, by a factor, in every camera that sees "
                    "them -- the inter-hip vector scaled; knees, root and everything else "
                    "untouched -- before the noise and before the keep mask",
            "why_not_toward_a_landmark": (
                "on the take the hip line halves while |hip_mid - root| holds (68.0-80.7 mm "
                "against 76.5) and both thighs hold (391.9-421.8 against 401.8 / 406.2). A "
                "collapse toward the pelvis landmark, which sits 76.5 mm off the hip line, "
                "would move both. An injection that lengthens the thighs would make the "
                "thigh rows fire, and on the real run they do not"),
            "fixture_person": injected_person,
            "factor": COLLAPSE_FACTOR,
            "frames": run_frames,
            "frame_count": len(run_frames),
            "landmark_travel_over_the_run": travel,
            "the_other_performer_is_a_control": "the other body is not injected; a hip-line "
                                                "fire on its hips is a false fire",
        },
        "scored_on": {
            "population": "the injected cells: the six scored landmarks of the injected "
                          "performer on the injected frames",
            "landmarks": list(SCORED_LANDMARKS),
            "same_denominator": "the cells are fixed by the injection, which is identical "
                                "across every arm, so no candidate can move the denominator",
        },
    }

    # ------------------------------------------------------------------------- the arms
    today_positions, today_raw, today_diag = run(
        cameras, collapsed_masked, TODAY_RULES,
        segment_length_ceiling_fraction=shipped, segment_length_mode="demote")
    mapping = temporal.pair_subjects(today_positions, truth19)
    report["subject_mapping"] = {str(k): int(v) for k, v in mapping.items()}
    subject = next(k for k, v in mapping.items() if int(v) == injected_person)
    report["injection"]["our_output_slot"] = int(subject)
    report["injection"]["slot_note"] = (
        "the fixture person that was injected, resolved into OUR output slot by "
        "`temporal.pair_subjects`. CLAUDE.md's crossed-subject-map lesson one lane over")

    cells = np.zeros((today_positions.shape[0], today_positions.shape[1],
                      len(SCORED_LANDMARKS)), dtype=bool)
    cells[subject, run_frames, :] = True
    report["scored_on"]["cells"] = int(cells.sum())

    def score(positions):
        row = d8s.score(positions, truth19, mapping, SCORED_LANDMARKS, cells)
        for group, landmarks in GROUPS.items():
            columns = [SCORED_LANDMARKS.index(name) for name in landmarks]
            group_cells = np.zeros_like(cells)
            group_cells[:, :, columns] = cells[:, :, columns]
            row[group] = d8s.score(positions, truth19, mapping, SCORED_LANDMARKS,
                                   group_cells)
        return row

    today_row = score(today_positions)
    today_row["fires"] = fire_counts(today_diag)
    today_row["cascade"] = cascade_under_the_hips(today_row["fires"])
    report["today_D8b"] = today_row

    raw_hip_error = temporal.error_mm(np.nan_to_num(today_raw, nan=np.inf), truth19,
                                      mapping, INJECTED)
    inside = np.zeros(raw_hip_error.shape, dtype=bool)
    inside[subject, run_frames, :] = True
    values = raw_hip_error[inside]
    values = values[np.isfinite(values)]
    report["injection"]["raw_triangulation_error_on_the_injected_hips_mm"] = {
        "cells": int(values.size),
        "median_mm": round(float(np.median(values)), 3) if values.size else None,
        "p95_mm": round(float(np.percentile(values, 95)), 3) if values.size else None,
        "what": "the RAW triangulated hip against exact truth on the injected frames with "
                "no rule of any kind applied -- the size of the fault the arms are trying "
                "to recover from",
    }
    print("injected fault (raw hips): "
          f"{report['injection']['raw_triangulation_error_on_the_injected_hips_mm']}")
    print(f"today (D8b): median {today_row.get('median_mm')} mm on {today_row.get('cells')} "
          f"cells, hips {today_row['hips'].get('median_mm')}")

    sweep: dict = {}
    errors: dict = {}
    for mode in MODES:
        rows = []
        for ceiling in CEILINGS:
            positions, raw, diagnostics = run(
                cameras, collapsed_masked, CANDIDATE_RULES,
                segment_length_ceiling_fraction=ceiling, segment_length_mode=mode)
            row = score(positions)
            row["ceiling_fraction"] = ceiling
            row["fires"] = fire_counts(diagnostics)
            row["cascade"] = cascade_under_the_hips(row["fires"])
            row["raw_bit_identical_to_today"] = bool(
                np.array_equal(raw, today_raw, equal_nan=True))
            row["hip_fires_outside_the_injected_run"] = {
                f"subject_{slot:02d}": [
                    frame for frame
                    in fires["length_rejected_frames_by_segment"].get("hip_line", [])
                    if slot != subject or frame not in set(run_frames)]
                for slot, fires in enumerate(row["fires"])}
            rows.append(row)
            errors[(mode, ceiling)] = temporal.error_mm(positions, truth19, mapping,
                                                        SCORED_LANDMARKS)
            print(f"  {mode:9s} ceiling {ceiling:5.3f}: median {row.get('median_mm')} mm, "
                  f"hips {row['hips'].get('median_mm')}, "
                  f"fires {[r['length_rejected_slots'] for r in row['fires']]}, "
                  f"held {[r['held_joint_fraction'] for r in row['fires']]}")
        sweep[mode] = {"label": MODE_LABEL[mode], "candidates": rows}
    report["sweep"] = sweep

    # ----------------------------------------------------- the selector, at one ceiling
    at_shipped = {mode: next(row for row in sweep[mode]["candidates"]
                             if abs(row["ceiling_fraction"] - shipped) < 1e-9)
                  for mode in MODES}
    today_errors = temporal.error_mm(today_positions, truth19, mapping, SCORED_LANDMARKS)
    hip_cells = np.zeros_like(cells)
    hip_columns = [SCORED_LANDMARKS.index(name) for name in INJECTED]
    hip_cells[:, :, hip_columns] = cells[:, :, hip_columns]
    margins = {}
    for mode in MODES:
        margins[f"{mode}_vs_today"] = temporal.block_bootstrap_pair(
            temporal.by_frame(errors[(mode, shipped)], hip_cells),
            temporal.by_frame(today_errors, hip_cells))
    for mode in ("reject", "best_ray"):
        margins[f"{mode}_vs_demote"] = temporal.block_bootstrap_pair(
            temporal.by_frame(errors[(mode, shipped)], hip_cells),
            temporal.by_frame(errors[("demote", shipped)], hip_cells))
    best_on_hips = min(MODES, key=lambda m: at_shipped[m]["hips"].get("median_mm", 1e9))
    best_pooled = min(MODES, key=lambda m: at_shipped[m].get("median_mm", 1e9))
    beats_today = bool(at_shipped[best_on_hips]["hips"].get("median_mm", 1e9)
                       < today_row["hips"].get("median_mm", 0.0))
    report["selector"] = {
        "rule": "lowest median 3D error on the INJECTED HIPS at the shipped ceiling, with "
                "the paired margins beside it on identical draws. D8b's fixture-v2 reading "
                "(the collapsed cell, not the pooled median), fixed before the numbers",
        "shipped_ceiling_fraction": shipped,
        "today_hips_median_mm": today_row["hips"].get("median_mm"),
        "per_mode_hips_median_mm": {mode: at_shipped[mode]["hips"].get("median_mm")
                                    for mode in MODES},
        "per_mode_pooled_median_mm": {mode: at_shipped[mode].get("median_mm")
                                      for mode in MODES},
        "per_group_median_mm": {
            group: {"today": today_row[group].get("median_mm"),
                    **{mode: at_shipped[mode][group].get("median_mm") for mode in MODES}}
            for group in GROUPS},
        "selected_mode": best_on_hips,
        "selected_mode_on_the_pooled_median": best_pooled,
        "the_two_readings_agree": bool(best_on_hips == best_pooled),
        "beats_today_on_the_hips": beats_today,
        "beats_today_on_the_pooled_median": bool(
            at_shipped[best_pooled].get("median_mm", 1e9) < today_row.get("median_mm", 0.0)),
        "selects_the_shipped_mode": bool(best_on_hips == "demote"),
        "stop_condition": ("if `selects_the_shipped_mode` is false the step STOPS and is "
                           "re-carded: a per-segment mode is a new constant and this file "
                           "may not register one"),
        "ceiling_argmin_on_the_hips": None,
        "ceilings_tied_at_the_argmin": None,
        "ceiling_note": ("the ceiling is CONFIRMED by 'fires on the injected run and not "
                         "outside it', not by the sweep's argmin. The argmin is reported "
                         "and moves nothing -- a ceiling chosen to minimise error on an "
                         "injected fault is a ceiling fitted to the injection"),
        "paired_margins": {
            "method": "temporal.block_bootstrap_pair -- I7's own moving-block bootstrap, "
                      "both arms resampled on IDENTICAL frame blocks. Per-frame agreement "
                      "in this lane has lag-1 autocorrelation 0.99, so ordinary resampling "
                      "is invalid (CLAUDE.md)",
            "scored_on": "the injected HIPS only",
            "pairs": margins,
        },
    }
    hips_by_ceiling = {c: next(row for row in sweep[best_on_hips]["candidates"]
                               if abs(row["ceiling_fraction"] - c) < 1e-9)["hips"].get(
                                   "median_mm", 1e9) for c in CEILINGS}
    lowest = min(hips_by_ceiling.values())
    report["selector"]["ceiling_argmin_on_the_hips"] = [
        c for c, v in hips_by_ceiling.items() if abs(v - lowest) < 1e-9]
    report["selector"]["ceilings_tied_at_the_argmin"] = {
        "hips_median_mm_by_ceiling": hips_by_ceiling,
        "note": "the sweep TIES across the top of the range: above a certain ceiling the "
                "same cells fire and the answer stops changing. Reporting one value as "
                "'the argmin' would quote a selection the sweep did not make"}
    print(f"\nselector (hips): today {today_row['hips'].get('median_mm')}; "
          + ", ".join(f"{m} {at_shipped[m]['hips'].get('median_mm')}" for m in MODES)
          + f" -> {best_on_hips}")

    # -------------------------------------------------------- the converter, S1's second half
    truth_slot = truth19[mapping[subject]]
    report["converter"] = {
        "what": "`positions_to_body_track` on the recovered positions against the SAME call "
                "on exact truth: the root translation and the hips' world rotation, per "
                "frame. The card requires it because a recovery that leaves the hips on the "
                "inward rays still moves the delivered root -- `_leg_root_offset` puts the "
                "rig's leg-root midpoint on the captured hip midpoint",
        "today_D8b": converter_delta(today_positions[subject], truth_slot, run_frames),
    }

    # The per-mode deltas need the POSITIONS, which the sweep did not retain, so each mode
    # is re-run at the shipped ceiling only -- the one ceiling the card asks the converter
    # question at. Re-running is cheap and it is the same code path; retaining eight
    # position arrays per mode is not.
    per_mode_converter = {}
    for mode in MODES:
        positions, _raw, _diag = run(
            cameras, collapsed_masked, CANDIDATE_RULES,
            segment_length_ceiling_fraction=shipped, segment_length_mode=mode)
        per_mode_converter[mode] = converter_delta(positions[subject], truth_slot,
                                                   run_frames)
    report["converter"]["per_mode"] = per_mode_converter

    # ------------------------------------------------------ S2, oracle 1: clean, fully seen
    clean = temporal.apply_keep_mask(records, np.ones_like(keep))
    oracle_today, oracle_raw_today, _ = run(cameras, clean, TODAY_RULES,
                                            segment_length_ceiling_fraction=shipped,
                                            segment_length_mode="demote")
    oracle_candidate, oracle_raw_candidate, oracle_diag = run(
        cameras, clean, CANDIDATE_RULES, segment_length_ceiling_fraction=shipped,
        segment_length_mode=best_on_hips)
    rows = fire_counts(oracle_diag)
    block = {
        "what": "every camera sees every landmark on every frame, no noise, no collapse. "
                "The hip row must do NOTHING: zero fires and output bit-identical to "
                "today's code",
        "length_rejected_slots": [row["length_rejected_slots"] for row in rows],
        "length_rejected_by_segment": [row["length_rejected_by_segment"] for row in rows],
        "hip_line_fires": [len(row["length_rejected_frames_by_segment"].get("hip_line", []))
                           for row in rows],
        "smoothed_bit_identical": bool(np.array_equal(oracle_today, oracle_candidate)),
        "raw_bit_identical": bool(np.array_equal(oracle_raw_today, oracle_raw_candidate,
                                                 equal_nan=True)),
    }
    block["passes"] = bool(sum(row or 0 for row in block["length_rejected_slots"]) == 0
                           and block["smoothed_bit_identical"]
                           and block["raw_bit_identical"])
    report["S2_oracle_clean_fully_seen"] = block

    # ------------------------------- S2, oracle 2: the collapsed clip's honest frames
    outside = at_shipped[best_on_hips]["hip_fires_outside_the_injected_run"]
    report["S2_oracle_uncollapsed_frames"] = {
        "what": "on the SAME collapsed clip, the frames outside the injected run must draw "
                "zero HIP-LINE fires -- a ceiling that fires on honest motion is a smoother "
                "wearing a reject's clothes",
        "mode": best_on_hips,
        "ceiling_fraction": shipped,
        "hip_fires_outside_the_run": outside,
        "run_frames": run_frames,
        "passes": not any(outside.values()),
        "all_length_fires_outside_the_run_any_segment": {
            f"subject_{slot:02d}": {
                name: [f for f in values
                       if slot != subject or f not in set(run_frames)]
                for name, values in fires["length_rejected_frames_by_segment"].items()
                if [f for f in values if slot != subject or f not in set(run_frames)]}
            for slot, fires in enumerate(at_shipped[best_on_hips]["fires"])},
    }

    # ---------------------------- S3, must-fail 1: a leg, arm or shoulder fire on the clean clip
    clean_rows = []
    for ceiling in CEILINGS:
        _positions, _raw, diagnostics = run(
            cameras, clean_masked, CANDIDATE_RULES,
            segment_length_ceiling_fraction=ceiling, segment_length_mode=best_on_hips)
        fires = fire_counts(diagnostics)
        by_segment: dict = {}
        for row in fires:
            for name, count in row["length_rejected_by_segment"].items():
                by_segment[name] = by_segment.get(name, 0) + count
        forbidden = sum(count for name, count in by_segment.items() if name != "hip_line")
        clean_rows.append({
            "ceiling_fraction": ceiling,
            "fires_by_segment": by_segment,
            "forbidden_fires_leg_arm_or_shoulder": forbidden,
            "hip_line_fires": by_segment.get("hip_line", 0),
            "all_length_rejects": [row["length_rejected_slots"] for row in fires],
        })
        print(f"  clean at {ceiling:5.3f}: forbidden {forbidden}, "
              f"hip line {by_segment.get('hip_line', 0)}")
    shipped_row = next(row for row in clean_rows
                       if abs(row["ceiling_fraction"] - shipped) < 1e-9)
    tight = [row for row in clean_rows if row["forbidden_fires_leg_arm_or_shoulder"] > 0]
    report["S3_must_fail_a_forbidden_fire_on_the_clean_clip"] = {
        "what": "the CLEAN arm is the NOISY, UNCOLLAPSED clip -- on the noise-free fixture "
                "every length is exactly constant and no ceiling could ever fire, so the "
                "control could not be demonstrated there. Any LEG, ARM or SHOULDER-LINE "
                "fire at the SHIPPED ceiling is a FAIL; a tight ceiling firing is the "
                "demonstration that this is not a band a constant can pass",
        "the_honest_hip_line_at_0_15_is_the_NEW_EXPOSURE": (
            "the card says so in as many words. A hip-line fire on the clean clip is "
            "REPORTED, not a fail: it is the cost of the row and the number belongs in the "
            "review"),
        "by_ceiling": clean_rows,
        "forbidden_fires_at_the_shipped_ceiling": shipped_row[
            "forbidden_fires_leg_arm_or_shoulder"],
        "hip_line_fires_at_the_shipped_ceiling": shipped_row["hip_line_fires"],
        "tight_ceilings_that_do_fire_forbidden_segments": [
            row["ceiling_fraction"] for row in tight],
        "passes": bool(shipped_row["forbidden_fires_leg_arm_or_shoulder"] == 0),
        "degenerate_demonstrated": bool(tight),
    }

    # ---------------------------------------- S3, must-fail 2: the whole-take hold
    candidate_positions, _raw, _diag = run(
        cameras, collapsed_masked, CANDIDATE_RULES,
        segment_length_ceiling_fraction=shipped, segment_length_mode=best_on_hips)
    frozen = candidate_positions.copy()
    for name in SCORED_LANDMARKS:
        frozen[:, :, cm.JOINT_INDEX[name]] = frozen[:, 0:1, cm.JOINT_INDEX[name]]
    def travel_of(landmarks):
        values = []
        for name in landmarks:
            joint = cm.JOINT_INDEX[name]
            series = truth19[mapping[subject], :, joint]
            for frame in run_frames:
                if np.isfinite(series[frame]).all() and np.isfinite(series[0]).all():
                    values.append(float(np.linalg.norm(series[frame] - series[0])) * 1000.0)
        if not values:
            return {"median": None, "max": None}
        return {"median": round(float(np.median(values)), 2),
                "max": round(float(np.max(values)), 2)}

    travel_values = {"pooled": travel_of(SCORED_LANDMARKS),
                     **{group: travel_of(landmarks)
                        for group, landmarks in GROUPS.items()}}
    frozen_score = score(frozen)
    report["S3_must_fail_whole_take_hold"] = {
        "what": "every scored landmark held at its own first frame for the whole take -- "
                "the degenerate any hold-based recovery must never become",
        "score": frozen_score,
        "shipped_score": at_shipped[best_on_hips],
        "how_far_the_landmarks_actually_travel_over_the_run_mm": travel_values,
        "and_the_pooled_figure_is_the_wrong_one_to_read": (
            "distance from frame 0's truth to each injected frame's truth, and it MUST be "
            "read per group. Pooled it is 494 mm, which is the ANKLES and KNEES of a squat "
            "clip swinging; THE HIPS TRAVEL 13 mm over the same frames. A frozen limb's "
            "error cannot exceed its own travel, so on the hips -- the cells the selector "
            "reads -- this control is WEAK by construction: it can only ever reach about "
            "13 mm, and it does (12.2). It fails as required against demote's 3.9, and it "
            "would fail against almost anything, so no claim rests on it. What carries the "
            "mode selection is the sweep and the two oracles"),
        "fails_as_required_on_the_hips": bool(
            frozen_score["hips"].get("median_mm", 0.0)
            > at_shipped[best_on_hips]["hips"].get("median_mm", 1e9)),
        "fails_as_required_on_the_pooled_median": bool(
            frozen_score.get("median_mm", 0.0)
            > at_shipped[best_on_hips].get("median_mm", 1e9)),
    }

    # --------------------------------------------------- the injection variants, REPORTED
    variants = []
    for label, factor, sides in (
            ("factor_0.35", 0.35, INJECTED), ("factor_0.50", COLLAPSE_FACTOR, INJECTED),
            ("factor_0.75", 0.75, INJECTED),
            ("one_hip_left_only_0.50", COLLAPSE_FACTOR, ("left_hip",)),
            ("stretch_1.25", STRETCH_FACTOR, INJECTED)):
        variant = temporal.apply_keep_mask(
            temporal.apply_displacements(
                noisy, hip_collapse_displacements(records, injected_person, run_frames,
                                                  factor, sides)), keep)
        row = {"label": label, "factor": factor, "sides": list(sides), "arms": {},
               "hip_fires": {}}
        positions, _raw, _diag = run(cameras, variant, TODAY_RULES,
                                     segment_length_ceiling_fraction=shipped,
                                     segment_length_mode="demote")
        base = d8s.score(positions, truth19, mapping, SCORED_LANDMARKS, hip_cells)
        row["today_hips_median_mm"] = base.get("median_mm")
        for mode in MODES:
            positions, _raw, diagnostics = run(
                cameras, variant, CANDIDATE_RULES,
                segment_length_ceiling_fraction=shipped, segment_length_mode=mode)
            row["arms"][mode] = d8s.score(positions, truth19, mapping, SCORED_LANDMARKS,
                                          hip_cells).get("median_mm")
            fires = fire_counts(diagnostics)
            row["hip_fires"][mode] = [
                len(f["length_rejected_frames_by_segment"].get("hip_line", []))
                for f in fires]
            if mode == "demote":
                row["converter"] = converter_delta(positions[subject], truth_slot,
                                                   run_frames)
                row["cascade"] = cascade_under_the_hips(fires)
        variants.append(row)
        print(f"  variant {label}: today {row['today_hips_median_mm']}, {row['arms']}")
    report["injection_variants"] = {
        "what": "the card's factor range, the ONE-HIP arm (the 84-86 mode) and the STRETCH "
                "arm (class (ii)'s sign), all at the shipped ceiling and all REPORTED. The "
                "selector runs on ONE factor so the three modes share one population",
        "the_stretch_arm_is_not_class_ii": (
            "a 2D-consistent widening in every view is not a two-view depth stretch. It "
            "says whether the symmetric rule fires on an outward departure; it says nothing "
            "about whether demote recovers a stretch that lives on the A-C baseline"),
        "rows": variants,
    }

    # -------------------------------------------------- the raw array, never modified
    report["raw_array_untouched"] = {
        "what": "the raw triangulation is captured before every rule and must be "
                "bit-identical between today's code and each candidate on the SAME input",
        "per_mode": {mode: all(row["raw_bit_identical_to_today"]
                               for row in sweep[mode]["candidates"]) for mode in MODES},
    }
    report["raw_array_untouched"]["bit_identical"] = all(
        report["raw_array_untouched"]["per_mode"].values())

    # ------------------------------------------- the fixture's own honest spread, S0's record
    _clean_positions, clean_raw, _clean_diag = run(
        cameras, clean_masked, CANDIDATE_RULES, segment_length_ceiling_fraction=None)
    report["S0_fixture_honest_hip_line"] = {
        "what": "the fixture's own UNCOLLAPSED hip line against its own take median, at the "
                "noise scale this run used -- the same statistic the instrument reports on "
                "the reference take",
        "noise_scale": scale,
        "measured": honest_hip_spread(clean_raw),
        "reference_take": TAKE_HIP_SPREAD,
        "target_used": list(TAKE_HIP_TARGET),
        "inside_the_takes_own_spread": None,
        "legs_for_comparison": d8b.honest_leg_spread(clean_raw),
    }
    measured = report["S0_fixture_honest_hip_line"]["measured"]
    report["S0_fixture_honest_hip_line"]["inside_the_takes_own_spread"] = bool(
        measured["worst_p5_fraction"] >= TAKE_HIP_TARGET[0]
        and measured["worst_p95_fraction"] <= TAKE_HIP_TARGET[1])

    report["verdict"] = "PASS" if (
        report["selector"]["selects_the_shipped_mode"]
        and report["selector"]["beats_today_on_the_hips"]
        and report["S2_oracle_clean_fully_seen"]["passes"]
        and report["S2_oracle_uncollapsed_frames"]["passes"]
        and report["S3_must_fail_a_forbidden_fire_on_the_clean_clip"]["passes"]
        and report["S3_must_fail_whole_take_hold"]["fails_as_required_on_the_hips"]
        and report["raw_array_untouched"]["bit_identical"]
    ) else "FAIL"

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nS0 fixture honest hip line:",
          json.dumps(report["S0_fixture_honest_hip_line"]["measured"]["per_subject"]))
    print("S0 inside the take's own spread:",
          report["S0_fixture_honest_hip_line"]["inside_the_takes_own_spread"])
    print("S1 selector :", json.dumps(report["selector"]["per_mode_hips_median_mm"]),
          "->", report["selector"]["selected_mode"],
          "| shipped mode selected:", report["selector"]["selects_the_shipped_mode"],
          "| beats today:", report["selector"]["beats_today_on_the_hips"])
    print("S2 oracle 1 :", report["S2_oracle_clean_fully_seen"]["passes"])
    print("S2 oracle 2 :", report["S2_oracle_uncollapsed_frames"]["passes"])
    print("S3 forbidden:", report["S3_must_fail_a_forbidden_fire_on_the_clean_clip"]["passes"],
          "degenerate demonstrated:",
          report["S3_must_fail_a_forbidden_fire_on_the_clean_clip"]["degenerate_demonstrated"])
    print("S3 frozen   :", report["S3_must_fail_whole_take_hold"]["fails_as_required_on_the_hips"])
    print("raw         :", report["raw_array_untouched"]["bit_identical"])
    print(f"\nverdict: {report['verdict']}\nwrote {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
