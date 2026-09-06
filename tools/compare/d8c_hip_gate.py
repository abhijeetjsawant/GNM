#!/usr/bin/env python3
"""D8c's gate: every band from the card, as its own keyed block, applied mechanically.

The card's bands, `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8c row, committed at
85b8113 BEFORE this branch's first instrument ran. They are carried into the report verbatim
under `preregistration` and each is answered in its own block:

    hygiene             today's code on the same inputs byte-identical to the shipped
                        delivery, 8 of 8, BEFORE any src change
    instrument first    `captured_limb_stability.py --reproduce d8c` reproduces the card's
                        figures on the shipped (D8b) build, before any src change
    SYNTHETIC           S0 the fixture's own honest hip line; S1 the selector; S2 the two
                        oracles; S3 the two must-fails
    B1                  part-wise silhouette, TORSO+LEGS, whole take AND the three runs
    B2                  the raw array byte-identical; every arm and shoulder-line fire count
                        IDENTICAL; thigh and shin identical except the counted cascade
    B3                  `delivered_vs_capture` against the RAW points and the repaired ones
    B4                  the frames-off count on a FIXED denominator, and the moving one
    B5                  MAMMA's joints on the three runs, hips and knees, REPORT ONLY
    B6                  `d3_skeleton_gate.py` bit-identical to post-merge-D8b -- a
                        SRC-HYGIENE TRIPWIRE, not a band on this rule
    R1-R4               the delivered root, the displacement envelope, the pelvis frame and
                        the hoist, from the two builds' own bytes. REPORTS with predictions

    merge rule, fixed before numbers: S1 selects `demote` at 0.15 on the hips' own geometry
    AND S2 both clauses AND S3 AND B1 on both performers AND B2 AND B4 on the fixed
    denominator AND B6; everything else reports

THE ONE THING EVERY READER MUST CARRY. The hip row sits between the raw triangulation and
the smoothed landmarks, the same place D8's three rules and D8b's nine sit, so **everything
downstream that compares two builds' SMOOTHED landmarks loses its byte-identical denominator
by design**. The RAW array is the fixed reference and B2 asserts it. Two blocks EXPECT to
report CHANGED and say so in their own text: `delivered_vs_capture`'s figures on performer 1,
and B1's smoothed-identity line. A CHANGED there is the step working.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_gate.py

Writes `artifacts/compare/d8c-hip/gate.json`. NOTHING is written under
`artifacts/commercial-multiview-soma77/`.
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

OUT_DIR = ROOT / "artifacts/compare/d8c-hip"
REPORT = OUT_DIR / "gate.json"
D8B_DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
D8C_DELIVERY = OUT_DIR / "delivery"
POST_MERGE_D8B = ROOT / "artifacts/compare/post-merge-D8b"
LOGS = OUT_DIR / "logs"
FIRST_FRAME_ID = 60

PREREGISTRATION = """\
**hygiene, before any src change:** today's code on the same inputs byte-identical to the
shipped delivery (8 of 8)

**instrument first:** `captured_limb_stability.py --reproduce d8c` must reproduce 30 raw /
23 smoothed frames off in the three runs, 124.9 mm raw / 139.7 smoothed at the minimum,
frame 113's 355 mm raw, the per-camera table on the HIPS and the root over 110-119 (A-C-D,
D 0.43-0.93, <= 8.1 px) and 158-168 (A and C only, D absent, 139-140 deg, <= 3.8 px), the
baseline split on 158-168 (along 231-275 / across 18-52 against honest 181 / 117) and the
honest-mask root->hip table above -- the classification landmarks become a flag rather than
a constant -- and repair the window block's frame ids, which are labelled by index offset
(it prints 60-94 for frames 85-119), an instrument defect found while writing this card

**SYNTHETIC on the REPAIRED fixture v2 (D8b's: amplitude scale 0.20, the squat clip at
stride 2), selector and bands, in this order:** (S0) the fixture's own honest hip line
measured FIRST against the take's -8.2 / +8.5 %; if it is wider the fixture is recalibrated
as a fixture parameter with `src/` byte-identical across it (the D8b precedent) and never
the ceiling; (S1) the injection reproduces the MEASURED mode, not D8b's: in every view that
sees them, both hip detections moved toward the 2D midpoint of the two hips along the line
between them by factor 0.5 (the inter-hip vector scaled; knees, root and everything else
untouched), for a run of 8-15 frames where both hips keep three or more supporting views,
before the noise and the keep mask; sensitivity at 0.35 and 0.75; a ONE-HIP arm (the 84-86
mode: one hip moved, the other exact) and a STRETCH arm at 1.25 (class (ii)) REPORTED beside
it; scored as 3D error against exact truth on the hips, the knees and the ankles AND through
the converter -- `positions_to_body_track` on the recovered positions against the same call
on truth: the root translation and the `R_hips` angle per frame -- for today's code, demote,
reject and best_ray at the shipped ceiling and the sweep 0.05-0.30; the thigh and shin fire
counts on the injected clip reported with the cascade separated (on the real run the thighs
never fired, and an injection that lengthens them is the wrong injection); (S2) oracle 1:
clean input -> zero fires and output bit-identical to today; oracle 2: the collapsed clip's
un-collapsed frames -> zero hip fires; (S3) must-fail: any LEG, ARM or shoulder-line fire on
the clean fixture at 0.15 = FAIL (the honest hip line at 0.15 is the new exposure), and a
whole-take hold

**REAL TAKE, bands the candidate cannot optimise:** (B1) the photographs: part-wise
silhouette, the TORSO+LEGS part, whole take AND the three runs (83-87, 109-119, 158-168) as
their own cut, not worse on either performer with the CI clear on the D7b/D8 predicate
(upper bound >= 0), improvement predicted on performer 1's runs and reported with its CI;
(B2) raw array byte-identical; every ARM and shoulder-line fire count IDENTICAL to D8b's,
and the THIGH and SHIN fire counts identical except the counted cascade under a newly marked
hip (the rule is order-independent within itself and the new row charges only the hips: any
other change means the addition did more than add a segment); (B3) `delivered_vs_capture
--reference raw` and vs the repaired points, D8b vs D8c, identical draws, hips, knees,
ankles and the root, report; R1-R4 above; (B4) captured hip line frames off > 15 % on a
FIXED denominator -- D8b's own medians, the `b4-fixed-denominator` pattern -- 23 -> <= 4
(performer 1) and 0 -> 0 (performer 0), the moving-denominator reading beside it; the
candidate optimises this directly, so it is IN THE MERGE RULE ONLY PAIRED with B1 (a
recovery that leaves the hips on the inward rays passes everything else and fails this; a
smoothing smear that moves the honest frames toward a new median fails it on the fixed
medians); must-fail: the D8b build reads 23; (B5) MAMMA's joints through `subject_map` on
the three runs, hips and knees, report only, predicted closer on run (i); (B6)
`d3_skeleton_gate.py` rerun, every figure bit-identical to `artifacts/compare/post-merge-D8b/`
(legs 0.05-0.07, arms 0.82-1.18, torso 8-11 mm) -- a SRC-HYGIENE TRIPWIRE, not a band on the
rule: the gate feeds truth to the converter and never runs the triangulation path, so the
length rule never executes there; the honest oracle for this row is S2; the head gate and
D7b's neck figure rerun and reported

**THE ROOT'S DEPENDENCE ON THE HIP MIDPOINT, pre-registered (mechanism first, numbers as
predictions only):** `_leg_root_offset` (D2b) puts the rig's leg-root midpoint on the
captured hip midpoint through `R_hips`, and `R_hips` is the D7 Kabsch of root, both hips and
the spine, per frame (`PELVIS_SMOOTHING_FRAMES` = 0) -- so a recovered hip line moves the
delivered root even when the midpoint stays put, by rotating the pelvis frame; the thighs
are aimed from the hips; `project_generated_foot_contacts` then translates the whole body.
What carries a change to unfired frames is `_fill_and_smooth_positions`' Savitzky-Golay
window (`SMOOTHING_WINDOW_FRAMES` = 9, +/-4 frames), not the 6-frame gap clause (that is the
hold for slots the solve never saw; demote keeps rays, so fired slots are candidates).
Registered as REPORTS with predictions, none of them a fail clause: (R1) D8b vs D8c root and
hips from the two GLBs' own bytes, AFTER subtracting each frame's own hoist, so a re-hoist is
not read as a body-wide move (prediction: the root's median move on the fired frames under
20 mm, the worst under half the worst width deficit, 45 mm); (R2) every landmark's
displacement plotted against its distance to the nearest RAW hip-line fire, with +/-4 frames
as the expected envelope -- frames outside it that move more than 1 mm are listed
(prediction: none beyond the envelope; if there are, the addition did more than add a
segment and the review says what); (R3) the per-frame `R_hips` delta (degrees) on the fired
frames, because a still midpoint can still move the root; (R4) the hoist per frame before
and after (its re-aim is the next step, not this one).

**Ships:** one row in `SEGMENT_LENGTH_RULES` -- `("hip_line", "left_hip", "right_hip",
("left_hip", "right_hip"))`; nothing else in `src/`; the ceiling (0.15) and the mode
(`demote`) are D8b's and are not re-selected -- the synthetic CONFIRMS them on the hips' own
geometry, and if it selects otherwise the step STOPS and is re-carded, because a per-segment
mode or ceiling is a new constant. No window, no new constant, one provenance entry
(`SEGMENT_LENGTH_RULES`'s open remedy closes; the ceiling's stays open).

**merge rule, fixed before numbers:** S1 selects `demote` at 0.15 on the hips' own geometry
AND S2 both clauses AND S3 AND B1 on both performers AND B2 AND B4 on the fixed denominator
AND B6; everything else reports
"""


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_report(directory: Path) -> dict:
    return json.loads((directory / "run-report.json").read_text())


def fires(directory: Path) -> list[dict]:
    rows = run_report(directory)["occlusion_repair"]
    return [{
        "length_rejected_slots": row.get("length_rejected_slots"),
        "length_rejected_by_segment": row.get("length_rejected_by_segment") or {},
        "length_rejected_frames_by_segment": row.get("length_rejected_frames_by_segment")
                                             or {},
        "length_rejected_by_joint": row.get("length_rejected_by_joint") or {},
        "cascade": row.get("children_marked_under_a_marked_parent_segment"),
        "segment_frames_measured": row.get("segment_frames_measured") or {},
        "segment_median_m": row.get("segment_median_m") or {},
    } for row in rows]


# --------------------------------------------------------------------- what the rule saw
def rule_visibility() -> dict:
    """The 30 raw off frames against the 18 the rule actually fired on, and why.

    CLAUDE.md, and the brief for this step repeats it: the rule runs on the array AFTER D8's
    conditioning gate has withheld slots, so a slot that is already NaN is INVISIBLE to it.
    A count of fires is therefore not a count of defects, and the difference is here rather
    than left to be discovered.
    """

    out: dict = {}
    after = fires(D8C_DELIVERY)
    for subject in (0, 1):
        with np.load(D8B_DELIVERY / f"subject-{subject:02d}.body-track.npz") as archive:
            raw = np.asarray(archive["raw_triangulated_world_positions_z_up_m"],
                             dtype=np.float64)
        a = raw[:, cm.JOINT_INDEX["left_hip"]]
        b = raw[:, cm.JOINT_INDEX["right_hip"]]
        usable = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        lengths = np.full(len(raw), np.nan)
        lengths[usable] = np.linalg.norm(a[usable] - b[usable], axis=1)
        median = float(np.median(lengths[usable]))
        off = np.flatnonzero(usable & (np.abs(lengths - median) / median > 0.15)).tolist()
        fired = after[subject]["length_rejected_frames_by_segment"].get("hip_line", [])
        measured = after[subject]["segment_frames_measured"].get("hip_line")
        out[f"subject_{subject:02d}"] = {
            "raw_hip_line_off_frames": len(off),
            "raw_hip_line_off_frame_ids": [f + FIRST_FRAME_ID for f in off],
            "frames_the_rule_could_measure": measured,
            "frames_of_150_the_rule_could_NOT_measure": 150 - int(measured or 0),
            "the_rule_fired_on": len(fired),
            "the_rule_fired_on_frame_ids": [f + FIRST_FRAME_ID for f in fired],
            "raw_off_frames_the_rule_did_NOT_fire_on": [
                f + FIRST_FRAME_ID for f in off if f not in set(fired)],
            "the_rules_own_reference_median_mm": round(
                1000.0 * float(after[subject]["segment_median_m"].get("hip_line", 0.0)), 3),
            "the_raw_arrays_own_median_mm": round(median * 1000.0, 3),
        }
    out["how_to_read_it"] = (
        "the rule runs on the array AFTER D8's conditioning and reachability gates have "
        "withheld slots, so a hip already NaN there is INVISIBLE to it and cannot be "
        "counted as a miss. It also runs on a slightly different population than the raw "
        "array, so its own median differs from the raw one and a frame can be off by 15 % "
        "of one and not the other. Both medians and both frame lists are here.")
    return out


# ---------------------------------------------------------------------------- B2
def b2() -> dict:
    before, after = fires(D8B_DELIVERY), fires(D8C_DELIVERY)
    raw_identical = {}
    for subject in (0, 1):
        with np.load(D8B_DELIVERY / f"subject-{subject:02d}.body-track.npz") as a, \
             np.load(D8C_DELIVERY / f"subject-{subject:02d}.body-track.npz") as b:
            raw_identical[f"subject_{subject:02d}"] = bool(np.array_equal(
                a["raw_triangulated_world_positions_z_up_m"],
                b["raw_triangulated_world_positions_z_up_m"], equal_nan=True))
    other = ("shoulder_line", "left_upper_arm", "right_upper_arm", "left_forearm",
             "right_forearm")
    legs = ("left_thigh", "right_thigh", "left_shin", "right_shin")
    rows: dict = {}
    identical_everywhere = True
    for subject in (0, 1):
        block = {}
        for name in other + legs:
            was = before[subject]["length_rejected_frames_by_segment"].get(name, [])
            now = after[subject]["length_rejected_frames_by_segment"].get(name, [])
            same = was == now
            identical_everywhere = identical_everywhere and same
            block[name] = {"D8b_frames": [f + FIRST_FRAME_ID for f in was],
                           "D8c_frames": [f + FIRST_FRAME_ID for f in now],
                           "identical": same,
                           "banded": "arm/shoulder: IDENTICAL" if name in other
                                     else "leg: identical except a counted cascade"}
        block["hip_line_the_new_row"] = {
            "D8b_frames": [f + FIRST_FRAME_ID for f in
                           before[subject]["length_rejected_frames_by_segment"].get(
                               "hip_line", [])],
            "D8c_frames": [f + FIRST_FRAME_ID for f in
                           after[subject]["length_rejected_frames_by_segment"].get(
                               "hip_line", [])]}
        block["cascade"] = {"D8b": before[subject]["cascade"],
                            "D8c": after[subject]["cascade"],
                            "unchanged": before[subject]["cascade"]
                                         == after[subject]["cascade"]}
        rows[f"subject_{subject:02d}"] = block
    return {
        "band": ("the raw array byte-identical; every ARM and shoulder-line fire count "
                 "IDENTICAL to D8b's; the THIGH and SHIN counts identical except the "
                 "counted cascade under a newly marked hip"),
        "raw_array_byte_identical": raw_identical,
        "per_segment": rows,
        "every_other_segments_fire_frames_identical": identical_everywhere,
        "why_it_matters": ("the rule computes every length on the array as it arrived, "
                           "before anything is withheld, so it is order-independent within "
                           "itself and the new row charges only the hips. ANY other change "
                           "would mean the addition did more than add a segment."),
        "verdict": "PASS" if (all(raw_identical.values()) and identical_everywhere)
                   else "FAIL",
    }


# ---------------------------------------------------------------------------- B5
def b5() -> dict:
    """MAMMA's hips and knees on the three runs. REPORT ONLY -- MAMMA never selects."""

    from subject_map import mamma_index_for  # noqa: E402
    import mamma_scoreboard as ms  # noqa: E402

    runs = {"83_87": (83, 87), "109_119": (109, 119), "158_168": (158, 168)}
    joints = ("left_hip", "right_hip", "left_knee", "right_knee")
    with np.load(D8B_DELIVERY / "subject-00.body-track.npz") as a, \
         np.load(D8B_DELIVERY / "subject-01.body-track.npz") as b:
        smoothed = np.stack([a["triangulated_world_positions_z_up_m"],
                             b["triangulated_world_positions_z_up_m"]])
    mapping = mamma_index_for(smoothed)
    mamma = {body: np.load(ms.MA3D / f"verts_joints_body_id-{body:02d}.npz",
                           allow_pickle=True)["pred_joints"] for body in (0, 1)}
    out: dict = {
        "band": "REPORT ONLY. MAMMA reports and never selects (CLAUDE.md).",
        "prediction_from_the_card": "closer on run (i), 83-87 and 109-119",
        "subject_map": {f"subject_{k:02d}": f"body_id-{v:02d}" for k, v in mapping.items()},
        "what": ("the CAPTURED hips and knees against MAMMA's own SMPL-X joints, in the "
                 "camera-rig world both live in, per run, before and after"),
        "blind_to": ("truth. MAMMA has none either -- its `gt_` arrays are byte-copies of "
                     "`pred_`. Moving toward it is not moving toward correct."),
        "subjects": {},
    }
    for subject in (0, 1):
        arrays = {}
        for label, directory in (("D8b", D8B_DELIVERY), ("D8c", D8C_DELIVERY)):
            with np.load(directory / f"subject-{subject:02d}.body-track.npz") as archive:
                arrays[label] = np.asarray(
                    archive["triangulated_world_positions_z_up_m"], dtype=np.float64)
        reference = mamma[mapping[subject]]
        block: dict = {}
        for run, (lo, hi) in runs.items():
            index = list(range(lo - FIRST_FRAME_ID, hi - FIRST_FRAME_ID + 1))
            cell = {}
            for name in joints:
                theirs = reference[index][:, ms.PAIRS[name]]
                for label in ("D8b", "D8c"):
                    ours = arrays[label][index][:, cm.JOINT_INDEX[name]]
                    distance = np.linalg.norm(ours - theirs, axis=1) * 1000.0
                    distance = distance[np.isfinite(distance)]
                    cell[f"{name}_{label}_median_mm"] = (
                        round(float(np.median(distance)), 2) if distance.size else None)
            cell["hips_D8b_median_mm"] = round(float(np.median([
                cell[f"{n}_D8b_median_mm"] for n in ("left_hip", "right_hip")])), 2)
            cell["hips_D8c_median_mm"] = round(float(np.median([
                cell[f"{n}_D8c_median_mm"] for n in ("left_hip", "right_hip")])), 2)
            cell["knees_D8b_median_mm"] = round(float(np.median([
                cell[f"{n}_D8b_median_mm"] for n in ("left_knee", "right_knee")])), 2)
            cell["knees_D8c_median_mm"] = round(float(np.median([
                cell[f"{n}_D8c_median_mm"] for n in ("left_knee", "right_knee")])), 2)
            cell["hips_moved_closer"] = bool(
                cell["hips_D8c_median_mm"] < cell["hips_D8b_median_mm"])
            block[run] = cell
        out["subjects"][f"subject_{subject:02d}"] = block

    # WHY THE PREDICTION FAILS, and it is not about the repair. MAMMA's `pred_joints`
    # indices 1 and 2 are SMPL-X's own pelvis joints on a body whose SHAPE is a per-subject
    # constant, so its hip line is the SAME NUMBER on every frame of the take -- and that
    # number is about half ours. Measured rather than asserted, below.
    widths = {}
    for subject in (0, 1):
        reference = mamma[mapping[subject]]
        series = np.linalg.norm(reference[:, ms.PAIRS["left_hip"]]
                                - reference[:, ms.PAIRS["right_hip"]], axis=1) * 1000.0
        with np.load(D8B_DELIVERY / f"subject-{subject:02d}.body-track.npz") as archive:
            ours = np.asarray(archive["triangulated_world_positions_z_up_m"],
                              dtype=np.float64)
        our_series = np.linalg.norm(ours[:, cm.JOINT_INDEX["left_hip"]]
                                    - ours[:, cm.JOINT_INDEX["right_hip"]], axis=1) * 1000.0
        widths[f"subject_{subject:02d}"] = {
            "mamma_hip_width_min_mm": round(float(series.min()), 3),
            "mamma_hip_width_max_mm": round(float(series.max()), 3),
            "mamma_hip_width_is_constant_over_the_take": bool(
                float(series.max() - series.min()) < 1e-6),
            "our_D8b_hip_width_median_mm": round(float(np.median(our_series)), 2),
            "our_D8b_hip_width_on_110_119_mm": [
                round(float(our_series[50:60].min()), 2),
                round(float(our_series[50:60].max()), 2)],
        }
    out["why_the_prediction_fails_and_it_is_not_the_repair"] = {
        "measured": widths,
        "the_finding": (
            "MAMMA's hip line is a CONSTANT -- 117.6 mm on all 150 frames for the falling "
            "performer and 114.8 for the other -- because it fits a rigid SMPL-X body whose "
            "shape betas are per-subject and per-take. Ours is a captured landmark pair "
            "whose take median is 214-215 mm. THE TWO ARE NOT THE SAME ANATOMICAL POINTS "
            "and the systematic gap is about 97 mm, several times anything this step "
            "changes. On 110-119 the D8b build's collapsed hip line read 140-172 mm, which "
            "is ACCIDENTALLY NEARER MAMMA's 118 mm constant than the honest 215 mm is. So "
            "agreeing with MAMMA on those frames REWARDS THE DEFECT, and repairing it "
            "necessarily scores worse against this reference. That is the whole of B5's "
            "'prediction failed' and it is a statement about the reference, not about the "
            "row. CLAUDE.md: numbers from different references never share an axis, and "
            "MAMMA reports and never selects."),
        "what_would_make_B5_readable": (
            "a MAMMA-side quantity that is not a bone width -- the pelvis CENTRE, which "
            "both sides define the same way. That is `root` in the scoreboard and it is "
            "already reported there."),
    }
    return out


# ---------------------------------------------------------------------------- B6
def b6() -> dict:
    ours = LOGS / "17-b6-d3-skeleton-gate.log"
    theirs = POST_MERGE_D8B / "d3_skeleton_gate.log"
    strip = lambda text: [line for line in text.splitlines()
                          if not line.strip().startswith("D3 verdict")]
    a = strip(ours.read_text()) if ours.exists() else None
    b = strip(theirs.read_text()) if theirs.exists() else None
    verdict_line = next((line for line in ours.read_text().splitlines()
                         if line.startswith("D3 verdict")), None) if ours.exists() else None
    return {
        "band": ("every figure bit-identical to artifacts/compare/post-merge-D8b/ -- a "
                 "SRC-HYGIENE TRIPWIRE, not a band on this rule"),
        "why_it_is_not_a_band_on_the_rule": (
            "`d3_skeleton_gate.py` builds landmarks from forward-kinematics TRUTH and calls "
            "`positions_to_body_track`; it never enters `reconstruct_multiview` or "
            "`_reject_inconsistent_segments`, so the length rule NEVER EXECUTES there. An "
            "exact rigid body cannot fire a length rule because the rule does not run. "
            "Bit-identity says only that the converter, the rest and the hoist were not "
            "edited. The honest oracle for this row is S2."),
        "every_line_compared": True,
        "identical_apart_from_the_elapsed_time": bool(a is not None and a == b),
        "our_verdict_line": verdict_line,
        "the_gates_own_standing_failures": (
            "this gate reads FAIL on four clauses on EVERY build since D7b and did so on "
            "post-merge-D8b too: two 'canonical unchanged' clauses pinned to references "
            "frozen at D2c/D3 that later steps moved by design, the exact-skeleton oracle's "
            "legs at 0.07 mm and arms at 1.17 mm, and the same-denominator clause. CLAUDE.md "
            "records the re-pin as owed. Nothing about it moved here, which is the point."),
        "verdict": "PASS" if (a is not None and a == b) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()

    hygiene = load(OUT_DIR / "delivery-hygiene-build.json") or {}
    build = load(OUT_DIR / "delivery-build.json") or {}
    instrument = load(OUT_DIR / "limb-stability-d8b.json") or {}
    after = load(OUT_DIR / "limb-stability-after.json") or {}
    synthetic = load(OUT_DIR / "synthetic.json") or {}
    synthetic_020 = load(OUT_DIR / "synthetic-noise-0.20.json") or {}
    calibration = load(OUT_DIR / "synthetic-noise-calibration.json") or {}
    silhouette = load(OUT_DIR / "silhouette-partwise.json") or {}
    placement = load(OUT_DIR / "placement.json") or {}
    b4 = load(OUT_DIR / "b4-fixed-denominator.json") or {}

    report: dict = {
        "title": "D8c gate -- the hip line, every pre-registered band applied mechanically",
        "step": "D8c",
        "branch": "ladder/D8c",
        "ships": "one row in SEGMENT_LENGTH_RULES: "
                 '("hip_line", "left_hip", "right_hip", ("left_hip", "right_hip")). '
                 "Nothing else in src/.",
        "preregistration": PREREGISTRATION,
        "preregistration_source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D8c row, "
                                   "committed at 85b8113 before this branch's first "
                                   "instrument ran. Frozen: no band, ceiling or clause "
                                   "moved during this step."),
        "mamma_free_in_every_selecting_arm": True,
        "src_change": {
            "files": ["src/autoanim_gnm/commercial_multiview.py"],
            "rows_before": 9, "rows_after": len(cm.SEGMENT_LENGTH_RULES),
            "the_row": list(cm.SEGMENT_LENGTH_RULES[-1][:3])
                       + [list(cm.SEGMENT_LENGTH_RULES[-1][3])],
            "ceiling_unchanged": float(cm.SEGMENT_LENGTH_CEILING_FRACTION) == 0.15,
            "modes_unchanged": list(cm.SEGMENT_LENGTH_MODES) == ["demote", "reject",
                                                                 "best_ray"],
        },
    }

    # ------------------------------------------------------------------------- hygiene
    report["hygiene_before_any_src_change"] = {
        "band": "8 of 8 delivered files byte-identical to the shipped delivery",
        "all_delivered_files_identical": (hygiene.get("hygiene") or {}).get(
            "all_delivered_files_identical"),
        "raw_byte_identical": (hygiene.get("hygiene") or {}).get(
            "raw_triangulation_byte_identical_same_denominator"),
        "smoothed_byte_identical": (hygiene.get("hygiene") or {}).get(
            "smoothed_triangulation_byte_identical"),
        "observations_byte_identical_to_the_shipped_build": (hygiene.get("hygiene") or {})
            .get("observations_byte_identical_to_the_shipped_build"),
        "log": "logs/01-delivery-hygiene.log",
        "verdict": "PASS" if (hygiene.get("verdict") == "PASS") else "FAIL",
    }

    # ---------------------------------------------------------------- instrument first
    clauses = (instrument.get("reproduction") or {}).get("clauses") or []
    report["instrument_first"] = {
        "band": ("`captured_limb_stability.py --reproduce d8c` reproduces the card's "
                 "figures on the shipped (D8b) build, committed BEFORE any src change"),
        "clauses_total": len(clauses),
        "clauses_matching": sum(1 for row in clauses if row["matches"]),
        "clauses": [{"clause": row["clause"], "card_says": row["card_says"],
                     "measured": row["measured"], "array": row["array"],
                     "matches": row["matches"], "note": row["note"]} for row in clauses],
        "card_figures_that_reproduce_only_in_part": [
            {"figure": "the thighs' high figure, 426 mm",
             "measured": "421.8 raw / 408.3 smoothed on the class (i) runs; 425.6 is "
                         "reachable only on 158-168, a different run and a different "
                         "failure",
             "band_moved": False},
            {"figure": "the across-the-baseline range 18-52 mm on 158-168",
             "measured": "reproduces on ten of the eleven frames; frame 158 reads 74.0",
             "band_moved": False},
            {"figure": "the hip-line-to-ray angle 18-28 deg on 158-168",
             "measured": "the upper figure reproduces exactly (28.3); the lower reads 15.8",
             "band_moved": False},
        ],
        "window_frame_id_repair": (
            "`segment_summary` labelled every reported frame as `array index + "
            "FIRST_FRAME_ID`, right for the whole-take call and wrong for every windowed "
            "one: the window block printed 60-94 for frames 85-119. Repaired. VERIFIED "
            "rather than asserted that the `d8` and `d8b` paths are otherwise untouched -- "
            "the committed instrument and this one were run on the same build under both "
            "flags and the reports diffed: 396 differing leaves, every one a +25 shift of "
            "`window/frames_off_ids` or `window/worst_frame_id`, same clause names, same "
            "card_says, same measured values, same verdicts (logs/05)."),
        "verdict": "PASS" if clauses and all(row["matches"] for row in clauses) else "FAIL",
        "log": "logs/02-limb-stability-d8c-reproduction.log",
    }

    # ----------------------------------------------------------------------- SYNTHETIC
    s0 = synthetic.get("S0_fixture_honest_hip_line") or {}
    selector = synthetic.get("selector") or {}
    report["S0_the_fixtures_own_honest_hip_line"] = {
        "band": ("measured FIRST, against the take's -8.2 / +8.5 %. If wider, the FIXTURE "
                 "is recalibrated as a fixture parameter with src/ byte-identical across "
                 "it -- never the ceiling."),
        "measured_at_D8bs_noise_scale_0.20": {
            "subject_01": ((synthetic_020.get("S0_fixture_honest_hip_line") or {})
                           .get("measured", {}).get("per_subject", {})
                           .get("subject_01", {}).get("p5_p95_fraction_of_median")),
            "subject_00": ((synthetic_020.get("S0_fixture_honest_hip_line") or {})
                           .get("measured", {}).get("per_subject", {})
                           .get("subject_00", {}).get("p5_p95_fraction_of_median")),
            "inside_the_takes_own_spread": (synthetic_020.get(
                "S0_fixture_honest_hip_line") or {}).get("inside_the_takes_own_spread"),
        },
        "recalibrated_to": s0.get("noise_scale"),
        "measured_at_the_calibrated_scale": s0.get("measured", {}).get("per_subject"),
        "inside_the_takes_own_spread": s0.get("inside_the_takes_own_spread"),
        "why_the_hip_line_needs_its_own_scale": (
            "the hip line is a ~200 mm segment where the legs D8b calibrated on are ~400, "
            "so the same pixel noise buys twice the fractional spread"),
        "calibration_sweep": calibration.get("candidates"),
        "calibration_rule": calibration.get("rule"),
        "src_byte_identical_across_the_recalibration": (
            "yes -- `src/` had not been touched at all when the sweep ran; the src change "
            "is a later commit on this branch"),
        "in_src": False,
        "verdict": "PASS" if s0.get("inside_the_takes_own_spread") else "FAIL",
        "log": "logs/04-s0-noise-calibration.log",
    }
    report["S1_the_selector"] = {
        "band": ("confirm D8b's mode and ceiling on the hips' own geometry. If the "
                 "selector chooses otherwise THE STEP STOPS and is re-carded."),
        "injection": {k: v for k, v in (synthetic.get("injection") or {}).items()
                      if k not in ("why_not_toward_a_landmark",)},
        "today_hips_median_mm": selector.get("today_hips_median_mm"),
        "per_mode_hips_median_mm": selector.get("per_mode_hips_median_mm"),
        "per_group_median_mm": selector.get("per_group_median_mm"),
        "selected_mode": selector.get("selected_mode"),
        "selects_the_shipped_mode": selector.get("selects_the_shipped_mode"),
        "beats_today_on_the_hips": selector.get("beats_today_on_the_hips"),
        "ceiling_argmin_on_the_hips": selector.get("ceiling_argmin_on_the_hips"),
        "ceilings_tied_at_the_argmin": selector.get("ceilings_tied_at_the_argmin"),
        "paired_margins": selector.get("paired_margins"),
        "converter": synthetic.get("converter"),
        "injection_variants": synthetic.get("injection_variants"),
        "cascade_on_the_injected_clip": next(
            (row["cascade"] for row in (synthetic.get("sweep") or {})
             .get("demote", {}).get("candidates", [])
             if abs(row["ceiling_fraction"] - 0.15) < 1e-9), None),
        "at_D8bs_noise_scale_0.20": {
            "per_mode_hips_median_mm": (synthetic_020.get("selector") or {}).get(
                "per_mode_hips_median_mm"),
            "selected_mode": (synthetic_020.get("selector") or {}).get("selected_mode"),
            "today_hips_median_mm": (synthetic_020.get("selector") or {}).get(
                "today_hips_median_mm"),
        },
        "verdict": "PASS" if (selector.get("selects_the_shipped_mode")
                              and selector.get("beats_today_on_the_hips")) else "FAIL",
        "log": "logs/06-synthetic-s1-s3.log",
    }
    o1 = synthetic.get("S2_oracle_clean_fully_seen") or {}
    o2 = synthetic.get("S2_oracle_uncollapsed_frames") or {}
    report["S2_the_two_oracles"] = {
        "band": ("clean input -> zero fires and output bit-identical to today; the "
                 "collapsed clip's un-collapsed frames -> zero hip fires"),
        "oracle_1_clean_fully_seen": {k: v for k, v in o1.items() if k != "what"},
        "oracle_2_uncollapsed_frames": {k: v for k, v in o2.items() if k != "what"},
        "at_D8bs_noise_scale_0.20": {
            "oracle_1": (synthetic_020.get("S2_oracle_clean_fully_seen") or {}).get(
                "passes"),
            "oracle_2": (synthetic_020.get("S2_oracle_uncollapsed_frames") or {}).get(
                "passes"),
            "oracle_2_fires_outside_the_run": (synthetic_020.get(
                "S2_oracle_uncollapsed_frames") or {}).get("hip_fires_outside_the_run"),
            "note": "kept beside the pass, unmoved. At the harder (uncalibrated) noise "
                    "scale oracle 2 fails by ONE frame, adjacent to the injected run -- a "
                    "fixture-noise boundary effect, and the reason S0 recalibrates the "
                    "fixture rather than the ceiling.",
        },
        "verdict": "PASS" if (o1.get("passes") and o2.get("passes")) else "FAIL",
    }
    m1 = synthetic.get("S3_must_fail_a_forbidden_fire_on_the_clean_clip") or {}
    m2 = synthetic.get("S3_must_fail_whole_take_hold") or {}
    report["S3_the_two_must_fails"] = {
        "band": ("any LEG, ARM or shoulder-line fire on the clean fixture at 0.15 = FAIL "
                 "(the honest hip line at 0.15 is the new exposure); and a whole-take hold"),
        "forbidden_fires_at_the_shipped_ceiling": m1.get(
            "forbidden_fires_at_the_shipped_ceiling"),
        "the_new_exposure_hip_line_fires_on_the_clean_clip": m1.get(
            "hip_line_fires_at_the_shipped_ceiling"),
        "degenerate_demonstrated_at": m1.get(
            "tight_ceilings_that_do_fire_forbidden_segments"),
        "by_ceiling": m1.get("by_ceiling"),
        "whole_take_hold": {
            "frozen_hips_median_mm": (m2.get("score") or {}).get("hips", {}).get(
                "median_mm"),
            "candidate_hips_median_mm": (m2.get("shipped_score") or {}).get(
                "hips", {}).get("median_mm"),
            "fails_as_required": m2.get("fails_as_required_on_the_hips"),
            "travel_mm": m2.get("how_far_the_landmarks_actually_travel_over_the_run_mm"),
            "the_control_is_WEAK_on_the_hips": m2.get(
                "and_the_pooled_figure_is_the_wrong_one_to_read"),
        },
        "verdict": "PASS" if (m1.get("passes")
                              and m2.get("fails_as_required_on_the_hips")) else "FAIL",
    }

    # ------------------------------------------------------------------------------ B1
    verdicts = silhouette.get("preregistered_clause_verdicts") or {}
    banded = [row for subject in ("subject_00", "subject_01")
              for name, row in (verdicts.get(subject) or {}).items()
              if name.startswith("clause_")]
    report["B1_the_photographs"] = {
        "band": silhouette.get("preregistered", {}).get("clause"),
        "clause_verdicts": verdicts,
        "torso_iou_by_cut": {
            subject: {cut: {"D8b": cell.get("torso_iou_D8b"),
                            "D8c": cell.get("torso_iou_D8c"),
                            "ORACLE_mamma_mesh": cell.get("torso_iou_ORACLE"),
                            "n": cell.get("n")}
                      for cut, cell in (silhouette.get("subjects", {}).get(subject, {})
                                        .get("cuts", {})).items()}
            for subject in ("subject_00", "subject_01")},
        "raw_byte_identical": silhouette.get("raw_triangulation_byte_identical"),
        "smoothed_byte_identical_EXPECTED_FALSE": silhouette.get(
            "smoothed_triangulation_byte_identical"),
        "what_it_cannot_settle": silhouette.get("preregistered", {}).get(
            "what_this_cannot_settle"),
        "clauses_total": len(banded),
        "clauses_passing": sum(1 for row in banded if row["verdict"] == "PASS"),
        "verdict": silhouette.get("verdict"),
        "log": "logs/14-b1-silhouette.log",
    }

    # ------------------------------------------------------------------------------ B2
    report["B2_the_raw_array_and_every_other_segments_fires"] = b2()
    report["what_the_rule_saw_and_what_it_could_not"] = rule_visibility()

    # ------------------------------------------------------------------------------ B3
    b3_raw = load(OUT_DIR / "delivered-vs-capture-raw.json") or {}
    b3_smooth = load(OUT_DIR / "delivered-vs-capture-smoothed.json") or {}
    report["B3_delivered_vs_capture"] = {
        "band": "REPORTED -- the card does not band this arm",
        "reference_is_biased_against_the_repair": (
            "under `--reference raw` the reference on the fired frames IS the point the "
            "rule judged wrong, so agreeing with it there scores well and disagreeing "
            "scores badly whatever the truth is. Read the LEGS, the FEET and the trunk "
            "floor, which the rule does not charge."),
        "vs_raw": b3_raw.get("subjects") or b3_raw,
        "vs_repaired_smoothed": b3_smooth.get("subjects") or b3_smooth,
        "log": ["logs/15-b3-delivered-vs-capture-raw.log",
                "logs/16-b3-delivered-vs-capture-smoothed.log"],
    }

    # ------------------------------------------------------------------------------ B4
    report["B4_the_frames_off_on_a_fixed_denominator"] = {
        "band": b4.get("B4", {}).get("band"),
        "must_fail": b4.get("B4", {}).get("must_fail"),
        "must_fail_holds": b4.get("B4", {}).get("must_fail_holds"),
        "measured": b4.get("B4", {}).get("measured"),
        "every_segment_both_arrays": b4.get("arrays"),
        "the_median_barely_moved": (
            "performer 1's hip-line median goes 214.40 -> 214.72 mm, 0.32 mm, so the count "
            "did not fall because the reference slid: the fixed and moving readings agree "
            "at 0 and the smear degenerate is excluded"),
        "the_candidate_optimises_this_directly": (
            "which is why the card puts it in the merge rule ONLY PAIRED with B1"),
        "verdict": "PASS" if b4.get("B4", {}).get("passes") else "FAIL",
        "log": "logs/13-b4-fixed-denominator.log",
    }

    # ------------------------------------------------------------------------------ B5
    report["B5_mamma_on_the_three_runs"] = b5()

    # ------------------------------------------------------------------------------ B6
    report["B6_the_d3_gate_src_hygiene_tripwire"] = b6()

    # ------------------------------------------------------------------------- R1 - R4
    report["R1_R4_the_delivered_root_pelvis_frame_and_hoist"] = {
        "band": "REPORTS with predictions. None of them is a fail clause.",
        "subjects": placement.get("subjects"),
        "R1_prediction": {"root median on the fired frames": "< 20 mm",
                          "root worst": "< 45 mm"},
        "R1_measured": {
            subject: (block.get("R1_root_and_hips_hoist_removed", {})
                      .get("meets_the_prediction"))
            for subject, block in (placement.get("subjects") or {}).items()},
        "R2_prediction": "no frame outside the +/-4-frame envelope moves more than 1 mm",
        "R2_measured": {
            subject: {
                "delivered_joints_outside_the_envelope_moving_more_than_1mm": len(
                    block.get("R2_displacement_vs_distance_to_a_fire", {})
                    .get("frames_outside_the_envelope_moving_more_than_1mm", [])),
                "CAPTURE_STAGE_landmarks_outside_the_envelope_moving_more_than_1mm": len(
                    block.get("R2_displacement_vs_distance_to_a_fire", {})
                    .get("attribution", {}).get("capture_stage_landmarks", {})
                    .get("frames_outside_the_envelope_moving_more_than_1mm", [])),
                "rest_bones_that_moved": (
                    block.get("R2_displacement_vs_distance_to_a_fire", {})
                    .get("attribution", {})
                    .get("delivery_stage_the_per_performer_rest", {})
                    .get("rest_bones_that_moved")),
                "worst_rest_bone_mm": (
                    block.get("R2_displacement_vs_distance_to_a_fire", {})
                    .get("attribution", {})
                    .get("delivery_stage_the_per_performer_rest", {})
                    .get("worst_bone_mm")),
            }
            for subject, block in (placement.get("subjects") or {}).items()},
        "R2_the_prediction_FAILS_and_this_is_what_did_it": (
            "at the CAPTURE stage the prediction holds exactly -- zero frames outside the "
            "envelope move more than 1 mm on either performer. The delivered JOINTS move on "
            "54 frames outside it, at 1.0-2.0 mm, and the mechanism is not a smoothing "
            "leak: D3's PER-PERFORMER REST SKELETON IS SIZED FROM THE WHOLE TAKE. Changing "
            "eighteen frames moves sixteen of the fifty-five rest bones by up to 0.96 mm "
            "(the right lower leg), so every frame's rotations solve on a slightly "
            "different body and every frame's forward kinematics lands somewhere else; the "
            "foot contacts change with it (24 -> 29 on performer 1). The envelope the card "
            "registered is the right envelope for the stage it names and the wrong one for "
            "the delivered file, and that is a property of the pipeline, not of this row."),
        "log": ["logs/11-placement-r1-r4.log", "logs/12-r2-attribution.log"],
    }

    # -------------------------------------------------------- head gate and D7b's neck
    head_log = LOGS / "18-head-gate.log"
    report["head_gate_and_the_D7b_neck_figure"] = {
        "band": "rerun and REPORTED",
        "head_gate_reads": (
            "artifacts/head-lane/head-solve-shipped.npz -- the head the PIPELINE delivers, "
            "read from disk. It does NOT see a branch build's landmarks (D8b review section "
            "7), so a rerun here says the head solve is where it was and nothing more."),
        "head_gate_lines": head_log.read_text().splitlines() if head_log.exists() else None,
        "d7b_trunk_length_floor": {
            subject: {label: (b3_raw.get("subjects", {}).get(subject, {})
                              .get("trunk_length_floor", {}).get(label))
                      for label in ("D8b", "D8c")}
            for subject in ("subject_00", "subject_01")},
        "log": "logs/18-head-gate.log",
    }

    # ------------------------------------------------------------------- the merge rule
    conjuncts = {
        "S1 selects demote at 0.15 on the hips' own geometry":
            report["S1_the_selector"]["verdict"],
        "S2 both oracle clauses": report["S2_the_two_oracles"]["verdict"],
        "S3 both must-fails": report["S3_the_two_must_fails"]["verdict"],
        "B1 on both performers": report["B1_the_photographs"]["verdict"],
        "B2": report["B2_the_raw_array_and_every_other_segments_fires"]["verdict"],
        "B4 on the fixed denominator":
            report["B4_the_frames_off_on_a_fixed_denominator"]["verdict"],
        "B6": report["B6_the_d3_gate_src_hygiene_tripwire"]["verdict"],
    }
    report["merge_rule"] = {
        "fixed_before_numbers": ("S1 selects `demote` at 0.15 on the hips' own geometry AND "
                                 "S2 both clauses AND S3 AND B1 on both performers AND B2 "
                                 "AND B4 on the fixed denominator AND B6; everything else "
                                 "reports"),
        "conjuncts": conjuncts,
        "reports_only": ["B3", "B5", "R1", "R2", "R3", "R4", "the head gate",
                         "D7b's neck figure", "the hygiene arm and the instrument-first "
                         "arm, which are preconditions and were met before any src change"],
        "mechanical_outcome": ("MERGE" if all(v == "PASS" for v in conjuncts.values())
                               else "DO NOT MERGE"),
        "failing_conjuncts": [k for k, v in conjuncts.items() if v != "PASS"],
    }
    report["verdict"] = ("PASS" if report["merge_rule"]["mechanical_outcome"] == "MERGE"
                         else "FAIL")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("D8c gate\n" + "=" * 78)
    for key in ("hygiene_before_any_src_change", "instrument_first",
                "S0_the_fixtures_own_honest_hip_line", "S1_the_selector",
                "S2_the_two_oracles", "S3_the_two_must_fails", "B1_the_photographs",
                "B2_the_raw_array_and_every_other_segments_fires",
                "B4_the_frames_off_on_a_fixed_denominator",
                "B6_the_d3_gate_src_hygiene_tripwire"):
        print(f"  {key:<52} {report[key].get('verdict')}")
    print(f"  {'B3_delivered_vs_capture':<52} REPORTED")
    print(f"  {'B5_mamma_on_the_three_runs':<52} REPORTED")
    print(f"  {'R1_R4':<52} REPORTED (R2's prediction fails; see the block)")
    print("\nmerge rule conjuncts")
    for name, value in conjuncts.items():
        print(f"  {value:<6} {name}")
    print(f"\nMECHANICAL OUTCOME: {report['merge_rule']['mechanical_outcome']}")
    print(f"verdict: {report['verdict']}")
    print(f"wrote {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
