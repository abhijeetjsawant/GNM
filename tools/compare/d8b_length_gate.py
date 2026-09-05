#!/usr/bin/env python3
"""D8b's gate: every band from the card, as its own keyed block, applied mechanically.

The card's bands, `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8b row, frozen 2026-09-06
before any number in this file existed. They are carried into the report verbatim under
`preregistration` and each is answered in its own block:

    instrument first    `tools/compare/captured_limb_stability.py` must reproduce the
                        card's figures on the shipped D9 build before any src change
    SYNTHETIC           the selector (today / (a) demote / (b) reject / (c) best_ray across
                        a ceiling sweep), both oracle clauses, and the two must-fails
    B1                  part-wise silhouette, ARMS and TORSO, window and whole take, D9 vs
                        D8b on identical draws, both performers, MAMMA mesh bit-identical
    B2                  the raw array byte-identical; leg rejects REPORTED, predicted 0
    B3                  `delivered_vs_capture` against the RAW points and against the
                        repaired ones, D9 vs D8b, identical draws, hands and elbows
    B4                  the instrument's frames-off counts before and after
    B5                  MAMMA's joints on the window through `subject_map`, report only;
                        the head gate rerun; the D3 closure

    merge rule, fixed before numbers: the synthetic selector holds for the shipped ceiling
    AND both oracle clauses AND B1 on both performers AND B2; the rest reports

THE ONE THING EVERY READER MUST CARRY. D8b's reject sits between the raw triangulation and
the smoothed landmarks -- the same place D8's three rules sit -- so **everything downstream
that compared two builds' SMOOTHED landmarks loses its byte-identical denominator by
design**. The RAW array is the fixed reference and B2 asserts it. Two blocks EXPECT to
report CHANGED and say so in their own text: `delivered_vs_capture`'s default mode, and the
D3 closure block's same-denominator clause. A CHANGED there is the step working.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8b_length_gate.py

Writes `artifacts/compare/d8b-length/gate.json`. NOTHING is written under
`artifacts/commercial-multiview-soma77/`.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
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

OUT_DIR = ROOT / "artifacts/compare/d8b-length"
REPORT = OUT_DIR / "gate.json"
D9_DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
D8B_DELIVERY = OUT_DIR / "delivery"
LEGS = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}

PREREGISTRATION = """\
**instrument first:** `captured_limb_stability.py` on the shipped delivery must reproduce
the figures above (16 / 4 / 4 / 0 frames off; 122-274 mm; A-C-D agreement to 1-11 px on
frames 110-122) before any src change

**SYNTHETIC (selector and bands):** I7's fixture with the real seen pattern replayed PLUS an
injected CONSISTENT collapse -- both shoulders moved toward the neck by a factor (0.35-0.75)
in every view that sees them for a run of 8-15 frames, the detector-bias mode measured here
-- scoring 3D error of the shoulders, elbows and wrists in the run for today's code, (a),
(b), both ceilings; oracle: clean input -> zero rejects and bit-identical output, AND on the
un-collapsed frames of the collapsed clip zero rejects (the ceiling must not fire on honest
motion); must-fail: a ceiling so tight it fires on the legs (any leg reject on the clean
fixture = FAIL) and a whole-take hold

COORDINATOR'S AMENDMENT, 2026-09-06, before any synthetic number existed: add a THIRD
candidate (c) KEEP THE ONE BEST RAY -- keep only the highest-confidence camera's ray for the
child landmark(s) and drop the others, so the existing single-ray recovery places the point;
and record why (b) is expected to reduce to the must-fail (a rejected slot keeps no ray, so
the solve cannot recover it and a run longer than MAXIMUM_INTERPOLATED_GAP_FRAMES is HELD on
the parent -- the frozen arm under another name). If (c) is selected, register nothing new
beyond the ceiling: the choice of ray is by the detector's own confidence.

**REAL TAKE, bands the candidate cannot optimise:** (B1) part-wise silhouette, arms AND
torso, window and whole take, not worse on either performer with the CI clear on the D7b/D8
predicate, improvement predicted on performer 1's window; (B2) raw array byte-identical; leg
rejects REPORTED, predicted 0 (a correct fire on the ankle's real NaN frames noted);
(B3) `delivered_vs_capture --reference raw` and vs the repaired points, D9 vs D8b, identical
draws, hands and elbows; (B4) captured shoulder line frames off >15 %: 16 -> <= 4
(performer 1), 4 -> <= 2 (performer 0), forearm 4 -> 0 -- the candidate optimises this
directly so it is paired with the photographs and the oracle; (B5) MAMMA's joints on the
window through `subject_map`, report only (predicted closer on the shoulders and elbows, as
D8 was)

**merge rule, fixed before numbers:** synthetic selector holds for the shipped ceiling AND
both oracle clauses AND B1 on both performers AND B2; the rest reports\
"""


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def arrays(delivery: Path, subject: int) -> dict[str, np.ndarray]:
    with np.load(delivery / f"subject-{subject:02d}.body-track.npz") as archive:
        return {"raw": np.asarray(archive["raw_triangulated_world_positions_z_up_m"]),
                "smoothed": np.asarray(archive["triangulated_world_positions_z_up_m"])}


# --------------------------------------------------------------------------------- B2
def b2_block(builds: dict[str, Path]) -> dict:
    same, smoothed_same = {}, {}
    for subject in (0, 1):
        a, b = arrays(builds["D9"], subject), arrays(builds["D8b"], subject)
        same[f"subject_{subject:02d}"] = bool(np.array_equal(a["raw"], b["raw"],
                                                             equal_nan=True))
        smoothed_same[f"subject_{subject:02d}"] = bool(
            np.array_equal(a["smoothed"], b["smoothed"]))
    run = load(builds["D8b"] / "run-report.json") or {}
    diagnostics = run.get("diagnostics", run)
    repair = diagnostics.get("occlusion_repair", [])
    rows = []
    for row in repair:
        by_joint = row.get("length_rejected_by_joint") or {}
        rows.append({
            "length_rejected_slots": row.get("length_rejected_slots"),
            "length_rejected_by_segment": row.get("length_rejected_by_segment"),
            "length_rejected_by_joint": by_joint,
            "length_rejected_legs": {k: v for k, v in by_joint.items() if k in LEGS},
            "children_marked_under_a_marked_parent_segment": row.get(
                "children_marked_under_a_marked_parent_segment"),
            "segment_median_m": row.get("segment_median_m"),
            "held_joint_fraction": row.get("held_joint_fraction"),
            "demoted_slots": row.get("demoted_slots"),
            "rejected_slots": row.get("rejected_slots"),
        })
    # The legs' own raw lengths on the frames the rule charged, so a CORRECT fire can be
    # told from a wrong one without leaving the report.
    detail = {}
    for subject in (0, 1):
        raw = arrays(builds["D9"], subject)["raw"]
        block = {}
        for name, parent, child, _charged in cm.SEGMENT_LENGTH_RULES:
            if child not in LEGS:
                continue
            frames = ((repair[subject].get("length_rejected_frames_by_segment") or {})
                      .get(name, []) if subject < len(repair) else [])
            if not frames:
                continue
            a = raw[:, cm.JOINT_INDEX[parent]]
            b = raw[:, cm.JOINT_INDEX[child]]
            usable = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            lengths = np.full(len(raw), np.nan)
            lengths[usable] = np.linalg.norm(a[usable] - b[usable], axis=1) * 1000.0
            median = float(np.nanmedian(lengths))
            block[name] = {
                "own_take_median_mm": round(median, 2),
                "fired_frames": [{"frame_index": int(f),
                                  "length_mm": round(float(lengths[f]), 2),
                                  "fraction_off": round(
                                      float(abs(lengths[f] - median) / median), 4)}
                                 for f in frames],
            }
        if block:
            detail[f"subject_{subject:02d}"] = block
    return {
        "band": "the RAW array byte-identical between the D9 and D8b builds; leg rejects "
                "REPORTED, predicted 0",
        "raw_bit_identical": same,
        "raw_is_the_fixed_reference": (
            "D8b's reject sits after `world` is captured, so this is the one array that "
            "must not move. Everything downstream loses its byte-identical denominator BY "
            "DESIGN, exactly as it did at D8."),
        "smoothed_bit_identical": smoothed_same,
        "smoothed_note": "EXPECTED false -- if it were true the step would have done nothing",
        "per_subject_length_reject": rows,
        "leg_prediction_held": not any(row["length_rejected_legs"] for row in rows),
        "leg_fires_in_detail": detail,
        "leg_note": ("the card puts the leg counts in the REPORTED column and predicts 0. "
                     "Where a leg fires, the frame's own raw length and its distance from "
                     "the performer's own median are here, so a CORRECT fire on a genuinely "
                     "broken capture can be told from an over-firing rule without leaving "
                     "the report."),
        "verdict": "PASS" if all(same.values()) else "FAIL",
        "verdict_note": (
            "B2's BAND is the raw array's byte-identity and nothing else. A leg that fires "
            "is a refuted PREDICTION, recorded in the review, and never a failed band -- "
            "reading it as a band would let a reported figure decide the merge, which the "
            "card does not say and the merge rule does not license."),
    }


# --------------------------------------------------------------------------------- B4
def b4_block() -> dict:
    before = load(OUT_DIR / "limb-stability-d9.json") or {}
    after = load(OUT_DIR / "limb-stability-after.json") or {}

    def segment(doc, subject, group, pair):
        return ((doc.get("subjects", {}).get(f"subject_{subject:02d}", {})
                 .get("segments", {}).get("smoothed", {}).get(group, {}).get(pair)) or {})

    def count(doc, subject, group, pair):
        return segment(doc, subject, group, pair).get(
            "frames_off_median_by_more_than_15pct")

    clauses = []
    for label, subject, group, pair, band, card_before in (
            ("performer 1 shoulder line", 1, "shoulder_line",
             "left_shoulder__right_shoulder", 4, 16),
            ("performer 0 shoulder line", 0, "shoulder_line",
             "left_shoulder__right_shoulder", 2, 4),
            ("performer 1 forearm L", 1, "arms", "left_elbow__left_wrist", 0, 4)):
        got_before = count(before, subject, group, pair)
        got_after = count(after, subject, group, pair)
        clauses.append({
            "clause": label,
            "card_before": card_before,
            "measured_before": got_before,
            "card_band_after": f"<= {band}",
            "measured_after": got_after,
            "verdict": ("PASS" if got_after is not None and got_after <= band else "FAIL"),
            "before_reproduces": got_before == card_before,
        })
    everything = {}
    for subject in (0, 1):
        rows = {}
        for group, pairs in (("arms", ("left_shoulder__left_elbow", "left_elbow__left_wrist",
                                       "right_shoulder__right_elbow",
                                       "right_elbow__right_wrist")),
                             ("legs", ("left_hip__left_knee", "left_knee__left_ankle",
                                       "right_hip__right_knee", "right_knee__right_ankle")),
                             ("shoulder_line", ("left_shoulder__right_shoulder",)),
                             ("hip_line", ("left_hip__right_hip",))):
            for pair in pairs:
                b, a = segment(before, subject, group, pair), segment(after, subject, group,
                                                                     pair)
                rows[f"{group}/{pair}"] = {
                    "frames_off_before": b.get("frames_off_median_by_more_than_15pct"),
                    "frames_off_after": a.get("frames_off_median_by_more_than_15pct"),
                    "min_mm_before": b.get("min_mm"), "min_mm_after": a.get("min_mm"),
                    "max_mm_before": b.get("max_mm"), "max_mm_after": a.get("max_mm"),
                    "p5_p95_before": b.get("p5_p95_fraction_of_median"),
                    "p5_p95_after": a.get("p5_p95_fraction_of_median"),
                }
        everything[f"subject_{subject:02d}"] = rows
    fixed = load(OUT_DIR / "b4-fixed-denominator.json") or {}
    return {
        "band": "captured shoulder line frames off >15 %: 16 -> <= 4 (performer 1), "
                "4 -> <= 2 (performer 0), forearm 4 -> 0",
        "same_denominator_defect_in_this_band": {
            "what": "the instrument recomputes each build's median from THAT BUILD's own "
                    "array, so a rule that withholds frames changes the reference the count "
                    "is taken against. B4 is the one band in this step whose denominator "
                    "moves with the candidate.",
            "found_by": "the coordinator's question about the forearm going 4 -> 5",
            "fixed_denominator_reading": fixed,
            "what_it_changes": (
                "exactly one cell. On performer 1's forearm the own-median reading is "
                "4 -> 5 and the fixed-denominator reading (D9's medians for both builds) is "
                "4 -> 4. Every other cell is identical under the two readings. The clause "
                "still FAILS either way -- the band asks for 0 -- but the apparent "
                "REGRESSION is an artefact of the moving median and not a real one."),
            "the_crossing_frame": {
                "frame_id": 83,
                "D9_length_mm": 291.35, "D9_median_mm": 255.36, "D9_fraction_off": 0.1410,
                "D8b_length_mm": 293.20, "D8b_median_mm": 253.10, "D8b_fraction_off": 0.1584,
                "D8b_length_against_D9_median": 0.1482,
                "was_the_elbow_withheld_on_this_frame": False,
                "what_actually_happened": (
                    "frame 83 was ALREADY 14.10 % off in D9 -- 0.9 % under the reporting "
                    "cut. The length rule fired on the shoulder line at ids 84-86 and the "
                    "upper arm at id 85, NOT on frame 83, so the elbow there was never "
                    "withheld; it moved 5.51 mm because its neighbours were withheld and "
                    "the sequence solve and the 9-frame Savitzky-Golay window carry that "
                    "in, lengthening the forearm by 1.85 mm. Holding the median at D9's "
                    "value that leaves the frame at 14.82 %, still under the cut. What "
                    "crosses it is the MEDIAN moving 255.36 -> 253.10 mm, because the rule "
                    "withheld 7 forearm frames elsewhere and changed the population the "
                    "median is taken over."),
            },
        },
        "the_candidate_optimises_this_directly": (
            "the card says so, and it is why this band is paired with the photographs (B1) "
            "and the synthetic oracles. A limb welded to its own median scores 0 here."),
        "clauses": clauses,
        "every_segment_before_and_after": everything,
        "hip_line_note": (
            "the HIP LINE is reported here and is NOT in the shipped rule's segment list, "
            "because the D8b card's list does not name it. On performer 1 it is off its own "
            "median by more than 15 % on 23 frames (min 139.7 mm against a 214.4 mm "
            "median) before AND after, which is the same class of defect on a segment this "
            "step does not act on."),
        "verdict": "PASS" if all(row["verdict"] == "PASS" for row in clauses) else "FAIL",
    }


# --------------------------------------------------------------------------------- B5
def d3_closure(builds: dict[str, Path]) -> dict:
    """D3's closure clause, computed here rather than by rebuilding a second delivery."""

    import d3_skeleton_gate as d3  # noqa: E402
    from autoanim_gnm.body import (  # noqa: E402
        BodyTrack, forward_kinematics_positions, skeleton_for_track)

    out: dict = {
        "band": "the delivered GLB agrees with forward kinematics of the track it carries, "
                "<= 1e-6 m. A WITHIN-build check, so D8b cannot excuse a failure of it.",
        "builds": {},
        "same_denominator_clause": {
            "what": "D3's same-denominator clause compares two builds' triangulated "
                    "landmarks byte for byte",
            "expected": "CHANGED -- D8b moves the smoothed landmarks by construction",
        },
    }
    for label, delivery in builds.items():
        rows = {}
        for subject in (0, 1):
            track = BodyTrack.from_dict(json.loads(
                (delivery / f"subject-{subject:02d}.body-track.json").read_text()))
            skeleton = skeleton_for_track(track)
            names, glb_positions, _rest = d3.glb_joint_positions(
                delivery / f"subject-{subject:02d}.glb")
            truth = forward_kinematics_positions(
                np.asarray(track.root_translation_m, np.float64),
                np.asarray(track.local_rotations_xyzw, np.float64),
                skeleton=skeleton).astype(np.float64)
            order = [list(skeleton.names).index(name) for name in names]
            worst = float(np.linalg.norm(glb_positions - truth[:, order], axis=2).max())
            rows[f"subject_{subject:02d}"] = {"closure_max_m": worst,
                                              "within_1e-6_m": bool(worst <= 1e-6)}
        out["builds"][label] = rows
    changed = {}
    for subject in (0, 1):
        changed[f"subject_{subject:02d}"] = not bool(np.array_equal(
            arrays(builds["D9"], subject)["smoothed"],
            arrays(builds["D8b"], subject)["smoothed"]))
    out["same_denominator_clause"]["smoothed_landmarks_changed"] = changed
    out["same_denominator_clause"]["verdict"] = (
        "CHANGED as expected" if all(changed.values())
        else "UNCHANGED -- which would mean the step did nothing")
    out["verdict"] = ("PASS" if all(row["within_1e-6_m"] for build in out["builds"].values()
                                    for row in build.values()) else "FAIL")
    return out


def head_gate_block() -> dict:
    before = load(OUT_DIR / "head-gate-shipped-before-d8b.json")
    after = load(OUT_DIR / "head-gate-shipped-after-d8b.json")
    strip = lambda doc: json.dumps(
        {k: v for k, v in (doc or {}).items() if k != "absolute_facing_not_a_band"},
        sort_keys=True)
    return {
        "what": "`tools/head/head_gate.py` rerun and its report compared before and after",
        "structurally_blind": (
            "it scores `artifacts/head-lane/head-solve-shipped.npz` and reads its torso "
            "frame from the SHIPPED delivery -- a head-lane artifact and a directory this "
            "build does not write -- so rerunning it reproduces its own figures and it is "
            "BLIND to D8b's effect on the delivered head. That is worth stating rather "
            "than quoting as reassurance: `_solve_head_for_subject` and `_thorax_frames` "
            "both read the SMOOTHED positions, so on a D8b build the head SOLVE itself "
            "changes. The figure that would see it is the delivered head's world "
            "orientation read from the two GLBs' own bytes, and it is not in the card."),
        "every_figure_byte_equal_before_and_after": bool(before and after
                                                         and strip(before) == strip(after)),
        "bands": (after or {}).get("bands"),
        "verdicts": {k: v for k, v in (after or {}).items() if k.startswith("subject")},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()
    builds = {"D9": D9_DELIVERY, "D8b": D8B_DELIVERY}
    for label, path in builds.items():
        if not path.exists():
            raise SystemExit(f"{label} delivery is missing at {path}")

    report: dict = {
        "title": "D8b -- captured segments that break the performer's own bone length: "
                 "every band from the card",
        "preregistration": PREREGISTRATION,
        "preregistration_source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D8b row, "
                                   "committed 2026-09-06 at 17e0dbc, before this branch's "
                                   "first instrument ran; plus the coordinator's amendment "
                                   "of the same day, received before any synthetic number "
                                   "existed and quoted in full above"),
        "builds": {label: str(path.relative_to(ROOT)) for label, path in builds.items()},
        "delivered_sha256": {
            label: {name: digest(path / name) for name in sorted(
                f"subject-{s:02d}{suffix}" for s in (0, 1)
                for suffix in (".glb", ".body-track.json", ".body-track.npz",
                               ".mapping.npz"))}
            for label, path in builds.items()},
        "denominator": (
            "the RAW triangulated landmarks, which D8b does not touch. The SMOOTHED "
            "landmarks move by construction, so two blocks that would otherwise assert "
            "byte-equality are EXPECTED to report CHANGED and say so in their own text."),
        "shipped_constants": {
            "SEGMENT_LENGTH_CEILING_FRACTION": cm.SEGMENT_LENGTH_CEILING_FRACTION,
            "SEGMENT_LENGTH_MODES": list(cm.SEGMENT_LENGTH_MODES),
            "shipped_mode": "demote",
            "RAY_PAIR_CONDITIONING_CEILING_DEG": cm.RAY_PAIR_CONDITIONING_CEILING_DEG,
            "REACHABILITY_SLACK_M": cm.REACHABILITY_SLACK_M,
            "MAXIMUM_INTERPOLATED_GAP_FRAMES": cm.MAXIMUM_INTERPOLATED_GAP_FRAMES,
        },
        "provenance_audit": {
            "report": "artifacts/compare/d8b-length/provenance.json",
            "verdict": (load(OUT_DIR / "provenance.json") or {}).get("verdict"),
            "leaks": len((load(OUT_DIR / "provenance.json") or {}).get("leaks", [])),
        },
    }

    # ------------------------------------------------------------- instrument first
    stability = load(OUT_DIR / "limb-stability-d9.json") or {}
    clauses = (stability.get("reproduction") or {}).get("clauses") or []
    report["instrument_first"] = {
        "band": "the instrument must reproduce the card's figures BEFORE any src change",
        "report": "artifacts/compare/d8b-length/limb-stability-d9.json",
        "committed_before_the_src_change": True,
        "clauses": clauses,
        "clauses_matching": f"{sum(1 for row in clauses if row['matches'])} of {len(clauses)}",
        "all_match": (stability.get("reproduction") or {}).get("all_match"),
        "the_one_that_does_not": (
            "the card says performer 1's shoulder line is off by more than 15 % on 16 "
            "frames after D8. Measured 18 on the shipped D9 build, and D8's own committed "
            "review (docs/reviews/occlusion-repair-2026-09-05.md section 5) records the "
            "figure as 27 -> 18. D9 is converter-only and moves no landmark, so 18 is what "
            "both builds carry and the card's 16 reproduces on neither. B4's band is on "
            "the AFTER value and is not moved by this."),
        "verdict": "PASS" if (stability.get("verdict") == "PASS") else "FAIL",
    }

    # ------------------------------------------------------------------- synthetic
    synthetic = load(OUT_DIR / "synthetic.json") or {}
    report["synthetic"] = {
        "band": "the selector (today / (a) / (b) / (c) across a ceiling sweep), BOTH oracle "
                "clauses, and the two must-fails",
        "report": "artifacts/compare/d8b-length/synthetic.json",
        "injection": synthetic.get("injection"),
        "selector": synthetic.get("selector"),
        "selector_metric_audit": synthetic.get("selector_metric_audit"),
        "ceiling_confirmation": synthetic.get("ceiling_confirmation"),
        "oracle_clean_fully_seen": synthetic.get("oracle_clean_fully_seen"),
        "oracle_uncollapsed_frames": synthetic.get("oracle_uncollapsed_frames"),
        "must_fail_ceiling_that_eats_the_legs": synthetic.get(
            "must_fail_ceiling_that_eats_the_legs"),
        "must_fail_frozen_arm": synthetic.get("must_fail_frozen_arm"),
        "fixture_honest_segment_spread": synthetic.get("fixture_honest_segment_spread"),
        "factor_sensitivity": synthetic.get("factor_sensitivity"),
        "raw_array_untouched_on_the_fixture": synthetic.get("raw_array_untouched"),
        "fixture_blind_to": (synthetic.get("fixture") or {}).get("blind_to"),
        "verdict": "PASS" if synthetic.get("verdict") == "PASS" else "FAIL",
    }

    # ---------------------------------------------------------------- synthetic v2
    # THE REVIEWER'S REPAIR, 2026-09-06. D8b was not merged; both failed clauses were traced
    # to measured defects in the v1 fixture, so the fixture was repaired and the SAME
    # pre-registered clauses were rerun on it. `synthetic` above is left exactly as it was.
    v2 = load(OUT_DIR / "synthetic-v2.json") or {}
    report["synthetic_v2"] = {
        "band": "THE SAME CLAUSES AS `synthetic`, rerun on a repaired fixture. No band, no "
                "ceiling and no line of `src/` was changed between the two runs -- only the "
                "fixture.",
        "report": "artifacts/compare/d8b-length/synthetic-v2.json",
        "what_was_repaired": {
            "defect_1_honest_frames_were_not_honest": {
                "measured_on_v1": "the fixture's own UNCOLLAPSED legs spread -13.2 %/+58.2 % "
                                  "at p5-p95 against the reference take's -5.1 %/+6.2 %",
                "repair": "a scale on I7's heavy-tail MAGNITUDE (the model and the sigma are "
                          "unchanged), CALIBRATED to 0.20 -- the largest scale whose honest "
                          "legs stay inside the take's own spread",
                "calibration_report":
                    "artifacts/compare/d8b-length/synthetic-v2-noise-calibration.json",
                "diagnosis": "at scale 0.00 the honest legs spread EXACTLY 0.0000, so the "
                             "geometry, the replayed mask and the two-view leg slots "
                             "contribute nothing: all of v1's spread was the noise "
                             "amplitude, applied at full sigma to all seventeen mapped "
                             "landmarks where I7 applies it to six",
                "result_on_v2": "-3.8 %/+5.5 % p5-p95 with 0 frames off by more than 15 % on "
                                "all eight leg segments, against the take's -5.1 %/+6.2 % "
                                "and 0 frames off",
                "is_a_fixture_parameter_not_a_shipped_constant": True,
            },
            "defect_2_the_collapse_ran_on_a_static_clip": {
                "measured_on_v1": "the six scored landmarks travelled 10.2 mm median over "
                                  "the injected run, so a frozen arm's error was bounded at "
                                  "19 mm against a 99 mm fault and the must-fail could not "
                                  "lose",
                "the_takes_own_figure": "188.1 mm median over frames 110-122 at 40.7 mm per "
                                        "frame of shoulder travel (performer 1, smoothed)",
                "repair": "the collapse is injected on the run with the MOST landmark "
                          "travel, on the fastest clip and stride the motion source has",
                "measured_across_every_clip_and_stride": (
                    "no motion source reaches the take's rate. The fastest usable is the "
                    "squat clip at stride 2 (21.3 mm/frame); the dialogue clip reaches 8.7 "
                    "and the acting clip 2.6. The shortfall is stated, not closed"),
                "result_on_v2": "105.7 mm median travel over the run (201.8 max), 10x v1 and "
                                "56 % of the take's; the frozen arm now scores 158.7 mm "
                                "against the candidate's 8.7",
                "two_travel_figures_and_they_differ": (
                    "105.7 mm is measured from the RUN's first frame (take frame 6); the "
                    "frozen control freezes at take frame 0, six frames earlier, so ITS "
                    "bound is 155.1 mm and it scores 158.7 -- at its bound, not beyond it. "
                    "Both numbers are in the v2 report and they answer different questions."),
                "what_actually_places_the_shoulders_on_v2": (
                    "D8's GAP CLAUSE, not D8b's ray handling. `demote` and `reject` produce "
                    "BIT-IDENTICAL shoulders on the run (0.0 mm) because the kept rays point "
                    "at the collapsed point and the solve's own reprojection gate refuses "
                    "the recovery, so the slot falls to `_hold_long_gaps_on_parent` under "
                    "both. Turning that clause off moves the shoulders 8.5 -> 33.9 mm from "
                    "truth. On this fixture D8b marks the slots and a D8 rule places them; "
                    "the two modes differ only at the cascade-marked elbows, by up to "
                    "45.3 mm."),
            },
        },
        "fixture": {k: (v2.get("fixture") or {}).get(k)
                    for k in ("clips", "stride", "frames", "noise_scale")},
        "injection": v2.get("injection"),
        "selector": v2.get("selector"),
        "selector_metric_audit": v2.get("selector_metric_audit"),
        "oracle_clean_fully_seen": v2.get("oracle_clean_fully_seen"),
        "oracle_uncollapsed_frames": v2.get("oracle_uncollapsed_frames"),
        "must_fail_ceiling_that_eats_the_legs": v2.get(
            "must_fail_ceiling_that_eats_the_legs"),
        "must_fail_frozen_arm": v2.get("must_fail_frozen_arm"),
        "fixture_honest_segment_spread": v2.get("fixture_honest_segment_spread"),
        "sweep": v2.get("sweep"),
        "factor_sensitivity": v2.get("factor_sensitivity"),
        "raw_array_untouched_on_the_fixture": v2.get("raw_array_untouched"),
        "verdict": "PASS" if v2.get("verdict") == "PASS" else "FAIL",
    }

    # ------------------------------------------------------------------------- B1
    silhouette = load(OUT_DIR / "silhouette-partwise.json") or {}
    report["B1_silhouette_arms_and_torso"] = {
        "band": "part-wise silhouette, ARMS and TORSO, window and whole take, not worse on "
                "either performer with the CI clear on the D7b/D8 predicate; improvement "
                "predicted on performer 1's window",
        "report": "artifacts/compare/d8b-length/silhouette-partwise.json",
        "clause_verdicts": silhouette.get("preregistered_clause_verdicts"),
        "subjects": silhouette.get("subjects"),
        "statistics": silhouette.get("statistics"),
        "raw_triangulation_byte_identical": silhouette.get(
            "raw_triangulation_byte_identical"),
        "smoothed_triangulation_byte_identical": silhouette.get(
            "smoothed_triangulation_byte_identical"),
        "blind_to": (silhouette.get("preregistered") or {}).get("what_this_cannot_settle"),
        "verdict": "PASS" if silhouette.get("verdict") == "PASS" else "FAIL",
    }

    # ------------------------------------------------------------------------- B2
    report["B2_raw_identity_and_legs"] = b2_block(builds)

    # ------------------------------------------------------------------------- B3
    raw_placement = load(OUT_DIR / "delivered-vs-capture-raw.json") or {}
    smoothed_placement = load(OUT_DIR / "delivered-vs-capture-smoothed.json") or {}
    wanted = ("LeftHand", "RightHand", "LeftLowerArm", "RightLowerArm", "LeftUpperArm",
              "RightUpperArm", "Neck", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot")

    def cut(payload):
        out = {}
        for subject_key, block in (payload.get("subjects") or {}).items():
            out[subject_key] = {joint: block["joints"][joint] for joint in wanted
                                if joint in block.get("joints", {})}
        return out

    report["B3_delivered_vs_capture"] = {
        "band": "`delivered_vs_capture` against the RAW points AND against the repaired "
                "ones, D9 vs D8b, identical draws, hands and elbows",
        "reports": ["artifacts/compare/d8b-length/delivered-vs-capture-raw.json",
                    "artifacts/compare/d8b-length/delivered-vs-capture-smoothed.json"],
        "vs_raw": {
            "reference": raw_placement.get("reference"),
            "same_denominator": raw_placement.get("same_denominator"),
            "joints": cut(raw_placement),
        },
        "vs_repaired": {
            "reference": smoothed_placement.get("reference"),
            "same_denominator": smoothed_placement.get("same_denominator"),
            "same_denominator_expected": (
                "FALSE. Each arm is scored against ITS OWN smoothed landmarks and D8b "
                "moves them, so this arm has no shared denominator by construction. It is "
                "reported because the card asks for it and because it says how far each "
                "delivery sits from the points it was actually solved onto -- never as a "
                "comparison between the two."),
            "joints": cut(smoothed_placement),
        },
        "why_raw_is_the_honest_one": (
            "scoring a D8b delivery against the SMOOTHED landmarks scores it against the "
            "very points D8b repaired. The raw array is bit-identical across the builds "
            "(B2) and is the only shared reference. AND IT IS BIASED AGAINST THE REPAIR on "
            "exactly the slots the step acts on: on a withheld slot the raw point is the "
            "one the step judged unreliable, so agreeing with it scores well and "
            "disagreeing scores badly whatever the truth is. D8 recorded this blindness "
            "(occlusion-repair-2026-09-05.md section 7.8) and it is why the card lists "
            "this band as reported."),
        "verdict": "REPORTED -- the card does not band this arm",
    }

    # ------------------------------------------------------------------------- B4
    report["B4_captured_frames_off"] = b4_block()

    # ------------------------------------------------------------------------- B5
    report["B5_reported_arms"] = {
        "band": "MAMMA's joints on the window through `subject_map`, report only "
                "(predicted closer on the shoulders and elbows, as D8 was); the head gate "
                "rerun; the D3 closure",
        "mamma_selects_nothing": ("CLAUDE.md: MAMMA is a measuring instrument, never in "
                                  "the shipping path. Nothing here selected anything, and "
                                  "no MAMMA figure entered src or a constant."),
        "subject_map": ("tools/head/subject_map.py -- MAMMA's body_id-00 is our subject 1; "
                        "pairing by index silently crosses the performers."),
        "scoreboard_D9": load(ROOT / "artifacts/compare/scoreboard-d9-shipped.json"),
        "scoreboard_D8b": load(ROOT / "artifacts/compare/scoreboard-d8b-length.json"),
        "head_gate": head_gate_block(),
        "d3_closure": d3_closure(builds),
        "verdict": "REPORTED",
    }

    # ----------------------------------------------------------------- the merge rule
    clause_verdicts = (silhouette.get("preregistered_clause_verdicts") or {})
    b1_both = all(
        value.get("verdict") == "PASS"
        for s in (0, 1)
        for name, value in clause_verdicts.get(f"subject_{s:02d}", {}).items()
        if name.startswith("clause_"))
    b2 = report["B2_raw_identity_and_legs"]["verdict"] == "PASS"

    def clauses_for(doc: dict) -> dict:
        selector = doc.get("selector") or {}
        return {
            "synthetic_selector_holds_for_the_shipped_ceiling": bool(
                selector.get("beats_today")),
            "oracle_clean_fully_seen": bool(
                (doc.get("oracle_clean_fully_seen") or {}).get("passes")),
            "oracle_uncollapsed_frames": bool(
                (doc.get("oracle_uncollapsed_frames") or {}).get("passes")),
            "B1_on_both_performers": bool(b1_both),
            "B2": b2,
        }

    on_v1, on_v2 = clauses_for(synthetic), clauses_for(v2)
    report["merge_rule"] = {
        "rule": "the synthetic selector holds for the shipped ceiling AND both oracle "
                "clauses AND B1 on both performers AND B2; the rest reports",
        "fixed": "before any number in this file existed, in the card at 17e0dbc",
        "applied_on": "FIXTURE v2 -- the repaired one. The clauses are the card's, "
                      "unchanged; the fixture they are scored on was rebuilt after the "
                      "reviewer traced both failures to it, and `on_fixture_v1` below keeps "
                      "the first outcome on the record.",
        "clauses": on_v2,
        "outcome": "MERGE" if all(on_v2.values()) else "DO NOT MERGE",
        "failed_clauses": [name for name, ok in on_v2.items() if not ok],
        "on_fixture_v1": {
            "clauses": on_v1,
            "outcome": "MERGE" if all(on_v1.values()) else "DO NOT MERGE",
            "failed_clauses": [name for name, ok in on_v1.items() if not ok],
            "why_it_failed": ("the v1 fixture's honest legs spread -13.2 %/+58.2 % against "
                              "the take's -5.1 %/+6.2 %, so `zero rejects on un-collapsed "
                              "frames` was asked of a body whose honest frames were already "
                              "broken; and its collapse ran on a clip where the landmarks "
                              "travelled 10.2 mm, so the pooled selector was bimodal by "
                              "construction and a frozen arm beat every candidate. Both are "
                              "measured instrument defects, and neither is evidence about "
                              "the rule."),
        },
        "what_changed_between_them": ("the FIXTURE only. No band was moved, no ceiling was "
                                      "changed, and `src/` is byte-identical between the two "
                                      "runs."),
        "selector_reading": ("on v2 the selector is read on the COLLAPSED-SHOULDER cell, "
                             "which is what the card names ('scoring 3D error of the "
                             "shoulders, elbows and wrists in the run'); the pooled median "
                             "is reported beside it and STILL reads worse than today "
                             "(6.09 -> 8.67), for the dilution reason recorded on v1. The "
                             "difference is that on v2 the pooled metric is no longer a "
                             "gate a constant can pass -- the frozen arm scores 158.7 mm "
                             "against the candidate's 8.7 -- so the refutation recorded on "
                             "v1 is preserved and no longer decides anything."),
    }
    report["verdict"] = report["merge_rule"]["outcome"]

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"merge_rule": report["merge_rule"]}, indent=1))
    for key in ("instrument_first", "synthetic", "synthetic_v2",
                "B1_silhouette_arms_and_torso",
                "B2_raw_identity_and_legs", "B3_delivered_vs_capture",
                "B4_captured_frames_off", "B5_reported_arms"):
        print(f"  {key:36s} {report.get(key, {}).get('verdict')}")
    print(f"\nwrote {out}")
    return 0 if report["verdict"] == "MERGE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
