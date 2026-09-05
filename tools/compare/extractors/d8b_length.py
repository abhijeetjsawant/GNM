#!/usr/bin/env python3
"""Ladder extractor for D8b -- captured segments that break the performer's own bone length.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d8b_length import x_segment_length_reject

and route its figures by what each REFERENCES: the `capture_` keys to rung 4 (the
triangulation, scored against the performer's own captured body), the `synthetic_` keys to
rung 4 as well (the same stage against a known answer), the `placement_` keys to rung 7 (the
converter, from the delivered file's own bytes) and the `silhouette_` keys to rung 1 (the
masks). One extractor call, three destinations.

It reads exactly one report, `artifacts/compare/d8b-length/gate.json`, plus the two
stability reports the gate names.

FOUR REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * THE PERFORMER'S OWN CAPTURED BODY. A shoulder line or a forearm against the median of
    that same performer's own take. A count of frames, not a millimetre. SELF-consistency,
    blind to truth by construction -- a limb welded to its own median scores zero -- which is
    why it is a headline and never a band on its own.
  * SYNTHETIC TRUTH, with an injected consistent collapse. Millimetres against a known
    answer, on the fixture only, and this is where the recovery MODE was selected.
  * OUR OWN RAW CAPTURED LANDMARKS, read against the delivered file's own bytes. The raw
    array is bit-identical between the D9 and D8b builds and is the one reference this step
    did not move.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE HEADLINE. On frames 110-122 the falling performer's captured shoulder line reads
122-274 mm against his own 364, while cameras A001 and C001 support both shoulders on all
thirteen frames, D001 on twelve, and all three agree with the collapsed point to 0.5-6.9 px.
It is a WELL-CONDITIONED triangulation of a point the 2D detector places wrongly in every
view, so no epipolar, reprojection or ray-angle gate can fire -- D8's conditioning gate
correctly does not. The performer's own bone lengths are the only evidence that can see it.

WHAT THE CHARTS CANNOT SAY, and it is the whole reason this step did not merge. The
frames-off-median count is self-referential: the candidate optimises it directly. The
synthetic bars sit on their own chart against a known answer -- and on that fixture the
step's OWN two honest-motion clauses failed, because the fixture's honest legs swing
-9.6 %/+25.6 % at p5-p95 against the reference take's -5.1 %/+6.2 %. The photographs cannot
resolve the change at all (differences of 1e-4 to 2e-3 IoU, every interval spanning zero).
See `docs/reviews/segment-length-2026-09-06.md`.

Self-check:  python3 tools/compare/extractors/d8b_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d8b-length/gate.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/captured_limb_stability.py "
    "--reproduce d8b --out artifacts/compare/d8b-length/limb-stability-d9.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8b_length_synthetic.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8b_length_delivery.py "
    "--out artifacts/compare/d8b-length/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/captured_limb_stability.py "
    "--landmarks-from artifacts/compare/d8b-length/delivery --skip-reproduction "
    "--out artifacts/compare/d8b-length/limb-stability-after.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/delivered_vs_capture.py "
    "--delivery D9=artifacts/commercial-multiview-soma77 "
    "--delivery D8b=artifacts/compare/d8b-length/delivery --reference raw "
    "--out artifacts/compare/d8b-length/delivered-vs-capture-raw.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8b_length_silhouette.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8b_length_gate.py"
)

REF_SELF = ("THE PERFORMER'S OWN CAPTURED BODY: each segment against the median length of "
            "that same performer over that same take. A count of frames, not a distance, "
            "and blind to truth by construction -- a limb welded to its own median scores "
            "perfectly")
REF_SYNTH = ("SYNTHETIC TRUTH with an injected consistent collapse: SOMASKEL77 clips posed "
             "through our own FK and projected into this rig, the real per-camera seen "
             "pattern replayed, our own detector's heavy-tail noise, and both shoulders "
             "moved toward the neck in EVERY view for twelve frames. The exact answer is "
             "known")
REF_RAW = ("OUR OWN RAW captured landmarks -- the delivery's own "
           "`raw_triangulated_world_positions_z_up_m`, NaNs intact, against the delivered "
           "GLB read from its own bytes. The one array D8b does not move")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"

BEFORE = "artifacts/compare/d8b-length/limb-stability-d9.json"
AFTER = "artifacts/compare/d8b-length/limb-stability-after.json"


def _subject_label(key: str) -> str:
    return key.replace("subject_", "performer ")


def _segment(doc: dict, subject: str, group: str, pair: str) -> dict:
    return ((doc.get("subjects", {}).get(subject, {}).get("segments", {})
             .get("smoothed", {}).get(group, {}).get(pair)) or {})


def x_segment_length_reject(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []

    # ------------------------------------------------- the captured body, self-referenced
    before = _load(BEFORE) or {}
    after = _load(AFTER) or {}
    for subject in ("subject_00", "subject_01"):
        label = _subject_label(subject)
        for doc, role, key, note in (
                (before, "control", "d9",
                 "the shipped D9 build -- the defect this step is about"),
                (after, "ours", "d8b", "after the segment-length reject")):
            block = _segment(doc, subject, "shoulder_line",
                             "left_shoulder__right_shoulder")
            entry = fig(
                f"D8b captured shoulder line, frames off the performer's own width by more "
                f"than 15 %, {label}, {key.upper()}",
                block.get("frames_off_median_by_more_than_15pct"), "frames of 150",
                REF_SELF, LOWER, key=f"capture_shoulder_off_{key}_{subject}", note=note)
            (figs if role == "ours" else ctrls).append(entry)
        for doc, role, key in ((before, "control", "d9"), (after, "ours", "d8b")):
            block = _segment(doc, subject, "arms", "left_elbow__left_wrist")
            entry = fig(
                f"D8b captured forearm L, frames off the performer's own length by more "
                f"than 15 %, {label}, {key.upper()}",
                block.get("frames_off_median_by_more_than_15pct"), "frames of 150",
                REF_SELF, LOWER, key=f"capture_forearm_off_{key}_{subject}")
            (figs if role == "ours" else ctrls).append(entry)
        # The legs are the control that says the instrument is not simply counting motion,
        # and the HIP LINE is the control that says the step is not finished: it breaks the
        # same way and is not in the shipped rule's segment list.
        thigh = _segment(before, subject, "legs", "left_hip__left_knee")
        ctrls.append(fig(
            f"D8b control: the captured thigh, same measure, {label}",
            thigh.get("frames_off_median_by_more_than_15pct"), "frames of 150",
            REF_SELF, LOWER, key=f"capture_thigh_off_d9_{subject}",
            note="the legs hold their length through the same frames on both performers; "
                 "0 frames off. The defect is the arms and the shoulder line, not motion"))
        hips = _segment(after, subject, "hip_line", "left_hip__right_hip")
        ctrls.append(fig(
            f"D8b left open: the captured hip line, same measure, {label}",
            hips.get("frames_off_median_by_more_than_15pct"), "frames of 150",
            REF_SELF, LOWER, key=f"capture_hipline_off_d8b_{subject}",
            note="the hip line collapses the same way on the falling performer and is NOT "
                 "in the shipped rule's segment list, because the card's list does not "
                 "name it. Unchanged by this step and recorded as open"))

    # --------------------------------------------------------------- synthetic truth
    groups = ((r.get("synthetic") or {}).get("selector") or {}).get(
        "per_mode_by_group_median_mm") or {}
    shoulders = groups.get("shoulders") or {}
    for name, role, key, note in (
            ("today", "control", "today", "the shipped D9 code on the same collapsed clip"),
            ("demote", "ours", "demote", "what the selector chose: the point withheld, the "
                                         "rays kept for the sequence solve"),
            ("reject", "alt", "reject", "the point AND the rays withheld"),
            ("best_ray", "alt", "best_ray", "only the highest-confidence camera's ray kept")):
        entry = fig(f"D8b synthetic: 3D error of the collapsed shoulders, {name}",
                    shoulders.get(name), "mm median", REF_SYNTH, LOWER,
                    key=f"synthetic_shoulder_error_{key}", note=note)
        (figs if role == "ours" else ctrls).append(entry)
    frozen = ((r.get("synthetic") or {}).get("must_fail_frozen_arm") or {})
    ctrls.append(fig(
        "D8b synthetic control: a frozen arm",
        ((frozen.get("score") or {}).get("shoulders") or {}).get("median_mm"),
        "mm median", REF_SYNTH, LOWER, key="synthetic_shoulder_error_frozen",
        note="every arm landmark held at its own first frame. On this fixture the arms "
             "travel only 10.2 mm over the injected run, so this control is UNINFORMATIVE "
             "here and is shown as the weak control it is, not as a pass"))

    # ------------------------------------------------------- the delivered file, vs RAW
    placement = ((r.get("B3_delivered_vs_capture") or {}).get("vs_raw") or {}).get(
        "joints") or {}
    for subject, block in placement.items():
        label = _subject_label(subject)
        for joint, short in (("LeftUpperArm", "left shoulder joint"),
                             ("LeftLowerArm", "left elbow")):
            row = block.get(joint) or {}
            for arm, role, key in (("D9", "control", "d9"), ("D8b", "ours", "d8b")):
                cell = row.get(arm) or {}
                entry = fig(
                    f"D8b delivered {short} to the RAW captured point, {label}, {arm}",
                    cell.get("median_mm"), "mm median", REF_RAW, LOWER,
                    key=f"placement_{joint.lower()}_median_{key}_{subject}",
                    note="read from the delivered GLB's own bytes. The raw point is the "
                         "one the step judged unreliable on exactly the frames it acts on, "
                         "so this reference is BIASED AGAINST the repair there")
                (figs if role == "ours" else ctrls).append(entry)

    # ------------------------------------------------------------------ the photographs
    silhouette = (r.get("B1_silhouette_arms_and_torso") or {}).get("subjects") or {}
    for subject, block in silhouette.items():
        label = _subject_label(subject)
        cell = (block.get("cuts") or {}).get("window") or {}
        figs.append(fig(f"D8b arms in the photographs, window frames, {label}",
                        cell.get("arm_iou_D8b"), "IoU median", REF_MASKS, HIGHER,
                        key=f"silhouette_arms_window_d8b_{subject}",
                        note="arms only, the 41 frames of the push-and-fall window"))
        ctrls.append(fig(f"D8b control: the same photographs before, {label}",
                         cell.get("arm_iou_D9"), "IoU median", REF_MASKS, HIGHER,
                         key=f"silhouette_arms_window_d9_{subject}"))
        ctrls.append(fig(f"D8b: the reference fitter's own mesh, arms, window, {label}",
                         cell.get("arm_iou_ORACLE"), "IoU median", REF_MASKS, HIGHER,
                         key=f"silhouette_arms_window_oracle_{subject}",
                         note="the ceiling a body of the wrong proportions can reach, "
                              "not a target"))
    return figs, ctrls


VISUALS = {
    "capture": [
        dict(title="D8b: how often the captured shoulders are the wrong width apart",
             plain="Each performer's own shoulder width, measured over their own take, is "
                   "the reference; the bar counts the frames where the captured shoulders "
                   "came out more than 15 % away from it. Lower is better. On the falling "
                   "performer the shoulders collapse to a third of his own width for a "
                   "dozen frames while three of the four cameras SEE him and all three "
                   "agree with the wrong point to a few pixels -- the detector places both "
                   "shoulders inward in every view, and no check made of geometry can tell. "
                   "The hatched bars are the thigh, measured the same way through the same "
                   "frames, and the hip line, which breaks the same way and which this step "
                   "does not act on.",
             better="lower",
             bars=[dict(label="After the reject, performer 1", role="ours", key="capture_shoulder_off_d8b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="capture_shoulder_off_d9_subject_01"),
                   dict(label="After the reject, performer 0", role="ours", key="capture_shoulder_off_d8b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="capture_shoulder_off_d9_subject_00"),
                   dict(label="The thigh, before, performer 1", role="control", key="capture_thigh_off_d9_subject_01"),
                   dict(label="The hip line, after, performer 1", role="control", key="capture_hipline_off_d8b_subject_01")]),
        dict(title="D8b: the same check on the forearm",
             plain="Same reference, same unit, lower is better. The forearm is the segment "
                   "this step did NOT fix: on the falling performer it goes from four bad "
                   "frames to five, against a target of none. The shoulders above it were "
                   "withheld and the elbow and wrist were re-solved from what was left, and "
                   "on this take that left the forearm no better.",
             better="lower",
             bars=[dict(label="After the reject, performer 1", role="ours", key="capture_forearm_off_d8b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="capture_forearm_off_d9_subject_01"),
                   dict(label="After the reject, performer 0", role="ours", key="capture_forearm_off_d8b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="capture_forearm_off_d9_subject_00")]),
    ],
    "synthetic": [
        dict(title="D8b: measured against a known answer, where truth exists",
             plain="Everything else here is measured against our own capture or against "
                   "photographs. This one is measured against an exactly known answer: a "
                   "synthetic performer, filmed by the same four cameras, with the real "
                   "pattern of camera dropouts replayed and both shoulders deliberately "
                   "pushed inward in every view for twelve frames -- the failure this step "
                   "is about, built on purpose. Lower is better, and the bars are the "
                   "collapsed shoulders only. The three ways of handling a rejected point "
                   "were compared here and the first one won. The hatched bars are the code "
                   "before this step and a deliberately wrong answer, an arm frozen in "
                   "place; on this fixture the arms barely move, so the frozen control is "
                   "weak and is shown as such rather than counted as a pass.",
             better="lower",
             bars=[dict(label="Withhold the point, keep the rays", role="ours", key="synthetic_shoulder_error_demote"),
                   dict(label="Before this step", role="control", key="synthetic_shoulder_error_today"),
                   dict(label="Withhold the point and the rays", role="alt", key="synthetic_shoulder_error_reject"),
                   dict(label="Keep only the best camera's ray", role="alt", key="synthetic_shoulder_error_best_ray"),
                   dict(label="Weak control: a frozen arm", role="control", key="synthetic_shoulder_error_frozen")]),
    ],
    "placement": [
        dict(title="D8b: how far the delivered shoulder sits from the point the cameras saw",
             plain="The reference is the raw triangulated point -- what the cameras "
                   "measured before any repair, and the one thing this step does not "
                   "change, so both bars are scored against exactly the same numbers. Lower "
                   "is better. Read it carefully: on the frames this step acts on, the raw "
                   "point is the one the step decided was wrong, so agreeing with it scores "
                   "well and disagreeing scores badly whatever the truth is. This chart is "
                   "biased against the repair by construction and is shown for completeness.",
             better="lower",
             bars=[dict(label="After the reject, performer 1", role="ours", key="placement_leftupperarm_median_d8b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="placement_leftupperarm_median_d9_subject_01"),
                   dict(label="After the reject, performer 0", role="ours", key="placement_leftupperarm_median_d8b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="placement_leftupperarm_median_d9_subject_00")]),
        dict(title="D8b: the same check at the elbow",
             plain="Same reference, same unit, same bias, lower is better. The elbow is the "
                   "joint below the shoulder, so a shoulder put back in the right place "
                   "should show here too.",
             better="lower",
             bars=[dict(label="After the reject, performer 1", role="ours", key="placement_leftlowerarm_median_d8b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="placement_leftlowerarm_median_d9_subject_01"),
                   dict(label="After the reject, performer 0", role="ours", key="placement_leftlowerarm_median_d8b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="placement_leftlowerarm_median_d9_subject_00")]),
    ],
    "masks": [
        dict(title="D8b: how well the arms cover the person in the photographs",
             plain="The reference fitter's person masks are the reference; higher is "
                   "better. Arms only, and only the 41 frames where the capture goes wrong. "
                   "The bars are level: this change moves the pixels by less than this "
                   "instrument can resolve, and every margin of uncertainty spans zero. "
                   "That is the honest reading -- the photographs neither support nor "
                   "contradict the step. The hatched bars are the build before this one and "
                   "the reference fitter's own mesh, which is the ceiling a body of the "
                   "wrong proportions can reach rather than a target.",
             better="higher",
             bars=[dict(label="After the reject, performer 1", role="ours", key="silhouette_arms_window_d8b_subject_01"),
                   dict(label="Before, performer 1", role="control", key="silhouette_arms_window_d9_subject_01"),
                   dict(label="Reference fitter's mesh, performer 1", role="control", key="silhouette_arms_window_oracle_subject_01"),
                   dict(label="After the reject, performer 0", role="ours", key="silhouette_arms_window_d8b_subject_00"),
                   dict(label="Before, performer 0", role="control", key="silhouette_arms_window_d9_subject_00"),
                   dict(label="Reference fitter's mesh, performer 0", role="control", key="silhouette_arms_window_oracle_subject_00")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_segment_length_reject({})
    for f in figs + ctrls:
        print(f"{f['key']:52s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
    keys = {f["key"] for f in figs + ctrls}
    missing = [b["key"] for group in VISUALS.values() for chart in group
               for b in chart["bars"] if b["key"] not in keys]
    print("VISUALS keys missing from the extractor:", missing or "NONE")
