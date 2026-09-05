#!/usr/bin/env python3
"""Ladder extractor for D8 -- the captured limbs under occlusion.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d8_occlusion import x_occlusion_repair

and route its figures by what each REFERENCES: the `capture_` keys to rung 4 (the
triangulation and the temporal stage, scored against the performer's own captured body),
the `placement_` keys to rung 7 (the converter, from the delivered file's own bytes) and
the `silhouette_` keys to rung 1 (the masks). One extractor call, three destinations.

It reads exactly one report, `artifacts/compare/d8-occlusion/gate.json`.

FOUR REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * THE PERFORMER'S OWN CAPTURED BODY. A shoulder line or a forearm against the median of
    that same performer's own take. A count of frames, not a millimetre. It is a
    SELF-consistency figure and it is blind to truth by construction -- a frozen limb
    scores perfectly -- which is why it is a headline and never a band.
  * OUR OWN RAW CAPTURED LANDMARKS, read against the delivered file's own bytes. The raw
    array is bit-identical between the D7b and D8 builds and is the one reference this step
    did not move; scoring against the smoothed landmarks would score D8 against the very
    points D8 repaired.
  * SYNTHETIC TRUTH. Millimetres against a known answer, on the fixture only. The only
    place a D8 constant was ever selected.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE HEADLINE. In the push-and-fall window cameras B001 and D001 lose one performer entirely
-- B001 on 20 of the window's 41 frames, D001 on 32 -- leaving A001 and C001, which sit
171-172 degrees apart at that subject. Two near-collinear rays fix a point across their
common axis and not along it, so the captured shoulder line collapses to 68 mm and stretches
to 554 while every epipolar and reprojection gate stays satisfied. It is a CONDITIONING
failure and no residual threshold can see one. D8 demotes such a slot to the sequence solve,
which recovers it from the same rays plus the performer's own limb lengths and continuity.

WHAT THE CHART CANNOT SAY. The frames-off-median count is self-referential: a limb welded to
its own median scores zero. That is why the synthetic bars sit on their own chart against a
known answer, and why the photographs are the band.

Self-check:  python3 tools/compare/extractors/d8_occlusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d8-occlusion/gate.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/captured_limb_stability.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8_occlusion_synthetic.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8_occlusion_delivery.py "
    "--out artifacts/compare/d8-occlusion/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8_occlusion_silhouette.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8_occlusion_gate.py"
)

REF_SELF = ("THE PERFORMER'S OWN CAPTURED BODY: each segment against the median length of "
            "that same performer over that same take. A count of frames, not a distance, "
            "and blind to truth by construction -- a frozen limb scores perfectly")
REF_RAW = ("OUR OWN RAW captured landmarks -- the delivery's own "
           "`raw_triangulated_world_positions_z_up_m`, NaNs intact, against the delivered "
           "GLB read from its own bytes. The one array D8 does not move, so it is the one "
           "honest reference for a D8 arm")
REF_SYNTH = ("SYNTHETIC TRUTH: SOMASKEL77 clips posed through our own FK and projected into "
             "this rig, with the real per-camera seen pattern replayed and our own "
             "detector's heavy-tail noise. The exact answer is known")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"


def _subject_label(key: str) -> str:
    return key.replace("subject_", "performer ")


def x_occlusion_repair(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []

    # ------------------------------------------------- the captured body, self-referenced
    stability = _load("artifacts/compare/d8-occlusion/limb-stability.json") or {}
    after = _load("artifacts/compare/d8-occlusion/limb-stability-after.json") or {}
    for subject in ("subject_00", "subject_01"):
        label = _subject_label(subject)
        for source, role, key, note in (
                (stability, "control", "before",
                 "the shipped D7b build -- the defect this step closes"),
                (after, "ours", "after", "after the occlusion repair")):
            block = ((source.get("subjects", {}).get(subject, {})
                      .get("segments", {}).get("smoothed", {})
                      .get("shoulder_line", {}).get("left_shoulder__right_shoulder")) or {})
            value = block.get("frames_off_median_by_more_than_15pct")
            entry = fig(
                f"D8 captured shoulder line, frames off the performer's own width by more "
                f"than 15 %, {label}, {key}",
                value, "frames of 150", REF_SELF, LOWER,
                key=f"capture_shoulder_off_{key}_{subject}", note=note)
            (figs if role == "ours" else ctrls).append(entry)
        for source, role, key in ((stability, "control", "before"), (after, "ours", "after")):
            block = ((source.get("subjects", {}).get(subject, {})
                      .get("segments", {}).get("smoothed", {})
                      .get("arms", {}).get("left_elbow__left_wrist")) or {})
            entry = fig(
                f"D8 captured forearm L, frames off the performer's own length by more "
                f"than 15 %, {label}, {key}",
                block.get("frames_off_median_by_more_than_15pct"), "frames of 150",
                REF_SELF, LOWER, key=f"capture_forearm_off_{key}_{subject}")
            (figs if role == "ours" else ctrls).append(entry)
        # The legs are the control that says the instrument is not simply counting motion.
        block = ((stability.get("subjects", {}).get(subject, {})
                  .get("segments", {}).get("smoothed", {})
                  .get("legs", {}).get("left_hip__left_knee")) or {})
        ctrls.append(fig(
            f"D8 control: the captured thigh, same measure, {label}",
            block.get("frames_off_median_by_more_than_15pct"), "frames of 150",
            REF_SELF, LOWER, key=f"capture_thigh_off_before_{subject}",
            note="the legs hold their length through the same window on both performers; "
                 "0 frames off. The defect is the arms and the shoulder line, not motion"))

    # ------------------------------------------------------- the delivered file, vs RAW
    placement = (r.get("B4_delivered_vs_raw_capture") or {}).get("joints") or {}
    for subject, block in placement.items():
        label = _subject_label(subject)
        for joint, short in (("LeftHand", "left hand"), ("LeftLowerArm", "left elbow")):
            row = block.get(joint) or {}
            for arm, role, key in (("D7b", "control", "d7b"), ("D8", "ours", "d8")):
                cell = row.get(arm) or {}
                entry = fig(
                    f"D8 delivered {short} to the RAW captured point, worst 5 % of frames, "
                    f"{label}, {arm}",
                    cell.get("p95_mm"), "mm p95", REF_RAW, LOWER,
                    key=f"placement_{joint.lower()}_p95_{key}_{subject}",
                    note="read from the delivered GLB's own bytes")
                (figs if role == "ours" else ctrls).append(entry)

    # --------------------------------------------------------------- synthetic truth
    arms = (r.get("synthetic") or {}).get("arms") or {}
    for name, role, key, note in (
            ("today", "control", "today", "the shipped D7b code on the same fixture"),
            ("conditioning", "alt", "conditioning", "the conditioning gate alone"),
            ("reachability", "alt", "reachability", "the reachability reject alone"),
            ("both", "ours", "both", "what ships: all three rules")):
        entry = fig(f"D8 synthetic: 3D error in the two-view window, {name}",
                    (arms.get(name) or {}).get("median_mm"), "mm median",
                    REF_SYNTH, LOWER, key=f"synthetic_error_{key}", note=note)
        (figs if role == "ours" else ctrls).append(entry)
    frozen = (r.get("synthetic") or {}).get("must_fail_frozen_arm") or {}
    ctrls.append(fig("D8 synthetic must-fail: a frozen arm",
                     (frozen.get("score") or {}).get("median_mm"), "mm median",
                     REF_SYNTH, LOWER, key="synthetic_error_frozen",
                     note="every arm landmark held at its own first frame for the whole "
                          "take -- the degenerate the gap clause must never become"))

    # ------------------------------------------------------------------ the photographs
    silhouette = (r.get("B1_silhouette_arms_on_the_window") or {}).get("subjects") or {}
    for subject, block in silhouette.items():
        label = _subject_label(subject)
        cell = (block.get("cuts") or {}).get("window") or {}
        figs.append(fig(f"D8 arms in the photographs, window frames, {label}",
                        cell.get("arm_iou_D8"), "IoU median", REF_MASKS, HIGHER,
                        key=f"silhouette_arms_window_d8_{subject}",
                        note="arms only, the 41 frames of the push-and-fall window"))
        ctrls.append(fig(f"D8 control: the same photographs before, {label}",
                         cell.get("arm_iou_D7b"), "IoU median", REF_MASKS, HIGHER,
                         key=f"silhouette_arms_window_d7b_{subject}"))
        ctrls.append(fig(f"D8: the reference fitter's own mesh, arms, window, {label}",
                         cell.get("arm_iou_ORACLE"), "IoU median", REF_MASKS, HIGHER,
                         key=f"silhouette_arms_window_oracle_{subject}",
                         note="the ceiling a body of the wrong proportions can reach, "
                              "not a target"))
    return figs, ctrls


VISUALS = {
    "capture": [
        dict(title="D8: how often the captured shoulders are the wrong width apart",
             plain="Each performer's own shoulder width, measured over their own take, is "
                   "the reference; the bar counts the frames where the captured shoulders "
                   "came out more than 15 % away from it. Lower is better. Where the two "
                   "bodies overlap, two of the four cameras lose a performer and the two "
                   "that remain are almost exactly opposite each other, so they agree "
                   "perfectly on a point that is wrong -- the shoulder line collapses to "
                   "68 mm and stretches to 554. The hatched bar is the thigh, measured the "
                   "same way through the same frames: it never goes wrong, which is how we "
                   "know this is not simply fast motion.",
             better="lower",
             bars=[dict(label="Ships (after the repair), performer 0", role="ours", key="capture_shoulder_off_after_subject_00"),
                   dict(label="Before, performer 0", role="control", key="capture_shoulder_off_before_subject_00"),
                   dict(label="Ships (after the repair), performer 1", role="ours", key="capture_shoulder_off_after_subject_01"),
                   dict(label="Before, performer 1", role="control", key="capture_shoulder_off_before_subject_01"),
                   dict(label="The thigh, before, performer 1", role="control", key="capture_thigh_off_before_subject_01")]),
        dict(title="D8: the same check on the forearm",
             plain="Same reference, same unit, lower is better. The forearm of the "
                   "performer who falls changes length on 18 frames of the take, all of "
                   "them inside the window where two cameras have lost her. A bone cannot "
                   "change length, so every one of those frames is an error we were "
                   "shipping.",
             better="lower",
             bars=[dict(label="Ships, performer 1", role="ours", key="capture_forearm_off_after_subject_01"),
                   dict(label="Before, performer 1", role="control", key="capture_forearm_off_before_subject_01"),
                   dict(label="Ships, performer 0", role="ours", key="capture_forearm_off_after_subject_00"),
                   dict(label="Before, performer 0", role="control", key="capture_forearm_off_before_subject_00")]),
    ],
    "placement": [
        dict(title="D8: how far the delivered left hand sits from the point the cameras actually saw",
             plain="The reference is the raw triangulated point -- what the cameras "
                   "measured before any repair, and the one thing this step does not "
                   "change, so both bars are scored against exactly the same numbers. "
                   "Lower is better, and this is the worst 5 % of frames rather than the "
                   "typical one, because the failure is confined to a few dozen frames and "
                   "a median would hide it. There is no floor bar here: unlike the trunk, "
                   "nothing about a hand's placement is impossible to reach.",
             better="lower",
             bars=[dict(label="Ships, performer 1", role="ours", key="placement_lefthand_p95_d8_subject_01"),
                   dict(label="Before, performer 1", role="control", key="placement_lefthand_p95_d7b_subject_01"),
                   dict(label="Ships, performer 0", role="ours", key="placement_lefthand_p95_d8_subject_00"),
                   dict(label="Before, performer 0", role="control", key="placement_lefthand_p95_d7b_subject_00")]),
        dict(title="D8: the same check at the elbow",
             plain="Same reference, same unit, worst 5 % of frames, lower is better. The "
                   "elbow is the joint above the hand, so a hand that has been put back "
                   "in the right place by moving the elbow would show here too.",
             better="lower",
             bars=[dict(label="Ships, performer 1", role="ours", key="placement_leftlowerarm_p95_d8_subject_01"),
                   dict(label="Before, performer 1", role="control", key="placement_leftlowerarm_p95_d7b_subject_01"),
                   dict(label="Ships, performer 0", role="ours", key="placement_leftlowerarm_p95_d8_subject_00"),
                   dict(label="Before, performer 0", role="control", key="placement_leftlowerarm_p95_d7b_subject_00")]),
        dict(title="D8: measured against a known answer, where truth exists",
             plain="Everything else on this page is measured against our own capture or "
                   "against photographs. This one is measured against an exactly known "
                   "answer: a synthetic performer, posed and filmed by the same four "
                   "cameras, with the real pattern of camera dropouts replayed on top so "
                   "the failure happens by construction. Lower is better. The hatched bars "
                   "are the code before this step and a deliberately wrong answer -- an arm "
                   "frozen in place for the whole take -- which has to score worse, and "
                   "does. Every number this step ships was chosen here and nowhere else.",
             better="lower",
             bars=[dict(label="Ships (all three rules)", role="ours", key="synthetic_error_both"),
                   dict(label="Before this step", role="control", key="synthetic_error_today"),
                   dict(label="The conditioning gate alone", role="alt", key="synthetic_error_conditioning"),
                   dict(label="The reachability reject alone", role="alt", key="synthetic_error_reachability"),
                   dict(label="Wrong answer: a frozen arm", role="control", key="synthetic_error_frozen")]),
    ],
    "masks": [
        dict(title="D8: how well the arms cover the person in the photographs, on the frames where it went wrong",
             plain="The reference fitter's person masks are the reference; higher is "
                   "better. Arms only, and only the 41 frames where two of the four "
                   "cameras have lost a performer -- the frames this step is about. The "
                   "hatched bars are the build before this one and the reference fitter's "
                   "own mesh, which is the ceiling a body of the wrong proportions can "
                   "reach rather than a target. A short window gives a wide margin of "
                   "uncertainty and that is stated on the technical page.",
             better="higher",
             bars=[dict(label="Ships, performer 1", role="ours", key="silhouette_arms_window_d8_subject_01"),
                   dict(label="Before, performer 1", role="control", key="silhouette_arms_window_d7b_subject_01"),
                   dict(label="Reference fitter's mesh, performer 1", role="control", key="silhouette_arms_window_oracle_subject_01"),
                   dict(label="Ships, performer 0", role="ours", key="silhouette_arms_window_d8_subject_00"),
                   dict(label="Before, performer 0", role="control", key="silhouette_arms_window_d7b_subject_00"),
                   dict(label="Reference fitter's mesh, performer 0", role="control", key="silhouette_arms_window_oracle_subject_00")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_occlusion_repair({})
    for f in figs + ctrls:
        print(f"{f['key']:52s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
    keys = {f["key"] for f in figs + ctrls}
    missing = [b["key"] for group in VISUALS.values() for chart in group
               for b in chart["bars"] if b["key"] not in keys]
    print("VISUALS keys missing from the extractor:", missing or "NONE")
