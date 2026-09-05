#!/usr/bin/env python3
"""Ladder extractor for D9 -- the arms aimed from their own origins.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d9_arms import x_arm_origin

and route its figures by what each REFERENCES: the `arm_` keys to rung 7 (the converter,
scored against our own captured landmarks), the `silhouette_` keys to rung 1 (the masks).
One extractor call, two destinations.

It reads exactly one report, `artifacts/compare/d9-arms/gate.json`.

THREE REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * OUR OWN CAPTURED LANDMARKS, read against the delivered file's own bytes. Millimetres.
  * THE ARM-AIM FLOOR, which is not a rival answer but the part of that same millimetre
    figure no aim can remove -- the bone's own LENGTH error, measured from the delivered
    `UpperArm` origin and chained through the placed elbow. It shares the axis with the
    bars above BY CONSTRUCTION (same quantity, same reference, same frames), which is why
    it may sit on the chart. It is bit-equal between the D8 and D9 builds, because D9 does
    not move that origin.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE HEADLINE. The delivered elbow and wrist come back onto the captured points: elbow
13.7 / 13.9 -> 7.0 / 6.8 mm on performer 0 and 13.4 / 15.1 -> 9.1 / 8.1 on performer 1;
wrist 14.7 / 15.6 -> 8.9 / 10.4 and 17.6 / 20.1 -> 10.9 / 11.9. All eight land within
0.0-1.6 mm of the floor, so what is left is the arm's own LENGTH error and belongs to a
fitted arm (D5), not to an aim. The `UpperArm` joint itself does NOT move -- its 10-13 mm
miss is the trunk chord plus a missing shoulder translation, also D5's -- and every joint
outside the four arm bones is bit-identical to the D8 build. The photographs agree: the
arms rise on both performers, whole take and window, and on performer 1's window the
interval is clear of zero.

ONE PREDICTION FAILS AS WRITTEN. The card predicted the elbow at 6-9 mm; performer 1's
left elbow reads 9.11, because its own floor there is 8.77 and no aim can go below it. The
band -- within 3 mm of the floor -- passes at 0.34 mm. The prediction is recorded as
refuted, and the band is not moved. See `docs/reviews/arm-origin-2026-09-05.md`.

Self-check:  python3 tools/compare/extractors/d9_arms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d9-arms/gate.json"
# Exactly what was run on the branch, in order. AFTER a merge the shipped delivery IS the
# D9 build, so the first two arms become `D8=artifacts/compare/delivered-before-d9-...`
# and `D9=artifacts/commercial-multiview-soma77`, and the labels keep their meaning.
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9_arm_delivery.py "
    "--out artifacts/compare/d9-arms/delivery-hygiene --expect-byte-identical  # src UNCHANGED && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9_arm_delivery.py "
    "--out artifacts/compare/d9-arms/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/delivered_vs_capture.py "
    "--delivery D8=artifacts/commercial-multiview-soma77 "
    "--delivery D9=artifacts/compare/d9-arms/delivery "
    "--out artifacts/compare/d9-arms/delivered-vs-capture.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9_arm_silhouette.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/mamma_scoreboard.py "
    "--tracks artifacts/compare/d9-arms/delivery --label d9-arms && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9_arm_gate.py"
)

REF_CAPTURE = ("OUR OWN captured landmarks: the delivery's own "
               "`triangulated_world_positions_z_up_m`, in absolute capture Z-up metres, "
               "against the delivered GLB read from its own bytes. Never MAMMA, never a "
               "re-detection, never a re-solve")
REF_FLOOR = (REF_CAPTURE + " -- the part of that same figure NO aim can remove: the bone's "
             "own LENGTH error, | ||captured_elbow - UpperArm_origin|| - L_upper |, and "
             "the wrist chained from the placed elbow")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"

SIDES = ("left", "right")


def _subject_label(key: str) -> str:
    return key.replace("subject_", "performer ")


def x_arm_origin(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []

    placement = r.get("B1_placement", {}).get("subjects", {})
    for subject, block in placement.items():
        label = _subject_label(subject)
        for side in SIDES:
            for part in ("elbow", "wrist"):
                cell = block.get(f"{side}_{part}") or {}
                medians = cell.get("median_mm", {})
                figs.append(fig(
                    f"D9 delivered {side} {part} to the captured {part}, {label}",
                    medians.get("D9"), "mm median", REF_CAPTURE, LOWER,
                    key=f"arm_{side}_{part}_after_{subject}",
                    note="read from the delivered GLB's own bytes"))
                ctrls.append(fig(
                    f"D9 control: the shipped D8 build's {side} {part}, {label}",
                    medians.get("D8"), "mm median", REF_CAPTURE, LOWER,
                    key=f"arm_{side}_{part}_d8_{subject}",
                    note="the defect D9 closes -- the arm was aimed by a "
                         "landmark-to-landmark direction from a displaced origin"))
                figs.append(fig(
                    f"D9: the aim FLOOR for the {side} {part}, {label}",
                    cell.get("floor_median_mm"), "mm median", REF_FLOOR, LOWER,
                    key=f"arm_{side}_{part}_floor_{subject}",
                    note="what a rigid bone cannot beat, whatever it is aimed at. "
                         "Bit-equal on both builds; handed to D5"))
                figs.append(fig(
                    f"D9 delivered {side} {part}, bent tercile, {label}",
                    (cell.get("bent_tercile_median_mm") or {}).get("D9"), "mm median",
                    REF_CAPTURE, LOWER, key=f"arm_{side}_{part}_after_bent_{subject}",
                    note="the third of frames with the most trunk tilt"))
                ctrls.append(fig(
                    f"D9 control: the D8 build's {side} {part}, bent tercile, {label}",
                    (cell.get("bent_tercile_median_mm") or {}).get("D8"), "mm median",
                    REF_CAPTURE, LOWER, key=f"arm_{side}_{part}_d8_bent_{subject}"))

    retained = r.get("B2_untouchable", {}).get("subjects", {})
    for subject, block in retained.items():
        label = _subject_label(subject)
        rows = block.get("from_the_delivered_file", {})
        for joint, key in (("LeftUpperArm", "shoulder"), ("Neck", "neck"),
                           ("LeftLowerLeg", "knee")):
            row = rows.get(joint, {})
            figs.append(fig(
                f"D9 retained: the delivered {key} to its landmark, {label}",
                row.get("D9_median_mm"), "mm median", REF_CAPTURE, LOWER,
                key=f"arm_{key}_after_{subject}",
                note="unchanged by construction: D9 aims two bones per arm and touches "
                     "nothing else. The shoulder's own miss belongs to D5"))
            ctrls.append(fig(
                f"D9 control: the same {key} on the D8 build, {label}",
                row.get("D8_median_mm"), "mm median", REF_CAPTURE, LOWER,
                key=f"arm_{key}_d8_{subject}", note="it must be IDENTICAL, and it is"))

    silhouette = r.get("B3_silhouette", {}).get("cuts", {})
    for subject, block in silhouette.items():
        label = _subject_label(subject)
        for cut in ("whole_take", "window"):
            cell = (block.get("cuts") or {}).get(cut, {})
            figs.append(fig(
                f"D9 silhouette, arms only, {cut.replace('_', ' ')}, {label}",
                cell.get("arm_iou_D9"), "IoU", REF_MASKS, HIGHER,
                key=f"silhouette_arms_after_{cut}_{subject}"))
            ctrls.append(fig(
                f"D9 control: the D8 build, arms only, {cut.replace('_', ' ')}, {label}",
                cell.get("arm_iou_D8"), "IoU", REF_MASKS, HIGHER,
                key=f"silhouette_arms_d8_{cut}_{subject}"))
            ctrls.append(fig(
                f"D9 oracle: MAMMA's own mesh, arms only, {cut.replace('_', ' ')}, {label}",
                cell.get("arm_iou_ORACLE"), "IoU", REF_MASKS, HIGHER,
                key=f"silhouette_arms_oracle_{cut}_{subject}",
                note="the CEILING a mesh of the wrong proportions can reach, not a target"))
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D9: how far the delivered elbow and wrist sit from the ones we measured, performer 0",
             plain="Our own measured elbow and wrist are the reference, at zero; lower is "
                   "better. Blue is what this step delivers. The hatched bars are the build "
                   "before it, which aimed each arm bone by the line between two measured "
                   "points while the bone actually turns about a joint somewhere else -- so "
                   "the whole arm was carried off by that gap. The aqua bars are the floor: "
                   "the part of the miss that a bone of fixed length can never remove, "
                   "whatever it is aimed at. Blue sits on the floor, so there is nothing "
                   "left here for an aim to fix; the rest is an arm cut to this performer, "
                   "and that is a later step.",
             better="lower",
             bars=[dict(label="Elbow, ships (aim from the arm's own joint)", role="ours",
                        key="arm_left_elbow_after_subject_00"),
                   dict(label="Elbow, before D9", role="control",
                        key="arm_left_elbow_d8_subject_00"),
                   dict(label="Elbow, floor", role="alt",
                        key="arm_left_elbow_floor_subject_00"),
                   dict(label="Wrist, ships", role="ours",
                        key="arm_left_wrist_after_subject_00"),
                   dict(label="Wrist, before D9", role="control",
                        key="arm_left_wrist_d8_subject_00"),
                   dict(label="Wrist, floor", role="alt",
                        key="arm_left_wrist_floor_subject_00")]),
        dict(title="D9: the same check on performer 1",
             plain="Same reference, same unit, lower is better. This performer's arms start "
                   "further out and come back the same way. The floor is higher here, and "
                   "the delivered arm sits on it: on the left elbow the floor is 8.8 mm and "
                   "the delivery reads 9.1, which is as close as a fixed-length bone can get.",
             better="lower",
             bars=[dict(label="Elbow, ships", role="ours",
                        key="arm_left_elbow_after_subject_01"),
                   dict(label="Elbow, before D9", role="control",
                        key="arm_left_elbow_d8_subject_01"),
                   dict(label="Elbow, floor", role="alt",
                        key="arm_left_elbow_floor_subject_01"),
                   dict(label="Wrist, ships", role="ours",
                        key="arm_left_wrist_after_subject_01"),
                   dict(label="Wrist, before D9", role="control",
                        key="arm_left_wrist_d8_subject_01"),
                   dict(label="Wrist, floor", role="alt",
                        key="arm_left_wrist_floor_subject_01")]),
        dict(title="D9: the right arm, both performers",
             plain="Same reference, lower is better. Both arms are changed the same way and "
                   "both land on their own floor, which is the point: this is one rule "
                   "applied twice, not a fit to one arm.",
             better="lower",
             bars=[dict(label="Elbow, ships, performer 0", role="ours",
                        key="arm_right_elbow_after_subject_00"),
                   dict(label="Elbow, before D9, performer 0", role="control",
                        key="arm_right_elbow_d8_subject_00"),
                   dict(label="Elbow, ships, performer 1", role="ours",
                        key="arm_right_elbow_after_subject_01"),
                   dict(label="Elbow, before D9, performer 1", role="control",
                        key="arm_right_elbow_d8_subject_01"),
                   dict(label="Wrist, ships, performer 1", role="ours",
                        key="arm_right_wrist_after_subject_01"),
                   dict(label="Wrist, before D9, performer 1", role="control",
                        key="arm_right_wrist_d8_subject_01")]),
        dict(title="D9: the shoulder, the neck and the knee, which this step must not touch",
             plain="Same reference, same unit, lower is better. This step aims two bones in "
                   "each arm and nothing else, so every pair of bars here has to be "
                   "identical -- and they are, float for float, read from the delivered "
                   "files. The shoulder is still 10-13 mm from the measured shoulder, and "
                   "that miss is not this step's: it comes from the collarbone having one "
                   "fixed length, and it is a later step's to close.",
             better="lower",
             bars=[dict(label="Shoulder, ships", role="ours",
                        key="arm_shoulder_after_subject_00"),
                   dict(label="Shoulder, before D9", role="control",
                        key="arm_shoulder_d8_subject_00"),
                   dict(label="Neck, ships", role="ours", key="arm_neck_after_subject_00"),
                   dict(label="Neck, before D9", role="control",
                        key="arm_neck_d8_subject_00"),
                   dict(label="Knee, ships", role="ours", key="arm_knee_after_subject_00"),
                   dict(label="Knee, before D9", role="control",
                        key="arm_knee_d8_subject_00")]),
    ],
    "masks": [
        dict(title="D9: how well the arms' outline covers the person's arms",
             plain="The reference fitter's person masks are the reference; higher is better. "
                   "Arms only, everything else taken out. Whole take and the stretch where "
                   "the two performers overlap. It rises on both performers in both cuts, "
                   "and on performer 1's overlap stretch the interval is clear of zero. The "
                   "hatched bars are the build before this one and the reference fitter's "
                   "own mesh -- the ceiling a body of the wrong proportions can reach, not "
                   "a target.",
             better="higher",
             bars=[dict(label="Whole take, ships", role="ours",
                        key="silhouette_arms_after_whole_take_subject_00"),
                   dict(label="Whole take, before D9", role="control",
                        key="silhouette_arms_d8_whole_take_subject_00"),
                   dict(label="Overlap stretch, ships", role="ours",
                        key="silhouette_arms_after_window_subject_00"),
                   dict(label="Overlap stretch, before D9", role="control",
                        key="silhouette_arms_d8_window_subject_00"),
                   dict(label="Overlap stretch, reference fitter's mesh", role="control",
                        key="silhouette_arms_oracle_window_subject_00")]),
        dict(title="D9: the same photographs on performer 1",
             plain="Same masks, same two cuts, higher is better. This is the performer whose "
                   "arms the previous step had the most trouble with, and it is where the "
                   "rise is largest and the only cell whose interval is clear of zero.",
             better="higher",
             bars=[dict(label="Whole take, ships", role="ours",
                        key="silhouette_arms_after_whole_take_subject_01"),
                   dict(label="Whole take, before D9", role="control",
                        key="silhouette_arms_d8_whole_take_subject_01"),
                   dict(label="Overlap stretch, ships", role="ours",
                        key="silhouette_arms_after_window_subject_01"),
                   dict(label="Overlap stretch, before D9", role="control",
                        key="silhouette_arms_d8_window_subject_01"),
                   dict(label="Overlap stretch, reference fitter's mesh", role="control",
                        key="silhouette_arms_oracle_window_subject_01")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_arm_origin({})
    for f in figs + ctrls:
        print(f"{f['key']:46s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
    keys = {f["key"] for f in figs + ctrls}
    missing = [b["key"] for group in VISUALS.values() for chart in group
               for b in chart["bars"] if b["key"] not in keys]
    print("VISUALS keys missing from the extractor:", missing or "NONE")
