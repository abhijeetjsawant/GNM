#!/usr/bin/env python3
"""Ladder extractor for D7b -- the trunk chain re-solved onto the captured neck.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d7b_trunk import x_trunk_resolve

and route its figures by what each REFERENCES: the `trunk_` keys to rung 7 (the
converter, scored against our own captured landmarks), the `silhouette_` keys to rung 1
(the masks). One extractor call, two destinations.

It reads exactly one report, `artifacts/compare/d7b-trunk/gate.json`.

THREE REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * OUR OWN CAPTURED LANDMARKS, read against the delivered file's own bytes. Millimetres.
    This is the new instrument, `delivered_vs_capture.py`, and it is the figure that saw
    the D7 defect.
  * THE LENGTH FLOOR, which is not a rival answer but the part of that same millimetre
    figure no aim can remove. It shares the axis with the bars above BY CONSTRUCTION --
    same quantity, same reference, same frames -- which is why it may sit on the chart.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE HEADLINE. The delivered `Neck` returns to its landmark: 58.9 -> 21.5 mm on performer 0
and 44.0 -> 18.3 on performer 1, and 131.3 -> 42.1 on performer 0's bent tercile. Every one
of those lands within 0.7 mm of the trunk-LENGTH floor, so what is left is not aim but a
rigid straight spine, and that share is handed to D5. The hips, knees and ankles are
bit-identical to D7 from the file. The photographs agree: performer 0's bent-tercile torso
rises +0.029 IoU with the interval clear of zero, and precision and recall BOTH rise, so it
is not the dilated-blob pattern.

ONE BAND FAILS AS WRITTEN. B5 asked for the head's world orientation unchanged to 1e-9 per
frame; the delivered track stores float32 and this step re-derives every local in the chain
above `Hips`, so the recomposed head moves by 4.5-5.1e-6 degrees -- under one float32 ULP,
and no further from the head solve's own float64 rotations than D7 is. The band is not
moved. See `docs/reviews/trunk-resolve-2026-09-05.md`.

Self-check:  python3 tools/compare/extractors/d7b_trunk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d7b-trunk/gate.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7b_trunk_delivery.py "
    "--out artifacts/compare/d7b-trunk/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7b_silhouette_partwise.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7b_trunk_gate.py"
)

REF_CAPTURE = ("OUR OWN captured landmarks: the delivery's own "
               "`triangulated_world_positions_z_up_m`, in absolute capture Z-up metres, "
               "against the delivered GLB read from its own bytes. Never MAMMA, never a "
               "re-detection, never a re-solve")
REF_FLOOR = (REF_CAPTURE + " -- the part of that same figure NO aim can remove: "
             "| L_rest - ||neck - Spine_origin|| |, a rigid straight spine's own length error")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"
REF_ORACLE = ("the head SOLVE's own float64 world rotations, through the converter's change "
              "of basis -- the head BOTH builds were handed")


def _subject_label(key: str) -> str:
    return key.replace("subject_", "performer ")


def x_trunk_resolve(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []

    placement = r.get("B1_placement", {}).get("subjects", {})
    for subject, block in placement.items():
        label = _subject_label(subject)
        whole = block.get("neck_median_mm", {})
        bent = block.get("neck_bent_tercile_median_mm", {})
        floor = block.get("length_floor_mm") or [None, None]
        figs.append(fig(f"D7b delivered Neck to the captured neck landmark, {label}",
                        whole.get("D7b"), "mm median", REF_CAPTURE, LOWER,
                        key=f"trunk_neck_after_{subject}",
                        note="read from the delivered GLB's own bytes"))
        figs.append(fig(f"D7b delivered Neck, bent tercile, {label}",
                        bent.get("D7b"), "mm median", REF_CAPTURE, LOWER,
                        key=f"trunk_neck_after_bent_{subject}",
                        note="the third of frames with the most trunk tilt"))
        ctrls.append(fig(f"D7b control: the shipped D7 build's Neck, {label}",
                         whole.get("D7"), "mm median", REF_CAPTURE, LOWER,
                         key=f"trunk_neck_d7_{subject}",
                         note="the defect D7b closes -- the pelvis frame moved the trunk "
                              "chain's origin and nothing re-solved the chain"))
        ctrls.append(fig(f"D7b control: the shipped D7 build's Neck, bent tercile, {label}",
                         bent.get("D7"), "mm median", REF_CAPTURE, LOWER,
                         key=f"trunk_neck_d7_bent_{subject}"))
        figs.append(fig(f"D7b alternative: before the pelvis frame (D3), Neck, {label}",
                        whole.get("D3"), "mm median", REF_CAPTURE, LOWER,
                        key=f"trunk_neck_d3_{subject}",
                        note="D3's chain lay on the trunk line by construction, so it was "
                             "close whole-take and far on bent frames"))
        figs.append(fig(f"D7b alternative: D3's Neck, bent tercile, {label}",
                        bent.get("D3"), "mm median", REF_CAPTURE, LOWER,
                        key=f"trunk_neck_d3_bent_{subject}"))
        figs.append(fig(f"D7b: the trunk LENGTH floor, {label}", floor[0], "mm median",
                        REF_FLOOR, LOWER, key=f"trunk_floor_{subject}",
                        note="what a straight rigid trunk cannot beat, whatever it is "
                             "aimed at. Handed to D5"))
        figs.append(fig(f"D7b: the trunk LENGTH floor, bent tercile, {label}", floor[1],
                        "mm median", REF_FLOOR, LOWER, key=f"trunk_floor_bent_{subject}"))

    retained = r.get("B2_untouchable", {}).get("subjects", {})
    for subject, block in retained.items():
        label = _subject_label(subject)
        rows = block.get("from_the_delivered_file", {})
        for joint, key in (("LeftUpperLeg", "hip"), ("LeftLowerLeg", "knee"),
                           ("LeftFoot", "ankle")):
            row = rows.get(joint, {})
            figs.append(fig(f"D7b retained: the delivered {key} to its landmark, {label}",
                            row.get("D7b_median_mm"), "mm median", REF_CAPTURE, LOWER,
                            key=f"trunk_{key}_after_{subject}",
                            note="bit-identical to D7 by construction: `Hips`, the root "
                                 "formula and the whole leg chain are untouched"))
            ctrls.append(fig(f"D7b control: the same {key} on the D7 build, {label}",
                             row.get("D7_median_mm"), "mm median", REF_CAPTURE, LOWER,
                             key=f"trunk_{key}_d7_{subject}",
                             note="it must be IDENTICAL, and it is"))

    arms = r.get("B3_arms", {}).get("subjects", {})
    for subject, rows in arms.items():
        label = _subject_label(subject)
        row = rows.get("RightHand", {})
        figs.append(fig(f"D7b: the delivered wrist to its landmark, bent tercile, {label}",
                        row.get("bent_tercile_median_mm", {}).get("D7b"), "mm median",
                        REF_CAPTURE, LOWER, key=f"trunk_wrist_after_bent_{subject}",
                        note="the arm hangs off `UpperChest`; move the trunk chain and the "
                             "whole arm follows, then pass C RE-SOLVES the elbow and wrist"))
        ctrls.append(fig(f"D7b control: the same wrist on the D7 build, bent tercile, {label}",
                         row.get("bent_tercile_median_mm", {}).get("D7"), "mm median",
                         REF_CAPTURE, LOWER, key=f"trunk_wrist_d7_bent_{subject}"))

    silhouette = r.get("B4_silhouette", {}).get("partwise_and_tercile", {})
    for subject, block in silhouette.items():
        label = _subject_label(subject)
        for tercile in ("upright", "middle", "bent"):
            cell = block.get("terciles", {}).get(tercile, {})
            figs.append(fig(f"D7b silhouette, body and legs, {tercile}, {label}",
                            cell.get("torso_iou_D7b"), "IoU", REF_MASKS, HIGHER,
                            key=f"silhouette_torso_after_{tercile}_{subject}"))
            ctrls.append(fig(f"D7b control: the D7 build, body and legs, {tercile}, {label}",
                             cell.get("torso_iou_D7"), "IoU", REF_MASKS, HIGHER,
                             key=f"silhouette_torso_d7_{tercile}_{subject}"))
            ctrls.append(fig(f"D7b oracle: MAMMA's own mesh, body and legs, {tercile}, {label}",
                             cell.get("torso_iou_ORACLE"), "IoU", REF_MASKS, HIGHER,
                             key=f"silhouette_oracle_torso_{tercile}_{subject}",
                             note="the CEILING a mesh of the wrong proportions can reach, "
                                  "not a target"))
        arms_cell = block.get("arm_iou_all_frames", {})
        figs.append(fig(f"D7b silhouette, arms only, {label}", arms_cell.get("D7b"), "IoU",
                        REF_MASKS, HIGHER, key=f"silhouette_arms_after_{subject}"))
        ctrls.append(fig(f"D7b control: the D7 build, arms only, {label}",
                         arms_cell.get("D7"), "IoU", REF_MASKS, HIGHER,
                         key=f"silhouette_arms_d7_{subject}"))

    head = r.get("B5_nothing_else_moved", {}).get("head_solve_oracle", {}).get("figures", {})
    for subject, block in head.items():
        ctrls.append(fig(f"D7b oracle: the delivered Head against the head solve's own "
                         f"float64 rotations, {_subject_label(subject)}",
                         block.get("D7b", {}).get("worst_deg"), "deg worst frame",
                         REF_ORACLE, LOWER, key=f"trunk_head_oracle_{subject}",
                         note="the pre-registered band (1e-9) FAILED as written; this is "
                              "under one float32 ULP and no further from the oracle than "
                              "the D7 build is"))
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D7b: how far the delivered neck sits from the neck we measured, performer 0",
             plain="Our own measured neck is the reference, at zero; lower is better. Blue is "
                   "what this step delivers. The hatched bar is the build before it, which "
                   "moved the neck off its landmark. The aqua bars are the build before THAT "
                   "and the floor -- the part of the miss that a straight, rigid spine can "
                   "never remove whatever it is aimed at. Blue sits on the floor, so there is "
                   "nothing left here for an aim to fix; the rest is a bendable spine, and "
                   "that is a later step.",
             better="lower",
             bars=[dict(label="Ships (aim from the spine's own origin)", role="ours", key="trunk_neck_after_subject_00"),
                   dict(label="Before D7b: the pelvis frame alone", role="control", key="trunk_neck_d7_subject_00"),
                   dict(label="Before D7: one rigid trunk", role="alt", key="trunk_neck_d3_subject_00"),
                   dict(label="Floor: what a rigid straight spine cannot beat", role="alt", key="trunk_floor_subject_00")]),
        dict(title="D7b: the same check on the third of frames where performer 0 is most bent over",
             plain="Same reference, same unit, lower is better -- but only the frames where the "
                   "performer is most bent over, which is where a pelvis that disagrees with "
                   "the trunk does the most damage. The neck comes back from 131 mm to 42, and "
                   "42 is the floor.",
             better="lower",
             bars=[dict(label="Ships (aim from the spine's own origin)", role="ours", key="trunk_neck_after_bent_subject_00"),
                   dict(label="Before D7b: the pelvis frame alone", role="control", key="trunk_neck_d7_bent_subject_00"),
                   dict(label="Before D7: one rigid trunk", role="alt", key="trunk_neck_d3_bent_subject_00"),
                   dict(label="Floor: what a rigid straight spine cannot beat", role="alt", key="trunk_floor_bent_subject_00")]),
        dict(title="D7b: the same check on performer 1",
             plain="Same reference, lower is better. This performer is bent over for most of "
                   "the take, and the neck returns from 44 mm to 18 -- again exactly the floor.",
             better="lower",
             bars=[dict(label="Ships (aim from the spine's own origin)", role="ours", key="trunk_neck_after_subject_01"),
                   dict(label="Before D7b: the pelvis frame alone", role="control", key="trunk_neck_d7_subject_01"),
                   dict(label="Before D7: one rigid trunk", role="alt", key="trunk_neck_d3_subject_01"),
                   dict(label="Floor: what a rigid straight spine cannot beat", role="alt", key="trunk_floor_subject_01")]),
        dict(title="D7b: the hips, knees and ankles, which this step must not touch",
             plain="Same reference, same unit, lower is better. This step changes the chain "
                   "ABOVE the hips and nothing below them, so every pair of bars here has to "
                   "be identical -- and they are, float for float, read from the delivered "
                   "files. That is the check that stops a good neck being bought with a worse "
                   "lower body.",
             better="lower",
             bars=[dict(label="Hip, ships", role="ours", key="trunk_hip_after_subject_00"),
                   dict(label="Hip, before D7b", role="control", key="trunk_hip_d7_subject_00"),
                   dict(label="Knee, ships", role="ours", key="trunk_knee_after_subject_00"),
                   dict(label="Knee, before D7b", role="control", key="trunk_knee_d7_subject_00"),
                   dict(label="Ankle, ships", role="ours", key="trunk_ankle_after_subject_00"),
                   dict(label="Ankle, before D7b", role="control", key="trunk_ankle_d7_subject_00")]),
        dict(title="D7b: the wrist, on the frames where performer 0 is most bent over",
             plain="Same reference, lower is better. Nothing was aimed at the arms here -- but "
                   "the arm hangs off the chest, so moving the chest back onto the body carries "
                   "the whole arm with it and the elbow and wrist are then re-solved onto their "
                   "own landmarks. This is that, and it was predicted before the numbers.",
             better="lower",
             bars=[dict(label="Ships", role="ours", key="trunk_wrist_after_bent_subject_00"),
                   dict(label="Before D7b", role="control", key="trunk_wrist_d7_bent_subject_00"),
                   dict(label="Ships, performer 1", role="ours", key="trunk_wrist_after_bent_subject_01"),
                   dict(label="Before D7b, performer 1", role="control", key="trunk_wrist_d7_bent_subject_01")]),
    ],
    "masks": [
        dict(title="D7b: how well the body's outline covers the person, by how far they are bent over",
             plain="The reference fitter's person masks are the reference; higher is better. "
                   "Body and legs only, arms taken out, performer 0. It rises in every band and "
                   "most where they are most bent over, and there the interval is clear of zero. "
                   "The hatched bars are the build before this one and the reference fitter's "
                   "own mesh -- the ceiling a body of the wrong proportions can reach, not a "
                   "target.",
             better="higher",
             bars=[dict(label="Upright, ships", role="ours", key="silhouette_torso_after_upright_subject_00"),
                   dict(label="Upright, before D7b", role="control", key="silhouette_torso_d7_upright_subject_00"),
                   dict(label="Middle, ships", role="ours", key="silhouette_torso_after_middle_subject_00"),
                   dict(label="Middle, before D7b", role="control", key="silhouette_torso_d7_middle_subject_00"),
                   dict(label="Most bent, ships", role="ours", key="silhouette_torso_after_bent_subject_00"),
                   dict(label="Most bent, before D7b", role="control", key="silhouette_torso_d7_bent_subject_00"),
                   dict(label="Most bent, reference fitter's mesh", role="control", key="silhouette_oracle_torso_bent_subject_00")]),
        dict(title="D7b: the same photographs on performer 1",
             plain="Same masks, same three bands, higher is better. Nothing falls on this "
                   "performer either; the gain is smaller because this take's pelvis and trunk "
                   "disagree less.",
             better="higher",
             bars=[dict(label="Upright, ships", role="ours", key="silhouette_torso_after_upright_subject_01"),
                   dict(label="Upright, before D7b", role="control", key="silhouette_torso_d7_upright_subject_01"),
                   dict(label="Middle, ships", role="ours", key="silhouette_torso_after_middle_subject_01"),
                   dict(label="Middle, before D7b", role="control", key="silhouette_torso_d7_middle_subject_01"),
                   dict(label="Most bent, ships", role="ours", key="silhouette_torso_after_bent_subject_01"),
                   dict(label="Most bent, before D7b", role="control", key="silhouette_torso_d7_bent_subject_01"),
                   dict(label="Most bent, reference fitter's mesh", role="control", key="silhouette_oracle_torso_bent_subject_01")]),
        dict(title="D7b: the arms in the photographs",
             plain="Same masks, higher is better, arms only. The arms move here -- they hang "
                   "off the chest this step turns -- and the check is that they do not get "
                   "worse. On both performers the change sits inside the range the measurement "
                   "itself is uncertain by.",
             better="higher",
             bars=[dict(label="Ships, performer 0", role="ours", key="silhouette_arms_after_subject_00"),
                   dict(label="Before D7b, performer 0", role="control", key="silhouette_arms_d7_subject_00"),
                   dict(label="Ships, performer 1", role="ours", key="silhouette_arms_after_subject_01"),
                   dict(label="Before D7b, performer 1", role="control", key="silhouette_arms_d7_subject_01")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_trunk_resolve({})
    for f in figs + ctrls:
        print(f"{f['key']:46s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
    keys = {f["key"] for f in figs + ctrls}
    missing = [b["key"] for group in VISUALS.values() for chart in group
               for b in chart["bars"] if b["key"] not in keys]
    print("VISUALS keys missing from the extractor:", missing or "NONE")
