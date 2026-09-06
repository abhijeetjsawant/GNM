#!/usr/bin/env python3
"""Ladder extractor for D8c -- the captured HIP LINE that breaks the performer's own width.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d8c_hip import x_hip_line_reject

and route its figures by what each REFERENCES: the `capture_` keys to rung 4 (the
triangulation, scored against the performer's own captured body), the `synthetic_` keys to
rung 4 as well (the same stage against a known answer), the `placement_` keys to rung 7 (the
converter, from the delivered file's own bytes) and the `silhouette_` keys to rung 1 (the
masks). One extractor call, three destinations.

It reads exactly one report, `artifacts/compare/d8c-hip/gate.json`, plus the two stability
reports and the placement report the gate names.

FOUR REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * THE PERFORMER'S OWN CAPTURED BODY. The hip line against the median of that same
    performer's own take. A count of frames, not a millimetre. SELF-consistency, blind to
    truth by construction -- a segment welded to its own median scores zero -- which is why
    it is a headline and is in the merge rule only PAIRED with the photographs.
  * SYNTHETIC TRUTH, with an injected consistent hip collapse. Millimetres against a known
    answer, on the fixture only, and this is where D8b's mode and ceiling were CONFIRMED on
    the hips' own geometry.
  * OUR OWN DELIVERED FILE, read back from its own bytes -- how far the delivered root and
    legs moved between the two builds once each frame's hoist is subtracted.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE HEADLINE. On the falling performer the captured hip line is off his own 215 mm width by
more than 15 % on 30 raw and 23 smoothed frames, in three runs, and CLASSIFIED PER CAMERA
THEY ARE TWO DIFFERENT FAILURES. On 110-119 cameras A, C and D all support both hips, agree
with the collapsed point to 8.1 px at ray angles of 157-161 degrees, and the width reads
125-174 mm: a well-conditioned triangulation of a point the detector places inward in every
view. On 158-168 only A and C see him, at 140 degrees -- under D8's 150-degree ceiling, so
the conditioning gate is correctly silent -- with the hip line 16-28 degrees off both
viewing rays and the whole error living on the pair's own baseline. One row in the
segment-length table fires on both, because `|L - median| / median` is symmetric.

WHAT THE CHARTS CANNOT SAY. The frames-off count is self-referential and the candidate
optimises it directly. The photographs SEE the first class (torso IoU +0.021 and +0.013 on
its two runs, both intervals clear of zero) and CANNOT SEE the second (-0.0008, spanning
zero) -- a body lying down, seen by two cameras whose baseline is the very axis the error
lives on, moves almost no pixels when its depth is corrected. And MAMMA cannot referee it at
all: its hip line is a CONSTANT 117.6 mm on all 150 frames, a rigid SMPL-X pelvis roughly
half our captured width apart, so the collapsed build was accidentally NEARER its convention
and repairing the collapse scores worse against it. See
`docs/reviews/hip-line-2026-09-06.md`.

Self-check:  python3 tools/compare/extractors/d8c_hip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d8c-hip/gate.json"
BEFORE = "artifacts/compare/d8c-hip/limb-stability-d8b.json"
AFTER = "artifacts/compare/d8c-hip/limb-stability-after.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_delivery.py "
    "--out artifacts/compare/d8c-hip/delivery-hygiene --expect-byte-identical && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/captured_limb_stability.py "
    "--reproduce d8c --out artifacts/compare/d8c-hip/limb-stability-d8b.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_synthetic.py --calibrate "
    "--out artifacts/compare/d8c-hip/synthetic-noise-calibration.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_synthetic.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_delivery.py "
    "--out artifacts/compare/d8c-hip/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/captured_limb_stability.py "
    "--landmarks-from artifacts/compare/d8c-hip/delivery --skip-reproduction "
    "--hip-geometry --out artifacts/compare/d8c-hip/limb-stability-after.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_silhouette.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_placement.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_fixed_denominator.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8c_hip_gate.py"
)

REF_SELF = ("THE PERFORMER'S OWN CAPTURED BODY: the hip line against the median width of "
            "that same performer over that same take. A count of frames, not a distance, "
            "and blind to truth by construction -- a segment welded to its own median "
            "scores perfectly")
REF_SELF_FIXED = ("THE PERFORMER'S OWN CAPTURED BODY, on a FIXED denominator: the D8c build "
                  "scored against the D8b build's own median, so the reference cannot slide "
                  "under the count when the rule withholds the frames furthest from it")
REF_SYNTH = ("SYNTHETIC TRUTH with an injected consistent hip collapse: SOMASKEL77 clips "
             "posed through our own FK and projected into this rig, the real per-camera "
             "seen pattern replayed, our own detector's heavy-tail noise calibrated against "
             "this take's own honest hip line, and both hips moved toward their own "
             "midpoint in EVERY view for ten frames. The exact answer is known")
REF_BYTES = ("OUR OWN DELIVERED FILE, read back from its own bytes: the D8b and D8c GLBs "
             "parsed and forward-kinematicked, with each frame's own foot-contact hoist "
             "subtracted so a re-hoist is not read as a body-wide move")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"


def _subject_label(key: str) -> str:
    return key.replace("subject_", "performer ")


def _hip(doc: dict, subject: str) -> dict:
    return ((doc.get("subjects", {}).get(subject, {}).get("segments", {})
             .get("smoothed", {}).get("hip_line", {})
             .get("left_hip__right_hip")) or {})


def _leg(doc: dict, subject: str, pair: str) -> dict:
    return ((doc.get("subjects", {}).get(subject, {}).get("segments", {})
             .get("smoothed", {}).get("legs", {}).get(pair)) or {})


def x_hip_line_reject(_: dict) -> tuple[list, list]:
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
                (before, "control", "d8b",
                 "the shipped D8b build -- the defect this step is about"),
                (after, "ours", "d8c", "after the hip-line row")):
            block = _hip(doc, subject)
            entry = fig(
                f"D8c captured hip line, frames off the performer's own width by more than "
                f"15 %, {label}, {key.upper()}",
                block.get("frames_off_median_by_more_than_15pct"), "frames of 150",
                REF_SELF, LOWER, key=f"capture_hipline_off_{key}_{subject}", note=note)
            (figs if role == "ours" else ctrls).append(entry)
        # The FIXED-denominator reading, which is the one in the merge rule.
        cell = ((r.get("B4_the_frames_off_on_a_fixed_denominator") or {})
                .get("measured", {}).get(f"{subject}/hip_line") or {})
        figs.append(fig(
            f"D8c captured hip line on a FIXED denominator, {label}, D8c",
            cell.get("after_fixed_denominator"), "frames of 150", REF_SELF_FIXED, LOWER,
            key=f"capture_hipline_off_fixed_d8c_{subject}",
            note="scored against the D8b build's own median, so a count that fell because "
                 "the reference slid would still show here. It does not: the median moved "
                 "0.32 mm"))
        ctrls.append(fig(
            f"D8c control: the same count on the build before, {label}, D8b",
            cell.get("before_own_median"), "frames of 150", REF_SELF_FIXED, LOWER,
            key=f"capture_hipline_off_fixed_d8b_{subject}"))
        # The legs are the control that says the instrument is not counting motion.
        thigh = _leg(before, subject, "left_hip__left_knee")
        ctrls.append(fig(
            f"D8c control: the captured thigh, same measure, {label}",
            thigh.get("frames_off_median_by_more_than_15pct"), "frames of 150",
            REF_SELF, LOWER, key=f"capture_thigh_off_d8b_{subject}",
            note="the legs hold their length through the same frames on both performers, "
                 "0 frames off, before and after. The defect is the hip line, not motion"))

    # --------------------------------------------------------------- synthetic truth
    selector = (r.get("S1_the_selector") or {})
    modes = selector.get("per_mode_hips_median_mm") or {}
    figs.append(fig("D8c synthetic: 3D error of the collapsed hips, demote",
                    modes.get("demote"), "mm median", REF_SYNTH, LOWER,
                    key="synthetic_hip_error_demote",
                    note="what the selector confirmed: the point withheld, the rays kept "
                         "for the sequence solve. D8b's mode, not re-selected here"))
    for name, key, note in (
            ("reject", "reject", "the point AND the rays withheld"),
            ("best_ray", "best_ray", "only the highest-confidence camera's ray kept")):
        ctrls.append(fig(f"D8c synthetic: 3D error of the collapsed hips, {name}",
                         modes.get(name), "mm median", REF_SYNTH, LOWER,
                         key=f"synthetic_hip_error_{key}", note=note))
    ctrls.append(fig("D8c synthetic: the code before this step",
                     selector.get("today_hips_median_mm"), "mm median", REF_SYNTH, LOWER,
                     key="synthetic_hip_error_today",
                     note="D8b's nine rows on the same collapsed clip"))
    frozen = ((r.get("S3_the_two_must_fails") or {}).get("whole_take_hold") or {})
    ctrls.append(fig("D8c synthetic control: the hips frozen at their first frame",
                     frozen.get("frozen_hips_median_mm"), "mm median", REF_SYNTH, LOWER,
                     key="synthetic_hip_error_frozen",
                     note="a WEAK control on this fixture and shown as one: over the "
                          "injected run the hips travel 13 mm, so a frozen hip can only "
                          "ever be about 13 mm wrong. It fails as required and nothing "
                          "rests on it"))

    # ------------------------------------------------------- the delivered file's bytes
    placement = ((r.get("R1_R4_the_delivered_root_pelvis_frame_and_hoist") or {})
                 .get("subjects") or {})
    for subject, block in placement.items():
        label = _subject_label(subject)
        row = ((block.get("R1_root_and_hips_hoist_removed") or {})
               .get("root_translation", {}).get("hoist_removed", {})
               .get("on_the_fired_frames") or {})
        figs.append(fig(
            f"D8c: how far the delivered root moved on the fired frames, {label}",
            row.get("median_mm"), "mm median", REF_BYTES, LOWER,
            key=f"placement_root_move_{subject}",
            note="NOT an error: there is no truth here. It is how much of the body the "
                 "repair actually moved, and the card predicted under 20 mm"))

    # ------------------------------------------------------------------ the photographs
    silhouette = (r.get("B1_the_photographs") or {}).get("torso_iou_by_cut") or {}
    for subject, cuts in silhouette.items():
        label = _subject_label(subject)
        for cut, short in (("run_109_119", "the collapse run"),
                           ("run_158_168", "the stretch run")):
            cell = cuts.get(cut) or {}
            figs.append(fig(
                f"D8c torso and legs in the photographs, {short}, {label}, D8c",
                cell.get("D8c"), "IoU median", REF_MASKS, HIGHER,
                key=f"silhouette_torso_{cut}_d8c_{subject}",
                note="torso, legs and head; the part the hip line moves"))
            ctrls.append(fig(
                f"D8c control: the same photographs before, {short}, {label}, D8b",
                cell.get("D8b"), "IoU median", REF_MASKS, HIGHER,
                key=f"silhouette_torso_{cut}_d8b_{subject}"))
            ctrls.append(fig(
                f"D8c: the reference fitter's own mesh, {short}, {label}",
                cell.get("ORACLE_mamma_mesh"), "IoU median", REF_MASKS, HIGHER,
                key=f"silhouette_torso_{cut}_oracle_{subject}",
                note="the ceiling a body of the wrong proportions can reach, not a target"))
    return figs, ctrls


VISUALS = {
    "capture": [
        dict(title="D8c: how often the captured hips are the wrong width apart",
             plain="Each performer's own hip width, measured over their own take, is the "
                   "reference; the bar counts the frames where the captured hips came out "
                   "more than 15 % away from it. Lower is better. On the falling performer "
                   "the hips collapse to two thirds of his own width for a dozen frames "
                   "while three of the four cameras SEE him and all three agree with the "
                   "wrong point to a few pixels, and stretch outward for another eleven "
                   "while only two cameras can see him at all. The hatched bars are the "
                   "thigh, measured the same way through the same frames, which does not "
                   "move: the fault is the hips, not the motion.",
             better="lower",
             bars=[dict(label="After the hip row, performer 1", role="ours", key="capture_hipline_off_d8c_subject_01"),
                   dict(label="Before, performer 1", role="control", key="capture_hipline_off_d8b_subject_01"),
                   dict(label="After the hip row, performer 0", role="ours", key="capture_hipline_off_d8c_subject_00"),
                   dict(label="Before, performer 0", role="control", key="capture_hipline_off_d8b_subject_00"),
                   dict(label="The thigh, before, performer 1", role="control", key="capture_thigh_off_d8b_subject_01")]),
        dict(title="D8c: the same count, measured against the OLD build's reference",
             plain="The rule withholds the frames furthest from the performer's own median, "
                   "which moves that median -- so a count that fell simply because the "
                   "reference slid toward the bad frames would look like a fix. This chart "
                   "removes that possibility by scoring the new build against the OLD "
                   "build's median. Lower is better. It reads the same as the chart above, "
                   "and the median itself moved by a third of a millimetre.",
             better="lower",
             bars=[dict(label="After, fixed reference, performer 1", role="ours", key="capture_hipline_off_fixed_d8c_subject_01"),
                   dict(label="Before, performer 1", role="control", key="capture_hipline_off_fixed_d8b_subject_01"),
                   dict(label="After, fixed reference, performer 0", role="ours", key="capture_hipline_off_fixed_d8c_subject_00"),
                   dict(label="Before, performer 0", role="control", key="capture_hipline_off_fixed_d8b_subject_00")]),
    ],
    "synthetic": [
        dict(title="D8c: measured against a known answer, where truth exists",
             plain="Everything else here is measured against our own capture or against "
                   "photographs. This one is measured against an exactly known answer: a "
                   "synthetic performer, filmed by the same four cameras, with the real "
                   "pattern of camera dropouts replayed and both hips deliberately pulled "
                   "toward each other in every view for ten frames -- the failure this step "
                   "is about, built on purpose. Lower is better, and the bars are the "
                   "collapsed hips only. The three ways of handling a withheld point were "
                   "compared here and the first one won by a wide margin. The hatched bars "
                   "are the code before this step and a deliberately wrong answer, hips "
                   "frozen in place; on this clip the hips barely move, so the frozen "
                   "control is weak and is shown as such rather than counted as a pass.",
             better="lower",
             bars=[dict(label="Withhold the point, keep the rays", role="ours", key="synthetic_hip_error_demote"),
                   dict(label="Before this step", role="control", key="synthetic_hip_error_today"),
                   dict(label="Withhold the point and the rays", role="alt", key="synthetic_hip_error_reject"),
                   dict(label="Keep only the best camera's ray", role="alt", key="synthetic_hip_error_best_ray"),
                   dict(label="Weak control: hips frozen", role="control", key="synthetic_hip_error_frozen")]),
    ],
    "placement": [
        dict(title="D8c: how far the delivered character actually moved",
             plain="Read from the delivered files themselves, with the automatic "
                   "floor-hoist taken out so a re-levelling is not counted as the body "
                   "moving. This is NOT an error -- there is no correct answer to compare "
                   "against on this footage. It is the size of the change: on the frames "
                   "the rule acts on, the character's root moves about three millimetres on "
                   "the falling performer and not at all on the other, whose hips were "
                   "never withheld.",
             better="lower",
             bars=[dict(label="Performer 1, fired frames", role="ours", key="placement_root_move_subject_01"),
                   dict(label="Performer 0, fired frames", role="ours", key="placement_root_move_subject_00")]),
    ],
    "masks": [
        dict(title="D8c: how well the body and legs cover the person in the photographs",
             plain="The reference fitter's person masks are the reference; higher is "
                   "better. Torso, legs and head, on the two runs where the hips are wrong. "
                   "On the collapse run -- three cameras agreeing on hips that are too "
                   "close together -- the mesh covers measurably more of the person after "
                   "the repair, and the margin of uncertainty is clear of zero. On the "
                   "stretch run it does not move at all, and that is expected rather than "
                   "disappointing: the performer is lying down, only two cameras can see "
                   "him, and the error is along the one direction those two cameras cannot "
                   "resolve -- correcting it barely changes any pixel. The hatched bars are "
                   "the build before this one and the reference fitter's own mesh, which is "
                   "the ceiling a body of the wrong proportions can reach rather than a "
                   "target.",
             better="higher",
             bars=[dict(label="After, collapse run, performer 1", role="ours", key="silhouette_torso_run_109_119_d8c_subject_01"),
                   dict(label="Before, collapse run, performer 1", role="control", key="silhouette_torso_run_109_119_d8b_subject_01"),
                   dict(label="Reference fitter's mesh, collapse run", role="control", key="silhouette_torso_run_109_119_oracle_subject_01"),
                   dict(label="After, stretch run, performer 1", role="ours", key="silhouette_torso_run_158_168_d8c_subject_01"),
                   dict(label="Before, stretch run, performer 1", role="control", key="silhouette_torso_run_158_168_d8b_subject_01"),
                   dict(label="Reference fitter's mesh, stretch run", role="control", key="silhouette_torso_run_158_168_oracle_subject_01")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_hip_line_reject({})
    for f in figs + ctrls:
        print(f"{f['key']:56s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
    keys = {f["key"] for f in figs + ctrls}
    missing = [b["key"] for group in VISUALS.values() for chart in group
               for b in chart["bars"] if b["key"] not in keys]
    print("VISUALS keys missing from the extractor:", missing or "NONE")
