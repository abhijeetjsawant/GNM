#!/usr/bin/env python3
"""Ladder extractor for D4 -- the body model in the delivery path (MHR replaces the MPFB body).

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the `x_*`-shaped
function and the proposed `VISUALS` entries. To register it, add to `ladder.py`:

    from extractors.d4_body_model import x_body_model

and route its figures by what each REFERENCES: the `bodymodel_sil_*` keys to rung 1 (the masks),
the `bodymodel_o1_*` and `bodymodel_place_*` keys to rung 7 (the converter). One call, two
destinations.

It reads exactly one report, `artifacts/compare/d4-body/gate.json`.

THREE REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * MAMMA's SAM2 person masks -- pixels of the footage, the one reference that is not
    model-mediated. B1's IoU lives here and nothing else does.
  * SYNTHETIC MHR TRUTH -- six bodies whose identity is a draw on MHR's own named scale
    channels, the pose a representable donor clamped to the model's own configured limits.
    Millimetres against a known answer. MAMMA-FREE.
  * OUR OWN CAPTURED LANDMARKS -- the triangulated array the delivery was fitted to. A
    placement figure against it cannot say whether the landmarks are right, only whether the
    delivered skeleton sits on them.

THE HEADLINE. The body stopped being a stock mesh stretched over a scaled rig. Silhouette IoU
0.647 / 0.652 -> 0.803 / 0.767, paired lower CI bounds +0.133 and +0.084, against MAMMA's own
mesh at 0.87 / 0.84 on the same rasteriser and the same masks.

WHAT BELONGS ON THE PAGE BESIDE THE BARS: **D4's ACCEPTANCE is FAIL.** O1, the exactness oracle,
reads 1.030 mm against a 1 mm band, and that blocks acceptance -- the "recorded exception" that
once stood here was an override and was withdrawn on 2026-09-22 (Astra's merge round, NO MERGE).
What merged is the `--body mhr` IMPLEMENTATION, opt-in behind an unchanged `rig` default, with
acceptance open and O1 re-registered prospectively as D4b. The band was set without measuring the
instrument's floor: with the TRUTH identity handed in and only the pose re-solved, the same
fixture reads 0.757 mm. And **most of the gain is the BODY MODEL, not the fit** -- MHR's own MEAN
body, driven by the same tracker, reads 0.766 / 0.771, and fitting the 68 scale channels adds
+0.037 on performer 0 and nothing distinguishable on performer 1.

Self-check:  python3 tools/compare/extractors/d4_body_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d4-body/gate.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python scripts/build_commercial_multiview_comparison.py "
    "--videos .cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos "
    "--calibration-yaml .cache/mamma/configs/examples/calib/iphones_outdoors.yaml "
    "--detector soma77 --body mhr --mhr-lod 2 --mhr-landmarks smoothed "
    "--output artifacts/compare/d4-body/delivery && "
    "/tmp/momenv/bin/python tools/fitter/d4_o1_exactness.py --drive "
    "--out artifacts/compare/d4-body/o1 --lod 2 && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4_body_gate.py "
    "--out artifacts/compare/d4-body/gate.json"
)

REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"
REF_SYNTH = ("SYNTHETIC MHR TRUTH: six bodies whose identity is a draw on MHR's own named "
             "scale_* channels within their configured limits, the donor pose clamped to the "
             "model's own limits. A known answer, MAMMA-FREE")
REF_CAPTURE = ("OUR OWN captured landmarks -- the triangulated array this delivery was fitted "
               "to, so a common-mode detector error is inside the reference")


def x_body_model(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    figs: list = []
    ctrls: list = []

    sil = report.get("B1_the_band", {}).get("measured", {})
    pooled = report.get("B1_pooled_medians_REPORTED", {})
    arms = report.get("B1_frozen_pose_control_below_the_candidate", {})
    for subject in ("00", "01"):
        figs.append(fig(f"D4 silhouette IoU, MHR delivered, performer {int(subject)}",
                        pooled.get("delivered_MHR_lod2", {}).get(f"subject_{subject}"),
                        "IoU", REF_MASKS, HIGHER,
                        key=f"bodymodel_sil_ours_subject_{subject}",
                        note="pooled over four cameras, whole take; the paired difference below "
                             "is the band"))
        ctrls.append(fig(f"D4 before: the D7c rig delivery, performer {int(subject)}",
                         pooled.get("baseline_D7c_rig", {}).get(f"subject_{subject}"),
                         "IoU", REF_MASKS, HIGHER,
                         key=f"bodymodel_sil_before_subject_{subject}"))
        ctrls.append(fig(f"D4 reported oracle: MAMMA's own mesh, performer {int(subject)}",
                         pooled.get("ORACLE_mamma_mesh", {}).get(f"subject_{subject}"),
                         "IoU", REF_MASKS, HIGHER,
                         key=f"bodymodel_sil_mamma_subject_{subject}",
                         note="the same rasteriser and the same masks, bit-identical to its "
                              "committed value on all 8 cells. Reported, never selected"))
        entry = sil.get(f"subject_{subject}", {})
        figs.append(fig(f"D4 paired IoU gain over the D7c rig, performer {int(subject)}",
                        entry.get("median_difference"), "IoU", REF_MASKS, HIGHER,
                        key=f"bodymodel_sil_gain_subject_{subject}",
                        note=f"the BAND: lower CI bound above zero. 95 % CI "
                             f"{entry.get('ci95')}, {entry.get('n')} frame-camera cells, "
                             f"moving-block bootstrap, block 15, identical draws"))
    frozen_pooled = min(
        (report.get("B1_silhouette_frozen_pooled", {}) or {}).values(), default=None)
    ctrls.append(fig("D4 control: the instrument's frozen-pose arm", frozen_pooled, "IoU",
                     REF_MASKS, HIGHER, key="bodymodel_sil_ctrl_frozen",
                     note=f"must stay below the candidate; it does in "
                          f"{arms.get('measured_cells_below')} of 8 cells (0.32-0.40 against "
                          f"0.65-0.82)"))
    ctrls.append(fig("D4 alternative: MHR's OWN mean body, same tracker",
                     pooled.get("control_MHR_mean_body_lod2", {}).get("subject_00"), "IoU",
                     REF_MASKS, HIGHER, key="bodymodel_sil_alt_mean_body",
                     note="performer 0; performer 1 reads "
                          f"{pooled.get('control_MHR_mean_body_lod2', {}).get('subject_01')}. "
                          "Most of the gain is the BODY MODEL, not the fit"))

    o1 = report.get("O1_exactness", {})
    figs.append(fig("D4 O1: fitted MHR against synthetic truth (worst of six seeds)",
                    o1.get("measured_max_over_seeds_mm"), "mm", REF_SYNTH, LOWER,
                    key="bodymodel_o1_ours",
                    note="band 1 mm -- FAILED at 1.030, and D4's ACCEPTANCE fails with it. The "
                         "band was set without measuring the instrument's floor; O1 is "
                         "re-registered prospectively as D4b"))
    ctrls.append(fig("D4 O1 floor: the TRUTH identity handed in, pose re-solved",
                     o1.get("exact_identity_floor_max_over_seeds_mm"), "mm", REF_SYNTH, LOWER,
                     key="bodymodel_o1_floor",
                     note="the best any fit could do through this solver on exact data. The "
                          "1 mm band leaves the identity fit 0.24 mm and it costs more"))
    ctrls.append(fig("D4 O1 must-fail: the MEAN body, pose re-solved on the same landmarks",
                     report.get("O1_must_fail_mean_body", {}).get("measured_min_over_seeds_mm"),
                     "mm", REF_SYNTH, LOWER, key="bodymodel_o1_ctrl_mean_body",
                     note="the arm that would otherwise pass B1 on its own; O1 rejects it by 9x "
                          "to 39x"))

    place = report.get("B3_placement_REPORTED", {})
    for subject in ("00", "01"):
        figs.append(fig(f"D4 delivered MHR joints against our landmarks, performer {int(subject)}",
                        place.get("all_landmark_median_mm", {}).get(f"subject_{subject}"),
                        "mm", REF_CAPTURE, LOWER, key=f"bodymodel_place_subject_{subject}",
                        note="17 mapped joints, median over the take. Reported, never banded: "
                             "the pinned offsets refuse to model the convention gap between "
                             "MHR's joints and SOMA-77's landmarks"))
    return figs, ctrls


VISUALS = {
    "masks": [
        dict(title="D4: does the delivered body cover the person in the photographs?",
             plain="The reference fitter's person masks are the reference; higher is better. "
                   "Before this step the delivered body was a stock mesh stretched over a "
                   "scaled skeleton. Now it is a real body model, fitted to this performer. "
                   "The orange bar is the reference fitter's own mesh through the identical "
                   "rasteriser -- it is still ahead, and the gap is what is left to win.",
             better="higher",
             bars=[dict(label="Before (the rig), performer 0", role="alt",
                        key="bodymodel_sil_before_subject_00"),
                   dict(label="After (MHR), performer 0", role="ours",
                        key="bodymodel_sil_ours_subject_00"),
                   dict(label="Reference fitter, performer 0", role="mamma",
                        key="bodymodel_sil_mamma_subject_00"),
                   dict(label="Before (the rig), performer 1", role="alt",
                        key="bodymodel_sil_before_subject_01"),
                   dict(label="After (MHR), performer 1", role="ours",
                        key="bodymodel_sil_ours_subject_01"),
                   dict(label="Reference fitter, performer 1", role="mamma",
                        key="bodymodel_sil_mamma_subject_01"),
                   dict(label="A frozen pose (deliberately wrong)", role="control",
                        key="bodymodel_sil_ctrl_frozen")]),
    ],
    "converter": [
        dict(title="D4: on a body we built on purpose, how close does the fit get?",
             plain="Six synthetic bodies with a known answer; lower is better. The blue bar is "
                   "our fit. The aqua bar beside it is the best this solver can do when it is "
                   "HANDED the right body and only has to find the pose -- that is the floor, "
                   "and our bar is close to it. The hatched bar is what happens if you skip "
                   "fitting the body at all: it is ten to forty times worse, which is the whole "
                   "point of the check.",
             better="lower",
             bars=[dict(label="Our fit (worst of six)", role="ours", key="bodymodel_o1_ours"),
                   dict(label="The floor: the right body handed in", role="alt",
                        key="bodymodel_o1_floor"),
                   dict(label="No body fit at all (deliberately wrong)", role="control",
                        key="bodymodel_o1_ctrl_mean_body")]),
        dict(title="D4: how close does the delivered skeleton sit to the points we tracked?",
             plain="Our own tracked landmarks are the reference; lower is better. This is "
                   "reported, not a pass mark: the fit is deliberately not allowed to slide its "
                   "markers, so wherever the body model's idea of a joint differs from the "
                   "tracker's, that difference shows up here as error.",
             better="lower",
             bars=[dict(label="Performer 0", role="ours", key="bodymodel_place_subject_00"),
                   dict(label="Performer 1", role="ours", key="bodymodel_place_subject_01")]),
    ],
}


if __name__ == "__main__":
    figures, controls = x_body_model({})
    for entry in figures + controls:
        print(f"{entry['key']:44s} {entry.get('value')!s:>12} {entry.get('unit', '')}")
    print(f"{len(figures)} figures, {len(controls)} controls")
