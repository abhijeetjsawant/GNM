#!/usr/bin/env python3
"""Ladder extractor for D4i -- the default flip to MHR and the instrument-roster migration.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner). This file supplies two
`x_*`-shaped functions and the proposed `VISUALS` entries. To register them, add to `ladder.py`:

    from extractors.d4i_flip import x_body_model_flip, x_body_model_flip_b4

and route `x_body_model_flip`'s `bodymodel_flip_sil_*` keys to rung 1 (the masks) and its `bodymodel_flip_place_*`
keys to the delivered rung; `x_body_model_flip_b4` is B4's OWN extractor (MAMMA's joints are their own reference and
never share an axis with the other two).

It reads the D4i gate (`artifacts/compare/d4i-flip/gate.json`, `tools/compare/d4i_flip_gate.py`) for the verdict
line, and the COORDINATOR'S RECORDED FINAL ROSTER run (`artifacts/compare/d4i-flip/final-roster/`, the entries of
`tools/compare/d4i_mhr_roster.py`) for the figures -- the post-migration measurement on the default build.

B1's delivered arm is named `D4d_fitted_MHR_lod2` in D4i (the coordinator's ruling: it reproduces D4d's B1 report
exactly). The rename to `delivered_MHR` belongs to D4i-b; this stub must change with it.

THREE REFERENCES, NEVER ONE AXIS:
  * MAMMA's SAM2 person masks (B1): pixels of the footage, not model-mediated;
  * OUR OWN CAPTURED LANDMARKS (B3): the array the delivery was fitted to -- a placement figure against it cannot say
    whether the landmarks are right, only whether the delivered skeleton sits on them;
  * MAMMA's `pred_joints` (B4): a research fitter's joints, CONVENTIONS, never truth -- a smaller number says the
    MHR joint sits nearer to where SMPL-X puts the joint of that name, not that it is better placed.

WHAT BELONGS ON THE PAGE BESIDE THE BARS, FIRST: **D4i: FAIL, on the oracle's JSON leg, read literally.** The
flipped default rebuilds D4d's MHR delivery byte for byte in every per-subject file except the two track JSONs,
which differ in one leaf, `body_model.assets`: the frozen fitter writes its own checkout's absolute path. Predicted
before the build; the coordinator's pre-registration error (the lane's sixth). Records only: the default is NOT
flipped on main. D4i-b re-registers the flip with that leaf named.

Self-check:  python3 tools/compare/extractors/d4i_flip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

GATE = "artifacts/compare/d4i-flip/gate.json"
ROSTER = "artifacts/compare/d4i-flip/final-roster"
POSITIVE = "artifacts/compare/d4i-flip/mustfail/v-positive"
ARM = "D4d_fitted_MHR_lod2"              # D4i's name for the delivered arm (D4i-b renames it)
REF_MASKS = ("MAMMA's SAM2 person masks (pixels of the footage, not model-mediated): median IoU over 150 frames x 4 "
             "cameras per performer, 960x540")
REF_CAPTURE = ("OUR OWN CAPTURED LANDMARKS (the triangulated array the delivery was fitted to): median distance of each "
               "declared landmark's MHR joint, median over the 17 landmarks")
REF_MAMMA = ("MAMMA's pred_joints (a research fitter's joint CONVENTIONS, never truth): 15 SMPL-X pairs through the "
             "declared map, the hip midpoint removed from both sides per frame")


def _verdict() -> str:
    gate = _load(GATE)
    return gate.get("line", "gate not run") if gate else "gate not run"


def x_body_model_flip(_: dict) -> tuple[list, list]:
    b1 = _load(f"{ROSTER}/b1.instrument.json")
    b3 = _load(f"{ROSTER}/b3.instrument.json")
    if not b1 or not b3:
        return [], []
    note = f"D4i {_verdict()} -- the default is NOT flipped on main (records only)"
    iou = lambda arm, s: b1.get("arms", {}).get(arm, {}).get(f"subject_{s:02d}", {}).get("pooled_median_iou")
    place = lambda report, s: (report or {}).get("subjects", {}).get(f"subject_{s:02d}", {}).get("all_landmarks_median_mm")
    shifted = _load(f"{POSITIVE}/b3.instrument.json")
    shifted_b1 = _load(f"{POSITIVE}/b1.instrument.json")
    figs = [fig(f"D4i B1: the default (MHR) build's mesh, performer {s}", iou(ARM, s), "IoU", REF_MASKS, HIGHER,
                key=f"bodymodel_flip_sil_p{s}", note=note) for s in (0, 1)]
    figs += [fig(f"D4i B3: the default build's skeleton on its own landmarks, performer {s}", place(b3, s), "mm",
                 REF_CAPTURE, LOWER, key=f"bodymodel_flip_place_p{s}", note="REPORTED, never banded") for s in (0, 1)]
    ctrls = [fig(f"D4i B1 alternative: the D7c rig mesh (the old default), performer {s}", iou("baseline_D7c_rig", s),
                 "IoU", REF_MASKS, HIGHER, key=f"bodymodel_flip_sil_alt_rig_p{s}",
                 note="the read-only i6 baseline, bound by sha256") for s in (0, 1)]
    ctrls += [fig(f"D4i positive control: the delivered GLB root moved 10 cm, performer {s}",
                  (shifted_b1 or {}).get("arms", {}).get(ARM, {}).get(f"subject_{s:02d}", {}).get("pooled_median_iou"),
                  "IoU", REF_MASKS, HIGHER, key=f"bodymodel_flip_sil_ctrl_shift_p{s}",
                  note="must-fail (v): a fresh export of the shifted GLBs; B1 FAILs") for s in (0, 1)]
    ctrls += [fig(f"D4i positive control: the delivered GLB root moved 10 cm, B3 performer {s}", place(shifted, s),
                  "mm", REF_CAPTURE, LOWER, key=f"bodymodel_flip_place_ctrl_shift_p{s}",
                  note="must-fail (v): the reading must move") for s in (0, 1)]
    return figs, ctrls


def x_body_model_flip_b4(_: dict) -> tuple[list, list]:
    """B4's own extractor: the default build's MHR joints against MAMMA's, root-relative. REPORTED, never selected."""
    b4 = _load(f"{ROSTER}/b4.instrument.json")
    if not b4:
        return [], []
    row = lambda s: b4.get("subjects", {}).get(f"subject_{s:02d}", {})
    figs = [fig(f"D4i B4: the default build's joints against MAMMA's, root-relative, performer {s}",
                row(s).get("root_relative_all_joint_median_mm"), "mm", REF_MAMMA, LOWER,
                key=f"bodymodel_flip_b4_rootrel_p{s}",
                note=(f"REPORTED, never selected. Bias {row(s).get('root_relative_all_joint_bias_median_mm')} mm (a "
                      f"convention gap), spread {row(s).get('root_relative_all_joint_spread_median_mm')} mm about it"))
            for s in (0, 1)]
    ctrls = [fig(f"D4i B4: the same, spread about the constant offset only, performer {s}",
                 row(s).get("root_relative_all_joint_spread_median_mm"), "mm", REF_MAMMA, LOWER,
                 key=f"bodymodel_flip_b4_spread_p{s}",
                 note="the part that moves frame to frame; the constant part is a convention difference")
             for s in (0, 1)]
    return figs, ctrls


VISUALS = {
    "masks": [
        dict(title="D4i: does the new default body (MHR) fill the performers' outlines better than the old rig?",
             plain="Yes, as in D4d: the MHR body covers the people in the footage far better than the old rig body. "
                   "But D4i itself FAILED a byte-for-byte check on a file path written inside two files, so the default "
                   "was not switched. Each bar is how much of the person's outline the body covers (IoU) -- higher is "
                   "better.",
             better="higher",
             bars=[dict(label="MHR default, performer 0", role="ours", key="bodymodel_flip_sil_p0"),
                   dict(label="MHR default, performer 1", role="ours", key="bodymodel_flip_sil_p1"),
                   dict(label="Old rig body, performer 0", role="alt", key="bodymodel_flip_sil_alt_rig_p0"),
                   dict(label="Old rig body, performer 1", role="alt", key="bodymodel_flip_sil_alt_rig_p1"),
                   dict(label="Body moved 10 cm on purpose, performer 0 (deliberately wrong)", role="control",
                        key="bodymodel_flip_sil_ctrl_shift_p0"),
                   dict(label="Body moved 10 cm on purpose, performer 1 (deliberately wrong)", role="control",
                        key="bodymodel_flip_sil_ctrl_shift_p1")]),
    ],
    "delivered": [
        dict(title="D4i: does the delivered skeleton sit on the landmarks it was fitted to?",
             plain="About 2 cm off on the median landmark, which is mostly where MHR puts a joint versus where the "
                   "detector puts it. Moving the body 10 cm on purpose shows the measure responds. Each bar is the "
                   "distance in millimetres -- lower is better, but it can only say the skeleton is on the landmarks, "
                   "not that the landmarks are right.",
             better="lower",
             bars=[dict(label="MHR default, performer 0", role="ours", key="bodymodel_flip_place_p0"),
                   dict(label="MHR default, performer 1", role="ours", key="bodymodel_flip_place_p1"),
                   dict(label="Moved 10 cm on purpose, performer 0 (deliberately wrong)", role="control",
                        key="bodymodel_flip_place_ctrl_shift_p0"),
                   dict(label="Moved 10 cm on purpose, performer 1 (deliberately wrong)", role="control",
                        key="bodymodel_flip_place_ctrl_shift_p1")]),
        dict(title="D4i B4: how far are our joints from MAMMA's joints of the same name?",
             plain="MAMMA is another research system, not the truth, and its joints follow a different convention, so "
                   "a constant offset is expected and says nothing about quality. The bars show the whole distance "
                   "and the part that changes frame to frame -- lower means closer to MAMMA, not necessarily better.",
             better="lower",
             bars=[dict(label="Ours vs MAMMA, performer 0", role="ours", key="bodymodel_flip_b4_rootrel_p0"),
                   dict(label="Ours vs MAMMA, performer 1", role="ours", key="bodymodel_flip_b4_rootrel_p1"),
                   dict(label="Frame-to-frame part only, performer 0", role="alt", key="bodymodel_flip_b4_spread_p0"),
                   dict(label="Frame-to-frame part only, performer 1", role="alt", key="bodymodel_flip_b4_spread_p1")]),
    ],
}


if __name__ == "__main__":
    figures, controls = x_body_model_flip({})
    b4_figures, b4_controls = x_body_model_flip_b4({})
    items = figures + controls + b4_figures + b4_controls
    for item in items:
        print(f"{item['key']:42s} {item['value']!s:>10} {item['unit']:4s} {item['note'][:80]}")
    keys = {item["key"] for item in items}
    missing = [bar["key"] for charts in VISUALS.values() for chart in charts for bar in chart["bars"]
               if bar["key"] not in keys]
    print("VISUALS keys resolved:", not missing, missing)
