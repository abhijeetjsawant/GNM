#!/usr/bin/env python3
"""Ladder extractor for D7c -- the pelvis on the RIG's own rest, not SOMA-77's.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or the
registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the `x_*`-shaped
function and the proposed `VISUALS` entries. To register it, add to `ladder.py`:

    from extractors.d7c_pelvis_rest import x_pelvis_rest

and route its figures by what each REFERENCES: the `pelvisrest_*` keys to rung 7 (the
converter), the `silhouette_*` keys to rung 1 (the masks). One extractor call, two
destinations.

It reads exactly one report, `artifacts/compare/d7c-pelvis-rest/gate.json`.

THREE REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * EXACT RIG TRUTH -- the D3 gate's six exact-skeleton bodies, where forward kinematics wrote
    the pelvis rotation and the rig's own rest triangle is CONGRUENT to the observed one.
    Degrees and millimetres against a known answer. MAMMA-FREE.
  * SYNTHETIC TRUTH UNDER OUR OWN DETECTOR'S NOISE -- the same bodies through the real
    cameras and the real triangulator, at a pixel sigma CALIBRATED to the take's own
    guard-kept pelvis-lever spread. The selector's arm.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not model-mediated.

THE HEADLINE. The delivered pelvis was pitched 6.865 degrees about the hip line on EVERY frame
of EVERY seed -- a constant of SOMA-77's convention, not noise, and a constant no rigid fit can
see. Reading the pelvis off the rig's OWN rest removes it exactly (1.5-1.8e-7 m of residual),
and the legs do not move (0.054-0.077 mm).

WHAT NONE OF IT SETTLES, and this belongs on the page beside the bars: nothing here resolves
the pelvis CONVENTION. Every figure is measured in the rig's own frame against the rig's own
rest; that SOMA-77's `Spine1` lies on the rig's `Hips`->`Spine` axis is an assumption this step
does not test and cannot. It goes to lane H's marker session.

Self-check:  python3 tools/compare/extractors/d7c_pelvis_rest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d7c-pelvis-rest/gate.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_delivery.py "
    "--out artifacts/compare/d7c-pelvis-rest/delivery && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_gate.py --oracle "
    "--oracle-save artifacts/compare/d7c-pelvis-rest/oracle-d7c "
    "--oracle-baseline artifacts/compare/d7c-pelvis-rest/oracle-shipped "
    "--take artifacts/compare/d7c-pelvis-rest/delivery "
    "--take-baseline artifacts/commercial-multiview-soma77 "
    "--out artifacts/compare/d7c-pelvis-rest/instrument-d7c.json && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_silhouette.py"
)

REF_EXACT = ("EXACT RIG TRUTH: the D3 gate's six exact-skeleton bodies, where forward "
             "kinematics wrote the pelvis rotation and the rig's own rest triangle is "
             "congruent to the observed one. MAMMA-FREE")
REF_NOISY = ("the same exact truth under OUR OWN detector's heavy-tail frame-correlated noise, "
             "injected in pixels through the real triangulator at a sigma CALIBRATED to the "
             "take's own guard-kept pelvis-lever spread (8.7636 mm). Never MAMMA's residual")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"


def x_pelvis_rest(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    figs: list = []
    ctrls: list = []

    oracle = report.get("O1_the_oracle", {})
    figs.append(fig("D7c pelvis vs exact truth, the shipping mode (worst seed)",
                    oracle.get("candidate_tilt_max_deg"), "deg", REF_EXACT, LOWER,
                    key="pelvisrest_tilt_ours",
                    note="band 0.01 deg; the shipped fit read 6.865 on every frame"))
    ctrls.append(fig("D7c before: SOMA-77's rest template (the delivered fit)",
                     oracle.get("C_tilt_median_deg"), "deg", REF_EXACT, LOWER,
                     key="pelvisrest_tilt_before",
                     note="a CONSTANT of the convention, identical on every frame of every "
                          "seed -- which is exactly why no rigid fit could see it"))
    ctrls.append(fig("D7c must-fail: a pelvis frozen upright",
                     oracle.get("frozen_upright_tilt_median_deg"), "deg", REF_EXACT, LOWER,
                     key="pelvisrest_tilt_ctrl_frozen_upright",
                     note="D7's own control, run through the identical code path"))
    ctrls.append(fig("D7c known blindness: the WRONG-ORIGIN template's apparent tilt",
                     oracle.get("wrong_origin_tilt_median_deg"), "deg", REF_EXACT, LOWER,
                     key="pelvisrest_tilt_ctrl_wrong_origin",
                     note="it reads ZERO. An 80 mm origin error is absorbed entirely by the "
                          "symmetric rest, so a tilt band passes it; only the metre residual "
                          "below sees it. The bar next to it is the point"))
    figs.append(fig("D7c unnormalised three-point residual, the shipping mode",
                    oracle.get("candidate_residual_max_m"), "m", REF_EXACT, LOWER,
                    key="pelvisrest_residual_ours", note="band 1e-6 m"))
    ctrls.append(fig("D7c: the same residual for the WRONG-ORIGIN template",
                     oracle.get("wrong_origin_residual_max_m"), "m", REF_EXACT, LOWER,
                     key="pelvisrest_residual_ctrl_wrong_origin",
                     note="0.000 deg of tilt and 80-97 mm of residual: the clause that "
                          "discriminates it"))
    ctrls.append(fig("D7c: the same residual for SOMA-77's template",
                     oracle.get("C_residual_max_m"), "m", REF_EXACT, LOWER,
                     key="pelvisrest_residual_before"))
    figs.append(fig("D7c `Spine` origin miss on exact truth, the shipping mode",
                    oracle.get("candidate_spine_max_mm"), "mm", REF_EXACT, LOWER,
                    key="pelvisrest_spine_ours", note="band 0.01 mm"))
    ctrls.append(fig("D7c `Spine` origin miss before (SOMA-77's template)",
                     oracle.get("C_spine_max_mm"), "mm", REF_EXACT, LOWER,
                     key="pelvisrest_spine_before", note="197 mm x sin 6.87 deg = 23.6"))
    figs.append(fig("D7c legs, feet and toes moved (O2)",
                    oracle.get("leg_move_max_mm"), "mm", "the shipped build's own forward "
                    "kinematics on the same six bodies", LOWER, key="pelvisrest_legs_moved",
                    note="band 0.1 mm; bit-identity is NOT claimed -- a pelvis frame is "
                         "whole-take"))

    selector = report.get("S_the_selector", {})
    for label, key, arm in (("ships (fit the rig's own pelvis triangle)", "pelvisrest_s_ours",
                             "a_kabsch_guarded"),
                            ("alternative (follow the hip line exactly)", "pelvisrest_s_alt",
                             "b_hipline_guarded")):
        figs.append(fig(f"D7c selector: pelvis error on bent frames, {label}",
                        selector.get(f"{arm}_bent_i_deg"), "deg", REF_NOISY, LOWER, key=key))
    for label, key, arm in (("before D7c (SOMA-77's template)", "pelvisrest_s_before",
                             "C_on_SOMA"),
                            ("deliberately wrong: pitch frozen to gravity",
                             "pelvisrest_s_ctrl_follower", "frozen_pitch_follower"),
                            ("deliberately wrong: the whole pelvis upright",
                             "pelvisrest_s_ctrl_vertical", "world_vertical"),
                            ("before D7 (the trunk line)", "pelvisrest_s_ctrl_thorax",
                             "thorax_as_pelvis")):
        ctrls.append(fig(f"D7c selector control: {label}",
                         selector.get(f"{arm}_bent_i_deg"), "deg", REF_NOISY, LOWER, key=key))
    guard = report.get("S_the_selector", {}).get("G2", {})
    figs.append(fig("D7c the lever guard, on corrupted frames",
                    guard.get("guarded_i_deg"), "deg", REF_NOISY, LOWER,
                    key="pelvisrest_guard_on"))
    ctrls.append(fig("D7c the same frames with the guard OFF",
                     guard.get("unguarded_i_deg"), "deg", REF_NOISY, LOWER,
                     key="pelvisrest_guard_off",
                     note="the guard is tested HERE, not in the main table: on the clean "
                          "fixture it rejects almost nothing and cannot lose"))

    photos = report.get("B1_the_photographs", {})
    for subject in ("subject_00", "subject_01"):
        label = subject.replace("subject_0", "performer ")
        for part in ("torso", "arm"):
            figs.append(fig(f"D7c silhouette, {part}, bent tercile, after, {label}",
                            photos.get(f"{subject}_{part}_bent_after"), "IoU", REF_MASKS,
                            HIGHER, key=f"silhouette_{part}_after_bent_{subject}"))
            ctrls.append(fig(f"D7c silhouette, {part}, bent tercile, before (D9b), {label}",
                             photos.get(f"{subject}_{part}_bent_before"), "IoU", REF_MASKS,
                             HIGHER, key=f"silhouette_{part}_before_bent_{subject}",
                             note="improvement is NOT predicted: a mesh can rotate inside "
                                  "its own outline, and no photograph resolves a CONSTANT "
                                  "change of frame"))
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D7c: how far the delivered pelvis is from the true pelvis, on bodies where "
                   "the true answer is known exactly",
             plain="Exact truth is the reference, at zero; lower is better. Blue is what ships. "
                   "Orange is what the code did before -- and note it is the SAME number on "
                   "every frame of every body, because it is a constant of the convention the "
                   "old code carried, not a wobble. The hatched bars are deliberately wrong "
                   "answers. One of them reads ZERO here, which is the whole reason the next "
                   "chart exists.",
             better="lower",
             bars=[dict(label="Ships (the rig's own rest)", role="ours", key="pelvisrest_tilt_ours"),
                   dict(label="Before D7c (SOMA-77's rest)", role="alt", key="pelvisrest_tilt_before"),
                   dict(label="Deliberately wrong: pelvis frozen upright", role="control", key="pelvisrest_tilt_ctrl_frozen_upright"),
                   dict(label="Deliberately wrong: the template taken from the wrong point", role="control", key="pelvisrest_tilt_ctrl_wrong_origin")]),
        dict(title="D7c: the measurement that catches the wrong answer the angle cannot",
             plain="Same bodies, same exact truth, but this measures how far the three pelvis "
                   "points land from where they should be, in metres. Lower is better. The "
                   "hatched bar on the right is the answer that looked perfect on the chart "
                   "above: taking the pelvis shape from a point 80 millimetres too high tilts "
                   "nothing, because the shape is symmetric -- it just puts everything in the "
                   "wrong place. An angle cannot see that. This can.",
             better="lower",
             bars=[dict(label="Ships", role="ours", key="pelvisrest_residual_ours"),
                   dict(label="Before D7c", role="alt", key="pelvisrest_residual_before"),
                   dict(label="Deliberately wrong: taken from the wrong point", role="control", key="pelvisrest_residual_ctrl_wrong_origin")]),
        dict(title="D7c: choosing between the two ways of reading the pelvis, under our own "
                   "camera noise",
             plain="Synthetic truth with our detector's own measured noise is the reference; "
                   "lower is better, and these are the frames where the performer is most bent "
                   "over. Blue ships. Aqua is the alternative that was genuinely in contention. "
                   "The hatched bars are answers built to be wrong -- including one that simply "
                   "keeps the pelvis level with the ground, which is the one a placement check "
                   "cannot tell apart from the real thing.",
             better="lower",
             bars=[dict(label="Ships", role="ours", key="pelvisrest_s_ours"),
                   dict(label="Alternative (follow the hip line exactly)", role="alt", key="pelvisrest_s_alt"),
                   dict(label="Before D7c", role="alt", key="pelvisrest_s_before"),
                   dict(label="Deliberately wrong: pitch frozen to gravity", role="control", key="pelvisrest_s_ctrl_follower"),
                   dict(label="Deliberately wrong: the whole pelvis upright", role="control", key="pelvisrest_s_ctrl_vertical"),
                   dict(label="Before D7: the trunk line", role="control", key="pelvisrest_s_ctrl_thorax")]),
        dict(title="D7c: does throwing away the broken spine readings help?",
             plain="Same reference, lower is better, measured only on frames deliberately given "
                   "a bad spine reading. Blue throws them away and fills the gap; hatched keeps "
                   "them. The gap is about tenfold. On ordinary frames the two are almost "
                   "identical, which is why this test was built with broken frames in it.",
             better="lower",
             bars=[dict(label="Ships (broken readings discarded)", role="ours", key="pelvisrest_guard_on"),
                   dict(label="Keeping the broken readings", role="control", key="pelvisrest_guard_off")]),
        dict(title="D7c: did the legs move? They were not supposed to",
             plain="The same character built by the previous code is the reference, at zero; "
                   "lower is better. The pelvis is rotated by about nine degrees and the legs "
                   "stay within a fifteenth of a millimetre, because the hips themselves are "
                   "placed by the skeleton's own geometry and not by the pelvis's rotation.",
             better="lower",
             bars=[dict(label="Legs, feet and toes moved", role="ours", key="pelvisrest_legs_moved")]),
    ],
    "masks": [
        dict(title="D7c: does the body's outline still cover the person, where they are most "
                   "bent over?",
             plain="The reference fitter's person masks are the reference; higher is better. "
                   "This step was NOT expected to improve them, and the reason is worth saying: "
                   "it rotates the whole trunk about the hip line, and a body can rotate inside "
                   "its own outline without moving a pixel. The check is here to catch the "
                   "opposite -- a step that broke something.",
             better="higher",
             bars=[dict(label="Body and legs, before, performer 0", role="alt", key="silhouette_torso_before_bent_subject_00"),
                   dict(label="Body and legs, after, performer 0", role="ours", key="silhouette_torso_after_bent_subject_00"),
                   dict(label="Body and legs, before, performer 1", role="alt", key="silhouette_torso_before_bent_subject_01"),
                   dict(label="Body and legs, after, performer 1", role="ours", key="silhouette_torso_after_bent_subject_01"),
                   dict(label="Arms, before, performer 0", role="alt", key="silhouette_arm_before_bent_subject_00"),
                   dict(label="Arms, after, performer 0", role="ours", key="silhouette_arm_after_bent_subject_00")]),
    ],
}


if __name__ == "__main__":
    figures, controls = x_pelvis_rest({})
    for entry in figures + controls:
        print(f"{entry['key']:46s} {entry.get('value')!s:>14} {entry.get('unit', '')}")
    print(f"{len(figures)} figures, {len(controls)} controls")
