#!/usr/bin/env python3
"""Ladder extractor for D7 -- `Hips` gets its own frame from the pelvis's own landmarks.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d7_pelvis_frame import x_pelvis_frame

and route its figures by what each REFERENCES: the `pelvis_` keys to rung 7 (the
converter), the `silhouette_` keys to rung 1 (the masks), the `rigidity_` keys to rung 7 as
a reference-free invariant. One extractor call, two destinations.

It reads exactly one report, `artifacts/compare/d7-pelvis-frame/gate.json`.

FOUR REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * SYNTHETIC TRUTH -- the posed SOMASKEL77 `Hips` world rotation, recovered by Kabsch of
    the clip's own rest pelvis offsets onto the posed ones. Degrees against exact truth.
  * A REFERENCE-FREE INVARIANT -- segment-length stability over the real take. Millimetres
    of spread, scored beside the body controls this lane already trusts.
  * the CONVERTER'S OWN OUTPUT -- the exact-recovery oracle and the canonical round trip.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE HEADLINE, AND IT IS A FAILURE. The pelvis frame beats today's code (the trunk line) by
18-24 degrees at every noise level tested. It does NOT beat a WORLD-VERTICAL pelvis at our
detector's own measured noise on this fixture's motion, and the pre-registered band said it
must. The truth pelvis on all five synthetic clips sits 1.1-12.7 degrees from vertical, so a
constant is a good approximation of it there; that is a property of THIS motion source and
the band was written before anyone had measured it. Reported, not smoothed away.

Self-check:  python3 tools/compare/extractors/d7_pelvis_frame.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d7-pelvis-frame/gate.json"
REGEN = (
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_pelvis_rigidity.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_pelvis_synthetic.py && "
    "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_pelvis_frame_gate.py"
)

REF_SYNTH = ("SYNTHETIC TRUTH: the posed SOMASKEL77 Hips world rotation, by Kabsch of the clip's "
             "OWN rest pelvis offsets onto the posed ones (4.2e-8 m residual). MAMMA-FREE")
REF_NOISY = (REF_SYNTH + ", with our own detector's measured heavy-tail frame-correlated noise "
             "injected in pixels at 3.20 px and recovered through the real triangulator")
REF_INVARIANT = ("a REFERENCE-FREE INVARIANT: a segment between two points rigid to one bone has "
                 "constant length whatever the pose. Spread in millimetres, never accuracy")
REF_ORACLE = ("the converter scored against its OWN posed output -- the pelvis landmarks are the "
              "shipped rest template rotated, so recovery is an IDENTITY and not a fit")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"


def _subject_label(key: str) -> str:
    return key.replace("subject_", "subject ").replace("/subject ", " / subject ")


def x_pelvis_frame(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []

    clean = r.get("B1_synthetic_clean", {})
    noisy = r.get("B2a_synthetic_noisy_the_selector", {})
    scores = noisy.get("selection_rule_outcome", {}).get("median_deg_by_candidate", {})
    controls = noisy.get("controls_median_deg", {})
    for name, key in (("A_root_to_spine1", "pelvis_noisy_root_to_spine1"),
                      ("B_hipmid_to_spine1", "pelvis_noisy_hipmid_to_spine1"),
                      ("C_kabsch_pelvis", "pelvis_noisy_kabsch")):
        figs.append(fig(f"D7 pelvis orientation error on bent frames, {name.split('_', 1)[1]}",
                        scores.get(name), "deg", REF_NOISY, LOWER, key=key,
                        note="squat clip, tilt > 20 deg, 5 seeds pooled"))
    ctrls.append(fig("D7 control: the TRUNK line as the pelvis (today's code)",
                     controls.get("thorax_as_pelvis"), "deg", REF_NOISY, LOWER,
                     key="pelvis_ctrl_thorax_as_pelvis",
                     note="the defect D7 closes; it must lose, and it does, by 18-24 deg"))
    ctrls.append(fig("D7 control: a WORLD-VERTICAL pelvis (the plausible shortcut)",
                     controls.get("world_vertical"), "deg", REF_NOISY, LOWER,
                     key="pelvis_ctrl_world_vertical",
                     note="PRE-REGISTERED as a degenerate that must lose. IT DOES NOT: the "
                          "band FAILED. The true pelvis on all five clips is 1.1-12.7 deg "
                          "from vertical, so a constant is a good approximation there"))
    ctrls.append(fig("D7 control: the best LUMBAR direction (root->Spine2 / the spine line)",
                     min([v for v in (controls.get("lumbar_root_to_spine2"),
                                      controls.get("lumbar_spine_line")) if v is not None],
                        default=None), "deg", REF_NOISY, LOWER, key="pelvis_ctrl_lumbar",
                     note="carries Spine1's flexion; a lumbar frame, not a pelvis one"))
    squat = clean.get("convention_residual_with_the_SHIPPED_constants_median_deg", {})
    for clip, row in squat.items():
        if "squat" not in clip:
            continue
        figs.append(fig("D7 clean synthetic: the shipped rest CONVENTION's own residual",
                        row.get("C_kabsch_pelvis"), "deg", REF_SYNTH, LOWER,
                        key="pelvis_convention_residual",
                        note="what the shipped rest template costs on a clip whose own rest "
                             "it is not. 0.44-0.63 deg across the five clips"))
    oracle = r.get("B5_exact_recovery_oracle", {})
    figs.append(fig("D7 exact-recovery oracle: the posed pelvis back out of the converter",
                    oracle.get("worst_deg"), "deg", REF_ORACLE, LOWER,
                    key="pelvis_oracle_worst", note=f"band {oracle.get('band_deg')} deg"))
    ctrls.append(fig("D7 must-fail: the same posed body with NO spine landmark",
                     oracle.get("must_fail_no_spine_input_median_deg"), "deg", REF_ORACLE, LOWER,
                     key="pelvis_oracle_ctrl_no_spine",
                     note="falls to the trunk line, which is the difference that was posed in"))

    rigidity = r.get("B3_rigidity_on_the_real_take", {}).get("per_subject", {})
    for subject, block in (rigidity or {}).items():
        if not block:
            continue
        t = _subject_label(subject)
        for label, key in (("CANDIDATE  root->Spine1", "rigidity_root_to_spine1"),
                           ("CANDIDATE  midhips->Spine1", "rigidity_midhips_to_spine1")):
            row = block.get(label, {})
            figs.append(fig(f"D7 rigidity: {label.split('  ')[1]} length spread, {t}",
                            row.get("sd_mm"), "mm sd", REF_INVARIANT, LOWER,
                            key=f"{key}_{subject}",
                            note="the HeadEnd instrument. The verdict is on sd_mm, never sd%"))
            ctrls.append(fig(f"D7 control: the WORST body control's length spread, {t}",
                             row.get("worst_body_control_sd_mm"), "mm sd", REF_INVARIANT, LOWER,
                             key=f"rigidity_ctrl_worst_body_{subject}",
                             note="shin, thigh, forearm, upper arm -- the controls this lane "
                                  "already trusts. The candidate must not exceed it"))

    # The PRE-REGISTERED cut: torso+legs and arms, by trunk tilt tercile, one denominator
    # (frames every camera scored, IoU averaged over the four).
    partwise = r.get("B7_silhouette", {}).get("partwise_and_tercile", {})
    for subject, row in partwise.items():
        t = _subject_label(subject)
        for tercile in ("upright", "middle", "bent"):
            cell = row.get("terciles", {}).get(tercile, {})
            figs.append(fig(f"D7 silhouette, torso+legs IoU, {tercile} tercile, before (D3), {t}",
                            cell.get("torso_iou_D3"), "IoU", REF_MASKS, HIGHER,
                            key=f"silhouette_torso_before_{tercile}_{subject}"))
            figs.append(fig(f"D7 silhouette, torso+legs IoU, {tercile} tercile, after (D7), {t}",
                            cell.get("torso_iou_D7"), "IoU", REF_MASKS, HIGHER,
                            key=f"silhouette_torso_after_{tercile}_{subject}",
                            note="pre-registered to RISE on the bent tercile"))
            ctrls.append(fig(f"D7 oracle: MAMMA's own mesh, torso+legs, {tercile}, {t}",
                             cell.get("torso_iou_ORACLE"), "IoU", REF_MASKS, HIGHER,
                             key=f"silhouette_oracle_torso_{tercile}_{subject}",
                             note="the ceiling this instrument can reach; it reads none of "
                                  "our track and is bit-identical between runs"))
        arm = row.get("arm_iou_all_frames", {})
        figs.append(fig(f"D7 silhouette, ARMS IoU, all frames, before (D3), {t}",
                        arm.get("D3"), "IoU", REF_MASKS, HIGHER,
                        key=f"silhouette_arms_before_{subject}"))
        figs.append(fig(f"D7 silhouette, ARMS IoU, all frames, after (D7), {t}",
                        arm.get("D7"), "IoU", REF_MASKS, HIGHER,
                        key=f"silhouette_arms_after_{subject}",
                        note="pre-registered to be UNCHANGED -- the thorax frame is untouched"))
    # the whole-person by-camera cut, superseded by the above but kept
    silhouette = r.get("B7_silhouette", {}).get("whole_person_by_camera_cell", {})
    for cell, row in silhouette.items():
        key = cell.replace("/", "_").replace("subject_", "s")
        figs.append(fig(f"D7 silhouette IoU before (D3), {cell}", row.get("iou_before_D3"),
                        "IoU", REF_MASKS, HIGHER, key=f"silhouette_before_{key}"))
        figs.append(fig(f"D7 silhouette IoU after (D7), {cell}", row.get("iou_after_D7"),
                        "IoU", REF_MASKS, HIGHER, key=f"silhouette_after_{key}"))

    reported = r.get("reported_never_banded", {}).get("hips_joint_offset_and_temporal", {})
    for subject, block in reported.items():
        t = _subject_label(subject)
        for when, tag in (("before_D3", "before"), ("after_D7", "after")):
            figs.append(fig(f"D7 Hips joint horizontal offset from the captured hip midpoint, "
                            f"bent tercile, {tag}, {t}",
                            block.get(when, {}).get("horizontal_offset_bent_tercile_median_mm"),
                            "mm", "our own capture's hip-landmark midpoint", LOWER,
                            key=f"pelvis_hips_offset_{tag}_{subject}",
                            note="REPORTED, never banded: it must move, and a constant can "
                                 "zero it"))
    return figs, ctrls


VISUALS = {
    "converter": [
        dict(title="D7: how far the delivered pelvis is from the true pelvis, on bent frames",
             plain="Exact synthetic truth is the reference, at zero; lower is better. Blue is what "
                   "ships. Orange is what the code did before -- the pelvis welded to the trunk. "
                   "The hatched bar is a deliberately wrong answer: a pelvis frozen upright. It was "
                   "supposed to lose and it did not, which is why this step's own gate says FAIL.",
             better="lower",
             bars=[dict(label="Ships (fit the whole pelvis)", role="ours", key="pelvis_noisy_kabsch"),
                   dict(label="Alternative (hips to lower spine)", role="alt", key="pelvis_noisy_hipmid_to_spine1"),
                   dict(label="Alternative (pelvis centre to lower spine)", role="alt", key="pelvis_noisy_root_to_spine1"),
                   dict(label="Before D7: the trunk line", role="alt", key="pelvis_ctrl_thorax_as_pelvis"),
                   dict(label="Deliberately wrong: frozen upright", role="control", key="pelvis_ctrl_world_vertical")]),
        dict(title="D7: is the lower-spine landmark stable enough to build a pelvis on?",
             plain="No reference at all here -- a bone does not change length, so the spread of a "
                   "segment's length over the take is how much the detector wobbles. Lower is "
                   "better. The hatched bars are the body parts this project already trusts; the "
                   "candidate has to be no worse than the worst of them.",
             better="lower",
             bars=[dict(label="Hips to lower spine, performer 0", role="ours", key="rigidity_midhips_to_spine1_subject_00"),
                   dict(label="Hips to lower spine, performer 1", role="ours", key="rigidity_midhips_to_spine1_subject_01"),
                   dict(label="Worst trusted body part, performer 0", role="control", key="rigidity_ctrl_worst_body_subject_00"),
                   dict(label="Worst trusted body part, performer 1", role="control", key="rigidity_ctrl_worst_body_subject_01")]),
        dict(title="D7: how far the rig's hip joint sits from the hips we measured, on bent frames",
             plain="Our own captured hip midpoint is the reference, at zero; lower is better. Before "
                   "D7 the offset leaned with the trunk. This is REPORTED and not a pass/fail: a "
                   "frozen answer could drive it to zero without being right.",
             better="lower",
             bars=[dict(label="Before D7, performer 0", role="alt", key="pelvis_hips_offset_before_subject_00"),
                   dict(label="After D7, performer 0", role="ours", key="pelvis_hips_offset_after_subject_00"),
                   dict(label="Before D7, performer 1", role="alt", key="pelvis_hips_offset_before_subject_01"),
                   dict(label="After D7, performer 1", role="ours", key="pelvis_hips_offset_after_subject_01")]),
    ],
    "masks": [
        dict(title="D7: how well the body's outline covers the person, by how far they are bent over",
             plain="The reference fitter's person masks are the reference; higher is better. This is "
                   "the body and legs only, with the arms taken out. On performer 0 it rises in every "
                   "band and most where they are most bent over, which is what this step was meant to "
                   "do. The hatched bars are the reference fitter's own mesh through the same "
                   "rasteriser -- the ceiling, not a target.",
             better="higher",
             bars=[dict(label="Upright, before", role="alt", key="silhouette_torso_before_upright_subject_00"),
                   dict(label="Upright, after", role="ours", key="silhouette_torso_after_upright_subject_00"),
                   dict(label="Middle, before", role="alt", key="silhouette_torso_before_middle_subject_00"),
                   dict(label="Middle, after", role="ours", key="silhouette_torso_after_middle_subject_00"),
                   dict(label="Most bent, before", role="alt", key="silhouette_torso_before_bent_subject_00"),
                   dict(label="Most bent, after", role="ours", key="silhouette_torso_after_bent_subject_00"),
                   dict(label="Reference fitter's mesh, most bent", role="control", key="silhouette_oracle_torso_bent_subject_00")]),
        dict(title="D7: the same check on performer 1, where nothing moved",
             plain="Same reference, same three bands, higher is better. On this performer the outline "
                   "did not change in any band -- and this is the same performer whose hip joint did "
                   "not move either, so the two checks agree about who this step reaches.",
             better="higher",
             bars=[dict(label="Upright, before", role="alt", key="silhouette_torso_before_upright_subject_01"),
                   dict(label="Upright, after", role="ours", key="silhouette_torso_after_upright_subject_01"),
                   dict(label="Middle, before", role="alt", key="silhouette_torso_before_middle_subject_01"),
                   dict(label="Middle, after", role="ours", key="silhouette_torso_after_middle_subject_01"),
                   dict(label="Most bent, before", role="alt", key="silhouette_torso_before_bent_subject_01"),
                   dict(label="Most bent, after", role="ours", key="silhouette_torso_after_bent_subject_01"),
                   dict(label="Reference fitter's mesh, most bent", role="control", key="silhouette_oracle_torso_bent_subject_01")]),
        dict(title="D7: the arms, which this step does not touch and which did not move",
             plain="Same masks, higher is better, arms only. This step changes the hips and nothing "
                   "below the collarbone, and the photographs agree: neither performer's arms moved "
                   "outside the range the measurement itself is uncertain by.",
             better="higher",
             bars=[dict(label="Before D7, performer 0", role="alt", key="silhouette_arms_before_subject_00"),
                   dict(label="After D7, performer 0", role="ours", key="silhouette_arms_after_subject_00"),
                   dict(label="Before D7, performer 1", role="alt", key="silhouette_arms_before_subject_01"),
                   dict(label="After D7, performer 1", role="ours", key="silhouette_arms_after_subject_01")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_pelvis_frame({})
    for f in figs + ctrls:
        print(f"{f['key']:52s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
