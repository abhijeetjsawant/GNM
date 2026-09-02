#!/usr/bin/env python3
"""Ladder extractor for D2 -- the clavicle origin.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entry. To register it, add to
`ladder.py`:

    from extractors.d2_clavicle import x_clavicle

and give the rungs that carry it `extract=x_clavicle`, `reports=[REPORT]`. It reads
exactly one report, `artifacts/compare/d2-clavicle/gate.json`.

THREE REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim so the page
cannot stack them:

  * the own-capture ROUND TRIP -- "a canonical-proportioned body BY CONSTRUCTION";
  * ARM B -- "a canonical body built from arm B's own output", a different body, different
    poses, a different joint convention (SMPL-X regressor vs the SOMA-77 adapter);
  * the SCOREBOARD -- MAMMA `pred_joints`, agreement with an instrument and not accuracy.

A fourth unit appears and is not millimetres at all: the TEMPORAL figures are degrees per
frame against human physiology, and the twist figures are degrees. They must never share a
chart with a millimetre.

THE ONE THING A READER MUST NOT TAKE FROM THIS RUNG. The round-trip arm gets WORSE after
D2 (41.57/47.05 -> 67.25/79.32 mm) and the delivered arm gets much BETTER (181/218 ->
124/90 mm). Both are true and the round trip's rise is not the clavicle: that instrument
feeds the rig's own UpperLeg origins back as hip landmarks while the converter puts `Hips`
on their midpoint 80 mm higher, so its second solve sits 80 mm low, and the clavicle is
the only direction measured in the rig's own frame and so the only one that can see it.
The `d2_ctrl_hip_drop_removed_*` control is that attribution and belongs on the chart
beside it.

Self-check:  python3 tools/compare/extractors/d2_clavicle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d2-clavicle/gate.json"
REGEN = "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d2_clavicle_gate.py"

REF_ROUNDTRIP = ("a canonical-proportioned body BY CONSTRUCTION -- the converter scored "
                 "against its OWN output, root-relative to the hip midpoint")
REF_OURS = ("our own triangulated capture, root-relative (hip midpoint) -- the converter "
            "scored against its OWN input, ONE solve, the delivered configuration")
REF_ARMB = ("a canonical body built from arm B's own output -- MAMMA `pred_joints` in. A "
            "different body, different poses, a different joint convention; NOT the same "
            "axis as the own-capture figures")
REF_MAMMA = ("MAMMA `pred_joints`, the scoreboard's own statistic (per-joint median over "
             "frames, then median over joints). Agreement with an instrument, not accuracy")
REF_PHYSICAL = ("the rig's own joint angles against human physiology -- a peak joint rate "
                "near 800 deg/s, which at 30 fps is 26.67 deg per frame. No reference "
                "fitter enters this number")
SUBJECTS = ("subject_00", "subject_01")


def _g(node, *path, default=None):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def x_clavicle(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    figs: list = []
    ctrls: list = []

    for s in SUBJECTS:
        d = _g(report, "subjects", s)
        if not d:
            continue
        t = s.replace("_", " ")

        # --- the DELIVERED configuration: one solve, real landmarks. The headline.
        for label, key, value in (
            ("before D2 (the 0.72 torso-axis anchor)", "d2_delivered_arms_before",
             _g(d, "on_our_capture_canonical", "before", "arms")),
            ("after D2 (the rig's own shoulder origin)", "d2_delivered_arms_after",
             _g(d, "on_our_capture_canonical", "after", "arms")),
        ):
            figs.append(fig(f"arm landmarks on our own capture, canonical rig, {label}, {t}",
                            value, "mm median", REF_OURS, LOWER, key=f"{key}_{s}",
                            note=f"regenerate: {REGEN}"))
        figs.append(fig(f"legs on our own capture, canonical rig, unchanged by D2, {t}",
                        _g(d, "on_our_capture_canonical", "after", "legs"), "mm median",
                        REF_OURS, LOWER, key=f"d2_delivered_legs_after_{s}",
                        note="identical before and after: no leg direction reads an origin"))

        # --- the sized rig, re-solved and replayed. These stopped being the same arm.
        figs.append(fig(f"same arms, rig sized to the performer, RE-SOLVED, before D2, {t}",
                        _g(d, "sized_rig", "resolved_before", "arms"), "mm median", REF_OURS,
                        LOWER, key=f"d2_sized_resolved_arms_before_{s}"))
        figs.append(fig(f"same arms, rig sized to the performer, RE-SOLVED, after D2, {t}",
                        _g(d, "sized_rig", "resolved_after", "arms"), "mm median", REF_OURS,
                        LOWER, key=f"d2_sized_resolved_arms_after_{s}",
                        note="before D2 a re-solve on a sized rig returned bit-identical "
                             "rotations to the canonical solve; after D2 it does not, "
                             "because sizing moves the origin the clavicle is measured from"))
        figs.append(fig(f"the SCOREBOARD's method instead -- canonical rotations REPLAYED "
                        f"on a sized rig, after D2, {t}",
                        _g(d, "sized_rig", "replayed_scoreboard_method_after", "arms"),
                        "mm median", REF_OURS, LOWER,
                        key=f"d2_sized_replayed_arms_after_{s}",
                        note="reported beside the re-solve because they are no longer the "
                             "same arm. The scoreboard's method is unchanged"))

        # --- the round trip. The BEFORE bar is a control, per the plan's key list.
        figs.append(fig(f"round-trip oracle, arms, after D2, {t}",
                        _g(d, "roundtrip", "after", "per_group", "arms"), "mm median",
                        REF_ROUNDTRIP, LOWER, key=f"d2_roundtrip_arms_after_{s}",
                        note="this went UP and it is not the clavicle -- see the hip-drop "
                             "control on the same axis"))
        figs.append(fig(f"round-trip oracle, legs, after D2, {t} (must be 0.00)",
                        _g(d, "roundtrip", "after", "per_group", "legs"), "mm median",
                        REF_ROUNDTRIP, LOWER, key=f"d2_roundtrip_legs_after_{s}"))
        ctrls.append(fig(f"CONTROL round-trip oracle, arms, BEFORE D2 (the 0.72 anchor), {t}",
                         _g(d, "roundtrip", "before_via_legacy_anchor_swap", "per_group", "arms"),
                         "mm median", REF_ROUNDTRIP, LOWER,
                         key=f"d2_roundtrip_arms_before_{s}",
                         note="produced by swapping ONLY the origin helper, so candidate and "
                              "control share a code path; it reproduces the committed I1 "
                              "report to 0.01 mm, which is what licenses the comparison"))

        # --- the controls that carry the argument
        ctrls.append(fig(f"ATTRIBUTION: the same round trip with the rig's own hip drop out "
                         f"of the RE-SOLVE's origin, arms, {t}",
                         _g(d, "CONTROL_hip_drop_removed_pass2", "per_group", "arms"),
                         "mm median", REF_ROUNDTRIP, LOWER,
                         key=f"d2_ctrl_hip_drop_removed_{s}",
                         note="the delivered path is untouched; only the instrument's own "
                              "re-solve changes. Under a millimetre, so ALL of the round "
                              "trip's rise is the root-placement convention (open D-lane "
                              "work) and none of it is the clavicle"))
        ctrls.append(fig(f"CONTROL the legacy anchor, reproduced by helper swap, arms, {t}",
                         _g(d, "CONTROL_legacy_anchor_reproduces_the_committed_report",
                            "swapped_arms_median_mm"), "mm median", REF_ROUNDTRIP, LOWER,
                         key=f"d2_ctrl_legacy_anchor_{s}",
                         note="must equal the committed I1 figure to 0.01 mm"))
        ctrls.append(fig(f"CONTROL the BEST of 17 on-torso-axis scalars, 0.40 to 1.20, "
                         f"arms, {t} (must still fail)",
                         _g(d, "CONTROL_legacy_scalar_sweep", "best_arms_median_mm"),
                         "mm median", REF_ROUNDTRIP, LOWER, key=f"d2_ctrl_best_scalar_{s}",
                         note="NO GATE A CONSTANT CAN PASS: the true origin is 110 mm OFF "
                              "the torso axis, so no point on that axis can land it. This "
                              "is also the 'tune 0.72 until a subject improves' degenerate. "
                              f"best scalar: "
                              f"{_g(d, 'CONTROL_legacy_scalar_sweep', 'best_scalar')}"))
        ctrls.append(fig(f"CONTROL the clavicle's PARENT origin (UpperChest), arms, {t} "
                         f"(must fail)",
                         _g(d, "CONTROL_upperchest_origin", "per_group", "arms"),
                         "mm median", REF_ROUNDTRIP, LOWER,
                         key=f"d2_ctrl_upperchest_origin_{s}",
                         note="the plausible off-by-one, 110 mm inboard of the right origin"))

        # --- arm B. Its own reference; it reports and never selects.
        b = _g(d, "arm_B_mamma_joints_in", default={})
        for label, key, value in (
            ("before D2", "d2_mammaB_arms_before", _g(b, "canonical_before", "arms")),
            ("after D2", "d2_mammaB_arms_after", _g(b, "canonical_after", "arms")),
        ):
            figs.append(fig(f"the converter on MAMMA's own joints, arms, canonical rig, "
                            f"{label}, {t}", value, "mm median", REF_ARMB, LOWER,
                            key=f"{key}_{s}"))
        ctrls.append(fig(f"ORACLE round trip inside arm B, arms, after D2, {t}",
                         _g(b, "roundtrip_after", "arms"), "mm median", REF_ARMB, LOWER,
                         key=f"d2_mammaB_roundtrip_after_{s}",
                         note="the same instrument defect as the own-capture round trip: "
                              f"with the hip drop out of the re-solve it reads "
                              f"{_g(b, 'roundtrip_after_hip_drop_removed_pass2', 'arms')}"))

        # --- the temporal cost. A DIFFERENT UNIT and a different reference.
        temporal = _g(d, "temporal", default={})
        chain = ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                 "LeftLowerArm", "RightLowerArm")
        for label, when, key in (("before D2", "before", "d2_temporal_over_ceiling_before"),
                                 ("after D2", "after", "d2_temporal_over_ceiling_after")):
            total = sum(_g(temporal, "joints", n, when,
                           "frames_over_the_physical_ceiling", default=0) for n in chain)
            figs.append(fig(f"clavicle-chain frames whose joint rate exceeds a human's peak, "
                            f"{label}, {t}", total, "frames of 149 steps", REF_PHYSICAL,
                            LOWER, key=f"{key}_{s}",
                            note="THE POSITIONAL SCORE CANNOT SEE THIS. D2 measures the "
                                 "direction from an origin 60-170 mm from the landmark "
                                 "where the anchor sat ~400 mm away, and a short lever arm "
                                 "turns landmark noise into direction noise. The arm root "
                                 "lands far better and travels worse"))
        ctrls.append(fig(f"CONTROL the same count on the LOWER LEG, {t} (must be identical "
                         f"before and after)",
                         _g(temporal, "joints", "LeftLowerLeg", "after",
                            "frames_over_the_physical_ceiling"), "frames of 149 steps",
                         REF_PHYSICAL, LOWER, key=f"d2_temporal_leg_control_{s}",
                         note="the blast radius stated temporally: legs, neck and head "
                              "step-angles are bit-identical"))

        # --- the scoreboard. A third reference.
        sc = _g(report, "scoreboard", "subjects", s, default={})
        for arm in ("canon", "sized"):
            for when, key in (("before_mm", "before"), ("after_mm", "after")):
                figs.append(fig(f"agreement with MAMMA, `{arm}` rig, all joints, {when[:-3]}"
                                f" D2, {t}", _g(sc, f"{arm}_all_joints", when), "mm median",
                                REF_MAMMA, LOWER, key=f"d2_scoreboard_{arm}_{key}_{s}"))
            figs.append(fig(f"agreement with MAMMA, `{arm}` rig, the SIX ARM joints only, "
                            f"after D2, {t}", _g(sc, f"{arm}_arms_six", "after_mm"),
                            "mm median", REF_MAMMA, LOWER,
                            key=f"d2_scoreboard_{arm}_arms_after_{s}"))
        ctrls.append(fig(f"CONTROL the scoreboard's `capture` arm must be UNCHANGED, {t}",
                         _g(sc, "capture_must_be_identical", "max_abs_difference_mm"),
                         "mm max difference", REF_MAMMA, LOWER,
                         key=f"d2_scoreboard_capture_unchanged_{s}",
                         note="it reads the triangulated positions, which D2 does not "
                              "touch. Anything but 0 means the rebuild is not the same data"))
    return figs, ctrls


if __name__ == "__main__":
    f, c = x_clavicle({})
    for row in f:
        print(f"FIG   {row['value']!s:>10}  {row['unit']:<24} {row['label']}")
    for row in c:
        print(f"CTRL  {row['value']!s:>10}  {row['unit']:<24} {row['label']}")
    print(f"\n{len(f)} figures, {len(c)} controls/oracles")


# ---------------------------------------------------------------------------------------
# Proposed `VISUALS` entry. `ladder.py`'s owner registers it; this is the entry to paste,
# not a registration. Roles use the validated palette: ours blue, mamma orange, alt aqua,
# control hatched.
#
# THREE charts because there are three units and three references, and a bar may never
# cross them: millimetres against our own capture, millimetres against MAMMA, and frames
# above a physiological rate. The round trip gets its own chart with its attribution
# control ON IT, because the number is misleading without the control beside it.
PROPOSED_VISUALS = {
    "converter": [
        dict(title="Where the arm root lands, on the delivered character",
             plain="How far the top of the arm ends up from the shoulder the cameras "
                   "actually saw, in millimetres, averaged over the take. Lower is better. "
                   "The rig used to aim the collarbone from a made-up point on the spine "
                   "while actually pivoting it at the shoulder; now it aims it from the "
                   "shoulder itself. Both performers improve, and the second one by more "
                   "than half. The legs are untouched and are shown to prove it.",
             better="lower",
             bars=[dict(label="Performer 1, before", role="control",
                        key="d2_delivered_arms_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2_delivered_arms_after_subject_00"),
                   dict(label="Performer 2, before", role="control",
                        key="d2_delivered_arms_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2_delivered_arms_after_subject_01"),
                   dict(label="Performer 1, legs (unchanged)", role="alt",
                        key="d2_delivered_legs_after_subject_00"),
                   dict(label="Performer 2, legs (unchanged)", role="alt",
                        key="d2_delivered_legs_after_subject_01")]),

        dict(title="The self-consistency check, and why it moved the wrong way",
             plain="This test feeds the rig its own output back and asks it to reproduce "
                   "itself. It got WORSE, and the last two bars say why: the test hands "
                   "back the rig's hip joints as hip landmarks while the converter treats "
                   "that midpoint as a point 80 mm higher up the pelvis, so on the second "
                   "pass the whole skeleton sits 80 mm low. Nothing could see that before, "
                   "because every other measurement is taken between two landmarks and a "
                   "shift cancels. Correct for it and the test closes to under a "
                   "millimetre, which is the real accuracy of the change. Lower is better.",
             better="lower",
             bars=[dict(label="Performer 1, before", role="control",
                        key="d2_roundtrip_arms_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2_roundtrip_arms_after_subject_00"),
                   dict(label="Performer 2, before", role="control",
                        key="d2_roundtrip_arms_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2_roundtrip_arms_after_subject_01"),
                   dict(label="Performer 1, with the 80 mm corrected", role="alt",
                        key="d2_ctrl_hip_drop_removed_subject_00"),
                   dict(label="Performer 2, with the 80 mm corrected", role="alt",
                        key="d2_ctrl_hip_drop_removed_subject_01"),
                   dict(label="Best of 17 tuned constants (still fails)", role="control",
                        key="d2_ctrl_best_scalar_subject_00")]),

        dict(title="What it cost: the arm now jitters more",
             plain="The number of frames in which a collarbone or arm joint turns faster "
                   "than a human joint can (about 800 degrees a second). Lower is better. "
                   "Aiming from the shoulder means aiming from a point only 6-17 cm from "
                   "the target instead of 40 cm away, and a short lever turns small "
                   "wobbles in the detected shoulder into large swings of the bone. The "
                   "arm now ends up in a much better PLACE and gets there less smoothly. "
                   "The leg bar is unchanged and is there to show the rest of the body was "
                   "not touched.",
             better="lower",
             bars=[dict(label="Performer 1, before", role="alt",
                        key="d2_temporal_over_ceiling_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2_temporal_over_ceiling_after_subject_00"),
                   dict(label="Performer 2, before", role="alt",
                        key="d2_temporal_over_ceiling_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2_temporal_over_ceiling_after_subject_01"),
                   dict(label="Performer 1, lower leg (untouched)", role="control",
                        key="d2_temporal_leg_control_subject_00")]),
    ],
    "delivered": [
        dict(title="Agreement with the reference fitter, before and after",
             plain="How far our delivered joints sit from MAMMA's, the research fitter we "
                   "measure against. Lower is better, and this is agreement with another "
                   "instrument rather than accuracy -- neither side has ground truth. The "
                   "sized rig improves for both performers. The first performer's "
                   "canonical bar does not move, and that is expected: the stock rig's "
                   "shoulders are far wider than these performers', so aiming the "
                   "collarbone correctly on a rig of the wrong width does not help until "
                   "the width is fixed.",
             better="lower",
             bars=[dict(label="Performer 1, stock rig, before", role="control",
                        key="d2_scoreboard_canon_before_subject_00"),
                   dict(label="Performer 1, stock rig, after", role="mamma",
                        key="d2_scoreboard_canon_after_subject_00"),
                   dict(label="Performer 1, sized rig, before", role="control",
                        key="d2_scoreboard_sized_before_subject_00"),
                   dict(label="Performer 1, sized rig, after", role="mamma",
                        key="d2_scoreboard_sized_after_subject_00"),
                   dict(label="Performer 2, stock rig, before", role="control",
                        key="d2_scoreboard_canon_before_subject_01"),
                   dict(label="Performer 2, stock rig, after", role="mamma",
                        key="d2_scoreboard_canon_after_subject_01"),
                   dict(label="Performer 2, sized rig, before", role="control",
                        key="d2_scoreboard_sized_before_subject_01"),
                   dict(label="Performer 2, sized rig, after", role="mamma",
                        key="d2_scoreboard_sized_after_subject_01")]),
    ],
}
