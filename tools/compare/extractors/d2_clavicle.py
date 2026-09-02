#!/usr/bin/env python3
"""Ladder extractor for D2 -- the clavicle origin.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entry. To register it, add to
`ladder.py`:

    from extractors.d2_clavicle import x_clavicle, x_root_placement

and give the rungs that carry them `extract=x_clavicle` / `extract=x_root_placement`,
`reports=[REPORT]`. Both read exactly one report,
`artifacts/compare/d2-clavicle/gate.json`. `x_clavicle` is D2 (the clavicle origin);
`x_root_placement` is D2b (the root placed on the captured hips) and is the one that
carries the ABSOLUTE placement figures and the silhouette.

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


# =========================================================================== D2b
# THE ROOT PLACED ON THE CAPTURED HIPS. Six references appear on this rung and NONE of them
# may share a chart with another:
#
#   1. the own-capture ROUND TRIP        millimetres, root-relative, our own output
#   2. our own CAPTURE, one solve        millimetres, root-relative, our own input
#   3. ARM B                             millimetres, a different body and joint convention
#   4. the SCOREBOARD / rung 11          millimetres, ABSOLUTE capture world, MAMMA's joints
#   5. ABSOLUTE HIP PLACEMENT            millimetres, absolute capture world, our own
#                                        captured hip landmarks -- the figure D2b owns
#   6. the SILHOUETTE                    a dimensionless overlap against SAM2's masks
#
# and a seventh unit that is not millimetres at all: frames above a physiological joint rate.
#
# THE ONE THING A READER MUST NOT TAKE FROM THIS RUNG. The round trip closing to 0.5 mm is a
# CONSISTENCY figure, not an accuracy figure -- the converter scored against its own output.
# What D2b actually claims is the ABSOLUTE row: the delivered rig's hip joints now sit on
# the captured hips horizontally (26/43 mm -> 0.0) and the tilt correlation collapses
# (0.97 -> 0.03 / -0.32). The vertical residue does NOT go away; it moves from the converter
# into the ground projection, where it is one uncapped scalar per take, and it is D5's.
# AND THE SILHOUETTE FELL, in all eight camera-subject cells, on both precision and recall.
# That belongs on the same page as the improvements, not in a footnote.
REF_ABS_HIPS = ("our own captured hip landmarks in ABSOLUTE capture world (Z-up metres), "
                "AFTER the ground projection. NOT root-relative: every own-capture figure "
                "in D2 removed exactly this vector by construction")
REF_SILHOUETTE_X = ("MAMMA's SAM2 masks -- the pixels of the actual footage, the one "
                    "retained artifact on this fixture that is not model-mediated. Blind "
                    "to depth, to a left/right mirror of a fore-aft symmetric pose, and to "
                    "everything inside the outline")


def x_root_placement(_: dict) -> tuple[list, list]:
    report = _load(REPORT)
    if not report:
        return [], []
    d2b = _g(report, "d2b_root_placement", default={})
    if not d2b:
        return [], []
    figs: list = []
    ctrls: list = []

    for s in SUBJECTS:
        d = _g(d2b, "subjects", s, default={})
        if not d:
            continue
        t = s.replace("_", " ")
        canon = _g(d, "rigs", "canonical", default={})
        sized = _g(d, "rigs", "sized_resolved", default={})

        # --- 1. the round trip, both rigs. D2's own pre-registered band, now met.
        for label, key, value in (
            ("D2 alone", "d2b_roundtrip_arms_before",
             _g(canon, "D2_alone", "roundtrip", "arms")),
            ("D2b, the root on the captured hips", "d2b_roundtrip_arms_after",
             _g(canon, "D2b_shipped", "roundtrip", "arms")),
        ):
            figs.append(fig(f"round-trip oracle, arms, canonical rig, {label}, {t}",
                            value, "mm median", REF_ROUNDTRIP, LOWER, key=f"{key}_{s}",
                            note="a CONSISTENCY figure and not an accuracy one: the "
                                 "converter is scored against its own output. "
                                 f"regenerate: {REGEN}"))
        figs.append(fig(f"round-trip oracle, arms, SIZED rig, D2b, {t}",
                        _g(sized, "D2b_shipped", "roundtrip", "arms"), "mm median",
                        REF_ROUNDTRIP, LOWER, key=f"d2b_roundtrip_arms_sized_after_{s}"))
        figs.append(fig(f"round-trip oracle, legs, D2b, {t} (must be 0.00)",
                        _g(canon, "D2b_shipped", "roundtrip", "legs"), "mm median",
                        REF_ROUNDTRIP, LOWER, key=f"d2b_roundtrip_legs_after_{s}",
                        note="0.00 here proves NOTHING about placement: the legs are "
                             "measured landmark-to-landmark and the score is "
                             "root-relative, so a translation cancels twice"))

        # --- 2. the delivered configuration, one solve, real landmarks
        for label, key, value in (
            ("D2 alone", "d2b_delivered_arms_before",
             _g(canon, "D2_alone", "delivered_on_our_capture", "arms")),
            ("D2b", "d2b_delivered_arms_after",
             _g(canon, "D2b_shipped", "delivered_on_our_capture", "arms")),
        ):
            figs.append(fig(f"arm landmarks on our own capture, canonical rig, {label}, {t}",
                            value, "mm median", REF_OURS, LOWER, key=f"{key}_{s}"))
        figs.append(fig(f"same arms, rig sized to the performer, RE-SOLVED, D2b, {t}",
                        _g(sized, "D2b_shipped", "delivered_on_our_capture", "arms"),
                        "mm median", REF_OURS, LOWER,
                        key=f"d2b_sized_resolved_arms_after_{s}"))
        figs.append(fig(f"legs on our own capture, canonical rig, D2b, {t}",
                        _g(canon, "D2b_shipped", "delivered_on_our_capture", "legs"),
                        "mm median", REF_OURS, LOWER, key=f"d2b_delivered_legs_after_{s}",
                        note="unchanged by D2b: no leg direction reads a root or an origin"))

        # --- 3. arm B, on its own reference. It reports and never selects.
        b = _g(d, "arm_B_mamma_joints_in", default={})
        for label, key in (("D2 alone", "d2b_mammaB_arms_before"),
                           ("D2b", "d2b_mammaB_arms_after")):
            arm = "D2_alone" if "before" in key else "D2b_shipped"
            figs.append(fig(f"the converter on MAMMA's own joints, arms, canonical rig, "
                            f"{label}, {t}", _g(b, f"canonical_{arm}", "arms"), "mm median",
                            REF_ARMB, LOWER, key=f"{key}_{s}"))

        # --- 5. THE FIGURE D2b OWNS. Absolute, and no root-relative instrument sees it.
        ab = _g(d2b, "absolute_hip_placement", "subjects", s, default={})
        for arm_label, key in (("delivered_before_D2", "before"), ("D2", "d2"),
                               ("D2b", "after")):
            cell = _g(ab, arm_label, default={})
            if not cell:
                continue
            figs.append(fig(f"delivered hip joints against the captured hips, HORIZONTAL, "
                            f"{arm_label}, {t}", _g(cell, "horizontal_mm", "median"),
                            "mm median, absolute", REF_ABS_HIPS, LOWER,
                            key=f"d2b_hip_abs_offset_horizontal_{key}_{s}",
                            note="the per-frame, tilt-dependent term. It is what the root "
                                 "fix removes, and the only own-capture figure in this "
                                 "lane that can see it at all"))
            figs.append(fig(f"the same, VERTICAL (capture +Z), {arm_label}, {t}",
                            _g(cell, "component_z_up_mm", "median"), "mm median, absolute",
                            REF_ABS_HIPS, LOWER,
                            key=f"d2b_hip_abs_offset_vertical_{key}_{s}",
                            note="dominated by the ground projection's ONE uncapped hoist, "
                                 "set by the worst frame of the take. D2b barely moves it; "
                                 "what remains is the legs' surplus length, which is D5"))
            figs.append(fig(f"the same, NORM, {arm_label}, {t}",
                            _g(cell, "norm_mm", "median"), "mm median, absolute",
                            REF_ABS_HIPS, LOWER,
                            key=f"d2b_hip_abs_offset_norm_{key}_{s}",
                            note=f"correlation with pelvis tilt: "
                                 f"{_g(cell, 'correlation_with_pelvis_tilt', 'norm')}"))

        # --- 4. rung 11, LEGS AND ARMS SEPARATED, absolute capture world
        r11 = _g(d2b, "rung11", "subjects", s, default={})
        for scope in ("all", "arms", "legs"):
            cell = _g(r11, f"canon_{scope}", default={})
            if not cell:
                continue
            margin = _g(cell, "margin_D2_minus_D2b", default={})
            for arm_label, key in (("delivered_before_D2_mm", "before"), ("D2_mm", "d2"),
                                   ("D2b_mm", "after")):
                figs.append(fig(f"agreement with MAMMA, canonical rig, {scope.upper()}, "
                                f"{arm_label[:-3]}, {t}", cell.get(arm_label), "mm median",
                                REF_MAMMA, LOWER,
                                key=f"d2b_rung11_{scope}_{key}_{s}",
                                note=f"D2 minus D2b margin {margin.get('median_mm')} mm, "
                                     f"95 % CI {margin.get('ci95_mm')}, p(wrong sign) "
                                     f"{margin.get('p_wrong_sign')}. Paired moving-block "
                                     f"bootstrap, block 15, 2000 draws, identical draws"))

        # --- the hoist that remains, and the contacts
        for arm_label, arm, key in (("D2 alone", "D2_alone", "before"),
                                    ("D2b", "D2b_shipped", "after")):
            figs.append(fig(f"the ground projection's uncapped vertical hoist, canonical "
                            f"rig, {arm_label}, {t}",
                            round(1000.0 * (_g(canon, arm, "ground_projection",
                                               "diagnostics",
                                               "ground_penetration_before_m") or 0.0), 2),
                            "mm", "the rig's own feet against the estimated floor", LOWER,
                            key=f"d2b_hoist_{key}_{s}",
                            note="body_projection.py:1209 adds this to the root's Y with "
                                 "NO cap. The capped per-contact term is a different, much "
                                 "smaller one. It falls because the rig is no longer sunk "
                                 "by its own hip convention"))

        # --- 6. the silhouette. It FELL, and that is on the chart.
        sil = _g(d2b, "silhouette", default={})
        for arm_key, key in (("before_per_camera_subject", "before"),
                             ("d2_alone_per_camera_subject", "d2"),
                             ("after_per_camera_subject", "after")):
            rows = _g(sil, arm_key, default={})
            cells = [v["iou_median"] for k, v in rows.items() if s in k]
            if not cells:
                continue
            figs.append(fig(f"silhouette IoU against MAMMA's SAM2 masks, {key}, {t}",
                            round(sum(cells) / len(cells), 4), "IoU, 0-1",
                            REF_SILHOUETTE_X, HIGHER, key=f"d2b_silhouette_iou_{key}_{s}",
                            note="mean over the four cameras of the per-camera median. "
                                 "HIGHER is better, and this one FELL"))
        for arm_key, key in (("before_per_camera_subject", "before"),
                             ("after_per_camera_subject", "after")):
            rows = _g(sil, arm_key, default={})
            for metric in ("precision_median", "recall_median"):
                cells = [v[metric] for k, v in rows.items() if s in k]
                if not cells:
                    continue
                figs.append(fig(f"silhouette {metric[:-7]}, {key}, {t}",
                                round(sum(cells) / len(cells), 4), f"{metric[:-7]}, 0-1",
                                REF_SILHOUETTE_X, HIGHER,
                                key=f"d2b_silhouette_{metric[:-7]}_{key}_{s}",
                                note="reported beside IoU and never collapsed into it: "
                                     "the degenerate a mesh instrument produces -- "
                                     "something too big -- buys recall with precision. "
                                     "BOTH fell here, so the mesh moved OFF the pixels"))
        ctrls.append(fig(f"ORACLE silhouette IoU, MAMMA's own mesh, must be UNCHANGED, {t}",
                         round(sum(v["iou_median"] for k, v in
                                   _g(sil, "after_oracle_mamma_mesh", default={}).items()
                                   if s in k) / 4.0, 4)
                         if _g(sil, "after_oracle_mamma_mesh") else None,
                         "IoU, 0-1", REF_SILHOUETTE_X, HIGHER,
                         key=f"d2b_silhouette_oracle_{s}",
                         note="MAMMA's own mesh through our scoring path. It reads none of "
                              "our track, so it must be bit-stable between the two runs -- "
                              "max difference "
                              f"{_g(sil, 'oracle_and_control_must_be_unchanged', 'max_abs_oracle_iou_difference')}. "
                              "It is also the ceiling this instrument can reach at all"))

        # --- 7. the temporal baseline for D2c. A DIFFERENT UNIT.
        temporal = _g(d, "temporal_baseline_for_D2c", default={})
        chain = ("LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
                 "LeftLowerArm", "RightLowerArm")
        for block, when, key in (("canonical_before_D2_vs_D2", "before", "before_d2"),
                                 ("canonical_D2_vs_D2b", "before", "d2"),
                                 ("canonical_D2_vs_D2b", "after", "d2b")):
            total = sum(_g(temporal, block, "joints", n, when,
                           "frames_over_the_physical_ceiling", default=0) for n in chain)
            figs.append(fig(f"clavicle-chain frames whose joint rate exceeds a human's "
                            f"peak, {key}, {t}", total, "frames of 149 steps",
                            REF_PHYSICAL, LOWER, key=f"d2b_temporal_over_ceiling_{key}_{s}",
                            note="D2b carries NO band for this. It is D2c's baseline, and "
                                 "the positional figures above cannot see it"))

        # --- the controls, all through the shipped code path
        c = _g(d, "controls", default={})
        ctrls.append(fig(f"CONTROL the lift with its SIGN FLIPPED (must fail), {t}",
                         _g(c, "b_sign_flipped", "roundtrip", "arms"), "mm median",
                         REF_ROUNDTRIP, LOWER, key=f"d2b_ctrl_sign_flipped_{s}",
                         note="same magnitude, same frame, opposite direction. It "
                              "separates 'a lift of the right size' from 'the right lift'"))
        ctrls.append(fig(f"CONTROL the same lift applied as a WORLD VERTICAL (must fail), {t}",
                         _g(c, "d_world_vertical_instead_of_the_hips_frame",
                            "roundtrip", "arms"), "mm median", REF_ROUNDTRIP, LOWER,
                         key=f"d2b_ctrl_world_vertical_{s}",
                         note="THE PLAUSIBLE SHORTCUT: identical to the derivation while "
                              "the pelvis is upright, wrong as it leans. On the "
                              "most-tilted quartile it delivers "
                              f"{_g(c, 'd_world_vertical_instead_of_the_hips_frame', 'delivered_arms_on_the_most_tilted_quartile_mm')} mm "
                              f"against the shipped "
                              f"{_g(c, 'd_world_vertical_instead_of_the_hips_frame', 'shipped_arms_on_the_most_tilted_quartile_mm')} mm"))
        sweep = _g(c, "c_lift_sweep_along_the_hips_up_axis", default={})
        best = None
        if sweep.get("per_lift"):
            # the best of the SIXTEEN OTHERS -- the skeleton's own value is the candidate,
            # not a control, and including it would make this bar the candidate's own bar.
            own = f"{sweep.get('the_skeletons_own_value_mm', 80.0):.0f}"
            others = {k: v for k, v in sweep["per_lift"].items() if k != own}
            if others:
                best = min(others.values(), key=lambda v: v["arms_median_mm"])
        ctrls.append(fig(f"CONTROL the BEST of the 16 OTHER tuned lifts, 0 to 160 mm, {t}",
                         best["arms_median_mm"] if best else None, "mm median",
                         REF_ROUNDTRIP, LOWER, key=f"d2b_ctrl_best_lift_{s}",
                         note="NO GATE A CONSTANT CAN PASS: the minimum of the sweep sits "
                              "at the skeleton's OWN value "
                              f"({sweep.get('the_skeletons_own_value_mm')} mm), which the "
                              "shipped code reads from `rest` and never writes down. "
                              f"Lifts clearing the 5 mm band on this subject: "
                              f"{sweep.get('values_passing_the_band')} -- where that is "
                              "more than one, the neighbouring value fails on the OTHER "
                              "subject, and a shipped constant is one number for both"))
        ctrls.append(fig(f"CONTROL round-trip legs under EVERY variant, {t} (must be 0.00)",
                         max(_g(c, "e_legs_and_torso_are_0.00_under_every_variant",
                                "sweep", default=[0.0])), "mm median", REF_ROUNDTRIP,
                         LOWER, key=f"d2b_ctrl_legs_zero_{s}",
                         note="untouched by construction, not by luck -- and, again, NOT "
                              "evidence that the placement is right"))
    return figs, ctrls


if __name__ == "__main__":
    f, c = x_clavicle({})
    f2, c2 = x_root_placement({})
    f, c = f + f2, c + c2
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


# ---------------------------------------------------------------------------------------
# D2b's proposed `VISUALS` entries. `ladder.py`'s owner registers them; this is the entry
# to paste, not a registration. Roles use the validated palette (ours blue, mamma orange,
# alt aqua, control hatched); do not repaint them.
#
# FIVE charts because there are five units and five references, and a bar may never cross
# them: millimetres against our own output (the round trip), millimetres against our own
# captured hips in ABSOLUTE world, millimetres against MAMMA's joints, a dimensionless
# overlap against SAM2's masks, and frames above a physiological joint rate. The silhouette
# chart is on the page BECAUSE it fell.
PROPOSED_VISUALS_D2B = {
    "converter": [
        dict(title="Where the character's hips actually sit, against the hips the cameras saw",
             plain="The rig has two hip joints, one at the top of each thigh, and the "
                   "cameras see where the performer's are. Until now the code put a "
                   "different bone -- the pelvis root, 8 cm higher -- on that spot, so the "
                   "whole skeleton sat 8 cm low on its own hips and the floor-contact stage "
                   "then hoisted it back up by about 14 cm. This chart is the sideways "
                   "(horizontal) miss, which is the part that changes as the performer "
                   "bends: it goes to zero. The up-down miss is a separate chart, because "
                   "it is a different problem and it does not go away.",
             better="lower",
             bars=[dict(label="Performer 1, before", role="control",
                        key="d2b_hip_abs_offset_horizontal_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2b_hip_abs_offset_horizontal_after_subject_00"),
                   dict(label="Performer 2, before", role="control",
                        key="d2b_hip_abs_offset_horizontal_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2b_hip_abs_offset_horizontal_after_subject_01")]),

        dict(title="The up-down miss, which this step moves rather than removes",
             plain="The same measurement, taken straight up. It barely changes -- and that "
                   "is the honest result. The floor-contact stage lifts the whole character "
                   "by one number per shot, chosen so the lowest foot just touches the "
                   "ground, and that lift simply re-adjusts. What is left is that the rig's "
                   "legs are about 3 cm too long for these performers, so a character whose "
                   "hips are in the right place has feet that are too low. Fixing the "
                   "proportions is a later step. Lower is better.",
             better="lower",
             bars=[dict(label="Performer 1, before", role="control",
                        key="d2b_hip_abs_offset_vertical_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2b_hip_abs_offset_vertical_after_subject_00"),
                   dict(label="Performer 2, before", role="control",
                        key="d2b_hip_abs_offset_vertical_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2b_hip_abs_offset_vertical_after_subject_01")]),

        dict(title="The self-consistency check, now closing",
             plain="This test feeds the rig its own output back and asks it to reproduce "
                   "itself. After the collarbone change it read 67 and 79 mm and we showed "
                   "the whole of that was the hip-placement problem; with the hips placed "
                   "it reads half a millimetre. The last two bars are deliberately wrong "
                   "answers run through exactly the same code: the same 8 cm lift applied "
                   "downwards, and the same 8 cm applied straight up in the world instead "
                   "of along the pelvis. Both fail, which is what makes the passing bar "
                   "mean something. Lower is better.",
             better="lower",
             bars=[dict(label="Performer 1, before", role="control",
                        key="d2b_roundtrip_arms_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2b_roundtrip_arms_after_subject_00"),
                   dict(label="Performer 2, before", role="control",
                        key="d2b_roundtrip_arms_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2b_roundtrip_arms_after_subject_01"),
                   dict(label="Wrong: lift reversed", role="control",
                        key="d2b_ctrl_sign_flipped_subject_00"),
                   dict(label="Wrong: lift in world, not pelvis", role="control",
                        key="d2b_ctrl_world_vertical_subject_00"),
                   dict(label="Best of the 16 other tuned lifts", role="alt",
                        key="d2b_ctrl_best_lift_subject_00")]),

        dict(title="What it cost: the outline fits the photographs worse",
             plain="This is the one measurement here that is scored against the actual "
                   "pixels of the footage rather than against another piece of software. "
                   "It got worse, on every camera and both performers, and both halves of "
                   "it fell -- the character is not too big or too small, it has moved off "
                   "the person. The most likely reason is that two errors used to partly "
                   "cancel: the hips were 8 cm low and the legs are 3 cm too long. Removing "
                   "one leaves the other exposed. Higher is better; the last bar is the "
                   "reference fitter's own mesh, which is the best this measurement can do.",
             better="higher",
             bars=[dict(label="Performer 1, before", role="alt",
                        key="d2b_silhouette_iou_before_subject_00"),
                   dict(label="Performer 1, after", role="ours",
                        key="d2b_silhouette_iou_after_subject_00"),
                   dict(label="Performer 2, before", role="alt",
                        key="d2b_silhouette_iou_before_subject_01"),
                   dict(label="Performer 2, after", role="ours",
                        key="d2b_silhouette_iou_after_subject_01"),
                   dict(label="Reference fitter's own mesh", role="mamma",
                        key="d2b_silhouette_oracle_subject_00")]),

        dict(title="Unchanged: the arm still jitters, and this step does not answer it",
             plain="The number of frames in which a collarbone or arm joint turns faster "
                   "than a human joint can. The collarbone change made this worse; placing "
                   "the hips brings it partway back. It is reported here as a baseline for "
                   "the step that owns the question, and there is no pass mark on it. "
                   "Lower is better.",
             better="lower",
             bars=[dict(label="Performer 1, before the collarbone fix", role="alt",
                        key="d2b_temporal_over_ceiling_before_d2_subject_00"),
                   dict(label="Performer 1, collarbone only", role="control",
                        key="d2b_temporal_over_ceiling_d2_subject_00"),
                   dict(label="Performer 1, hips placed too", role="ours",
                        key="d2b_temporal_over_ceiling_d2b_subject_00"),
                   dict(label="Performer 2, before the collarbone fix", role="alt",
                        key="d2b_temporal_over_ceiling_before_d2_subject_01"),
                   dict(label="Performer 2, collarbone only", role="control",
                        key="d2b_temporal_over_ceiling_d2_subject_01"),
                   dict(label="Performer 2, hips placed too", role="ours",
                        key="d2b_temporal_over_ceiling_d2b_subject_01")]),
    ],
    "delivered": [
        dict(title="Agreement with the reference fitter, arms and legs kept apart",
             plain="How far our delivered joints sit from MAMMA's, the research fitter we "
                   "measure against, in real world coordinates rather than relative to the "
                   "hips -- so unlike every earlier chart this one can see where the "
                   "character actually is. Lower is better. This is agreement with another "
                   "instrument, not accuracy: neither side has ground truth. The arms "
                   "improve a lot and the legs only a little, which is what was written "
                   "down before the numbers existed, because only the sideways part of the "
                   "hip error leaves.",
             better="lower",
             bars=[dict(label="Performer 1, arms, before", role="control",
                        key="d2b_rung11_arms_d2_subject_00"),
                   dict(label="Performer 1, arms, after", role="mamma",
                        key="d2b_rung11_arms_after_subject_00"),
                   dict(label="Performer 1, legs, before", role="control",
                        key="d2b_rung11_legs_d2_subject_00"),
                   dict(label="Performer 1, legs, after", role="mamma",
                        key="d2b_rung11_legs_after_subject_00"),
                   dict(label="Performer 2, arms, before", role="control",
                        key="d2b_rung11_arms_d2_subject_01"),
                   dict(label="Performer 2, arms, after", role="mamma",
                        key="d2b_rung11_arms_after_subject_01"),
                   dict(label="Performer 2, legs, before", role="control",
                        key="d2b_rung11_legs_d2_subject_01"),
                   dict(label="Performer 2, legs, after", role="mamma",
                        key="d2b_rung11_legs_after_subject_01")]),
    ],
}
