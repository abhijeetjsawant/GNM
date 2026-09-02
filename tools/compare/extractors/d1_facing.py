#!/usr/bin/env python3
"""Ladder extractor for D1 (locate) -- where the facing defect lives.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN §2). This file supplies the `x_*`-shaped
function and the proposed `VISUALS` entry, and is imported by the instrument's tests. To
register it, add to `ladder.py`:

    from extractors.d1_facing import x_facing

and give the `delivered` rung (n=11) `extract=x_facing`, `reports=[REPORT]`.

AXIS DISCIPLINE. Every figure here is a DOT PRODUCT of two unit directions, in [-1, +1].
That is its own axis and it shares it with nothing else on the page -- not with I6's IoU,
not with I1's millimetres. The two references (our own triangulated capture, and MAMMA's
`pred_joints`) are reported as separate figures and never averaged together, because they
are two independent estimates and neither is truth.

THE HANDEDNESS FIGURES ARE SIGNS, not scores. They are carried as `COUNT` so nothing tries
to rank them: -1 is what every real human on this fixture measures, +1 is the mirrored
convention, and the whole finding is that one delivered arm reads each.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = "artifacts/compare/facing-location.json"

sys.path.insert(0, str(ROOT / "tools" / "compare"))
try:  # pragma: no cover - exercised by whichever import path is available
    from ladder import COUNT, HIGHER, LOWER, fig  # type: ignore
except Exception:  # pragma: no cover
    LOWER = "lower is better"
    HIGHER = "higher is better"
    COUNT = "exposure count, not a score"

    def fig(label, value, unit, reference, better=LOWER, key=None, note=""):
        return {"key": key or label, "label": label, "value": value, "unit": unit,
                "reference": reference, "better": better, "note": note}


REF_CAPTURE = ("our own triangulated capture's forward direction, unit(left x up) from the "
               "hips and neck -- an estimate, not truth")
REF_MAMMA = ("MAMMA `pred_joints` forward direction, built the same way from its own hips "
             "and neck -- a second independent estimate, never a target")
SUBJECTS = ("subject_00", "subject_01")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def x_facing(_: dict) -> tuple[list, list]:
    path = ROOT / REPORT
    if not path.exists():
        return [], []
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return [], []

    dots = report.get("forward_dot", {})
    triples = report.get("triple_product", {}).get("arms", {})
    figs: list[dict] = []
    ctrls: list[dict] = []

    def dot(group: str, reference: str) -> float | None:
        return _mean([
            dots[s][group][reference]["median"] for s in SUBJECTS
            if group in dots.get(s, {}) and reference in dots[s][group]
        ])

    def sign(arm: str) -> float | None:
        return _mean([
            triples[arm][s]["sign_median"] for s in SUBJECTS
            if arm in triples and s in triples[arm]
        ])

    # --- the headline: which way each part of the delivered character points ------------
    # Reported per part, never pooled: the whole finding is that the parts DISAGREE, and a
    # mean over them would hide exactly the thing this instrument exists to show.
    parts = (
        ("delivered_torso_Hips", "which way the delivered PELVIS points"),
        ("delivered_torso_Chest", "which way the delivered CHEST points"),
        ("delivered_Head", "which way the delivered HEAD points"),
        ("delivered_MESH_nose", "which way the delivered MESH's NOSE points -- measured on "
                                "the shipped GLB through the real skinning, with the vertex "
                                "sets picked from the bind pose by geometry, so no joint "
                                "name enters this number"),
        ("delivered_feet", "which way the delivered FEET point -- solved from the "
                           "triangulated ToeBase since f6a4973, and the one part that is "
                           "NOT carried by the yawed torso frame"),
    )
    for key, label in parts:
        figs.append(fig(f"{label} (+1 = with the performer, -1 = exactly opposed)",
                        dot(key, "vs_our_capture_forward"), "direction cosine", REF_CAPTURE,
                        HIGHER, key=f"fwd_{key}_capture"))
        if dot(key, "vs_mamma_forward") is not None:
            figs.append(fig(f"{label}, against MAMMA's independent answer",
                            dot(key, "vs_mamma_forward"), "direction cosine", REF_MAMMA,
                            HIGHER, key=f"fwd_{key}_mamma"))

    figs.append(fig("the delivered MESH's +X side against the performer's LEFT -- negative "
                    "means the mesh's left arm is on the performer's right. Negative here "
                    "AND at the nose is a YAW; only one of the two would be a reflection",
                    dot("delivered_MESH_plus_x_side_vs_performer_LEFT", "vs_our_capture_left"),
                    "direction cosine", REF_CAPTURE, HIGHER, key="fwd_mesh_side"))

    # --- handedness: the one thing no silhouette and no by-name score can see -----------
    for arm, label in (
        ("our_triangulated_capture", "our capture"),
        ("mamma_pred_joints", "MAMMA's joints"),
        ("delivered_rig_forward_from_its_FEET", "the delivered rig, forward read from its FEET"),
        ("delivered_rig_forward_from_its_TORSO", "THE SAME delivered rig, forward read from "
                                                 "its TORSO -- this arm and the one above "
                                                 "disagreeing IS the defect, stated as one bit"),
        ("delivered_MESH_surface", "the delivered mesh's own skin"),
    ):
        figs.append(fig(f"handedness sign of {label} (-1 on every real human here; +1 is "
                        f"the mirrored convention)", sign(arm), "sign", REF_CAPTURE, COUNT,
                        key=f"hand_{arm}"))

    rest = report.get("rest_convention", {})
    for asset in ("asset_mpfb_neutral_body", "code_DETAILED_HUMANOID"):
        figs.append(fig(f"handedness sign of the REST skeleton in `{asset}` -- no capture, "
                        f"no reference, no camera: the mirror read straight off the asset",
                        rest.get(asset, {}).get("handedness_triple_product_sign"), "sign",
                        "the asset file itself", COUNT, key=f"hand_rest_{asset}"))

    # --- the oracle, and the controls ---------------------------------------------------
    ctrls.append(fig("ORACLE: MAMMA's forward against our capture's forward -- two "
                     "independent estimates of which way the performers face. This is the "
                     "ceiling, and a delivered part sitting near -1 against it is not a "
                     "disagreement about degrees, it is a reversal",
                     dot("ORACLE_mamma_forward_vs_our_capture_forward", "vs_our_capture_forward"),
                     "direction cosine", REF_CAPTURE, HIGHER, key="oracle_forward"))

    for arm, label, better in (
        ("CONTROL_yaw180_about_the_subjects_own_up",
         "control: our capture turned 180 degrees about each subject's own vertical -- a "
         "PROPER rotation, so it must keep the handedness sign and fail the forward-dot",
         LOWER),
        ("CONTROL_yaw90_about_the_subjects_own_up",
         "control: our capture turned 90 degrees -- must fail the forward-dot", LOWER),
        ("CONTROL_sagittal_mirror",
         "control: our capture mirrored in each subject's own sagittal plane -- left and "
         "right exchanged, facing untouched. It PASSES the forward-dot by construction and "
         "its silhouette is unchanged, so only the handedness sign rejects it. This is why "
         "the gate needs both bands, and why the D1 gate card's 'a mirrored human must fail "
         "the forward-dot' cannot be met", HIGHER),
    ):
        entry = triples.get(arm, {})
        ctrls.append(fig(f"{label} [forward-dot]",
                         _mean([entry[s]["forward_dot_vs_capture_median"] for s in SUBJECTS
                                if s in entry and "forward_dot_vs_capture_median" in entry[s]]),
                         "direction cosine", REF_CAPTURE, better, key=f"{arm}_fwd"))
        ctrls.append(fig(f"{label} [handedness sign]", sign(arm), "sign", REF_CAPTURE, COUNT,
                         key=f"{arm}_hand"))
    return figs, ctrls


# ---------------------------------------------------------------------------------------
# Proposed `VISUALS` entry for the `delivered` rung (n=11). Fable owns `ladder.py`; this is
# the entry to paste, not a registration. Roles use the validated palette: ours blue, mamma
# orange, alt aqua, control hatched.
#
# One chart, not two: the finding is a DISAGREEMENT BETWEEN PARTS OF ONE CHARACTER, so the
# parts belong on one axis where the reader can see the pelvis at -1 and the feet at +0.9
# side by side. Every bar is a direction cosine against the same reference.
PROPOSED_VISUALS = {
    "delivered": [
        dict(title="Which way the delivered character faces, part by part",
             plain="+1 means a part points the same way the performer does in the footage; "
                   "-1 means it points exactly the opposite way. The pelvis, chest, head "
                   "and the mesh's own nose are all reversed; the feet, which are solved "
                   "separately from the toes, are right. MAMMA's independent reading of the "
                   "performers agrees with ours at +0.99, so this is not two instruments "
                   "disagreeing about a few degrees -- the body is turned round.",
             better="higher",
             bars=[dict(label="Delivered pelvis", role="ours", key="fwd_delivered_torso_Hips_capture"),
                   dict(label="Delivered chest", role="ours", key="fwd_delivered_torso_Chest_capture"),
                   dict(label="Delivered head", role="ours", key="fwd_delivered_Head_capture"),
                   dict(label="The delivered mesh's nose", role="ours", key="fwd_delivered_MESH_nose_capture"),
                   dict(label="Delivered feet (solved from the toes)", role="alt", key="fwd_delivered_feet_capture"),
                   dict(label="MAMMA's reading of the performers", role="mamma", key="oracle_forward"),
                   dict(label="Our capture turned 180 degrees (the wrong answer)", role="control",
                        key="CONTROL_yaw180_about_the_subjects_own_up_fwd")]),
    ],
}
