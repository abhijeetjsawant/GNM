#!/usr/bin/env python3
"""Ladder extractor for D3 -- one rest skeleton per performer, carried end to end.

A STUB, deliberately: `tools/compare/ladder.py` owns the `RUNGS` registry (one owner, or
the registrations collide -- LADDER_EXECUTION_PLAN section 2). This file supplies the
`x_*`-shaped function and the proposed `VISUALS` entries. To register it, add to
`ladder.py`:

    from extractors.d3_skeleton import x_skeleton

and route its figures by what each REFERENCES, exactly as `_d2_figures` does: the
`rung11_` keys to rung 11 (delivered), the `silhouette_` keys to rung 1 (masks), the
`closure_` / `oracle_` / `rest_` keys to rung 6 (shape -- the body the delivery is built
on), and the `converter_` keys to rung 7. One extractor call, four destinations.

It reads exactly one report, `artifacts/compare/d3-skeleton/gate.json`.

FOUR REFERENCES, NEVER ONE AXIS, and the `reference` strings differ verbatim:

  * CLOSURE -- the track's OWN forward kinematics: the delivered GLB, parsed and
    forward-kinematicked node by node, against `forward_kinematics_positions` on the rest
    the track carries. Metres of disagreement between two halves of one delivery.
  * SYNTHETIC TRUTH -- a body we built with independently perturbed per-bone scale
    channels, recovered through the converter, the real exporter and the real asset.
  * our OWN CAPTURE, root-relative -- the delivered rig against the landmarks it was
    solved from.
  * MAMMA `pred_joints` -- agreement with an instrument, never accuracy.
  * MAMMA's SAM2 masks -- pixels of the footage, the one reference that is not
    model-mediated.

THE FINDING THIS RUNG CLOSES. Before D3 the delivery carried TWO skeletons: the converter
and every joint instrument used the code rig, the exporter wrote the MPFB asset's own rest,
and the delivered GLB's joints sat 81-195 mm (median 131) from the forward kinematics every
joint instrument scored. The closure band is the proof that there is now one body: the GLB
reproduces the track's FK to 5e-7 m, and the pre-D3 exporter on the same track misses by
93-96 mm median. The silhouette against the photographs -- the instrument that FELL 8 of 8
under D2b -- ROSE 8 of 8 here, 0.521 -> 0.627, with the MAMMA oracle bit-identical, which
is the two-skeleton explanation for D2b's fall confirmed by the photographs themselves.

Self-check:  python3 tools/compare/extractors/d3_skeleton.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ladder import HIGHER, LOWER, _load, fig  # noqa: E402

REPORT = "artifacts/compare/d3-skeleton/gate.json"
REGEN = "PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d3_skeleton_gate.py"

REF_CLOSURE = ("the track's OWN forward kinematics on the rest it carries -- the delivered GLB "
               "parsed and forward-kinematicked against it. Two halves of ONE delivery")
REF_SYNTH = ("SYNTHETIC TRUTH: a body built with independently perturbed per-bone scale channels, "
             "posed through our own FK, recovered through the converter, the real exporter and "
             "the real MPFB asset")
REF_OURS = ("our own triangulated capture, root-relative (hip midpoint) -- the converter scored "
            "against its OWN input, ONE solve, the delivered configuration")
REF_MAMMA = ("MAMMA `pred_joints`, the scoreboard's own statistic (per-joint median over frames, "
             "then median over joints). Agreement with an instrument, not accuracy")
REF_MASKS = "MAMMA's SAM2 person masks -- pixels of the footage, not model-mediated"
REF_REST = "limb lengths measured from OUR capture (estimate_limb_lengths_m)"


def _subject_label(key: str) -> str:
    return key.replace("subject_", "subject ")


def x_skeleton(_: dict) -> tuple[list, list]:
    r = _load(REPORT)
    if not r:
        return [], []
    figs: list = []
    ctrls: list = []
    closure = r.get("closure", {}).get("subjects", {})
    for s, block in closure.items():
        t = _subject_label(s)
        figs.append(fig(f"D3 closure: delivered GLB vs the track's own FK, max over frames and joints, {t}",
                        block.get("max_error_m"), "m", REF_CLOSURE, key=f"closure_max_{s}",
                        note=f"band {r['closure'].get('band_m')} m"))
        ctrls.append(fig(f"D3 must-fail: the PRE-D3 exporter on the same track, median, {t}",
                         block.get("mustfail_pre_d3_exporter", {}).get("median_error_mm"), "mm", REF_CLOSURE, LOWER,
                         key=f"closure_ctrl_pre_d3_{s}", note="must fail the band, and does"))
    oracle = r.get("exact_skeleton_oracle", {})
    figs.append(fig("D3 exact-skeleton oracle: worst arms over 6 perturbed bodies (GLB vs posed truth)",
                    oracle.get("worst_arms_mm"), "mm", REF_SYNTH, key="oracle_worst_arms",
                    note=f"band {oracle.get('bands', {}).get('arms_mm')} mm; FAIL as pre-registered "
                         "-- D2's clavicle residual on a longer lever (0.43 at 0.78x span, 0.69 at 1.26x)"))
    figs.append(fig("D3 exact-skeleton oracle: worst legs over 6 perturbed bodies",
                    oracle.get("worst_legs_mm"), "mm", REF_SYNTH, key="oracle_worst_legs"))
    mf = oracle.get("mustfail_worst_mm", {})
    ctrls.append(fig("D3 must-fail: the same body arriving CANONICAL downstream, best arms over seeds",
                     mf.get("arms"), "mm", REF_SYNTH, LOWER, key="oracle_ctrl_canonical_arms"))
    ctrls.append(fig("D3 must-fail: the same body arriving CANONICAL downstream, best legs over seeds",
                     mf.get("legs"), "mm", REF_SYNTH, LOWER, key="oracle_ctrl_canonical_legs"))
    delivered = r.get("delivered", {}).get("subjects", {})
    for s, block in delivered.items():
        t = _subject_label(s)
        b = block.get("delivered_on_our_capture_before_mm", {})
        a = block.get("delivered_on_our_capture_after_mm", {})
        for group in ("arms", "legs", "torso"):
            figs.append(fig(f"D3 delivered rig vs our capture, {group}, before, {t}", b.get(group), "mm",
                            REF_OURS, key=f"converter_{group}_before_{s}"))
            figs.append(fig(f"D3 delivered rig vs our capture, {group}, after, {t}", a.get(group), "mm",
                            REF_OURS, key=f"converter_{group}_after_{s}"))
        spans_b = block.get("rest_spans_before_m", {})
        spans_a = block.get("rest_spans_after_m", {})
        for span in ("shoulder_span_m", "hip_span_m", "thigh_m", "shin_m"):
            figs.append(fig(f"D3 rest {span.replace('_m', '').replace('_', ' ')}, canonical rig, {t}",
                            spans_b.get(span), "m", REF_REST, key=f"rest_{span}_canonical_{s}"))
            figs.append(fig(f"D3 rest {span.replace('_m', '').replace('_', ' ')}, delivered (own), {t}",
                            spans_a.get(span), "m", REF_REST, key=f"rest_{span}_delivered_{s}"))
    hoist = r.get("ground_projection", {}).get("subjects", {})
    for s, block in hoist.items():
        t = _subject_label(s)
        figs.append(fig(f"D3 ground hoist, median, before, {t}", block.get("before", {}).get("hoist_median_mm"),
                        "mm", "the projection's own floor estimate", key=f"converter_hoist_before_{s}"))
        figs.append(fig(f"D3 ground hoist, median, after, {t}", block.get("after", {}).get("hoist_median_mm"),
                        "mm", "the projection's own floor estimate", key=f"converter_hoist_after_{s}"))
    ext = r.get("external_reports", {})
    for when in ("before", "after"):
        head = ext.get(f"rung11_{when}", {}).get("headline", {})
        for s in ("subject_00", "subject_01"):
            t = _subject_label(s)
            figs.append(fig(f"D3 rung 11, delivered body vs MAMMA, all 15 joints, {when}, {t}",
                            head.get(f"{s}/median_mm/sized"), "mm median", REF_MAMMA,
                            key=f"rung11_delivered_{when}_{s}"))
            if when == "after":
                figs.append(fig(f"D3 rung 11, the same rotations REPLAYED on the canonical rig, {t}",
                                head.get(f"{s}/median_mm/canon"), "mm median", REF_MAMMA,
                                key=f"rung11_canonical_replay_after_{s}"))
    sil = ext.get("silhouette_summary", {})
    for s, block in sil.get("per_subject", {}).items():
        t = _subject_label(s)
        figs.append(fig(f"D3 silhouette IoU vs SAM2 masks, before, {t}", block.get("before_median_iou"), "IoU",
                        REF_MASKS, HIGHER, key=f"silhouette_before_{s}"))
        figs.append(fig(f"D3 silhouette IoU vs SAM2 masks, after, {t}", block.get("after_median_iou"), "IoU",
                        REF_MASKS, HIGHER, key=f"silhouette_after_{s}"))
    if sil:
        ctrls.append(fig("D3 silhouette: MAMMA's own mesh through the same rasteriser (oracle), 8-cell median",
                         sil.get("oracle_median_iou"), "IoU", REF_MASKS, HIGHER, key="silhouette_oracle",
                         note="bit-identical between the before and after runs" if
                         sil.get("oracle_bit_identical_between_runs") else "NOT bit-identical -- runs not comparable"))
    return figs, ctrls


# Proposed VISUALS entries (for the registry owner). Bars on one chart share one reference.
PROPOSED_VISUALS = {
    "shape": [
        dict(title="D3: does the delivered character carry the body the joints were solved on?",
             plain="The reference is the track's own forward kinematics, so the right answer is zero. "
                   "The hatched bar is the exporter as it was before D3, on the same track: it put the "
                   "mesh's joints about nine centimetres from where every joint instrument scored them.",
             better="lower",
             bars=[dict(label="Delivered GLB vs its own FK, performer 0", role="ours", key="closure_max_subject_00"),
                   dict(label="Pre-D3 exporter, performer 0", role="control", key="closure_ctrl_pre_d3_subject_00"),
                   dict(label="Delivered GLB vs its own FK, performer 1", role="ours", key="closure_max_subject_01"),
                   dict(label="Pre-D3 exporter, performer 1", role="control", key="closure_ctrl_pre_d3_subject_01")]),
        dict(title="D3: a body we built, recovered exactly through the whole delivery",
             plain="Synthetic truth is the reference. Six bodies with every bone length, span and spine "
                   "segment drawn independently, posed, solved, exported and read back. The hatched bar "
                   "is the same body arriving canonical downstream, which is what shipped before D3.",
             better="lower",
             bars=[dict(label="Worst arms, 6 bodies", role="ours", key="oracle_worst_arms"),
                   dict(label="Worst legs, 6 bodies", role="ours", key="oracle_worst_legs"),
                   dict(label="Arriving canonical, arms", role="control", key="oracle_ctrl_canonical_arms"),
                   dict(label="Arriving canonical, legs", role="control", key="oracle_ctrl_canonical_legs")]),
        dict(title="D3: the rig's shoulder span, fixed rig against the performer's own",
             plain="Limb lengths measured from our own capture are the reference. The fixed rig was 540 mm "
                   "across the shoulders; the delivered body is now each performer's own width.",
             better="lower",
             bars=[dict(label="Fixed rig, performer 0", role="alt", key="rest_shoulder_span_m_canonical_subject_00"),
                   dict(label="Delivered body, performer 0", role="ours", key="rest_shoulder_span_m_delivered_subject_00"),
                   dict(label="Fixed rig, performer 1", role="alt", key="rest_shoulder_span_m_canonical_subject_01"),
                   dict(label="Delivered body, performer 1", role="ours", key="rest_shoulder_span_m_delivered_subject_01")]),
    ],
    "pose": [
        dict(title="D3: where the delivered rig lands on our own capture, before and after",
             plain="Our own capture is the reference, root-relative. Before: the fixed rig. After: the "
                   "rig sized to the performer, which is now what ships.",
             better="lower",
             bars=[dict(label="Arms before, performer 0", role="alt", key="converter_arms_before_subject_00"),
                   dict(label="Arms after, performer 0", role="ours", key="converter_arms_after_subject_00"),
                   dict(label="Legs before, performer 0", role="alt", key="converter_legs_before_subject_00"),
                   dict(label="Legs after, performer 0", role="ours", key="converter_legs_after_subject_00"),
                   dict(label="Arms before, performer 1", role="alt", key="converter_arms_before_subject_01"),
                   dict(label="Arms after, performer 1", role="ours", key="converter_arms_after_subject_01"),
                   dict(label="Legs before, performer 1", role="alt", key="converter_legs_before_subject_01"),
                   dict(label="Legs after, performer 1", role="ours", key="converter_legs_after_subject_01")]),
    ],
    "delivered": [
        dict(title="D3: the delivered character against MAMMA's joints, before and after",
             plain="MAMMA's joints are the reference, at zero; agreement with an instrument, not accuracy. "
                   "Before: the fixed rig. After: the performer's own body. The aqua bar replays the new "
                   "rotations on the old fixed rig, which is what a consumer that ignored the skeleton would show.",
             better="lower",
             bars=[dict(label="Before, performer 0", role="alt", key="rung11_delivered_before_subject_00"),
                   dict(label="After, performer 0", role="ours", key="rung11_delivered_after_subject_00"),
                   dict(label="Replayed on the fixed rig, 0", role="alt", key="rung11_canonical_replay_after_subject_00"),
                   dict(label="Before, performer 1", role="alt", key="rung11_delivered_before_subject_01"),
                   dict(label="After, performer 1", role="ours", key="rung11_delivered_after_subject_01"),
                   dict(label="Replayed on the fixed rig, 1", role="alt", key="rung11_canonical_replay_after_subject_01")]),
    ],
    "masks": [
        dict(title="D3: how well the delivered outline covers the person in the footage, before and after",
             plain="MAMMA's person masks are the reference; higher is better. This is the instrument that "
                   "FELL under D2b. It rose in all eight cells once the mesh carried the same skeleton the "
                   "joints were solved on. The hatched bar is MAMMA's own mesh through the same rasteriser.",
             better="higher",
             bars=[dict(label="Before, performer 0", role="alt", key="silhouette_before_subject_00"),
                   dict(label="After, performer 0", role="ours", key="silhouette_after_subject_00"),
                   dict(label="Before, performer 1", role="alt", key="silhouette_before_subject_01"),
                   dict(label="After, performer 1", role="ours", key="silhouette_after_subject_01"),
                   dict(label="MAMMA's mesh (oracle)", role="control", key="silhouette_oracle")]),
    ],
}


if __name__ == "__main__":
    figs, ctrls = x_skeleton({})
    for f in figs + ctrls:
        print(f"{f['key']:48s} {f.get('value')!s:>12} {f.get('unit', '')}")
    print(f"{len(figs)} figures, {len(ctrls)} controls")
