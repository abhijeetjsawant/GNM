#!/usr/bin/env python3
"""THE SUBSTITUTION LADDER. MAMMA's pipeline, stage by stage, with ours beside each.

MAMMA is a stack of separable parts, every one of which leaves an artifact on disk:
calibration (`ma_cap`), person masks (`ma_masks`), dense 2D landmarks (`ma_2d`), and inside
`ma_3d` a triangulation, a cross-view re-ID, a per-subject shape and a multi-run pose fit
that ends in a mesh. Ours has a counterpart for each -- or a gap. This file is the
registry of those pairs and of the ONE instrument that scores each pair, and it does
three things every time it runs:

  1. reads every rung's figures from the JSON report its instrument wrote under
     `artifacts/` -- never from a document, never typed in -- and marks a rung whose
     report is missing as exactly that;
  2. appends a line to `docs/ladder-history.jsonl` (committed, unlike artifacts/) so a
     later run shows what improved and what got worse, per rung, against the last entry;
  3. renders `docs/substitution-ladder.html`, the living page.

THE METHOD, and it is a standing rule (docs/SUBSTITUTION_LADDER.md §1): replace ONE part
at a time. Every other stage in a rung is supplied by MAMMA's retained output, so that
when a number moves exactly one thing moved it. That makes every rung instrument-only:
a configuration containing any MAMMA stage never ships, and no shipped constant may be
fitted on one (CLAUDE.md).

Numbers from different references NEVER share an axis. Each figure below carries its
reference beside it, and the page has no column called "mm vs MAMMA".

    .venv/bin/python tools/compare/ladder.py            # aggregate + render
    .venv/bin/python tools/compare/ladder.py --note "..."  # force a history line
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
CMP = ART / "compare"
HISTORY = ROOT / "docs/ladder-history.jsonl"
PAGE = ROOT / "docs/substitution-ladder.html"
MAMMA_OUT = ART / "mamma/mamma-4cam-five-second-v2/output"
FIXTURE = "pushing_and_lifting_from_ground"

LOWER = "lower is better"
HIGHER = "higher is better"
COUNT = "exposure count, not a score"


def _load(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _stamp(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {"path": rel, "exists": False}
    return {"path": rel, "exists": True,
            "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}


def fig(label: str, value: float | int | None, unit: str, reference: str, better: str = LOWER,
        key: str | None = None, note: str = "") -> dict:
    return {"key": key or label, "label": label, "value": value, "unit": unit,
            "reference": reference, "better": better, "note": note}


# --------------------------------------------------------------------------- extractors
# Each returns (figures, controls) from on-disk reports, or ([], []) if the report is
# missing. Figures are the headline numbers the history tracks; controls are the arms
# that must fail (degenerate solutions) or must pass (oracles) for the rung to mean
# anything. A rung with figures and no controls is flagged on the page.

def x_swap_2d(_: dict) -> tuple[list, list]:
    r = _load("artifacts/compare/swap-2d-into-our-triangulation.json")
    if not r:
        return [], []
    ref = "MAMMA's exact 512 landmarks (verts_512 @ pred_vertices)"
    u = r["arms"].get("uniform (positions only)", {})
    v = r["arms"].get("MAMMA visibility", {})
    figs = [fig("our triangulation, MAMMA 2D, uniform confidence, subject 0", u.get("subject_00", {}).get("median_mm"),
                "mm median", ref, key="uniform_s0"),
            fig("same, subject 1", u.get("subject_01", {}).get("median_mm"), "mm median", ref, key="uniform_s1"),
            fig("coverage under uniform confidence", u.get("subject_00", {}).get("landmarks_triangulated_of_512"),
                "of 512 landmarks", "same population both arms", HIGHER, key="coverage_s0")]
    ctrls = [fig("visibility-as-confidence arm, subject 0 (survivorship: fewer landmarks)",
                 v.get("subject_00", {}).get("median_mm"), "mm median", ref, key="vis_s0",
                 note=f"on {v.get('subject_00', {}).get('landmarks_triangulated_of_512', '?'):.0f} landmarks -- "
                      "a different population; the 0.16 mm same-denominator value is in BATTLE1_COMPONENT_SWAP.md")]
    return figs, ctrls


def x_assoc(_: dict) -> tuple[list, list]:
    r = _load("artifacts/compare/association-swap.json")
    if not r:
        return [], []
    ref = "MAMMA's subject labels in ma_2d (truth-grade for this stage)"
    a = r["arms"]
    figs = [fig("identity switches, uniform confidence, no history", a.get("uniform conf", {}).get("switches"),
                f"of {r['frames'] * 2} slot-frames", ref, key="switches"),
            fig("switches with camera 0 shifted +1 frame", a.get("+1 frame", {}).get("switches"),
                f"of {r['frames'] * 2}", ref, key="switches_p1"),
            fig("switches with camera 0 shifted -1 frame", a.get("-1 frame", {}).get("switches"),
                f"of {r['frames'] * 2}", ref, key="switches_m1")]
    e = r["epipolar_margin_px_symmetric_native"]
    d = r["degenerate_check"]
    ctrls = [fig("degenerate check: frames with surfaces within 50 mm", d["contact_frames_within_50mm"],
                 f"of {r['frames']} frames", "MAMMA's fitted vertices", HIGHER, key="contact_frames",
                 note="zero switches is passable by a constant when the bodies are apart; this says they were not"),
             fig("epipolar margin in contact, p05 (swapped minus correct, must stay > 0)",
                 e["in_contact"]["p05"], "px symmetric at 3840", "same 19 landmarks the associator sees", HIGHER,
                 key="margin_contact_p05"),
             fig("camera pairs where the wrong pairing was cheaper", e["all"]["wrong_pairing_cheaper"],
                 f"of {e['all']['n']}", "same", LOWER, key="wrong_cheaper")]
    return figs, ctrls


def x_detector(_: dict) -> tuple[list, list]:
    r = _load("artifacts/commercial-multiview-soma77/run-report.json")
    if not r:
        return [], []
    figs = [fig("our detector's reprojection through our own solve, median",
                r.get("median_reprojection_error_px"), f"px at detector width {r.get('detector_width')}",
                "our own triangulation -- self-consistency, blind to coherent bias", key="reproj_p50"),
            fig("same, p95", r.get("p95_reprojection_error_px"), "px", "same", key="reproj_p95"),
            fig("frames deferred to exhaustive association", r.get("frames_deferred_to_exhaustive_association"),
                f"of {r.get('frame_count')}", "our pipeline's own diagnostic", COUNT, key="deferred")]
    return figs, []


def x_sequence(_: dict) -> tuple[list, list]:
    r = _load("artifacts/commercial-multiview-soma77/run-report.json")
    if not r:
        return [], []
    figs = [fig("joints filled by interpolation", r.get("interpolated_joint_fraction"), "fraction",
                "our pipeline's own diagnostic", COUNT, key="interp_frac"),
            fig("joints recovered by the sequence constraint", r.get("constraint_recovered_joint_fraction"),
                "fraction", "same", COUNT, key="constraint_frac",
                note="an exposure count: the stage fires only on single-ray slots and this fixture has ~20")]
    return figs, []


def x_shape(_: dict) -> tuple[list, list]:
    r = _load("artifacts/compare/smplx-shape-fit.json")
    if not r:
        return [], []
    ref = "limb lengths measured from OUR capture (estimate_limb_lengths_m)"
    figs, ctrls = [], []
    for s in ("subject_00", "subject_01"):
        d = r.get(s, {})
        e = d.get("mean_abs_error_mm", {})
        figs.append(fig(f"SMPL-X 10-beta fit, mean abs limb error, {s.replace('_', ' ')}",
                        e.get("smplx"), "mm", ref, key=f"smplx_{s}"))
        figs.append(fig(f"our canonical rig, same limbs, {s.replace('_', ' ')}",
                        e.get("our_rig"), "mm", ref, key=f"rig_{s}"))
        ctrls.append(fig(f"head above pelvis, {s.replace('_', ' ')} (the overfit that was caught read -624)",
                         d.get("head_above_pelvis_mm"), "mm", "the fitted body's own geometry", HIGHER,
                         key=f"head_above_{s}",
                         note="plausible" if d.get("anatomically_plausible") else "NOT PLAUSIBLE"))
    return figs, ctrls


def x_pose(_: dict) -> tuple[list, list]:
    sb = _load("artifacts/compare/scoreboard-commercial-multiview-soma77.json")
    pf = _load("artifacts/compare/smplx-pose-fit.json")
    if not sb or not pf:
        return [], []
    ref = "MAMMA's pred_joints, 15 body joints, 150 frames, subject map from pelvis agreement"
    figs, ctrls = [], []
    for s in ("subject_00", "subject_01"):
        m = sb["subjects"][s]["median_mm"]
        p = pf["subjects"][s]
        t = s.replace("_", " ")
        figs.append(fig(f"our raw triangulation, {t}", m["capture"], "mm median", ref, key=f"capture_{s}"))
        figs.append(fig(f"our capture driving SMPL-X (rung 2), {t}", p["median_vs_mamma_mm"], "mm median", ref,
                        key=f"smplx_{s}"))
        ctrls.append(fig(f"rung 2 against OUR OWN capture, {t} (must stay small: it follows our data, not MAMMA's)",
                         p["median_vs_our_capture_mm"], "mm median", "our triangulated joints", LOWER,
                         key=f"smplx_vs_ours_{s}"))
    return figs, ctrls


def x_sequence_and_oracle(spec: dict) -> tuple[list, list]:
    """Rung 5: the production run's exposure counts plus I2's perfect-2D oracle
    (`tools/compare/extractors/i2_oracle.py`, wired by the registry owner)."""
    figs, ctrls = x_sequence(spec)
    sys.path.insert(0, str(ROOT / "tools/compare"))
    from extractors.i2_oracle import x_oracle_2d  # noqa: E402
    f2, c2 = x_oracle_2d(spec)
    return figs + f2, ctrls + c2


def x_feet_bar(spec: dict) -> tuple[list, list]:
    """Rung 10: I4's extractor, `tools/compare/extractors/i4_feet.py`, wired by the registry owner."""
    sys.path.insert(0, str(ROOT / "tools/compare"))
    from extractors.i4_feet import x_feet_bar as _x  # noqa: E402
    return _x(spec)


def x_pose_and_retarget(spec: dict) -> tuple[list, list]:
    """Rung 7 carries two instruments: the scoreboard / SMPL-X arms and I1's retarget split.

    I1's extractor lives in `tools/compare/extractors/i1_retarget.py` (the agent's stub,
    wired here by the registry owner). Both report against two references that never
    share an axis; each `fig` carries its own.
    """
    figs, ctrls = x_pose(spec)
    sys.path.insert(0, str(ROOT / "tools/compare"))
    from extractors.i1_retarget import x_retarget  # noqa: E402
    f2, c2 = x_retarget(spec)
    return figs + f2, ctrls + c2


def x_delivered(_: dict) -> tuple[list, list]:
    sb = _load("artifacts/compare/scoreboard-commercial-multiview-soma77.json")
    if not sb:
        return [], []
    ref = "MAMMA's pred_joints, 15 body joints, 150 frames -- the SAME reference as the pose rung"
    figs = []
    for s in ("subject_00", "subject_01"):
        m = sb["subjects"][s]["median_mm"]
        t = s.replace("_", " ")
        figs.append(fig(f"our rig as delivered (canonical proportions), {t}", m["canon"], "mm median", ref,
                        key=f"canon_{s}"))
        figs.append(fig(f"our rig sized to the performer, {t}", m["sized"], "mm median", ref, key=f"sized_{s}"))
    lim = sb["subjects"]["subject_00"]["measured_limbs_mm"]
    ctrls = [fig("shoulder span the delivered rig carries", lim.get("shoulder_span_canonical_mm"), "mm",
                 "against measured 346 / 363 and a full SMPL-X shape range topping out near 367", LOWER,
                 key="rig_shoulder_span")]
    return figs, ctrls


def x_head(_: dict) -> tuple[list, list]:
    r = _load("artifacts/head-lane/head-gate-shipped.json")
    if not r:
        return [], []
    ref = "MAMMA's head orientation in our thorax frame, gated frames (P1 median)"
    figs, ctrls = [], []
    for s in ("subject_00", "subject_01"):
        arms = r[s].get("gated_arms_verdict_not_binding", {})
        t = s.replace("_", " ")
        c = arms.get("candidate_multiview_fit", {})
        figs.append(fig(f"shipped head fit, {t}", c.get("P1_agreement_with_mamma_deg", {}).get("median"),
                        "deg median", ref, key=f"head_{s}", note=f"gate verdict {c.get('verdict')}"))
        o = arms.get("ORACLE_mamma_head_our_thorax", {})
        k = arms.get("C1_locked_head_constant", {})
        ctrls.append(fig(f"oracle: MAMMA's own head through our frames, {t} (the floor our frame definitions impose)",
                         o.get("P1_agreement_with_mamma_deg", {}).get("median"), "deg median", ref, LOWER,
                         key=f"oracle_{s}", note=f"verdict {o.get('verdict')}"))
        ctrls.append(fig(f"control: the locked-head constant, {t} (must fail)",
                         k.get("P1_agreement_with_mamma_deg", {}).get("median"), "deg median", ref, LOWER,
                         key=f"constant_{s}", note=f"verdict {k.get('verdict')}"))
    return figs, ctrls


def x_feet(_: dict) -> tuple[list, list]:
    r = _load("artifacts/feet-lane/delivered-foot.json")
    if not r:
        return [], []
    ref = "OUR triangulated Foot->ToeBase direction -- both sides ours; MAMMA's feet are unscored"
    figs, ctrls = [], []
    for s in ("subject_00", "subject_01"):
        for side in ("L", "R"):
            d = r[s][side]
            t = f"{s.replace('_', ' ')} {side}"
            # COUNT, not LOWER: since f6a4973 the delivered foot is solved FROM this direction, so a
            # drop here is the solver agreeing with its own input. No direction may render on it.
            figs.append(fig(f"delivered foot axis vs triangulated ToeBase direction (its own input since f6a4973), offset removed, {t}",
                            d["after_removing_constant_offset"]["median_deg"], "deg median", ref, COUNT,
                            key=f"foot_{s}_{side}", note="tautology after f6a4973; not a pass"))
            ctrls.append(fig(f"control: one fixed axis, {t} (delivered must beat it)",
                             d["constant_axis_control"]["median_deg"], "deg median", ref, LOWER,
                             key=f"foot_const_{s}_{side}"))
    return figs, ctrls


# ------------------------------------------------------------------------------ registry
# Order is MAMMA's own execution order: ma_cap, ma_masks, ma_2d, then the sub-stages of
# ma_3d in the order `fit()` runs them, then the mesh, then ma_vis.

RUNGS: list[dict[str, Any]] = [
    dict(
        id="calibration", n=0, title="Capture, calibration, sync, undistortion",
        mamma="`ma_cap`: cam_int / cam_ext per camera from the rig's calibration files, undistortion, "
              "frame window [60, 210). Emits `ma_cap/<take>/gt/<cam>.npz` and `global.npz`.",
        ours="Nothing owned. `camera-rig.json` is MAMMA's `ma_cap` calibration converted. No sync residual "
             "or distortion model has ever been measured on anything of ours.",
        interface="intrinsics (3x3), extrinsics (4x4 world->cam), image size, frame window",
        supplied_by_mamma="this stage, in EVERY rung below -- which is why every rung is instrument-only",
        instrument=None,
        instrument_missing="an owned multicam fixture: checkerboard calibration, a flash-band sync measurement, "
                           "a measured distortion model. Plan of record: BODY_LANE_PLAN.md §0.",
        reference="-", blind="everything: no instrument exists",
        status="not owned", extract=None, reports=[],
        both_directions="MAMMA->ours: done (we consume its rig). Ours->MAMMA: nothing to feed it.",
    ),
    dict(
        id="masks", n=1, title="Person masks and per-view identity",
        mamma="`ma_masks`: YOLO boxes + SAM2 tracklets per camera, a CLIP feature bank and an epipolar anchor "
              "to give the same person the same id across views. Emits masks per camera per frame.",
        ours="None. SOMA-77 detects people top-down with its own boxes; identity is decided later, on "
             "keypoints, by our cross-view association. We consume no masks.",
        interface="per-camera per-frame person id + mask",
        supplied_by_mamma="ma_2d arrives pre-split by subject, so the masks' identity work is inside every "
                          "rung that consumes ma_2d",
        instrument=None,
        instrument_missing="mask IoU per view against `ma_masks`, and per-view id stability -- buildable from "
                           "retained files, deliberately not started: nothing of ours consumes masks.",
        reference="-", blind="-", status="not built, deliberately", extract=None, reports=[],
        both_directions="MAMMA->ours: implicit (pre-split ma_2d). Ours->MAMMA: nothing to feed it.",
    ),
    dict(
        id="detector", n=2, regenerate="python -m autoanim_gnm ... (the production build writes run-report.json; see docs/BODY_LANE_PLAN.md)",
        title="2D landmarks",
        mamma="`ma_2d`: MammaNet, 512 dense surface landmarks per person with a log-sigma channel, a "
              "visibility channel and contact / floor-contact channels, at native 3840x2160, per subject.",
        ours="SOMA-77 via NVIDIA GEM-X: 77 skeletal joints at 1280 width, of which the adapter maps 17 body "
             "joints, 5 head landmarks and the toe joints. Confidence in [0.895, 0.992] whether or not occluded.",
        interface="per camera, per frame, per subject: (landmark, x, y, confidence)",
        supplied_by_mamma="calibration; subject labels (pre-split)",
        instrument="`artifacts/commercial-multiview-soma77/run-report.json` (reprojection through our own solve) "
                   "-- a self-consistency figure only",
        instrument_missing="the cross-detector figures that discriminate (bulk ratio 2.2x, ~18.6 px coherent "
                           "bias) live in `tools/swap-harness/mamma_residuals.py` and `sam3d_ladder.py`, which "
                           "print and write no report. Give them JSON output and register them here.",
        reference="our own triangulation",
        blind="coherent per-camera bias -- passes through triangulation untouched; the tail; anything MAMMA's "
              "2D does differently, because this rung has no MAMMA arm on disk yet",
        status="measured, no discriminating instrument on disk", extract=x_detector,
        reports=["artifacts/commercial-multiview-soma77/run-report.json"],
        both_directions="MAMMA->ours: done, see the triangulation rung. Ours->MAMMA: OPEN. MAMMA's fitter consumes "
                        "its own 512 landmarks; `use_sparse_body_ldmks` selects a subset of THOSE (`self.body_idx`), "
                        "not arbitrary joints. Feeding our 17 joints needs a mapping onto SMPL-X joint targets "
                        "plus a Modal run of `run_ma_3d.py`. Not built.",
    ),
    dict(
        id="association", n=3, regenerate="python3 tools/swap-harness/association_swap.py   # SYSTEM python3: verts_512.pkl needs torch",
        title="Cross-view association / re-ID",
        mamma="Inside `ma_3d`: mvpose-style epipolar association with temporal grouping "
              "(`mvpose_style_associate_and_triangulate_temporal`) that re-assigns `pts2d` when a camera's "
              "label swapped. Runs once, before the fit.",
        ours="`associate_frame_graph`: cycle-consistent graph matching on 19-joint epipolar cost, with an "
             "exhaustive fallback on ambiguous frames.",
        interface="per frame: which detection in each camera is which subject",
        supplied_by_mamma="calibration; 2D landmarks (ma_2d, labels stripped and shuffled)",
        instrument="`tools/swap-harness/association_swap.py` -> `artifacts/compare/association-swap.json`",
        reference="MAMMA's own subject labels",
        blind="tested with a geometrically perfect, count-correct detector: the ambiguous and ghost-detection "
              "paths were never exercised. Says nothing about association on OUR 2D.",
        status="measured", extract=x_assoc, reports=["artifacts/compare/association-swap.json"],
        both_directions="MAMMA->ours: done. Ours->MAMMA: would need our detections in MAMMA's 512 format; not built.",
    ),
    dict(
        id="triangulation", n=4, regenerate="python3 tools/swap-harness/swap_true.py   # SYSTEM python3: verts_512.pkl needs torch",
        title="Triangulation",
        mamma="Inside `ma_3d`: `triangulate_batch`, DLT over all cameras per landmark, rows weighted by "
              "visibility, all 512 points. Emits `triangulated_3d_pts` (150, 512, 3).",
        ours="`triangulate_point`: DLT with sqrt-confidence row weighting, a confidence floor and an inlier gate.",
        interface="per frame per landmark: a world point or nothing",
        supplied_by_mamma="calibration; 2D landmarks; subject labels",
        instrument="`tools/swap-harness/swap_true.py` -> `artifacts/compare/swap-2d-into-our-triangulation.json`",
        reference="MAMMA's exact 512 landmark positions regressed from its fitted mesh",
        blind="anything under ~10 mm: the reference is a regularised fit that itself moves away from raw "
              "triangulation by that order. Compares geometry only; never exercises the sequence solve or "
              "the smoother.",
        status="measured, at the instrument's floor", extract=x_swap_2d,
        reports=["artifacts/compare/swap-2d-into-our-triangulation.json"],
        both_directions="MAMMA->ours: done. Ours->MAMMA: `triangulated_3d_pts` could be replaced by our points as "
                        "the fitter's 3D target, but only with 512-landmark input; not built.",
    ),
    dict(
        id="temporal", n=5, regenerate=".venv/bin/python tools/compare/oracle_2d.py   # ~3.5 min; run-report.json comes from the production build",
        title="Temporal solve and smoothing",
        mamma="No separate stage. All frames are optimised jointly, and runs 3 and 4 of the fit carry "
              "`angular_acc_loss` (0.003 / 0.005) and `pts3d_temp_loss` (0.03) -- verified in the retained "
              "`run_config.yaml`; runs 1-2 have every temporal term commented out.",
        ours="`solve_sequence_positions` (limb-length constraint on single-ray slots) then fill and "
             "Savitzky-Golay. Touches every joint, 5.6-7.9 mm median, and is the de facto outlier filter.",
        interface="per frame per joint: a world point, now on every frame",
        supplied_by_mamma="in the I2 oracle: its `pred_joints`, projected through the rig into all four cameras "
                          "as the 2D, so that association, triangulation and this stage run on a perfect detector",
        instrument="`tools/compare/oracle_2d.py` -> `artifacts/compare/oracle-2d.json` (I2, 2026-09-02): the whole "
                   "pipeline (`reconstruct_multiview`, wrapped, never re-implemented) on MAMMA's own skeleton "
                   "projected into the four cameras. Raw triangulation recovers the projected joints to 1e-8 mm, "
                   "so everything the oracle costs is THIS stage -- 1.15 mm median, largest on the wrists and ankles. "
                   "A noise arm injects MAMMA's own 2D residual deciles i.i.d. over 5 seeds.",
        instrument_missing="on REAL footage, still none valid: MAMMA's mesh is itself smoothed and its "
                           "`triangulated_3d_pts` are correlated with the 2D being smoothed. The oracle measures "
                           "the stage on a take with no dropped views, and its job is recovering joints a camera "
                           "loses; I7's synthetic fixture with injected single-ray slots is what scores that.",
        reference="MAMMA's pred_joints, projected and reconstructed -- exact by construction",
        blind="everything upstream of 2D: detector, calibration, distortion, sync, soft tissue and, above all, "
              "joint-definition error, which dominates every real-footage figure here. The noise arm is a lower "
              "bound twice over (MAMMA's residual is regularised toward its own landmarks, and it is injected "
              "i.i.d. while real detector error is correlated across joints, frames and cameras). No dropped "
              "views, so the sequence solve's actual job is never exercised. Head and toe solves not attempted.",
        status="measured on perfect 2D (1.15 mm, all of it this stage); dark on real footage",
        extract=x_sequence_and_oracle,
        reports=["artifacts/commercial-multiview-soma77/run-report.json", "artifacts/compare/oracle-2d.json"],
        both_directions="MAMMA->ours: done (I2) -- the oracle arm for rungs 3-7: association and triangulation pass at "
                        "0.00 on perfect 2D; a one-frame camera shift costs 6.7-6.8 mm, about what MAMMA-grade 2D "
                        "noise costs (4.6 mm), and a crossed subject pairing 1,243 mm. Ours->MAMMA: no reference.",
    ),
    dict(
        id="shape", n=6, regenerate=".venv/bin/python tools/compare/fit_smplx_to_capture.py",
        title="Body shape per performer",
        mamma="Inside `ma_3d`, run 2: 16 SMPL-X betas free with `body_shape_prior_loss`, per subject, jointly "
              "with pose. Emits `smplx_betas` (1, 16).",
        ours="Nothing in the delivery path: the rig is one fixed body (540 mm shoulders). Instruments: rung 1, "
             "`fit_smplx_to_capture.py`, 10 betas to our own measured limb lengths with the model's own "
             "standardised-beta prior; and an MHR + momentum prototype (`tools/fitter/`, 20.4 vs 26.1 mm, "
             "printed, no report).",
        interface="a per-subject body: bone lengths (and, for MAMMA, a surface)",
        supplied_by_mamma="the body MODEL (SMPL-X, research-licensed) -- not its data: the targets are our capture",
        instrument="`tools/compare/fit_smplx_to_capture.py` -> `artifacts/compare/smplx-shape-fit.json`",
        reference="limb lengths measured from our own capture",
        blind="surface entirely -- 11 limb targets say nothing about girth. And this is 10 betas with an L2 "
              "prior, not MAMMA's 16 with its prior: our SMPL-X instrument is not MAMMA's SMPL-X configuration.",
        status="measured (instrument), absent (delivery)", extract=x_shape,
        reports=["artifacts/compare/smplx-shape-fit.json"],
        both_directions="MAMMA->ours: its `smplx_betas` could size our rig -- pointless, the rig has no shape "
                        "space. Ours->MAMMA: our limb lengths into its fitter as a fixed shape -- not built.",
    ),
    dict(
        id="pose", n=7, regenerate=".venv/bin/python tools/compare/mamma_scoreboard.py && .venv/bin/python tools/compare/fit_smplx_pose.py",
        title="Pose recovery",
        mamma="Inside `ma_3d`: four Adam runs on Geman-McClure reprojection of 512 landmarks plus an L2 to the "
              "triangulated points (run 1), pose + translation free throughout. Emits `smplx_pose` (150, 165).",
        ours="`positions_to_body_track`: analytic, closed-form positions->rotations onto the rig. No optimiser. "
             "Instrument: rung 2, `fit_smplx_pose.py`, a least-squares fit of SMPL-X to our 15 triangulated "
             "joints with a pose prior on the unobserved DoF.",
        interface="per frame: a posed skeleton in the capture world",
        supplied_by_mamma="the body model only; detector, association, triangulation are ours",
        instrument="`tools/compare/mamma_scoreboard.py` + `tools/compare/fit_smplx_pose.py` -> "
                   "`artifacts/compare/scoreboard-*.json`, `smplx-pose-fit.json`",
        reference="MAMMA's pred_joints on 15 body joints, subject map from pelvis agreement",
        blind="accuracy: the reference is itself SMPL-X, so fitting SMPL-X to our points shares its joint "
              "regressor with the thing scored against -- some of a gain can be convention convergence. Hands, "
              "feet, head: not in the 15.",
        status="measured; converter split by I1 (2026-09-02)", extract=x_pose_and_retarget,
        reports=["artifacts/compare/scoreboard-commercial-multiview-soma77.json", "artifacts/compare/smplx-pose-fit.json",
                 "artifacts/compare/retarget-cost.json"],
        both_directions="MAMMA->ours (I1 arm B, built 2026-09-02): its `pred_joints` mapped through the scoreboard's "
                        "`PAIRS` into `positions_to_body_track` -- the 4 unmapped joints (eyes, ears) filled with the "
                        "nose position, proved inert (bit-identical rotations under a different filler); the converter "
                        "REJECTS NaN, so the plan's NaN adapter was wrong. On MAMMA's rigid joints the sized rig lands "
                        "at 52 / 51 mm on the arms and 3.4 mm on the legs; its own round-trip is 22 / 36 on the arms. "
                        "On our capture the sized rig lands at 85 / 92 (arms) and the round-trip floor is 30 / 36 -- "
                        "the gap above the floor is the input's bone-length wander (21 / 35 mm std across frames; "
                        "MAMMA's is 0.0) plus the landmark-to-joint-origin convention, and nothing on our capture can "
                        "split those two. Two plan premises fell: 're-solving' on a sized skeleton is bit-identical to "
                        "replaying canonical rotations (the converter turns rest DIRECTIONS), and the Hips 0.98 literal "
                        "is already gone. Ours->MAMMA: see detector rung.",
    ),
    dict(
        id="hands", n=8, title="Hands",
        mamma="Hand pose (45 + 45 axis-angle) inside the same SMPL-X fit, from the hand region of the 512 "
              "landmarks. Recovers fingers on this footage (BATTLE1_INCREMENT5_HAND_FIT_RESULT.md).",
        ours="Delivered: a constant rest curl per finger joint, because independent triangulation of SOMA-77's "
             "15 finger joints fails (226 % bone-length sd). Prototype: `hand_fit.py`, a constrained MHR "
             "hand-chain fit -- 35.0 mm held-out reprojection, 43-88 mm fingertip agreement with MAMMA, "
             "with an open thrashing defect (temporal term 1.9 % of the objective).",
        interface="per frame: 15 finger joints per hand",
        supplied_by_mamma="reference only",
        instrument=None,
        instrument_missing="the hand fit prints and writes arrays (`artifacts/handfit-arrays/`) but no report. "
                           "Its held-out-camera protocol is the right instrument; give it JSON output and register it.",
        reference="held-out camera (protocol); MAMMA fingertips (agreement)",
        blind="contact precision (an open hand and a fist differ by 60-80 mm, so grasp state clears the noise "
              "and touch does not)",
        status="prototype, figures in a document only", extract=None, reports=[],
        both_directions="MAMMA->ours: its hand landmarks into our chain fit -- would price the detector for "
                        "hands; not built.",
    ),
    dict(
        id="head", n=9, regenerate=".venv/bin/python tools/head/gate_the_shipped_head.py",
        title="Head orientation",
        mamma="A locked-head SMPL-X: jaw and eyes zero, head orientation is the body pose's neck/head joints "
              "driven by the 512 landmarks' head region.",
        ours="`_solve_head_for_subject`: rigid 5-landmark template (Head, HeadEnd, Jaw, both eyes) fitted over "
             "the take, anchored to the neck, temporal prior in the neck frame, 60 deg/frame physical reject.",
        interface="per frame: a head rotation relative to the thorax",
        supplied_by_mamma="reference only (its head through our thorax frame is the ORACLE arm)",
        instrument="`tools/head/gate_the_shipped_head.py` -> `artifacts/head-lane/head-gate-shipped.json`",
        reference="MAMMA's head orientation expressed in our thorax frame",
        blind="absolute orientation (a tracking gate mean-removes each take); accuracy (parity only); the "
              "jitter band, which the solver regularises directly",
        status="measured, one performer misses P1 p95 by 1.18 deg", extract=x_head,
        reports=["artifacts/head-lane/head-gate-shipped.json"],
        both_directions="MAMMA->ours: done (oracle arm). Ours->MAMMA: not meaningful; its head is locked.",
    ),
    dict(
        id="feet", n=10, regenerate=".venv/bin/python tools/feet/mamma_feet_bar.py",
        title="Feet, toes and ground contact",
        mamma="`ma_2d` emits contact and floor-contact probabilities per landmark; `ma_3d` carries them as "
              "`smplx_contact` / `smplx_floor_contact` and run 4 adds an intersection loss. Foot orientation "
              "comes from the fit's ankle and toe joints.",
        ours="Delivered foot orientation is solved from SOMA-77's `ToeBase` since 2026-09-01 (commit f6a4973, "
             "`commercial_multiview.py` ~1588); before that it was welded to the torso frame. `ToeEnd` fails "
             "(12-145 % length) and is not used, so the `Toes` channel rides the foot. Contact flags off by design.",
        interface="per frame: ankle + toe rotations, contact state",
        supplied_by_mamma="the reference only: its ankle (7/8) -> foot (10/11) direction in OUR shin frame; its own "
                          "foot through our frames is the oracle arm",
        instrument="`tools/feet/mamma_feet_bar.py` -> `artifacts/feet-lane/mamma-feet-bar.json` (I4, 2026-09-02). "
                   "Shin frame: origin ankle, knee->ankle, anterior from the cross with the pelvic axis. Offset and "
                   "spread reported separately; block bootstrap on identical draws, P(candidate beats control) per "
                   "foot. The earlier `delivered-foot.json` scored the solver against its own input and is retired.",
        reference="MAMMA's pred_joints foot direction in our shin frame -- parity, never truth",
        blind="accuracy entirely (both sides are estimates); inversion/eversion, which is roll about the foot's "
              "long axis and no ankle->foot DIRECTION carries it; toe articulation (SMPL-X's chain ends at the "
              "ball joint and our `Toes` channel is the identity); absolute orientation beyond the 90 deg gate; "
              "hip rotation leaking into ab/adduction through the pelvic reference (the oracle bounds it, it does "
              "not remove it); ground contact -- MAMMA's contact channel saturates near 0.25-0.35 planted or not, "
              "so no agreement figure exists and no threshold was chosen.",
        status="measured against MAMMA; a constant 17-24 deg inward toe offset on every foot", extract=x_feet_bar,
        reports=["artifacts/feet-lane/mamma-feet-bar.json"],
        both_directions="MAMMA->ours: done (I4) -- its foot direction is the bar and its foot through our shin "
                        "frames is the oracle (2.8-4.6 deg spread, so the frame definitions cost little). "
                        "Ours->MAMMA: not meaningful; its feet come out of a whole-body fit.",
    ),
    dict(
        id="delivered", n=11, regenerate=".venv/bin/python tools/compare/mamma_scoreboard.py",
        title="Mesh and delivered skeleton -- the end-to-end rung, deliberately last",
        mamma="`get_smplx_forward` -> `pred_vertices` (150, 10475, 3) and `pred_joints` (150, 127, 3); then "
              "`ma_vis` overlays. Research-licensed; never ships.",
        ours="Retarget onto the fixed rig, MPFB mesh, exported GLB/FBX. Carries the 180 deg facing defect "
             "(BODY_LANE_PLAN.md) until fixed.",
        interface="what a user receives",
        supplied_by_mamma="nothing -- this is our whole pipeline, scored against MAMMA's whole pipeline",
        instrument="`tools/compare/mamma_scoreboard.py` (canon / sized arms)",
        reference="MAMMA's pred_joints on 15 body joints -- the same reference as the pose rung, so those two "
                  "rungs and only those two may sit on one axis",
        blind="everything the per-stage rungs are blind to, compounded; and it cannot attribute -- that is what "
              "the rungs above are for",
        status="measured", extract=x_delivered,
        reports=["artifacts/compare/scoreboard-commercial-multiview-soma77.json"],
        both_directions="-",
    ),
]


# ---------------------------------------------------------------------------- history
def git_state() -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=ROOT, text=True).strip())
    except Exception:
        sha, dirty = "unknown", True
    return {"sha": sha, "dirty": dirty}


def headline(rungs: list[dict]) -> dict:
    out = {}
    for r in rungs:
        vals = {f["key"]: round(float(f["value"]), 2) for f in r["figures"] + r["controls"]
                if f["value"] is not None}
        if vals:
            out[r["id"]] = vals
    return out


def history_entries() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()]


def baseline(entries: list[dict], head: dict) -> dict | None:
    """The last entry whose headline DIFFERS from the current state.

    A re-render after nothing changed must still show the deltas of the last real
    change, or the page reports an improvement exactly once and then forgets it.
    """
    for e in reversed(entries):
        if e.get("headline") != head:
            return e
    return None


def deltas(rungs: list[dict], prev: dict | None) -> None:
    for r in rungs:
        p = (prev or {}).get("headline", {}).get(r["id"], {})
        for f in r["figures"] + r["controls"]:
            f["delta"] = None
            if f["value"] is None or f["key"] not in p or p[f["key"]] is None:
                continue
            d = round(float(f["value"]), 2) - round(float(p[f["key"]]), 2)
            if f["better"] == COUNT or abs(d) < 0.005:
                f["delta"] = {"value": d, "direction": "same"}
            else:
                good = d < 0 if f["better"] == LOWER else d > 0
                f["delta"] = {"value": d, "direction": "better" if good else "worse"}


# ----------------------------------------------------------------------------- render
CSS = """
:root{--bg:#F5F6F8;--card:#FFFFFF;--rule:#D9DEE5;--ink:#141A22;--muted:#5B6776;--faint:#8A96A6;
 --mamma:#7A4DD6;--ours:#1F7FD1;--good:#1E8E4C;--bad:#C93B3B;--warn:#B7791F;--dark:#3E4A5A;
 --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0B0F14;--card:#141A22;--rule:#273240;--ink:#E8EDF3;--muted:#98A5B6;--faint:#6B7889;--mamma:#B08CFF;--ours:#5FB3F5;--good:#4CC77A;--bad:#F06A6A;--warn:#E0A33A;--dark:#AAB6C5}}
:root[data-theme="dark"]{--bg:#0B0F14;--card:#141A22;--rule:#273240;--ink:#E8EDF3;--muted:#98A5B6;--faint:#6B7889;--mamma:#B08CFF;--ours:#5FB3F5;--good:#4CC77A;--bad:#F06A6A;--warn:#E0A33A;--dark:#AAB6C5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:76rem;margin:0 auto;padding:2rem 1.2rem 4rem}
.eyebrow{font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin:0 0 .4rem}
h1{font-size:clamp(1.5rem,3vw,2.1rem);line-height:1.1;margin:0 0 .5rem;letter-spacing:-.02em}
.lede{color:var(--muted);max-width:70ch;margin:0 0 1.2rem}
.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(16rem,1fr));gap:.7rem;margin:0 0 1.6rem}
.rule{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:.7rem .9rem;font-size:.86rem}
.rule b{display:block;margin-bottom:.15rem}
.meta{font-family:var(--mono);font-size:.72rem;color:var(--faint);margin:0 0 1.4rem}
.rung{background:var(--card);border:1px solid var(--rule);border-radius:8px;margin:0 0 1.1rem;overflow:hidden}
.rung header{display:flex;align-items:baseline;gap:.8rem;padding:.8rem 1rem;border-bottom:1px solid var(--rule)}
.rung header .n{font-family:var(--mono);font-size:.75rem;color:var(--faint)}
.rung header h2{font-size:1.05rem;margin:0;flex:1}
.status{font-family:var(--mono);font-size:.68rem;padding:.2rem .5rem;border-radius:3px;border:1px solid var(--rule);color:var(--muted);white-space:nowrap}
.status.measured{color:var(--good);border-color:var(--good)}.status.missing{color:var(--warn);border-color:var(--warn)}
.status.none{color:var(--bad);border-color:var(--bad)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
.col{padding:.8rem 1rem;font-size:.86rem}.col+.col{border-left:1px solid var(--rule)}
.col .who{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;margin:0 0 .3rem}
.col.m .who{color:var(--mamma)}.col.o .who{color:var(--ours)}
.col p{margin:0}
.band{padding:.7rem 1rem;border-top:1px solid var(--rule);font-size:.84rem}
.band .k{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);display:block;margin-bottom:.25rem}
.figs{width:100%;border-collapse:collapse;font-size:.84rem}
.figs td{padding:.3rem .4rem;border-top:1px solid var(--rule);vertical-align:top}
.figs td.v{font-family:var(--mono);white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}
.figs td.r{color:var(--muted);font-size:.78rem}
.figs tr.ctrl td{color:var(--muted)}
.d{font-family:var(--mono);font-size:.72rem;margin-left:.4rem}.d.better{color:var(--good)}.d.worse{color:var(--bad)}.d.same{color:var(--faint)}
.warn{color:var(--warn)}.miss{color:var(--warn);font-style:italic}
code{font-family:var(--mono);font-size:.92em;background:rgba(127,127,127,.12);padding:.05em .3em;border-radius:3px}
.prov{font-family:var(--mono);font-size:.7rem;color:var(--faint)}
@media (max-width:48rem){.cols{grid-template-columns:1fr}.col+.col{border-left:0;border-top:1px solid var(--rule)}}
footer{margin-top:2rem;color:var(--faint);font-size:.8rem;border-top:1px solid var(--rule);padding-top:1rem}
"""


def _md(s: str) -> str:
    """Escape, then turn `code` spans into <code>."""
    out = html.escape(s)
    parts = out.split("`")
    return "".join(p if i % 2 == 0 else f"<code>{p}</code>" for i, p in enumerate(parts))


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def _rows(items: list[dict], cls: str = "") -> str:
    rows = []
    for f in items:
        d = f.get("delta")
        dl = ""
        if d and d["direction"] != "same":
            dl = f'<span class="d {d["direction"]}">{d["value"]:+.2f} {html.escape(d["direction"])}</span>'
        elif d:
            dl = '<span class="d same">= last</span>'
        note = f' <span class="r">{_md(f["note"])}</span>' if f.get("note") else ""
        rows.append(f'<tr class="{cls}"><td>{_md(f["label"])}{note}</td>'
                    f'<td class="v">{_fmt(f["value"])} {html.escape(f["unit"])}{dl}</td>'
                    f'<td class="r">vs {_md(f["reference"])}</td></tr>')
    return "".join(rows)


def render(rungs: list[dict], state: dict, prev: dict | None, now: str) -> str:
    cards = []
    for r in rungs:
        st = r["status"]
        cls = "measured" if st.startswith("measured") else ("none" if st in ("not owned", "dark") else "missing")
        figs = r["figures"]
        ctrls = r["controls"]
        body = f'''
<article class="rung" id="{r["id"]}">
<header><span class="n">rung {r["n"]:02d}</span><h2>{_md(r["title"])}</h2><span class="status {cls}">{html.escape(st)}</span></header>
<div class="cols">
<div class="col m"><p class="who">MAMMA</p><p>{_md(r["mamma"])}</p></div>
<div class="col o"><p class="who">Ours</p><p>{_md(r["ours"])}</p></div>
</div>
<div class="band"><span class="k">Interface crossed</span>{_md(r["interface"])}
&nbsp;·&nbsp; <span class="k" style="display:inline">Supplied by MAMMA in this rung</span> {_md(r["supplied_by_mamma"])}</div>'''
        if r.get("instrument"):
            body += f'<div class="band"><span class="k">Instrument · reference</span>{_md(r["instrument"])}<br>reference: {_md(r["reference"])}</div>'
        if r.get("instrument_missing"):
            body += f'<div class="band"><span class="k">Instrument missing</span><span class="miss">{_md(r["instrument_missing"])}</span></div>'
        if figs or ctrls:
            body += '<div class="band"><span class="k">Figures — each with its own reference; no shared axis</span><table class="figs">'
            body += _rows(figs)
            if ctrls:
                body += '<tr><td colspan="3" class="r" style="padding-top:.6rem"><b>controls and oracles</b> — the arms that make the figures mean something</td></tr>'
                body += _rows(ctrls, "ctrl")
            else:
                body += '<tr><td colspan="3" class="warn">no control arm on disk — a figure without a failing degenerate solution is a claim, not a measurement</td></tr>'
            body += "</table></div>"
        elif r.get("reports"):
            body += '<div class="band"><span class="k">Figures</span><span class="miss">report missing on disk — ' + \
                    (f'run <code>{html.escape(r["regenerate"])}</code> to write ' if r.get("regenerate") else "the instrument must write ") + \
                    ", ".join(f"<code>{html.escape(p)}</code>" for p in r["reports"]) + "</span></div>"
        body += f'<div class="band"><span class="k">Blind to</span>{_md(r["blind"])}</div>'
        body += f'<div class="band"><span class="k">Substitution, both directions</span>{_md(r["both_directions"])}</div>'
        if r.get("reports"):
            prov = " · ".join(f'{html.escape(s["path"])} ({"missing" if not s["exists"] else s["mtime"]})' for s in r["provenance"])
            regen = f'<br>regenerate: <code>{html.escape(r["regenerate"])}</code>' if r.get("regenerate") else ""
            body += f'<div class="band prov">provenance: {prov}{regen}</div>'
        body += "</article>"
        cards.append(body)

    prev_line = (f'deltas against the last distinct state, {html.escape(prev.get("ts", "?"))} at '
                 f'{html.escape(prev.get("git", {}).get("sha", "?"))}'
                 if prev else "no earlier distinct state in the history — deltas appear after the first change")
    return f'''<meta charset="utf-8">
<title>Substitution Ladder</title>
<style>{CSS}</style>
<div class="wrap">
<p class="eyebrow">Body lane · MAMMA as the benchmark · one part at a time</p>
<h1>The Substitution Ladder</h1>
<p class="lede">MAMMA's pipeline divided into the stages its own code runs, ours beside each, and one instrument per pair. Every rung changes exactly one part and lets MAMMA's retained output supply the rest, so when a number moves, one thing moved it. Every configuration with a MAMMA part in it is an instrument and never ships.</p>
<div class="rules">
<div class="rule"><b>One part at a time.</b> End-to-end is the last rung, not the first. A number measured while two parts changed attributes to neither.</div>
<div class="rule"><b>No gate a constant can pass.</b> Every figure ships beside a control that must fail, or an oracle that must pass. A rung without one is flagged.</div>
<div class="rule"><b>Same denominator.</b> Both arms of a comparison are scored on the same population, and every figure names its reference. Nothing here is accuracy: neither side has ground truth.</div>
<div class="rule"><b>Regenerable or absent.</b> Figures come only from JSON reports under <code>artifacts/</code>. A number that lives in a document is listed as missing until its instrument writes a report.</div>
</div>
<p class="meta">rendered {html.escape(now)} · git {html.escape(state["sha"])}{" (dirty)" if state["dirty"] else ""} · fixture {html.escape(FIXTURE)}, 4 cameras, 150 frames, 2 performers · {prev_line} · history: <code>docs/ladder-history.jsonl</code> · plan: <code>docs/SUBSTITUTION_LADDER.md</code></p>
{"".join(cards)}
<footer>Generated by <code>tools/compare/ladder.py</code>. Edit the registry there, never this file. MAMMA and SMPL-X are research-licensed instruments; nothing they produce enters a delivered artifact, trained weights or a shipped constant.</footer>
</div>
'''


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default=None, help="record a history line even if nothing changed, with this note")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()

    now = dt.datetime.now().isoformat(timespec="seconds")
    state = git_state()
    rungs = []
    for spec in RUNGS:
        r = dict(spec)
        figs, ctrls = spec["extract"](spec) if spec["extract"] else ([], [])
        r["figures"], r["controls"] = figs, ctrls
        r["provenance"] = [_stamp(p) for p in spec.get("reports", [])]
        if spec["extract"] and not figs and spec.get("reports"):
            r["status"] = "report missing"
        rungs.append(r)

    head = headline(rungs)
    entries = history_entries()
    prev = baseline(entries, head)
    deltas(rungs, prev)

    record = {"ts": now, "git": state, "note": args.note, "headline": head}
    changed = not entries or entries[-1].get("headline") != head
    if not args.no_history and (changed or args.note):
        with HISTORY.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"history: appended ({'changed' if changed else 'note'}) -> {HISTORY.relative_to(ROOT)}")
    else:
        print("history: unchanged, no line appended")

    CMP.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in record.items()}
    out["rungs"] = [{k: v for k, v in r.items() if k != "extract"} for r in rungs]
    (CMP / "ladder.json").write_text(json.dumps(out, indent=2, default=str))
    PAGE.write_text(render(rungs, state, prev, now))
    print(f"wrote {CMP / 'ladder.json'} and {PAGE.relative_to(ROOT)}")

    # console summary
    print(f"\n{'rung':<14}{'status':<44}figures")
    for r in rungs:
        n = len(r["figures"])
        flag = "" if (n == 0 or r["controls"]) else "  NO CONTROL ARM"
        print(f"{r['id']:<14}{r['status']:<44}{n}{flag}")
        for f in r["figures"]:
            d = f.get("delta")
            ds = "" if not d or d["direction"] == "same" else f"   {d['value']:+.2f} {d['direction']}"
            print(f"    {f['label'][:70]:<70} {_fmt(f['value']):>8} {f['unit']}{ds}")


if __name__ == "__main__":
    main()
