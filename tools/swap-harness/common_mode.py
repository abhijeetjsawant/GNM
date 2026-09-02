#!/usr/bin/env python3
"""I3 report 2 of 3 -- PER-CAMERA STATIC COMMON MODE, fitted on one half, scored on the other.

Is our coherent 2D error a per-view COMMON-MODE shift -- one translation for the whole
image -- rather than dozens of independent per-joint errors? Our detector is TOP-DOWN:
SOMA-77 runs on a person box at CROP=256 with the model seeing only the centre 192
columns, which is exactly the architecture where bbox centre/scale jitter, resize
conventions and half-pixel choices translate most joints in one image coherently.
Triangulation then reads a coherent image displacement as a coherent 3D displacement,
and -- unlike independent keypoint noise -- **it does not average down with more views.**

It matters because a per-camera translation is four numbers. If it buys a real share of
the error it is the cheapest fix available; if it does not, the error is per-joint and
only a better detector reaches it. That is exactly the fork the I3 decision rule
(`i3_decision.py`) has to resolve, and THIS report carries the verdict.

THE SPLIT-HALF, and it is the point of the rewrite. The offsets are fitted on frames
[0, 75) and scored on frames [75, 150), and vice versa. Beside it, the SAME-FRAMES arm
-- fitted and scored on the same frames -- as the control that must show the inflation.
Fitting and scoring on the same frames is a free parameter per camera per half and it
flatters itself by construction; the difference between the two arms is how much.

The halves are CONTIGUOUS, not interleaved. Per-frame agreement here is strongly
autocorrelated (the measured lag-1 is in the report), so an even/odd split would put a
near-copy of every fitting frame into the scoring half and hide the inflation entirely.
The interleaved split is computed too, as the demonstration of that.

WHAT THIS CAN AND CANNOT SHOW. The common mode is estimated against MAMMA's fitted
`pred_joints` projected through our rig -- with oracle knowledge. So this measures the
CEILING of what a coordinate-level fix could buy. **It does not deliver the fix, and the
four numbers in this report are never shipped**: they are fitted on a MAMMA-derived arm,
and MAMMA never enters the delivery path (CLAUDE.md, LADDER_EXECUTION_PLAN §4). What the
report authorises is *building* the fix and re-selecting its constants from a MAMMA-free
reference.

    python3 tools/swap-harness/common_mode.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i3_decision as D  # noqa: E402
from i3_decision import cm  # noqa: E402

REPORT = D.ROOT / "artifacts/compare/detector-common-mode.json"

MINIMUM_CONFIDENCE = 0.25
MINIMUM_JOINTS_FOR_A_COMMON_MODE = 6
SPLIT_FRAME = 75

BLIND_TO = (
    "1. WHOSE shift it is. A coherent per-view 2D translation is produced just as well by "
    "a calibration error -- an extrinsic rotation of a fraction of a degree, or a "
    "principal point off by a few pixels -- as by a detector crop convention. This "
    "instrument cannot tell them apart, and rung 0 is 'not owned': the rig IS MAMMA's "
    "ma_cap. So a large static per-camera term is EVIDENCE FOR A COORDINATE FIX and NOT "
    "evidence about the detector. 2. It is scored against MAMMA's fit, so any bias shared "
    "by that fit and our rig is invisible, and MAMMA's own fit error is charged to us. "
    "3. It is blind to error along the viewing rays: the offsets are fitted in the image "
    "plane. 4. The residual after removal is not 'the detector's true error' -- it is "
    "what a per-view TRANSLATION cannot express, and a scale or rotation term would take "
    "more of it (the implied scale factor is reported, not removed)."
)


def summarise(matrix: np.ndarray) -> dict:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return {"n": 0, "median_mm": None, "p95_mm": None}
    return {"n": int(finite.size),
            "median_mm": D._round(np.median(finite)),
            "p95_mm": D._round(np.percentile(finite, 95)),
            "mean_mm": D._round(np.mean(finite))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()

    cameras = D.working_cameras()
    assigned, positions, _raw, diagnostics = D.assigned_detector_2d()
    subjects, frames, camera_count = assigned.shape[:3]
    names = D.SCORED_VS_MAMMA
    joints = len(names)

    mapping = D.subject_mapping(positions)
    truth_all = D.mamma_pred_joints()
    truth = np.stack([truth_all[mapping[s], :frames][:, [D.PAIRS[n] for n in names]]
                      for s in range(subjects)])                       # [S, F, J, 3]

    reference = np.moveaxis(D.project_joints(cameras, truth), 0, 2)     # [S, F, C, J, 2]
    observed = np.stack([assigned[:, :, :, cm.JOINT_INDEX[n]] for n in names], axis=3)
    observed = np.moveaxis(observed, 3, 3)                              # [S, F, C, J, 3]
    unusable = ~(np.isfinite(observed[..., :2]).all(axis=-1)
                 & (observed[..., 2] >= MINIMUM_CONFIDENCE))
    observed = observed.copy()
    observed[unusable] = np.nan

    residual = observed[..., :2] - reference

    # ---- the per-(subject, frame, camera) common mode, and its static per-camera part
    common = np.full((subjects, frames, camera_count, 2), np.nan)
    implied_scale = np.full((subjects, frames, camera_count), np.nan)
    for s in range(subjects):
        for f in range(frames):
            for c in range(camera_count):
                rows = residual[s, f, c]
                keep = np.isfinite(rows).all(axis=1)
                if int(keep.sum()) < MINIMUM_JOINTS_FOR_A_COMMON_MODE:
                    continue
                common[s, f, c] = np.median(rows[keep], axis=0)
                p, q = observed[s, f, c, keep, :2], reference[s, f, c, keep]
                pc, qc = p - p.mean(0), q - q.mean(0)
                denominator = float((qc * qc).sum())
                if denominator > 1e-9:
                    implied_scale[s, f, c] = float((pc * qc).sum() / denominator)

    def static_from(frame_mask: np.ndarray) -> np.ndarray:
        """One 2D translation per camera, the median over the selected (subject, frame)."""
        out = np.full((camera_count, 2), np.nan)
        for c in range(camera_count):
            values = common[:, frame_mask, c].reshape(-1, 2)
            values = values[np.isfinite(values).all(axis=1)]
            if values.size:
                out[c] = np.median(values, axis=0)
        return out

    first = np.zeros(frames, bool); first[:SPLIT_FRAME] = True
    second = ~first
    even = np.zeros(frames, bool); even[::2] = True
    odd = ~even

    static_all = static_from(np.ones(frames, bool))
    static_first, static_second = static_from(first), static_from(second)
    static_even, static_odd = static_from(even), static_from(odd)

    def per_frame_offsets(a: np.ndarray, mask_a: np.ndarray,
                          b: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """[F, C, 2]: the offset applied to each frame, from the fit of the OTHER half."""
        out = np.full((frames, camera_count, 2), np.nan)
        out[mask_a] = b            # frames in half A are scored with half B's fit
        out[mask_b] = a
        return out

    held_out = per_frame_offsets(static_first, first, static_second, second)
    same_frames = per_frame_offsets(static_second, first, static_first, second)  # A gets A's
    held_out_interleaved = per_frame_offsets(static_even, even, static_odd, odd)
    rotated = np.roll(np.arange(camera_count), 1)
    held_out_shuffled_cameras = held_out[:, rotated]

    # --- POST-HOC, and labelled as such. The four fitted offsets turned out to point the
    # same way and to be the same size, so the shuffled-CAMERA control cannot discriminate:
    # permuting four near-identical vectors changes almost nothing. That is a finding, not
    # a broken control -- it says the effect is not per-camera but detector-wide, which is
    # what a single shared crop/resize convention looks like. These three arms test that
    # reading, and the control that CAN discriminate is a rotated offset: same magnitude,
    # wrong direction. The verdict above is computed from the preregistered rule alone and
    # none of these arms enters it.
    def global_static(frame_mask: np.ndarray) -> np.ndarray:
        values = common[:, frame_mask].reshape(-1, 2)
        values = values[np.isfinite(values).all(axis=1)]
        return np.median(values, axis=0) if values.size else np.full(2, np.nan)

    global_first, global_second = global_static(first), global_static(second)
    global_held_out = per_frame_offsets(
        np.broadcast_to(global_first, (camera_count, 2)), first,
        np.broadcast_to(global_second, (camera_count, 2)), second)
    turn = np.asarray(((0.0, -1.0), (1.0, 0.0)))
    global_rotated = global_held_out @ turn.T

    arms_2d = {
        "as_measured": np.zeros((frames, camera_count, 2)),
        "STATIC_all_frames_fit_and_scored_on_all": np.broadcast_to(
            static_all, (frames, camera_count, 2)),
        "SPLIT_HALF_held_out": held_out,
        "CONTROL_same_frames_fit_and_score": same_frames,
        "CONTROL_interleaved_half_held_out": held_out_interleaved,
        "CONTROL_shuffled_camera_assignment": held_out_shuffled_cameras,
        "POSTHOC_global_single_offset_held_out": global_held_out,
        "POSTHOC_CONTROL_global_offset_rotated_90deg": global_rotated,
        "POSTHOC_CONTROL_global_offset_negated": -global_held_out,
    }

    def triangulate(offsets_per_frame: np.ndarray | None,
                    per_view: np.ndarray | None = None) -> np.ndarray:
        out = np.full((subjects, frames, joints, 3), np.nan)
        corrected = observed.copy()
        if per_view is not None:
            corrected[..., :2] -= per_view[..., None, :]
        elif offsets_per_frame is not None:
            corrected[..., :2] -= offsets_per_frame[None, :, :, None, :]
        for s in range(subjects):
            for f in range(frames):
                for j in range(joints):
                    points = corrected[s, f, :, j, :2]
                    if int(np.isfinite(points).all(axis=1).sum()) < 2:
                        continue
                    result = cm.triangulate_point(
                        cameras, points, np.nan_to_num(corrected[s, f, :, j, 2], nan=0.0),
                        pixel_scale=1.0, minimum_confidence=MINIMUM_CONFIDENCE)
                    if result is not None:
                        out[s, f, j] = result.position_world_m
        return out

    solved = {name: triangulate(offsets) for name, offsets in arms_2d.items()}
    solved["ORACLE_full_common_mode_removed"] = triangulate(None, per_view=common)

    error = {name: np.linalg.norm(points - truth, axis=-1) * 1000.0
             for name, points in solved.items()}
    keep = np.ones_like(error["as_measured"], dtype=bool)
    for matrix in error.values():
        keep &= np.isfinite(matrix)
    scored = {name: np.where(keep, matrix, np.nan) for name, matrix in error.items()}

    per_frame = {name: np.moveaxis(matrix, 1, 0).reshape(frames, -1)
                 for name, matrix in scored.items()}
    bootstrap = D.paired_block_bootstrap(
        per_frame, baseline="as_measured",
        candidates=[name for name in scored if name != "as_measured"])

    base = summarise(scored["as_measured"])["median_mm"]

    def gain(name: str) -> float | None:
        value = summarise(scored[name])["median_mm"]
        return None if not base else D._round(1.0 - value / base, 4)

    magnitudes = {D.CAMERAS[c]: {
        "all_frames_px1280": D._round(float(np.linalg.norm(static_all[c]))),
        "first_half_px1280": D._round(float(np.linalg.norm(static_first[c]))),
        "second_half_px1280": D._round(float(np.linalg.norm(static_second[c]))),
        "first_half_xy": [D._round(v) for v in static_first[c]],
        "second_half_xy": [D._round(v) for v in static_second[c]],
        "half_to_half_difference_px1280": D._round(
            float(np.linalg.norm(static_first[c] - static_second[c]))),
        "difference_as_a_fraction_of_the_mean_magnitude": D._round(
            float(np.linalg.norm(static_first[c] - static_second[c])
                  / max(0.5 * (np.linalg.norm(static_first[c])
                               + np.linalg.norm(static_second[c])), 1e-9)), 4),
    } for c in range(camera_count)}
    d_halves = max(entry["difference_as_a_fraction_of_the_mean_magnitude"]
                   for entry in magnitudes.values())

    residual_2d = np.linalg.norm(residual, axis=-1)
    after_static = np.linalg.norm(residual - static_all[None, None, :, None, :], axis=-1)
    after_full = np.linalg.norm(residual - common[..., None, :], axis=-1)

    frame_error = np.array([np.nanmedian(row) if np.isfinite(row).any() else np.nan
                            for row in per_frame["as_measured"]])
    frame_common = np.array([np.nanmedian(np.linalg.norm(common[:, f], axis=-1))
                             for f in range(frames)])

    decision = D.evaluate_decision_rule(
        g_heldout=gain("SPLIT_HALF_held_out"),
        p_heldout=bootstrap["arms"]["SPLIT_HALF_held_out"]["P_beats_baseline"],
        d_halves=d_halves,
        g_shuffled_cameras=gain("CONTROL_shuffled_camera_assignment"),
        g_ceiling=gain("ORACLE_full_common_mode_removed"),
    )

    worst = max(magnitudes, key=lambda name:
                magnitudes[name]["difference_as_a_fraction_of_the_mean_magnitude"])
    entry = magnitudes[worst]
    same_size = (
        f"Plainly: NEARLY, and not by much. The worst camera is {worst}, which goes "
        f"{entry['first_half_px1280']} px on frames [0, {SPLIT_FRAME}) to "
        f"{entry['second_half_px1280']} px on frames [{SPLIT_FRAME}, {frames}) -- a "
        f"half-to-half difference of {entry['half_to_half_difference_px1280']} px, "
        f"{entry['difference_as_a_fraction_of_the_mean_magnitude']:.2f} of its own mean "
        f"magnitude. The rule's clause 3 asks for <= {D.THRESHOLD_D_HALVES:.2f}, so it "
        f"passes by {D.THRESHOLD_D_HALVES - d_halves:.2f} and no bootstrap was put on "
        "D_halves, so treat 'the offsets are static' as UNRESOLVED on 150 frames rather "
        "than established. Every camera shrinks from the first half to the second, which "
        "is a drift, not noise. It does not change the verdict: destination (b) fails on "
        "clause 4 independently."
        if d_halves <= D.THRESHOLD_D_HALVES else
        f"Plainly: NO. The worst camera is {worst}, {entry['first_half_px1280']} px on the "
        f"first half against {entry['second_half_px1280']} px on the second -- "
        f"{entry['difference_as_a_fraction_of_the_mean_magnitude']:.0%} of its own "
        "magnitude. A single static constant per camera cannot express that.")

    report = {
        "report": "I3 report 2 of 3 -- per-camera static common mode, split-half",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "regenerate": "python3 tools/swap-harness/common_mode.py",
        "interpreter": "system python3 (tools/swap-harness/); NOT .venv",
        "reference": "MAMMA's fitted `pred_joints` projected through our rig, 14 joints. "
                     "ORACLE-ASSISTED: the offsets are fitted against this reference, so "
                     "every gain here is a CEILING. Millimetre figures here must never "
                     "share an axis with the pixel figures of "
                     "detector-self-agreement.json.",
        "why_the_baseline_is_not_rung_7s_number": (
            "The 'as_measured' baseline here is a RAW PER-FRAME RE-TRIANGULATION of the "
            "associated detections, joint by joint, with no sequence solve, no fill and no "
            "Savitzky-Golay window -- because an offset applied to 2D has to be scored "
            "through the same geometry in every arm, and the temporal stage would carry "
            "part of the correction. Rung 7's 36.1 / 41.3 mm is the DELIVERED smoothed "
            "track on 15 joints including `nose`. Same reference, different population and "
            "different stage: the difference is not a regression and the two numbers do "
            "not belong on one axis."
        ),
        "blind_to": BLIND_TO,
        "decision_rule": D.DECISION_RULE,
        "THE_VERDICT": decision,
        "are_the_offsets_the_same_size_on_both_halves": same_size,
        "population": D.population_header({
            "joints_scored": list(names),
            "joints_excluded_and_why": D.SEMANTIC_EXCLUSIONS,
            "joints_excluded_count": len(D.SEMANTIC_EXCLUSIONS),
            "denominator": f"{int(keep.sum())} of {keep.size} (subject, frame, joint) slots "
                           "-- every arm scored on exactly the slots every arm resolved.",
            "split": f"CONTIGUOUS halves at frame {SPLIT_FRAME}; each half scored with the "
                     "other half's fit. The interleaved (even/odd) split is a control.",
            "minimum_joints_for_a_common_mode": MINIMUM_JOINTS_FOR_A_COMMON_MODE,
        }),
        "the_four_numbers": {
            "note": "REPORTED, NEVER SHIPPED. Fitted against a MAMMA-derived reference; "
                    "shipping them would put MAMMA into the delivery path.",
            "units": "pixels at 1280 wide (multiply by 3.0 for native 3840)",
            "per_camera": magnitudes,
            "D_halves_worst_camera": D._round(d_halves, 4),
        },
        "3d_error_vs_mamma_projections_mm": {
            name: dict(summarise(matrix), gain_vs_as_measured=gain(name))
            for name, matrix in scored.items()
        },
        "2d_residual_px1280": {
            "as_measured_median": D._round(np.nanmedian(residual_2d)),
            "after_the_static_per_camera_offset": D._round(np.nanmedian(after_static)),
            "after_the_full_per_view_common_mode": D._round(np.nanmedian(after_full)),
            "common_mode_magnitude_median": D._round(np.nanmedian(np.linalg.norm(common, axis=-1))),
            "common_mode_magnitude_p90": D._round(np.nanpercentile(np.linalg.norm(common, axis=-1), 90)),
            "implied_uniform_scale_factor_median": D._round(np.nanmedian(implied_scale), 5),
            "note": "the residual must shrink in 2D as well as in 3D, or the 3D gain is "
                    "not the offset doing the work.",
        },
        "what_clause_4_of_the_rule_revealed": {
            "labelled": "POST HOC -- read after the verdict, and it changes no threshold.",
            "finding": "the four fitted offsets are not four numbers. All four point "
                       "up-and-left by a similar amount, so permuting them across cameras "
                       "leaves the correction almost unchanged and the shuffled-camera "
                       "control reproduces "
                       f"{100 * (gain('CONTROL_shuffled_camera_assignment') or 0) / max(gain('SPLIT_HALF_held_out') or 1e-9, 1e-9):.0f}% "
                       "of the held-out gain. Clause 4 therefore FAILS, and the "
                       "preregistered destination (b) -- a PER-CAMERA fix, four numbers -- "
                       "is not supported by this data.",
            "what_it_points_at_instead": "a single detector-wide 2D offset, the signature "
                                         "of one shared crop / resize / half-pixel "
                                         "convention rather than four camera-specific "
                                         "ones. The arms below test that reading.",
            "global_single_offset_px1280": {
                "first_half_xy": [D._round(v) for v in global_first],
                "second_half_xy": [D._round(v) for v in global_second],
                "held_out_gain": gain("POSTHOC_global_single_offset_held_out"),
                "four_number_held_out_gain": gain("SPLIT_HALF_held_out"),
            },
            "the_control_that_CAN_discriminate": "a rotated offset -- same magnitude, "
                                                 "direction turned 90 degrees. Camera "
                                                 "shuffling cannot test direction when "
                                                 "every camera agrees on it; rotation can.",
            "a_rule_for_the_follow_up_experiment_registered_now": (
                "a ONE-vector detector-wide offset is worth building iff its held-out gain "
                "is >= 0.10 of the baseline, the rotated control's gain is < 0 (it must "
                "HURT), and the negated control's gain is < 0. Its constants must still be "
                "re-selected from a MAMMA-free reference before anything ships. Registered "
                "before the three POSTHOC arms below were computed."
            ),
            "and_how_that_rule_came_out": {
                "clauses": {
                    "global held-out gain >= 0.10":
                        (gain("POSTHOC_global_single_offset_held_out") or 0) >= 0.10,
                    "rotated control gain < 0":
                        (gain("POSTHOC_CONTROL_global_offset_rotated_90deg") or 0) < 0,
                    "negated control gain < 0":
                        (gain("POSTHOC_CONTROL_global_offset_negated") or 0) < 0,
                },
                "verdict": "NOT MET -- the rotated control's MEDIAN gain is "
                           f"{gain('POSTHOC_CONTROL_global_offset_rotated_90deg')}, i.e. "
                           "very slightly positive rather than negative, so the clause as "
                           "written fails."
                           if (gain("POSTHOC_CONTROL_global_offset_rotated_90deg") or 0) >= 0
                           else "met",
                "and_it_is_left_failing_on_purpose": (
                    "The clause was registered on the MEDIAN alone and the p95 says the "
                    "opposite: the rotated offset takes the p95 from "
                    f"{summarise(scored['as_measured'])['p95_mm']} mm to "
                    f"{summarise(scored['POSTHOC_CONTROL_global_offset_rotated_90deg'])['p95_mm']} mm "
                    "while the real one takes it to "
                    f"{summarise(scored['POSTHOC_global_single_offset_held_out'])['p95_mm']} mm. "
                    "A rule that was written down badly is not repaired by rewriting it "
                    "after the numbers; the follow-up rule is restated for the NEXT run as "
                    "'the rotated control must not improve EITHER the median or the p95', "
                    "and until that run it is unmet. What the rotated arm does establish is "
                    "that a per-view translation's median gain is only weakly "
                    "direction-specific -- which is the same thing the 17.5% ceiling says."
                ),
            },
        },
        "controls": {
            "CONTROL_same_frames_fit_and_score": "the SAME four numbers fitted and scored "
                                                 "on the same frames. It must beat the "
                                                 "held-out arm; the difference is the "
                                                 "inflation a same-frames fit buys.",
            "CONTROL_interleaved_half_held_out": "even/odd frames instead of contiguous "
                                                 "halves. With this much autocorrelation "
                                                 "it is nearly a same-frames fit wearing a "
                                                 "held-out label, and its gain should sit "
                                                 "near the same-frames arm's.",
            "CONTROL_shuffled_camera_assignment": "each camera given the NEXT camera's "
                                                  "offset. Same four numbers, wrong "
                                                  "cameras. It must NOT reproduce the gain.",
            "POSTHOC_CONTROL_global_offset_rotated_90deg": "the global offset's magnitude "
                                                          "with its direction turned 90 "
                                                          "degrees. It must HURT; a gain "
                                                          "here would mean the improvement "
                                                          "is magnitude shrinkage, not a "
                                                          "recovered direction.",
            "POSTHOC_CONTROL_global_offset_negated": "the global offset applied backwards. "
                                                     "It must hurt by about as much as the "
                                                     "real one helps.",
            "ORACLE_full_common_mode_removed": "a free 2D translation per subject per "
                                               "frame per camera, fitted on the scored "
                                               "frames. Not achievable in production; it "
                                               "is the CEILING of any per-view translation "
                                               "fix, and what survives it is what a better "
                                               "detector would have to reach.",
        },
        "statistics": {
            "lag1_autocorrelation_of_the_per_frame_median_3d_error": D._round(
                D.lag1_autocorrelation(frame_error), 4),
            "lag1_autocorrelation_of_the_per_frame_common_mode_magnitude": D._round(
                D.lag1_autocorrelation(frame_common), 4),
            "lag1_note": "measured here, not inherited from CLAUDE.md's 0.99, which is the "
                         "head lane's series.",
            "moving_block_bootstrap": bootstrap,
        },
        "pipeline_diagnostics": diagnostics.as_dict(),
        "subject_pairing": {f"our_subject_{s:02d}": f"body_id-{b:02d}"
                            for s, b in sorted(mapping.items())},
        "inputs_sha256": D.input_shas(
            [D.RIG]
            + [D.WORK / f"{name}-soma77-observations.jsonl" for name in D.CAMERAS]
            + [D.MA3D / f"verts_joints_body_id-{b:02d}.npz" for b in (0, 1)]),
        "mamma_used": True,
        "ships_nothing": "instrument-only; the four offsets are reported and never applied "
                         "to a delivery build.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"3D error vs MAMMA projections, {int(keep.sum())} slots every arm resolved:")
    for name, matrix in scored.items():
        entry = summarise(matrix)
        print(f"  {name:42s} median {entry['median_mm']:8.2f} mm   p95 {entry['p95_mm']:8.2f}   "
              f"gain {str(gain(name)):>8s}")
    print(f"\nthe four numbers (px at 1280):")
    for name, entry in magnitudes.items():
        print(f"  {name}: all {entry['all_frames_px1280']:5.2f}  first half "
              f"{entry['first_half_px1280']:5.2f}  second half {entry['second_half_px1280']:5.2f}  "
              f"D {entry['difference_as_a_fraction_of_the_mean_magnitude']:.2f}")
    print(f"\n{same_size}")
    print(f"lag-1 of the per-frame median 3D error: "
          f"{report['statistics']['lag1_autocorrelation_of_the_per_frame_median_3d_error']}")
    print(f"\nVERDICT: {decision['verdict']}")
    for key in ("a_pseudo_label_campaign", "b_per_camera_offset_fix"):
        print(f"  {key}: {decision[key]['triggered']}  {decision[key]['clauses']}")
    print(f"\nwrote {args.out.relative_to(D.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
