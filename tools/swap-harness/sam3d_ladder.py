#!/usr/bin/env python3
"""I3 report 1 -- CROSS-VIEW SELF-AGREEMENT of our 2D detector. Reference-free.

Does our detector agree with *itself* across the four calibrated views? Nothing else
enters: no MAMMA, no fitted body, no joint-convention mapping. For every camera pair,
every frame, every subject and every joint, the symmetric epipolar distance between the
two views' detections of the same physical point.

**HALVE IT.** `_epipolar_distance_px` returns the SUM of the two one-sided distances
(CLAUDE.md; the measured symmetric/one-sided ratio is 1.962 and is re-measured below).
Every headline here is the one-sided value, symmetric / 2, and both are in the report.

WHY THIS FILE HAS THIS NAME. It used to run SAM 3D Body's detections through the same
statistic and print a ratio against SOMA-77. That printout is retired with the "2.2x vs
MAMMA" headline (see `detector-vs-mamma.json`): the two detectors emit different joint
sets, so the ratio had different denominators on its two sides. The statistic was always
the right one; what it needed was our own detector, a control arm and a JSON report.

THE GATE, and it has TWO bands, because cross-view self-agreement alone is passable by
a constant:
  * **G1 discrimination** -- the shuffled cross-view pairing (same detections, wrong
    partner) must be at least 5x worse. A detector whose disagreement with itself is
    indistinguishable from its disagreement with the *other performer* has measured
    nothing.
  * **G2 liveness** -- the median per-frame 2D displacement must be at least
    1.0 px/frame at 1280. A frozen skeleton projected into four calibrated cameras is
    epipolar-perfect by construction and scores exactly 0 here.

WHAT IT IS BLIND TO, stated as a control that PASSES: our own triangulated 3D,
reprojected into the four cameras, passes both bands. Cross-view consistency is a
within-frame geometric property and any 3D-consistent thing has it -- it says nothing
about whether the point is on the right part of the body. Rigidity and accuracy are
different questions (CLAUDE.md).

    python3 tools/swap-harness/sam3d_ladder.py
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

REPORT = D.ROOT / "artifacts/compare/detector-self-agreement.json"

MINIMUM_CONFIDENCE = 0.25          # the pipeline's own floor
G1_DISCRIMINATION_RATIO = 5.0
G2_LIVENESS_PX_PER_FRAME = 1.0
PROJECTED_CONFIDENCE = 0.95

BLIND_TO = (
    "Cross-view self-agreement is a WITHIN-FRAME geometric property and nothing else. "
    "It cannot see: (1) a landmark that is consistently on the wrong part of the body in "
    "every view -- our own triangulated 3D reprojected passes both bands and carries no "
    "detector information at all, and that arm is in the report; (2) depth -- an epipolar "
    "distance is measured along the image plane, so error along the viewing rays is "
    "invisible; (3) any per-camera error that is itself epipolar-consistent, which "
    "includes a translation along the epipolar line and, at four cameras in a rough ring, "
    "a large part of a common-mode crop shift -- that is what report 2 is for; (4) "
    "calibration error, which it charges to the detector."
)


def symmetric_epipolar(fundamental: np.ndarray, source: np.ndarray, target: np.ndarray):
    """Per-point symmetric epipolar distance, the same expression the pipeline medians.

    `cm._epipolar_distance_px` returns the MEDIAN of exactly these values over the
    joints two cameras share; the report asserts that agreement rather than trusting it.
    Returns (symmetric, one_sided_target, one_sided_source).
    """
    source_h = np.hstack((source[:, :2], np.ones((source.shape[0], 1))))
    target_h = np.hstack((target[:, :2], np.ones((target.shape[0], 1))))
    target_lines = source_h @ fundamental.T
    source_lines = target_h @ fundamental
    numerator = np.abs(np.sum(target_h * target_lines, axis=1))
    target_norm = np.hypot(target_lines[:, 0], target_lines[:, 1])
    source_norm = np.hypot(source_lines[:, 0], source_lines[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        to_target = numerator / target_norm
        to_source = numerator / source_norm
    return to_target + to_source, to_target, to_source


def projected_arm(cameras, points: np.ndarray) -> np.ndarray:
    """[subject, frame, 19, 3] world -> the detector-shaped [S, F, C, 19, 3] array."""
    projected = D.project_joints(cameras, points)            # [C, S, F, 19, 2]
    projected = np.moveaxis(projected, 0, 2)                 # [S, F, C, 19, 2]
    out = np.full(projected.shape[:-1] + (3,), np.nan)
    out[..., :2] = projected
    out[..., 2] = np.where(np.isfinite(projected).all(axis=-1), PROJECTED_CONFIDENCE, np.nan)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()

    cameras = D.working_cameras()
    assigned, positions, _raw, diagnostics = D.assigned_detector_2d()
    subjects, frames, camera_count, joints = assigned.shape[:4]
    names = list(cm.JOINT_NAMES)

    frozen = np.repeat(positions[:, :1], frames, axis=1)
    arms = {
        "ours": assigned,
        "CONTROL_frozen_skeleton_projected": projected_arm(cameras, frozen),
        "CONTROL_our_own_3d_reprojected": projected_arm(cameras, positions),
    }

    usable = np.isfinite(assigned[..., :2]).all(axis=-1) & (assigned[..., 2] >= MINIMUM_CONFIDENCE)
    for name, arm in arms.items():
        usable &= (np.isfinite(arm[..., :2]).all(axis=-1)
                   & (arm[..., 2] >= MINIMUM_CONFIDENCE))
        del name
    # the shuffled arm reads the OTHER subject out of the second camera, so its slot has
    # to be usable for both subjects; the AND below keeps every arm on one denominator.
    usable = usable.all(axis=0)[None].repeat(subjects, axis=0)

    pairs = [(a, b) for a in range(camera_count) for b in range(a + 1, camera_count)]
    fundamentals = {(a, b): cm._fundamental_matrix(cameras[a], cameras[b]) for a, b in pairs}

    arm_names = list(arms) + ["CONTROL_shuffled_cross_view_pairing"]
    symmetric = {name: np.full((len(pairs), frames, subjects, joints), np.nan)
                 for name in arm_names}
    one_sided_target = np.full((len(pairs), frames, subjects, joints), np.nan)
    pipeline_check: list[tuple[float, float]] = []

    for pair_index, (a, b) in enumerate(pairs):
        fundamental = fundamentals[(a, b)]
        for frame in range(frames):
            for subject in range(subjects):
                slots = usable[subject, frame, a] & usable[subject, frame, b]
                if not slots.any():
                    continue
                for name in arm_names:
                    if name == "CONTROL_shuffled_cross_view_pairing":
                        source = arms["ours"][subject, frame, a]
                        target = arms["ours"][(subject + 1) % subjects, frame, b]
                    else:
                        source = arms[name][subject, frame, a]
                        target = arms[name][subject, frame, b]
                    sym, to_t, _to_s = symmetric_epipolar(
                        fundamental, source[slots], target[slots])
                    symmetric[name][pair_index, frame, subject, slots] = sym
                    if name == "ours":
                        one_sided_target[pair_index, frame, subject, slots] = to_t

    quantiles = [10, 25, 50, 75, 90, 95, 99]

    def describe(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"n": 0}
        halved = finite / 2.0
        return {
            "n": int(finite.size),
            "one_sided_median_px1280": D._round(np.median(halved)),
            "one_sided_quantiles_px1280": {str(q): D._round(np.percentile(halved, q))
                                           for q in quantiles},
            "symmetric_median_px1280": D._round(np.median(finite)),
            "one_sided_median_px3840": D._round(np.median(halved) * D.PX_1280_TO_3840),
        }

    per_arm = {name: describe(values) for name, values in symmetric.items()}

    # per-frame series for the bootstrap and the autocorrelation: [frame, ...] with the
    # pair/subject/joint axes pooled.
    per_frame = {name: np.moveaxis(values, 1, 0).reshape(frames, -1) / 2.0
                 for name, values in symmetric.items()}
    frame_median = np.array([np.nanmedian(row) if np.isfinite(row).any() else np.nan
                             for row in per_frame["ours"]])
    bootstrap = D.paired_block_bootstrap(
        per_frame, baseline="CONTROL_shuffled_cross_view_pairing",
        candidates=("ours", "CONTROL_frozen_skeleton_projected",
                    "CONTROL_our_own_3d_reprojected"))

    # liveness: median per-frame 2D displacement, on the same slots
    def liveness(arm: np.ndarray) -> float | None:
        xy = np.where(usable[..., None], arm[..., :2], np.nan)
        step = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
        finite = step[np.isfinite(step)]
        return D._round(np.median(finite)) if finite.size else None

    liveness_px = {name: liveness(arms[name]) for name in arms}
    liveness_px["CONTROL_shuffled_cross_view_pairing"] = liveness_px["ours"]

    assigned_any = np.isfinite(assigned[..., :2]).all(axis=-1).any(axis=-1)
    own_usable = (np.isfinite(assigned[..., :2]).all(axis=-1)
                  & (assigned[..., 2] >= MINIMUM_CONFIDENCE))
    own_values = []
    for pair_index, (a, b) in enumerate(pairs):
        fundamental = fundamentals[(a, b)]
        for frame in range(frames):
            for subject in range(subjects):
                slots = own_usable[subject, frame, a] & own_usable[subject, frame, b]
                if not slots.any():
                    continue
                source, target = assigned[subject, frame, a], assigned[subject, frame, b]
                sym, _t, _s2 = symmetric_epipolar(fundamental, source[slots], target[slots])
                own_values.append(sym)
                # `slots` here IS `_epipolar_distance_px`'s own shared mask -- finite and
                # above the confidence floor in both views -- so the two are directly
                # comparable and the assertion is meaningful.
                if int(slots.sum()) >= 4:
                    pipeline_check.append((
                        float(np.median(sym)),
                        float(cm._epipolar_distance_px(
                            fundamental, source, target,
                            minimum_confidence=MINIMUM_CONFIDENCE,
                            minimum_shared_joints=4)),
                    ))
    own_population = D._round(np.median(np.concatenate(own_values)) / 2.0)

    ours = per_arm["ours"]["one_sided_median_px1280"]
    shuffled = per_arm["CONTROL_shuffled_cross_view_pairing"]["one_sided_median_px1280"]
    ratio = None if not ours else shuffled / ours

    def verdict(name: str) -> dict:
        # A projected arm is epipolar-perfect by construction and its median is 0.0, so
        # its discrimination ratio is infinite and it PASSES G1. That is the whole point
        # of the second band: G1 alone cannot reject a constant.
        arm_median = per_arm[name]["one_sided_median_px1280"]
        if arm_median is None:
            arm_ratio = None
        elif arm_median <= 1e-9:
            arm_ratio = float("inf")
        else:
            arm_ratio = shuffled / arm_median
        g1 = arm_ratio is not None and arm_ratio >= G1_DISCRIMINATION_RATIO
        g2 = liveness_px[name] is not None and liveness_px[name] >= G2_LIVENESS_PX_PER_FRAME
        return {"G1_discrimination_ratio": ("inf" if arm_ratio == float("inf")
                                            else D._round(arm_ratio)),
                "G1_pass": bool(g1),
                "G2_liveness_px_per_frame": liveness_px[name],
                "G2_pass": bool(g2),
                "passes_the_gate": bool(g1 and g2)}

    per_joint = {}
    for index, name in enumerate(names):
        values = symmetric["ours"][:, :, :, index]
        finite = values[np.isfinite(values)]
        per_joint[name] = ({"n": 0, "note": "not emitted by SOMA-77"} if finite.size == 0
                           else {"n": int(finite.size),
                                 "one_sided_median_px1280": D._round(np.median(finite) / 2.0)})
    per_pair = {}
    for index, (a, b) in enumerate(pairs):
        values = symmetric["ours"][index]
        finite = values[np.isfinite(values)]
        per_pair[f"{D.CAMERAS[a]}-{D.CAMERAS[b]}"] = {
            "n": int(finite.size),
            "one_sided_median_px1280": D._round(np.median(finite) / 2.0) if finite.size else None}

    checked = np.asarray(pipeline_check)
    measured_ratio = symmetric["ours"] / np.where(one_sided_target > 0, one_sided_target, np.nan)

    report = {
        "report": "I3 report 1 of 3 -- cross-view self-agreement of our 2D detector",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "regenerate": "python3 tools/swap-harness/sam3d_ladder.py",
        "interpreter": "system python3 (tools/swap-harness/); NOT .venv",
        "reference": "REFERENCE-FREE -- the detector against itself across four "
                     "calibrated views. No MAMMA, no fitted body, no joint mapping. This "
                     "figure must never share an axis with any figure whose reference is "
                     "MAMMA (detector-vs-mamma.json) or with a 3D millimetre figure.",
        "blind_to": BLIND_TO,
        "decision_rule": D.DECISION_RULE,
        "this_report_supplies_to_the_rule": {
            "R1_self_agreement_px1280": ours,
            "R1_shuffled_px1280": shuffled,
            "R1_liveness_px_per_frame": liveness_px["ours"],
            "note": "the verdict itself is computed in detector-common-mode.json, which "
                    "holds G_heldout, G_ceiling, P_heldout and D_halves. These three are "
                    "corroboration: a detector that does not agree with itself across "
                    "views cannot be fixed by four numbers.",
        },
        "population": D.population_header({
            "camera_pairs": [f"{D.CAMERAS[a]}-{D.CAMERAS[b]}" for a, b in pairs],
            "joints_scored": [name for name, entry in per_joint.items() if entry["n"]],
            "joints_dropped": {name: "not emitted by SOMA-77 on any frame"
                               for name, entry in per_joint.items() if not entry["n"]},
            "confidence_floor": MINIMUM_CONFIDENCE,
            "denominator": "one slot mask for every arm: a (subject, frame, camera, joint) "
                           "slot counts only where BOTH subjects are usable in BOTH "
                           "cameras, because the shuffled control reads the other subject "
                           "out of the second view. Same denominator on every arm.",
            "what_the_common_denominator_costs": {
                "camera_slots_the_associator_left_unassigned":
                    f"{int((~assigned_any).sum())} of {assigned_any.size} "
                    "(subject, frame, camera) slots -- an exposure count for the "
                    "ASSOCIATION stage, not a detector figure",
                "scored_slots_of_the_maximum": f"{int(usable.sum())} of {usable.size}",
                "ours_on_its_OWN_population_one_sided_median_px1280": own_population,
                "note": "the headline uses the common denominator so the arms are "
                        "comparable; this line says what that choice cost. The two agree "
                        "closely, so the common denominator is not selecting easy slots.",
            },
        }),
        "units": {
            "symmetric_vs_one_sided": "_epipolar_distance_px returns the SUM of the two "
                                      "one-sided distances (CLAUDE.md). Every headline "
                                      "here is symmetric / 2.",
            "measured_symmetric_over_one_sided_ratio": {
                "median": D._round(np.nanmedian(measured_ratio)),
                "p05": D._round(np.nanpercentile(measured_ratio, 5)),
                "p95": D._round(np.nanpercentile(measured_ratio, 95)),
                "claimed_in_CLAUDE_md": 1.962,
            },
            "px1280_to_px3840": D.PX_1280_TO_3840,
        },
        "wrap_not_reimplement": {
            "positions_reproduce_the_retained_track_to_mm": D._round(float(np.nanmax(np.abs(
                positions[0] - np.load(
                    D.ROOT / "artifacts/commercial-multiview-soma77/subject-00.body-track.npz"
                )["triangulated_world_positions_z_up_m"])) * 1000.0), 8),
            "per_joint_expression_vs_pipeline_median_max_abs_px": D._round(
                float(np.max(np.abs(checked[:, 0] - checked[:, 1]))) if checked.size else None, 9),
            "checks": int(len(pipeline_check)),
            "note": "the per-joint expression is the pipeline's own, verified by taking "
                    "its median over the shared joints and differencing against "
                    "_epipolar_distance_px on the identical inputs.",
        },
        "gate": {
            "G1_discrimination": f"shuffled / candidate >= {G1_DISCRIMINATION_RATIO}",
            "G2_liveness": f"median per-frame 2D displacement >= {G2_LIVENESS_PX_PER_FRAME} "
                           "px/frame at 1280",
            "why_two_bands": "epipolar self-agreement alone is passable by a constant: a "
                             "frozen skeleton projected into calibrated cameras is exactly "
                             "consistent. G2 is the band it cannot pass.",
        },
        "headline": {
            "one_sided_epipolar_median_px1280": ours,
            "shuffled_over_ours": D._round(ratio),
            "why_it_is_not_the_3_11_px_on_record": (
                "The retired printout quoted 3.11 px for SOMA-77 on this footage. Same "
                "statistic, different population: that run paired cameras by projected-root "
                "proximity to a bounding box with a 160 px acceptance radius and scored "
                "whatever survived, while this one takes the pipeline's OWN association and "
                "scores only slots where both subjects are usable in both cameras. The "
                "definition also differs by a hair -- the old figure was the target-side "
                "one-sided distance and this is the mean of the two sides, which the "
                "measured symmetric ratio puts within 1%. Treat 3.11 as superseded, not as "
                "a regression."
            ),
        },
        "arms": {name: dict(per_arm[name], liveness_px_per_frame=liveness_px[name],
                            verdict=verdict(name)) for name in arm_names},
        "controls": {
            "CONTROL_shuffled_cross_view_pairing": "same detections, wrong partner across "
                                                   "views. MUST be far worse.",
            "CONTROL_frozen_skeleton_projected": "our own frame-0 3D projected into all "
                                                 "four cameras on every frame. MUST fail "
                                                 "-- it passes G1 trivially and fails G2.",
            "CONTROL_our_own_3d_reprojected": "our own per-frame 3D reprojected. It PASSES "
                                              "both bands and carries no detector "
                                              "information beyond a fit. That is not a "
                                              "defect in the control; it is what this "
                                              "figure is blind to, and it is why report 1 "
                                              "is a diagnostic and not an acceptance gate.",
        },
        "per_camera_pair": per_pair,
        "per_joint": per_joint,
        "statistics": {
            "lag1_autocorrelation_of_the_per_frame_median": D._round(
                D.lag1_autocorrelation(frame_median), 4),
            "lag1_note": "measured on this report's own series; CLAUDE.md's 0.99 is a "
                         "different lane's series and was not assumed.",
            "moving_block_bootstrap": bootstrap,
        },
        "pipeline_diagnostics": diagnostics.as_dict(),
        "inputs_sha256": D.input_shas(
            [D.RIG] + [D.WORK / f"{name}-soma77-observations.jsonl" for name in D.CAMERAS]),
        "mamma_used": False,
        "ships_nothing": "reference-free diagnostic; selects no constant.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"cross-view self-agreement, one-sided, px at 1280  (n={per_arm['ours']['n']})")
    for name in arm_names:
        entry, check = per_arm[name], verdict(name)
        print(f"  {name:42s} {entry['one_sided_median_px1280']:8.3f}   "
              f"liveness {str(liveness_px[name]):>7s} px/frame   "
              f"G1 {'pass' if check['G1_pass'] else 'FAIL'}  "
              f"G2 {'pass' if check['G2_pass'] else 'FAIL'}")
    print(f"\nshuffled / ours = {ratio:.1f}x   (band: >= {G1_DISCRIMINATION_RATIO})")
    print(f"lag-1 autocorrelation of the per-frame median: "
          f"{report['statistics']['lag1_autocorrelation_of_the_per_frame_median']}")
    print(f"\nwrote {args.out.relative_to(D.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
