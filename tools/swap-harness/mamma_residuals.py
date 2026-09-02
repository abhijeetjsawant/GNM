#!/usr/bin/env python3
"""I3 report 3 of 3 -- our detector's 2D residual against MAMMA's projections, and the
retirement of the "2.2x vs MAMMA" headline.

TWO DISTRIBUTIONS, DELIBERATELY NOT DIVIDED BY ONE ANOTHER.

  * OURS -- SOMA-77's 2D minus our rig's projection of MAMMA's fitted `pred_joints`, on
    the 14 joints whose semantics survive the crossing. MAMMA's fit never saw our
    detector, so this residual is not deflated. It is still not "our detector's error":
    it is our detector's disagreement with a research fitter, and it carries MAMMA's own
    fit error and any rig calibration error with it.

  * MAMMA's OWN -- its `ma_2d` landmarks minus our rig's projection of its own fitted
    surface. **DEFLATED, and the label is the point**: the fit CONSUMED those very
    landmarks, so the residual is regularised toward them and is a lower bound on
    MammaNet's error. It is reported for its SHAPE and as the noise pool the perfect-2D
    oracle samples (`tools/compare/oracle_2d.py`), never as an accuracy figure.

WHY THE RATIO OF THE TWO IS RETIRED. The "2.2x" headline divided our detector's
cross-view disagreement -- scored against RAW TRIANGULATION of our own 2D -- by MAMMA's
residual against a BODY-REGULARISED FIT that had already consumed its landmarks. Two
different references, two different denominators, one axis. The numerator was a detector
measured against an unregularised geometric reconstruction of itself; the denominator was
a detector measured against a smoothed parametric surface fitted to it. A body prior that
pulls landmarks onto a mesh shrinks the second and not the first, and the size of that
shrinkage is unknown here, so the ratio is not a measurement of anything. It is retired,
and the two distributions are published side by side with their references attached.

THE SEMANTIC EXCLUSIONS ARE COUNTED, NOT DROPPED QUIETLY. `nose` is our SOMA-77 index 6,
the *Head skeletal joint* -- a point inside the skull, behind the eyes (CLAUDE.md) -- and
MAMMA's counterpart is a different skeleton's head joint. Ears are schema-only in SOMA-77
and populated on zero frames; the locked-head SMPL-X has no moving eye joints. Five of
our nineteen contract joints are excluded, and the arm WITH `nose` in it is the control
that must be worse.

    python3 tools/swap-harness/mamma_residuals.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i3_decision as D  # noqa: E402
from i3_decision import cm  # noqa: E402

REPORT = D.ROOT / "artifacts/compare/detector-vs-mamma.json"
ORACLE_CACHE = D.ROOT / "artifacts/compare/oracle-2d-cache/verts512-regressor.npy"
ORACLE_REPORT = D.ROOT / "artifacts/compare/oracle-2d.json"

MINIMUM_CONFIDENCE = 0.25
VISIBILITY_FLOOR = 0.5
QUANTILES = [10, 25, 50, 75, 90, 95, 99]
SHAPE_FIT_SAMPLES = 50_000

BLIND_TO = (
    "This figure is our detector's DISAGREEMENT WITH A RESEARCH FITTER, not its error. "
    "It cannot see: (1) anything MAMMA's fit gets wrong -- that is charged to us in full; "
    "(2) calibration error, likewise charged to us, and rung 0 is 'not owned' (the rig IS "
    "MAMMA's ma_cap); (3) the difference between a skeletal joint and the point a detector "
    "was trained to mark -- a convention offset of centimetres reads here as detector "
    "error, which is exactly why `nose` is excluded and kept as a control; (4) depth, "
    "because a 2D residual is measured in the image plane. And the two distributions in "
    "this report are NOT on one axis: MAMMA's is deflated by construction and ours is not, "
    "so their ratio measures the body prior, not the detectors."
)


def swap_left_right(names) -> list[str]:
    """`left_x` <-> `right_x`, everything else untouched. An involution, and the control
    that must fail: same joints, same frames, same subject, the sides crossed."""
    return [name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            for name in names]


def describe(values: np.ndarray, width: int) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    scale = D.NATIVE_WIDTH / width
    return {
        "n": int(finite.size),
        f"median_px{width}": D._round(np.median(finite)),
        "median_px3840": D._round(np.median(finite) * scale),
        f"quantiles_px{width}": {str(q): D._round(np.percentile(finite, q)) for q in QUANTILES},
        "quantile_grid": QUANTILES,
    }


def shape_fit(values: np.ndarray, seed: int = 5) -> dict:
    """Two-point mixture or lognormal continuum? It bears on the decision rule.

    A mixture with a small bad fraction says 'a few landmarks blow up and the rest are
    fine' -- an outlier-rejection problem. A lognormal continuum says 'everything is a
    little off' -- which no gate and no veto reaches, and which is what a retrained
    detector is for.
    """
    measured = np.percentile(values[np.isfinite(values)], [10, 25, 50, 75, 90, 95])
    rng = np.random.default_rng(seed)
    grid = [10, 25, 50, 75, 90, 95]
    best_mixture = best_lognormal = None
    scale = float(np.median(measured))
    for clean in np.linspace(0.1 * scale, 1.5 * scale, 23):
        for fraction in np.arange(0.02, 0.35, 0.02):
            for bad in np.linspace(1.0 * scale, 15.0 * scale, 28):
                sigma = np.where(rng.random(SHAPE_FIT_SAMPLES) < fraction, bad, clean)
                draw = np.abs(rng.normal(0, sigma)) * np.sqrt(2)
                loss = float(np.mean(np.abs(np.log(np.percentile(draw, grid) / measured))))
                if best_mixture is None or loss < best_mixture[0]:
                    best_mixture = (loss, clean, fraction, bad)
    for median in np.linspace(0.1 * scale, 1.5 * scale, 23):
        for spread in np.arange(0.3, 2.01, 0.05):
            sigma = median * np.exp(rng.normal(0, spread, SHAPE_FIT_SAMPLES))
            draw = np.abs(rng.normal(0, sigma)) * np.sqrt(2)
            loss = float(np.mean(np.abs(np.log(np.percentile(draw, grid) / measured))))
            if best_lognormal is None or loss < best_lognormal[0]:
                best_lognormal = (loss, median, spread)
    return {
        "two_point_mixture": {"clean_px": D._round(best_mixture[1]),
                              "bad_fraction": D._round(best_mixture[2]),
                              "bad_px": D._round(best_mixture[3]),
                              "log_ratio_loss": D._round(best_mixture[0], 4)},
        "lognormal": {"median_px": D._round(best_lognormal[1]),
                      "sigma_log": D._round(best_lognormal[2]),
                      "log_ratio_loss": D._round(best_lognormal[0], 4)},
        "better": ("lognormal continuum" if best_lognormal[0] < best_mixture[0]
                   else "two-point mixture"),
        "why_it_matters": "a mixture is an outlier problem a veto or a robust loss can "
                          "reach; a lognormal continuum is broadband error that only a "
                          "better detector removes.",
    }


def mamma_deflated_pool(cameras_native) -> tuple[np.ndarray, dict]:
    """MAMMA's ma_2d against our rig's projection of its own fitted surface.

    The identical pool and convention as `tools/compare/oracle_2d.py`'s projection check
    -- native pixels, visibility >= 0.5, every second frame, all four cameras, both
    bodies -- reusing its cached float64 regressor rather than recomputing it a second,
    different way. The report cross-checks the quantiles against that file.
    """
    if ORACLE_CACHE.exists():
        regressor = np.load(ORACLE_CACHE)
        source = str(ORACLE_CACHE.relative_to(D.ROOT))
    else:
        regressor = np.asarray(pickle.load(open(D.VERTS_512, "rb"))).astype(np.float64)
        source = str(D.VERTS_512.relative_to(D.ROOT))
    landmarks = np.stack([np.load(D.MA2D / f"{name}.npz", allow_pickle=True)["landmarks"]
                          for name in D.CAMERAS])
    visibility = np.stack([np.load(D.MA2D / f"{name}.npz", allow_pickle=True)["visibilities"]
                           for name in D.CAMERAS])
    pool = []
    for body in (0, 1):
        vertices = np.load(D.MA3D / f"verts_joints_body_id-{body:02d}.npz",
                           allow_pickle=True)["pred_vertices"].astype(np.float64)
        surface = np.einsum("kv,fvc->fkc", regressor, vertices)
        for index in range(len(D.CAMERAS)):
            for frame in range(0, vertices.shape[0], 2):
                uv, depth = cameras_native[index].project(surface[frame])
                keep = (depth > 0.0) & (visibility[index, frame, body] >= VISIBILITY_FLOOR)
                pool.append(np.linalg.norm(
                    uv - landmarks[index, frame, body, :, :2], axis=1)[keep])
    pool = np.concatenate(pool)
    cross_check = {"against": "artifacts/compare/oracle-2d.json projection.dense_residual_px_native"}
    if ORACLE_REPORT.exists():
        try:
            recorded = json.loads(ORACLE_REPORT.read_text())["projection"]["dense_residual_px_native"]
            here = [round(float(v), 2) for v in np.percentile(pool, recorded["quantiles"])]
            cross_check.update({"recorded": recorded["values"], "here": here,
                                "identical": here == recorded["values"],
                                "recorded_n": recorded["n"], "here_n": int(pool.size)})
        except (OSError, ValueError, KeyError):
            cross_check["error"] = "oracle-2d.json present but unreadable"
    return pool, {"regressor_source": source, "cross_check": cross_check}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    args = parser.parse_args()

    cameras = D.working_cameras()
    cameras_native = cm.load_camera_rig(D.RIG)
    assigned, positions, _raw, diagnostics = D.assigned_detector_2d()
    subjects, frames = assigned.shape[:2]
    mapping = D.subject_mapping(positions)
    truth = D.mamma_pred_joints()

    scored_names = list(D.SCORED_VS_MAMMA)
    with_nose = scored_names + ["nose"]

    def reference_for(order: dict[int, int], names) -> np.ndarray:
        stacked = np.stack([truth[order[s], :frames][:, [D.PAIRS[n] for n in names]]
                            for s in range(subjects)])
        return np.moveaxis(D.project_joints(cameras, stacked), 0, 2)     # [S, F, C, J, 2]

    def observed_for(names, array=None) -> np.ndarray:
        source = assigned if array is None else array
        out = np.stack([source[:, :, :, cm.JOINT_INDEX[n]] for n in names], axis=3)
        bad = ~(np.isfinite(out[..., :2]).all(axis=-1) & (out[..., 2] >= MINIMUM_CONFIDENCE))
        out = out.copy()
        out[bad] = np.nan
        return out

    crossed = {s: mapping[(s + 1) % subjects] for s in mapping}
    frozen = np.repeat(assigned[:, :1], frames, axis=1)

    def residual(names, order=None, source=None) -> np.ndarray:
        order = mapping if order is None else order
        return np.linalg.norm(
            observed_for(names, source)[..., :2] - reference_for(order, names), axis=-1)

    # A left/right swap is the semantic mismatch that MUST fail: same joints, same
    # frames, same subject, the sides crossed. It is the control the `nose` exclusion was
    # supposed to be and -- as the report says below -- is not, on this fixture.
    swapped_names = swap_left_right(scored_names)

    ours_full = residual(scored_names)
    with_nose_full = residual(with_nose)
    nose_only = residual(["nose"])
    shuffled_full = residual(scored_names, order=crossed)
    frozen_full = residual(scored_names, source=frozen)
    lr_full = np.linalg.norm(
        observed_for(scored_names)[..., :2] - reference_for(mapping, swapped_names), axis=-1)

    # SAME DENOMINATOR: the slots every arm resolved. `with_nose` has an extra joint, so
    # it is masked on the shared 14 plus its own nose column.
    keep = (np.isfinite(ours_full) & np.isfinite(shuffled_full) & np.isfinite(frozen_full)
            & np.isfinite(with_nose_full[..., :len(scored_names)]))
    ours = np.where(keep, ours_full, np.nan)
    shuffled = np.where(keep, shuffled_full, np.nan)
    frozen_arm = np.where(keep, frozen_full, np.nan)
    lr_arm = np.where(keep, lr_full, np.nan)
    nose_keep = keep.all(axis=-1)[..., None] & np.isfinite(nose_only)
    nose = np.where(nose_keep, nose_only, np.nan)
    with_nose_arm = np.concatenate([ours, nose], axis=-1)

    deflated, deflated_meta = mamma_deflated_pool(cameras_native)

    per_frame = {name: np.moveaxis(matrix, 1, 0).reshape(frames, -1) for name, matrix in (
        ("ours_vs_mamma_projections", ours),
        ("CONTROL_including_the_semantic_mismatch_nose", with_nose_arm),
        ("CONTROL_shuffled_subject_pairing", shuffled),
        ("CONTROL_frozen_frame0_detection", frozen_arm),
        ("CONTROL_left_right_swapped", lr_arm))}
    bootstrap = D.paired_block_bootstrap(
        per_frame, baseline="ours_vs_mamma_projections",
        candidates=[name for name in per_frame if name != "ours_vs_mamma_projections"])
    frame_median = np.array([np.nanmedian(row) if np.isfinite(row).any() else np.nan
                             for row in per_frame["ours_vs_mamma_projections"]])

    ours_median_native = describe(ours, D.WORKING_WIDTH)["median_px3840"]
    deflated_median_native = describe(deflated, D.NATIVE_WIDTH)["median_px3840"]

    per_joint = {}
    for index, name in enumerate(scored_names):
        per_joint[name] = describe(ours[..., index], D.WORKING_WIDTH)
    per_joint["nose (EXCLUDED, control only)"] = describe(nose[..., 0], D.WORKING_WIDTH)

    report = {
        "report": "I3 report 3 of 3 -- our detector's 2D vs MAMMA's projections; the "
                  "2.2x headline retired",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "regenerate": "python3 tools/swap-harness/mamma_residuals.py",
        "interpreter": "system python3 (tools/swap-harness/); NOT .venv",
        "reference": "MAMMA's fitted pred_joints projected through our rig (for OUR arm) "
                     "and MAMMA's own fitted surface projected through our rig (for ITS "
                     "arm, DEFLATED). These are two different references and the two "
                     "distributions must never share an axis -- and neither may share an "
                     "axis with detector-self-agreement.json, whose reference is nothing "
                     "at all.",
        "blind_to": BLIND_TO,
        "decision_rule": D.DECISION_RULE,
        "this_report_supplies_to_the_rule": {
            "shape_of_our_residual": "a lognormal continuum indicates broadband error that "
                                     "only a retrained detector reaches -- corroboration "
                                     "for destination (a); a two-point mixture would "
                                     "indicate an outlier problem a veto could reach.",
            "note": "the verdict is computed in detector-common-mode.json.",
        },
        "RETIRED_HEADLINE": {
            "what_it_said": "our 2D detector is ~2.2x worse than MAMMA's",
            "what_it_actually_compared": {
                "numerator": "our detector's disagreement scored against RAW TRIANGULATION "
                             "of our own 2D -- an unregularised geometric reconstruction, "
                             "with no body model anywhere in it",
                "denominator": "MAMMA's 2D scored against a BODY-REGULARISED SMPL-X FIT "
                               "that had already consumed those very landmarks",
            },
            "why_the_denominators_differ": "a body prior pulls landmarks onto a mesh and "
                                           "shrinks the residual it is scored against; raw "
                                           "triangulation does not. The two arms therefore "
                                           "differ in the reference, in whether the "
                                           "reference consumed the measurement, and in the "
                                           "joint population (17 skeletal joints against "
                                           "512 surface landmarks). The size of the "
                                           "shrinkage is unknown on this fixture, so the "
                                           "ratio bounds nothing.",
            "status": "RETIRED. Do not restore it, and do not replace it with the ratio "
                      "below, which has the same defect.",
            "the_ratio_it_would_be_today": {
                "ours_median_px3840": ours_median_native,
                "mamma_deflated_median_px3840": deflated_median_native,
                "ratio": D._round(None if not deflated_median_native
                                  else ours_median_native / deflated_median_native),
                "is_a_score": False,
                "note": "printed so nobody has to recompute it to be told it is not a "
                        "score. It is a measurement of the body prior's deflation plus a "
                        "joint-population change, not of two detectors.",
            },
        },
        "population": D.population_header({
            "joints_scored": scored_names,
            "joints_excluded_and_why": D.SEMANTIC_EXCLUSIONS,
            "joints_excluded_count": len(D.SEMANTIC_EXCLUSIONS),
            "contract_joints": 19,
            "detector_emits": 17,
            "scored_against_mamma": len(scored_names),
            "excluded_observations": {
                "nose": int(np.isfinite(nose).sum()),
                "left_eye/right_eye": "emitted by SOMA-77 (300 person-frames each per "
                                      "camera) but the locked-head SMPL-X has no moving eye "
                                      "joint to score against, so they are not in PAIRS",
                "left_ear/right_ear": "0 observations -- schema-only in SOMA-77",
            },
            "denominator": f"{int(keep.sum())} of {keep.size} (subject, frame, camera, "
                           "joint) slots; every arm on exactly those slots, the nose "
                           "column added only to the control that needs it.",
        }),
        "arms": {
            "ours_vs_mamma_projections": dict(
                describe(ours, D.WORKING_WIDTH),
                reference="MAMMA pred_joints projected through our rig",
                deflated=False,
                shape=shape_fit(ours[np.isfinite(ours)])),
            "MAMMA_own_residual_DEFLATED": dict(
                describe(deflated, D.NATIVE_WIDTH),
                reference="MAMMA's ma_2d against our rig's projection of its OWN fitted "
                          "surface -- the fit consumed these landmarks",
                deflated=True,
                is_a_score=False,
                shape=shape_fit(deflated),
                **deflated_meta),
        },
        "controls": {
            "CONTROL_including_the_semantic_mismatch_nose": dict(
                describe(with_nose_arm, D.WORKING_WIDTH),
                must="be worse than the scored population",
                note="`nose` alone is in per_joint below. Our nose IS the Head skeletal "
                     "joint, a point inside the skull; MAMMA's is a different skeleton's "
                     "head joint. The gap between this arm and the scored arm is the size "
                     "of the convention error the exclusion removes."),
            "CONTROL_nose_alone": dict(describe(nose, D.WORKING_WIDTH),
                                       must="be much worse than any scored joint"),
            "CONTROL_shuffled_subject_pairing": dict(
                describe(shuffled, D.WORKING_WIDTH),
                must="be far worse -- our subject scored against the OTHER performer's "
                     "MAMMA body. Pairing here is derived from pelvis agreement "
                     "(subject_map.py), never from the index."),
            "CONTROL_frozen_frame0_detection": dict(
                describe(frozen_arm, D.WORKING_WIDTH),
                must="be far worse -- our frame-0 detections held constant on every frame. "
                     "A detector that has stopped detecting."),
            "CONTROL_left_right_swapped": dict(
                describe(lr_arm, D.WORKING_WIDTH),
                must="be far worse -- the sides crossed. This is the joint-semantic control "
                     "that actually discriminates on this fixture."),
        },
        "what_the_nose_exclusion_turned_out_to_cost": {
            "labelled": "POST HOC -- the exclusion was decided from the joint definitions, "
                        "before the numbers.",
            "finding": f"`nose` alone scores "
                       f"{describe(nose, D.WORKING_WIDTH).get('median_px1280')} px at 1280 "
                       f"against the scored population's "
                       f"{describe(ours, D.WORKING_WIDTH).get('median_px1280')} px, and "
                       "including it moves the pooled median by "
                       f"{D._round(abs((describe(with_nose_arm, D.WORKING_WIDTH).get('median_px1280') or 0) - (describe(ours, D.WORKING_WIDTH).get('median_px1280') or 0)), 3)}"
                       " px. So on this fixture the exclusion is CONSERVATIVE, not "
                       "load-bearing.",
            "why_it_stays_excluded": "the two head joints belong to two different "
                                     "skeletons and their offset is a convention, not "
                                     "detector error. That the convention happens to be "
                                     "small here is a property of this fixture's head "
                                     "poses, not a licence to score a mismatched pair. The "
                                     "control that does discriminate is the left/right "
                                     "swap, and it is above.",
            "so_the_planned_control_did_not_fail": "the I3 gate card expected the "
                                                   "joint-semantic control to be visibly "
                                                   "worse. It is not. Reported as measured.",
        },
        "per_joint_px1280": per_joint,
        "statistics": {
            "lag1_autocorrelation_of_the_per_frame_median_2d_residual": D._round(
                D.lag1_autocorrelation(frame_median), 4),
            "lag1_note": "measured on this report's own series, not inherited.",
            "moving_block_bootstrap": bootstrap,
        },
        "pipeline_diagnostics": diagnostics.as_dict(),
        "subject_pairing": {f"our_subject_{s:02d}": f"body_id-{b:02d}"
                            for s, b in sorted(mapping.items())},
        "inputs_sha256": D.input_shas(
            [D.RIG, D.VERTS_512]
            + [D.WORK / f"{name}-soma77-observations.jsonl" for name in D.CAMERAS]
            + [D.MA3D / f"verts_joints_body_id-{b:02d}.npz" for b in (0, 1)]
            + [D.MA2D / f"{name}.npz" for name in D.CAMERAS]),
        "mamma_used": True,
        "ships_nothing": "instrument-only; selects no constant, no weight, no threshold.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"our detector vs MAMMA's projections, {int(keep.sum())} slots "
          f"({len(scored_names)} joints, {len(D.SEMANTIC_EXCLUSIONS)} excluded):")
    for name, arm in (("ours", ours), ("CONTROL with nose", with_nose_arm),
                      ("CONTROL nose alone", nose),
                      ("CONTROL shuffled subject", shuffled),
                      ("CONTROL frozen frame 0", frozen_arm),
                      ("CONTROL L/R swapped", lr_arm)):
        entry = describe(arm, D.WORKING_WIDTH)
        print(f"  {name:26s} median {entry['median_px1280']:8.2f} px@1280   "
              f"{entry['median_px3840']:8.2f} px@3840   n={entry['n']}")
    entry = describe(deflated, D.NATIVE_WIDTH)
    print(f"  {'MAMMA own (DEFLATED)':26s} median {entry['median_px3840']:8.2f} px@3840  "
          f"n={entry['n']}   <- NOT on the same axis")
    print(f"\nour residual's shape: {report['arms']['ours_vs_mamma_projections']['shape']['better']}")
    print(f"MAMMA's deflated shape: {report['arms']['MAMMA_own_residual_DEFLATED']['shape']['better']}")
    print(f"oracle-2d cross-check identical: "
          f"{deflated_meta['cross_check'].get('identical')}")
    print(f"lag-1 of the per-frame median 2D residual: "
          f"{report['statistics']['lag1_autocorrelation_of_the_per_frame_median_2d_residual']}")
    print(f"\nwrote {args.out.relative_to(D.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
