#!/usr/bin/env python3
"""D8's gate: every band from the card, as its own keyed block, applied mechanically.

The card's bands, `docs/LADDER_EXECUTION_PLAN.md` section 2, the D8 row, frozen 2026-09-05
before any number in this file existed. They are carried into the report verbatim under
`preregistration` and each is answered in its own block:

    instrument first    `tools/compare/captured_limb_stability.py` must reproduce the
                        card's figures before any src change
    SYNTHETIC           selector (four arms), oracle, and the two must-fails
    B1                  part-wise silhouette, arms, on the window frames vs D7b
    B2                  held-out camera reprojection on the frames that HAVE a third view
    B3                  the raw array bit-identical; leg demotions and rejections reported
    B4                  `delivered_vs_capture --reference raw`, D7b vs D8
    B5                  MAMMA's joints through `subject_map`, report only; the head gate
                        and the D7b neck figures rerun and reported; the D3 closure block
                        rerun with its same-denominator clause EXPECTED to report CHANGED

    merge rule, fixed before numbers: synthetic selector holds for the shipped combination
    AND the oracle passes AND B1 on both performers AND B3; everything else reports

THE ONE THING EVERY READER MUST CARRY. D8's repair sits between the raw triangulation and
the smoothed landmarks, so **everything downstream loses its byte-identical denominator by
design** and the card says so. The RAW array replaces it as the fixed reference, and two
blocks that used to assert byte-equality now EXPECT to report CHANGED and say so in their
own text: the D3 closure block's same-denominator clause, and `delivered_vs_capture`'s
landmark identity check. A CHANGED there is the step working, not a failure -- and the
thing that would be a failure, a moved RAW array, is B3 and is asserted.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d8_occlusion_gate.py

Writes `artifacts/compare/d8-occlusion/gate.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

import autoanim_gnm.commercial_multiview as cm  # noqa: E402

OUT_DIR = ROOT / "artifacts/compare/d8-occlusion"
REPORT = OUT_DIR / "gate.json"
D7B_DELIVERY = ROOT / "artifacts/commercial-multiview-soma77"
D8_DELIVERY = OUT_DIR / "delivery"
CAMERAS = ("A001", "B001", "C001", "D001")
SAMPLE_RATE_HZ = 30
WORKING = (1280, 720)
FIRST_FRAME_ID = 60
WINDOW_FIRST_ID, WINDOW_LAST_ID = 85, 125

PREREGISTRATION = """\
**instrument first (the D7b way):** `tools/compare/captured_limb_stability.py` reads the
shipped npz and the per-camera observation files and reports, per performer / landmark /
frame, segment length vs the performer's own median, camera support, per-view confidence
and reprojection residual of the raw point, ray-pair angles, NaN runs; it must reproduce
the figures above (legs +/-5 %, performer 1 forearm 18 frames, shoulder line 68-554, B/D
dropout, A-C 172 deg) before any src change

**SYNTHETIC TRUTH (selector and bands):** I7's fixture -- SOMASKEL77 posed clips projected
into THIS rig with the REAL per-camera seen pattern replayed (`replayed_keep`, so the
A-C-only window occurs by construction) and the detector's own heavy-tail noise; selector:
3D error of the landmarks in the two-view window vs truth, for the conditioning gate, the
reachability reject, both, and today's code; oracle: clean fully-seen input -> ZERO
demotions, ZERO rejections, output bit-identical; must-fail: the whole-take hold (frozen
arm) and a step test in place of reachability (the two-big-steps plateau)

**REAL TAKE, bands the candidate cannot optimise:** (B1) part-wise silhouette, arms, on the
window frames vs D7b, not worse on either performer with the CI clear and predicted to rise
on performer 1; (B2) held-out camera reprojection (the I5 pattern) on the frames that HAVE
a third view (9-22 of 41 in the window; the two-view frames have no held-out arm and are
reported as such); (B3) raw array bit-identical between builds; legs and root: number of
leg demotions/rejections REPORTED (predicted 0; the ankle has real NaN frames so a correct
fire there is not a fail); (B4) delivered hand/elbow from the file vs the RAW finite points
(`delivered_vs_capture --reference raw`), never vs the repaired points; (B5) MAMMA's joints
through `subject_map` on the window: oracle arm, report only; head gate and D7b neck figure
rerun and reported

**merge rule, fixed before numbers:** synthetic selector holds for the shipped combination
AND the oracle passes AND B1 on both performers AND B3; everything else reports\
"""


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def window_mask(frames: int) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    mask[WINDOW_FIRST_ID - FIRST_FRAME_ID:WINDOW_LAST_ID - FIRST_FRAME_ID + 1] = True
    return mask


def arrays(delivery: Path, subject: int) -> dict[str, np.ndarray]:
    with np.load(delivery / f"subject-{subject:02d}.body-track.npz") as archive:
        return {"raw": np.asarray(archive["raw_triangulated_world_positions_z_up_m"]),
                "smoothed": np.asarray(archive["triangulated_world_positions_z_up_m"])}


# --------------------------------------------------------------------------------- B2
def held_out_camera(builds: dict[str, Path]) -> dict:
    """The I5 pattern: reconstruct from three cameras, score in the fourth. Four folds.

    The 3D under test never saw the held-out camera, so its reprojection there is not a
    residual the solve minimised -- which is the whole point, and the reason this band
    survives the objection that the silhouette and the placement figures are both scored
    against things the pipeline itself produced.

    ONLY the frames that HAVE a third view can be scored. On a two-view frame, removing a
    camera leaves ONE ray and there is no reconstruction to hold anything out of. Those
    frames are counted and reported as "no held-out arm", never scored and never filled.
    """

    from autoanim_gnm.commercial_multiview import load_observation_jsonl, load_camera_rig

    rig = tuple(camera.scaled(*WORKING)
                for camera in load_camera_rig(D7B_DELIVERY / "camera-rig.json"))
    records = [load_observation_jsonl(D7B_DELIVERY / "work" / f"{name}-soma77-observations.jsonl")
               for name in CAMERAS]
    frames = len(records[0])
    window = window_mask(frames)

    # Which cameras support each slot, from the pipeline's own association.
    captured: list[np.ndarray] = []

    def recording(*args, **kwargs):
        result = cm.associate_frame_graph(*args, **kwargs)
        captured.append(np.array(result[0], copy=True))
        return result

    cm.reconstruct_multiview(rig, records, subject_count=2,
                             sample_rate_hz=SAMPLE_RATE_HZ, associator=recording)
    assigned = np.stack(captured, axis=1)
    seen = (np.isfinite(assigned[..., :2]).all(axis=-1) & (assigned[..., 2] >= 0.25))

    settings = {
        "D7b": dict(ray_pair_conditioning_ceiling_deg=None, reachability_reject=False,
                    maximum_interpolated_gap_frames=None),
        "D8": dict(),
    }
    out: dict = {
        "what": "reconstruct from three cameras, reproject the result into the fourth, "
                "score against that camera's own detections. Four folds, both builds, the "
                "SAME frames and the same landmarks.",
        "blind_to": ["DEPTH along the held-out camera's own viewing ray, by construction",
                     "a common-mode detector error, which is in both the arm and the score"],
        "folds": {},
    }
    landmarks = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist")
    columns = [cm.JOINT_INDEX[n] for n in landmarks]

    for held in range(len(CAMERAS)):
        kept = [index for index in range(len(CAMERAS)) if index != held]
        subset_rig = tuple(rig[index] for index in kept)
        subset_records = [records[index] for index in kept]
        fold: dict = {"held_out": CAMERAS[held], "builds": {}}
        # A slot is scorable when the held-out camera saw it AND at least two of the other
        # three did -- that is "has a third view", counted here rather than assumed.
        support_other = seen[:, :, kept][:, :, :, columns].sum(axis=2)
        scorable = seen[:, :, held][:, :, columns] & (support_other >= 2)
        fold["window_frames_with_a_third_view"] = {
            f"subject_{s:02d}": {landmarks[i]: int(scorable[s, window, i].sum())
                                 for i in range(len(landmarks))} for s in (0, 1)}
        fold["window_frames_without_a_held_out_arm"] = {
            f"subject_{s:02d}": {landmarks[i]: int(window.sum())
                                 - int(scorable[s, window, i].sum())
                                 for i in range(len(landmarks))} for s in (0, 1)}
        for label, kwargs in settings.items():
            _tracks, _diag, smoothed, _raw = cm.reconstruct_multiview(
                subset_rig, subset_records, subject_count=2,
                sample_rate_hz=SAMPLE_RATE_HZ, **kwargs)
            residuals = {"window": [], "whole_take": []}
            for subject in range(2):
                for position, joint in enumerate(columns):
                    for frame in range(frames):
                        if not scorable[subject, frame, position]:
                            continue
                        point = smoothed[subject, frame, joint]
                        if not np.isfinite(point).all():
                            continue
                        projected, depth = rig[held].project(point)
                        if depth <= 0.0:
                            continue
                        value = float(np.linalg.norm(
                            projected - assigned[subject, frame, held, joint, :2]))
                        residuals["whole_take"].append(value)
                        if window[frame]:
                            residuals["window"].append(value)
            fold["builds"][label] = {
                cut: ({"n": len(values),
                       "median_px1280": round(float(np.median(values)), 3),
                       "p95_px1280": round(float(np.percentile(values, 95)), 3)}
                      if values else {"n": 0})
                for cut, values in residuals.items()}
        out["folds"][CAMERAS[held]] = fold

    # The headline: pooled over folds, window only.
    pooled = {}
    for label in settings:
        medians = [out["folds"][c]["builds"][label]["window"].get("median_px1280")
                   for c in CAMERAS
                   if out["folds"][c]["builds"][label]["window"].get("n")]
        pooled[label] = round(float(np.median(medians)), 3) if medians else None
    out["window_median_of_the_four_fold_medians_px1280"] = pooled
    out["direction"] = "lower is better"
    out["verdict"] = "REPORTED -- the card does not band this arm"
    return out


# --------------------------------------------------------------------------------- B3
def raw_identity_and_leg_counts(builds: dict[str, Path]) -> dict:
    same = {}
    for subject in (0, 1):
        a = arrays(builds["D7b"], subject)["raw"]
        b = arrays(builds["D8"], subject)["raw"]
        same[f"subject_{subject:02d}"] = bool(np.array_equal(a, b, equal_nan=True))
    smoothed_same = {}
    for subject in (0, 1):
        smoothed_same[f"subject_{subject:02d}"] = bool(np.array_equal(
            arrays(builds["D7b"], subject)["smoothed"],
            arrays(builds["D8"], subject)["smoothed"]))
    run = load(builds["D8"] / "run-report.json") or {}
    diagnostics = run.get("diagnostics", run)
    repair = diagnostics.get("occlusion_repair", [])
    legs = {"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"}
    leg_counts = []
    for row in repair:
        leg_counts.append({
            "demoted_legs": {k: v for k, v in row.get("demoted_by_joint", {}).items()
                             if k in legs},
            "rejected_legs": {k: v for k, v in row.get("rejected_by_joint", {}).items()
                              if k in legs},
            "demoted_total": row.get("demoted_slots"),
            "rejected_total": row.get("rejected_slots"),
            "demoted_by_joint": row.get("demoted_by_joint"),
            "rejected_by_joint": row.get("rejected_by_joint"),
            "held_joint_fraction": row.get("held_joint_fraction"),
        })
    # The ankle's own real NaN frames, so a correct fire there can be told from a wrong one.
    ankle_nans = {}
    for subject in (0, 1):
        raw = arrays(builds["D7b"], subject)["raw"]
        ankle_nans[f"subject_{subject:02d}"] = {
            name: int((~np.isfinite(raw[:, cm.JOINT_INDEX[name]]).all(axis=1)).sum())
            for name in ("left_ankle", "right_ankle", "left_wrist", "right_wrist")}
    return {
        "band": "the RAW array bit-identical between the D7b and D8 builds; leg demotions "
                "and rejections REPORTED, predicted 0",
        "raw_bit_identical": same,
        "raw_is_the_fixed_reference": (
            "D8's three rules all sit after `world` is captured, so this is the one array "
            "that must not move. Everything downstream loses its byte-identical "
            "denominator BY DESIGN and the card says so."),
        "smoothed_bit_identical": smoothed_same,
        "smoothed_note": "EXPECTED false -- if it were true the step would have done nothing",
        "per_subject_repair": leg_counts,
        "real_nan_frames_in_the_raw_array": ankle_nans,
        "nan_note": "the ankle and wrist have real missing frames in the raw array, so a "
                    "reject that fires there is not necessarily wrong; the counts are here "
                    "so the two can be told apart",
        "verdict": ("PASS" if all(same.values())
                    and not any(row["demoted_legs"] or row["rejected_legs"]
                                for row in leg_counts)
                    else "FAIL"),
        "verdict_note": "PASS requires the raw array identical on both performers AND no "
                        "leg demoted or rejected; the leg clause was predicted to hold and "
                        "a failure of it would be reported, not moved",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--skip-held-out", action="store_true")
    args = parser.parse_args()
    builds = {"D7b": D7B_DELIVERY, "D8": D8_DELIVERY}
    for label, path in builds.items():
        if not path.exists():
            raise SystemExit(f"{label} delivery is missing at {path}")

    report: dict = {
        "title": "D8 -- the captured limbs under occlusion: every band from the card",
        "preregistration": PREREGISTRATION,
        "preregistration_source": ("docs/LADDER_EXECUTION_PLAN.md section 2, the D8 row, "
                                   "committed 2026-09-05 at 7edd23f, before this branch's "
                                   "first instrument ran"),
        "builds": {label: str(path.relative_to(ROOT)) for label, path in builds.items()},
        "delivered_sha256": {
            label: {name: digest(path / name) for name in sorted(
                f"subject-{s:02d}{suffix}" for s in (0, 1)
                for suffix in (".glb", ".body-track.json", ".body-track.npz",
                               ".mapping.npz"))}
            for label, path in builds.items()},
        "denominator": (
            "the RAW triangulated landmarks, which D8 does not touch. Up to D7b the "
            "SMOOTHED landmarks were the shared denominator because a converter change "
            "cannot move a landmark; D8 moves them by construction, so two blocks that "
            "used to assert byte-equality are EXPECTED to report CHANGED here and say so "
            "in their own text."),
        "shipped_constants": {
            "RAY_PAIR_CONDITIONING_CEILING_DEG": cm.RAY_PAIR_CONDITIONING_CEILING_DEG,
            "REACHABILITY_SLACK_M": cm.REACHABILITY_SLACK_M,
            "MAXIMUM_INTERPOLATED_GAP_FRAMES": cm.MAXIMUM_INTERPOLATED_GAP_FRAMES,
            "REACHABILITY_SPEED_CEILING_M_S": cm.REACHABILITY_SPEED_CEILING_M_S,
        },
    }

    # ------------------------------------------------------------- instrument first
    stability = load(OUT_DIR / "limb-stability.json")
    report["instrument_first"] = {
        "band": "the instrument must reproduce the card's figures BEFORE any src change",
        "report": "artifacts/compare/d8-occlusion/limb-stability.json",
        "committed_before_the_src_change": True,
        "clauses": (stability or {}).get("reproduction", {}).get("clauses"),
        "all_match": (stability or {}).get("reproduction", {}).get("all_match"),
        "legs_length_band_seed": (stability or {}).get(
            "reproduction", {}).get("legs_length_band_seed"),
        "verdict": "PASS" if (stability or {}).get("verdict") == "PASS" else "FAIL",
    }

    # ------------------------------------------------------------------- synthetic
    synthetic = load(OUT_DIR / "synthetic.json") or {}
    report["synthetic"] = {
        "band": "selector (four arms), oracle (zero demotions, zero rejections, "
                "bit-identical), and the two must-fails",
        "report": "artifacts/compare/d8-occlusion/synthetic.json",
        "selected": synthetic.get("selected"),
        "shipped": synthetic.get("shipped"),
        "shipped_equals_selected": synthetic.get("shipped_equals_selected"),
        "arms": {label: {k: v for k, v in row.items() if k != "settings"}
                 for label, row in (synthetic.get("arms") or {}).items()},
        "paired_margins": synthetic.get("paired_margins"),
        "selector_verdict": synthetic.get("selector_verdict"),
        "oracle": synthetic.get("oracle_clean_fully_seen"),
        "must_fail_frozen_arm": synthetic.get("must_fail_frozen_arm"),
        "must_fail_step_test": synthetic.get("must_fail_step_test"),
        "raw_array_untouched_on_the_fixture": synthetic.get("raw_array_untouched"),
        "selection": synthetic.get("selection"),
        "fixture_blind_to": (synthetic.get("fixture") or {}).get("blind_to"),
        "verdict": "PASS" if synthetic.get("verdict") == "PASS" else "FAIL",
    }

    # ------------------------------------------------------------------------- B1
    silhouette = load(OUT_DIR / "silhouette-partwise.json") or {}
    report["B1_silhouette_arms_on_the_window"] = {
        "band": "part-wise silhouette, arms, on the window frames vs D7b: not worse on "
                "either performer with the CI clear; predicted to rise on performer 1",
        "report": "artifacts/compare/d8-occlusion/silhouette-partwise.json",
        "clause_verdicts": silhouette.get("preregistered_clause_verdicts"),
        "subjects": silhouette.get("subjects"),
        "statistics": silhouette.get("statistics"),
        "blind_to": silhouette.get("preregistered", {}).get("what_this_cannot_settle"),
        "verdict": "PASS" if silhouette.get("verdict") == "PASS" else "FAIL",
    }

    # ------------------------------------------------------------------------- B2
    if not args.skip_held_out:
        report["B2_held_out_camera"] = held_out_camera(builds)
    else:
        report["B2_held_out_camera"] = {"verdict": "SKIPPED"}

    # ------------------------------------------------------------------------- B3
    report["B3_raw_identity_and_legs"] = raw_identity_and_leg_counts(builds)

    # ------------------------------------------------------------------------- B4
    placement = load(OUT_DIR / "delivered-vs-capture-raw.json") or {}
    hand_elbow = {}
    for subject_key, block in (placement.get("subjects") or {}).items():
        hand_elbow[subject_key] = {
            joint: block["joints"][joint]
            for joint in ("LeftHand", "RightHand", "LeftLowerArm", "RightLowerArm",
                          "LeftUpperArm", "RightUpperArm", "Neck",
                          "LeftUpperLeg", "LeftLowerLeg", "LeftFoot")
            if joint in block.get("joints", {})}
    report["B4_delivered_vs_raw_capture"] = {
        "band": "delivered hand/elbow from the file against the RAW finite points, never "
                "against the repaired ones. D7b vs D8",
        "report": "artifacts/compare/d8-occlusion/delivered-vs-capture-raw.json",
        "reference": placement.get("reference"),
        "reference_mode": placement.get("reference_mode"),
        "same_denominator": placement.get("same_denominator"),
        "joints": hand_elbow,
        "why_raw": ("scoring a D8 delivery against the SMOOTHED landmarks would score it "
                    "against the very points D8 repaired -- the candidate would be marking "
                    "its own paper. The raw array is bit-identical across the builds (B3) "
                    "and is the only honest reference here."),
        "verdict": "REPORTED -- the card does not band this arm",
    }

    # ------------------------------------------------------------------------- B5
    report["B5_reported_arms"] = {
        "band": "MAMMA's joints through `subject_map` on the window: ORACLE arm, report "
                "only. The head gate and the D7b neck figures rerun and reported. The D3 "
                "closure block rerun, with its same-denominator clause EXPECTED to report "
                "CHANGED.",
        "mamma_selects_nothing": ("CLAUDE.md: MAMMA is a measuring instrument, never in "
                                  "the shipping path. Nothing here selected anything."),
        "d3_closure_same_denominator_expected_changed": (
            "the D3 closure gate asserts the delivered GLB agrees with forward kinematics "
            "of the track it carries -- that clause is unaffected and is a real check. Its "
            "SAME-DENOMINATOR clause compares the two builds' triangulated landmarks and "
            "will report CHANGED, because D8 moves them. That is the step working."),
        "scoreboard": load(OUT_DIR / "scoreboard-d8.json"),
        "head_gate": load(OUT_DIR / "head-gate-d8.json"),
        "d3_closure": load(OUT_DIR / "d3-closure-d8.json"),
        "delivered_vs_capture_smoothed": load(
            OUT_DIR / "delivered-vs-capture-smoothed.json"),
        "verdict": "REPORTED",
    }

    # ----------------------------------------------------------------- the merge rule
    def passed(key: str) -> bool:
        return report.get(key, {}).get("verdict") == "PASS"

    b1 = report["B1_silhouette_arms_on_the_window"].get("clause_verdicts") or {}
    b1_both = all(
        (b1.get(f"subject_{s:02d}", {})
         .get("clause_1_arms_on_the_window_not_worse_than_D7b", {}).get("verdict") == "PASS")
        for s in (0, 1))
    selector_holds = bool((synthetic.get("selector_verdict") or {}).get("holds"))
    oracle_passes = bool((synthetic.get("oracle_clean_fully_seen") or {}).get("passes"))
    clauses = {
        "synthetic_selector_holds_for_the_shipped_combination": selector_holds,
        "oracle_passes": oracle_passes,
        "B1_on_both_performers": b1_both,
        "B3": passed("B3_raw_identity_and_legs"),
    }
    report["merge_rule"] = {
        "rule": "synthetic selector holds for the shipped combination AND the oracle "
                "passes AND B1 on both performers AND B3; everything else reports",
        "fixed": "before any number in this file existed, in the card at 7edd23f",
        "clauses": clauses,
        "outcome": "MERGE" if all(clauses.values()) else "DO NOT MERGE",
        "failed_clauses": [name for name, ok in clauses.items() if not ok],
    }
    report["verdict"] = report["merge_rule"]["outcome"]

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"merge_rule": report["merge_rule"]}, indent=1))
    for key in ("instrument_first", "synthetic", "B1_silhouette_arms_on_the_window",
                "B2_held_out_camera", "B3_raw_identity_and_legs",
                "B4_delivered_vs_raw_capture", "B5_reported_arms"):
        print(f"  {key:36s} {report.get(key, {}).get('verdict')}")
    print(f"\nwrote {out}")
    return 0 if report["verdict"] == "MERGE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
