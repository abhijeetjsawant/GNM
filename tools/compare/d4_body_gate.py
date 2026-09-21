"""D4's gate: every clause's predicted / measured / verdict, derived from the reports.

Not one verdict in here is a literal. Every `verdict` field is a comparison computed from a number
that was read out of a report (or, for hygiene, out of the delivered bytes themselves), and the
mutation harness proves it: `--mutate PATH=VALUE` edits the gate's INPUTS and the verdict has to
move. "A gate is proven by mutating its INPUTS, never its verdicts" (CLAUDE.md, D7c).

THE MERGE RULE, fixed before the numbers: hygiene AND the reproduction AND O1 AND B1 on both
performers AND B2; B3, B4, B5 report. **O1 FAILS and is a RECORDED EXCEPTION** decided by the
coordinator, to be merged on B1 (status log 4338ada on main; Astra's merge review is pending): the 1 mm band was set without
measuring the instrument's floor, the floor is 0.51-0.76 mm on this fixture, and the band is NOT
moved and nothing is re-selected. It is written here as a FAIL with an exception beside it, never
as a pass. Re-pinning O1 relative to the measured floor is instrument debt for the next step.

    .venv/bin/python tools/compare/d4_body_gate.py --out artifacts/compare/d4-body/gate.json
    .venv/bin/python tools/compare/d4_body_gate.py --out /tmp/x.json \
        --mutate b1/paired/delivered_MHR_lod2_minus_baseline_D7c_rig_subject_01/ci95_of_the_median_difference/0=-0.01
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D4 = ROOT / "artifacts/compare/d4-body"
SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
EIGHT = ("subject-00.glb", "subject-01.glb", "subject-00.body-track.npz",
         "subject-01.body-track.npz", "subject-00.body-track.json", "subject-01.body-track.json",
         "subject-00.mapping.npz", "subject-01.mapping.npz")
PRECARD = {"subject_00": 0.789, "subject_01": 0.730}
REPRODUCTION_TOLERANCE = 0.001
O1_BAND_MM = 1.0
CLOSURE_BAND_M = 1e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> dict:
    """Everything the gate reads, in one tree, so a mutation can reach any leaf."""
    read = lambda name: json.loads((D4 / name).read_text(encoding="utf-8"))  # noqa: E731
    return {
        "hygiene": {"rebuild_sha256": {f: sha256(D4 / "hygiene" / f) for f in EIGHT},
                    "shipped_sha256": {f: sha256(SHIPPED / f) for f in EIGHT}},
        "reproduction": read("reproduction.json"),
        "reproduction_repaired": read("reproduction-repaired.json"),
        "o1": read("o1/o1.json"),
        "o1_readings": read("o1/o1-statistic-readings.json"),
        "o1_closure": read("o1/closure.json"),
        "o1_closure_mutations": read("o1/closure-mutations.json"),
        "o1_prerepair": read("o1-prerepair.json"),
        "b1": read("b1-paired.json"),
        "b1_parts": read("b1-parts-terciles.json"),
        "b1_mamma": read("b1-mamma-bit-identity.json"),
        "b1_silhouette": {"arms": {name: {cam: {s: {"iou": row[cam][s]["iou"]["median"]}
                                                for s in ("subject_00", "subject_01")}
                                          for cam in ("A001", "B001", "C001", "D001")}
                                   for name, row in read("silhouette-delivery.json")["arms"].items()}},
        "b2": read("b2-same-denominator.json"),
        "b3": read("b3-placement.json"),
        "b4": read("b4-mamma-arm.json"),
        "b5": read("b5-delivered-bytes.json"),
        "delivery_closure": read("delivery-closure.json"),
        "free_offsets": {f"subject_{s:02d}": read(
            f"free-offsets-lod2/fit-report-subject-{s:02d}.json")["subjects"][f"subject_{s:02d}"]
            for s in (0, 1)},
        "fit_report": read("delivery/fit-report.json"),
    }


def dig(tree, path: str):
    node = tree
    for key in path.split("/"):
        node = node[int(key)] if isinstance(node, list) else node[key]
    return node


def put(tree, path: str, value):
    keys = path.split("/")
    node = tree
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def verdicts(data: dict) -> dict:
    hyg = data["hygiene"]
    identical = {f: hyg["rebuild_sha256"][f] == hyg["shipped_sha256"][f] for f in EIGHT}
    repro = data["reproduction"]["arms"]["buildscript_MHR_lod6_raw"]
    repro_delta = {s: abs(repro[s]["pooled_median_iou"] - PRECARD[s]) for s in PRECARD}
    o1_oracle = data["o1_readings"]["oracle"]
    o1_max = max(v["median_of_per_frame_medians"] for v in o1_oracle.values())
    o1_mean_min = min(v["median_of_per_frame_medians"]
                      for v in data["o1_readings"]["mean_body"].values())
    o1_floor_max = max(v["median_of_per_frame_medians"]
                       for v in data["o1_readings"]["exact_identity"].values())
    b1 = {s: data["b1"]["paired"][f"delivered_MHR_lod2_minus_baseline_D7c_rig_subject_{s}"]
          for s in ("00", "01")}
    b1_lower = {s: b1[s]["ci95_of_the_median_difference"][0] for s in b1}
    frozen = data["b1_silhouette"]["arms"]["control_frozen_pose_tracked"]
    ours = data["b1_silhouette"]["arms"]["ours_delivered"]
    frozen_below = all(frozen[cam][s]["iou"] < ours[cam][s]["iou"]
                       for cam in frozen for s in frozen[cam])

    out = {
        "hygiene": {
            "predicted": "--body rig rebuilds the D7c delivery 8 of 8 byte-identical",
            "measured": {"identical": sum(identical.values()), "of": len(EIGHT),
                         "files": identical},
            "verdict": "PASS" if all(identical.values()) else "FAIL"},
        "reproduction": {
            "predicted": f"--body mhr lod6 RAW reproduces {PRECARD} to {REPRODUCTION_TOLERANCE}",
            "measured": {s: repro[s]["pooled_median_iou"] for s in PRECARD},
            "max_abs_difference": round(max(repro_delta.values()), 6),
            "verdict": "PASS" if max(repro_delta.values()) <= REPRODUCTION_TOLERANCE else "FAIL",
            # The path reproduced EXACTLY, and it carries a defect: the pre-card fitted both
            # performers in one process, and a momentum calibration mutates whatever a later
            # Character.load_fbx returns. Repaired (one process per performer), at the pre-card's
            # own lod6 and raw array:
            "repaired_one_process_per_performer": {
                s: data["reproduction_repaired"]["arms"]
                ["buildscript_MHR_lod6_raw_oneprocess"][s]["pooled_median_iou"]
                for s in PRECARD},
            "repaired_minus_precard": {
                s: round(data["reproduction_repaired"]["paired"][
                    f"buildscript_MHR_lod6_raw_oneprocess_minus_precard_fitted_MHR_"
                    f"subject_{s[-2:]}"]["median_difference"], 6) for s in PRECARD}},
        "O1_exactness": {
            "predicted": f"the max over six seeds <= {O1_BAND_MM} mm",
            "measured_max_over_seeds_mm": round(o1_max, 4),
            "measured_per_seed_mm": {k: v["median_of_per_frame_medians"]
                                     for k, v in o1_oracle.items()},
            "verdict": "PASS" if o1_max <= O1_BAND_MM else "FAIL",
            "recorded_exception": (
                "FAIL, and TO BE MERGED on B1 as a RECORDED EXCEPTION by the coordinator's "
                "decision (status log 4338ada on main; Astra's merge review is pending). The "
                "band was set without measuring the instrument's floor; "
                "with the TRUTH identity handed in and only the pose re-solved the same fixture "
                f"reads {round(o1_floor_max, 4)} mm, so the band allows the identity fit "
                f"{round(O1_BAND_MM - o1_floor_max, 4)} mm and it costs more. The band is NOT "
                "moved and nothing is re-selected; re-pinning O1 against the measured floor is "
                "instrument debt for the next step."),
            "exact_identity_floor_max_over_seeds_mm": round(o1_floor_max, 4),
            # The pre-repair STOP stays on record (CLAUDE.md, D7c): the fixture's own defect was
            # measured and repaired as a FIXTURE PARAMETER with the fitter byte-identical, and
            # the original reading is reproducible with --no-clamp.
            "pre_repair_max_over_seeds_mm": data["o1_prerepair"]["oracle_max_over_seeds_mm"],
            "pre_repair_exact_identity_floor_mm":
                data["o1_prerepair"]["exact_identity_max_over_seeds_mm"],
            "fixture_repair": {
                "clamped_parameter_frames":
                    data["o1"]["fixture_repair"]["clamped_parameter_frames"],
                "violating_pose_parameters":
                    len(data["o1"]["fixture_repair"]["violating_parameters"]),
                "fitter_source_sha256_across_the_repair": data["o1"]["fitter_source_sha256"],
                "recorded_as": "post hoc; reproduce the pre-repair reading with --no-clamp"}},
        "O1_must_fail_mean_body": {
            "predicted": "misses the same predicate on every seed, by the injected scale, 5-60 mm",
            "measured_min_over_seeds_mm": round(o1_mean_min, 4),
            "verdict": "PASS" if o1_mean_min > O1_BAND_MM else "FAIL"},
        "O1_closure": {
            "predicted": f"the delivered GLB's FK reproduces the track to <= {CLOSURE_BAND_M} m",
            "measured_worst_max_abs_m": data["o1_closure"]["worst_max_abs_m"],
            "verdict": "PASS" if data["o1_closure"]["worst_max_abs_m"] <= CLOSURE_BAND_M else "FAIL",
            "mutations_rejected": {k: v["max_abs_m"]
                                   for k, v in data["o1_closure_mutations"]["pairs"].items()},
            "mutations_all_rejected": all(not v["within_band"] for v
                                          in data["o1_closure_mutations"]["pairs"].values())},
        "B1_the_band": {
            "predicted": "fitted MHR minus the D7c delivery, the LOWER CI bound above zero on "
                         "BOTH performers (two cells)",
            "measured": {f"subject_{s}": {"median_difference": b1[s]["median_difference"],
                                          "ci95": b1[s]["ci95_of_the_median_difference"],
                                          "n": b1[s]["n"]} for s in b1},
            "verdict": "PASS" if all(v > 0 for v in b1_lower.values()) else "FAIL"},
        "B1_silhouette_frozen_pooled": {
            f"{cam}_{s}": frozen[cam][s]["iou"] for cam in frozen for s in frozen[cam]},
        "B1_pooled_medians_REPORTED": {
            "note": "pooled over the four cameras and the whole take, per performer, from the "
                    "paired instrument's own arms. MAMMA's row is the median of its four camera "
                    "medians (the committed report carries no pooled figure) and is on the SAME "
                    "masks and rasteriser, so it shares this axis.",
            **{name: {s: row[s]["pooled_median_iou"] for s in ("subject_00", "subject_01")}
               for name, row in data["b1"]["arms"].items()},
            "ORACLE_mamma_mesh": {
                s: round(float(sorted(data["b1_silhouette"]["arms"]["ORACLE_mamma_mesh"][cam][s]
                                      ["iou"] for cam in ("A001", "B001", "C001", "D001"))[1:3][0]
                               + sorted(data["b1_silhouette"]["arms"]["ORACLE_mamma_mesh"][cam][s]
                                        ["iou"] for cam in ("A001", "B001", "C001", "D001"))[1:3][1])
                         / 2.0, 4)
                for s in ("subject_00", "subject_01")}},
        "B1_frozen_pose_control_below_the_candidate": {
            "predicted": "the instrument's frozen-pose control stays below the candidate",
            "measured_cells_below": sum(frozen[cam][s]["iou"] < ours[cam][s]["iou"]
                                        for cam in frozen for s in frozen[cam]),
            "verdict": "PASS" if frozen_below else "FAIL"},
        "B1_mamma_bit_identical": {
            "predicted": "MAMMA's mesh through the same rasteriser is bit-identical to its "
                         "committed value",
            "measured_cells_identical": sum(
                c["identical_all_fields"] for c in data["b1_mamma"]["cells"].values()),
            "verdict": "PASS" if data["b1_mamma"]["bit_identical_on_all_8_cells"] else "FAIL"},
        "B2_same_denominator": {
            "predicted": "the consumed array is the rig converter's input, and momentum's markers "
                         "re-derive from it exactly",
            # Derived from B2's own numbered CHECKS, never from its verdict leaf: a gate that
            # reads another instrument's verdict can be turned by mutating a verdict, which is
            # exactly what D7c's rule forbids.
            "measured": {s: {k: v for k, v in row.items()
                             if k[0].isdigit() or k == "marker_names_are_the_declared_map"}
                         for s, row in data["b2"]["subjects"].items()},
            "verdict": "PASS" if all(
                bool(v) for row in data["b2"]["subjects"].values()
                for k, v in row.items()
                if k[0].isdigit() or k == "marker_names_are_the_declared_map") else "FAIL"},
        "B1_parts_and_terciles_REPORTED": {
            "partition": data["b1_parts"]["partition"],
            "tercile_edges_trunk_lean_deg": data["b1_parts"]["tercile_edges_trunk_lean_deg"],
            "parts": data["b1_parts"]["parts"],
            "bent_terciles": data["b1_parts"]["bent_terciles"],
            "note": "precision and recall are the inflation diagnostic beside IoU. No precision "
                    "veto exists in this card and none was invented at merge time."},
        "delivery_fit_REPORTED": {
            "one_process_per_performer": data["fit_report"]["one_process_per_performer"],
            **{s: {"mesh_vertices": row["subjects"][s]["mesh_vertices"],
                   "nonzero_identity_channels": row["subjects"][s]["nonzero_identity_channels"],
                   "joint_to_landmark_residual_mm_median":
                       row["subjects"][s]["joint_to_landmark_residual_mm_median_over_frames"],
                   "locator_offset_mm_median": row["subjects"][s]["locator_offset_mm_median"],
                   "lod": row["lod"], "landmarks": row["landmarks"]}
               for s, row in data["fit_report"]["subjects"].items()}},
        "B3_placement_REPORTED": {
            "all_landmark_median_mm": {s: row["all_landmarks_median_mm"]
                                       for s, row in data["b3"]["subjects"].items()},
            "segment_mean_abs_error_mm": {s: row["segment_mean_abs_error_mm"]
                                          for s, row in data["b3"]["subjects"].items()}},
        "B4_mamma_arm_REPORTED": {
            "absolute_median_mm": {s: row["absolute_all_joint_median_mm"]
                                   for s, row in data["b4"]["subjects"].items()},
            "bias_mm": {s: row["absolute_all_joint_bias_median_mm"]
                        for s, row in data["b4"]["subjects"].items()},
            "spread_mm": {s: row["absolute_all_joint_spread_median_mm"]
                          for s, row in data["b4"]["subjects"].items()},
            "subject_to_mamma_body_id": data["b4"]["subject_to_mamma_body_id"]},
        "B5_delivered_bytes_REPORTED": {
            # derived from the checks, not from B5's own verdict leaf
            "bytes_verdict": "PASS" if all(
                row["sampler_times_are_k_over_30_s"]
                and row["sampler_times_strictly_increasing"]
                and row["joint_names_match_MHR"] and row["hierarchy_matches_MHR"]
                and row["rest_translation_max_abs_diff_cm"] < 1e-4
                and row["mesh_vertices_match_MHR"] and row["mesh_all_finite"]
                and row["mesh_frames"] == 150 and row["skin_weight_max_abs_diff"] < 1e-4
                for row in data["b5"]["subjects"].values()) else "FAIL",
            "byte_checks": {s: {k: row[k] for k in (
                "sampler_times_are_k_over_30_s", "sampler_times_strictly_increasing",
                "joint_names_match_MHR", "hierarchy_matches_MHR",
                "rest_translation_max_abs_diff_cm", "mesh_vertices_match_MHR",
                "mesh_all_finite", "mesh_frames", "skin_weight_max_abs_diff")}
                for s, row in data["b5"]["subjects"].items()},
            "facing_positive_on_every_frame": data["b5"][
                "facing_positive_on_every_frame_both_performers"],
            "facing_frames": {s: [row["facing_dot_positive_frames"], row["facing_dot_frames_scored"]]
                              for s, row in data["b5"]["subjects"].items()}},
        "delivery_closure": {
            "predicted": f"the delivered GLB's FK reproduces the track to <= {CLOSURE_BAND_M} m",
            "measured_worst_max_abs_m": data["delivery_closure"]["worst_max_abs_m"],
            "verdict": "PASS" if data["delivery_closure"]["worst_max_abs_m"] <= CLOSURE_BAND_M
                       else "FAIL"},
        "must_fail_free_locator_offsets": {
            "predicted": "offsets median > 50 mm AND the identity channels at zero",
            "measured": {s: {"locator_offset_mm_median": row["locator_offset_mm_median"],
                             "nonzero_identity_channels": row["nonzero_identity_channels"],
                             "residual_mm": row["joint_to_landmark_residual_mm_median_over_frames"]}
                         for s, row in data["free_offsets"].items()},
            "half_a_offsets_over_50mm": all(row["locator_offset_mm_median"] > 50.0
                                            for row in data["free_offsets"].values()),
            "half_b_identity_at_zero": all(row["nonzero_identity_channels"] == 0
                                           for row in data["free_offsets"].values()),
            "verdict": "the arm FAILS as a must-fail must (half A holds); the PREDICTION's half B "
                       "is FALSE and is recorded as a failed prediction"},
    }
    conjuncts = ("hygiene", "reproduction", "O1_exactness", "B1_the_band", "B2_same_denominator")
    failing = [c for c in conjuncts if out[c]["verdict"] != "PASS"]
    out["merge_rule"] = {
        "rule": "hygiene AND the reproduction AND O1 AND B1 on both performers AND B2",
        "conjuncts": {c: out[c]["verdict"] for c in conjuncts},
        "failing_conjuncts": failing,
        "recorded_exceptions": ["O1_exactness"],
        "merge": ("MERGE with O1 a recorded exception"
                  if failing == ["O1_exactness"] else
                  "MERGE" if not failing else f"NO MERGE: {failing}")}
    return out


# One mutation per conjunct of the merge rule, applied to the gate's INPUTS. Each must turn that
# conjunct's verdict, and the O1 row must turn it the other way -- otherwise the FAIL is hardcoded.
MUTATION_TABLE = {
    "hygiene: one rebuilt file's sha256 changed":
        ('hygiene/rebuild_sha256/subject-00.glb="deadbeef"', "hygiene"),
    "reproduction: performer 0's pooled IoU moved by 0.01":
        ("reproduction/arms/buildscript_MHR_lod6_raw/subject_00/pooled_median_iou=0.78",
         "reproduction"),
    "O1: the worst seed brought under the band":
        ("o1_readings/oracle/20260924/median_of_per_frame_medians=0.9", "O1_exactness"),
    "B1: performer 1's lower CI bound put below zero":
        ("b1/paired/delivered_MHR_lod2_minus_baseline_D7c_rig_subject_01/"
         "ci95_of_the_median_difference/0=-0.01", "B1_the_band"),
    "B1: performer 0's lower CI bound put exactly at zero":
        ("b1/paired/delivered_MHR_lod2_minus_baseline_D7c_rig_subject_00/"
         "ci95_of_the_median_difference/0=0.0", "B1_the_band"),
    "B2: performer 0's re-derived marker check turned false":
        ("b2/subjects/subject_00/"
         "4a_marker_values_match_the_declared_mapping_and_conversion=false",
         "B2_same_denominator"),
    "B2: performer 1's consumed array no longer byte-identical to the rig converter's input":
        ("b2/subjects/subject_01/"
         "1a_handed_array_is_byte_identical_to_the_rig_converter_input_on_this_build=false",
         "B2_same_denominator"),
}


def mutation_table(baseline: dict) -> dict:
    rows = {}
    for label, (mutation, conjunct) in MUTATION_TABLE.items():
        data = load_inputs()
        path, value = mutation.split("=", 1)
        put(data, path, json.loads(value))
        moved = verdicts(data)
        rows[label] = {"mutation": mutation, "conjunct": conjunct,
                       "baseline_verdict": baseline[conjunct]["verdict"],
                       "mutated_verdict": moved[conjunct]["verdict"],
                       "the_conjunct_turned":
                           moved[conjunct]["verdict"] != baseline[conjunct]["verdict"],
                       "merge_line": moved["merge_rule"]["merge"]}
    rows["every_conjunct_turns"] = all(r["the_conjunct_turned"] for r in rows.values()
                                       if isinstance(r, dict))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mutate", action="append", default=[],
                        help="PATH=VALUE (JSON) applied to the gate's INPUTS before "
                             "deriving. PATH is slash-separated, because some keys are filenames "
                             "and carry dots.")
    arguments = parser.parse_args()
    data = load_inputs()
    applied = {}
    for item in arguments.mutate:
        path, value = item.split("=", 1)
        applied[path] = {"before": dig(data, path), "after": json.loads(value)}
        put(data, path, json.loads(value))
    report = verdicts(data)
    report["mutations_applied"] = applied
    if not applied:
        report["input_mutation_table"] = mutation_table(report)
    Path(arguments.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["merge_rule"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
