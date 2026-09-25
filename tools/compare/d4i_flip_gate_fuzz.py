"""D4i: the gate's fuzz. Every mutation changes the UNDERLYING EVIDENCE -- a delivered file's bytes, an instrument's
own report, an entry report, the roster, the tree -- on a COPY, and re-runs `d4i_flip_gate.verdicts` with that one
path pointed at the copy. No mutation ever supplies or edits a PASS flag the gate reads as a conclusion; where a
mutation edits a verdict field, it is the INSTRUMENT'S own field, which is the evidence the card says to read.

Outcome classes, kept apart:
  TURNED   the targeted leg was True and the mutation made it False;
  UNMOVED  it did not (a failure of the gate, unless the case is a negative control);
  CRASH    the gate could not evaluate the conjunct (it fails closed, and it is counted as its own class, never
           as TURNED);
Negative controls must leave every conjunct's legs exactly as they were. The COUNTERFACTUAL is reported, not
scored: with D4d's own track-JSON bytes in place of the default's, does the oracle hold? (It shows the FAIL is those
two files and nothing else; it is not a normalisation admitted into the verdict.)

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4i_flip_gate_fuzz.py --out artifacts/compare/d4i-flip/fuzz.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/compare"))
import d4i_flip_gate as gate  # noqa: E402

SCRATCH = ROOT / "artifacts/compare/d4i-flip/fuzz-scratch"


def clone(key: str, tag: str) -> Path:
    src = Path(gate.PATHS[key])
    dst = SCRATCH / tag / src.name
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    dst.parent.mkdir(parents=True)
    subprocess.run(["cp", "-Rc" if src.is_dir() else "-c", str(src), str(dst)], check=True)
    return dst


def scratch_worktree(tag: str) -> Path:
    root = SCRATCH / tag / "worktree"
    if root.parent.exists():
        shutil.rmtree(root.parent)
    (root / "tools/fitter").mkdir(parents=True)
    (root / "tools/compare").mkdir(parents=True)
    subprocess.run(["cp", "-Rc", str(ROOT / "src"), str(root / "src")], check=True)
    for rel in ("tools/fitter/mhr_delivery.py", "tools/compare/post_merge.sh", "tools/compare/ladder.py"):
        subprocess.run(["cp", "-c", str(ROOT / rel), str(root / rel)], check=True)
    return root


def flip_byte(path: Path, at: int = -100) -> None:
    raw = bytearray(path.read_bytes())
    raw[at] ^= 0x01
    path.write_bytes(bytes(raw))


def edit_json(path: Path, fn: Callable[[Any], None]) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data, indent=1))


def leg(conjunct: str, name: str) -> Callable[[dict], bool]:
    return lambda r: r["conjuncts"][conjunct]["legs"][name]


def file_leg(name: str) -> Callable[[dict], bool]:
    return lambda r: r["conjuncts"]["oracle"]["files"][name]["byte_identical"]


# (id, conjunct, target extractor, the evidence key, the mutation on the copy)
def mutations() -> list[tuple[str, str, Callable, str, Callable[[Path], None]]]:
    fr = lambda p, name: p / name
    out = [
        ("oracle: a GLB byte", "oracle", file_leg("subject-00.glb"), "default",
         lambda p: flip_byte(p / "subject-00.glb")),
        ("oracle: a markers byte", "oracle", file_leg("subject-01.markers.npz"), "default",
         lambda p: flip_byte(p / "subject-01.markers.npz")),
        ("oracle: the calibration pass count", "oracle", file_leg("subject-00.calibration-passes.json"), "default",
         lambda p: edit_json(p / "subject-00.calibration-passes.json", lambda d: d.__setitem__("passes", 1))),
        ("oracle: the track npz's mean body (the roster cannot see it; the oracle must)", "oracle",
         file_leg("subject-00.body-track.npz"), "default", lambda p: _resave_rest(p / "subject-00.body-track.npz")),
        ("oracle: a delivered file missing", "oracle", leg("oracle", "manifest_equal"), "default",
         lambda p: (p / "fit-report.json").unlink()),
        ("oracle: a converter-input byte", "oracle", leg("oracle", "reference_hashes_equal"), "default",
         lambda p: flip_byte(p / "converter-inputs/subject-00-consumed.npz")),
        ("oracle: the silhouette's identity block", "oracle", leg("oracle", "subject_identities_equal"), "final_roster",
         lambda p: edit_json(p / "silhouette.instrument.json", lambda d: d["identity"]["our_subject_to_body_id"]
                             .__setitem__("subject_00", "body_id-00"))),
        ("oracle: B4's subject map crossed", "oracle", leg("oracle", "subject_identities_equal"), "final_roster",
         lambda p: edit_json(p / "b4.instrument.json", lambda d: d.__setitem__("subject_to_mamma_body_id",
                                                                              {"0": 0, "1": 1}))),
        ("oracle: B1's seed", "oracle", leg("oracle", "draws_equal"), "final_roster",
         lambda p: edit_json(p / "b1.instrument.json", lambda d: d.__setitem__("seed", 1))),
        ("oracle: one B1 paired row", "oracle", leg("oracle", "scored_rows_equal"), "final_roster",
         lambda p: edit_json(p / "b1.instrument.json", lambda d: d["paired"][
             "D4d_fitted_MHR_lod2_minus_D4c_fitted_MHR_lod2_subject_01"].__setitem__("median_difference", 0.0))),
        ("oracle: one silhouette figure", "oracle", leg("oracle", "scored_rows_equal"), "final_roster",
         lambda p: edit_json(p / "silhouette.instrument.json", lambda d: d["arms"]["ours_delivered"]["A001"]
                             ["subject_00"]["iou"].__setitem__("median", 0.5))),
        ("oracle: the closure residual", "oracle", leg("oracle", "closure_reproduced"), "final_roster",
         lambda p: edit_json(p / "closure.instrument.json", lambda d: d.__setitem__("worst_max_abs_m", 1e-3))),
        ("oracle: a B2 check", "oracle", leg("oracle", "b2_reproduced"), "final_roster",
         lambda p: edit_json(p / "b2.instrument.json", lambda d: d["subjects"]["subject_00"]
                             .__setitem__("3_frame_order_identical", False))),
        ("oracle: a B5 figure", "oracle", leg("oracle", "b5_reproduced"), "final_roster",
         lambda p: edit_json(p / "b5.instrument.json", lambda d: d["subjects"]["subject_00"]
                             .__setitem__("skin_weight_max_abs_diff", 0.5))),
        ("B1: the band from the rows", "B1_reproduced_exactly", leg("B1_reproduced_exactly", "band_from_the_rows"),
         "final_roster", lambda p: edit_json(p / "b1.instrument.json", lambda d: d["paired"][
             "D4d_fitted_MHR_lod2_minus_baseline_D7c_rig_subject_01"]["ci95_of_the_median_difference"]
             .__setitem__(0, -0.01))),
        ("B1: the entry's verdict", "B1_reproduced_exactly", leg("B1_reproduced_exactly", "entry_verdict_PASS"),
         "final_roster", lambda p: edit_json(p / "b1.json", lambda d: d.__setitem__("verdict", "FAIL"))),
        ("B1: a cached (not fresh) mesh", "B1_reproduced_exactly",
         leg("B1_reproduced_exactly", "mesh_is_a_fresh_bound_export_of_this_delivery"), "final_roster",
         lambda p: edit_json(p / "silhouette.json", lambda d: d["mesh_cache"].__setitem__("fresh_export", False))),
        ("B1: a mesh bound to other GLBs", "B1_reproduced_exactly",
         leg("B1_reproduced_exactly", "mesh_is_a_fresh_bound_export_of_this_delivery"), "final_roster",
         lambda p: edit_json(p / "silhouette.json", lambda d: d["mesh_cache"]["binding"]["glb_sha256"]
                             .__setitem__("subject-00.glb", "0" * 64))),
        ("B2: the instrument's own verdict", "B2_PASS", leg("B2_PASS", "instrument_verdict"), "final_roster",
         lambda p: edit_json(p / "b2.instrument.json", lambda d: d.__setitem__("verdict", "FAIL"))),
        ("closure: the instrument's own band", "closure_PASS", leg("closure_PASS", "instrument_all_within_band"),
         "final_roster", lambda p: edit_json(p / "closure.instrument.json",
                                             lambda d: d.__setitem__("all_within_band", False))),
        ("hygiene: a rig-arm byte", "hygiene", leg("hygiene", "rig_arm_8_of_8_against_the_shipped_D7c"), "rig_arm",
         lambda p: flip_byte(p / "subject-00.mapping.npz")),
        ("hygiene: a src byte", "hygiene", leg("hygiene", "src_and_the_fitter_byte_identical_to_the_base"), "worktree",
         lambda p: _append(p / "src/autoanim_gnm/body.py", b"\n")),
        ("hygiene: a file added under src", "hygiene", leg("hygiene", "src_and_the_fitter_byte_identical_to_the_base"),
         "worktree", lambda p: (p / "src/autoanim_gnm/new_module.py").write_text("x = 1\n")),
        ("hygiene: the fitter", "hygiene", leg("hygiene", "src_and_the_fitter_byte_identical_to_the_base"), "worktree",
         lambda p: _append(p / "tools/fitter/mhr_delivery.py", b"# edit\n")),
        ("hygiene: an i6 byte", "hygiene", leg("hygiene", "i6_baseline_unchanged"), "i6",
         lambda p: flip_byte(p / "mean-body-00.npy")),
        ("must-fail (i): a scope refusal that ran instead", "must_fails_i_to_v",
         leg("must_fails_i_to_v", "every_entry_report_turned_or_stayed_as_required"), "mustfail",
         lambda p: edit_json(p / "i-scope/rig-build-b3.json", lambda d: d.__setitem__("verdict", "PASS"))),
        ("must-fail (iii): the negative control moved a figure", "must_fails_i_to_v",
         leg("must_fails_i_to_v", "every_entry_report_turned_or_stayed_as_required"), "mustfail",
         lambda p: edit_json(p / "iii-negative/b3.instrument.json", lambda d: d["subjects"]["subject_00"]
                             .__setitem__("all_landmarks_median_mm", 1.0))),
        ("must-fail (population): a report missing", "must_fails_i_to_v",
         leg("must_fails_i_to_v", "every_entry_report_turned_or_stayed_as_required"), "mustfail",
         lambda p: (p / "population/truncated-b3.json").unlink()),
        ("must-fail (v): the positive control did not move B3", "must_fails_i_to_v",
         leg("must_fails_i_to_v", "every_entry_report_turned_or_stayed_as_required"), "mustfail",
         lambda p: shutil.copyfile(gate.PATHS["final_roster"] / "b3.instrument.json",
                                   p / "v-positive/b3.instrument.json")),
        ("must-fail (ii): a record-read case", "must_fails_i_to_v", leg("must_fails_i_to_v", "record_read_cases_held"),
         "mustfail_record", lambda p: edit_json(p, lambda d: d["ii_mixed"]["the_default_build_refuses_a_stale_mapping"]
                                               .__setitem__("held", False))),
        ("population: a mismatch in B3's entry", "population_and_own_verdicts",
         leg("population_and_own_verdicts", "every_entry"), "final_roster",
         lambda p: edit_json(p / "b3.json", lambda d: d["population"]["mismatches"].append("subject_00/capture_frames"))),
        ("population: an entry run on another delivery", "population_and_own_verdicts",
         leg("population_and_own_verdicts", "every_entry"), "final_roster",
         lambda p: edit_json(p / "b5.json", lambda d: d.__setitem__("delivery", str(gate.PATHS["d4d_delivery"])))),
        ("population: an entry without its own verdict field", "population_and_own_verdicts",
         leg("population_and_own_verdicts", "every_entry"), "final_roster",
         lambda p: edit_json(p / "b4.json", lambda d: d.__setitem__("instrument_verdict_field", {}))),
        ("population: the recorded run disagrees", "population_and_own_verdicts",
         leg("population_and_own_verdicts", "the_recorded_run_agrees_with_the_entry_reports"), "final_roster_record",
         lambda p: edit_json(p, lambda d: d["entries"]["b4"].__setitem__("value", "FAIL"))),
        ("roster: a missing post-migration evidence", "roster_complete",
         leg("roster_complete", "every_instrument_has_all_three_fields"), "roster",
         lambda p: edit_json(p, lambda d: d["instruments"]["b3"]["post_migration"].__setitem__("evidence", ""))),
        ("roster: a (d) instrument kept", "roster_complete", leg("roster_complete", "every_instrument_has_all_three_fields"),
         "roster", lambda p: edit_json(p, lambda d: d["instruments"]["head_gate"]["final_action"]
                                      .__setitem__("action", "keep"))),
        ("coordinator: post_merge.sh reverted", "coordinator_migration_recorded",
         leg("coordinator_migration_recorded", "post_merge_sh_changed_on_the_branch"), "worktree",
         lambda p: (p / "tools/compare/post_merge.sh").write_bytes(gate.git_bytes("tools/compare/post_merge.sh"))),
        ("coordinator: the recorded run not all PASS", "coordinator_migration_recorded",
         leg("coordinator_migration_recorded", "final_roster_run_recorded"), "final_roster_record",
         lambda p: edit_json(p, lambda d: d.__setitem__("mhr_all_pass", False))),
        ("coordinator: the recorded run log missing", "coordinator_migration_recorded",
         leg("coordinator_migration_recorded", "final_roster_run_recorded"), "final_roster_log",
         lambda p: p.unlink()),
    ]
    return out


def _resave_rest(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["rest_positions_z_up_m"] = arrays["rest_positions_z_up_m"] + np.float32(0.01)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _append(path: Path, data: bytes) -> None:
    path.write_bytes(path.read_bytes() + data)


CRASHES = [
    ("crash: B1's report unreadable", "B1_reproduced_exactly", "final_roster",
     lambda p: (p / "b1.instrument.json").write_text("{not json")),
    ("crash: the closure's report missing", "closure_PASS", "final_roster",
     lambda p: (p / "closure.instrument.json").unlink()),
    ("crash: the roster unreadable", "roster_complete", "roster", lambda p: p.write_text("")),
]
NEGATIVE = [
    ("negative: every delivered file's mtime moved", "default",
     lambda p: [os.utime(f, (1, 1)) for f in p.rglob("*") if f.is_file()]),
    ("negative: the review page (not a per-subject file) edited", "default",
     lambda p: _append(p / "review.html", b"<!-- -->")),
    ("negative: a line appended to the recorded run's log", "final_roster_log", lambda p: _append(p, b"\n# note\n")),
    ("negative: B3's REPORTED figures in the ENTRY report (not the instrument's) edited", "final_roster",
     lambda p: edit_json(p / "b3.json", lambda d: d["reported"]["subject_00"].__setitem__("all_landmarks_median_mm", 0))),
]


def legs_of(report: dict) -> dict:
    return {n: (c.get("legs"), "crash" in c) for n, c in report["conjuncts"].items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = gate.verdicts()
    base_legs = legs_of(baseline)
    rows = []
    for i, (name, conjunct, target, key, mutate) in enumerate(mutations()):
        tag = f"m{i:02d}"
        path = scratch_worktree(tag) if key == "worktree" else clone(key, tag)
        mutate(path)
        report = gate.verdicts({key: path})
        c = report["conjuncts"][conjunct]
        before = target(baseline)
        try:
            after = target(report)
            outcome = ("CRASH" if "crash" in c else "TURNED" if (before is True and after is False) else "UNMOVED")
        except Exception as error:
            after, outcome = f"{type(error).__name__}: {error}", "CRASH"
        others = [n for n, v in legs_of(report).items() if n != conjunct and v != base_legs[n]]
        rows.append({"mutation": name, "conjunct": conjunct, "evidence": key, "before": before, "after": after,
                     "outcome": outcome, "conjunct_holds_after": c["holds"], "verdict_after": report["verdict"],
                     "other_conjuncts_moved": others})
        print(f"{outcome:8s} {name}" + (f"   (also moved: {others})" if others else ""))
    crashes = []
    for i, (name, conjunct, key, mutate) in enumerate(CRASHES):
        path = clone(key, f"c{i:02d}")
        mutate(path)
        report = gate.verdicts({key: path})
        c = report["conjuncts"][conjunct]
        crashes.append({"mutation": name, "conjunct": conjunct, "class": "CRASH" if "crash" in c else "NOT A CRASH",
                        "fails_closed": c["holds"] is False, "verdict_after": report["verdict"], "crash": c.get("crash")})
        print(f"{crashes[-1]['class']:8s} {name}")
    negatives = []
    for i, (name, key, mutate) in enumerate(NEGATIVE):
        path = clone(key, f"n{i:02d}")
        mutate(path)
        report = gate.verdicts({key: path})
        moved = [n for n, v in legs_of(report).items() if v != base_legs[n]]
        negatives.append({"mutation": name, "moved": moved, "unmoved": not moved})
        print(f"{'UNMOVED' if not moved else 'MOVED':8s} {name}")
    swap = clone("default", "counterfactual")
    for s in (0, 1):
        shutil.copyfile(gate.PATHS["d4d_delivery"] / f"subject-{s:02d}.body-track.json",
                        swap / f"subject-{s:02d}.body-track.json")
    counter = gate.verdicts({"default": swap})["conjuncts"]["oracle"]
    counterfactual = {"what": "the default's two track JSONs replaced by D4d's bytes (the ONLY difference being "
                              "body_model.assets); REPORTED, never admitted into the verdict",
                      "oracle_holds": counter["holds"], "legs": counter["legs"]}
    print("COUNTERFACTUAL oracle holds with D4d's track-JSON bytes:", counter["holds"])
    summary = {"targeted": len(rows), "turned": sum(r["outcome"] == "TURNED" for r in rows),
               "unmoved": [r["mutation"] for r in rows if r["outcome"] == "UNMOVED"],
               "crash_in_targeted": [r["mutation"] for r in rows if r["outcome"] == "CRASH"],
               "crash_class": {"cases": len(crashes), "all_crash_and_fail_closed":
                               all(c["class"] == "CRASH" and c["fails_closed"] for c in crashes)},
               "negative_controls_unmoved": all(n["unmoved"] for n in negatives),
               "baseline_verdict": baseline["line"]}
    args.out.write_text(json.dumps({"baseline": baseline["line"], "targeted": rows, "crashes": crashes,
                                    "negatives": negatives, "counterfactual": counterfactual,
                                    "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))
    shutil.rmtree(SCRATCH, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
