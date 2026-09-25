"""D4i: THE gate. One verdict for the default flip to MHR and the instrument-roster migration.

Every conjunct is DERIVED here from the evidence files -- the two deliveries' bytes, the instruments' own reports,
the roster's entry reports, the must-fail entry reports, the git tree -- and never from a PASS flag some other
program wrote. Where the card names a per-file byte result, it is read LITERALLY: no normalisation is admitted.

The card's verdict (docs/LADDER_EXECUTION_PLAN.md, "D4i the body model becomes the default"):
    PASS iff the oracle (the full per-subject file manifest, reference hashes, subject identities, draws and the
    SCORED ROWS equal) AND B1 reproduced exactly AND B2 PASS AND closure PASS AND hygiene AND must-fails i-v AND
    every final class-(a) entry matches the frozen population and reports its own verdict AND the roster JSON is
    complete with all three fields AND the coordinator's post_merge.sh / ladder.py changes are on the branch with
    the final roster run recorded.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4i_flip_gate.py --out artifacts/compare/d4i-flip/gate.json

`verdicts(PATHS)` is the whole gate; `tools/compare/d4i_flip_gate_fuzz.py` calls it on mutated copies of the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "acceb55"          # the step's base: the last commit before D4i touched anything
SUBJECTS = (0, 1)
CARD_FILES = tuple(f"subject-{s:02d}.{x}" for s in SUBJECTS
                   for x in ("glb", "body-track.npz", "body-track.json", "markers.npz", "calibration-start.json",
                             "calibration-passes.json"))
REFERENCE_FILES = ("camera-rig.json", "converter-inputs/subject-00-consumed.npz",
                   "converter-inputs/subject-01-consumed.npz", "converter-inputs/rig-converter-input.npz")
RIG_EIGHT = tuple(f"subject-{s:02d}.{x}" for s in SUBJECTS for x in ("glb", "body-track.npz", "body-track.json",
                                                                    "mapping.npz"))
MHR_ENTRIES = ("silhouette", "b1", "b2", "b3", "b4", "closure", "b5", "verifier")
FROZEN = ("src", "tools/fitter/mhr_delivery.py")
MIGRATED = ("tools/compare/post_merge.sh", "tools/compare/ladder.py")
A = "artifacts/compare"
PATHS: dict[str, Path] = {
    "default": ROOT / f"{A}/d4i-flip/default",
    "d4d_delivery": ROOT / f"{A}/d4d-twopass/delivery",
    "d4d_b1": ROOT / f"{A}/d4d-twopass/b1-paired.json",
    "d4d_silhouette": ROOT / f"{A}/d4d-twopass/silhouette-delivery.json",
    "d4d_closure": ROOT / f"{A}/d4d-twopass/delivery-closure.json",
    "d4d_b2": ROOT / f"{A}/d4d-twopass/b2-same-denominator.json",
    # B5 was never run by D4d: its reference reading is Phase 1's, on D4d's own delivery (the shadow run)
    "phase1": ROOT / f"{A}/d4i-flip/phase1/shadow/artifacts/compare/post-merge-D4i-observed",
    "final_roster": ROOT / f"{A}/d4i-flip/final-roster",
    "final_roster_record": ROOT / "docs/reviews/body-model-flip-records/final-roster-verdicts.json",
    "final_roster_log": ROOT / "docs/reviews/body-model-flip-records/final-roster-coordinator-run.log",
    "rig_arm": ROOT / f"{A}/soma77-rig-d4i",
    "shipped": ROOT / "artifacts/commercial-multiview-soma77",
    "i6": ROOT / f"{A}/i6",
    "i6_hashes": ROOT / "docs/reviews/body-model-flip-records/i6-baseline.sha256",
    "mustfail": ROOT / f"{A}/d4i-flip/mustfail",
    "mustfail_record": ROOT / "docs/reviews/body-model-flip-records/mustfail.json",
    "roster": ROOT / "docs/reviews/body-model-flip-records/roster.json",
    "worktree": ROOT,
}


def sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def strip(obj: Any, drop: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [strip(v, drop) for v in obj]
    return obj


def leaves(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def manifest(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*")
                  if p.is_file() and "work" not in p.relative_to(root).parts)


def git_bytes(path: str) -> bytes | None:
    done = subprocess.run(["git", "show", f"{BASE_COMMIT}:{path}"], cwd=ROOT, capture_output=True)
    return done.stdout if done.returncode == 0 else None


def git_files(prefix: str) -> list[str]:
    done = subprocess.run(["git", "ls-tree", "-r", "--name-only", BASE_COMMIT, prefix], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    return done.stdout.split()


# ------------------------------------------------------------------------------------------------ conjuncts

def oracle(P: dict) -> dict:
    default, d4d = Path(P["default"]), Path(P["d4d_delivery"])
    files = {}
    for name in CARD_FILES:
        a, b = sha256(default / name), sha256(d4d / name)
        row = {"default_sha256": a, "d4d_sha256": b, "byte_identical": a is not None and a == b}
        if not row["byte_identical"] and name.endswith(".json") and a and b:
            la, lb = dict(leaves(load(default / name))), dict(leaves(load(d4d / name)))
            row["differing_leaves"] = sorted(k for k in set(la) | set(lb) if la.get(k, "<absent>") != lb.get(k, "<absent>"))
        files[name] = row
    references = {name: sha256(default / name) is not None and sha256(default / name) == sha256(d4d / name)
                  for name in REFERENCE_FILES}
    fr = Path(P["final_roster"])
    b1_now, b1_d4d = load(fr / "b1.instrument.json"), load(P["d4d_b1"])
    sil_now, sil_d4d = load(fr / "silhouette.instrument.json"), load(P["d4d_silhouette"])
    b4_now = load(fr / "b4.instrument.json")
    legs = {
        "per_subject_files_byte_identical": all(r["byte_identical"] for r in files.values()),
        "manifest_equal": manifest(default) == manifest(d4d),
        "reference_hashes_equal": all(references.values()),
        "subject_identities_equal": (sil_now.get("identity") == sil_d4d.get("identity")
                                     and b4_now.get("subject_to_mamma_body_id") == {"0": 1, "1": 0}),
        "draws_equal": (b1_now.get("seed") == b1_d4d.get("seed")
                        and b1_now.get("arm_file_sha256") == b1_d4d.get("arm_file_sha256")),
        "scored_rows_equal": (strip(b1_now, {"arm_files"}) == strip(b1_d4d, {"arm_files"})
                              and strip(sil_now, {"scope", "mesh_cache"}) == sil_d4d),
        "closure_reproduced": strip(load(fr / "closure.instrument.json"), {"glb", "track"})
                              == strip(load(P["d4d_closure"]), {"glb", "track"}),
        "b2_reproduced": strip(load(fr / "b2.instrument.json"), {"delivery", "rig_build"})
                         == strip(load(P["d4d_b2"]), {"delivery", "rig_build"}),
        "b5_reproduced": strip(load(fr / "b5.instrument.json"), {"delivery", "reference"})
                         == strip(load(Path(P["phase1"]) / "b5-delivered-bytes.json"), {"delivery", "reference"}),
    }
    failing = [n for n, r in files.items() if not r["byte_identical"]]
    return {"holds": all(legs.values()), "legs": legs, "files": files, "references": references,
            "failing_files": failing,
            "failing_leaves": {n: files[n].get("differing_leaves") for n in failing},
            "read": "LITERALLY: a per-subject file either is byte-identical or it is not; no normalisation"}


def b1(P: dict) -> dict:
    fr = Path(P["final_roster"])
    now, d4d, entry = load(fr / "b1.instrument.json"), load(P["d4d_b1"]), load(fr / "b1.json")
    lower = {k: now["paired"][f"D4d_fitted_MHR_lod2_minus_baseline_D7c_rig_{k}"]["ci95_of_the_median_difference"][0]
             for k in ("subject_00", "subject_01")}
    legs = {"every_value_equal_to_D4d_except_the_arm_paths": strip(now, {"arm_files"}) == strip(d4d, {"arm_files"}),
            "same_keys": set(now) == set(d4d),
            "entry_verdict_PASS": entry.get("verdict") == "PASS",
            "band_from_the_rows": all(v > 0.0 for v in lower.values()),
            "mesh_is_a_fresh_bound_export_of_this_delivery":
                load(fr / "silhouette.json").get("mesh_cache", {}).get("fresh_export") is True
                and load(fr / "silhouette.json").get("mesh_cache", {}).get("binding", {}).get("glb_sha256")
                == {f"subject-{s:02d}.glb": sha256(Path(P["default"]) / f"subject-{s:02d}.glb") for s in SUBJECTS}}
    return {"holds": all(legs.values()), "legs": legs, "lower_ci": lower}


def own_entry(P: dict, name: str, field: str, want: Any) -> dict:
    fr = Path(P["final_roster"])
    entry, raw = load(fr / f"{name}.json"), load(fr / f"{name}.instrument.json")
    legs = {f"instrument_{field}": raw.get(field) == want, "entry_verdict_PASS": entry.get("verdict") == "PASS"}
    return {"holds": all(legs.values()), "legs": legs, "instrument_field": {field: raw.get(field)}}


def hygiene(P: dict) -> dict:
    rig, shipped, work = Path(P["rig_arm"]), Path(P["shipped"]), Path(P["worktree"])
    eight = {n: sha256(rig / n) is not None and sha256(rig / n) == sha256(shipped / n) for n in RIG_EIGHT}
    frozen: dict[str, bool] = {}
    for prefix in FROZEN:
        for path in git_files(prefix):
            here = work / path
            frozen[path] = here.is_file() and here.read_bytes() == git_bytes(path)
    extra = sorted(str(p.relative_to(work)) for p in (work / "src").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".DS_Store"))
                   and str(p.relative_to(work)) not in frozen)
    i6 = {}
    for line in Path(P["i6_hashes"]).read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        i6[name] = sha256(Path(P["i6"]) / name) == digest
    legs = {"rig_arm_8_of_8_against_the_shipped_D7c": all(eight.values()) and len(eight) == 8,
            "src_and_the_fitter_byte_identical_to_the_base": all(frozen.values()) and not extra and len(frozen) > 0,
            "i6_baseline_unchanged": all(i6.values()) and len(i6) == 7}
    return {"holds": all(legs.values()), "legs": legs, "rig_eight": eight,
            "frozen_changed": sorted(k for k, v in frozen.items() if not v), "src_files_added": extra, "i6": i6}


def mustfails(P: dict) -> dict:
    """Each must-fail re-derived from its ENTRY REPORT on disk where one exists; the few cases with no entry report
    (the build's own refusals, direct silhouette and verifier invocations) are read from the must-fail record, and
    said so."""
    mf, fr = Path(P["mustfail"]), Path(P["final_roster"])
    record = load(P["mustfail_record"])
    rep = lambda rel: load(mf / rel) if (mf / rel).is_file() else {"verdict": "NO REPORT"}
    turned: dict[str, bool] = {}
    for e in MHR_ENTRIES:
        for rel in (f"i-scope/rig-build-{e}.json", f"i-scope/swapped-{e}.json", f"ii-mixed/stale-{e}.json"):
            r = rep(rel)
            turned[rel] = r.get("verdict") == "REFUSED" and "instrument" not in r
    provenance = {"delivery", "glb", "track", "track_sha256", "reference", "arm_files", "mesh_cache", "rig_build",
                  "artifact"}
    for e in MHR_ENTRIES:
        r, raw = rep(f"iii-negative/{e}.json"), mf / f"iii-negative/{e}.instrument.json"
        turned[f"iii-negative/{e}.json (unchanged)"] = (
            r.get("verdict") == "PASS" and raw.is_file()
            and strip(load(raw), provenance) == strip(load(fr / f"{e}.instrument.json"), provenance))
    for name in ("head_removed_from_the_not_consumed_list", "head_marked_consumed", "the_marks_stripped_as_D4d_wrote_it"):
        turned[f"iv-head/{name}-entry.json"] = rep(f"iv-head/{name}-entry.json").get("verdict") == "FAIL"
    b3_new, b3_now = rep("v-positive/b3.instrument.json"), load(fr / "b3.instrument.json")
    turned["v-positive/b3 (the reading moves)"] = all(
        b3_new.get("subjects", {}).get(s, {}).get("all_landmarks_median_mm")
        not in (None, b3_now["subjects"][s]["all_landmarks_median_mm"]) for s in ("subject_00", "subject_01"))
    b1_new, b1_now = rep("v-positive/b1.instrument.json"), load(fr / "b1.instrument.json")
    turned["v-positive/b1 (the reading moves, on a fresh export)"] = (
        rep("v-positive/silhouette.json").get("mesh_cache", {}).get("fresh_export") is True
        and all(b1_new.get("arms", {}).get("D4d_fitted_MHR_lod2", {}).get(s, {}).get("pooled_median_iou")
                not in (None, b1_now["arms"]["D4d_fitted_MHR_lod2"][s]["pooled_median_iou"])
                for s in ("subject_00", "subject_01")))
    turned["v-positive/closure.json"] = rep("v-positive/closure.json").get("verdict") == "FAIL"
    for e in ("b2", "b3", "b4", "closure", "b5", "verifier"):
        turned[f"population/missing-subject-{e}.json"] = rep(f"population/missing-subject-{e}.json").get("verdict") != "PASS"
        want_pass = e == "closure"      # the closure does not read the captured landmarks
        got = rep(f"population/truncated-{e}.json").get("verdict")
        turned[f"population/truncated-{e}.json"] = (got == "PASS") if want_pass else (got not in ("PASS", "NO REPORT"))
    turned["population/missing-glb-b3.json"] = rep("population/missing-glb-b3.json").get("verdict") == "CRASH"
    for label in ("substituted_reference", "reference_nudged_1mm_on_one_value"):
        for e in ("b2", "b3", "b4", "b5"):
            got = rep(f"population/{label}-{e}.json").get("verdict")
            turned[f"population/{label}-{e}.json"] = got not in ("PASS", "NO REPORT")
    turned["population/b1-short-mesh.json"] = rep("population/b1-short-mesh.json").get("verdict") == "CRASH"
    turned["population/b1-substituted-baseline.json"] = rep("population/b1-substituted-baseline.json").get("verdict") == "FAIL"
    turned["population/b5-unbound-mesh.json"] = rep("population/b5-unbound-mesh.json").get("verdict") == "FAIL"
    record_read = {f"{g}/{n}": c["held"] for g in ("i_scope", "ii_mixed", "iv_head")
                   for n, c in record.get(g, {}).items()
                   if n.startswith(("silhouette_mhr", "silhouette_rig", "verifier_mhr", "the_default_build",
                                    "a_rig_build", "the_verifier"))
                   or (g == "iv_head" and "verifier" in c.get("observed", {}))}
    iv_verifier = {f"iv_head/{n}": c["observed"]["verifier"].get("status") == "fail"
                   for n, c in record.get("iv_head", {}).items()}
    record_read.update(iv_verifier)
    missed = [f"{g}/{n}" for g, cases in record.items() if isinstance(cases, dict) and g != "summary"
              for n, c in cases.items() if isinstance(c, dict) and c.get("held") is False]
    legs = {"every_entry_report_turned_or_stayed_as_required": all(turned.values()) and len(turned) >= 60,
            "record_read_cases_held": all(record_read.values()) and len(record_read) >= 9}
    return {"holds": all(legs.values()), "legs": legs, "entry_report_cases": len(turned),
            "not_turned": sorted(k for k, v in turned.items() if not v), "record_read": record_read,
            "prediction_misses_reported": missed}


def population(P: dict) -> dict:
    """The entries are bound to the evidence's default build BY CONTENT: the delivery each entry names must hold the
    same per-subject bytes as the default under test (a path match alone passed a copy and failed a moved mtime --
    the fuzz's negative controls found it)."""
    fr, default = Path(P["final_roster"]), Path(P["default"])
    want = {name: sha256(default / name) for name in CARD_FILES}
    rows = {}
    for e in MHR_ENTRIES:
        r = load(fr / f"{e}.json")
        named = Path(r.get("delivery", "/nonexistent"))
        rows[e] = {"verdict_PASS": r.get("verdict") == "PASS",
                   "population_checked_and_matched": bool(r.get("population", {}).get("expected"))
                                                     and r.get("population", {}).get("mismatches") == [],
                   "own_verdict_field_reported": bool(r.get("instrument_verdict_field")),
                   "scope_checked": r.get("scope") == "mhr" and bool(r.get("schema_declared")),
                   "on_the_default_build_by_content":
                       all(v is not None for v in want.values())
                       and {name: sha256(named / name) for name in CARD_FILES} == want}
    record = load(P["final_roster_record"])["entries"]
    agrees = all(record.get(e, {}).get("value") == load(fr / f"{e}.json").get("verdict") for e in MHR_ENTRIES)
    legs = {"every_entry": all(all(v.values()) for v in rows.values()),
            "the_recorded_run_agrees_with_the_entry_reports": agrees}
    return {"holds": all(legs.values()), "legs": legs, "entries": rows}


def roster_complete(P: dict) -> dict:
    roster = load(P["roster"])["instruments"]
    rows = {}
    for name, inst in roster.items():
        o, f, m = inst.get("observed") or {}, inst.get("final_action") or {}, inst.get("post_migration") or {}
        rows[name] = {"observed": bool(o.get("class")) and bool(o.get("evidence")) and bool(o.get("command")),
                      "final_action": f.get("action") in ("keep", "relabel", "replace", "remove") and bool(f.get("why")),
                      "post_migration": bool(m.get("class")) and bool(m.get("evidence")) and bool(m.get("command")),
                      "d_is_removed": o.get("class") != "d" or f.get("action") == "remove"}
    legs = {"every_instrument_has_all_three_fields": all(all(v.values()) for v in rows.values()) and len(rows) >= 18}
    return {"holds": all(legs.values()), "legs": legs, "instruments": len(rows),
            "incomplete": sorted(k for k, v in rows.items() if not all(v.values()))}


def coordinator(P: dict) -> dict:
    work = Path(P["worktree"])
    changed = {path: (work / path).is_file() and (work / path).read_bytes() != git_bytes(path) for path in MIGRATED}
    record = load(P["final_roster_record"])
    legs = {"post_merge_sh_changed_on_the_branch": changed["tools/compare/post_merge.sh"],
            "ladder_py_changed_on_the_branch": changed["tools/compare/ladder.py"],
            "final_roster_run_recorded": Path(P["final_roster_log"]).is_file()
                                         and record.get("mhr_all_pass") is True and record.get("no_entry_crashed") is True
                                         and record == load(Path(P["final_roster"]) / "roster-verdicts.json")}
    return {"holds": all(legs.values()), "legs": legs,
            "note": "ladder.py is the coordinator's, edited after stage 7 by the coordinator's own order; rerun the gate then"}


CONJUNCTS = {"oracle": oracle, "B1_reproduced_exactly": b1,
             "B2_PASS": lambda P: own_entry(P, "b2", "verdict", "PASS"),
             "closure_PASS": lambda P: own_entry(P, "closure", "all_within_band", True),
             "hygiene": hygiene, "must_fails_i_to_v": mustfails, "population_and_own_verdicts": population,
             "roster_complete": roster_complete, "coordinator_migration_recorded": coordinator}


def verdicts(P: dict | None = None) -> dict:
    P = {**PATHS, **(P or {})}
    report: dict[str, Any] = {"conjuncts": {}}
    for name, fn in CONJUNCTS.items():
        try:
            report["conjuncts"][name] = fn(P)
        except Exception as error:   # evidence the gate cannot read fails closed, as its own class
            report["conjuncts"][name] = {"holds": False, "crash": f"{type(error).__name__}: {error}"}
    failed = [n for n, c in report["conjuncts"].items() if not c["holds"]]
    crashed = [n for n, c in report["conjuncts"].items() if "crash" in c]
    report["failed"] = failed
    report["crashed"] = crashed
    report["verdict"] = "PASS" if not failed else "FAIL"
    report["line"] = ("VERDICT: PASS (every conjunct holds)" if not failed
                      else f"VERDICT: FAIL (failed: {', '.join(failed)})")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verdicts()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, c in report["conjuncts"].items():
        print(f"{name:34s} {'HOLDS' if c['holds'] else 'FAILS'}  " + json.dumps(c.get("legs", c.get("crash")))[:200])
    oracle_row = report["conjuncts"].get("oracle", {})
    if oracle_row.get("failing_files"):
        print("oracle, failing per-subject files:", json.dumps(oracle_row["failing_leaves"]))
    print(report["line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
