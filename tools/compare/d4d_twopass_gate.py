#!/usr/bin/env python3
"""D4d a second calibration pass on the landmark start: ONE verdict, derived from the inputs.

The card is the "D4d a second calibration pass" row of `docs/LADDER_EXECUTION_PLAN.md` §2 (copy:
`docs/reviews/body-model-twopass-card-2026-09-25.md`). This gate reads only what the producers wrote -- D4c's frozen
drawn set, the stage-1 provenance, the Phase-1 cells (new) and D4c's retained cells (bound by content to D4c's own
records), the committed Phase-1 decision JSON, the Phase-2 cells and their arrays, the closure report, the B1 reports,
the delivery's own files (B2 re-derived here), the tripwire's files themselves and the fitter's source at the D4c tag
-- and derives every clause from them. No verdict is a literal.

    verdict = STOP     if precondition 0 fails (D4c's drawn set, by sha256: the spine not drawn)
            = FAIL     else if Phase 1's population, provenance or construction is short, missing or mismatched
            = STOP     else if stage 0b fails (the spine control passes L as scored on any Phase-1 fixture)
            = STOP     else if WARM leaves the trunk beyond its paired tolerance on any of D4c's 12 (D4e)
            = STOP     else if WARM holds but TWO-PASS does not bring every scored segment within tolerance on all 12
            = FAIL     else if the decision JSON is not committed, or not re-derivable from the Phase-1 cells
            = FAIL     else if Phase 2's population, provenance, construction or freeze order is short or mismatched
            = INVALID  else if the exact_identity arm's pooled statistic exceeds 1.0 mm on any Phase-2 fixture
            = STOP     else if init-only PASSES L on any Phase-2 fixture
            = PASS     else iff L AND closure AND must-fails i-iv AND B1 AND B2 AND hygiene
            = FAIL     otherwise

THE BAND L (D4b's, unchanged; the arithmetic imported from `d4b_o1_gate`): per fixture, every segment a drawn channel
moves -- the rest length at MHR's zero pose with the identity retained, |fitted - truth| in mm -- within the sum of its
two endpoints' PAIRED floors (the per-joint median over frames of the same fixture's exact_identity arm).

Instrument debt repaired here (the card): the tripwire's equality is RE-DERIVED from the files (byte equality, else
only the named normalisation for that file kind), and B2's numbered checks are RE-DERIVED from the delivery's own
files -- the producers' booleans are never read. The fuzz (`d4d_twopass_gate_fuzz.py`) keeps CRASH as its own class.

ONE verdict line is printed. D4's consequence is derived from it in the JSON (`consequence`).

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --precondition-0 --out RECORDS/precondition-0.json
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --hygiene --out RECORDS/hygiene.json
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --phase1 --out RECORDS/phase1-decision.json
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --manifest --out RECORDS/acceptance-manifest.json
    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --out artifacts/compare/d4d-twopass/gate.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/fitter"))
sys.path.insert(0, str(ROOT / "tools/compare"))
import d4c_start_gate as d4cg  # noqa: E402

RECORDS = ROOT / "docs/reviews/body-model-twopass-records"
D4C_RECORDS = ROOT / "docs/reviews/body-model-start-records"
D4C_DRAWN_SET = D4C_RECORDS / "drawn-set.json"
D4C_DRAWN_SET_SHA256 = "430f1f67ea27123969ea5bafde01eb607b0092cb7a47d6a69f451038052163b7"   # D4c's stage-1 record


OUT_DIR = ROOT / "artifacts/compare/d4d-twopass"
FITTER = ROOT / "tools/fitter/mhr_delivery.py"
BASE_TAG = "ladder/D4c-fail-1a89cc7"
BASE_COMMIT = "1a89cc73b2ced93543e9cbfb191d79208aef423b"
BASE_FITTER_SHA256 = "dd54443dffcdc9821b58f8e82091ebe709f7b80cee33e9aede197eab4c009389"   # the tag's fitter
SHIPPED = ROOT / "artifacts/commercial-multiview-soma77"
EIGHT = ("subject-00.glb", "subject-01.glb", "subject-00.body-track.npz", "subject-01.body-track.npz",
         "subject-00.body-track.json", "subject-01.body-track.json", "subject-00.mapping.npz",
         "subject-01.mapping.npz")
D4C_DELIVERY = ROOT / "artifacts/compare/d4c-start/delivery"
D4C_ACCEPTANCE = ROOT / "artifacts/compare/d4c-start/acceptance"
D4C_DELIVERY_FILES = tuple(f"subject-{s}.{k}" for s in ("00", "01")
                           for k in ("glb", "body-track.npz", "body-track.json", "markers.npz",
                                     "calibration-start.json")) + ("fit-report-subject-00.json",
                                                                   "fit-report-subject-01.json")
D4C_ACCEPTANCE_FILES = 204          # 72 cell JSONs, 72 arrays, 12 GLBs, 12 track npz, 12 track JSONs, 24 debug logs
# The ONLY normalisations the tripwire allows, named: a worktree-rooted assets path in a body-track JSON, the fitter's
# own sha256 in a D4c cell's provenance (the file changed by design; its behaviour at one pass is what is tested),
# and momentum's wall-clock log timestamps.
NORMALISED = {"*.body-track.json": ["body_model.assets"], "cell-*.json": ["provenance.fitter_sha256"],
              "*.calibration-debug.log": ["momentum timestamps [YYYY-MM-DD HH:MM:SS.mmm]"]}
TIMESTAMP = re.compile(rb"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\]")
EXPECTED_SECOND_PASS = """if passes == 2:
    first = identity.copy()
    mt.calibrate_markers(character, first.copy(), markers, stage_a)
    identity, _, _ = mt.calibrate_markers(character, first.copy(), markers, calibration)
    identity = np.asarray(identity, np.float32)
elif passes != 1:
    raise ValueError(f'passes is 1 or 2, never swept (got {passes!r})')"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def git_bytes(spec: str) -> bytes:
    return subprocess.run(["git", "show", spec], cwd=ROOT, capture_output=True, check=False).stdout


# ------------------------------------------------------------------------------------------ hygiene

def normalised_equal(name: str, a: bytes | None, b: bytes | None) -> tuple[bool, str]:
    """(equal, how) -- byte equality first; else ONLY the named normalisation for that file kind."""
    if a is None or b is None:
        return False, "missing"
    if a == b:
        return True, "bytes"
    try:
        if name.endswith(".body-track.json"):
            x, y = json.loads(a), json.loads(b)
            x["body_model"]["assets"] = y["body_model"]["assets"] = "<normalised>"
            return x == y, "normalised body_model.assets"
        if name.startswith("cell-") and name.endswith(".json"):
            x, y = json.loads(a), json.loads(b)
            x["provenance"]["fitter_sha256"] = y["provenance"]["fitter_sha256"] = "<normalised>"
            return x == y, "normalised provenance.fitter_sha256"
        if name.endswith(".calibration-debug.log"):
            return TIMESTAMP.sub(b"[T]", a) == TIMESTAMP.sub(b"[T]", b), "normalised momentum timestamps"
    except (ValueError, KeyError, TypeError):
        return False, "unreadable"
    return False, "bytes differ"


def tripwire(files: dict) -> dict:
    """passes = 1 through the new code against D4c's pinned delivery and acceptance cells, from the FILES
    (`files` maps a label to (D4c bytes, tripwire bytes)); every set is checked by identity."""
    rows = {k: dict(zip(("equal", "how"), normalised_equal(k.rsplit("/", 1)[-1], *v))) for k, v in files.items()}
    delivery = [k for k in rows if k.startswith("delivery/")]
    cells = [k for k in rows if k.startswith("acceptance/")]
    sets = {"delivery_files": sorted(k.split("/", 1)[1] for k in delivery) == sorted(D4C_DELIVERY_FILES),
            "acceptance_files": len(cells) == D4C_ACCEPTANCE_FILES}
    return {"rows": rows, "sets_by_identity": sets,
            "byte_identical": sum(r["how"] == "bytes" for r in rows.values()),
            "equal_after_the_named_normalisation": sum(r["equal"] and r["how"] != "bytes" for r in rows.values()),
            "unequal": sorted(k for k, r in rows.items() if not r["equal"]),
            "holds": bool(all(sets.values()) and all(r["equal"] for r in rows.values())),
            "normalisations_allowed": NORMALISED}


def tripwire_files(out_dir: Path = OUT_DIR) -> dict:
    files = {}
    for name in D4C_DELIVERY_FILES:
        a, b = D4C_DELIVERY / name, out_dir / "tripwire/delivery" / name
        files[f"delivery/{name}"] = (a.read_bytes() if a.is_file() else None, b.read_bytes() if b.is_file() else None)
    names = sorted({p.name for p in D4C_ACCEPTANCE.iterdir()} | {p.name for p in (out_dir / "tripwire/acceptance").iterdir()}) \
        if (out_dir / "tripwire/acceptance").is_dir() else sorted(p.name for p in D4C_ACCEPTANCE.iterdir())
    for name in names:
        a, b = D4C_ACCEPTANCE / name, out_dir / "tripwire/acceptance" / name
        files[f"acceptance/{name}"] = (a.read_bytes() if a.is_file() else None, b.read_bytes() if b.is_file() else None)
    return files


def source_diff(base_source: str, new_source: str) -> dict:
    """Attribution: `fit_one` differs from the D4c tag's ONLY by the `passes` keyword (last, default 1), its
    docstring, and ONE statement -- the second calibration pass, its own copy of the first pass's identity, exactly
    `EXPECTED_SECOND_PASS` -- placed directly after the first pass. Compared as ASTs after undoing exactly those."""
    def fit_one(source: str):
        return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "fit_one")

    try:
        old, new = fit_one(base_source), fit_one(new_source)
    except (StopIteration, SyntaxError, TypeError, ValueError):
        return {"only_the_second_pass": False, "why": "fit_one not found"}
    old_args, new_args = [a.arg for a in old.args.kwonlyargs], [a.arg for a in new.args.kwonlyargs]
    default = new.args.kw_defaults[-1] if new.args.kw_defaults else None
    args_ok = (new_args == old_args + ["passes"] and isinstance(default, ast.Constant) and default.value == 1)
    new.args.kwonlyargs, new.args.kw_defaults = new.args.kwonlyargs[:-1], new.args.kw_defaults[:-1]
    for f in (old, new):
        if f.body and isinstance(f.body[0], ast.Expr) and isinstance(getattr(f.body[0], "value", None), ast.Constant):
            f.body = f.body[1:]
    expected = ast.unparse(ast.parse(EXPECTED_SECOND_PASS))
    removed, placed = 0, False
    for node in ast.walk(new):
        for field in ("body", "orelse"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            kept = []
            for i, statement in enumerate(block):
                if isinstance(statement, ast.If) and ast.unparse(statement) == expected:
                    removed += 1
                    before = block[i - 2:i] if i >= 2 else []
                    placed = [ast.unparse(b) for b in before] == [
                        "identity, _, _ = mt.calibrate_markers(character, start.copy(), markers, calibration)",
                        "identity = np.asarray(identity, np.float32)"]
                    continue
                kept.append(statement)
            setattr(node, field, kept)
    same = ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    return {"only_the_second_pass": bool(args_ok and removed == 1 and placed and same),
            "kwonly_args_added": [a for a in new_args if a not in old_args], "passes_default_is_1": args_ok,
            "second_pass_blocks_removed": removed, "placed_after_the_first_pass": placed,
            "ast_equal_after_undoing_it": same, "base": f"{BASE_COMMIT}:tools/fitter/mhr_delivery.py",
            "base_sha256": sha256_bytes(base_source.encode())}


def rig_rebuild(rig: dict) -> dict:
    rows = {f: {"rebuild": (rig.get(f) or [None, None])[0], "shipped": (rig.get(f) or [None, None])[1]} for f in EIGHT}
    for r in rows.values():
        r["identical"] = r["rebuild"] is not None and r["rebuild"] == r["shipped"]
    return {"rows": rows, "identical_count": sum(r["identical"] for r in rows.values()),
            "holds": all(r["identical"] for r in rows.values())}


def load_hygiene(out_dir: Path = OUT_DIR) -> dict:
    return {"fitter_source": FITTER.read_text(encoding="utf-8"),
            "base_source": git_bytes(f"{BASE_COMMIT}:tools/fitter/mhr_delivery.py").decode(),
            "files": tripwire_files(out_dir),
            "rig": {f: [sha256_file(out_dir / "hygiene" / f), sha256_file(SHIPPED / f)] for f in EIGHT}}


def hygiene(h: dict | None = None) -> dict:
    h = load_hygiene() if h is None else h
    trip = tripwire(h["files"])
    diff = source_diff(h["base_source"], h["fitter_source"])
    rig = rig_rebuild(h["rig"])
    base_ok = diff.get("base_sha256") == BASE_FITTER_SHA256
    return {"pass": bool(trip["holds"] and diff["only_the_second_pass"] and rig["holds"] and base_ok),
            "tripwire": trip, "source_diff": diff, "rig_rebuild": rig, "base_fitter_is_the_tag_s": base_ok}


def precondition_0(drawn_bytes: bytes) -> dict:
    """D4c's frozen limit-aware drawn set, reused by sha256; the spine must be drawn (and the trunk scored)."""
    bound = sha256_bytes(drawn_bytes) == D4C_DRAWN_SET_SHA256
    try:
        drawn = json.loads(drawn_bytes)
    except (ValueError, TypeError):
        drawn = {}
    result = d4cg.precondition_0(drawn)
    return dict(result, reused_by_sha256=bound, drawn_set_sha256=sha256_bytes(drawn_bytes),
                holds=bool(result["holds"] and bound))


# --------------------------------------------------------------------------------------------- Phase 1

import math  # noqa: E402

import numpy as np  # noqa: E402

import d4b_o1_gate as d4bg  # noqa: E402
import d4d_fixture as fx4  # noqa: E402  (momentum-free at import)
from d4b_identifiability import CHANNELS, MAPPED_JOINTS  # noqa: E402

PHASE1_DIR = OUT_DIR / "phase1"
PROVENANCE = RECORDS / "provenance.json"
DECISION = RECORDS / "phase1-decision.json"
FRAMES = 150
IDENTITY_CHANNELS = 68
LAYOUT_AGREEMENT_CM = 1e-6
LANDMARK_AGREEMENT_CM = 1e-9
START_AGREEMENT = 1e-4          # |the fitter's float32 start - the gate's float64 recomputation|, units (D4c's)
SETTINGS = dict(fx4.SETTINGS)
D4C_SCHEMA = "d4c-start-cell/1"
CALLS = {1: 2, 2: 4}
RULE = ("Frozen by the card: STOP (D4e takes the objective) if WARM leaves the trunk beyond its paired tolerance on ANY "
        "of D4c's 12; the candidate is TWO-PASS iff WARM holds on all 12 AND two-pass brings every scored segment "
        "within tolerance on all 12; otherwise STOP, reported (SW* is then the lead for a landmark-derived "
        "shoulder-width start, which would need its own registration). A missing, short or non-finite cell is FAIL. "
        "Stage 0b: the spine control fails L as scored on every Phase-1 fixture, else STOP.")


def phase1_population() -> list[tuple[str, int, int]]:
    return [(p, s, d) for p in fx4.PHASE1 for s, d in fx4.POPULATIONS[p]]


def stage1_sha(prefix: str, stage1: dict) -> str | None:
    hits = [v for k, v in (stage1.get("sha256") or {}).items() if k.startswith(prefix)]
    return hits[0] if len(hits) == 1 else None


def load_phase1(phase1_dir: Path = PHASE1_DIR) -> dict:
    inputs = {"drawn_bytes": D4C_DRAWN_SET.read_bytes(),
              "stage1": json.loads(PROVENANCE.read_text(encoding="utf-8")),
              "fitter_sha256": sha256_file(FITTER),
              "fixture_sha256": sha256_file(ROOT / "tools/fitter/d4d_fixture.py"),
              "d4c_fixture_sha256": sha256_file(ROOT / "tools/fitter/d4c_fixture.py"),
              "d4c_manifest_bytes": fx4.D4C_MANIFEST.read_bytes(),
              "d4c_development_bytes": fx4.D4C_DEVELOPMENT.read_bytes(),
              "new": {}, "new_files": {}, "arrays": {}, "retained": {}}
    # REPORTED only: D4b's own WARM cells (the truth start handed in through D4b's proxy on the D4 fitter)
    inputs["d4b_warm"] = {}
    for p, s, d in phase1_population():
        path = ROOT / "artifacts/compare/d4b-o1" / ("burned" if p == "d4" else "fresh") / f"cell-{s}-d{d}-warm.json"
        if p != "d4c" and path.is_file():
            inputs["d4b_warm"][f"{p}/{s}_d{d}"] = json.loads(path.read_text(encoding="utf-8")).get("fitted_identity")
    # the decision JSON: the bytes on disk and the bytes of the commit that ADDED it
    commit = added_commit(DECISION) if DECISION.exists() else ""
    inputs["decision_bytes"] = DECISION.read_bytes() if DECISION.exists() else b""
    inputs["decision_committed_bytes"] = git_bytes(f"{commit}:{DECISION.relative_to(ROOT)}") if commit else b""
    for path in sorted(Path(phase1_dir).glob("cell-*.json")):
        inputs["new"][path.name] = json.loads(path.read_text(encoding="utf-8"))
        inputs["new_files"][path.name] = sha256_file(path)
    for path in sorted(Path(phase1_dir).glob("fixture-*.arrays.npz")):
        inputs["new_files"][path.name] = sha256_file(path)
        if "-two_pass." in path.name:
            data = np.load(path)          # no pickle: plain arrays written by the fixture
            inputs["arrays"][path.name] = {
                "consumed_landmarks_z_up_m": np.asarray(data["consumed_landmarks_z_up_m"]),
                "consumed_joint_names": [str(n) for n in data["consumed_joint_names"]],
                "truth_mapped_cm": np.asarray(data["truth_mapped_cm"])}
    for p, s, d in phase1_population():
        for a in fx4.PHASE1_RETAINED_ARMS:
            path = fx4.retained_cell(p, s, d, a)
            if path.is_file():
                data = path.read_bytes()
                record = json.loads(data)
                inputs["retained"][f"{p}/{s}/d{d}/{a}"] = {
                    "name": path.name, "stage": "acceptance" if p == "d4c" else "development", "record": record,
                    "file_sha256": sha256_bytes(data),
                    "sorted_dump_sha256": sha256_bytes(json.dumps(record, sort_keys=True).encode())}
    return inputs


def _cell_arrays(record: dict) -> dict | None:
    out = {"distance": d4bg.finite_array(record.get("distance_mm"), (FRAMES, len(MAPPED_JOINTS))),
           "truth_rest": d4bg.finite_array(record.get("truth_rest_mapped_cm"), (len(MAPPED_JOINTS), 3)),
           "fitted_rest": d4bg.finite_array(record.get("fitted_rest_mapped_cm"), (len(MAPPED_JOINTS), 3)),
           "truth_id": d4bg.finite_array(record.get("truth_identity"), (IDENTITY_CHANNELS,)),
           "fitted_id": d4bg.finite_array(record.get("fitted_identity"), (IDENTITY_CHANNELS,))}
    return None if any(v is None for v in out.values()) else out


def _names_ok(names: list) -> bool:
    return (len(names) == IDENTITY_CHANNELS and len(set(names)) == IDENTITY_CHANNELS
            and all(str(n).startswith("scale_") for n in names) and set(CHANNELS) <= set(names))


def expected_new_provenance(inputs: dict) -> dict:
    st = inputs["stage1"]
    return {"fitter_sha256": inputs["fitter_sha256"],
            "d4_fixture_sha256": stage1_sha("D4 fixture ", st), "d4b_fixture_sha256": stage1_sha("D4b fixture ", st),
            "d4c_fixture_sha256": stage1_sha("D4c fixture ", st), "this_file_sha256": inputs["fixture_sha256"],
            "donor_sha256": {"0": stage1_sha("donor 0 ", st), "1": stage1_sha("donor 1 ", st)},
            "model_sha256": stage1_sha("model ", st), "fbx_sha256": stage1_sha("fbx ", st),
            "d4c_drawn_set_sha256": stage1_sha("D4c drawn set ", st),
            "d4b_drawn_set_sha256": stage1_sha("D4b frozen drawn set ", st),
            "d4c_development_json_sha256": stage1_sha("D4c development JSON ", st),
            "pymomentum": (st.get("environment") or {}).get("pymomentum-cpu")}


def check_calibrating(record: dict, arrays: dict, names: list[str], passes: int, bad: list[str]) -> None:
    """A calibrating arm: settings, call count, per-call labels and the hand-offs of the identity between calls."""
    settings = record.get("settings") or {}
    if settings.get("passes") != passes or settings.get("calibration_debug_captured") is not True:
        bad.append("settings passes / debug capture")
    if record.get("calibrate_markers_calls") != CALLS[passes]:
        bad.append("calibrate_markers calls")
    calls = record.get("calibration_calls") or []
    if [c.get("stage") for c in calls] != list(fx4.CALL_LABELS[passes]):
        bad.append("per-call stage labels")
        return
    handed = [d4bg.finite_array(c.get("identity_handed"), (IDENTITY_CHANNELS,)) for c in calls]
    returned = [d4bg.finite_array(c.get("identity_returned"), (IDENTITY_CHANNELS,)) for c in calls]
    rests = [d4bg.finite_array(c.get("rest_mapped_cm_after"), (len(MAPPED_JOINTS), 3)) for c in calls]
    if any(v is None for v in handed + returned + rests):
        bad.append("per-call identities or rests short or non-finite")
        return
    if [bool(c.get("locators_only")) for c in calls] != [True, False] * passes:
        bad.append("per-call locators_only is not A, B per pass")
    start = d4bg.finite_array((record.get("start") or {}).get("start_identity"), (IDENTITY_CHANNELS,))
    kind = (record.get("start") or {}).get("kind")
    if kind == "zero":
        start = np.zeros(IDENTITY_CHANNELS) if (record.get("start") or {}).get("start_identity") is None else None
    if start is None:
        bad.append("start identity missing")
        return
    # each pass's two stages are handed the same identity: the start (pass 1), pass 1's stage-B identity (pass 2)
    for k in range(passes):
        want = start if k == 0 else returned[2 * k - 1]
        if not (np.array_equal(handed[2 * k], want) and np.array_equal(handed[2 * k + 1], want)):
            bad.append(f"pass {k + 1} was not handed {'the start' if k == 0 else 'pass 1s identity'}")
    if not np.array_equal(returned[-1], arrays["fitted_id"]):
        bad.append("the fitted identity is not the last stage-B return")
    it = record.get("calibration_iterations") or {}
    if (it.get("calls") != CALLS[passes] or it.get("passes") != passes or it.get("max_iter") != SETTINGS["max_iter"]
            or [c.get("stage") for c in it.get("per_call") or []] != list(fx4.CALL_LABELS[passes])):
        bad.append("the cap-exhaustion record is not per stage and per pass")


def check_phase1(inputs: dict, problems: list[str]) -> dict:
    """Population, provenance and construction for Phase 1. Returns the valid cells by (population, s, d, arm)."""
    drawn = json.loads(inputs["drawn_bytes"])
    st = inputs["stage1"]
    if sha256_bytes(inputs["d4c_manifest_bytes"]) != stage1_sha("D4c acceptance manifest ", st):
        problems.append("phase 1: D4c's acceptance manifest is not the one stage 1 recorded")
    if sha256_bytes(inputs["d4c_development_bytes"]) != stage1_sha("D4c development JSON ", st):
        problems.append("phase 1: D4c's development JSON is not the one stage 1 recorded")
    if sha256_bytes(inputs["drawn_bytes"]) != D4C_DRAWN_SET_SHA256:
        problems.append("phase 1: the drawn set is not D4c's")
    manifest = (json.loads(inputs["d4c_manifest_bytes"]) or {}).get("cells_sha256") or {}
    development = (json.loads(inputs["d4c_development_bytes"]) or {}).get("cells_sha256") or {}
    statistic = json.loads(inputs["d4c_development_bytes"]).get("chosen_trunk_statistic")
    prov = expected_new_provenance(inputs)
    population = phase1_population()
    expected = {f"cell-{s}-d{d}-{a}.json" for _, s, d in population for a in fx4.PHASE1_NEW_ARMS}
    for name in sorted(expected - set(inputs["new"])):
        problems.append(f"phase 1: {name} missing")
    for name in sorted(set(inputs["new"]) - expected):
        problems.append(f"phase 1: {name} is not in the named population")
    cells, canonical = {}, None
    for p, s, d in population:
        # retained D4c cells, bound by content to D4c's own records
        for a in fx4.PHASE1_RETAINED_ARMS:
            got = inputs["retained"].get(f"{p}/{s}/d{d}/{a}")
            tag = f"phase 1 retained {p} {s}/d{d}/{a}"
            if got is None:
                problems.append(f"{tag}: missing")
                continue
            bound = (manifest.get(got["name"]) == got["file_sha256"] if got["stage"] == "acceptance"
                     else development.get(got["name"]) == got["sorted_dump_sha256"])
            record = got["record"]
            arr = _cell_arrays(record)
            want_arm = {"d4c_start": "candidate" if p == "d4c" else "candidate_p90"}.get(a, a)
            want_pop = "acceptance" if p == "d4c" else p
            bad = []
            if not bound:
                bad.append("not bound by content to D4c's record")
            if (record.get("schema"), record.get("seed"), record.get("donor"), record.get("arm"),
                    record.get("population")) != (D4C_SCHEMA, s, d, want_arm, want_pop):
                bad.append("identity fields")
            if record.get("lod") != 2 or record.get("frames") != FRAMES or list(record.get("mapped_joints") or []) != list(MAPPED_JOINTS):
                bad.append("lod, frames or mapped joints")
            if arr is None:
                bad.append("arrays short, missing or non-finite")
            if a == "d4c_start" and (record.get("start") or {}).get("trunk_statistic") != statistic:
                bad.append("the D4c start's trunk statistic is not D4c's frozen choice")
            if bad:
                problems.append(f"{tag}: " + "; ".join(bad))
                continue
            names = list(record.get("identity_channel_names") or [])
            cells[(p, s, d, a)] = dict(arr, record=record, names=names)
        for a in fx4.PHASE1_NEW_ARMS:
            record = inputs["new"].get(f"cell-{s}-d{d}-{a}.json")
            if record is None:
                continue
            tag, bad = f"phase 1 {p} {s}/d{d}/{a}", []
            if (record.get("schema"), record.get("seed"), record.get("donor"), record.get("arm"),
                    record.get("population")) != (fx4.SCHEMA, s, d, a, p):
                bad.append("identity fields")
            if record.get("lod") != 2 or record.get("frames") != FRAMES or list(record.get("mapped_joints") or []) != list(MAPPED_JOINTS):
                bad.append("lod, frames or mapped joints")
            if record.get("phase1_decision_sha256") is not None:
                bad.append("a Phase-1 cell carries a decision hash")
            arr = _cell_arrays(record)
            if arr is None:
                bad.append("arrays short, missing or non-finite")
            names = list(record.get("identity_channel_names") or [])
            if not _names_ok(names):
                bad.append("identity channel names")
            elif canonical is None:
                canonical = names
            elif names != canonical:
                bad.append("identity channel names differ between cells")
            layout = record.get("truth_rest_full_vs_simplified_character_max_abs_cm")
            if not isinstance(layout, (int, float)) or not math.isfinite(layout) or layout > LAYOUT_AGREEMENT_CM:
                bad.append("the truth rest differs between the 204- and 178-parameter characters")
            got = record.get("provenance") or {}
            for field, value in prov.items():
                if value is None or got.get(field) != value:
                    bad.append(f"provenance {field}")
            settings = record.get("settings") or {}
            for field, value in SETTINGS.items():
                if settings.get(field) != value:
                    bad.append(f"settings {field}")
            if inputs["new_files"].get(record.get("arrays")) != record.get("arrays_sha256") or not record.get("arrays_sha256"):
                bad.append("arrays file hash")
            start = record.get("start") or {}
            if start.get("trunk_statistic") != statistic or start.get("kind") != fx4.CALIBRATING[a][0]:
                bad.append("start kind or trunk statistic")
            if arr is not None and not bad:
                check_calibrating(record, arr, names, fx4.CALIBRATING[a][1], bad)
            if bad:
                problems.append(f"{tag}: " + "; ".join(bad))
                continue
            cells[(p, s, d, a)] = dict(arr, record=record, names=names)
    # fixture identity and each arm's construction, verified
    for p, s, d in population:
        group = {a: cells.get((p, s, d, a)) for a in fx4.PHASE1_ARMS}
        if any(c is None for c in group.values()):
            continue
        tag = f"phase 1 fixture {p} {s}/d{d}"
        names = group["warm"]["names"]
        truth = group["exact_identity"]
        if any(c["names"] != names for c in group.values()):
            problems.append(f"{tag}: the arms' identity layouts differ")
            continue
        if any(not np.array_equal(c["truth_id"], truth["truth_id"]) or not np.array_equal(c["truth_rest"], truth["truth_rest"])
               for c in group.values()):
            problems.append(f"{tag}: the arms do not share one truth")
            continue
        sha = {(c["record"].get("fixture") or {}).get("truth_motion_sha256") for c in group.values()}
        if len(sha) != 1 or None in sha:
            problems.append(f"{tag}: the arms' truth motions differ")
        if p == "d4c":
            regenerated = fx4.fx.acceptance_draw(s, d, list(drawn["drawn_set"]), drawn["configured_limits"], CHANNELS)
            if any(np.float32(truth["truth_id"][i]) != np.float32(regenerated.get(n, 0.0)) for i, n in enumerate(names)):
                problems.append(f"{tag}: the truth is not the regenerated draw")
        if not np.array_equal(truth["fitted_id"], truth["truth_id"]):
            problems.append(f"{tag}: exact_identity is not the truth")
        spine = names.index("scale_spine_length")
        sd = group["spine_displaced"]
        other = np.arange(len(names)) != spine
        if (not np.array_equal(sd["fitted_id"][other], sd["truth_id"][other])
                or abs(sd["fitted_id"][spine] - fx4.fx.displaced_spine(float(sd["truth_id"][spine]))) > 1e-6):
            problems.append(f"{tag}: spine_displaced is not the truth with the spine moved")
        # the landmark start: recomputed from the consumed landmarks (D4c's momentum-free rule), shared by D4c's
        # one-pass cell and every new arm that starts from it
        arr = inputs["arrays"].get(group["two_pass"]["record"].get("arrays"))
        if arr is None:
            problems.append(f"{tag}: the two-pass consumed landmarks are not readable")
            continue
        consumed, joint_names, truth_cm = arr["consumed_landmarks_z_up_m"], arr["consumed_joint_names"], arr["truth_mapped_cm"]
        as_capture = np.stack([truth_cm[..., 0], -truth_cm[..., 2], truth_cm[..., 1]], axis=-1) / 100.0
        columns = [joint_names.index(l) for l in d4cg.MAP]
        if consumed.shape != (FRAMES, len(joint_names), 3) or np.nanmax(np.abs(consumed[:, columns] - as_capture)) * 100.0 > LANDMARK_AGREEMENT_CM:
            problems.append(f"{tag}: the consumed landmarks are not the truth's joints")
            continue
        recomputed = d4cg.recompute_start(consumed, joint_names, drawn, statistic)
        want = np.array([recomputed.get(n, 0.0) for n in names])
        landmark = np.asarray(group["two_pass"]["record"]["start"]["landmark_start_identity"], float)
        d4c_start = np.asarray((group["d4c_start"]["record"].get("start") or {}).get("start_identity") or [np.nan] * len(names), float)
        if np.max(np.abs(landmark - want)) > START_AGREEMENT or not np.array_equal(landmark, d4c_start):
            problems.append(f"{tag}: the landmark start is not the frozen rule's, or not D4c's one-pass start")
        for a in fx4.PHASE1_NEW_ARMS:
            if not np.array_equal(np.asarray(group[a]["record"]["start"]["landmark_start_identity"], float), landmark):
                problems.append(f"{tag}/{a}: a different landmark start")
        starts = {a: np.asarray(group[a]["record"]["start"]["start_identity"], float) for a in fx4.PHASE1_NEW_ARMS}
        sw = names.index("scale_shoulder_width")
        want_sw = landmark.copy()
        want_sw[sw] = float(np.float32(truth["truth_id"][sw]))
        if not np.array_equal(starts["two_pass"], landmark):
            problems.append(f"{tag}: two_pass does not start from the landmark start")
        if not np.array_equal(starts["warm"].astype(np.float32), truth["truth_id"].astype(np.float32)):
            problems.append(f"{tag}: warm does not start from the truth")
        if not np.array_equal(starts["sw_star"], want_sw):
            problems.append(f"{tag}: sw_star is not the landmark start with only shoulder width at the truth")
        # pass 1 of two-pass IS D4c's one pass (the same code at passes = 1, proved by the tripwire)
        pass1 = np.asarray(group["two_pass"]["record"]["calibration_calls"][1]["identity_returned"], float)
        if not np.array_equal(pass1, group["d4c_start"]["fitted_id"]):
            problems.append(f"{tag}: two-pass's first pass is not D4c's one-pass identity")
    return cells


def call_readings(cell: dict, floor: np.ndarray, scored: list[str]) -> list[dict]:
    names = cell["names"]
    spine = names.index("scale_spine_length")
    out = []
    for call in cell["record"].get("calibration_calls") or []:
        rows = d4bg.segment_rows(cell["truth_rest"], np.asarray(call["rest_mapped_cm_after"], float), floor, scored)
        out.append({"stage": call["stage"],
                    "spine_error": float(call["identity_returned"][spine] - cell["truth_id"][spine]),
                    "shoulder_width_error": float(call["identity_returned"][names.index("scale_shoulder_width")]
                                                  - cell["truth_id"][names.index("scale_shoulder_width")]),
                    "trunk_ratio": rows["trunk"]["error_mm"] / rows["trunk"]["tolerance_mm"],
                    "worst_scored_segment_ratio": max(r["error_mm"] / r["tolerance_mm"] for r in rows.values() if r["scored"]),
                    "L_passes": d4bg.l_passes(rows),
                    "locator_offset_mm_max_after": call.get("locator_offset_mm_max_after")})
    return out


def phase1(inputs: dict) -> dict:
    problems: list[str] = []
    pre0 = precondition_0(inputs["drawn_bytes"])
    cells = check_phase1(inputs, problems)
    scored = pre0["scored"]
    drawn = pre0["drawn_set"]
    population = phase1_population()
    rows = {}
    for p, s, d in population:
        if not all((p, s, d, a) in cells for a in fx4.PHASE1_ARMS):
            continue
        floor = d4bg.floors_mm(cells[(p, s, d, "exact_identity")]["distance"])
        names = cells[(p, s, d, "warm")]["names"]
        truth = cells[(p, s, d, "exact_identity")]["truth_id"]
        spine_truth = float(truth[names.index("scale_spine_length")])
        row = {"population": p, "spine_draw": spine_truth, "spine_tercile": fx4.spine_tercile(spine_truth),
               "floor_mm": {j: float(floor[i]) for i, j in enumerate(MAPPED_JOINTS)},
               "segments": {a: d4bg.segment_rows(cells[(p, s, d, a)]["truth_rest"], cells[(p, s, d, a)]["fitted_rest"],
                                                 floor, scored) for a in fx4.PHASE1_ARMS},
               "pooled_mm": {a: d4bg.pooled_mm(cells[(p, s, d, a)]["distance"]) for a in fx4.PHASE1_ARMS}}
        start_of = {"d4c_start": np.asarray(cells[(p, s, d, "d4c_start")]["record"]["start"]["start_identity"], float)}
        for a in fx4.PHASE1_NEW_ARMS:
            start_of[a] = np.asarray(cells[(p, s, d, a)]["record"]["start"]["start_identity"], float)
        row["start_error"] = {a: {c: float(start_of[a][names.index(c)] - truth[names.index(c)]) for c in drawn}
                              for a in start_of}
        row["recovered_error"] = {a: {c: float(cells[(p, s, d, a)]["fitted_id"][names.index(c)] - truth[names.index(c)])
                                      for c in drawn} for a in start_of}
        row["per_stage"] = {a: call_readings(cells[(p, s, d, a)], floor, scored) for a in fx4.PHASE1_NEW_ARMS}
        row["cap_exhaustion"] = {a: [{k: c.get(k) for k in ("stage", "solves", "stopped_below_the_cap",
                                                             "stopped_at_the_configured_cap",
                                                             "stopped_above_the_configured_cap", "max_index")}
                                     for c in cells[(p, s, d, a)]["record"]["calibration_iterations"]["per_call"]]
                                 for a in fx4.PHASE1_NEW_ARMS}
        rows[f"{p}/{s}_d{d}"] = row
    complete = len(rows) == len(population) and not problems

    def ratio(row: dict, arm: str, segment: str = "trunk") -> float:
        r = row["segments"][arm][segment]
        return r["error_mm"] / r["tolerance_mm"] if r["tolerance_mm"] > 0 else math.inf

    stage_0b = {k: bool(v["segments"]["spine_displaced"]["trunk"]["scored"]
                        and v["segments"]["spine_displaced"]["trunk"]["error_mm"] > v["segments"]["spine_displaced"]["trunk"]["tolerance_mm"])
                for k, v in rows.items()}
    fork_rows = {k: v for k, v in rows.items() if v["population"] == "d4c"}
    warm_within = {k: bool(v["segments"]["warm"]["trunk"]["error_mm"] <= v["segments"]["warm"]["trunk"]["tolerance_mm"])
                   for k, v in fork_rows.items()}
    two_pass_l = {k: d4bg.l_passes(v["segments"]["two_pass"]) for k, v in fork_rows.items()}
    warm_holds = complete and len(warm_within) == 12 and all(warm_within.values())
    two_pass_closes = complete and len(two_pass_l) == 12 and all(two_pass_l.values())
    if not pre0["holds"]:
        selected = "STOP-precondition-0"
    elif not complete:
        selected = "FAIL-population"
    elif not all(stage_0b.values()):
        selected = "STOP-stage-0b"
    elif not warm_holds:
        selected = "STOP-WARM-drift"
    elif two_pass_closes:
        selected = "TWO-PASS"
    else:
        selected = "STOP-two-pass-does-not-close"

    def summary(arm: str, keep) -> dict:
        chosen = {k: v for k, v in rows.items() if keep(v)}
        return {"fixtures": len(chosen),
                "L_passes": sum(d4bg.l_passes(v["segments"][arm]) for v in chosen.values()),
                "trunk_within": sum(v["segments"][arm]["trunk"]["within"] for v in chosen.values()),
                "worst_trunk_ratio": max((ratio(v, arm) for v in chosen.values()), default=None),
                "worst_scored_segment_ratio": max((max(r["error_mm"] / r["tolerance_mm"] for r in v["segments"][arm].values()
                                                       if r["scored"]) for v in chosen.values()), default=None)}

    arms = ("d4c_start", "warm", "two_pass", "sw_star", "spine_displaced")
    by_population = {p: {a: summary(a, lambda v, p=p: v["population"] == p) for a in arms} for p in fx4.PHASE1}
    by_tercile = {p: {t: {a: summary(a, lambda v, p=p, t=t: v["population"] == p and v["spine_tercile"] == t)
                          for a in arms} for t in ("short", "middle", "long")} for p in fx4.PHASE1}
    return {
        "step": "D4d", "stage": 3,
        "what": "PHASE 1, the diagnostic, on burned fixtures only: the frozen fork decided by WARM and TWO-PASS on D4c's "
                "12; D4's 6 and D4b's 12 reported",
        "rule": RULE,
        "fork_population": [k for k in rows if rows[k]["population"] == "d4c"] if complete else None,
        "precondition_0_holds": pre0["holds"],
        "complete": complete,
        "problems": problems,
        "stage_0b": {"spine_displaced_fails_L_at_the_trunk_as_scored": stage_0b,
                     "fixtures_failing": sum(stage_0b.values()), "fixtures": len(stage_0b),
                     "holds": complete and all(stage_0b.values())},
        "fork": {"warm_trunk_within_tolerance": warm_within, "warm_holds_on_all_12": warm_holds,
                 "warm_trunk_ratio": {k: ratio(v, "warm") for k, v in fork_rows.items()},
                 "two_pass_L_passes": two_pass_l, "two_pass_closes_every_segment_on_all_12": two_pass_closes,
                 "two_pass_worst_segment_ratio": {k: max(r["error_mm"] / r["tolerance_mm"] for r in v["segments"]["two_pass"].values()
                                                         if r["scored"]) for k, v in fork_rows.items()}},
        "selected": selected,
        "phase_2_licensed": selected == "TWO-PASS",
        "spine_tercile_rule": "scale_spine_length's configured limit [-1.1, 1.1] in equal thirds (short < -0.3667 < "
                              "middle < 0.3667 < long), frozen in d4d_fixture.spine_tercile before any reading",
        "by_population": by_population,
        "by_spine_tercile": by_tercile,
        "fixtures": rows,
        "REPORTED_warm_equals_D4b_s_retained_warm_fitted_identity": {
            k: (v is not None and (k.split("/")[0], int(k.split("/")[1].split("_d")[0]), int(k.split("_d")[1]), "warm") in cells
                and np.array_equal(np.asarray(v, float), cells[(k.split("/")[0], int(k.split("/")[1].split("_d")[0]),
                                                                int(k.split("_d")[1]), "warm")]["fitted_id"]))
            for k, v in sorted(inputs.get("d4b_warm", {}).items())},
        "cells_sha256": dict(sorted(inputs["new_files"].items())),
        "retained_cells_bound": {k: {"name": v["name"], "file_sha256": v["file_sha256"],
                                     "sorted_dump_sha256": v["sorted_dump_sha256"]}
                                 for k, v in sorted(inputs["retained"].items())},
        "ran_on": {"fitter_sha256": inputs["fitter_sha256"], "fixture_sha256": inputs["fixture_sha256"]},
    }


# --------------------------------------------------------------------------------------------- Phase 2

PHASE2_DIR = OUT_DIR / "phase2"
MANIFEST = RECORDS / "acceptance-manifest.json"
VALIDITY_CEILING_MM = 1.0
CLOSURE_BAND_M = 1e-4
EXACT_ZERO_MM = 1e-9
B1_CANDIDATE, B1_BASELINE, B1_D4C = "D4d_fitted_MHR_lod2", "baseline_D7c_rig", "D4c_fitted_MHR_lod2"
B1_CAMERAS = ("A001", "B001", "C001", "D001")
SUBJECTS = ("subject_00", "subject_01")
B2_KEYS = ("1a_handed_array_is_byte_identical_to_the_rig_converter_input_on_this_build",
           "1b_handed_array_is_byte_identical_to_the_rig_BUILD_s_delivered_array",
           "1c_the_array_is_the_SMOOTHED_repaired_one", "2a_joint_names_and_order_identical_everywhere",
           "2b_validity_mask_identical", "3_frame_order_identical",
           "4a_marker_values_match_the_declared_mapping_and_conversion", "4b_occlusion_flags_match",
           "marker_names_are_the_declared_map")


def added_commit(path: Path) -> str:
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else ""


def load_phase2(out_dir: Path = OUT_DIR) -> dict:
    phase2_dir = out_dir / "phase2"
    inputs = {"cells": {}, "files": {}, "arrays": {}}
    for path in sorted(phase2_dir.glob("cell-*.json")):
        inputs["cells"][path.name] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(phase2_dir.glob("fixture-*")):
        if path.suffix in (".glb", ".npz"):
            inputs["files"][path.name] = sha256_file(path)
        if path.name.endswith(".arrays.npz") and any(f"-{a}." in path.name for a in ("candidate", "init_only", "one_pass")):
            data = np.load(path)          # no pickle: plain arrays written by the fixture
            inputs["arrays"][path.name] = {
                "consumed_landmarks_z_up_m": np.asarray(data["consumed_landmarks_z_up_m"]),
                "consumed_joint_names": [str(n) for n in data["consumed_joint_names"]],
                "truth_mapped_cm": np.asarray(data["truth_mapped_cm"])}
    inputs["cell_file_sha256"] = {n: sha256_file(phase2_dir / n) for n in sorted(inputs["cells"])}
    inputs["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    decision_commit, manifest_commit = added_commit(DECISION), added_commit(MANIFEST) if MANIFEST.exists() else ""
    inputs["commits"] = {
        "decision": decision_commit, "manifest": manifest_commit,
        "decision_time": int(subprocess.run(["git", "show", "-s", "--format=%ct", decision_commit], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip() or 0) if decision_commit else 0,
        "manifest_time": int(subprocess.run(["git", "show", "-s", "--format=%ct", manifest_commit], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip() or 0) if manifest_commit else 0,
        "decision_is_ancestor": bool(decision_commit and manifest_commit and subprocess.run(
            ["git", "merge-base", "--is-ancestor", decision_commit, manifest_commit], cwd=ROOT).returncode == 0),
        "decision_committed_bytes_sha256": sha256_bytes(git_bytes(f"{decision_commit}:{DECISION.relative_to(ROOT)}"))
        if decision_commit else None,
        "manifest_committed_bytes_sha256": sha256_bytes(git_bytes(f"HEAD:{MANIFEST.relative_to(ROOT)}"))
        if manifest_commit else None,
    }
    inputs["manifest_bytes_sha256"] = sha256_file(MANIFEST)
    closure = out_dir / "closure.json"
    inputs["closure"] = json.loads(closure.read_text(encoding="utf-8")) if closure.exists() else {}
    for name in ("b1-paired.json", "silhouette-delivery.json"):
        path = out_dir / name
        inputs[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    inputs["silhouette_committed"] = json.loads((ROOT / "artifacts/compare/silhouette.json").read_text(encoding="utf-8"))
    delivery = out_dir / "delivery"
    inputs["b1_files"] = {"D4d_mesh": sha256_file(out_dir / "work-delivery/delivered-mesh.npz"),
                          "silhouette_mesh": sha256_file(out_dir / "work-silhouette/delivered-mesh.npz"),
                          "D4c_mesh": sha256_file(ROOT / "artifacts/compare/d4c-start/work-delivery/delivered-mesh.npz"),
                          **{f"delivery/subject-{s:02d}.glb": sha256_file(delivery / f"subject-{s:02d}.glb") for s in (0, 1)}}
    inputs["delivery"] = load_delivery(delivery, out_dir / "hygiene")
    return inputs


def load_delivery(delivery: Path, rig_build: Path) -> dict:
    """Every file B2 and the combined-fitter binding read, loaded once (the repository's OWN build output under
    artifacts/, npz with object arrays: allow_pickle; CLAUDE.md), with its sha256."""
    out = {"files_sha256": {}, "subjects": {}}
    inputs = delivery / "converter-inputs"

    def load(path: Path):
        out["files_sha256"][str(path.relative_to(OUT_DIR)) if OUT_DIR in path.parents else str(path)] = sha256_file(path)
        return np.load(path, allow_pickle=True) if path.is_file() else None

    rig_side = load(inputs / "rig-converter-input.npz")
    out["rig_converter_input"] = None if rig_side is None else {k: np.asarray(rig_side[k]) for k in rig_side.files}
    for s in (0, 1):
        row = {}
        for label, path in (("consumed", inputs / f"subject-{s:02d}-consumed.npz"),
                            ("rig_track", rig_build / f"subject-{s:02d}.body-track.npz"),
                            ("markers", delivery / f"subject-{s:02d}.markers.npz"),
                            ("track", delivery / f"subject-{s:02d}.body-track.npz")):
            data = load(path)
            row[label] = None if data is None else {k: np.asarray(data[k]) for k in data.files}
        for label, name in (("track_json", f"subject-{s:02d}.body-track.json"),
                            ("start_json", f"subject-{s:02d}.calibration-start.json"),
                            ("passes_json", f"subject-{s:02d}.calibration-passes.json")):
            path = delivery / name
            out["files_sha256"][f"delivery/{name}"] = sha256_file(path)
            row[label] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        out["subjects"][f"subject_{s:02d}"] = row
    return out


def _mhr_cm(p_zup_m: np.ndarray) -> np.ndarray:
    """Capture Z-up metres -> MHR Y-up centimetres, `(x, z, -y) * 100`, written out from the card's words."""
    return np.stack([p_zup_m[..., 0], p_zup_m[..., 2], -p_zup_m[..., 1]], axis=-1) * 100.0


def _bytes_equal(a, b) -> bool:
    return a is not None and b is not None and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def b2_rederived(delivery: dict) -> dict:
    """D4's B2, every numbered check RE-DERIVED here from the files (never the producer's booleans)."""
    checks = {}
    rig_side = delivery.get("rig_converter_input")
    for s in SUBJECTS:
        row = delivery["subjects"].get(s) or {}
        need = ("consumed", "rig_track", "markers", "track", "track_json")
        if rig_side is None or any(row.get(k) is None for k in need):
            checks[s] = {k: False for k in B2_KEYS}
            continue
        consumed, rig_track, markers, track = row["consumed"], row["rig_track"], row["markers"], row["track"]
        try:
            key = str(markers["source_array_key"])
            handed = np.asarray(consumed[key])
            declared = row["track_json"]["landmark_to_joint"]
            names_consumed = [str(n) for n in consumed["joint_names"]]
            rig_input = np.asarray(rig_side[f"{s}_triangulated_world_positions_z_up_m"])
            rig_delivered = np.asarray(rig_track["triangulated_world_positions_z_up_m"])
            array_cm = _mhr_cm(np.asarray(handed, np.float64))
            landmarks = list(declared)
            expected = np.zeros((handed.shape[0], len(landmarks), 3))
            occluded = np.zeros((handed.shape[0], len(landmarks)), bool)
            for column, landmark in enumerate(landmarks):
                values = array_cm[:, names_consumed.index(landmark)]
                finite = np.isfinite(values).all(axis=1)
                expected[finite, column] = values[finite]
                occluded[:, column] = ~finite
            checks[s] = {
                B2_KEYS[0]: _bytes_equal(handed, rig_input),
                B2_KEYS[1]: _bytes_equal(handed, rig_delivered),
                B2_KEYS[2]: key == "triangulated_world_positions_z_up_m",
                B2_KEYS[3]: (names_consumed == [str(n) for n in markers["source_joint_names"]]
                             == [str(n) for n in track["consumed_joint_names"]] == [str(n) for n in rig_side["joint_names"]]),
                B2_KEYS[4]: bool(np.array_equal(np.isfinite(handed), np.isfinite(rig_delivered))),
                B2_KEYS[5]: bool(np.array_equal(np.asarray(consumed["ticks"]), np.asarray(rig_track["ticks"]))),
                B2_KEYS[6]: bool(np.array_equal(expected, np.asarray(markers["marker_positions_mhr_cm"]))),
                B2_KEYS[7]: bool(np.array_equal(occluded, np.asarray(markers["marker_occluded"]))),
                B2_KEYS[8]: [str(n) for n in markers["marker_names"]] == landmarks,
            }
        except (KeyError, ValueError, TypeError, IndexError):
            checks[s] = {k: False for k in B2_KEYS}
    return {"pass": all(all(v is True for v in checks[s].values()) and set(checks[s]) == set(B2_KEYS) for s in SUBJECTS),
            "checks": checks}


def combined_fitter_binding(delivery: dict, drawn: dict, statistic: str | None) -> dict:
    """The rebuilt delivery IS the combined fitter's: two passes, the landmark start, and that start equal to the
    frozen rule recomputed here from the consumed (smoothed) array."""
    rows = {}
    for s in SUBJECTS:
        row = delivery["subjects"].get(s) or {}
        passes = (row.get("passes_json") or {}).get("passes")
        start = row.get("start_json") or {}
        ok_start = False
        try:
            consumed = row["consumed"]
            names = [str(n) for n in consumed["joint_names"]]
            recomputed = d4cg.recompute_start(np.asarray(consumed["triangulated_world_positions_z_up_m"]), names, drawn,
                                              statistic)
            recorded = start.get("identity") or {}
            ok_start = (start.get("mode") == "landmark" and start.get("trunk_statistic") == statistic
                        and start.get("consumed_array_key") == "triangulated_world_positions_z_up_m"
                        and all(abs(recorded.get(c, 0.0) - v) <= START_AGREEMENT for c, v in recomputed.items())
                        and set(recorded) <= set(recomputed))
        except (KeyError, TypeError, ValueError):
            ok_start = False
        rows[s] = {"passes": passes, "two_passes": passes == 2, "landmark_start_is_the_frozen_rule_s": bool(ok_start)}
    return {"holds": all(r["two_passes"] and r["landmark_start_is_the_frozen_rule_s"] for r in rows.values()), "rows": rows}


def b1_clauses(inputs: dict, binding: dict) -> dict:
    paired, sil, committed = inputs["b1-paired.json"], inputs["silhouette-delivery.json"], inputs["silhouette_committed"]
    problems = []
    pop = paired.get("population") or {}
    consumed = pop.get("frames_consumed_per_arm") or {}
    arms = (B1_CANDIDATE, B1_BASELINE, B1_D4C)
    for arm in arms:
        if any((consumed.get(arm) or {}).get(s) != FRAMES for s in SUBJECTS):
            problems.append(f"B1: arm {arm} did not consume 150 frames on both performers")
    excluded = pop.get("cells_excluded_by_the_mask_cache") or {}
    required = {s: len(B1_CAMERAS) * FRAMES - len(excluded.get(s, [None] * 10**6)) for s in SUBJECTS}
    rows = {}
    for s in SUBJECTS:
        for cand, ref in ((B1_CANDIDATE, B1_BASELINE), (B1_CANDIDATE, B1_D4C)):
            row = (paired.get("paired") or {}).get(f"{cand}_minus_{ref}_{s}") or {}
            ci = row.get("ci95_of_the_median_difference")
            if row.get("n") != required[s] or pop.get("cells_required", {}).get(s) != required[s] or required[s] <= 0:
                problems.append(f"B1: {cand} - {ref} {s} not scored on the named population")
            if not (isinstance(ci, list) and len(ci) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in ci)):
                problems.append(f"B1: {cand} - {ref} {s} carries no finite CI")
                ci = [math.nan, math.nan]
            if row.get("block_length") != 15 or row.get("resamples") != 2000:
                problems.append(f"B1: {cand} - {ref} {s} is not block 15 x 2000")
            rows[f"{cand}_minus_{ref}_{s}"] = {"median": row.get("median_difference"), "ci95": ci}
    band = {s: rows[f"{B1_CANDIDATE}_minus_{B1_BASELINE}_{s}"]["ci95"][0] > 0 for s in SUBJECTS}
    files = inputs["b1_files"]
    arm_sha = paired.get("arm_file_sha256") or {}
    bind = {
        "paired_candidate_mesh_is_the_file": arm_sha.get(B1_CANDIDATE) == files["D4d_mesh"] and files["D4d_mesh"] is not None,
        "paired_d4c_mesh_is_d4c_s_file": arm_sha.get(B1_D4C) == files["D4c_mesh"] and files["D4c_mesh"] is not None,
        "silhouette_mesh_is_the_same_file": files["silhouette_mesh"] == files["D4d_mesh"],
        "silhouette_read_the_rebuilt_glbs": all(
            (sil.get("input_sha256") or {}).get(f"subject-{s:02d}.glb") == files[f"delivery/subject-{s:02d}.glb"]
            and files[f"delivery/subject-{s:02d}.glb"] is not None for s in (0, 1)),
        "paired_candidate_medians_equal_the_silhouette_run": all(
            ((paired.get("arms") or {}).get(B1_CANDIDATE) or {}).get(s, {}).get(cam)
            == round(float(((sil.get("arms") or {}).get("ours_delivered") or {}).get(cam, {}).get(s, {}).get(
                "iou", {}).get("median", math.nan)), 4) for s in SUBJECTS for cam in B1_CAMERAS),
        "the_delivery_is_the_combined_fitter_s": binding["holds"],
    }
    ours = (sil.get("arms") or {}).get("ours_delivered") or {}
    frozen = (sil.get("arms") or {}).get("control_frozen_pose_tracked") or {}
    frozen_below = {f"{cam}_{s}": bool((frozen.get(cam, {}).get(s, {}).get("iou", {}).get("median", math.inf))
                                       < (ours.get(cam, {}).get(s, {}).get("iou", {}).get("median", -math.inf)))
                    for cam in B1_CAMERAS for s in SUBJECTS}
    mamma_new = (sil.get("arms") or {}).get("ORACLE_mamma_mesh") or {}
    mamma_old = (committed.get("arms") or {}).get("ORACLE_mamma_mesh") or {}
    mamma_same = {f"{cam}_{s}": bool(mamma_new.get(cam, {}).get(s) is not None
                                     and mamma_new.get(cam, {}).get(s) == mamma_old.get(cam, {}).get(s))
                  for cam in B1_CAMERAS for s in SUBJECTS}
    ok = not problems and all(band.values()) and all(bind.values()) and all(frozen_below.values()) and all(mamma_same.values())
    return {"pass": bool(ok), "band_lower_ci_above_zero": band, "binding": bind, "frozen_pose_below": frozen_below,
            "mamma_bit_identical": mamma_same, "pairs": rows, "problems": problems,
            "arms_pooled_median_iou": {a: {s: ((paired.get("arms") or {}).get(a) or {}).get(s, {}).get("pooled_median_iou")
                                           for s in SUBJECTS} for a in arms},
            "REPORTED_D4d_minus_D4c": {s: rows[f"{B1_CANDIDATE}_minus_{B1_D4C}_{s}"] for s in SUBJECTS}}


def check_phase2(inputs: dict, phase1_inputs: dict, decision_sha: str, problems: list[str]) -> dict:
    drawn = json.loads(phase1_inputs["drawn_bytes"])
    statistic = json.loads(phase1_inputs["d4c_development_bytes"]).get("chosen_trunk_statistic")
    prov = expected_new_provenance(phase1_inputs)
    population = fx4.POPULATIONS["phase2"]
    arms = fx4.PHASE2_ARMS
    expected = {f"cell-{s}-d{d}-{a}.json" for s, d in population for a in arms}
    for name in sorted(expected - set(inputs["cells"])):
        problems.append(f"phase 2: {name} missing")
    for name in sorted(set(inputs["cells"]) - expected):
        problems.append(f"phase 2: {name} is not in the named population")
    cells, canonical = {}, None
    for s, d in population:
        for a in arms:
            record = inputs["cells"].get(f"cell-{s}-d{d}-{a}.json")
            if record is None:
                continue
            tag, bad = f"phase 2 {s}/d{d}/{a}", []
            if (record.get("schema"), record.get("seed"), record.get("donor"), record.get("arm"),
                    record.get("population")) != (fx4.SCHEMA, s, d, a, "phase2"):
                bad.append("identity fields")
            if record.get("lod") != 2 or record.get("frames") != FRAMES or list(record.get("mapped_joints") or []) != list(MAPPED_JOINTS):
                bad.append("lod, frames or mapped joints")
            if record.get("phase1_decision_sha256") != decision_sha or not decision_sha:
                bad.append("the cell does not carry the committed Phase-1 decision's sha256")
            arr = _cell_arrays(record)
            if arr is None:
                bad.append("arrays short, missing or non-finite")
            names = list(record.get("identity_channel_names") or [])
            if not _names_ok(names):
                bad.append("identity channel names")
            elif canonical is None:
                canonical = names
            elif names != canonical:
                bad.append("identity channel names differ between cells")
            layout = record.get("truth_rest_full_vs_simplified_character_max_abs_cm")
            if not isinstance(layout, (int, float)) or not math.isfinite(layout) or layout > LAYOUT_AGREEMENT_CM:
                bad.append("the truth rest differs between the 204- and 178-parameter characters")
            got = record.get("provenance") or {}
            for field, value in prov.items():
                if value is None or got.get(field) != value:
                    bad.append(f"provenance {field}")
            settings = record.get("settings") or {}
            for field, value in SETTINGS.items():
                if settings.get(field) != value:
                    bad.append(f"settings {field}")
            if inputs["files"].get(record.get("arrays")) != record.get("arrays_sha256") or not record.get("arrays_sha256"):
                bad.append("arrays file hash")
            start = record.get("start") or {}
            if start.get("trunk_statistic") != statistic:
                bad.append("start: trunk statistic")
            if a in fx4.CALIBRATING:
                if start.get("kind") != fx4.CALIBRATING[a][0]:
                    bad.append("start kind")
                if arr is not None and not bad:
                    check_calibrating(record, arr, names, fx4.CALIBRATING[a][1], bad)
            else:
                if record.get("calibrate_markers_calls") != 0 or (settings.get("passes"), settings.get(
                        "calibration_debug_captured")) != (0, False) or record.get("calibration_calls"):
                    bad.append("a non-calibrating arm calibrated")
            if a != "spine_displaced" and a != "sw_star" and record.get("arm_note") != {}:
                bad.append("arm_note on an arm that carries none")
            if a == "candidate":
                for key, sha_key in (("glb", "glb_sha256"), ("track", "track_sha256")):
                    if not record.get(key) or inputs["files"].get(record.get(key)) != record.get(sha_key):
                        bad.append(f"{key} file hash")
            if bad:
                problems.append(f"{tag}: " + "; ".join(bad))
                continue
            cells[(s, d, a)] = dict(arr, record=record, names=names)
    truths, clamps = {}, {}
    for s, d in population:
        group = {a: cells.get((s, d, a)) for a in arms}
        if any(c is None for c in group.values()):
            continue
        tag = f"phase 2 fixture {s}/d{d}"
        first = group["exact_identity"]
        names = first["names"]
        fixture = first["record"].get("fixture") or {}
        if any(c["record"].get("fixture") != fixture or not np.array_equal(c["truth_id"], first["truth_id"])
               or not np.array_equal(c["truth_rest"], first["truth_rest"]) for c in group.values()):
            problems.append(f"{tag}: the arms do not share one truth")
            continue
        if fixture.get("generator") != f"numpy default_rng([{s}, {d}]) (seeded by the pair)" or list(
                fixture.get("drawn_channels") or []) != list(drawn["drawn_set"]):
            problems.append(f"{tag}: generator or drawn channels")
        regenerated = fx4.fx.acceptance_draw(s, d, list(drawn["drawn_set"]), drawn["configured_limits"], CHANNELS)
        if any(np.float32(first["truth_id"][i]) != np.float32(regenerated.get(n, 0.0)) for i, n in enumerate(names)):
            problems.append(f"{tag}: the truth is not the regenerated draw")
        clamps.setdefault(d, set()).add(fixture.get("clamped_parameter_frames"))
        truths[(s, d)] = first["truth_id"]
        if not np.array_equal(group["exact_identity"]["fitted_id"], first["truth_id"]):
            problems.append(f"{tag}: exact_identity is not the truth")
        if np.any(group["mean_body"]["fitted_id"] != 0.0):
            problems.append(f"{tag}: mean_body is not zero")
        spine = names.index("scale_spine_length")
        sd = group["spine_displaced"]
        other = np.arange(len(names)) != spine
        if (not np.array_equal(sd["fitted_id"][other], sd["truth_id"][other])
                or abs(sd["fitted_id"][spine] - fx4.fx.displaced_spine(float(sd["truth_id"][spine]))) > 1e-6):
            problems.append(f"{tag}: spine_displaced is not the truth with the spine moved")
        arr = inputs["arrays"].get(group["candidate"]["record"].get("arrays"))
        if arr is None:
            problems.append(f"{tag}: the consumed landmarks are not readable")
            continue
        consumed, joint_names, truth_cm = arr["consumed_landmarks_z_up_m"], arr["consumed_joint_names"], arr["truth_mapped_cm"]
        as_capture = np.stack([truth_cm[..., 0], -truth_cm[..., 2], truth_cm[..., 1]], axis=-1) / 100.0
        columns = [joint_names.index(l) for l in d4cg.MAP]
        if consumed.shape != (FRAMES, len(joint_names), 3) or np.nanmax(np.abs(consumed[:, columns] - as_capture)) * 100.0 > LANDMARK_AGREEMENT_CM:
            problems.append(f"{tag}: the consumed landmarks are not the truth's joints")
            continue
        recomputed = d4cg.recompute_start(consumed, joint_names, drawn, statistic)
        want = np.array([recomputed.get(n, 0.0) for n in names])
        landmark = np.asarray(group["candidate"]["record"]["start"]["landmark_start_identity"], float)
        if np.max(np.abs(landmark - want)) > START_AGREEMENT:
            problems.append(f"{tag}: the landmark start is not the frozen rule's")
        for a in arms:
            if not np.array_equal(np.asarray(group[a]["record"]["start"]["landmark_start_identity"], float), landmark):
                problems.append(f"{tag}/{a}: a different landmark start")
        for a in ("candidate", "one_pass"):
            if not np.array_equal(np.asarray(group[a]["record"]["start"]["start_identity"], float), landmark):
                problems.append(f"{tag}/{a}: does not start from the landmark start")
        if group["legacy"]["record"]["start"].get("start_identity") is not None:
            problems.append(f"{tag}/legacy: not the zero start")
        held = np.asarray(group["init_only"]["record"]["start"].get("held_identity") or [], float)
        if held.shape != landmark.shape or not np.array_equal(held, landmark) or not np.array_equal(
                group["init_only"]["fitted_id"].astype(np.float32), landmark.astype(np.float32)):
            problems.append(f"{tag}/init_only: not the landmark start held")
        pass1 = np.asarray(group["candidate"]["record"]["calibration_calls"][1]["identity_returned"], float)
        if not np.array_equal(pass1, group["one_pass"]["fitted_id"]):
            problems.append(f"{tag}: the candidate's first pass is not the one-pass arm's identity")
    for d, counts in clamps.items():
        if len(counts) != 1 or not all(isinstance(c, int) and c > 0 for c in counts):
            problems.append(f"phase 2 donor {d}: the fixture clamp is not one donor-determined count")
    keys = list(truths)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.array_equal(truths[keys[i]], truths[keys[j]]):
                problems.append(f"phase 2: fixtures {keys[i]} and {keys[j]} share one identity draw")
    return cells


# ---------------------------------------------------------------------------------------------- build

def build(phase1_inputs: dict, phase2_inputs: dict | None, hygiene_inputs: dict | None = None) -> dict:
    problems: list[str] = []
    pre0 = precondition_0(phase1_inputs["drawn_bytes"])
    ph1 = phase1(phase1_inputs)
    # the committed decision must be the one these cells re-derive, byte for byte in content
    decision_bytes = phase1_inputs.get("decision_bytes") or b""
    committed = phase1_inputs.get("decision_committed_bytes") or b""
    decision_sha = sha256_bytes(decision_bytes) if decision_bytes else None
    try:
        decision = json.loads(decision_bytes) if decision_bytes else {}
    except ValueError:
        decision = {}
    decision_ok = bool(decision_bytes and committed == decision_bytes
                       and decision == json.loads(json.dumps(ph1, default=float)))
    if not decision_ok:
        problems.append("phase 1: the committed decision JSON is missing, differs from its committed bytes, or is not "
                        "re-derivable from the Phase-1 cells")
    selected = ph1["selected"]
    stage_0b = ph1["stage_0b"]["holds"]
    rule_two_pass = selected == "TWO-PASS"
    conjuncts = {"precondition_0": pre0["holds"], "phase1_complete": ph1["complete"], "stage_0b": stage_0b,
                 "the_rule_selects_TWO_PASS": rule_two_pass, "phase1_decision_committed_and_rederived": decision_ok}
    clauses: dict = {}
    reported: dict = {"phase1_by_population": ph1["by_population"], "phase1_by_spine_tercile": ph1["by_spine_tercile"],
                      "phase1_fork": ph1["fork"]}
    cells, rows = {}, {}
    if rule_two_pass and decision_ok and phase2_inputs is not None:
        order = phase2_inputs["commits"]
        order_ok = bool(order.get("decision") and order.get("manifest") and order["decision"] != order["manifest"]
                        and order.get("decision_is_ancestor") and order["decision_time"] <= order["manifest_time"]
                        and order.get("decision_committed_bytes_sha256") == decision_sha
                        and order.get("manifest_committed_bytes_sha256") == phase2_inputs.get("manifest_bytes_sha256"))
        if not order_ok:
            problems.append("freeze order: the decision JSON's commit is not older than the acceptance manifest's, or a "
                            "file on disk is not its committed copy")
        manifest = phase2_inputs.get("manifest") or {}
        if (manifest.get("cells_sha256") != phase2_inputs.get("cell_file_sha256") or not manifest.get("cells_sha256")
                or manifest.get("phase1_decision_sha256") != decision_sha):
            problems.append("freeze order: the acceptance manifest does not name the Phase-2 cells by content, or not "
                            "under the committed decision")
        cells = check_phase2(phase2_inputs, phase1_inputs, decision_sha, problems)
        scored = pre0["scored"]
        for s, d in fx4.POPULATIONS["phase2"]:
            if not all((s, d, a) in cells for a in fx4.PHASE2_ARMS):
                continue
            floor = d4bg.floors_mm(cells[(s, d, "exact_identity")]["distance"])
            names = cells[(s, d, "candidate")]["names"]
            spine = float(cells[(s, d, "candidate")]["truth_id"][names.index("scale_spine_length")])
            rows[f"{s}_d{d}"] = {
                "spine_draw": spine, "spine_tercile": fx4.spine_tercile(spine),
                "floor_mm": {j: float(floor[i]) for i, j in enumerate(MAPPED_JOINTS)},
                "pooled_mm": {a: d4bg.pooled_mm(cells[(s, d, a)]["distance"]) for a in fx4.PHASE2_ARMS},
                "segments": {a: d4bg.segment_rows(cells[(s, d, a)]["truth_rest"], cells[(s, d, a)]["fitted_rest"], floor, scored)
                             for a in fx4.PHASE2_ARMS},
                "identity_error": {a: {n: float(cells[(s, d, a)]["fitted_id"][i] - cells[(s, d, a)]["truth_id"][i])
                                       for i, n in enumerate(names) if n in pre0["drawn_set"]} for a in fx4.PHASE2_ARMS},
                "per_stage": {a: call_readings(cells[(s, d, a)], floor, scored) for a in ("candidate", "one_pass", "legacy")},
                "cap_exhaustion": {a: cells[(s, d, a)]["record"]["calibration_iterations"]["per_call"]
                                   for a in ("candidate", "one_pass", "legacy")}}
    population = fx4.POPULATIONS["phase2"]
    population_ok = bool(rule_two_pass and decision_ok and not problems and len(rows) == len(population))

    def ratio(row: dict, arm: str) -> float:
        r = row["segments"][arm]["trunk"]
        return r["error_mm"] / r["tolerance_mm"] if r["tolerance_mm"] > 0 else math.inf

    closure_rows, closure_problems = {}, []
    if population_ok:
        pairs = (phase2_inputs.get("closure") or {}).get("pairs") or {}
        for s, d in population:
            tag, record = f"{s}_d{d}", cells[(s, d, "candidate")]["record"]
            pair = pairs.get(tag) or {}
            ok = (pair.get("glb_sha256") == record.get("glb_sha256") == phase2_inputs["files"].get(record.get("glb"))
                  and pair.get("track_sha256") == record.get("track_sha256") == phase2_inputs["files"].get(record.get("track"))
                  and phase2_inputs["files"].get(record.get("glb")) is not None
                  and pair.get("frames_in_glb") == FRAMES and pair.get("frames_in_track") == FRAMES
                  and pair.get("joints_missing_from_the_glb") == [] and (pair.get("joints_compared") or 0) >= 127)
            value = pair.get("max_abs_m")
            finite = isinstance(value, (int, float)) and math.isfinite(value)
            closure_rows[tag] = {"max_abs_m": value, "bound_by_content": ok,
                                 "within": bool(ok and finite and value <= CLOSURE_BAND_M)}
            if not ok:
                closure_problems.append(f"closure {tag}: not this cell's GLB and track by content, or not a full read")
    validity = population_ok and all(v["pooled_mm"]["exact_identity"] <= VALIDITY_CEILING_MM for v in rows.values())
    l_fixture = {k: d4bg.l_passes(v["segments"]["candidate"]) for k, v in rows.items()}
    mean_misses = {k: d4bg.misses_l(v["segments"]["mean_body"]) for k, v in rows.items()}
    spine_fails = {k: bool(v["segments"]["spine_displaced"]["trunk"]["scored"]
                           and v["segments"]["spine_displaced"]["trunk"]["error_mm"]
                           > v["segments"]["spine_displaced"]["trunk"]["tolerance_mm"]) for k, v in rows.items()}
    exact_zero = {k: max(r["error_mm"] for r in v["segments"]["exact_identity"].values()) <= EXACT_ZERO_MM
                  and d4bg.l_passes(v["segments"]["exact_identity"]) for k, v in rows.items()}
    init_misses = {k: d4bg.misses_l(v["segments"]["init_only"]) for k, v in rows.items()}
    closure_ok = population_ok and len(closure_rows) == len(population) and all(
        r["within"] for r in closure_rows.values()) and not closure_problems
    if population_ok:
        binding = combined_fitter_binding(phase2_inputs["delivery"], json.loads(phase1_inputs["drawn_bytes"]),
                                          json.loads(phase1_inputs["d4c_development_bytes"]).get("chosen_trunk_statistic"))
        b1 = b1_clauses(phase2_inputs, binding)
        b2 = b2_rederived(phase2_inputs["delivery"])
    else:
        binding, b1, b2 = {"holds": False}, {"pass": False, "REPORTED_D4d_minus_D4c": None}, {"pass": False}
    hyg = hygiene(hygiene_inputs)
    conjuncts.update({
        "validity": validity,
        "L": population_ok and all(l_fixture.values()),
        "closure": closure_ok,
        "must_fail_i_mean_body_misses_L": population_ok and all(mean_misses.values()),
        "must_fail_ii_spine_displaced_fails_L_at_the_trunk": population_ok and all(spine_fails.values()),
        "must_fail_iii_exact_identity_reads_zero_and_passes": population_ok and all(exact_zero.values()),
        "must_fail_iv_init_only_misses_L": population_ok and all(init_misses.values()),
        "B1": bool(b1["pass"]), "B2": bool(b2["pass"]), "hygiene": bool(hyg["pass"]),
    })
    if not pre0["holds"]:
        verdict, reason = "STOP", "precondition 0: the spine is not drawn"
    elif not ph1["complete"]:
        verdict, reason = "FAIL", "phase 1: population, provenance or construction"
    elif not stage_0b:
        verdict, reason = "STOP", "stage 0b: the spine control passes L as scored on a Phase-1 fixture"
    elif selected == "STOP-WARM-drift":
        verdict, reason = "STOP", "the WARM fork: WARM leaves the trunk beyond tolerance on one of D4c's 12 (D4e takes the objective)"
    elif selected == "STOP-two-pass-does-not-close":
        verdict, reason = "STOP", "the fork: WARM holds but TWO-PASS does not close every segment on D4c's 12"
    elif not decision_ok:
        verdict, reason = "FAIL", "the Phase-1 decision JSON is not the committed, re-derivable one"
    elif not population_ok:
        verdict, reason = "FAIL", "phase 2: population, provenance, construction or freeze order"
    elif not validity:
        verdict, reason = "INVALID", "the exact_identity floor exceeds 1.0 mm on a fixture"
    elif not all(init_misses.values()):
        verdict, reason = "STOP", "init-only passes L on an acceptance fixture: the oracle cannot score the calibration"
    elif all(conjuncts.values()):
        verdict, reason = "PASS", "every conjunct holds"
    else:
        verdict, reason = "FAIL", "failed: " + ", ".join(k for k, v in conjuncts.items() if not v)
    consequence = {
        "PASS": "D4's O1 is superseded by the combined fitter (D4c's landmark start + two passes) with B1 re-shown on it; "
                "D4, D4c and D4d close; everything merges; --body mhr carries the combined fitter; the default stays rig",
        "FAIL": "records only on main; the branch is pinned by tag; D4 stays open",
        "INVALID": "records only on main; D4 stays open; INVALID never licenses replacing a fixture",
        "STOP": "records only on main; the branch is pinned by tag; D4 stays open" + (
            "; D4e takes the objective (the basin is its HYPOTHESIS)" if selected == "STOP-WARM-drift" else
            "; SW* is the lead for a landmark-derived shoulder-width start, which needs its own registration"
            if selected == "STOP-two-pass-does-not-close" else ""),
    }[verdict]
    if rows:
        reported.update({
            "one_pass_D4c_fitter": {k: {"L_passes": d4bg.l_passes(v["segments"]["one_pass"]), "trunk_ratio": ratio(v, "one_pass")}
                                    for k, v in rows.items()},
            "legacy_zero_start": {k: {"L_passes": d4bg.l_passes(v["segments"]["legacy"]), "trunk_ratio": ratio(v, "legacy")}
                                  for k, v in rows.items()},
            "candidate_trunk_ratio": {k: ratio(v, "candidate") for k, v in rows.items()},
            "worst_scored_segment_ratio": {a: {k: max(r["error_mm"] / r["tolerance_mm"] for r in v["segments"][a].values()
                                                      if r["scored"]) for k, v in rows.items()} for a in fx4.PHASE2_ARMS},
            "by_spine_tercile": {t: {"fixtures": [k for k, v in rows.items() if v["spine_tercile"] == t],
                                     "candidate_worst_trunk_ratio": max((ratio(v, "candidate") for v in rows.values()
                                                                         if v["spine_tercile"] == t), default=None)}
                                 for t in ("short", "middle", "long")},
            "J_candidate_minus_paired_floor_mm": {k: {j: float(np.median(cells[(int(k.split("_d")[0]), int(k.split("_d")[1]),
                                                                              "candidate")]["distance"][:, i]) - v["floor_mm"][j])
                                                      for i, j in enumerate(MAPPED_JOINTS)} for k, v in rows.items()},
            "cap_exhaustion_per_stage_and_pass": {k: v["cap_exhaustion"] for k, v in rows.items()},
            "per_stage_readings": {k: v["per_stage"] for k, v in rows.items()},
            "empty_intersection_of_allowable_rest_length_intervals": {
                seg: (lambda lows, highs: {"max_low_mm": max(lows), "min_high_mm": min(highs), "empty": max(lows) > min(highs)})(
                    [v["segments"]["candidate"][seg]["truth_mm"] - v["segments"]["candidate"][seg]["tolerance_mm"] for v in rows.values()],
                    [v["segments"]["candidate"][seg]["truth_mm"] + v["segments"]["candidate"][seg]["tolerance_mm"] for v in rows.values()])
                for seg in pre0["scored"]},
            "segments_all_arms": {k: v["segments"] for k, v in rows.items()},
            "pooled_statistic_mm_per_arm": {k: v["pooled_mm"] for k, v in rows.items()},
            "identity_error": {k: v["identity_error"] for k, v in rows.items()},
            "B1_REPORTED_D4d_minus_D4c": b1.get("REPORTED_D4d_minus_D4c"),
            "the_delivery_is_the_combined_fitter_s": binding,
        })
    worst_candidate = max((ratio(v, "candidate") for v in rows.values()), default=None)
    worst_row = max(rows.items(), key=lambda kv: ratio(kv[1], "candidate"))[0] if rows else None
    clauses = {
        "precondition_0": {"predicted": "(carried) the spine drawn", "measured": {"drawn": pre0["drawn_set"]},
                           "verdict": "HOLDS" if pre0["holds"] else "STOP", "prediction": "HELD" if pre0["holds"] else "FAILED"},
        "stage_0b": {"predicted": "the spine control fails L as scored on every Phase-1 fixture",
                     "measured": {"fixtures_failing": ph1["stage_0b"]["fixtures_failing"], "fixtures": ph1["stage_0b"]["fixtures"]},
                     "verdict": "HOLDS" if stage_0b else "STOP", "prediction": "HELD" if stage_0b else "FAILED"},
        "phase1_WARM": {"predicted": "WARM holds on 12/12",
                        "measured": {"within": sum(ph1["fork"]["warm_trunk_within_tolerance"].values()),
                                     "fixtures": len(ph1["fork"]["warm_trunk_within_tolerance"]),
                                     "worst_trunk_ratio": max(ph1["fork"]["warm_trunk_ratio"].values(), default=None),
                                     "20261106_d0_trunk_ratio": ph1["fork"]["warm_trunk_ratio"].get("d4c/20261106_d0")},
                        "verdict": "HOLDS" if ph1["fork"]["warm_holds_on_all_12"] else "STOP",
                        "prediction": "HELD" if ph1["fork"]["warm_holds_on_all_12"] else "FAILED"},
        "phase1_TWO_PASS": {"predicted": "two-pass brings D4c's 1.10x below 1 with every segment within tolerance on all 12",
                            "measured": {"L_passes": sum(ph1["fork"]["two_pass_L_passes"].values()),
                                         "fixtures": len(ph1["fork"]["two_pass_L_passes"]),
                                         "worst_segment_ratio": max(ph1["fork"]["two_pass_worst_segment_ratio"].values(), default=None),
                                         "20261106_d0_worst_segment_ratio": ph1["fork"]["two_pass_worst_segment_ratio"].get("d4c/20261106_d0")},
                            "verdict": "SELECTED" if rule_two_pass else "NOT SELECTED",
                            "prediction": "HELD" if rule_two_pass else "FAILED"},
    }
    if rule_two_pass:
        clauses.update({
            "validity": {"predicted": "exact_identity pooled <= 1.0 mm on every fixture",
                         "measured": {"max_mm": max((v["pooled_mm"]["exact_identity"] for v in rows.values()), default=None)},
                         "verdict": "PASS" if validity else ("INVALID" if population_ok else "FAIL"),
                         "prediction": "HELD" if validity else "FAILED"},
            "L": {"predicted": "PASS, the worst trunk on a strongly negative spine",
                  "measured": {"fixtures_passing": sum(l_fixture.values()), "fixtures": len(l_fixture),
                               "worst_trunk_ratio": worst_candidate, "worst_trunk_fixture": worst_row,
                               "worst_trunk_fixture_spine_draw": rows[worst_row]["spine_draw"] if worst_row else None,
                               "segments_failing": sorted({s_ for v in rows.values() for s_, r in v["segments"]["candidate"].items()
                                                           if r["scored"] and not r["within"]})},
                  "verdict": "PASS" if conjuncts["L"] else "FAIL",
                  "prediction": "HELD" if conjuncts["L"] and worst_row and rows[worst_row]["spine_tercile"] == "short" else "FAILED"},
            "closure": {"predicted": "<= 1e-4 m on every candidate cell, bound by content",
                        "measured": {"rows": closure_rows, "problems": closure_problems},
                        "verdict": "PASS" if closure_ok else "FAIL", "prediction": "HELD" if closure_ok else "FAILED"},
            "must_fail_i": {"predicted": "the mean body misses L on every fixture",
                            "measured": {"fixtures_missing": sum(mean_misses.values()), "fixtures": len(mean_misses)},
                            "verdict": "PASS" if conjuncts["must_fail_i_mean_body_misses_L"] else "FAIL",
                            "prediction": "HELD" if conjuncts["must_fail_i_mean_body_misses_L"] else "FAILED"},
            "must_fail_ii": {"predicted": "the spine displaced 0.149 misses L at the trunk everywhere",
                             "measured": {"fixtures_failing_at_the_trunk": sum(spine_fails.values()), "fixtures": len(spine_fails)},
                             "verdict": "PASS" if conjuncts["must_fail_ii_spine_displaced_fails_L_at_the_trunk"] else "FAIL",
                             "prediction": "HELD" if conjuncts["must_fail_ii_spine_displaced_fails_L_at_the_trunk"] else "FAILED"},
            "must_fail_iii": {"predicted": "the exact identity reads 0 and passes", "measured": {"fixtures": sum(exact_zero.values())},
                              "verdict": "PASS" if conjuncts["must_fail_iii_exact_identity_reads_zero_and_passes"] else "FAIL",
                              "prediction": "HELD" if conjuncts["must_fail_iii_exact_identity_reads_zero_and_passes"] else "FAILED"},
            "must_fail_iv": {"predicted": "init-only misses L (STOP if it passes on any)",
                             "measured": {"fixtures_missing": sum(init_misses.values()), "fixtures": len(init_misses)},
                             "verdict": "PASS" if conjuncts["must_fail_iv_init_only_misses_L"] else "STOP",
                             "prediction": "HELD" if conjuncts["must_fail_iv_init_only_misses_L"] else "FAILED"},
            "B1": {"predicted": "combined - D7c lower CI > 0 on both performers; combined - D4c within +-0.01 (reported)",
                   "measured": {k: b1.get(k) for k in ("pairs", "binding", "problems", "frozen_pose_below",
                                                       "mamma_bit_identical", "arms_pooled_median_iou")},
                   "verdict": "PASS" if b1["pass"] else "FAIL",
                   "prediction": "HELD" if b1["pass"] and all(abs((r or {}).get("median") or math.inf) <= 0.01 for r in
                                                              (b1.get("REPORTED_D4d_minus_D4c") or {}).values()) else "FAILED"},
            "B2": {"predicted": "(carried) the consumed array is the rig converter's input; the markers re-derive",
                   "measured": b2, "verdict": "PASS" if b2["pass"] else "FAIL", "prediction": "HELD" if b2["pass"] else "FAILED"},
        })
    clauses["hygiene"] = {"predicted": "--body rig 8/8; passes = 1 reproduces D4c's delivery and acceptance cells; the "
                                       "source diff is the second pass only",
                          "measured": {"rig_identical": hyg["rig_rebuild"]["identical_count"],
                                       "tripwire_unequal": hyg["tripwire"]["unequal"],
                                       "tripwire_byte_identical": hyg["tripwire"]["byte_identical"],
                                       "tripwire_equal_after_the_named_normalisation": hyg["tripwire"]["equal_after_the_named_normalisation"],
                                       "source_diff": hyg["source_diff"]},
                          "verdict": "PASS" if hyg["pass"] else "FAIL", "prediction": "HELD" if hyg["pass"] else "FAILED"}
    clauses["overall"] = {"predicted": "PASS", "measured": verdict, "verdict": verdict,
                          "prediction": "HELD" if verdict == "PASS" else "FAILED"}
    return {"step": "D4d", "verdict": verdict, "reason": reason, "consequence": consequence, "conjuncts": conjuncts,
            "phase1_selected": selected, "problems": problems + closure_problems + ph1["problems"],
            "population": {"phase1_fixtures": len(ph1["fixtures"]), "phase1_arms": list(fx4.PHASE1_ARMS),
                           "phase2_fixtures": [f"{s}_d{d}" for s, d in population] if rule_two_pass else None,
                           "phase2_arms": list(fx4.PHASE2_ARMS), "frames": FRAMES, "joints": len(MAPPED_JOINTS),
                           "phase2_cells_valid": len(cells)},
            "phase1_decision_sha256": decision_sha, "clauses": clauses, "reported": reported}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precondition-0", action="store_true")
    parser.add_argument("--hygiene", action="store_true", help="stage 2: the tripwire, the source diff and the rig "
                                                              "rebuild, re-derived from the files")
    parser.add_argument("--manifest", action="store_true", help="stage 4: name the Phase-2 cells by content")
    parser.add_argument("--phase1", action="store_true", help="stage 3: the Phase-1 record and the frozen fork's "
                                                              "decision, written as the decision JSON")
    arguments = parser.parse_args()
    if arguments.phase1:
        record = phase1(load_phase1())
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        for problem in record["problems"][:30]:
            print("  PROBLEM:", problem)
        print("stage 0b:", record["stage_0b"]["fixtures_failing"], "of", record["stage_0b"]["fixtures"])
        for p, arms in record["by_population"].items():
            for a, v in arms.items():
                print(f"  {p:4s} {a:16s} L {v['L_passes']}/{v['fixtures']}  trunk within {v['trunk_within']}  worst trunk "
                      f"{v['worst_trunk_ratio']:.3f}  worst segment {v['worst_scored_segment_ratio']:.3f}")
        print("fork: WARM holds on all 12:", record["fork"]["warm_holds_on_all_12"], "| TWO-PASS closes every segment "
              "on all 12:", record["fork"]["two_pass_closes_every_segment_on_all_12"])
        print("SELECTED:", record["selected"], "| Phase 2 licensed:", record["phase_2_licensed"])
        return 0
    if arguments.hygiene:
        record = dict(hygiene(), step="D4d", stage=2)
        record["tripwire"]["rows"] = {k: v for k, v in record["tripwire"]["rows"].items()}
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        t = record["tripwire"]
        print(f"tripwire: {t['byte_identical']} byte-identical, {t['equal_after_the_named_normalisation']} equal after "
              f"the named normalisation, unequal {t['unequal']}, sets {t['sets_by_identity']}")
        print("source diff:", {k: v for k, v in record["source_diff"].items() if k != "base"})
        print("rig rebuild:", record["rig_rebuild"]["identical_count"], "of 8 byte-identical")
        print("hygiene:", "HOLDS" if record["pass"] else "FAILS")
        return 0
    if arguments.precondition_0:
        record = dict(precondition_0(D4C_DRAWN_SET.read_bytes()), step="D4d", stage=1,
                      what="precondition 0: D4c's frozen limit-aware drawn set reused by sha256 and re-derived by "
                           "D4c's own rule (imported); STOP if the spine is not drawn")
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print("precondition 0:", "HOLDS" if record["holds"] else "STOP", "| drawn", record["drawn_set"],
              "| trunk scored", record["trunk_scored"], "| reused by sha256", record["reused_by_sha256"])
        return 0
    if arguments.manifest:
        cells = sorted((OUT_DIR / "phase2").glob("cell-*.json"))
        record = {"step": "D4d", "stage": 4,
                  "what": "the Phase-2 cells, named by content, under the committed Phase-1 decision; this file's commit "
                          "must be younger than the decision JSON's (the gate checks the git order)",
                  "phase1_decision_sha256": sha256_file(DECISION),
                  "cells_sha256": {c.name: sha256_file(c) for c in cells}, "cells": len(cells)}
        arguments.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print("manifest:", len(cells), "cells")
        return 0
    phase1_inputs = load_phase1()
    phase2_inputs = load_phase2() if (OUT_DIR / "phase2").is_dir() else None
    report = build(phase1_inputs, phase2_inputs)
    arguments.out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    for name, clause in report["clauses"].items():
        if name != "overall":
            print(f"  {name:16s} {clause['verdict']:12s} prediction {clause['prediction']}")
    for problem in report["problems"][:30]:
        print("  PROBLEM:", problem)
    print(f"VERDICT: {report['verdict']} ({report['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
