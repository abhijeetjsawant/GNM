#!/usr/bin/env python3
"""D4d a second calibration pass on the landmark start: ONE verdict, derived from the inputs.

Stage 1 of the build writes precondition 0 only (D4c's frozen drawn set, reused by sha256, re-derived by D4c's own
rule, imported). The later stages extend this file.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --precondition-0 \
        --out docs/reviews/body-model-twopass-records/precondition-0.json
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


def rig_rebuild(out_dir: Path = OUT_DIR) -> dict:
    rows = {f: {"rebuild": sha256_file(out_dir / "hygiene" / f), "shipped": sha256_file(SHIPPED / f)} for f in EIGHT}
    for r in rows.values():
        r["identical"] = r["rebuild"] is not None and r["rebuild"] == r["shipped"]
    return {"rows": rows, "identical_count": sum(r["identical"] for r in rows.values()),
            "holds": all(r["identical"] for r in rows.values())}


def hygiene(out_dir: Path = OUT_DIR, fitter_source: str | None = None, base_source: str | None = None,
            files: dict | None = None) -> dict:
    fitter_source = FITTER.read_text(encoding="utf-8") if fitter_source is None else fitter_source
    base_source = git_bytes(f"{BASE_COMMIT}:tools/fitter/mhr_delivery.py").decode() if base_source is None else base_source
    trip = tripwire(tripwire_files(out_dir) if files is None else files)
    diff = source_diff(base_source, fitter_source)
    rig = rig_rebuild(out_dir)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precondition-0", action="store_true")
    parser.add_argument("--hygiene", action="store_true", help="stage 2: the tripwire, the source diff and the rig "
                                                              "rebuild, re-derived from the files")
    arguments = parser.parse_args()
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
    raise SystemExit("only --precondition-0 exists at stage 1")


if __name__ == "__main__":
    raise SystemExit(main())
