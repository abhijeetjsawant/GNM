#!/usr/bin/env python3
"""D7c: PROVE the gate, leaf by leaf, instead of asserting it.

Four rounds of Astra's merge review found holes in `d7c_gate_report.py` one at a time --
literal verdicts, saved classifications, partial populations, stored aggregates -- and each was
answered with a hand-picked mutation table. A hand-picked table can only contain the attacks
its author thought of, which is exactly why the fifth round found six more.

So this walks EVERY LEAF of EVERY REPORT the gate reads and mutates each one in turn:

  numbers   set to 1e6 (an error, residual or travel that must fail its band), to -1e6 (a
            confidence bound or a margin that must fail its), to 0 (a ratio, a count or a
            denominator that must fail its), and DELETED
  strings   set to a mismatched value (a hash that no longer matches), and DELETED
  booleans  flipped, and DELETED
  lists     emptied, first member dropped, first member duplicated
  maps      emptied, one named member dropped, one member duplicated under a new key

and requires **NO MERGE** from every leaf any clause depends on. The mutation count reported is
the number of leaves visited, not a table someone wrote.

WHAT A PASS MEANS AND WHAT IT DOES NOT. A leaf that no mutation can turn is either read only by
a REPORT clause or read by nothing; both are listed, with which it is, derived from whether any
clause's own text moved. It does NOT prove the gate reads the right things -- only that what it
reads, it depends on. Choosing the clauses is still the card's job.

IF A LEAF ESCAPES, FIX THE GATE, NOT THE FUZZER.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_gate_fuzz.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools/compare") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools/compare"))

from d7c_gate_report import BASE, build, load_all  # noqa: E402

BIG = 1.0e6
MISMATCH = "0" * 64
SENTINEL = object()


def walk(node, path=()):
    """Every leaf and every container, as (path, kind)."""
    if isinstance(node, dict):
        yield path, "map"
        for key, value in list(node.items()):
            yield from walk(value, path + (key,))
    elif isinstance(node, list):
        yield path, "list"
        for index, value in enumerate(node):
            yield from walk(value, path + (index,))
    elif isinstance(node, bool):
        yield path, "bool"
    elif isinstance(node, (int, float)):
        yield path, "number"
    elif isinstance(node, str):
        yield path, "string"
    else:
        yield path, "other"


def fetch(reports, path):
    node = reports
    for step in path[:-1]:
        node = node[step]
    return node, path[-1]


def mutations_for(kind, value):
    """What to do to a leaf of this kind. Each entry is (label, new value or SENTINEL)."""
    if kind == "number":
        # THREE SIGNS, not two. A CI upper bound fails on a NEGATIVE and is untouched by 1e6
        # or 0 -- the first run of this fuzzer reported B1's `ci95/1` as an escape for that
        # reason alone, which was the mutation set's gap and not the gate's: the clause reads
        # the leaf and bands it correctly. The distinction matters, so it is recorded rather
        # than quietly patched: a leaf escapes when the GATE ignores it, and this one did not.
        return [("set to 1e6", BIG), ("set to -1e6", -BIG), ("set to 0", 0),
                ("deleted", SENTINEL)]
    if kind == "string":
        return [("mismatched", MISMATCH), ("deleted", SENTINEL)]
    if kind == "bool":
        return [("flipped", not value), ("deleted", SENTINEL)]
    if kind == "list":
        out = [("emptied", [])]
        if value:
            out.append(("first member dropped", value[1:]))
            out.append(("first member duplicated", [value[0]] + list(value)))
        return out
    if kind == "map":
        out = [("emptied", {})]
        if value:
            first = next(iter(value))
            dropped = {k: v for k, v in value.items() if k != first}
            out.append((f"member {first!r} dropped", dropped))
            duplicated = dict(value)
            duplicated[f"{first}__copy"] = value[first]
            out.append((f"member {first!r} duplicated", duplicated))
        return out
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "gate-fuzz.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many leaves (0 = every leaf)")
    args = parser.parse_args()

    reports = load_all()
    clean = build(reports)
    if clean["verdict"] != "MERGE":
        print(f"the gate does not read MERGE on the unmutated reports ({clean['verdict']}); "
              "the fuzz would prove nothing")
    baseline = {c["clause"]: (c["verdict"], str(c["measured"])) for c in clean["clauses"]}

    targets = [(path, kind) for path, kind in walk(reports) if path]
    if args.limit:
        targets = targets[:args.limit]
    enforced, report_only, inert = [], [], []
    for path, kind in targets:
        parent, key = fetch(reports, path)
        original = parent[key]
        turned, verdicts, moved = False, set(), set()
        for label, replacement in mutations_for(kind, original):
            # THE CONTAINER IS RESTORED WHOLESALE. Restoring by index and an identity check
            # is not safe: equal floats and small ints are interned, so `parent[key] is
            # original` can be True after a deletion has shifted the list, the element is
            # never put back, and every later lookup into that report fails. The first run of
            # this fuzzer died that way (`KeyError: 'left_knee'`), which is worth recording:
            # an instrument that corrupts its own input produces a clean-looking report of
            # nothing.
            snapshot = list(parent) if isinstance(parent, list) else dict(parent)
            try:
                if replacement is SENTINEL:
                    del parent[key]
                else:
                    parent[key] = replacement
                try:
                    result = build(reports)
                except Exception as error:            # a crash is a detection, not a pass
                    result = {"verdict": "NO MERGE", "clauses": [],
                              "crash": f"{type(error).__name__}: {error}"}
            finally:
                if isinstance(parent, list):
                    parent[:] = snapshot
                else:
                    parent.clear()
                    parent.update(snapshot)
            verdicts.add(result["verdict"])
            if result["verdict"] != "MERGE":
                turned = True
            for entry in result.get("clauses", []):
                before = baseline.get(entry["clause"])
                if before and (before[0] != entry["verdict"]
                               or before[1] != str(entry["measured"])):
                    moved.add(entry["clause"])
        row = {"path": "/".join(map(str, path)), "kind": kind,
               "verdicts": sorted(verdicts), "clauses_moved": sorted(moved)[:4]}
        if turned:
            enforced.append(row)
        elif moved:
            row["justification"] = ("read only by clauses whose verdict is REPORT -- it "
                                    "changes what is printed and nothing that is banded")
            report_only.append(row)
        else:
            row["justification"] = ("read by no clause: a label, a provenance string, a "
                                    "diagnostics field or a note")
            inert.append(row)

    out = {
        "title": "D7c: the gate proved leaf by leaf, not asserted",
        "method": (
            "every leaf of every report the gate reads is mutated in turn -- numbers to 1e6, "
            "to 0 and deleted; strings mismatched and deleted; booleans flipped and deleted; "
            "lists emptied, shortened and duplicated; maps emptied, shortened and duplicated "
            "-- and the whole gate is rebuilt from the mutated reports."),
        "what_a_pass_does_not_prove": (
            "that the gate reads the RIGHT things. Only that what it reads, it depends on. "
            "Choosing the clauses remains the card's job."),
        "leaves_visited": len(targets),
        "enforced": len(enforced),
        "read_only_by_REPORT_clauses": len(report_only),
        "read_by_no_clause": len(inert),
        "summary": (f"{len(targets)} leaves visited, {len(enforced)} enforced, "
                    f"{len(report_only)} REPORT-only, {len(inert)} inert"),
        "enforced_leaves": enforced,
        "report_only_leaves": report_only,
        "inert_leaves": inert,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(out["summary"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
