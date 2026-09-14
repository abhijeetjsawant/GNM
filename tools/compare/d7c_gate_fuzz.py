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

HOW A LEAF IS CLASSIFIED, and the first version of this got it wrong. "Did any clause's text
move?" is not a classification: it put 8 leaves that move ENFORCED P1 control clauses into the
REPORT-only bucket, and mixed the excluded diagnostics clause and the preserved sigma-1 FAIL in
with genuine report rows. A leaf is now classified by WHICH clauses it moves -- their status
(PASS / FAIL / REPORT) and whether they belong to a merge conjunct:

  ENFORCED                 some mutation turns the gate to NO MERGE
  moves a conjunct clause  it moves a clause inside the merge rule but no mutation turned the
                           verdict -- this is a GAP and is reported as one, not as a pass
  REPORT-only              every clause it moves has status REPORT
  diagnostics              it moves only the clause deliberately excluded from the predicate
  historical FAIL          it moves only a preserved recorded STOP, which is meant to stay FAIL
  read by no clause        no clause's text moves at all

AND FOR ENFORCED NUMERIC LEAVES, A MONOTONE CHECK -- and its first version was mis-specified,
which is worth recording because it produced 387 "failures" that were the check's fault. It
required BOTH +1e6 and -1e6 to fail if either did, and most bands here are ONE-SIDED: an error
that must be small correctly PASSES when driven to -1e6, because that is the good direction.
What monotonicity actually means is: whichever extreme fails, pushing SIX MORE ORDERS OF
MAGNITUDE the same way must fail too. A leaf that fails at +1e6 and passes at +1e12 is not
being banded; it is being matched.

WHAT A PASS DOES NOT PROVE: that the gate reads the RIGHT things. Only that what it reads, it
depends on. Choosing the clauses is still the card's job.

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

from d7c_gate_report import (  # noqa: E402
    BASE, CONJUNCTS, OUTSIDE, RECORDED_STOPS, UNREAD_MEASUREMENTS_JUSTIFIED, build,
    classify_leaf, coverage_audit, justification_for, kind_of, load_all, normalise)

BIG = 1.0e6
FAR = 1.0e12
MISMATCH = "0" * 64
SENTINEL = object()


def walk(node, path=()):
    """Every leaf and every container, as (path, kind). The kinds are the gate's own."""
    yield path, kind_of(node)
    if isinstance(node, dict):
        for key, value in list(node.items()):
            yield from walk(value, path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, path + (index,))


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


BUCKETS = ("enforced", "moves_a_conjunct_clause_without_turning_the_verdict",
           "report_only", "diagnostics_clause_only", "historical_FAIL_clause_only",
           "read_by_no_clause")


def merge(parts, out_path) -> int:
    """Join the chunk reports into the one report `gate.json` embeds. Nothing is recomputed:
    the rows are concatenated and the counts summed, and the chunks must tile the target list
    exactly -- a missing or overlapping chunk is an error, not a smaller fuzz."""
    chunks = sorted((json.loads(path.read_text()) for path in parts),
                    key=lambda c: c["visited"]["start"])
    at, joined = 0, None
    for chunk in chunks:
        if chunk["visited"]["start"] != at:
            raise SystemExit(f"chunk starts at {chunk['visited']['start']}, not {at}: the "
                             "chunks do not tile the target list")
        at += chunk["visited"]["total"]
        if joined is None:
            joined = {k: v for k, v in chunk.items() if k not in BUCKETS}
            joined.update({name: list(chunk[name]) for name in BUCKETS})
            continue
        for name in BUCKETS:
            joined[name].extend(chunk[name])
        for key in ("leaves", "containers", "total"):
            joined["visited"][key] += chunk["visited"][key]
        for label, count in chunk["unread_by_class"].items():
            joined["unread_by_class"][label] = (
                joined["unread_by_class"].get(label, 0) + count)
    if at != joined["visited"]["of"]:
        raise SystemExit(f"the chunks cover {at} of {joined['visited']['of']} targets")
    gaps = joined["moves_a_conjunct_clause_without_turning_the_verdict"]
    failures = [row for row in joined["enforced"] if "monotone_check" in row]
    joined["counts"] = {name: len(joined[name]) for name in BUCKETS}
    joined["gaps"] = len(gaps)
    joined["monotone_check_failures"] = len(failures)
    joined["monotone_check_failing_leaves"] = failures[:20]
    joined["summary"] = (
        f"{joined['visited']['total']} visited ({joined['visited']['leaves']} leaves + "
        f"{joined['visited']['containers']} containers): {len(joined['enforced'])} enforced, "
        f"{len(gaps)} gaps, {len(joined['report_only'])} REPORT-only, "
        f"{len(joined['diagnostics_clause_only'])} diagnostics, "
        f"{len(joined['historical_FAIL_clause_only'])} historical-FAIL, "
        f"{len(joined['read_by_no_clause'])} read by no clause")
    joined["merged_from"] = [str(path) for path in parts]
    out_path.write_text(json.dumps(joined, indent=1))
    print(joined["summary"])
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "gate-fuzz.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many leaves (0 = every leaf)")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many targets first, so the run can be SPLIT INTO "
                             "CHUNKS. The whole fuzz is ~64,000 gate builds at 9 ms each, "
                             "which is at the shell's own timeout; backgrounding it is worse "
                             "(the harness restarts a backgrounded run), so it is split and "
                             "merged instead.")
    parser.add_argument("--merge", type=Path, nargs="*", default=None,
                        help="merge chunk reports written by --start/--limit into one")
    args = parser.parse_args()
    if args.merge:
        return merge(args.merge, args.out)

    reports = load_all()
    clean = build(reports)
    if clean["verdict"] != "MERGE":
        print(f"the gate does not read MERGE on the unmutated reports ({clean['verdict']}); "
              "the fuzz would prove nothing")
    baseline = {c["clause"]: (c["verdict"], str(c["measured"])) for c in clean["clauses"]}

    status_of = {c["clause"]: c["verdict"] for c in clean["clauses"]}
    in_conjunct = set()
    for _name, prefixes in CONJUNCTS:
        for prefix in prefixes:
            in_conjunct.update(c for c in status_of if c.startswith(prefix))
    excluded = set(OUTSIDE)
    # THE PRESERVED STOPS BY NAME, not "whichever clauses fail today". Deriving this class
    # from the baseline verdicts would let a new clause that accidentally FAILS on the
    # unmutated reports absorb every leaf it reads into a class excused by construction.
    historical = set(RECORDED_STOPS)
    failing_now = {c for c in status_of if status_of[c] == "FAIL"}
    if failing_now != historical:
        print("the baseline FAIL set is not the two recorded STOPs: "
              f"{sorted(failing_now ^ historical)}")
    read_reports = {p.split("/")[0] for p in clean["touched"]}

    targets = [(path, kind) for path, kind in walk(reports) if path]
    total_targets = len(targets)
    targets = targets[args.start:]
    if args.limit:
        targets = targets[:args.limit]
    buckets = {"enforced": [], "moves_a_conjunct_clause_without_turning_the_verdict": [],
               "report_only": [], "diagnostics_clause_only": [],
               "historical_FAIL_clause_only": [], "read_by_no_clause": []}
    leaves = containers = 0
    unread_classes: dict[str, int] = {}
    for path, kind in targets:
        parent, key = fetch(reports, path)
        original = parent[key]
        turned, verdicts, moved, directions = False, set(), set(), {}
        for label, replacement in mutations_for(kind, original):
            # THE CONTAINER IS RESTORED WHOLESALE, because restoring by index under an
            # identity test is unsound in general: after a deletion has shifted a list,
            # `parent[key] is original` can be true of a DIFFERENT element -- Python caches
            # small ints and both booleans, so a list of counts or flags can satisfy it --
            # and the restore is then skipped and the report left corrupt. An earlier version
            # of this fuzzer did that and died with `KeyError: 'left_knee'`; THE EXACT TRIGGER
            # WAS NOT ISOLATED, and an earlier note here blamed interned equal floats, which
            # does not reproduce (separately decoded equal floats are distinct objects).
            # Restoring the whole container removes the class without needing the diagnosis.
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
            failed = result["verdict"] != "MERGE"
            if failed:
                turned = True
            if label in ("set to 1e6", "set to -1e6"):
                directions[label] = failed
            for entry in result.get("clauses", []):
                before = baseline.get(entry["clause"])
                if before and (before[0] != entry["verdict"]
                               or before[1] != str(entry["measured"])):
                    moved.add(entry["clause"])
        # THE MONOTONE CHECK: whichever extreme failed, six more orders the same way must
        # fail too. Only the FAILING direction is tested -- a one-sided band is supposed to
        # pass when pushed the good way.
        monotone = {}
        for label, further in (("set to 1e6", FAR), ("set to -1e6", -FAR)):
            if not directions.get(label):
                continue
            snapshot = list(parent) if isinstance(parent, list) else dict(parent)
            try:
                parent[key] = further
                try:
                    beyond = build(reports)["verdict"] != "MERGE"
                except Exception:
                    beyond = True
            finally:
                if isinstance(parent, list):
                    parent[:] = snapshot
                else:
                    parent.clear()
                    parent.update(snapshot)
            monotone[label] = beyond
        row = {"path": "/".join(map(str, path)), "kind": kind,
               "verdicts": sorted(verdicts), "clauses_moved": sorted(moved)[:4]}
        if kind in ("list", "map"):
            containers += 1
        else:
            leaves += 1
        if turned:
            if monotone and not all(monotone.values()):
                row["monotone_check"] = monotone
                row["note"] = ("fails at 1e6 and PASSES at 1e12 in the SAME direction -- the "
                               "clause is matching this leaf, not banding it")
            buckets["enforced"].append(row)
        elif moved & in_conjunct:
            row["justification"] = ("GAP: it moves a clause inside the merge rule and no "
                                    "mutation turned the verdict")
            buckets["moves_a_conjunct_clause_without_turning_the_verdict"].append(row)
        elif moved and moved <= excluded:
            row["justification"] = ("moves only the diagnostics clause, which the card does "
                                    "not band and which is listed as deliberately outside "
                                    "the predicate")
            buckets["diagnostics_clause_only"].append(row)
        elif moved and moved <= historical:
            row["justification"] = ("moves only a preserved recorded STOP, which is MEANT to "
                                    "stay FAIL and is outside the merge rule")
            buckets["historical_FAIL_clause_only"].append(row)
        elif moved and all(status_of.get(c) == "REPORT" for c in moved):
            row["justification"] = "every clause it moves has status REPORT"
            buckets["report_only"].append(row)
        elif moved:
            row["justification"] = (f"moves {sorted(moved)[:3]}, none of them a conjunct "
                                    "clause, a REPORT clause, the diagnostics clause or a "
                                    "preserved STOP")
            buckets["moves_a_conjunct_clause_without_turning_the_verdict"].append(row)
        else:
            # THE UNREAD CLASS, INVERTED (Astra round 7). "Read by no clause" was a single
            # bucket of 15,618 rows and a reviewer found four measurements hiding in it in
            # one round, which is the definition of a class that proves nothing. Every
            # unread leaf is now classified by WHAT IT IS, and every MEASUREMENT leaf under
            # a report some clause reads is a GAP unless the gate's own inventory names its
            # family.
            label = classify_leaf(path, kind)
            row["class"] = label
            row["subtree"] = str(path[0])
            if label == "MEASUREMENT" and str(path[0]) in read_reports:
                pattern, reason = justification_for(normalise(path),
                                                    UNREAD_MEASUREMENTS_JUSTIFIED)
                if pattern is None:
                    row["justification"] = ("GAP: an unread MEASUREMENT leaf under a report "
                                            "a clause reads, with no justification named in "
                                            "the gate's inventory")
                    buckets["moves_a_conjunct_clause_without_turning_the_verdict"].append(row)
                    unread_classes["MEASUREMENT justified"] = unread_classes.get(
                        "MEASUREMENT justified", 0)
                    continue
                row["justification"] = f"{pattern}: {reason}"
                label = "MEASUREMENT justified"
            else:
                row["justification"] = {
                    "LABEL": "a name, a rule or a note; not a measurement",
                    "PROVENANCE": "identifies an input or an output; not a measurement",
                    "DIAGNOSTIC": "inside a subtree the card puts outside the predicate",
                    "MEASUREMENT": "in a report no clause reads at all",
                    "CONTAINER": "a container whose leaves are classified individually",
                }[label]
            unread_classes[label] = unread_classes.get(label, 0) + 1
            buckets["read_by_no_clause"].append(row)

    gaps = buckets["moves_a_conjunct_clause_without_turning_the_verdict"]
    direction_failures = [row for row in buckets["enforced"] if "monotone_check" in row]
    out = {
        "title": "D7c: the gate proved leaf by leaf, not asserted",
        "method": (
            "every leaf of every report the gate reads is mutated in turn -- numbers to 1e6, "
            "to -1e6, to 0 and deleted; strings mismatched and deleted; booleans flipped and "
            "deleted; lists and maps emptied, shortened and duplicated -- and the whole gate "
            "is rebuilt from the mutated reports."),
        "enforced_means": (
            "at least one mutation of that leaf turns the gate's verdict away from MERGE"),
        "classification": (
            "by WHICH clauses a leaf moves -- their status and whether they belong to a merge "
            "conjunct -- not by whether anything moved at all. An earlier version used the "
            "latter and put 8 leaves that move ENFORCED P1 control clauses into the "
            "REPORT-only bucket."),
        "what_a_pass_does_not_prove": (
            "that the gate reads the RIGHT things. Only that what it reads, it depends on. "
            "Choosing the clauses remains the card's job."),
        "the_unread_class_inverted": (
            "every leaf no clause reads is classified LABEL / PROVENANCE / DIAGNOSTIC / "
            "MEASUREMENT and tagged with the report it lives in. A MEASUREMENT leaf under a "
            "report some clause reads is a GAP unless the gate's own inventory names its "
            "family -- which is the check that ends the class Astra found four members of in "
            "each of rounds 6 and 7."),
        "unread_by_class": unread_classes,
        "coverage_audit": coverage_audit(load_all(), clean),
        "visited": {"total": len(targets), "leaves": leaves, "containers": containers,
                    "of": total_targets, "start": args.start},
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "gaps": len(gaps),
        "monotone_check": (
            "for every enforced numeric leaf, whichever extreme fails is pushed six more "
            "orders of magnitude the SAME way and must fail again. Only the failing direction "
            "is tested: a one-sided band is supposed to pass when pushed the good way."),
        "monotone_check_failures": len(direction_failures),
        "monotone_check_failing_leaves": direction_failures[:20],
        "summary": (f"{len(targets)} visited ({leaves} leaves + {containers} containers): "
                    f"{len(buckets['enforced'])} enforced, {len(gaps)} gaps, "
                    f"{len(buckets['report_only'])} REPORT-only, "
                    f"{len(buckets['diagnostics_clause_only'])} diagnostics, "
                    f"{len(buckets['historical_FAIL_clause_only'])} historical-FAIL, "
                    f"{len(buckets['read_by_no_clause'])} read by no clause"),
        **{name: rows for name, rows in buckets.items()},
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(out["summary"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
