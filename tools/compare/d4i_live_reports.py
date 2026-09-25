"""D4i: hash the SHARED live reports every roster instrument could overwrite, and check them later.

`artifacts/` is shared between the main checkout and every worktree, and the legacy instruments write
their reports to fixed paths the ladder reads (`artifacts/compare/*.json`, `d3-skeleton/gate.json`,
`d8-occlusion/limb-stability.json`, `feet-lane/`, `head-lane/`), plus the shipped delivery and the D7c
silhouette baseline. D4i's Phase 1 runs those instruments on D4d's MHR delivery from a shadow root, so
none of them may move. This records the set before and proves it after.

    .venv/bin/python tools/compare/d4i_live_reports.py record OUT.json
    .venv/bin/python tools/compare/d4i_live_reports.py check  BEFORE.json [--out AFTER.json]

`check` exits 0 and prints `LIVE REPORTS UNCHANGED` only when every path hashes as recorded and no path was
added or removed; its verdict field is `verdict`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
# Directories hashed recursively (the shipped delivery INCLUDING its work/ detections, the D7c baseline,
# and every per-lane report directory a legacy instrument writes into).
TREES = ("commercial-multiview-soma77", "compare/i6", "compare/d3-skeleton", "compare/d8-occlusion",
         "compare/oracle-2d-cache", "feet-lane", "head-lane")
# Top-level report files under artifacts/compare (the ladder's inputs), hashed non-recursively.
FLAT = ("compare",)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> dict[str, str]:
    out: dict[str, str] = {}
    for tree in TREES:
        base = ART / tree
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    out[str(path.relative_to(ART))] = _sha(path)
    for flat in FLAT:
        for path in sorted((ART / flat).iterdir()):
            if path.is_file():
                out[str(path.relative_to(ART))] = _sha(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("out", type=Path)
    check = sub.add_parser("check")
    check.add_argument("before", type=Path)
    check.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    now = snapshot()
    if arguments.command == "record":
        arguments.out.write_text(json.dumps({"root": str(ART.resolve()), "trees": TREES, "flat": FLAT,
                                             "files": now}, indent=1), encoding="utf-8")
        print(f"recorded {len(now)} live files -> {arguments.out}")
        return 0
    before = json.loads(arguments.before.read_text(encoding="utf-8"))["files"]
    changed = sorted(k for k in before if k in now and now[k] != before[k])
    removed = sorted(k for k in before if k not in now)
    added = sorted(k for k in now if k not in before)
    verdict = "UNCHANGED" if not (changed or removed or added) else "CHANGED"
    report = {"before": str(arguments.before), "files_checked": len(before), "changed": changed,
              "removed": removed, "added": added, "verdict": verdict}
    if arguments.out:
        arguments.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"LIVE REPORTS {verdict}: {len(before)} checked, {len(changed)} changed, "
          f"{len(removed)} removed, {len(added)} added")
    for key in changed + removed + added:
        print("  ", key)
    return 0 if verdict == "UNCHANGED" else 1


if __name__ == "__main__":
    sys.exit(main())
