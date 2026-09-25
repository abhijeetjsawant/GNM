#!/usr/bin/env python3
"""D4d a second calibration pass on the landmark start: ONE verdict, derived from the inputs.

Stage 1 of the build writes precondition 0 only (D4c's frozen drawn set, reused by sha256, re-derived by D4c's own
rule, imported). The later stages extend this file.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d4d_twopass_gate.py --precondition-0 \
        --out docs/reviews/body-model-twopass-records/precondition-0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    arguments = parser.parse_args()
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
