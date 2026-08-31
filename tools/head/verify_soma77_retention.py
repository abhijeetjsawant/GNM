#!/usr/bin/env python3
"""Tripwire check: do the retained observations really carry all 77 SOMA joints,
in SOMA-77 order?

`docs/HEAD_FEET_HANDS_PLAN.md` §2 states that answering the HeadEnd question
"needs a **worker re-run**, not a re-read", because the observations were written
after the adapter dropped 60 joints. That is false for the *file*: the worker
retains `landmarks_soma77` alongside the 17-joint dict (see the module docstring
of `workers/commercial_multiview/soma77_pose.py`).

Before building on that, prove the array is in SOMA-77 order rather than merely
77 long. The 17 mapped joints have known indices, so every mapped joint must be
byte-identical to `landmarks_soma77[index]` on every person-frame. If the array
had been reordered or written from a different model, that check fails.

Blind to: whether the *values* are correct. This establishes indexing only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workers.commercial_multiview.soma77_pose import SOMA77_TO_AUTOANIM  # noqa: E402

WORK = Path("artifacts/soma77-full/work")
CAMERAS = ("A001", "B001", "C001", "D001")
SKEL = json.loads(Path("src/autoanim_gnm/data/somaskel77-v1.json").read_text())


def main() -> int:
    names = SKEL["joint_names"]
    total = mismatched = 0
    lengths: set[int] = set()
    frames: set[int] = set()
    for camera in CAMERAS:
        for line in (WORK / f"{camera}-observations.jsonl").read_text().splitlines():
            record = json.loads(line)
            frames.add(record["frame_index"])
            for person in record["people"]:
                marks = person["landmarks_soma77"]
                lengths.add(len(marks))
                for name, index in SOMA77_TO_AUTOANIM.items():
                    joint = person["joints"][name]
                    x, y, confidence = marks[index]
                    total += 1
                    if (joint["x"], joint["y"], joint["confidence"]) != (x, y, confidence):
                        mismatched += 1
    print(f"array lengths seen: {sorted(lengths)}")
    print(f"frame_index range: {min(frames)}..{max(frames)} ({len(frames)} distinct)")
    print(f"mapped-joint identity checks: {total}, mismatched: {mismatched}")
    for index in (4, 6, 7, 8, 9, 10):
        print(f"  index {index:2d} = {names[index]}")
    return 0 if mismatched == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
