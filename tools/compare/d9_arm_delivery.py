#!/usr/bin/env python3
"""D9: rebuild the delivery through the REAL build script, into this step's own directory.

Modelled line for line on `tools/compare/d8_occlusion_delivery.py`, which was modelled on
`d7b_trunk_delivery.py` and that on `d7_world_vertical_delivery.py`.
`scripts/build_commercial_multiview_comparison.py` is imported as a module and its real
`main()` runs, so the detector feed, the association, the triangulation, the head solve, the
toe solve, the sizing, the exporter and the report are the delivered ones. The only thing
that differs between the two runs this file makes is the state of `src/`.

**NOTHING is written under `artifacts/commercial-multiview-soma77/`.** The delivered
build's `work/` -- its cached per-camera detections -- is COPIED (never symlinked) into the
output directory, so the detector feed is byte-identical and the build reuses it rather
than re-detecting, and the copy's hashes are asserted unchanged before and after.

WHY A NEW FILE AND NOT A PARAMETER. `d8_occlusion_delivery.py` carries D8's clauses and its
committed report is the record of that pass; D8 did not edit D7b's file for the same
reason. This file differs from it in exactly one substantive place, and that place is the
whole point of D9:

  THE SAME-DENOMINATOR VERDICT IS THE **SMOOTHED** ARRAY AGAIN, BESIDE THE RAW ONE.
  D8's conditioning gate, reachability reject and gap clause sit BETWEEN the raw
  triangulation and the smoothed array, so D8 had to fall back on the raw array as its
  shared denominator. D9 is a CONVERTER-ONLY change: it aims two arm bones from their own
  forward-kinematic origins and touches nothing that can move a landmark. So
  `triangulated_world_positions_z_up_m` must be byte-identical between the D8 build and the
  D9 build, exactly as it was up to D7b, and `raw_triangulated_world_positions_z_up_m` must
  be byte-identical too. BOTH are verdicts here, on BOTH arms. If either reads False the
  change did more than aiming and the step is wrong, whatever the placement figures say
  (`docs/LADDER_EXECUTION_PLAN.md` section 2, the D9 card: "Converter-only, so the smoothed
  landmarks are byte-identical between D8 and D9").

TWO ARMS, and the ORDER matters:

  * `--out artifacts/compare/d9-arms/delivery-hygiene --expect-byte-identical` run with
    `src/` UNCHANGED, BEFORE any D9 change lands. Every GLB, track and npz must come out
    byte-identical to `artifacts/commercial-multiview-soma77`. That is the hygiene
    demonstration: it proves the rebuild harness reproduces the shipped delivery exactly,
    so any later difference belongs to the src change and to nothing else.
  * `--out artifacts/compare/d9-arms/delivery` run AFTER the change. Its landmarks -- raw
    AND smoothed -- must still be byte-identical to the shipped ones; the four delivered
    files are expected to DIFFER, in the arm rotations and in nothing else, and B2 of
    `d9_arm_gate.py` is what proves the "nothing else".

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9_arm_delivery.py \
        --out artifacts/compare/d9-arms/delivery-hygiene --expect-byte-identical
"""

from __future__ import annotations

import argparse
import json
import shutil
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _relative in ("src", "tools/compare", "tools/head", "scripts"):
    sys.path.insert(0, str(ROOT / _relative))

import autoanim_gnm  # noqa: E402

if not str(Path(autoanim_gnm.__file__).resolve()).startswith(str(ROOT)):
    raise SystemExit(
        f"PYTHONPATH trap: autoanim_gnm resolved to {autoanim_gnm.__file__}, not this "
        f"worktree ({ROOT}). Re-run with PYTHONPATH=$PWD/src.")

DELIVERED = ROOT / "artifacts/commercial-multiview-soma77"
VIDEOS = ROOT / ".cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos"
CALIBRATION = ROOT / ".cache/mamma/configs/examples/calib/iphones_outdoors.yaml"

# The delivered files a byte-comparison covers. `run-report.json` and the HTML reviews are
# deliberately excluded: they carry paths and timings.
DELIVERED_FILES = tuple(
    f"subject-{s:02d}{suffix}"
    for s in (0, 1)
    for suffix in (".glb", ".body-track.json", ".body-track.npz", ".mapping.npz")
)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def observation_hashes(directory: Path) -> dict[str, str]:
    return {path.name: digest(path)
            for path in sorted((directory / "work").glob("*observations.jsonl"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-byte-identical", action="store_true",
                        help="assert every delivered file matches the shipped delivery; "
                             "the hygiene arm, run with src/ UNCHANGED")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.resolve() == DELIVERED.resolve():
        raise SystemExit("refusing to write into the shipped delivery")

    out.mkdir(parents=True, exist_ok=True)
    if not (out / "work").exists():
        print(f"copying work/ (never symlinked) into {out}")
        shutil.copytree(DELIVERED / "work", out / "work")
    before = observation_hashes(out)
    delivered_hashes = observation_hashes(DELIVERED)

    import build_commercial_multiview_comparison as build  # noqa: E402

    argv = sys.argv[:]
    sys.argv = [
        "build_commercial_multiview_comparison.py",
        "--videos", str(VIDEOS),
        "--calibration-yaml", str(CALIBRATION),
        "--detector", "soma77",
        "--output", str(out),
    ]
    try:
        code = build.main()
    finally:
        sys.argv = argv
    if code != 0:
        raise SystemExit(f"the build exited {code}")

    after = observation_hashes(out)
    files = {name: {"rebuild": digest(out / name), "shipped": digest(DELIVERED / name)}
             for name in DELIVERED_FILES}
    for row in files.values():
        row["identical"] = row["rebuild"] == row["shipped"]
    raw_identical = {}
    smoothed_identical = {}
    for subject in (0, 1):
        with np.load(out / f"subject-{subject:02d}.body-track.npz") as a, \
             np.load(DELIVERED / f"subject-{subject:02d}.body-track.npz") as b:
            raw_identical[f"subject_{subject:02d}"] = bool(np.array_equal(
                a["raw_triangulated_world_positions_z_up_m"],
                b["raw_triangulated_world_positions_z_up_m"],
                equal_nan=True))
            smoothed_identical[f"subject_{subject:02d}"] = bool(np.array_equal(
                a["triangulated_world_positions_z_up_m"],
                b["triangulated_world_positions_z_up_m"]))
    run = json.loads((out / "run-report.json").read_text())
    diagnostics = run.get("diagnostics", run)
    report = {
        "title": ("D9 rebuild -- the real build script, this step's own output directory, "
                  "the shipped delivery's own cached detections copied in"),
        "output": str(out.relative_to(ROOT)),
        "src_state": ("UNCHANGED (hygiene arm)" if args.expect_byte_identical
                      else "the arms aimed from their own origins"),
        "work_copied_never_symlinked": True,
        "work_source": "artifacts/commercial-multiview-soma77/work",
        "hygiene": {
            "observations_byte_identical_before_and_after_the_build": before == after,
            "observations_byte_identical_to_the_shipped_build": after == delivered_hashes,
            "raw_triangulation_byte_identical_same_denominator": raw_identical,
            "smoothed_triangulation_byte_identical_same_denominator": smoothed_identical,
            "observation_files": sorted(after),
            "both_landmark_arrays_are_verdicts_here": (
                "D9 is converter-only. It cannot move a landmark, so BOTH the raw and the "
                "smoothed arrays must be byte-identical to the shipped build on both arms. "
                "D8 could only assert the raw one, because its repair sits between them."),
            "delivered_files_vs_shipped": files,
            "all_delivered_files_identical": all(r["identical"] for r in files.values()),
            "delivered_files_note": (
                "EXPECTED True on the hygiene arm and False on the D9 arm -- the arm "
                "rotations move, which is the step. WHAT else moved is B2's question, and "
                "`d9_arm_gate.py` answers it joint by joint rather than from these hashes."),
        },
        "diagnostics": {key: diagnostics.get(key) for key in
                        ("pelvis_frame", "spine_triangulation", "head_orientation",
                         "toe_triangulation", "occlusion_repair",
                         "held_joint_fraction", "interpolated_joint_fraction")},
    }
    ok = (report["hygiene"]["observations_byte_identical_before_and_after_the_build"]
          and report["hygiene"]["observations_byte_identical_to_the_shipped_build"]
          and all(raw_identical.values())
          and all(smoothed_identical.values()))
    if args.expect_byte_identical:
        ok = ok and report["hygiene"]["all_delivered_files_identical"]
    report["verdict"] = "PASS" if ok else "FAIL"
    destination = out.parent / f"{out.name}-build.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "diagnostics"}, indent=1))
    print(f"\nwrote {destination}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
