#!/usr/bin/env python3
"""D9b: rebuild the delivery through the REAL build script, into this step's own directory.

Modelled line for line on `tools/compare/d8c_hip_delivery.py`, which was modelled on
`d8b_length_delivery.py`, that on `d8_occlusion_delivery.py`, that on `d7b_trunk_delivery.py`
and that on `d7_world_vertical_delivery.py`. `scripts/build_commercial_multiview_comparison.py`
is imported as a module and its real `main()` runs, so the detector feed, the association, the
triangulation, the head solve, the toe solve, the sizing, the exporter and the report are the
delivered ones. Between two runs of this file the only things that differ are the state of
`src/` and the `--mode` wrapper below.

**NOTHING is written under `artifacts/commercial-multiview-soma77/`.** The delivered build's
`work/` -- its cached per-camera detections -- is COPIED (never symlinked) into the output
directory, so the detector feed is byte-identical and the build reuses it rather than
re-detecting, and the copy's hashes are asserted unchanged before and after.

WHY A NEW FILE AND NOT A PARAMETER: as D8c said of D8b's, this file carries D9b's clauses and
its committed report is the record of this pass. It differs from D8c's in two places: the
shipped build it compares against is now the **D8c** delivery, and it carries the `--mode`
wrapper and the two WATCHERS this step needs.

THE WATCHERS ARE OBSERVERS, and they are installed on EVERY arm including the hygiene one,
because the quantity D9b's B2 compares -- pass B's accepted set on the SHIPPED build -- can
only be produced by a run of the shipped src. `cm.project_generated_foot_contacts` and
`cm._reachable_clavicle_sequence` are wrapped, never re-implemented: the real functions run
and their inputs and outputs are recorded (CLAUDE.md: wrap the pipeline to instrument it).

THE MODES, each a wrapper around the REAL projection and nothing else:

  * `shipped`      -- no wrapper. The delivery arm, and the hygiene arm when `src/` is
                      unchanged.
  * `zero-hoist`   -- the projection runs in full (contacts derived, the foot LOCAL lock
                      applied, `correction` computed) and then the returned root is replaced
                      by the root that went IN. This is the REFACTOR TRIPWIRE's arm: run it
                      under the old src and under the new src and the eight delivered files
                      must be byte-identical, which is the only way to see a re-solve that
                      reordered float operations (the D7b seven-normalisations lesson --
                      pass A and the re-solve share a helper, so an in-process "the re-solve
                      is a no-op" check would pass on a refactor that moved BOTH). It is
                      also the card's lock-without-correction DEGENERATE: it zeroes B1 by
                      undoing the plant, and must fail B2 (the root) and B4 (the plant).
  * `no-correction`-- `maximum_root_correction_m=0`. The card's other degenerate: every run
                      `continue`s before `contacts[...] = True`, so contacts go to (0, 0) and
                      the foot lock never lands. Passes B1 and O1; B4 exposes it.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d9b_hoist_delivery.py \
        --out artifacts/compare/d9b-hoist/delivery-hygiene --expect-byte-identical
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import shutil
import time
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

from autoanim_gnm import commercial_multiview as cm  # noqa: E402

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


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(values.max()), 4)}


def install_watchers(mode: str) -> dict:
    """Wrap the projection and the clavicle reject. The real functions do the work."""

    log: dict = {"projection_calls": [], "clavicle_calls": []}
    real_projection = cm.project_generated_foot_contacts
    real_reject = cm._reachable_clavicle_sequence

    def projection(track, **kwargs):
        call_kwargs = dict(kwargs)
        if mode == "no-correction":
            call_kwargs["maximum_root_correction_m"] = 0.0
        projected, diagnostics = real_projection(track, **call_kwargs)
        if mode == "zero-hoist":
            projected = replace(
                projected,
                root_translation_m=np.array(track.root_translation_m, copy=True))
        hoist = (np.asarray(projected.root_translation_m, dtype=np.float64)
                 - np.asarray(track.root_translation_m, dtype=np.float64))
        magnitude = 1e3 * np.linalg.norm(hoist, axis=1)
        log["projection_calls"].append({
            "mode": mode,
            "kwargs": {k: float(v) for k, v in call_kwargs.items()},
            "frames": int(len(magnitude)),
            "contacts": [int(v) for v in np.asarray(projected.foot_contacts).sum(0)],
            "contact_runs": _runs_of(np.asarray(projected.foot_contacts)),
            "diagnostics": diagnostics.as_dict(),
            "hoist_mm": _summary(magnitude),
            "frames_over_0_5mm": int((magnitude > 0.5).sum()),
            "frames_under_0_001mm": int((magnitude < 1e-3).sum()),
            "hoisted_frames": [int(i) for i in np.flatnonzero(magnitude > 0.5)],
        })
        return projected, diagnostics

    def reject(local_rotations, parent_world_rotations, ceiling_deg_per_frame):
        replaced, accepted = real_reject(
            local_rotations, parent_world_rotations, ceiling_deg_per_frame)
        log["clavicle_calls"].append({
            "call_index": len(log["clavicle_calls"]),
            "frames": int(len(accepted)),
            "rejected_count": int((~accepted).sum()),
            "rejected_frames": [int(i) for i in np.flatnonzero(~accepted)],
            "accepted_sha256": sha256(
                np.ascontiguousarray(accepted).tobytes()).hexdigest(),
        })
        return replaced, accepted

    cm.project_generated_foot_contacts = projection
    cm._reachable_clavicle_sequence = reject
    log["restore"] = (real_projection, real_reject)
    return log


def _runs_of(contacts: np.ndarray) -> list[list[list[int]]]:
    runs: list[list[list[int]]] = []
    for side in range(contacts.shape[1]):
        on = np.flatnonzero(contacts[:, side])
        if not on.size:
            runs.append([])
            continue
        splits = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
        runs.append([[int(r[0]), int(r[-1])] for r in splits])
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", default="shipped",
                        choices=("shipped", "zero-hoist", "no-correction"))
    parser.add_argument("--src-state", default="",
                        help="what src/ carried on this arm, recorded verbatim")
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

    log = install_watchers(args.mode)
    argv = sys.argv[:]
    sys.argv = [
        "build_commercial_multiview_comparison.py",
        "--videos", str(VIDEOS),
        "--calibration-yaml", str(CALIBRATION),
        "--detector", "soma77",
        "--output", str(out),
    ]
    started = time.time()
    try:
        code = build.main()
    finally:
        sys.argv = argv
        cm.project_generated_foot_contacts, cm._reachable_clavicle_sequence = log.pop(
            "restore")
    elapsed = time.time() - started
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
        "title": ("D9b rebuild -- the real build script, this step's own output directory, "
                  "the shipped delivery's own cached detections copied in"),
        "output": str(out.relative_to(ROOT)),
        "mode": args.mode,
        "src_state": args.src_state or ("UNCHANGED (hygiene arm)"
                                        if args.expect_byte_identical else "unrecorded"),
        "build_seconds": round(elapsed, 1),
        "work_copied_never_symlinked": True,
        "work_source": "artifacts/commercial-multiview-soma77/work",
        "hygiene": {
            "observations_byte_identical_before_and_after_the_build": before == after,
            "observations_byte_identical_to_the_shipped_build": after == delivered_hashes,
            "observation_files": sorted(after),
            "raw_triangulation_byte_identical_same_denominator": raw_identical,
            "smoothed_triangulation_byte_identical": smoothed_identical,
            "landmark_note": (
                "D9b is a CONVERTER-ONLY change on byte-identical landmarks: BOTH the raw "
                "and the smoothed arrays must be byte-identical on every arm, hygiene and "
                "candidate alike. If either moves, the change did more than re-aim and the "
                "same-denominator clause has not returned to PASS."),
            "delivered_files_vs_shipped": files,
            "all_delivered_files_identical": all(r["identical"] for r in files.values()),
        },
        "watchers": {
            "note": ("the REAL `project_generated_foot_contacts` and "
                     "`_reachable_clavicle_sequence` ran; these are recordings of their "
                     "inputs and outputs, never a re-implementation"),
            "projection_calls": log["projection_calls"],
            "clavicle_calls": log["clavicle_calls"],
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
