#!/usr/bin/env python3
"""D7c: rebuild the delivery through the REAL build script, into this step's own directory.

Modelled line for line on `tools/compare/d9b_hoist_delivery.py`, which was modelled on
`d8c_hip_delivery.py`, that on `d8b_length_delivery.py`, that on `d8_occlusion_delivery.py`,
that on `d7b_trunk_delivery.py` and that on `d7_world_vertical_delivery.py`.
`scripts/build_commercial_multiview_comparison.py` is imported as a module and its real
`main()` runs, so the detector feed, the association, the triangulation, the head solve, the
toe solve, the sizing, the exporter and the report are the delivered ones. Between two runs of
this file the only things that differ are the state of `src/` and the `--mode` wrapper below.

**NOTHING is written under `artifacts/commercial-multiview-soma77/`.** The delivered build's
`work/` -- its cached per-camera detections -- is COPIED (never symlinked) into the output
directory, so the detector feed is byte-identical and the build reuses it rather than
re-detecting, and the copy's hashes are asserted unchanged before and after.

WHY A NEW FILE AND NOT A PARAMETER: as D9b said of D8c's, this file carries D7c's clauses and
its committed report is the record of this pass. It differs from D9b's in three places: the
shipped build it compares against is the **D9b** delivery; the projection watcher SAVES THE
RETURNED TRACK (D9b's recorded summaries of it, which cannot answer P1); and the two modes are
D7c's projection controls rather than D9b's hoist degenerates.

THE WATCHERS ARE OBSERVERS, installed on EVERY arm including the hygiene one.
`cm.project_generated_foot_contacts` and `cm.positions_to_body_track` are wrapped, never
re-implemented: the real functions run and their inputs and returns are recorded (CLAUDE.md:
wrap the pipeline to instrument it).

  * the projection watcher writes its RETURN to `projection-snapshot-NN.npz` -- the
    post-projection track, immediately after the SINGLE projection -- and P1 authenticates
    the delivered body-track against it;
  * the converter watcher writes its INPUTS to `converter-inputs/call-NN.npz`. The delivered
    `.npz` carries the 19 landmarks but NOT `spine_world_z_up_m`, and every take-level pelvis
    figure is measured against SOMA-77's `Spine1`; without this dump the take block would have
    to re-derive the spine point, which is a re-implementation.

THE MODES, each a wrapper around the REAL projection and nothing else:

  * `shipped`                 -- no wrapper beyond the recording one. The hygiene arm when
                                 `src/` is unchanged, and the delivery arm afterwards.
  * `control-overwrite-locals`-- P1 CONTROL 1. After the projection returns, the foot and toe
                                 LOCAL rotations are overwritten with the pre-projection ones.
                                 A protected channel is verified to have actually changed
                                 (the control asserts it, so a no-op cannot masquerade as a
                                 control). It must FAIL P1; it may PASS P2, because a foot's
                                 own rotation does not move its origin -- which is the whole
                                 reason P1 exists and the anchor check alone does not suffice.
  * `control-clear-contacts`  -- P1 CONTROL 2. After the projection returns, the NONEMPTY
                                 contact mask is cleared (and the root is put back, because
                                 `BodyTrack.__post_init__` runs `validate_body_track` and a
                                 hoisted root with no contacts is fine, but a contact with a
                                 moved foot is refused -- see D9b's `zero-hoist` note). It
                                 must FAIL P1 on the mask; it may PASS P2, because the
                                 geometry stays planted.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7c_pelvis_rest_delivery.py \
        --out artifacts/compare/d7c-pelvis-rest/delivery-hygiene --expect-byte-identical
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

if str(ROOT / "tools/compare") not in sys.path:                      # noqa: E402
    sys.path.insert(0, str(ROOT / "tools/compare"))
from d7c_source_fingerprint import fingerprint_now as source_fingerprint  # noqa: E402

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

FOOT_JOINTS = ("LeftFoot", "LeftToes", "RightFoot", "RightToes")


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


def install_watcher(mode: str, snapshots: Path, inputs: Path) -> dict:
    """Wrap the projection and the converter. The real functions do the work."""

    log: dict = {"projection_calls": [], "converter_calls": []}
    real_projection = cm.project_generated_foot_contacts
    real_converter = cm.positions_to_body_track

    def converter(positions, **kwargs):
        index = len(log["converter_calls"])
        skeleton = kwargs.get("skeleton")
        names = list(skeleton.names) if skeleton is not None else []
        spine = kwargs.get("spine_world_z_up_m")
        toes = kwargs.get("toe_world_z_up_m")
        # The real caller already passes its own `pelvis_report_out` dict (:3791) and the
        # run-report publishes it. Read THAT dict after the call; never substitute one, or
        # the delivered report loses its `pelvis_frame` block.
        track = real_converter(positions, **kwargs)
        report = dict(kwargs.get("pelvis_report_out") or {})
        np.savez(
            inputs / f"call-{index:02d}.npz",
            positions_world_z_up_m=np.asarray(positions, dtype=np.float64),
            spine_world_z_up_m=(np.asarray(spine, dtype=np.float64)
                                if spine is not None else np.zeros(0)),
            toe_world_z_up_m=(np.asarray(toes, dtype=np.float64)
                              if toes is not None else np.zeros(0)),
            rest_translations_m=(np.asarray(skeleton.rest_translations_m, dtype=np.float64)
                                 if skeleton is not None else np.zeros(0)),
            joint_names=np.array(names),
        )
        log["converter_calls"].append({
            "call_index": index,
            "frames": int(len(np.asarray(positions))),
            "spine_supplied": spine is not None,
            "pelvis_report": {k: (v if isinstance(v, (int, float, str, bool, list))
                                  else str(v)) for k, v in report.items()},
            "inputs": f"converter-inputs/call-{index:02d}.npz",
        })
        return track

    def projection(track, **kwargs):
        projected, diagnostics = real_projection(track, **kwargs)
        # THE SNAPSHOT IS TAKEN HERE, from the function's OWN RETURN, BEFORE any control
        # mutates it. Taking it afterwards was a real defect and it made a control
        # undetectable: `control-clear-contacts` cleared the mask on the delivered track AND
        # on the snapshot, so P1 compared two copies of the same mutation and read PASS on a
        # build that was mutated by construction. A control that its own instrument cannot
        # see is worse than no control.
        honest = {
            "root_translation_m": np.array(projected.root_translation_m, copy=True),
            "local_rotations_xyzw": np.array(projected.local_rotations_xyzw, copy=True),
            "foot_contacts": np.array(projected.foot_contacts, copy=True),
        }
        note = ""
        if mode == "control-overwrite-locals":
            names = list(projected.joint_names)
            before = np.array(projected.local_rotations_xyzw, copy=True)
            rotations = np.array(projected.local_rotations_xyzw, copy=True)
            pre = np.asarray(track.local_rotations_xyzw)
            for name in FOOT_JOINTS:
                rotations[:, names.index(name)] = pre[:, names.index(name)]
            changed = int((~np.all(rotations == before, axis=-1)).sum())
            if changed == 0:
                raise SystemExit(
                    "control-overwrite-locals is a no-op on this build: the projection "
                    "did not change a foot local, so the control does not exercise P1")
            note = f"overwrote {changed} foot/toe local samples with the pre-projection ones"
            projected = replace(projected, local_rotations_xyzw=rotations)
        elif mode == "control-clear-contacts":
            contacts = np.asarray(projected.foot_contacts)
            if not contacts.any():
                raise SystemExit(
                    "control-clear-contacts is a no-op on this build: the projection "
                    "returned an empty mask, so the control does not exercise P1")
            note = (f"cleared a nonempty mask ({[int(v) for v in contacts.sum(0)]} frames) "
                    "and restored the pre-projection root so the validator accepts it")
            projected = replace(
                projected,
                foot_contacts=np.zeros_like(contacts),
                root_translation_m=np.array(track.root_translation_m, copy=True))
        index = len(log["projection_calls"])
        hoist = (np.asarray(projected.root_translation_m, dtype=np.float64)
                 - np.asarray(track.root_translation_m, dtype=np.float64))
        magnitude = 1e3 * np.linalg.norm(hoist, axis=1)
        # THE SNAPSHOT. Written from the RETURNED track, at the dtype the dataclass stores
        # (float32 for the root and the rotations, bool for the mask), so P1's comparison
        # against the delivered npz is a comparison of the same numbers and not of a cast.
        np.savez(
            snapshots / f"projection-snapshot-{index:02d}.npz",
            root_translation_m=honest["root_translation_m"],
            local_rotations_xyzw=honest["local_rotations_xyzw"],
            foot_contacts=honest["foot_contacts"],
            joint_names=np.array(list(projected.joint_names)),
            pre_root_translation_m=np.asarray(track.root_translation_m),
            pre_local_rotations_xyzw=np.asarray(track.local_rotations_xyzw),
        )
        log["projection_calls"].append({
            "call_index": index,
            "mode": mode,
            "control_note": note,
            "frames": int(len(magnitude)),
            "contacts": [int(v) for v in np.asarray(projected.foot_contacts).sum(0)],
            "contact_runs": _runs_of(np.asarray(projected.foot_contacts)),
            "diagnostics": diagnostics.as_dict(),
            "hoist_mm": _summary(magnitude),
            "frames_over_0_5mm": int((magnitude > 0.5).sum()),
            "hoisted_frames": [int(i) for i in np.flatnonzero(magnitude > 0.5)],
            "snapshot": f"projection-snapshot-{index:02d}.npz",
        })
        return projected, diagnostics

    cm.project_generated_foot_contacts = projection
    cm.positions_to_body_track = converter
    log["restore"] = (real_projection, real_converter)
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", default="shipped",
                        choices=("shipped", "control-overwrite-locals",
                                 "control-clear-contacts"))
    parser.add_argument("--pelvis-mode", default=None,
                        help="hold `cm.PELVIS_FRAME_SOURCE` at this value for the build. "
                             "The REFACTOR TRIPWIRE's mechanism: with it at "
                             "`C_kabsch_pelvis` the refactored `_pelvis_world_frames` must "
                             "reproduce the D9b delivery bit for bit, 8 of 8.")
    parser.add_argument("--src-state", default="",
                        help="what src/ carried on this arm, recorded verbatim")
    parser.add_argument("--src-stage", choices=("pre_change", "refactored"), required=True,
                        help="WHICH SOURCE STAGE this build is, stated by the operator and "
                             "verified by the gate against the converter's hash and git. It "
                             "is an input, never inferred: a producer that decided its own "
                             "stage from the gate's own reference file would only ever agree "
                             "with itself.")
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
    snapshots = out / "projection-snapshots"
    if snapshots.exists():
        shutil.rmtree(snapshots)
    snapshots.mkdir()
    inputs = out / "converter-inputs"
    if inputs.exists():
        shutil.rmtree(inputs)
    inputs.mkdir()
    before = observation_hashes(out)
    delivered_hashes = observation_hashes(DELIVERED)

    import build_commercial_multiview_comparison as build  # noqa: E402

    print(f"resolved autoanim_gnm: {Path(autoanim_gnm.__file__).resolve()}")
    print(f"resolved commercial_multiview: {Path(cm.__file__).resolve()}")
    print(f"resolved build script: {Path(build.__file__).resolve()}")
    print(f"PELVIS_FRAME_SOURCE = {cm.PELVIS_FRAME_SOURCE!r}")

    saved_source = cm.PELVIS_FRAME_SOURCE
    if args.pelvis_mode is not None:
        cm.PELVIS_FRAME_SOURCE = args.pelvis_mode
        print(f"PELVIS_FRAME_SOURCE held at {cm.PELVIS_FRAME_SOURCE!r} for this build")
    log = install_watcher(args.mode, snapshots, inputs)
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
        cm.project_generated_foot_contacts, cm.positions_to_body_track = log.pop(
            "restore")
        cm.PELVIS_FRAME_SOURCE = saved_source
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
        "title": ("D7c rebuild -- the real build script, this step's own output directory, "
                  "the shipped delivery's own cached detections copied in"),
        "output": str(out.relative_to(ROOT)),
        "mode": args.mode,
        "pelvis_frame_source_at_build_time": (args.pelvis_mode
                                              if args.pelvis_mode is not None
                                              else saved_source),
        "pelvis_mode_held": args.pelvis_mode,
        "resolved_module": str(Path(cm.__file__).resolve()),
        # THE SOURCE THAT PRODUCED THIS BUILD, by content and not by location. A path prefix
        # accepts a nested checkout and rejects an identical one elsewhere; a hash of every
        # loaded `autoanim_gnm` module says which code actually ran. See
        # `tools/compare/d7c_source_fingerprint.py` for why, and for the three stages the
        # gate must keep apart.
        "source_fingerprint": source_fingerprint(
            args.pelvis_mode if args.pelvis_mode is not None else saved_source,
            stage=args.src_stage),
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
                "D7c is a CONVERTER-ONLY change on byte-identical landmarks: BOTH the raw "
                "and the smoothed arrays must be byte-identical on every arm, hygiene and "
                "candidate alike. If either moves, the change did more than refit the "
                "pelvis and the same-denominator clause has not returned to PASS."),
            "delivered_files_vs_shipped": files,
            "all_delivered_files_identical": all(r["identical"] for r in files.values()),
        },
        "watcher": {
            "note": ("the REAL `project_generated_foot_contacts` ran; this is a recording "
                     "of its input and its RETURN, never a re-implementation. The return "
                     "is saved under projection-snapshots/ and is P1's reference."),
            "projection_calls": log["projection_calls"],
            "converter_calls": log["converter_calls"],
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
