#!/usr/bin/env python3
"""D7 option (c): the frozen-upright pelvis, RENDERED as a control delivery.

**THIS DELIVERY NEVER SHIPS.** It is a CONTROL, and every artifact it writes is labelled
one. It exists to answer the question section 7b of
`docs/reviews/pelvis-frame-2026-09-04.md` forced and could not settle: performer 0's
torso+legs silhouette rise on the bent tercile is not attributed between *"the pelvis is
measured"* and *"the pelvis is merely not the thorax"*. The photographs are the one
reference neither candidate can optimise, so the constant gets an arm in them.

HOW, AND WHY IT IS A SUBSTITUTION AND NOT A COPY. `commercial_multiview._pelvis_world_frames`
is module level and called by bare name -- exactly as `_leg_root_offset` and `_joint_origin`
are, and for exactly this reason: an instrument substitutes it and the control runs through
the converter's IDENTICAL call site, the same `_frame_alignment`, the same secondary axis
(the hip line), the same root formula, the same everything below. Only the primary target
differs: the rig's +Y, which IS world up after `positions_to_body_track`'s change of basis.
A re-implementation of the converter would be a different pipeline wearing the same name.

NOTHING ENTERS `src/`. No shipped constant moves. `scripts/build_commercial_multiview_
comparison.py` is imported as a module and its real `main()` runs, so the detector feed, the
association, the triangulation, the head solve, the toe solve, the sizing, the exporter and
the report are all the delivered ones.

    PYTHONPATH=$PWD/src .venv/bin/python tools/compare/d7_world_vertical_delivery.py

Writes `artifacts/compare/d7-pelvis-frame/delivery-world-vertical/` and
`artifacts/compare/d7-pelvis-frame/world-vertical-build.json` (the hygiene assertions).
NOTHING is written under `artifacts/commercial-multiview-soma77/` or `artifacts/compare/i6/`.
"""

from __future__ import annotations

import json
import shutil
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

OUT_DIR = ROOT / "artifacts/compare/d7-pelvis-frame"
CONTROL = OUT_DIR / "delivery-world-vertical"
DELIVERED = ROOT / "artifacts/commercial-multiview-soma77"
REPORT = OUT_DIR / "world-vertical-build.json"

VIDEOS = ROOT / ".cache/mamma/data/mamma_example/pushing_and_lifting_from_ground/videos"
CALIBRATION = ROOT / ".cache/mamma/configs/examples/calib/iphones_outdoors.yaml"


def world_vertical_frames(points_rig_y_up_m, spine1_rig_y_up_m, *, mode=None,
                          smoothing_frames=None):
    """THE CONTROL. A pelvis frozen bolt upright, through the converter's own construction.

    Same signature as `_pelvis_world_frames`, same call site, same `_frame_alignment`, same
    secondary axis (the captured hip line, so yaw still follows the performer). ONLY the
    primary target differs: the rig's +Y. It ignores the spine landmark entirely, which is
    the point -- this is what a constant pelvis pitch looks like, and section 0.6 of the
    review pre-registered that a constant can zero the `Hips`-offset figure. It can.
    """
    points = np.asarray(points_rig_y_up_m, dtype=np.float64)
    frames = len(points)
    left = points[:, cm.JOINT_INDEX["left_hip"]]
    right = points[:, cm.JOINT_INDEX["right_hip"]]
    quaternions = np.zeros((frames, 4), dtype=np.float64)
    for frame in range(frames):
        quaternions[frame] = cm._frame_alignment(
            (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
            np.asarray((0.0, 1.0, 0.0)), left[frame] - right[frame])
    for frame in range(1, frames):
        if float(np.dot(quaternions[frame], quaternions[frame - 1])) < 0.0:
            quaternions[frame] *= -1.0
    return quaternions, {
        "status": "solved",
        "mode": "world_vertical_CONTROL_NEVER_SHIPS",
        "smoothing_frames": 0,
        "resolved_fraction": 1.0,
        "interpolated_frames": 0,
        "warning": "this is a DEGENERATE control delivery. It ignores the spine landmark.",
    }


def _delivered_diagnostics() -> dict:
    """The delivered run report's diagnostics, whether it nests them or not."""
    report = json.loads((DELIVERED / "run-report.json").read_text())
    return report.get("diagnostics", report)


def observation_hashes(directory: Path) -> dict[str, str]:
    from hashlib import sha256
    return {path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted((directory / "work").glob("*observations.jsonl"))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (CONTROL / "work").exists():
        (CONTROL).mkdir(parents=True, exist_ok=True)
        print(f"copying work/ (never symlinked) into {CONTROL}")
        shutil.copytree(DELIVERED / "work", CONTROL / "work")
    before = observation_hashes(CONTROL)
    delivered_hashes = observation_hashes(DELIVERED)

    import build_commercial_multiview_comparison as build  # noqa: E402

    shipped = cm._pelvis_world_frames
    cm._pelvis_world_frames = world_vertical_frames
    argv = sys.argv[:]
    sys.argv = [
        "build_commercial_multiview_comparison.py",
        "--videos", str(VIDEOS),
        "--calibration-yaml", str(CALIBRATION),
        "--detector", "soma77",
        "--output", str(CONTROL),
    ]
    try:
        code = build.main()
    finally:
        cm._pelvis_world_frames = shipped
        sys.argv = argv
    if code != 0:
        raise SystemExit(f"the control build exited {code}")

    after = observation_hashes(CONTROL)
    run = json.loads((CONTROL / "run-report.json").read_text())
    diagnostics = run.get("diagnostics", run)
    triangulation = {}
    for subject in (0, 1):
        with np.load(CONTROL / f"subject-{subject:02d}.body-track.npz") as a, \
             np.load(DELIVERED / f"subject-{subject:02d}.body-track.npz") as b:
            triangulation[f"subject_{subject:02d}"] = bool(np.array_equal(
                a["triangulated_world_positions_z_up_m"],
                b["triangulated_world_positions_z_up_m"]))
    report = {
        "title": "D7 option (c) -- the frozen-upright pelvis, rendered as a CONTROL",
        "THIS_DELIVERY_NEVER_SHIPS": True,
        "output": str(CONTROL.relative_to(ROOT)),
        "how": ("`commercial_multiview._pelvis_world_frames` substituted at module level; "
                "the REAL build script's main() ran with the same invocation D7's rebuild "
                "used. Nothing entered src/."),
        "work_copied_never_symlinked": True,
        "hygiene": {
            "observations_byte_identical_before_and_after_the_build": before == after,
            "observations_byte_identical_to_the_delivered_build": after == delivered_hashes,
            "observation_files": sorted(after),
            "triangulation_byte_identical_same_denominator": triangulation,
        },
        "pelvis_frame_diagnostics": diagnostics.get("pelvis_frame"),
        "spine_triangulation_diagnostics": diagnostics.get("spine_triangulation"),
        # The delivered run report is FLAT and the rebuild's is nested, so both sides go
        # through the same `.get("diagnostics", report)` fallback. An earlier version did
        # not, read None on one side, and reported a difference that does not exist.
        "head_orientation_unchanged": bool(
            json.dumps(diagnostics.get("head_orientation"), sort_keys=True)
            == json.dumps(_delivered_diagnostics().get("head_orientation"), sort_keys=True)),
        "toe_triangulation_unchanged": bool(
            json.dumps(diagnostics.get("toe_triangulation"), sort_keys=True)
            == json.dumps(_delivered_diagnostics().get("toe_triangulation"), sort_keys=True)),
    }
    ok = (report["hygiene"]["observations_byte_identical_before_and_after_the_build"]
          and report["hygiene"]["observations_byte_identical_to_the_delivered_build"]
          and all(triangulation.values()))
    report["verdict"] = "PASS" if ok else "FAIL"
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "spine_triangulation_diagnostics"},
                     indent=1))
    print(f"\nwrote {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
