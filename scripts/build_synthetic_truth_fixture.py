#!/usr/bin/env python3
"""Synthetic ground truth for the multiview body pipeline.

Every accuracy number this project has produced is reprojection, held-out
reprojection, or agreement with MAMMA. None of them is error against truth. This
script makes truth: it poses a skeleton through the pipeline's own forward
kinematics, projects it through the real calibrated rig, and writes observations
in the ordinary contract. The joints it posed *are* the answer, exactly.

What that measures and what it does not:

* the **estimator** -- triangulation, association, the sequence solve -- exactly,
  in millimetres, today;
* the **detector** not at all, because there are no images here. Arm (i) of
  `docs/BATTLE2_SYNTHETIC_TRUTH_FIXTURE.md` renders; this is the no-render path
  that the design found unblocks three of its four arms.

Absent by construction, and therefore never claimable from these numbers:
calibration error, lens distortion, sync error, soft-tissue artefact, and
joint-definition error -- the last of which dominated Battles 0 and 1 on real
footage. **This does not replace the marker session.**

    python scripts/build_synthetic_truth_fixture.py --out artifacts/synthetic-truth

Gate G1, the zero-noise positive control, runs by default: reconstruct from
noiseless projections and recover the truth to under 1 mm. Anything larger is a
convention bug -- an axis swap, a quaternion reorder, a mirrored principal point
-- not an accuracy result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers" / "commercial_multiview"))

from autoanim_gnm import commercial_multiview as cm
from autoanim_gnm.soma_motion import soma_forward_kinematics
from soma77_pose import SOMA77_TO_AUTOANIM

SCHEMA_VERSION = "autoanim.body-observations/1.1"
DETECTOR = "synthetic_truth"
CAMERAS = ("A001", "B001", "C001", "D001")
WORKING_WIDTH, WORKING_HEIGHT = 1280, 720

# The SOMA rest frame is Y-up; the calibrated rig world is Z-up. A +90 degree
# rotation about X carries one to the other with determinant +1, so it rotates
# rather than mirrors -- a mirror here would swap left and right limbs and every
# downstream number would be quietly wrong while looking entirely plausible.
SOMA_TO_WORLD = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Full-body clips only. csg-dialogue-upper-body spans 0.82 m vertically -- it is
# truncated at the waist, and half a skeleton would make the leg joints unposed
# rather than merely unobserved.
MOTION_ROOT = Path(".cache/autoanim_gnm/gem-x/outputs")
FULL_BODY_CLIPS = (
    "autoanim_dialogue/amy-cuddy-dialogue-body",
    "autoanim_squat/research-squat-640",
    "autoanim_will_acting/will-stephen-acting-body",
    "autoanim_real/autoanim_fixture",
    "cpu_smoke/autoanim_fixture",
)


def load_clip(name: str) -> np.ndarray:
    """Forward-kinematic joint positions for one clip, in world axes."""

    path = MOTION_ROOT / name / "soma_motion.npz"
    data = np.load(path, allow_pickle=True)
    positions = soma_forward_kinematics(
        data["root_translation_m"],
        data["local_rotations_xyzw"],
        data["rest_joint_positions_m"],
        data["rest_world_rotations_xyzw"] if "rest_world_rotations_xyzw" in data.files else None,
    )
    return positions @ SOMA_TO_WORLD.T


def build_take(clip: str, frames: int | None, ground_xy: np.ndarray) -> np.ndarray:
    """One subject's take: a single clip, placed in the volume.

    Deliberately **one** clip. An earlier draft concatenated several to reach a
    target length, and it made the truth non-physical in two ways at once: each
    recording carries its own `rest_joint_positions_m`, so the performer's bones
    changed length at every boundary, and the takes are unrelated, so the body
    teleported. `estimate_limb_lengths_m` fixes one limb length per subject per
    take and then constrains against it, so the first artefact put the estimator
    in a fight it cannot win -- and the resulting error would have been read as
    estimator error. It showed up as a 130 mm maximum against a 0.61 mm median.

    The cost is take length: the longest full-body clip on disk is 96 frames.
    Two subjects with different rest skeletons is a feature, not a compromise --
    gate G9 needs limb lengths that differ between subjects.
    """

    take = load_clip(clip)
    if frames is not None:
        take = take[:frames]
    # Stand the subject where a real performer stood, feet on the floor. Placement
    # comes from the measured take so the synthetic subject is inside every
    # frustum by construction rather than by hope.
    take = take - take[:, :1, :]                       # hips to the origin
    take[:, :, :2] += ground_xy[None, None, :]
    take[:, :, 2] -= take[..., 2].min()
    return take


def observations_for(
    cameras: tuple[cm.CalibratedCamera, ...], subjects: np.ndarray
) -> dict[str, list[dict]]:
    """Project truth into the ordinary observation contract, noiselessly."""

    frames = subjects.shape[1]
    records: dict[str, list[dict]] = {name: [] for name in CAMERAS}
    for camera_index, camera in enumerate(cameras):
        for frame in range(frames):
            people = []
            for subject in range(subjects.shape[0]):
                joints = {}
                for name, soma in SOMA77_TO_AUTOANIM.items():
                    uv, depth = camera.project(subjects[subject, frame, soma])
                    if depth <= 0.0:
                        continue
                    joints[name] = {
                        "x": float(uv[0]),
                        "y": float(uv[1]),
                        # Uniform and high: there is no detector here to be
                        # uncertain, and a fabricated confidence would leak a
                        # signal the real pipeline never gets.
                        "confidence": 0.95,
                    }
                if joints:
                    people.append({"index": len(people), "joints": joints})
            records[CAMERAS[camera_index]].append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "detector": DETECTOR,
                    "frame_index": frame,
                    "width": camera.width,
                    "height": camera.height,
                    "image_path": f"synthetic://{CAMERAS[camera_index]}/{frame:06d}",
                    "people": people,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/synthetic-truth"))
    parser.add_argument("--rig", type=Path, default=Path("artifacts/soma77-full/camera-rig.json"))
    # None means "the natural length of the shortest clip". Concatenating to reach
    # a longer take is what made the truth non-physical; see build_take.
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--placement", type=Path,
                        default=Path("artifacts/handfit-arrays/body-track.npz"))
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    rig = cm.load_camera_rig(args.rig)
    cameras = tuple(camera.scaled(WORKING_WIDTH, WORKING_HEIGHT) for camera in rig)

    if args.placement.exists():
        track = np.load(args.placement, allow_pickle=True)["positions"]
        roots = track[:, :, cm.JOINT_INDEX["root"], :2]
        ground = np.asarray([np.nanmedian(roots[s], axis=0) for s in range(min(2, len(roots)))])
    else:
        ground = np.asarray([[0.5, 4.9], [-0.9, 4.7]])
    print(f"placement (world x, y): {np.round(ground, 2).tolist()}")

    takes = [build_take(FULL_BODY_CLIPS[0], args.frames, ground[0]),
             build_take(FULL_BODY_CLIPS[2], args.frames, ground[1])]
    length = min(len(t) for t in takes)
    subjects = np.stack([t[:length] for t in takes])
    print(f"truth: {subjects.shape[0]} subjects x {subjects.shape[1]} frames x "
          f"{subjects.shape[2]} joints")

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    records = observations_for(cameras, subjects)
    for name, rows in records.items():
        (work / f"{name}-observations.jsonl").write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8"
        )
    np.savez_compressed(args.out / "truth.npz", subjects_soma77_m=subjects,
                        joint_map=np.array(list(SOMA77_TO_AUTOANIM.items()), dtype=object))
    (args.out / "camera-rig.json").write_text(args.rig.read_text(encoding="utf-8"), encoding="utf-8")
    seen = [sum(len(r["people"]) for r in rows) for rows in records.values()]
    print(f"observations written: {seen} person-frames per camera")

    if args.skip_gate:
        return 0
    return gate_g1(cameras, records, subjects, args.out)


def gate_g1(cameras, records, subjects, out: Path) -> int:
    """Zero-noise positive control. Truth in, truth out, under 1 mm."""

    observations = [records[name] for name in CAMERAS]
    tracks, diagnostics, positions, _ = cm.reconstruct_multiview(
        cameras, observations, subject_count=2, sample_rate_hz=30
    )
    # SOMA-77 has no ear landmarks, so the 19-joint contract's ears are absent
    # here exactly as they are absent on real footage. Scoring them would measure
    # the reconstruction's fallback behaviour, not its accuracy.
    scored = [name for name in cm.JOINT_INDEX if name in SOMA77_TO_AUTOANIM]
    columns = [cm.JOINT_INDEX[name] for name in scored]
    truth = np.stack([
        np.stack([subjects[s, :, SOMA77_TO_AUTOANIM[name]] for name in scored], axis=1)
        for s in range(subjects.shape[0])
    ])
    positions = positions[:, :, columns, :]
    print(f"scoring {len(scored)} of {len(cm.JOINT_INDEX)} joints; "
          f"absent from SOMA-77: {sorted(set(cm.JOINT_INDEX) - set(scored))}")
    # Subject order is not guaranteed to match; pair on median root distance.
    root = scored.index("root")
    cost = np.asarray([
        [np.nanmedian(np.linalg.norm(positions[a, :, root] - truth[b, :, root], axis=1))
         for b in range(2)] for a in range(2)
    ])
    pair = (0, 1) if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else (1, 0)
    errors = np.concatenate([
        np.linalg.norm(positions[a] - truth[pair[a]], axis=2).ravel() for a in range(2)
    ]) * 1000.0
    finite = errors[np.isfinite(errors)]
    coverage = len(finite) / len(errors)
    median, p95, worst = (float(np.median(finite)), float(np.percentile(finite, 95)),
                          float(finite.max()))
    verdict = "PASS" if median < 1.0 and coverage > 0.99 else "FAIL"
    print(f"\nG1 zero-noise positive control: median {median:.4f} mm, p95 {p95:.4f} mm, "
          f"max {worst:.4f} mm, coverage {coverage*100:.2f}%  -> {verdict}")
    if verdict == "FAIL":
        print("  A convention bug, not an accuracy result: check the Y-up to Z-up "
              "rotation, the quaternion order, and the principal-point sign.")
    (out / "g1-report.json").write_text(json.dumps({
        "gate": "G1 zero-noise positive control", "threshold_mm": 1.0,
        "median_mm": median, "p95_mm": p95, "max_mm": worst,
        "coverage": coverage, "verdict": verdict,
        "subject_pairing": list(pair), "joints_scored": scored,
        "diagnostics": diagnostics.as_dict(),
    }, indent=1), encoding="utf-8")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
