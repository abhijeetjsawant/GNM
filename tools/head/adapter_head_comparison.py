#!/usr/bin/env python3
"""Score the head landmarks of every already-integrated detector on THIS footage.

`docs/HEAD_FEET_HANDS_PLAN.md` §2 requires this before anyone proposes a new
detector: enumerate what the integrated adapters emit and score them on the same
frames, same cameras, same estimator.

  * `soma77_pose.py`   -- Head(6), HeadEnd(7), Jaw(8), LeftEye(9), RightEye(10).
                          **No ears at all**; `left_ear`/`right_ear` are schema
                          only and populated on zero frames.
  * `apple_vision_pose.swift` -- nose, both eyes and **both ears** (`.leftEar`,
                          `.rightEar`, lines 59-60). Its detections for this
                          exact window are already on disk: they are the boxes
                          the SOMA-77 worker was driven from.
  * `mediapipe_pose.py`  -- also carries ears (lines 71-72). No detections exist
                          for this window; noted, not run.

The ear baseline is the point of the exercise: ~145 mm against the eye
baseline's 63 mm, so if head orientation is limited by feature size in pixels an
ear axis should be ~2.3x better conditioned.

Both detectors are scored through `reconstruct_multiview` itself, so the numbers
come from the production estimator with production gates, and the body controls
in each row are that detector's own -- never crossed.

Blind to: joint convention. Apple Vision predicts *surface* landmarks and
SOMA-77 predicts *skeletal joint centres*; an ear and an eye are different
points on different definitions. This compares how STABLE each is, which is
convention-free, and says nothing about which is closer to anatomical truth.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX,
    load_camera_rig,
    load_observation_jsonl,
)

CAMERAS = ("A001", "B001", "C001", "D001")
RIG = Path("artifacts/soma77-full/camera-rig.json")
SOURCES = {
    "apple_vision": Path("artifacts/commercial-multiview-soma77/work/{cam}-observations.jsonl"),
    "soma77": Path("artifacts/soma77-full/work/{cam}-observations.jsonl"),
}
PAIRS = [
    ("HEAD  ear axis", "right_ear", "left_ear"),
    ("HEAD  eye baseline", "right_eye", "left_eye"),
    ("HEAD  nose->neck", "nose", "neck"),
    ("CONTROL shoulder width", "right_shoulder", "left_shoulder"),
    ("CONTROL forearm L", "left_elbow", "left_wrist"),
    ("CONTROL shin L", "left_knee", "left_ankle"),
]


def rotations(vectors: np.ndarray, ok: np.ndarray) -> dict:
    unit = np.full_like(vectors, np.nan)
    unit[ok] = vectors[ok] / np.linalg.norm(vectors[ok], axis=1, keepdims=True)
    index = np.flatnonzero(ok)
    pairs = index[1:][np.diff(index) == 1]
    if pairs.size == 0:
        return {"n": 0}
    dots = np.einsum("ni,ni->n", unit[pairs], unit[pairs - 1])
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return {
        "n": int(angles.size),
        "median_deg": float(np.median(angles)),
        "p95_deg": float(np.percentile(angles, 95)),
        "max_deg": float(angles.max()),
    }


def main() -> None:
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    report: dict = {}
    for detector, template in SOURCES.items():
        observations = [load_observation_jsonl(str(template).format(cam=name)) for name in CAMERAS]
        width, height = observations[0][0]["width"], observations[0][0]["height"]
        cameras = [rig[name].scaled(width, height) for name in CAMERAS]
        _, _, _, raw = cm.reconstruct_multiview(
            cameras, observations, subject_count=2, sample_rate_hz=30
        )
        detector_block: dict = {}
        for subject in range(raw.shape[0]):
            world = raw[subject]
            have = np.isfinite(world).all(axis=2)
            needed = sorted({JOINT_INDEX[a] for _, a, b in PAIRS} | {JOINT_INDEX[b] for _, a, b in PAIRS})
            # Same denominator inside a detector: one frame set where every
            # landmark this detector is scored on resolved.
            common = have[:, needed].all(axis=1)
            block: dict = {
                "frames_common": int(common.sum()),
                "frames_total": int(world.shape[0]),
                "coverage": {},
                "segments": {},
                "axis_rotation": {},
            }
            for label, a, b in PAIRS:
                ia, ib = JOINT_INDEX[a], JOINT_INDEX[b]
                block["coverage"][label] = {
                    a: int(have[:, ia].sum()), b: int(have[:, ib].sum())
                }
                if common.sum() < 5:
                    continue
                d = np.linalg.norm(world[common, ia] - world[common, ib], axis=1) * 1000.0
                block["segments"][label] = {
                    "mean_mm": float(d.mean()),
                    "sd_mm": float(d.std()),
                    "sd_pct": float(100.0 * d.std() / d.mean()),
                }
                vectors = world[:, ib] - world[:, ia]
                block["axis_rotation"][label] = rotations(vectors, common)
            # The §1b opposition test, on whichever head axis the detector has.
            shoulder = world[:, JOINT_INDEX["left_shoulder"]] - world[:, JOINT_INDEX["right_shoulder"]]
            for label, a, b in (("ear", "right_ear", "left_ear"), ("eye", "right_eye", "left_eye")):
                axis = world[:, JOINT_INDEX[b]] - world[:, JOINT_INDEX[a]]
                ok = common & np.isfinite(axis).all(axis=1)
                if not ok.any():
                    continue
                dots = np.einsum(
                    "ni,ni->n",
                    axis[ok] / np.linalg.norm(axis[ok], axis=1, keepdims=True),
                    shoulder[ok] / np.linalg.norm(shoulder[ok], axis=1, keepdims=True),
                )
                block.setdefault("opposes_shoulders", {})[label] = {
                    "frames": int(ok.sum()), "opposing": int((dots < 0).sum())
                }
            detector_block[f"subject_{subject:02d}"] = block
        report[detector] = detector_block

    Path("artifacts/head-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/head-lane/adapter-head-comparison.json").write_text(json.dumps(report, indent=2))

    for detector, subjects in report.items():
        for subject, block in subjects.items():
            print(f"\n=== {detector} / {subject}  ({block['frames_common']} common of "
                  f"{block['frames_total']}) ===")
            print(f"{'segment':24s} {'mean mm':>9s} {'sd mm':>8s} {'sd%':>7s}   "
                  f"{'rot med':>8s} {'p95':>8s}")
            for label in block["segments"]:
                s = block["segments"][label]
                r = block["axis_rotation"].get(label, {})
                print(f"{label:24s} {s['mean_mm']:9.1f} {s['sd_mm']:8.1f} {s['sd_pct']:6.1f}%   "
                      f"{r.get('median_deg', float('nan')):8.2f} {r.get('p95_deg', float('nan')):8.2f}")
            if "opposes_shoulders" in block:
                print("  opposes shoulders:", json.dumps(block["opposes_shoulders"]))
            print("  coverage:", json.dumps(
                {k: v for k, v in block["coverage"].items() if k.startswith("HEAD")}))


if __name__ == "__main__":
    main()
