#!/usr/bin/env python3
"""Apply the SAME discriminating probe to Apple Vision's ears that rejected HeadEnd.

`head_landmark_noise.py` rejected SOMA-77's head chain on cross-view
disagreement: triangulate each landmark from every supporting camera *pair* and
measure how far apart those independent solutions land. Accepting Apple Vision's
ears on a weaker test than the one that rejected HeadEnd would be exactly the
asymmetry this lane keeps catching, so the ears face the same instrument.

The probe is the one `FINGER_TRIANGULATION_GATE.md` identified as decisive: a
per-view 2D template dressed as a landmark reprojects beautifully and
triangulates inconsistently, because "20 px right of the head centre" is a
different physical direction in a camera 90 degrees away. Cross-view agreement
is what separates an image measurement from a crop-conditioned guess.

Blind to: bias, as always. Consistent is not correct.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_INDEX, JOINT_NAMES, _person_array, load_camera_rig,
    load_observation_jsonl, triangulate_point,
)

CAMERAS = ("A001", "B001", "C001", "D001")
RIG = Path("artifacts/soma77-full/camera-rig.json")
AV = "artifacts/commercial-multiview-soma77/work/{cam}-observations.jsonl"
WATCH = ["left_ear", "right_ear", "left_eye", "right_eye", "nose", "neck",
         "left_shoulder", "right_shoulder", "left_elbow", "left_wrist",
         "left_knee", "left_ankle"]


def main() -> None:
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    observations = [load_observation_jsonl(AV.format(cam=name)) for name in CAMERAS]
    frames = len(observations[0])
    cameras = [rig[name].scaled(observations[0][0]["width"], observations[0][0]["height"])
               for name in CAMERAS]

    calls: list[np.ndarray] = []
    real = cm.associate_frame_graph

    def recording(*args, **kwargs):
        associated, cost = real(*args, **kwargs)
        calls.append(np.array(associated, copy=True))
        return associated, cost

    _, _, _, raw = cm.reconstruct_multiview(
        cameras, observations, subject_count=2, sample_rate_hz=30, associator=recording
    )
    used = np.isfinite(raw).all(axis=3).any(axis=2).T  # [frame, subject]

    report: dict = {}
    for subject in range(2):
        rows: dict[str, dict] = {}
        for name in WATCH:
            joint = JOINT_INDEX[name]
            spread, support, confidence = [], [], []
            for frame in range(frames):
                if not used[frame, subject]:
                    continue
                row = calls[frame][subject]  # [camera, joint, 3]
                points, weights = row[:, joint, :2], row[:, joint, 2]
                full = triangulate_point(cameras, points, np.nan_to_num(weights), pixel_scale=1.0)
                if full is None:
                    continue
                inliers = list(full.used_camera_indices)
                support.append(len(inliers))
                confidence.append(float(np.mean(weights[inliers])))
                solutions = []
                for a, b in combinations(inliers, 2):
                    pair_points = np.full((len(CAMERAS), 2), np.nan)
                    pair_weights = np.zeros(len(CAMERAS))
                    for camera in (a, b):
                        pair_points[camera] = points[camera]
                        pair_weights[camera] = weights[camera]
                    pair = triangulate_point(cameras, pair_points, pair_weights,
                                             pixel_scale=1.0, inlier_threshold_px=1e6)
                    if pair is not None:
                        solutions.append(pair.position_world_m)
                if len(solutions) >= 2:
                    stack = np.asarray(solutions)
                    spread.append(float(max(np.linalg.norm(p - q)
                                            for p, q in combinations(stack, 2)) * 1000.0))
            values = np.asarray(spread)
            rows[name] = {
                "frames_triangulated": len(support),
                "frames_with_three_or_more_views": int(values.size),
                "camera_support_mean": float(np.mean(support)) if support else None,
                "detector_confidence_mean": float(np.mean(confidence)) if confidence else None,
                "two_view_spread_mm_median": float(np.median(values)) if values.size else None,
                "two_view_spread_mm_p95": float(np.percentile(values, 95)) if values.size else None,
            }
        report[f"subject_{subject:02d}"] = rows

    Path("artifacts/head-lane/av-ear-probe.json").write_text(json.dumps(report, indent=2))
    for subject, rows in report.items():
        print(f"\n=== apple_vision / {subject} ===")
        print(f"{'landmark':16s} {'frames':>7s} {'>=3cam':>7s} {'conf':>6s} {'views':>6s} "
              f"{'2view med mm':>13s} {'p95':>9s}")
        for name, row in rows.items():
            median = row["two_view_spread_mm_median"]
            p95 = row["two_view_spread_mm_p95"]
            print(f"{name:16s} {row['frames_triangulated']:7d} "
                  f"{row['frames_with_three_or_more_views']:7d} "
                  f"{row['detector_confidence_mean']:6.3f} {row['camera_support_mean']:6.2f} "
                  f"{'-' if median is None else format(median, '13.1f'):>13s} "
                  f"{'-' if p95 is None else format(p95, '9.1f'):>9s}")


if __name__ == "__main__":
    main()
