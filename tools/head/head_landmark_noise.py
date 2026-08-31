#!/usr/bin/env python3
"""Which head landmark is actually noisy, and is it noise or is it depth?

`headend_gate.py` found the head-chain segments split sharply: Chest->Neck1 is
6.2/6.9 % stable -- body-control class -- while every segment touching `Head`,
`Jaw`, `HeadEnd` or an eye runs 66-272 %. A segment statistic cannot say which
of its two endpoints is at fault, and it confounds detector noise with real
motion. This separates both.

Two probes, neither of which can be flattered by reprojection:

**A. Cross-view disagreement.** Triangulate each landmark from every *pair* of
supporting cameras independently and measure the spread of those solutions. Four
views of one real 3D point agree; four independent per-view guesses do not. This
is the discriminator `FINGER_TRIANGULATION_GATE.md` identified as the one that
separates an image-measured point from a crop-conditioned prior, and it is
**blind to real motion** -- it is a within-frame statistic.

**B. Rigid-skull residual.** The skull is rigid, so the distance from a
well-tracked neck-base landmark to each head landmark is a constant over the
take. Its spread is that landmark's own error, in millimetres, with the body's
motion divided out. Uses `Neck1` as the anchor, which probe A must first show is
clean -- if it is not, this probe is void and says so.

Blind to: bias. A landmark can be cross-view consistent and rigid-skull constant
while sitting systematically 30 mm from where it is named, which is exactly the
convention-offset term `FITTER_PLAN.md` §4 says dominates on real footage.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.commercial_multiview import triangulate_point  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from associate import CAMERAS, OUT, SUBJECT_COUNT, load  # noqa: E402

LANDMARKS = {
    "Hips": 0, "Chest": 3, "Neck1": 4, "Neck2": 5, "Head": 6, "HeadEnd": 7,
    "Jaw": 8, "LeftEye": 9, "RightEye": 10,
    "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14,
    "LeftShin": 68, "LeftFoot": 69,
}
ANCHOR = "Neck1"


def main() -> None:
    cameras, observations = load()
    state = np.load(OUT / "association.npz")
    assignment, used = state["assignment"], state["used"]
    frames = len(observations[0])
    names = list(LANDMARKS)

    marks = [
        [[np.asarray(p["landmarks_soma77"], dtype=np.float64) for p in values[f]["people"]]
         for f in range(frames)]
        for values in observations
    ]

    report: dict = {}
    for subject in range(SUBJECT_COUNT):
        rows: dict[str, dict] = {}
        anchor_xyz = np.full((frames, 3), np.nan)
        full_xyz = {name: np.full((frames, 3), np.nan) for name in names}
        spread: dict[str, list[float]] = {name: [] for name in names}
        spread_max: dict[str, list[float]] = {name: [] for name in names}
        rms: dict[str, list[float]] = {name: [] for name in names}
        pairs_used: dict[str, list[int]] = {name: [] for name in names}
        confidence: dict[str, list[float]] = {name: [] for name in names}
        pixel_span: dict[str, list[float]] = {name: [] for name in names}
        support_counts: dict[str, list[int]] = {name: [] for name in names}

        for frame in range(frames):
            if not used[frame, subject]:
                continue
            for name in names:
                index = LANDMARKS[name]
                points = np.full((len(CAMERAS), 2), np.nan)
                weights = np.zeros(len(CAMERAS))
                for camera in range(len(CAMERAS)):
                    person = int(assignment[frame, subject, camera])
                    if person < 0:
                        continue
                    x, y, c = marks[camera][frame][person][index]
                    points[camera], weights[camera] = (x, y), c
                full = triangulate_point(cameras, points, weights, pixel_scale=1.0)
                if full is None:
                    continue
                full_xyz[name][frame] = full.position_world_m
                inliers = list(full.used_camera_indices)
                support_counts[name].append(len(inliers))
                confidence[name].append(float(np.mean(weights[inliers])))
                pixel_span[name].append(
                    float(np.max(np.linalg.norm(points[inliers] - points[inliers].mean(axis=0), axis=1)))
                )
                # --- probe A: every 2-view solution from the accepted views ----
                solutions = []
                for a, b in combinations(inliers, 2):
                    pair_points = np.full((len(CAMERAS), 2), np.nan)
                    pair_weights = np.zeros(len(CAMERAS))
                    for camera in (a, b):
                        pair_points[camera], pair_weights[camera] = points[camera], weights[camera]
                    pair = triangulate_point(
                        cameras, pair_points, pair_weights, pixel_scale=1.0,
                        inlier_threshold_px=1e6,  # a 2-view solve has no outlier to reject
                    )
                    if pair is not None:
                        solutions.append(pair.position_world_m)
                if len(solutions) >= 2:
                    stack = np.asarray(solutions)
                    gaps = [np.linalg.norm(p - q) for p, q in combinations(stack, 2)]
                    # MAX over pairs scales with the NUMBER of pairs, so a landmark
                    # seen by 4 cameras (6 pairs) reads worse than one seen by 3
                    # (3 pairs) at identical noise -- a composition artefact, since
                    # camera support differs by landmark and by detector. Median
                    # pair distance and RMS about the centroid do not, so both are
                    # carried and the median is the one to quote.
                    spread[name].append(float(np.median(gaps) * 1000.0))
                    spread_max[name].append(float(max(gaps) * 1000.0))
                    centroid = stack.mean(axis=0)
                    rms[name].append(
                        float(np.sqrt(np.mean(np.sum((stack - centroid) ** 2, axis=1))) * 1000.0)
                    )
                    pairs_used[name].append(len(gaps))
            anchor_xyz[frame] = full_xyz[ANCHOR][frame]

        for name in names:
            values = np.asarray(spread[name])
            ok = np.isfinite(full_xyz[name]).all(axis=1) & np.isfinite(anchor_xyz).all(axis=1)
            rigid = np.linalg.norm(full_xyz[name][ok] - anchor_xyz[ok], axis=1) * 1000.0
            rows[name] = {
                "frames": int(ok.sum()),
                "camera_support_mean": float(np.mean(support_counts[name])) if support_counts[name] else None,
                "detector_confidence_mean": float(np.mean(confidence[name])) if confidence[name] else None,
                "two_view_spread_mm": {
                    "n": int(values.size),
                    "pairs_per_frame_mean": float(np.mean(pairs_used[name])) if pairs_used[name] else None,
                    "median_of_pair_distances": float(np.median(values)) if values.size else None,
                    "p95_of_pair_distances": float(np.percentile(values, 95)) if values.size else None,
                    "rms_about_centroid_median": float(np.median(rms[name])) if rms[name] else None,
                    "max_over_pairs_median": float(np.median(spread_max[name])) if spread_max[name] else None,
                },
                f"distance_to_{ANCHOR}_mm": {
                    "mean": float(rigid.mean()) if rigid.size else None,
                    "sd": float(rigid.std()) if rigid.size else None,
                },
            }
        report[f"subject_{subject:02d}"] = rows

    Path("artifacts/head-lane").mkdir(parents=True, exist_ok=True)
    Path("artifacts/head-lane/landmark-noise.json").write_text(json.dumps(report, indent=2))

    for subject, rows in report.items():
        print(f"\n=== {subject} ===")
        print(f"{'landmark':12s} {'conf':>6s} {'views':>6s} {'pairs':>6s} "
              f"{'median pair':>12s} {'rms':>8s} {'max(old)':>9s} {'p95':>8s}")
        for name, row in rows.items():
            block = row["two_view_spread_mm"]
            fmt = lambda v, w: "-".rjust(w) if v is None else format(v, f"{w}.1f")
            print(f"{name:12s} {row['detector_confidence_mean']:6.3f} "
                  f"{row['camera_support_mean']:6.2f} "
                  f"{block['pairs_per_frame_mean'] or 0:6.2f} "
                  f"{fmt(block['median_of_pair_distances'], 12)} "
                  f"{fmt(block['rms_about_centroid_median'], 8)} "
                  f"{fmt(block['max_over_pairs_median'], 9)} "
                  f"{fmt(block['p95_of_pair_distances'], 8)}")


if __name__ == "__main__":
    main()
