#!/usr/bin/env python3
"""Is the head's failure in the 2D, or in the estimator? The finger gate's step 3.

`HEAD_ORIENTATION_MEASURED.md` §7.3 named this as one of two tests that must precede a
model-constrained head fit, and it is the one that decides whether such a fit can work
at all.

`FINGER_TRIANGULATION_GATE.md`'s decisive structure was that the fingers **passed**
cross-view reprojection at 1.2x the body control while failing the physical invariant at
4.4x. That combination localises the failure in **depth** -- the 2D was fine, the
estimator was wrong -- which is why a model-constrained fit rescued them.

If the head's 2D instead **fails** epipolar consistency, the failure is in the detector,
and no fit can repair 2D observations that do not correspond to one 3D point. Same test,
same footage, same association, run per landmark rather than per person.

`_epipolar_distance_px` in the pipeline returns the **symmetric** distance -- the sum of
two one-sided distances -- and CLAUDE.md records the measured ratio at 1.962. This halves
it, so every number below is a one-sided pixel distance at 1280 width.

Blind to: error along the epipolar line, which is exactly the depth direction. That is
the point -- this test is *designed* to be blind to depth, so that a failure here means
the 2D disagree in the one direction depth cannot explain.
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
    JOINT_INDEX, _fundamental_matrix, _person_array, load_camera_rig, load_observation_jsonl,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from associate import CAMERAS, OUT, RIG, load  # noqa: E402

MIN_CONFIDENCE = 0.25
SOMA = {"Neck2": 5, "Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10,
        "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14, "LeftShin": 68, "LeftFoot": 69}
SOMA_HEAD = {"Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"}
AV = {name: JOINT_INDEX[name] for name in
      ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
       "left_shoulder", "left_elbow", "left_wrist", "left_knee", "left_ankle")}
AV_HEAD = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}


def one_sided_px(fundamental: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
    """Half the symmetric point-to-epipolar-line distance, in pixels."""
    s = np.append(source, 1.0)
    t = np.append(target, 1.0)
    target_line = fundamental @ s
    source_line = fundamental.T @ t
    numerator = abs(float(t @ target_line))
    tn = float(np.hypot(target_line[0], target_line[1]))
    sn = float(np.hypot(source_line[0], source_line[1]))
    if tn < 1e-9 or sn < 1e-9:
        return float("nan")
    return 0.5 * (numerator / tn + numerator / sn)


def score(points: dict[str, np.ndarray], cameras, head_names: set[str]) -> dict:
    """points[name] is [frame, camera, 3] of (x, y, confidence)."""
    fundamentals = {
        (a, b): _fundamental_matrix(cameras[a], cameras[b])
        for a, b in combinations(range(len(cameras)), 2)
    }
    rows: dict[str, dict] = {}
    for name, array in points.items():
        distances = []
        for frame in range(array.shape[0]):
            for a, b in fundamentals:
                sa, sb = array[frame, a], array[frame, b]
                if not (np.isfinite(sa).all() and np.isfinite(sb).all()):
                    continue
                if sa[2] < MIN_CONFIDENCE or sb[2] < MIN_CONFIDENCE:
                    continue
                value = one_sided_px(fundamentals[(a, b)], sa[:2], sb[:2])
                if np.isfinite(value):
                    distances.append(value)
        values = np.asarray(distances)
        rows[name] = {
            "region": "head" if name in head_names else "body control",
            "pairs": int(values.size),
            "median_px": float(np.median(values)) if values.size else None,
            "p95_px": float(np.percentile(values, 95)) if values.size else None,
        }
    body = [r["median_px"] for n, r in rows.items() if r["region"] == "body control"]
    control = float(np.median(body))
    for row in rows.values():
        row["ratio_to_body_control"] = row["median_px"] / control if row["median_px"] else None
    return {"body_control_median_px": control, "landmarks": rows}


def main() -> None:
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    state = np.load(OUT / "association.npz")
    assignment, used = state["assignment"], state["used"]
    report: dict = {}

    # --- SOMA-77, on the association recovered from the shipped pipeline ---------
    cameras, observations = load()
    frames = len(observations[0])
    marks = [[[np.asarray(p["landmarks_soma77"], dtype=np.float64) for p in v[f]["people"]]
              for f in range(frames)] for v in observations]
    for subject in range(2):
        points = {name: np.full((frames, len(CAMERAS), 3), np.nan) for name in SOMA}
        for frame in range(frames):
            if not used[frame, subject]:
                continue
            for camera in range(len(CAMERAS)):
                person = int(assignment[frame, subject, camera])
                if person < 0:
                    continue
                for name, index in SOMA.items():
                    points[name][frame, camera] = marks[camera][frame][person][index]
        report[f"soma77_subject_{subject:02d}"] = score(points, cameras, SOMA_HEAD)

    # --- Apple Vision, associated by its own run --------------------------------
    av_obs = [load_observation_jsonl(
        f"artifacts/commercial-multiview-soma77/work/{name}-observations.jsonl") for name in CAMERAS]
    av_cameras = [rig[name].scaled(av_obs[0][0]["width"], av_obs[0][0]["height"]) for name in CAMERAS]
    calls: list[np.ndarray] = []
    real = cm.associate_frame_graph

    def recording(*args, **kwargs):
        associated, cost = real(*args, **kwargs)
        calls.append(np.array(associated, copy=True))
        return associated, cost

    cm.reconstruct_multiview(av_cameras, av_obs, subject_count=2,
                             sample_rate_hz=30, associator=recording)
    associated = np.stack(calls)  # [frame, subject, camera, joint, 3]
    for subject in range(2):
        points = {name: associated[:, subject, :, index] for name, index in AV.items()}
        report[f"apple_vision_subject_{subject:02d}"] = score(points, av_cameras, AV_HEAD)

    Path("artifacts/head-lane/head-epipolar-gate.json").write_text(json.dumps(report, indent=2))
    for key, block in report.items():
        print(f"\n=== {key}   (body control median {block['body_control_median_px']:.2f} px) ===")
        print(f"{'landmark':14s} {'region':13s} {'pairs':>7s} {'median px':>10s} "
              f"{'p95 px':>8s} {'ratio':>7s}")
        for name, row in sorted(block["landmarks"].items(), key=lambda kv: kv[1]["region"]):
            print(f"{name:14s} {row['region']:13s} {row['pairs']:7d} "
                  f"{row['median_px']:10.2f} {row['p95_px']:8.2f} "
                  f"{row['ratio_to_body_control']:7.2f}")


if __name__ == "__main__":
    main()
