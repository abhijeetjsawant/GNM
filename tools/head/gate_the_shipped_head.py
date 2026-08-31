#!/usr/bin/env python3
"""Score the head the PIPELINE actually delivers, not the one the prototype produced.

**Why this exists.** The gate passed on rotations from `tools/head/select_smoothing.py`;
the pipeline ships rotations from `src/autoanim_gnm/head_orientation.py`. Those are two
estimators with two selection paths, and on the reference fixture they chose *different*
temporal weights for subject 0 -- 100 in the prototype, 30 in the pipeline. "The head
passes the gate" and "the head is on the delivery path" were therefore two true statements
about two different heads, which is this lane's recurring defect exactly one sentence away.

This dumps the pipeline's own solve in the format `head_gate.py` reads, so from here the
gate scores what ships.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import load_camera_rig, load_observation_jsonl  # noqa: E402
from associate import CAMERAS, OUT, RIG, WORK  # noqa: E402

HEAD_INDICES = {"Head": 6, "HeadEnd": 7, "Jaw": 8, "LeftEye": 9, "RightEye": 10}


def main() -> None:
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    observations = [load_observation_jsonl(WORK / f"{name}-observations.jsonl") for name in CAMERAS]
    cameras = [
        rig[name].scaled(observations[0][0]["width"], observations[0][0]["height"])
        for name in CAMERAS
    ]
    head = [
        [
            [
                np.asarray([person["landmarks_soma77"][i] for i in HEAD_INDICES.values()], float)
                for person in record["people"]
            ]
            for record in camera
        ]
        for camera in observations
    ]
    tracks, diagnostics, _, _ = cm.reconstruct_multiview(
        cameras, observations, subject_count=2, sample_rate_hz=30,
        head_landmarks_by_camera=head, head_landmark_names=tuple(HEAD_INDICES),
    )
    payload: dict[str, np.ndarray] = {}
    report = []
    for subject, entry in enumerate(diagnostics.head_orientation):
        report.append({"subject": subject, **entry})
        if entry.get("status") != "solved":
            raise SystemExit(f"subject {subject} did not solve: {entry}")
    # Re-solve to retain the rotations themselves; reconstruct_multiview keeps only the
    # track. Cheap relative to the run above and uses the identical code path.
    from autoanim_gnm.head_orientation import solve_head_orientation
    from autoanim_gnm.commercial_multiview import JOINT_INDEX, _thorax_frames

    state = np.load(OUT / "association.npz")
    assignment = state["assignment"]
    smoothed = {
        s: np.load(f"artifacts/commercial-multiview-soma77/subject-{s:02d}.body-track.npz")[
            "triangulated_world_positions_z_up_m"
        ]
        for s in (0, 1)
    }
    for subject in (0, 1):
        frames = len(observations[0])
        obs = np.full((frames, len(CAMERAS), len(HEAD_INDICES), 3), np.nan)
        for frame in range(frames):
            for camera in range(len(CAMERAS)):
                person = int(assignment[frame, subject, camera])
                if person >= 0:
                    obs[frame, camera] = head[camera][frame][person]
        solved = solve_head_orientation(
            cameras, obs, tuple(HEAD_INDICES),
            thorax_world=_thorax_frames(smoothed[subject]),
            neck_origin_world_m=smoothed[subject][:, JOINT_INDEX["neck"]],
        )
        payload[f"subject_{subject:02d}_head_world"] = solved.rotations_world
        payload[f"subject_{subject:02d}_head_position_m"] = solved.positions_world_m
        payload[f"subject_{subject:02d}_template_m"] = solved.template_m
        report[subject]["resolved_weight"] = solved.temporal_weight
        report[subject]["resolved_reprojection_px"] = solved.reprojection_px
    # The pipeline's rotations carry the ANATOMICAL GAUGE, so the head's local frame is
    # canonical rather than the raw template frame. The gate's P3 must use the canonical
    # up axis; this flag tells it so.
    payload["gauge_applied"] = np.asarray([1])
    np.savez(OUT / "head-solve-shipped.npz", **payload)
    (OUT / "head-solve-shipped.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
