#!/usr/bin/env python3
"""Recover the pipeline's cross-view person assignment so the *unmapped* SOMA-77
landmarks can be triangulated on exactly the subjects the rest of the lane uses.

The 17 mapped joints and the 60 discarded ones share one observation record, so
once a detection is assigned to a subject the whole 77-point array comes with
it. What the reconstruction never retains is *which* detection it chose.

**Do not re-implement the association loop.** A hand-replication of
`reconstruct_multiview`'s loop was tried first and drifted from the retained
tracks by a median worst-joint 9-19 mm, which would have been invisible in the
head numbers downstream. This instead runs the real function with the associator
wrapped, so the assignment is the one the pipeline actually made, by
construction.

Naive index matching does not work here and was checked: person 0 in each camera
is not the same person, and taking it as such lands 1.84 m from the retained
track on frame 0.

Provenance gate: the run asserts that `reconstruct_multiview` on these
observations reproduces `artifacts/commercial-multiview-soma77`'s retained
`raw_triangulated_world_positions_z_up_m` **exactly** -- same NaN pattern, 0.0 mm
maximum. If a future edit to the pipeline breaks that, this refuses to write.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import autoanim_gnm.commercial_multiview as cm  # noqa: E402
from autoanim_gnm.commercial_multiview import (  # noqa: E402
    JOINT_NAMES,
    _person_array,
    load_camera_rig,
    load_observation_jsonl,
)

CAMERAS = ("A001", "B001", "C001", "D001")
RIG = Path("artifacts/soma77-full/camera-rig.json")
WORK = Path("artifacts/soma77-full/work")
TRACKS = Path("artifacts/commercial-multiview-soma77")
OUT = Path("artifacts/head-lane")
SUBJECT_COUNT = 2


def load() -> tuple[list, list[list[dict]]]:
    rig = {camera.name: camera for camera in load_camera_rig(RIG)}
    observations = [load_observation_jsonl(WORK / f"{name}-observations.jsonl") for name in CAMERAS]
    width, height = observations[0][0]["width"], observations[0][0]["height"]
    return [rig[name].scaled(width, height) for name in CAMERAS], observations


def recover() -> tuple[np.ndarray, np.ndarray, dict]:
    cameras, observations = load()
    frames = len(observations[0])
    calls: list[np.ndarray] = []

    real = cm.associate_frame_graph

    def recording(*args, **kwargs):
        associated, cost = real(*args, **kwargs)
        calls.append(np.array(associated, copy=True))
        return associated, cost

    tracks, diagnostics, _, raw = cm.reconstruct_multiview(
        cameras, observations, subject_count=SUBJECT_COUNT, sample_rate_hz=30, associator=recording
    )
    if len(calls) != frames:
        raise SystemExit(f"associator ran {len(calls)} times for {frames} frames")

    # --- provenance gate: this must BE the retained run, not merely resemble it
    provenance: dict[str, dict] = {}
    for subject in range(SUBJECT_COUNT):
        retained = np.load(TRACKS / f"subject-{subject:02d}.body-track.npz")[
            "raw_triangulated_world_positions_z_up_m"
        ]
        same_nan = bool(
            np.array_equal(np.isnan(raw[subject]).any(axis=2), np.isnan(retained).any(axis=2))
        )
        delta = np.linalg.norm(raw[subject] - retained, axis=2) * 1000.0
        worst = float(np.nanmax(delta))
        provenance[f"subject_{subject:02d}"] = {
            "nan_pattern_identical": same_nan,
            "max_delta_mm": worst,
        }
        if not same_nan or worst > 0.0:
            raise SystemExit(
                f"subject {subject}: replay does not reproduce the retained track "
                f"(nan_identical={same_nan}, max={worst} mm) -- refusing to write"
            )

    # --- map each associated row back to its source detection ------------------
    assignment = np.full((frames, SUBJECT_COUNT, len(CAMERAS)), -1, dtype=np.int64)
    unmatched = 0
    for frame in range(frames):
        people = [
            [_person_array(person) for person in values[frame]["people"]] for values in observations
        ]
        for subject in range(SUBJECT_COUNT):
            for camera in range(len(CAMERAS)):
                row = calls[frame][subject, camera]
                if not np.isfinite(row).any():
                    continue
                hits = [
                    index
                    for index, candidate in enumerate(people[camera])
                    if np.array_equal(row, candidate, equal_nan=True)
                ]
                if len(hits) == 1:
                    assignment[frame, subject, camera] = hits[0]
                else:
                    unmatched += 1

    # A subject-frame the pipeline rejected on its temporal gate contributes no
    # world position, so its landmarks must not contribute either.
    used = np.isfinite(raw).all(axis=3).any(axis=2)  # [subject, frame]
    used = used.T  # [frame, subject]

    report = {
        "frames": frames,
        "rows_not_uniquely_matched": unmatched,
        "subject_frames_used": {
            f"subject_{s:02d}": int(used[:, s].sum()) for s in range(SUBJECT_COUNT)
        },
        "cameras_per_subject_frame_mean": float((assignment >= 0).sum(axis=2)[used].mean()),
        "provenance_vs_retained_track": provenance,
        "diagnostics_match_run_report": {
            "valid_joint_fraction": diagnostics.valid_joint_fraction,
            "median_reprojection_error_px": diagnostics.median_reprojection_error_px,
        },
    }
    return assignment, used, report


def main() -> None:
    assignment, used, report = recover()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "association.npz", assignment=assignment, used=used)
    (OUT / "association-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
