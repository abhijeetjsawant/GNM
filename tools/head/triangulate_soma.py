#!/usr/bin/env python3
"""Triangulate arbitrary SOMA-77 landmarks under the pipeline's own association.

Shared by every head measurement here so the eye baseline, the skull axis and
the body controls are all produced by the *same* estimator with the *same*
gates -- `triangulate_point` at its production settings, on the frames the
pipeline accepted. Nothing is scored against a landmark the pipeline would have
rejected, and nothing enjoys a threshold the mapped joints do not.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from autoanim_gnm.commercial_multiview import triangulate_point  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from associate import CAMERAS, OUT, SUBJECT_COUNT, load  # noqa: E402


def triangulate(indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions[subject, frame, k, 3], support[subject, frame, k], used[frame, subject])."""
    cameras, observations = load()
    state = np.load(OUT / "association.npz")
    assignment, used = state["assignment"], state["used"]
    frames = len(observations[0])

    marks = [
        [
            [np.asarray(person["landmarks_soma77"], dtype=np.float64) for person in values[frame]["people"]]
            for frame in range(frames)
        ]
        for values in observations
    ]

    positions = np.full((SUBJECT_COUNT, frames, len(indices), 3), np.nan)
    support = np.zeros((SUBJECT_COUNT, frames, len(indices)), dtype=np.int64)
    for frame in range(frames):
        for subject in range(SUBJECT_COUNT):
            if not used[frame, subject]:
                continue
            for slot, index in enumerate(indices):
                points = np.full((len(CAMERAS), 2), np.nan)
                weights = np.zeros(len(CAMERAS))
                for camera in range(len(CAMERAS)):
                    person = int(assignment[frame, subject, camera])
                    if person < 0:
                        continue
                    x, y, confidence = marks[camera][frame][person][index]
                    points[camera], weights[camera] = (x, y), confidence
                result = triangulate_point(cameras, points, weights, pixel_scale=1.0)
                if result is not None:
                    positions[subject, frame, slot] = result.position_world_m
                    support[subject, frame, slot] = len(result.used_camera_indices)
    return positions, support, used
