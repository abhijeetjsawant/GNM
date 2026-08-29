"""Timing and facial-ownership contracts for MAMMA-backed character takes.

MAMMA returns a body-only SMPL-X sequence.  A connected AutoAnim character
therefore needs an explicit source-frame clock and an equally explicit facial
performance mode before it can be composed with Google GNM.  This module keeps
that information inspectable; it never invents facial acting from body motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acting import TICKS_PER_SECOND
from .body import BodyTrack
from .serialization import write_npz


class MammaCompositionError(ValueError):
    """A MAMMA take cannot be given a safe, exact composition timeline."""


@dataclass(frozen=True, slots=True)
class MammaCaptureTimeline:
    """The original-video frame mapping for one MAMMA body sequence."""

    input_sha256: str
    source_pts: np.ndarray
    ticks: np.ndarray
    source_time_base_numerator: int
    source_time_base_denominator: int


def mamma_capture_timeline(
    body_track: BodyTrack,
    *,
    input_sha256: str,
    source_start_frame: int,
    source_frame_rate_hz: int = 30,
    source_time_base_denominator: int = 15_360,
) -> MammaCaptureTimeline:
    """Bind MAMMA samples to their original camera-frame PTS values.

    ``source_start_frame`` is deliberately required.  MAMMA may process a
    window from a longer clip, and resetting its source frame numbers to zero
    would falsely claim that its output corresponds to the start of the source
    video.
    """

    if (
        not isinstance(input_sha256, str)
        or len(input_sha256) != 64
        or any(character not in "0123456789abcdef" for character in input_sha256)
        or type(source_start_frame) is not int
        or source_start_frame < 0
        or type(source_frame_rate_hz) is not int
        or source_frame_rate_hz <= 0
        or type(source_time_base_denominator) is not int
        or source_time_base_denominator <= 0
        or source_time_base_denominator % source_frame_rate_hz
        or body_track.sample_rate_hz != source_frame_rate_hz
        or body_track.ticks_per_second != TICKS_PER_SECOND
    ):
        raise MammaCompositionError("MAMMA source timing contract is invalid")
    expected_ticks = np.arange(len(body_track.ticks), dtype=np.int64) * (
        TICKS_PER_SECOND // source_frame_rate_hz
    )
    if not np.array_equal(body_track.ticks, expected_ticks):
        raise MammaCompositionError("MAMMA body ticks are not a zero-based source clock")
    pts_per_frame = source_time_base_denominator // source_frame_rate_hz
    source_frames = source_start_frame + np.arange(len(body_track.ticks), dtype=np.int64)
    return MammaCaptureTimeline(
        input_sha256=input_sha256,
        source_pts=source_frames * pts_per_frame,
        ticks=body_track.ticks.copy(),
        source_time_base_numerator=1,
        source_time_base_denominator=source_time_base_denominator,
    )


def write_neutral_gnm_face_performance(
    path: str,
    *,
    timeline: MammaCaptureTimeline,
    identity: np.ndarray | None = None,
) -> None:
    """Write a neutral GNM face track that follows body neck/head motion.

    The resulting track deliberately contains no estimated expression, lipsync,
    tongue, or eye motion.  It is suitable only for checking the attachment and
    the body-owned head transform on a non-dialogue MAMMA take.
    """

    frames = len(timeline.ticks)
    face_identity = (
        np.zeros(253, dtype=np.float32)
        if identity is None
        else np.asarray(identity, dtype=np.float32)
    )
    if (
        frames < 2
        or face_identity.shape != (253,)
        or not np.isfinite(face_identity).all()
        or timeline.source_pts.shape != (frames,)
        or timeline.ticks.shape != (frames,)
        or np.any(np.diff(timeline.source_pts) <= 0)
        or np.any(np.diff(timeline.ticks) <= 0)
    ):
        raise MammaCompositionError("Neutral GNM face performance inputs are invalid")
    write_npz(
        path,
        identity=face_identity,
        expression=np.zeros((frames, 383), dtype=np.float32),
        rotations=np.zeros((frames, 4, 3), dtype=np.float32),
        translation=np.zeros((frames, 3), dtype=np.float32),
        timestamps_seconds=timeline.ticks.astype(np.float64) / TICKS_PER_SECOND,
        source_pts=timeline.source_pts,
    )


__all__ = [
    "MammaCaptureTimeline",
    "MammaCompositionError",
    "mamma_capture_timeline",
    "write_neutral_gnm_face_performance",
]
