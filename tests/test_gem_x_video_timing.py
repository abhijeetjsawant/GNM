from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from workers.gem_x.video_timing import (
    decode_video_with_timing,
    load_timing,
    verify_decoded_frames,
    write_timing,
)


class _Frame:
    def __init__(self, pts: int | None, value: int) -> None:
        self.pts = pts
        self.value = value

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        return np.full((2, 3, 3), self.value, dtype=np.uint8)


class _Container:
    def __init__(self, frames: list[_Frame]) -> None:
        self.frames = frames
        self.streams = SimpleNamespace(
            video=[SimpleNamespace(time_base=Fraction(1, 30))]
        )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def decode(self, _stream):
        return iter(self.frames)


def _av(frames: list[_Frame]):
    return SimpleNamespace(open=lambda _path: _Container(frames))


def test_decode_records_exact_pts_and_rgb_identity(tmp_path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")

    frames, timing = decode_video_with_timing(
        video,
        av_module=_av([_Frame(10, 1), _Frame(11, 2)]),
    )

    assert frames.shape == (2, 2, 3, 3)
    assert timing["source_pts"] == [10, 11]
    assert timing["source_time_base"] == {"numerator": 1, "denominator": 30}
    verify_decoded_frames(frames, timing)


@pytest.mark.parametrize(
    "frames",
    [
        [_Frame(None, 1), _Frame(1, 2)],
        [_Frame(2, 1), _Frame(2, 2)],
        [_Frame(2, 1), _Frame(1, 2)],
    ],
)
def test_decode_rejects_missing_or_nonmonotonic_pts(tmp_path, frames) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")

    with pytest.raises(RuntimeError, match="PTS"):
        decode_video_with_timing(video, av_module=_av(frames))


def test_timing_rejects_input_mutation_and_frame_mismatch(tmp_path) -> None:
    video = tmp_path / "input.mp4"
    timing_path = tmp_path / "timing.json"
    video.write_bytes(b"original")
    frames, timing = decode_video_with_timing(
        video,
        av_module=_av([_Frame(0, 1), _Frame(1, 2)]),
    )
    write_timing(timing_path, timing)
    mutated = frames.copy()
    mutated[1, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="frame identity"):
        verify_decoded_frames(mutated, timing)

    video.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="not bound"):
        load_timing(timing_path, video)
