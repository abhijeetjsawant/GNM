"""Exact, observation-only audio-sample to video-PTS evidence.

The video-follow lane must not infer speech, change animation controls, or hide
container timing behind the normalized browser proxy.  This module inspects the
retained source, hashes a deterministic native-rate mono PCM decode, and joins
that sample clock to exact display-frame intervals.  Unavailable audio evidence
is represented explicitly and is never treated as permission to fuse signals.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from .errors import AutoAnimError
from .serialization import write_json


AUDIO_VIDEO_TIMING_SCHEMA_VERSION = "autoanim.audio-video-timing.v1"
AUDIO_VIDEO_TIMING_POLICY = "observation_only_no_motion_effect"
MAX_AUDIO_VIDEO_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_FFPROBE_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_DECODED_PCM_BYTES = 2 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _TimingBlocked(Exception):
    def __init__(self, status: str, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_source(source: Path, directory: Path) -> tuple[Path, str, int]:
    """Copy one descriptor-bound source into a private immutable tool snapshot."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise AutoAnimError(
            "INPUT_INVALID",
            "Retained video source could not be opened safely",
        ) from exc
    snapshot = directory / "retained-source.bin"
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source_handle, snapshot.open(
            "xb"
        ) as target:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                byte_count += len(block)
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    except OSError as exc:
        snapshot.unlink(missing_ok=True)
        raise AutoAnimError(
            "INPUT_INVALID",
            "Retained video source could not be snapshotted safely",
        ) from exc
    if byte_count <= 0:
        snapshot.unlink(missing_ok=True)
        raise AutoAnimError("INPUT_INVALID", "Retained video source is empty")
    os.chmod(snapshot, 0o400)
    return snapshot, digest.hexdigest(), byte_count


def _redact_tool_diagnostic(message: str, command: Sequence[str]) -> str:
    """Remove source/tool filesystem paths from persisted failure evidence."""

    output = message
    for value in sorted(set(command), key=len, reverse=True):
        if not value or ("/" not in value and "\\" not in value):
            continue
        output = output.replace(value, "<REDACTED_PATH>")
        try:
            output = output.replace(str(Path(value).resolve()), "<REDACTED_PATH>")
        except (OSError, ValueError):
            pass
    # Failure evidence needs a typed reason, not arbitrary multi-line decoder
    # logs. Keeping one bounded line also prevents control/log injection.
    return output.replace("\r", " ").replace("\n", " ").strip()[:1024]


def _serialized_evidence_size(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ) + 1


def _tool_version(executable: str) -> str:
    try:
        result = subprocess.run(
            (executable, "-version"),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _TimingBlocked(
            "blocked_dependency_missing",
            "DEPENDENCY_MISSING",
            f"Could not run {executable}: {exc}",
        ) from exc
    lines = result.stdout.splitlines()
    if not lines:
        raise _TimingBlocked(
            "blocked_dependency_missing",
            "DEPENDENCY_MISSING",
            f"{executable} did not report a version",
        )
    return lines[0].strip()


def _fraction(value: object, *, field: str, positive: bool = False) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            f"Invalid {field}: {value!r}",
        ) from exc
    if positive and result <= 0:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            f"{field} must be positive",
        )
    return result


def _rational(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _milliseconds(value: Fraction) -> float:
    return float(value * 1000)


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _covering_sample_span(
    start: Fraction,
    end: Fraction,
    *,
    audio_start: Fraction,
    sample_rate: int,
    sample_count: int,
) -> tuple[int, int]:
    if end <= audio_start or start >= audio_start + Fraction(sample_count, sample_rate):
        boundary = 0 if end <= audio_start else sample_count
        return boundary, boundary
    relative_start = (start - audio_start) * sample_rate
    relative_end = (end - audio_start) * sample_rate
    first = min(sample_count, max(0, _floor(relative_start)))
    end_exclusive = min(sample_count, max(0, _ceil(relative_end)))
    if end_exclusive < first:
        end_exclusive = first
    return first, end_exclusive


def _probe_command(source: Path, ffprobe_bin: str) -> tuple[str, ...]:
    return (
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,time_base,start_pts,duration_ts,"
            "sample_fmt,sample_rate,channels,channel_layout:"
            "frame=media_type,stream_index,best_effort_timestamp,duration,"
            "pkt_duration,nb_samples,sample_rate,channels,channel_layout"
        ),
        "-show_streams",
        "-show_frames",
        "-of",
        "json",
        str(source),
    )


def _probe_source(command: tuple[str, ...]) -> dict[str, Any]:
    try:
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(command, stdout=output, stderr=error_log)
            deadline = time.monotonic() + 180.0
            exceeded = False
            timed_out = False
            while process.poll() is None:
                if os.fstat(output.fileno()).st_size > MAX_FFPROBE_OUTPUT_BYTES:
                    exceeded = True
                    process.kill()
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(0.01)
            process.wait()
            output_size = os.fstat(output.fileno()).st_size
            if exceeded or output_size > MAX_FFPROBE_OUTPUT_BYTES:
                raise _TimingBlocked(
                    "blocked_probe_output_limit",
                    "LIMIT_EXCEEDED",
                    "FFprobe timing output exceeds the configured evidence limit",
                )
            if timed_out:
                raise _TimingBlocked(
                    "blocked_probe_timeout",
                    "LIMIT_EXCEEDED",
                    "FFprobe timed out while inspecting source A/V timing",
                )
            error_log.seek(0)
            error_text = error_log.read(4096).decode("utf-8", errors="replace")
            if process.returncode != 0:
                diagnostic = _redact_tool_diagnostic(error_text, command)
                raise _TimingBlocked(
                    "blocked_unexplained_timing",
                    "MEDIA_INVALID",
                    "FFprobe could not inspect source A/V timing"
                    + (f": {diagnostic}" if diagnostic else ""),
                )
            output.seek(0)
            with open(output.fileno(), "r", encoding="utf-8", closefd=False) as text_output:
                payload = json.load(text_output)
    except FileNotFoundError as exc:
        raise _TimingBlocked(
            "blocked_dependency_missing",
            "DEPENDENCY_MISSING",
            f"FFprobe is unavailable: {command[0]}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            "FFprobe returned invalid JSON for source A/V timing",
        ) from exc
    if not isinstance(payload, dict):
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            "FFprobe A/V timing root is not an object",
        )
    return payload


def _pcm_command(source: Path, ffmpeg_bin: str) -> tuple[str, ...]:
    return (
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    )


def _command_template(command: Sequence[str], source: Path) -> list[str]:
    """Remove the retained source's workstation path from served evidence."""

    source_value = str(source)
    return ["<RETAINED_SOURCE>" if value == source_value else value for value in command]


def _hash_pcm(command: tuple[str, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with tempfile.TemporaryFile() as error_log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=error_log,
            )
            assert process.stdout is not None
            try:
                for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
                    byte_count += len(block)
                    if byte_count > MAX_DECODED_PCM_BYTES:
                        process.kill()
                        raise _TimingBlocked(
                            "blocked_decode_limit",
                            "LIMIT_EXCEEDED",
                            "Decoded mono PCM exceeds the configured evidence limit",
                        )
                    digest.update(block)
                return_code = process.wait(timeout=900)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise _TimingBlocked(
                    "blocked_decode_timeout",
                    "LIMIT_EXCEEDED",
                    "Audio PCM decoding timed out",
                ) from exc
            error_log.seek(0)
            error_text = error_log.read(4096).decode("utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise _TimingBlocked(
            "blocked_dependency_missing",
            "DEPENDENCY_MISSING",
            f"FFmpeg is unavailable: {command[0]}",
        ) from exc
    if return_code != 0:
        diagnostic = _redact_tool_diagnostic(error_text, command)
        raise _TimingBlocked(
            "blocked_decode_failed",
            "MEDIA_INVALID",
            "Could not decode primary audio stream to mono PCM"
            + (f": {diagnostic}" if diagnostic else ""),
        )
    if byte_count <= 0 or byte_count % 2:
        raise _TimingBlocked(
            "blocked_decode_failed",
            "MEDIA_INVALID",
            "Decoded mono PCM is empty or not aligned to signed 16-bit samples",
        )
    return digest.hexdigest(), byte_count // 2


def _streams(payload: Mapping[str, Any], codec_type: str) -> list[dict[str, Any]]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return []
    return [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type
    ]


def _frames(payload: Mapping[str, Any], stream_index: int) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    return [
        frame
        for frame in frames
        if isinstance(frame, dict) and frame.get("stream_index") == stream_index
    ]


def _base_evidence(
    *,
    source_name: str,
    source_bytes: int,
    source_sha256: str,
    ffprobe_version: str | None,
    ffprobe_command: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schemaVersion": AUDIO_VIDEO_TIMING_SCHEMA_VERSION,
        "kind": "audio_sample_video_pts_join",
        "policy": AUDIO_VIDEO_TIMING_POLICY,
        "sourceMode": "video_follow",
        "consumedByRetargeting": False,
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "bytes": source_bytes,
        },
        "tools": {
            "ffprobe": {
                "version": ffprobe_version,
                "command": list(ffprobe_command),
            },
            "ffmpeg": None,
        },
        "claims": {
            "changesFinalGNMMotion": False,
            "audioContentClassified": False,
            "speechContentInferred": False,
            "lipSyncCorrected": False,
            "emotionInferred": False,
            "productionValidated": False,
        },
    }


def _blocked_evidence(
    base: Mapping[str, Any],
    blocked: _TimingBlocked,
) -> dict[str, Any]:
    evidence = dict(base)
    evidence.update(
        {
            "status": blocked.status,
            "audioVideoJoin": None,
            "fusionGate": {
                "status": "blocked",
                "reasons": [blocked.code],
                "exactClockJoinAvailable": False,
                "audioSemanticEvidenceAvailable": False,
            },
            "failure": {
                "code": blocked.code,
                "message": blocked.message,
            },
            "caveats": [
                "Visual-only video_follow remains valid; unavailable A/V evidence is not motion input.",
                "No audio content, speech, viseme, or emotion claim is made.",
            ],
        }
    )
    return evidence


def _stream_index(stream: Mapping[str, Any], *, field: str) -> int:
    try:
        value = int(stream["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            f"Primary {field} stream has no integer index",
        ) from exc
    if value < 0:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            f"Primary {field} stream index is negative",
        )
    return value


def _integer(value: object, *, field: str, positive: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            f"Invalid {field}: {value!r}",
        ) from exc
    if positive and result <= 0:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            f"{field} must be positive",
        )
    return result


def _build_available_join(
    payload: Mapping[str, Any],
    *,
    source: Path,
    base: dict[str, Any],
    expected_video_source_pts: tuple[int, ...],
    expected_video_time_base: Fraction,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    video_streams = _streams(payload, "video")
    if not video_streams:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            "Retained source has no primary video stream",
        )
    audio_streams = _streams(payload, "audio")
    if not audio_streams:
        raise _TimingBlocked(
            "unavailable_no_audio",
            "AUDIO_STREAM_UNAVAILABLE",
            "Retained source has no decodable primary audio stream",
        )
    video_stream = video_streams[0]
    audio_stream = audio_streams[0]
    video_index = _stream_index(video_stream, field="video")
    audio_index = _stream_index(audio_stream, field="audio")
    video_time_base = _fraction(
        video_stream.get("time_base"), field="video time_base", positive=True
    )
    if video_time_base != expected_video_time_base:
        raise _TimingBlocked(
            "blocked_source_binding_mismatch",
            "INTEGRITY_FAILED",
            "A/V probe video time base does not match the retained capture",
        )
    video_frames = _frames(payload, video_index)
    try:
        video_pts = tuple(int(frame["best_effort_timestamp"]) for frame in video_frames)
    except (KeyError, TypeError, ValueError) as exc:
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            "A/V probe is missing exact display-frame PTS",
        ) from exc
    if video_pts != expected_video_source_pts:
        raise _TimingBlocked(
            "blocked_source_binding_mismatch",
            "INTEGRITY_FAILED",
            "A/V probe display-frame PTS do not match the retained capture",
        )
    if len(video_pts) > 1 and any(
        right <= left for left, right in zip(video_pts, video_pts[1:])
    ):
        raise _TimingBlocked(
            "blocked_unexplained_timing",
            "MEDIA_INVALID",
            "Display-frame PTS are not strictly increasing",
        )

    display_intervals: list[tuple[Fraction, Fraction, str]] = []
    for index, source_pts in enumerate(video_pts):
        start = Fraction(source_pts) * video_time_base
        # FFprobe's integer `duration` and `pkt_duration` frame fields use the
        # selected stream's declared time base.  The similarly named
        # `duration_time` fields are decimal renderings and are intentionally
        # not requested because they would discard rational precision.
        duration_value = video_frames[index].get(
            "duration", video_frames[index].get("pkt_duration")
        )
        if duration_value is not None:
            duration = _integer(
                duration_value,
                field=f"video frame {index} duration",
                positive=True,
            )
            end = start + Fraction(duration) * video_time_base
            duration_source = "ffprobe_frame_duration_ticks"
        elif index + 1 < len(video_pts):
            end = Fraction(video_pts[index + 1]) * video_time_base
            duration_source = "next_display_frame_pts"
        else:
            if duration_value is None and video_stream.get("duration_ts") is not None:
                stream_duration = _integer(
                    video_stream["duration_ts"],
                    field="video stream duration_ts",
                    positive=True,
                )
                stream_start = _integer(
                    video_stream.get("start_pts", video_pts[0]),
                    field="video stream start_pts",
                )
                duration_value = stream_start + stream_duration - source_pts
            duration = _integer(
                duration_value,
                field="last video frame duration",
                positive=True,
            )
            end = start + Fraction(duration) * video_time_base
            duration_source = "ffprobe_last_frame_duration_ticks"
        if end <= start:
            raise _TimingBlocked(
                "blocked_unexplained_timing",
                "MEDIA_INVALID",
                f"Video frame {index} has a non-positive display interval",
            )
        if index + 1 < len(video_pts):
            next_start = Fraction(video_pts[index + 1]) * video_time_base
            if end > next_start + video_time_base:
                raise _TimingBlocked(
                    "blocked_unexplained_timing",
                    "MEDIA_INVALID",
                    f"Video frame {index} duration overlaps the next display frame",
                )
        display_intervals.append((start, end, duration_source))

    sample_rate = _integer(
        audio_stream.get("sample_rate"), field="audio sample_rate", positive=True
    )
    source_channels = _integer(
        audio_stream.get("channels"), field="audio channels", positive=True
    )
    audio_time_base = _fraction(
        audio_stream.get("time_base"), field="audio time_base", positive=True
    )
    audio_frames = _frames(payload, audio_index)
    if not audio_frames:
        raise _TimingBlocked(
            "blocked_missing_audio_timing",
            "MEDIA_INVALID",
            "Primary audio stream has no decoded frame timing",
        )

    audio_frame_records: list[dict[str, Any]] = []
    decoded_sample_start = 0
    first_audio_pts: int | None = None
    previous_audio_pts: int | None = None
    maximum_residual = Fraction(0)
    for index, frame in enumerate(audio_frames):
        pts = _integer(
            frame.get("best_effort_timestamp"),
            field=f"audio frame {index} PTS",
        )
        sample_count = _integer(
            frame.get("nb_samples"),
            field=f"audio frame {index} sample count",
            positive=True,
        )
        frame_rate = frame.get("sample_rate")
        if frame_rate is not None and _integer(
            frame_rate, field=f"audio frame {index} sample_rate", positive=True
        ) != sample_rate:
            raise _TimingBlocked(
                "blocked_unexplained_timing",
                "MEDIA_INVALID",
                f"Audio sample rate changes at decoded frame {index}",
            )
        frame_channels = frame.get("channels")
        if frame_channels is not None and _integer(
            frame_channels, field=f"audio frame {index} channels", positive=True
        ) != source_channels:
            raise _TimingBlocked(
                "blocked_unexplained_timing",
                "MEDIA_INVALID",
                f"Audio channel count changes at decoded frame {index}",
            )
        if previous_audio_pts is not None and pts <= previous_audio_pts:
            raise _TimingBlocked(
                "blocked_nonmonotonic_audio_timing",
                "MEDIA_INVALID",
                f"Decoded audio PTS are not strictly increasing at frame {index}",
            )
        if first_audio_pts is None:
            first_audio_pts = pts
        observed_start = Fraction(pts) * audio_time_base
        modeled_start = (
            Fraction(first_audio_pts) * audio_time_base
            + Fraction(decoded_sample_start, sample_rate)
        )
        residual = observed_start - modeled_start
        maximum_residual = max(maximum_residual, abs(residual))
        if abs(residual) >= audio_time_base:
            raise _TimingBlocked(
                "blocked_unexplained_audio_timing",
                "MEDIA_INVALID",
                "Decoded audio PTS depart from the contiguous native sample clock by at least "
                f"one source time-base tick at frame {index}",
            )
        audio_frame_records.append(
            {
                "frameIndex": index,
                "sourcePTS": pts,
                "sourceStartExactRational": _rational(observed_start),
                "decodedSampleStartIndex": decoded_sample_start,
                "decodedSampleEndExclusive": decoded_sample_start + sample_count,
                "sampleCount": sample_count,
                "contiguousClockResidualExactRational": _rational(residual),
            }
        )
        decoded_sample_start += sample_count
        previous_audio_pts = pts

    assert first_audio_pts is not None
    ffmpeg_version = _tool_version(ffmpeg_bin)
    pcm_command = _pcm_command(source, ffmpeg_bin)
    pcm_sha256, pcm_sample_count = _hash_pcm(pcm_command)
    if pcm_sample_count != decoded_sample_start:
        raise _TimingBlocked(
            "blocked_decode_timing_mismatch",
            "MEDIA_INVALID",
            "Decoded mono PCM sample count does not match FFprobe decoded audio frames "
            f"({pcm_sample_count} PCM samples, {decoded_sample_start} timed samples)",
        )
    base["tools"]["ffmpeg"] = {
        "version": ffmpeg_version,
        "command": _command_template(pcm_command, source),
    }

    audio_start = Fraction(first_audio_pts) * audio_time_base
    audio_duration = Fraction(pcm_sample_count, sample_rate)
    audio_end = audio_start + audio_duration
    video_start = display_intervals[0][0]
    video_end = display_intervals[-1][1]
    video_duration = video_end - video_start
    start_offset = audio_start - video_start
    end_offset = audio_end - video_end
    duration_drift = end_offset - start_offset

    joined_video_frames: list[dict[str, Any]] = []
    complete_coverage_count = 0
    frames_with_audio = 0
    for index, ((start, end, duration_source), source_pts) in enumerate(
        zip(display_intervals, video_pts, strict=True)
    ):
        sample_start, sample_end = _covering_sample_span(
            start,
            end,
            audio_start=audio_start,
            sample_rate=sample_rate,
            sample_count=pcm_sample_count,
        )
        has_audio = sample_end > sample_start
        coverage_complete = start >= audio_start and end <= audio_end
        frames_with_audio += int(has_audio)
        complete_coverage_count += int(coverage_complete)
        joined_video_frames.append(
            {
                "frameIndex": index,
                "sourcePTS": source_pts,
                "displayStartExactRational": _rational(start),
                "displayEndExactRational": _rational(end),
                "displayDurationSource": duration_source,
                "coveringAudioSampleSpan": [sample_start, sample_end],
                "hasAudioSampleCoverage": has_audio,
                "audioCoverageComplete": coverage_complete,
            }
        )

    fusion_reasons = ["audio_content_not_classified"]
    if start_offset:
        fusion_reasons.append("nonzero_av_start_offset")
    if duration_drift:
        fusion_reasons.append("nonzero_av_duration_drift")
    if complete_coverage_count != len(video_pts):
        fusion_reasons.append("incomplete_video_audio_coverage")

    evidence = dict(base)
    evidence.update(
        {
            "status": "available_observation",
            "audioVideoJoin": {
                "video": {
                    "streamIndex": video_index,
                    "codec": str(video_stream.get("codec_name", "unknown")),
                    "timeBase": _rational(video_time_base),
                    "streamStartPTS": (
                        int(video_stream["start_pts"])
                        if video_stream.get("start_pts") is not None
                        else None
                    ),
                    "frameCount": len(video_pts),
                    "displayStartExactRational": _rational(video_start),
                    "displayEndExactRational": _rational(video_end),
                    "displayDurationExactRational": _rational(video_duration),
                    "frames": joined_video_frames,
                },
                "audio": {
                    "streamIndex": audio_index,
                    "availableAudioStreamCount": len(audio_streams),
                    "codec": str(audio_stream.get("codec_name", "unknown")),
                    "sourceSampleFormat": audio_stream.get("sample_fmt"),
                    "timeBase": _rational(audio_time_base),
                    "streamStartPTS": (
                        int(audio_stream["start_pts"])
                        if audio_stream.get("start_pts") is not None
                        else None
                    ),
                    "decodedFirstFramePTS": first_audio_pts,
                    "decodedStartExactRational": _rational(audio_start),
                    "decodedEndExactRational": _rational(audio_end),
                    "sampleRate": sample_rate,
                    "sourceChannels": source_channels,
                    "sourceChannelLayout": audio_stream.get("channel_layout"),
                    "decodedFrames": audio_frame_records,
                    "maximumContiguousClockResidualExactRational": _rational(
                        maximum_residual
                    ),
                    "maximumResidualExplanation": (
                        "strictly_less_than_one_source_time_base_tick"
                    ),
                    "monoPcm": {
                        "format": "s16le",
                        "sampleRate": sample_rate,
                        "channels": 1,
                        "sampleCount": pcm_sample_count,
                        "byteCount": pcm_sample_count * 2,
                        "sha256": pcm_sha256,
                        "channelConversion": (
                            "ffmpeg_-ac_1_default_downmix_matrix_version_bound"
                        ),
                    },
                },
                "sync": {
                    "audioMinusVideoStartOffsetExactRational": _rational(start_offset),
                    "audioMinusVideoStartOffsetMs": _milliseconds(start_offset),
                    "audioMinusVideoEndOffsetExactRational": _rational(end_offset),
                    "audioMinusVideoEndOffsetMs": _milliseconds(end_offset),
                    "audioMinusVideoDurationDriftExactRational": _rational(
                        duration_drift
                    ),
                    "audioMinusVideoDurationDriftMs": _milliseconds(duration_drift),
                    "videoDurationExactRational": _rational(video_duration),
                    "audioDurationExactRational": _rational(audio_duration),
                    "videoFramesWithAudioSampleCoverage": frames_with_audio,
                    "videoFramesWithCompleteAudioCoverage": complete_coverage_count,
                    "completeAudioCoverageForAllVideoFrames": (
                        complete_coverage_count == len(video_pts)
                    ),
                },
            },
            "fusionGate": {
                "status": "blocked",
                "reasons": fusion_reasons,
                "exactClockJoinAvailable": True,
                "audioSemanticEvidenceAvailable": False,
            },
            "failure": None,
            "caveats": [
                "Decoded samples are timed from the first decoded audio PTS at the native sample rate.",
                "Per-frame audio PTS quantization is accepted only below one declared audio time-base tick.",
                "Covering sample spans may include boundary samples that only partially overlap a video frame.",
                "No audio content, speech, viseme, emotion, or automatic timing correction is inferred.",
                "This artifact is diagnostic and is not consumed by retargeting in video_follow mode.",
            ],
        }
    )
    return evidence


def build_audio_video_timing_evidence(
    source_path: str | Path,
    *,
    expected_source_sha256: str,
    expected_video_source_pts: Sequence[int],
    expected_video_time_base: Fraction,
    require_available: bool = False,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    """Inspect retained source timing without modifying visual motion.

    When ``require_available`` is false, missing or ambiguous audio evidence is
    represented by a fusion-blocking artifact so visual-only capture can still
    succeed.  Source/capture binding mismatches always raise.
    """

    source = Path(source_path).expanduser().absolute()
    if not source.is_file() or source.stat().st_size <= 0:
        raise AutoAnimError("INPUT_INVALID", "Retained video source is unavailable")
    expected_pts = tuple(int(value) for value in expected_video_source_pts)
    if not expected_pts or any(
        right <= left for left, right in zip(expected_pts, expected_pts[1:])
    ):
        raise AutoAnimError(
            "MEDIA_INVALID", "Expected display-frame PTS must be non-empty and monotonic"
        )
    video_time_base = Fraction(expected_video_time_base)
    if video_time_base <= 0:
        raise AutoAnimError("MEDIA_INVALID", "Expected video time base must be positive")

    source_name = source.name
    with tempfile.TemporaryDirectory(prefix="autoanim-av-source-") as snapshot_dir:
        snapshot, source_sha256, source_bytes = _snapshot_source(
            source,
            Path(snapshot_dir),
        )
        if source_sha256 != expected_source_sha256:
            raise AutoAnimError(
                "INTEGRITY_FAILED",
                "Retained source hash does not match capture provenance",
            )
        command = _probe_command(snapshot, ffprobe_bin)
        base = _base_evidence(
            source_name=source_name,
            source_bytes=source_bytes,
            source_sha256=source_sha256,
            ffprobe_version=None,
            ffprobe_command=tuple(_command_template(command, snapshot)),
        )
        try:
            ffprobe_version = _tool_version(ffprobe_bin)
            base["tools"]["ffprobe"]["version"] = ffprobe_version
            payload = _probe_source(command)
            evidence = _build_available_join(
                payload,
                source=snapshot,
                base=base,
                expected_video_source_pts=expected_pts,
                expected_video_time_base=video_time_base,
                ffmpeg_bin=ffmpeg_bin,
            )
            if (
                snapshot.stat().st_size != source_bytes
                or _file_sha256(snapshot) != source_sha256
            ):
                raise _TimingBlocked(
                    "blocked_source_binding_mismatch",
                    "INTEGRITY_FAILED",
                    "Immutable A/V tool snapshot changed during inspection",
                )
            if _serialized_evidence_size(evidence) > MAX_AUDIO_VIDEO_EVIDENCE_BYTES:
                raise _TimingBlocked(
                    "blocked_evidence_output_limit",
                    "LIMIT_EXCEEDED",
                    "A/V timing evidence exceeds the configured artifact limit",
                )
        except _TimingBlocked as blocked:
            if blocked.code == "INTEGRITY_FAILED":
                raise AutoAnimError(blocked.code, blocked.message) from blocked
            if require_available:
                raise AutoAnimError(blocked.code, blocked.message) from blocked
            evidence = _blocked_evidence(base, blocked)
        if _serialized_evidence_size(evidence) > MAX_AUDIO_VIDEO_EVIDENCE_BYTES:
            raise AutoAnimError(
                "LIMIT_EXCEEDED",
                "Blocked A/V timing evidence exceeds the configured artifact limit",
            )
        return evidence


def write_audio_video_timing_evidence(
    path: str | Path,
    source_path: str | Path,
    *,
    expected_source_sha256: str,
    expected_video_source_pts: Sequence[int],
    expected_video_time_base: Fraction,
    require_available: bool = False,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    evidence = build_audio_video_timing_evidence(
        source_path,
        expected_source_sha256=expected_source_sha256,
        expected_video_source_pts=expected_video_source_pts,
        expected_video_time_base=expected_video_time_base,
        require_available=require_available,
        ffprobe_bin=ffprobe_bin,
        ffmpeg_bin=ffmpeg_bin,
    )
    return write_json(path, evidence)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON member: {key}")
        result[key] = value
    return result


def _verified_rational(value: object, *, field: str) -> Fraction:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or value[1] <= 0
    ):
        raise ValueError(f"{field} must be one exact rational pair")
    result = Fraction(value[0], value[1])
    if value != [result.numerator, result.denominator]:
        raise ValueError(f"{field} rational must be reduced")
    return result


def load_verified_audio_video_timing_evidence(
    path: str | Path,
    *,
    expected_source_sha256: str,
    expected_video_source_pts: Sequence[int],
    expected_pcm_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify source binding and all exact sample-span arithmetic."""

    source_path = Path(path)
    size = source_path.stat().st_size
    if size <= 0 or size > MAX_AUDIO_VIDEO_EVIDENCE_BYTES:
        raise ValueError("A/V timing evidence size is outside the accepted bounds")
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON number: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("A/V timing evidence must be canonical UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("A/V timing evidence root must be an object")
    source = payload.get("source")
    claims = payload.get("claims")
    fusion_gate = payload.get("fusionGate")
    if (
        payload.get("schemaVersion") != AUDIO_VIDEO_TIMING_SCHEMA_VERSION
        or payload.get("kind") != "audio_sample_video_pts_join"
        or payload.get("policy") != AUDIO_VIDEO_TIMING_POLICY
        or payload.get("sourceMode") != "video_follow"
        or payload.get("consumedByRetargeting") is not False
        or not isinstance(source, dict)
        or source.get("sha256") != expected_source_sha256
        or not isinstance(claims, dict)
        or claims.get("changesFinalGNMMotion") is not False
        or claims.get("audioContentClassified") is not False
        or claims.get("speechContentInferred") is not False
        or claims.get("lipSyncCorrected") is not False
        or claims.get("productionValidated") is not False
        or not isinstance(fusion_gate, dict)
        or fusion_gate.get("status") != "blocked"
        or fusion_gate.get("audioSemanticEvidenceAvailable") is not False
    ):
        raise ValueError("Unsupported or unbound A/V timing evidence contract")

    status = payload.get("status")
    if status != "available_observation":
        status_codes = {
            "unavailable_no_audio": "AUDIO_STREAM_UNAVAILABLE",
            "blocked_dependency_missing": "DEPENDENCY_MISSING",
            "blocked_probe_timeout": "LIMIT_EXCEEDED",
            "blocked_unexplained_timing": "MEDIA_INVALID",
            "blocked_missing_audio_timing": "MEDIA_INVALID",
            "blocked_nonmonotonic_audio_timing": "MEDIA_INVALID",
            "blocked_unexplained_audio_timing": "MEDIA_INVALID",
            "blocked_decode_timing_mismatch": "MEDIA_INVALID",
            "blocked_decode_failed": "MEDIA_INVALID",
            "blocked_decode_timeout": "LIMIT_EXCEEDED",
            "blocked_decode_limit": "LIMIT_EXCEEDED",
        }
        failure = payload.get("failure")
        expected_failure_code = status_codes.get(status)
        if (
            expected_failure_code is None
            or payload.get("audioVideoJoin") is not None
            or fusion_gate.get("exactClockJoinAvailable") is not False
            or not isinstance(failure, dict)
            or failure.get("code") != expected_failure_code
            or not isinstance(failure.get("message"), str)
            or not failure["message"]
            or fusion_gate.get("reasons") != [expected_failure_code]
        ):
            raise ValueError("Invalid unavailable A/V timing evidence")
        return payload

    join = payload.get("audioVideoJoin")
    if not isinstance(join, dict):
        raise ValueError("Available A/V timing evidence has no join")
    video = join.get("video")
    audio = join.get("audio")
    sync = join.get("sync")
    if not isinstance(video, dict) or not isinstance(audio, dict) or not isinstance(sync, dict):
        raise ValueError("Available A/V timing evidence has incomplete stream records")
    expected_pts = tuple(int(value) for value in expected_video_source_pts)
    video_frames = video.get("frames")
    video_time_base = _verified_rational(video.get("timeBase"), field="video time base")
    if (
        not isinstance(video_frames, list)
        or len(video_frames) != len(expected_pts)
        or video.get("frameCount") != len(expected_pts)
        or not expected_pts
    ):
        raise ValueError("A/V timing video frames are not bound to the capture")

    mono_pcm = audio.get("monoPcm")
    if not isinstance(mono_pcm, dict):
        raise ValueError("A/V timing evidence has no mono PCM binding")
    sample_rate = mono_pcm.get("sampleRate")
    sample_count = mono_pcm.get("sampleCount")
    pcm_sha256 = mono_pcm.get("sha256")
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
        or audio.get("sampleRate") != sample_rate
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count <= 0
        or mono_pcm.get("byteCount") != sample_count * 2
        or mono_pcm.get("channels") != 1
        or mono_pcm.get("format") != "s16le"
        or not isinstance(pcm_sha256, str)
        or _SHA256_RE.fullmatch(pcm_sha256) is None
        or (expected_pcm_sha256 is not None and pcm_sha256 != expected_pcm_sha256)
    ):
        raise ValueError("A/V timing mono PCM binding is invalid")
    audio_start = _verified_rational(
        audio.get("decodedStartExactRational"), field="audio start"
    )
    audio_end = _verified_rational(
        audio.get("decodedEndExactRational"), field="audio end"
    )
    if audio_end != audio_start + Fraction(sample_count, sample_rate):
        raise ValueError("A/V timing decoded audio duration does not match its sample count")
    audio_time_base = _verified_rational(audio.get("timeBase"), field="audio time base")
    if audio_time_base <= 0:
        raise ValueError("A/V timing audio time base must be positive")
    decoded_first_pts = audio.get("decodedFirstFramePTS")
    decoded_frames = audio.get("decodedFrames")
    maximum_residual = _verified_rational(
        audio.get("maximumContiguousClockResidualExactRational"),
        field="maximum audio clock residual",
    )
    if (
        not isinstance(decoded_first_pts, int)
        or isinstance(decoded_first_pts, bool)
        or audio_start != Fraction(decoded_first_pts) * audio_time_base
        or not isinstance(decoded_frames, list)
        or not decoded_frames
        or audio.get("maximumResidualExplanation")
        != "strictly_less_than_one_source_time_base_tick"
        or mono_pcm.get("channelConversion")
        != "ffmpeg_-ac_1_default_downmix_matrix_version_bound"
    ):
        raise ValueError("A/V timing decoded audio clock binding is invalid")

    decoded_sample_cursor = 0
    previous_audio_pts: int | None = None
    verified_maximum_residual = Fraction(0)
    for index, frame in enumerate(decoded_frames):
        if not isinstance(frame, dict):
            raise ValueError(f"A/V timing audio frame {index} is not an object")
        pts = frame.get("sourcePTS")
        frame_sample_start = frame.get("decodedSampleStartIndex")
        frame_sample_end = frame.get("decodedSampleEndExclusive")
        frame_sample_count = frame.get("sampleCount")
        if (
            frame.get("frameIndex") != index
            or not isinstance(pts, int)
            or isinstance(pts, bool)
            or not isinstance(frame_sample_start, int)
            or isinstance(frame_sample_start, bool)
            or not isinstance(frame_sample_end, int)
            or isinstance(frame_sample_end, bool)
            or not isinstance(frame_sample_count, int)
            or isinstance(frame_sample_count, bool)
            or frame_sample_count <= 0
            or frame_sample_start != decoded_sample_cursor
            or frame_sample_end != frame_sample_start + frame_sample_count
            or (previous_audio_pts is not None and pts <= previous_audio_pts)
        ):
            raise ValueError(f"A/V timing audio frame {index} has invalid sample timing")
        observed_start = Fraction(pts) * audio_time_base
        modeled_start = audio_start + Fraction(frame_sample_start, sample_rate)
        residual = observed_start - modeled_start
        if (
            _verified_rational(
                frame.get("sourceStartExactRational"),
                field=f"audio frame {index} source start",
            )
            != observed_start
            or _verified_rational(
                frame.get("contiguousClockResidualExactRational"),
                field=f"audio frame {index} clock residual",
            )
            != residual
            or abs(residual) >= audio_time_base
        ):
            raise ValueError(f"A/V timing audio frame {index} has invalid source clock")
        verified_maximum_residual = max(verified_maximum_residual, abs(residual))
        decoded_sample_cursor = frame_sample_end
        previous_audio_pts = pts
    if (
        decoded_sample_cursor != sample_count
        or verified_maximum_residual != maximum_residual
    ):
        raise ValueError("A/V timing decoded audio frame summary is inconsistent")

    previous_start: Fraction | None = None
    last_end: Fraction | None = None
    complete_count = 0
    with_audio_count = 0
    for index, (frame, expected_pts_value) in enumerate(
        zip(video_frames, expected_pts, strict=True)
    ):
        if not isinstance(frame, dict):
            raise ValueError(f"A/V timing video frame {index} is not an object")
        start = _verified_rational(
            frame.get("displayStartExactRational"), field=f"video frame {index} start"
        )
        end = _verified_rational(
            frame.get("displayEndExactRational"), field=f"video frame {index} end"
        )
        span = frame.get("coveringAudioSampleSpan")
        expected_span = _covering_sample_span(
            start,
            end,
            audio_start=audio_start,
            sample_rate=sample_rate,
            sample_count=sample_count,
        )
        coverage_complete = start >= audio_start and end <= audio_end
        duration_source = frame.get("displayDurationSource")
        duration_in_ticks = (end - start) / video_time_base
        duration_valid = (
            duration_source
            in {
                "ffprobe_frame_duration_ticks",
                "ffprobe_last_frame_duration_ticks",
            }
            and duration_in_ticks.denominator == 1
            and duration_in_ticks > 0
        ) or (
            duration_source == "next_display_frame_pts"
            and index + 1 < len(expected_pts)
            and end == Fraction(expected_pts[index + 1]) * video_time_base
        )
        overlap_valid = (
            index + 1 == len(expected_pts)
            or end
            <= Fraction(expected_pts[index + 1]) * video_time_base + video_time_base
        )
        if (
            frame.get("frameIndex") != index
            or frame.get("sourcePTS") != expected_pts_value
            or start != Fraction(expected_pts_value) * video_time_base
            or end <= start
            or (previous_start is not None and start <= previous_start)
            or not duration_valid
            or not overlap_valid
            or span != list(expected_span)
            or frame.get("hasAudioSampleCoverage") is not (expected_span[1] > expected_span[0])
            or frame.get("audioCoverageComplete") is not coverage_complete
        ):
            raise ValueError(f"A/V timing video frame {index} has invalid sample timing")
        with_audio_count += int(expected_span[1] > expected_span[0])
        complete_count += int(coverage_complete)
        previous_start = start
        last_end = end

    assert last_end is not None
    video_start = Fraction(expected_pts[0]) * video_time_base
    video_end = last_end
    video_duration = video_end - video_start
    if (
        _verified_rational(video.get("displayStartExactRational"), field="video start")
        != video_start
        or _verified_rational(video.get("displayEndExactRational"), field="video end")
        != video_end
        or _verified_rational(
            video.get("displayDurationExactRational"), field="video duration"
        )
        != video_duration
    ):
        raise ValueError("A/V timing top-level video interval is inconsistent")
    audio_duration = audio_end - audio_start
    start_offset = audio_start - video_start
    end_offset = audio_end - video_end
    drift = audio_duration - video_duration
    exact_checks = {
        "audioMinusVideoStartOffsetExactRational": start_offset,
        "audioMinusVideoEndOffsetExactRational": end_offset,
        "audioMinusVideoDurationDriftExactRational": drift,
        "videoDurationExactRational": video_duration,
        "audioDurationExactRational": audio_duration,
    }
    for field, expected in exact_checks.items():
        if _verified_rational(sync.get(field), field=field) != expected:
            raise ValueError(f"A/V timing sync field {field} is inconsistent")
    float_checks = {
        "audioMinusVideoStartOffsetMs": _milliseconds(start_offset),
        "audioMinusVideoEndOffsetMs": _milliseconds(end_offset),
        "audioMinusVideoDurationDriftMs": _milliseconds(drift),
    }
    for field, expected in float_checks.items():
        value = sync.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            raise ValueError(f"A/V timing sync field {field} is inconsistent")
    if (
        sync.get("videoFramesWithAudioSampleCoverage") != with_audio_count
        or sync.get("videoFramesWithCompleteAudioCoverage") != complete_count
        or sync.get("completeAudioCoverageForAllVideoFrames")
        is not (complete_count == len(expected_pts))
        or fusion_gate.get("exactClockJoinAvailable") is not True
    ):
        raise ValueError("A/V timing coverage summary is inconsistent")
    expected_fusion_reasons = ["audio_content_not_classified"]
    if start_offset:
        expected_fusion_reasons.append("nonzero_av_start_offset")
    if drift:
        expected_fusion_reasons.append("nonzero_av_duration_drift")
    if complete_count != len(expected_pts):
        expected_fusion_reasons.append("incomplete_video_audio_coverage")
    if (
        fusion_gate.get("reasons") != expected_fusion_reasons
        or payload.get("failure") is not None
    ):
        raise ValueError("A/V timing fusion gate is inconsistent")
    return payload


__all__ = [
    "AUDIO_VIDEO_TIMING_POLICY",
    "AUDIO_VIDEO_TIMING_SCHEMA_VERSION",
    "MAX_AUDIO_VIDEO_EVIDENCE_BYTES",
    "build_audio_video_timing_evidence",
    "load_verified_audio_video_timing_evidence",
    "write_audio_video_timing_evidence",
]
